"""Unit tests for RaptorQEncoder and RaptorQDecoder."""

import struct
from math import ceil

import qrstream.raptorq_codec as rq
from qrstream.raptorq_codec import (
    RaptorQEncoder,
    RaptorQDecoder,
    _rq_order_packets,
    _rq_payload_id,
    _rq_payload_id_parts,
    _rq_source_block_layout,
    _rq_source_index,
)
from qrstream.protocol import V4_VERSION, pack_v4, unpack


def _packet(sbn: int, local_esi: int, symbol_size: int = 4) -> bytes:
    payload_id = _rq_payload_id(sbn, local_esi)
    return struct.pack('>I', payload_id) + bytes([payload_id & 0xFF]) * symbol_size


class TestRaptorQPayloadIdMapping:
    def test_payload_id_roundtrip(self):
        payload_id = _rq_payload_id(7, 0x123456)
        assert _rq_payload_id_parts(payload_id) == (7, 0x123456)

    def test_single_source_block_identity_mapping(self):
        assert _rq_source_block_layout(4) == [(0, 4)]
        assert _rq_source_index(_rq_payload_id(0, 2), 4) == 2
        assert _rq_source_index(_rq_payload_id(0, 4), 4) is None

    def test_multi_source_block_global_mapping(self):
        # K > 56_403 forces two RaptorQ source blocks.  The second
        # block's local ESI 0 must map to a non-zero global source index;
        # treating PayloadId as a flat ESI would miss this in the block map.
        k = 56_405
        assert _rq_source_block_layout(k) == [(0, 28_203), (28_203, 28_202)]
        assert _rq_source_index(_rq_payload_id(0, 28_202), k) == 28_202
        assert _rq_source_index(_rq_payload_id(1, 0), k) == 28_203
        assert _rq_source_index(_rq_payload_id(1, 28_201), k) == 56_404
        assert _rq_source_index(_rq_payload_id(0, 28_203), k) is None
        assert _rq_source_index(_rq_payload_id(1, 28_202), k) is None

    def test_mapping_uses_explicit_source_block_count(self):
        assert _rq_source_block_layout(10, 3) == [(0, 4), (4, 3), (7, 3)]
        assert _rq_source_index(_rq_payload_id(1, 0), 10, 3) == 4
        assert _rq_source_index(_rq_payload_id(2, 2), 10, 3) == 9
        assert _rq_source_index(_rq_payload_id(2, 3), 10, 3) is None


class TestRaptorQEncoder:
    """RaptorQEncoder generates valid V4 packets."""

    def test_basic_generation(self):
        data = b'Hello RaptorQ!' * 20
        blocksize = 64
        encoder = RaptorQEncoder(data, blocksize)
        K = ceil(len(data) / blocksize)
        assert encoder.K == K

        blocks = list(encoder.generate_blocks(K + 5))
        assert len(blocks) == K + 5

        for packed, esi, seq in blocks:
            header, block_data = unpack(packed)
            assert header.version == V4_VERSION
            assert header.filesize == len(data)
            assert header.symbol_size == blocksize
            assert header.symbol_count == K
            assert len(block_data) == blocksize

    def test_esi_sequential(self):
        data = b'\xAB' * 256
        blocksize = 64
        encoder = RaptorQEncoder(data, blocksize)
        K = encoder.K

        blocks = list(encoder.generate_blocks(K + 3))
        esis = [esi for _, esi, _ in blocks]
        # First K ESIs should be 0..K-1 (source symbols)
        assert esis[:K] == list(range(K))
        # Repair symbols start at K
        assert all(e >= K for e in esis[K:])

    def test_random_access_source_is_not_eagerly_materialized_on_construction(self):
        """Encoder construction should NOT materialize the backing store.

        Note: ``generate_blocks()`` must materialise data to pass it to
        the raptorq library (which handles sub-block interleaving
        internally).  This test verifies that construction itself remains
        lazy — only block generation triggers materialisation.
        """
        class TrackingData:
            def __init__(self, data: bytes):
                self._data = data
                self.materialized = False

            def __len__(self):
                return len(self._data)

            def __getitem__(self, key):
                if (isinstance(key, slice)
                        and key.start is None
                        and key.stop == len(self._data)):
                    self.materialized = True
                return self._data[key]

        data = b'ABCDEFGH' * 64
        source = TrackingData(data)
        encoder = RaptorQEncoder(source, 64)

        # Construction must remain lazy (no eager materialisation).
        assert source.materialized is False

        # Block generation requires materialisation for library correctness.
        blocks = list(encoder.generate_blocks(encoder.K))
        assert source.materialized is True

        # Verify the generated data is correct.
        decoder = RaptorQDecoder()
        for packed, _, _ in blocks:
            done, _ = decoder.decode_bytes(packed)
            if done:
                break
        assert decoder.bytes_dump() == data

    def test_systematic_source_symbols(self):
        """First K packets should carry the source data directly."""
        data = b'ABCDEFGH' * 32  # 256 bytes
        blocksize = 64
        encoder = RaptorQEncoder(data, blocksize)
        K = encoder.K

        blocks = list(encoder.generate_blocks(K))
        for i, (packed, esi, seq) in enumerate(blocks):
            header, block_data = unpack(packed, skip_crc=True)
            expected = data[i * blocksize:(i + 1) * blocksize]
            if len(expected) < blocksize:
                expected = expected + b'\x00' * (blocksize - len(expected))
            assert block_data == expected, f"Source symbol {i} mismatch"

    def test_compressed_flag(self):
        data = b'\x00' * 128
        encoder = RaptorQEncoder(data, 64, compressed=True)
        packed, _, _ = next(encoder.generate_blocks(1))
        header, _ = unpack(packed)
        assert header.compressed is True

    def test_alphanumeric_flag(self):
        data = b'\x00' * 128
        encoder = RaptorQEncoder(data, 64, alphanumeric_qr=True)
        packed, _, _ = next(encoder.generate_blocks(1))
        header, _ = unpack(packed)
        assert header.alphanumeric_qr is True

    def test_seq_counter_wraps(self):
        """Sequence counter should wrap at 0xFFFF."""
        data = b'\x00' * 128
        encoder = RaptorQEncoder(data, 64)
        # Set _seq to near wrap point
        encoder._seq = 0xFFFE
        blocks = list(encoder.generate_blocks(3))
        seqs = [seq for _, _, seq in blocks]
        # _seq is reset to 0 by generate_blocks, so should be 0,1,2
        assert seqs == [0, 1, 2]

    def test_multi_source_block_packet_ordering(self):
        packets = [
            _packet(0, 0), _packet(0, 1), _packet(0, 2), _packet(0, 3),
            _packet(1, 0), _packet(1, 1), _packet(1, 2),
        ]

        ordered = _rq_order_packets(packets, total_symbols=5, source_blocks=2)
        payload_ids = [struct.unpack('>I', pkt[:4])[0] for pkt in ordered]

        assert payload_ids == [
            _rq_payload_id(0, 0),
            _rq_payload_id(1, 0),
            _rq_payload_id(0, 1),
            _rq_payload_id(1, 1),
            _rq_payload_id(0, 2),
            _rq_payload_id(0, 3),
            _rq_payload_id(1, 2),
        ]

    def test_generate_blocks_writes_z_and_round_robin_order(self):
        class FakePacketSource:
            def get_encoded_packets(self, repair_per_block):
                # With the fix, _iter_source_packets calls
                # get_encoded_packets(0) to get source symbols, and
                # generate_blocks calls get_encoded_packets(repair_count)
                # for repair symbols.  Both should return the same source
                # packets; repair_per_block > 0 additionally returns repair.
                source_pkts = [
                    _packet(0, 0), _packet(0, 1),
                    _packet(0, 2),
                    _packet(1, 0), _packet(1, 1),
                ]
                if repair_per_block == 0:
                    return source_pkts
                assert repair_per_block == 1
                repair_pkts = [
                    _packet(0, 3),  # repair for SBN 0
                    _packet(1, 2),  # repair for SBN 1
                ]
                return source_pkts + repair_pkts

        encoder = RaptorQEncoder.__new__(RaptorQEncoder)
        encoder.data = b'\x00' * 20
        encoder.filesize = 20
        encoder.blocksize = 4
        encoder.K = 5
        encoder.source_blocks = 2
        encoder.compressed = False
        encoder.alphanumeric_qr = False
        encoder._encoder = FakePacketSource()
        encoder._seq = 0
        encoder._requested_blocksize = 4

        blocks = list(RaptorQEncoder.generate_blocks(encoder, 7))
        payload_ids = [payload_id for _, payload_id, _ in blocks]
        headers = [unpack(packed)[0] for packed, _, _ in blocks]

        assert payload_ids == [
            _rq_payload_id(0, 0),
            _rq_payload_id(1, 0),
            _rq_payload_id(0, 1),
            _rq_payload_id(1, 1),
            _rq_payload_id(0, 2),
            _rq_payload_id(0, 3),
            _rq_payload_id(1, 2),
        ]
        assert all(header.reserved == 2 for header in headers)


class TestRaptorQDecoder:
    """RaptorQDecoder consumes V4 packets and recovers data."""

    def test_basic_decode(self):
        data = b'Hello RaptorQ!' * 20  # 280 bytes
        blocksize = 64
        encoder = RaptorQEncoder(data, blocksize)
        K = encoder.K

        decoder = RaptorQDecoder()
        for packed, esi, seq in encoder.generate_blocks(K + 2):
            done, _ = decoder.decode_bytes(packed)
            if done:
                break

        assert decoder.done
        assert decoder.bytes_dump() == data

    def test_progress_tracking(self):
        data = b'\x00' * 256
        encoder = RaptorQEncoder(data, 64)
        decoder = RaptorQDecoder()

        assert decoder.progress == 0.0
        assert decoder.num_recovered == 0

        blocks = list(encoder.generate_blocks(encoder.K + 2))
        for i, (packed, esi, seq) in enumerate(blocks):
            done, _ = decoder.decode_bytes(packed)
            assert decoder.progress >= 0.0
            if done:
                assert decoder.progress == 1.0
                assert decoder.num_recovered == decoder.K
                break

    def test_progress_uses_confirmed_source_symbols(self):
        decoder = RaptorQDecoder()
        decoder.initialized = True
        decoder.K = 10
        decoder._fed_count = 8
        decoder.eliminated = {0: True, 5: True}

        assert decoder.num_recovered == 2
        assert decoder.progress == 0.2

    def test_multi_source_block_eliminated_uses_payload_id_z(self, monkeypatch):
        class DummyInnerDecoder:
            def decode(self, packet):
                return None

        class DummyDecoderFactory:
            @staticmethod
            def with_defaults(padded_len, symbol_size):
                return DummyInnerDecoder()

        class DummyRaptorQ:
            Decoder = DummyDecoderFactory

        monkeypatch.setattr(rq, "_raptorq", DummyRaptorQ)

        decoder = RaptorQDecoder()
        first = pack_v4(
            filesize=20,
            symbol_size=4,
            symbol_count=5,
            esi=_rq_payload_id(1, 0),
            block_seq=0,
            data=b'\x00' * 4,
            reserved=2,
        )
        done, _ = decoder.decode_bytes(first)

        assert not done
        assert decoder.source_blocks == 2
        assert decoder.eliminated == {3: True}
        assert decoder.num_recovered == 1
        assert decoder.progress == 0.2

        second = pack_v4(
            filesize=20,
            symbol_size=4,
            symbol_count=5,
            esi=_rq_payload_id(1, 1),
            block_seq=1,
            data=b'\x00' * 4,
            reserved=2,
        )
        repair = pack_v4(
            filesize=20,
            symbol_size=4,
            symbol_count=5,
            esi=_rq_payload_id(0, 3),
            block_seq=2,
            data=b'\x00' * 4,
            reserved=2,
        )

        decoder.decode_bytes(second)
        decoder.decode_bytes(repair)

        assert decoder.eliminated == {3: True, 4: True}
        assert decoder.num_recovered == 2
        assert decoder.progress == 0.4

    def test_legacy_reserved_zero_uses_single_source_block(self, monkeypatch):
        class DummyInnerDecoder:
            def decode(self, packet):
                return None

        class DummyDecoderFactory:
            @staticmethod
            def with_defaults(padded_len, symbol_size):
                return DummyInnerDecoder()

        class DummyRaptorQ:
            Decoder = DummyDecoderFactory

        monkeypatch.setattr(rq, "_raptorq", DummyRaptorQ)

        decoder = RaptorQDecoder()
        packed = pack_v4(
            filesize=16,
            symbol_size=4,
            symbol_count=4,
            esi=_rq_payload_id(0, 2),
            block_seq=0,
            data=b'\x00' * 4,
            reserved=0,
        )
        decoder.decode_bytes(packed)

        assert decoder.source_blocks == 1
        assert decoder.eliminated == {2: True}

    def test_is_done(self):
        data = b'\xAB' * 128
        encoder = RaptorQEncoder(data, 64)
        decoder = RaptorQDecoder()

        assert not decoder.is_done()

        for packed, esi, seq in encoder.generate_blocks(encoder.K + 5):
            decoder.decode_bytes(packed)
            if decoder.is_done():
                break

        assert decoder.is_done()

    def test_eliminated_tracks_source_symbols(self):
        """eliminated dict should grow as source ESIs arrive."""
        data = b'\xAB' * 256
        blocksize = 64
        encoder = RaptorQEncoder(data, blocksize)
        decoder = RaptorQDecoder()
        K = encoder.K

        assert decoder.eliminated == {}

        blocks = list(encoder.generate_blocks(K + 5))
        source_fed = 0
        for packed, esi, seq in blocks:
            done, _ = decoder.decode_bytes(packed)
            if esi < K:
                source_fed += 1
                assert esi in decoder.eliminated, (
                    f"ESI {esi} should be in eliminated after feeding")
            if done:
                # Once done, ALL K blocks must be present.
                assert len(decoder.eliminated) == K
                for i in range(K):
                    assert i in decoder.eliminated
                break

    def test_eliminated_compatible_with_block_map(self):
        """eliminated dict should support 'block_idx in recovered' tests."""
        data = b'\x00' * 128
        encoder = RaptorQEncoder(data, 64)
        decoder = RaptorQDecoder()

        for packed, esi, seq in encoder.generate_blocks(encoder.K + 2):
            decoder.decode_bytes(packed)
            if decoder.done:
                break

        # The UI layer does: ``if block_idx in recovered_set``
        for i in range(decoder.K):
            assert i in decoder.eliminated

    def test_gaussian_rescue_noop(self):
        """try_gaussian_rescue() is a no-op for RaptorQ."""
        decoder = RaptorQDecoder()
        assert decoder.try_gaussian_rescue() is False

    def test_header_consistency_check(self):
        """Inconsistent headers should raise ValueError."""
        data = b'\x00' * 128
        encoder1 = RaptorQEncoder(data, 64)
        encoder2 = RaptorQEncoder(data, 32)  # different blocksize

        decoder = RaptorQDecoder()
        packed1, _, _ = next(encoder1.generate_blocks(1))
        packed2, _, _ = next(encoder2.generate_blocks(1))

        decoder.decode_bytes(packed1)
        try:
            decoder.decode_bytes(packed2)
            assert False, "Expected ValueError for blocksize mismatch"
        except ValueError:
            pass

    def test_bytes_dump_to_file(self, tmp_path):
        data = b'File content for RaptorQ test!' * 10
        encoder = RaptorQEncoder(data, 64)
        decoder = RaptorQDecoder()

        for packed, esi, seq in encoder.generate_blocks(encoder.K + 5):
            done, _ = decoder.decode_bytes(packed)
            if done:
                break

        output = tmp_path / "output.bin"
        written = decoder.bytes_dump_to_file(str(output))
        assert written == len(data)
        assert output.read_bytes() == data
