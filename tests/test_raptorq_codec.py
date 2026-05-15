"""Unit tests for RaptorQEncoder and RaptorQDecoder."""

import struct
from math import ceil

from qrstream.raptorq_codec import RaptorQEncoder, RaptorQDecoder
from qrstream.protocol import V4_VERSION, unpack


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
