"""End-to-end roundtrip tests for RaptorQ codec (no video I/O)."""

import random
import zlib
from math import ceil

from qrstream.raptorq_codec import RaptorQEncoder, RaptorQDecoder
from qrstream.protocol import V4Header, unpack


class TestRaptorQRoundtrip:
    """Encode and decode without video -- pure RaptorQ fountain code roundtrip."""

    def _roundtrip(self, data: bytes, overhead: float = 1.2,
                   compress: bool = False):
        if compress:
            payload = zlib.compress(data)
        else:
            payload = data

        filesize = len(payload)
        blocksize = 64
        K = ceil(filesize / blocksize)
        num_blocks = max(K, int(K * overhead))

        encoder = RaptorQEncoder(
            payload,
            blocksize,
            compressed=compress,
        )
        decoder = RaptorQDecoder()

        for packed, esi, seq in encoder.generate_blocks(num_blocks):
            done, _ = decoder.decode_bytes(packed)
            if done:
                return decoder.bytes_dump()

        return None

    def test_small_data(self):
        data = b"Hello, QRStream with RaptorQ!"
        result = self._roundtrip(data)
        assert result == data

    def test_exact_blocksize_multiple(self):
        data = b'\xAB' * 256
        result = self._roundtrip(data)
        assert result == data

    def test_non_aligned_data(self):
        data = b'\xCD' * 100
        result = self._roundtrip(data)
        assert result == data

    def test_larger_data(self):
        data = random.Random(0x10241024).randbytes(1024)
        result = self._roundtrip(data, overhead=1.2)
        assert result == data

    def test_compressed_roundtrip(self):
        data = b'A' * 500
        result = self._roundtrip(data, compress=True)
        assert result == data

    def test_binary_data(self):
        data = bytes(range(256)) * 2
        result = self._roundtrip(data)
        assert result == data

    def test_minimal_overhead(self):
        """RaptorQ should decode with very little overhead."""
        data = random.Random(42).randbytes(2048)
        # RaptorQ should decode with essentially K packets.
        result = self._roundtrip(data, overhead=1.05)
        assert result == data

    def test_packet_loss_recovery(self):
        """Simulate packet loss and verify recovery with repair symbols."""
        data = random.Random(123).randbytes(1024)
        blocksize = 64
        K = ceil(len(data) / blocksize)
        overhead = 1.5  # 50% extra packets

        encoder = RaptorQEncoder(data, blocksize)
        all_blocks = list(encoder.generate_blocks(int(K * overhead)))

        # Drop 30% of blocks randomly
        rng = random.Random(456)
        drop_count = int(len(all_blocks) * 0.30)
        indices = list(range(len(all_blocks)))
        rng.shuffle(indices)
        kept = sorted(indices[drop_count:])
        kept_blocks = [all_blocks[i] for i in kept]

        decoder = RaptorQDecoder()
        for packed, esi, seq in kept_blocks:
            done, _ = decoder.decode_bytes(packed)
            if done:
                assert decoder.bytes_dump() == data
                return

        # Should have decoded
        assert False, f"Failed to decode with {len(kept_blocks)} of {len(all_blocks)} blocks"


class TestRaptorQProtocolVersions:
    """V4 protocol correctness in roundtrip context."""

    def test_encoder_produces_v4_blocks(self):
        data = b"test data for protocol"
        blocksize = 32
        encoder = RaptorQEncoder(data, blocksize)

        for packed, esi, seq in encoder.generate_blocks(5):
            header, block_data = unpack(packed)
            assert isinstance(header, V4Header)
            assert header.version == 0x04
            assert header.symbol_size == blocksize
            assert header.esi == esi
            assert header.block_seq == seq
            assert len(block_data) == blocksize


class TestRaptorQDecoderProgress:
    """Progress reporting for RaptorQ decoder."""

    def test_progress_starts_at_zero(self):
        decoder = RaptorQDecoder()
        assert decoder.progress == 0.0

    def test_progress_increases(self):
        data = b'\x00' * 256
        blocksize = 64
        encoder = RaptorQEncoder(data, blocksize)
        decoder = RaptorQDecoder()

        prev_progress = 0.0
        for packed, esi, seq in encoder.generate_blocks(encoder.K + 5):
            done, _ = decoder.decode_bytes(packed)
            assert decoder.progress >= prev_progress
            prev_progress = decoder.progress
            if done:
                assert decoder.progress == 1.0
                break


class TestMixedV3V4Detection:
    """Verify that decode_blocks auto-detects V3 vs V4."""

    def test_v4_blocks_decode_via_decode_blocks(self):
        from qrstream.decoder import decode_blocks

        data = b"Testing auto-detection of V4 blocks!" * 5
        blocksize = 64
        encoder = RaptorQEncoder(data, blocksize)
        K = encoder.K

        blocks_raw = [packed for packed, _, _ in
                      encoder.generate_blocks(K + 5)]

        result = decode_blocks(blocks_raw)
        assert result == data

    def test_v3_blocks_still_decode_via_decode_blocks(self):
        from qrstream.encoder import LTEncoder
        from qrstream.decoder import decode_blocks

        data = b"Testing LT backward compatibility!" * 5
        blocksize = 64
        encoder = LTEncoder(data, blocksize)
        K = encoder.K

        blocks_raw = [packed for packed, _, _ in
                      encoder.generate_blocks(int(K * 3.0))]

        result = decode_blocks(blocks_raw)
        assert result == data
