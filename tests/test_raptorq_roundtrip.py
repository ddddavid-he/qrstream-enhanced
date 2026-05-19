"""End-to-end roundtrip tests for RaptorQ codec (no video I/O)."""

import hashlib
import random
import zlib
from math import ceil

import pytest

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


class TestRaptorQLargeDataInterleave:
    """Regression tests for sub-block interleaving with large K.

    The raptorq library (RFC 6330 Section 5.6) uses sub-block column
    interleaving when the transfer length exceeds an internal threshold.
    Source symbols from the library are NOT simple linear slices of the
    input data in this regime.  These tests verify that encode→decode
    roundtrips remain correct for parameters that trigger this behaviour.

    See: https://github.com/ddddavid-he/qrstream-enhanced — bug report
    "RaptorQ large data silent data corruption"
    """

    def _roundtrip_sha256(self, data: bytes, blocksize: int,
                          overhead: float = 1.0) -> bytes:
        """Encode→decode and verify SHA256 integrity."""
        expected_sha = hashlib.sha256(data).hexdigest()

        encoder = RaptorQEncoder(data, blocksize)
        num_blocks = max(encoder.K, int(encoder.K * overhead))

        decoder = RaptorQDecoder()
        for packed, esi, seq in encoder.generate_blocks(num_blocks):
            done, _ = decoder.decode_bytes(packed)
            if done:
                result = decoder.bytes_dump()
                actual_sha = hashlib.sha256(result).hexdigest()
                assert actual_sha == expected_sha, (
                    f"SHA256 mismatch: blocksize={blocksize}, "
                    f"data_len={len(data)}, K={encoder.K}")
                return result

        raise AssertionError(
            f"Decode did not converge: blocksize={blocksize}, "
            f"data_len={len(data)}, K={encoder.K}, fed={num_blocks}")

    @pytest.mark.parametrize("blocksize", [256, 512, 936])
    def test_large_k_subblock_interleave_roundtrip(self, blocksize):
        """10MB roundtrip with blocksizes that trigger sub-block interleaving.

        This is the exact scenario described in the bug report where
        K ≥ ~11131 with blocksize=936 causes the raptorq library to
        apply column interleaving.
        """
        data_size = 10 * 1024 * 1024  # 10MB
        # Use deterministic pseudo-random data so mismatches are reproducible.
        rng = random.Random(0xDEADBEEF)
        data = rng.randbytes(data_size)

        result = self._roundtrip_sha256(data, blocksize)
        assert result == data

    def test_blocksize_936_1mb_roundtrip(self):
        """1MB with blocksize=936 — triggers interleaving at lower threshold."""
        data_size = 1 * 1024 * 1024
        rng = random.Random(42)
        data = rng.randbytes(data_size)

        result = self._roundtrip_sha256(data, blocksize=936)
        assert result == data

    def test_blocksize_936_boundary_k(self):
        """Test near the exact K boundary where interleaving activates.

        For blocksize=936, the raptorq library activates sub-block
        interleaving at K=11131 (data_size=10,418,616 bytes).
        """
        # Just above threshold: K=11131 triggers interleaving
        data_size = 11131 * 936
        rng = random.Random(0xCAFEBABE)
        data = rng.randbytes(data_size)

        result = self._roundtrip_sha256(data, blocksize=936)
        assert result == data

    def test_non_aligned_large_data(self):
        """Large data not aligned to blocksize boundary."""
        # 10MB + 473 bytes — not a multiple of 936
        data_size = 10 * 1024 * 1024 + 473
        rng = random.Random(0xBAADF00D)
        data = rng.randbytes(data_size)

        result = self._roundtrip_sha256(data, blocksize=936)
        assert result == data

    def test_large_data_with_packet_loss(self):
        """Large data roundtrip with simulated packet loss."""
        data_size = 10 * 1024 * 1024
        blocksize = 936
        rng = random.Random(0x12345678)
        data = rng.randbytes(data_size)

        encoder = RaptorQEncoder(data, blocksize)
        # Request 20% overhead to survive packet loss
        all_blocks = list(encoder.generate_blocks(int(encoder.K * 1.2)))

        # Drop 15% of packets randomly
        drop_rng = random.Random(999)
        drop_count = int(len(all_blocks) * 0.15)
        indices = list(range(len(all_blocks)))
        drop_rng.shuffle(indices)
        kept = sorted(indices[drop_count:])
        kept_blocks = [all_blocks[i] for i in kept]

        decoder = RaptorQDecoder()
        for packed, esi, seq in kept_blocks:
            done, _ = decoder.decode_bytes(packed)
            if done:
                result = decoder.bytes_dump()
                assert hashlib.sha256(result).hexdigest() == \
                    hashlib.sha256(data).hexdigest()
                assert result == data
                return

        raise AssertionError(
            f"Failed to decode with {len(kept_blocks)} of "
            f"{len(all_blocks)} blocks")
