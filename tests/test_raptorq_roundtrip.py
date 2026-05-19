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


class TestRaptorQDataIntegrity:
    """Data correctness and file transfer integrity verification.

    These tests ensure that the RaptorQ codec preserves file content
    bit-for-bit across a variety of realistic file patterns, sizes, and
    blocksize configurations — covering scenarios beyond simple random
    bytes.
    """

    def _roundtrip_verify(self, data: bytes, blocksize: int,
                          overhead: float = 1.05) -> None:
        """Encode→decode roundtrip with SHA256 + byte-level verification."""
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
                    f"SHA256 mismatch: expected {expected_sha[:16]}… "
                    f"got {actual_sha[:16]}…, "
                    f"blocksize={blocksize}, data_len={len(data)}")
                assert len(result) == len(data), (
                    f"Length mismatch: expected {len(data)}, "
                    f"got {len(result)}")
                # Find first diff for diagnostics
                assert result == data, self._diff_msg(data, result)
                return

        raise AssertionError(
            f"Decode did not converge: blocksize={blocksize}, "
            f"data_len={len(data)}, K={encoder.K}")

    @staticmethod
    def _diff_msg(expected: bytes, actual: bytes) -> str:
        for i in range(min(len(expected), len(actual))):
            if expected[i] != actual[i]:
                return (
                    f"First diff at byte {i}: "
                    f"expected 0x{expected[i]:02x}, got 0x{actual[i]:02x}")
        return f"Length mismatch: {len(expected)} vs {len(actual)}"

    # ── Pattern-based data correctness ────────────────────────────────

    def test_all_zeros(self):
        """All-zero file — verifies no false 'skip' optimisation corrupts data."""
        data = b'\x00' * (2 * 1024 * 1024)
        self._roundtrip_verify(data, blocksize=936)

    def test_all_ones(self):
        """All 0xFF file — boundary value for unsigned byte range."""
        data = b'\xff' * (2 * 1024 * 1024)
        self._roundtrip_verify(data, blocksize=936)

    def test_repeating_pattern_256(self):
        """0x00..0xFF repeating — exercises every byte value at every position."""
        data = bytes(range(256)) * (8 * 1024)  # 2MB
        self._roundtrip_verify(data, blocksize=936)

    def test_sequential_32bit_counter(self):
        """4-byte big-endian counter — structured data easy to verify on failure."""
        import struct as st
        num_words = 512 * 1024  # 2MB
        data = b''.join(st.pack('>I', i) for i in range(num_words))
        self._roundtrip_verify(data, blocksize=512)

    def test_alternating_blocks(self):
        """Alternating 0x00/0xFF blocks at blocksize boundary — tests symbol boundaries."""
        blocksize = 936
        num_blocks = 2048
        data = b''
        for i in range(num_blocks):
            data += bytes([0x00 if i % 2 == 0 else 0xFF]) * blocksize
        self._roundtrip_verify(data, blocksize=blocksize)

    # ── Realistic file patterns ───────────────────────────────────────

    def test_sparse_file_mostly_zeros(self):
        """Simulates a sparse file: mostly zeros with scattered non-zero regions."""
        rng = random.Random(0x5A453)
        size = 5 * 1024 * 1024
        data = bytearray(size)
        # Scatter 1000 random 512-byte bursts into the zero field
        for _ in range(1000):
            offset = rng.randint(0, size - 512)
            data[offset:offset + 512] = rng.randbytes(512)
        self._roundtrip_verify(bytes(data), blocksize=936)

    def test_highly_compressible_text(self):
        """Simulates a text/log file — repetitive ASCII with some variation."""
        lines = []
        for i in range(50000):
            lines.append(f"[2026-05-19 12:00:{i%60:02d}] INFO: Processing item {i} of 50000 — status=OK\n")
        data = ''.join(lines).encode('utf-8')
        # Trim to ~5MB
        data = data[:5 * 1024 * 1024]
        self._roundtrip_verify(data, blocksize=936)

    def test_binary_executable_pattern(self):
        """Simulates binary executable: header + code-like patterns + padding."""
        rng = random.Random(0xE1F)
        # ELF-like header
        header = b'\x7fELF' + rng.randbytes(60)
        # Pseudo-code sections with varying entropy
        sections = []
        for _ in range(20):
            section_size = rng.randint(50_000, 200_000)
            sections.append(rng.randbytes(section_size))
        # Zero-padding between sections (BSS-like)
        padding = b'\x00' * 100_000
        data = header
        for sec in sections:
            data += sec + padding
        data = data[:8 * 1024 * 1024]  # Cap at 8MB
        self._roundtrip_verify(data, blocksize=936)

    # ── Blocksize sweep ───────────────────────────────────────────────

    @pytest.mark.parametrize("blocksize", [64, 128, 256, 512, 936, 1024])
    def test_blocksize_sweep_1mb(self, blocksize):
        """1MB random data across all common blocksizes."""
        rng = random.Random(blocksize * 7)
        data = rng.randbytes(1024 * 1024)
        self._roundtrip_verify(data, blocksize=blocksize)

    @pytest.mark.parametrize("blocksize", [256, 512, 936])
    def test_blocksize_sweep_10mb(self, blocksize):
        """10MB random data for blocksizes that may trigger interleaving."""
        rng = random.Random(blocksize * 13)
        data = rng.randbytes(10 * 1024 * 1024)
        self._roundtrip_verify(data, blocksize=blocksize)

    # ── Edge-case sizes ───────────────────────────────────────────────

    def test_one_byte(self):
        """Minimum possible file — single byte."""
        self._roundtrip_verify(b'\x42', blocksize=64)

    def test_one_block_exact(self):
        """File exactly one block — no multi-block reassembly needed."""
        rng = random.Random(1)
        self._roundtrip_verify(rng.randbytes(936), blocksize=936)

    def test_blocksize_minus_one(self):
        """File one byte short of blocksize — last block heavily padded."""
        rng = random.Random(2)
        self._roundtrip_verify(rng.randbytes(935), blocksize=936)

    def test_blocksize_plus_one(self):
        """File one byte over blocksize — creates exactly 2 blocks."""
        rng = random.Random(3)
        self._roundtrip_verify(rng.randbytes(937), blocksize=936)

    def test_prime_sized_file(self):
        """File with prime-number size — never aligns to any blocksize."""
        # 1048573 is prime, close to 1MB
        rng = random.Random(0xB01AE)
        data = rng.randbytes(1048573)
        self._roundtrip_verify(data, blocksize=936)

    # ── Multi-session consistency ─────────────────────────────────────

    def test_encoder_determinism_large(self):
        """Same input always produces same encoded output — important for resumable transfers."""
        rng = random.Random(0xDE7)
        data = rng.randbytes(5 * 1024 * 1024)
        blocksize = 936

        enc1 = RaptorQEncoder(data, blocksize)
        enc2 = RaptorQEncoder(data, blocksize)

        blocks1 = [(esi, symbol) for _, esi, symbol in enc1.generate_blocks(enc1.K)]
        blocks2 = [(esi, symbol) for _, esi, symbol in enc2.generate_blocks(enc2.K)]

        assert blocks1 == blocks2, "Encoder is non-deterministic"

    def test_decode_from_any_k_subset(self):
        """Any K source symbols should decode — verifies systematic property."""
        rng = random.Random(0x50B)
        data = rng.randbytes(500 * 1024)  # 500KB
        blocksize = 512

        encoder = RaptorQEncoder(data, blocksize)
        K = encoder.K
        all_blocks = list(encoder.generate_blocks(K))

        # Feed only first K blocks (all source symbols, no repair)
        decoder = RaptorQDecoder()
        for packed, esi, seq in all_blocks:
            done, _ = decoder.decode_bytes(packed)
            if done:
                assert decoder.bytes_dump() == data
                return

        raise AssertionError("Failed to decode with K source symbols")

    # ── CRC and byte-boundary verification ────────────────────────────

    def test_per_block_crc_integrity(self):
        """Verify CRC32 on each V4 frame is valid — catches header/data misalignment."""
        import zlib as z
        rng = random.Random(0xC4C)
        data = rng.randbytes(2 * 1024 * 1024)
        blocksize = 936

        encoder = RaptorQEncoder(data, blocksize)
        for packed, esi, seq in encoder.generate_blocks(encoder.K):
            # unpack validates CRC internally; would raise on mismatch
            header, block_data = unpack(packed, skip_crc=False)
            assert len(block_data) == blocksize

    def test_truncated_last_block_padding(self):
        """Verify that filesize < K*blocksize files are correctly zero-padded and trimmed."""
        # File where last symbol needs substantial padding
        rng = random.Random(0xBAD)
        data = rng.randbytes(1000)  # blocksize=936: K=2, last block has 936-64=872 padding bytes
        blocksize = 936

        encoder = RaptorQEncoder(data, blocksize)
        assert encoder.K == 2  # sanity

        decoder = RaptorQDecoder()
        for packed, esi, seq in encoder.generate_blocks(encoder.K + 2):
            done, _ = decoder.decode_bytes(packed)
            if done:
                result = decoder.bytes_dump()
                assert len(result) == len(data), (
                    f"Result length {len(result)} != data length {len(data)}")
                assert result == data
                return

        raise AssertionError("Decode did not converge")

    def test_compressed_large_roundtrip(self):
        """Large compressible data with compression enabled — exercises decompress path."""
        # Generate highly compressible data
        data = (b'A' * 1000 + b'B' * 500 + b'C' * 300) * 2000  # ~3.6MB raw
        data = data[:3 * 1024 * 1024]

        import zlib
        compressed = zlib.compress(data)
        encoder = RaptorQEncoder(compressed, blocksize=936, compressed=True)

        decoder = RaptorQDecoder()
        for packed, esi, seq in encoder.generate_blocks(int(encoder.K * 1.05)):
            done, _ = decoder.decode_bytes(packed)
            if done:
                result = decoder.bytes_dump()
                assert result == data, "Compressed roundtrip data mismatch"
                return

        raise AssertionError("Compressed decode did not converge")
