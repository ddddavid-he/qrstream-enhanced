"""End-to-end roundtrip test: encode → decode using pure data (no video)."""

import os
import zlib
from math import ceil

import pytest

from qrstream.lt_codec import PRNG, DEFAULT_C, DEFAULT_DELTA
from qrstream.protocol import pack_v2, auto_blocksize, unpack, V2Header
from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder


class TestDataRoundtrip:
    """Test encoding and decoding without video — pure LT fountain code roundtrip."""

    def _roundtrip(self, data: bytes, overhead: float = 3.0,
                   compress: bool = False):
        """Helper: encode data into LT blocks and decode them back."""
        if compress:
            payload = zlib.compress(data)
        else:
            payload = data

        filesize = len(payload)
        blocksize = 64
        K = ceil(filesize / blocksize)
        num_blocks = int(K * overhead)

        encoder = LTEncoder(payload, blocksize, compressed=compress)
        decoder = LTDecoder()

        for packed, seed, seq in encoder.generate_blocks(num_blocks):
            try:
                done, _ = decoder.decode_bytes(packed)
                if done:
                    result = decoder.bytes_dump()
                    if compress:
                        return result  # already decompressed by bytes_dump
                    return result
            except ValueError:
                # CRC error — shouldn't happen in clean roundtrip
                raise

        # If not done, return None
        return None

    def test_small_data(self):
        data = b"Hello, QRStream!"
        result = self._roundtrip(data)
        assert result == data

    def test_exact_blocksize_multiple(self):
        data = b'\xAB' * 256  # 4 blocks of 64
        result = self._roundtrip(data)
        assert result == data

    def test_non_aligned_data(self):
        data = b'\xCD' * 100  # 100 bytes, not aligned to 64
        result = self._roundtrip(data)
        assert result == data

    def test_larger_data(self):
        data = os.urandom(1024)
        result = self._roundtrip(data, overhead=3.0)
        assert result == data

    def test_compressed_roundtrip(self):
        # Highly compressible data
        data = b'A' * 500
        result = self._roundtrip(data, compress=True)
        assert result == data

    def test_binary_data(self):
        data = bytes(range(256)) * 2  # 512 bytes of all byte values
        result = self._roundtrip(data)
        assert result == data


class TestV2Protocol:
    """Test V2 protocol pack/unpack with encoder-generated blocks."""

    def test_encoder_produces_valid_v2(self):
        data = b"test data for protocol"
        blocksize = 32
        encoder = LTEncoder(data, blocksize)

        for packed, seed, seq in encoder.generate_blocks(5):
            header, block_data = unpack(packed)
            assert isinstance(header, V2Header)
            assert header.version == 0x02
            assert header.blocksize == blocksize
            assert header.seed == seed
            assert header.block_seq == seq
            assert len(block_data) == blocksize

    def test_auto_blocksize_reasonable(self):
        for size in [100, 1000, 10000, 100000]:
            bs = auto_blocksize(size)
            K = ceil(size / bs)
            assert K <= 65535
            assert bs > 0


class TestDecoderProgress:
    def test_progress_starts_at_zero(self):
        decoder = LTDecoder()
        assert decoder.progress == 0.0

    def test_progress_increases(self):
        data = b'\x00' * 256
        blocksize = 64
        encoder = LTEncoder(data, blocksize)
        decoder = LTDecoder()

        prev_progress = 0.0
        for packed, seed, seq in encoder.generate_blocks(20):
            try:
                done, _ = decoder.decode_bytes(packed)
                assert decoder.progress >= prev_progress
                prev_progress = decoder.progress
                if done:
                    assert decoder.progress == 1.0
                    break
            except ValueError:
                pass
