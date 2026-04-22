"""End-to-end roundtrip tests without video I/O."""

import os
import zlib
from math import ceil

from qrstream.protocol import V3Header, auto_blocksize, unpack
from qrstream.encoder import LTEncoder, MmapDataSource, _load_payload
from qrstream.decoder import LTDecoder


class TestDataRoundtrip:
    """Test encoding and decoding without video — pure LT fountain code roundtrip."""

    def _roundtrip(self, data: bytes, overhead: float = 3.0,
                   compress: bool = False):
        if compress:
            payload = zlib.compress(data)
        else:
            payload = data

        filesize = len(payload)
        blocksize = 64
        K = ceil(filesize / blocksize)
        num_blocks = int(K * overhead)

        encoder = LTEncoder(
            payload,
            blocksize,
            compressed=compress,
        )
        decoder = LTDecoder()

        for packed, seed, seq in encoder.generate_blocks(num_blocks):
            done, _ = decoder.decode_bytes(packed)
            if done:
                return decoder.bytes_dump()

        return None

    def test_small_data(self):
        data = b"Hello, QRStream!"
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
        data = os.urandom(1024)
        result = self._roundtrip(data, overhead=3.0)
        assert result == data

    def test_compressed_roundtrip(self):
        data = b'A' * 500
        result = self._roundtrip(data, compress=True)
        assert result == data

    def test_binary_data(self):
        data = bytes(range(256)) * 2
        result = self._roundtrip(data)
        assert result == data


class TestProtocolVersions:
    def test_encoder_produces_valid_v3_by_default(self):
        data = b"test data for protocol"
        blocksize = 32
        encoder = LTEncoder(data, blocksize)

        for packed, seed, seq in encoder.generate_blocks(5):
            header, block_data = unpack(packed)
            assert isinstance(header, V3Header)
            assert header.version == 0x03
            assert header.blocksize == blocksize
            assert header.seed == seed
            assert header.block_seq == seq
            assert len(block_data) == blocksize

    def test_auto_blocksize_reasonable(self):
        for size in [100, 1000, 10000, 100000]:
            bs = auto_blocksize(size)
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
            done, _ = decoder.decode_bytes(packed)
            assert decoder.progress >= prev_progress
            prev_progress = decoder.progress
            if done:
                assert decoder.progress == 1.0
                break


class TestStreamingPaths:
    def test_load_payload_uses_mmap_for_large_uncompressed_files(self, tmp_path):
        input_path = tmp_path / "large.bin"
        input_path.write_bytes(b'A' * (10 * 1024 * 1024 + 1))

        payload, compressed, used_mmap, raw_size = _load_payload(
            str(input_path),
            compress=False,
            verbose=False,
        )
        try:
            assert isinstance(payload, MmapDataSource)
            assert compressed is False
            assert used_mmap is True
            assert raw_size == len(payload)
            assert payload[:8] == b'A' * 8
        finally:
            payload.close()

    def test_load_payload_disables_compression_for_large_input(self, tmp_path):
        input_path = tmp_path / "large.bin"
        input_path.write_bytes(b'B' * (10 * 1024 * 1024 + 1))

        payload, compressed, used_mmap, _ = _load_payload(
            str(input_path),
            compress=True,
            verbose=False,
        )
        try:
            assert compressed is False
            assert used_mmap is True
            assert isinstance(payload, MmapDataSource)
        finally:
            payload.close()
