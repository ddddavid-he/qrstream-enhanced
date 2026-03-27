"""Tests for the V1/V2 protocol serialization and deserialization."""

import struct
import zlib
import pytest

from qrstream.protocol import (
    pack_v2, unpack, auto_blocksize,
    V1Header, V2Header, V1_HEADER_SIZE, V2_HEADER_SIZE,
)


class TestV2PackUnpack:
    def test_roundtrip(self):
        data = b'\xAB' * 100
        packed = pack_v2(
            filesize=1000, blocksize=100, block_count=10,
            seed=42, block_seq=0, data=data, compressed=False,
        )
        header, unpacked_data = unpack(packed)
        assert isinstance(header, V2Header)
        assert header.version == 0x02
        assert header.compressed is False
        assert header.filesize == 1000
        assert header.blocksize == 100
        assert header.block_count == 10
        assert header.seed == 42
        assert header.block_seq == 0
        assert unpacked_data == data

    def test_compressed_flag(self):
        data = b'\x00' * 50
        packed = pack_v2(
            filesize=500, blocksize=50, block_count=10,
            seed=1, block_seq=1, data=data, compressed=True,
        )
        header, _ = unpack(packed)
        assert header.compressed is True

    def test_crc_validation(self):
        data = b'\xFF' * 80
        packed = pack_v2(
            filesize=800, blocksize=80, block_count=10,
            seed=99, block_seq=5, data=data,
        )
        # Corrupt one data byte
        corrupted = bytearray(packed)
        corrupted[-1] ^= 0xFF
        with pytest.raises(ValueError, match="CRC32 mismatch"):
            unpack(bytes(corrupted))

    def test_crc_header_corruption(self):
        data = b'\x01' * 60
        packed = pack_v2(
            filesize=600, blocksize=60, block_count=10,
            seed=7, block_seq=3, data=data,
        )
        # Corrupt a header byte (flags)
        corrupted = bytearray(packed)
        corrupted[1] ^= 0xFF
        with pytest.raises(ValueError, match="CRC32 mismatch"):
            unpack(bytes(corrupted))


class TestV1Compat:
    def test_v1_unpack(self):
        # Build a V1 packet manually: magic=0x01, filesize, blocksize, seed
        magic = 0x01
        filesize = 256
        blocksize = 64
        seed = 12345
        data = b'\xCC' * 64
        raw = struct.pack('!BIII', magic, filesize, blocksize, seed) + data
        header, unpacked_data = unpack(raw)
        assert isinstance(header, V1Header)
        assert header.compressed is True  # bit0 set
        assert header.filesize == 256
        assert header.blocksize == 64
        assert header.seed == 12345
        assert unpacked_data == data

    def test_v1_uncompressed(self):
        magic = 0x00
        raw = struct.pack('!BIII', magic, 100, 50, 999) + b'\x00' * 50
        header, _ = unpack(raw)
        assert isinstance(header, V1Header)
        assert header.compressed is False


class TestAutoBlocksize:
    def test_returns_positive(self):
        for ec in range(4):
            bs = auto_blocksize(10000, ec)
            assert bs > 0

    def test_larger_ec_smaller_block(self):
        bs_l = auto_blocksize(10000, 0)
        bs_h = auto_blocksize(10000, 3)
        assert bs_l >= bs_h

    def test_large_file_block_count_fits_uint16(self):
        # 100MB file
        bs = auto_blocksize(100_000_000, 1)
        from math import ceil
        k = ceil(100_000_000 / bs)
        assert k <= 65535

    def test_too_short_block(self):
        with pytest.raises(ValueError):
            unpack(b'\x02\x00')
