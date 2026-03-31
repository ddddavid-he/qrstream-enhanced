"""Tests for the V2/V3 protocol serialization and deserialization."""

from math import ceil

import pytest

from qrstream.protocol import (
    V2Header,
    V2_HEADER_SIZE,
    V3Header,
    V3_BLOCK_OVERHEAD,
    auto_blocksize,
    pack_v2,
    pack_v3,
    unpack,
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

    def test_binary_qr_flag(self):
        data = b'\x00' * 50
        packed = pack_v2(
            filesize=500, blocksize=50, block_count=10,
            seed=1, block_seq=1, data=data, binary_qr=True,
        )
        header, _ = unpack(packed)
        assert header.binary_qr is True

    def test_crc_validation(self):
        data = b'\xFF' * 80
        packed = pack_v2(
            filesize=800, blocksize=80, block_count=10,
            seed=99, block_seq=5, data=data,
        )
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
        corrupted = bytearray(packed)
        corrupted[1] ^= 0xFF
        with pytest.raises(ValueError, match="CRC32 mismatch"):
            unpack(bytes(corrupted))

    def test_skip_crc(self):
        data = b'\xFF' * 80
        packed = pack_v2(
            filesize=800, blocksize=80, block_count=10,
            seed=99, block_seq=5, data=data,
        )
        corrupted = bytearray(packed)
        corrupted[-1] ^= 0xFF
        header, _ = unpack(bytes(corrupted), skip_crc=True)
        assert header.seed == 99


class TestV3PackUnpack:
    def test_roundtrip(self):
        data = b'\xCD' * 100
        packed = pack_v3(
            filesize=1000, blocksize=100, block_count=10,
            seed=123, block_seq=9, data=data, compressed=False,
            binary_qr=True,
        )
        header, unpacked_data = unpack(packed)
        assert isinstance(header, V3Header)
        assert header.version == 0x03
        assert header.compressed is False
        assert header.binary_qr is True
        assert header.filesize == 1000
        assert header.blocksize == 100
        assert header.block_count == 10
        assert header.seed == 123
        assert header.block_seq == 9
        assert unpacked_data == data

    def test_crc_validation(self):
        data = b'\xEE' * 80
        packed = pack_v3(
            filesize=800, blocksize=80, block_count=10,
            seed=77, block_seq=5, data=data,
        )
        corrupted = bytearray(packed)
        corrupted[-1] ^= 0xFF
        with pytest.raises(ValueError, match="CRC32 mismatch"):
            unpack(bytes(corrupted))

    def test_skip_crc(self):
        data = b'\xDD' * 80
        packed = pack_v3(
            filesize=800, blocksize=80, block_count=10,
            seed=88, block_seq=6, data=data,
        )
        corrupted = bytearray(packed)
        corrupted[-1] ^= 0xFF
        header, _ = unpack(bytes(corrupted), skip_crc=True)
        assert header.seed == 88

    def test_truncated_data_rejected(self):
        packed = pack_v3(
            filesize=64, blocksize=64, block_count=1,
            seed=1, block_seq=0, data=b'A' * 64,
        )
        with pytest.raises(ValueError, match="data length mismatch"):
            unpack(packed[:-1])


class TestAutoBlocksize:
    def test_returns_positive(self):
        for ec in range(4):
            bs = auto_blocksize(10000, ec)
            assert bs > 0

    def test_larger_ec_smaller_block(self):
        bs_l = auto_blocksize(10000, 0)
        bs_h = auto_blocksize(10000, 3)
        assert bs_l >= bs_h

    def test_v2_large_file_rejected_when_capacity_impossible(self):
        with pytest.raises(ValueError, match="File too large for V2"):
            auto_blocksize(100_000_000, 1, protocol_version=2)

    def test_v3_large_file_allows_large_block_count(self):
        bs = auto_blocksize(100_000_000, 1, protocol_version=3)
        assert ceil(100_000_000 / bs) > 65535

    def test_binary_gives_larger_blocksize(self):
        bs_b64 = auto_blocksize(10000, binary_qr=False)
        bs_bin = auto_blocksize(10000, binary_qr=True)
        assert bs_bin > bs_b64

    def test_v3_overhead_is_larger_than_v2(self):
        assert V3_BLOCK_OVERHEAD > V2_HEADER_SIZE

    def test_unsupported_version_rejected(self):
        with pytest.raises(ValueError, match="Unsupported block version"):
            unpack(b'\x01' + b'\x00' * 19)

    def test_too_short_block(self):
        with pytest.raises(ValueError, match="Block too short"):
            unpack(b'\x02\x00')
