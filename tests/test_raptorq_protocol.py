"""Tests for V4 (RaptorQ) protocol serialization."""

import struct

from qrstream.protocol import (
    V4_VERSION,
    V4Header,
    V4_BLOCK_OVERHEAD,
    pack_v4,
    unpack_v4,
    unpack,
)


class TestV4PackUnpack:
    """Round-trip V4 (RaptorQ) block serialization."""

    def test_basic_roundtrip(self):
        data = b'\xAB' * 64
        packed = pack_v4(
            filesize=1024,
            symbol_size=64,
            symbol_count=16,
            esi=42,
            block_seq=7,
            data=data,
        )
        header, unpacked_data = unpack_v4(packed)
        assert isinstance(header, V4Header)
        assert header.version == V4_VERSION
        assert header.filesize == 1024
        assert header.symbol_size == 64
        assert header.symbol_count == 16
        assert header.esi == 42
        assert header.block_seq == 7
        assert unpacked_data == data

    def test_reserved_roundtrip(self):
        data = b'\xAB' * 64
        packed = pack_v4(
            filesize=1024,
            symbol_size=64,
            symbol_count=16,
            esi=0x01000000,
            block_seq=7,
            data=data,
            reserved=2,
        )
        header, unpacked_data = unpack_v4(packed)
        assert header.esi == 0x01000000
        assert header.reserved == 2
        assert unpacked_data == data

    def test_compressed_flag(self):
        data = b'\x00' * 32
        packed = pack_v4(
            filesize=100,
            symbol_size=32,
            symbol_count=4,
            esi=0,
            block_seq=0,
            data=data,
            compressed=True,
        )
        header, _ = unpack_v4(packed)
        assert header.compressed is True

    def test_alphanumeric_flag(self):
        data = b'\x00' * 32
        packed = pack_v4(
            filesize=100,
            symbol_size=32,
            symbol_count=4,
            esi=1,
            block_seq=0,
            data=data,
            alphanumeric_qr=True,
        )
        header, _ = unpack_v4(packed)
        assert header.binary_qr is True
        assert header.alphanumeric_qr is True

    def test_prng_version_is_minus_one(self):
        """V4 headers have no PRNG; prng_version should be -1."""
        data = b'\x00' * 32
        packed = pack_v4(
            filesize=100,
            symbol_size=32,
            symbol_count=4,
            esi=0,
            block_seq=0,
            data=data,
        )
        header, _ = unpack_v4(packed)
        assert header.prng_version == -1

    def test_compatibility_properties(self):
        """V4Header provides V3-compatible property aliases."""
        data = b'\x00' * 32
        packed = pack_v4(
            filesize=200,
            symbol_size=32,
            symbol_count=7,
            esi=5,
            block_seq=3,
            data=data,
        )
        header, _ = unpack_v4(packed)
        # V3 aliases
        assert header.blocksize == 32
        assert header.block_count == 7
        assert header.seed == 5

    def test_crc_validation(self):
        """Corrupt CRC should raise ValueError."""
        data = b'\x00' * 32
        packed = pack_v4(
            filesize=100,
            symbol_size=32,
            symbol_count=4,
            esi=0,
            block_seq=0,
            data=data,
        )
        # Corrupt the CRC (last 4 bytes)
        corrupted = packed[:-4] + b'\xff\xff\xff\xff'
        try:
            unpack_v4(corrupted)
            assert False, "Expected ValueError for CRC mismatch"
        except ValueError as exc:
            assert "CRC32 mismatch" in str(exc)

    def test_skip_crc(self):
        """skip_crc=True should accept blocks with bad CRC."""
        data = b'\x00' * 32
        packed = pack_v4(
            filesize=100,
            symbol_size=32,
            symbol_count=4,
            esi=0,
            block_seq=0,
            data=data,
        )
        corrupted = packed[:-4] + b'\xff\xff\xff\xff'
        header, unpacked_data = unpack_v4(corrupted, skip_crc=True)
        assert unpacked_data == data


class TestUnpackDispatch:
    """unpack() auto-detects V3 vs V4 by version byte."""

    def test_v4_dispatch(self):
        data = b'\x00' * 32
        packed = pack_v4(
            filesize=100,
            symbol_size=32,
            symbol_count=4,
            esi=0,
            block_seq=0,
            data=data,
        )
        header, _ = unpack(packed)
        assert isinstance(header, V4Header)
        assert header.version == V4_VERSION

    def test_v3_still_works(self):
        from qrstream.protocol import pack_v3, V3Header
        data = b'\x00' * 32
        packed = pack_v3(
            filesize=100,
            blocksize=32,
            block_count=4,
            seed=1,
            block_seq=0,
            data=data,
        )
        header, _ = unpack(packed)
        assert isinstance(header, V3Header)


class TestV4Validation:
    """Edge cases and validation for V4 packing."""

    def test_data_longer_than_symbol_size_raises(self):
        try:
            pack_v4(filesize=100, symbol_size=32, symbol_count=4,
                    esi=0, block_seq=0, data=b'\x00' * 64)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_esi_exceeds_uint32_raises(self):
        try:
            pack_v4(filesize=100, symbol_size=32, symbol_count=4,
                    esi=0x1_0000_0000, block_seq=0, data=b'\x00' * 32)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_reserved_exceeds_uint16_raises(self):
        try:
            pack_v4(filesize=100, symbol_size=32, symbol_count=4,
                    esi=0, block_seq=0, data=b'\x00' * 32,
                    reserved=0x1_0000)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_overhead_bytes(self):
        """V4 overhead should be 28 bytes (same as V3)."""
        assert V4_BLOCK_OVERHEAD == 28
