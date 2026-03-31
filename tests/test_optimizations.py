"""Tests for performance optimizations to ensure correctness."""

import os
import zlib
from math import ceil

import numpy as np
import pytest

from qrstream.lt_codec import (
    BlockGraph, xor_bytes, _to_np, _xor_np, _xor_np_inplace,
)
from qrstream.protocol import (
    pack_v2, unpack, auto_blocksize, V2Header, V2_HEADER_SIZE,
    cobs_encode, cobs_decode,
)
from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder
from qrstream.qr_utils import (
    generate_qr_image, reset_strategy_stats, try_decode_qr,
)


class TestSkipCrc:
    """Test CRC skip parameter for pre-validated blocks."""

    def test_skip_crc_valid_block(self):
        data = b'\xAB' * 64
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=42, block_seq=0, data=data)
        # Normal unpack (validates CRC)
        header1, data1 = unpack(packed, skip_crc=False)
        # Skip CRC unpack
        header2, data2 = unpack(packed, skip_crc=True)
        assert data1 == data2
        assert header1.seed == header2.seed

    def test_skip_crc_corrupt_block(self):
        data = b'\xAB' * 64
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=42, block_seq=0, data=data)
        # Corrupt one byte
        corrupted = bytearray(packed)
        corrupted[25] ^= 0xFF
        corrupted = bytes(corrupted)

        # Normal unpack detects corruption
        with pytest.raises(ValueError, match="CRC32"):
            unpack(corrupted, skip_crc=False)

        # Skip CRC does NOT detect corruption (by design)
        header, data = unpack(corrupted, skip_crc=True)
        assert header.seed == 42


class TestNumpyBlockGraph:
    """Test numpy-based BlockGraph correctness."""

    def test_stores_numpy_arrays(self):
        bg = BlockGraph(2)
        bg.add_block({0}, b'\x01\x02\x03')
        assert isinstance(bg.eliminated[0], np.ndarray)
        assert bg.eliminated[0].dtype == np.uint8

    def test_numpy_xor_consistency(self):
        """Verify numpy XOR matches bytes XOR."""
        a = os.urandom(128)
        b = os.urandom(128)
        result_bytes = xor_bytes(a, b)
        result_np = _xor_np(_to_np(a), _to_np(b)).tobytes()
        assert result_bytes == result_np

    def test_inplace_xor(self):
        a = np.array([0x01, 0x02, 0x03], dtype=np.uint8)
        b = np.array([0xFF, 0x00, 0x0F], dtype=np.uint8)
        expected = np.array([0xFE, 0x02, 0x0C], dtype=np.uint8)
        _xor_np_inplace(a, b)
        np.testing.assert_array_equal(a, expected)

    def test_large_block_graph_recovery(self):
        """Test BlockGraph with many blocks."""
        data = os.urandom(2048)
        blocksize = 64
        K = ceil(len(data) / blocksize)
        encoder = LTEncoder(data, blocksize)
        decoder = LTDecoder()

        num_blocks = int(K * 3.0)
        for packed, seed, seq in encoder.generate_blocks(num_blocks):
            done, _ = decoder.decode_bytes(packed)
            if done:
                result = decoder.bytes_dump()
                assert result == data
                return
        pytest.fail("Decoding did not complete")


class TestBatchXor:
    """Test batch XOR in encoder's generate_block."""

    def test_batch_xor_produces_correct_output(self):
        """Verify batch XOR (numpy reduce) matches sequential XOR."""
        data = os.urandom(256)
        blocksize = 64
        encoder = LTEncoder(data, blocksize)

        # Generate blocks and verify they decode correctly
        decoder = LTDecoder()
        for packed, seed, seq in encoder.generate_blocks(30):
            done, _ = decoder.decode_bytes(packed)
            if done:
                assert decoder.bytes_dump() == data
                return
        pytest.fail("Decoding did not complete")


class TestAutoBlocksizeBinaryQr:
    """Test auto_blocksize with binary QR mode."""

    def test_binary_mode_gives_larger_blocksize(self):
        filesize = 10000
        bs_base64 = auto_blocksize(filesize, ec_level=1, qr_version=20)
        bs_binary = auto_blocksize(filesize, ec_level=1, qr_version=20,
                                   binary_qr=True)
        assert bs_binary > bs_base64

    def test_binary_mode_capacity_ratio(self):
        """Binary mode should give roughly 33% more capacity."""
        filesize = 100000
        bs_base64 = auto_blocksize(filesize, ec_level=1, qr_version=20)
        bs_binary = auto_blocksize(filesize, ec_level=1, qr_version=20,
                                   binary_qr=True)
        ratio = bs_binary / bs_base64
        assert 1.2 < ratio < 1.5  # roughly 33% more


class TestBinaryQrFlag:
    """Test binary_qr flag in V2 protocol."""

    def test_binary_qr_flag_roundtrip(self):
        data = b'\x00' * 64
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=1, block_seq=0, data=data,
                         binary_qr=True)
        header, unpacked_data = unpack(packed)
        assert isinstance(header, V2Header)
        assert header.binary_qr is True

    def test_no_binary_qr_flag(self):
        data = b'\x00' * 64
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=1, block_seq=0, data=data)
        header, _ = unpack(packed)
        assert header.binary_qr is False


class TestQrGeneration:
    """Test QR code generation paths."""

    def test_opencv_qr_generation(self):
        data = pack_v2(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64)
        img = generate_qr_image(data, ec_level=1, version=20,
                                use_legacy=False)
        assert img is not None
        assert len(img.shape) == 3
        assert img.shape[2] == 3  # BGR

    def test_legacy_qr_generation(self):
        data = pack_v2(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64)
        img = generate_qr_image(data, ec_level=1, version=20,
                                use_legacy=True)
        assert img is not None
        assert len(img.shape) == 3

    def test_binary_qr_generation(self):
        data = pack_v2(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64,
                       binary_qr=True)
        img = generate_qr_image(data, ec_level=1, version=20,
                                binary_mode=True)
        assert img is not None
        assert len(img.shape) == 3


class TestWeChatDetector:
    """Test WeChatQRCode detector integration."""

    def test_reset(self):
        reset_strategy_stats()

    def test_wechat_detects_base64_qr(self):
        """WeChatQRCode can decode standard base64-encoded QR."""
        import base64, zlib
        data = os.urandom(64)
        packed = pack_v2(filesize=100, blocksize=64, block_count=2,
                         seed=42, block_seq=0, data=data)
        img = generate_qr_image(packed, ec_level=1, version=20)
        result = try_decode_qr(img)
        assert result is not None
        block = base64.b64decode(result)
        assert block[0] == 0x02
        stored = int.from_bytes(block[16:20], 'big')
        computed = zlib.crc32(block[:16] + block[20:]) & 0xFFFFFFFF
        assert stored == computed

    def test_wechat_detects_cobs_binary_qr(self):
        """WeChatQRCode can decode COBS binary QR and payload validates."""
        import zlib
        data = os.urandom(64)
        packed = pack_v2(filesize=100, blocksize=64, block_count=2,
                         seed=99, block_seq=0, data=data, binary_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20,
                                binary_mode=True)
        result = try_decode_qr(img)
        assert result is not None, "WeChatQRCode should detect COBS QR"
        # Decode COBS: latin-1 -> cobs_decode -> raw bytes
        raw = result.encode('latin-1')
        block = cobs_decode(raw)
        assert block[0] == 0x02
        stored = int.from_bytes(block[16:20], 'big')
        computed = zlib.crc32(block[:16] + block[20:]) & 0xFFFFFFFF
        assert stored == computed, "COBS binary QR CRC must match"

    def test_wechat_binary_qr_with_null_heavy_data(self):
        """Binary QR with data full of null bytes still works via COBS."""
        import zlib
        data = b'\x00' * 64  # worst case for COBS
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=7, block_seq=0, data=data, binary_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20,
                                binary_mode=True)
        result = try_decode_qr(img)
        assert result is not None
        raw = result.encode('latin-1')
        block = cobs_decode(raw)
        assert block == packed


class TestCobs:
    """Test COBS encode/decode correctness."""

    def test_roundtrip_simple(self):
        data = b'Hello, World!'
        assert cobs_decode(cobs_encode(data)) == data

    def test_roundtrip_with_nulls(self):
        data = b'\x00\x00\x00'
        encoded = cobs_encode(data)
        assert b'\x00' not in encoded
        assert cobs_decode(encoded) == data

    def test_roundtrip_mixed(self):
        data = b'\x01\x00\x02\x00\x03'
        encoded = cobs_encode(data)
        assert b'\x00' not in encoded
        assert cobs_decode(encoded) == data

    def test_roundtrip_no_nulls(self):
        data = b'\x01\x02\x03\x04\x05'
        encoded = cobs_encode(data)
        assert b'\x00' not in encoded
        assert cobs_decode(encoded) == data

    def test_roundtrip_binary(self):
        data = bytes(range(256)) * 3  # all byte values including \x00
        encoded = cobs_encode(data)
        assert b'\x00' not in encoded
        assert cobs_decode(encoded) == data

    def test_roundtrip_random(self):
        data = os.urandom(1000)
        encoded = cobs_encode(data)
        assert b'\x00' not in encoded
        assert cobs_decode(encoded) == data

    def test_empty(self):
        assert cobs_decode(cobs_encode(b'')) == b''

    def test_single_null(self):
        data = b'\x00'
        encoded = cobs_encode(data)
        assert b'\x00' not in encoded
        assert cobs_decode(encoded) == data

    def test_overhead_is_small(self):
        """COBS overhead should be at most ~0.4%."""
        data = os.urandom(10000)
        encoded = cobs_encode(data)
        overhead = len(encoded) - len(data)
        max_expected = ceil(len(data) / 254) + 1
        assert overhead <= max_expected

    def test_v2_block_roundtrip_with_cobs(self):
        """Full V2 block → COBS encode → COBS decode → unpack."""
        block_data = os.urandom(64)
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=42, block_seq=0, data=block_data,
                         binary_qr=True)
        encoded = cobs_encode(packed)
        assert b'\x00' not in encoded
        decoded = cobs_decode(encoded)
        assert decoded == packed
        header, data = unpack(decoded)
        assert header.seed == 42
        assert data == block_data


class TestBinaryQrCobsGeneration:
    """Test binary QR generation uses COBS (no null bytes in QR payload)."""

    def test_binary_qr_generates_image(self):
        data = pack_v2(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\x00' * 64,
                       binary_qr=True)
        img = generate_qr_image(data, ec_level=1, version=20,
                                binary_mode=True)
        assert img is not None
        assert len(img.shape) == 3

    def test_binary_qr_detectable_by_wechat(self):
        """COBS-encoded binary QR should be decodable by WeChatQRCode."""
        from qrstream.protocol import cobs_encode

        block_data = os.urandom(64)
        packed = pack_v2(filesize=100, blocksize=64, block_count=2,
                         seed=1, block_seq=0, data=block_data,
                         binary_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20,
                                binary_mode=True)

        qr_str = try_decode_qr(img)
        assert qr_str, "WeChatQRCode should detect the COBS QR code"

        # Decode: latin-1 -> COBS decode -> original packed bytes
        raw = qr_str.encode('latin-1')
        recovered = cobs_decode(raw)
        assert recovered == packed
