"""Tests for performance optimizations and advanced features."""

import os
from math import ceil

import numpy as np
import pytest

from qrstream.lt_codec import (
    BlockGraph, xor_bytes, _to_np, _xor_np, _xor_np_inplace,
)
from qrstream.protocol import (
    pack_v2, unpack, V2_HEADER_SIZE,
    cobs_encode, cobs_decode,
)
from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder
from qrstream.qr_utils import (
    generate_qr_image, reset_strategy_stats, try_decode_qr,
)


class TestNumpyBlockGraph:
    """Test numpy-based BlockGraph correctness."""

    def test_stores_numpy_arrays(self):
        bg = BlockGraph(2)
        bg.add_block({0}, b'\x01\x02\x03')
        assert isinstance(bg.eliminated[0], np.ndarray)
        assert bg.eliminated[0].dtype == np.uint8

    def test_numpy_xor_consistency(self):
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
        data = os.urandom(2048)
        blocksize = 64
        encoder = LTEncoder(data, blocksize)
        decoder = LTDecoder()
        for packed, seed, seq in encoder.generate_blocks(int(ceil(len(data) / blocksize) * 3)):
            done, _ = decoder.decode_bytes(packed)
            if done:
                assert decoder.bytes_dump() == data
                return
        pytest.fail("Decoding did not complete")


class TestBatchXor:
    """Test batch XOR in encoder's generate_block."""

    def test_batch_xor_produces_correct_output(self):
        data = os.urandom(256)
        blocksize = 64
        encoder = LTEncoder(data, blocksize)
        decoder = LTDecoder()
        for packed, seed, seq in encoder.generate_blocks(30):
            done, _ = decoder.decode_bytes(packed)
            if done:
                assert decoder.bytes_dump() == data
                return
        pytest.fail("Decoding did not complete")


class TestQrGeneration:
    """Test QR code generation paths."""

    def test_opencv_qr_generation(self):
        data = pack_v2(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64)
        img = generate_qr_image(data, ec_level=1, version=20, use_legacy=False)
        assert img is not None
        assert img.shape[2] == 3

    def test_legacy_qr_generation(self):
        data = pack_v2(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64)
        img = generate_qr_image(data, ec_level=1, version=20, use_legacy=True)
        assert img is not None
        assert img.shape[2] == 3

    def test_binary_qr_generation(self):
        data = pack_v2(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64, binary_qr=True)
        img = generate_qr_image(data, ec_level=1, version=20, binary_mode=True)
        assert img is not None
        assert img.shape[2] == 3


class TestWeChatDetector:
    """Test WeChatQRCode detector integration."""

    def test_reset(self):
        reset_strategy_stats()

    def test_wechat_detects_base64_qr(self):
        import base64, zlib
        data = os.urandom(64)
        packed = pack_v2(filesize=100, blocksize=64, block_count=2,
                         seed=42, block_seq=0, data=data)
        img = generate_qr_image(packed, ec_level=1, version=20)
        result = try_decode_qr(img)
        assert result is not None
        block = base64.b64decode(result)
        stored = int.from_bytes(block[16:20], 'big')
        computed = zlib.crc32(block[:16] + block[20:]) & 0xFFFFFFFF
        assert stored == computed

    def test_wechat_detects_cobs_binary_qr(self):
        import zlib
        data = os.urandom(64)
        packed = pack_v2(filesize=100, blocksize=64, block_count=2,
                         seed=99, block_seq=0, data=data, binary_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20, binary_mode=True)
        result = try_decode_qr(img)
        assert result is not None
        block = cobs_decode(result.encode('latin-1'))
        stored = int.from_bytes(block[16:20], 'big')
        computed = zlib.crc32(block[:16] + block[20:]) & 0xFFFFFFFF
        assert stored == computed

    def test_wechat_binary_qr_with_null_heavy_data(self):
        data = b'\x00' * 64
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=7, block_seq=0, data=data, binary_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20, binary_mode=True)
        result = try_decode_qr(img)
        assert result is not None
        block = cobs_decode(result.encode('latin-1'))
        assert block == packed

    def test_binary_qr_full_roundtrip(self):
        """Binary QR: encode image -> WeChatQRCode detect -> COBS decode -> unpack."""
        block_data = os.urandom(64)
        packed = pack_v2(filesize=100, blocksize=64, block_count=2,
                         seed=1, block_seq=0, data=block_data, binary_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20, binary_mode=True)
        qr_str = try_decode_qr(img)
        assert qr_str
        recovered = cobs_decode(qr_str.encode('latin-1'))
        assert recovered == packed


class TestCobs:
    """Test COBS encode/decode correctness."""

    def test_roundtrip_simple(self):
        assert cobs_decode(cobs_encode(b'Hello, World!')) == b'Hello, World!'

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

    def test_roundtrip_all_bytes(self):
        data = bytes(range(256)) * 3
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
        data = os.urandom(10000)
        encoded = cobs_encode(data)
        overhead = len(encoded) - len(data)
        assert overhead <= ceil(len(data) / 254) + 1

    def test_v2_block_roundtrip_with_cobs(self):
        block_data = os.urandom(64)
        packed = pack_v2(filesize=64, blocksize=64, block_count=1,
                         seed=42, block_seq=0, data=block_data, binary_qr=True)
        encoded = cobs_encode(packed)
        assert b'\x00' not in encoded
        decoded = cobs_decode(encoded)
        assert decoded == packed
        header, data = unpack(decoded)
        assert header.seed == 42
        assert data == block_data
