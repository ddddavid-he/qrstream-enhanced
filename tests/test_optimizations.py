"""Tests for performance optimizations and advanced features."""

import random
from math import ceil

import numpy as np
import pytest

from qrstream.lt_codec import (
    BlockGraph, xor_bytes, _to_np, _xor_np, _xor_np_inplace,
)
from qrstream.protocol import (
    pack_v3, unpack,
    base45_encode, base45_decode,
)
from qrstream import __version__
from qrstream.cli import build_parser
from qrstream.encoder import LTEncoder, _resolve_border_modules
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
        rng = random.Random(0xA1B2C3D4)
        a = rng.randbytes(128)
        b = rng.randbytes(128)
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
        data = random.Random(0xE5E5E5E5).randbytes(2048)
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
        data = random.Random(0xBA7C1135).randbytes(256)
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

    def test_alphanumeric_qr_generation(self):
        data = pack_v3(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64,
                       alphanumeric_qr=True)
        img = generate_qr_image(data, ec_level=1, version=20,
                                alphanumeric=True)
        assert img is not None
        assert img.shape[2] == 3

    def test_base64_qr_generation(self):
        data = pack_v3(filesize=100, blocksize=64, block_count=2,
                       seed=1, block_seq=0, data=b'\xAA' * 64)
        img = generate_qr_image(data, ec_level=1, version=20,
                                alphanumeric=False)
        assert img is not None
        assert img.shape[2] == 3


class TestCli:
    def test_version_flag_prints_package_version(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['-v'])
        assert exc_info.value.code == 0
        assert capsys.readouterr().out.strip() == f'qrstream {__version__}'

    def test_verbose_flag_stays_on_subcommands(self):
        parser = build_parser()
        args = parser.parse_args(['encode', 'input.bin', '-o', 'out.mp4', '-V'])
        assert args.verbose is True

    def test_output_mode_defaults_to_auto(self):
        parser = build_parser()
        args = parser.parse_args(
            ['decode', 'video.mp4', '-o', 'out.bin'])
        assert args.output_mode == 'auto'
        assert args.verbose is False

    def test_output_mode_accepts_all_modes(self):
        parser = build_parser()
        for mode in ('auto', 'interactive', 'log', 'quiet', 'verbose'):
            args = parser.parse_args(
                ['encode', 'input.bin', '-o', 'out.mp4',
                 '--output-mode', mode])
            assert args.output_mode == mode

    def test_encode_border_default_uses_standard_quiet_zone(self):
        parser = build_parser()
        args = parser.parse_args(['encode', 'input.bin', '-o', 'out.mp4'])
        assert args.border is None


class TestBorderDefaults:
    def test_default_border_resolves_to_standard_quiet_zone(self):
        assert _resolve_border_modules(20, None) == 4.0

    def test_percentage_border_is_still_supported(self):
        assert _resolve_border_modules(20, 10.0) == pytest.approx(9.7)


class TestZxingDetector:
    """Test zxing-cpp detector integration."""

    def test_reset(self):
        reset_strategy_stats()

    def test_detects_base64_qr(self):
        import base64
        # Deterministic mixed-byte payload: using a fixed payload avoids
        # flaky CI failures from random payloads that land on detector
        # edge cases. Tests encode → zxing detect → base64 decode → unpack.
        reset_strategy_stats()
        data = bytes((i * 37 + 11) % 256 for i in range(64))
        packed = pack_v3(filesize=100, blocksize=64, block_count=2,
                         seed=42, block_seq=0, data=data)
        img = generate_qr_image(packed, ec_level=1, version=20,
                                alphanumeric=False)
        result = try_decode_qr(img)
        assert result is not None
        block = base64.b64decode(result)
        # CRC integrity: unpack validates CRC and raises on mismatch.
        header, recovered = unpack(block)
        assert header.seed == 42
        assert recovered == data

    def test_detects_alphanumeric_qr(self):
        """New default: base45 payload in QR alphanumeric mode."""
        reset_strategy_stats()
        data = bytes((i * 53 + 7) % 256 for i in range(64))
        packed = pack_v3(filesize=100, blocksize=64, block_count=2,
                         seed=99, block_seq=0, data=data,
                         alphanumeric_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20,
                                alphanumeric=True)
        result = try_decode_qr(img)
        assert result is not None
        block = base45_decode(result)
        header, recovered = unpack(block)
        assert header.seed == 99
        assert header.alphanumeric_qr is True
        assert recovered == data

    def test_alphanumeric_qr_with_null_heavy_data(self):
        """base45 happily carries all-zero payloads (0x00 is legal)."""
        data = b'\x00' * 64
        packed = pack_v3(filesize=64, blocksize=64, block_count=1,
                         seed=7, block_seq=0, data=data,
                         alphanumeric_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20,
                                alphanumeric=True)
        result = try_decode_qr(img)
        assert result is not None
        block = base45_decode(result)
        assert block == packed

    def test_alphanumeric_qr_full_roundtrip(self):
        """base45 QR: encode image -> zxing detect -> base45 decode -> unpack."""
        reset_strategy_stats()
        # Use deterministic mixed bytes to avoid flaky CI failures from
        # unlucky random payloads that occasionally reduce detector stability.
        block_data = bytes((i * 37 + 11) % 256 for i in range(64))
        packed = pack_v3(filesize=100, blocksize=64, block_count=2,
                         seed=1, block_seq=0, data=block_data,
                         alphanumeric_qr=True)
        img = generate_qr_image(packed, ec_level=1, version=20,
                                alphanumeric=True)
        qr_str = try_decode_qr(img)
        assert qr_str is not None
        recovered = base45_decode(qr_str)
        assert recovered == packed



class TestBase45:
    """Test base45 encode/decode correctness."""

    def test_roundtrip_simple(self):
        assert base45_decode(base45_encode(b'Hello, World!')) == b'Hello, World!'

    def test_roundtrip_with_nulls(self):
        data = b'\x00\x00\x00'
        assert base45_decode(base45_encode(data)) == data

    def test_roundtrip_mixed(self):
        data = b'\x01\x00\x02\x00\x03'
        assert base45_decode(base45_encode(data)) == data

    def test_roundtrip_all_bytes(self):
        data = bytes(range(256)) * 3
        assert base45_decode(base45_encode(data)) == data

    def test_empty(self):
        assert base45_decode(base45_encode(b'')) == b''

    def test_roundtrip_random(self):
        data = random.Random(0x0BC0E5E1).randbytes(10000)
        assert base45_decode(base45_encode(data)) == data

    def test_v3_block_roundtrip_with_base45(self):
        """base45 roundtrip through V3 block serialization."""
        block_data = random.Random(0xC0B5B10C).randbytes(64)
        packed = pack_v3(filesize=64, blocksize=64, block_count=1,
                         seed=42, block_seq=0, data=block_data,
                         alphanumeric_qr=True)
        encoded = base45_encode(packed)
        decoded = base45_decode(encoded)
        assert decoded == packed
        header, data = unpack(decoded)
        assert header.seed == 42
        assert data == block_data

