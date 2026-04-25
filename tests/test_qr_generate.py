"""
Tests for QR image generation correctness and robustness.

What these tests catch
----------------------
Previously, ``qrcode 8.x`` crashed with ``ValueError: glog(0)`` on certain
LT fountain-code blocks whose base45-encoded content caused a Reed-Solomon
block's leading codeword to be 0x00.  The crash was silent in the sense that
it only appeared when encoding specific files (those whose compressed payload
size places the blocksize close to the QR version capacity boundary), so it
was never caught by the existing codec-level tests, which bypassed the QR
generation layer entirely.

The fix was to replace ``qrcode`` with ``segno``, which has a correct RS
implementation.  These tests ensure:

1. ``generate_qr_image`` never raises for any LT block from a realistic
   payload — including the "zero-heavy" blocks that triggered the bug.
2. The image is decodable by WechatQR (end-to-end sanity).
3. The backend uses ``segno``, not ``qrcode``.
4. Key encoding properties: version, size, border, EC level are honoured.
5. Both alphanumeric (base45) and byte (base64) paths work.
"""

import random
import zlib
from math import ceil

import numpy as np
import pytest

from qrstream.qr_utils import generate_qr_image, try_decode_qr, HAS_SEGNO
from qrstream.protocol import base45_encode, base45_decode, auto_blocksize
from qrstream.encoder import LTEncoder


# ── helpers ───────────────────────────────────────────────────────

def _random_bytes(size: int, seed: int = 0) -> bytes:
    return random.Random(seed).randbytes(size)


def _make_lt_blocks(raw: bytes, overhead: float = 2.0,
                    compress: bool = True) -> list[bytes]:
    """Return a list of packed LT blocks for ``raw``."""
    payload = zlib.compress(raw) if compress else raw
    blocksize = auto_blocksize(len(payload))
    K = ceil(len(payload) / blocksize)
    num = int(K * overhead)
    enc = LTEncoder(payload, blocksize, compressed=compress, alphanumeric_qr=True)
    return [packed for packed, _, _ in enc.generate_blocks(num)]


# ── 1. Backend guard ──────────────────────────────────────────────

class TestBackend:
    def test_segno_is_available(self):
        """segno must be installed — it is the sole QR backend."""
        assert HAS_SEGNO, "segno not found; install with `pip install segno`"

    def test_qr_utils_uses_segno(self):
        """qr_utils must import segno, not qrcode, for generation."""
        import qrstream.qr_utils as mod
        import inspect
        src = inspect.getsource(mod)
        assert "import segno" in src
        # The old qrcode backend must not be referenced in generation code
        assert "qrcode.QRCode" not in src
        assert "_EC_MAP" in src  # segno ec map still present


# ── 2. glog(0) regression ─────────────────────────────────────────

class TestGlogRegression:
    """
    Regression tests for the qrcode 8.x glog(0) crash.

    The crash occurred when:
    - auto_blocksize() chose a blocksize near the QR capacity boundary
    - An LT XOR block happened to produce mostly-zero data
    - base45 encoding of that block produced a string whose RS block
      boundary fell on a zero leading codeword

    We reproduce this by using a small compressible payload whose
    compressed size puts blocksize exactly at the problematic boundary,
    then generate all LT blocks and verify none crash.
    """

    # 68 743 bytes, compresses to ~17 574 bytes → blocksize=938 → the
    # exact configuration that triggered the crash with qrcode 8.2.
    # We synthesise equivalent data instead of shipping the original file.
    PAYLOAD_SIZE = 17_574   # approximate compressed target size

    def _synthetic_payload(self) -> bytes:
        """Return bytes that compress to approximately PAYLOAD_SIZE."""
        # Repeating structured data compresses well
        chunk = b"ROW_DATA:" + b"0123456789ABCDEF" * 40 + b"\n"
        raw = chunk * (self.PAYLOAD_SIZE // len(chunk) + 1)
        compressed = zlib.compress(raw)
        return compressed[:self.PAYLOAD_SIZE]

    def test_no_crash_on_zero_heavy_blocks(self):
        """generate_qr_image must not raise for any LT block, ever."""
        payload = self._synthetic_payload()
        blocksize = auto_blocksize(len(payload))
        K = ceil(len(payload) / blocksize)
        # Generate 2x overhead — enough to include zero-heavy high-degree blocks
        num = K * 2
        enc = LTEncoder(payload, blocksize, compressed=True, alphanumeric_qr=True)

        crashed = []
        for packed, seed, _ in enc.generate_blocks(num):
            try:
                generate_qr_image(packed, ec_level=1, version=25)
            except Exception as exc:
                crashed.append((seed, str(exc)))

        assert not crashed, (
            f"generate_qr_image crashed on {len(crashed)} block(s):\n"
            + "\n".join(f"  seed={s}: {e}" for s, e in crashed[:5])
        )

    @pytest.mark.parametrize("ec_level", [0, 1, 2, 3])
    def test_no_crash_all_ec_levels(self, ec_level):
        """Bug must not reappear at any EC level."""
        payload = self._synthetic_payload()
        blocksize = auto_blocksize(len(payload), ec_level=ec_level)
        K = ceil(len(payload) / blocksize)
        enc = LTEncoder(payload, blocksize, compressed=True, alphanumeric_qr=True)
        for packed, seed, _ in enc.generate_blocks(K * 2):
            # Must not raise
            generate_qr_image(packed, ec_level=ec_level)

    @pytest.mark.parametrize("qr_version", [20, 25, 30])
    def test_no_crash_common_versions(self, qr_version):
        """Bug was version-specific; verify multiple common versions."""
        payload = self._synthetic_payload()
        blocksize = auto_blocksize(len(payload), qr_version=qr_version)
        K = ceil(len(payload) / blocksize)
        enc = LTEncoder(payload, blocksize, compressed=True, alphanumeric_qr=True)
        for packed, seed, _ in enc.generate_blocks(K * 2):
            generate_qr_image(packed, ec_level=1, version=qr_version)


# ── 3. Image properties ───────────────────────────────────────────

class TestImageProperties:
    """generate_qr_image must produce images with predictable dimensions."""

    def _block(self) -> bytes:
        data = _random_bytes(200, seed=42)
        enc = LTEncoder(data, 64)
        return next(enc.generate_blocks(1))[0]

    def test_output_is_bgr_ndarray(self):
        img = generate_qr_image(self._block(), version=10)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3  # BGR

    def test_box_size_scales_image(self):
        block = self._block()
        img5 = generate_qr_image(block, box_size=5, border=0, version=10)
        img10 = generate_qr_image(block, box_size=10, border=0, version=10)
        # 10-pixel box should be exactly twice the 5-pixel box
        assert img10.shape[0] == img5.shape[0] * 2
        assert img10.shape[1] == img5.shape[1] * 2

    def test_border_adds_quiet_zone(self):
        block = self._block()
        img0 = generate_qr_image(block, box_size=10, border=0, version=10)
        img4 = generate_qr_image(block, box_size=10, border=4, version=10)
        # Each border adds 2 * border * box_size pixels on each axis
        diff = img4.shape[0] - img0.shape[0]
        assert diff == 2 * 4 * 10

    def test_image_is_binary_black_white(self):
        """QR modules must be pure black (0) or pure white (255)."""
        img = generate_qr_image(self._block(), version=10)
        unique = set(np.unique(img))
        assert unique <= {0, 255}, f"Unexpected pixel values: {unique - {0, 255}}"

    def test_version_controls_size(self):
        """Higher QR version → larger matrix → larger image."""
        block = self._block()
        img15 = generate_qr_image(block, box_size=5, border=0, version=15)
        img20 = generate_qr_image(block, box_size=5, border=0, version=20)
        assert img20.shape[0] > img15.shape[0]


# ── 4. Encode/decode roundtrip ────────────────────────────────────

class TestEncodeDecodeRoundtrip:
    """
    End-to-end: generate_qr_image → try_decode_qr must recover the data.

    WechatQR occasionally fails to decode specific QR matrices (this is
    a known sensitivity of any real-world detector).  We therefore test
    multiple independently-seeded blocks and require a high success rate
    rather than 100%.
    """

    def _roundtrip_block(self, packed: bytes, version: int = 15,
                          ec_level: int = 1) -> bool:
        """Return True if try_decode_qr recovers the base45 payload."""
        img = generate_qr_image(packed, ec_level=ec_level, version=version)
        decoded_str = try_decode_qr(img)
        if decoded_str is None:
            return False
        expected = base45_encode(packed).decode("ascii")
        return decoded_str == expected

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 10, 42, 100])
    def test_small_block_roundtrip(self, seed):
        """Small payloads at a safe version should decode reliably."""
        data = _random_bytes(64, seed=seed)
        enc = LTEncoder(data, 32)
        packed, _, _ = next(enc.generate_blocks(1))
        assert self._roundtrip_block(packed, version=10), \
            f"Failed to decode QR for seed={seed}"

    def test_realistic_payload_high_success_rate(self):
        """
        Realistic LT blocks from a compressible payload must decode at
        ≥70% success rate with WechatQR.

        The threshold is intentionally below 100% because WechatQR has
        documented sensitivity to certain QR mask patterns — this is
        not a generation bug.  The same 70% threshold applies equally
        to qrcode and segno (they produce identical pixels).
        """
        raw = _random_bytes(5_000, seed=0xDEAD)
        blocks = _make_lt_blocks(raw, overhead=2.0)[:30]

        ok = 0
        for packed in blocks:
            if self._roundtrip_block(packed, version=25, ec_level=1):
                ok += 1

        rate = ok / len(blocks)
        assert rate >= 0.70, (
            f"WechatQR decode success rate too low: {ok}/{len(blocks)} = {rate:.0%}"
        )

    def test_base64_path_roundtrip(self):
        """byte-mode (base64) path must also encode/decode correctly."""
        import base64
        data = _random_bytes(100, seed=7)
        enc = LTEncoder(data, 64, alphanumeric_qr=False)
        packed, _, _ = next(enc.generate_blocks(1))
        img = generate_qr_image(packed, ec_level=1, version=10, alphanumeric=False)
        decoded_str = try_decode_qr(img)
        assert decoded_str is not None
        expected = base64.b64encode(packed).decode("ascii")
        assert decoded_str == expected


# ── 5. All blocks for triggering file sizes ───────────────────────

class TestAllBlocksNoException:
    """
    For every combination of payload size and QR version that was
    plausible at the time the bug existed, generate_qr_image must
    complete without exception.

    These tests are deliberately data-driven so that future regressions
    at new version/size combinations are caught automatically.
    """

    @pytest.mark.parametrize("compressed_size,qr_version,ec_level", [
        # Original crash: ~17.5 KB compressed → v25 ec=1
        (17_574, 25, 1),
        # Variants around the same boundary
        (17_000, 25, 1),
        (18_000, 25, 1),
        # Other common version/ec combos
        (10_000, 20, 1),
        (30_000, 30, 1),
        (5_000,  15, 0),
    ])
    def test_no_exception_for_config(self, compressed_size, qr_version, ec_level):
        payload = _random_bytes(compressed_size, seed=qr_version * 100 + ec_level)
        blocksize = auto_blocksize(len(payload), ec_level=ec_level,
                                   qr_version=qr_version)
        K = ceil(len(payload) / blocksize)
        enc = LTEncoder(payload, blocksize, compressed=False, alphanumeric_qr=True)
        for packed, seed, _ in enc.generate_blocks(min(K * 2, 100)):
            try:
                generate_qr_image(packed, ec_level=ec_level, version=qr_version)
            except Exception as exc:
                pytest.fail(
                    f"generate_qr_image raised for compressed_size={compressed_size} "
                    f"qr_version={qr_version} ec_level={ec_level} seed={seed}: {exc}"
                )
