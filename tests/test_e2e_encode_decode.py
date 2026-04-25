"""
End-to-end encode → video → decode pipeline tests.

These tests exercise the full stack:
  input_file → encode_to_video → extract_qr_from_video
             → decode_blocks_to_file → SHA256 verify

Marked ``@pytest.mark.e2e`` — excluded from the default ``pytest`` run
(see ``pyproject.toml addopts``).  The dedicated CI workflow
``.github/workflows/e2e-encode-decode.yml`` opts in with ``-m e2e``.

Why these tests are necessary
------------------------------
The unit-test suite (``test.yml``) calls ``LTEncoder`` / ``LTDecoder``
directly and never touches QR image generation.  This meant the
``qrcode 8.x glog(0)`` crash was invisible to CI: it only fired when
``encode_to_video`` was called on a file whose payload size fell close
to the QR-version capacity boundary (blocksize=938, v25 EC_M, K=19).

These tests close that gap by running the complete pipeline on files
specifically sized to hit that boundary, plus a sweep of common sizes.

Sizes
-----
User requirement: 10 KB, 100 KB, 500 KB (raw input).

The glog-trigger test uses a 17 574-byte raw file encoded *without*
compression — this is the exact payload size that triggers blocksize=938
and K=19 (the configuration that caused the original crash).
"""

import hashlib
import pathlib
import random
from math import ceil

import pytest

from qrstream.protocol import auto_blocksize


# ── helpers ───────────────────────────────────────────────────────

def _random_bytes(size: int, seed: int = 0) -> bytes:
    return random.Random(seed).randbytes(size)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_decode_verify(raw: bytes, tmp_path: pathlib.Path,
                           label: str = "file",
                           compress: bool = True) -> None:
    """Complete pipeline: raw bytes → MP4 → recovered bytes → assert SHA256."""
    from qrstream.encoder import encode_to_video
    from qrstream.decoder import extract_qr_from_video, decode_blocks_to_file

    src = tmp_path / f"{label}.bin"
    mp4 = tmp_path / f"{label}.mp4"
    out = tmp_path / f"{label}_out.bin"

    src.write_bytes(raw)
    src_hash = _sha256(src)

    encode_to_video(str(src), str(mp4), compress=compress, verbose=False)
    assert mp4.exists() and mp4.stat().st_size > 0, \
        f"encode_to_video produced no output for {label}"

    blocks = extract_qr_from_video(str(mp4), verbose=False)
    written = decode_blocks_to_file(blocks, str(out), verbose=False)

    assert out.exists(), f"decode produced no output file for {label}"
    assert written == len(raw), \
        f"{label}: written={written} != expected={len(raw)}"
    assert _sha256(out) == src_hash, \
        f"{label}: SHA256 mismatch — data corrupted after encode/decode"


# ── tests ─────────────────────────────────────────────────────────

@pytest.mark.e2e
class TestE2EEncodeDecode:
    """Full encode→video→decode roundtrip with SHA256 verification."""

    # ── glog(0) regression ────────────────────────────────────────

    def test_glog_trigger_config(self, tmp_path):
        """
        17 574-byte payload encoded without compression.

        ``auto_blocksize(17574)`` returns 938 → K=19 blocks, v25 EC_M.
        This is the exact (payload_size, blocksize, version) triple that
        triggered the ``qrcode 8.x glog(0)`` crash.  With segno as the
        QR backend it must complete and recover the file byte-exactly.
        """
        # Verify the trigger condition is still active
        bs = auto_blocksize(17_574)
        assert bs == 938, f"auto_blocksize changed: {bs} (expected 938)"
        assert ceil(17_574 / bs) == 19

        raw = _random_bytes(17_574, seed=0x616C6F67)  # "alog" in hex
        _encode_decode_verify(raw, tmp_path, "glog_trigger", compress=False)

    # ── user-requested sizes ──────────────────────────────────────

    def test_10k(self, tmp_path):
        """10 KB random input."""
        _encode_decode_verify(_random_bytes(10_000, seed=10), tmp_path, "10k")

    def test_100k(self, tmp_path):
        """100 KB random input."""
        _encode_decode_verify(_random_bytes(100_000, seed=100), tmp_path, "100k")

    def test_500k(self, tmp_path):
        """500 KB random input."""
        _encode_decode_verify(_random_bytes(500_000, seed=500), tmp_path, "500k")

    # ── additional edge cases ─────────────────────────────────────

    def test_exact_block_boundary(self, tmp_path):
        """
        Payload whose size is an exact multiple of blocksize — exercises
        the last-block path where no zero-padding is needed.
        """
        raw = _random_bytes(40_000, seed=4)
        bs = auto_blocksize(len(raw))
        k = ceil(len(raw) / bs)
        _encode_decode_verify(raw[:k * bs], tmp_path, "exact_boundary",
                               compress=False)
