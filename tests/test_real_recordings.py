"""End-to-end smoke tests against real phone-recorded QR videos.

These are slow (seconds each, not milliseconds) and intentionally
excluded from the default ``pytest tests/`` run via the ``slow``
marker so the ~80 unit tests still complete in under a second.

Invoke explicitly with one of::

    uv run pytest -m slow -v
    uv run pytest tests/test_real_recordings.py -v

The CI workflow also runs these explicitly in a dedicated job.

Fixtures live in ``tests/fixtures/``; see that directory's README for
how they were recorded and re-encoded.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from qrstream.decoder import extract_qr_from_video, decode_blocks_to_file


_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Each entry: (video filename, input filename, expected sha256 of
# the decoded bytes (which must equal the sha256 of input.bin)).
#
# The SHA-256 values below are recomputed and committed together
# with the fixture files, so if you ever regenerate the inputs the
# decoded data MUST still match byte-for-byte.
_CASES = [
    pytest.param(
        "testcase-v061.mp4",
        "testcase-v061.input.bin",
        "4a440b6da851a9a2e35eacca95b7b2fe29e3560c169b0a57211fccc2f5469443",
        id="v061-30KB-V25-60fps-phone",
    ),
    pytest.param(
        "testcase-v070.mp4",
        "testcase-v070.input.bin",
        "2a20b62e35bf4b3a7f5fa4854397eeafea99d1efb8db38737cda4df55a4d5b8d",
        id="v070-300KB-V25-30fps-phone",
    ),
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.slow
@pytest.mark.parametrize("video_name, input_name, expected_sha", _CASES)
def test_phone_recording_roundtrip(
    video_name: str, input_name: str, expected_sha: str
) -> None:
    """Decode a real phone-captured recording and verify byte-exact match.

    Guards against regressions in:
      - base45 / QR alphanumeric decode path
      - decoder's pipelined frame-read + worker-pool scheduling
        (Tier 1.2 change in v0.7.0)
      - WeChatQRCode integration / OpenCV version drift
      - LT belief-propagation correctness over lossy input
    """
    video_path = _FIXTURES_DIR / video_name
    input_path = _FIXTURES_DIR / input_name

    assert video_path.exists(), f"missing fixture video: {video_path}"
    assert input_path.exists(), f"missing fixture input: {input_path}"

    # Sanity: the committed input.bin must still hash to the expected
    # value. If this fails, someone tampered with the fixture and the
    # test can't trust its own oracle.
    assert _sha256_file(input_path) == expected_sha, (
        f"fixture input {input_name} has drifted from its committed "
        f"SHA-256; the decoded-bytes assertion would be meaningless."
    )

    # Decode: video → unique blocks → output bytes.
    blocks = extract_qr_from_video(
        str(video_path), sample_rate=0, verbose=False, workers=None)
    assert blocks, f"decoder returned no blocks for {video_name}"

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        written = decode_blocks_to_file(
            blocks, str(out_path), verbose=False)
        assert written == input_path.stat().st_size, (
            f"decoded size {written} != input size "
            f"{input_path.stat().st_size}")
        assert _sha256_file(out_path) == expected_sha, (
            f"decoded bytes do not match expected SHA-256 for "
            f"{video_name}")
    finally:
        if out_path.exists():
            out_path.unlink()
