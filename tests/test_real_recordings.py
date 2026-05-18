"""End-to-end smoke tests against real phone-recorded QR videos.

These are slow (seconds each, not milliseconds) and intentionally
excluded from the default ``pytest tests/`` run via the ``slow``
marker so the unit tests still complete in well under a second.

Because the captures take noticeable wall-clock time and exercise
OpenCV / WeChatQRCode rather than any Python-version-specific
logic, the project runs them in a **dedicated GitHub Actions
workflow** (``.github/workflows/real-world-tests.yml``) instead of
the per-Python-version unit-test matrix. Running the slow layer
once per architecture on Python 3.13 is enough to catch an
OpenCV / WeChatQR regression.

Invoke locally with either::

    uv run pytest -m slow -v
    uv run pytest tests/test_real_recordings.py -v

Fixtures live under ``tests/fixtures/`` in layered sub-dirs:

* ``real-phone-v4/`` — captures produced with the qrstream ≥ 0.8
  default path (``prng_version=1`` flag set; SplitMix64 mixer,
  GE rescue available).  Recorded at ``--overhead 1.5 --fps 10``
  then re-encoded with HEVC / CRF 32-36 / 720×720 / 12-15 fps to
  keep the repo footprint manageable.
* ``real-phone-current/`` — current codec fixtures: one LT and one
  RaptorQ phone recording of the deterministic first 1,000,000
  digits after π's decimal point.  These assert by SHA-256 only, so
  no source ``.input.bin`` needs to be committed.

See ``tests/fixtures/README.md`` for the full recording and
re-encoding procedure used to produce each case.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from qrstream.decoder import extract_qr_from_video, decode_blocks_to_file


_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Each entry declares (subdir, video, optional input_bin,
# expected_sha, expected_size).  Most historical fixtures commit the
# source ``.input.bin`` as the oracle.  Current pi fixtures instead
# assert against the deterministic decoded bytes by SHA-256 and size,
# avoiding a committed 1 MB source file per codec.
_GATING_CASES = [
    # real-phone-v4: the qrstream 0.8+ LT default path.
    pytest.param(
        "real-phone-v4", "v073-100kB.mp4",
        "v073-100kB.input.bin",
        "6fbf396baedd1233f4c8486e8a4a4cc43b9a1283e19ae4dcb3cd27c4ad4dbed2",
        102_400,
        id="v4-v073-100kB-V25-15fps-phone",
    ),
    pytest.param(
        "real-phone-v4", "v073-300kB.mp4",
        "v073-300kB.input.bin",
        "115e32de92187eb5cc544e04b5bb5ed953577d6c75489d8e4c1f2b1c374380fb",
        307_200,
        id="v4-v073-300kB-V25-12fps-phone",
    ),
    # Current codec fixtures: deterministic π digits, no source file.
    pytest.param(
        "real-phone-current", "v092-lt-pi-1MB.mp4",
        None,
        "7806ee47461b49ef1f578e14461b2c83c09c6d7a9a914275da1d71e9cbbf7069",
        1_000_000,
        id="current-lt-pi-1MB-phone",
    ),
    pytest.param(
        "real-phone-current", "v092-raptorq-pi-1MB.mp4",
        None,
        "7806ee47461b49ef1f578e14461b2c83c09c6d7a9a914275da1d71e9cbbf7069",
        1_000_000,
        id="current-raptorq-pi-1MB-phone",
    ),
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_case(subdir: str, video_name: str, input_name: str | None,
              expected_sha: str, expected_size: int) -> None:
    video_path = _FIXTURES_DIR / subdir / video_name
    input_path = _FIXTURES_DIR / subdir / input_name if input_name else None

    assert video_path.exists(), f"missing fixture video: {video_path}"
    if input_path is not None:
        assert input_path.exists(), f"missing fixture input: {input_path}"

        # Sanity gate: the committed input.bin must still hash to the
        # oracle value. If this fails, the test can't trust its own
        # ground truth.
        assert _sha256_file(input_path) == expected_sha, (
            f"fixture input {input_name} has drifted from its committed "
            f"SHA-256; the decoded-bytes assertion would be meaningless."
        )
        assert input_path.stat().st_size == expected_size

    # Decode: video → unique blocks → output bytes.
    blocks = extract_qr_from_video(
        str(video_path), sample_rate=0, verbose=False, workers=None)
    assert blocks, f"decoder returned no blocks for {video_name}"

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        written = decode_blocks_to_file(
            blocks, str(out_path), verbose=False)
        assert written == expected_size, (
            f"decoded size {written} != expected size {expected_size}")
        assert _sha256_file(out_path) == expected_sha, (
            f"decoded bytes do not match expected SHA-256 for "
            f"{video_name}")
    finally:
        if out_path.exists():
            out_path.unlink()


@pytest.mark.slow
@pytest.mark.parametrize(
    "subdir, video_name, input_name, expected_sha, expected_size",
    _GATING_CASES)
def test_phone_recording_roundtrip_gating(
    subdir: str,
    video_name: str,
    input_name: str | None,
    expected_sha: str,
    expected_size: int,
) -> None:
    """Gating end-to-end: any failure blocks the real-world test job.

    Guards against regressions in:
      - base45 / QR alphanumeric decode path
      - decoder's pipelined frame-read + worker-pool scheduling
      - WeChatQRCode integration / OpenCV version drift
      - LT belief-propagation + Gauss-Jordan rescue correctness
    """
    _run_case(subdir, video_name, input_name, expected_sha, expected_size)
