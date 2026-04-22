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

Fixtures live under ``tests/fixtures/`` in two layered sub-dirs:

* ``real-phone-v3/`` — captures produced with the qrstream ≤ 0.7
  protocol path (``prng_version=0`` flag cleared; LCG PRNG with
  5 warmup rounds).  Kept so the decoder's legacy-compat path
  stays covered even after v0.8+ makes ``prng_version=1`` the
  default.
* ``real-phone-v4/`` — captures produced with the qrstream ≥ 0.8
  default path (``prng_version=1`` flag set; SplitMix64 mixer,
  GE rescue available).  Recorded at ``--overhead 1.5 --fps 10``
  then re-encoded with HEVC / CRF 32-36 / 720×720 / 12-15 fps to
  keep the repo footprint manageable.

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

# Each entry declares (label, subdir, video, input_bin, expected_sha,
# strict). ``strict=False`` means a failure is reported but does not
# fail the job — used for legacy captures whose quality is known to
# be marginal and which we refuse to block releases on (see the
# v070 note below).
#
# SHA-256 values are computed against the committed ``.input.bin``
# files; if an input is ever regenerated the matching hash here must
# be updated as well or the test becomes a tautology.
_GATING_CASES = [
    pytest.param(
        "real-phone-v3", "v061.mp4",
        "v061.input.bin",
        "4a440b6da851a9a2e35eacca95b7b2fe29e3560c169b0a57211fccc2f5469443",
        id="v3-v061-30KB-V25-60fps-phone",
    ),
    # real-phone-v4: the qrstream 0.8+ default path. All three are
    # gating — if any of these regress, the fix has broken something
    # user-visible.
    pytest.param(
        "real-phone-v4", "v073-10kB.mp4",
        "v073-10kB.input.bin",
        "897d28b6b6e8540e08cb2e10f790a7cd40c84d56840e03349fef4a05a95ee8a4",
        id="v4-v073-10kB-V25-15fps-phone",
    ),
    pytest.param(
        "real-phone-v4", "v073-100kB.mp4",
        "v073-100kB.input.bin",
        "6fbf396baedd1233f4c8486e8a4a4cc43b9a1283e19ae4dcb3cd27c4ad4dbed2",
        id="v4-v073-100kB-V25-15fps-phone",
    ),
    pytest.param(
        "real-phone-v4", "v073-300kB.mp4",
        "v073-300kB.input.bin",
        "115e32de92187eb5cc544e04b5bb5ed953577d6c75489d8e4c1f2b1c374380fb",
        id="v4-v073-300kB-V25-12fps-phone",
    ),
]

# v070 is a known-marginal capture — an early-prototype phone
# recording of a 300 kB payload at a suboptimal distance / focus.
# It decodes on most hardware / OpenCV builds but occasionally
# misses ~0.5% of QR frames on bespoke ffmpeg builds, which can
# push it below the LT convergence threshold.  We keep it as a
# smoke signal of the decoder's worst-case behaviour but mark it
# ``xfail(strict=False)`` so a red result shows up in the workflow
# summary without blocking a release.  If you need to re-gate it,
# remove the ``strict=False`` line below.
_NON_GATING_CASES = [
    pytest.param(
        "real-phone-v3", "v070.mp4",
        "v070.input.bin",
        "2a20b62e35bf4b3a7f5fa4854397eeafea99d1efb8db38737cda4df55a4d5b8d",
        id="v3-v070-300KB-V25-30fps-phone",
        marks=pytest.mark.xfail(
            strict=False,
            reason="v070 is a low-quality capture kept as a "
                   "worst-case smoke signal; see the _NON_GATING_CASES "
                   "block in this file for details.",
        ),
    ),
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_case(subdir: str, video_name: str, input_name: str,
              expected_sha: str) -> None:
    video_path = _FIXTURES_DIR / subdir / video_name
    input_path = _FIXTURES_DIR / subdir / input_name

    assert video_path.exists(), f"missing fixture video: {video_path}"
    assert input_path.exists(), f"missing fixture input: {input_path}"

    # Sanity gate: the committed input.bin must still hash to the
    # oracle value. If this fails, the test can't trust its own
    # ground truth.
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "subdir, video_name, input_name, expected_sha", _GATING_CASES)
def test_phone_recording_roundtrip_gating(
    subdir: str, video_name: str, input_name: str, expected_sha: str,
) -> None:
    """Gating end-to-end: any failure blocks the real-world test job.

    Guards against regressions in:
      - base45 / QR alphanumeric decode path
      - decoder's pipelined frame-read + worker-pool scheduling
      - WeChatQRCode integration / OpenCV version drift
      - LT belief-propagation + Gauss-Jordan rescue correctness
    """
    _run_case(subdir, video_name, input_name, expected_sha)


@pytest.mark.slow
@pytest.mark.parametrize(
    "subdir, video_name, input_name, expected_sha", _NON_GATING_CASES)
def test_phone_recording_roundtrip_non_gating(
    subdir: str, video_name: str, input_name: str, expected_sha: str,
) -> None:
    """Non-gating end-to-end smoke: marked ``xfail(strict=False)``
    so a failure is visible in the test report but does not fail
    the workflow. See ``_NON_GATING_CASES`` for why each entry is
    here."""
    _run_case(subdir, video_name, input_name, expected_sha)
