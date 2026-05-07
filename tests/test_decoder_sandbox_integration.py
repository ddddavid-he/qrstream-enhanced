"""Tests for detect_isolation parameter behaviour after zxing-cpp migration.

Since qr_sandbox is deprecated and detect_isolation is now a no-op,
these tests verify:

  1. extract_qr_from_video still accepts 'on' and 'off' without error
     (but emits DeprecationWarning).
  2. Both modes produce correctly decoded output (same underlying detector).
  3. Invalid isolation modes are still rejected with ValueError.
  4. The DeprecationWarning is emitted whenever detect_isolation is passed.
"""

import pathlib
import warnings

import pytest

from qrstream.decoder import (
    LTDecoder,
    extract_qr_from_video,
)


FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "real-phone-v4" / "v073-10kB.mp4"
FIXTURE_INPUT = pathlib.Path(__file__).parent / "fixtures" / "real-phone-v4" / "v073-10kB.input.bin"


def _decode_blocks_to_bytes(blocks):
    dec = LTDecoder()
    for b in blocks:
        try:
            done, _ = dec.decode_bytes(b)
            if done:
                break
        except ValueError:
            continue
    if not dec.done:
        dec.try_gaussian_rescue()
    return dec.bytes_dump() if dec.done else None


@pytest.mark.slow
def test_extract_with_isolation_on_matches_off():
    if not FIXTURE.exists():
        pytest.skip("fixture video missing")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        blocks_off = extract_qr_from_video(
            str(FIXTURE), sample_rate=0, verbose=False,
            detect_isolation='off',
        )
        blocks_on = extract_qr_from_video(
            str(FIXTURE), sample_rate=0, verbose=False,
            detect_isolation='on',
        )

    out_off = _decode_blocks_to_bytes(blocks_off)
    out_on = _decode_blocks_to_bytes(blocks_on)

    assert out_off is not None, "isolation=off: LT decode failed"
    assert out_on is not None, "isolation=on: LT decode failed"
    expected = FIXTURE_INPUT.read_bytes()
    assert out_off == expected
    assert out_on == expected


def test_extract_rejects_invalid_isolation_mode(tmp_path):
    # ValueError is still raised for unknown values to catch typos.
    bogus = tmp_path / "does-not-exist.mp4"
    with pytest.raises(ValueError, match="detect_isolation"):
        extract_qr_from_video(
            str(bogus), sample_rate=0, verbose=False,
            detect_isolation='auto',
        )


def test_detect_isolation_on_emits_deprecation_warning(tmp_path):
    """Passing detect_isolation='on' must raise DeprecationWarning."""
    bogus = tmp_path / "does-not-exist.mp4"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            extract_qr_from_video(
                str(bogus), sample_rate=0, verbose=False,
                detect_isolation='on',
            )
        except FileNotFoundError:
            pass  # expected — bogus path
    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)
                 and "detect_isolation" in str(w.message).lower()]
    assert dep_warns, "Expected DeprecationWarning for detect_isolation='on'"


def test_detect_isolation_off_emits_deprecation_warning(tmp_path):
    """Passing detect_isolation='off' must also raise DeprecationWarning."""
    bogus = tmp_path / "does-not-exist.mp4"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            extract_qr_from_video(
                str(bogus), sample_rate=0, verbose=False,
                detect_isolation='off',
            )
        except FileNotFoundError:
            pass
    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)
                 and "detect_isolation" in str(w.message).lower()]
    assert dep_warns, "Expected DeprecationWarning for detect_isolation='off'"
