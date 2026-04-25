"""Integration tests for decoder ↔ sandbox wiring.

These tests verify:

  1. ``extract_qr_from_video(detect_isolation='on')`` and ``'off'``
     produce decoded block lists that reconstruct the same payload.
  2. The module-level ``_dispatch_detect`` hook is restored after each
     call, regardless of mode.
  3. Invalid isolation modes are rejected.
  4. A sandbox construction failure degrades gracefully to in-process
     detection rather than crashing the decode.
"""

import pathlib

import pytest

from qrstream import decoder as _decoder_mod
from qrstream.decoder import (
    LTDecoder,
    _in_process_detect,
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
    # Both should reconstruct the same original input file.
    expected = FIXTURE_INPUT.read_bytes()
    assert out_off == expected
    assert out_on == expected


@pytest.mark.slow
def test_dispatch_detect_is_restored_after_decode():
    if not FIXTURE.exists():
        pytest.skip("fixture video missing")

    assert _decoder_mod._dispatch_detect is _in_process_detect

    extract_qr_from_video(
        str(FIXTURE), sample_rate=0, verbose=False,
        detect_isolation='on',
    )
    assert _decoder_mod._dispatch_detect is _in_process_detect, (
        "dispatch hook was not restored after isolation='on' decode"
    )

    extract_qr_from_video(
        str(FIXTURE), sample_rate=0, verbose=False,
        detect_isolation='off',
    )
    assert _decoder_mod._dispatch_detect is _in_process_detect, (
        "dispatch hook was not restored after isolation='off' decode"
    )


def test_extract_rejects_invalid_isolation_mode(tmp_path):
    # Use a non-existent path so we fail fast at the validator step
    # before any I/O. The validator runs before VideoCapture.
    bogus = tmp_path / "does-not-exist.mp4"
    with pytest.raises(ValueError, match="detect_isolation"):
        extract_qr_from_video(
            str(bogus), sample_rate=0, verbose=False,
            detect_isolation='auto',
        )


@pytest.mark.slow
def test_extract_falls_back_when_sandbox_init_fails(monkeypatch):
    if not FIXTURE.exists():
        pytest.skip("fixture video missing")

    class _BoomSandbox:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated sandbox init failure")

    monkeypatch.setattr(
        _decoder_mod.qr_sandbox, "SandboxedDetector", _BoomSandbox
    )

    # Should not raise — should fall back to in-process detection and
    # still produce decodable blocks.
    blocks = extract_qr_from_video(
        str(FIXTURE), sample_rate=0, verbose=False,
        detect_isolation='on',
    )
    out = _decode_blocks_to_bytes(blocks)
    expected = FIXTURE_INPUT.read_bytes()
    assert out == expected
