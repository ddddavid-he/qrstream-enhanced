"""Tests for the CLI's ``--overhead`` floor/warning behaviour.

The LT codec cannot converge below ~1.2x overhead regardless of
capture quality. RaptorQ needs ~1.02x minimum.  The CLI must reject
values below the active codec's floor (exit 2) so users don't waste
a long encode on an un-decodable output.
"""

from __future__ import annotations

import sys

import pytest

from qrstream.cli import (
    _MIN_OVERHEAD,
    _MIN_OVERHEAD_LT,
    _MIN_OVERHEAD_RQ,
    _RECOMMENDED_OVERHEAD,
    _RECOMMENDED_OVERHEAD_LT,
    _RECOMMENDED_OVERHEAD_RQ,
    build_parser,
    cmd_encode,
)


def _args(overhead: float, input_path: str, output_path: str,
          fountain_codec: str = 'raptorq'):
    parser = build_parser()
    return parser.parse_args(
        ['encode', input_path, '-o', output_path,
         '--overhead', str(overhead),
         '--fountain-codec', fountain_codec]
    )


def test_cli_rejects_overhead_below_floor(tmp_path, capsys):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")
    out = tmp_path / "out.mp4"

    # Test with LT codec (original behaviour)
    args = _args(0.9, str(src), str(out), fountain_codec='lt')
    with pytest.raises(SystemExit) as exc_info:
        cmd_encode(args)
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert "below the LT codec" in captured.out


def test_cli_rejects_overhead_below_raptorq_floor(tmp_path, capsys):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")
    out = tmp_path / "out.mp4"

    args = _args(0.9, str(src), str(out), fountain_codec='raptorq')
    with pytest.raises(SystemExit) as exc_info:
        cmd_encode(args)
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert "below the RaptorQ codec" in captured.out


def test_cli_warns_between_floor_and_recommended(tmp_path, capsys, monkeypatch):
    """1.2 <= overhead < 1.5 is allowed for LT but must emit a warning."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 256)
    out = tmp_path / "out.mp4"

    # Stub the actual encoder — we only care about the CLI guard,
    # not about building a real video.
    called = {}

    def fake_encode(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("qrstream.cli.encode_to_video",
                         fake_encode, raising=False)
    # The encoder is imported lazily inside cmd_encode, so patch the
    # module-level binding that cmd_encode will resolve.
    import qrstream.encoder as enc_mod
    monkeypatch.setattr(enc_mod, "encode_to_video", fake_encode)

    args = _args(1.3, str(src), str(out), fountain_codec='lt')
    cmd_encode(args)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert called.get("overhead") == 1.3


def test_cli_silent_at_or_above_recommended(tmp_path, capsys, monkeypatch):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 256)
    out = tmp_path / "out.mp4"

    def fake_encode(**kwargs):
        pass

    import qrstream.encoder as enc_mod
    monkeypatch.setattr(enc_mod, "encode_to_video", fake_encode)

    args = _args(_RECOMMENDED_OVERHEAD_LT, str(src), str(out), fountain_codec='lt')
    cmd_encode(args)
    captured = capsys.readouterr()
    assert "Warning" not in captured.out
    assert "below the LT codec" not in captured.out


def test_cli_floor_matches_prng_v1_convergence_floor():
    """Pin down the CLI floor — if someone tweaks the mixer or the
    PRNG and convergence degrades, we want this test to force them
    to consciously revisit _MIN_OVERHEAD rather than let decoding
    silently regress below a safe number."""
    assert 1.20 <= _MIN_OVERHEAD < _RECOMMENDED_OVERHEAD
    assert _RECOMMENDED_OVERHEAD <= 2.0
