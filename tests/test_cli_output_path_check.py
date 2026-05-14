"""Tests for the CLI's early ``-o / --output`` writability gate.

encode/decode are long-running jobs — a bad output path should
fail in the first second, not after a multi-minute probe + scan.
These tests exercise the ``_check_output_path_writable`` helper
plus its integration at the ``cmd_encode`` / ``cmd_decode`` entry
points.

The ``cmd_*`` tests stub out the actual encoder/decoder so we
measure the guard itself, not the heavy pipeline.
"""

from __future__ import annotations

import os
import stat

import pytest

from qrstream.cli import (
    _check_output_path_writable,
    _close_reporter,
    build_parser,
    cmd_decode,
    cmd_encode,
)


def _skip_if_mode_bits_are_not_enforced():
    if os.name == "nt":
        pytest.skip("Windows does not enforce POSIX chmod write bits")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses file-mode permission checks")


# ── _check_output_path_writable ────────────────────────────────


class TestOutputPathCheck:
    def test_writable_path_returns_none(self, tmp_path):
        # Clean writable directory + non-existent target: the
        # happy path every real invocation takes.
        target = tmp_path / "out.bin"
        assert _check_output_path_writable(str(target)) is None

    def test_empty_output_is_rejected(self):
        assert _check_output_path_writable("") == "output path is empty"

    def test_missing_parent_directory_is_rejected(self, tmp_path):
        # Use a nested path whose parent doesn't exist — we
        # deliberately *don't* mkdir -p so a typo is loud.
        bogus = tmp_path / "typo_dir" / "out.bin"
        msg = _check_output_path_writable(str(bogus))
        assert msg is not None
        assert "does not exist" in msg

    def test_parent_directory_not_writable(self, tmp_path):
        # Create a directory and drop its write bits.  This uses
        # POSIX mode semantics, so skip where those bits are not
        # enforced for the current process.
        _skip_if_mode_bits_are_not_enforced()
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x, no write
        try:
            target = ro / "out.bin"
            msg = _check_output_path_writable(str(target))
            assert msg is not None
            assert "not writable" in msg
        finally:
            ro.chmod(stat.S_IRWXU)  # restore so tmp cleanup works

    def test_output_is_existing_directory(self, tmp_path):
        # A naked directory path (user forgot the filename) must
        # be caught — otherwise encoder would blow up mid-run.
        msg = _check_output_path_writable(str(tmp_path))
        assert msg is not None
        assert "existing directory" in msg

    def test_existing_readonly_file_is_rejected(self, tmp_path):
        _skip_if_mode_bits_are_not_enforced()
        out = tmp_path / "out.bin"
        out.write_bytes(b"old")
        out.chmod(stat.S_IRUSR)  # read-only for owner
        try:
            msg = _check_output_path_writable(str(out))
            assert msg is not None
            assert "not writable" in msg
        finally:
            out.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_existing_writable_file_is_ok(self, tmp_path):
        # Overwriting a writable file is the normal "re-run with
        # same -o" flow — must succeed.
        out = tmp_path / "out.bin"
        out.write_bytes(b"old")
        assert _check_output_path_writable(str(out)) is None

    def test_relative_path_uses_cwd(self, tmp_path, monkeypatch):
        # A bare filename must be interpreted relative to CWD;
        # we shouldn't need the user to always type an absolute
        # path for the gate to work.
        monkeypatch.chdir(tmp_path)
        assert _check_output_path_writable("out.bin") is None


# ── cmd_encode / cmd_decode integration ────────────────────────


class TestCmdEncodeGate:
    def test_encode_fails_fast_on_missing_parent(
            self, tmp_path, capsys, monkeypatch):
        """cmd_encode must short-circuit *before* touching the
        encoder when the output parent is missing."""
        src = tmp_path / "src.bin"
        src.write_bytes(b"hello")

        # Stub the encoder so a regression that bypasses the gate
        # would at least be detectable (and fast).
        called: dict = {}
        import qrstream.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "encode_to_video",
            lambda **kw: called.setdefault("hit", True),
        )

        parser = build_parser()
        args = parser.parse_args([
            "encode", str(src),
            "-o", str(tmp_path / "missing_dir" / "out.mp4"),
            "--overhead", "2.0",
        ])

        with pytest.raises(SystemExit) as exc_info:
            cmd_encode(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "does not exist" in captured.err
        # The encoder MUST NOT have been reached.
        assert not called, (
            "cmd_encode called the encoder despite a bad output "
            "path — the gate must run first"
        )

    def test_encode_happy_path_passes_gate(
            self, tmp_path, monkeypatch):
        src = tmp_path / "src.bin"
        src.write_bytes(b"hello")
        out = tmp_path / "out.mp4"

        called: dict = {}
        import qrstream.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "encode_to_video",
            lambda **kw: called.setdefault("kw", kw),
        )

        parser = build_parser()
        args = parser.parse_args([
            "encode", str(src), "-o", str(out), "--overhead", "2.0",
        ])
        cmd_encode(args)
        assert called, "encoder was not invoked on the happy path"
        assert called["kw"]["output_path"] == str(out)

    def test_encode_still_rejects_exact_same_input_output_path(
            self, tmp_path, capsys, monkeypatch):
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"hello")

        called: dict = {}
        import qrstream.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "encode_to_video",
            lambda **kw: called.setdefault("hit", True),
        )

        parser = build_parser()
        args = parser.parse_args([
            "encode", str(src), "-o", str(src), "--codec", "h264",
            "--overhead", "2.0",
        ])

        with pytest.raises(SystemExit) as exc_info:
            cmd_encode(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "same as the input file" in captured.out
        assert not called

    def test_encode_display_only_skips_output_gate(
            self, tmp_path, monkeypatch):
        src = tmp_path / "src.bin"
        src.write_bytes(b"hello")

        called: dict = {}
        import qrstream.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "encode_to_display",
            lambda **kw: called.setdefault("kw", kw),
        )
        monkeypatch.setattr(
            enc_mod, "encode_to_video",
            lambda **kw: called.setdefault("video", kw),
        )

        parser = build_parser()
        args = parser.parse_args([
            "encode", str(src), "--display", "--overhead", "2.0",
        ])
        cmd_encode(args)

        assert "kw" in called
        assert "video" not in called
        assert called["kw"]["input_path"] == str(src)
        assert called["kw"]["output_path"] is None

    def test_encode_display_with_output_saves_after_display(
            self, tmp_path, monkeypatch):
        src = tmp_path / "src.bin"
        out = tmp_path / "out.mp4"
        src.write_bytes(b"hello")

        called: dict = {}
        import qrstream.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "encode_to_display",
            lambda **kw: called.setdefault("display", kw),
        )
        monkeypatch.setattr(
            enc_mod, "encode_to_video",
            lambda **kw: called.setdefault("video", kw),
        )

        parser = build_parser()
        args = parser.parse_args([
            "encode", str(src), "--display", "-o", str(out),
        ])
        cmd_encode(args)

        assert "display" in called
        assert "video" not in called
        assert called["display"]["output_path"] == str(out)
        assert called["display"]["codec"] == "h264"
        assert called["display"]["report_display_done"] is False

    def test_encode_without_output_defaults_to_display(
            self, tmp_path, monkeypatch):
        src = tmp_path / "src.bin"
        src.write_bytes(b"hello")

        called: dict = {}
        import qrstream.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "encode_to_display",
            lambda **kw: called.setdefault("display", kw),
        )
        monkeypatch.setattr(
            enc_mod, "encode_to_video",
            lambda **kw: called.setdefault("video", kw),
        )

        parser = build_parser()
        args = parser.parse_args(["encode", str(src)])
        cmd_encode(args)

        assert "display" in called
        assert "video" not in called
        assert called["display"]["output_path"] is None


class TestCmdDecodeGate:
    def test_decode_fails_fast_on_missing_parent(
            self, tmp_path, capsys, monkeypatch):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"not really a video")

        called: dict = {}
        import qrstream.decoder as dec_mod
        monkeypatch.setattr(
            dec_mod, "extract_qr_from_video",
            lambda *a, **k: called.setdefault("hit", True) or [],
        )
        monkeypatch.setattr(
            dec_mod, "decode_blocks_to_file",
            lambda *a, **k: called.setdefault("write", True) or 0,
        )

        parser = build_parser()
        args = parser.parse_args([
            "decode", str(video),
            "-o", str(tmp_path / "missing" / "out.bin"),
        ])

        with pytest.raises(SystemExit) as exc_info:
            cmd_decode(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "does not exist" in captured.err
        assert not called, (
            "cmd_decode ran extraction despite a bad output "
            "path — the gate must run first"
        )

    def test_decode_fails_fast_on_readonly_file(
            self, tmp_path, capsys, monkeypatch):
        _skip_if_mode_bits_are_not_enforced()
        video = tmp_path / "in.mp4"
        video.write_bytes(b"not really a video")
        out = tmp_path / "out.bin"
        out.write_bytes(b"old")
        out.chmod(stat.S_IRUSR)  # read-only

        called: dict = {}
        import qrstream.decoder as dec_mod
        monkeypatch.setattr(
            dec_mod, "extract_qr_from_video",
            lambda *a, **k: called.setdefault("hit", True) or [],
        )

        parser = build_parser()
        args = parser.parse_args([
            "decode", str(video), "-o", str(out),
        ])

        try:
            with pytest.raises(SystemExit) as exc_info:
                cmd_decode(args)
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "not writable" in captured.err
            assert not called
        finally:
            out.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_close_reporter_swallows_reporter_close_errors():
    class _BrokenReporter:
        def close(self):
            raise RuntimeError("boom")

    _close_reporter(_BrokenReporter())
