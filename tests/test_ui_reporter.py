"""Unit tests for qrstream.ui (progress/status reporter layer)."""

from __future__ import annotations

import io
import time

import pytest

from qrstream.ui import (
    LogReporter,
    OutputMode,
    QuietReporter,
    SlidingHitWindow,
    compute_block_map_cells,
    compute_range_strip_cells,
    render_block_map_plain,
    render_range_strip_plain,
    resolve_output_mode,
)


# ── SlidingHitWindow ─────────────────────────────────────────────


class TestSlidingHitWindow:
    def test_empty_ratio_is_zero(self):
        assert SlidingHitWindow().ratio == 0.0

    def test_push_and_ratio(self):
        w = SlidingHitWindow(capacity=4)
        for v in (True, False, True, True):
            w.push(v)
        assert w.ratio == pytest.approx(0.75)

    def test_capacity_caps_samples(self):
        w = SlidingHitWindow(capacity=3)
        for v in (True, True, True, False, False, False):
            w.push(v)
        assert w.ratio == pytest.approx(0.0)

    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError):
            SlidingHitWindow(capacity=0)


# ── Block map renderer ───────────────────────────────────────────


class TestBlockMap:
    def test_empty_inputs(self):
        assert compute_block_map_cells({}, k=0, width=10) == []
        assert compute_block_map_cells({}, k=10, width=0) == []

    def test_full_recovery_is_all_full_blocks(self):
        cells = compute_block_map_cells(set(range(10)), k=10, width=10)
        assert len(cells) == 10
        # All buckets 100% recovered → final tier (█, bright_green).
        for ch, style, density in cells:
            assert density == pytest.approx(1.0)
            assert ch == "█"
            assert style == "bright_green"

    def test_no_recovery_is_all_empty(self):
        cells = compute_block_map_cells(set(), k=10, width=10)
        for ch, style, density in cells:
            assert density == 0.0
            assert ch == "░"
            assert style == "grey35"

    def test_partial_recovery_bucketing(self):
        # k=10, width=5 → each bucket covers 2 source blocks.
        # Recovered = {0,1,2,3} means first 2 buckets are 100%, next
        # one is 0%, last two 0%.
        cells = compute_block_map_cells(
            {0, 1, 2, 3}, k=10, width=5)
        assert len(cells) == 5
        densities = [d for _, _, d in cells]
        assert densities == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0])

    def test_plain_map_string(self):
        assert render_block_map_plain({0, 1, 2, 3}, 10, 5) == "██░░░"


# ── Range strip renderer ─────────────────────────────────────────


class TestRangeStrip:
    def test_empty_width_returns_empty(self):
        assert compute_range_strip_cells([], 100, 0) == []

    def test_zero_total_frames_is_idle(self):
        cells = compute_range_strip_cells([(0, 10)], 0, 5)
        for ch, style in cells:
            assert ch == "·"
            assert style == "grey35"

    def test_current_range_overrides_pending(self):
        s = render_range_strip_plain(
            [(10, 20)], 100, 20, current=(10, 20))
        assert "▶" in s  # current marker present
        # The current range covers 10-20 (inclusive) over 100 frames on
        # a width-20 strip → cells 2..4 get the current marker.
        assert s[:2] == "··"

    def test_scanned_range_marks_done(self):
        s = render_range_strip_plain(
            [(40, 60)], 100, 20, scanned=[(0, 20)])
        assert "█" in s  # scanned (done) marker
        assert "▁" in s  # pending marker on segments


# ── Quiet reporter ───────────────────────────────────────────────


class TestQuietReporter:
    def test_quiet_emits_only_errors_and_save(self):
        buf = io.StringIO()
        r = QuietReporter(stream=buf)
        r.info("hello")
        r.debug("diag")
        r.probe_start()
        r.probe_done(sample=2, detect=0.7, repeat=2.0, crop_reduction=None)
        r.scan_update(video_pct=50.0, hit_window=0.5,
                      file_pct=30.0, recovered=set(), k=10)
        assert buf.getvalue() == ""

        r.warn("minor")
        r.error("boom")
        r.save_done(output_path="out.bin", bytes_written=1024)
        out = buf.getvalue()
        assert "Warning: minor" in out
        assert "Error: boom" in out
        assert "Saved: out.bin" in out


# ── Log reporter ─────────────────────────────────────────────────


class TestLogReporter:
    def _last_line(self, buf: io.StringIO) -> str:
        lines = [ln for ln in buf.getvalue().splitlines() if ln]
        return lines[-1] if lines else ""

    def test_log_emits_key_value_format(self):
        buf = io.StringIO()
        r = LogReporter(stream=buf)
        r.probe_start()
        r.probe_done(sample=2, detect=0.68, repeat=1.9,
                     crop_reduction=0.64)
        lines = buf.getvalue().splitlines()
        assert any("phase=probe status=start" in ln for ln in lines)
        done = next(ln for ln in lines if "status=done" in ln)
        assert "sample=2" in done
        assert "detect=68%" in done
        assert "repeat=1.9" in done
        assert "crop_reduction=64%" in done

    def test_log_throttles_scan_updates(self):
        buf = io.StringIO()
        r = LogReporter(stream=buf, pct_step=5.0, min_interval_sec=60.0)
        r.scan_start(total_frames=1000)
        # First update always emits; second at +1% within 60s must
        # NOT emit; third at +10% must emit.
        r.scan_update(video_pct=10.0, hit_window=0.5,
                      file_pct=10.0, recovered=set(), k=10)
        r.scan_update(video_pct=11.0, hit_window=0.5,
                      file_pct=10.5, recovered=set(), k=10)
        r.scan_update(video_pct=22.0, hit_window=0.5,
                      file_pct=20.0, recovered=set(), k=10)
        scans = [ln for ln in buf.getvalue().splitlines()
                 if "phase=scan" in ln and "video=" in ln]
        assert len(scans) == 2

    def test_log_escapes_values_with_spaces(self):
        buf = io.StringIO()
        r = LogReporter(stream=buf)
        r.save_done(output_path="path with spaces.bin",
                    bytes_written=42)
        line = self._last_line(buf)
        assert 'output="path with spaces.bin"' in line

    def test_log_verbose_includes_map_in_scan(self):
        buf = io.StringIO()
        r = LogReporter(stream=buf, verbose=True,
                        pct_step=0.0, min_interval_sec=0.0)
        r.scan_start(total_frames=10)
        r.scan_update(video_pct=50.0, hit_window=0.5,
                      file_pct=50.0,
                      recovered={0, 1, 2, 3, 4}, k=10)
        # At least one scan line should carry map=...
        assert any("map=" in ln for ln in buf.getvalue().splitlines())


# ── Resolver ─────────────────────────────────────────────────────


class TestResolver:
    def test_quiet_mode_returns_quiet(self):
        r = resolve_output_mode(OutputMode.QUIET, stderr_isatty=True)
        assert isinstance(r, QuietReporter)

    def test_log_mode_returns_log(self):
        r = resolve_output_mode(OutputMode.LOG, stderr_isatty=True)
        assert isinstance(r, LogReporter)

    def test_auto_mode_falls_back_to_log_when_not_tty(self):
        r = resolve_output_mode(OutputMode.AUTO, stderr_isatty=False)
        assert isinstance(r, LogReporter)

    def test_auto_mode_accepts_string(self):
        r = resolve_output_mode("quiet", stderr_isatty=True)
        assert isinstance(r, QuietReporter)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            resolve_output_mode("banana")

    def test_interactive_hard_fails_when_rich_missing(self, monkeypatch):
        """Explicit --output-mode=interactive must not silently become log."""
        import qrstream.ui as ui
        monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
        monkeypatch.setattr(
            ui, "_RICH_IMPORT_ERROR",
            ImportError("cannot import name 'TaskProgressColumn'"),
            raising=False,
        )
        with pytest.raises(RuntimeError) as exc_info:
            ui.resolve_output_mode(
                OutputMode.INTERACTIVE,
                stderr_isatty=True,
                explicit=True,
            )
        msg = str(exc_info.value)
        assert "rich" in msg.lower()
        assert "TaskProgressColumn" in msg  # reason surfaced

    def test_interactive_falls_back_when_not_explicit(self, monkeypatch):
        """Internal callers (explicit=False) still get a graceful fallback."""
        import qrstream.ui as ui
        monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
        monkeypatch.setattr(
            ui, "_RICH_IMPORT_ERROR", None, raising=False,
        )
        r = ui.resolve_output_mode(
            OutputMode.INTERACTIVE,
            stderr_isatty=True,
            explicit=False,
        )
        assert isinstance(r, LogReporter)

    def test_verbose_falls_back_to_log_verbose(self, monkeypatch, capsys):
        """Explicit verbose with no Rich → LogReporter(verbose=True) + warn."""
        import qrstream.ui as ui
        monkeypatch.setattr(ui, "_RICH_AVAILABLE", False)
        monkeypatch.setattr(
            ui, "_RICH_IMPORT_ERROR", ImportError("no rich"),
            raising=False,
        )
        r = ui.resolve_output_mode(
            OutputMode.VERBOSE,
            stderr_isatty=True,
            explicit=True,
        )
        assert isinstance(r, LogReporter)
        assert r._verbose is True
        # A single warning line should be emitted to stderr.
        captured = capsys.readouterr()
        assert "Rich UI" in captured.err
        assert "pip install" in captured.err
