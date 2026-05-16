"""Unit tests for qrstream.ui (progress/status reporter layer)."""

from __future__ import annotations

import io
import time

import pytest

from qrstream.ui import (
    LogReporter,
    OutputMode,
    QuietReporter,
    RichReporter,
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
        r.ge_start(stage="scan", recovered=3, k=10)
        r.ge_done(success=False, recovered=3, k=10)
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

    def test_log_reports_ge_checkpoint(self):
        buf = io.StringIO()
        r = LogReporter(stream=buf)
        r.ge_start(stage="scan", recovered=4, k=10)
        r.ge_done(success=True, recovered=10, k=10)
        lines = buf.getvalue().splitlines()
        assert any("phase=ge status=start" in ln for ln in lines)
        assert any("stage=scan" in ln for ln in lines)
        done = next(ln for ln in lines if "phase=ge status=done" in ln)
        assert "result=success" in done
        assert "recovered=10" in done
        assert "total=10" in done


# ── Calibration progress integration ─────────────────────────────


class TestCalibrationProgressIntegration:
    def test_generate_calibration_output_uses_semantic_reporter(
            self, monkeypatch, tmp_path):
        import qrstream.calibrate as cal_mod

        events: list[tuple[str, dict]] = []

        class Recorder:
            def info(self, message):
                raise AssertionError(f"unexpected info progress: {message}")

            def calibrate_generate_start(self, **kw):
                events.append(("start", kw))

            def calibrate_generate_update(self, **kw):
                events.append(("update", kw))

            def calibrate_generate_done(self, **kw):
                events.append(("done", kw))

        monkeypatch.setattr(
            cal_mod,
            "_build_frame_sequence",
            lambda _config: [object(), object(), object()],
        )

        def fake_generate_video(config, frame_seq, output_path, codec, reporter):
            assert config.preset_name == "fast"
            assert len(frame_seq) == 3
            assert output_path == str(tmp_path / "cal.mp4")
            assert codec == "h264"
            reporter.calibrate_generate_update(progress_pct=100.0)

        monkeypatch.setattr(cal_mod, "_generate_video", fake_generate_video)

        out = str(tmp_path / "cal.mp4")
        cal_mod.generate_calibration(
            preset_name="fast",
            output_path=out,
            display_hz=60,
            reporter=Recorder(),
        )

        assert events == [
            ("start", {"preset": "fast", "total_frames": 3}),
            ("update", {"progress_pct": 100.0}),
            ("done", {"output_path": out}),
        ]


# ── Resolver ─────────────────────────────────────────────────────


class TestRichReporter:
    def test_scan_and_file_rows_align_in_interactive_output(self):
        buf = io.StringIO()
        r = RichReporter(stream=buf)
        r.scan_start(total_frames=100, total_blocks=32)
        r.scan_update(video_pct=100.0, hit_window=0.93,
                      file_pct=100.0,
                      recovered=set(range(32)), k=32)
        r.close()

        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        scan_line = [ln for ln in lines if ln.lstrip().startswith("Scan")][-1]
        file_line = [ln for ln in lines if ln.lstrip().startswith("File")][-1]

        def _first_bar_col(line: str) -> int:
            cols = [line.find(ch) for ch in "━█▓▒░" if ch in line]
            return min(col for col in cols if col >= 0)

        def _last_bar_col(line: str) -> int:
            cols = [line.rfind(ch) for ch in "━█▓▒░╸╺" if ch in line]
            return max(col for col in cols if col >= 0)

        # Label starts: Scan / File rows both begin at column 0 and
        # carry the padded label set by ``_pad_status_label``.
        assert scan_line.lstrip().startswith("Scan")
        assert file_line.lstrip().startswith("File")
        # The shared bar column must start at exactly the same
        # column on both rows — this is what "progress bars are
        # aligned" means to the user.
        assert _first_bar_col(scan_line) == _first_bar_col(file_line)
        # …and end at the same column too.  This guards against
        # regressions where one of the rows grows a wider stats
        # suffix and drags the bar shorter on that row alone.
        assert _last_bar_col(scan_line) == _last_bar_col(file_line)
        # Scan stats: at 100% the ETA/fps fields are suppressed
        # (no smoothed estimate available for an instant scan) so
        # only "det N%" is present — abbreviated from "detect" to
        # keep the stats cell inside the width budget.  The detect
        # percent is rendered with a fixed 3-character slot
        # ("  9%" / " 83%" / "100%") so the stats width stays
        # constant, so match with a regex rather than a literal.
        import re
        assert re.search(r"100\.0% \(det\s+93%\)", scan_line), (
            f"scan_line: {scan_line!r}"
        )
        # File stats: N/K blocks counter is shown next to the %.
        assert "32/32 blocks" in file_line
        assert "100.0%" in file_line

    def test_file_row_renders_as_pure_block_map(self):
        """File row is the qBittorrent-style block map, with NO
        extra tip / cursor / overlay glyph.

        Covers the regression where an earlier design added a
        1/8-cell "progress tip" on top of the map; users found it
        visually confusing and wanted the block map to speak for
        itself.  This test both guards against that regression and
        makes the "block map is the single visual indicator"
        contract explicit.
        """
        buf = io.StringIO()
        r = RichReporter(stream=buf)
        r.scan_start(total_frames=1000, total_blocks=32)
        r.scan_update(video_pct=5.0, hit_window=0.5,
                      file_pct=0.6,      # <1%: nothing recovered yet
                      recovered=set(), k=32)
        r.close()
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        file_line = next(ln for ln in lines
                         if ln.lstrip().startswith("File"))
        # qBittorrent-style block-map characters present.
        assert any(ch in file_line for ch in "░▒▓█")
        # No sub-cell tip glyphs (the removed 1/8-width blocks).
        assert not any(ch in file_line for ch in "▏▎▍▌▋▊▉"), (
            f"tip glyph leaked into File row: {file_line!r}"
        )
        # N/K counter reflects the recovered set size.
        assert "0/32 blocks" in file_line

    def test_file_row_full_recovery_uses_block_map_colours(self):
        """Full recovery → block-map paints ‘█’, nothing else."""
        buf = io.StringIO()
        r = RichReporter(stream=buf)
        r.scan_start(total_frames=100, total_blocks=16)
        r.scan_update(video_pct=100.0, hit_window=1.0,
                      file_pct=100.0,
                      recovered=set(range(16)), k=16)
        r.close()
        file_line = next(
            ln for ln in buf.getvalue().splitlines()
            if ln.lstrip().startswith("File")
        )
        # Filled buckets use '█' (top density tier) — the classic
        # qBittorrent "all chunks present" look.
        assert "█" in file_line
        # No tip / cursor glyphs.
        assert not any(ch in file_line for ch in "▏▎▍▌▋▊▉")
        assert "16/16 blocks" in file_line

    def test_scan_row_reports_eta_after_warmup(self):
        """After ≥1 s and >1% progress, Scan shows ETA + det%.

        fps is intentionally NOT surfaced in the Scan stats cell
        (it's still computed internally to feed the ETA EWMA).
        This test both verifies the positive case (ETA + det
        appear in the expected order) and locks in the "no fps"
        contract so a future refactor can't silently put it back.
        """
        import qrstream.ui as ui_mod

        buf = io.StringIO()
        r = RichReporter(stream=buf)
        r.scan_start(total_frames=1000)
        # Fast-forward the reporter's internal clock so the ETA
        # estimator has the warm-up it needs.  We monkey-patch
        # ``time.monotonic`` inside the ui module to keep the test
        # deterministic (no real sleeps).
        r._scan_started_at = 0.0  # anchor "start" at t=0
        real_monotonic = ui_mod.time.monotonic
        try:
            ui_mod.time.monotonic = lambda: 5.0  # type: ignore[assignment]
            r.scan_update(video_pct=25.0, hit_window=0.8,
                          file_pct=25.0,
                          recovered=set(range(4)), k=16)
        finally:
            ui_mod.time.monotonic = real_monotonic
        r.close()
        out = buf.getvalue()
        scan_line = [ln for ln in out.splitlines()
                     if ln.lstrip().startswith("Scan")][-1]
        # ETA + det appear inside the parenthesised stats.
        # det uses a fixed-width 3-char slot so the stats cell
        # doesn't jitter when the detect-rate digit count
        # changes — match with a regex rather than a literal.
        import re
        assert "ETA" in scan_line
        assert re.search(r"det\s+80%", scan_line), (
            f"scan_line: {scan_line!r}"
        )
        # fps is NOT rendered — neither the unit nor a bare
        # number before it.  Guard against regressions that would
        # re-add a derived metric users explicitly asked to drop.
        assert "fps" not in scan_line
        # Order contract: ETA precedes det.
        assert scan_line.index("ETA") < scan_line.index("det ")

    def test_scan_stats_width_constant_across_detect_rates(self):
        """det N% must occupy a fixed-width slot so the bar
        doesn't twitch as the detect rate crosses digit-count
        boundaries (9% → 10% → 100%).

        This test renders three scan_updates with wildly different
        detect rates at the same video percent and asserts that
        the bar end column and overall line length are identical
        on every frame.  A future change that drops the ``>3`` in
        ``det {hit*100:>3.0f}%`` would re-introduce the jitter
        and trip this test.
        """
        import qrstream.ui as ui_mod

        def render_scan_line(hit: float) -> str:
            buf = io.StringIO()
            r = RichReporter(stream=buf)
            # Pin console width so we're measuring the stats
            # column alone, not terminal resize.
            r._console.width = 120  # type: ignore[attr-defined]
            r._console._width = 120  # type: ignore[attr-defined]
            r.scan_start(total_frames=1000, total_blocks=100)
            r._scan_started_at = 0.0
            real = ui_mod.time.monotonic
            try:
                ui_mod.time.monotonic = lambda: 10.0  # type: ignore[assignment]
                r.scan_update(video_pct=25.0, hit_window=hit,
                              file_pct=10.0,
                              recovered=set(range(10)), k=100)
            finally:
                ui_mod.time.monotonic = real
            r.close()
            for ln in buf.getvalue().splitlines():
                if ln.lstrip().startswith("Scan"):
                    return ln
            raise AssertionError("Scan line not found")

        lines = [render_scan_line(h) for h in (0.05, 0.09, 0.83, 1.00)]
        lengths = {len(ln) for ln in lines}
        assert len(lengths) == 1, (
            f"scan lines have varying length across detect rates "
            f"(stats cell is jittering): {lengths}; lines={lines}"
        )
        # Bar end column must also match — guards against the
        # case where only trailing spaces change but the bar
        # itself is being recomputed.
        def _bar_end(ln: str) -> int:
            return max(ln.rfind("━"), ln.rfind("╺"), ln.rfind("╸"))

        bar_ends = {_bar_end(ln) for ln in lines}
        assert len(bar_ends) == 1, (
            f"bar right edge shifts across detect rates "
            f"(indicates width-jitter regression): {bar_ends}"
        )

    def test_ge_checkpoint_prints_simple_result(self):
        buf = io.StringIO()
        r = RichReporter(stream=buf)
        r.ge_start(stage="scan", recovered=4, k=10)
        r.ge_done(success=True, recovered=10, k=10)
        r.close()
        out = buf.getvalue()
        assert "GE" in out
        assert "recovered 10/10 blocks" in out

    def test_probe_done_emits_probe_and_plan_lines(self):
        """probe_done splits observations and plan into two lines."""
        buf = io.StringIO()
        r = RichReporter(stream=buf)
        r.probe_done(sample=1, detect=0.78, repeat=2.5,
                     crop_reduction=0.43,
                     observed=281, total_probed=360,
                     max_dim=1080)
        r.close()
        out = buf.getvalue()
        lines = [ln for ln in out.splitlines() if ln.strip()]
        probe_line = next(ln for ln in lines
                          if ln.lstrip().startswith("Probe"))
        plan_line = next(ln for ln in lines
                         if ln.lstrip().startswith("Plan"))
        # Observations on the Probe line.
        assert "281/360" in probe_line
        assert "detect 78%" in probe_line
        # Plan parameters on the second line, not mixed into the
        # first — this is the readability fix users asked for.
        assert "sample=1" in plan_line
        assert "repeat=2.5" in plan_line
        assert "-43%" in plan_line
        assert "1080px" in plan_line
        # Cross-check: the plan's parameters MUST NOT leak back
        # into the Probe line.
        assert "sample=" not in probe_line
        assert "repeat=" not in probe_line

    def test_probe_done_without_optional_fields_is_graceful(self):
        """Legacy probe_done calls (no observed / max_dim) still render."""
        buf = io.StringIO()
        r = RichReporter(stream=buf)
        r.probe_done(sample=2, detect=0.5, repeat=1.8,
                     crop_reduction=None)
        r.close()
        out = buf.getvalue()
        assert "Probe" in out
        assert "Plan" in out
        assert "detect 50%" in out
        assert "crop=off" in out or "crop=" in out
        assert "max_dim=" not in out  # only shown when provided


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
