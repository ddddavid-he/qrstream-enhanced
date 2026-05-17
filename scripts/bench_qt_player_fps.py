#!/usr/bin/env python3
"""Benchmark the Qt display player at different target frame rates.

This script measures the playback path used by ``encode --display`` on the
current machine. It pre-fills a ``ModuleFrameCache`` so QR generation is not a
bottleneck, then drives the real Qt window and records:
- target fps
- effective frame-advance fps
- tick interval p50/p95
- display update cost p50/p95
- late-reset count/rate (player fell behind and reset its deadline)
- wall-clock drift vs theoretical media duration

Usage:
  uv run python scripts/bench_qt_player_fps.py --fps 30,60,90,120 --frames 800
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qrstream.display_cache import (  # noqa: E402
    DisplayProducerState,
    ModuleFrameCache,
    pack_module_image,
)
import qrstream.display_player_qt as qt_player  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


@dataclass
class BenchResult:
    ok: bool
    target_fps: int
    total_frames: int
    module_side: int
    expected_sec: float
    elapsed_sec: float
    effective_fps: float
    drift_sec: float
    tick_count: int
    tick_p50_ms: float
    tick_p95_ms: float
    display_p50_ms: float
    display_p95_ms: float
    late_reset_count: int
    late_reset_rate: float
    frames_advanced: int
    finished: bool
    frames_skipped: int = 0
    error: str = ""


class BenchWindow(qt_player._QRStreamWindow):
    def __init__(
        self,
        cache: ModuleFrameCache,
        state: DisplayProducerState,
        fps: int,
        config: qt_player.DisplayPlayerQtConfig,
        max_runtime_sec: float,
    ):
        self._bench_start_ts: float | None = None
        self._bench_end_ts: float | None = None
        self._tick_ts: list[float] = []
        self._display_costs: list[float] = []
        self._advance_ts: list[float] = []
        self._late_reset_count = 0
        self._finished = False
        self._frame_sequence: list[int] = []  # track presented frames
        super().__init__(cache, state, fps, config)
        self._safety = qt_player.QTimer(self)
        self._safety.setSingleShot(True)
        self._safety.timeout.connect(self._abort_due_timeout)
        self._safety.start(max(1000, int(max_runtime_sec * 1000)))
        qt_player.QTimer.singleShot(0, self._start_benchmark)

    def _start_benchmark(self) -> None:
        self._bench_start_ts = time.perf_counter()
        if self._cache.total_frames <= 1:
            self._finish(success=True)
            return
        self._playing = True
        self._play_btn.setText("⏸")
        self._next_frame_ts = time.monotonic() + self._frame_interval

    def _abort_due_timeout(self) -> None:
        self._finish(success=False)

    def _finish(self, success: bool) -> None:
        if self._bench_end_ts is not None:
            return
        self._finished = success
        self._bench_end_ts = time.perf_counter()
        self._state.request_cancel()
        self._timer.stop()
        self.close()

    def _update_display(self) -> None:
        t0 = time.perf_counter()
        super()._update_display()
        if self._bench_start_ts is not None:
            self._display_costs.append(time.perf_counter() - t0)

    def _tick(self) -> None:
        if self._bench_start_ts is not None:
            self._tick_ts.append(time.perf_counter())
        now = time.monotonic()

        if self._playing and now >= self._next_frame_ts:
            nxt = self._frame_index + 1
            if nxt >= self._cache.total_frames:
                self._playing = False
                self._play_btn.setText("▶")
            elif self._cache.has_frame(nxt):
                self._frame_index = nxt
                self._advance_ts.append(time.perf_counter())
                self._frame_sequence.append(nxt)
                # Accumulative + clamp (no skip, fast recovery)
                self._next_frame_ts += self._frame_interval
                if self._next_frame_ts < now:
                    self._next_frame_ts = now
            else:
                self._playing = False
                self._play_btn.setText("▶")

        can = qt_player._can_play(self._cache, self._state, self._frame_index,
                                  self._fps, self._config)
        if self._playing and not can:
            self._playing = False
            self._play_btn.setText("▶")

        self._update_controls()
        self._update_display()

        if self._frame_index >= self._cache.total_frames - 1 and not self._playing:
            self._finish(success=True)


def _build_cache(total_frames: int, module_side: int) -> ModuleFrameCache:
    cache = ModuleFrameCache(total_frames=total_frames, module_side=module_side)
    rng = np.random.default_rng(0)
    module_img = np.where(
        rng.random((module_side, module_side)) > 0.5,
        0,
        255,
    ).astype(np.uint8)
    packed = pack_module_image(module_img)
    for index in range(total_frames):
        cache.put_packed(index, packed)
    cache.mark_done()
    return cache


def _run_single(args: argparse.Namespace) -> BenchResult:
    qt_player.require_pyside6()
    app = qt_player.QApplication.instance()
    if app is None:
        app = qt_player.QApplication([])
        app.setApplicationName("QRStream Qt FPS Bench")

    cache = _build_cache(args.frames, args.module_side)
    state = DisplayProducerState(args.frames)
    state.mark_produced(args.frames)
    state.mark_done()

    config = qt_player.DisplayPlayerQtConfig(
        title=f"QRStream Qt Bench — {args.fps} fps",
        metadata=qt_player.DisplayMetadata(
            file_name="bench",
            total_frames=args.frames,
            module_side=args.module_side,
            fps=args.fps,
        ),
        lock_window_size=True,
        initial_screen_fraction=args.screen_fraction,
        ignore_saved_geometry=True,
    )
    window = BenchWindow(cache, state, args.fps, config, args.max_runtime_sec)
    window.show()
    app.exec()

    if window._bench_start_ts is None:
        raise RuntimeError("benchmark never started")
    end_ts = window._bench_end_ts or time.perf_counter()
    elapsed = max(1e-6, end_ts - window._bench_start_ts)
    frames_advanced = max(0, window._frame_index)
    expected = max(0.0, (args.frames - 1) / max(1, args.fps))
    tick_intervals = [
        (window._tick_ts[i] - window._tick_ts[i - 1]) * 1000.0
        for i in range(1, len(window._tick_ts))
    ]
    display_ms = [value * 1000.0 for value in window._display_costs]
    late_reset_rate = (
        window._late_reset_count / max(1, frames_advanced)
    )

    # Detect frame skips in the recorded sequence
    seq = window._frame_sequence
    frames_skipped = 0
    for i in range(1, len(seq)):
        gap = seq[i] - seq[i - 1]
        if gap != 1:
            frames_skipped += gap - 1

    return BenchResult(
        ok=True,
        target_fps=args.fps,
        total_frames=args.frames,
        module_side=args.module_side,
        expected_sec=expected,
        elapsed_sec=elapsed,
        effective_fps=frames_advanced / elapsed,
        drift_sec=elapsed - expected,
        tick_count=len(window._tick_ts),
        tick_p50_ms=_percentile(tick_intervals, 0.50),
        tick_p95_ms=_percentile(tick_intervals, 0.95),
        display_p50_ms=_percentile(display_ms, 0.50),
        display_p95_ms=_percentile(display_ms, 0.95),
        late_reset_count=window._late_reset_count,
        late_reset_rate=late_reset_rate,
        frames_advanced=frames_advanced,
        finished=window._finished,
        frames_skipped=frames_skipped,
    )


def _parse_fps_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("at least one fps value is required")
    return values


def _run_child_for_fps(script_path: Path, base_args: argparse.Namespace,
                       fps: int) -> BenchResult:
    cmd = [
        sys.executable,
        str(script_path),
        "--child",
        "--fps",
        str(fps),
        "--frames",
        str(base_args.frames),
        "--module-side",
        str(base_args.module_side),
        "--screen-fraction",
        str(base_args.screen_fraction),
        "--max-runtime-sec",
        str(base_args.max_runtime_sec),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"child exited {proc.returncode}")
    return BenchResult(**json.loads(proc.stdout))


def _print_summary(results: list[BenchResult], usable_threshold: float,
                   max_reset_rate: float) -> None:
    print(f"{'target':>6} {'eff':>8} {'done':>5} {'drift':>8} {'tick95':>8} {'draw95':>8} {'resets':>8} {'skips':>6} {'usable':>7}")
    print("-" * 78)
    usable_max = None
    for result in results:
        usable = (
            result.finished
            and result.effective_fps >= result.target_fps * usable_threshold
            and result.late_reset_rate <= max_reset_rate
            and result.frames_skipped == 0
        )
        if usable:
            usable_max = result.target_fps
        print(
            f"{result.target_fps:>6} "
            f"{result.effective_fps:>8.1f} "
            f"{('yes' if result.finished else 'no'):>5} "
            f"{result.drift_sec:>8.3f} "
            f"{result.tick_p95_ms:>8.2f} "
            f"{result.display_p95_ms:>8.2f} "
            f"{result.late_reset_count:>8} "
            f"{result.frames_skipped:>6} "
            f"{('yes' if usable else 'no'):>7}"
        )
    print()
    if usable_max is None:
        print("No tested fps met the usable threshold on this machine.")
    else:
        print(f"Maximum usable fps on this machine: {usable_max}")
        print(f"Rule: effective_fps >= {usable_threshold:.0%} target and late_reset_rate <= {max_reset_rate:.0%}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", default="30,60,90,120",
                        help="comma-separated target fps values")
    parser.add_argument("--frames", type=int, default=800,
                        help="frame count per run")
    parser.add_argument("--module-side", type=int, default=185,
                        help="module image side length")
    parser.add_argument("--screen-fraction", type=float, default=0.70,
                        help="initial window size as a fraction of screen")
    parser.add_argument("--max-runtime-sec", type=float, default=30.0,
                        help="abort a single run after this many seconds")
    parser.add_argument("--usable-threshold", type=float, default=0.95,
                        help="minimum effective_fps/target_fps ratio to count as usable")
    parser.add_argument("--max-reset-rate", type=float, default=0.05,
                        help="maximum late-reset rate to count as usable")
    parser.add_argument("--child", action="store_true",
                        help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.child:
        if isinstance(args.fps, str):
            fps_values = _parse_fps_list(args.fps)
            if len(fps_values) != 1:
                raise ValueError("child mode expects a single fps value")
            args.fps = fps_values[0]
        try:
            result = _run_single(args)
        except Exception as exc:
            result = BenchResult(
                ok=False,
                target_fps=int(args.fps),
                total_frames=args.frames,
                module_side=args.module_side,
                expected_sec=0.0,
                elapsed_sec=0.0,
                effective_fps=0.0,
                drift_sec=0.0,
                tick_count=0,
                tick_p50_ms=0.0,
                tick_p95_ms=0.0,
                display_p50_ms=0.0,
                display_p95_ms=0.0,
                late_reset_count=0,
                late_reset_rate=0.0,
                frames_advanced=0,
                finished=False,
                error=str(exc),
            )
        print(json.dumps(asdict(result)))
        return 0 if result.ok else 1

    fps_values = _parse_fps_list(args.fps)
    script_path = Path(__file__).resolve()
    results = [_run_child_for_fps(script_path, args, fps) for fps in fps_values]
    _print_summary(results, args.usable_threshold, args.max_reset_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
