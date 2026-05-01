"""
Unified progress / status UI layer for qrstream.

This module owns ALL user-facing progress, phase, and status output
produced by the encoder / decoder pipelines.  Business functions do
not call ``print`` or ``tqdm`` directly — they emit semantic events
(``probe_start``, ``scan_update``, ``encode_done``, …) through a
:class:`ProgressReporter` protocol object, and three concrete
implementations decide how to render them:

* :class:`RichReporter` — animated, colour, pip-style thin progress
  bars plus a qBittorrent-style block map for file recovery.  Also
  handles ``verbose`` mode on TTY (debug lines are routed through
  :meth:`rich.console.Console.log`).

* :class:`LogReporter` — single-line, appendable ``key=value``
  records for CI / ``tee`` / remote log capture; throttles
  ``scan_update`` / ``recover_update`` / ``encode_update`` to avoid
  spamming log files.

* :class:`QuietReporter` — emits errors and the final success line
  only; used for scripted invocations.

Error / warning messages that must remain captured by ``capsys`` in
existing tests continue to be ``print``-ed by ``cli.py`` directly —
the reporter ``error`` / ``warn`` methods are layered on top and
route to ``stderr`` for interactive / log mode.

Concrete renderers encapsulated here:

* :class:`SlidingHitWindow` — fixed-length ``deque`` tracking
  per-frame detect success for the *hit* metric shown beside the
  Scan / Recover bars.
* :func:`render_block_map` — source-block location map (colourised
  via Rich ``Text`` spans when available).
* :func:`render_range_strip` — targeted-recovery segment strip
  projected onto the video timeline.
"""

from __future__ import annotations

import enum
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

_RICH_IMPORT_ERROR: Exception | None = None
try:  # Rich is a hard dependency (declared in pyproject.toml).
    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text
    _RICH_AVAILABLE = True
except Exception as _exc:  # pragma: no cover — rich missing/old → log/quiet
    _RICH_AVAILABLE = False
    _RICH_IMPORT_ERROR = _exc


# ── Enums ────────────────────────────────────────────────────────


class OutputMode(str, enum.Enum):
    """User-selectable output modes (``--output-mode`` on the CLI).

    ``AUTO`` resolves at runtime: interactive on TTY, log otherwise.
    """

    AUTO = "auto"
    INTERACTIVE = "interactive"
    LOG = "log"
    QUIET = "quiet"
    VERBOSE = "verbose"


class Phase(str, enum.Enum):
    PROBE = "probe"
    SCAN = "scan"
    RECOVER = "recover"
    SAVE = "save"
    ENCODE = "encode"


# ── Sliding hit window ───────────────────────────────────────────


class SlidingHitWindow:
    """Fixed-length 0/1 deque tracking recent detect success.

    Used by Scan / Recover to show the ``hit`` metric — a windowed
    average of "QR decoded this frame?" over the last *N* processed
    frames.
    """

    __slots__ = ("_samples",)

    def __init__(self, capacity: int = 128):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._samples: deque[int] = deque(maxlen=capacity)

    def push(self, hit: bool) -> None:
        self._samples.append(1 if hit else 0)

    @property
    def ratio(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def clear(self) -> None:
        self._samples.clear()


# ── Block map / Range strip renderers ────────────────────────────


# Density thresholds for bucket rendering.  Matches the design doc's
# 0 / ≤33 / ≤66 / ≤99 / 100 buckets.
_DENSITY_CHAR_AND_STYLE: tuple[tuple[float, str, str], ...] = (
    (0.0,    "░", "grey35"),
    (0.33,   "▒", "blue"),
    (0.66,   "▓", "cyan"),
    (0.999,  "▓", "green"),
    (1.0,    "█", "bright_green"),
)


def _density_cell(density: float) -> tuple[str, str]:
    """Return ``(char, rich_style)`` for a bucket density in [0, 1]."""
    density = max(0.0, min(1.0, density))
    for thresh, ch, style in _DENSITY_CHAR_AND_STYLE:
        if density <= thresh:
            return ch, style
    return _DENSITY_CHAR_AND_STYLE[-1][1], _DENSITY_CHAR_AND_STYLE[-1][2]


def compute_block_map_cells(
    recovered: Iterable[int] | set[int] | dict[int, object],
    k: int,
    width: int,
) -> list[tuple[str, str, float]]:
    """Compute ``(char, style, density)`` cells for a block map.

    Parameters
    ----------
    recovered
        Container of recovered source-block indices (``set`` /
        ``dict`` / iterable).  Membership test via ``in``.
    k
        Total number of source blocks.
    width
        Target number of cells (terminal columns) to render.

    Notes
    -----
    The returned cell count always matches ``width`` so the visual
    block strip can align with progress bars even when ``k`` is much
    smaller than the available terminal width.
    """
    if k <= 0 or width <= 0:
        return []

    if isinstance(recovered, dict):
        recovered_set = recovered  # dict supports ``in`` on keys
    else:
        recovered_set = (
            recovered if hasattr(recovered, "__contains__")
            else set(recovered)
        )

    cells: list[tuple[str, str, float]] = []
    scale = k / width
    for cell_idx in range(width):
        start = cell_idx * scale
        end = (cell_idx + 1) * scale
        lo = int(math.floor(start))
        hi = min(k - 1, int(math.ceil(end) - 1))
        covered = 0.0
        for block_idx in range(lo, hi + 1):
            overlap = min(end, block_idx + 1.0) - max(start, float(block_idx))
            if overlap > 0.0 and block_idx in recovered_set:
                covered += overlap
        density = covered / scale if scale > 0 else 0.0
        ch, style = _density_cell(density)
        cells.append((ch, style, density))
    return cells


def render_block_map_plain(
    recovered: Iterable[int] | set[int] | dict[int, object],
    k: int,
    width: int,
) -> str:
    """Plain-text block map (no colour codes) for log mode."""
    cells = compute_block_map_cells(recovered, k, width)
    return "".join(ch for ch, _, _ in cells)


def render_block_map_rich(
    recovered: Iterable[int] | set[int] | dict[int, object],
    k: int,
    width: int,
) -> "Text":
    """Rich ``Text`` object with per-cell colour spans."""
    cells = compute_block_map_cells(recovered, k, width)
    text = Text()
    for ch, style, _ in cells:
        text.append(ch, style=style)
    return text


# Range-strip characters.  Keep it short and terminal-safe.
_RANGE_IDLE = "·"
_RANGE_PENDING = "▁"
_RANGE_CURRENT = "▶"
_RANGE_DONE = "█"


def compute_range_strip_cells(
    segments: Sequence[tuple[int, int]],
    total_frames: int,
    width: int,
    *,
    current: tuple[int, int] | None = None,
    scanned: Iterable[tuple[int, int]] = (),
) -> list[tuple[str, str]]:
    """Project recovery segments onto a fixed-width video timeline.

    Parameters
    ----------
    segments
        Pending / planned recovery ranges as ``(start, end)``
        inclusive frame indices.
    total_frames
        Length of the source video in frames.
    width
        Target number of cells.
    current
        Active range being scanned (rendered with ``current`` style).
    scanned
        Already-finished ranges (rendered with ``done`` style).
    """
    if width <= 0:
        return []
    if total_frames <= 0:
        return [(_RANGE_IDLE, "grey35") for _ in range(width)]

    cells: list[tuple[str, str]] = [(_RANGE_IDLE, "grey35") for _ in range(width)]

    def _mark(start: int, end: int, ch: str, style: str) -> None:
        lo = max(0, min(total_frames - 1, start))
        hi = max(0, min(total_frames - 1, end))
        if hi < lo:
            return
        a = int(lo * width // total_frames)
        b = int(hi * width // total_frames)
        a = max(0, min(width - 1, a))
        b = max(0, min(width - 1, b))
        for i in range(a, b + 1):
            cells[i] = (ch, style)

    # Planning order: pending < scanned < current so current wins.
    for s, e in segments:
        _mark(s, e, _RANGE_PENDING, "yellow")
    for s, e in scanned:
        _mark(s, e, _RANGE_DONE, "green")
    if current is not None:
        _mark(current[0], current[1], _RANGE_CURRENT, "bright_cyan")
    return cells


def render_range_strip_plain(
    segments: Sequence[tuple[int, int]],
    total_frames: int,
    width: int,
    *,
    current: tuple[int, int] | None = None,
    scanned: Iterable[tuple[int, int]] = (),
) -> str:
    cells = compute_range_strip_cells(
        segments, total_frames, width,
        current=current, scanned=scanned,
    )
    return "".join(ch for ch, _ in cells)


def render_range_strip_rich(
    segments: Sequence[tuple[int, int]],
    total_frames: int,
    width: int,
    *,
    current: tuple[int, int] | None = None,
    scanned: Iterable[tuple[int, int]] = (),
) -> "Text":
    cells = compute_range_strip_cells(
        segments, total_frames, width,
        current=current, scanned=scanned,
    )
    text = Text()
    for ch, style in cells:
        text.append(ch, style=style)
    return text


# ── Reporter protocol ────────────────────────────────────────────


class ProgressReporter(Protocol):
    """Semantic progress/status event sink.

    Reporter methods must be cheap (called per-frame in hot paths)
    and side-effect-safe; concrete implementations absorb exceptions
    internally so UI issues never interrupt encode/decode.
    """

    # Generic
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def debug(self, message: str) -> None: ...
    def close(self) -> None: ...

    # Decode
    def probe_start(self) -> None: ...
    def probe_update(self, *, scanned: int, total: int,
                     detect: float, phase: str) -> None: ...
    def probe_done(self, *, sample: int, detect: float,
                   repeat: float,
                   crop_reduction: float | None) -> None: ...

    def scan_start(self, *, total_frames: int,
                   total_blocks: int | None = None) -> None: ...
    def scan_update(self, *, video_pct: float, hit_window: float,
                    file_pct: float,
                    recovered: object,
                    k: int | None) -> None: ...
    def scan_done(self) -> None: ...

    def recover_start(self, *, level: str,
                      segments: Sequence[tuple[int, int]],
                      total_frames: int) -> None: ...
    def recover_update(self, *, progress_pct: float, hit_window: float,
                       file_pct: float, recovered: object,
                       k: int | None,
                       current_range: tuple[int, int] | None) -> None: ...
    def recover_done(self) -> None: ...

    def save_done(self, *, output_path: str, bytes_written: int) -> None: ...

    # Encode
    def encode_start(self, *, duration_sec: float, fps: int,
                     qr_version: int, mode: str,
                     overhead: float) -> None: ...
    def encode_update(self, *, progress_pct: float,
                      speed_fps: float,
                      eta_sec: float) -> None: ...
    def encode_done(self, *, output_path: str,
                    size_bytes: int) -> None: ...


# ── Helpers ──────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _fmt_size(n: int) -> str:
    if n is None or n < 0:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB")
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    if i == 0:
        return f"{int(x)} {units[i]}"
    return f"{x:.1f} {units[i]}"


def _fmt_pct(x: float) -> str:
    return f"{max(0.0, min(100.0, x)):.1f}%"


_STATUS_LABEL_WIDTH = 8


def _pad_status_label(label: str) -> str:
    """Return a fixed-width left-aligned label for status rows."""
    return f"{label:<{_STATUS_LABEL_WIDTH}}"


def _pad_stacked_status_label(label: str) -> str:
    """Match the extra inter-column gap Rich inserts after progress labels."""
    return _pad_status_label(label) + " "


def _log_escape(value: object) -> str:
    """Escape a value for ``key=value`` log output."""
    s = str(value)
    if not s:
        return '""'
    if any(c in s for c in (" ", "=", '"', "\t")):
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s


# ── Quiet reporter ───────────────────────────────────────────────


class QuietReporter:
    """Emits nothing but errors and one final success line."""

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stderr

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            if not text.endswith("\n"):
                self._stream.write("\n")
            self._stream.flush()
        except Exception:
            pass

    # Generic ------------------------------------------------------
    def info(self, message: str) -> None:  # pragma: no cover — silent
        return

    def warn(self, message: str) -> None:
        self._write(f"Warning: {message}")

    def error(self, message: str) -> None:
        self._write(f"Error: {message}")

    def debug(self, message: str) -> None:  # pragma: no cover — silent
        return

    def close(self) -> None:  # pragma: no cover — nothing to close
        return

    # Decode -------------------------------------------------------
    def probe_start(self) -> None: return
    def probe_update(self, **_kw) -> None: return
    def probe_done(self, **_kw) -> None: return
    def scan_start(self, **_kw) -> None: return
    def scan_update(self, **_kw) -> None: return
    def scan_done(self) -> None: return
    def recover_start(self, **_kw) -> None: return
    def recover_update(self, **_kw) -> None: return
    def recover_done(self) -> None: return

    def save_done(self, *, output_path: str, bytes_written: int) -> None:
        self._write(f"Saved: {output_path} ({_fmt_size(bytes_written)})")

    # Encode -------------------------------------------------------
    def encode_start(self, **_kw) -> None: return
    def encode_update(self, **_kw) -> None: return

    def encode_done(self, *, output_path: str, size_bytes: int) -> None:
        self._write(f"Encoded: {output_path} ({_fmt_size(size_bytes)})")


# ── Log reporter (key=value) ─────────────────────────────────────


@dataclass
class _ThrottleState:
    last_ts: float = 0.0
    last_pct: float = -1.0


class LogReporter:
    """Append-only ``key=value`` line log.

    Throttles per-frame updates to at most once every 2s or every 5%
    progress delta (whichever comes first).  ``phase=...
    status=start|done`` events always emit.  In ``verbose`` mode the
    throttle is relaxed (min 0.5s) and ``map`` / ``debug`` keys are
    included.
    """

    def __init__(self, stream=None, *, verbose: bool = False,
                 pct_step: float = 5.0,
                 min_interval_sec: float = 2.0):
        self._stream = stream if stream is not None else sys.stderr
        self._verbose = verbose
        self._pct_step = pct_step
        self._min_interval = 0.5 if verbose else min_interval_sec
        self._state: dict[str, _ThrottleState] = {}

    # ── internal helpers ─────────────────────────────────────
    def _timestamp(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime())

    def _write_line(self, **fields) -> None:
        try:
            parts = [f"{k}={_log_escape(v)}"
                     for k, v in fields.items() if v is not None]
            line = f"[{self._timestamp()}] " + " ".join(parts)
            self._stream.write(line)
            if not line.endswith("\n"):
                self._stream.write("\n")
            self._stream.flush()
        except Exception:
            pass

    def _should_emit(self, key: str, pct: float) -> bool:
        now = time.monotonic()
        state = self._state.setdefault(key, _ThrottleState())
        if state.last_pct < 0:
            state.last_ts = now
            state.last_pct = pct
            return True
        if abs(pct - state.last_pct) >= self._pct_step:
            state.last_ts = now
            state.last_pct = pct
            return True
        if (now - state.last_ts) >= self._min_interval:
            state.last_ts = now
            state.last_pct = pct
            return True
        return False

    # ── generic ───────────────────────────────────────────────
    def info(self, message: str) -> None:
        self._write_line(event="info", msg=message)

    def warn(self, message: str) -> None:
        self._write_line(event="warn", msg=message)

    def error(self, message: str) -> None:
        self._write_line(event="error", msg=message)

    def debug(self, message: str) -> None:
        if self._verbose:
            self._write_line(event="debug", msg=message)

    def close(self) -> None:
        return

    # ── decode ────────────────────────────────────────────────
    def probe_start(self) -> None:
        self._write_line(phase="probe", status="start")

    def probe_update(self, *, scanned: int, total: int,
                     detect: float, phase: str) -> None:
        if not self._should_emit("probe", scanned / total * 100 if total else 0):
            return
        self._write_line(phase="probe", status=phase,
                         scanned=scanned, total=total,
                         detect=f"{detect * 100:.0f}%")

    def probe_done(self, *, sample: int, detect: float, repeat: float,
                   crop_reduction: float | None) -> None:
        fields = {
            "phase": "probe",
            "status": "done",
            "sample": sample,
            "detect": f"{detect * 100:.0f}%",
            "repeat": f"{repeat:.1f}",
        }
        if crop_reduction is None:
            fields["crop_reduction"] = "off"
        else:
            fields["crop_reduction"] = f"{crop_reduction * 100:.0f}%"
        self._write_line(**fields)

    def scan_start(self, *, total_frames: int,
                   total_blocks: int | None = None) -> None:
        fields = {"phase": "scan", "status": "start",
                  "total_frames": total_frames}
        if total_blocks is not None:
            fields["total_blocks"] = total_blocks
        self._write_line(**fields)
        self._state.pop("scan", None)

    def scan_update(self, *, video_pct: float, hit_window: float,
                    file_pct: float, recovered: object,
                    k: int | None) -> None:
        if not self._should_emit("scan", video_pct):
            return
        fields = {
            "phase": "scan",
            "video": _fmt_pct(video_pct),
            "file": _fmt_pct(file_pct),
            "hit_window": f"{hit_window * 100:.0f}%",
        }
        if self._verbose and k and k > 0:
            try:
                term_cols = os.get_terminal_size().columns
            except (OSError, ValueError):
                term_cols = 80
            width = min(96, max(16, term_cols - 24))
            fields["map"] = render_block_map_plain(recovered, k, width)
        self._write_line(**fields)

    def scan_done(self) -> None:
        self._write_line(phase="scan", status="done")

    def recover_start(self, *, level: str,
                      segments: Sequence[tuple[int, int]],
                      total_frames: int) -> None:
        self._write_line(phase="recover", status="start",
                         level=level, segments=len(segments),
                         total_frames=total_frames)
        self._state.pop("recover", None)

    def recover_update(self, *, progress_pct: float, hit_window: float,
                       file_pct: float, recovered: object,
                       k: int | None,
                       current_range: tuple[int, int] | None) -> None:
        if not self._should_emit("recover", progress_pct):
            return
        fields = {
            "phase": "recover",
            "progress": _fmt_pct(progress_pct),
            "file": _fmt_pct(file_pct),
            "hit_window": f"{hit_window * 100:.0f}%",
        }
        if self._verbose and k and k > 0:
            fields["map"] = render_block_map_plain(recovered, k, 48)
        self._write_line(**fields)

    def recover_done(self) -> None:
        self._write_line(phase="recover", status="done")

    def save_done(self, *, output_path: str, bytes_written: int) -> None:
        self._write_line(phase="save", status="done",
                         output=output_path, bytes=bytes_written)

    # ── encode ────────────────────────────────────────────────
    def encode_start(self, *, duration_sec: float, fps: int,
                     qr_version: int, mode: str, overhead: float) -> None:
        self._write_line(phase="encode", status="start",
                         duration=f"{duration_sec:.1f}s",
                         fps=fps, qr=f"v{qr_version}",
                         mode=mode, overhead=f"{overhead:.1f}x")
        self._state.pop("encode", None)

    def encode_update(self, *, progress_pct: float, speed_fps: float,
                      eta_sec: float) -> None:
        if not self._should_emit("encode", progress_pct):
            return
        self._write_line(phase="encode",
                         progress=_fmt_pct(progress_pct),
                         speed=f"{speed_fps:.1f}fps",
                         eta=_fmt_duration(eta_sec))

    def encode_done(self, *, output_path: str, size_bytes: int) -> None:
        self._write_line(phase="encode", status="done",
                         output=output_path, size=_fmt_size(size_bytes))


# ── Rich reporter ────────────────────────────────────────────────


if _RICH_AVAILABLE:

    class _HitColumn(ProgressColumn):
        """Show sliding-window detection rate: ``(detect 93%)``."""

        def render(self, task):  # type: ignore[override]
            hit = task.fields.get("hit") if task.fields else None
            if hit is None:
                return Text("", style="dim")
            return Text(f"(detect {hit * 100:.0f}%)", style="bright_cyan")


    class _EncodeStatsColumn(ProgressColumn):
        """Show fps + ETA in parens, comma-separated: ``(61.0 fps, ETA 00:05)``."""

        def render(self, task):  # type: ignore[override]
            fields = task.fields or {}
            fps = fields.get("fps")
            eta_override = fields.get("eta_override")
            if fps is None and eta_override is None:
                return Text("", style="dim")
            parts: list[str] = []
            if fps is not None:
                parts.append(f"{fps:.1f} fps")
            if eta_override is not None:
                parts.append(f"ETA {_fmt_duration(eta_override)}")
            return Text(f"({', '.join(parts)})", style="bright_magenta")

else:  # pragma: no cover — exercised only when rich is unavailable
    _HitColumn = _EncodeStatsColumn = None  # type: ignore[assignment]


class RichReporter:
    """Animated Rich-driven reporter (interactive / verbose-on-tty).

    Scan / Recover / Encode all use a short pip/rich-style thin bar.
    File recovery is rendered as a wide coloured block map under
    the bar; targeted recovery adds a segment strip.
    """

    _MAP_MIN_WIDTH = 24
    _MAP_MAX_WIDTH = 80
    _MAP_LABEL = _pad_stacked_status_label("File")
    _RANGE_LABEL = _pad_stacked_status_label("Range")
    # Throttle map updates so per-frame scan callbacks don't churn
    # the Live renderer when the bucket output hasn't changed.
    _MAP_QUANT_BUCKETS = 200.0  # 0.5% quantisation on file pct

    def __init__(self, *, verbose: bool = False, stream=None):
        if not _RICH_AVAILABLE:
            raise RuntimeError("rich is required for RichReporter; "
                               "install with `pip install rich`.")
        self._console = Console(
            file=stream if stream is not None else sys.stderr,
            stderr=stream is None,
            highlight=False,
            soft_wrap=False,
        )
        self._verbose = verbose
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: int | None = None
        # Cached last state for file / range maps.
        self._file_map_text: Text | None = None
        self._range_map_text: Text | None = None
        self._last_map_bucket: int = -1
        self._last_map_k: int = -1
        self._last_map_recovered_len: int = -1
        # Recover state
        self._recover_segments: Sequence[tuple[int, int]] = ()
        self._recover_total_frames: int = 0
        self._recover_scanned: list[tuple[int, int]] = []
        self._probe_spinner_progress: Progress | None = None
        self._probe_task_id: int | None = None

    # ── internal ──────────────────────────────────────────────
    def _map_width(self) -> int:
        try:
            term_w = self._console.size.width
        except Exception:
            term_w = 80
        # Leave room for label + trailing ' 37.8%'
        body = max(self._MAP_MIN_WIDTH, term_w - len(self._MAP_LABEL) - 8)
        return min(self._MAP_MAX_WIDTH, body)

    def _stop_live(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        self._progress = None
        self._task_id = None
        self._file_map_text = None
        self._range_map_text = None
        self._last_map_bucket = -1
        self._last_map_k = -1
        self._last_map_recovered_len = -1

    def _build_group(self) -> Group:
        parts: list[object] = []
        if self._progress is not None:
            parts.append(self._progress)
        if self._range_map_text is not None:
            range_line = Text()
            range_line.append(self._RANGE_LABEL, style="bold")
            range_line.append_text(self._range_map_text)
            parts.append(range_line)
        if self._file_map_text is not None:
            parts.append(self._file_map_text)
        return Group(*parts)

    def _refresh(self) -> None:
        if self._live is None:
            return
        try:
            self._live.update(self._build_group())
        except Exception:
            pass

    # ── generic ───────────────────────────────────────────────
    def info(self, message: str) -> None:
        self._console.print(message)

    def warn(self, message: str) -> None:
        self._console.print(f"[yellow]Warning:[/yellow] {message}")

    def error(self, message: str) -> None:
        self._console.print(f"[bold red]Error:[/bold red] {message}")

    def debug(self, message: str) -> None:
        if self._verbose:
            self._console.log(f"[dim]{message}[/dim]")

    def close(self) -> None:
        self._stop_live()

    # ── decode: probe ────────────────────────────────────────
    def probe_start(self) -> None:
        self._stop_live()
        self._probe_spinner_progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn(
                f"[bold cyan]{_pad_status_label('Probe')}[/bold cyan]"
            ),
            TextColumn("[cyan]{task.description}[/cyan]"),
            console=self._console,
            transient=True,
        )
        self._probe_task_id = self._probe_spinner_progress.add_task(
            "starting...", total=None)
        self._live = Live(
            self._probe_spinner_progress,
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def probe_update(self, *, scanned: int, total: int,
                     detect: float, phase: str) -> None:
        if self._probe_spinner_progress is None or self._probe_task_id is None:
            return
        if phase == "reading":
            desc = f"reading frames {scanned}/{total}"
        elif phase == "scanning":
            desc = f"scanning {scanned}/{total}, detect {detect * 100:.0f}%"
        elif phase == "calibrating":
            desc = f"calibrating (detect {detect * 100:.0f}%)"
        else:
            desc = phase
        try:
            self._probe_spinner_progress.update(
                self._probe_task_id, description=desc)
        except Exception:
            pass

    def probe_done(self, *, sample: int, detect: float, repeat: float,
                   crop_reduction: float | None) -> None:
        self._stop_live()
        self._probe_spinner_progress = None
        self._probe_task_id = None
        if crop_reduction is None:
            crop_str = "[dim]crop=off[/dim]"
        else:
            crop_str = f"crop=[green]-{crop_reduction * 100:.0f}%[/green]"
        self._console.print(
            f"[bold cyan]Probe[/bold cyan]  [green]✓[/green]  "
            f"sample=[bold]{sample}[/bold]  "
            f"detect=[bold]{detect * 100:.0f}%[/bold]  "
            f"repeat=[bold]{repeat:.1f}[/bold]  "
            f"{crop_str}"
        )

    # ── decode: scan ──────────────────────────────────────────
    def scan_start(self, *, total_frames: int,
                   total_blocks: int | None = None) -> None:
        self._stop_live()
        self._progress = Progress(
            TextColumn(
                f"[bold cyan]{_pad_status_label('Scan')}[/bold cyan]"
            ),
            BarColumn(bar_width=None, complete_style="cyan",
                      finished_style="bright_cyan",
                      pulse_style="bright_cyan"),
            TaskProgressColumn(),
            TextColumn("  "),
            _HitColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "scan", total=max(1, total_frames), hit=0.0)
        self._live = Live(
            self._build_group(),
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.start()

    def scan_update(self, *, video_pct: float, hit_window: float,
                    file_pct: float, recovered: object,
                    k: int | None) -> None:
        if self._progress is None or self._task_id is None:
            return
        total = self._progress.tasks[self._task_id].total or 1
        completed = max(0.0, min(total, video_pct / 100.0 * total))
        try:
            self._progress.update(self._task_id, completed=completed,
                                  hit=hit_window)
        except Exception:
            pass
        self._update_file_map(file_pct, recovered, k)
        self._refresh()

    def scan_done(self) -> None:
        if self._progress is not None and self._task_id is not None:
            try:
                total = self._progress.tasks[self._task_id].total or 1
                self._progress.update(self._task_id, completed=total)
            except Exception:
                pass
        self._stop_live()

    # ── decode: recover ──────────────────────────────────────
    def recover_start(self, *, level: str,
                      segments: Sequence[tuple[int, int]],
                      total_frames: int) -> None:
        self._stop_live()
        self._recover_segments = list(segments)
        self._recover_total_frames = max(1, total_frames)
        self._recover_scanned = []
        self._progress = Progress(
            TextColumn(
                f"[bold yellow]Recover[/bold yellow]  "
                f"[dim]{level}[/dim] "
            ),
            BarColumn(bar_width=None, complete_style="yellow",
                      finished_style="bright_yellow",
                      pulse_style="bright_yellow"),
            TaskProgressColumn(),
            TextColumn("  "),
            _HitColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "recover", total=1000, hit=0.0)
        width = self._map_width()
        self._range_map_text = render_range_strip_rich(
            self._recover_segments,
            self._recover_total_frames,
            width,
        )
        self._live = Live(
            self._build_group(),
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.start()

    def recover_update(self, *, progress_pct: float, hit_window: float,
                       file_pct: float, recovered: object,
                       k: int | None,
                       current_range: tuple[int, int] | None) -> None:
        if self._progress is None or self._task_id is None:
            return
        try:
            self._progress.update(self._task_id,
                                  completed=max(0.0, min(1000.0,
                                                         progress_pct * 10.0)),
                                  hit=hit_window)
        except Exception:
            pass
        # Range strip: rebuild only when current range changes.
        width = self._map_width()
        self._range_map_text = render_range_strip_rich(
            self._recover_segments,
            self._recover_total_frames,
            width,
            current=current_range,
            scanned=self._recover_scanned,
        )
        self._update_file_map(file_pct, recovered, k)
        self._refresh()

    def recover_done(self) -> None:
        if self._progress is not None and self._task_id is not None:
            try:
                self._progress.update(self._task_id, completed=1000)
            except Exception:
                pass
        self._stop_live()

    # ── save ──────────────────────────────────────────────────
    def save_done(self, *, output_path: str, bytes_written: int) -> None:
        self._stop_live()
        self._console.print(
            f"[bold green]Save[/bold green]   [green]✓[/green]  "
            f"{output_path}  [dim]{_fmt_size(bytes_written)}[/dim]"
        )

    # ── encode ────────────────────────────────────────────────
    def encode_start(self, *, duration_sec: float, fps: int,
                     qr_version: int, mode: str, overhead: float) -> None:
        self._stop_live()
        self._console.print(
            f"[bold green]Encode[/bold green]  "
            f"video=[bold]{_fmt_duration(duration_sec)}[/bold]  "
            f"fps=[bold]{fps}[/bold]  "
            f"qr=[bold]V{qr_version}[/bold]  "
            f"mode=[bold]{mode}[/bold]  "
            f"overhead=[bold]{overhead:.1f}x[/bold]"
        )
        self._progress = Progress(
            TextColumn(
                f"[bold green]{_pad_status_label('Encode')}[/bold green]"
            ),
            BarColumn(bar_width=None, complete_style="green",
                      finished_style="bright_green",
                      pulse_style="bright_green"),
            TaskProgressColumn(),
            TextColumn(" "),
            _EncodeStatsColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "encode", total=1000, fps=0.0, eta_override=None)
        self._live = Live(
            self._progress,
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.start()

    def encode_update(self, *, progress_pct: float, speed_fps: float,
                      eta_sec: float) -> None:
        if self._progress is None or self._task_id is None:
            return
        try:
            self._progress.update(
                self._task_id,
                completed=max(0.0, min(1000.0, progress_pct * 10.0)),
                fps=speed_fps,
                eta_override=eta_sec,
            )
        except Exception:
            pass

    def encode_done(self, *, output_path: str, size_bytes: int) -> None:
        if self._progress is not None and self._task_id is not None:
            try:
                self._progress.update(self._task_id, completed=1000)
            except Exception:
                pass
        self._stop_live()
        self._console.print(
            f"[bold green]Done[/bold green]   {output_path}  "
            f"[dim]{_fmt_size(size_bytes)}[/dim]"
        )

    # ── helpers ───────────────────────────────────────────────
    def _update_file_map(self, file_pct: float,
                         recovered: object,
                         k: int | None) -> None:
        if k is None or k <= 0:
            return
        bucket = int(file_pct * (self._MAP_QUANT_BUCKETS / 100.0))
        rec_len = -1
        if isinstance(recovered, dict):
            rec_len = len(recovered)
        elif hasattr(recovered, "__len__"):
            try:
                rec_len = len(recovered)  # type: ignore[arg-type]
            except Exception:
                rec_len = -1
        if (bucket == self._last_map_bucket and k == self._last_map_k
                and rec_len == self._last_map_recovered_len
                and self._file_map_text is not None):
            return
        width = self._map_width()
        map_text = render_block_map_rich(recovered, k, width)
        final = Text()
        final.append(self._MAP_LABEL, style="bold")
        final.append_text(map_text)
        final.append(f"  {file_pct:4.1f}%", style="bold")
        self._file_map_text = final
        self._last_map_bucket = bucket
        self._last_map_k = k
        self._last_map_recovered_len = rec_len


# ── Resolver ─────────────────────────────────────────────────────


def _rich_unavailable_reason() -> str:
    if _RICH_AVAILABLE:
        return ""
    if _RICH_IMPORT_ERROR is None:
        return "rich is not installed"
    exc = _RICH_IMPORT_ERROR
    return f"{type(exc).__name__}: {exc}"


def resolve_output_mode(mode: OutputMode | str,
                        *, stderr_isatty: bool | None = None,
                        explicit: bool = True) -> "ProgressReporter":
    """Construct the appropriate reporter for ``mode``.

    ``AUTO`` chooses :class:`RichReporter` when stderr is a TTY (and
    Rich imported cleanly), else :class:`LogReporter`.

    When the caller explicitly asked for ``interactive`` or ``verbose``
    (``explicit=True``, the default) but Rich is unavailable, a single
    warning line is written to stderr so the user understands why
    they're seeing log-style output instead of the animated UI.
    """
    if isinstance(mode, str):
        try:
            mode = OutputMode(mode)
        except ValueError as exc:
            raise ValueError(
                f"Unknown output mode: {mode!r}. Expected one of "
                f"{[m.value for m in OutputMode]}."
            ) from exc

    if stderr_isatty is None:
        try:
            stderr_isatty = bool(sys.stderr.isatty())
        except Exception:
            stderr_isatty = False

    def _warn_rich_unavailable(requested: str) -> None:
        if not explicit:
            return
        reason = _rich_unavailable_reason() or "unknown"
        try:
            sys.stderr.write(
                f"Warning: --output-mode={requested} requested but Rich UI "
                f"is unavailable ({reason}); falling back to log output. "
                f"Install/upgrade with `pip install -U 'rich>=13.0.0'`.\n"
            )
            sys.stderr.flush()
        except Exception:
            pass

    if mode is OutputMode.QUIET:
        return QuietReporter()
    if mode is OutputMode.LOG:
        return LogReporter(verbose=False)
    if mode is OutputMode.VERBOSE:
        if _RICH_AVAILABLE and stderr_isatty:
            return RichReporter(verbose=True)
        if not _RICH_AVAILABLE:
            _warn_rich_unavailable("verbose")
        return LogReporter(verbose=True)
    if mode is OutputMode.INTERACTIVE:
        if _RICH_AVAILABLE:
            return RichReporter(verbose=False)
        # Explicit interactive request with no Rich → hard fail so the
        # user knows immediately instead of silently getting log output.
        if explicit:
            reason = _rich_unavailable_reason() or "unknown"
            raise RuntimeError(
                f"--output-mode=interactive requires the Rich library "
                f"but it is unavailable ({reason}). Install or upgrade "
                f"with `pip install -U 'rich>=13.0.0'`, or pass "
                f"`--output-mode=log` / `--output-mode=auto` instead."
            )
        _warn_rich_unavailable("interactive")
        return LogReporter(verbose=False)
    # AUTO
    if _RICH_AVAILABLE and stderr_isatty:
        return RichReporter(verbose=False)
    return LogReporter(verbose=False)


__all__ = [
    "OutputMode",
    "Phase",
    "ProgressReporter",
    "RichReporter",
    "LogReporter",
    "QuietReporter",
    "SlidingHitWindow",
    "resolve_output_mode",
    "render_block_map_plain",
    "render_block_map_rich",
    "render_range_strip_plain",
    "render_range_strip_rich",
    "compute_block_map_cells",
    "compute_range_strip_cells",
]
