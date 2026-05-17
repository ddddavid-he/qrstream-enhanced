"""PySide6 playback loop for display-only encoding.

Provides a modern dark-themed player with native timeline scrubber,
system font rendering, and full keyboard/mouse control. PySide6 is a
runtime dependency of the default qrstream package.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .display_cache import (
    ModuleFrameCache,
)


# ── Check for PySide6 availability ────────────────────────────────

_PYSIDE6_AVAILABLE = False
_PYSIDE6_IMPORT_ERROR: str = ""

try:
    from PySide6.QtCore import Qt, QTimer, QSettings  # noqa: F401
    from PySide6.QtGui import (  # noqa: F401
        QColor,
        QImage,
        QKeySequence,
        QPainter,
        QPixmap,
        QShortcut,
    )
    from PySide6.QtWidgets import (  # noqa: F401
        QApplication,
        QDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )
    _PYSIDE6_AVAILABLE = True
except ImportError as exc:
    _PYSIDE6_IMPORT_ERROR = str(exc)


def require_pyside6() -> None:
    """Raise a clear error if PySide6 is not installed."""
    if _PYSIDE6_AVAILABLE:
        return
    raise ImportError(
        "PySide6 is required for the Qt display player and is included "
        "in the default qrstream package. Reinstall qrstream or install "
        "PySide6-Essentials directly.\n\n"
        "    pip install --upgrade qrstream\n"
        "    pip install PySide6-Essentials\n\n"
        f"Details: {_PYSIDE6_IMPORT_ERROR}"
    )


# ── Config (no PySide6 dependency) ────────────────────────────────


@dataclass
class DisplayMetadata:
    """File and encoding metadata shown in the info panel."""

    file_name: str = ""
    file_size: int = 0
    payload_size: int = 0
    compressed: bool = False
    data_blocks: int = 0
    total_blocks: int = 0
    block_size: int = 0
    total_frames: int = 0
    qr_version: int = 0
    ec_level: int = 0
    module_side: int = 0
    fps: int = 0
    high_density: bool = False


@dataclass
class DisplayPlayerQtConfig:
    title: str = "QRStream"
    min_prebuffer_seconds: float = 3.0
    producer_fps_window_seconds: float = 3.0
    producer_grace_factor: float = 1.05
    metadata: DisplayMetadata | None = None
    lock_window_size: bool = False
    integer_scale: bool = False
    initial_screen_fraction: float = 0.70
    ignore_saved_geometry: bool = False


# ── Helpers (no PySide6 dependency) ───────────────────────────────


def _can_play(
    cache: ModuleFrameCache,
    state,
    frame_index: int,
    fps: int,
    config: DisplayPlayerQtConfig,
) -> bool:
    contiguous = cache.contiguous_from(frame_index)
    if contiguous <= 0:
        return False
    if cache.is_done() or state.is_done():
        return True
    min_frames = max(1, int(config.min_prebuffer_seconds * max(1, fps)))
    if contiguous < min_frames:
        return False
    producer_fps = state.producer_fps(config.producer_fps_window_seconds)
    return producer_fps >= fps * config.producer_grace_factor


__all__ = [
    "DisplayMetadata",
    "DisplayPlayerQtConfig",
    "play_display_qt",
    "require_pyside6",
]

# ── Everything below requires PySide6 ─────────────────────────────

if not _PYSIDE6_AVAILABLE:
    def play_display_qt(*args, **kwargs) -> None:  # pragma: no cover
        require_pyside6()
else:
    # ── QSS Dark Theme ───────────────────────────────────────────

    _DARK_QSS = """
    QMainWindow {
        background-color: #181818;
    }
    QLabel {
        color: #cccccc;
        font-size: 13px;
    }
    QLabel#timeLabel {
        font-size: 12px;
        font-family: "Menlo", "Consolas", monospace;
        color: #999999;
        padding: 0 2px;
    }
    QLabel#loopIndicator {
        font-size: 11px;
        color: #4c9eff;
        padding: 0 2px;
    }
    QPushButton {
        background-color: transparent;
        color: #cccccc;
        border: none;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #333333;
    }
    QPushButton:pressed {
        background-color: #444444;
    }
    QPushButton#playButton {
        font-size: 14px;
        padding: 2px 6px;
        min-width: 24px;
        max-width: 24px;
    }
    QWidget#controlBar {
        background-color: #1e1e1e;
    }
    QStatusBar {
        background-color: #181818;
        color: #444444;
        font-size: 11px;
        border-top: 1px solid #2a2a2a;
        padding: 1px 8px;
    }
    QDialog {
        background-color: #1e1e1e;
        border: 1px solid #333333;
    }
    QDialog QLabel {
        color: #cccccc;
        font-size: 13px;
    }
    QDialog QPushButton {
        background-color: #333333;
        border: 1px solid #444444;
        padding: 6px 20px;
    }
    QDialog QPushButton:hover {
        background-color: #444444;
        border-color: #555555;
    }
    """

    # ── Help dialog ──────────────────────────────────────────────

    _HELP_TEXT = """
    <style>
      body { color: #cccccc; font-family: -apple-system, sans-serif; }
      h2 { color: #4c9eff; margin-bottom: 8px; font-size: 15px; }
      table { margin: 8px 0; }
      td { padding: 3px 14px 3px 0; }
      td:first-child {
          color: #4c9eff; font-weight: 600; white-space: nowrap;
          font-family: "SF Mono", "Menlo", "Consolas", monospace;
          font-size: 12px;
      }
      td:last-child { color: #999999; font-size: 13px; }
    </style>
    <h2>Keyboard Shortcuts</h2>
    <table>
    <tr><td>Space</td><td>Play / Pause</td></tr>
    <tr><td>&larr; &rarr;</td><td>Step frame backward / forward</td></tr>
    <tr><td>J &nbsp; K</td><td>Jump &minus;1s / +1s</td></tr>
    <tr><td>+ &nbsp; &minus;</td><td>Zoom in / out</td></tr>
    <tr><td>L</td><td>Toggle loop mode</td></tr>
    <tr><td>I</td><td>File info panel</td></tr>
    <tr><td>F</td><td>Toggle fullscreen</td></tr>
    <tr><td>H &nbsp; ?</td><td>Show this help</td></tr>
    <tr><td>Q &nbsp; Esc</td><td>Quit</td></tr>
    </table>
    """

    # ── Helpers (PySide6-dependent) ──────────────────────────────

    class _TimelineSlider(QSlider):
        """QSlider with a YouTube-style gray buffer bar behind the groove."""

        def __init__(self, orientation, parent=None):
            super().__init__(orientation, parent)
            self._buffer_pct = 0.0  # 0..1

        def set_buffer(self, pct: float) -> None:
            self._buffer_pct = max(0.0, min(1.0, pct))
            self.update()

        def paintEvent(self, event) -> None:  # noqa: N802
            # Paint buffer bar, then let QSlider paint on top
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Groove geometry: centered strip
            groove_h = 4
            handle_w = 12
            margin = handle_w // 2
            y = (self.height() - groove_h) // 2
            usable = self.width() - handle_w

            # Background groove
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#333333"))
            p.drawRoundedRect(margin, y, usable, groove_h, 2, 2)

            # Buffer bar (gray, like YouTube loading)
            if self._buffer_pct > 0:
                buf_w = int(usable * self._buffer_pct)
                p.setBrush(QColor("#555555"))
                p.drawRoundedRect(margin, y, buf_w, groove_h, 2, 2)

            # Played bar (accent)
            rng = self.maximum() - self.minimum()
            if rng > 0:
                frac = (self.value() - self.minimum()) / rng
                play_w = int(usable * frac)
                p.setBrush(QColor("#4c9eff"))
                p.drawRoundedRect(margin, y, play_w, groove_h, 2, 2)

            # Handle
            if rng > 0:
                frac = (self.value() - self.minimum()) / rng
            else:
                frac = 0.0
            hx = margin + int(usable * frac) - handle_w // 2
            hy = (self.height() - handle_w) // 2
            p.setBrush(QColor("#ffffff"))
            p.drawEllipse(hx, hy, handle_w, handle_w)

            p.end()

    class _PixmapCache:
        """O(1) LRU cache for scaled QPixmap objects."""

        def __init__(self, max_entries: int = 128):
            from collections import OrderedDict as _OD
            self._max = max(1, max_entries)
            self._cache: _OD[tuple[int, int], QPixmap] = _OD()

        def get(self, key: tuple[int, int]) -> QPixmap | None:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

        def put(self, key: tuple[int, int], pixmap: QPixmap) -> None:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._max:
                self._cache.popitem(last=False)
            self._cache[key] = pixmap

        def clear(self) -> None:
            self._cache.clear()

    def _numpy_to_qimage(arr: np.ndarray) -> QImage:
        """Convert a 2-D 0/255 grayscale numpy array to QImage (zero copy)."""
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)

    def _fmt_time(seconds: float) -> str:
        """Format seconds as m:ss or h:mm:ss."""
        s = max(0, int(seconds))
        if s < 3600:
            return f"{s // 60}:{s % 60:02d}"
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    # ── Main Window ──────────────────────────────────────────────

    class _QRStreamWindow(QMainWindow):
        """PySide6 window for QR code frame playback."""

        def __init__(
            self,
            cache: ModuleFrameCache,
            state,
            fps: int,
            config: DisplayPlayerQtConfig,
        ):
            super().__init__()
            self._cache = cache
            self._state = state
            self._fps = max(1, fps)
            self._config = config
            self._frame_interval = 1.0 / self._fps

            self._module_side = cache.module_side
            self._frame_index = 0
            self._playing = False
            self._looping = False
            self._next_frame_ts = 0.0

            self._presentation = _PixmapCache(max_entries=128)

            self._settings = QSettings("QRStream", "DisplayPlayer")

            self.setWindowTitle(config.title)
            self.setMinimumSize(320, 400)

            self._setup_ui()
            self._apply_theme()
            self._setup_shortcuts()

            if (config.ignore_saved_geometry or config.lock_window_size
                    or not self._restore_geometry()):
                self._auto_size()
            if config.lock_window_size:
                self.setFixedSize(self.size())

            self._setup_timer()
            self._update_display()

        # ── UI Construction ─────────────────────────────────────

        def _setup_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._qr_label = QLabel()
            self._qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._qr_label.setStyleSheet("background-color: #111111;")
            self._qr_label.setMinimumSize(100, 100)
            layout.addWidget(self._qr_label, stretch=1)

            # ── Single-row control bar ──────────────────────────
            control_bar = QWidget()
            control_bar.setObjectName("controlBar")
            control_bar.setFixedHeight(36)
            row = QHBoxLayout(control_bar)
            row.setContentsMargins(6, 0, 10, 0)
            row.setSpacing(6)

            self._play_btn = QPushButton("▶")
            self._play_btn.setObjectName("playButton")
            self._play_btn.clicked.connect(self._toggle_play)
            row.addWidget(self._play_btn)

            self._time_label = QLabel("0:00 / 0:00")
            self._time_label.setObjectName("timeLabel")
            row.addWidget(self._time_label)

            self._slider = _TimelineSlider(Qt.Orientation.Horizontal)
            self._slider.setRange(0, max(0, self._cache.total_frames - 1))
            self._slider.setValue(0)
            self._slider.setFixedHeight(20)
            self._slider.sliderPressed.connect(self._on_slider_pressed)
            self._slider.sliderReleased.connect(self._on_slider_released)
            self._slider.valueChanged.connect(self._on_slider_moved)
            row.addWidget(self._slider, stretch=1)

            self._loop_label = QLabel("")
            self._loop_label.setObjectName("loopIndicator")
            row.addWidget(self._loop_label)

            layout.addWidget(control_bar)

            self._status = QStatusBar()
            self.setStatusBar(self._status)
            self._status.showMessage(
                "Space play/pause  ←→ frame  J/K ±1s  "
                "+/- zoom  L loop  I info  F fullscreen  H help  Q quit")

        def _apply_theme(self) -> None:
            self.setStyleSheet(_DARK_QSS)

        def _setup_shortcuts(self) -> None:
            QShortcut(QKeySequence(Qt.Key.Key_Space),
                      self, self._toggle_play)
            QShortcut(QKeySequence(Qt.Key.Key_Left),
                      self, lambda: self._step(-1))
            QShortcut(QKeySequence(Qt.Key.Key_Right),
                      self, lambda: self._step(1))
            QShortcut(QKeySequence(Qt.Key.Key_A),
                      self, lambda: self._step(-1))
            QShortcut(QKeySequence(Qt.Key.Key_D),
                      self, lambda: self._step(1))
            QShortcut(QKeySequence(Qt.Key.Key_J),
                      self, lambda: self._jump(-1))
            QShortcut(QKeySequence(Qt.Key.Key_K),
                      self, lambda: self._jump(1))
            QShortcut(QKeySequence(Qt.Key.Key_Down),
                      self, lambda: self._jump(-1))
            QShortcut(QKeySequence(Qt.Key.Key_Up),
                      self, lambda: self._jump(1))
            QShortcut(QKeySequence(Qt.Key.Key_Plus),
                      self, lambda: self._zoom(1))
            QShortcut(QKeySequence(Qt.Key.Key_Equal),
                      self, lambda: self._zoom(1))
            QShortcut(QKeySequence(Qt.Key.Key_Minus),
                      self, lambda: self._zoom(-1))
            QShortcut(QKeySequence(Qt.Key.Key_F),
                      self, self._toggle_fullscreen)
            QShortcut(QKeySequence(Qt.Key.Key_H),
                      self, self._show_help)
            QShortcut(QKeySequence(Qt.Key.Key_Question),
                      self, self._show_help)
            QShortcut(QKeySequence(Qt.Key.Key_L),
                      self, self._toggle_loop)
            QShortcut(QKeySequence(Qt.Key.Key_I),
                      self, self._show_info)
            QShortcut(QKeySequence(Qt.Key.Key_Q),
                      self, self._quit)
            QShortcut(QKeySequence(Qt.Key.Key_Escape),
                      self, self._quit)

        def _setup_timer(self) -> None:
            self._timer = QTimer(self)
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.timeout.connect(self._tick)
            self._timer.start(max(1, 1000 // self._fps))

        def _restore_geometry(self) -> bool:
            """Restore saved window geometry. Returns True if restored."""
            geom = self._settings.value("window/geometry")
            if geom is not None:
                self.restoreGeometry(geom)
                return True
            return False

        def _auto_size(self) -> None:
            """Size the window from the configured screen fraction."""
            screen = self.screen()
            if screen is None:
                self.resize(800, 860)
                return
            avail = screen.availableGeometry()
            fraction = max(0.1, min(1.0, self._config.initial_screen_fraction))
            max_qr_side = max(320, min(avail.width(), avail.height() - 60))
            target = int(max_qr_side * fraction)
            target = max(400, min(target, max_qr_side))
            self.resize(target, target + 60)
            # centre on screen
            frame = self.frameGeometry()
            frame.moveCenter(avail.center())
            self.move(frame.topLeft())

        def closeEvent(self, event) -> None:  # noqa: N802
            if not self._config.lock_window_size:
                self._settings.setValue("window/geometry", self.saveGeometry())
            self._timer.stop()
            super().closeEvent(event)

        # ── Frame update ────────────────────────────────────────

        def _tick(self) -> None:
            now = time.monotonic()

            if self._playing and now >= self._next_frame_ts:
                nxt = self._frame_index + 1
                if nxt >= self._cache.total_frames:
                    if self._looping and self._cache.has_frame(0):
                        self._frame_index = 0
                        self._next_frame_ts = now + self._frame_interval
                    else:
                        self._playing = False
                        self._play_btn.setText("▶")
                elif self._cache.has_frame(nxt):
                    self._frame_index = nxt
                    self._next_frame_ts += self._frame_interval
                    if self._next_frame_ts < now - self._frame_interval:
                        self._next_frame_ts = now + self._frame_interval
                else:
                    self._playing = False
                    self._play_btn.setText("▶")

            can = _can_play(self._cache, self._state, self._frame_index,
                            self._fps, self._config)
            if self._playing and not can:
                self._playing = False
                self._play_btn.setText("▶")

            self._update_controls()
            self._update_display()

        def _update_controls(self) -> None:
            total = max(1, self._cache.total_frames)
            cur_sec = self._frame_index / max(1, self._fps)
            tot_sec = total / max(1, self._fps)
            self._time_label.setText(
                f"{_fmt_time(cur_sec)} / {_fmt_time(tot_sec)}")
            self._slider.blockSignals(True)
            self._slider.setValue(self._frame_index)
            self._slider.blockSignals(False)

            self._slider.set_buffer(self._cache.valid_count / total)

        def _update_display(self) -> None:
            label_size = self._qr_label.size()
            side = min(label_size.width(), label_size.height())
            if side < 1:
                return

            module_img = self._cache.get_module_image(self._frame_index)
            key = (self._frame_index, side)
            if module_img is not None:
                cached = self._presentation.get(key)
                if cached is not None:
                    self._qr_label.setPixmap(cached)
                    return
                qimg = _numpy_to_qimage(module_img)
            else:
                placeholder = np.full(
                    (self._module_side, self._module_side),
                    255, dtype=np.uint8)
                qimg = _numpy_to_qimage(placeholder)

            pixmap = QPixmap.fromImage(qimg)
            if self._config.integer_scale and module_img is not None:
                module_side = max(module_img.shape[0], module_img.shape[1])
                scale = max(1, side // max(1, module_side))
                target_side = min(side, module_side * scale)
            else:
                target_side = side
            scaled = pixmap.scaled(
                target_side, target_side,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            if module_img is not None:
                self._presentation.put(key, scaled)
            self._qr_label.setPixmap(scaled)

        # ── Actions ─────────────────────────────────────────────

        def _toggle_play(self) -> None:
            if self._playing:
                self._playing = False
                self._play_btn.setText("▶")
                return
            can = _can_play(self._cache, self._state, self._frame_index,
                            self._fps, self._config)
            if can:
                self._playing = True
                self._play_btn.setText("⏸")
                self._next_frame_ts = (time.monotonic()
                                       + self._frame_interval)

        def _step(self, delta: int) -> None:
            target = max(0, min(self._cache.total_frames - 1,
                                self._frame_index + delta))
            if target != self._frame_index and self._cache.has_frame(target):
                self._frame_index = target
                self._update_display()
            self._playing = False
            self._play_btn.setText("▶")

        def _jump(self, delta_seconds: int) -> None:
            frames = delta_seconds * self._fps
            target = max(0, min(self._cache.total_frames - 1,
                                self._frame_index + frames))
            if target != self._frame_index:
                for idx in range(target, -1, -1):
                    if self._cache.has_frame(idx):
                        self._frame_index = idx
                        self._update_display()
                        break
            self._playing = False
            self._play_btn.setText("▶")

        def _zoom(self, delta: int) -> None:
            if self._config.lock_window_size:
                return
            step = 80
            w = self.width() + delta * step
            h = self.height() + delta * step
            w = max(self.minimumWidth(), w)
            h = max(self.minimumHeight(), h)
            self.resize(w, h)

        def _toggle_fullscreen(self) -> None:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        def _toggle_loop(self) -> None:
            self._looping = not self._looping
            self._loop_label.setText("LOOP" if self._looping else "")

        def _show_info(self) -> None:
            meta = self._config.metadata
            if meta is None:
                meta = DisplayMetadata()

            ec_names = {0: "L (7%)", 1: "M (15%)", 2: "Q (25%)", 3: "H (30%)"}
            ec_str = ec_names.get(meta.ec_level, f"Level {meta.ec_level}")
            mode_str = "Alphanumeric (Base45)" if meta.high_density else "Binary (Base64)"

            def _fmt_size(n: int) -> str:
                if n < 1024:
                    return f"{n} B"
                elif n < 1024 * 1024:
                    return f"{n / 1024:.1f} KiB"
                else:
                    return f"{n / (1024 * 1024):.2f} MiB"

            duration = meta.total_frames / max(1, meta.fps)

            html = f"""
            <style>
              body {{ color: #cccccc; font-family: -apple-system, sans-serif; }}
              h2 {{ color: #ffffff; margin: 0 0 14px 0; font-size: 15px;
                    font-weight: 600; }}
              .section {{ margin-bottom: 16px; }}
              .section-title {{
                  color: #4c9eff; font-size: 10px; font-weight: 700;
                  text-transform: uppercase; letter-spacing: 1.2px;
                  margin-bottom: 6px; padding-bottom: 4px;
                  border-bottom: 1px solid #333333;
              }}
              table {{ width: 100%; border-collapse: collapse; }}
              td {{ padding: 3px 8px 3px 0; vertical-align: top; }}
              td.key {{
                  color: #777777; font-size: 12px; width: 110px;
                  white-space: nowrap;
              }}
              td.val {{ color: #dddddd; font-size: 13px; }}
              .mono {{ font-family: "SF Mono", "Menlo", monospace;
                       color: #4c9eff; }}
            </style>
            <h2>File Information</h2>
            <div class="section">
              <div class="section-title">Source</div>
              <table>
                <tr><td class="key">File</td>
                    <td class="val">{meta.file_name}</td></tr>
                <tr><td class="key">Original size</td>
                    <td class="val">{_fmt_size(meta.file_size)}</td></tr>
                <tr><td class="key">Payload size</td>
                    <td class="val">{_fmt_size(meta.payload_size)}
                    {"(compressed)" if meta.compressed else ""}</td></tr>
              </table>
            </div>
            <div class="section">
              <div class="section-title">Encoding</div>
              <table>
                <tr><td class="key">Data blocks</td>
                    <td class="val">{meta.data_blocks}</td></tr>
                <tr><td class="key">Total blocks</td>
                    <td class="val">{meta.total_blocks}</td></tr>
                <tr><td class="key">Block size</td>
                    <td class="val">{meta.block_size} bytes</td></tr>
                <tr><td class="key">Mode</td>
                    <td class="val">{mode_str}</td></tr>
              </table>
            </div>
            <div class="section">
              <div class="section-title">QR Code</div>
              <table>
                <tr><td class="key">Version</td>
                    <td class="val">{meta.qr_version}</td></tr>
                <tr><td class="key">EC Level</td>
                    <td class="val">{ec_str}</td></tr>
                <tr><td class="key">Module side</td>
                    <td class="val"><span class="mono">{meta.module_side}×{meta.module_side}</span></td></tr>
              </table>
            </div>
            <div class="section">
              <div class="section-title">Playback</div>
              <table>
                <tr><td class="key">Total frames</td>
                    <td class="val">{meta.total_frames}</td></tr>
                <tr><td class="key">Frame rate</td>
                    <td class="val">{meta.fps} fps</td></tr>
                <tr><td class="key">Duration</td>
                    <td class="val">{duration:.1f}s</td></tr>
              </table>
            </div>
            """

            dlg = QDialog(self)
            dlg.setWindowTitle("Info — QRStream")
            dlg.setMinimumWidth(380)
            dlg.setMaximumWidth(480)
            layout = QVBoxLayout(dlg)
            layout.setContentsMargins(16, 16, 16, 12)
            label = QLabel(html)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            layout.addWidget(label)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn,
                             alignment=Qt.AlignmentFlag.AlignRight)
            QShortcut(QKeySequence(Qt.Key.Key_Escape), dlg, dlg.accept)
            QShortcut(QKeySequence(Qt.Key.Key_I), dlg, dlg.accept)
            dlg.setStyleSheet(_DARK_QSS)
            dlg.exec()

        def _show_help(self) -> None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Help — QRStream")
            dlg.setMinimumWidth(380)
            layout = QVBoxLayout(dlg)
            label = QLabel(_HELP_TEXT)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            layout.addWidget(label)
            close_btn = QPushButton("Close (Esc)")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn,
                             alignment=Qt.AlignmentFlag.AlignRight)
            QShortcut(QKeySequence(Qt.Key.Key_Escape), dlg, dlg.accept)
            dlg.setStyleSheet(_DARK_QSS)
            dlg.exec()

        def _quit(self) -> None:
            self._state.request_cancel()
            self._timer.stop()
            self.close()

        # ── Slider handlers ─────────────────────────────────────

        _slider_dragging: bool = False

        def _on_slider_pressed(self) -> None:
            self._slider_dragging = True
            was_playing = self._playing
            self._playing = False
            self._play_btn.setText("▶")
            self._was_playing_before_drag = was_playing

        def _on_slider_released(self) -> None:
            self._slider_dragging = False
            val = self._slider.value()
            for idx in range(val, -1, -1):
                if self._cache.has_frame(idx):
                    self._frame_index = idx
                    break
            self._update_display()
            if getattr(self, '_was_playing_before_drag', False):
                self._toggle_play()

        def _on_slider_moved(self, value: int) -> None:
            if not self._slider_dragging:
                return
            for idx in range(value, -1, -1):
                if self._cache.has_frame(idx):
                    self._frame_index = idx
                    break
            self._update_display()

        # ── Resize ──────────────────────────────────────────────

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._presentation.clear()
            self._update_display()

    # ── Public entry point ───────────────────────────────────────

    def play_display_qt(
        cache: ModuleFrameCache,
        state,
        fps: int,
        config: DisplayPlayerQtConfig | None = None,
    ) -> None:
        """Play cached module frames in a PySide6 window.

        Blocks until the user closes the window or the producer finishes.
        Requires PySide6, which is included in the default qrstream package.
        """
        require_pyside6()

        if config is None:
            config = DisplayPlayerQtConfig()

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
            app.setApplicationName("QRStream")

        window = _QRStreamWindow(cache, state, fps, config)
        window.show()

        app.exec()

        if QApplication.instance() is app:
            try:
                app.quit()
            except Exception:
                pass