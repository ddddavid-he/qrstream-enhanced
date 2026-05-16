# `qrs encode --display` Qt 播放器方案

状态：已实现；本文档记录完整设计方案与决策过程。

当前工作分支：`feature/display-mode`，按 `BRANCHING.md` 从 `dev` 拉出。

## 背景

第一版 display-only 播放器 (`play_display_cache`) 使用 OpenCV `cv2.imshow` 实现。该实现存在以下 UI/UX 问题：

1. **字体不可控** — OpenCV 仅提供 6 款位图字体（FONT_HERSHEY_SIMPLEX 等），无抗锯齿，字号极小。
2. **窗口布局差** — QR 画面 + 固定 96px 白色状态栏，观感像开发者调试窗口。
3. **提示文字拥挤** — 三行快捷键提示永远贴在底部，占用画面高度。
4. **缺少交互控件** — 无时间线滑块、无鼠标拖拽 seek、无全屏切换。
5. **缩放方式原始** — 仅 +/- 逐像素缩放，无预设档位。

## 目标

1. 用 PySide6 实现现代深色主题播放器，替换 OpenCV `cv2.imshow`。
2. 提供原生时间线滑块（QSlider），支持拖拽 seek。
3. 使用系统 TTF 字体，完整抗锯齿 + CJK 支持。
4. 快捷键通过 QShortcut 绑定，按 H 弹出帮助对话框。
5. 全屏切换（F）、缩放（+/-）、播放/暂停（Space）全部由 Qt 原生控件接管。
6. QR 画面区域**零遮挡** — 所有 UI 控件在独立布局区域，不叠在 QR 图上。
7. 不改动 `ModuleFrameCache` / `DisplayProducerState` — 播放器只做渲染层替换。

## 非目标

1. 不做 cv2 播放器的 fallback。PySide6 缺失时直接报错，引导用户修复默认安装环境或单独安装 `PySide6-Essentials`。
2. 第一版不做鼠标滚轮缩放（Qt 支持但优先级低）。
3. 不做速度控制、循环播放等高级功能 — 留待后续 TODO。
4. 不替换 `encode_to_video` 路径 — 仅影响 `--display` 模式。

## 依赖策略

```toml
[project]
dependencies = [
    "PySide6-Essentials>=6.7.0",
]
```

- `pip install qrstream` → 完整 GUI：PySide6 播放器随默认包安装。
- `pip install qrstream[gui]` → 兼容旧安装脚本；`gui` extra 现在是 no-op。
- 核心依赖仍使用 `opencv-python-headless` — Qt 不需要 OpenCV 的 highgui。
- 不引入 Pillow；module image 直接走 numpy → QImage → QPixmap。

### 许可证

PySide6 采用 **LGPLv3**（用户可选 GPLv2/GPLv3）。LGPLv3 的核心要求：

- 动态链接 PySide6（Python `import` 天然满足）。
- 允许用户替换 PySide6 库文件（`pip install --upgrade PySide6-Essentials` 满足）。
- 附 LGPL 文本，声明 PySide6 部分适用 LGPL。

qrstream 本身是 MIT，MIT + LGPLv3 完全兼容。对于开源项目，LGPL 没有任何实质限制。

## 架构

```
src/qrstream/
├── display_cache.py          # ModuleFrameCache / PresentationFrameCache（不变）
├── display_player.py         # cv2 播放器（保留，不再作为默认）
├── display_player_qt.py      # 新增：PySide6 播放器
├── encoder.py                # encode_to_display → play_display_qt
└── cli.py                    # --display 帮助文本更新，ImportError 捕获
```

两个播放器后端共享 `ModuleFrameCache` + `DisplayProducerState` + `PresentationFrameCache`。差异仅在渲染层和事件循环。

### 模块级 PySide6 检测

```python
# display_player_qt.py — 模块顶部
_PYSIDE6_AVAILABLE = False
try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap, QShortcut, ...
    from PySide6.QtWidgets import QApplication, QMainWindow, ...
    _PYSIDE6_AVAILABLE = True
except ImportError as exc:
    _PYSIDE6_IMPORT_ERROR = str(exc)
```

- `require_pyside6()` 始终可用 — 不可用时抛 `ImportError`。
- `DisplayPlayerQtConfig` / `_display_side` / `_can_play` 无 PySide6 依赖，始终可用。
- `_QRStreamWindow` / `play_display_qt` 仅在 `_PYSIDE6_AVAILABLE` 时定义。
- 不可用时 `play_display_qt` 是一个 stub，调用即抛 `require_pyside6()`。

### encode_to_display 调用路径

```python
# encoder.py
def encode_to_display(input_path, ..., player=None):
    if player is None:
        require_pyside6()               # ← 缺失即报错
    # ... 编码逻辑不变 ...
    if player is not None:
        player(cache, state, fps)       # ← 测试注入
    else:
        config = DisplayPlayerQtConfig(
            title=f"QRStream — {os.path.basename(input_path)}")
        play_display_qt(cache, state, fps, config=config)
```

- `player` 参数保留为测试/程序化调用注入点。
- 正常路径始终走 PySide6，不做 cv2 fallback。

## PySide6 窗口布局

```
┌──────────────────────────────────────────────────┐
│  QMainWindow  —  title: "QRStream — data.bin"    │
│  ┌────────────────────────────────────────────┐  │
│  │                                            │  │
│  │  QLabel (QR frame)                         │  │
│  │  background-color: #0f0f23                 │  │
│  │  scaledContents via QPixmap.scaled()       │  │
│  │                                            │  │
│  ├────────────────────────────────────────────┤  │
│  │  QWidget (control bar, bg: #16213e)        │  │
│  │  [▶] [0:14 / 0:50] [══════╸════════] [LOOP]│  │
│  │  QPushButton  QLabel  QSlider       QLabel │  │
│  ├────────────────────────────────────────────┤  │
│  │  QStatusBar (bg: #16213e)                  │  │
│  │  "Space play/pause · ←→ frame · ..."       │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 控件清单

| 控件 | PySide6 类 | 用途 |
|------|-----------|------|
| QR 画面 | `QLabel` + `QPixmap` | 显示 module 图最近邻放大 |
| 播放按钮 | `QPushButton("▶"/"⏸")` | 播放/暂停切换 |
| 时间线 | `QSlider(Qt.Horizontal)` | 拖拽 seek，实时预览 |
| 时间显示 | `QLabel("0:14 / 0:50")` | 当前时间 / 总时长 |
| LOOP 标记 | `QLabel("LOOP")` | 循环播放状态 |
| 状态栏 | `QStatusBar` | 快捷键提示 |

### numpy → QPixmap 转换

```python
def _numpy_to_qimage(arr: np.ndarray) -> QImage:
    h, w = arr.shape
    return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    # ↑ 直接引用 numpy 内存，零拷贝
```

- `ModuleFrameCache.get_module_image()` 返回 `uint8` 灰度图（0/255）。
- `QImage(arr.data, ...)` 引用 numpy buffer，不做拷贝。
- `QPixmap.fromImage()` 做一次像素格式转换（GPU 端）。
- `pixmap.scaled()` 最近邻放大，保持 QR module 边界清晰。

## 快捷键

| 按键 | 动作 | 实现 |
|------|------|------|
| Space | 播放/暂停 | `QShortcut(Qt.Key_Space, ...)` |
| ← → / A D | 逐帧步进 | `QShortcut(Qt.Key_Left/Right, ...)` |
| J / L / ↑ ↓ | 跳转 ±1s | `QShortcut(Qt.Key_J/L, ...)` |
| + / - | 缩放 | `QShortcut(Qt.Key_Plus/Minus, ...)` |
| F | 全屏切换 | `QShortcut(Qt.Key_F, ...)` |
| H / ? | 帮助弹窗 | `QShortcut(Qt.Key_H, ...)` |
| Q / Esc | 退出 | `QShortcut(Qt.Key_Q/Escape, ...)` |

### 帮助弹窗

按下 H 时弹出 `QDialog`，显示快捷键列表（HTML 格式，与窗口同款深色主题）。按 Esc 或点击 Close 关闭。

### 时间线拖拽

- `sliderPressed` → 暂停播放，记录 `_was_playing_before_drag`。
- `sliderMoved` → 实时预览滑条位置的缓存帧（不提交 frame_index）。
- `sliderReleased` → 提交最终位置，恢复播放（如果拖拽前在播放）。

## 深色主题 (QSS)

```css
QMainWindow  { background-color: #1a1a2e }
QLabel       { color: #e0e0e0; font-size: 13px }
QPushButton  { background: #2a2a4a; border: 1px solid #3a3a5a; border-radius: 4px }
QSlider      { groove: #2a2a4a; handle: #6a6aaa }
QStatusBar   { background: #16213e; color: #8888aa; border-top: 1px solid #2a2a4a }
```

全部通过 `setStyleSheet(_DARK_QSS)` 一次性应用，无需逐控件设置。

## 播放门控

与第一版 cv2 播放器完全相同的逻辑，通过 `_can_play()` 函数判定：

```python
contiguous = cache.contiguous_from(frame_index)
min_frames = int(min_prebuffer_seconds * fps)
producer_fps = state.producer_fps(window_seconds)
return (contiguous >= min_frames
        and producer_fps >= fps * grace_factor)
```

- 不满足条件 → 播放按钮无响应，状态不变。
- 播放中缓存缺失 → 自动暂停。

## 帧驱动

```python
self._timer = QTimer(self)
self._timer.setTimerType(Qt.TimerType.PreciseTimer)
self._timer.timeout.connect(self._tick)
self._timer.start(1000 // self._fps)  # ~100ms @ 10fps
```

`_tick()` 中：
1. 如果正在播放且到达下一帧时间 → 前进一帧。
2. 检查门控条件。
3. 更新 slider / label / cache bar（`blockSignals` 避免反馈循环）。

## PresentationFrameCache 复用

```python
key = (frame_index, display_side)
cached = self._presentation.get(key)
if cached is not None:
    self._qr_label.setPixmap(cached)  # 命中缓存，跳过解包+缩放
    return
# 否则：get_module_image → QImage → QPixmap → scaled → put
```

- 缩放后的 QPixmap 缓存在 `PresentationFrameCache` 中（LRU，64 MiB 上限）。
- 缩放级别改变时 `self._presentation.clear()` 清空重建。
- 与第一版 cv2 播放器共享同一个 `PresentationFrameCache` 类。

## CLI 错误处理

```python
# cli.py — cmd_encode()
except ImportError as exc:
    print(f"Error: {exc}", file=sys.stderr)
    sys.exit(3)
```

- 错误信息由 `require_pyside6()` 生成，格式清晰：
  ```
  PySide6 is required for the Qt display player and is included in the default qrstream package.
  Reinstall qrstream or install PySide6-Essentials directly.

      pip install --upgrade qrstream
      pip install PySide6-Essentials

  Details: No module named 'PySide6'
  ```
- CLI help 已更新：`--display` 说明使用内置 GUI 播放器，不再提示安装 extra。

## 改动清单

| 文件 | 改动 |
|------|------|
| `pyproject.toml` | 将 `PySide6-Essentials` 加入默认依赖；保留 no-op `gui` extra 兼容旧脚本 |
| `src/qrstream/display_player_qt.py` | **新建** — PySide6 播放器 (~830 行) |
| `src/qrstream/encoder.py` | `play_display_cache` → `play_display_qt`；`player` 测试钩子 |
| `src/qrstream/cli.py` | `--display` help 更新；`ImportError` 捕获 |
| `tests/test_display_encode.py` | 通过 `player=` 注入 fake player |

## 测试

- 286 个单元测试全部通过（含 1 skip, 34 deselected slow/e2e）。
- `test_display_encode` 通过 `player=fake_player` 注入 mock，不依赖 PySide6。
- 无 PySide6 环境下 `play_display_qt()` 调用抛清晰的 `ImportError`。

## 后续 TODO

1. ~~**鼠标滚轮缩放**~~ — 不做，+/- 缩放已改为调整窗口大小。
2. ~~**速度控制**~~ — 不做。
3. ~~**循环播放**~~ — ✅ 已实现。L 键切换，控件栏显示 "LOOP" 文字指示。
4. ~~**文件元数据面板**~~ — ✅ 已实现。I 键弹出美观深色 QDialog（Source / Encoding / QR Code / Playback 四区）。
5. ~~**终端侧编码进度**~~ — ✅ 已实现。CLI AUTO 模式自动选择 RichReporter，display 模式补充 `encode_done` 收尾。
6. ~~**窗口尺寸记忆**~~ — ✅ 已实现。QSettings 保存/恢复窗口 geometry。
7. **PIL 字体升级** — 未做，当前 QSS + 系统字体已满足需求。

### 额外改进（实现期间新增）

- **自适应窗口大小** — 首次打开取屏幕短边 70% 居中；后续恢复上次 geometry。
- **QR 跟随窗口缩放** — pixmap 始终适配 QLabel 实际尺寸，拖拽窗口即时重绘。
- **YouTube 式缓冲条** — 自绘 `_TimelineSlider`，灰色 buffer bar 叠在 groove 下。
- **时间轴显示时间** — `m:ss / m:ss` 格式替代帧号。
- **J/K 快捷键** — J 后退 1s、K 前进 1s。
- **单行控制栏** — 播放按钮 + 时间 + slider + loop 指示，高度仅 36px。
- **PresentationFrameCache 类型修复** — 原 np.ndarray 缓存不兼容 QPixmap，改用专用 `_PixmapCache`。