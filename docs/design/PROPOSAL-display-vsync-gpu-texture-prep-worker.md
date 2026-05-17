# 设计提案 — Display 渲染路径可选升级：Vsync 对齐 / GPU 纹理 / 后台预渲染

**Date**: 2026-05-17
**Status**: 按需启动（On-Hold）——当前 120fps 已达标，无迫切需求
**前置依赖**: `fix/display-fps-phase1`（子进程 producer + 零跳帧保证，已完成）

---

## 背景：为什么写这份文档

Phase 1 修复了 `encode --display` 的帧率瓶颈（GIL 争抢导致
120fps 目标仅达 27fps）。修复后 120Hz ProMotion 设备实测达到
120fps，零跳帧。

原始性能报告建议了进一步优化方向（vsync 驱动渲染、GPU 纹理路
径等）。经过分析，这些方向**在当前场景下没有实际收益**，但可能
在未来特定条件下变得必要。本文档记录分析结论和对接点，供未来
决策参考。

---

## Phase 1 完成后的性能基线

| 指标 | Phase 1 前 | Phase 1 后 |
|------|-----------|-----------|
| 60fps 有效帧率 | 55.3 fps | ~60 fps |
| 120fps 有效帧率 | ~27 fps | ~120 fps |
| 跳帧 | 存在 late-reset 跳帧 | 零跳帧保证 |
| 每帧渲染成本（GUI 线程） | 3.09ms (GIL 争抢) | ~0.25ms (子进程模式) |
| 帧预算占用率 (120fps) | 37% (3.09/8.33) | 3% (0.25/8.33) |

---

## 可选升级方向

### 方向 A: Vsync 对齐（帧时序均匀化）

**解决什么问题**：QTimer 是"尽力而为"的调度，帧间隔有抖动
（目标 8.33ms，实际可能 7-10ms 交替）。Vsync 对齐让每帧精确
落在显示器 blanking interval，帧间隔严格均匀（± <0.1ms）。

**对 QRStream 的意义**：每个 QR 码获得等长曝光时间，降低短帧
被相机漏捕的概率。

**实现方式**：`QOpenGLWidget` + `swapBuffers()` 自动对齐 vsync。
Qt 封装了后端差异（macOS Metal / Windows ANGLE-D3D / Linux Mesa）。

**当前收益**：**边际。** 零跳帧保证已确保每帧展示；帧时序不均
匀目前未造成实际问题。

**何时启动**：实测发现相机因帧时序不均匀而漏帧。

---

### 方向 B: 后台预渲染（Prep Worker）

**解决什么问题**：当前每帧在 GUI 线程做 unpack → QImage →
QPixmap → scale（0.25ms），占帧预算 3%。Prep Worker 在后台线
程提前完成这些工作，GUI 线程只做 `setPixmap()`（0.01ms）。

**实现方式**：
```
SharedFrameBuffer ──► Prep Worker 线程 ──► ring buffer of QImage
                                                  │
GUI _tick() ◄──── QPixmap.fromImage + setPixmap ──┘
```

Phase 1 的 `SharedFrameBuffer` 扁平内存布局已为此预留。

**当前收益**：**极低。** 0.25ms/帧远非瓶颈。

**何时启动**：
- 需要 4K 分辨率下的高帧率（scale 成本上升）
- 需要同时运行其他 CPU 密集型 GUI 操作
- `_update_display` p95 超过帧预算的 30%

---

### 方向 C: OpenGL 纹理路径

**解决什么问题**：跳过 CPU scale 路径，用 GPU 直接 texture
upload + fragment shader scaling。

**跨平台兼容性**：

| 平台 | Qt 后端 | GPU 厂商 | 状态 |
|------|---------|---------|------|
| macOS | Metal (MoltenVK) | Apple Silicon / Intel | ⚠ OpenGL 已废弃，Qt 走 Metal 但 Python binding 偶有兼容问题 |
| Windows | ANGLE (D3D11→ES) | NVIDIA / AMD / Intel | ✅ 稳定 |
| Linux X11 | GLX | NVIDIA / AMD / Intel (Mesa) | ✅ 稳定 |
| Linux Wayland | EGL | 同上 | ⚠ NVIDIA 专有驱动历史上有问题，2025 后改善 |
| CI headless | 无 GPU | — | 需要 `QT_QPA_PLATFORM=offscreen` 跳过 GL |

PySide6 `QOpenGLWidget` + `QOpenGLTexture` 写一次代码跑所有
平台（OpenGL ES 2.0 公分母），但需要处理上述 edge case。

**当前收益**：**无。** 185×185 → 900px CPU scale 只需 0.1ms。

**何时启动**：
- module_side 增长到 300+
- 需要支持 240Hz+ 显示器
- 需要 4K+ 分辨率渲染

---

## Phase 1 已预留的对接点

| Phase 1 组件 | 对接方式 |
|-------------|---------|
| `SharedFrameBuffer` 扁平内存 | Prep Worker 直接读取；GPU 路径可 mmap 到 PBO |
| `SharedBufferCacheAdapter` duck-typed 接口 | 任何新渲染器可替换 `_QRStreamWindow` 而不影响 adapter |
| `_update_display()` | 替换为 GL paint 或 prep worker 消费路径 |
| `_PixmapCache` (OrderedDict LRU) | Prep Worker 模式下改为 ring buffer |
| 零跳帧帧推进 (`_tick`) | Vsync 模式下由 `swapBuffers` 驱动 |

---

## 推荐优先级（如需启动时）

1. **Prep Worker** — 最简单、风险最低，优先在 4K 需求出现时
2. **Vsync 对齐** — 仅当帧时序问题导致实际拍摄失败
3. **OpenGL 纹理** — 最后手段，仅当 CPU 路径成为瓶颈

---

## 参考

- 性能报告: `display-fps-strict/2026-05-17`
- Phase 1 分支: `fix/display-fps-phase1`
- ProcessPool 历史实验: `dev/ENCODER_PROCESSPOOL_ABANDONED.md`
- GIL 分析: `docs/GIL_ANALYSIS.md`
