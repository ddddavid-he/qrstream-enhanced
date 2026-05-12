# `qrs encode --display` 第一版设计方案

状态：待审阅；本文档仅整理方案，审阅通过后再开始实现。

当前工作分支：`feature/display-mode`，按 `BRANCHING.md` 从 `dev` 拉出。

## 目标

1. 让 `qrs encode --display` 成为真正的 display-only 编码播放模式。
2. 第一版禁止 `--display` 与 `-o/--output` 同时使用。
3. `--display` 模式不写最终视频文件，不读正在写入或已写完的 MP4。
4. 将编码、QR module 渲染、缓存、播放统一在同一流程中完成。
5. 当 module 生成/渲染速度不足以支撑目标播放帧率且缓存不足时，不允许播放，避免录屏卡顿。
6. 使用紧凑 module-level 缓存，避免缓存完整 BGR 像素帧。

## 非目标

1. 第一版不支持 `--display + -o/--output`。
2. 第一版不做视频文件边写边读。
3. 第一版不做普通视频 GOP/keyframe 缓存；QR 帧天然独立，不需要视频关键帧模型。
4. 第一版不优先支持复杂编辑/拖拽时间轴；以稳定播放和录屏友好为主。

## CLI 行为

| 命令 | 第一版行为 |
|---|---|
| `qrs encode input.bin -o out.mp4` | 保持现有行为，生成视频文件 |
| `qrs encode input.bin --display` | display-only：编码、缓存、播放，不生成最终文件 |
| `qrs encode input.bin -o out.mp4 --display` | 报错，禁止共用 |
| `qrs encode input.bin` | 报错，必须指定 `-o` 或 `--display` |

建议错误信息：

```text
--display cannot be used together with -o/--output yet.
TODO: future versions may support generating the final video from display cache after encoding completes.
```

## 总体流水线

```text
input bytes
  -> LT encoder blocks
  -> QR module image at scale=1, including quiet zone
  -> bit-packed module frame cache
  -> playback-time nearest-neighbor integer upscale
  -> OpenCV display window
```

关键点：

- 缓存对象是 module 图，不是最终显示像素图。
- module 图每个 module 只占 1 个像素，包含 quiet zone。
- 缓存时进一步 bit-pack，每个 module 只占 1 bit。
- 播放时按窗口尺寸做最近邻整数倍放大。
- 不重复调用完整 BGR 渲染路径，避免抢占 `-o` 视频写入路径的 CPU；display-only 使用自己的轻量渲染路径。

## Module frame 表示

默认配置下：

- `qr_version = 25`
- QR 内容区 module side = `117`
- quiet zone = `4`
- 缓存 module side = `117 + 2 * 4 = 125`
- bit-packed row bytes = `ceil(125 / 8) = 16`
- 单帧缓存大小 = `125 * 16 = 2000 bytes`

建议结构：

```text
ModuleFrameCache
  module_side: int
  row_bytes: int
  total_frames: int
  chunk_size: int = 256
  chunks: dict[int, np.ndarray]
  valid: bitset / bytearray
  mode: full | window
  memory_budget_bytes: int
```

chunk 形态：

```text
uint8[chunk_size, module_side, row_bytes]
```

默认 V25 下一个 chunk：

```text
256 * 2000 bytes ~= 500 KiB
```

这样可以避免“每帧一个 Python 对象”的额外开销。

## 缓存策略

### 估算公式

```text
module_side = qr_content_side + 2 * border_modules
row_bytes = ceil(module_side / 8)
frame_bytes = module_side * row_bytes
total_cache_bytes = total_frames * frame_bytes
```

### 第一版默认策略

1. 若 `estimated_cache_bytes <= 128 MiB`：使用 full compact module cache。
2. 若 `duration <= 3600s` 且 `estimated_cache_bytes <= 192 MiB`：仍使用 full compact module cache。
3. 超过上述阈值：进入 window cache，优先保证从当前播放位置向后的连续窗口。

理由：

- 当前典型场景视频不超过 1 小时。
- 默认配置 1 小时仅约 `68.7 MiB` 原始 bit-packed 数据，含 chunk/状态开销约 `75~90 MiB`。
- `128 MiB` soft limit 已能覆盖大多数默认 display 场景。
- `192 MiB` 作为 1 小时内的保守上限，避免过早退化到复杂策略。

### Presentation cache

主缓存仍然使用 bit-packed module frame；同时允许维护一个有容量上限的完整显示帧缓存，用于减少播放时重复 unpack 和 resize 的成本。

建议第一版完整显示帧缓存容量：

```text
presentation_cache_budget = 64 MiB
```

策略：

- 只缓存播放窗口尺寸下的完整显示帧。
- 窗口尺寸变化后清空该缓存，重新按新尺寸生成。
- 使用 LRU 或环形窗口，优先保留当前播放位置附近的帧。
- 达到 `64 MiB` 后淘汰旧帧，不影响底层 module cache。

说明：该缓存是播放层优化，不替代 module cache，也不用于最终视频生成。即使关闭完整显示帧缓存，display-only 流程仍可正常工作。

## 播放控制与禁播条件

### 初始状态

- 窗口打开后默认暂停。
- 编码线程/任务开始生成 module frame 并写入缓存。
- UI 展示当前缓存进度、估算 producer FPS、目标播放 FPS。

### 允许播放条件

允许播放需满足至少一个条件：

1. 所有帧已经缓存完成。
2. 从播放位置开始存在足够连续缓存，并且滚动 producer FPS 不低于目标 FPS。

建议第一版参数：

```text
min_prebuffer_seconds = 3
producer_fps_window_seconds = 2 ~ 5
grace_factor = 1.05
```

判定：

```text
contiguous_cached_seconds >= min_prebuffer_seconds
and producer_fps >= target_fps * grace_factor
```

若 producer FPS 持续低于播放 FPS，播放按钮保持不可用或按键无效，只继续缓存。

### 播放中下溢处理

若播放中遇到目标帧未缓存：

1. 自动暂停。
2. 显示 buffering 状态。
3. 等待重新满足允许播放条件。

## 显示缩放

播放时使用最近邻缩放：

```python
cv2.resize(module_img, (display_side, display_side), interpolation=cv2.INTER_NEAREST)
```

推荐整数倍 module scale：

```text
module_px = floor(available_display_side / module_side)
display_side = module_side * max(1, module_px)
```

这样可以保持 QR module 边界清晰，避免插值造成识别风险。

## 需要新增/调整的代码模块

### `src/qrstream/qr_utils.py`

新增轻量 module 渲染函数，避免走完整 BGR 输出：

```text
generate_qr_module_image(...): np.ndarray[uint8]
```

返回值：

- 形状：`(module_side, module_side)`
- 值：`0/255` 或 bool-like 黑白值
- 包含 quiet zone
- 不做 BGR 转换
- 不做大尺寸 `box_size` 放大

可选再新增：

```text
pack_module_image(module_img) -> bytes/np.ndarray
unpack_module_frame(packed, module_side) -> np.ndarray
```

### `src/qrstream/display_cache.py`（新增）

职责：

- 估算缓存大小。
- 管理 full/window cache。
- 按 frame index 写入和读取 bit-packed module frame。
- 查询从某个 frame index 开始的连续缓存长度。

### `src/qrstream/display.py` 或 `src/qrstream/display_player.py`（新增）

职责：

- OpenCV 窗口管理。
- 播放/暂停/退出按键。
- 缓存进度和状态 overlay。
- 按播放时钟读取 cache 并放大显示。

### `src/qrstream/encoder.py`

新增 display-only 编码入口或复用现有 LT block 生成逻辑：

```text
encode_to_display(...)
```

职责：

- 生成 LT blocks。
- 调用 module 渲染。
- 写入 `ModuleFrameCache`。
- 与 player 协调 producer 状态。

### `src/qrstream/cli.py`

调整 `encode` 参数校验：

- `-o/--output` 从 argparse required 改为运行期条件校验。
- `--display` 与 `-o/--output` 同时出现时报错。
- 两者都不存在时报错。

## 未来 TODO：兼容 `--display + -o`

未来版本可以支持：

```text
qrs encode input.bin --display -o out.mp4
```

但实现方式应为：

1. display 阶段继续使用 module cache。
2. 编码/缓存完成后，从 module cache 统一生成最终视频文件。
3. 不读取正在写入的 MP4。
4. 不在 display 过程中维护大规模或持久化的完整 BGR frame cache；播放层可保留受 `64 MiB` 限制的临时 presentation cache。

这样可以保持 display 体验，同时避免边写边读视频文件带来的不稳定性。

## 后续 TODO

1. 优化 display UI/UX：
   - 状态信息继续避免遮挡 QR 区域。
   - 优化播放控制提示、进度条、当前帧/总帧展示。
   - 评估更适合录屏场景的默认窗口尺寸、缩放策略和快捷键布局。
2. 实现 `--display + -o/--output` 兼容：
   - display 阶段继续使用 module cache。
   - 编码/缓存完成后，从缓存统一生成最终视频文件。
   - 不读取正在写入的 MP4。
3. 对 `64 MiB` presentation cache、`128 MiB / 192 MiB` module cache 阈值做实际 benchmark，再决定默认值。

## 测试计划

按用户规则，构建和测试使用 `podman` 执行。

建议覆盖：

1. CLI 参数校验：
   - `--display + -o` 报错。
   - 无 `--display` 且无 `-o` 报错。
   - 仅 `-o` 保持现有行为。
   - 仅 `--display` 进入 display-only 路径。
2. module 渲染：
   - module side 正确。
   - quiet zone 正确。
   - bit-pack/unpack 后图像一致。
3. cache：
   - 估算大小正确。
   - chunk 写入/读取正确。
   - contiguous cached frames 查询正确。
   - full/window 模式阈值选择正确。
4. playback gating：
   - producer FPS 低于目标 FPS 且缓存不足时不可播放。
   - 缓存完成后允许播放。
   - 播放中 cache miss 自动暂停。
5. 回归：
   - 原有 `-o` encode/decode 流程不变。

## 实现顺序建议

1. CLI 参数语义调整，先保护第一版边界。
2. 增加 module 渲染与 bit-pack/unpack 单元测试。
3. 实现 `ModuleFrameCache`。
4. 实现 display-only producer。
5. 实现 OpenCV player 与播放 gating。
6. 用 podman 跑单元测试与现有关键回归。

## 审阅关注点

请重点确认：

1. 第一版是否只实现 display-only，不兼容 `-o`。
2. module cache 的 `128 MiB / 192 MiB / 1h` 阈值，以及 presentation cache 暂定 `64 MiB` 上限是否合适；这些数值均需实际测试后再定。
3. window cache 是否需要第一版完整实现，还是先在超阈值时提示用户降低参数或等待全量缓存。
4. 播放按键和 UI overlay 是否需要更明确的交互规格。
5. 是否接受新增 `display_cache.py` 与 `display_player.py` 两个模块。
