# Discovery — `encode --display` 时间轴降速与 Qt Player FPS 瓶颈

**Date**: 2026-05-17  
**Branch**: `dev`  
**Author**: investigation note for follow-up implementation / review

---

## 问题描述

用户观测到：

- 运行 `encode --display --fps=90` 时，display 窗口中时间轴推进 `1s` 的真实耗时明显大于 `1s`
- `--fps=60` 时也存在类似现象
- 这意味着 display 侧的实际推进 fps 低于命令行指定的 fps

本报告只回答两个问题：

1. **是谁卡住了整个流程，导致时间轴慢于配置 fps？**
2. **当前实现能在屏幕上实际渲染到多高的帧率？**

---

## 结论摘要

### 结论 1：拖慢时间轴的主瓶颈是 Qt 播放器主线程，不是 encode 生产线程

在真实 `encode --display` 路径中：

- producer 线程能够稳定产生约 `275-279 fps` 的模块帧
- cache 连续缓冲充足，没有出现播放被 `_can_play()` 频繁停掉的证据
- 但 Qt 主线程的实际 tick 频率明显低于目标 fps
- 且 `_tick()` 每次最多只推进 1 帧，不会追帧

因此时间轴按 `frame_index / fps` 计算后会自然变慢。

### 结论 2：当前 display 功能的 fps **不是严格保证值**

`--display --fps=N` 在当前实现中更接近“目标播放节奏”，而不是“严格保证的实际显示帧率”。

原因：

- `QTimer` 只是按目标 interval 触发 tick
- 每个 tick 还要承担 UI 更新、模块图解包、`QPixmap` 创建、缩放、`setPixmap()` 等主线程工作
- 一旦 tick 实际频率低于目标 fps，时间轴就会慢于墙钟时间

### 结论 3：当前机器屏幕的实际可见渲染上限是 `120 Hz`

通过 `QScreen.refreshRate()` 查询当前主屏：

- `Built-in Retina Display`
- `refreshRate = 120.0`

所以即使应用内部可以推进更高频率，**真正屏幕上可见的刷新上限仍约为 `120 fps`**。

---

## 代码证据

### 1. 时间轴按 `frame_index / fps` 计算

Qt display 时间标签直接来自：

- `src/qrstream/display_player_qt.py:541-546`

```python
cur_sec = self._frame_index / max(1, self._fps)
tot_sec = total / max(1, self._fps)
```

这意味着：

- 如果 `_frame_index` 增长慢于 `fps`
- 那么时间轴推进一定慢于真实时间

### 2. `_tick()` 每次最多只推进 1 帧

关键逻辑：

- `src/qrstream/display_player_qt.py:511-527`

```python
if self._playing and now >= self._next_frame_ts:
    nxt = self._frame_index + 1
    ...
    self._frame_index = nxt
```

即使某次 tick 晚到了很多，也只推进 1 帧，不会在一个 tick 内追多帧。

因此只要 tick 实际到达频率低于目标 fps，播放进度就会永久落后。

### 3. 每个 tick 都会做完整 UI / 渲染工作

- `src/qrstream/display_player_qt.py:538-539`

```python
self._update_controls()
self._update_display()
```

其中 `_update_display()` 继续触发：

- 从 cache 取模块图：`src/qrstream/display_player_qt.py:559`
- 解包模块帧：`src/qrstream/display_cache.py:280-284`
- `QPixmap.fromImage(...)`：`src/qrstream/display_player_qt.py:573`
- `pixmap.scaled(...)`：`src/qrstream/display_player_qt.py:580-584`
- `setPixmap(...)`：`src/qrstream/display_player_qt.py:587`

这整条链路都跑在 Qt 主线程。

### 4. `QTimer` 是目标节拍，不是完成保证

- `src/qrstream/display_player_qt.py:472-476`

```python
self._timer.start(max(1, 1000 // self._fps))
```

例如：

- `fps=60` -> interval 约 `16 ms`
- `fps=90` -> interval 约 `11 ms`

如果单次 tick + 渲染总耗时超过这个 interval，后续 tick 就会自然延后。

### 5. CLI 上显示的 encode `speed_fps` 不是 display 真实推进 fps

CLI encode 进度显示速度来自：

- `src/qrstream/encoder.py:937-952`

```python
speed = produced / elapsed
```

这是 producer 的平均产出速度，不是 Qt 窗口实际显示帧率，也不是时间轴推进速度。

---

## 实验设计

### A. 先测 producer 线程是否够快

方法：

- 调用 `encode_to_display()`
- 用 fake player 替代 Qt 播放器
- 只旁观 `state.produced / state.producer_fps() / cache.valid_count`

目标：排除“encode 线程本身不够快”的可能。

### B. 再测真实 end-to-end：encode + Qt display

方法：

- 保持真实 `encode_to_display()` + Qt player 路径
- 对 `_QRStreamWindow` 做只用于实验的轻量探针
- 记录：
  - effective fps
  - tick p95
  - draw/update p95
  - producer_fps
  - contiguous buffer
  - pause 事件

目标：区分是 producer gating 卡住，还是 Qt 主线程 tick / render 卡住。

### C. 查询屏幕刷新率

方法：

- 使用 `QApplication.primaryScreen().refreshRate()` 查询主屏

目标：确认“屏幕实际可见渲染上限”。

---

## 实验结果

### 1. producer 线程吞吐（不启用 Qt 播放）

测试条件：

- 2 MiB 随机输入
- `qr_version=40`
- `fps=90`
- `compress=False`

结果：

- 总帧数：`1129`
- producer 平均速度：**`278.5 fps`**
- 最近窗口 producer 速度稳定在约 **`278 fps`**

结论：

- producer 明显快于 `90 fps`
- 也显著高于 `_can_play()` 所需门槛 `90 * 1.05 = 94.5 fps`
- 因此 **不是 producer 跟不上**

### 2. 真实 end-to-end（encode + Qt display）

#### `--fps=60`

结果：

- effective fps: **`55.3`**
- tick p95: **`24.82 ms`**
- draw p95: **`7.80 ms`**
- producer p50: **`279.0 fps`**
- contiguous buffer p50: **`520.0`**
- pause events: **`0`**

解释：

- 目标 60 fps 对应理想 interval 为 `16.67 ms`
- 但 tick p95 已达到 `24.82 ms`
- 说明 Qt 主线程实际回调节奏明显慢于目标

#### `--fps=90`

结果：

- effective fps: **`78.8`**
- tick p95: **`25.72 ms`**
- draw p95: **`7.96 ms`**
- producer p50: **`275.9 fps`**
- contiguous buffer p50: **`520.5`**
- pause events: **`0`**

解释：

- 目标 90 fps 对应理想 interval 为 `11.11 ms`
- 但 tick p95 达到 `25.72 ms`
- producer 仍有约 `276 fps`
- contiguous buffer 也足够大
- 且没有 pause events

这说明：

- 不是 `_can_play()` 频繁停播
- 不是 producer 供帧不足
- **是 Qt 主线程 tick + render 路径跑不到目标节拍**

### 3. 屏幕刷新率

结果：

- primary screen: `Built-in Retina Display`
- `refreshRate = 120.0`

结论：

- 当前机器屏幕的实际可见刷新上限约为 **`120 fps`**
- 应用内部推进速度高于 120 时，也不会让屏幕真正显示超过 120 个不同可见刷新

---

## 归因结论

### 直接责任链条

1. `QTimer` 以目标 interval 触发 `_tick()`：`src/qrstream/display_player_qt.py:472-476`
2. `_tick()` 每次都要执行 UI 控件更新和图像渲染：`src/qrstream/display_player_qt.py:538-587`
3. 主线程实际无法稳定达到 60 / 90 fps 所要求的 tick 频率
4. `_tick()` 又不会追帧，只会最多推进 1 帧：`src/qrstream/display_player_qt.py:514-527`
5. 时间轴按 `_frame_index / fps` 计算：`src/qrstream/display_player_qt.py:543-546`
6. 最终表现为：**时间轴 1 秒的真实耗时大于 1 秒**

### 谁不是主瓶颈

- **不是 encode producer 线程**：实测约 `278 fps`
- **不是 `_can_play()` gating**：真实播放中 `pause_events = 0`，contiguous buffer 充足
- **不是屏幕 120Hz 限制导致 60 / 90fps 无法达到**：60 / 90 都远低于 120Hz

### 谁是主瓶颈

- **Qt player 主线程的 tick / render 路径**
- 尤其是每个 tick 上的：
  - `_update_controls()`
  - `_update_display()`
  - `get_module_image()` / `unpack_module_frame()`
  - `QPixmap.fromImage()`
  - `pixmap.scaled()`
  - `setPixmap()`

---

## 最终判断

当前 `display` 功能应被视为：

- **以 fps 为目标值的播放器**，而不是 **严格保证显示 fps 的播放器**
- 在高 fps 下存在明确的主线程瓶颈
- 因此 display 时间轴进度可能慢于用户指定 fps

也就是说：

> `--display --fps=N` 当前并不严格等价于“窗口将以 N fps 稳定显示并按真实时间推进”。

---

## 备注

本次调研已新增实验脚本：

- `scripts/bench_qt_player_fps.py`

它适合继续做以下验证：

- 不同窗口大小对 tick / draw 的影响
- 不同 `qr_version` 对主线程开销的影响
- `integer_scale` / window resize 对渲染成本的影响
- 后续优化前后的回归对比
