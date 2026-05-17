# Discovery — Calibrate FPS Ceiling、相位漂移与单向链路约束

**Date**: 2026-05-18  
**Branch**: `dev`  
**Author**: investigation note for follow-up implementation / review

---

## 问题描述

用户在运行 `calibrate` 后，根据推荐参数录制 QR 流，解码结果极差：

- 校准视频（IMG_9634.MOV）：录像设备实际帧率 59.94fps
- 校准给出推荐：`fps=60`
- 按推荐录制的视频（IMG_9635.MOV）：时长 330s，仅恢复 **38.6%**（6383/16545 blocks）

解码日志：

```text
Decoding incomplete: 6383/16545 source blocks recovered from 7676 encoded blocks.
Try recording the QR stream longer to capture more unique frames.
```

这个问题最初看起来像是一个简单的 `round()` / `floor()` bug，但继续往下看会发现，里面其实叠了三层问题：

1. 录像端 fps ceiling 算错了，错误地放行了 `60fps`
2. 校准窗口太短，看不出长时间相位漂移
3. 当前系统是“两台设备 + 单向链路”，分析侧实际上**拿不到发送端真实显示刷新率**，因此不能把 display-side cadence 当成推荐输入

---

## 结论摘要

### 结论 1：`_capture_fps_ceiling()` 使用了 `round()` 而非 `floor()`

当前代码（`src/qrstream/calibrate.py:726-732`）：

```python
return max(1, int(video_metadata.fps + 0.5))   # ← 等价于 round()
```

对于 NTSC 标准 59.94fps 的录像设备：

```text
int(59.94 + 0.5) = int(60.44) = 60
```

结果是 `fps_ceiling = 60`，导致 `fps=60` 通过过滤，被推荐为编码帧率。

### 结论 2：短校准窗口掩盖了相位漂移，校准对“等速边界”不敏感

`standard` preset 下，60fps 测试段只有约 24 帧，持续约 **0.4 秒**。

相机（59.94fps）与显示器（60fps）之间的相位漂移速率大约是：

```text
drift_rate = |60.0 - 59.94| / 60.0 ≈ 0.001 帧/帧 = 0.1%
```

在 0.4 秒窗口内：

```text
accumulated_miss ≈ 24 帧 × 0.001 = 0.024 帧
```

这个量级几乎不可能被当前校准统计观察到，所以 60fps 段在短窗口里看起来是“正常”的。

### 结论 3：长时间录制时，60fps 编码会产生系统性唯一帧损失

在 330 秒录制中，纯时钟偏差理论上就会累计：

```text
accumulated_miss ≈ 330s × 60fps × 0.001 ≈ 20 帧
```

但更糟的不是这 20 帧本身，而是 miss 的分布不是平滑均匀的，而是会在相位接近边界时成簇出现：

- 相机与显示器没有相位锁定（PLL / genlock）
- 初始相位是随机的，频率还有 ppm 级偏差
- 相位接近边界时，更容易采到混合帧、撕裂帧、模糊帧
- 这些帧对 QR 检测不是“轻微降质”，而是直接不可解

因此结果会表现为“看似录了很久，但唯一可用帧不够”，与用户这次实测完全一致。

### 结论 4：即使相机精确运行在 60.000fps，也不能把 60fps 显示流当作可靠推荐

要稳定抓到每个唯一显示帧，采样端不能只是“名义上等于”信号端。

对本项目这个场景：

- 设备 A 的屏幕上播放 `fps = x` 的 QR 视频
- 设备 B 用 `fps = y` 的摄像头录制
- 后续只靠录制结果解码

当 `x == y` 时，两个独立时钟之间仍然存在：

1. 相位偏移（phase offset）
2. 频率漂移（frequency drift）
3. 抖动（jitter）

所以 `camera_fps = display_fps` 不是安全边界，最多只能说“看起来接近边界”。作为推荐值，应该要求：

```text
recommended_encode_fps < measured_capture_fps
```

### 结论 5：在当前“两台设备 + 单向链路”架构下，分析侧无法把显示器刷新率当作推荐输入

这是这次讨论里最关键的架构结论。

当前工作流本质上是：

1. 设备 A 生成或播放 calibration 视频
2. 设备 B 录制这个视频
3. 在分析阶段，只拿到录制文件本身

分析侧真正已知的，只有：

- 录制视频的 metadata（例如 `29.97/59.94fps`）
- calibration 流里编码进去的 step / ladder 信息
- 实测 detect rate

分析侧**不知道**的包括：

- 设备 A 实际用的是哪块屏幕
- 那块屏幕真实刷新率是多少
- 视频播放时的实际推进 cadence 是多少
- calibration 视频是否是在一个设备上生成、再拿到另一台设备播放
- 生成时写入的 `display_hz` 与播放时真实 `display_hz` 是否一致

因此：

- **发送端**的 `--display-hz` 仍然可以作为“生成校准视频时的 ladder cap”输入
- **分析端**不能把某个 `display_hz` 当成可验证的 ground truth 来参与推荐

换句话说，**依赖分析侧知道发送端显示器刷新率来做推荐，这个设计在当前单向协议下不可行**。

### 结论 6：在当前整数 FPS ladder 下，真正决定结果的往往不是“留 1fps 还是留 5%”，而是 ladder 本身太粗

当前主要 ladder：

- `fast`: `[10, 15, 20, 25, 30, 45, 60]`
- `standard`: `[10, 12, 15, 18, 20, 25, 30, 45, 60]`
- `high`: `[10, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 100, 120]`

如果推荐规则改成“严格小于 capture fps”，那么在当前 ladder 下会直接落到这些档位：

| 录像设备 fps | `fast/standard/full` 最高安全候选 | `high` 最高安全候选 |
|---|---:|---:|
| ~29.97 / 30.00 | 25 | 25 |
| ~59.94 / 60.00 | 45 | 50 |
| ~119.88 / 120.00 | 60 | 100 |

这意味着：

- `59.94 -> 59` 与 `59.94 * 0.95 -> 56`，在 `standard` preset 下最后都会落到 `45`
- `30.00 -> 29` 与 `30.00 * 0.95 -> 28`，最后都会落到 `25`

所以在现有 ladder 下，**先把“严格不等于 capture fps”做对，比精细争论 margin 是 `-1fps` 还是 `-5%` 更重要**。

---

## 代码定位

### Bug 位置：录像端 FPS ceiling 算错

**`src/qrstream/calibrate.py:726-732`** — `_capture_fps_ceiling()`

```python
def _capture_fps_ceiling(video_metadata: VideoMetadata | None) -> int | None:
    if video_metadata is None or video_metadata.fps is None:
        return None
    if video_metadata.fps <= 0:
        return None
    # Phone videos often report 29.97/59.94; treat them as 30/60 ceilings.
    return max(1, int(video_metadata.fps + 0.5))   # ← BUG：round() 而非 floor()
```

注释里“treat them as 30/60 ceilings”的意图本身就不对。  
59.94fps 不应被提升成 60fps ceiling；如果系统要安全，反而应保证推荐值严格低于真实 capture fps。

### 使用位置 1：分析阶段过滤 `effective_fps_detect_rates`

**`src/qrstream/calibrate.py:899-913`**

```python
fps_ceiling = _capture_fps_ceiling(video_metadata)
if fps_data_reliable and fps_ceiling is not None:
    filtered = {
        fps: rate for fps, rate in effective_fps_detect_rates.items()
        if fps <= fps_ceiling
    }
```

只要 ceiling 被算高，`60 <= 60` 就会放行。

### 使用位置 2：optimizer 过滤候选 FPS

**`src/qrstream/calibration_optimizer.py:176-177`**

```python
if config.capture_fps_ceiling is not None:
    fps_values = [fps for fps in fps_values if fps <= config.capture_fps_ceiling]
```

这里也完全依赖 `capture_fps_ceiling` 的正确性。

### 设计错位位置：分析阶段把 `analysis_display_hz` 当成显示侧代理值

**`src/qrstream/calibrate.py:1567-1577`**

```python
analysis_display_hz = (
    max(inferred_fps_ladder)
    if inferred_fps_ladder else
    (max(metadata_fps_values) if metadata_fps_values else 60)
)

config = resolve_preset(preset_name, display_hz=analysis_display_hz)
```

这个 `analysis_display_hz` 本质上只是：

- 从 calibration ladder 推出来的最大 tested fps，或
- 一个兜底默认值

它**不是**发送端真实显示器刷新率。

后面又把这个值继续传给：

**`src/qrstream/calibrate.py:1648-1658`**

```python
result = compute_recommendations(
    ...
    display_hz=analysis_display_hz,
)
```

在“两台设备 + 单向链路”的约束下，这个值不能被解释成显示侧 ground truth。它最多只是分析时为了恢复 preset 配置而用的 ladder 上界近似值。

### 可行但需要区分的部分：生成阶段使用 `display_hz` 仍然合理

**`src/qrstream/calibrate.py:1122-1125`**

```python
config = resolve_preset(
    preset_name, display_hz=_get_display_refresh_rate())
```

这部分是**发送端自己**用自己的显示器信息去裁剪测试 ladder。  
这个用途没有问题。

问题不在“发送端能不能知道自己的显示器刷新率”，而在“分析端能不能把这个值当成自己知道的事实继续推荐”。后者在当前协议下做不到。

---

## 实验数据

### IMG_9634.MOV（校准视频）

```text
average_rate : 59.959211 fps
base_rate    : 59.940060 fps  (NTSC 标准：60000/1001)
time_base    : 1/600
frames       : 3675
duration     : 61.3s
```

当前 `_capture_fps_ceiling()` 的计算：

```text
int(59.959 + 0.5) = int(60.459) = 60   ← 错误地允许了 fps=60
```

最小修复后的 `floor()` 计算：

```text
int(59.959) = 59   ← 至少不会再错误放行 60
```

### 相位漂移估算

| 场景 | 校准段（0.4s） | 实际传输（330s） |
|---|---:|---:|
| 纯时钟偏差导致 miss | ~0.024 帧 | ~20 帧 |
| 可观测性 | 几乎不可见 | 显著 |
| 边界附近突发 miss | 极少触发 | 多次触发 |

### 当前 ladder 下，严格 `< capture_fps` 的实际效果

| 规则 | 59.94fps 设备 | 60.00fps 设备 | 29.97fps 设备 | 30.00fps 设备 |
|---|---:|---:|---:|---:|
| `round()` | 60 | 60 | 30 | 30 |
| `floor()` | 59 | 60 | 29 | 30 |
| “最大整数且严格小于 raw fps” | 59 | 59 | 29 | 29 |

对整数 ladder 而言，最后一行才准确表达了当前系统真正需要的安全边界。

---

## 外部资料补充（定性支持，非精确定量证明）

本次额外查阅到的一些行业资料，能支持“无同步的 camera/display cadence 会在相位边界处产生可见 artifacts”这一点，但**不足以推导出一个通用且精确的 margin 公式**。

- Image Engineering, *Camera Timing Parameters*  
  <https://www.image-engineering.de/library/image-quality/factors/1294-timing>  
  该文把 `display refresh rate` 和 `rolling shutter speed` 都明确归为 timing 参数，指出较低刷新率会带来基于 display 信息的对齐问题，而 rolling shutter 的逐行读取会引入时序相关失真。

- Brompton, *Genlock Settings*  
  <https://www.bromptontech.com/online-help/Content/Tessera%20User%20Manual/03.%20Feature%20Topics/12.3.1%20-%20Genlock.htm>  
  文档明确写到：没有 genlock 时，屏幕在相机里可能出现 rolling dark lines；同时还需要微调 shutter speed 和 phase offset 才能减少 artifacts。这直接支持“同 nominal fps 但不同步时，画面并不稳定可采”的判断。

- INFiLED, *Broadcast LED Walls: How to Eliminate Flicker, Moiré, and Camera Issues*  
  <https://www.infiled.com/blog/broadcast-studio-led-walls-how-to-eliminate-flicker-moire-and-camera-issues/>  
  文章强调很多 on-camera artifacts 实际根源是 LED wall 的 timing、scan characteristics 和 sync 问题，而不是“相机参数写成同样的 fps 就万事大吉”。

这些资料共同支持的，是**定性结论**：

- 无 genlock / 无 phase lock 时，camera 与 display 的时序边界会产生 artifacts
- 这些 artifacts 与刷新、快门、rolling shutter、相位偏移共同相关
- “名义 fps 相等”并不等于“稳定可用”

但它们**不能**告诉我们一个可以普适复用的精确 margin，例如：

- 留 `1fps`
- 留 `3fps`
- 留 `5%`
- 留 `10%`

因此本项目的 margin 设计，应该优先追求：

1. 规则解释简单
2. 与当前 ladder 结构匹配
3. 在单向链路下可实现且可验证

而不是伪精确地引入一个“看起来科学”的百分比。

---

## 修复方向

### 方案 A：最小必要修复，把 `round()` 改成 `floor()`

```python
def _capture_fps_ceiling(video_metadata):
    ...
    return max(1, int(video_metadata.fps))   # 对正数等价于 floor()
```

效果：

- 59.94 不再被错误放行到 60
- 29.97 不再被错误放行到 30
- 这是最小、正确、低风险的修复

但它仍然保留了一个问题：**精确整数 fps 的设备仍可能推荐等速值**。

### 方案 B：推荐阶段改成“严格小于 capture fps”（当前架构下更合理的产品策略）

如果系统继续使用整数 fps ladder，那么“允许的最大整数候选”应该是：

```python
strict_ceiling = max(1, math.ceil(raw_capture_fps) - 1)
```

例子：

- 59.94 -> 59
- 60.00 -> 59
- 29.97 -> 29
- 30.00 -> 29
- 120.00 -> 119

这比简单的 `int(raw) - 1` 更准确，因为：

- `int(59.94) - 1 = 58` 过于保守
- 我们真正要表达的是“允许的最大整数，且它必须严格小于 raw fps”

**这是当前我更倾向的推荐策略**，原因是：

- 它不依赖分析侧知道发送端显示器刷新率
- 它与“等速边界不安全”的物理直觉一致
- 在当前 coarse ladder 下，它的实际推荐档位与更复杂的百分比 margin 大多相同

### 方案 C：如果觉得吞吐损失太大，优先改 ladder，而不是重新依赖 `display_hz`

如果团队觉得：

- `60 -> 45`
- `30 -> 25`

这个台阶太大，那么更合理的改进方向是：

- 在 `fast/standard/full` ladder 中加入中间档位，例如 `50/55`
- 适当拉长 fps step 的持续时间，让边界问题在校准阶段更容易暴露

而不是试图让分析侧重新“猜”发送端 display cadence。

---

## 不推荐的方向

- 不要在分析侧继续把 `analysis_display_hz` 解释成发送端真实显示刷新率。
- 不要因为“生成端可以知道 display_hz”就推导出“分析端也可以据此做安全推荐”。这两件事在当前协议里不是同一件事。
- 不要把 `capture_fps_ceiling` 改成 float 然后在 optimizer 里大面积透传；当前 ladder 本身是整数，收益很有限，改动面却很大。
- 不要执着于一个看起来精确的比例 margin（例如固定 5%）。在现有 ladder 下，这个比例大多不会改变最终候选，反而会制造伪精度。

---

## 影响范围

- 所有使用 NTSC 制式相机（iPhone、大多数 Android、GoPro 等）的用户
- 所有“发送端屏幕播放、接收端相机录制、后续离线分析”的两设备单向链路场景
- `fast` / `standard` / `full` preset 都会受影响，因为它们都包含 30 / 60 这些边界点
- `high` preset 受影响更复杂，但同样存在“等速边界不安全”的问题，只是 ladder 更细一些

---

## 最终建议

1. 先做最小 bug fix：`round()` -> `floor()`。
2. 设计层面明确：**分析侧推荐不再依赖 display-side refresh rate**。
3. 产品策略上采用：**recommended fps 必须严格小于 measured capture fps**。
4. 对整数 ladder，推荐用“最大整数且严格小于 raw capture fps”这一规则来实现，而不是 `floor()-1`。
5. 如果吞吐损失不能接受，优先增加中间 fps ladder 档位并延长校准段，而不是回退到 display-hz-based recommendation。

