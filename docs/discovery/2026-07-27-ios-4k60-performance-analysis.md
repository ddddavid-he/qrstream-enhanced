# iOS 4K60 检测性能分析与实施边界

## 结论

当前 iOS App 不能宣称满足 4K60 逐帧检测，也尚未与 PyPI
`qrstream==0.10.4` 的有效吞吐能力对齐。

差距主要在相机格式选择、ZXingCpp 图像入口和 reader 策略，不在
RaptorQ/UniFFI：

1. App 固定使用 `.hd1920x1080`，没有选择设备支持的 4K60
   `AVCaptureDevice.Format` 和帧时长。
2. App 请求 BGRA。zxing-cpp 的 Apple wrapper 对 BGRA 每帧执行
   `CVPixelBuffer -> CIImage -> CGImage -> grayscale buffer`；NV12
   `420v/420f` 可以直接读取 Y plane。
3. `tryHarder=true` 被无条件用于每一帧，在这组 fixture 中没有提高最终恢复
   正确性，却显著增加检测延迟。
4. `alwaysDiscardsLateVideoFrames=true` 允许系统在 detector 落后时静默丢帧，
   与逐帧检测要求冲突。
5. SwiftPM 构建的 zxing-cpp 在一段固定录像帧上触发内部 assertion 并终止
   进程；PyPI wheel 没有以同样方式退出。
6. Rust `consumeQrText` 的耗时远低于 ZXing 检测，不是当前优先优化对象。

## 性能契约

移动端按以下顺序选择运行档位：

1. `3840x2160 @ 60 fps`
2. `1920x1080 @ 60 fps`
3. `1920x1080 @ 30 fps`

分辨率优先降级，因此不选择 `3840x2160 @ 30 fps`。如果设备连
1080p30 都不支持，可使用设备最大能力做 best effort，但必须明确报告未满足
最低性能要求。

一个档位只有同时满足设备能力和 detector 持续处理能力时才可被选中。
“逐帧检测”定义为：

- 不主动抽帧，每个采集并交付的帧都调用一次 ZXing reader。
- `captureOutput(_:didDrop:from:)` 计数为 0。
- 完成恢复前 `capturedFrameCount == detectionAttemptCount`。
- detector 的持续处理速率不低于输入速率，待处理队列不持续增长。
- 60 fps 档的单帧检测 `P95 < 16.67 ms`，工程目标为 `< 10 ms`。
- fixture 完整恢复且输出大小和 SHA-256 与 source 一致。
- 10 分钟循环回放中不崩溃，也不因 thermal throttling 失去当前档位。

完成恢复后的主动停止不算丢帧；应停止 detector 计数和相机帧投递，避免无效
功耗。

## 可重复输入

分析使用四段用户录制的 4K60、10-bit HEVC 视频。源 QRStream 视频由
`scripts/make_ios_replay_recording_kit.py` 生成，payload 是确定性的，并已
离线验证恢复文件 SHA-256。

第一轮 Swift 回放由 `AVAssetReader` 统一输出 1920x1080。以下数据只能证明
优化方向和 1080p 余量，不能作为 iPhone 4K60 达标证据。最终结论必须来自
目标 iPhone 上的原生 4K NV12 相机/回放链路。

## Swift 检测路径对照

`balanced` fixture、同一 Swift wrapper、1920x1080：

| Pixel format | tryHarder | P50 | P95 | QR hit frames | Unique payloads | SHA |
|---|---:|---:|---:|---:|---:|---|
| BGRA | true | 17.63 ms | 20.51 ms | 1014 | 545 | match |
| NV12 | true | 12.05 ms | 13.78 ms | 939 | 544 | match |
| BGRA | false | 7.90 ms | 9.85 ms | 1014 | 545 | match |
| NV12 | false | 3.04 ms | 3.83 ms | 939 | 544 | match |

相对当前 `BGRA + tryHarder=true`，改为 `NV12 + tryHarder=false` 后 P50
约快 5.8 倍，唯一 payload 只从 545 变为 544，最终恢复不受影响。

四段 fixture 的 `NV12 + tryHarder=false` 结果：

| Case | ZXing P50 | ZXing P95 | Unique payloads | 模拟 60 fps | 模拟 30 fps |
|---|---:|---:|---:|---|---|
| baseline | 2.64 ms | 3.32 ms | 276 | complete, SHA match | complete |
| balanced | 3.04 ms | 3.83 ms | 544 | complete, SHA match | complete |
| dense | 4.09 ms | 5.17 ms | 705 | complete, SHA match | complete |
| throughput | 4.00 ms | 5.17 ms | 565 | complete, SHA match | incomplete |

当前 `BGRA + tryHarder=true` 的 P50 为 16.31～20.11 ms，在模拟 60 fps
latest-frame 调度下产生 1.4%～17.1% 背压丢帧。

`throughput` 源视频以 45 fps 发送 QR。模拟 30 fps capture 时，即使 detector
足够快也不能恢复；60 fps capture 可以恢复。这说明显式选择 60 fps 相机格式
是协议有效吞吐的必要条件。

## 稳定性发现

App 当前使用的 Swift zxing-cpp wrapper 在 `balanced` 的标准化第 390 帧
稳定终止：

```text
Assertion failed: (l1.isValid() && l2.isValid()),
function intersect, file RegressionLine.h, line 153.
```

`tryHarder=false` 和 wrapper 3.0.0 仍会触发。给本地 Release 构建定义
`NDEBUG` 后四段可以完成，但这只用于继续测量，不能单独视为完整修复。生产
修复需要保留 crash frame、核对上游修复，并验证无 abort 也没有无效识别结果。

## PyPI 0.10.4 对照

PyPI 能正确读取四段 4K60 文件并恢复相同 SHA，但它会使用 crop、缩放、自适应
采样和 early termination，因此不能描述为“对每个原始 4K 帧做全分辨率
ZXing 检测”。

| Case | 默认 workers 墙钟 | workers=1 墙钟 | 结果 |
|---|---:|---:|---|
| baseline | 28.01 s | 16.44 s | SHA match |
| balanced | 19.95 s | 19.60 s | SHA match |
| dense | 17.33 s | 17.49 s | SHA match |
| throughput | 17.38 s | 18.79 s | SHA match |

线程池没有稳定获益，因此 iOS 不应先照搬 PyPI worker 数。先消除 BGRA 转换和
无条件 `tryHarder`，再根据真机数据决定是否需要 ROI 或并行 detector。

Rust/UniFFI `consumeQrText` P50 约 0.02～0.06 ms、P95 约
0.02～0.08 ms；只有最终完成包出现约 1 ms 峰值。

## 实施顺序

### 第一阶段：正确选择并观测采集档位

1. 枚举后置摄像头格式，按 4K60、1080p60、1080p30 选择。
2. 同时设置 active format、min/max frame duration 和 NV12 pixel format。
3. 增加实际采集 FPS、detector FPS、drop callback、队列深度、当前档位和
   thermal state 指标。
4. 档位选择失败或持续处理能力不足时，记录原因后降级。

### 第二阶段：缩短逐帧检测路径

1. 使用 NV12 Y plane 零拷贝入口。
2. 固定 `tryHarder=false`；普通检测失败、返回空结果或抛错时立即丢弃当前帧，
   不在串行检测队列中执行 hard-mode 重试。
3. 完成恢复后停止识别和 capture delivery。
4. 修复 zxing-cpp assertion，并将触发帧纳入回归。

### 第三阶段：回放与真机 gate

1. 摄像头和 `AVAssetReader` 必须复用同一帧处理器。
2. 回放保留视频原始 4K 分辨率和 NV12 输出，不再标准化到 1080p。
3. 四段 fixture 每档至少三轮，并循环到 10 分钟观察温控。
4. Mac 回放只做提交间相对回归；4K60 达标结论只来自目标 iPhone。

ROI、temporal gate 和多 detector worker 会改变“每个完整帧都执行一次检测”
的含义，暂不进入第一阶段。若真机 4K60 仍不达标，需要先重新确认是否允许
ROI 内逐帧检测，再引入这些策略。

## 实施后本地验证

实现改为 NV12、normal reader 和精确的 zxing 防御性补丁后，Mac 回放不再把
录像标准化到 1080p，而是让 `AVAssetReader` 输出全部原始 3840x2160 帧。
Release 构建仍保留 zxing assertion。

| Case | 全帧处理 FPS | ZXing P50 | ZXing P95 | 全帧恢复 |
|---|---:|---:|---:|---|
| baseline | 121.74 | 7.89 ms | 9.82 ms | complete, SHA match |
| balanced | 113.70 | 8.69 ms | 10.77 ms | complete, SHA match |
| dense | 101.81 | 9.65 ms | 11.68 ms | complete, SHA match |
| throughput | 97.68 | 9.36 ms | 12.25 ms | complete, SHA match |

四段的平均全帧处理能力都超过 60 fps，P95 都低于 16.67 ms。旧的容量 1
latest-frame 模拟在 throughput 的少量长尾帧上仍会覆盖 18 帧；新契约使用
全帧 FIFO，允许可消化的瞬时积压，因此不以 latest-frame 结果判定失败。

这组结果仍然只证明 Mac 和算法路径有 4K60 余量。已发现的 iPhone 15 Pro
当时处于 `unavailable`，所以相机真实 delivered FPS、drop callback、10 分钟
thermal state 和运行时降级仍需在设备重新连接后验证。

同时完成：

- 7 个 Swift 单元测试通过。
- arm64 iOS Simulator Debug 构建通过。
- arm64 iOS Simulator Release 构建通过。
- `balanced` 原 assertion 帧在 assertion 保持启用时不再终止进程。
