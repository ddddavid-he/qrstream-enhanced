# iOS 视频回放实时解码基准与录制 Fixture 方案

## 1. 目标

当前 iOS App 通过 `AVCaptureVideoDataOutput` 接收摄像头帧，在串行
`detectionQueue` 上调用 ZXingCpp，然后将唯一 QR 载荷交给
`RustDecodeSession`。手持拍摄适合验证真实信道，但不适合作为日常性能基准：
输入不可重复、人工成本高，而且很难区分信道质量与解码器性能。

本方案用一组固定录制视频作为可重复输入，覆盖两类验证：

1. **实时行为验证**：按录制视频时间戳模拟摄像头持续产帧，验证 4K60、
   1080p60、1080p30 三档是否能逐帧检测。
2. **性能与正确性回归**：测量 ZXingCpp、去重和 Rust 解码阶段的耗时，
   并以源载荷 SHA-256 验证最终输出。

录制视频不能完全替代摄像头测试。它不覆盖相机 ISP、自动对焦、自动曝光和
新的手抖轨迹，但它能把一次真实拍摄固定下来，使后续提交在相同像素输入上
反复比较。

## 2. 测试链路

生产摄像头和视频回放必须复用同一个帧处理器，避免形成只在 benchmark 中
存在的第二套识别实现：

```text
AVCaptureVideoDataOutput ─┐
                          ├─> QRFrameProcessor
AVAssetReader Replay ─────┘      ├─ ZXingCpp read
                                 ├─ payload dedupe
                                 └─ RustDecodeSession.consumeQrText
```

建议从 `ScannerViewController` 抽取以下职责：

- `QRFrameProcessor`
  - 持有并复用同一个 `ZXIBarcodeReader`
  - 接受 `CVPixelBuffer`
  - 执行 ZXingCpp 识别、格式过滤和 payload 去重
  - 将唯一 payload 交给 decode session
- `ReplayFrameSource`
  - 使用 `AVAssetReaderTrackOutput`
  - 输出相机链路相同的 NV12 `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`
  - 保留每帧 presentation timestamp（PTS）
- `ReplayMetricsCollector`
  - 只记录元数据和计时，不记录 QR payload 或恢复文件内容
  - 导出版本化 JSON；需要人工查看时再转换成 CSV

摄像头与回放应使用一致的 ZXingCpp 参数：

- QR Code only
- `tryHarder = false`；仅在连续普通检测失败后低频启用 hard fallback
- `tryRotate = false`
- `tryInvert = false`
- `tryDownscale = true`
- `maxNumberOfSymbols = 4`

## 3. 回放模式

### 3.1 `realtimeAllFrames`：主要验收模式

按视频 PTS 以 1.0 倍速产生帧。生产者和识别消费者彼此独立，每个调度帧
必须执行一次检测：

1. 到达下一帧 PTS 时，生产者提交该帧。
2. 帧进入有界 FIFO，识别器按 PTS 顺序处理。
3. 每轮结束时 `scheduledFrameCount` 必须等于 `processedFrameCount`。
4. 队列深度如果持续增长，当前档位判定不达标并进入下一降级档位。

不能通过无限缓存把慢 detector 伪装成逐帧实时处理。报告必须记录最大队列
深度和回放结束时的处理拖尾；持续处理 FPS 必须不低于输入 FPS。

回放读取应在独立队列上进行，并记录 `sourceStarvationCount`。如果
`AVAssetReader` 来不及在 PTS 前准备下一帧，该轮结果不能解释为纯识别性能
退化。

### 3.2 `realtimeLatest`：兼容诊断模式

使用容量为 1 的 latest-frame 缓冲复现旧版
`alwaysDiscardsLateVideoFrames = true` 行为。该模式用于量化历史实现的背压
丢帧，不用于 4K60 逐帧验收。

### 3.3 `unthrottledAllFrames`：能力上限模式

忽略 PTS，不主动丢帧，顺序处理全部视频帧。用于测量：

- 单设备最大持续处理 FPS
- ZXingCpp 延迟分布
- Rust consume 开销
- 与 `realtimeLatest` 的差异

该模式不是实时摄像头行为，不用于判定线上丢帧率。

## 4. 指标口径

### 4.1 输入和调度

- `sourceFrameCount`：`AVAssetReader` 成功读出的帧数
- `scheduledFrameCount`：进入实时调度器的帧数
- `processedFrameCount`：实际调用 ZXingCpp 的帧数
- `backpressureDroppedFrames`：latest-frame 缓冲覆盖的帧数
- `captureDroppedFrames`：`AVCaptureVideoDataOutput` drop callback 帧数
- `maximumQueueDepth`：逐帧实时队列的最大深度
- `processingTailMs`：最后一个输入 PTS 到最后一次检测完成的拖尾
- `sourceStarvationCount`：读取端没有按 PTS 准备好下一帧的次数
- `effectiveProcessingFPS`：`processedFrameCount / activeWallTime`

`sourceFrameCount` 不等于录制设备真正曝光的唯一画面数；重复画面仍是独立的
视频帧。

### 4.2 ZXingCpp

- `readLatencyMs`：平均值、P50、P95、P99、最大值
- `qrHitFrames`：至少识别出一个 QR 的处理帧数
- `qrHitRate`：`qrHitFrames / processedFrameCount`
- `decodedPayloadCount`：识别出的 QR payload 总数
- `uniquePayloadCount`：去重后交给 Rust 的 payload 数
- `duplicatePayloadCount`
- `readerErrorCount`

延迟百分位必须来自单帧样本，不能由分桶平均值反推。

### 4.3 Rust 与端到端恢复

- `consumeLatencyMs`：平均值、P50、P95、P99、最大值
- `acceptedPayloadCount`
- `rejectedPayloadCount`
- `rustDuplicateCount`
- `timeToFirstQRMs`
- `timeToFirstAcceptedSymbolMs`
- `timeToCompleteMs`
- `recoveredSymbolsPerSecond`
- `numRecovered`、`symbolCount`、`progress`
- `resultSize`
- `resultSHA256`
- `expectedSHA256`
- `outputMatchesExpected`

如果录制内容不足以完成恢复，测试仍输出性能指标，但正确性状态应明确为
`incomplete`，不能把它记成 SHA 失败。

### 4.4 运行环境

- App Git commit、构建配置
- 设备型号、OS 版本
- 视频文件 SHA-256、分辨率、FPS、时长、编码格式
- 回放模式和速率
- `ProcessInfo.processInfo.thermalState`
- 内存峰值（能够稳定采集时）

性能报告不得包含 QR payload 文本、源文件内容或恢复文件内容。

## 5. Fixture 结构与生命周期

录制前生成的 source kit：

```text
qrstream-ios-replay-recording-kit-YYYYMMDD/
├── README-recording.md
├── manifest.json
├── sources/
│   ├── ios-replay-baseline-v20-15fps.bin
│   └── ...
└── videos/
    ├── ios-replay-baseline-v20-15fps.mp4
    └── ...
```

录制回来供本地分析的 fixture：

```text
inputs/ios-replay-YYYYMMDD/
├── README.md
├── manifest.json
├── sources/                 # 可提交小载荷，或只保留 size + SHA-256
└── recordings/
    ├── <case>__<device>__<camera-mode>__take01.mov
    └── ...
```

`manifest.json` 是事实来源，至少记录：

- case ID 和用途
- 完整 encode 参数
- 源载荷大小及 SHA-256
- 播放源视频大小、SHA-256 和 ffprobe 元数据
- 录制设备、OS、相机模式、take 编号
- 录制文件 SHA-256
- 离线解码结果与输出 SHA-256

`inputs/` 已由仓库忽略，原始手机录制文件不加入 Git。若未来为了 CI 体积选取
最小帧 fixture 或转码，需要单独评审数据体积和隐私，并验证处理后的 fixture
仍能完成恢复。不能只因为文件能识别出部分 QR 就认为 fixture 有效。

## 6. Source 视频矩阵

第一版固定生成以下四个 RaptorQ V4、base45 alphanumeric、H.264 用例。
载荷由固定算法生成且不压缩，便于跨机器复现：

| Case | 载荷 | QR version | FPS | Overhead | 目的 |
|---|---:|---:|---:|---:|---|
| `baseline-v20-15fps` | 128 KiB | 20 | 15 | 1.30 | 低压力基线和基本正确性 |
| `balanced-v30-25fps` | 512 KiB | 30 | 25 | 1.35 | 接近日常高吞吐参数 |
| `dense-v40-30fps` | 1 MiB | 40 | 30 | 1.50 | 高密度识别延迟和成功率 |
| `throughput-v40-45fps` | 1 MiB | 40 | 45 | 2.00 | 主动制造背压并观察晚帧丢弃 |

所有视频包含 2 秒白色 lead-in 和 10% quiet-zone border。45 fps 用例的高
冗余用于容忍发送显示、录像和识别三处的帧损失；它的目标不是要求接收端
识别每一帧。

生成命令：

```bash
UV_CACHE_DIR=/tmp/qrstream-uv-cache \
  uv run python scripts/make_ios_replay_recording_kit.py \
  --output-dir /tmp/qrstream-ios-replay-recording-kit-YYYYMMDD \
  --verify
```

脚本拒绝覆盖已有目录。生成结果复制到桌面前，应确认四个 source 视频都能
由当前 Python 离线解码器恢复，且输出 SHA-256 与源载荷一致。

## 7. 录制规范

每个 case 单独录制：

1. 播放设备关闭自动锁屏和通知，亮度固定，视频全屏显示。
2. 录像设备优先使用 4K60，并在 manifest 中记录实际分辨率和 FPS。
3. QR 图案完整入镜，quiet zone 不被画面裁切；QR 内容约占录像短边的
   60%～80%。
4. 在播放前至少提前 2 秒开始录像，播放结束后至少延后 2 秒停止。
5. 不暂停、不拖动进度条、不在一次录像中拼接多个 case。
6. 首批性能 fixture 使用固定支架或稳定摆放，减少新的随机信道变量。
7. 如需手持鲁棒性集，另录 `handheld` take，不覆盖固定机位基准。
8. 保留相机原始文件，不经聊天软件或相册导出选项二次压缩。

建议命名：

```text
<case>__<device>__4k60__fixed__take01.mov
<case>__<device>__4k60__handheld__take01.mov
```

## 8. 基准执行规则

- 每个视频、每种模式至少运行 3 次，报告中位数及单轮数据。
- 在计时轮之前运行一次不计入结果的 warm-up。
- 记录每轮开始和结束 thermal state。
- 设备进入 `serious` 或 `critical` 温控状态时停止性能轮，保留该轮但标记
  `thermallyInvalid = true`。
- 同一提交内比较使用相同设备、相同 OS、相同 fixture 和相同构建配置。
- 正确性失败立即作为 gate；性能回归使用多轮中位数，避免单轮噪声。

第一版正确性与性能 gate：

- 所有原本可完整恢复的 fixture 仍须 `outputMatchesExpected = true`
- 不允许崩溃、死锁或 reader error 持续增长
- `realtimeAllFrames` 不允许主动抽帧或 capture drop
- 完成恢复前 `scheduledFrameCount == processedFrameCount`
- 60 fps 档持续处理 FPS 不低于输入 FPS，P95 ZXingCpp 延迟低于 16.67 ms
- 档位顺序固定为 4K60、1080p60、1080p30，不选择 4K30
- 10 分钟循环回放不得因崩溃或温控降频失去当前档位
- `unthrottledAllFrames` 的有效处理 FPS 回归超过 10% 时告警
- P95 工程目标为 10 ms；回归阈值在积累至少 5 次同设备基线后再收紧

## 9. 实现与验收顺序

1. 抽取共享 `QRFrameProcessor`，摄像头和回放共用 NV12 检测路径。
2. 实现 4K60、1080p60、1080p30 的设备能力选择和运行时降级。
3. 添加可注入时钟的逐帧 FIFO 调度器，并用单元测试验证无丢帧及过载语义。
4. 保留 latest-frame 调度器，只用于对照旧实现。
5. 添加 `AVAssetReader` 回放源和 Debug-only 文件选择入口。
6. 添加版本化 JSON 指标导出。
7. 用本方案的四个录制 fixture 在真实设备上建立首版基线。
8. 再增加 Mac 批处理入口，用于快速提交前回归；Mac 绝对性能不与 iPhone
   混合比较。

验收时需要同时证明：

- 摄像头和视频回放调用同一个 ZXingCpp/Rust 帧处理器
- 实时模式的生产节奏不被识别耗时串行阻塞，detector 队列不持续增长
- 完成恢复前每个调度帧都执行一次检测
- 重复运行不会复用上一轮 dedupe 或 Rust session 状态
- 报告中没有 payload/文件内容
- 完整恢复 case 的结果大小和 SHA-256 正确
