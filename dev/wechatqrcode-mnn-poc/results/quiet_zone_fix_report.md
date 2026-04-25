# MNN tight-crop → quiet-zone padding 修复报告

## 背景

M1 接入后端到端 benchmark 显示当前分支 `--mnn` 比 OpenCV 更慢：
`IMG_9425.MOV`（1016 帧、M4 Pro）上，MNN CPU 端到端 `30.05s`、MNN
Metal `48.31s`，OpenCV 只要 `23.41s`。`DetectorRouter` 统计进一步
表明 MNN CPU **0/779 命中**、Metal `103/780` 命中，其余全部回退。

## 剖析结论

抽样 `200` 帧分 `6` 条独立路径测命中率：

| 路径 | 给出 bbox | 解码命中 |
|---|---:|---:|
| OpenCV 全帧（参考上限） | 188/200 | **188/200** |
| MNN CPU detect-only | **200/200** | — |
| MNN CPU detect + cpu_decode（tight crop） | 200/200 | **0/200** |
| MNN Metal detect-only | **200/200** | — |
| MNN Metal detect + cpu_decode（tight crop） | 200/200 | 10/200 |

detector 本身毫无问题（比 OpenCV 全帧还多给了 12 帧）。瓶颈在
`_run_detector → _clamp_bbox → frame[y0:y1,x0:x1] → _cpu_decode`
这段：**MNN SSD 的 bbox 紧贴 QR 外部 finder，把 ISO/IEC 18004
要求的 4 模块 quiet zone 切掉了，ZXing 在裁剪内找不到 Finder Pattern**。

### Padding 对照实验（100 帧）

| 裁剪策略 | 命中 |
|---|---:|
| `pad=0%`（原生产路径等价） | **0 / 100** |
| `pad=5%` | **93 / 100** |
| `pad=15%` | **95 / 100** |
| `pad=30%` | 95 / 100 |
| `pad=50%` | 95 / 100 |
| `pad=0%` + `resize 2×`（模拟 SR 清晰化） | 0 / 100 |
| OpenCV 全帧（上限） | 95 / 100 |

跳变锐利、与分辨率无关，原因明确落在 quiet zone。

## 修复

`src/qrstream/detector/mnn_detector.py`：

1. 新增 `_QUIET_ZONE_PAD_RATIO = 0.15`。
2. 新增 helper `_pad_bbox(x0,y0,x1,y1,img_w,img_h,ratio)`：按 bbox
   短边扩展，再 clamp 到图像边界，最小 1 像素。
3. `MNNQrDetector.detect` 在 `_clamp_bbox` 后调 `_pad_bbox` 再裁剪；
   SR 阈值改用 padded 尺寸；若 padded crop 解码失败，**再退回 tight
   crop 跑一次** 作为兜底（数据表明几乎永不触发，成本受限）。

`tests/test_detector.py` 加 `TestPadBboxForQuietZone` 共 7 条单元
回归，锁定：

- `_QUIET_ZONE_PAD_RATIO > 0` 且 `≥ 0.10`（4 模块静区的底线）
- 正常情况下四边等量外扩
- clamp 不逃出 `[0, img_w/h]`
- 短边驱动，wide-narrow bbox 也拿到对称 margin
- `ratio=0` 时为恒等（反向兼容）
- tiny bbox（4×4）最小 1 px 外扩
- 退化输入保持退化，交给调用方跳过

## 验证

### 命中率（200 帧抽样）

| 路径 | 修复前 | 修复后 |
|---|---:|---:|
| MNN CPU 完整链路 | **0 / 200** | **189 / 200** |
| MNN Metal 完整链路 | **10 / 200** | **189 / 200** |
| OpenCV 全帧（参考） | 188 / 200 | 188 / 200 |

MNN 两条路径的完整链路命中率均达到 `~94.5%`，与 OpenCV 上限持平
并略高（detector 本身多找到的帧数转化成了端到端命中）。`avg_ms`
从 ~300 ms（每次 MNN miss 再走 OpenCV 回退）降到 ~73 ms。

### 端到端（IMG_9425.MOV，所有输出 SHA256 匹配源文件）

| 路径 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| OpenCV 全帧 | 23.41 s | 25.56 s | 波动 |
| MNN CPU | 30.05 s | **24.12 s** | **–19.7%** |
| MNN Metal | 48.31 s | **27.78 s** | **–42.5%** |

MNN CPU 端到端已与 OpenCV 持平；MNN Metal 因本视频量级下每条线都要
吃一次 MNN 初始化 + DetectionOutput layer fallback 固定开销，追平
OpenCV 还需配合后续 (a) `DetectorRouter.opencv_fallback` 动态关闭、
(b) M5 batch pipeline。关键是**根因已完全解决**，此后所有 detector
/ GPU 优化才具备被放大的前提。

## 后续建议

1. 修完本项后，M1 的 `fallback=779` 断崖式下降，可在 `DetectorRouter`
   里引入“前 N 帧命中率统计 → 动态关闭 opencv_fallback”，消除剩下
   的额外 OpenCV 调用成本。
2. `MNNQrDetector._cpu_decode` 当前仍调用 `cv2.wechat_qrcode_WeChatQRCode`，
   这条路径仍承担 ZXing native crash 风险；后续可考虑用更轻的 QR
   解码器（如 pyzbar / zxing-cpp python binding）替换。
3. 本轮仅在 `IMG_9425.MOV` 上验证；接下来用 `tests/fixtures/` 里
   `real-phone-v4/*.mp4` 复跑一次 detect_breakdown 看收益是否跨样本
   一致。

## 产物

- 代码：`src/qrstream/detector/mnn_detector.py`
- 测试：`tests/test_detector.py::TestPadBboxForQuietZone`
- 脚本：`.bench/detect_breakdown.py`、`.bench/probe_crop_padding.py`、
  `.bench/bench_current_decode.py`
- 数据：`.bench/results-host/detect_breakdown.json`（修复前）、
  `.bench/results-host/detect_breakdown_after_padding.json`（修复后）、
  `.bench/results-host/crop_padding.json`（padding 对照）、
  `.bench/results-host/after_padding/*.bin`（端到端解码产物，
  `sha256 = 3a3a30b00b3cbaa9ad88ef54f881c1c310a58f40212d6c29c1c5947b7fca7968`）
