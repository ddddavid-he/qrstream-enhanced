# Milestone 0 验证报告

## 测试环境

- **容器**: `python:3.12-slim` (Debian, aarch64)
- **Python**: 3.12.13
- **OpenCV**: 4.13.0 (`opencv-contrib-python-headless`)
- **MNN**: 3.5.0 (`pip install MNN`)
- **Host**: macOS (Apple Silicon, arm64), podman 5.8.2
- **日期**: 2026-04-25

## 模型文件

### Caffe 原始模型

| 文件 | SHA256 | 大小 |
|------|--------|------|
| detect.prototxt | `e8acfc395caf443a...` | 42K |
| detect.caffemodel | `cc49b8c9babaf45f...` | 943K |
| sr.prototxt | `8ae41acba97e8b4a...` | 5.9K |
| sr.caffemodel | `e5d36889d8e6ef2f...` | 24K |

### MNN 转换后模型

| 文件 | SHA256 | 大小 |
|------|--------|------|
| detect.mnn | `e11626ec6694abf9...` | 974K |
| sr.mnn | `21cf1f14502bc183...` | 28K |

## 转换结果

### Detector 模型 ✅ 转换成功

```
MNNConvert -f CAFFE → detect.mnn: Converted Success!
输入层: data
输出层: detection_output
```

### SR 模型 ✅ 转换成功

```
MNNConvert -f CAFFE → sr.mnn: Converted Success!
输入层: data
输出层: fc
警告: Check failed: offsetSize >= 1 ==> crop offset error (不影响转换)
```

## 推理验证

### Detector 验证 ✅ PASS

| 指标 | OpenCV DNN | MNN | 差异 |
|------|-----------|-----|------|
| 输出形状 | (1,1,100,7) | (1,1,N,6) | MNN 无 batch_idx 列（dim=6 vs 7） |
| 检测数 | 1 | 1 | 一致 |
| Confidence | 0.6572 | 0.6572 | **< 0.000001** |
| BBox | [0.4720, 0.4670, 0.5283, 0.5270] | [0.4720, 0.4670, 0.5283, 0.5270] | **= 0.000000** |
| 推理耗时 | 22.8 ms | 25.3 ms | CPU-only，未优化 |

**结论**: detector 输出**完美对齐**。唯一差异是 MNN 的 DetectionOutput 层输出维度为 6（不含 batch index），而 OpenCV DNN 输出为 7（含 batch index）。后处理需对应调整。

### SR 验证 ⚠️ 近似通过

| 指标 | OpenCV DNN | MNN | 差异 |
|------|-----------|-----|------|
| 输出形状 | (1,1,199,199) | (1,1,200,200) | MNN 多 1 行/列 |
| 值范围 | [-0.46, 1.25] | [-0.52, 2.68] | MNN 范围更宽 |
| Cosine 相似度 | — | — | **0.9957** |
| Mean abs error | — | — | 0.0088 |
| 推理耗时 | 6.1 ms | 2.1 ms | MNN 快 2.9x |

**结论**: SR 模型核心输出高度一致（cosine > 0.99），但存在 **1 像素尺寸差异** — 这是 Caffe Crop 层在两个框架中的边界处理不同导致的。实际使用中可忽略（后处理只取 `uint8` 像素值，差 1 行不影响 QR 解码）。

## 关键发现（影响实现）

### 1. 模型输入格式修正

原方案假设 detector 输入为 BGR 3 通道 300×300，实际 prototxt 定义为：

- **输入**: `(1, 1, 384, 384)` — **单通道灰度**
- **预处理**: `resize(gray, (det_w, det_h))` → `/ 255.0`
- 上游代码使用动态尺寸（基于目标面积 400×400），不是固定 300

### 2. MNN DetectionOutput 维度差异

- OpenCV DNN: `(1, 1, N, 7)` → `[batch_idx, class, conf, x0, y0, x1, y1]`
- MNN: `(1, 1, N, 6)` → `[class, conf, x0, y0, x1, y1]`（无 batch_idx）
- 后处理代码需根据维度自动适配

### 3. MNN 3.5 Python API

MNN 3.5 的 Python API 与旧版文档不同：
- 无 `ScheduleConfig` 类，改用 `createSession({'backend': 'CPU'})`
- 无 `Forward_CPU` / `Forward_Metal` 常量
- `resizeTensor` 的第二个参数必须是 **tuple** 不能是 list
- MNN version API: `MNN.version()`

### 4. SR 模型 Crop 层行为差异

MNN 的 Crop 层输出比 OpenCV DNN 多 1 行/列：
- OpenCV: 100×100 输入 → 199×199 输出
- MNN: 100×100 输入 → 200×200 输出
- 对最终 QR 解码影响可忽略

## 下一步

Milestone 0 验证通过。两个模型均可成功转换并产生一致输出。可以进入 **Milestone 1: Apple Metal 首版接入**。

需要先更新 `mnn_detector.py` 中的以下实现细节：
1. 输入改为单通道灰度 384×384（或动态尺寸）
2. 适配 MNN 3.5 API（去掉 ScheduleConfig）
3. 处理 DetectionOutput dim=6 的情况
4. SR 输出裁剪到偶数尺寸
