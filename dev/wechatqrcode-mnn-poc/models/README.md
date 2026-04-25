# WeChatQRCode 模型文件

## 来源

模型文件来自 [WeChatCV/opencv_3rdparty](https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode) 仓库，与 `opencv-contrib-python` 内置的 `cv2.wechat_qrcode_WeChatQRCode()` 使用的是同一套。

## 许可证

模型文件遵循 [Apache License 2.0](https://github.com/WeChatCV/opencv_3rdparty/blob/wechat_qrcode/LICENSE)。

## 文件列表

### Caffe 原始模型（`caffe/`）

| 文件 | 说明 |
|------|------|
| `detect.prototxt` | SSD QR 码检测网络定义 |
| `detect.caffemodel` | SSD QR 码检测网络权重 |
| `sr.prototxt` | 超分辨率网络定义 |
| `sr.caffemodel` | 超分辨率网络权重 |

### MNN 转换后模型（`mnn/`）

| 文件 | 说明 |
|------|------|
| `detect.mnn` | SSD 检测网络（MNN 格式） |
| `sr.mnn` | 超分辨率网络（MNN 格式） |

## 获取方式

```bash
# 1. 下载 Caffe 模型
bash scripts/fetch_models.sh

# 2. 转换为 MNN 格式
bash scripts/convert_to_mnn.sh

# 3. 验证输出一致性
python scripts/verify_models.py --image test_image.png
```

## 模型推理细节

### SSD Detector

- **输入**: BGR uint8 图像
- **预处理**: `resize(300, 300, INTER_CUBIC)` → `/ 255.0` → NCHW blob
- **输出**: `(1, 1, N, 7)` tensor
  - `[batch, channel, detection_idx, (unused, class_id, confidence, x0, y0, x1, y1)]`
  - `class_id == 1` → QR 码
  - `confidence > 1e-5` → 有效检测
  - 坐标为归一化值 `[0, 1]`，需乘以原图尺寸

### Super-Resolution

- **输入**: 灰度图 `(H, W)` → `/ 255.0` → `(1, 1, H, W)` float32
- **输出**: `(1, 1, 2H, 2W)` float32
- **后处理**: `* 255.0` → `clip(0, 255)` → uint8
- **触发条件**: `sqrt(roi_area) < 160` 时才调用（小 ROI 需要增强）

## 注意事项

- 模型文件不纳入 Git 版本管理（见 `.gitignore`）
- 本地开发需手动运行 `fetch_models.sh` 获取
- CI/CD 中模型文件通过缓存或临时下载获取
