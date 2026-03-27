# QRStream 编解码使用手册

## 前置准备

```bash
# 安装依赖
uv sync
```

确保系统已安装 Python >= 3.10。

---

## 编码：将文件转为 QR 码视频

将任意文件编码为包含 QR 码序列的 MP4 视频。

### 基础用法

```bash
uv run qrstream encode <输入文件> -o <输出视频>
```

```bash
# 示例：编码一个 PDF
uv run qrstream encode report.pdf -o report.mp4
```

### 常用选项

```bash
# 提高冗余倍率（默认 2.0，拍摄场景建议 3.0+）
uv run qrstream encode file.pdf -o file.mp4 --overhead 3.0

# 降低 QR 密度 + 提高纠错（适合手机拍屏）
uv run qrstream encode file.pdf -o file.mp4 --qr-version 15 --ec-level 3

# 提高帧率以缩短视频时长
uv run qrstream encode file.pdf -o file.mp4 --fps 15

# 查看详细编码进度
uv run qrstream encode file.pdf -o file.mp4 -v
```

### 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--overhead` | `2.0` | 冗余倍率。越高容错越好，视频越长 |
| `--fps` | `10` | 视频帧率。越高视频越短，但对播放/录制要求更高 |
| `--ec-level` | `1` | QR 纠错：0=L(7%), 1=M(15%), 2=Q(25%), 3=H(30%) |
| `--qr-version` | `20` | QR 版本 1-40。越小 QR 码越简单但每帧数据量越少 |
| `--no-compress` | - | 禁用压缩（已压缩的文件如 .zip 可用此选项） |
| `-w` | 自动 | 并行工作进程数 |
| `-v` | - | 显示详细进度 |

### 场景推荐配置

**屏幕直传**（解码端直接录屏）：
```bash
uv run qrstream encode file.pdf -o file.mp4 --overhead 2.0 --fps 10
```

**手机拍屏**（相机对着屏幕拍）：
```bash
uv run qrstream encode file.pdf -o file.mp4 --overhead 3.0 --fps 5 --ec-level 3 --qr-version 15
```

**大文件快速传输**（高质量屏幕录制）：
```bash
uv run qrstream encode file.bin -o file.mp4 --overhead 2.0 --fps 15 --qr-version 30
```

---

## 解码：从 QR 码视频还原文件

从视频中识别 QR 码并重建原始文件。

### 基础用法

```bash
uv run qrstream decode <视频文件> -o <输出文件>
```

```bash
# 示例：解码视频还原 PDF
uv run qrstream decode report.mp4 -o report_recovered.pdf
```

### 常用选项

```bash
# 自动检测采样率（默认行为，推荐）
uv run qrstream decode video.mp4 -o output.pdf

# 手动指定采样率（每帧都扫描，最慢但最全）
uv run qrstream decode video.mp4 -o output.pdf -s 1

# 查看详细解码进度
uv run qrstream decode video.mp4 -o output.pdf -v
```

### 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output` | `decoded_output` | 输出文件路径 |
| `-s, --sample-rate` | `0`（自动） | 每 N 帧采样。0=自动探测，1=逐帧扫描 |
| `-w` | 自动 | 并行工作进程数 |
| `-v` | - | 显示详细进度 |

---

## 完整工作流示例

### 1. 编码端

```bash
# 将机密文件编码为 QR 视频
uv run qrstream encode secret.docx -o transfer.mp4 --overhead 2.5 -v
```

输出：
```
Input: secret.docx (45230 bytes)
Compressed: 45230 → 31024 bytes (68.6%)
Blocks: K=63, blocksize=493, total=126 (overhead=2.0x)
QR frame size: 870x870, video FPS: 10, workers: 8
Encoding frames: 100%|████████████████████| 126/126
Output: transfer.mp4 (1904093 bytes, 126 frames)
```

### 2. 传输

在编码端播放 `transfer.mp4`，解码端通过以下任一方式获取视频：
- **屏幕录制**：直接录屏
- **手机拍摄**：对着屏幕拍摄视频
- **文件传输**：直接拷贝 MP4 文件

### 3. 解码端

```bash
# 从视频还原文件
uv run qrstream decode transfer.mp4 -o secret_recovered.docx -v
```

输出：
```
Processing: transfer.mp4
Extracting QR codes...
Probe complete: 60 frames, avg repeat=1.0, auto sample_rate=1
Scanning frames: 100%|████████████████████| 126/126
Extraction done (early termination): 126 frames, 72 unique blocks
Decoded after 72/72 blocks (filesize=31024, K=63, compressed=True)

Success! Saved to: secret_recovered.docx (45230 bytes)
```

---

## 脚本调用方式

除了 `uv run qrstream` 命令，也可以通过脚本入口调用：

```bash
# 等效于 uv run qrstream encode ...
uv run main.py encode file.pdf -o file.mp4

# 等效于 uv run qrstream decode ...
uv run main.py decode video.mp4 -o output.pdf
```

---

## 常见问题

**Q: 解码失败，提示 "Decoding incomplete"**
- 增大编码时的 `--overhead`（如 3.0 或 4.0）
- 降低 `--fps` 让每帧显示更久
- 解码时用 `-s 1` 逐帧扫描
- 确保视频清晰，QR 码完整显示在画面中

**Q: 编码时应该选什么 QR 版本？**
- 屏幕录制：`--qr-version 20`（默认，平衡密度与可读性）
- 手机拍摄：`--qr-version 10~15`（更大模块，更易识别）
- 高质量传输：`--qr-version 30~40`（更多数据，视频更短）

**Q: 纠错等级怎么选？**
- 屏幕录制：`--ec-level 1`（M，默认，足够）
- 手机拍摄：`--ec-level 2` 或 `3`（Q/H，应对拍摄失真）
- 注意：更高纠错 = 更少有效数据 = 视频更长
