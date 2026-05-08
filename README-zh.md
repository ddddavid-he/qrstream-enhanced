# QRStream

[English](https://github.com/ddddavid-he/qrstream-enhanced/blob/main/README.md)

通过 QR 码视频流传输任意文件。基于 **LT 喷泉码（Luby Transform Fountain Codes）** 实现可靠的无反馈信道数据传输——即使丢失部分帧也能完整恢复原始文件。

## 原理概览

```
编码端                                    解码端
┌──────────┐   LT 喷泉码    ┌──────────┐   录屏/拍摄   ┌──────────┐   QR 识别    ┌──────────┐
│ 原始文件  │ ──────────── → │ QR 码视频 │ ──────────→ │ 视频文件  │ ──────────→ │ 还原文件  │
└──────────┘   zlib + base45 └──────────┘              └──────────┘   LT 解码   └──────────┘
```

1. **编码**：将文件（可选 zlib 压缩）分块，通过 LT 喷泉码生成冗余编码块，每块序列化为 V3 协议帧，经 base45 编码后嵌入 QR 码的 alphanumeric 模式，最终输出 MP4 视频。
2. **解码**：使用 zxing-cpp 从视频中提取 QR 码（原生 C++，快速、鲁棒、无崩溃风险），base45 解码后 CRC32 校验去除损坏帧，喂入 LT 解码器进行信念传播（peeling），恢复所有源块后重建原始文件。旧版 base64/COBS 视频（v0.6 之前）会走 fallback 路径继续兼容。

**核心优势**：
- **LT 喷泉码**：无码率纠删码，天然容忍帧丢失、模糊、遮挡
- **Base45 + QR Alphanumeric 模式**：RFC 9285 base45 让数据落到 QR 的 alphanumeric 模式（每字符 5.5 bit，byte 模式 8 bit），在同一 QR version 下比 base64 更密、视频更小、编解码更快
- **zxing-cpp 检测器**：原生 C++ QR 检测器（v0.9 起取代 WeChatQRCode）——释放 GIL 支持真正并行检测，对噪声帧可重入且无崩溃，速度提升 4–10×，检测率持平
- **自适应采样率**：根据检测率和帧重复数自动选择最优采样策略
- **定向恢复**：首轮扫描后针对缺失块的时间位置精准补扫
- **低内存路径**：mmap 编码 + 流式写文件解码，支持大文件场景

## 安装

### 通过 pip 从 PyPI 安装

```bash
pip install qrstream
```

安装后可直接使用以下任一命令：

```bash
qrstream <command> [options]
# 或
qrs <command> [options]
```

也可以通过模块方式运行：

```bash
python -m qrstream <command> [options]
```

### 通过 uv 从 PyPI 安装

```bash
uv tool install qrstream
```

安装后运行：

```bash
qrstream <command> [options]
```

如果只想临时执行而不常驻安装：

```bash
uvx qrstream <command> [options]
```

### 开发环境安装

```bash
git clone https://github.com/ddddavid-he/qrstream-enhanced.git && cd qrstream-enhanced
uv sync --dev
```

### 系统要求

- Python >= 3.10（已测试 3.10 – 3.14）
- 依赖：`opencv-contrib-python`, `numpy`, `rich`, `zxing-cpp`

## 使用方式

```bash
qrstream <command> [options]
qrstream -V
qrstream --version
```

同时保留 `qrs` 这个短命令别名，也支持 `python -m qrstream`。

### 编码（文件 → QR 码视频）

```bash
qrstream encode <file> -o output.mp4 [options]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<file>` | - | 输入文件路径 |
| `-o, --output` | **必填** | 输出视频路径 |
| `--overhead` | `2.0` | 编码冗余倍率（源块数的倍数） |
| `--fps` | `10` | 输出视频帧率 |
| `--ec-level` | `1` | **已废弃**（v0.9 起隐藏，v0.10.0 将移除）：QR 纠错等级。在 qrstream 管线中实际多余——帧丢失已由 LT `--overhead` 处理。旧脚本可继续使用，但建议停止设置此参数。 |
| `--qr-version` | `25` | QR 码版本 1-40（越大密度越高） |
| `--border` | 标准 4 模块静区 | 静区宽度，按 QR 内容宽度百分比计算（`--border 10` = 10%，`--border 0` 可关闭） |
| `--lead-in-seconds` | `0.0` | 在首个 QR 帧前插入白色引导帧，便于开始录屏 |
| `--no-compress` | - | 禁用 zlib 压缩 |
| `--force-compress` | - | 对大文件的 V3 编码强制整体压缩（会占用更多内存） |
| `--qr-mode` | `alphanumeric` | QR 载荷编码：`alphanumeric`（base45，默认，更密）或 `base64`（byte 模式，fallback） |
| `--legacy-qr` | - | 仅作 CLI 向后兼容保留，不再影响行为 |
| `--codec` | `h264` | 视频编码器：`h264`（默认，压缩率好）、`mp4v` 或 `mjpeg`（编码更快，文件更大）。qrstream 会显式写入匹配的容器格式，并保留你提供的文件后缀；若后缀看起来不匹配，会给出 warning。 |
| `-w, --workers` | `1` | QR 生成的并行工作线程数。默认保持为 1，因为完整编码管线通常瓶颈在视频写出阶段，虽然 QR 矩阵生成（`zxingcpp.create_barcode()`）本身是原生 C++、不持 GIL。只有在你的机器上实测确认收益时，才建议手动调大。 |
| `--output-mode` | `auto` | 进度/状态渲染方式：`auto`（TTY 时 Rich 交互，否则 `log`）、`log`（CI 友好的 `key=value` 追加行）、`quiet`（仅输出错误和最终路径）、`verbose`（完整诊断输出） |
| `-v, --verbose` | - | `--output-mode verbose` 的别名（向后兼容保留） |

### 解码（QR 码视频 → 文件）

```bash
qrstream decode <video> -o output_file [options]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<video>` | - | 输入视频路径（MP4, MOV 等） |
| `-o, --output` | **必填** | 输出文件路径 |
| `-s, --sample-rate` | `0`（自动） | 每 N 帧采样一次（0=自适应探测） |
| `-w, --workers` | 全部 CPU 核心 | QR 识别的并行工作线程数。zxing-cpp 是原生 C++ 实现，执行期间释放 GIL，多线程能真正并行。 |
| `--output-mode` | `auto` | 进度/状态渲染方式：`auto`、`log`、`quiet`、`verbose`（与编码端同） |
| `-v, --verbose` | - | `--output-mode verbose` 的别名（向后兼容保留） |

### 示例

```bash
# 编码 PDF 文件（默认 base45 alphanumeric 模式，2 倍冗余，h264）
qrstream encode report.pdf -o report.mp4 --overhead 2.0 --output-mode verbose

# 解码视频（自适应采样率 + 定向恢复）
qrstream decode report.mp4 -o report_recovered.pdf --output-mode verbose

# 编码时使用更高 QR 版本（适合手机拍屏场景）
qrstream encode data.bin -o data.mp4 --qr-version 20

# 录屏场景：加大静区 + 预留白屏起录
qrstream encode slides.zip -o slides.mp4 --border 10 --lead-in-seconds 1.5

# CI 场景：log 模式解码
qrstream decode recording.mov -o out.bin --output-mode log
```

### 编程接口

```python
from qrstream.encoder import encode_to_video
from qrstream.decoder import extract_qr_from_video, decode_blocks, decode_blocks_to_file

# 编码（默认使用 base45 alphanumeric 模式）
encode_to_video("input.bin", "output.mp4", overhead=2.0, verbose=True)

# 录屏场景：加大静区 + 白屏引导
encode_to_video("input.bin", "output.mp4", border=10.0, lead_in_seconds=1.5)

# 解码到内存
blocks = extract_qr_from_video("output.mp4", verbose=True)
result = decode_blocks(blocks, verbose=True)

# 更适合大文件：直接写文件，降低额外内存占用
written = decode_blocks_to_file(blocks, "recovered.bin", verbose=True)
print(f"wrote {written} bytes")
```

## 项目结构

```
project-root/
├── pyproject.toml             # 项目配置与依赖
├── src/qrstream/
│   ├── cli.py                 # CLI 入口（encode/decode 子命令）
│   ├── encoder.py             # LT 编码 → QR 帧生成 → MP4 视频写入
│   ├── decoder.py             # 视频帧提取 → QR 检测 → LT 解码 → 文件重建
│   ├── lt_codec.py            # LT 喷泉码原语（PRNG、RSD、BlockGraph）
│   ├── protocol.py            # V3 协议序列化 + base45 编解码（解码端兼容旧版 base64/COBS）
│   └── qr_utils.py            # QR 生成 + 检测（zxing-cpp）
├── tests/
│   ├── test_lt_codec.py       # LT 编解码器单元测试
│   ├── test_protocol.py       # V3 协议 + base45 测试
│   ├── test_roundtrip.py      # 端到端回环测试
│   ├── test_qr_generate.py    # QR 生成正确性 + glog(0) 回归测试
│   ├── test_e2e_encode_decode.py  # 完整编码→视频→解码 SHA256 验证
│   └── test_optimizations.py  # 性能优化 + zxing-cpp + legacy fallback 测试
└── dev/
    ├── benchmark.py           # 性能基准测试
    ├── perf-profile/          # cProfile 热点分析脚本
    ├── test-container/        # Podman 测试容器
    └── wechatqrcode-mnn-poc/  # 历史 WeChatQRCode MNN 加速 POC（已归档）
```

## 技术细节

### V3 协议格式（24 字节头部 + 4 字节尾部 CRC）

```
Offset  Size  Field
  0      1    version      0x03
  1      1    flags        bit0=zlib 压缩, bit1=高密度模式（base45 alphanumeric）
  2      8    filesize     uint64 BE（编码载荷大小；压缩时为压缩后大小）
 10      2    blocksize    uint16 BE
 12      4    block_count  uint32 BE  K = ceil(filesize / blocksize)
 16      4    seed         uint32 BE  PRNG 种子
 20      2    block_seq    uint16 BE  单调递增序号
 22      2    reserved     预留（当前为 0）
 24      ...  data         blocksize 字节的编码数据
 ...     4    crc32        CRC32（header[0:24] + data）
```

- 默认编码使用 **V3 + base45 alphanumeric QR**。
- 解码端按 base45 → base64 → COBS 顺序尝试，保留对 v0.6 之前老视频的兼容。
- V3 将 `filesize` 扩展为 `uint64`，`block_count` 扩展为 `uint32`，适合更大的文件和块数。

### 编码模式

| 模式 | QR 内容 | QR 模式 | 字符开销 | 默认 |
|------|---------|---------|----------|------|
| Base45 alphanumeric | raw bytes → base45 → `0-9A-Z $%*+-./:` | Alphanumeric（5.5 bit/字符） | ~67%（但落在更密的 QR 模式 → **净密度更高**） | 是 |
| Base64 | raw bytes → base64 string | Byte（8 bit/字符） | ~33% | 否（`--qr-mode base64`） |
| COBS（legacy） | raw bytes → COBS → latin-1 string | Byte | ~0.4% | **v0.6 起移除编码路径**，仅解码端保留以兼容旧视频 |

Base45（RFC 9285）成为默认是因为 QR 的 alphanumeric 模式每字符承载的 bit 比 byte 模式更多——V25/M 下 base45 的单帧载荷比 base64 大约 30%，实测视频小 20~25%、编解码快 10~20%。

### 大文件与低内存路径

- 对于较大的 **V3** 输入文件，编码端会优先使用 `mmap` 做随机访问，避免把原文件整体复制进内存。
- 当输入足够大时，V3 编码默认会关闭整体 `zlib` 压缩，以保留低内存路径；如需强制压缩可使用 `--force-compress`。
- 解码端在恢复完成后支持直接写文件，并在压缩模式下使用增量解压，降低额外内存占用。
- 解码交互界面会实时刷新两行联动的进度：视频扫描进度条（百分比、ETA、滑动窗口检测率）与 qBittorrent 风格的文件块地图（按桶着色的密度图 + `N/K blocks` 计数）。非交互场景可使用 `--output-mode log` 输出 `key=value` 行，或 `--output-mode quiet` 用于脚本化调用。

### 解码管线

1. **探测阶段**：在视频中段的 3 个分散窗口中采样（默认每窗 120 帧），分别测量检测率和重复度，并取最保守的 `sample_rate`；完成时以 `Probe` + `Plan` 两行分别展示观测指标与决策参数
2. **主扫描**：按自适应采样率并行检测 QR 码，实时喂入 LT 解码器，`Scan` 行展示视频进度 / ETA / 检测率，`File` 行实时刷新块地图与 `N/K blocks` 计数
3. **定向恢复**：若首轮未恢复完整，定位缺失 seed 对应的视频时间段精准补扫
4. **LT 解码**：信念传播（peeling）算法恢复所有源块
5. **输出写回**：按序写回恢复块；压缩模式下使用增量解压

### LT 喷泉码参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 度分布 | Robust Soliton Distribution | c=0.1, delta=0.5 |
| PRNG | SplitMix64 混淆 + LCG (a=16807, m=2^31-1) | 非线性种子混淆消除序列种子相关性 |
| XOR | numpy 向量化 + 原地操作 | 比纯 Python 快 10-50x |
| 解码 | Belief Propagation (Peeling) | 基于二部图的迭代消元 |

## 测试

```bash
# 单元测试（默认，不含视频 I/O，速度快）
uv run pytest tests/ -v

# 端到端编码→视频→解码测试（10 KB、100 KB、500 KB + glog 回归）
uv run pytest -m e2e -v

# 真实手机录像测试（需要 fixture 视频文件）
uv run pytest -m slow -v
```

### 工具命令

```bash
# 显示交互界面使用的色彩调色板
qrstream colors
```

## 许可证

MIT
