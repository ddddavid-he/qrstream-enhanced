# QRStream

[English](https://github.com/ddddavid-he/qrstream-enhanced/blob/main/README.md)

通过 QR 码视频流传输任意文件。基于 **LT 喷泉码（Luby Transform Fountain Codes）** 实现可靠的无反馈信道数据传输——即使丢失部分帧也能完整恢复原始文件。

## 原理概览

```
编码端                                    解码端
┌──────────┐   LT 喷泉码    ┌──────────┐   录屏/拍摄   ┌──────────┐   QR 识别    ┌──────────┐
│ 原始文件  │ ──────────── → │ QR 码视频 │ ──────────→ │ 视频文件  │ ──────────→ │ 还原文件  │
└──────────┘   zlib + COBS  └──────────┘              └──────────┘   LT 解码    └──────────┘
```

1. **编码**：将文件（可选 zlib 压缩）分块，通过 LT 喷泉码生成冗余编码块，每块序列化为 V3 协议帧，经 COBS 编码后嵌入 QR 码，最终输出 MP4 视频。
2. **解码**：使用 WeChatQRCode 从视频中高鲁棒性地提取 QR 码，COBS 解码后 CRC32 校验去除损坏帧，喂入 LT 解码器进行信念传播（peeling），恢复所有源块后重建原始文件。解码端会自动兼容 V2/V3 协议。

**核心优势**：
- **LT 喷泉码**：无码率纠删码，天然容忍帧丢失、模糊、遮挡
- **COBS 编码**：仅 0.4% overhead，比 base64 节省 33% 容量
- **WeChatQRCode 检测器**：对手机拍摄场景（透视、摩尔纹、光照）鲁棒性远超标准 QR 检测器
- **自适应采样率**：根据检测率和帧重复数自动选择最优采样策略
- **定向恢复**：首轮扫描后针对缺失块的时间位置精准补扫

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

- Python >= 3.10
- 依赖：`opencv-contrib-python`, `numpy`, `tqdm`, `qrcode[pil]`

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
| `-o, --output` | `<filename>.mp4` | 输出视频路径 |
| `--overhead` | `2.0` | 编码冗余倍率（源块数的倍数） |
| `--fps` | `10` | 输出视频帧率 |
| `--ec-level` | `1` | QR 纠错等级：0=L(7%), 1=M(15%), 2=Q(25%), 3=H(30%) |
| `--qr-version` | `20` | QR 码版本 1-40（越大密度越高） |
| `--border` | 标准 4 模块静区 | 静区宽度，按 QR 内容宽度百分比计算（`--border 10` = 10%，`--border 0` 可关闭） |
| `--no-compress` | - | 禁用 zlib 压缩 |
| `--force-compress` | - | 对大文件的 V3 编码强制整体压缩（会占用更多内存） |
| `--base64-qr` | - | 使用 base64 编码代替 COBS（兼容性更好但容量少 33%） |
| `--legacy-qr` | - | 使用 qrcode 库生成 QR（更慢但参数控制更精细） |
| `--codec` | `mp4v` | 视频编码器：`mp4v` 或 `mjpeg`（更快但文件更大） |
| `--protocol` | `v3` | 编码协议版本：`v3`（默认）或 `v2` |
| `-w, --workers` | CPU 核心数 | 并行 QR 生成的工作进程数 |
| `-v, --verbose` | - | 输出额外详细信息（进度条始终显示） |

### 解码（QR 码视频 → 文件）

```bash
qrstream decode <video> -o output_file [options]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<video>` | - | 输入视频路径（MP4, MOV 等） |
| `-o, --output` | `decoded_output` | 输出文件路径 |
| `-s, --sample-rate` | `0`（自动） | 每 N 帧采样一次（0=自适应探测） |
| `-w, --workers` | 全部 CPU 核心 | 并行 QR 识别的工作进程数 |
| `-v, --verbose` | - | 输出详细进度信息；大任务会显示 probe、扫描、LT 解码和写文件进度 |

### 示例

```bash
# 编码 PDF 文件（默认 COBS 二进制模式，2 倍冗余）
qrstream encode report.pdf -o report.mp4 --overhead 2.0 -v

# 解码视频（自适应采样率 + 定向恢复）
qrstream decode report.mp4 -o report_recovered.pdf -v

# 编码时使用高纠错等级（适合手机拍屏场景）
qrstream encode data.bin -o data.mp4 --ec-level 3 --qr-version 15
```

### 编程接口

```python
from qrstream.encoder import encode_to_video
from qrstream.decoder import extract_qr_from_video, decode_blocks, decode_blocks_to_file

# 编码（默认使用 COBS 二进制模式）
encode_to_video("input.bin", "output.mp4", overhead=2.0, verbose=True)

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
│   ├── protocol.py            # V2/V3 协议序列化 + COBS 编解码
│   └── qr_utils.py            # QR 生成（OpenCV）+ 检测（WeChatQRCode）
├── tests/
│   ├── test_lt_codec.py       # LT 编解码器单元测试
│   ├── test_protocol.py       # V2/V3 协议 + COBS 测试
│   ├── test_roundtrip.py      # 端到端回环测试
│   └── test_optimizations.py  # 性能优化 + WeChatQR + COBS 测试
└── benchmarks/
    └── benchmark.py           # 性能基准测试
```

## 技术细节

### V3 协议格式（24 字节头部 + 4 字节尾部 CRC）

```
Offset  Size  Field
  0      1    version      0x03
  1      1    flags        bit0=zlib 压缩, bit1=COBS 二进制模式
  2      8    filesize     uint64 BE（编码载荷大小；压缩时为压缩后大小）
 10      2    blocksize    uint16 BE
 12      4    block_count  uint32 BE  K = ceil(filesize / blocksize)
 16      4    seed         uint32 BE  PRNG 种子
 20      2    block_seq    uint16 BE  单调递增序号
 22      2    reserved     预留（当前为 0）
 24      ...  data         blocksize 字节的编码数据
 ...     4    crc32        CRC32（header[0:24] + data）
```

- 默认编码使用 **V3**。
- 解码器会自动兼容 **V2** 和 **V3**。
- V3 将 `filesize` 扩展为 `uint64`，`block_count` 扩展为 `uint32`，适合更大的文件和块数。

### 编码模式

| 模式 | QR 内容 | 容量开销 | 默认 |
|------|---------|----------|------|
| COBS 二进制 | raw bytes → COBS → latin-1 string | ~0.4% | 是 |
| Base64 | raw bytes → base64 string | ~33% | 否（`--base64-qr`） |

COBS（Consistent Overhead Byte Stuffing）消除所有 `\x00` 字节，使数据可安全通过 QR 字符串接口传递。

### 大文件与低内存路径

- 对于较大的 **V3** 输入文件，编码端会优先使用 `mmap` 做随机访问，避免把原文件整体复制进内存。
- 当输入足够大时，V3 编码默认会关闭整体 `zlib` 压缩，以保留低内存路径；如需强制压缩可使用 `--force-compress`。
- 解码端在恢复完成后支持直接写文件，并在压缩模式下使用增量解压，降低额外内存占用。
- 大文件解码会额外显示 **LT block 解码进度** 和 **输出写入进度**，避免在提取 QR 完成后长时间无输出。

### 解码管线

1. **探测阶段**：在视频中段的 3 个分散窗口中采样（默认每窗 120 帧），分别测量检测率和重复度，并取最保守的 `sample_rate`
2. **主扫描**：按自适应采样率并行检测 QR 码，实时喂入 LT 解码器
3. **定向恢复**：若首轮未恢复完整，定位缺失 seed 对应的视频时间段精准补扫
4. **LT 解码**：信念传播（peeling）算法恢复所有源块，并对大任务显示 block 解码进度
5. **输出写回**：按序写回恢复块；压缩模式下使用增量解压，并显示写文件进度

### LT 喷泉码参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 度分布 | Robust Soliton Distribution | c=0.1, delta=0.5 |
| PRNG | LCG (a=16807, m=2^31-1) | 5 轮预热消除序列种子偏差 |
| XOR | numpy 向量化 + 原地操作 | 比纯 Python 快 10-50x |
| 解码 | Belief Propagation (Peeling) | 基于二部图的迭代消元 |

## 测试

```bash
uv run pytest tests/ -v
```

## 许可证

MIT
