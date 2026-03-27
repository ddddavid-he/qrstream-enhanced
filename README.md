# QRStream

通过 QR 码视频流传输任意文件。基于 **LT 喷泉码（Luby Transform Fountain Codes）** 实现可靠的无反馈信道数据传输——即使丢失部分帧也能完整恢复原始文件。

## 原理概览

```
编码端                                    解码端
┌──────────┐   LT 喷泉码    ┌──────────┐   录屏/拍摄   ┌──────────┐   QR 识别    ┌──────────┐
│ 原始文件  │ ──────────── → │ QR 码视频 │ ──────────→ │ 视频文件  │ ──────────→ │ 还原文件  │
└──────────┘   zlib + V2    └──────────┘              └──────────┘   LT 解码    └──────────┘
```

1. **编码**：将文件（可选 zlib 压缩）分块，通过 LT 喷泉码生成冗余编码块，每块序列化为 V2 协议帧并嵌入 QR 码，最终输出 MP4 视频。
2. **解码**：从视频中逐帧提取 QR 码，CRC32 校验去除损坏帧，喂入 LT 解码器进行信念传播（peeling），恢复所有源块后重建原始文件。

**核心优势**：LT 喷泉码是一种无码率（rateless）纠删码，理论上只需接收约 K×1.05 个编码块即可恢复 K 个源块，天然适应 QR 码视频传输中的帧丢失、模糊、遮挡等问题。

## 安装

```bash
# 克隆项目
git clone <repo-url> && cd qrstream-enhanced

# 安装依赖（推荐使用 uv）
uv sync

# 安装开发依赖（含 pytest）
uv sync --extra dev
```

### 系统要求

- Python >= 3.10
- 运行时依赖：`opencv-python`, `numpy`, `tqdm`, `qrcode[pil]`

## 使用方式

QRStream 支持两种等效的调用方式：

```bash
# 方式一：脚本调用
uv run main.py <command> [options]

# 方式二：命令调用（通过 console_scripts 入口点）
uv run qrstream <command> [options]
```

### 编码（文件 → QR 码视频）

```bash
uv run qrstream encode <file> -o output.mp4 [options]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<file>` | - | 输入文件路径 |
| `-o, --output` | `<filename>.mp4` | 输出视频路径 |
| `--overhead` | `2.0` | 编码冗余倍率（源块数的倍数） |
| `--fps` | `10` | 输出视频帧率 |
| `--ec-level` | `1` | QR 纠错等级：0=L(7%), 1=M(15%), 2=Q(25%), 3=H(30%) |
| `--qr-version` | `20` | QR 码版本 1-40（越大密度越高） |
| `--no-compress` | - | 禁用 zlib 压缩 |
| `-w, --workers` | CPU 核心数 | 并行 QR 生成的工作进程数（上限 8） |
| `-v, --verbose` | - | 输出详细进度信息 |

### 解码（QR 码视频 → 文件）

```bash
uv run qrstream decode <video> -o output_file [options]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<video>` | - | 输入视频路径（MP4, MOV 等） |
| `-o, --output` | `decoded_output` | 输出文件路径 |
| `-s, --sample-rate` | `0`（自动） | 每 N 帧采样一次（0=自动探测） |
| `-w, --workers` | CPU 核心数 | 并行 QR 识别的工作进程数（上限 8） |
| `-v, --verbose` | - | 输出详细进度信息 |

### 示例

```bash
# 编码一个 PDF 文件，2.5 倍冗余，15fps
uv run qrstream encode report.pdf -o report.mp4 --overhead 2.5 --fps 15 -v

# 解码视频，自动采样率
uv run qrstream decode report.mp4 -o report_recovered.pdf -v

# 编码时使用高纠错等级（适合屏幕拍摄场景）
uv run qrstream encode data.bin -o data.mp4 --ec-level 3 --qr-version 15
```

### 编程接口

```python
from qrstream.encoder import encode_to_video
from qrstream.decoder import extract_qr_from_video, decode_blocks

# 编码
encode_to_video("input.bin", "output.mp4", overhead=2.0, fps=10, verbose=True)

# 解码
blocks = extract_qr_from_video("output.mp4", sample_rate=0, verbose=True)
result = decode_blocks(blocks, verbose=True)
with open("recovered.bin", "wb") as f:
    f.write(result)
```

## 项目结构

```
qrstream-enhanced/
├── main.py                    # 脚本入口点 (uv run main.py)
├── pyproject.toml             # 项目配置与依赖
├── src/qrstream/
│   ├── __init__.py            # 包版本信息
│   ├── __main__.py            # python -m qrstream 支持
│   ├── cli.py                 # CLI 命令解析（encode/decode 子命令）
│   ├── encoder.py             # LT 编码器 → QR 帧 → MP4 视频
│   ├── decoder.py             # 视频 → QR 提取 → LT 解码 → 文件重建
│   ├── lt_codec.py            # LT 喷泉码原语（PRNG、RSD、BlockGraph）
│   ├── protocol.py            # V1/V2 协议帧序列化与反序列化
│   └── qr_utils.py            # QR 码生成与多策略检测
├── tests/
│   ├── test_lt_codec.py       # LT 编解码单元测试
│   ├── test_protocol.py       # 协议序列化测试
│   └── test_roundtrip.py      # 端到端编解码测试
├── inputs/                    # 输入测试数据（不纳入版本控制）
└── outputs/                   # 输出结果（不纳入版本控制）
```

## 技术细节

### V2 协议格式（20 字节头部）

```
Offset  Size  Field
  0      1    version      0x02
  1      1    flags        bit0=zlib 压缩
  2      4    filesize     uint32 BE（压缩后大小）
  6      2    blocksize    uint16 BE
  8      2    block_count  uint16 BE  K = ceil(filesize / blocksize)
 10      4    seed         uint32 BE  PRNG 种子
 14      2    block_seq    uint16 BE  单调递增序号
 16      4    crc32        CRC32（header[0:16] + data）
 20      ...  data         blocksize 字节的编码数据
```

### LT 喷泉码参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 度分布 | Robust Soliton Distribution (RSD) | c=0.1, delta=0.5 |
| PRNG | LCG (a=16807, m=2^31-1) | 5 轮预热消除序列种子偏差 |
| XOR | numpy 向量化 | 比纯 Python 快 10-50x |
| 解码算法 | Belief Propagation (Peeling) | 基于二部图的迭代消元 |

### QR 码容量参考

| QR 版本 | 纠错 L | 纠错 M | 纠错 Q | 纠错 H |
|---------|--------|--------|--------|--------|
| 10      | 271    | 213    | 151    | 119    |
| 20      | 858    | 666    | 482    | 382    |
| 30      | 1732   | 1370   | 982    | 742    |
| 40      | 2953   | 2331   | 1663   | 1273   |

单位：字节（byte mode）。实际可用数据量需扣除 V2 头部 20 字节和 base64 编码 4/3 膨胀。

## 测试

```bash
uv run pytest tests/ -v
```

## 许可证

MIT
