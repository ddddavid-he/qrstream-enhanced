# QRStream

[English](https://github.com/ddddavid-he/qrstream-enhanced/blob/main/README.md)

通过 QR 码视频流传输任意文件。基于 **RaptorQ（RFC 6330）** 和 **LT 喷泉码** 实现可靠的无反馈信道数据传输——即使丢失部分帧也能完整恢复原始文件。

## 原理概览

```
编码端                                    解码端
┌──────────┐ RaptorQ / LT   ┌──────────┐   录屏/拍摄   ┌──────────┐   QR 识别    ┌──────────┐
│ 原始文件  │ ──────────── → │ QR 码视频 │ ──────────→ │ 视频文件  │ ──────────→ │ 还原文件  │
└──────────┘   zlib + base45 └──────────┘              └──────────┘   RQ / LT   └──────────┘
```

1. **编码**：将文件（可选 zlib 压缩）分块，通过 RaptorQ（默认）或 LT 喷泉码生成冗余编码块，每块序列化为 V3/V4 协议帧，经 base45 编码后嵌入 QR 码的 alphanumeric 模式，最终输出 MP4 视频。
2. **解码**：使用 zxing-cpp 从视频中提取 QR 码（原生 C++，快速、鲁棒、无崩溃风险），base45 解码后 CRC32 校验去除损坏帧，喂入 RaptorQ 或 LT 解码器进行恢复，重建原始文件。旧版 base64/COBS 视频（v0.6 之前）会走 fallback 路径继续兼容。

**核心优势**：
- **RaptorQ（RFC 6330）**：默认喷泉码——系统性编码，仅需收到任意 K 个包即可高概率恢复；LT 喷泉码作为传统选项仍可使用
- **Base45 + QR Alphanumeric 模式**：RFC 9285 base45 让数据落到 QR 的 alphanumeric 模式（每字符 5.5 bit，byte 模式 8 bit），在同一 QR version 下比 base64 更密、视频更小、编解码更快
- **zxing-cpp 检测器**：原生 C++ QR 检测器，释放 GIL 支持真正并行检测；相比历史 OpenCV/WeChatQRCode 路径，对噪声帧可重入且无崩溃，速度更快且检测率持平
- **自适应采样率**：根据检测率和帧重复数自动选择最优采样策略
- **定向恢复 + GE 救援**：主扫描后 LT 解码器可先运行 GF(2) 高斯消元 checkpoint，提前完成已经满秩但 peeling 卡住的 LT 图；必要时再只补扫缺失 seed 所在的视频片段。RaptorQ 在解码器内部处理恢复，无需 GE。
- **低内存路径**：RaptorQ/LT 源符号 mmap 编码 + 流式写文件解码，支持大文件场景
- **显示模式**：`qrstream encode` 省略 `-o` 时会将生成的 QR 帧直接传送至内置 Qt 播放器；`--display -o` 会优先保障显示流畅度，同时确保最终生成完整视频文件

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

### 默认包含 GUI

`qrstream` 默认安装 Qt 显示播放器。省略 `-o` 的编码会直接打开播放器，
`--display -o` 则会同时提供实时显示和完整视频输出。旧的 `qrstream[gui]`
extra 仍会被接受，以兼容既有安装脚本，但已不再是必需项。

### 开发环境安装

```bash
git clone https://github.com/ddddavid-he/qrstream-enhanced.git && cd qrstream-enhanced
uv sync --dev
```

### 开发文档

- [CONTRIBUTING.md](docs/CONTRIBUTING.md)：分支策略、提交/PR 约定、CI 触发/skip 规则与发布流程
- [ARCH.md](docs/ARCH.md)：架构说明、核心模块索引、协议/UI/校准设计与测试入口

仓库根目录下旧的 `BRANCHING.md` 已下线，分支与开发流程规则统一维护在 `docs/CONTRIBUTING.md`。

### 系统要求

- Python >= 3.10（已测试 3.10 – 3.14）
- 依赖：`opencv-python-headless`, `numpy`, `rich`, `zxing-cpp`, `av`, `PySide6-Essentials`, `raptorq`

## 使用方式

```bash
qrstream <command> [options]
qrstream -V
qrstream --version
```

同时保留 `qrs` 这个短命令别名，也支持 `python -m qrstream`。

### 编码（文件 → QR 码视频）

```bash
qrstream encode <file> [-o output.mp4] [--display] [options]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<file>` | - | 输入文件路径 |
| `-o, --output` | 可选 | 输出视频路径；省略时编码默认进入屏幕显示模式。 |
| `--display` | - | 直接在内置 GUI 播放器中显示生成的 QR 帧。与 `-o/--output` 同时使用时，优先保障显示流畅度；如后台写入未完成，关闭显示窗口后会继续完成剩余视频输出。 |
| `--overhead` | `1.2`（RaptorQ）/ `2.0`（LT） | 编码冗余倍率（源块数的倍数）。默认值取决于 `--fountain-codec`。 |
| `--fps` | `10` | 输出视频帧率 |
| `--ec-level` | `1` | **已废弃并隐藏**：QR 纠错等级。在 qrstream 管线中实际多余——帧丢失已由 LT `--overhead` 处理。旧脚本在废弃窗口内仍可继续使用，但建议停止设置此参数。 |
| `--qr-version` | `25` | QR 码版本 1-40（越大密度越高） |
| `--border` | 标准 4 模块静区 | 静区宽度，按 QR 内容宽度百分比计算（`--border 10` = 10%，`--border 0` 可关闭） |
| `--lead-in-seconds` | `0.0` | 在首个 QR 帧前插入白色引导帧，便于开始录屏 |
| `--no-compress` | - | 禁用 zlib 压缩 |
| `--force-compress` | - | 对大文件的 V3 编码强制整体压缩（会占用更多内存） |
| `--qr-mode` | `alphanumeric` | QR 载荷编码：`alphanumeric`（base45，默认，更密）或 `base64`（byte 模式，fallback） |
| `--legacy-qr` | - | 仅作 CLI 向后兼容保留，不再影响行为 |
| `--auto-mask` | - | 仅作 CLI 向后兼容保留，不再影响行为（zxing-cpp 会在原生代码中自动评估全部 QR mask） |
| `--codec` | `h264` | 视频编码器：`h264`（默认，压缩率好）、`mp4v` 或 `mjpeg`（编码更快，文件更大）。qrstream 会显式写入匹配的容器格式，并保留你提供的文件后缀；若后缀看起来不匹配，会给出 warning。 |
| `--fountain-codec` | `raptorq` | 喷泉码：`raptorq`（默认，RFC 6330，近最优恢复）或 `lt`（传统 LT 喷泉码） |
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

### 校准（自动校准信道参数）

```bash
qrstream calibrate [--display | -o output.mp4 | -i video.mp4] [options]
```

不带模式参数时，`calibrate` 默认为 `--display`（在屏幕上播放校准序列）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--display` | 默认 | 通过内置 Qt 播放器播放校准序列（编码端） |
| `-o, --output` | - | 将校准视频写入文件（编码端） |
| `-i, --input` | - | 分析已录制的校准视频（解码端） |
| `--precision` | `standard` | 校准预设：`low`（弱信道）、`fast`（~15s）、`standard`（~30s）、`full`（~60s）、`high`（~60s，更高 FPS） |
| `--display-hz` | 自动 / 60 | 覆盖显示器刷新率（显示模式自动检测，视频模式默认 60） |
| `--codec` | `h264` | 校准输出的视频编码器 |
| `--target-size` | - | 用于文件特定冗余估算的目标载荷大小（分析模式；如 `100M`、`1.5GiB`） |
| `--target-file` | - | 用于文件特定冗余估算的目标载荷文件（分析模式） |
| `--fountain-codec` | `raptorq` | 冗余估算使用的喷泉码模型（分析模式） |
| `--confidence` | - | 覆盖所有档位的解码成功率目标（分析模式；如 `0.95`） |
| `-w, --workers` | 自动 | 分析的并行工作线程数 |
| `--output-mode` | `auto` | 进度/状态渲染方式（与编码/解码端同） |
| `-v, --verbose` | - | `--output-mode verbose` 的别名（向后兼容保留） |

### 示例

```bash
# 编码 PDF 文件（默认 base45 alphanumeric 模式，RaptorQ，1.2 倍冗余，h264）
qrstream encode report.pdf -o report.mp4 --output-mode verbose

# 使用 LT 喷泉码编码，更高冗余应对噪声信道
qrstream encode report.pdf -o report.mp4 --fountain-codec lt --overhead 2.0

# 解码视频（自适应采样率 + 定向恢复）
qrstream decode report.mp4 -o report_recovered.pdf --output-mode verbose

# 编码时使用更高 QR 版本（适合手机拍屏场景）
qrstream encode data.bin -o data.mp4 --qr-version 20

# 录屏场景：加大静区 + 预留白屏起录
qrstream encode slides.zip -o slides.mp4 --border 10 --lead-in-seconds 1.5

# 直接显示 QR 帧；省略 -o 时默认进入显示模式
qrstream encode data.zip

# 直接显示 QR 帧；如后台写入未完成，关闭窗口后继续完成视频输出
qrstream encode data.zip --display -o data.mp4

# CI 场景：log 模式解码
qrstream decode recording.mov -o out.bin --output-mode log

# 校准：在屏幕上播放校准序列（默认）
qrstream calibrate

# 校准：为特定显示器生成校准视频
qrstream calibrate -o calib.mp4 --display-hz 120

# 校准：分析已录制的校准视频
qrstream calibrate -i recording.mov --target-size 100M
```

### 最佳实践

```bash
# ── 小文件（< 10 MB）：直接编解码 ─────────────────────────────────
# 默认参数即可；RaptorQ 1.2x 冗余足够。
qrstream encode small_file.zip -o small_file.mp4
qrstream decode small_file.mp4 -o recovered.zip

# ── 大文件：先校准获取最佳参数，再编解码 ─────────────────────────
# 第 1 步 — 生成与你的显示设备匹配的校准视频
qrstream calibrate -o calib.mp4 --display-hz 60

# 第 2 步 — 用手机/相机录制校准视频

# 第 3 步 — 分析录制结果；记下推荐的参数设置
qrstream calibrate -i recording.mov --target-size 500M

# 第 4 步 — 使用推荐的 QR version、FPS 和冗余进行编码
#          （使用分析步骤输出的值）
qrstream encode large_file.bin -o large_file.mp4 \
    --qr-version 30 --fps 15 --overhead 1.3

# 第 5 步 — 解码
qrstream decode large_file.mp4 -o recovered.bin

# ── 噪声信道（光照不足、手持拍摄等）─────────────────────────────
# 使用 LT 喷泉码并加大冗余，恢复更稳健
qrstream encode important.pdf -o important.mp4 \
    --fountain-codec lt --overhead 2.5

# ── 方便录制的编码 ───────────────────────────────────────────────
# 加宽静区有助于模糊拍摄时的检测；
# 白屏引导帧为开始录屏留出准备时间
qrstream encode slides.zip -o slides.mp4 \
    --border 15 --lead-in-seconds 2

# ── 直接屏幕显示（无需生成视频文件）─────────────────────────────
# 在屏幕上播放 QR 帧，接收端直接拍屏即可。
# 省略 -o 为纯显示模式；加 -o 可同时生成视频文件。
qrstream encode data.zip
qrstream encode data.zip --display -o data.mp4
```

### 编程接口

```python
from qrstream.encoder import encode_to_video
from qrstream.decoder import extract_qr_from_video, decode_blocks, decode_blocks_to_file

# 编码（默认使用 base45 alphanumeric 模式，RaptorQ，1.2x 冗余）
encode_to_video("input.bin", "output.mp4", overhead=1.2, verbose=True)

# 录屏场景：加大静区 + 白屏引导
encode_to_video("input.bin", "output.mp4", border=10.0, lead_in_seconds=1.5)

# 解码到内存
blocks = extract_qr_from_video("output.mp4", verbose=True)
result = decode_blocks(blocks, verbose=True)

# 更适合大文件：直接写文件，降低额外内存占用
written = decode_blocks_to_file(blocks, "recovered.bin", verbose=True)
print(f"wrote {written} bytes")

# 进阶：复用提取阶段已经完成的 decoder（例如 scan 阶段 GE 已成功）
blocks, completed_decoder = extract_qr_from_video(
    "output.mp4", verbose=True, return_decoder=True)
written = decode_blocks_to_file(
    blocks, "recovered.bin", decoder=completed_decoder)
```

## 项目结构

```
project-root/
├── pyproject.toml             # 项目配置与依赖
├── src/qrstream/
│   ├── cli.py                 # CLI 入口（encode/decode/calibrate/colors 子命令）
│   ├── encoder.py             # RaptorQ/LT 编码 → QR 帧生成 → MP4 视频写入
│   ├── decoder.py             # 视频帧提取 → QR 检测 → RaptorQ/LT 解码 → 文件重建
│   ├── raptorq_codec.py       # RaptorQ（RFC 6330）编码/解码器
│   ├── lt_codec.py            # LT 喷泉码原语（PRNG、RSD、BlockGraph、GF(2) 救援）
│   ├── protocol.py            # V3/V4 协议序列化 + base45 编解码（解码端兼容旧版 base64/COBS）
│   ├── qr_utils.py            # QR 生成 + 检测（zxing-cpp）
│   ├── calibrate.py           # 校准视频生成与分析
│   ├── calibration_optimizer.py # 联合 QR version/FPS/冗余优化
│   ├── overhead_policy.py     # 共享喷泉码冗余策略常量
│   ├── ui.py                  # 统一进度/状态 UI 层
│   ├── display_cache.py       # 有界显示模式帧缓存
│   ├── display_player*.py     # Qt 显示模式播放器
│   └── _compat.py             # 平台兼容性辅助
├── tests/
│   ├── test_lt_codec.py       # LT 编解码器单元测试
│   ├── test_raptorq_codec.py  # RaptorQ 编解码器单元测试
│   ├── test_raptorq_protocol.py # V4 协议测试
│   ├── test_raptorq_roundtrip.py # RaptorQ 回环测试
│   ├── test_protocol.py       # V3 协议 + base45 测试
│   ├── test_gaussian_rescue.py # GE 救援 fallback 测试
│   ├── test_roundtrip.py      # 纯 LT codec 回环测试（不含视频 I/O）
│   ├── test_qr_generate*.py   # QR 生成正确性 + mask/glog 回归测试
│   ├── test_e2e_encode_decode.py  # 完整编码→视频→解码 SHA256 验证
│   ├── test_display_*.py      # 显示模式缓存/播放器测试
│   ├── test_optimizations.py  # 性能优化 + zxing-cpp + legacy fallback 测试
│   ├── test_calibrate.py      # 校准子命令测试
│   ├── test_calibration_optimizer.py # 校准优化器测试
│   ├── test_prng_v2.py        # PRNG 版本测试
│   ├── test_ppm_learning.py   # PPM 阈值学习测试
│   ├── test_probe_adaptation.py # 自适应探测测试
│   ├── test_cli_*.py          # CLI 验证测试
│   └── test_ui_reporter.py    # UI 报告器测试
└── docs/
    ├── CONTRIBUTING.md        # 开发流程、分支/提交/CI 规则
    ├── ARCH.md                # 架构参考文档
    ├── discovery/             # 调研记录与发现
    └── tooling/               # benchmark、profiling 与本地容器辅助工具
```

## 技术细节

### V3 协议格式（24 字节头部 + 4 字节尾部 CRC）

```
Offset  Size  Field
  0      1    version      0x03
  1      1    flags        bit0=zlib 压缩, bit1=高密度模式（base45 alphanumeric）,
                           bit2=prng_version（1=SplitMix64, 0=传统 LCG）
  2      8    filesize     uint64 BE（编码载荷大小；压缩时为压缩后大小）
 10      2    blocksize    uint16 BE
 12      4    block_count  uint32 BE  K = ceil(filesize / blocksize)
 16      4    seed         uint32 BE  PRNG 种子
 20      2    block_seq    uint16 BE  单调递增序号
 22      2    reserved     预留（当前为 0）
 24      ...  data         blocksize 字节的编码数据
 ...     4    crc32        CRC32（header[0:24] + data）
```

### V4 协议格式（RaptorQ，相同的 24 字节头部 + 4 字节尾部 CRC）

```
Offset  Size  Field
  0      1    version      0x04
  1      1    flags        bit0=zlib 压缩, bit1=高密度模式（base45 alphanumeric）
  2      8    filesize     uint64 BE
 10      2    symbol_size  uint16 BE（与 V3 blocksize 位置相同）
 12      4    symbol_count uint32 BE  K（与 V3 block_count 位置相同）
 16      4    esi          uint32 BE  RaptorQ PayloadId（SBN || local ESI）
 20      2    block_seq    uint16 BE
 22      2    source_blocks uint16 BE  Z；0 = 传统/单源块
 24      ...  data         symbol_size 字节的编码数据
 ...     4    crc32        CRC32（header[0:24] + data）
```

- 默认编码使用 **V4 + RaptorQ + base45 alphanumeric QR**。
- 解码端从版本字节自动检测 V4（RaptorQ）和 V3（LT），并按 base45 → base64 → COBS 顺序尝试，保留对 v0.6 之前老视频的兼容。
- V3/V4 将 `filesize` 扩展为 `uint64`，`block_count`/`symbol_count` 扩展为 `uint32`，适合更大的文件和块数。

### 编码模式

| 模式 | QR 内容 | QR 模式 | 字符开销 | 默认 |
|------|---------|---------|----------|------|
| Base45 alphanumeric | raw bytes → base45 → `0-9A-Z $%*+-./:` | Alphanumeric（5.5 bit/字符） | ~67%（但落在更密的 QR 模式 → **净密度更高**） | 是 |
| Base64 | raw bytes → base64 string | Byte（8 bit/字符） | ~33% | 否（`--qr-mode base64`） |
| COBS（legacy） | raw bytes → COBS → latin-1 string | Byte | ~0.4% | **v0.6 起移除编码路径**，仅解码端保留以兼容旧视频 |

Base45（RFC 9285）成为默认是因为 QR 的 alphanumeric 模式每字符承载的 bit 比 byte 模式更多——V25/M 下 base45 的单帧载荷比 base64 大约 30%，实测视频小 20~25%、编解码快 10~20%。

### 大文件与低内存路径

- 对于较大的 **V3/V4** 输入文件，共用加载器会优先使用 `mmap` 做随机访问。LT 会直接消费映射输入，RaptorQ 也会用它生成系统性源符号；仅在上游 `raptorq` API 需要生成修复符号时才物化连续缓冲区。
- 当输入足够大时，编码默认会关闭整体 `zlib` 压缩，以保留低内存路径；如需强制压缩可使用 `--force-compress`。
- 解码端在恢复完成后支持直接写文件，并在压缩模式下使用增量解压，降低额外内存占用。
- 解码交互界面会实时刷新两行联动的进度：视频扫描进度条（百分比、ETA、滑动窗口检测率）与 qBittorrent 风格的文件块地图（按桶着色的密度图 + `N/K blocks` 计数）。非交互场景可使用 `--output-mode log` 输出 `key=value` 行，或 `--output-mode quiet` 用于脚本化调用。

### 解码管线

1. **探测阶段**：三阶段管线——裁剪探索（全分辨率突发帧）、PPM 分辨率扫描（多分辨率检测以学习自适应缩放）与采样率估计（流水线式读取+检测分散窗口）。计算自适应 `sample_rate`、`max_dim` 和裁剪 ROI；完成时以 `Probe` + `Plan` 两行分别展示观测指标与决策参数
2. **主扫描**：按自适应采样率并行检测 QR 码，实时喂入喷泉解码器（自动检测 V4/RaptorQ 或 V3/LT），`Scan` 行展示视频进度 / ETA / 检测率，`File` 行实时刷新块地图与 `N/K blocks` 计数
3. **GE checkpoint**（仅 LT）：若主扫描后 peeling 卡住，先对已累积的 LT 方程运行 GF(2) Gauss-Jordan 救援；如果方程已经覆盖全部缺失源块，则直接完成解码，并复用该 decoder 写文件。RaptorQ 在内部处理恢复，跳过此步骤
4. **定向恢复**：若喷泉解码器尚未收敛，则基于已观测的（seed/ESI, frame）关系定位缺失符号对应的视频时间段精准补扫（含 CLAHE 对比度增强）；每个 recovery level 新增 unique block 后，升级到下一层前会再尝试一次 GE checkpoint（LT）
5. **LT 解码 fallback**：仅传入 raw blocks 的编程接口仍保留最终的 peeling + GE rescue 兜底
6. **输出写回**：按序写回恢复块；压缩模式下使用增量解压

### 喷泉码参数

#### RaptorQ（RFC 6330，默认）

| 参数 | 值 | 说明 |
|------|-----|------|
| 码类型 | 系统性 RaptorQ | 源符号直接传输；修复符号提供冗余 |
| 恢复能力 | 近最优 | 收到任意 K 个包即可高概率解码 |
| 载荷标识 | SBN \|\| local ESI（4 字节） | 源块编号 + 局部编码符号 ID |
| 每源块最大符号数 | 56,403 | 超过则拆分为多个源块 |
| 默认冗余 | 1.2x（最低 1.05x，推荐 ≥1.10x） | 近最优恢复所需冗余远低于 LT |
| 解码 | 内部 RaptorQ 解码器 | 无需 Gauss-Jordan 救援 |

#### LT 喷泉码（传统）

| 参数 | 值 | 说明 |
|------|-----|------|
| 度分布 | Robust Soliton Distribution | c=0.1, delta=0.5 |
| PRNG | SplitMix64 混淆 + LCG (a=16807, m=2^31-1) | 非线性种子混淆消除序列种子相关性 |
| XOR | numpy 向量化 + 原地操作 | 比纯 Python 快 10-50x |
| 解码 | Belief Propagation (Peeling) + GF(2) GE rescue | Peeling 是快速路径；Gauss-Jordan checkpoint 可恢复卡住但满秩的 LT 图 |
| 默认冗余 | 2.0x（最低 1.20x，推荐 ≥1.50x） | LT 收敛次优需要更高冗余 |

## 测试

```bash
# 默认测试集（速度快；pyproject.toml 默认排除 slow 和 e2e marker）
uv run pytest

# 端到端编码→视频→解码测试
uv run pytest -m e2e

# 真实手机录像测试（需要 fixture 视频文件）
uv run pytest -m slow
```

### 工具命令

```bash
# 显示交互界面使用的色彩调色板
qrstream colors
```

## 许可证

MIT
