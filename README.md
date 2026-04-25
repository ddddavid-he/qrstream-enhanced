# QRStream

[中文文档](https://github.com/ddddavid-he/qrstream-enhanced/blob/main/README-zh.md)

Transfer arbitrary files through QR code video streams. Built on **LT Fountain Codes (Luby Transform)** for reliable, feedback-free data transmission — the original file can be fully recovered even if some frames are lost.

## How It Works

```
Encoder                                     Decoder
┌──────────┐   LT Fountain    ┌──────────┐   Screen cap   ┌──────────┐   QR detect    ┌──────────┐
│   File    │ ────────────── → │ QR Video │ ──────────── → │  Video   │ ────────────→ │ Recovered│
└──────────┘   zlib + base45  └──────────┘                └──────────┘   LT decode    │   File   │
                                                                                       └──────────┘
```

1. **Encode**: Split the file (optionally zlib-compressed) into blocks, generate redundant coded blocks via LT fountain codes, serialize each into a V3 protocol frame, base45-encode into QR alphanumeric-mode symbols, and output an MP4 video.
2. **Decode**: Extract QR codes from video using WeChatQRCode (highly robust), base45-decode, CRC32-validate to discard corrupted frames, feed into the LT decoder for belief propagation (peeling), and reconstruct the original file. Legacy base64 and COBS videos (pre-v0.6) continue to decode via a fallback path.

**Key Features**:
- **LT Fountain Codes**: Rateless erasure codes — naturally tolerant of frame loss, blur, and occlusion
- **Base45 + QR Alphanumeric Mode**: RFC 9285 base45 packs data into QR alphanumeric mode (5.5 bits/char vs 8 for byte mode); smaller and faster than base64 at the same QR version
- **WeChatQRCode Detector**: Far more robust than standard QR detectors for phone-captured screens (perspective, moire, lighting)
- **Adaptive Sample Rate**: Automatically selects optimal sampling strategy based on detection rate and frame repetition
- **Targeted Recovery**: After initial scan, precisely re-scans video segments where missing blocks are expected
- **Low-Memory Paths**: mmap-backed encoding and streaming decode-to-file for large inputs

## Installation

### From PyPI with pip

```bash
pip install qrstream
```

Use either command after installation:

```bash
qrstream <command> [options]
# or
qrs <command> [options]
```

You can also run it as a module:

```bash
python -m qrstream <command> [options]
```

### From PyPI with uv

```bash
uv tool install qrstream
```

Then run:

```bash
qrstream <command> [options]
```

For one-off execution without a persistent install:

```bash
uvx qrstream <command> [options]
```

### Development Install

```bash
git clone https://github.com/ddddavid-he/qrstream-enhanced.git && cd qrstream-enhanced
uv sync --dev
```

### Requirements

- Python >= 3.10 (3.10 – 3.14 tested)
- Dependencies: `opencv-contrib-python`, `numpy`, `tqdm`, `segno`

## Usage

```bash
qrstream <command> [options]
qrstream -V
qrstream --version
```

`qrs` is kept as a short alias, and `python -m qrstream` works as well.

### Encode (File → QR Video)

```bash
qrstream encode <file> -o output.mp4 [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `<file>` | - | Input file path |
| `-o, --output` | **required** | Output video path |
| `--overhead` | `2.0` | Encoding redundancy ratio (multiple of source block count) |
| `--fps` | `10` | Output video frame rate |
| `--ec-level` | `1` | QR error correction: 0=L(7%), 1=M(15%), 2=Q(25%), 3=H(30%) |
| `--qr-version` | `25` | QR code version 1-40 (higher = denser) |
| `--border` | standard 4-module quiet zone | Quiet-zone width as a percentage of QR content width (`--border 10` = 10%, `--border 0` disables it) |
| `--lead-in-seconds` | `0.0` | Insert white lead-in frames before the first QR frame |
| `--no-compress` | - | Disable zlib compression |
| `--force-compress` | - | Force compression for large V3 inputs (higher memory usage) |
| `--qr-mode` | `alphanumeric` | QR payload encoding: `alphanumeric` (base45, default, denser) or `base64` (byte mode, fallback) |
| `--legacy-qr` | - | Accepted but ignored (kept for CLI backward compatibility) |
| `--codec` | `mp4v` | Video codec: `mp4v` or `mjpeg` (faster but larger files) |
| `-w, --workers` | `min(CPU count, 4)` | Parallel worker threads for QR generation. The auto-picked default is capped at 4 because QR matrix generation (`segno.make()`) is pure-Python (GIL-bound), so more than ~4 worker threads mostly contend on the GIL without adding real parallelism. Pass a larger value explicitly to override the cap on CPU-rich machines if profiling shows benefit. |
| `-v, --verbose` | - | Print extra detail (progress bars always shown) |

### Decode (QR Video → File)

```bash
qrstream decode <video> -o output_file [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `<video>` | - | Input video path (MP4, MOV, etc.) |
| `-o, --output` | **required** | Output file path |
| `-s, --sample-rate` | `0` (auto) | Sample every Nth frame (0 = adaptive probing) |
| `-w, --workers` | All CPU cores | Parallel worker threads for QR detection. `WeChatQRCode` is implemented in C++ and releases the GIL during detection, so more threads scale close to linearly on multi-core machines. |
| `-v, --verbose` | - | Print detailed progress |

### Examples

```bash
# Encode a PDF (default: base45 alphanumeric mode, 2x redundancy)
qrstream encode report.pdf -o report.mp4 --overhead 2.0 -v

# Decode video (adaptive sample rate + targeted recovery)
qrstream decode report.mp4 -o report_recovered.pdf -v

# Encode with high error correction (for phone screen capture)
qrstream encode data.bin -o data.mp4 --ec-level 3 --qr-version 15

# Add a larger quiet zone and white lead-in before recording
qrstream encode slides.zip -o slides.mp4 --border 10 --lead-in-seconds 1.5
```

### Python API

```python
from qrstream.encoder import encode_to_video
from qrstream.decoder import extract_qr_from_video, decode_blocks, decode_blocks_to_file

# Encode (default: base45 alphanumeric mode)
encode_to_video("input.bin", "output.mp4", overhead=2.0, verbose=True)

# Add recording-friendly quiet zone and white lead-in
encode_to_video("input.bin", "output.mp4", border=10.0, lead_in_seconds=1.5)

# Decode to memory
blocks = extract_qr_from_video("output.mp4", verbose=True)
result = decode_blocks(blocks, verbose=True)

# Better for large files: stream directly to file with incremental decompression
written = decode_blocks_to_file(blocks, "recovered.bin", verbose=True)
print(f"wrote {written} bytes")
```

## Project Structure

```
project-root/
├── pyproject.toml             # Project config & dependencies
├── src/qrstream/
│   ├── cli.py                 # CLI entry (encode/decode subcommands)
│   ├── encoder.py             # LT encode → QR frame generation → MP4 video
│   ├── decoder.py             # Video frame extraction → QR detect → LT decode → file rebuild
│   ├── lt_codec.py            # LT fountain code primitives (PRNG, RSD, BlockGraph)
│   ├── protocol.py            # V3 protocol serialization + base45 codec (legacy base64/COBS decode supported)
│   └── qr_utils.py            # QR generation (segno) + detection (WeChatQRCode)
├── tests/
│   ├── test_lt_codec.py       # LT codec unit tests
│   ├── test_protocol.py       # V3 protocol + base45 tests
│   ├── test_decoder.py        # Decoder validation + probe strategy tests
│   ├── test_roundtrip.py      # Pure LT codec roundtrip tests (no video I/O)
│   ├── test_qr_generate.py    # QR generation correctness + glog(0) regression
│   ├── test_e2e_encode_decode.py  # End-to-end encode→video→decode SHA256 tests
│   └── test_optimizations.py  # Perf optimizations + WeChatQR + legacy-fallback tests
└── dev/
    ├── benchmark.py           # Performance benchmarks
    ├── perf-profile/          # cProfile hotspot analysis scripts
    ├── test-container/        # Podman test container
    └── wechatqrcode-mnn-poc/  # WeChatQRCode MNN acceleration POC
```

## Technical Details

### V3 Protocol Format (24-byte header + 4-byte trailing CRC)

```
Offset  Size  Field
  0      1    version      0x03
  1      1    flags        bit0=zlib compressed, bit1=high-density mode (base45 alphanumeric)
  2      8    filesize     uint64 BE (encoded payload size; compressed size when zlib is on)
 10      2    blocksize    uint16 BE
 12      4    block_count  uint32 BE  K = ceil(filesize / blocksize)
 16      4    seed         uint32 BE  PRNG seed
 20      2    block_seq    uint16 BE  monotonically increasing sequence number
 22      2    reserved     reserved (currently 0)
 24      ...  data         blocksize bytes of encoded data
 ...     4    crc32        CRC32(header[0:24] + data)
```

- Default encoding uses **V3 + base45 alphanumeric QR**.
- The decoder tries base45 → base64 → COBS in order, preserving compatibility with pre-v0.6 videos.
- V3 extends `filesize` to `uint64` and `block_count` to `uint32`, supporting larger files and block counts.

### Encoding Modes

| Mode | QR Content | QR Mode | Overhead | Default |
|------|-----------|---------|----------|---------|
| Base45 alphanumeric | raw bytes → base45 → `0-9A-Z $%*+-./:` | Alphanumeric (5.5 bits/char) | ~67% (but uses denser QR mode → **net denser** than byte mode) | Yes |
| Base64 | raw bytes → base64 string | Byte (8 bits/char) | ~33% | No (`--qr-mode base64`) |
| COBS (legacy) | raw bytes → COBS → latin-1 string | Byte | ~0.4% | **Removed** in v0.6; decode-only fallback for old videos |

Base45 (RFC 9285) is the default because QR's alphanumeric mode is denser per character than byte mode — at V25/M the base45 payload per frame is ~30% larger than base64, and in practice produces 20–25% smaller videos and 10–20% faster encode/decode.

### Large Files & Low-Memory Paths

- For large **V3** inputs, the encoder uses `mmap` for random access, avoiding loading the entire file into memory.
- When the input is large enough, V3 encoding automatically disables `zlib` compression to preserve the low-memory path; use `--force-compress` to override.
- The decoder supports streaming writes with incremental decompression, reducing memory overhead.
- Large file decoding shows **LT block decoding progress** and **output write progress** bars.

### Decoding Pipeline

1. **Probe phase**: Sample 3 spread-out windows in the video (120 frames each by default), measure detection rate and repetition per window, pick the most conservative `sample_rate`
2. **Main scan**: Detect QR codes in parallel at the adaptive sample rate, feeding into the LT decoder in real time
3. **Targeted recovery**: If the first pass didn't recover all blocks, use linear regression on observed (seed, frame) pairs to locate missing seeds and re-scan those segments precisely
4. **LT decode**: Belief propagation (peeling) to recover all source blocks
5. **Output writeback**: Write recovered blocks sequentially; incremental decompression in compressed mode

### LT Fountain Code Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Degree distribution | Robust Soliton Distribution | c=0.1, delta=0.5 |
| PRNG | SplitMix64 mixer + LCG (a=16807, m=2^31-1) | Non-linear seed mixing eliminates sequential-seed correlation |
| XOR | numpy vectorized + in-place | 10-50x faster than pure Python |
| Decoding | Belief Propagation (Peeling) | Iterative elimination on bipartite graph |

## Testing

```bash
# Unit tests (default — fast, no video I/O)
uv run pytest tests/ -v

# End-to-end encode→video→decode tests (10 KB, 100 KB, 500 KB + glog regression)
uv run pytest -m e2e -v

# Real-world phone-recording tests (requires fixture videos)
uv run pytest -m slow -v
```

## License

MIT
