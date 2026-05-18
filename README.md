# QRStream

[中文文档](https://github.com/ddddavid-he/qrstream-enhanced/blob/main/README-zh.md)

Transfer arbitrary files through QR code video streams. Built on **RaptorQ (RFC 6330)** and **LT Fountain Codes** for reliable, feedback-free data transmission — the original file can be fully recovered even if some frames are lost.

## How It Works

```
Encoder                                     Decoder
┌──────────┐ RaptorQ / LT    ┌──────────┐   Screen cap   ┌──────────┐   QR detect    ┌──────────┐
│   File    │ ────────────── → │ QR Video │ ──────────── → │  Video   │ ────────────→ │ Recovered│
└──────────┘   zlib + base45  └──────────┘                └──────────┘   RQ / LT      │   File   │
                                                                                       └──────────┘
```

1. **Encode**: Split the file (optionally zlib-compressed) into blocks, generate redundant coded blocks via RaptorQ (default) or LT fountain codes, serialize each into a V3/V4 protocol frame, base45-encode into QR alphanumeric-mode symbols, and output an MP4 video.
2. **Decode**: Extract QR codes from video using zxing-cpp (fast, robust, crash-free), base45-decode, CRC32-validate to discard corrupted frames, feed into the RaptorQ or LT decoder for recovery, and reconstruct the original file. Legacy base64 videos continue to decode via a fallback path.

**Key Features**:
- **RaptorQ (RFC 6330)**: Default fountain codec — systematic, near-optimal recovery with any K packets; LT codes also available as a legacy option
- **Base45 + QR Alphanumeric Mode**: RFC 9285 base45 packs data into QR alphanumeric mode (5.5 bits/char vs 8 for byte mode); smaller and faster than base64 at the same QR version
- **zxing-cpp Detector**: Native C++ QR detector — releases the GIL for true parallel detection, reentrant and crash-free on noisy inputs, faster than the historical OpenCV/WeChatQRCode path with equivalent detection rate
- **Adaptive Sample Rate**: Automatically selects optimal sampling strategy based on detection rate and frame repetition
- **Targeted Recovery + GE Rescue**: After the main scan, the LT decoder can run a GF(2) Gaussian-elimination checkpoint to finish stalled LT graphs early; if needed, it re-scans only video segments where missing seeds are expected. RaptorQ handles recovery internally without GE.
- **Low-Memory Paths**: mmap-backed RaptorQ/LT source-symbol encoding and streaming decode-to-file for large inputs
- **Display Mode**: `qrstream encode` without `-o` streams generated QR frames directly to the built-in Qt player; `--display -o` prioritises smooth display while still completing the output video

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

### GUI included by default

`qrstream` installs the Qt display player by default.  Encoding without `-o`
opens the player directly, and `--display -o` combines live display with a
complete output video.  The legacy `qrstream[gui]` extra remains accepted for
older install scripts, but it is no longer required.

### Development Install

```bash
git clone https://github.com/ddddavid-he/qrstream-enhanced.git && cd qrstream-enhanced
uv sync --dev
```

### Developer Docs

- [Contributing Guide](docs/CONTRIBUTING.md): Branching strategy, commit/PR rules, CI trigger/skip conventions, and release flow
- [Architecture Reference](docs/ARCH.md): Module map, protocol layout, calibration/UI design, and testing/tooling reference

The old top-level `BRANCHING.md` has been retired; branch and workflow rules now live in `docs/CONTRIBUTING.md`.

### Requirements

- Python >= 3.10 (3.10 – 3.14 tested)
- Dependencies: `opencv-python-headless`, `numpy`, `rich`, `zxing-cpp`, `av`, `PySide6-Essentials`, `raptorq`

## Usage

```bash
qrstream <command> [options]
qrstream -V
qrstream --version
```

`qrs` is kept as a short alias, and `python -m qrstream` works as well.

### Encode (File → QR Video)

```bash
qrstream encode <file> [-o output.mp4] [--display] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `<file>` | - | Input file path |
| `-o, --output` | optional | Output video path. If omitted, encode defaults to on-screen display. |
| `--display` | - | Display generated QR frames in the built-in GUI player. When combined with `-o/--output`, display smoothness is prioritised and the output video is completed after the display window closes if needed. |
| `--overhead` | `1.2` (RaptorQ) / `2.0` (LT) | Encoding redundancy ratio (multiple of source block count). Default depends on `--fountain-codec`. |
| `--fps` | `10` | Output video frame rate |
| `--ec-level` | `1` | **Deprecated and hidden**: QR error correction level. Redundant — LT `--overhead` already handles frame loss. Existing scripts continue to work during the deprecation window but should stop using this option. |
| `--qr-version` | `25` | QR code version 1-40 (higher = denser) |
| `--border` | standard 4-module quiet zone | Quiet-zone width as a percentage of QR content width (`--border 10` = 10%, `--border 0` disables it) |
| `--lead-in-seconds` | `0.0` | Insert white lead-in frames before the first QR frame |
| `--no-compress` | - | Disable zlib compression |
| `--force-compress` | - | Force compression for large V3 inputs (higher memory usage) |
| `--qr-mode` | `alphanumeric` | QR payload encoding: `alphanumeric` (base45, default, denser) or `base64` (byte mode, fallback) |
| `--legacy-qr` | - | Accepted but ignored (kept for CLI backward compatibility) |
| `--auto-mask` | - | Accepted but ignored (zxing-cpp always evaluates all QR mask patterns in native code) |
| `--codec` | `h264` | Video codec: `h264` (default, good compression), `mp4v`, or `mjpeg` (faster encode, larger files). qrstream writes the matching container explicitly and keeps your filename suffix unchanged; if the suffix looks inconsistent, it emits a warning. |
| `--fountain-codec` | `raptorq` | Fountain code: `raptorq` (default, RFC 6330, near-optimal recovery) or `lt` (legacy LT codes) |
| `-w, --workers` | `1` | Parallel worker threads for QR generation. The default stays at 1 because the full encode pipeline is typically video-writer-bound even though QR matrix generation (`zxingcpp.create_barcode()`) is native C++ (GIL-free). Pass a larger value explicitly only if profiling on your machine shows a win. |
| `--output-mode` | `auto` | Progress/status rendering: `auto` (Rich interactive on TTY, `log` otherwise), `log` (append-only `key=value` lines for CI), `quiet` (errors and final path only), `verbose` (full diagnostic output) |
| `-v, --verbose` | - | Alias for `--output-mode verbose` (kept for backward compatibility) |

### Decode (QR Video → File)

```bash
qrstream decode <video> -o output_file [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `<video>` | - | Input video path (MP4, MOV, etc.) |
| `-o, --output` | **required** | Output file path |
| `-s, --sample-rate` | `0` (auto) | Sample every Nth frame (0 = adaptive probing) |
| `-w, --workers` | All CPU cores | Parallel worker threads for QR detection. zxing-cpp is native C++ and releases the GIL during detection, so more threads scale close to linearly on multi-core machines. |
| `--output-mode` | `auto` | Progress/status rendering: `auto`, `log`, `quiet`, or `verbose` (same as encode) |
| `-v, --verbose` | - | Alias for `--output-mode verbose` (kept for backward compatibility) |

### Calibrate (Auto-calibrate Channel Parameters) — *experimental*

```bash
qrstream calibrate [--display | -o output.mp4 | -i video.mp4] [options]
```

> **Note:** `calibrate` is experimental and may change in future releases.

Without a mode argument, `calibrate` defaults to `--display` (play a calibration sequence on screen).

| Option | Default | Description |
|--------|---------|-------------|
| `--display` | default | Play calibration sequence via the built-in Qt player (encoder side) |
| `-o, --output` | - | Write calibration video to file (encoder side) |
| `-i, --input` | - | Analyze a captured calibration video (decoder side) |
| `--precision` | `standard` | Calibration preset: `low` (weak channels), `fast` (~15s), `standard` (~30s), `full` (~60s), `high` (~60s, higher FPS) |
| `--display-hz` | auto / 60 | Override display refresh rate in Hz (auto in display mode, 60 in video mode) |
| `--codec` | `h264` | Video codec for calibration output |
| `--target-size` | - | Target payload size for file-specific overhead estimates (analysis mode; e.g. `100M`, `1.5GiB`) |
| `--target-file` | - | Target payload file for file-specific overhead estimates (analysis mode) |
| `--fountain-codec` | `raptorq` | Fountain code model for overhead estimates (analysis mode) |
| `--confidence` | - | Override decode-success target across all tiers (analysis mode; e.g. `0.95`) |
| `-w, --workers` | auto | Parallel workers for analysis |
| `--output-mode` | `auto` | Progress/status rendering (same as encode/decode) |
| `-v, --verbose` | - | Alias for `--output-mode verbose` |

### Examples

```bash
# Encode a PDF (default: base45 alphanumeric mode, RaptorQ, 1.2x redundancy, h264)
qrstream encode report.pdf -o report.mp4 --output-mode verbose

# Encode with LT codes and higher overhead for noisy channels
qrstream encode report.pdf -o report.mp4 --fountain-codec lt --overhead 2.0

# Decode video (adaptive sample rate + targeted recovery)
qrstream decode report.mp4 -o report_recovered.pdf --output-mode verbose

# Encode with high QR version for phone screen capture
qrstream encode data.bin -o data.mp4 --qr-version 20

# Add a larger quiet zone and white lead-in before recording
qrstream encode slides.zip -o slides.mp4 --border 10 --lead-in-seconds 1.5

# Display QR frames directly; omitting -o defaults to display mode
qrstream encode data.zip

# Display QR frames and still save a complete video after display closes if needed
qrstream encode data.zip --display -o data.mp4

# CI-friendly decode with log output
qrstream decode recording.mov -o out.bin --output-mode log

# Calibrate: display calibration sequence on screen (default)
qrstream calibrate

# Calibrate: generate calibration video for a specific display
qrstream calibrate -o calib.mp4 --display-hz 120

# Calibrate: analyze a captured calibration video
qrstream calibrate -i recording.mov --target-size 100M
```

### Best Practices

```bash
# ── Small files (< 10 MB): encode and decode directly ──────────────
# Default parameters work well; RaptorQ 1.2x overhead is sufficient.
qrstream encode small_file.zip -o small_file.mp4
qrstream decode small_file.mp4 -o recovered.zip

# ── Large files: calibrate first, then encode with optimal params ──
# Step 1 — Generate a calibration video matching your display setup
qrstream calibrate -o calib.mp4 --display-hz 60

# Step 2 — Record the calibration video with your phone/camera

# Step 3 — Analyze the recording; note the recommended settings
qrstream calibrate -i recording.mov --target-size 500M

# Step 4 — Encode with the recommended QR version, FPS, and overhead
#          (use the values reported by the analysis step)
qrstream encode large_file.bin -o large_file.mp4 \
    --qr-version 30 --fps 15 --overhead 1.3

# Step 5 — Decode
qrstream decode large_file.mp4 -o recovered.bin

# ── Noisy channel (poor lighting, hand-held recording, etc.) ───────
# Use LT codes with higher overhead for more robust recovery
qrstream encode important.pdf -o important.mp4 \
    --fountain-codec lt --overhead 2.5

# ── Recording-friendly encode ──────────────────────────────────────
# Wider quiet zone helps detectors on blurry captures;
# white lead-in gives time to start recording before QR frames appear
qrstream encode slides.zip -o slides.mp4 \
    --border 15 --lead-in-seconds 2

# ── Direct screen display (no video file needed) ───────────────────
# Show QR frames on screen for the receiver to capture directly.
# No -o means display-only mode; add -o to also produce a video file.
qrstream encode data.zip
qrstream encode data.zip --display -o data.mp4
```

### Python API

```python
from qrstream.encoder import encode_to_video
from qrstream.decoder import extract_qr_from_video, decode_blocks, decode_blocks_to_file

# Encode (default: base45 alphanumeric mode, RaptorQ, 1.2x overhead)
encode_to_video("input.bin", "output.mp4", overhead=1.2, verbose=True)

# Add recording-friendly quiet zone and white lead-in
encode_to_video("input.bin", "output.mp4", border=10.0, lead_in_seconds=1.5)

# Decode to memory
blocks = extract_qr_from_video("output.mp4", verbose=True)
result = decode_blocks(blocks, verbose=True)

# Better for large files: stream directly to file with incremental decompression
written = decode_blocks_to_file(blocks, "recovered.bin", verbose=True)
print(f"wrote {written} bytes")

# Advanced: reuse a decoder completed during extraction (e.g. by scan-phase GE)
blocks, completed_decoder = extract_qr_from_video(
    "output.mp4", verbose=True, return_decoder=True)
written = decode_blocks_to_file(
    blocks, "recovered.bin", decoder=completed_decoder)
```

## Project Structure

```
project-root/
├── pyproject.toml             # Project config & dependencies
├── src/qrstream/
│   ├── cli.py                 # CLI entry (encode/decode/calibrate/colors subcommands)
│   ├── encoder.py             # RaptorQ/LT encode → QR frame generation → MP4 video
│   ├── decoder.py             # Video frame extraction → QR detect → RaptorQ/LT decode → file rebuild
│   ├── raptorq_codec.py       # RaptorQ (RFC 6330) encoder/decoder
│   ├── lt_codec.py            # LT fountain code primitives (PRNG, RSD, BlockGraph, GF(2) rescue)
│   ├── protocol.py            # V3/V4 protocol serialization + base45 codec (legacy base64 decode supported)
│   ├── qr_utils.py            # QR generation + detection (zxing-cpp)
│   ├── calibrate.py           # Calibration video generation and analysis
│   ├── calibration_optimizer.py # Joint QR version/FPS/overhead optimization
│   ├── overhead_policy.py     # Shared fountain-code overhead policy constants
│   ├── ui.py                  # Unified progress/status UI layer
│   ├── display_cache.py       # Bounded display-mode frame cache
│   ├── display_player*.py     # Qt display-mode players
│   └── _compat.py             # Platform compatibility helpers
├── tests/
│   ├── test_lt_codec.py       # LT codec unit tests
│   ├── test_raptorq_codec.py  # RaptorQ codec unit tests
│   ├── test_raptorq_protocol.py # V4 protocol tests
│   ├── test_raptorq_roundtrip.py # RaptorQ roundtrip tests
│   ├── test_protocol.py       # V3 protocol + base45 tests
│   ├── test_decoder.py        # Decoder validation + probe strategy tests
│   ├── test_gaussian_rescue.py # GE rescue fallback tests
│   ├── test_roundtrip.py      # Pure LT codec roundtrip tests (no video I/O)
│   ├── test_qr_generate*.py   # QR generation correctness + mask/glog regressions
│   ├── test_e2e_encode_decode.py  # End-to-end encode→video→decode SHA256 tests
│   ├── test_display_*.py      # Display-mode cache/player tests
│   ├── test_optimizations.py  # Perf optimizations + zxing-cpp + legacy-fallback tests
│   ├── test_calibrate.py      # Calibration subcommand tests
│   ├── test_calibration_optimizer.py # Calibration optimizer tests
│   ├── test_prng_v2.py        # PRNG version tests
│   ├── test_ppm_learning.py   # PPM threshold learning tests
│   ├── test_probe_adaptation.py # Adaptive probe tests
│   ├── test_cli_*.py          # CLI validation tests
│   └── test_ui_reporter.py    # UI reporter tests
└── docs/
    ├── CONTRIBUTING.md        # Contributor workflow, branch/commit/CI rules
    ├── ARCH.md                # Architecture reference
    ├── discovery/             # Investigation notes and findings
    └── tooling/               # Benchmark, profiling, and local container helpers
```

## Technical Details

### V3 Protocol Format (24-byte header + 4-byte trailing CRC)

```
Offset  Size  Field
  0      1    version      0x03
  1      1    flags        bit0=zlib compressed, bit1=high-density mode (base45 alphanumeric),
                           bit2=prng_version (1=SplitMix64, 0=legacy LCG)
  2      8    filesize     uint64 BE (encoded payload size; compressed size when zlib is on)
 10      2    blocksize    uint16 BE
 12      4    block_count  uint32 BE  K = ceil(filesize / blocksize)
 16      4    seed         uint32 BE  PRNG seed
 20      2    block_seq    uint16 BE  monotonically increasing sequence number
 22      2    reserved     reserved (currently 0)
 24      ...  data         blocksize bytes of encoded data
 ...     4    crc32        CRC32(header[0:24] + data)
```

### V4 Protocol Format (RaptorQ, same 24-byte header + 4-byte trailing CRC)

```
Offset  Size  Field
  0      1    version      0x04
  1      1    flags        bit0=zlib compressed, bit1=high-density mode (base45 alphanumeric)
  2      8    filesize     uint64 BE
 10      2    symbol_size  uint16 BE (same position as V3 blocksize)
 12      4    symbol_count uint32 BE  K (same position as V3 block_count)
 16      4    esi          uint32 BE  RaptorQ PayloadId (SBN || local ESI)
 20      2    block_seq    uint16 BE
 22      2    source_blocks uint16 BE  Z; 0 = legacy/single source block
 24      ...  data         symbol_size bytes of encoded data
 ...     4    crc32        CRC32(header[0:24] + data)
```

- Default encoding uses **V4 + RaptorQ + base45 alphanumeric QR**.
- The decoder auto-detects V4 (RaptorQ) vs V3 (LT) from the version byte, and tries base45 → base64 in order, preserving compatibility with older videos.
- V3/V4 extend `filesize` to `uint64` and `block_count`/`symbol_count` to `uint32`, supporting larger files and block counts.

### Encoding Modes

| Mode | QR Content | QR Mode | Overhead | Default |
|------|-----------|---------|----------|---------|
| Base45 alphanumeric | raw bytes → base45 → `0-9A-Z $%*+-./:` | Alphanumeric (5.5 bits/char) | ~67% (but uses denser QR mode → **net denser** than byte mode) | Yes |
| Base64 | raw bytes → base64 string | Byte (8 bits/char) | ~33% | No (`--qr-mode base64`) |

Base45 (RFC 9285) is the default because QR's alphanumeric mode is denser per character than byte mode — at V25/M the base45 payload per frame is ~30% larger than base64, and in practice produces 20–25% smaller videos and 10–20% faster encode/decode.

### Large Files & Low-Memory Paths

- For large **V3/V4** inputs, the shared loader uses `mmap` for random access. LT consumes the mapped input directly, and RaptorQ uses it for systematic source-symbol emission; repair-symbol generation materializes a contiguous buffer only when required by the upstream `raptorq` API.
- When the input is large enough, encoding automatically disables `zlib` compression to preserve the low-memory path; use `--force-compress` to override.
- The decoder supports streaming writes with incremental decompression, reducing memory overhead.
- During decode, the interactive UI shows a two-row live block: a video-scan progress bar (percent, ETA, live detection rate) and a qBittorrent-style file-recovery block map (per-bucket colour density plus an `N/K blocks` counter).  Use `--output-mode log` for CI-friendly `key=value` lines or `--output-mode quiet` for scripted invocations.

### Decoding Pipeline

1. **Probe phase**: Three-phase pipeline — crop exploration (full-res burst), PPM resolution sweep (multi-resolution detection to learn adaptive downscale), and sample rate estimation (pipelined read+detect on spread-out windows). Computes adaptive `sample_rate`, `max_dim`, and crop ROI; completion prints a `Probe` + `Plan` summary
2. **Main scan**: Detect QR codes in parallel at the adaptive sample rate, feeding into the fountain decoder (auto-detected V4/RaptorQ or V3/LT) in real time.  The interactive UI shows a `Scan` row (video progress / ETA / detection rate) and a `File` row (qBittorrent-style block map + `N/K blocks` counter), aligned in a shared table
3. **GE checkpoint** (LT only): If peeling stalls after the main scan, try a GF(2) Gauss-Jordan rescue over the accumulated LT equations.  When the equations already span the missing source blocks, decoding stops here and the completed decoder is reused for writeback. RaptorQ handles recovery internally and skips this step
4. **Targeted recovery**: If the fountain decoder hasn't converged, use linear regression on observed (seed/ESI, frame) pairs to locate missing symbols and re-scan those segments precisely (with CLAHE contrast boost). After each recovery level that adds new unique blocks, run another GE checkpoint (LT) before escalating
5. **LT decode fallback**: Programmatic callers that pass only raw blocks still get the traditional final peeling + GE rescue in `decode_blocks()` / `decode_blocks_to_file()`
6. **Output writeback**: Write recovered blocks sequentially; incremental decompression in compressed mode

### Fountain Code Parameters

#### RaptorQ (RFC 6330, default)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Code type | Systematic RaptorQ | Source symbols transmitted directly; repair symbols for redundancy |
| Recovery | Near-optimal | Decode with high probability from any K received packets |
| Payload ID | SBN \|\| local ESI (4 bytes) | Source Block Number + local Encoding Symbol ID |
| Max source symbols per block | 56,403 | Files with more symbols are split into multiple source blocks |
| Default overhead | 1.2x (min 1.05x, recommended ≥1.10x) | Much lower than LT due to near-optimal recovery |
| Decoding | Internal RaptorQ decoder | No Gauss-Jordan rescue needed |

#### LT Fountain Codes (legacy)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Degree distribution | Robust Soliton Distribution | c=0.1, delta=0.5 |
| PRNG | SplitMix64 mixer + LCG (a=16807, m=2^31-1) | Non-linear seed mixing eliminates sequential-seed correlation |
| XOR | numpy vectorized + in-place | 10-50x faster than pure Python |
| Decoding | Belief Propagation (Peeling) + GF(2) GE rescue | Peeling is the fast path; Gauss-Jordan checkpoints recover stalled but full-rank LT graphs |
| Default overhead | 2.0x (min 1.20x, recommended ≥1.50x) | Higher overhead needed due to sub-optimal LT convergence |

## Testing

```bash
# Default test set (fast; excludes slow and e2e markers via pyproject.toml)
uv run pytest

# End-to-end encode→video→decode tests
uv run pytest -m e2e

# Real-world phone-recording tests (requires fixture videos)
uv run pytest -m slow
```

### Utility Commands

```bash
# Display the colour palette used by the interactive UI
qrstream colors
```

## License

MIT
