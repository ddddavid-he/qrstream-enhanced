# Process vs Thread Model: Research Index

This directory contains comprehensive research on why QRStream uses ThreadPoolExecutor instead of ProcessPoolExecutor for parallel workloads.

## Quick Reference

**TL;DR**: ProcessPool was benchmarked and abandoned. Isolated speedup (2.8-3.0x) didn't translate to end-to-end improvement (0% gain or regression). IPC overhead outweighs CPU gains. Current ThreadPool architecture is optimal.

## Documents

### 1. **PROCESS_VS_THREAD_EXECUTIVE_SUMMARY.txt** (Executive Level)
- **Audience**: Anyone wanting to understand the decision without deep technical details
- **Length**: ~12 KB
- **Contains**:
  - 5 pieces of evidence backing the decision
  - Clear "do this, don't do that" recommendations
  - Where to look for encoder/decoder speedup improvements
  - When to revisit threading decisions

**Start here if you**: Want a quick understanding of why threads were chosen

---

### 2. **RESEARCH_PROCESS_VS_THREAD_HISTORY.md** (Technical Deep-Dive)
- **Audience**: Engineers, architects, future maintainers
- **Length**: ~15 KB
- **Contains**:
  - 11 sections covering complete history
  - Git commit analysis (0de4395, 2a3a579, 6ea15a1, 9862417, 78f2ef6)
  - Detailed benchmarks and performance data
  - CPU-bound vs I/O-bound breakdown
  - GIL timeline and impact analysis
  - Shared memory & IPC patterns
  - C extensions and GIL behavior
  - Design rationale with evidence

**Start here if you**: Need technical justification for architectural decisions

---

### 3. **Console Output Summary** (Visual Quick Reference)
- Displayed above in formatted ASCII art
- Contains:
  - Performance tables
  - Architecture diagrams
  - Key findings in bullet format
  - Bottleneck analysis

**Refer to this for**: Quick visual reference of performance numbers

## Key Findings

### Finding #1: ProcessPool Was Tested and Abandoned
- **Commit**: 0de4395 (2026-05-03)
- **Document**: `dev/ENCODER_PROCESSPOOL_ABANDONED.md`
- **Isolated speedup**: 2.8-3.0x faster QR generation
- **Full pipeline result**: 0% improvement (IPC overhead dominates)
- **Status**: Do not revisit

### Finding #2: Video Muxing is the Bottleneck
- Muxing: 50-100 ms per frame
- QR generation: 1.7 ms per frame
- Parallelizing QR gen doesn't help when muxing serializes you
- Real win: +30% from dedicated writer thread (Tier 1.1)

### Finding #3: zxing-cpp Changed Everything (v0.9+)
- Provides 3.6x faster QR generation
- Releases GIL during operations
- Enables real thread parallelism
- But encoder still defaults to workers=1 (muxing bottleneck)

### Finding #4: Decoder Benefits from Threading
- Detection shows 3.73x speedup with 4 workers
- ProcessPool would be devastating (11.6x slower)
- Default: workers = cpu_count (use all cores)

### Finding #5: Measured Verdict
| Scenario | ThreadPool | ProcessPool | Winner |
|----------|-----------|-------------|--------|
| QR generation (isolated) | 2.377s | 1.026s | ProcessPool 2.8x faster |
| Full encode (realistic) | 2.36s | abandoned | ThreadPool 0% regression |
| QR detection | 0.1347s | 1.5707s | ThreadPool 11.6x faster |

## Core Documents Referenced

### Original Decision Logs
- `dev/ENCODER_PROCESSPOOL_ABANDONED.md` - ProcessPool experiment results
- `docs/discovery/DISCOVERY-decode-perf-optimizations-2026-05-07.md` - Decode analysis

### Implementation
- `src/qrstream/encoder.py` (lines 544-722) - Threading + worker implementation
- `src/qrstream/decoder.py` (lines 1812-1819) - Worker count strategy

### Key Commits
1. **9862417** (2026-04-22) - Dedicated writer thread (+30% speedup)
2. **2a3a579** (2026-05-03) - Set encoder default workers to 1
3. **0de4395** (2026-05-03) - Archive ProcessPool experiment
4. **6ea15a1** (2026-05-07) - zxing-cpp backend (GIL-free, 3.6x faster)
5. **78f2ef6** (2026-05-07) - FFmpeg threading deadlock fix

## Decision Summary

### ✅ Why ThreadPoolExecutor
- GIL-free operations (zxing-cpp, PyAV, NumPy, OpenCV)
- Shared memory access (efficient for video I/O)
- No IPC overhead
- Measured benchmarks prove no end-to-end improvement with ProcessPool
- Simpler debugging and deployment

### ❌ Why Not ProcessPoolExecutor
- IPC overhead outweighs isolated CPU gains
- Detection path: 11.6x slower on pure work due to frame serialization
- Encoding path: 0% end-to-end improvement despite 2.8x isolated speedup
- No crash isolation benefit (zxing-cpp doesn't crash)
- Memory overhead (process address space duplication)

### Encoder Strategy
- **Default**: workers=1 (muxing bottleneck, not QR generation)
- **Real win**: Dedicated writer thread overlapping I/O (+30%)
- **Why not more workers**: Doesn't overcome muxer serialization

### Decoder Strategy
- **Default**: workers=cpu_count (detection scales linearly)
- **Why**: zxing-cpp releases GIL, enables real parallelism
- **Evidence**: 3.73x speedup with 4 workers on detection

## For Future Maintainers

### Do NOT Revisit Unless:
1. ❌ You want to use ProcessPool (tested, doesn't help)
2. ❌ You want to increase encoder workers > 1 by default (tested, no win)
3. ❌ You want to optimize QR generation further (already 1.7 ms with zxing-cpp)

### DO Look Here For Speedup:
1. ✅ Codec selection (mjpeg faster than x264)
2. ✅ Video muxer optimization (currently 50-100 ms bottleneck)
3. ✅ Hardware acceleration (GPU encoding)
4. ✅ Decoder: frame prep optimization (if profiles show benefit)

### When to Revisit Decisions:
1. ✅ If PEP 703 (free-threaded Python) becomes mainstream
2. ✅ If you eliminate the video muxer bottleneck
3. ✅ If a new C extension crashes like WeChatQRCode did

## Research Methodology

This research was conducted by:
1. **Git history analysis**: Commits, branch history, archived experiments
2. **Document review**: Benchmarking notes, design docs, discovery reports
3. **Code analysis**: Implementation of threading model in encoder.py and decoder.py
4. **Performance data**: Measured benchmarks from commits and documentation
5. **Architectural assessment**: IPC patterns, shared memory usage, GIL impact

All findings are backed by measured evidence, not speculation.

## Questions?

Each document answers different levels of detail:
- **"Should I use processes?"** → See Executive Summary
- **"Why specifically was ProcessPool abandoned?"** → See Technical History
- **"What's the bottleneck in encode/decode?"** → See Performance Breakdown
- **"How does threading work in the code?"** → See src/qrstream/encoder.py

---

**Generated**: 2026-05-17  
**Research Scope**: Process vs Thread model decisions in QRStream  
**Evidence Base**: Git history, benchmarks, code analysis  
**Status**: Complete and documented
