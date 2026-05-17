# QRStream Research Documentation Index

## 🎯 Quick Navigation

### Start Here
- **[COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md)** ⭐ MAIN DOCUMENT
  - Executive summary of all findings
  - Complete analysis of GIL contention, threading vs processes, recommendations
  - Read this first for the full picture

### Detailed Technical Analysis
1. **[GIL_README.md](./GIL_README.md)** - GIL Analysis Overview
   - Navigation guide for GIL documentation
   - Key findings summary
   - Verification commands

2. **[GIL_QUICK_REFERENCE.md](./GIL_QUICK_REFERENCE.md)** - GIL Quick Facts
   - TL;DR for busy developers
   - Timeline visualization
   - Library GIL status table
   - Action items by priority

3. **[GIL_ANALYSIS.md](./GIL_ANALYSIS.md)** - Deep Dive GIL Analysis
   - 20-minute comprehensive technical analysis
   - All 7 dependencies with GIL status
   - 4 detailed pipeline stages with timings
   - Complete timeline diagram
   - Contention analysis at 30/60/120/240 fps
   - Optimization recommendations with priorities

4. **[GIL_CODE_LOCATIONS.md](./GIL_CODE_LOCATIONS.md)** - Code Reference
   - File-by-file breakdown with line numbers
   - GIL status and duration per code section
   - Code snippets for each stage
   - Verification commands
   - Code modification guide (Cython, C extensions, PyO3)

### Historical Context & Experiments
- **archive/encoder-processpool** (git branch tag)
  - Abandoned ProcessPool experiment (May 3, 2026)
  - Command: `git show archive/encoder-processpool:dev/ENCODER_PROCESSPOOL_ABANDONED.md`
  - Key Finding: ProcessPool showed 0% full-pipeline speedup despite 2.9x isolated QR gen speedup
  - Reason: VideoWriter is the actual bottleneck

### Performance Profiling Tools
- **[docs/tooling/perf-profile/](./tooling/perf-profile/)**
  - Profile encode: `python docs/tooling/perf-profile/profile_encode.py`
  - Profile decode: `python docs/tooling/perf-profile/profile_decode.py`
  - Profile hotpaths: `python docs/tooling/perf-profile/profile_hotpaths.py`
  - Results: `docs/tooling/perf-profile/results/`

### Discovery Documents
- **[discovery/DISCOVERY-display-player-fps-bottleneck-2026-05-17.md](./discovery/DISCOVERY-display-player-fps-bottleneck-2026-05-17.md)**
  - Latest display player FPS bottleneck analysis
  - Frame pacing investigation
  - GUI rendering performance

- **[discovery/DISCOVERY-decode-perf-optimizations-2026-05-07.md](./discovery/DISCOVERY-decode-perf-optimizations-2026-05-07.md)**
  - Decode performance optimization opportunities
  - Gauss-Jordan elimination analysis
  - Peeling graph performance

## 📊 Key Findings Summary

### GIL Contention
- **Status**: Negligible for real workloads
- **GIL-Holding**: Only base45_encode (pure Python) holds GIL for ~0.3ms per frame
- **GIL-Released**: 93-98% of execution time
- **Contention at 30fps**: <1% probability
- **Contention at 240fps**: ~7% probability (still low)

### Threading vs Processes
- **Current Model**: ThreadPoolExecutor (threads)
- **Experiment Result**: ProcessPoolExecutor (processes) added 0% end-to-end speedup
- **Isolated QR Gen**: Processes show 2.9x speedup, but it's not the bottleneck
- **Real Bottleneck**: VideoWriter/muxing, not Python code
- **Recommendation**: Keep threading model, don't switch to processes

### Default Workers Setting
- **Current**: `workers=1` (single-threaded QR generation)
- **Why**: No performance benefit from parallelism, reduces memory overhead
- **History**: Changed from `min(os.cpu_count(), 4)` after ProcessPool experiment
- **Status**: Optimal for the use case

### Performance Bottleneck
- **Not**: GIL contention
- **Not**: QR generation speed
- **Actually**: VideoWriter/codec output (I/O bound)
- **Solution Path**: Optimize codec/video writing, not Python encoding

## 🔍 Investigation Points Completed

✅ **GIL Analysis**
- All 7 dependencies checked for GIL behavior
- All 4 pipeline stages analyzed with timings
- Contention probability calculated for multiple frame rates
- Codebase location mapping with line numbers

✅ **Process Model History**
- Git history analyzed for threading decisions
- ProcessPool experiment discovered and analyzed
- Conclusion documented with experimental data
- Future optimization paths identified

✅ **Encoder Architecture**
- DisplayProducer threading model documented
- Worker pool implementation analyzed
- Producer-consumer pattern confirmed
- GUI thread interaction verified

✅ **Dependencies and Bottlenecks**
- All C/C++/Rust extensions identified
- Pure Python operations quantified
- Real bottleneck (VideoWriter) confirmed
- Optimization recommendations provided

## 📋 File Manifest

| Document | Type | Size | Focus |
|---|---|---|---|
| COMPREHENSIVE_ANALYSIS.md | Summary | ~180 lines | Main findings and recommendations |
| GIL_README.md | Index | ~200 lines | GIL documentation navigation |
| GIL_QUICK_REFERENCE.md | Cheat Sheet | ~250 lines | Quick facts and timelines |
| GIL_ANALYSIS.md | Deep Dive | ~500 lines | Comprehensive technical analysis |
| GIL_CODE_LOCATIONS.md | Reference | ~400 lines | Code locations with line numbers |

**Total Documentation**: ~1,500 lines of detailed analysis

## 🎯 Next Steps by User Role

### For Performance-Focused Users
1. Read [COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md)
2. If targeting 30-60fps: No action needed, architecture is optimal
3. If targeting 120+ fps: See "Optimization Options" in comprehensive analysis
4. For profiles: Run `docs/tooling/perf-profile/profile_encode.py`

### For Core Developers
1. Read [COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md) for context
2. Review [GIL_ANALYSIS.md](./GIL_ANALYSIS.md) for technical depth
3. Check [GIL_CODE_LOCATIONS.md](./GIL_CODE_LOCATIONS.md) for exact code references
4. Reference existing architecture: `src/qrstream/encoder.py` DisplayProducer class

### For Contributors Considering Changes
1. Understand current architecture: [COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md)
2. Before parallelizing: Check ProcessPool experiment conclusion
3. Before optimizing: Profile with `docs/tooling/perf-profile/profile_encode.py`
4. Consider bottleneck: VideoWriter/codec, not Python code

### For Research/Understanding
1. Read all GIL documentation in order (README → Quick Ref → Analysis → Code Locations)
2. Check ProcessPool experiment on archive branch
3. Profile current workload with profiling tools
4. Review performance discovery documents

## 🔗 External References

### Key Commits
- **0de4395** (May 3, 2026): "archive encoder processpool experiment"
  - Contains: ProcessPool implementation and results
  - Branch: archive/encoder-processpool
  - Status: Do not merge, for reference only

### Key Files to Understand
- `src/qrstream/encoder.py`: DisplayProducer, threading model
- `src/qrstream/protocol.py`: base45_encode (GIL bottleneck)
- `src/qrstream/lt_codec.py`: PRNG and XOR operations
- `src/qrstream/raptorq_codec.py`: RaptorQ encoder/decoder
- `src/qrstream/qr_utils.py`: QR generation (uses zxing-cpp)
- `src/qrstream/display_cache.py`: Frame caching (numpy operations)

### Related Documentation
- [README.md](../README.md): Project overview
- [BRANCHING.md](../BRANCHING.md): Branch strategy and experimental branches
- [pyproject.toml](../pyproject.toml): Dependencies and versions

## 📞 Questions This Research Answers

### "Should we use ProcessPoolExecutor instead of ThreadPoolExecutor?"
**Answer**: No. The ProcessPool experiment showed 0% full-pipeline speedup despite isolating and parallelizing QR generation. The real bottleneck is VideoWriter/codec output.

### "Is GIL contention a problem?"
**Answer**: No. Only base45_encode holds the GIL (~0.3ms per frame). Contention probability is <1% at 30fps, <7% at 240fps. Not a practical concern.

### "Why is workers=1 the default?"
**Answer**: Because adding more workers doesn't accelerate the full pipeline. Single worker reduces memory overhead while maintaining responsive GUI through ThreadPoolExecutor's producer-consumer pattern.

### "What should we optimize for 120+ fps?"
**Answer**: Not GIL contention, not QR generation parallelism, but VideoWriter codec configuration. Profile with perf-profile tools to see where time is actually spent.

### "Could we switch to free-threaded Python 3.13+"?
**Answer**: Unlikely to help. The bottleneck is I/O (VideoWriter), and GIL contention is already negligible. Free-threaded Python would add complexity with minimal benefit.

---

**Research Completed**: May 17, 2026
**Status**: All investigation points completed and documented
**Recommendation**: Archive this research as reference for future performance decisions
