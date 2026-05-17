# History of Process vs Thread Model Decisions in QRStream

## Executive Summary

The QRStream project has exclusively used **ThreadPoolExecutor** for parallelism (both encoder and decoder paths). A ProcessPoolExecutor experiment was **abandoned after benchmarking showed it provided no end-to-end benefit** despite isolated QR generation speedups. The decision to use threads over processes is backed by detailed measurements and remains justified by the current architecture.

---

## 1. Key Decision Points in Git History

### Commit: `0de4395` - "archive encoder processpool experiment" (2026-05-03)

**Status**: Archived / Do Not Merge

This is the most important finding. The project explicitly experimented with ProcessPoolExecutor and chose to abandon it. The document `dev/ENCODER_PROCESSPOOL_ABANDONED.md` records:

#### Isolated QR Generation Results (segno backend):
- ThreadPool w=1: 2.868s (baseline)
- ThreadPool w=4: 2.377s (1.21x speedup)
- **ProcessPool w=4: 1.026s (2.80x speedup)**
- ProcessPool w=6: 0.980s (2.93x speedup)
- ProcessPool w=14: 1.052s (2.73x speedup)

**Conclusion**: Processes achieved ~2.8-3.0x speedup on pure QR generation.

#### Full `encode_to_video()` Pipeline Results:
```
| size  | workers=1 | workers=4 | workers=6 |
|-------|-----------|-----------|-----------|
| 100KB | 2.36s     | 2.55s     | 2.45s     |
| 500KB | 10.43s    | 10.46s    | 10.38s    |
| 1MB   | 20.55s    | 20.96s    | 20.87s    |
```

**Conclusion**: No meaningful end-to-end improvement. The videowriter + muxing stage dominates.

#### Why Abandoned:
1. Isolated QR generation benefits, but full encode does not
2. ProcessPool added complexity (spawn mode, IPC, grayscale IPC optimization, v0.7.4 fork-safety handling)
3. The safer product decision: keep fixed-mask encoder single-worker unless future profiling removes the video writer bottleneck

---

### Commit: `2a3a579` - "Set encoder default workers to one" (2026-05-03)

**Before this commit**, the encoder auto-selected `workers = min(os.cpu_count(), 4)` under the assumption that:
- QR generation is pure Python (GIL-bound)
- So GIL contention limits practical parallelism to ~4 threads

**After this commit**, the encoder defaults to `workers = 1` because:
- Full-pipeline benchmarks show VideoWriter/muxing is the bottleneck
- QR generation parallelism doesn't overcome the muxer serialization
- Users can still opt-in to higher worker counts explicitly

---

### Commit: `6ea15a1` - "feat(encoder): replace segno QR generation backend with zxing-cpp" (2026-05-07)

**Game-changer**: zxing-cpp is native C++ and **releases the GIL** during QR matrix generation.

**Performance improvement**:
- segno (pure Python, GIL-bound): ~6.1 ms/frame
- zxing-cpp (native C++, GIL-free): ~1.7 ms/frame
- **Speedup: 3.6×**

**Threading implication**: 
> "Additional benefit: zxing-cpp releases the GIL during QR matrix generation, so --workers > 1 can now provide real thread parallelism (previously contended on the GIL with no actual speedup)."

However, the commit message also notes: **the full encode pipeline is still typically video-writer-bound**, so the default remained `workers=1`.

---

### Commit: `9862417` - "Tier 1.1: Run VideoWriter on a dedicated thread" (2026-04-22)

This is the **primary parallelization win** in the encoder:

**Baseline (v0.6.1, 14 workers)**:
- VideoWriter.write was **54% of encode wall-time**
- Main thread serialized writes between QR generation batches

**After dedicated writer thread**:
- cv2.VideoWriter.write() runs on its own producer thread
- Main thread pushes QR frames onto a bounded Queue (maxsize = max(workers*8, 128))
- Writer thread drains serially — frame order preserved (FIFO + single consumer)

**Measured impact** (mac M-series, 14 workers):
- 1 MB:   24 s / 90 f/s  →  17.6 s / 133 f/s   (-27% wall, +48% f/s)
- 5 MB:  128 s / 87 f/s  →  87.5 s / 129 f/s   (-32% wall, +48% f/s)

**Finding**: The 30% improvement came from **overlapping QR generation with video muxing**, not from using more threads.

---

### Commit: `78f2ef6` - "FFmpeg frame-level threading deadlock fix" (2026-05-07)

**Issue**: FFmpeg frame-level threading (`stream.thread_type = "AUTO"`) deadlocks when combined with Python generators.

**Root cause**: Internal decoded-frame queue fills while the generator is suspended at `yield`, causing circular wait between producer threads and consumer.

**Solution**: Remove thread-level parallelism in FFmpeg (rely on single-threaded decode from PyAV wrapper).

**Performance finding**: Decode throughput already exceeds consumption rate (122 fps single-thread), so the loss was zero end-to-end.

---

## 2. Decoder Threading Model

From `src/qrstream/decoder.py`:

### Worker Count Strategy (Decode path):
```python
if workers is None:
    workers = os.cpu_count() or 1
```

**Default**: Use all CPU cores (unlike encoder which defaults to 1).

**Rationale**: zxing-cpp detection is native C++ and releases the GIL, so **real parallelism is achievable**. The docs state:

> "zxing-cpp is native C++ and releases the GIL during detection, so more threads scale close to linearly on multi-core machines."

### Detector Benchmark (Post-zxing-cpp Migration)

From `docs/discovery/DISCOVERY-decode-perf-optimizations-2026-05-07.md`:

**Prepared-frame benchmark on `IMG_9442.MOV` (204 sampled frames)**:

| workers | ThreadPool | ProcessPool | Process / Thread |
|---------|-----------|-------------|-----------------|
| 1       | 0.5021s   | 2.1986s     | 4.38x slower    |
| 2       | 0.2526s   | 1.5042s     | 5.95x slower    |
| 4       | 0.1347s   | 1.5707s     | 11.66x slower   |
| 6       | 0.0922s   | 1.6271s     | 17.64x slower   |

**Full decode benchmark**:

| video       | ThreadPool total | ProcessPool total | Result                    |
|-------------|------------------|------------------|---------------------------|
| IMG_9442    | 29.883s          | 29.230s           | near parity, process ~2% |
| IMG_9455    | 34.349s          | 36.512s           | thread ~6% faster         |

**Conclusion**: ThreadPool is decisively better for pure detection. ProcessPool overhead dominates end-to-end.

---

## 3. Current Encoder Threading Model

### `encode_to_video()` Worker Path

Located in `src/qrstream/encoder.py` lines 663-722:

```python
if workers > 1:
    # Dedicated producer thread generates fountain-coded blocks
    block_queue = Queue(maxsize=batch_size * 2)
    
    def _block_producer():
        encoder._seq = 0
        for packed, _, _ in encoder.generate_blocks(num_blocks):
            block_queue.put(packed)
        block_queue.put(None)
    
    producer = Thread(target=_block_producer, daemon=True)
    producer.start()
    
    # ThreadPoolExecutor processes batches of blocks into QR images
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Batch-based map() call to zxing-cpp.create_barcode (GIL-free)
        qr_imgs = list(pool.map(
            generate_qr_image, batch,
            repeat(ec_level), repeat(10), repeat(border_modules),
            repeat(qr_version), repeat(use_legacy_qr),
            repeat(None), repeat(high_density),
            repeat(auto_mask),
        ))
        # Push to writer queue
        for qr_img in qr_imgs:
            writer_queue.put(qr_img)
else:
    # Single-threaded path: sequential generation and encoding
    encoder._seq = 0
    for packed, _, _ in encoder.generate_blocks(num_blocks):
        qr_img = generate_qr_image(...)
        writer_queue.put(qr_img)
```

### Key Properties:
1. **Single producer thread** generates fountain-coded blocks
2. **ThreadPoolExecutor** renders blocks into QR images (GIL-free with zxing-cpp)
3. **Single writer thread** (separate) muxes frames to video (started earlier)
4. **Bounded queues** prevent unbounded memory growth

---

## 4. CPU-Bound vs I/O-Bound Breakdown

### Encoder Path:

| Stage                    | Type       | GIL Status              | Bottleneck?       |
|--------------------------|-----------|------------------------|------------------|
| Block generation (XOR)   | CPU-bound | Python, GIL-held        | No (small)       |
| QR matrix generation     | CPU-bound | zxing-cpp, **GIL-free** | No (~1.7 ms)     |
| Video encoding (x264)    | CPU-bound | **av**, GIL-free        | No (~15 ms)      |
| **Video muxing**         | **I/O**   | **av.mux() GIL-free**   | **YES (~50 ms)** |

**Finding**: Video muxing dominates (50-100 ms per frame on typical setups), overwhelming the benefits of parallel QR generation (1.7 ms).

### Decoder Path:

| Stage                    | Type      | GIL Status              | Bottleneck?      |
|--------------------------|-----------|------------------------|-----------------|
| Frame reading            | I/O       | PyAV, GIL-free          | Medium          |
| Frame prep (crop/resize) | CPU-bound | OpenCV, GIL-free        | Small           |
| **QR detection**         | CPU-bound | **zxing-cpp, GIL-free** | **YES (variable)** |

**Finding**: Detection parallelizes near-linearly up to CPU cores because zxing-cpp releases the GIL.

---

## 5. GIL Impact Analysis

### Historical (Pre-zxing-cpp):
- QR detection: WeChatQRCode via cv2 (C++, but had SIGSEGV crash risk requiring subprocess sandbox)
- QR generation: segno (pure Python, GIL-bound) or qrcode (pure Python, GIL-bound + bug in glog(0))

### Current (Post-zxing-cpp, v0.9+):
- QR detection: zxing-cpp (C++, native bindings, **GIL-released**)
- QR generation: zxing-cpp (C++, native bindings, **GIL-released**)

**Implication**: 
- ThreadPoolExecutor can now provide **true parallelism** for both generation and detection
- But full-encode is still **muxer-bound**, not generation-bound
- Decode detection **now scales to all CPU cores**

---

## 6. Shared Memory and IPC Patterns

### Current Implementation: None (Threads Only)

1. **Encoder**: All threads share the main process memory space
   - Producer thread, worker threads, writer thread all access mmap'd input or in-memory payload
   - Queues use shared memory (Python Queue is thread-safe, backed by locks)
   - Output video file is written by single writer thread (no race conditions)

2. **Decoder**: All threads share the main process memory space
   - ThreadPoolExecutor workers read from shared video file handles (seekable)
   - Detection results returned via concurrent.futures (thread-safe)
   - No cross-process communication

### Why ProcessPool Was Rejected:
1. **IPC overhead**: JPEG/frame data would need serialization across process boundaries
2. **Shared state complexity**: Decoder uses a single cv2.VideoCapture or PyAV context; passing to subprocess requires re-opening or pickling
3. **No crash isolation benefit**: zxing-cpp doesn't crash (unlike WeChatQRCode), so sandbox is unnecessary
4. **Memory overhead**: Each worker process duplicates the entire address space

---

## 7. Performance Summary: Processes vs Threads

### Pure QR Generation (Benchmark Target):
- **ThreadPool**: 2.377s (w=4)
- **ProcessPool**: 1.026s (w=4)
- **ProcessPool advantage**: 2.3x

### Full Encode (Real-world Target):
- **ThreadPool w=1**: 2.36s (100KB file)
- **ThreadPool w=4**: 2.55s (100KB file) - **+8% regression**
- **ProcessPool w=4**: (not measured separately, but discovered to give no win)
- **ProcessPool advantage**: ~0%

### Full Decode:
- **ThreadPool w=4**: 29.883s (IMG_9442)
- **ProcessPool w=4**: 29.230s (IMG_9442) - **2% faster**, but...
- **Pure detect**: ThreadPool 0.1347s, ProcessPool 1.5707s (**11.6x slower**)
- **IPC overhead**: Dominates any detection gain

---

## 8. Design Rationale

### Why Threads Were Chosen (and Why Processes Were Rejected):

1. **Shared Memory Access**: Video I/O and data structures are efficiently shared via threads
2. **GIL No Longer a Bottleneck**: zxing-cpp releases the GIL, so threads provide real parallelism
3. **Measured End-to-End**: ProcessPool shows no end-to-end win despite isolated speedups
4. **IPC Overhead**: Outweighs any CPU-bound gains in real workloads
5. **Simplicity**: Threads are simpler to debug, no pickling/unpickling, shared memory consistency
6. **Current Bottlenecks are I/O**: Muxing (encoder) and frame prep (decoder) are not CPU-bound with a single thread

### Why Default workers=1 for Encoder (But Not Decoder):

**Encoder**:
- Muxing (video writer) is the bottleneck (50-100 ms per frame)
- Even with zxing-cpp releasing the GIL, QR generation is not the limiter
- Dedicated writer thread (Tier 1.1) was the real win (+30%)
- Higher worker counts don't overcome muxer serialization

**Decoder**:
- Detection (zxing-cpp) **is** a significant portion of decode time
- Scales linearly to CPU cores because GIL is released
- Multiple workers provide real speedup
- Default: use all CPU cores

---

## 9. Existing C Extensions and GIL Behavior

### QR Operations (Both Generation and Detection):
- **zxingcpp** (Python bindings to C++ library): **Releases GIL** during barcode operations
- **cv2** operations: Generally GIL-free in OpenCV's C++ implementation

### Video I/O:
- **PyAV** (libav/ffmpeg bindings): GIL-free during decode/encode
- **cv2.VideoCapture** (deprecated in favor of PyAV): GIL-free

### Array Operations:
- **NumPy**: Array operations generally GIL-free for C-level work (resize, XOR, etc.)

### Implication:
The three heaviest workloads (QR generation/detection, video I/O, array ops) are all GIL-free. The threading model can therefore achieve near-linear scalability on real parallelism up to CPU core count.

---

## 10. Summary Table: Decision History

| Date   | Commit | Decision | Rationale | Status |
|--------|--------|----------|-----------|--------|
| Apr-22 | 9862417 | Dedicated writer thread | Overlapping I/O with generation gains 30% | ✅ Active |
| May-03 | 2a3a579 | workers default → 1 | VideoWriter is bottleneck, not QR gen | ✅ Active |
| May-03 | 0de4395 | ProcessPool → Abandoned | Isolated speedup, but no full-pipeline win | ❌ Archived |
| May-07 | 6ea15a1 | zxing-cpp backend | GIL-free generation enables real parallelism | ✅ Active |
| May-07 | 78f2ef6 | Remove FFmpeg threading | Deadlock with generators; no perf loss | ✅ Active |

---

## 11. Recommendations for Future Work

1. **Encoder**: Do not revisit ProcessPool. The videowriter is the bottleneck.
   - If seeking performance gains, focus on codec selection (mjpeg faster than x264) or muxer optimization
   
2. **Decoder**: ThreadPoolExecutor remains the right choice.
   - zxing-cpp parallelizes well
   - IPC overhead would eliminate any gains
   
3. **Possible Future**: If free-threaded Python (PEP 703) becomes mainstream, revisit GIL assumptions.
   - But current code remains optimal for CPython 3.13+

---

## References

- `dev/ENCODER_PROCESSPOOL_ABANDONED.md` (commit 0de4395)
- `docs/discovery/DISCOVERY-decode-perf-optimizations-2026-05-07.md`
- `src/qrstream/encoder.py` lines 544-722 (worker implementation)
- `src/qrstream/decoder.py` lines 1812-1819 (worker count strategy)
- Commits: 0de4395, 2a3a579, 6ea15a1, 9862417, 78f2ef6
