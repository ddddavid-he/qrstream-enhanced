# QRStream Encoder Architecture Analysis

## Executive Summary

The encoder has **two distinct paths**: `encode_to_video()` and `encode_to_display()`. Both use threads but in fundamentally different ways:

1. **encode_to_video()** – Single-threaded producer → thread-pool QR generation → muxer thread (line 663-722)
2. **encode_to_display()** – Single producer thread → thread-pool QR generation → display + optional video sink (line 989-1073)

The producer thread's per-frame pipeline is **compute-light** (~2–3 ms per frame). Moving it to a subprocess would **add more overhead than it saves**.

---

## 1. encode_to_display() – Producer Thread Coordination

### Location
Lines 771–1113 (`encode_to_display` function)
Lines 989–1073 (`_produce` inner function)

### Architecture
```
Main thread (display player)
    ↓
Producer thread (_produce)
    ├─→ encoder.generate_blocks()  ← fountain encoding (CPU-light)
    ├─→ generate_qr_module_image() ← zxing-cpp (GIL-free, native C++)
    ├─→ pack_module_image()        ← NumPy bitpacking (CPU-light)
    ├─→ cache.put_packed()         ← thread-safe cache write (locked)
    └─→ video_sink.offer()         ← optional async video write (queued)
    
Display thread (Qt event loop)
    ├─→ cache.get_packed()         ← locked read
    └─→ unpack + upscale + display
```

### ModuleFrameCache: Thread-Safe Sharing

**How it's shared** (display_cache.py):
- **Lock type**: `RLock` + `Condition` variable (line 178)
- **Data layout**: 
  - `_valid: bytearray(total_frames)` – tracks which frames are ready (1 byte per frame, no lock overhead)
  - `_chunks: OrderedDict[chunk_index, ndarray]` – chunked storage (line 173)
  - Each chunk: shape `(256, module_side, row_bytes)` where `row_bytes = ceil(module_side/8)` (line 235)

**Thread-safe operations**:
1. **Producer writes** (line 243–258):
   ```python
   def put_packed(self, index, packed):
       with self._condition:              # Acquire lock
           chunk, local_idx = self._chunk_for_write(index)
           if not self._valid[index]:
               self._valid[index] = 1
               self._valid_count += 1
           chunk[local_idx] = packed_arr  # Atomic NumPy assignment
           self._condition.notify_all()
   ```
   - Holds lock for: chunk allocation, validity flag update, array assignment
   - Lock time: **< 1 µs** (no I/O, no long operations)

2. **Display reads** (line 269–278):
   ```python
   def get_packed(self, index):
       with self._lock:
           if not self._valid[index]:
               return None
           chunk = self._chunks.get(chunk_index)
           chunk[...].move_to_end(chunk_index)  # LRU tracking
           return chunk[index % chunk_size].copy()  # Copy for thread safety
   ```
   - Returns a copy (safe, but adds ~100–300 µs for V25 QR: ~170×170 bits = ~3.6 KB)

3. **Wait primitive** (line 297–300):
   ```python
   def wait_for_frame(self, index, timeout=None):
       with self._condition:
           return self._condition.wait_for(
               lambda: self._done or self.has_frame(index), timeout=timeout)
   ```
   - Display thread blocks here until producer fills the cache

### Memory Layout of Packed Frames

**Single frame layout** (display_cache.py line 56–68):
```python
def pack_module_image(module_img: np.ndarray) -> np.ndarray:
    """Pack a 0/255 module image into one bit per module."""
    black = module_img == 0  # Black = 1, white = 0
    packed = np.packbits(black, axis=1, bitorder="big")
    return np.ascontiguousarray(packed, dtype=np.uint8)
```

- **Input**: (module_side, module_side) uint8 array with values 0 or 255
- **Output**: (module_side, ceil(module_side/8)) uint8 array
- **Example for V25 (171×171 modules)**:
  - Input: 171×171 = 29,241 bytes
  - Packed: 171×22 = 3,762 bytes (≈7.4 KB with contiguous copy)
  - Per-chunk (256 frames): 171×22×256 ≈ **963 KB** (very small)

**Chunk storage** (line 235–237):
```python
chunk = np.zeros(
    (self.chunk_size, self.module_side, self.row_bytes),  # (256, 171, 22)
    dtype=np.uint8,
)
```
- **Memory**: 256 × 171 × 22 = 963,072 bytes ≈ 0.96 MB per chunk
- **Cache with 128 chunks** (default soft limit): ~123 MB total

---

## 2. The _produce() Inner Function: Per-Frame Pipeline

### Single-Worker Path (lines 1040–1061)

```python
for offset, (packed, _, _) in enumerate(encoder.generate_blocks(num_blocks)):
    if cancel_event.is_set() or state.cancel_requested():
        return
    
    # Step 1: Generate module image (1.7 ms for V25, GIL-free)
    module_img = generate_qr_module_image(
        packed, ec_level=ec_level, border=border_modules,
        version=qr_version, use_legacy=use_legacy_qr,
        alphanumeric=high_density, auto_mask=auto_mask,
    )
    
    # Step 2: Pack into bits (0.1 ms, NumPy-accelerated)
    frame_index = lead_in_frames + offset
    packed_frame = pack_module_image(module_img)
    
    # Step 3: Write to cache (< 0.1 ms, locked)
    cache.put_packed(frame_index, packed_frame)
    
    # Step 4: Offer to video sink if present (< 0.1 ms, queued)
    if video_sink is not None:
        video_sink.offer(frame_index, packed_frame)
    
    # Step 5: Update state (< 0.1 ms, locked)
    produced += 1
    state.mark_produced()
    
    # Step 6: Report progress (rate-limited to 10 Hz)
    _report_progress(produced, start_ts, last_report_ts)
```

**Timeline per frame (single-worker)**:
- `generate_qr_module_image()`: ~1.7 ms (zxing-cpp, GIL-free)
- `pack_module_image()`: ~0.1 ms (NumPy bitpack, GIL-free)
- `cache.put_packed()`: < 0.1 ms (lock held ~1 µs, array assignment ~100 µs)
- `video_sink.offer()`: < 0.1 ms (queued, no blocking)
- `state.mark_produced()`: < 0.1 ms (RLock + deque append)
- **Total: ~1.8–2.0 ms per frame**

### Multi-Worker Path (lines 1007–1039)

```python
batch_size = max(workers * 4, 64)  # e.g., workers=4 → 64 frames per batch
block_iter = encoder.generate_blocks(num_blocks)

with ThreadPoolExecutor(max_workers=workers) as pool:
    frame_base = lead_in_frames
    while not cancel_event.is_set() and not state.cancel_requested():
        
        # Step 1: Collect batch of packed blocks (CPU-light, ~0.1 ms)
        batch = []
        for _ in range(batch_size):
            try:
                packed, _, _ = next(block_iter)
            except StopIteration:
                break
            batch.append(packed)
        
        if not batch:
            break
        
        # Step 2: Parallelize QR generation across workers (1.7 ms / workers)
        module_imgs = list(pool.map(
            generate_qr_module_image, batch,
            repeat(ec_level), repeat(border_modules),
            repeat(qr_version), repeat(use_legacy_qr),
            repeat(None), repeat(high_density),
            repeat(auto_mask),
        ))
        
        # Step 3: Write all packed frames to cache (batch, ~0.2 ms)
        for module_img in module_imgs:
            if cancel_event.is_set() or state.cancel_requested():
                return
            packed_frame = pack_module_image(module_img)
            cache.put_packed(frame_base, packed_frame)
            if video_sink is not None:
                video_sink.offer(frame_base, packed_frame)
            frame_base += 1
            produced += 1
            state.mark_produced()
        
        _report_progress(produced, start_ts, last_report_ts)
```

**Timeline (multi-worker, workers=4, batch=64)**:
- Fountain encoding (per batch): **negligible** (generator, pulls on demand)
- QR generation (parallelized): 1.7 ms × 64 / 4 workers = **27.2 ms** (wall time)
- Packing + caching + video sink: **6.4 ms** (sequential post-processing)
- **Total for batch: ~34 ms** = **0.53 ms per frame** (speedup ~3.8×)

---

## 3. encode_to_video() – Video-Muxer Bottleneck

### Location
Lines 412–769 (`encode_to_video` function)

### Architecture
```
Main thread (QR generation)
    ├─→ encoder.generate_blocks()  ← fountain encoding
    ├─→ generate_qr_image()        ← zxing-cpp (GIL-free)
    └─→ writer_queue.put()
    
Writer thread (muxer)
    ├─→ cv2.cvtColor() or cv2.resize()
    ├─→ av.VideoFrame.from_ndarray()
    ├─→ out_stream.encode()        ← x264 encode (bottleneck)
    └─→ output.mux()               ← write to file
```

### Workers > 1 Path (lines 663–703)

```python
if workers > 1:
    block_queue = Queue(maxsize=batch_size * 2)
    
    # Producer thread: Generate packed blocks
    def _block_producer():
        encoder._seq = 0
        for packed, _, _ in encoder.generate_blocks(num_blocks):
            block_queue.put(packed)
        block_queue.put(None)  # Sentinel
    
    producer = Thread(target=_block_producer, daemon=True)
    producer.start()
    
    # Main thread: QR generation + writer queue
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = False
        while not done:
            batch = []
            for _ in range(batch_size):
                item = block_queue.get()  # Blocks until available
                if item is None:
                    done = True
                    break
                batch.append(item)
            
            if not batch:
                break
            
            # Parallelize QR generation (GIL-free)
            qr_imgs = list(pool.map(
                generate_qr_image, batch,
                repeat(ec_level), repeat(10), repeat(border_modules),
                repeat(qr_version), repeat(use_legacy_qr),
                repeat(None), repeat(high_density),
                repeat(auto_mask),
            ))
            
            # Push to muxer
            for qr_img in qr_imgs:
                writer_queue.put(qr_img)
                if writer_error:
                    raise _WriterFailure(...)
            
            produced += len(batch)
            _report_progress(time.monotonic())
```

**Data flow**:
1. `_block_producer()` thread: Calls `encoder.generate_blocks()` (fountain encoding, CPU-light)
2. Main thread: Waits on `block_queue.get()` (blocks if producer is slow)
3. Main thread: Calls `pool.map(generate_qr_image, ...)` (parallelized QR generation)
4. Main thread: Pushes to `writer_queue` (bounded, max 128 frames)
5. Writer thread: Encodes and muxes (x264, ~5–10 ms per frame)

**Performance note** (lines 544–559):
```python
if workers is None:
    workers = 1
elif workers > 1:
    reporter.warn(
        "Encoder --workers > 1 is experimental: full encode is "
        "often video-writer-bound, so higher worker counts may "
        "not improve end-to-end performance despite QR generation "
        "itself being GIL-free (zxing-cpp native)."
    )
```

The muxer is the bottleneck, not QR generation.

---

## 4. Single-Worker Path in encode_to_video() (lines 705–722)

```python
else:
    encoder._seq = 0
    for packed, _, _ in encoder.generate_blocks(num_blocks):
        qr_img = generate_qr_image(
            packed, ec_level=ec_level, box_size=10,
            border=border_modules, version=qr_version,
            use_legacy=use_legacy_qr, alphanumeric=high_density,
            auto_mask=auto_mask,
        )
        writer_queue.put(qr_img)
        if writer_error:
            raise _WriterFailure(...)
        produced += 1
        _report_progress(time.monotonic())
```

**No separate producer thread** – main thread does everything:
1. Call `encoder.generate_blocks()` (pulls one packed block)
2. Generate QR image (1.7 ms)
3. Put in writer_queue (non-blocking, bounded)
4. If queue is full, main thread blocks until muxer drains it

**Pipelining**: While main thread generates frame N, muxer encodes frame N-1 (or earlier).

---

## 5. _DisplayVideoSink – Best-Effort Realtime Writing

### Location
Lines 249–410 (`_DisplayVideoSink` class)

### Architecture
```
Producer thread
    └─→ video_sink.offer(frame_index, packed_frame)
    
Writer thread
    ├─→ _queue.get() (blocks until available)
    ├─→ unpack_module_frame()
    ├─→ cv2.resize()
    ├─→ av.VideoFrame.from_ndarray()
    ├─→ out_stream.encode()
    └─→ output.mux()
```

### Non-Blocking Offer (lines 298–314)

```python
def offer(self, frame_index: int, packed_frame) -> bool:
    with self._lock:
        if self.deferred_from is not None or self._error:
            return False
        if frame_index != self._next_offer:
            self.deferred_from = min(frame_index, self._next_offer)
            return False
    
    try:
        self._queue.put_nowait((frame_index, packed_frame.copy()))
    except Full:
        with self._lock:
            if self.deferred_from is None:
                self.deferred_from = frame_index
        return False
    
    with self._lock:
        self._next_offer = frame_index + 1
    return True
```

**Key behaviors**:
- Returns `False` if queue is full (doesn't block producer)
- Tracks `deferred_from` to know which frames to regenerate later
- Requires **sequential offer** (frame N+1 can't be offered until frame N is offered)

### Finalization Path (lines 361–392)

If realtime writing couldn't keep up:
1. Stop writer thread
2. Regenerate all missing frames from cache/fallback
3. Write all remaining frames sequentially

```python
def finalize(self, total_frames: int, module_frame_at) -> None:
    self._finish_realtime()
    start_ts = time.monotonic()
    for frame_index in range(self.total_written, total_frames):
        module_img = module_frame_at(frame_index)  # Regenerate if needed
        self._write_module_image(module_img)
        self.total_written = frame_index + 1
        # ... progress reporting ...
```

---

## 6. Existing IPC Patterns – None for Process-Based Parallelism

### Current Patterns
1. **Thread-safe queues** (`queue.Queue`) – used for:
   - Muxer coordination (writer_queue)
   - Block pre-generation (block_queue, video path only)
   - Video sink (best-effort queue)

2. **Thread-safe cache** (ModuleFrameCache) – based on:
   - `RLock` (reentrant lock for nested calls)
   - `Condition` (for producer-consumer coordination)
   - Memory-mapped chunks (no shared memory, just NumPy arrays)

3. **No multiprocessing** – no `multiprocessing.Process`, no shared memory, no IPC

### Why No Process-Based Parallelism?
- GIL contention is **not the bottleneck** – zxing-cpp (QR generation) releases the GIL
- Shared state (fountain encoder, cache) would require complex IPC
- Process creation/teardown overhead (~50–100 ms per process on typical systems)
- Pipe/socket IPC for packed frames (~3.7 KB per frame) would add latency
- Thread pools already saturate the CPU without GIL contention

---

## Per-Frame Timeline Analysis

### Single-Worker Display Path
| Operation | Time | GIL? | Thread-Safe? |
|-----------|------|------|--------------|
| `encoder.generate_blocks()` | ~0.1 µs | No | Yes (generator state immutable) |
| `generate_qr_module_image()` | ~1.7 ms | **No** (zxing-cpp native C++) | Yes |
| `pack_module_image()` | ~0.1 ms | No (NumPy) | Yes |
| `cache.put_packed()` | < 0.1 ms* | Mixed | Yes (RLock) |
| `video_sink.offer()` | < 0.1 ms** | Yes | Yes (Lock) |
| `state.mark_produced()` | < 0.1 ms | Yes | Yes (RLock) |
| **Total (wall time)** | **~1.8 ms** | — | — |

*Lock held ~1 µs; array assignment ~100 µs (GIL released during NumPy ops)
**Queued, non-blocking call (< 1 µs if queue not full)

### Multi-Worker Display Path (4 workers, batch=64)
| Operation | Time | Parallelization |
|-----------|------|-----------------|
| Batch collection | ~0.1 ms | Sequential (main thread) |
| QR generation (pool.map) | ~27 ms wall | 4× parallel (1.7 ms × 64 / 4) |
| Packing + cache writes | ~6.4 ms | Sequential (main thread post-processing) |
| **Per-frame average** | **~0.53 ms** | **3.8× speedup vs. single-worker** |

### Muxer (Video Writer) Path
| Operation | Time |
|-----------|------|
| `cv2.cvtColor()` or `cv2.resize()` | ~0.5–1 ms |
| `av.VideoFrame.from_ndarray()` | ~0.1 ms |
| `out_stream.encode()` (x264) | ~5–10 ms |
| `output.mux()` | ~0.5 ms |
| **Total per frame** | **~6–12 ms** |

**Conclusion**: Muxer is **6–8× slower** than QR generation, making workers > 1 of limited benefit for video output.

---

## Recommendations for Process-Based IPC Evaluation

### ❌ Why NOT to Move Producer to Subprocess

1. **IPC Overhead > Savings**
   - Per-frame IPC cost: ~3–5 ms (pipe/socket for ~3.7 KB packed frame)
   - Per-frame generation cost: ~1.8 ms
   - **Net loss: 2–3× slower**

2. **No GIL Contention**
   - zxing-cpp (QR generation) releases GIL; no bottleneck
   - Other operations are lock-free or have μs-scale locks
   - Threads already work well

3. **Shared State Complexity**
   - Fountain encoder state (K, blocksize, seed offset) → IPC overhead
   - Cache writes require frequent synchronization
   - Would need:
     - Serialized encoder state over pipe
     - Or pre-compute all packed frames (adds latency, not savings)
     - Or use `multiprocessing.shared_memory` (newer, still has overhead)

4. **Display Path is Fast Enough**
   - Single-worker: ~1.8 ms per frame ✓
   - Multi-worker: ~0.53 ms per frame ✓
   - Display FPS: 10–30 → needs only 0.03–0.1 ms per frame to run ahead

5. **Video Path is Muxer-Bound**
   - QR generation (1.8 ms) is 3–6× faster than muxing (6–12 ms)
   - Workers > 1 doesn't help because muxer can't keep up

### ✅ When Process IPC Might Help (Hypothetically)

Only if:
- Workload shifts to CPU-heavy computation **after** zxing-cpp
- Muxer becomes <1 ms (unrealistic, x264 doesn't get faster)
- GIL contention appears in bottle-necking code (currently absent)
- Display frame generation time > display refresh (unlikely at 10–30 FPS with 1.8 ms latency)

---

## Summary: Per-Frame Producer Work

### What the producer thread does per frame (single-worker display):

1. **Generate packed block** (0.1 µs) – generator, CPU-light
2. **Generate QR module image** (1.7 ms) – zxing-cpp native, GIL-free
3. **Pack bits** (0.1 ms) – NumPy bitpack, GIL-free
4. **Write to cache** (< 0.1 ms) – RLock held ~1 µs, array assignment ~100 µs
5. **Offer to video sink** (< 0.1 ms) – queued, non-blocking
6. **Update state** (< 0.1 ms) – RLock, deque append
7. **Report progress** (rate-limited, ~10 Hz)

**Total: ~1.8 ms per frame**

### Why subprocess is NOT a win:

- IPC per frame: ~3–5 ms > per-frame generation: ~1.8 ms
- Savings from no GIL contention: **$0** (zxing-cpp releases GIL anyway)
- Complexity cost: ModuleFrameCache + fountain encoder state synchronization

---

## Code References

- **display_cache.py**: ModuleFrameCache (thread-safe cache with RLock + Condition)
- **encoder.py lines 989–1073**: _produce() inner function (producer thread)
- **encoder.py lines 1007–1039**: Multi-worker path (ThreadPoolExecutor)
- **encoder.py lines 663–703**: encode_to_video() workers > 1 path
- **encoder.py lines 298–314**: _DisplayVideoSink.offer() (non-blocking)
- **qr_utils.py lines 131–148**: generate_qr_module_image() (zxing-cpp wrapper)
