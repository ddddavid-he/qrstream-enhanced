# GIL Analysis: Code Location Reference

## File Structure & GIL-Relevant Sections

### 1. raptorq_codec.py - RaptorQ Encoding (Rust → GIL Released ✅)

**File**: `src/qrstream/raptorq_codec.py`

| Section | Lines | Operation | GIL Status | Notes |
|---------|-------|-----------|-----------|-------|
| `RaptorQEncoder.__init__` | 169-210 | Initialization & probing | RELEASED ✅ | Rust: `_raptorq.Encoder.with_defaults()` |
| `_ensure_encoder()` | 222-228 | Encoder caching | RELEASED ✅ | Creates Rust encoder once |
| `generate_blocks()` | 258-327 | Packet generation | RELEASED ✅ | Rust: `get_encoded_packets()` (line 277) |
| `_iter_source_packets()` | 242-250 | Systematic packets | RELEASED ✅ | Python loop, but Rust data |
| Decoder.consume_block() | 377-455 | Decoding | RELEASED ✅ | Rust: `_rq_decoder.decode()` (line 438) |

**Key Call**: Line 277
```python
packets = self._ensure_encoder().get_encoded_packets(repair_per_block)
```
⬅️ **This is a Rust call that releases the GIL**

---

### 2. lt_codec.py - LT Encoding (Mixed: Python PRNG + numpy XOR)

**File**: `src/qrstream/lt_codec.py`

| Section | Lines | Operation | GIL Status | Notes |
|---------|-------|-----------|-----------|-------|
| `splitmix64_mix()` | 59-74 | PRNG seed mixer | HOLDS ❌ | Pure Python bit ops (~1 μs) |
| `PRNG.set_seed()` | 146-147 | State initialization | HOLDS ❌ | Trivial, not a bottleneck |
| `PRNG.get_src_blocks()` | 149-176 | Block selection | HOLDS ❌ | Lines 167: splitmix64_mix() holds GIL ~1 μs |
| `xor_bytes()` | 181-195 | XOR operation | RELEASED ✅ | **np.bitwise_xor() at line 195** |
| `BlockGraph.add_block()` | 245-271 | Graph insertion | MIXED | np.xor + pure Python logic |
| `BlockGraph.eliminate()` | 273-286 | Peeling step | MIXED | np.xor operations in loop |
| `try_gaussian_rescue()` | 303-412 | Gauss-Jordan solver | RELEASED ✅ | Heavy numpy: packbits (l.357), bitwise_xor (l.395) |

**Key Lines**:
- Line 167: `self.state = splitmix64_mix(blockseed)` → holds GIL (<1 μs)
- Line 195: `return bytes(np.bitwise_xor(arr_a, arr_b))` → releases GIL ✅

---

### 3. protocol.py - Serialization & Encoding

**File**: `src/qrstream/protocol.py`

#### Section A: base45_encode() ⚠️ **MAIN GIL CONTENTION SPOT**

| Lines | Operation | GIL Status | Duration |
|-------|-----------|-----------|----------|
| 100-125 | base45_encode() | **HOLDS ❌** | ~200-500 μs |

**Code**:
```python
def base45_encode(data: bytes) -> bytes:    # Line 100
    out = bytearray()
    i = 0
    length = len(data)
    while i + 2 <= length:                  # Line 109 - Pure Python loop
        n = (data[i] << 8) | data[i + 1]    # Line 110 - Holds GIL
        c = n // 2025                       # Line 111 - Holds GIL (division)
        n -= c * 2025
        b = n // 45
        a = n - b * 45
        out.append(_B45_BYTES[a])           # Line 115 - Array indexing (Holds GIL)
        out.append(_B45_BYTES[b])
        out.append(_B45_BYTES[c])
        i += 2
    # ... odd-byte handling ...
    return bytes(out)                       # Line 125
```

**⚠️ Impact**: Holds GIL for entire 100-125 loop (no C extension calls)

---

#### Section B: base45_decode() ⚠️ **Same Issue as encode**

| Lines | Operation | GIL Status | Notes |
|-------|-----------|-----------|-------|
| 128-166 | base45_decode() | **HOLDS ❌** | Similar pure Python loop |

---

#### Section C: pack_v3() & pack_v4() ✅ **Mostly GIL-Free**

| Lines | Operation | GIL Status | Duration |
|-------|-----------|-----------|----------|
| 344-391 | pack_v3() | MIXED | ~20-50 μs |
| 563-608 | pack_v4() | MIXED | ~20-50 μs |

**Components**:
```python
def pack_v4(...) -> bytes:                         # Line 563
    # Validation (lines 575-587) - Pure Python, brief, holds GIL
    flags = 0x00                                   # Line 589 - Holds GIL
    # ...
    header = struct.pack(                          # Line 596 - RELEASES GIL ✅
        '>BBQHIIHH',
        V4_VERSION,
        flags,
        filesize,
        symbol_size,
        symbol_count,
        esi,
        block_seq,
        reserved,
    )
    crc = zlib.crc32(header + data) & 0xFFFFFFFF   # Line 607 - RELEASES GIL ✅
    return header + data + struct.pack('>I', crc)  # Line 608 - RELEASES GIL ✅
```

**Breakdown**:
- Lines 589-594: Flags assembly (pure Python) → holds GIL <1 μs
- Line 596: `struct.pack()` → **RELEASES GIL ✅** ~1-2 μs
- Line 607: `zlib.crc32()` → **RELEASES GIL ✅** ~10-50 μs

---

### 4. qr_utils.py - QR Code Generation

**File**: `src/qrstream/qr_utils.py`

#### Section A: _encode_qr_payload() ⚠️ **Calls base45_encode()**

| Lines | Operation | GIL Status |
|-------|-----------|-----------|
| 71-89 | _encode_qr_payload() | DEPENDS |

**Code**:
```python
def _encode_qr_payload(data: bytes, ...) -> tuple[str, bool]:
    if use_alphanumeric:
        from .protocol import base45_encode            # Line 85
        payload = base45_encode(data).decode("ascii")  # Line 86 - **HOLDS GIL ❌**
    else:
        payload = _b64lib.b64encode(data).decode("ascii")  # Line 88 - RELEASES GIL ✅
    return payload, use_alphanumeric
```

**⚠️ When alphanumeric=True**: Calls base45_encode() → **HOLDS GIL**
**✅ When alphanumeric=False**: Calls base64.b64encode() → **RELEASES GIL**

---

#### Section B: _render_qr_gray() ✅ **All Native Code**

| Lines | Operation | GIL Status | Duration |
|-------|-----------|-----------|----------|
| 161-204 | _render_qr_gray() | RELEASED ✅ | ~5 ms |

**Code**:
```python
def _render_qr_gray(payload: str, ...) -> np.ndarray:
    # Lines 176-178: validation (brief, holds GIL <1 μs)
    kwargs: dict = {'ec_level': ec}
    
    bc = zxingcpp.create_barcode(              # Line 185 - **RELEASES GIL ✅**
        payload,
        zxingcpp.BarcodeFormat.QRCode,
        **kwargs,
    )
    
    zimg = bc.to_image(scale=bs, ...)          # Line 194 - **RELEASES GIL ✅**
    qr_arr = np.array(zimg, dtype=np.uint8)    # Line 195 - **RELEASES GIL ✅**
    
    img = np.full((side, side), 255, ...)      # Line 201 - **RELEASES GIL ✅**
    img[bd_px:..., bd_px:...] = qr_arr         # Line 202 - **RELEASES GIL ✅**
    
    return img
```

**✅ All operations are C++/numpy: GIL completely released**

---

### 5. display_cache.py - Frame Packing

**File**: `src/qrstream/display_cache.py`

#### pack_module_image() ✅ **Pure Numpy**

| Lines | Operation | GIL Status | Duration |
|-------|-----------|-----------|----------|
| 56-68 | pack_module_image() | RELEASED ✅ | ~0.5 ms |

**Code**:
```python
def pack_module_image(module_img: np.ndarray) -> np.ndarray:  # Line 56
    arr = np.asarray(module_img)               # Line 61 - RELEASES GIL ✅
    black = arr == 0                           # Line 66 - RELEASES GIL ✅ (numpy ufunc)
    packed = np.packbits(black, ...)           # Line 67 - RELEASES GIL ✅ (numpy C ext)
    return np.ascontiguousarray(packed, ...)   # Line 68 - RELEASES GIL ✅
```

**✅ All operations are numpy ufuncs: GIL completely released**

---

#### unpack_module_frame() ✅ **Pure Numpy**

| Lines | Operation | GIL Status |
|-------|-----------|-----------|
| 71-80 | unpack_module_frame() | RELEASED ✅ |

**Code**:
```python
def unpack_module_frame(packed: np.ndarray, module_side: int) -> np.ndarray:  # Line 71
    arr = np.asarray(packed, ...)              # Line 73 - RELEASES GIL ✅
    bits = np.unpackbits(arr, ...)             # Line 79 - RELEASES GIL ✅
    return np.where(bits, 0, 255)...           # Line 80 - RELEASES GIL ✅
```

**✅ All numpy operations: GIL completely released**

---

### 6. encoder.py - Producer Pipeline Orchestration

**File**: `src/qrstream/encoder.py`

#### DisplayProducer.encode_display() - Main Producer Loop

| Lines | Operation | Sub-components |
|-------|-----------|-----------------|
| ~968-1055 | Main encoding loop | See sub-sections below |

**Sub-section 1**: Generating blocks (line ~1009)
```python
block_iter = encoder.generate_blocks(num_blocks)  # Line ~1009
```
→ Calls `RaptorQEncoder.generate_blocks()` → **RELEASES GIL ✅**

**Sub-section 2**: Generating QR (line ~1023 or ~1045)
```python
module_img = generate_qr_module_image(          # Line ~1045
    packed_frame,
    ec_level=self.ec_level,
    version=self.qr_version,
    alphanumeric=self.alphanumeric,
)
```
→ Calls `qr_utils.generate_qr_module_image()` → **RELEASES GIL ✅** (except base45 in payload encoding)

**Sub-section 3**: Packing frame (line ~1032 or ~1055)
```python
packed_frame = pack_module_image(module_img)    # Line ~1032 or ~1055
```
→ Calls `display_cache.pack_module_image()` → **RELEASES GIL ✅**

---

## Summary: GIL Contention Points

### ❌ GIL HELD (Contention Risk)
1. **base45_encode()** - `protocol.py:100-125` (200-500 μs)
2. **base45_decode()** - `protocol.py:128-166` (similar)
3. **PRNG.get_src_blocks()** - `lt_codec.py:149-176` (1 μs, negligible)
4. **pack_v4() validation** - `protocol.py:575-587` (<1 μs)

### ✅ GIL RELEASED (Safe)
1. **_raptorq encoder** - `raptorq_codec.py:277` (Rust)
2. **zxingcpp QR gen** - `qr_utils.py:185-194` (C++)
3. **numpy operations** - `display_cache.py:56-80` (C)
4. **struct.pack()** - `protocol.py:596` (C)
5. **zlib.crc32()** - `protocol.py:607` (C)
6. **base64.b64encode()** - `qr_utils.py:88` (C)

---

## Verification Commands

### Check if base45 is actually a bottleneck
```bash
cd /Users/ddddavid/workspace/qrstream-enhanced
python -c "
import time
from src.qrstream.protocol import base45_encode
import numpy as np

data = np.random.bytes(300)
start = time.perf_counter()
for _ in range(1000):
    base45_encode(data)
elapsed = time.perf_counter() - start
print(f'1000 iterations: {elapsed*1000:.1f} ms ({elapsed:.1f} μs per call)')
"
```

Expected output: ~300-500 μs per call

### Profile the producer pipeline
```bash
python -m py_spy record -o profile.svg --gil -- qrs encode test.bin --output output.mp4
```

Then inspect `profile.svg` to see which functions hold the GIL.

---

## Code Modification Guide

### To optimize base45_encode():

**Option 1: Use Cython** (Recommended for quick fix)
```cython
# protocol.pyx
def base45_encode(data: bytes) -> bytes:
    # ... same logic, but Cython compiles to C ...
```

**Option 2: Use C extension** (Recommended for production)
```c
// protocol_c.c
PyObject* base45_encode(PyObject* self, PyObject* args) {
    // ... C implementation ...
    // Releases GIL with: Py_BEGIN_ALLOW_THREADS / Py_END_ALLOW_THREADS
}
```

**Option 3: Use PyO3 bindings** (If rewriting in Rust)
```rust
// In a .rs file compiled with PyO3
#[pyfunction]
pub fn base45_encode(data: &[u8]) -> PyResult<Vec<u8>> {
    // Rust implementation (auto-releases GIL)
}
```

