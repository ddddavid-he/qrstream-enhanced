# QRStream Protocol & Codec Analysis Report
Generated: 2026-05-15

## EXECUTIVE SUMMARY

The QRStream V3 protocol is **explicitly LT-codec-aware**. While the wire format (header + data + CRC) is codec-agnostic, the protocol embeds LT-specific semantics through the `seed` field and PRNG version flag. To support alternative codecs like RaptorQ, you need:

1. **Add codec_type to header flags** (use bits 0x08-0x0F, currently reserved)
2. **Create codec abstraction interface** (BlockCodec ABC with encode/decode methods)
3. **Refactor encoder/decoder to use codec factory** (swap implementations at runtime)
4. **Keep wire format unchanged** (backward compatibility maintained)

---

## 1. V3 BLOCK FORMAT (DETAILED SPECIFICATION)

### Layout Overview
```
Total block size = 24-byte header + data + 4-byte trailing CRC32
Total overhead = 28 bytes (V3_BLOCK_OVERHEAD)
```

### Header Structure (24 bytes total)
Uses big-endian (`>`) struct format: `>BBQHIIHH`

| Offset | Name | Type | Size | Purpose |
|--------|------|------|------|---------|
| 0 | version | uint8 | 1 B | Protocol version (0x03 for V3) |
| 1 | flags | uint8 | 1 B | Feature flags (see below) |
| 2-9 | filesize | uint64_be | 8 B | Total uncompressed payload size (0 to 2^64-1) |
| 10-11 | blocksize | uint16_be | 2 B | Size of each block in bytes (1 to 65535) |
| 12-15 | block_count | uint32_be | 4 B | Number of source blocks K = ceil(filesize/blocksize) |
| 16-19 | seed | uint32_be | 4 B | PRNG seed for this encoded block (determines degree/src blocks) |
| 20-21 | block_seq | uint16_be | 2 B | Sequence number within encoded block stream (0-65535) |
| 22-23 | reserved | uint16_be | 2 B | Reserved for future use (currently 0) |

**Total header: 24 bytes**

### Flags Byte (offset 1)
```
Bit 0x01: compression flag
  - 0 = raw payload
  - 1 = payload was zlib-compressed before splitting into blocks

Bit 0x02: high-density QR encoding flag
  - 0 = base64 encoding for QR payload (byte mode)
  - 1 = base45 alphanumeric encoding (higher density)
  - LEGACY: older videos used COBS+latin-1; decoder tries both
  - NOTE: This flag only affects QR rendering, NOT LT/CRC layout

Bit 0x04: PRNG version flag
  - 0 = legacy LCG with 5 warmup rounds (qrstream ≤ 0.7)
  - 1 = SplitMix64 seed-mixer (qrstream ≥ 0.8, recommended)
  - Affects how seed maps to PRNG state and degree/src_blocks

Bits 0x08-0xFF: reserved (must be 0)
  - Available for future codec selection or feature flags
```

### Data Section
- **Size**: variable (0 to blocksize bytes)
- **Content**: Encoded fountain block payload
- **Structure**: XOR of source block(s) selected by seed-to-indices mapping

### Trailing CRC32 Section
- **Size**: 4 bytes (big-endian uint32)
- **Algorithm**: zlib.crc32(header + data) & 0xFFFFFFFF
- **Validation**: Decoder verifies before unpacking; corrupted blocks are silently discarded

---

## 2. SEED / BLOCK_COUNT / BLOCKSIZE PACKING

### Derivation Pipeline
```python
# Encoder side (LTEncoder.__init__)
filesize = len(payload)                    # Total bytes to encode
blocksize = auto_blocksize(filesize)       # Choose optimal block size
K = ceil(filesize / blocksize)            # Number of source blocks
block_count = K                           # Packed into header

# Per-block generation (LTEncoder.generate_blocks)
for i in range(num_encoded_blocks):
    seed = i + 1                          # Sequential: 1, 2, 3, ...
    _, _, src_blocks = PRNG.get_src_blocks(seed=seed)
    block_data = XOR of self._get_block(idx) for idx in src_blocks
    pack_v3(..., seed=seed, block_count=K, blocksize=blocksize, ...)
```

### PRNG Seed → Degree/Src_blocks Mapping

**PRNG_VERSION=0 (legacy, ≤0.7):**
```
1. state = seed
2. For _ in range(5):  # PRNG_WARMUP_ROUNDS
     state = (16807 * state) % (2^31 - 1)
3. Decode degree d from RSD CDF bin
4. Generate d unique src block indices via LCG sampling
```

⚠️ **Problem**: Sequential seeds (1,2,3...) produce correlated outputs
   - All small seeds map to low-degree CDF buckets
   - Peeling graph lacks degree-1 check nodes early
   - Stalls at K=1827 with overhead=1.5× (user-reported failure)

**PRNG_VERSION=1 (modern, ≥0.8, recommended):**
```
1. state = splitmix64_mix(seed)  # Non-linear SplitMix64 avalanche
2. No warmup needed
3. Decode degree d from RSD CDF bin
4. Generate d unique src block indices via LCG sampling
```

✓ **Benefit**: Non-linear mix decorrelates sequential seeds
   - 1,2,3 now map to diverse degree buckets
   - Healthy peeling graph with diverse degrees across full range
   - No stalls even at extreme K values

### Blocksize Selection
```python
def auto_blocksize(filesize: int, ec_level: int = 1, qr_version: int = 25,
                   alphanumeric_qr: bool = True) -> int:
    # Choose blocksize to maximize per-QR frame capacity
    # Base45 alphanumeric: ~646 B usable per QR @ V20/M
    # Base64 byte mode: ~499 B usable per QR @ V20/M
    # Accounts for 28-byte protocol overhead
    # Ensures last block of last frame fits within QR capacity
    max_blocksize = capacity - V3_BLOCK_OVERHEAD - 1  # 1B safety margin
    return min(max_blocksize, filesize)
```

---

## 3. ENCODE_BLOCK() AND DECODE_BLOCK() SIGNATURES

### encode_block() - LTEncoder.generate_block(seed)
```python
def generate_block(self, seed: int) -> tuple[bytes, int]:
    """Generate one encoded block for a given PRNG seed.
    
    Returns: (block_data, seq)
      block_data: XOR of selected source blocks (bytes)
      seq: sequence number within encoded stream (uint16)
    
    Implementation:
      1. PRNG.get_src_blocks(seed) → (blockseed, degree, src_indices)
      2. Retrieve source blocks by index
      3. XOR all selected blocks together (vectorized via numpy)
      4. Return packed bytes and sequence number
    """
    _, _, src_blocks = self.prng.get_src_blocks(seed=seed)
    
    if len(src_blocks) == 1:
        result = self._get_block(next(iter(src_blocks)))
    elif len(src_blocks) == 2:
        result = xor_bytes(self._get_block(a), self._get_block(b))
    else:
        # Vectorized XOR via numpy for 3+ blocks
        blocks_array = np.empty((len(src_blocks), self.blocksize),
                                dtype=np.uint8)
        for i, idx in enumerate(src_blocks):
            blocks_array[i] = np.frombuffer(self._get_block(idx),
                                            dtype=np.uint8)
        result = bytes(np.bitwise_xor.reduce(blocks_array, axis=0))
    
    seq = self._seq & 0xFFFF
    self._seq += 1
    return result, seq
```

### decode_block() - LTDecoder.consume_block(header, data)
```python
def consume_block(self, header: V3Header, data: bytes) -> tuple[bool, bool]:
    """Feed a parsed block (header + data bytes) into the decoder.
    
    Args:
      header: Unpacked V3Header with all fields
      data: Block payload bytes (0 to blocksize)
    
    Returns: (done, compressed)
      done: True if all K source blocks recovered
      compressed: compression flag from header
    
    Process:
      1. Initialize block_graph if first block
      2. Validate header consistency (filesize, blocksize, K, prng_version)
      3. Reconstruct src_blocks via PRNG(seed, prng_version)
      4. Add to BipartiteGraph for belief-propagation decoding
      5. Return done status
      
    The PRNG must be EXACTLY the same version as encoder (same prng_version)
    or the seed→src_blocks mapping will differ and decoding fails.
    """
    filesize = header.filesize
    blocksize = header.blocksize
    block_count = header.block_count
    seed = header.seed
    compressed = header.compressed
    
    # Initialize on first block
    if not self.initialized:
        self.protocol_version = header.version
        self.prng_version = header.prng_version
        self.filesize = filesize
        self.blocksize = blocksize
        self.K = block_count
        self.compressed = compressed
        self.block_graph = BlockGraph(self.K)
        self.prng = PRNG(self.K, delta=self.delta, c=self.c,
                         prng_version=self.prng_version)
        self.initialized = True
    else:
        # Validate consistency across all blocks
        if header.version != self.protocol_version:
            raise ValueError(f"version mismatch: {header.version}")
        if header.prng_version != self.prng_version:
            raise ValueError(f"prng_version mismatch: {header.prng_version}")
        # ... other validations
    
    # Reconstruct src_blocks from seed using SAME PRNG VERSION
    _, _, src_blocks = self.prng.get_src_blocks(seed=seed)
    
    # Normalize data length
    if len(data) < self.blocksize:
        data = data + b'\x00' * (self.blocksize - len(data))
    elif len(data) > self.blocksize:
        data = data[:self.blocksize]
    
    # Feed into belief-propagation graph
    self.done = self.block_graph.add_block(src_blocks, data)
    return self.done, self.compressed
```

---

## 4. LT-SPECIFIC ASSUMPTIONS IN THE PROTOCOL

### Protocol DOES Make LT Assumptions:
1. **Seed determinism**: seed → (degree, src_blocks) must be deterministic
   - PRNG generates reproducible sequences from seed
   - Decoder can replay the exact same block selection
   - Mismatch in prng_version → hard decode failure
   
2. **PRNG version encoded in flags**: Decoder must know which seed→state mapping to use
   - Flag bit 0x04 toggles between LCG-warmup (v0) and SplitMix64 (v1)
   - Blocks must be consistent within a stream (mixing versions is an error)
   
3. **Sequential block encoding**: Encoder produces seed=1, seed=2, seed=3, ...
   - LTEncoder.generate_blocks loop: `seed = i + 1`
   - Not required by protocol, but baked into current encoder implementation
   
4. **XOR combination operator**: All encoded blocks are XOR of source blocks
   - encode_block() XORs src blocks
   - decode_block via BlockGraph.add_block() uses XOR to combine
   - XOR property is fundamental to LT decoding

### Protocol DOESN'T Assume:
1. ✓ Block graph structure (bipartite, belief-propagation)
   - Protocol just specifies src_block indices; how they're combined is codec-specific
   
2. ✓ Degree distribution (Robust Soliton with specific C/delta)
   - PRNG_VERSION selects seed-to-state mapping, but CDF parameters aren't protocol-mandated
   
3. ✓ Specific failure recovery (Gaussian elimination is optional)
   - GE is LT-specific; RaptorQ uses different recovery
   
4. ✓ Specific QR encoding (base45/base64/COBS are flags, not protocol essence)
   - These are QR rendering choices, not fountain-code essentials

---

## 5. CODEC ABSTRACTION REQUIREMENTS

### What MUST Change to Support Different Codecs (e.g., RaptorQ)

**Protocol layer (protocol.py):**
1. Add codec_type field to header
   - Option A: Use bits 0x08-0x0F in flags byte for codec selector (0=LT, 1=RaptorQ)
   - Option B: Replace unused reserved byte with codec_type byte
   - Recommendation: Use reserved byte (simpler, no bit-packing)
   
2. Make seed→src_blocks mapping codec-agnostic
   - Current: `_, _, src_blocks = PRNG.get_src_blocks(seed)`
   - Future: `src_blocks = codec.get_src_blocks_from_seed(seed, K, prng_version)`
   - Each codec interprets seed/block_count/blocksize differently

**Encoder/Decoder layer (encoder.py, decoder.py):**
1. Replace hardcoded LTEncoder/LTDecoder with generic Encoder/Decoder
2. Use codec factory based on codec_type from header
3. Call codec-specific encode/decode methods

**Codec implementation layer (new codecs/*.py):**
1. Each codec implements BlockCodec interface
2. LT codec moves from monolithic to pluggable
3. RaptorQ implementation adds new codec module

### What MUST Stay in Protocol (Backward Compatible)

1. ✓ File layout (header + data + CRC)
2. ✓ Blocksize/block_count/filesize encoding (all codecs need to split files)
3. ✓ Seed field (meaning changes per codec, but field presence stays)
4. ✓ Flags for compression and QR encoding (orthogonal to codec choice)
5. ✓ CRC32 validation (works for any codec)

### What Moves to Codec-Specific Implementation

1. `encode_block(seed)` → codec-specific implementation
2. `decode_block(seed, data)` → codec-specific add_block logic
3. PRNG variant → RaptorQ doesn't need PRNG, uses different seed interpretation
4. Degree distribution → RaptorQ uses systematic code properties
5. Recovery strategy → RaptorQ has built-in recovery, no need for GE fallback

---

## 6. CURRENT CODEC COUPLING: Zero Abstraction Layer

### State of the Codebase
- **LTEncoder and LTDecoder are hardcoded to LT codec**
- **No strategy pattern or ABC base class**
- PRNG imported directly: `from .lt_codec import PRNG, BlockGraph, ...`
- pack_v3/unpack_v3 don't know about codecs; they only handle protocol layer

### Current Tightly Coupled Flow:
```
encoder.py:LTEncoder
    ↓ imports
lt_codec.py:PRNG
    ↓ uses
protocol.py:pack_v3(seed=..., block_count=...)

decoder.py:LTDecoder
    ↓ imports
lt_codec.py:PRNG, BlockGraph
    ↓ uses
protocol.py:unpack_v3 → V3Header(seed=..., block_count=...)
```

### To Support Multiple Codecs, Would Need:

```
BlockCodec (abstract base)
├── LTCodec(BlockCodec)
│   ├── PRNG
│   ├── BlockGraph
│   └── Gaussian elimination fallback
└── RaptorQCodec(BlockCodec)
    ├── RaptorQ state machine
    └── Systematic code properties

encoder.py:Encoder (generic)
    ↓ uses factory
CodecFactory.create(codec_type)
    ↓ returns
BlockCodec instance

protocol.py:
    - Add codec_type field to header
    - pack_v3(codec_type, ...) 
    - unpack_v3(...) → extract codec_type
    - Seed interpretation varies per codec
```

---

## 7. TEST PATTERNS & COVERAGE

### test_roundtrip.py
- **Purpose**: Pure LT fountain-code roundtrips without video
- **Pattern**: 
  ```python
  encoder = LTEncoder(data, blocksize)
  decoder = LTDecoder()
  for packed, seed, seq in encoder.generate_blocks(num_blocks):
      done, _ = decoder.decode_bytes(packed)
      if done: break
  result = decoder.bytes_dump()
  assert result == data
  ```
- **Coverage**: 
  - Small/large/random data
  - Exact block boundaries
  - Compression/no compression
  - Progress tracking
  - Protocol version consistency
- **Gap**: No codec selection tests (no abstraction)

### test_lt_codec.py
- **Purpose**: Primitive unit tests (PRNG, BlockGraph, xor_bytes)
- **Pattern**:
  ```python
  prng = PRNG(K=50)
  for seed in range(1, 100):
      _, d, blocks = prng.get_src_blocks(seed=seed)
      assert 1 <= d <= 50
      assert len(blocks) == d
  ```
- **Coverage**: 
  - Determinism with same seed
  - Different outputs for different seeds
  - Degree in valid range [1, K]
  - Source block indices in range [0, K)
  - BlockGraph belief-propagation
  - XOR byte operations
  - Gauss-Jordan bit-packing
- **Gap**: No codec abstraction tests

### test_e2e_encode_decode.py (@pytest.mark.e2e)
- **Purpose**: Full encode→video→decode pipeline
- **Pattern**:
  ```python
  encode_to_video(input_file, output.mp4)
  blocks = extract_qr_from_video(output.mp4)
  decoded = decode_blocks_to_file(blocks, output.bin)
  assert SHA256(decoded) == SHA256(input_file)
  ```
- **Coverage**: 
  - 10KB/100KB/500KB files
  - QR version sweep (v10/20/30/40)
  - Regression: glog(0) crash condition (blocksize=938, K=19)
  - SHA256 integrity
  - Path handling
- **Gap**: No multi-codec E2E tests (none possible without abstraction)

### test_gaussian_rescue.py
- **Purpose**: Gauss-Jordan fallback when peeling stalls
- **Pattern**:
  ```python
  enc = LTEncoder(..., prng_version=0)  # Force stalling config
  dec = LTDecoder()
  for packed, _, _ in enc.generate_blocks(K * 1.5):
      dec.decode_bytes(packed)
  
  assert dec.num_recovered < K  # Peeling stalled
  assert dec.try_gaussian_rescue()  # GE finishes it
  assert dec.bytes_dump() == payload
  ```
- **Coverage**: 
  - GE rescue on stalled peeling (v0 configs)
  - GE is no-op when peeling converged
  - GE rejects insufficient-info cases
  - High-level decode_blocks auto-triggers GE
- **Gap**: No codec-specific rescue strategies

---

## 8. DEPENDENCY STATUS

### Current pyproject.toml
```
Core:
- numpy>=2.0.0             (vectorized XOR, BitPacked GE matrices)
- opencv-python-headless   (QR frame extraction)
- zxing-cpp>=3.0.0         (QR decoding)
- av>=17.0.0               (video I/O)
- PySide6-Essentials       (GUI)
- rich>=13.7.0             (CLI reporting)

No RaptorQ dependency currently.
```

### If Adding RaptorQ:
```
Candidate libraries:
- raptorq (pure Python, slow)
- raptorq-rust (FFI via ctypes, fast) — NOT YET PUBLISHED as py pkg
- raptorq-python (unmaintained)
- Custom minimal implementation (recommended for this project)

Recommendation: Implement custom minimal RaptorQ for:
  1. Educational value
  2. No external dependency risk
  3. Can be tuned for QRStream use case
  4. Easier to maintain
```

---

## VISUAL SUMMARY

```
┌─────────────────────────────────────────────────────────┐
│  V3 Block: Header(24B) + Data(0-N) + CRC(4B)           │
│                                                         │
│  Seed = i+1 (encoder loop i=0→∞)                       │
│    ↓ PRNG(seed, K, prng_version)                        │
│    ↓ → (degree, src_indices)                            │
│    ↓ → XOR of blocks[idx] for idx in src_indices       │
│    ↓ → pack_v3(...) → wire bytes                        │
│                                                         │
│  On decoder: unpack → seed → PRNG (same version!)      │
│    ↓ → (degree, src_indices) must match encoder        │
│    ↓ → add to BlockGraph → belief-prop / GE            │
│    ↓ → recover source blocks                            │
└─────────────────────────────────────────────────────────┘

LT-SPECIFIC parts:
  ✓ seed → (degree, indices) mapping via PRNG
  ✓ XOR as combination operator
  ✓ Belief-propagation + GE recovery

CODEC-AGNOSTIC parts:
  ✓ Block layout (24+data+4 bytes)
  ✓ CRC32 validation
  ✓ filesize/blocksize/block_count encoding
  ✓ Compression flag
  ✓ QR encoding flag
```

---

## RECOMMENDATIONS

### To Support RaptorQ or Other Codecs:

1. **Minimal Protocol Change**: Add codec_type to flags (bits 0x08-0x0F)
2. **Create BlockCodec ABC**: Define encode/decode interface
3. **Refactor encoder/decoder**: Use factory pattern to instantiate codec
4. **Implement LTCodec wrapper**: Move current LT code into codec module
5. **Add RaptorQCodec**: Plug in new codec with different seed interpretation
6. **Extend test suite**: Add codec selection and alternate-codec roundtrip tests

### Breaking Change Policy:

- **Backward compatible**: Old V3 blocks (codec_type=0) still decode with LT codec
- **Future-proof**: codec_type in header allows graceful codec negotiation
- **No wire-format change**: Just reinterpret seed field per codec

---

## CONCLUSION

The V3 protocol is **explicitly coupled to LT semantics** but **not structurally dependent** on them. The coupling is through the PRNG version flag and seed interpretation, not the wire format. To support multiple codecs, you need to:

1. Extend protocol.py with codec_type metadata
2. Extract codec logic into pluggable modules
3. Create a simple codec interface (3-4 methods)
4. Keep the wire format unchanged

The existing test suite provides a solid foundation; you'll just need to add codec-selection and alternate-codec roundtrip tests.
