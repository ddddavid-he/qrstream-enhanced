# Rust V4 Core PoC Discovery — 2026-05-30

## Context

QRStream's long-term mobile path is a platform-native scanner feeding QR payloads into a shared core decoder. LT/V3 is legacy and remains Python-only. Rust core exploration focuses on V4/RaptorQ and session/protocol boundaries.

This discovery was performed on branch `explore/rust-v4-core-poc`.

## Prototype

Added a PyO3 crate under `rust/qrstream-rs` exposing:

```python
qrstream_rs.V4DecodeSession
```

Implemented V4-only core responsibilities:

- base45/base64 QR payload decode
- V4 header parse
- CRC32 validation
- duplicate ESI tracking
- RaptorQ packet reconstruction and decode via Rust `raptorq` crate
- compressed payload zlib decompression
- progress/snapshot/result APIs
- per-block and batch block consumption

Python `DecodeSession` gained an experimental flag:

```python
DecodeSession(use_rust_v4=True)
```

Default behavior remains unchanged. V3/LT stays on Python.

## Functional validation

After release-build installing the Rust extension:

```bash
uvx maturin develop --release --manifest-path rust/qrstream-rs/Cargo.toml
uv run pytest tests/test_rust_v4_core_poc.py tests/test_decode_session.py tests/test_raptorq_codec.py tests/test_raptorq_roundtrip.py tests/test_raptorq_protocol.py
```

Result:

```text
104 passed
```

## Performance results

### Synthetic V4/RaptorQ blocks, release build

| Payload | Blocks | Python DecodeSession | Python `use_rust_v4` | Rust direct | Rust batch |
|---:|---:|---:|---:|---:|---:|
| 64 KiB | 76 | 0.198 ms | 0.122 ms (1.62x) | 0.022 ms (8.82x) | 0.023 ms (8.74x) |
| 1 MiB | 1228 | 3.073 ms | 2.042 ms (1.50x) | 0.392 ms (7.83x) | 0.387 ms (7.95x) |
| 5 MiB | 6144 | 15.679 ms | 10.455 ms (1.50x) | 2.107 ms (7.44x) | 2.259 ms (6.94x) |

### Real-world phone RaptorQ fixture

Fixture:

```text
tests/fixtures/real-phone-v092/v092-raptorq-pi-1MB.mp4
```

Extraction produced:

```text
455 unique blocks, V4, filesize=1,000,000, K=451
```

Decode-only benchmark:

| Path | Median | Speedup |
|---|---:|---:|
| Python DecodeSession | 2.531 ms | 1.00x |
| Python `use_rust_v4` | 2.030 ms | 1.25x |
| Rust direct | 1.285 ms | 1.97x |
| Rust batch | 1.248 ms | 2.03x |

All paths reproduced the expected SHA-256:

```text
7806ee47461b49ef1f578e14461b2c83c09c6d7a9a914275da1d71e9cbbf7069
```

## Findings

1. Rust core is viable for V4/RaptorQ.
2. Dev builds are misleading; release build is required for meaningful results.
3. Avoiding Python-side pre-parse/CRC matters. A V4 fast-path improved `DecodeSession(use_rust_v4=True)` from slower-than-Python to about 1.25–1.5x faster.
4. Direct Rust session is much faster than the Python wrapper for synthetic data and about 2x faster on the real-world RaptorQ fixture.
5. Batch API did not materially beat Rust direct in these tests. The main wins are release build and reducing Python wrapper work, not batching alone.
6. The decode-only cost is already very small compared with real-world QR/video extraction; the mobile value is more about shared core and lower latency ceiling than desktop CLI speed.

## Recommendation

Keep this as an exploration branch for now. Do not enable Rust backend by default in the Python package yet.

Next steps if continuing:

1. Stabilize a Rust-first API shape matching future mobile needs:
   - `consume_qr_text`
   - `consume_block`
   - `snapshot`
   - `result_bytes`
2. Add a thin compatibility layer for Python tests, but avoid designing around PyO3 performance.
3. Explore UniFFI after the Rust API stabilizes.
4. Add optional Rust tests gated by extension availability rather than making CI depend on Rust packaging immediately.
5. Do not port LT/V3 to Rust.
