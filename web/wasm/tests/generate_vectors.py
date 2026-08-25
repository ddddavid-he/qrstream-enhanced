"""Generate V4 RaptorQ test vectors for the Rust WASM port.

Uses the same pipeline as `encode_to_video` (auto_blocksize +
RaptorQEncoder) to produce raw V4 frames, then wraps them as base45 and
base64 QR payload strings plus expected outputs as JSON.
"""

import base64
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qrstream.encoder import _DEFAULT_QR_EC_LEVEL  # noqa: E402
from qrstream.protocol import auto_blocksize, base45_encode  # noqa: E402
from qrstream.raptorq_codec import RaptorQEncoder  # noqa: E402

QR_VERSION = 30


def make_case(name: str, data: bytes, *, compressed: bool = False, overhead: float = 2.0) -> dict:
    payload = zlib.compress(data) if compressed else data
    blocksize = auto_blocksize(
        len(payload), _DEFAULT_QR_EC_LEVEL, QR_VERSION, alphanumeric_qr=True)
    enc = RaptorQEncoder(payload, blocksize,
                         compressed=compressed, alphanumeric_qr=True)
    from math import ceil
    count = ceil(enc.K * overhead) + 4
    frames = [packed for packed, _, _ in enc.generate_blocks(count)]
    return {
        "name": name,
        "compressed": compressed,
        "data_b64": base64.b64encode(data).decode("ascii"),
        "payload_b64": base64.b64encode(payload).decode("ascii"),
        "filesize": len(data),
        "blocksize": blocksize,
        "K": enc.K,
        "num_frames": len(frames),
        "frames": [base64.b64encode(f).decode("ascii") for f in frames],
        "qr_texts_base45": [base45_encode(f).decode("ascii") for f in frames],
        "qr_texts_base64": [base64.b64encode(f).decode("ascii") for f in frames],
    }


def main() -> None:
    rng = random.Random(42)
    cases = [
        # tiny payload (single symbol)
        make_case("tiny", b"Hello, QRStream!", overhead=1.0),
        # small random payload (~1 KB)
        make_case("small_1k", bytes(rng.randrange(256) for _ in range(1024)), overhead=1.5),
        # medium random payload (~50 KB, multiple symbols)
        make_case("medium_50k", bytes(rng.randrange(256) for _ in range(50_000)), overhead=1.2),
        # compressed text payload
        make_case("compressed_text", b"The quick brown fox jumps over the lazy dog. " * 400,
                  compressed=True, overhead=1.5),
        # compressible zeros payload
        make_case("compressed_zeros", bytes(20_000), compressed=True, overhead=1.5),
    ]

    out_path = os.path.join(os.path.dirname(__file__), "vectors.json")
    with open(out_path, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"Wrote {len(cases)} cases, {os.path.getsize(out_path)} bytes -> {out_path}")
    for c in cases:
        print(f"  {c['name']}: filesize={c['filesize']} K={c['K']} "
              f"blocksize={c['blocksize']} frames={c['num_frames']}")


if __name__ == "__main__":
    main()
