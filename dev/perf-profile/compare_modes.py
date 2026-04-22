"""
End-to-end comparison of QR encoding modes.

Modes compared:
  1. base64              — current default when binary_qr=False
                           (33% inflation, pure ASCII, WeChatQR works,
                           forces QR byte mode)
  2. binary_str          — current default when binary_qr=True
                           (cobs -> latin-1 str -> qrcode -> UTF-8 bloat
                           -> QR version auto-upgraded by qrcode lib)
  3. binary_bytes_v20    — cobs -> qrcode.add_data(bytes) -> QR stays at
                           requested version. Requires QRCodeDetector
                           (WeChatQR can't handle non-UTF-8 bytes).
                           [Optional, only if user wants to see the
                            "true binary" numbers.]
  4. ascii85             — 4 bytes -> 5 ASCII chars (0x21..0x75).
                           25% inflation, pure ASCII, forces byte mode.
  5. base45              — 2 bytes -> 3 chars from the 45-char QR
                           alphanumeric set.  Nominal inflation 50% but
                           QR alphanumeric mode uses only 5.5 bits/char
                           (vs 8 for byte mode), so the *physical*
                           capacity at V20/M jumps from 499B (base64)
                           to 614B — even bigger than the 635B that
                           binary_str fakes by silently upgrading QR
                           version. WeChatQR decodes cleanly.

Metrics per file size:
  - video file bytes
  - encode wall time
  - decode wall time
  - QR version actually used
  - total frames
  - per-frame QR pixel area

Usage:
    python dev/perf-profile/compare_modes.py --size 100
    python dev/perf-profile/compare_modes.py --sizes 10,100,1024
    python dev/perf-profile/compare_modes.py --skip-binary-bytes
"""

import argparse
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from qrstream.encoder import (  # noqa: E402
    LTEncoder, _load_payload, _resolve_border_modules,
)
from qrstream.protocol import (  # noqa: E402
    V3_VERSION, auto_blocksize, cobs_encode, cobs_decode, unpack,
)
from qrstream.qr_utils import try_decode_qr, generate_qr_image  # noqa: E402
from qrstream.decoder import extract_qr_from_video, LTDecoder  # noqa: E402
from qrstream.lt_codec import PRNG, BlockGraph  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Encoding helpers — one QR-image function per mode
# ─────────────────────────────────────────────────────────────

import qrcode  # noqa: E402
from qrcode.constants import ERROR_CORRECT_M, ERROR_CORRECT_L  # noqa: E402


_EC_MAP = {0: 1, 1: 0, 2: 3, 3: 2}  # ec_level (0=L,1=M,2=Q,3=H) → qrcode consts


def _qrcode_to_bgr(qr: "qrcode.QRCode") -> np.ndarray:
    pil_img = qr.make_image(fill_color="black", back_color="white")
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def make_qr_base64(packed: bytes, ec_level: int, version: int,
                   box_size: int, border: float) -> tuple[np.ndarray, int]:
    """Mode 1: base64(packed) as ASCII string."""
    import base64 as b64lib
    b64 = b64lib.b64encode(packed)  # bytes → bytes (ASCII-safe)
    q = qrcode.QRCode(
        version=version,
        error_correction=_EC_MAP[ec_level],
        box_size=box_size, border=round(border),
    )
    q.add_data(b64)
    q.make(fit=True)
    return _qrcode_to_bgr(q), q.version


def make_qr_binary_str(packed: bytes, ec_level: int, version: int,
                       box_size: int, border: float) -> tuple[np.ndarray, int]:
    """Mode 2: current default — cobs(packed).decode('latin-1') as str."""
    cobs = cobs_encode(packed)
    payload = cobs.decode("latin-1")  # ← triggers UTF-8 bloat inside qrcode
    q = qrcode.QRCode(
        version=version,
        error_correction=_EC_MAP[ec_level],
        box_size=box_size, border=round(border),
    )
    q.add_data(payload)
    q.make(fit=True)
    return _qrcode_to_bgr(q), q.version


def make_qr_binary_bytes(packed: bytes, ec_level: int, version: int,
                          box_size: int, border: float) -> tuple[np.ndarray, int]:
    """Mode 3: cobs(packed) passed as bytes; QR stays at requested version."""
    cobs = cobs_encode(packed)
    q = qrcode.QRCode(
        version=version,
        error_correction=_EC_MAP[ec_level],
        box_size=box_size, border=round(border),
    )
    q.add_data(cobs)  # bytes → no UTF-8 bloat
    q.make(fit=True)
    return _qrcode_to_bgr(q), q.version


# ── Ascii85 encoding ───────────────────────────────────────────
# 4 raw bytes → 5 ASCII chars (range 0x21..0x75).
# Inflation: exactly 25% (base64 is 33.6%, so ~6% more usable payload
# per frame vs base64).
# Output is pure ASCII — WeChatQR decodes it fine, no UTF-8 bloat.
import base64 as _b64lib  # noqa: E402


def ascii85_encode(data: bytes) -> bytes:
    return _b64lib.a85encode(data)


def ascii85_decode(data: bytes) -> bytes:
    return _b64lib.a85decode(data)


def make_qr_ascii85(packed: bytes, ec_level: int, version: int,
                    box_size: int, border: float) -> tuple[np.ndarray, int]:
    """Mode 4: ascii85 → pure ASCII, WeChatQR-compatible, 25% inflation."""
    payload = ascii85_encode(packed)  # bytes, all in 0x21..0x75 ASCII
    q = qrcode.QRCode(
        version=version,
        error_correction=_EC_MAP[ec_level],
        box_size=box_size, border=round(border),
    )
    q.add_data(payload)
    q.make(fit=True)
    return _qrcode_to_bgr(q), q.version


# ── Base45 encoding (QR alphanumeric mode) ─────────────────────
# QR's alphanumeric mode packs every 2 chars into 11 bits (vs 8 bits
# per char for byte mode).  The 45-char alphabet is fixed by the QR
# spec: "0-9A-Z $%*+-./:".  By mapping every 2 raw bytes into 3 of
# these chars (base45 as per RFC 9285), we carry 16 bits of data in
# 3 * 5.5 = 16.5 bits of QR payload — essentially 1:1.  Compared with
# byte-mode base64 (8 bits of QR per 6 bits of data, i.e. 0.75
# efficiency), base45 is ~29% more data-dense.
_B45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
_B45_IDX = {c: i for i, c in enumerate(_B45)}
assert len(_B45) == 45


def base45_encode(data: bytes) -> bytes:
    """Encode bytes as a base45 ASCII string (RFC 9285).

    Returns bytes for a consistent API with ``base64.b64encode``.
    """
    out = bytearray()
    i = 0
    while i + 2 <= len(data):
        n = (data[i] << 8) | data[i + 1]
        c = n // (45 * 45)
        n -= c * 45 * 45
        b = n // 45
        a = n - b * 45
        out.append(ord(_B45[a]))
        out.append(ord(_B45[b]))
        out.append(ord(_B45[c]))
        i += 2
    if i < len(data):
        n = data[i]
        b = n // 45
        a = n - b * 45
        out.append(ord(_B45[a]))
        out.append(ord(_B45[b]))
    return bytes(out)


def base45_decode(data: bytes) -> bytes:
    """Inverse of ``base45_encode``. Accepts bytes or str."""
    if isinstance(data, bytes):
        s = data.decode("ascii")
    else:
        s = data
    out = bytearray()
    i = 0
    while i + 3 <= len(s):
        a = _B45_IDX[s[i]]
        b = _B45_IDX[s[i + 1]]
        c = _B45_IDX[s[i + 2]]
        n = a + b * 45 + c * 45 * 45
        if n > 0xFFFF:
            raise ValueError("invalid base45 triplet")
        out.append((n >> 8) & 0xFF)
        out.append(n & 0xFF)
        i += 3
    if i + 2 == len(s):
        a = _B45_IDX[s[i]]
        b = _B45_IDX[s[i + 1]]
        n = a + b * 45
        if n > 0xFF:
            raise ValueError("invalid base45 tail")
        out.append(n)
    elif i != len(s):
        raise ValueError("invalid base45 length")
    return bytes(out)


def make_qr_base45(packed: bytes, ec_level: int, version: int,
                   box_size: int, border: float) -> tuple[np.ndarray, int]:
    """Mode 5: base45 → QR alphanumeric mode → ~29% more payload than base64."""
    payload = base45_encode(packed)
    # Passing as str so qrcode picks alphanumeric mode (mode=2).
    # Passing bytes would force byte mode and lose the density win.
    payload_str = payload.decode("ascii")
    q = qrcode.QRCode(
        version=version,
        error_correction=_EC_MAP[ec_level],
        box_size=box_size, border=round(border),
    )
    q.add_data(payload_str)
    q.make(fit=True)
    return _qrcode_to_bgr(q), q.version


# ─────────────────────────────────────────────────────────────
# End-to-end encode → mp4 → decode for one mode
# ─────────────────────────────────────────────────────────────

def _choose_blocksize_for_mode(mode: str, payload_size: int,
                                ec_level: int, qr_version: int) -> int:
    """Pick a blocksize that the mode can realistically fit at the
    requested version.  For modes that accept ~the ISO table (base64
    and binary_str before inflation), we just use auto_blocksize.

    For binary_bytes we use auto_blocksize(binary_qr=True) — packed
    fits 1:1 in byte mode.

    For base128 we use auto_blocksize(binary_qr=False) with adjustment
    (base128 has 14% inflation vs base64's 33%, so we can afford more
    data per frame).
    """
    if mode in ("base64",):
        return auto_blocksize(payload_size, ec_level, qr_version,
                              binary_qr=False, protocol_version=V3_VERSION)
    if mode in ("binary_str", "binary_bytes"):
        # binary_qr=True gives the aggressive blocksize that the ISO
        # table allows for raw bytes.  binary_str will then overflow
        # and get upgraded by qrcode; binary_bytes stays put.
        return auto_blocksize(payload_size, ec_level, qr_version,
                              binary_qr=True, protocol_version=V3_VERSION)
    if mode == "ascii85":
        # Solve: ceil(N/4)*5 <= qr_capacity  →  N <= qr_capacity * 4/5
        from qrstream.protocol import _QR_CAPACITY, _QR_CAPACITY_V40
        qr_capacity = _QR_CAPACITY.get(
            (qr_version, ec_level),
            _QR_CAPACITY_V40.get(ec_level, 2331),
        )
        max_usable = (qr_capacity * 4) // 5
        from qrstream.protocol import _block_overhead
        overhead = _block_overhead(V3_VERSION)
        max_blocksize = max(max_usable - overhead, 64)
        blocksize = max(min(max_blocksize, payload_size), 64)
        return blocksize
    if mode == "base45":
        # QR alphanumeric-mode capacity per ISO/IEC 18004, ECC-M only.
        # base45 produces ceil(N/2)*3 chars (odd tail uses 2 chars for
        # 1 byte), so max raw bytes = (cap // 3) * 2 + (1 if cap % 3 == 2 else 0).
        alnum_cap_M = {
            10: 513,  15: 815,  20: 970,  25: 1125, 30: 1351,
            35: 1637, 40: 1852,
        }
        # Nearest version at or above qr_version.
        alnum_cap = alnum_cap_M.get(qr_version)
        if alnum_cap is None:
            # Fallback: derive ratio (alphanumeric cap ≈ byte cap × 1.46).
            from qrstream.protocol import _QR_CAPACITY, _QR_CAPACITY_V40
            byte_cap = _QR_CAPACITY.get(
                (qr_version, ec_level),
                _QR_CAPACITY_V40.get(ec_level, 2331),
            )
            alnum_cap = int(byte_cap * 1.46)
        triplets = alnum_cap // 3
        tail = 1 if alnum_cap % 3 == 2 else 0
        max_usable = triplets * 2 + tail
        from qrstream.protocol import _block_overhead
        overhead = _block_overhead(V3_VERSION)
        # Leave 1-byte margin; qrcode lib occasionally refuses inputs
        # exactly at the boundary for certain byte values.
        max_blocksize = max(max_usable - overhead - 1, 64)
        blocksize = max(min(max_blocksize, payload_size), 64)
        return blocksize
    raise ValueError(mode)


_MAKE_QR = {
    "base64": make_qr_base64,
    "binary_str": make_qr_binary_str,
    "binary_bytes": make_qr_binary_bytes,
    "ascii85": make_qr_ascii85,
    "base45": make_qr_base45,
}


def encode_video_for_mode(input_path: str, output_path: str,
                          mode: str, overhead: float = 2.0,
                          fps: int = 10, ec_level: int = 1,
                          qr_version: int = 20) -> dict:
    """Single-process, in-process encode.  Returns stats dict."""
    raw = open(input_path, "rb").read()
    payload_size = len(raw)

    blocksize = _choose_blocksize_for_mode(mode, payload_size, ec_level, qr_version)
    K = (payload_size + blocksize - 1) // blocksize
    num_blocks = int(K * overhead)

    encoder = LTEncoder(raw, blocksize, compressed=False,
                         binary_qr=(mode != "base64"),
                         protocol_version=V3_VERSION)

    border = _resolve_border_modules(qr_version, None)
    make_fn = _MAKE_QR[mode]

    # Probe the first frame to get dimensions and actual version used.
    encoder._seq = 0
    first_packed = next(encoder.generate_blocks(1))[0]
    first_img, actual_version = make_fn(first_packed, ec_level, qr_version,
                                        box_size=10, border=border)
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cannot open writer")

    t0 = time.perf_counter()
    encoder._seq = 0
    max_version = actual_version
    for packed, _, _ in encoder.generate_blocks(num_blocks):
        img, v = make_fn(packed, ec_level, qr_version,
                         box_size=10, border=border)
        if v > max_version:
            max_version = v
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        writer.write(img)
    writer.release()
    encode_time = time.perf_counter() - t0

    video_size = os.path.getsize(output_path)
    return {
        "mode": mode,
        "blocksize": blocksize,
        "K": K,
        "num_blocks": num_blocks,
        "requested_version": qr_version,
        "actual_version": max_version,
        "frame_size": (w, h),
        "frame_pixels": w * h,
        "encode_time": encode_time,
        "video_bytes": video_size,
        "encode_fps": num_blocks / encode_time if encode_time > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────
# Mode-aware decode
# ─────────────────────────────────────────────────────────────

def _decode_video_for_mode(video_path: str, mode: str) -> dict:
    """Decode a video using the appropriate QR detector + payload decoder.

    For modes whose QR payload survives WeChatQR's UTF-8 roundtrip we
    use WeChatQR (fast). For binary_bytes we must use QRCodeDetector.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if mode == "binary_bytes":
        detector = cv2.QRCodeDetector()
        def detect(frame):
            s, _, _ = detector.detectAndDecode(frame)
            return s if s else None
    else:
        w = cv2.wechat_qrcode_WeChatQRCode()
        def detect(frame):
            try:
                results, _ = w.detectAndDecode(frame)
                return results[0] if results else None
            except UnicodeDecodeError:
                return None

    def payload_to_block(qr_str: str) -> bytes | None:
        if qr_str is None:
            return None
        if mode == "base64":
            import base64 as b64lib
            try:
                return b64lib.b64decode(qr_str)
            except Exception:
                return None
        if mode == "binary_str":
            try:
                return cobs_decode(qr_str.encode("latin-1"))
            except Exception:
                return None
        if mode == "binary_bytes":
            try:
                return cobs_decode(qr_str.encode("latin-1"))
            except Exception:
                return None
        if mode == "ascii85":
            try:
                raw_bytes = qr_str.encode("latin-1")
                return ascii85_decode(raw_bytes)
            except Exception:
                return None
        if mode == "base45":
            try:
                return base45_decode(qr_str)
            except (KeyError, ValueError):
                return None
        return None

    # Read + decode + feed LT
    lt = LTDecoder()
    seen_seeds: set[int] = set()

    t0 = time.perf_counter()
    frame_idx = 0
    decoded_ok = 0
    detected = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        qr = detect(frame)
        if qr is not None:
            detected += 1
            block = payload_to_block(qr)
            if block is not None:
                try:
                    header, _ = unpack(block)
                    if header.seed not in seen_seeds:
                        seen_seeds.add(header.seed)
                        done, _ = lt.decode_bytes(block, skip_crc=True)
                        decoded_ok += 1
                        if done:
                            break
                except Exception:
                    pass
        frame_idx += 1
    cap.release()
    decode_time = time.perf_counter() - t0

    return {
        "decode_time": decode_time,
        "total_frames_read": frame_idx,
        "frames_detected": detected,
        "blocks_decoded": decoded_ok,
        "lt_done": lt.done,
        "K": lt.K,
        "progress": lt.progress,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_size(size_kb: int, modes: list[str], qr_version: int = 20) -> list[dict]:
    """Run all modes for one input size; return list of result dicts."""
    size = size_kb * 1024
    fd, input_path = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        f.write(os.urandom(size))
    results = []
    try:
        for mode in modes:
            video_path = tempfile.mktemp(suffix=".mp4")
            try:
                enc = encode_video_for_mode(input_path, video_path, mode,
                                             qr_version=qr_version)
                dec = _decode_video_for_mode(video_path, mode)
                enc.update(dec)
                enc["input_kb"] = size_kb
                results.append(enc)
                print(
                    f"  {mode:14}  req_v={enc['requested_version']} "
                    f"act_v={enc['actual_version']:<3} "
                    f"blocksize={enc['blocksize']:4}  "
                    f"frames={enc['num_blocks']:5}  "
                    f"{enc['frame_size'][0]}x{enc['frame_size'][1]}  "
                    f"video={enc['video_bytes'] / 1024 / 1024:6.2f}MB  "
                    f"enc={enc['encode_time']:6.2f}s "
                    f"({enc['encode_fps']:.1f} f/s)  "
                    f"dec={enc['decode_time']:6.2f}s  "
                    f"recovered={'YES' if enc['lt_done'] else 'NO'} "
                    f"({enc['blocks_decoded']}/{enc['K']})",
                    flush=True,
                )
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="10,100",
                    help="Comma-separated KB sizes (default: 10,100)")
    ap.add_argument("--qr-version", type=int, default=20)
    ap.add_argument("--skip-binary-bytes", action="store_true",
                    help="Skip the slow binary_bytes+QRCodeDetector mode.")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated subset of modes.")
    args = ap.parse_args()

    default_modes = ["base64", "binary_str", "ascii85", "base45"]
    if not args.skip_binary_bytes:
        default_modes.append("binary_bytes")
    if args.only:
        modes = [m.strip() for m in args.only.split(",")]
    else:
        modes = default_modes

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    print(f"\nComparing modes: {modes}")
    print(f"QR version request: V{args.qr_version}")
    print(f"Input sizes (KB): {sizes}\n")

    all_results: list[dict] = []
    for sz in sizes:
        print(f"=== input size {sz} KB ===")
        all_results.extend(run_size(sz, modes, qr_version=args.qr_version))
        print()

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"{'size':>5}  {'mode':<14}  {'ver':>5}  {'frames':>6}  "
          f"{'video_MB':>8}  {'enc_s':>6}  {'dec_s':>6}  "
          f"{'total_s':>7}  {'ok':>3}")
    for r in all_results:
        ok = "YES" if r["lt_done"] else "NO"
        total = r["encode_time"] + r["decode_time"]
        print(f"{r['input_kb']:>4}K  {r['mode']:<14}  "
              f"V{r['actual_version']:<4} {r['num_blocks']:>6}  "
              f"{r['video_bytes'] / 1024 / 1024:>8.2f}  "
              f"{r['encode_time']:>6.2f}  {r['decode_time']:>6.2f}  "
              f"{total:>7.2f}  {ok:>3}")


if __name__ == "__main__":
    main()
