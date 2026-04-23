"""
Micro-benchmarks for suspected hot paths.

These measure isolated components (no pipeline overhead) to determine
per-call cost of each stage.  Combined with profile_encode/profile_decode
staged timings, they make it easy to see what dominates.

Covered:
  - generate_qr_image        (encode: QR matrix → BGR image)
  - LTEncoder.generate_block (encode: PRNG + XOR)
  - _downscale_frame         (decode: frame normalisation)
  - try_decode_qr            (decode: WeChatQR / QRCodeDetector)
  - cobs_encode / decode     (encode & decode: protocol layer)
  - unpack (protocol)        (decode: CRC + header parse)
  - BlockGraph.add_block     (decode: belief propagation step)
  - PRNG.get_src_blocks      (both paths)
  - VideoWriter.write (mp4v) (encode: video muxing)
"""

import os
import sys
import tempfile
import time
from math import ceil
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from qrstream.encoder import LTEncoder  # noqa: E402
from qrstream.lt_codec import BlockGraph, PRNG  # noqa: E402
from qrstream.protocol import (  # noqa: E402
    auto_blocksize, base45_decode, base45_encode, cobs_decode, cobs_encode,
    pack_v3, unpack,
)
from qrstream.qr_utils import generate_qr_image, try_decode_qr  # noqa: E402


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _timed_loop(fn, iterations: int) -> dict:
    """Run fn() `iterations` times, return stats in µs."""
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return {
        "iter": iterations,
        "mean_us": mean(samples),
        "median_us": median(samples),
        "min_us": min(samples),
        "p95_us": sorted(samples)[int(len(samples) * 0.95) - 1],
    }


def _format_row(name: str, stats: dict, extra: str = "") -> str:
    return (
        f"  {name:<34} iter={stats['iter']:>5}  "
        f"median={stats['median_us']:8.1f}µs  "
        f"mean={stats['mean_us']:8.1f}µs  "
        f"min={stats['min_us']:8.1f}µs  "
        f"p95={stats['p95_us']:8.1f}µs  {extra}"
    )


# ─────────────────────────────────────────────────────────────

def bench_generate_qr_image() -> list[str]:
    """Test generate_qr_image with realistic packed blocks per QR version.

    Uses v0.6.0 default path (base45 + QR alphanumeric mode).
    """
    out = ["\n--- generate_qr_image (encode hot path, base45/alphanumeric) ---"]
    for version in (20, 25, 40):
        auto_bs = auto_blocksize(
            10 * 1024 * 1024, ec_level=1, qr_version=version,
            alphanumeric_qr=True)
        # Use a conservative blocksize well below auto_bs.
        blocksize = max(64, int(auto_bs * 0.5))
        dummy_data = os.urandom(blocksize * 4)
        encoder = LTEncoder(dummy_data, blocksize,
                            alphanumeric_qr=True)
        packed, _, _ = next(encoder.generate_blocks(1))
        # warm-up
        generate_qr_image(packed, ec_level=1, version=version,
                          box_size=10, border=4.0, alphanumeric=True)
        stats = _timed_loop(
            lambda p=packed, v=version: generate_qr_image(
                p, ec_level=1, version=v, box_size=10, border=4.0,
                alphanumeric=True),
            iterations=30,
        )
        out.append(_format_row(
            f"v{version} packed={len(packed)}B", stats,
            f"blocksize={blocksize}/auto={auto_bs}"))
    return out


def bench_lt_generate_block() -> list[str]:
    """Pure LT encoding for various K."""
    out = ["\n--- LTEncoder.generate_block (encode) ---"]
    for K in [10, 100, 1000, 10000]:
        blocksize = 666
        data = os.urandom(K * blocksize)
        encoder = LTEncoder(data, blocksize)
        # warm
        encoder.generate_block(1)
        seed_counter = [1]

        def _call():
            encoder.prng.set_seed(seed_counter[0])
            encoder.generate_block(seed_counter[0])
            seed_counter[0] += 1

        iters = 200 if K <= 1000 else 50
        stats = _timed_loop(_call, iterations=iters)
        out.append(_format_row(f"K={K}, blocksize={blocksize}", stats))
    return out


def bench_downscale_frame() -> list[str]:
    from qrstream.decoder import _downscale_frame
    out = ["\n--- _downscale_frame (decode frame prep) ---"]
    for (h, w) in [(720, 720), (1080, 1080), (2160, 2160), (3840, 2160)]:
        frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        stats = _timed_loop(lambda: _downscale_frame(frame), iterations=30)
        out.append(_format_row(f"{w}x{h}", stats))
    return out


def bench_try_decode_qr() -> list[str]:
    """Detect a real QR code (most realistic). Uses v0.6.0 default path."""
    out = ["\n--- try_decode_qr (decode worker hot path) ---"]
    for version in (20, 25, 40):
        auto_bs = auto_blocksize(
            10 * 1024 * 1024, ec_level=1, qr_version=version,
            alphanumeric_qr=True)
        blocksize = max(64, int(auto_bs * 0.5))
        dummy_data = os.urandom(blocksize * 4)
        encoder = LTEncoder(dummy_data, blocksize,
                            alphanumeric_qr=True)
        packed, _, _ = next(encoder.generate_blocks(1))
        qr_bgr = generate_qr_image(packed, ec_level=1, version=version,
                                    box_size=10, border=4.0,
                                    alphanumeric=True)
        # warm-up (WeChatQR model load)
        try_decode_qr(qr_bgr)
        stats = _timed_loop(
            lambda q=qr_bgr: try_decode_qr(q), iterations=30)
        out.append(_format_row(
            f"v{version} packed={len(packed)}B", stats,
            f"frame={qr_bgr.shape[1]}x{qr_bgr.shape[0]}"))
    return out


def bench_base45() -> list[str]:
    """base45 encode/decode — new default encoding layer (replaces COBS)."""
    out = ["\n--- base45_encode / base45_decode (new default) ---"]
    for size in [100, 500, 1000, 2000]:
        data = os.urandom(size)
        encoded = base45_encode(data)
        enc_stats = _timed_loop(
            lambda d=data: base45_encode(d), iterations=500)
        dec_stats = _timed_loop(
            lambda e=encoded: base45_decode(e), iterations=500)
        out.append(_format_row(
            f"base45_encode {size}B", enc_stats, f"→ {len(encoded)}B"))
        out.append(_format_row(
            f"base45_decode {size}B", dec_stats))
    return out


def bench_cobs() -> list[str]:
    """Kept for reference; COBS is now decode-only legacy fallback."""
    out = ["\n--- cobs_encode / cobs_decode (legacy fallback) ---"]
    for size in [100, 500, 1000, 2000]:
        data = os.urandom(size)
        encoded = cobs_encode(data)
        enc_stats = _timed_loop(
            lambda d=data: cobs_encode(d), iterations=500)
        dec_stats = _timed_loop(
            lambda e=encoded: cobs_decode(e), iterations=500)
        out.append(_format_row(
            f"cobs_encode {size}B", enc_stats, f"→ {len(encoded)}B"))
        out.append(_format_row(
            f"cobs_decode {size}B", dec_stats))
    return out


def bench_unpack() -> list[str]:
    out = ["\n--- protocol.unpack (decode, with CRC) ---"]
    blocksize = 666
    K = 1000
    data = os.urandom(K * blocksize)
    encoder = LTEncoder(data, blocksize)
    packed, _, _ = next(encoder.generate_blocks(1))
    stats = _timed_loop(
        lambda p=packed: unpack(p), iterations=2000)
    out.append(_format_row("unpack V3", stats, f"payload={len(packed)}B"))
    stats = _timed_loop(
        lambda p=packed: unpack(p, skip_crc=True), iterations=2000)
    out.append(_format_row("unpack V3 (skip_crc)", stats))
    return out


def bench_blockgraph_add_block() -> list[str]:
    out = ["\n--- BlockGraph.add_block (decode belief prop.) ---"]
    for K in [100, 1000, 5000]:
        blocksize = 666
        data = os.urandom(K * blocksize)
        encoder = LTEncoder(data, blocksize)
        # Pre-generate packed blocks (enough for decode).
        packed_blocks = []
        for packed, _, _ in encoder.generate_blocks(int(K * 2.0)):
            packed_blocks.append(packed)

        # Time: full decode loop from scratch (single run = one observation).
        samples = []
        for _ in range(3):
            graph = BlockGraph(K)
            prng = PRNG(K)
            t0 = time.perf_counter()
            for packed in packed_blocks:
                header, data_bytes = unpack(packed, skip_crc=True)
                _, _, src = prng.get_src_blocks(seed=header.seed)
                if len(data_bytes) < blocksize:
                    data_bytes = data_bytes + b"\x00" * (blocksize - len(data_bytes))
                if graph.add_block(src, data_bytes):
                    break
            samples.append((time.perf_counter() - t0) * 1e6)
        stats = {
            "iter": 3,
            "mean_us": mean(samples),
            "median_us": median(samples),
            "min_us": min(samples),
            "p95_us": max(samples),
        }
        out.append(_format_row(
            f"full decode K={K}",
            stats,
            f"(≈{stats['median_us'] / K:.1f}µs/block avg)",
        ))
    return out


def bench_prng_get_src_blocks() -> list[str]:
    out = ["\n--- PRNG.get_src_blocks (both paths) ---"]
    for K in [100, 1000, 10000]:
        prng = PRNG(K)
        seed_counter = [1]

        def _call():
            prng.get_src_blocks(seed=seed_counter[0])
            seed_counter[0] += 1

        stats = _timed_loop(_call, iterations=500)
        out.append(_format_row(f"K={K}", stats))
    return out


def bench_video_writer() -> list[str]:
    out = ["\n--- cv2.VideoWriter.write (encode muxing) ---"]
    # Use a representative QR frame at v0.6.0 default (V25, base45).
    data = os.urandom(666)
    qr_bgr = generate_qr_image(data, ec_level=1, version=25, box_size=10,
                                border=4.0, alphanumeric=True)
    h, w = qr_bgr.shape[:2]
    for codec_name, fourcc_str in [("mp4v", "mp4v"), ("mjpeg", "MJPG")]:
        output_path = tempfile.mktemp(suffix=".mp4" if codec_name == "mp4v" else ".avi")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, 10, (w, h))
        if not writer.isOpened():
            out.append(f"  {codec_name}: could not open writer, skipped")
            continue
        # warm-up
        writer.write(qr_bgr)
        stats = _timed_loop(lambda w_=writer, q=qr_bgr: w_.write(q), iterations=200)
        writer.release()
        if os.path.exists(output_path):
            os.remove(output_path)
        out.append(_format_row(
            f"{codec_name} {w}x{h}", stats,
            ""))
    return out


# ─────────────────────────────────────────────────────────────

def main() -> None:
    all_lines: list[str] = ["QRStream hot-path micro-benchmarks", "=" * 70]
    all_lines.extend(bench_generate_qr_image())
    all_lines.extend(bench_lt_generate_block())
    all_lines.extend(bench_downscale_frame())
    all_lines.extend(bench_try_decode_qr())
    all_lines.extend(bench_base45())
    all_lines.extend(bench_cobs())
    all_lines.extend(bench_unpack())
    all_lines.extend(bench_blockgraph_add_block())
    all_lines.extend(bench_prng_get_src_blocks())
    all_lines.extend(bench_video_writer())

    text = "\n".join(all_lines) + "\n"
    print(text)
    (RESULTS_DIR / "hotpaths_report.txt").write_text(text)
    print(f"Saved: {RESULTS_DIR / 'hotpaths_report.txt'}")


if __name__ == "__main__":
    main()
