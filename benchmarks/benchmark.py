"""
QRStream performance benchmark suite.

Measures encoding and decoding throughput across different file sizes.
Run before and after optimizations to compare.
"""

import os
import sys
import time
import tempfile
from math import ceil

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qrstream.encoder import encode_to_video, LTEncoder
from qrstream.decoder import decode_blocks, LTDecoder
from qrstream.lt_codec import PRNG, BlockGraph, xor_bytes
from qrstream.protocol import auto_blocksize
from qrstream.qr_utils import generate_qr_image


def benchmark_lt_codec(data_size: int = 4096, blocksize: int = 64,
                       overhead: float = 3.0, iterations: int = 5):
    """Benchmark pure LT encode + decode (no QR/video)."""
    data = os.urandom(data_size)
    K = ceil(data_size / blocksize)
    num_blocks = int(K * overhead)

    # Encode benchmark
    times_encode = []
    for _ in range(iterations):
        encoder = LTEncoder(data, blocksize)
        start = time.perf_counter()
        blocks = [packed for packed, _, _ in encoder.generate_blocks(num_blocks)]
        elapsed = time.perf_counter() - start
        times_encode.append(elapsed)

    # Decode benchmark
    times_decode = []
    for _ in range(iterations):
        decoder = LTDecoder()
        start = time.perf_counter()
        for block_bytes in blocks:
            done, _ = decoder.decode_bytes(block_bytes)
            if done:
                _ = decoder.bytes_dump()
                break
        elapsed = time.perf_counter() - start
        times_decode.append(elapsed)

    return {
        'encode_min': min(times_encode),
        'encode_avg': sum(times_encode) / len(times_encode),
        'decode_min': min(times_decode),
        'decode_avg': sum(times_decode) / len(times_decode),
    }


def benchmark_xor(block_size: int = 512, iterations: int = 10000):
    """Benchmark xor_bytes throughput."""
    a = os.urandom(block_size)
    b = os.urandom(block_size)

    start = time.perf_counter()
    for _ in range(iterations):
        xor_bytes(a, b)
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    throughput_mb = (block_size * iterations) / elapsed / 1e6
    return {
        'ops_per_sec': ops_per_sec,
        'throughput_mb_s': throughput_mb,
        'total_time': elapsed,
    }


def benchmark_qr_generation(data_size: int = 400, iterations: int = 50):
    """Benchmark QR code image generation."""
    data = os.urandom(data_size)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        img = generate_qr_image(data, ec_level=1, version=20)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'min': min(times),
        'avg': sum(times) / len(times),
        'total': sum(times),
        'frame_size': img.shape[:2],
    }


def benchmark_block_graph(K: int = 500, blocksize: int = 128,
                          overhead: float = 2.5, iterations: int = 3):
    """Benchmark BlockGraph belief-propagation performance."""
    data = os.urandom(K * blocksize)

    times = []
    for _ in range(iterations):
        encoder = LTEncoder(data, blocksize)
        num_blocks = int(K * overhead)

        # Pre-generate blocks
        packed_blocks = []
        for packed, seed, seq in encoder.generate_blocks(num_blocks):
            packed_blocks.append(packed)

        # Benchmark decode (belief propagation)
        decoder = LTDecoder()
        start = time.perf_counter()
        for block_bytes in packed_blocks:
            done, _ = decoder.decode_bytes(block_bytes)
            if done:
                break
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'K': K,
        'blocksize': blocksize,
        'min': min(times),
        'avg': sum(times) / len(times),
    }


def benchmark_encode_to_video(file_size_kb: int = 10, iterations: int = 2):
    """Benchmark full encode pipeline (file → video)."""
    test_data = os.urandom(file_size_kb * 1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(test_data)
        input_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        output_path = f.name

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        encode_to_video(input_path, output_path, overhead=2.0,
                        fps=10, verbose=False, workers=4)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if os.path.exists(output_path):
            os.remove(output_path)

    os.remove(input_path)

    return {
        'file_size_kb': file_size_kb,
        'min': min(times),
        'avg': sum(times) / len(times),
    }


def main():
    print("=" * 60)
    print("QRStream Performance Benchmark")
    print("=" * 60)

    # XOR benchmark
    print("\n--- XOR Throughput ---")
    for bs in [128, 512, 1024]:
        result = benchmark_xor(block_size=bs)
        print(f"  blocksize={bs}: {result['ops_per_sec']:.0f} ops/s, "
              f"{result['throughput_mb_s']:.1f} MB/s")

    # QR generation benchmark
    print("\n--- QR Generation ---")
    result = benchmark_qr_generation()
    print(f"  avg={result['avg']*1000:.1f}ms, "
          f"min={result['min']*1000:.1f}ms, "
          f"frame={result['frame_size']}")

    # LT codec benchmark
    print("\n--- LT Codec (encode + decode) ---")
    for size in [1024, 4096, 16384]:
        result = benchmark_lt_codec(data_size=size)
        print(f"  {size}B: encode={result['encode_avg']*1000:.1f}ms, "
              f"decode={result['decode_avg']*1000:.1f}ms")

    # BlockGraph benchmark
    print("\n--- BlockGraph Belief Propagation ---")
    for K in [100, 500, 1000]:
        result = benchmark_block_graph(K=K)
        print(f"  K={K}: avg={result['avg']*1000:.1f}ms, "
              f"min={result['min']*1000:.1f}ms")

    # Full encode pipeline
    print("\n--- Full Encode Pipeline (file → video) ---")
    for size_kb in [5, 10]:
        result = benchmark_encode_to_video(file_size_kb=size_kb, iterations=2)
        print(f"  {size_kb}KB: avg={result['avg']:.2f}s, "
              f"min={result['min']:.2f}s")

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == '__main__':
    main()
