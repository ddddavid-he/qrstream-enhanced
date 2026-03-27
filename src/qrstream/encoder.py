"""
LT Fountain Code Encoder: file → LT encoded blocks → QR frames → MP4 video.
"""

import os
import zlib
from math import ceil
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from .lt_codec import PRNG, DEFAULT_C, DEFAULT_DELTA
from .protocol import pack_v2, auto_blocksize, V2_HEADER_SIZE
from .qr_utils import generate_qr_image


class LTEncoder:
    """Encodes a file into an infinite stream of LT fountain-coded blocks."""

    def __init__(self, data: bytes, blocksize: int,
                 compressed: bool = False,
                 c: float = DEFAULT_C, delta: float = DEFAULT_DELTA):
        self.data = data
        self.filesize = len(data)
        self.blocksize = blocksize
        self.compressed = compressed
        self.K = ceil(self.filesize / blocksize)
        self.prng = PRNG(self.K, delta=delta, c=c)
        self._seq = 0

    def _get_block(self, index: int) -> bytes:
        """Get the i-th source block (zero-padded if last block is short)."""
        start = index * self.blocksize
        end = start + self.blocksize
        block = self.data[start:end]
        if len(block) < self.blocksize:
            block = block + b'\x00' * (self.blocksize - len(block))
        return block

    def generate_block(self, seed: int) -> tuple[bytes, int]:
        """Generate one encoded block for a given PRNG seed.

        Returns (encoded_data_bytes, block_seq).
        """
        _, _, src_blocks = self.prng.get_src_blocks(seed=seed)

        # XOR all source blocks together
        from .lt_codec import xor_bytes
        result = b'\x00' * self.blocksize
        for idx in src_blocks:
            result = xor_bytes(result, self._get_block(idx))

        seq = self._seq
        self._seq += 1
        return result, seq

    def generate_blocks(self, count: int):
        """Generate `count` encoded blocks as packed V2 byte strings.

        Yields (packed_bytes, seed, seq) tuples.
        Seeds are sequential starting from 1.
        """
        for i in range(count):
            seed = i + 1
            self.prng.set_seed(seed)
            block_data, seq = self.generate_block(seed)
            packed = pack_v2(
                filesize=self.filesize,
                blocksize=self.blocksize,
                block_count=self.K,
                seed=seed,
                block_seq=seq,
                data=block_data,
                compressed=self.compressed,
            )
            yield packed, seed, seq


def encode_to_video(input_path: str, output_path: str,
                    overhead: float = 2.0,
                    fps: int = 10,
                    ec_level: int = 1,
                    qr_version: int = 20,
                    compress: bool = True,
                    verbose: bool = False,
                    workers: int | None = None):
    """Encode a file to a QR-code video using LT fountain codes.

    Args:
        input_path:  Path to input file
        output_path: Path to output .mp4 video
        overhead:    Ratio of encoded blocks to source blocks (default 2.5x)
        fps:         Frames per second in output video
        ec_level:    QR error correction level (0=L, 1=M, 2=Q, 3=H)
        qr_version:  QR code version 1-40 (default: 20)
        compress:    Whether to zlib-compress the data first
        verbose:     Print progress details
        workers:     Number of parallel workers for QR generation (default: CPU count)
    """
    # Read input
    with open(input_path, 'rb') as f:
        raw_data = f.read()

    if verbose:
        print(f"Input: {input_path} ({len(raw_data)} bytes)")

    # Optionally compress
    if compress:
        data = zlib.compress(raw_data)
        if verbose:
            ratio = len(data) / len(raw_data) * 100
            print(f"Compressed: {len(raw_data)} → {len(data)} bytes ({ratio:.1f}%)")
    else:
        data = raw_data

    filesize = len(data)
    blocksize = auto_blocksize(filesize, ec_level, qr_version)
    K = ceil(filesize / blocksize)
    num_blocks = int(K * overhead)

    if verbose:
        print(f"Blocks: K={K}, blocksize={blocksize}, "
              f"total={num_blocks} (overhead={overhead}x)")

    # Create encoder
    encoder = LTEncoder(data, blocksize, compressed=compress)

    # Generate first QR to determine frame size
    first_packed, _, _ = next(encoder.generate_blocks(1))
    first_qr = generate_qr_image(first_packed, ec_level=ec_level,
                                   version=qr_version)
    h, w = first_qr.shape[:2]

    if workers is None:
        workers = min(os.cpu_count() or 1, 8)

    if verbose:
        print(f"QR frame size: {w}x{h}, video FPS: {fps}, workers: {workers}")
        print(f"Estimated duration: {num_blocks / fps:.1f}s")

    # Pre-generate all packed blocks (lightweight, fast)
    encoder._seq = 0
    all_packed = [packed for packed, _, _ in encoder.generate_blocks(num_blocks)]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path}")

    # Batch size for parallel QR generation to control memory usage
    batch_size = max(workers * 4, 64)

    try:
        progress = tqdm(total=num_blocks, desc="Encoding frames",
                        disable=not verbose)

        if workers > 1:
            # Parallel: generate QR images in batches using ProcessPoolExecutor
            ec_levels = [ec_level] * batch_size
            versions = [qr_version] * batch_size
            # Use default box_size=10 and border=4
            box_sizes = [10] * batch_size
            borders = [4] * batch_size

            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i in range(0, num_blocks, batch_size):
                    batch = all_packed[i:i + batch_size]
                    n = len(batch)
                    qr_imgs = list(pool.map(
                        generate_qr_image, batch,
                        ec_levels[:n], box_sizes[:n],
                        borders[:n], versions[:n],
                    ))
                    for qr_img in qr_imgs:
                        if qr_img.shape[:2] != (h, w):
                            qr_img = cv2.resize(qr_img, (w, h),
                                                interpolation=cv2.INTER_NEAREST)
                        writer.write(qr_img)
                    progress.update(n)
        else:
            # Serial fallback
            for packed in all_packed:
                qr_img = generate_qr_image(packed, ec_level=ec_level,
                                            version=qr_version)
                if qr_img.shape[:2] != (h, w):
                    qr_img = cv2.resize(qr_img, (w, h),
                                        interpolation=cv2.INTER_NEAREST)
                writer.write(qr_img)
                progress.update(1)

        progress.close()
    finally:
        writer.release()

    output_size = os.path.getsize(output_path)
    if verbose:
        print(f"Output: {output_path} ({output_size} bytes, "
              f"{num_blocks} frames)")
    else:
        print(f"Encoded {input_path} → {output_path} "
              f"({num_blocks} frames, {num_blocks / fps:.1f}s)")
