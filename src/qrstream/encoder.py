"""
LT Fountain Code Encoder: file → LT encoded blocks → QR frames → MP4 video.
"""

import mmap
import os
import zlib
from itertools import repeat
from math import ceil
from queue import Queue
from threading import Thread
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from .lt_codec import PRNG, DEFAULT_C, DEFAULT_DELTA, xor_bytes
from .protocol import pack_v2, auto_blocksize, V2_HEADER_SIZE
from .qr_utils import generate_qr_image


# Threshold for memory-mapped file reading (10 MB)
_MMAP_THRESHOLD = 10 * 1024 * 1024


class LTEncoder:
    """Encodes a file into an infinite stream of LT fountain-coded blocks."""

    def __init__(self, data: bytes, blocksize: int,
                 compressed: bool = False,
                 binary_qr: bool = False,
                 c: float = DEFAULT_C, delta: float = DEFAULT_DELTA):
        self.data = data
        self.filesize = len(data)
        self.blocksize = blocksize
        self.compressed = compressed
        self.binary_qr = binary_qr
        self.K = ceil(self.filesize / blocksize)
        self.prng = PRNG(self.K, delta=delta, c=c)
        self._seq = 0
        self._cached_last_block = None

    def _get_block(self, index: int) -> bytes:
        """Get the i-th source block (zero-padded if last block is short)."""
        start = index * self.blocksize
        end = start + self.blocksize
        block = self.data[start:end]
        if len(block) < self.blocksize:
            if self._cached_last_block is None:
                self._cached_last_block = block + b'\x00' * (self.blocksize - len(block))
            return self._cached_last_block
        return block

    def generate_block(self, seed: int) -> tuple[bytes, int]:
        """Generate one encoded block for a given PRNG seed.

        Returns (encoded_data_bytes, block_seq).
        Uses batch numpy XOR when multiple source blocks are involved.
        """
        _, _, src_blocks = self.prng.get_src_blocks(seed=seed)

        if len(src_blocks) == 1:
            result = self._get_block(next(iter(src_blocks)))
        elif len(src_blocks) == 2:
            it = iter(src_blocks)
            result = xor_bytes(self._get_block(next(it)),
                               self._get_block(next(it)))
        else:
            # Batch XOR: stack all source blocks into a 2D numpy array
            # and reduce with bitwise_xor in one vectorized operation
            blocks_array = np.empty((len(src_blocks), self.blocksize),
                                    dtype=np.uint8)
            for i, idx in enumerate(src_blocks):
                block = self._get_block(idx)
                blocks_array[i] = np.frombuffer(block, dtype=np.uint8)
            result = bytes(np.bitwise_xor.reduce(blocks_array, axis=0))

        seq = self._seq & 0xFFFF  # Wrap to uint16 range (0-65535)
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
                binary_qr=self.binary_qr,
            )
            yield packed, seed, seq


def _read_file(input_path: str) -> bytes:
    """Read file, using mmap for large files to reduce memory pressure."""
    file_size = os.path.getsize(input_path)
    if file_size > _MMAP_THRESHOLD:
        with open(input_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = mm[:]
            mm.close()
        return data
    else:
        with open(input_path, 'rb') as f:
            return f.read()


# Codec map for video output
_CODEC_MAP = {
    'mp4v': ('mp4v', '.mp4'),
    'mjpeg': ('MJPG', '.avi'),
}


def encode_to_video(input_path: str, output_path: str,
                    overhead: float = 2.0,
                    fps: int = 10,
                    ec_level: int = 1,
                    qr_version: int = 20,
                    compress: bool = True,
                    verbose: bool = False,
                    workers: int | None = None,
                    use_legacy_qr: bool = False,
                    codec: str = 'mp4v',
                    binary_qr: bool = False):
    """Encode a file to a QR-code video using LT fountain codes.

    Args:
        input_path:  Path to input file
        output_path: Path to output video
        overhead:    Ratio of encoded blocks to source blocks (default 2.0x)
        fps:         Frames per second in output video
        ec_level:    QR error correction level (0=L, 1=M, 2=Q, 3=H)
        qr_version:  QR code version 1-40 (default: 20)
        compress:    Whether to zlib-compress the data first
        verbose:     Print progress details
        workers:     Number of parallel workers for QR generation (default: CPU count)
        use_legacy_qr: Use qrcode library instead of OpenCV (slower, more control)
        codec:       Video codec - 'mp4v' (default) or 'mjpeg' (faster, larger files)
        binary_qr:   Embed raw bytes in QR (skip base64, 33% more capacity)
    """
    raw_data = _read_file(input_path)

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
    blocksize = auto_blocksize(filesize, ec_level, qr_version,
                               binary_qr=binary_qr)
    K = ceil(filesize / blocksize)
    num_blocks = int(K * overhead)

    if verbose:
        mode_str = "binary" if binary_qr else "base64"
        print(f"Blocks: K={K}, blocksize={blocksize}, "
              f"total={num_blocks} (overhead={overhead}x, {mode_str})")

    # Create encoder
    encoder = LTEncoder(data, blocksize, compressed=compress,
                        binary_qr=binary_qr)

    # Generate first QR to determine frame size
    first_packed, _, _ = next(encoder.generate_blocks(1))
    first_qr = generate_qr_image(first_packed, ec_level=ec_level,
                                   version=qr_version,
                                   use_legacy=use_legacy_qr,
                                   binary_mode=binary_qr)
    h, w = first_qr.shape[:2]

    if workers is None:
        workers = min(os.cpu_count() or 1, 8)

    if verbose:
        print(f"QR frame size: {w}x{h}, video FPS: {fps}, workers: {workers}")
        print(f"Estimated duration: {num_blocks / fps:.1f}s")

    # Resolve video codec
    fourcc_str, default_ext = _CODEC_MAP.get(codec, ('mp4v', '.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    # Adjust output extension if needed
    if codec == 'mjpeg' and output_path.endswith('.mp4'):
        output_path = output_path[:-4] + default_ext

    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path}")

    # Batch size for parallel QR generation
    batch_size = max(workers * 4, 64)

    try:
        progress = tqdm(total=num_blocks, desc="Encoding frames",
                        disable=not verbose)

        if workers > 1:
            # Streaming: produce blocks in a background thread,
            # consume them in batches for parallel QR generation.
            block_queue = Queue(maxsize=batch_size * 2)

            def _block_producer():
                enc = LTEncoder(data, blocksize, compressed=compress,
                                binary_qr=binary_qr)
                enc._seq = 0
                for packed, _, _ in enc.generate_blocks(num_blocks):
                    block_queue.put(packed)
                block_queue.put(None)  # sentinel

            producer = Thread(target=_block_producer, daemon=True)
            producer.start()

            with ProcessPoolExecutor(max_workers=workers) as pool:
                done = False
                while not done:
                    batch = []
                    for _ in range(batch_size):
                        item = block_queue.get()
                        if item is None:
                            done = True
                            break
                        batch.append(item)
                    if not batch:
                        break
                    qr_imgs = list(pool.map(
                        generate_qr_image, batch,
                        repeat(ec_level), repeat(10), repeat(4),
                        repeat(qr_version), repeat(use_legacy_qr),
                        repeat(binary_qr)
                    ))
                    for qr_img in qr_imgs:
                        if qr_img.shape[:2] != (h, w):
                            qr_img = cv2.resize(qr_img, (w, h),
                                                interpolation=cv2.INTER_NEAREST)
                        writer.write(qr_img)
                    progress.update(len(batch))

            producer.join(timeout=5)
        else:
            # Serial fallback (streaming, no pre-allocation)
            encoder._seq = 0
            for packed, _, _ in encoder.generate_blocks(num_blocks):
                qr_img = generate_qr_image(packed, ec_level=ec_level,
                                            version=qr_version,
                                            use_legacy=use_legacy_qr,
                                            binary_mode=binary_qr)
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
