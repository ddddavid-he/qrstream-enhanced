"""
LT Fountain Code Encoder: file → LT encoded blocks → QR frames → video.
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
from .protocol import (
    V2_VERSION,
    V3_VERSION,
    auto_blocksize,
    pack_v2,
    pack_v3,
)
from .qr_utils import generate_qr_image


# Prefer mmap-backed random access for larger uncompressed inputs.
_MMAP_THRESHOLD = 10 * 1024 * 1024


class MmapDataSource:
    """Random-access file-backed data source backed by mmap."""

    def __init__(self, input_path: str):
        self._file = open(input_path, 'rb')
        try:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError:
            self._file.close()
            raise
        self.size = len(self._mmap)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, key):
        return self._mmap[key]

    def close(self):
        self._mmap.close()
        self._file.close()


class LTEncoder:
    """Encodes a payload into an LT fountain-coded block stream."""

    def __init__(self, data, blocksize: int,
                 compressed: bool = False,
                 binary_qr: bool = False,
                 protocol_version: int = V3_VERSION,
                 c: float = DEFAULT_C, delta: float = DEFAULT_DELTA):
        self.data = data
        self.filesize = len(data)
        self.blocksize = blocksize
        self.compressed = compressed
        self.binary_qr = binary_qr
        self.protocol_version = protocol_version
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
        """Generate one encoded block for a given PRNG seed."""
        _, _, src_blocks = self.prng.get_src_blocks(seed=seed)

        if len(src_blocks) == 1:
            result = self._get_block(next(iter(src_blocks)))
        elif len(src_blocks) == 2:
            it = iter(src_blocks)
            result = xor_bytes(self._get_block(next(it)),
                               self._get_block(next(it)))
        else:
            blocks_array = np.empty((len(src_blocks), self.blocksize),
                                    dtype=np.uint8)
            for i, idx in enumerate(src_blocks):
                block = self._get_block(idx)
                blocks_array[i] = np.frombuffer(block, dtype=np.uint8)
            result = bytes(np.bitwise_xor.reduce(blocks_array, axis=0))

        seq = self._seq & 0xFFFF
        self._seq += 1
        return result, seq

    def generate_blocks(self, count: int):
        """Generate `count` encoded blocks as packed byte strings."""
        pack_fn = pack_v3 if self.protocol_version == V3_VERSION else pack_v2

        for i in range(count):
            seed = i + 1
            self.prng.set_seed(seed)
            block_data, seq = self.generate_block(seed)
            packed = pack_fn(
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


def _read_file_bytes(input_path: str) -> bytes:
    with open(input_path, 'rb') as f:
        return f.read()


def _load_payload(input_path: str, compress: bool,
                  protocol_version: int,
                  force_compress: bool = False,
                  verbose: bool = False):
    """Load the LT source payload with a low-memory path when possible.

    Returns (payload, effective_compress, used_mmap, raw_size).
    """
    raw_size = os.path.getsize(input_path)

    if compress:
        if (protocol_version == V3_VERSION and raw_size > _MMAP_THRESHOLD
                and not force_compress):
            if verbose:
                print("Compression disabled for large V3 input to keep memory usage low.")
            compress = False
        else:
            raw_data = _read_file_bytes(input_path)
            data = zlib.compress(raw_data)
            return data, True, False, raw_size

    if raw_size > _MMAP_THRESHOLD:
        return MmapDataSource(input_path), False, True, raw_size
    return _read_file_bytes(input_path), False, False, raw_size


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
                    binary_qr: bool = True,
                    protocol_version: int = V3_VERSION,
                    force_compress: bool = False):
    """Encode a file to a QR-code video using LT fountain codes."""
    payload = None
    writer = None

    try:
        payload, compress, used_mmap, raw_size = _load_payload(
            input_path,
            compress=compress,
            protocol_version=protocol_version,
            force_compress=force_compress,
            verbose=verbose,
        )

        payload_size = len(payload)
        if verbose:
            source_desc = "mmap" if used_mmap else "memory"
            print(f"Input: {input_path} ({raw_size} bytes, source={source_desc})")
            if compress:
                ratio = payload_size / raw_size * 100 if raw_size else 0.0
                print(f"Compressed: {raw_size} → {payload_size} bytes ({ratio:.1f}%)")

        blocksize = auto_blocksize(
            payload_size,
            ec_level,
            qr_version,
            binary_qr=binary_qr,
            protocol_version=protocol_version,
        )
        K = ceil(payload_size / blocksize)
        num_blocks = int(K * overhead)

        if verbose:
            mode_str = "binary" if binary_qr else "base64"
            protocol_str = f"V{protocol_version}"
            print(f"Blocks: K={K}, blocksize={blocksize}, total={num_blocks} "
                  f"(overhead={overhead}x, {mode_str}, {protocol_str})")

        encoder = LTEncoder(
            payload,
            blocksize,
            compressed=compress,
            binary_qr=binary_qr,
            protocol_version=protocol_version,
        )

        first_packed, _, _ = next(encoder.generate_blocks(1))
        first_qr = generate_qr_image(
            first_packed,
            ec_level=ec_level,
            version=qr_version,
            use_legacy=use_legacy_qr,
            binary_mode=binary_qr,
        )
        h, w = first_qr.shape[:2]

        if workers is None:
            workers = os.cpu_count() or 1

        if verbose:
            print(f"QR frame size: {w}x{h}, video FPS: {fps}, workers: {workers}")
            print(f"Estimated duration: {num_blocks / fps:.1f}s")

        fourcc_str, default_ext = _CODEC_MAP.get(codec, ('mp4v', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        if codec == 'mjpeg' and output_path.endswith('.mp4'):
            output_path = output_path[:-4] + default_ext

        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {output_path}")

        batch_size = max(workers * 4, 64)
        progress = tqdm(total=num_blocks, desc="Encoding frames")

        if workers > 1:
            block_queue = Queue(maxsize=batch_size * 2)

            def _block_producer():
                encoder._seq = 0
                for packed, _, _ in encoder.generate_blocks(num_blocks):
                    block_queue.put(packed)
                block_queue.put(None)

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
                        repeat(binary_qr),
                    ))
                    for qr_img in qr_imgs:
                        if qr_img.shape[:2] != (h, w):
                            qr_img = cv2.resize(qr_img, (w, h),
                                                interpolation=cv2.INTER_NEAREST)
                        writer.write(qr_img)
                    progress.update(len(batch))

            producer.join(timeout=5)
        else:
            encoder._seq = 0
            for packed, _, _ in encoder.generate_blocks(num_blocks):
                qr_img = generate_qr_image(
                    packed,
                    ec_level=ec_level,
                    version=qr_version,
                    use_legacy=use_legacy_qr,
                    binary_mode=binary_qr,
                )
                if qr_img.shape[:2] != (h, w):
                    qr_img = cv2.resize(qr_img, (w, h),
                                        interpolation=cv2.INTER_NEAREST)
                writer.write(qr_img)
                progress.update(1)

        progress.close()
    finally:
        if writer is not None:
            writer.release()
        if payload is not None:
            close = getattr(payload, 'close', None)
            if callable(close):
                close()

    output_size = os.path.getsize(output_path)
    if verbose:
        print(f"Output: {output_path} ({output_size} bytes, {num_blocks} frames)")
    else:
        print(f"Encoded {input_path} → {output_path} "
              f"({num_blocks} frames, {num_blocks / fps:.1f}s)")
