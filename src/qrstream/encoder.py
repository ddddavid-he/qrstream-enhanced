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
    _resolve_alphanumeric_flag,
    auto_blocksize,
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
                 alphanumeric_qr: bool | None = None,
                 c: float = DEFAULT_C, delta: float = DEFAULT_DELTA):
        self.data = data
        self.filesize = len(data)
        self.blocksize = blocksize
        self.compressed = compressed
        # ``binary_qr`` and ``alphanumeric_qr`` both map to the same
        # header flag bit (0x02); prefer the alphanumeric_qr name.
        self.alphanumeric_qr = _resolve_alphanumeric_flag(
            binary_qr, alphanumeric_qr)
        self.K = ceil(self.filesize / blocksize)
        self.prng = PRNG(self.K, delta=delta, c=c)
        self._seq = 0
        self._cached_last_block = None

    # Keep ``binary_qr`` as a read-only attribute alias so code that
    # inspects the encoder state (older tests, scripts) keeps working.
    @property
    def binary_qr(self) -> bool:
        return self.alphanumeric_qr

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
        for i in range(count):
            seed = i + 1
            self.prng.set_seed(seed)
            block_data, seq = self.generate_block(seed)
            packed = pack_v3(
                filesize=self.filesize,
                blocksize=self.blocksize,
                block_count=self.K,
                seed=seed,
                block_seq=seq,
                data=block_data,
                compressed=self.compressed,
                alphanumeric_qr=self.alphanumeric_qr,
            )
            yield packed, seed, seq


def _read_file_bytes(input_path: str) -> bytes:
    with open(input_path, 'rb') as f:
        return f.read()


def _load_payload(input_path: str, compress: bool,
                  force_compress: bool = False,
                  verbose: bool = False):
    """Load the LT source payload with a low-memory path when possible.

    Returns (payload, effective_compress, used_mmap, raw_size).
    """
    raw_size = os.path.getsize(input_path)

    if compress:
        if raw_size > _MMAP_THRESHOLD and not force_compress:
            if verbose:
                print("Compression disabled for large input to keep memory usage low.")
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


def _resolve_border_modules(qr_version: int, border: float | None) -> float:
    """Resolve CLI/API border input to QR quiet-zone width in modules."""
    if border is None:
        return 4.0
    return round((qr_version - 1) * 4 + 21) * border / 100.0


def encode_to_video(input_path: str, output_path: str,
                    overhead: float = 2.0,
                    fps: int = 10,
                    ec_level: int = 1,
                    qr_version: int = 25,
                    border: float | None = None,
                    lead_in_seconds: float = 0.0,
                    compress: bool = True,
                    verbose: bool = False,
                    workers: int | None = None,
                    use_legacy_qr: bool = False,
                    codec: str = 'mp4v',
                    binary_qr: bool = True,
                    alphanumeric_qr: bool | None = None,
                    force_compress: bool = False):
    """Encode a file to a QR-code video using LT fountain codes.

    ``binary_qr`` and ``alphanumeric_qr`` are aliases for the
    high-density QR mode flag; prefer ``alphanumeric_qr`` in new code.
    When enabled (default), frames are encoded via base45 into QR
    alphanumeric mode, carrying ~29% more payload per frame than base64.
    """
    high_density = _resolve_alphanumeric_flag(binary_qr, alphanumeric_qr)
    payload = None
    writer = None
    writer_thread = None
    writer_queue: Queue | None = None

    try:
        payload, compress, used_mmap, raw_size = _load_payload(
            input_path,
            compress=compress,
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
            alphanumeric_qr=high_density,
        )
        border_modules = _resolve_border_modules(qr_version, border)
        K = ceil(payload_size / blocksize)
        num_blocks = int(K * overhead)
        lead_in_frames = max(0, round(lead_in_seconds * fps))
        total_frames = num_blocks + lead_in_frames

        if verbose:
            mode_str = "alphanumeric/base45" if high_density else "base64"
            print(f"Blocks: K={K}, blocksize={blocksize}, total={num_blocks} "
                  f"(overhead={overhead}x, {mode_str})")

        encoder = LTEncoder(
            payload,
            blocksize,
            compressed=compress,
            alphanumeric_qr=high_density,
        )

        first_packed, _, _ = next(encoder.generate_blocks(1))
        first_qr = generate_qr_image(
            first_packed,
            ec_level=ec_level,
            box_size=10,
            border=border_modules,
            version=qr_version,
            use_legacy=use_legacy_qr,
            alphanumeric=high_density,
        )
        h, w = first_qr.shape[:2]

        if workers is None:
            workers = os.cpu_count() or 1

        if verbose:
            print(f"QR frame size: {w}x{h}, video FPS: {fps}, workers: {workers}")
            print(f"Estimated duration: {total_frames / fps:.1f}s")

        fourcc_str, default_ext = _CODEC_MAP.get(codec, ('mp4v', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        if codec == 'mjpeg' and output_path.endswith('.mp4'):
            output_path = output_path[:-4] + default_ext

        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {output_path}")

        # ── VideoWriter runs on its own thread ──────────────────────
        # Measured baseline (v0.6.1, 10 MB input, 14 workers):
        #   VideoWriter.write was 54% of encode wall-time, blocking
        #   the main thread between batches. Moving the write loop
        #   to a dedicated thread lets pool.map() keep the workers
        #   busy while the previous batch is being muxed.
        writer_queue: Queue = Queue(maxsize=max(workers * 8, 128))

        def _writer_loop():
            while True:
                frame = writer_queue.get()
                if frame is None:
                    return
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h),
                                       interpolation=cv2.INTER_NEAREST)
                writer.write(frame)

        writer_thread = Thread(target=_writer_loop, daemon=False)
        writer_thread.start()

        if lead_in_frames:
            blank_frame = np.full((h, w, 3), 255, dtype=first_qr.dtype)
            for _ in range(lead_in_frames):
                writer_queue.put(blank_frame)

        batch_size = max(workers * 4, 64)
        progress = tqdm(total=num_blocks, desc="Encode", unit="f",
                        dynamic_ncols=True)

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
                    # generate_qr_image signature:
                    #   (data, ec_level, box_size, border, version,
                    #    use_legacy, binary_mode, alphanumeric)
                    qr_imgs = list(pool.map(
                        generate_qr_image, batch,
                        repeat(ec_level), repeat(10), repeat(border_modules),
                        repeat(qr_version), repeat(use_legacy_qr),
                        repeat(None), repeat(high_density),
                    ))
                    for qr_img in qr_imgs:
                        writer_queue.put(qr_img)
                    progress.update(len(batch))

            producer.join(timeout=5)
        else:
            encoder._seq = 0
            for packed, _, _ in encoder.generate_blocks(num_blocks):
                qr_img = generate_qr_image(
                    packed,
                    ec_level=ec_level,
                    box_size=10,
                    border=border_modules,
                    version=qr_version,
                    use_legacy=use_legacy_qr,
                    alphanumeric=high_density,
                )
                writer_queue.put(qr_img)
                progress.update(1)

        progress.close()

        # Flush writer: signal sentinel and wait for disk writes to drain
        writer_queue.put(None)
        writer_thread.join()
        writer_thread = None
    finally:
        # On the exception path, make sure we don't leave the writer
        # thread blocked on an empty queue (daemon=False would keep the
        # process alive after an error).
        if writer_thread is not None and writer_thread.is_alive():
            if writer_queue is not None:
                writer_queue.put(None)
            writer_thread.join(timeout=5)
        if writer is not None:
            writer.release()
        if payload is not None:
            close = getattr(payload, 'close', None)
            if callable(close):
                close()

    output_size = os.path.getsize(output_path)
    if verbose:
        print(f"Output: {output_path} ({output_size} bytes, {total_frames} frames)")
    else:
        print(f"Encoded {input_path} → {output_path} "
              f"({total_frames} frames, {total_frames / fps:.1f}s)")
