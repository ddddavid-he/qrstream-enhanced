"""
Fountain Code Encoder: file -> encoded blocks -> QR frames -> video.

Supports both LT fountain codes (legacy) and RaptorQ (RFC 6330).
"""

import mmap
import os
import tempfile
import time
import zlib
from itertools import repeat
from math import ceil
from queue import Full, Queue
from threading import Event, Lock, Thread
from concurrent.futures import ThreadPoolExecutor

from ._compat import suppress_native_stderr

with suppress_native_stderr():
    import cv2
    import numpy as np
    import av

# Suppress verbose FFmpeg log output (info/warning level).
av.logging.set_level(av.logging.FATAL)

from .lt_codec import PRNG, DEFAULT_C, DEFAULT_DELTA, xor_bytes
from .protocol import (
    _resolve_alphanumeric_flag,
    auto_blocksize,
    pack_v3,
)
from .raptorq_codec import RaptorQEncoder
from .display_cache import (
    ModuleFrameCache,
    pack_module_image,
    plan_module_cache,
    unpack_module_frame,
)
from .qr_utils import generate_qr_image, generate_qr_module_image
from .ui import ProgressReporter, QuietReporter


# Prefer mmap-backed random access for larger uncompressed inputs.
_MMAP_THRESHOLD = 10 * 1024 * 1024


class _WriterFailure(RuntimeError):
    """Raised when the background video writer fails."""


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
                 c: float = DEFAULT_C, delta: float = DEFAULT_DELTA,
                 prng_version: int = 1):
        self.data = data
        self.filesize = len(data)
        self.blocksize = blocksize
        self.compressed = compressed
        # ``binary_qr`` and ``alphanumeric_qr`` both map to the same
        # header flag bit (0x02); prefer the alphanumeric_qr name.
        self.alphanumeric_qr = _resolve_alphanumeric_flag(
            binary_qr, alphanumeric_qr)
        # PRNG schema version. 1 = SplitMix64 (default, qrstream ≥
        # 0.8); 0 = legacy LCG warmup (kept so tests / tooling can
        # reproduce old fixtures on demand).
        #
        # TODO(v0.10.0): remove the prng_version kwarg entirely
        # once legacy v0 encode support is dropped. See
        # ``protocol.py`` for the full removal checklist.
        if prng_version not in (0, 1):
            raise ValueError(f"Unsupported prng_version: {prng_version}")
        self.prng_version = prng_version
        self.K = ceil(self.filesize / blocksize)
        self.prng = PRNG(self.K, delta=delta, c=c,
                         prng_version=prng_version)
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
                prng_version=self.prng_version,
            )
            yield packed, seed, seq


def _read_file_bytes(input_path: str) -> bytes:
    with open(input_path, 'rb') as f:
        return f.read()


def _load_payload(input_path: str, compress: bool,
                  force_compress: bool = False,
                  verbose: bool = False,
                  reporter: ProgressReporter | None = None):
    """Load the source payload with a low-memory path when possible.

    Returns (payload, effective_compress, used_mmap, raw_size).
    """
    raw_size = os.path.getsize(input_path)

    if compress:
        if raw_size > _MMAP_THRESHOLD and not force_compress:
            if verbose and reporter is not None:
                reporter.debug(
                    "Compression disabled for large input to keep memory usage low."
                )
            compress = False
        else:
            raw_data = _read_file_bytes(input_path)
            data = zlib.compress(raw_data)
            return data, True, False, raw_size

    if raw_size > _MMAP_THRESHOLD:
        return MmapDataSource(input_path), False, True, raw_size
    return _read_file_bytes(input_path), False, False, raw_size


# Codec map for video output
# PyAV codec mapping: user-facing name → (pyav_codec, pix_fmt, ext, stream_options).
_PYAV_CODEC_MAP = {
    'h264': ('libx264', 'yuv420p', '.mp4',
             {"preset": "ultrafast", "tune": "stillimage", "crf": "23"}),
    'mp4v': ('mpeg4', 'yuv420p', '.mp4', {}),
    'mjpeg': ('mjpeg', 'yuvj420p', '.avi', {"q:v": "2"}),
}

_PYAV_CONTAINER_FORMAT = {
    'h264': 'mp4',
    'mp4v': 'mp4',
    'mjpeg': 'avi',
}


def _warn_if_output_extension_mismatches_codec(
    output_path: str,
    codec: str,
    reporter: ProgressReporter,
) -> None:
    """Warn when the requested filename extension disagrees with the container."""
    codec_info = _PYAV_CODEC_MAP.get(codec)
    if codec_info is None:
        raise ValueError(
            f"Unsupported codec: {codec!r}. "
            f"Choose from: {list(_PYAV_CODEC_MAP)}"
        )

    _, _, expected_ext, _ = codec_info
    actual_ext = os.path.splitext(output_path)[1].lower()
    if not actual_ext or actual_ext == expected_ext:
        return

    reporter.warn(
        f"Output filename extension {actual_ext!r} does not match codec "
        f"{codec!r} (expected {expected_ext!r}). The file will be written "
        f"using the {codec!r} container/codec settings while keeping the "
        f"user-provided path unchanged."
    )


def _resolve_border_modules(qr_version: int, border: float | None) -> float:
    """Resolve CLI/API border input to QR quiet-zone width in modules."""
    if border is None:
        return 4.0
    return round((qr_version - 1) * 4 + 21) * border / 100.0


class _DisplayVideoSink:
    """Best-effort realtime writer for ``--display -o``.

    ``offer`` never blocks display production.  If the bounded queue fills,
    realtime writing is deferred; ``finalize`` then regenerates all missing
    frames and still produces a complete output file.
    """

    def __init__(self, output_path: str, fps: int, codec: str,
                 module_side: int, reporter: ProgressReporter,
                 box_size: int = 10, queue_frames: int | None = None):
        codec_info = _PYAV_CODEC_MAP.get(codec)
        if codec_info is None:
            raise ValueError(
                f"Unsupported codec: {codec!r}. Choose from: {list(_PYAV_CODEC_MAP)}"
            )
        pyav_codec, pix_fmt, _default_ext, stream_opts = codec_info
        container_format = _PYAV_CONTAINER_FORMAT[codec]
        _warn_if_output_extension_mismatches_codec(output_path, codec, reporter)

        parent = os.path.dirname(os.path.abspath(output_path)) or "."
        prefix = f".{os.path.basename(output_path)}."
        tmp = tempfile.NamedTemporaryFile(
            prefix=prefix, suffix=".tmp", dir=parent, delete=False)
        tmp.close()

        self.output_path = output_path
        self.tmp_path = tmp.name
        self.reporter = reporter
        self.module_side = module_side
        self.side = module_side * max(1, int(box_size))
        self.total_written = 0
        self.deferred_from: int | None = None
        self._next_offer = 0
        self._error: list[BaseException] = []
        self._lock = Lock()
        self._queue: Queue = Queue(
            maxsize=max(1, queue_frames or max(128, fps * 4)))
        self._output = av.open(self.tmp_path, "w", format=container_format)
        self._stream = self._output.add_stream(pyav_codec, rate=fps)
        self._stream.width = self.side
        self._stream.height = self.side
        self._stream.pix_fmt = pix_fmt
        if stream_opts:
            self._stream.options = stream_opts
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        self._finalized = False

    def offer(self, frame_index: int, packed_frame) -> bool:
        with self._lock:
            if self.deferred_from is not None or self._error:
                return False
            if frame_index != self._next_offer:
                self.deferred_from = min(frame_index, self._next_offer)
                return False
        try:
            self._queue.put_nowait((frame_index, packed_frame.copy()))
        except Full:
            with self._lock:
                if self.deferred_from is None:
                    self.deferred_from = frame_index
            return False
        with self._lock:
            self._next_offer = frame_index + 1
        return True

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                frame_index, packed_frame = item
                if frame_index != self.total_written:
                    raise RuntimeError(
                        f"video writer expected frame {self.total_written}, "
                        f"got {frame_index}"
                    )
                self._write_packed_frame(packed_frame)
                self.total_written += 1
        except BaseException as exc:
            with self._lock:
                self._error.append(exc)

    def _write_packed_frame(self, packed_frame) -> None:
        self._write_module_image(
            unpack_module_frame(packed_frame, self.module_side))

    def _write_module_image(self, module_img) -> None:
        scaled = cv2.resize(
            module_img,
            (self.side, self.side),
            interpolation=cv2.INTER_NEAREST,
        )
        frame = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        frame_av = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self._stream.encode(frame_av):
            self._output.mux(packet)

    def _finish_realtime(self) -> None:
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except Full:
                if self._error:
                    break
        self._thread.join()
        if self._error:
            raise _WriterFailure("video writer thread failed") from self._error[0]

    def finalize(self, total_frames: int, module_frame_at) -> None:
        try:
            self._finish_realtime()
            start_ts = time.monotonic()
            last_report_ts = start_ts
            for frame_index in range(self.total_written, total_frames):
                module_img = module_frame_at(frame_index)
                self._write_module_image(module_img)
                self.total_written = frame_index + 1
                now = time.monotonic()
                if now - last_report_ts >= 0.1 or self.total_written == total_frames:
                    elapsed = max(1e-6, now - start_ts)
                    self.reporter.encode_update(
                        progress_pct=(self.total_written / total_frames * 100)
                        if total_frames else 100.0,
                        speed_fps=(self.total_written / elapsed),
                        eta_sec=0.0,
                    )
                    last_report_ts = now

            for packet in self._stream.encode():
                self._output.mux(packet)
            self._output.close()
            self._output = None
            os.replace(self.tmp_path, self.output_path)
            self._finalized = True
            output_size = os.path.getsize(self.output_path)
            self.reporter.encode_done(
                output_path=self.output_path, size_bytes=output_size)
        except BaseException:
            self.discard()
            raise

    def discard(self) -> None:
        try:
            self._finish_realtime()
        except Exception:
            pass
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                pass
            self._output = None
        if not self._finalized:
            try:
                os.unlink(self.tmp_path)
            except OSError:
                pass


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
                    codec: str = 'h264',
                    binary_qr: bool = True,
                    alphanumeric_qr: bool | None = None,
                    force_compress: bool = False,
                    auto_mask: bool = False,
                    reporter: ProgressReporter | None = None,
                    fountain_codec: str = 'raptorq'):
    """Encode a file to a QR-code video using fountain codes.

    ``fountain_codec`` selects the erasure code: ``'raptorq'``
    (default, RFC 6330, near-optimal recovery) or ``'lt'`` (legacy
    LT fountain codes with SplitMix64 PRNG).

    .. deprecated:: 0.9
       The ``'lt'`` codec is legacy.  It will be removed in v1.0.0,
       leaving RaptorQ as the sole fountain codec.

    ``binary_qr`` and ``alphanumeric_qr`` are aliases for the
    high-density QR mode flag; prefer ``alphanumeric_qr`` in new code.
    When enabled (default), frames are encoded via base45 into QR
    alphanumeric mode, carrying ~29% more payload per frame than base64.

    ``reporter`` --- optional :class:`qrstream.ui.ProgressReporter` used
    for progress/status rendering.  When ``None`` a :class:`QuietReporter`
    is used so the function stays side-effect-free for programmatic use.

    .. deprecated:: 0.8
        ``ec_level`` is redundant with ``overhead`` in qrstream's
        pipeline and will be removed in v0.10.0.  QR-level Reed-Solomon
        only rescues *bit* errors within a detected frame, but
        WeChatQRCode either decodes a frame's payload or returns
        ``None``; borderline frames are handled by fountain overhead
        at the video level.  The CLI already hides ``--ec-level``; the
        API keyword is retained for one deprecation window.
    """
    if reporter is None:
        reporter = QuietReporter()

    high_density = _resolve_alphanumeric_flag(binary_qr, alphanumeric_qr)
    payload = None
    output = None
    writer_thread = None
    writer_queue: Queue | None = None
    writer_error: list[BaseException] = []
    final_output_path = output_path

    try:
        payload, compress, used_mmap, raw_size = _load_payload(
            input_path,
            compress=compress,
            force_compress=force_compress,
            verbose=verbose,
            reporter=reporter,
        )

        payload_size = len(payload)
        if verbose:
            source_desc = "mmap" if used_mmap else "memory"
            reporter.debug(
                f"Input: {input_path} ({raw_size} bytes, source={source_desc})"
            )
            if compress:
                ratio = payload_size / raw_size * 100 if raw_size else 0.0
                reporter.debug(
                    f"Compressed: {raw_size} → {payload_size} bytes "
                    f"({ratio:.1f}%)"
                )

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
            reporter.debug(
                f"Blocks: K={K}, blocksize={blocksize}, "
                f"total={num_blocks} "
                f"(overhead={overhead}x, {mode_str})"
            )

        if fountain_codec == 'raptorq':
            encoder = RaptorQEncoder(
                payload,
                blocksize,
                compressed=compress,
                alphanumeric_qr=high_density,
            )
        elif fountain_codec == 'lt':
            encoder = LTEncoder(
                payload,
                blocksize,
                compressed=compress,
                alphanumeric_qr=high_density,
            )
        else:
            raise ValueError(
                f"Unsupported fountain_codec: {fountain_codec!r}. "
                f"Choose 'raptorq' or 'lt'."
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
            auto_mask=auto_mask,
        )
        h, w = first_qr.shape[:2]

        if workers is None:
            # Full-pipeline benchmarks show the default encoder path is
            # usually VideoWriter-bound.  QR generation now uses
            # zxing-cpp (native C++, releases the GIL), so
            # ThreadPoolExecutor can provide real parallelism.  However
            # the muxer thread is still the typical bottleneck, so keep
            # the automatic default conservative; users can opt in to
            # higher worker counts explicitly.
            workers = 1
        elif workers > 1:
            reporter.warn(
                "Encoder --workers > 1 is experimental: full encode is "
                "often video-writer-bound, so higher worker counts may "
                "not improve end-to-end performance despite QR generation "
                "itself being GIL-free (zxing-cpp native)."
            )

        if verbose:
            reporter.debug(
                f"QR frame size: {w}x{h}, video FPS: {fps}, workers: {workers}"
            )

        # User-facing encode summary (always shown, even without verbose):
        #   video duration, fps, QR version, QR mode, overhead.
        duration_sec = total_frames / fps if fps else 0.0
        mode_str = "base45" if high_density else "base64"
        reporter.encode_start(
            duration_sec=duration_sec,
            fps=fps,
            qr_version=qr_version,
            mode=mode_str,
            overhead=overhead,
            input_path=input_path,
            file_size=raw_size,
        )

        codec_info = _PYAV_CODEC_MAP.get(codec)
        if codec_info is None:
            raise ValueError(
                f"Unsupported codec: {codec!r}. "
                f"Choose from: {list(_PYAV_CODEC_MAP)}"
            )
        pyav_codec, pix_fmt, _default_ext, stream_opts = codec_info
        container_format = _PYAV_CONTAINER_FORMAT[codec]

        final_output_path = output_path
        if os.path.abspath(input_path) == os.path.abspath(final_output_path):
            raise ValueError(
                f"Output path is the same as the input file: {final_output_path}. "
                "Choose a different output path."
            )

        _warn_if_output_extension_mismatches_codec(
            final_output_path, codec, reporter)

        output = av.open(output_path, "w", format=container_format)
        out_stream = output.add_stream(pyav_codec, rate=fps)
        out_stream.width = w
        out_stream.height = h
        out_stream.pix_fmt = pix_fmt
        if stream_opts:
            out_stream.options = stream_opts

        # ── Encoder/muxer runs on its own thread ────────────────────
        # x264 encode + mux overlaps with QR generation on the main
        # thread so the pipeline stays compute-bound on the slowest
        # stage rather than alternating between them.
        writer_queue: Queue = Queue(maxsize=max(workers * 8, 128))

        def _writer_loop():
            try:
                while True:
                    frame = writer_queue.get()
                    if frame is None:
                        return
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h),
                                           interpolation=cv2.INTER_NEAREST)
                    frame_av = av.VideoFrame.from_ndarray(frame, format="bgr24")
                    for packet in out_stream.encode(frame_av):
                        output.mux(packet)
            except BaseException as exc:
                writer_error.append(exc)

        writer_thread = Thread(target=_writer_loop, daemon=True)
        writer_thread.start()

        if lead_in_frames:
            blank_frame = np.full((h, w, 3), 255, dtype=first_qr.dtype)
            for _ in range(lead_in_frames):
                writer_queue.put(blank_frame)
            if writer_error:
                raise _WriterFailure("video writer thread failed") from writer_error[0]

        batch_size = max(workers * 4, 64)

        # ── Progress tracking ───────────────────────────────────
        produced = 0
        start_ts = time.monotonic()
        last_report_ts = start_ts

        def _report_progress(now: float) -> None:
            nonlocal last_report_ts
            # Rate-limit to ~10 Hz to keep the Rich Live renderer
            # out of the hot path; LogReporter has its own throttle.
            if now - last_report_ts < 0.1 and produced < num_blocks:
                return
            elapsed = max(1e-6, now - start_ts)
            speed = produced / elapsed
            remaining = max(0, num_blocks - produced)
            eta = remaining / speed if speed > 1e-6 else 0.0
            pct = (produced / num_blocks * 100) if num_blocks else 100.0
            reporter.encode_update(
                progress_pct=pct,
                speed_fps=speed,
                eta_sec=eta,
            )
            last_report_ts = now

        if workers > 1:
            block_queue = Queue(maxsize=batch_size * 2)

            def _block_producer():
                encoder._seq = 0
                for packed, _, _ in encoder.generate_blocks(num_blocks):
                    block_queue.put(packed)
                block_queue.put(None)

            producer = Thread(target=_block_producer, daemon=True)
            producer.start()

            with ThreadPoolExecutor(max_workers=workers) as pool:
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
                    #    use_legacy, binary_mode, alphanumeric, auto_mask)
                    qr_imgs = list(pool.map(
                        generate_qr_image, batch,
                        repeat(ec_level), repeat(10), repeat(border_modules),
                        repeat(qr_version), repeat(use_legacy_qr),
                        repeat(None), repeat(high_density),
                        repeat(auto_mask),
                    ))
                    for qr_img in qr_imgs:
                        writer_queue.put(qr_img)
                        if writer_error:
                            raise _WriterFailure("video writer thread failed") from writer_error[0]
                    produced += len(batch)
                    _report_progress(time.monotonic())

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
                    auto_mask=auto_mask,
                )
                writer_queue.put(qr_img)
                if writer_error:
                    raise _WriterFailure("video writer thread failed") from writer_error[0]
                produced += 1
                _report_progress(time.monotonic())

        # Final 100% tick so the bar cleanly lands on 100.
        if num_blocks > 0:
            reporter.encode_update(
                progress_pct=100.0,
                speed_fps=produced / max(1e-6, time.monotonic() - start_ts),
                eta_sec=0.0,
            )

        # Flush writer: signal sentinel and wait for disk writes to drain
        writer_queue.put(None)
        writer_thread.join()
        writer_thread = None
        if writer_error:
            raise _WriterFailure("video writer thread failed") from writer_error[0]

        # Flush encoder: drain remaining frames and close the file.
        for packet in out_stream.encode():
            output.mux(packet)
        output.close()
        output = None  # Prevent double-close in finally block.
    finally:
        # On the exception path, make sure we don't leave the writer
        # thread blocked on an empty queue (daemon=True would let it die
        # with the process, but we still want a clean shutdown attempt).
        if writer_thread is not None and writer_thread.is_alive():
            if writer_queue is not None:
                writer_queue.put(None)
            writer_thread.join(timeout=5)
        if output is not None:
            try:
                output.close()
            except Exception:
                pass
        if payload is not None:
            close = getattr(payload, 'close', None)
            if callable(close):
                close()

    output_size = os.path.getsize(final_output_path)
    reporter.encode_done(output_path=final_output_path, size_bytes=output_size)
    if verbose:
        reporter.debug(
            f"Output: {final_output_path} ({output_size} bytes, "
            f"{total_frames} frames)"
        )


def encode_to_display(input_path: str,
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
                      binary_qr: bool = True,
                      alphanumeric_qr: bool | None = None,
                      force_compress: bool = False,
                      auto_mask: bool = False,
                      reporter: ProgressReporter | None = None,
                      player=None,
                      _player=None,
                      output_path: str | None = None,
                      codec: str = 'h264',
                      video_queue_frames: int | None = None,
                      report_display_done: bool = True,
                      fountain_codec: str = 'raptorq') -> ModuleFrameCache:
    """Encode a file into a display module-frame cache and play it.

    QR frames are rendered at one pixel per module, bit-packed into
    :class:`ModuleFrameCache`, then upscaled by the display player at
    presentation time.  When ``output_path`` is provided, a best-effort
    realtime writer records frames without blocking display; any missing
    suffix is regenerated after the display window closes.
    """
    # Lazy imports — avoid pulling PySide6 when only encode_to_video is used
    # (allows qrstream-headless to work without GUI dependencies).
    from .display_cache import DisplayProducerState  # noqa: F811
    from .display_player_qt import (  # noqa: F811
        DisplayMetadata,
        DisplayPlayerQtConfig,
        play_display_qt,
        require_pyside6,
    )

    if reporter is None:
        reporter = QuietReporter()

    if _player is None:
        _player = player
    if _player is None:
        require_pyside6()

    high_density = _resolve_alphanumeric_flag(binary_qr, alphanumeric_qr)
    payload = None
    producer_thread: Thread | None = None
    producer_error: list[BaseException] = []
    video_sink = None
    cancel_event = Event()

    try:
        payload, compress, used_mmap, raw_size = _load_payload(
            input_path,
            compress=compress,
            force_compress=force_compress,
            verbose=verbose,
            reporter=reporter,
        )

        payload_size = len(payload)
        if verbose:
            source_desc = "mmap" if used_mmap else "memory"
            reporter.debug(
                f"Input: {input_path} ({raw_size} bytes, source={source_desc})"
            )
            if compress:
                ratio = payload_size / raw_size * 100 if raw_size else 0.0
                reporter.debug(
                    f"Compressed: {raw_size} → {payload_size} bytes "
                    f"({ratio:.1f}%)"
                )

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
            reporter.debug(
                f"Blocks: K={K}, blocksize={blocksize}, total={num_blocks} "
                f"(overhead={overhead}x, {mode_str})"
            )

        if fountain_codec == 'raptorq':
            encoder = RaptorQEncoder(
                payload,
                blocksize,
                compressed=compress,
                alphanumeric_qr=high_density,
            )
        elif fountain_codec == 'lt':
            encoder = LTEncoder(
                payload,
                blocksize,
                compressed=compress,
                alphanumeric_qr=high_density,
            )
        else:
            raise ValueError(
                f"Unsupported fountain_codec: {fountain_codec!r}. "
                f"Choose 'raptorq' or 'lt'."
            )
        first_packed, _, _ = next(encoder.generate_blocks(1))
        first_module = generate_qr_module_image(
            first_packed,
            ec_level=ec_level,
            border=border_modules,
            version=qr_version,
            use_legacy=use_legacy_qr,
            alphanumeric=high_density,
            auto_mask=auto_mask,
        )
        module_side = int(first_module.shape[0])
        plan = plan_module_cache(total_frames, module_side, fps)
        cache = ModuleFrameCache.from_plan(total_frames, module_side, plan)
        state = DisplayProducerState(total_frames)
        if output_path is not None:
            if os.path.abspath(input_path) == os.path.abspath(output_path):
                raise ValueError(
                    f"Output path is the same as the input file: {output_path}. "
                    "Choose a different output path."
                )
            video_sink = _DisplayVideoSink(
                output_path=output_path,
                fps=fps,
                codec=codec,
                module_side=module_side,
                reporter=reporter,
                queue_frames=video_queue_frames,
            )

        if workers is None:
            workers = 1
        elif workers > 1:
            reporter.warn(
                "Display --workers > 1 is experimental: QR module "
                "generation can parallelise, but playback still depends on "
                "continuous cache availability."
            )

        duration_sec = total_frames / fps if fps else 0.0
        mode_str = "base45" if high_density else "base64"
        reporter.encode_start(
            duration_sec=duration_sec,
            fps=fps,
            qr_version=qr_version,
            mode=mode_str,
            overhead=overhead,
            input_path=input_path,
            file_size=raw_size,
        )

        def _report_progress(produced: int, start_ts: float,
                             last_report_ts: list[float]) -> None:
            now = time.monotonic()
            if now - last_report_ts[0] < 0.1 and produced < total_frames:
                return
            elapsed = max(1e-6, now - start_ts)
            speed = produced / elapsed
            remaining = max(0, total_frames - produced)
            eta = remaining / speed if speed > 1e-6 else 0.0
            pct = (produced / total_frames * 100) if total_frames else 100.0
            reporter.encode_update(
                progress_pct=pct,
                speed_fps=speed,
                eta_sec=eta,
            )
            last_report_ts[0] = now

        blank_module = np.full((module_side, module_side), 255, dtype=np.uint8)

        # Pre-generate all packed frames for the ``_module_frame_at``
        # fallback path.  LTEncoder can regenerate any block by seed,
        # but RaptorQEncoder requires batch generation; caching the
        # packed bytes unifies both codecs.
        _packed_cache: dict[int, bytes] = {}

        def _ensure_packed_cache():
            """Lazily populate the packed-frame cache if not yet built."""
            if _packed_cache:
                return
            encoder._seq = 0
            for offset, (packed, _, _) in enumerate(
                    encoder.generate_blocks(num_blocks)):
                _packed_cache[offset] = packed

        def _module_frame_at(frame_index: int):
            if frame_index < lead_in_frames:
                return blank_module.copy()
            block_index = frame_index - lead_in_frames
            _ensure_packed_cache()
            packed = _packed_cache.get(block_index)
            if packed is None:
                return blank_module.copy()
            return generate_qr_module_image(
                packed,
                ec_level=ec_level,
                border=border_modules,
                version=qr_version,
                use_legacy=use_legacy_qr,
                alphanumeric=high_density,
                auto_mask=auto_mask,
            )

        def _produce() -> None:
            produced = 0
            start_ts = time.monotonic()
            last_report_ts = [start_ts]
            try:
                if lead_in_frames:
                    packed_blank = pack_module_image(blank_module)
                    for frame_index in range(lead_in_frames):
                        if cancel_event.is_set() or state.cancel_requested():
                            return
                        cache.put_packed(frame_index, packed_blank)
                        if video_sink is not None:
                            video_sink.offer(frame_index, packed_blank)
                        produced += 1
                        state.mark_produced()
                    _report_progress(produced, start_ts, last_report_ts)

                encoder._seq = 0
                if workers > 1:
                    batch_size = max(workers * 4, 64)
                    block_iter = encoder.generate_blocks(num_blocks)
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        frame_base = lead_in_frames
                        while not cancel_event.is_set() and not state.cancel_requested():
                            batch = []
                            for _ in range(batch_size):
                                try:
                                    packed, _, _ = next(block_iter)
                                except StopIteration:
                                    break
                                batch.append(packed)
                            if not batch:
                                break
                            module_imgs = list(pool.map(
                                generate_qr_module_image, batch,
                                repeat(ec_level), repeat(border_modules),
                                repeat(qr_version), repeat(use_legacy_qr),
                                repeat(None), repeat(high_density),
                                repeat(auto_mask),
                            ))
                            for module_img in module_imgs:
                                if cancel_event.is_set() or state.cancel_requested():
                                    return
                                packed_frame = pack_module_image(module_img)
                                cache.put_packed(frame_base, packed_frame)
                                if video_sink is not None:
                                    video_sink.offer(frame_base, packed_frame)
                                frame_base += 1
                                produced += 1
                                state.mark_produced()
                            _report_progress(produced, start_ts, last_report_ts)
                else:
                    for offset, (packed, _, _) in enumerate(
                            encoder.generate_blocks(num_blocks)):
                        if cancel_event.is_set() or state.cancel_requested():
                            return
                        module_img = generate_qr_module_image(
                            packed,
                            ec_level=ec_level,
                            border=border_modules,
                            version=qr_version,
                            use_legacy=use_legacy_qr,
                            alphanumeric=high_density,
                            auto_mask=auto_mask,
                        )
                        frame_index = lead_in_frames + offset
                        packed_frame = pack_module_image(module_img)
                        cache.put_packed(frame_index, packed_frame)
                        if video_sink is not None:
                            video_sink.offer(frame_index, packed_frame)
                        produced += 1
                        state.mark_produced()
                        _report_progress(produced, start_ts, last_report_ts)

                if total_frames > 0:
                    reporter.encode_update(
                        progress_pct=100.0,
                        speed_fps=produced / max(1e-6, time.monotonic() - start_ts),
                        eta_sec=0.0,
                    )
            except BaseException as exc:
                producer_error.append(exc)
            finally:
                cache.mark_done()
                state.mark_done()

        producer_thread = Thread(target=_produce, daemon=True)
        producer_thread.start()
        try:
            if _player is not None:
                _player(cache, state, fps)
            else:
                player_meta = DisplayMetadata(
                    file_name=os.path.basename(input_path),
                    file_size=raw_size,
                    payload_size=payload_size,
                    compressed=compress,
                    data_blocks=int(K),
                    total_blocks=num_blocks,
                    block_size=blocksize,
                    total_frames=total_frames,
                    qr_version=qr_version,
                    ec_level=ec_level,
                    module_side=module_side,
                    fps=fps,
                    high_density=high_density,
                )
                player_config = DisplayPlayerQtConfig(
                    title=f"QRStream — {os.path.basename(input_path)}",
                    metadata=player_meta,
                )
                play_display_qt(cache, state, fps, config=player_config)
        finally:
            cancel_event.set()
            producer_thread.join()
        if producer_error:
            raise producer_error[0]

        if video_sink is not None:
            video_sink.finalize(total_frames, _module_frame_at)
            video_sink = None
        elif (report_display_done and not state.cancel_requested()
                and cache.valid_count >= total_frames):
            reporter.encode_done(output_path="(display)", size_bytes=0)
        return cache
    finally:
        if producer_thread is not None and producer_thread.is_alive():
            cancel_event.set()
            producer_thread.join()
        if video_sink is not None:
            video_sink.discard()
        if payload is not None:
            close = getattr(payload, 'close', None)
            if callable(close):
                close()
