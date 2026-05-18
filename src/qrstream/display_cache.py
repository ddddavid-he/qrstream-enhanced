"""Compact frame caches for ``qrs encode --display``."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Condition, Event, RLock
import time

import numpy as np


DEFAULT_MODULE_CACHE_SOFT_LIMIT = 128 * 1024 * 1024
DEFAULT_MODULE_CACHE_ONE_HOUR_LIMIT = 192 * 1024 * 1024
DEFAULT_CACHE_ONE_HOUR_SECONDS = 3600


@dataclass(frozen=True)
class ModuleCachePlan:
    """Sizing decision for the bit-packed module-frame cache."""

    mode: str
    total_bytes: int
    frame_bytes: int
    memory_budget_bytes: int


def module_row_bytes(module_side: int) -> int:
    if module_side <= 0:
        raise ValueError("module_side must be positive")
    return (module_side + 7) // 8


def estimate_module_cache_bytes(total_frames: int, module_side: int) -> int:
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    return total_frames * module_side * module_row_bytes(module_side)


def plan_module_cache(total_frames: int, module_side: int, fps: int,
                      soft_limit_bytes: int = DEFAULT_MODULE_CACHE_SOFT_LIMIT,
                      one_hour_limit_bytes: int = DEFAULT_MODULE_CACHE_ONE_HOUR_LIMIT,
                      one_hour_seconds: int = DEFAULT_CACHE_ONE_HOUR_SECONDS
                      ) -> ModuleCachePlan:
    """Choose full or windowed module cache using the documented heuristics."""
    total_bytes = estimate_module_cache_bytes(total_frames, module_side)
    frame_bytes = module_side * module_row_bytes(module_side)
    duration = total_frames / fps if fps > 0 else float("inf")
    if total_bytes <= soft_limit_bytes:
        return ModuleCachePlan("full", total_bytes, frame_bytes, total_bytes)
    if duration <= one_hour_seconds and total_bytes <= one_hour_limit_bytes:
        return ModuleCachePlan("full", total_bytes, frame_bytes, total_bytes)
    return ModuleCachePlan("window", total_bytes, frame_bytes, soft_limit_bytes)


def pack_module_image(module_img: np.ndarray) -> np.ndarray:
    """Pack a 0/255 module image into one bit per module.

    Black modules are stored as bit ``1`` and white modules as bit ``0``.
    """
    arr = np.asarray(module_img)
    if arr.ndim != 2:
        raise ValueError("module image must be a 2D array")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError("module image must be non-empty")
    black = arr == 0
    packed = np.packbits(black, axis=1, bitorder="big")
    return np.ascontiguousarray(packed, dtype=np.uint8)


def unpack_module_frame(packed: np.ndarray, module_side: int) -> np.ndarray:
    """Unpack a bit-packed module frame into a 0/255 grayscale image."""
    arr = np.asarray(packed, dtype=np.uint8)
    expected_shape = (module_side, module_row_bytes(module_side))
    if arr.shape != expected_shape:
        raise ValueError(
            f"packed frame shape {arr.shape} does not match {expected_shape}"
        )
    bits = np.unpackbits(arr, axis=1, count=module_side, bitorder="big")
    return np.where(bits, 0, 255).astype(np.uint8)


class DisplayProducerState:
    """Thread-safe producer progress shared with display players."""

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self._lock = RLock()
        self._done = Event()
        self._cancel = Event()
        self._started = time.monotonic()
        self._produced = 0
        self._samples: deque[tuple[float, int]] = deque()

    def mark_produced(self, count: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            self._produced += count
            self._samples.append((now, self._produced))
            self._trim_samples(now, 10.0)

    def mark_done(self) -> None:
        self._done.set()

    def request_cancel(self) -> None:
        self._cancel.set()

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait_done(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    @property
    def produced(self) -> int:
        with self._lock:
            return self._produced

    @property
    def progress_pct(self) -> float:
        if self.total_frames <= 0:
            return 100.0
        return min(100.0, self.produced / self.total_frames * 100.0)

    def producer_fps(self, window_seconds: float = 3.0) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim_samples(now, max(window_seconds, 0.1))
            if len(self._samples) >= 2:
                first_ts, first_count = self._samples[0]
                last_ts, last_count = self._samples[-1]
                elapsed = max(1e-6, last_ts - first_ts)
                return max(0.0, (last_count - first_count) / elapsed)
            elapsed = max(1e-6, now - self._started)
            return self._produced / elapsed

    def _trim_samples(self, now: float, window_seconds: float) -> None:
        cutoff = now - window_seconds
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()


class ModuleFrameCache:
    """Thread-safe chunked cache for bit-packed QR module frames."""

    def __init__(self, total_frames: int, module_side: int,
                 memory_budget_bytes: int | None = None,
                 chunk_size: int = 256,
                 mode: str = "full"):
        if total_frames < 0:
            raise ValueError("total_frames must be non-negative")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if mode not in {"full", "window"}:
            raise ValueError("mode must be 'full' or 'window'")

        self.total_frames = total_frames
        self.module_side = module_side
        self.row_bytes = module_row_bytes(module_side)
        self.frame_bytes = module_side * self.row_bytes
        self.chunk_size = chunk_size
        self.mode = mode
        self.memory_budget_bytes = (
            estimate_module_cache_bytes(total_frames, module_side)
            if memory_budget_bytes is None else int(memory_budget_bytes)
        )

        chunk_bytes = self.chunk_size * self.frame_bytes
        self._max_chunks = max(1, self.memory_budget_bytes // max(1, chunk_bytes))
        self._chunks: OrderedDict[int, np.ndarray] = OrderedDict()
        self._valid = bytearray(total_frames)
        self._valid_count = 0
        self._done = False
        self._lock = RLock()
        self._condition = Condition(self._lock)

    @classmethod
    def from_plan(cls, total_frames: int, module_side: int,
                  plan: ModuleCachePlan, chunk_size: int = 256
                  ) -> "ModuleFrameCache":
        return cls(
            total_frames=total_frames,
            module_side=module_side,
            memory_budget_bytes=plan.memory_budget_bytes,
            chunk_size=chunk_size,
            mode=plan.mode,
        )

    @property
    def valid_count(self) -> int:
        with self._lock:
            return self._valid_count

    @property
    def cached_bytes(self) -> int:
        with self._lock:
            return len(self._chunks) * self.chunk_size * self.frame_bytes

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= self.total_frames:
            raise IndexError("frame index out of range")

    def _chunk_bounds(self, chunk_index: int) -> tuple[int, int]:
        start = chunk_index * self.chunk_size
        end = min(self.total_frames, start + self.chunk_size)
        return start, end

    def _drop_chunk_validity(self, chunk_index: int) -> None:
        start, end = self._chunk_bounds(chunk_index)
        for frame_index in range(start, end):
            if self._valid[frame_index]:
                self._valid[frame_index] = 0
                self._valid_count -= 1

    def _evict_if_needed(self, protected_chunk: int) -> None:
        if self.mode != "window":
            return
        while len(self._chunks) > self._max_chunks:
            old_chunk, _ = next(iter(self._chunks.items()))
            if old_chunk == protected_chunk and len(self._chunks) > 1:
                self._chunks.move_to_end(old_chunk)
                old_chunk, _ = next(iter(self._chunks.items()))
            self._chunks.pop(old_chunk)
            self._drop_chunk_validity(old_chunk)

    def _chunk_for_write(self, index: int) -> tuple[np.ndarray, int]:
        chunk_index = index // self.chunk_size
        local_index = index % self.chunk_size
        chunk = self._chunks.get(chunk_index)
        if chunk is None:
            chunk = np.zeros(
                (self.chunk_size, self.module_side, self.row_bytes),
                dtype=np.uint8,
            )
            self._chunks[chunk_index] = chunk
        self._chunks.move_to_end(chunk_index)
        self._evict_if_needed(protected_chunk=chunk_index)
        return chunk, local_index

    def put_packed(self, index: int, packed: np.ndarray) -> None:
        packed_arr = np.asarray(packed, dtype=np.uint8)
        expected_shape = (self.module_side, self.row_bytes)
        if packed_arr.shape != expected_shape:
            raise ValueError(
                f"packed frame shape {packed_arr.shape} does not match "
                f"{expected_shape}"
            )
        with self._condition:
            self._check_index(index)
            chunk, local_index = self._chunk_for_write(index)
            if not self._valid[index]:
                self._valid[index] = 1
                self._valid_count += 1
            chunk[local_index] = packed_arr
            self._condition.notify_all()

    def put_module_image(self, index: int, module_img: np.ndarray) -> None:
        self.put_packed(index, pack_module_image(module_img))

    def has_frame(self, index: int) -> bool:
        with self._lock:
            if index < 0 or index >= self.total_frames:
                return False
            return bool(self._valid[index])

    def get_packed(self, index: int) -> np.ndarray | None:
        with self._lock:
            if index < 0 or index >= self.total_frames or not self._valid[index]:
                return None
            chunk_index = index // self.chunk_size
            chunk = self._chunks.get(chunk_index)
            if chunk is None:
                return None
            self._chunks.move_to_end(chunk_index)
            return chunk[index % self.chunk_size].copy()

    def get_module_image(self, index: int) -> np.ndarray | None:
        packed = self.get_packed(index)
        if packed is None:
            return None
        return unpack_module_frame(packed, self.module_side)

    def contiguous_from(self, start_index: int) -> int:
        with self._lock:
            if start_index < 0 or start_index >= self.total_frames:
                return 0
            count = 0
            for index in range(start_index, self.total_frames):
                if not self._valid[index]:
                    break
                count += 1
            return count

    def wait_for_frame(self, index: int, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._done or self.has_frame(index), timeout=timeout)

    def mark_done(self) -> None:
        with self._condition:
            self._done = True
            self._condition.notify_all()

    def is_done(self) -> bool:
        with self._lock:
            return self._done


# ── Shared-memory frame buffer for subprocess producer ──────────


class SharedFrameBuffer:
    """Process-safe flat frame buffer backed by shared memory.

    Layout::

        ┌──────────────────────────────────────┐
        │ valid_flags: uint8[total_frames]     │
        ├──────────────────────────────────────┤
        │ frame_data: uint8[total_frames       │
        │             × module_side × row_bytes]│
        └──────────────────────────────────────┘

    The producer subprocess writes packed frames and sets validity
    flags.  The GUI process reads them without any locking — single-
    byte flag writes are atomic on all supported platforms, and the
    producer writes frame data *before* setting the flag to 1.
    """

    def __init__(self, total_frames: int, module_side: int,
                 name: str | None = None):
        from multiprocessing.shared_memory import SharedMemory

        self.total_frames = total_frames
        self.module_side = module_side
        self.row_bytes = module_row_bytes(module_side)
        self.frame_bytes = module_side * self.row_bytes
        self._flags_size = total_frames
        self._data_size = total_frames * self.frame_bytes
        total_size = max(1, self._flags_size + self._data_size)

        if name is None:
            self._shm = SharedMemory(create=True, size=total_size)
            self._owner = True
            self._shm.buf[:self._flags_size] = b'\x00' * self._flags_size
        else:
            self._shm = SharedMemory(name=name, create=False)
            self._owner = False

    @property
    def name(self) -> str:
        return self._shm.name

    def put_packed(self, index: int, packed: np.ndarray) -> None:
        """Write a packed frame (called from producer subprocess)."""
        if index < 0 or index >= self.total_frames:
            raise IndexError(
                f"frame index {index} out of range [0, {self.total_frames})"
            )
        flat = np.asarray(packed, dtype=np.uint8).ravel()
        if flat.nbytes != self.frame_bytes:
            raise ValueError(
                f"packed frame size {flat.nbytes} does not match "
                f"expected {self.frame_bytes} bytes "
                f"(module_side={self.module_side}, row_bytes={self.row_bytes})"
            )
        offset = self._flags_size + index * self.frame_bytes
        self._shm.buf[offset:offset + self.frame_bytes] = flat.tobytes()
        # Set flag *after* data is fully written.
        self._shm.buf[index] = 1

    def has_frame(self, index: int) -> bool:
        if index < 0 or index >= self.total_frames:
            return False
        return bool(self._shm.buf[index])

    def get_packed(self, index: int) -> np.ndarray | None:
        if index < 0 or index >= self.total_frames:
            return None
        if not self._shm.buf[index]:
            return None
        offset = self._flags_size + index * self.frame_bytes
        raw = bytes(self._shm.buf[offset:offset + self.frame_bytes])
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            self.module_side, self.row_bytes)

    def close(self) -> None:
        self._shm.close()
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


class SharedProducerState:
    """Process-safe producer state using multiprocessing primitives.

    Drop-in replacement for :class:`DisplayProducerState` when the
    producer runs in a subprocess.
    """

    def __init__(self, total_frames: int):
        import multiprocessing as _mp
        self.total_frames = total_frames
        self._produced = _mp.Value('i', 0)
        self._done = _mp.Event()
        self._cancel = _mp.Event()
        self._started = _mp.Value('d', 0.0)
        # Error flag: 0 = ok, 1 = producer failed.
        self._error_flag = _mp.Value('i', 0)
        # Truncated error message (up to 512 bytes).
        self._error_msg = _mp.Array('c', 512)

    def mark_error(self, message: str) -> None:
        """Record an error from the producer subprocess."""
        self._error_flag.value = 1
        encoded = message.encode('utf-8', errors='replace')[:511]
        self._error_msg[:len(encoded)] = encoded

    def has_error(self) -> bool:
        return bool(self._error_flag.value)

    def get_error(self) -> str | None:
        if not self._error_flag.value:
            return None
        raw = bytes(self._error_msg).split(b'\x00', 1)[0]
        return raw.decode('utf-8', errors='replace')

    def mark_produced(self, count: int = 1) -> None:
        with self._produced.get_lock():
            self._produced.value += count

    def mark_done(self) -> None:
        self._done.set()

    def request_cancel(self) -> None:
        self._cancel.set()

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait_done(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    @property
    def produced(self) -> int:
        return self._produced.value

    @property
    def progress_pct(self) -> float:
        if self.total_frames <= 0:
            return 100.0
        return min(100.0, self.produced / self.total_frames * 100.0)

    def producer_fps(self, window_seconds: float = 3.0) -> float:
        """Approximate producer fps from total produced / elapsed."""
        started = self._started.value
        if started <= 0:
            return 0.0
        elapsed = max(1e-6, time.monotonic() - started)
        return max(0.0, self.produced / elapsed)


class SharedBufferCacheAdapter:
    """Read-only cache interface over :class:`SharedFrameBuffer`.

    Provides the same duck-typed interface that
    :class:`_QRStreamWindow` expects from :class:`ModuleFrameCache`,
    so the Qt player works unchanged regardless of whether the
    producer is a thread or a subprocess.
    """

    def __init__(self, buf: SharedFrameBuffer,
                 state: SharedProducerState):
        self._buf = buf
        self._state = state
        self.total_frames = buf.total_frames
        self.module_side = buf.module_side
        self.row_bytes = buf.row_bytes
        self.frame_bytes = buf.frame_bytes
        self.mode = "full"

    @property
    def valid_count(self) -> int:
        return self._state.produced

    def has_frame(self, index: int) -> bool:
        return self._buf.has_frame(index)

    def get_packed(self, index: int) -> np.ndarray | None:
        return self._buf.get_packed(index)

    def get_module_image(self, index: int) -> np.ndarray | None:
        packed = self.get_packed(index)
        if packed is None:
            return None
        return unpack_module_frame(packed, self.module_side)

    def contiguous_from(self, start_index: int) -> int:
        if start_index < 0 or start_index >= self.total_frames:
            return 0
        count = 0
        for i in range(start_index, self.total_frames):
            if not self._buf.has_frame(i):
                break
            count += 1
        return count

    def is_done(self) -> bool:
        return self._state.is_done()

    def mark_done(self) -> None:
        self._state.mark_done()
