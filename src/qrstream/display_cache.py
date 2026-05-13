"""Compact frame caches for ``qrs encode --display``."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import ceil
from threading import Condition, RLock

import numpy as np


DEFAULT_MODULE_CACHE_SOFT_LIMIT = 128 * 1024 * 1024
DEFAULT_MODULE_CACHE_ONE_HOUR_LIMIT = 192 * 1024 * 1024
DEFAULT_PRESENTATION_CACHE_BUDGET = 64 * 1024 * 1024
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


class PresentationFrameCache:
    """Small LRU cache for playback-sized display frames."""

    def __init__(self, budget_bytes: int = DEFAULT_PRESENTATION_CACHE_BUDGET):
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")
        self.budget_bytes = int(budget_bytes)
        self._frames: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._bytes = 0

    @property
    def current_bytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        self._frames.clear()
        self._bytes = 0

    def get(self, key: tuple[int, int]) -> np.ndarray | None:
        frame = self._frames.get(key)
        if frame is None:
            return None
        self._frames.move_to_end(key)
        return frame.copy()

    def put(self, key: tuple[int, int], frame: np.ndarray) -> None:
        arr = np.ascontiguousarray(frame)
        size = int(arr.nbytes)
        if self.budget_bytes <= 0 or size > self.budget_bytes:
            return
        old = self._frames.pop(key, None)
        if old is not None:
            self._bytes -= int(old.nbytes)
        self._frames[key] = arr.copy()
        self._bytes += size
        while self._bytes > self.budget_bytes and self._frames:
            _, evicted = self._frames.popitem(last=False)
            self._bytes -= int(evicted.nbytes)
