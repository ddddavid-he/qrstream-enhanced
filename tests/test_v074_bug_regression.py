"""
Regression tests for the v0.7.4 encoding bug.

== Root cause of v0.7.4 encoding failure ==

v0.7.4 used ``ProcessPoolExecutor`` (with the OS default ``fork`` start method
on Linux) to parallelise QR image generation inside ``encode_to_video()``.
At the same time, a background ``Thread`` (_block_producer) was running in the
main process feeding encoded blocks into a queue that the pool workers would
convert to QR images.

Forking a multi-threaded process is unsafe:
- The child inherits a snapshot of the parent where only the calling thread
  exists.  Mutexes/locks held by the background thread in the parent are
  frozen in their locked state inside the child.
- Python 3.12+ warns: "This process is multi-threaded, use of fork() may lead
  to deadlocks in the child."
- On CI (Linux, default fork), ~75% of decoded blocks returned from workers
  were corrupt — LT peeling stalled even at high overhead.

v0.7.5 fix: replace the unsafe fork-backed ``ProcessPoolExecutor`` path with
``ThreadPoolExecutor``.  The encoder may now use ``ProcessPoolExecutor`` again,
but only with an explicit ``spawn`` context so workers do not inherit a
multi-threaded parent snapshot.

== Why previous tests did NOT catch this ==

All existing roundtrip tests (test_roundtrip.py, test_lt_codec.py, …) exercise
the pure LT codec layer — ``LTEncoder.generate_blocks()`` + ``LTDecoder`` —
without touching the video I/O path (``encode_to_video`` / ``extract_qr_from_video``).
That layer uses ``ProcessPoolExecutor`` only inside the full video pipeline.
The unit-level tests bypassed the broken code entirely.

== The ``if seed:`` change in lt_codec.py ==

v0.7.5 also changed ``if seed:`` → ``if seed is not None:`` in
``PRNG.get_src_blocks()``.  This is a correctness fix for any future caller
passing seed=0, but it has **zero effect on the current encode/decode
pipeline** because:

1. ``LTEncoder.generate_blocks()`` uses seeds 1, 2, 3, … (never 0).
2. ``LTDecoder.consume_block()`` passes ``header.seed`` which is exactly what
   the encoder wrote (also never 0).

So those tests in the original PR are testing a hypothetical future edge case,
not the real v0.7.4 failure.

== What these tests cover ==

- LT codec roundtrip at all requested sizes (50k–10M) — pure codec, no video
- Full video encode→decode pipeline at small sizes exercising
  ``encode_to_video`` + ``extract_qr_from_video`` directly
- Regression guard: any encoder process pool must use ``spawn`` rather than
  Linux's unsafe default ``fork``
- ``PRNG.get_src_blocks(seed=0)`` correctness (latent bug, not v0.7.4 trigger)
"""

import inspect
import random
import re
from math import ceil
from pathlib import Path

import pytest

import qrstream.encoder as encoder_mod
from qrstream.lt_codec import PRNG, DEFAULT_C, DEFAULT_DELTA
from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_random_bytes(size: int, seed: int = 0xDEADBEEF) -> bytes:
    return random.Random(seed).randbytes(size)


def _roundtrip_lt(data: bytes, overhead: float = 3.5,
                  blocksize: int | None = None,
                  prng_version: int = 1) -> bytes | None:
    """Pure LT encode→decode without QR or video I/O."""
    filesize = len(data)
    if blocksize is None:
        blocksize = max(64, min(512, filesize // 500 + 1))
    K = ceil(filesize / blocksize)
    num_blocks = int(K * overhead)
    encoder = LTEncoder(data, blocksize, prng_version=prng_version)
    decoder = LTDecoder()
    for packed, seed, seq in encoder.generate_blocks(num_blocks):
        done, _ = decoder.decode_bytes(packed)
        if done:
            return decoder.bytes_dump()
    return None


# ─────────────────────────────────────────────────────────────────
# 1. Regression guard: process pools must never use unsafe fork
# ─────────────────────────────────────────────────────────────────

class TestEncoderPoolStrategy:
    """
    The v0.7.4 bug was caused by a fork-backed ``ProcessPoolExecutor`` being
    created while background threads were alive.  The encoder can use processes
    for GIL-bound segno work, but it must force ``spawn``.
    """

    def _source_of(self, module_name: str) -> str:
        import importlib
        mod = importlib.import_module(module_name)
        return inspect.getsource(mod)

    def test_encoder_process_pool_uses_spawn(self):
        src = self._source_of("qrstream.encoder")
        assert "ProcessPoolExecutor" in src, \
            "encoder.py should use ProcessPoolExecutor for GIL-bound segno work"
        assert "get_context(_PROCESS_START_METHOD)" in src
        assert '_PROCESS_START_METHOD = "spawn"' in src

    def test_encoder_default_workers_stay_single_for_fixed_mask(self, monkeypatch):
        monkeypatch.setattr(encoder_mod, "_is_free_threaded_python", lambda: False)
        monkeypatch.setattr(encoder_mod.os, "cpu_count", lambda: 14)
        assert encoder_mod._default_encode_workers(auto_mask=False) == 1

    def test_encoder_default_workers_use_processes_for_auto_mask(self, monkeypatch):
        monkeypatch.setattr(encoder_mod, "_is_free_threaded_python", lambda: False)
        monkeypatch.setattr(encoder_mod.os, "cpu_count", lambda: 14)
        assert encoder_mod._default_encode_workers(auto_mask=True) == 6

    def test_encoder_default_workers_are_capped_for_free_threaded(self, monkeypatch):
        monkeypatch.setattr(encoder_mod, "_is_free_threaded_python", lambda: True)
        monkeypatch.setattr(encoder_mod.os, "cpu_count", lambda: 14)
        assert encoder_mod._default_encode_workers(auto_mask=False) == 8

    def test_encoder_executor_selection_tracks_gil_state(self, monkeypatch):
        monkeypatch.setattr(encoder_mod, "_is_free_threaded_python", lambda: False)
        executor_cls, executor_kwargs, label = encoder_mod._qr_executor_config(4)
        assert executor_cls is encoder_mod.ProcessPoolExecutor
        assert label == "processes/spawn"
        assert executor_kwargs["mp_context"].get_start_method() == "spawn"

        monkeypatch.setattr(encoder_mod, "_is_free_threaded_python", lambda: True)
        executor_cls, executor_kwargs, label = encoder_mod._qr_executor_config(4)
        assert executor_cls is encoder_mod.ThreadPoolExecutor
        assert executor_kwargs == {}
        assert label == "threads"

    def test_decoder_stays_thread_pool_only(self):
        src = self._source_of("qrstream.decoder")
        assert "ThreadPoolExecutor" in src, \
            "decoder.py must use ThreadPoolExecutor"
        assert "ProcessPoolExecutor" not in src, \
            "decoder.py must NOT use ProcessPoolExecutor"


# ─────────────────────────────────────────────────────────────────
# 2. Full pipeline: encode_to_video + extract_qr_from_video
#    (exercises the code path that was broken in v0.7.4)
# ─────────────────────────────────────────────────────────────────

class TestVideoRoundtrip:
    """
    End-to-end video encode→decode pipeline.

    v0.7.4 used ProcessPoolExecutor here, causing corrupt QR frames when
    workers > 1 on Linux (fork + live background thread = unsafe).

    We test with workers=2 explicitly to catch any regression.
    """

    @pytest.fixture
    def tmp_video(self, tmp_path):
        return str(tmp_path / "test.mp4")

    def _video_roundtrip(self, data: bytes, tmp_path: Path,
                         workers: int = 2) -> bytes | None:
        from qrstream.encoder import encode_to_video
        from qrstream.decoder import extract_qr_from_video, decode_blocks

        input_path = str(tmp_path / "input.bin")
        output_path = str(tmp_path / "output.mp4")
        Path(input_path).write_bytes(data)

        encode_to_video(
            input_path, output_path,
            overhead=3.0, fps=10, ec_level=1, qr_version=15,
            compress=False, verbose=False, workers=workers,
        )

        blocks = extract_qr_from_video(output_path, verbose=False, workers=workers)
        return decode_blocks(blocks, verbose=False)

    @pytest.mark.parametrize("size_kb", [20, 50])
    def test_video_roundtrip_multiworker(self, size_kb, tmp_path):
        """Multi-worker video encode+decode — exercises the encoder pool path."""
        data = _make_random_bytes(size_kb * 1024, seed=size_kb)
        result = self._video_roundtrip(data, tmp_path, workers=2)
        assert result is not None, \
            f"Video decode stalled at {size_kb}KB (workers=2) — encoder pool regression?"
        assert result == data, f"Data mismatch at {size_kb}KB"

    def test_video_roundtrip_single_worker(self, tmp_path):
        """Single worker (workers=1) also worked in v0.7.4 — must keep working."""
        data = _make_random_bytes(20 * 1024, seed=0xABCD)
        result = self._video_roundtrip(data, tmp_path, workers=1)
        assert result is not None, "Video decode stalled with workers=1"
        assert result == data


# ─────────────────────────────────────────────────────────────────
# 3. LT codec roundtrip at all requested sizes
#    (pure codec layer, no video I/O)
# ─────────────────────────────────────────────────────────────────

class TestRoundtripMultiSize:
    """
    LT encode→decode roundtrip at sizes 50k / 100k / 500k / 1M / 2M / 10M.

    These tests bypass video I/O so they are fast. They verify that the
    core LT peeling graph is correct across all scales — if any future
    regression breaks the codec layer, these will catch it.

    Note: these tests were NOT sufficient to detect the v0.7.4 bug because
    that bug lived in the video pipeline (ProcessPoolExecutor), not here.
    """

    @pytest.mark.parametrize("size_kb", [50, 100, 500, 1000, 2000])
    def test_roundtrip_kb(self, size_kb):
        data = _make_random_bytes(size_kb * 1024, seed=size_kb)
        result = _roundtrip_lt(data)
        assert result is not None, \
            f"LT decode stalled at {size_kb}KB"
        assert result == data

    def test_roundtrip_10mb(self):
        data = _make_random_bytes(10 * 1024 * 1024, seed=0x10_0000)
        result = _roundtrip_lt(data, overhead=2.5)
        assert result is not None, "LT decode stalled at 10MB"
        assert result == data

    @pytest.mark.parametrize("size_kb", [50, 100, 500, 1000])
    def test_roundtrip_prng_v0(self, size_kb):
        """Legacy prng_version=0 path."""
        data = _make_random_bytes(size_kb * 1024, seed=size_kb + 0x100)
        result = _roundtrip_lt(data, prng_version=0, overhead=4.0)
        assert result is not None, \
            f"LT decode stalled at {size_kb}KB (prng_v0)"
        assert result == data

    @pytest.mark.parametrize("size_kb,blocksize", [
        (50,   64),
        (100,  128),
        (500,  256),
        (1000, 512),
        (2000, 512),
    ])
    def test_roundtrip_fixed_blocksize(self, size_kb, blocksize):
        data = _make_random_bytes(size_kb * 1024, seed=size_kb ^ blocksize)
        result = _roundtrip_lt(data, blocksize=blocksize, overhead=3.5)
        assert result is not None, \
            f"LT decode stalled {size_kb}KB / bs={blocksize}"
        assert result == data


# ─────────────────────────────────────────────────────────────────
# 4. Encoder determinism across scales
# ─────────────────────────────────────────────────────────────────

class TestEncoderDeterminism:
    """Encode the same data twice, verify identical block streams."""

    @pytest.mark.parametrize("size_kb", [50, 100, 500, 1000, 2000])
    def test_deterministic_kb(self, size_kb):
        data = _make_random_bytes(size_kb * 1024, seed=size_kb + 0x200)
        blocksize = max(64, data.__len__() // 500 + 1)
        blocks_a = list(LTEncoder(data, blocksize).generate_blocks(20))
        blocks_b = list(LTEncoder(data, blocksize).generate_blocks(20))
        assert blocks_a == blocks_b, \
            f"Non-deterministic encoder at {size_kb}KB"

    def test_deterministic_10mb(self):
        data = _make_random_bytes(10 * 1024 * 1024, seed=0xC0FFEE)
        blocks_a = list(LTEncoder(data, 512).generate_blocks(50))
        blocks_b = list(LTEncoder(data, 512).generate_blocks(50))
        assert blocks_a == blocks_b


# ─────────────────────────────────────────────────────────────────
# 5. Latent PRNG bug (if seed: vs if seed is not None:)
#    Not the v0.7.4 trigger, but a real correctness issue for seed=0.
# ─────────────────────────────────────────────────────────────────

class TestPRNGSeedZero:
    """
    ``get_src_blocks(seed=0)`` correctness.

    The v0.7.4 code used ``if seed:`` which silently skips resetting state
    when seed==0.  This does NOT affect the current encoder/decoder (seeds
    start at 1), but would break any caller passing seed=0 explicitly.
    The fix (``if seed is not None:``) is already in v0.7.5.
    """

    def test_seed_zero_is_deterministic(self):
        """seed=0 must give the same result regardless of prior PRNG state."""
        p = PRNG(K=200, delta=DEFAULT_DELTA, c=DEFAULT_C, prng_version=1)
        p.get_src_blocks(seed=12345)               # pollute state
        _, d1, s1 = p.get_src_blocks(seed=0)

        p2 = PRNG(K=200, delta=DEFAULT_DELTA, c=DEFAULT_C, prng_version=1)
        _, d2, s2 = p2.get_src_blocks(seed=0)      # clean state

        assert (d1, s1) == (d2, s2), \
            "seed=0 result depends on prior state — if seed: bug is back"

    def test_seed_zero_differs_from_seed_one(self):
        p = PRNG(K=200, delta=DEFAULT_DELTA, c=DEFAULT_C, prng_version=1)
        _, _, s0 = p.get_src_blocks(seed=0)
        _, _, s1 = p.get_src_blocks(seed=1)
        assert s0 != s1

    def test_encoder_seeds_never_zero(self):
        """Current encoder must not produce seed=0 — confirms the safe range."""
        data = b'\xAB' * 256
        seeds = [s for _, s, _ in LTEncoder(data, 64).generate_blocks(10)]
        assert all(s >= 1 for s in seeds), f"Encoder produced seed < 1: {seeds}"
        assert seeds == list(range(1, 11))


# ─────────────────────────────────────────────────────────────────
# 6. Overhead sensitivity
# ─────────────────────────────────────────────────────────────────

class TestOverheadSensitivity:
    @pytest.mark.parametrize("overhead", [2.0, 2.5, 3.0, 4.0])
    def test_100kb_overhead_variants(self, overhead):
        data = _make_random_bytes(100 * 1024, seed=int(overhead * 100))
        result = _roundtrip_lt(data, overhead=overhead, blocksize=128)
        assert result is not None, f"Stalled 100KB overhead={overhead}"
        assert result == data

    @pytest.mark.parametrize("overhead", [2.5, 3.0])
    def test_1mb_overhead_variants(self, overhead):
        data = _make_random_bytes(1024 * 1024, seed=int(overhead * 1000))
        result = _roundtrip_lt(data, overhead=overhead, blocksize=256)
        assert result is not None, f"Stalled 1MB overhead={overhead}"
        assert result == data
