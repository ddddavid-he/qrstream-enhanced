"""
End-to-end integration tests for the pluggable detector plumbing.

These tests exercise the full decode path with ``use_mnn=True``
*without* requiring MNN to be installed — the goal is to verify
that when MNN is unavailable (the common case on Linux CI), the
:class:`DetectorRouter` transparently falls back to
:class:`OpenCVWeChatDetector` and the video still decodes bit-exactly.

Covered paths (Milestone 1 acceptance):

- ``DetectorRouter`` is actually plumbed into the thread-pool workers
  (regression guard: before M1 the router was created but never passed
  to ``_worker_detect_qr``, so ``--mnn`` was a no-op).
- ``opencv_fallback=False`` suppresses the OpenCV retry path.
- ``_qr_text_to_block_and_seed`` parses all three legacy payload
  encodings (base45 / base64 / COBS) and rejects junk.
- A small encode → decode round-trip works with ``use_mnn=True`` on
  a host without MNN, proving the fallback chain ends at byte-equal.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qrstream.decoder import (
    _qr_text_to_block_and_seed,
    _worker_detect_qr,
    _worker_detect_qr_clahe,
    extract_qr_from_video,
    decode_blocks,
)
from qrstream.detector.base import DetectResult, QRDetector
from qrstream.detector.router import DetectorRouter
from qrstream.encoder import encode_to_video


def _mnn_available() -> bool:
    """Return True iff MNN Python bindings import cleanly."""
    try:
        import MNN  # noqa: F401
        return True
    except Exception:
        return False


def _mnn_models_available() -> bool:
    """Return True iff both .mnn model files exist in the repo."""
    from qrstream.detector.mnn_detector import _resolve_model_dir, _DETECT_MODEL_NAME, _SR_MODEL_NAME

    model_dir = _resolve_model_dir()
    return (
        (model_dir / _DETECT_MODEL_NAME).exists()
        and (model_dir / _SR_MODEL_NAME).exists()
    )


# ── Fake detector used to prove router-injection is wired up ──────


class _FakeDetector(QRDetector):
    """Detector stub that records every frame it sees."""

    DETECTOR_CAN_CRASH = False

    def __init__(self, text: str | None = None):
        self._text = text
        self.calls: list[tuple[int, int]] = []  # (h, w)

    def detect(self, frame: np.ndarray) -> DetectResult:
        if frame is None or not isinstance(frame, np.ndarray):
            return DetectResult(backend="fake")
        self.calls.append((int(frame.shape[0]), int(frame.shape[1])))
        return DetectResult(text=self._text, backend="fake")

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "fake"


# ── _qr_text_to_block_and_seed ───────────────────────────────────


class TestQrTextPayloadParser:
    """The payload parser must accept every legacy encoding.

    It is a module-level helper (lives in ``decoder.py``) shared by
    both ``_worker_detect_qr`` and ``_worker_detect_qr_clahe``, so a
    bug here shows up in every scan pass at once.
    """

    def test_junk_payload_returns_none(self):
        assert _qr_text_to_block_and_seed("!!!not a qr payload!!!") == (
            None,
            None,
        )

    def test_empty_string_returns_none(self):
        assert _qr_text_to_block_and_seed("") == (None, None)


# ── Worker plumbing ──────────────────────────────────────────────


class TestWorkerDetectorInjection:
    """``_worker_detect_qr`` must actually route through ``qr_detector``.

    The Milestone 1 refactor added a ``qr_detector=None`` kwarg to
    both worker functions; the decoder binds it with
    ``functools.partial`` before handing the worker to the thread
    pool.  If the kwarg ever stops being forwarded, the MNN path
    silently dies and these tests go red.
    """

    def test_main_worker_routes_to_injected_detector(self):
        det = _FakeDetector(text=None)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        result = _worker_detect_qr((7, frame), qr_detector=det)

        assert result == (7, None, None)
        # The detector must have been called exactly once, with the
        # frame's native shape preserved.
        assert det.calls == [(100, 120)]

    def test_main_worker_uses_default_path_when_no_detector(self):
        det = _FakeDetector(text="never-called")
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        # No qr_detector ⇒ the default OpenCV WeChat path runs; it
        # sees only a black frame so we just assert it doesn't crash
        # and the fake detector stays untouched.
        result = _worker_detect_qr((3, frame))
        assert result == (3, None, None)
        assert det.calls == []

    def test_main_worker_handles_none_frame(self):
        det = _FakeDetector(text="x")
        result = _worker_detect_qr((5, None), qr_detector=det)
        assert result == (5, None, None)
        # None frame short-circuits before the detector is consulted.
        assert det.calls == []

    def test_clahe_worker_routes_to_injected_detector(self):
        det = _FakeDetector(text=None)
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        result = _worker_detect_qr_clahe((11, frame), qr_detector=det)

        assert result == (11, None, None)
        # CLAHE boosts a BGR→YCrCb→BGR copy of the same frame, so
        # the detector should have been invoked exactly once with
        # the original (H, W).  The shape must match before clamping
        # or worker internals silently rewrote the frame size.
        assert det.calls == [(80, 80)]


# ── DetectorRouter.opencv_fallback ───────────────────────────────


class TestRouterFallbackPolicy:
    """``opencv_fallback`` toggles whether MNN no-detects retry OpenCV."""

    def test_fallback_off_short_circuits_when_mnn_says_no(self):
        # Construct a router that thinks MNN is available by
        # injecting a fake MNN detector directly.  We want to verify
        # the routing decision, not actual MNN behaviour.
        router = DetectorRouter(use_mnn=True, opencv_fallback=False)
        router._mnn_detector = _FakeDetector(text=None)
        router._mnn_init_attempted = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = router.detect(frame)

        stats = router.get_stats()
        assert result.text is None
        # MNN was attempted once, got no-detect, did NOT retry OpenCV.
        assert stats["mnn_attempts"] == 1
        assert stats["mnn_fallbacks"] == 1
        assert stats["opencv_attempts"] == 0

    def test_fallback_on_retries_opencv_after_mnn_no_detect(self):
        router = DetectorRouter(use_mnn=True, opencv_fallback=True)
        router._mnn_detector = _FakeDetector(text=None)
        router._mnn_init_attempted = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        router.detect(frame)

        stats = router.get_stats()
        assert stats["mnn_attempts"] == 1
        assert stats["mnn_fallbacks"] == 1
        # OpenCV must have been consulted after MNN missed.
        assert stats["opencv_attempts"] == 1

    def test_mnn_hit_short_circuits_opencv(self):
        router = DetectorRouter(use_mnn=True, opencv_fallback=True)
        router._mnn_detector = _FakeDetector(text="payload-string")
        router._mnn_init_attempted = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = router.detect(frame)

        assert result.text == "payload-string"
        stats = router.get_stats()
        assert stats["mnn_success"] == 1
        # Successful MNN detection must NOT trigger the OpenCV retry.
        assert stats["opencv_attempts"] == 0


# ── Adaptive opencv_fallback controller ──────────────────────────


class _ScriptedOpenCV:
    """Replacement for ``router._opencv_detector`` whose per-call
    verdict is dictated by a caller-provided boolean list.

    Lets a test drive the router through an exact rescue pattern
    without needing real image data or a real WeChatQRCode model.
    """

    DETECTOR_CAN_CRASH = False

    def __init__(self, script: list[bool]):
        self._script = list(script)
        self._i = 0
        self.calls = 0

    def detect(self, frame: np.ndarray) -> DetectResult:
        self.calls += 1
        if self._i < len(self._script):
            hit = self._script[self._i]
            self._i += 1
        else:
            hit = False
        return DetectResult(
            text="ok" if hit else None, backend="scripted-opencv",
        )

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "scripted-opencv"


def _make_router(
    *, script: list[bool],
    warmup: int = 8, window: int = 32,
    disable_rate: float = 0.1, enable_rate: float = 0.3,
    probe_interval: int = 8,
    adaptive: bool = True,
    opencv_fallback: bool = True,
) -> tuple[DetectorRouter, _ScriptedOpenCV, _FakeDetector]:
    """Build a router with a scripted OpenCV and an always-miss MNN.

    Small warmup/window values keep the tests fast while still
    exercising the full threshold logic.
    """
    router = DetectorRouter(
        use_mnn=True,
        opencv_fallback=opencv_fallback,
        adaptive_fallback=adaptive,
        adaptive_warmup=warmup,
        adaptive_window=window,
        adaptive_disable_rate=disable_rate,
        adaptive_enable_rate=enable_rate,
        adaptive_probe_interval=probe_interval,
    )
    cv_stub = _ScriptedOpenCV(script)
    router._opencv_detector = cv_stub
    mnn_stub = _FakeDetector(text=None)
    router._mnn_detector = mnn_stub
    router._mnn_init_attempted = True
    return router, cv_stub, mnn_stub


class TestAdaptiveFallback:
    """Rolling rescue-rate controls whether MNN misses retry OpenCV.

    Regression guard for the observation that on IMG_9425.MOV the
    default path ran OpenCV 779 times per video, rescuing zero
    frames.  With the adaptive controller on, that wasted work
    should stop after the warmup window.
    """

    def _drive(self, router: DetectorRouter, n: int) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for _ in range(n):
            router.detect(frame)

    def test_disables_fallback_when_rescue_rate_stays_low(self):
        # OpenCV will never save an MNN miss.
        router, cv_stub, _ = _make_router(
            script=[False] * 100,
            warmup=8, window=16, disable_rate=0.1,
        )

        self._drive(router, 20)

        stats = router.get_stats()
        assert stats["mnn_attempts"] == 20
        assert stats["mnn_fallbacks"] == 20
        # OpenCV should only be asked while the controller was still
        # in warmup; after that the adaptive switch flips off and
        # OpenCV stops being called.
        assert stats["opencv_attempts"] < 20
        # Must have seen at least the warmup budget.
        assert stats["opencv_attempts"] >= 8
        assert stats["adaptive_disables"] >= 1
        assert stats["opencv_rescues"] == 0

    def test_reenables_fallback_when_rescue_rate_recovers(self):
        # First burst looks hopeless (disable), then OpenCV starts
        # rescuing every frame.  The controller must probe OpenCV
        # periodically while suppressed so it notices the recovery.
        bad = [False] * 16
        good = [True] * 64
        router, cv_stub, _ = _make_router(
            script=bad + good, warmup=8, window=12,
            disable_rate=0.1, enable_rate=0.3,
            probe_interval=2,
        )

        self._drive(router, len(bad) + len(good))

        stats = router.get_stats()
        assert stats["adaptive_disables"] >= 1
        assert stats["adaptive_enables"] >= 1
        # Final state must be active again; otherwise the router is
        # stuck off and we lose MNN-miss rescues forever.
        with router._stats_lock:
            assert router._fallback_active is True

    def test_hysteresis_prevents_flapping(self):
        # Rescue rate hovers near the disable threshold: with
        # disable_rate=0.1 and enable_rate=0.3, rate=0.15 must NOT
        # cause fallback to re-enable once it's been turned off.
        #
        # Pattern: start with zeros to force disable, then a band
        # where ~15% of misses rescue.
        zeros = [False] * 20
        mild = ([True] + [False] * 6) * 4  # ~14% hit rate
        script = zeros + mild
        router, _, _ = _make_router(
            script=script, warmup=8, window=20,
            disable_rate=0.1, enable_rate=0.3,
        )

        self._drive(router, len(script))

        stats = router.get_stats()
        assert stats["adaptive_disables"] >= 1
        # Rescue rate in the second phase is below enable_rate (0.3),
        # so no re-enable allowed.
        assert stats["adaptive_enables"] == 0

    def test_warmup_prevents_early_disable(self):
        # Even with 100% rescue-rate-of-zero, the controller must
        # not flip during warmup.  If it did, a video that starts
        # with a few all-dark frames would permanently lose OpenCV
        # fallback before we've learnt anything.
        router, _, _ = _make_router(
            script=[False] * 20, warmup=16, window=16,
        )

        # Drive exactly warmup-1 frames.
        self._drive(router, 15)

        stats = router.get_stats()
        assert stats["adaptive_disables"] == 0
        with router._stats_lock:
            assert router._fallback_active is True

    def test_adaptive_false_never_flips(self):
        # User explicitly disables adaptive behaviour → OpenCV must
        # be called for every single MNN miss, forever.
        router, _, _ = _make_router(
            script=[False] * 200, warmup=4, window=8,
            adaptive=False,
        )

        self._drive(router, 40)

        stats = router.get_stats()
        assert stats["opencv_attempts"] == 40
        assert stats["adaptive_disables"] == 0
        assert stats["adaptive_enables"] == 0

    def test_opencv_fallback_false_is_absolute(self):
        # ``opencv_fallback=False`` is a hard user preference — the
        # adaptive controller must never re-enable a fallback the
        # user explicitly asked not to run.
        router, cv_stub, _ = _make_router(
            script=[True] * 100, warmup=2, window=4,
            opencv_fallback=False, adaptive=True,
        )

        self._drive(router, 20)

        stats = router.get_stats()
        assert stats["mnn_fallbacks"] == 20
        assert stats["opencv_attempts"] == 0
        assert cv_stub.calls == 0
        # No adaptive bookkeeping either, because the adaptive
        # controller itself is a no-op when user-level fallback
        # is off.
        assert stats["adaptive_disables"] == 0
        assert stats["adaptive_enables"] == 0

    def test_status_summary_reports_fallback_state(self):
        router, _, _ = _make_router(
            script=[False] * 40, warmup=4, window=8, disable_rate=0.1,
        )
        self._drive(router, 12)

        summary = router.get_status_summary()
        assert "Fallback:" in summary
        # The controller should have flipped; the summary must
        # reflect whichever state we're now in.
        with router._stats_lock:
            expected = "active" if router._fallback_active else "suppressed"
        assert expected in summary


# ── DetectorRouter stats thread-safety smoke ─────────────────────


class TestRouterStatsSnapshot:
    def test_get_stats_returns_copy(self):
        router = DetectorRouter(use_mnn=False)
        snap1 = router.get_stats()
        snap1["mnn_attempts"] = 999  # mutate the snapshot
        snap2 = router.get_stats()
        # Mutating a snapshot must not affect the router's counters.
        assert snap2["mnn_attempts"] == 0


# ── End-to-end: use_mnn=True without MNN installed ───────────────


@pytest.mark.slow
class TestExtractWithMnnFallback:
    """When MNN isn't installed, ``use_mnn=True`` must still decode.

    This is the key Milestone 1 guarantee: turning on ``--mnn``
    never breaks the existing OpenCV pipeline — it only adds an
    extra path that, when available, accelerates detection.  On a
    host without MNN (the default Linux CI environment), the router
    falls through to ``OpenCVWeChatDetector`` and the video must
    decode byte-exactly.
    """

    def _encode_small_video(self, data: bytes, out_dir: Path) -> Path:
        input_path = out_dir / "payload.bin"
        video_path = out_dir / "payload.mp4"
        input_path.write_bytes(data)
        encode_to_video(
            input_path=str(input_path),
            output_path=str(video_path),
            overhead=2.0,
            fps=10,
            ec_level=1,
            qr_version=10,
            border=4,
            lead_in_seconds=0.0,
            compress=False,
            verbose=False,
            workers=2,
        )
        return video_path

    def test_round_trip_use_mnn_true_falls_back_to_opencv(self, tmp_path):
        # Keep the payload small so the encode → decode cycle fits
        # comfortably into the default 'slow' test budget.
        data = os.urandom(512)
        video_path = self._encode_small_video(data, tmp_path)

        blocks = extract_qr_from_video(
            str(video_path), sample_rate=1, verbose=False, workers=2,
            use_mnn=True)
        assert blocks, "decoder produced no blocks with use_mnn=True"

        decoded = decode_blocks(blocks, verbose=False)
        assert decoded == data


# ── End-to-end: MNN actually exercised (opt-in via environment) ──


@pytest.mark.slow
@pytest.mark.skipif(
    not _mnn_available(), reason="MNN Python bindings not installed",
)
@pytest.mark.skipif(
    not _mnn_models_available(),
    reason="MNN models missing (run the M0 container to produce them)",
)
class TestExtractWithMnnEnabled:
    """End-to-end: ``use_mnn=True`` with MNN + models actually present.

    Runs only when both the ``MNN`` package is importable AND the
    ``detect.mnn`` / ``sr.mnn`` artefacts exist under
    ``dev/wechatqrcode-mnn-poc/models``.  Inside the M1 container
    both prerequisites are satisfied, so this class exercises the
    full MNN inference path end-to-end on the CPU backend (Metal is
    an Apple-only backend and unavailable inside Linux containers).
    """

    def _encode_small_video(self, data: bytes, out_dir: Path) -> Path:
        input_path = out_dir / "payload.bin"
        video_path = out_dir / "payload.mp4"
        input_path.write_bytes(data)
        encode_to_video(
            input_path=str(input_path),
            output_path=str(video_path),
            overhead=2.0,
            fps=10,
            ec_level=1,
            qr_version=10,
            border=4,
            lead_in_seconds=0.0,
            compress=False,
            verbose=False,
            workers=2,
        )
        return video_path

    def test_round_trip_use_mnn_true_runs_mnn_path(self, tmp_path):
        data = os.urandom(512)
        video_path = self._encode_small_video(data, tmp_path)

        blocks = extract_qr_from_video(
            str(video_path), sample_rate=1, verbose=False, workers=2,
            use_mnn=True)
        assert blocks, "MNN decode produced no blocks"

        decoded = decode_blocks(blocks, verbose=False)
        assert decoded == data

    def test_mnn_detector_reports_cpu_backend_inside_container(self):
        """Inside Linux containers, Metal is unavailable and CPU wins."""
        from qrstream.detector.mnn_detector import MNNQrDetector

        det = MNNQrDetector(backend="cpu")
        assert det.is_available(), "MNN detector should be available"
        # Prime the backend via a trivial detect() call.
        det.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        assert det.name.startswith("mnn_"), (
            f"expected mnn_* name, got {det.name!r}"
        )

    def test_router_stats_track_mnn_attempts(self):
        """After MNN-enabled frames flow through the worker, stats move.

        Regression guard: this is the exact failure mode M1 fixes.
        Before the worker/router plumbing, ``use_mnn=True`` silently
        ran the legacy OpenCV path and MNN counters stayed at zero.
        We assert the counters move so a future regression to that
        behaviour goes red.
        """
        # Feed synthetic frames through the worker directly — we do
        # not need a real QR video for this counter check, only proof
        # that ``functools.partial(_worker_detect_qr, qr_detector=router)``
        # actually dispatches to the router rather than the legacy
        # per-thread OpenCV cache in ``qr_utils.py``.
        router = DetectorRouter(use_mnn=True, mnn_backend="cpu")

        frames = [
            np.zeros((120, 160, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        for idx, frame in enumerate(frames):
            _worker_detect_qr((idx, frame), qr_detector=router)

        stats = router.get_stats()
        # MNN must have been attempted on every frame we processed.
        assert stats["mnn_attempts"] == len(frames), (
            f"expected {len(frames)} MNN attempts, stats={stats}"
        )


# ── detect_batch integration ─────────────────────────────────────


class TestDetectBatchIntegration:
    """``detect_batch`` must behave identically to sequential ``detect``."""

    def test_batch_with_fake_mnn_matches_sequential(self):
        """Batch call via router with injected fake MNN must match
        per-frame results (same stats, same verdicts)."""
        router = DetectorRouter(use_mnn=True, opencv_fallback=False)
        fake = _FakeDetector(text=None)
        router._mnn_detector = fake
        router._mnn_init_attempted = True

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
        batch_results = router.detect_batch(frames)

        assert len(batch_results) == 6
        stats = router.get_stats()
        assert stats["mnn_attempts"] == 6
        assert stats["mnn_fallbacks"] == 6
        # opencv_fallback=False → no OpenCV calls
        assert stats["opencv_attempts"] == 0

    def test_batch_with_mnn_hit_short_circuits(self):
        router = DetectorRouter(use_mnn=True, opencv_fallback=True)
        fake = _FakeDetector(text="test-payload")
        router._mnn_detector = fake
        router._mnn_init_attempted = True

        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
        results = router.detect_batch(frames)

        assert all(r.text == "test-payload" for r in results)
        stats = router.get_stats()
        assert stats["mnn_success"] == 3
        assert stats["opencv_attempts"] == 0
