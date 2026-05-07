"""
Tests for ``qrstream.qr_sandbox.SandboxedDetector`` in isolation.

These tests synthesise QR frames in-memory via
``qrstream.qr_utils.generate_qr_image`` so no fixture video / image
files are required.  Crash-injection tests use dedicated subclass
helpers whose helper-loop function lives at module scope (required by
the ``spawn`` start method: all targets must be pickle-safe top-level
callables).

.. deprecated::
    qr_sandbox is deprecated since v0.9 (zxing-cpp backend, no crashes).
    This entire test module is skipped.  It is retained only so that
    ``git blame`` still shows why the sandbox existed.
"""
from __future__ import annotations

import pytest
pytestmark = pytest.mark.skip(reason="qr_sandbox deprecated since v0.9 (zxing-cpp backend)")

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from qrstream.qr_sandbox import SandboxedDetector, _STOP
from qrstream.qr_utils import generate_qr_image, try_decode_qr


# ── frame-synthesis helpers ─────────────────────────────────────
# We intentionally use ``alphanumeric=False`` (base64 / QR byte mode)
# rather than the project default (base45 alphanumeric). The decoded
# QR string is then exactly ``base64(payload).decode('ascii')``, which
# is trivial to predict / compare in assertions. The default base45
# path works too, but makes test expectations awkward.

def _make_qr(payload: bytes, *, version: int = 5) -> np.ndarray:
    """Synthesise a BGR QR image for ``payload`` via zxing-cpp.

    Returns the image; the decoded string from zxing-cpp will be
    ``base64(payload).decode('ascii')``.
    """
    return generate_qr_image(
        payload, ec_level=1, version=version, alphanumeric=False,
    )


def _expected_for(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


# ── helper-loop variants used by crash-injection tests ──────────
# All three live at module scope so ``multiprocessing.spawn`` can
# pickle them by reference.

_CRASH_SENTINEL_PREFIX = "CRASHME"
# Raw bytes payload whose base64 encoding starts with ``CRASHME``. The
# base64 alphabet is [A-Za-z0-9+/] so ``CRASHME`` is valid base64 text;
# any payload starting with the pre-image of ``CRASHM`` will do. We
# use a fixed pre-image so the test is deterministic.
_CRASH_SENTINEL_RAW_PAYLOAD = base64.b64decode("CRASHMEA=")  # 6 bytes


def _helper_loop_crash_on_sentinel(in_q, out_q):
    """Helper that dies with os._exit(134) when the decoded QR text
    starts with the sentinel prefix, and returns "ok" otherwise."""
    while True:
        try:
            item = in_q.get()
        except (EOFError, KeyboardInterrupt):
            return
        if item == _STOP:
            return
        try:
            frame_idx, shape, dtype_str, raw = item
        except Exception:
            continue
        try:
            frame = np.frombuffer(
                raw, dtype=np.dtype(dtype_str)
            ).reshape(shape)
            decoded = try_decode_qr(frame)
        except Exception:
            decoded = None

        if decoded is not None and decoded.startswith(
            _CRASH_SENTINEL_PREFIX
        ):
            # Simulate a native crash.  os._exit bypasses atexit/finally
            # handlers, matching the real SIGSEGV/SIGTRAP path.
            os._exit(134)

        try:
            out_q.put(
                (frame_idx, "ok" if decoded is not None else None, None)
            )
        except (BrokenPipeError, OSError):
            return


def _helper_loop_always_crash(in_q, out_q):
    """Helper that dies on every frame, used to trigger the abort path."""
    del out_q
    while True:
        try:
            item = in_q.get()
        except (EOFError, KeyboardInterrupt):
            return
        if item == _STOP:
            return
        os._exit(134)


def _helper_loop_sleep_forever(in_q, out_q):
    """Helper that never replies, to exercise the detect() timeout path."""
    del out_q
    while True:
        try:
            item = in_q.get()
        except (EOFError, KeyboardInterrupt):
            return
        if item == _STOP:
            return
        # Sleep until killed; never put anything on out_q.
        time.sleep(3600)


# ── sandbox subclasses for crash injection ──────────────────────

class _CrashOnSentinelSandbox(SandboxedDetector):
    def _helper_target(self):
        return _helper_loop_crash_on_sentinel


class _AlwaysCrashSandbox(SandboxedDetector):
    def _helper_target(self):
        return _helper_loop_always_crash


class _SleepSandbox(SandboxedDetector):
    def _helper_target(self):
        return _helper_loop_sleep_forever


# ── tests ───────────────────────────────────────────────────────

def test_detect_roundtrip_returns_same_as_inprocess():
    img = _make_qr(b"hello world", version=5)
    expected = try_decode_qr(img)
    assert expected == _expected_for(b"hello world")

    with SandboxedDetector(pool_size=1) as sb:
        got = sb.detect(0, img)
    assert got == expected


def test_detect_with_bbox_returns_text_and_geometry():
    """detect_with_bbox round-trips both the decoded text and a
    sensible (4, 2) float32 bbox through the helper subprocess.

    This is the primary regression guard for the arm64 segfault that
    motivated routing the probe path through the sandbox: if the IPC
    serialisation of the bbox tuple breaks, the adaptive-downscale
    probe silently degrades to "no observations" and the legacy
    1080-cap fallback kicks in.
    """
    img = _make_qr(b"bbox-roundtrip", version=10)
    h, w = img.shape[:2]
    expected_text = _expected_for(b"bbox-roundtrip")

    with SandboxedDetector(pool_size=1) as sb:
        got = sb.detect_with_bbox(0, img)

    assert got is not None, "detect_with_bbox returned None on a clean QR"
    text, bbox = got
    assert text == expected_text
    assert bbox is not None, "bbox should be populated on a clean QR"
    assert bbox.shape == (4, 2)
    assert bbox.dtype == np.float32
    # Each corner must lie within the source frame; a degenerate
    # all-zeros bbox would also satisfy "within frame" but would
    # carry zero area, so check both.
    assert (bbox[:, 0] >= 0).all() and (bbox[:, 0] <= w).all()
    assert (bbox[:, 1] >= 0).all() and (bbox[:, 1] <= h).all()
    side_x = float(np.linalg.norm(bbox[1] - bbox[0]))
    side_y = float(np.linalg.norm(bbox[3] - bbox[0]))
    assert side_x > 1.0 and side_y > 1.0, \
        f"bbox is degenerate: {bbox.tolist()}"


def test_detect_with_bbox_returns_none_on_blank_frame():
    """Blank frames must surface as None rather than a (None, None)
    tuple — callers branch on ``is None`` to skip no-detect frames."""
    blank = np.full((400, 400, 3), 255, dtype=np.uint8)
    with SandboxedDetector(pool_size=1) as sb:
        got = sb.detect_with_bbox(0, blank)
    assert got is None


def test_multiple_frames_round_trip_concurrently():
    payloads = {i: f"frame-{i}".encode() for i in range(20)}
    frames = [(i, _make_qr(payloads[i], version=5)) for i in range(20)]

    with SandboxedDetector(pool_size=3) as sb:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(sb.detect, idx, img): idx
                for idx, img in frames
            }
            results = {}
            for fut in futures:
                idx = futures[fut]
                results[idx] = fut.result(timeout=60)

    for idx, _ in frames:
        assert results[idx] == _expected_for(payloads[idx]), (
            f"frame {idx} got wrong routing: {results[idx]!r}"
        )


def test_worker_crash_is_recovered():
    # 10 frames, exactly one of which triggers the crash helper.
    good_frames = [
        (i, _make_qr(f"payload-{i}".encode(), version=5))
        for i in range(9)
    ]
    # The crash helper compares the *decoded* QR string against the
    # sentinel prefix. Since we encode in base64, we embed the sentinel
    # by making the decoded output start with the prefix.
    crash_payload = _CRASH_SENTINEL_RAW_PAYLOAD
    crash_frame = (
        99,
        _make_qr(crash_payload, version=5),
    )
    all_frames = good_frames + [crash_frame]

    with _CrashOnSentinelSandbox(pool_size=2) as sb:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(sb.detect, idx, img, 30.0): idx
                for idx, img in all_frames
            }
            results = {}
            for fut, idx in futures.items():
                results[idx] = fut.result(timeout=60)

        assert results[99] is None, (
            "sentinel frame should be satisfied with None (helper crashed)"
        )
        for idx, _ in good_frames:
            assert results[idx] == "ok", (
                f"non-sentinel frame {idx} got {results[idx]!r}, expected 'ok'"
            )
        assert sb.crash_count >= 1

    # Implicit .close() via context manager must have returned quickly.


def test_repeated_crashes_trigger_abort():
    img = _make_qr(b"whatever", version=5)

    with _AlwaysCrashSandbox(
        pool_size=1,
        crash_abort_threshold=3,
        crash_abort_window=10.0,
    ) as sb:
        aborted = False
        for _ in range(20):
            try:
                sb.detect(0, img, timeout=5.0)
            except RuntimeError as e:
                assert "repeatedly crashing" in str(e)
                aborted = True
                break
            except TimeoutError:
                # Back-pressure while helpers keep dying is fine;
                # keep trying until abort fires.
                pass
        assert aborted, "abort threshold never tripped"


def test_close_is_idempotent():
    sb = SandboxedDetector(pool_size=1)
    sb.close()
    sb.close()  # must not raise


def test_close_after_no_use():
    sb = SandboxedDetector(pool_size=1)
    sb.close()
    for proc in getattr(sb, "_helpers", []):
        assert not proc.is_alive()


def test_detect_after_close_raises():
    img = _make_qr(b"x", version=5)
    sb = SandboxedDetector(pool_size=1)
    sb.close()
    with pytest.raises(RuntimeError):
        sb.detect(0, img)


def test_context_manager_closes_on_exit():
    with SandboxedDetector(pool_size=1) as sb:
        helpers = list(sb._helpers)
    # Give helpers a moment to exit after _STOP.
    for proc in helpers:
        proc.join(timeout=5.0)
        assert not proc.is_alive(), (
            "helper still alive after context manager exit"
        )


def test_detect_timeout_returns_none_not_raises():
    img = _make_qr(b"unreachable", version=5)
    with _SleepSandbox(pool_size=1) as sb:
        got = sb.detect(0, img, timeout=0.5)
    assert got is None


def test_out_q_routing_under_many_in_flight():
    payloads = {i: f"idx-{i}".encode() for i in range(100)}
    frames = [(i, _make_qr(payloads[i], version=5)) for i in range(100)]

    with SandboxedDetector(pool_size=3) as sb:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(sb.detect, idx, img, 60.0): idx
                for idx, img in frames
            }
            results = {}
            for fut, idx in futures.items():
                results[idx] = fut.result(timeout=120)

    mismatches = [
        (idx, got) for idx, got in results.items()
        if got != _expected_for(payloads[idx])
    ]
    assert not mismatches, f"routing mismatches: {mismatches[:5]}"
