"""Guard tests for the CLAHE-based targeted recovery wiring.

These tests directly target the v0.7.1 → v0.7.2 bug-fix delta:

  - ``_worker_detect_qr_clahe`` must exist in the decoder module and
    be a plain module-level callable (so ``ThreadPoolExecutor.submit``
    can dispatch it; keeping the worker module-level also preserves
    clarity and test-time introspection).
  - ``_stream_scan`` must accept an injectable ``worker_fn`` so that
    main scan and targeted recovery can run different detectors on
    the same frame pipeline.
  - ``_targeted_recovery`` must wire the CLAHE worker through.

If any of these assertions fail, the v070 amd64 regression comes
back.  These tests don't need a fixture video because they verify
module structure + call wiring, which is what the regression was
actually about.
"""

from __future__ import annotations

import inspect

import numpy as np

from qrstream import decoder as dec_mod


def test_clahe_worker_is_defined_and_callable():
    """The CLAHE recovery worker must exist as a module-level callable."""
    assert hasattr(dec_mod, "_worker_detect_qr_clahe"), (
        "_worker_detect_qr_clahe is missing; the v070 recovery path "
        "will be dead even if _targeted_recovery is triggered."
    )
    fn = dec_mod._worker_detect_qr_clahe
    assert callable(fn)
    # Must be a module-level function (not a lambda / closure) so it
    # is trivially introspectable; ThreadPoolExecutor can dispatch it.
    assert inspect.isfunction(fn)
    assert fn.__module__ == "qrstream.decoder"


def test_clahe_worker_handles_none_and_noisy_frames():
    """The CLAHE worker must fail gracefully, mirroring the main worker.

    Two shapes of input are important:
      1. ``(frame_idx, None)`` — skipped frame.
      2. A frame with no QR code — CLAHE runs, WeChat returns nothing.

    Both should return ``(frame_idx, None, None)``.
    """
    fn = dec_mod._worker_detect_qr_clahe

    # Case 1: frame is None (skipped).
    assert fn((7, None)) == (7, None, None)

    # Case 2: random-noise frame, no QR signal.  Deterministic rng
    # so CI reruns behave identically.
    rng = np.random.default_rng(12345)
    noise = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    idx, block, seed = fn((42, noise))
    assert idx == 42
    # WeChat must not hallucinate a QR out of noise; if it does, the
    # subsequent unpack() would raise and we'd still return (_, None, None).
    assert block is None
    assert seed is None


def test_stream_scan_accepts_worker_fn():
    """``_stream_scan`` must take a ``worker_fn`` keyword.

    This is what lets recovery swap in the CLAHE worker without
    duplicating the entire pipelined-submit loop.
    """
    sig = inspect.signature(dec_mod._stream_scan)
    assert "worker_fn" in sig.parameters, (
        "_stream_scan must accept worker_fn for recovery to use CLAHE."
    )
    # Default must be None (or _worker_detect_qr) so main-scan
    # behaviour is unchanged.
    default = sig.parameters["worker_fn"].default
    assert default is None or default is dec_mod._worker_detect_qr


def test_targeted_recovery_gate_no_longer_depends_on_sample_rate():
    """Regression guard: the old ``sample_rate > 1`` guard was the
    reason v070 never even *entered* recovery on amd64 (probe picked
    sample_rate=1).  Make sure nobody reintroduces it.

    We can't dynamically re-run the gate here without a fixture, so
    we inspect the non-comment source of ``extract_qr_from_video``
    for any ``if`` statement that conditions recovery on
    ``sample_rate > 1``.
    """
    src = inspect.getsource(dec_mod.extract_qr_from_video)
    # Strip out full-line comments so our documentation of the bug
    # fix (which legitimately mentions ``sample_rate > 1``) does not
    # trigger this guard.
    code_only = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    # The forbidden pattern is a boolean condition ANDed with the
    # recovery trigger.  Any re-introduction would look like
    # ``and sample_rate > 1``.
    assert "and sample_rate > 1" not in code_only, (
        "The `sample_rate > 1` gate on _targeted_recovery was "
        "reintroduced; v070 amd64 will fail again."
    )


def test_ge_checkpoint_runs_rescue_with_reporter_events():
    """Phase-boundary GE checkpoints should wrap the decoder rescue."""

    class _FakeDecoder:
        initialized = True
        done = False
        K = 10

        def __init__(self):
            self.called = False
            self._num_recovered = 4

        @property
        def num_recovered(self):
            return self._num_recovered

        def try_gaussian_rescue(self):
            self.called = True
            self.done = True
            self._num_recovered = self.K
            return True

    class _Reporter:
        def __init__(self):
            self.events = []

        def ge_start(self, **kwargs):
            self.events.append(("start", kwargs))

        def ge_done(self, **kwargs):
            self.events.append(("done", kwargs))

    decoder = _FakeDecoder()
    reporter = _Reporter()

    assert dec_mod._attempt_ge_checkpoint(decoder, reporter, "scan") is True
    assert decoder.called
    assert reporter.events[0] == (
        "start", {"stage": "scan", "recovered": 4, "k": 10})
    assert reporter.events[1] == (
        "done", {"success": True, "recovered": 10, "k": 10})
