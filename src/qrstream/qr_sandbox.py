"""
Subprocess-isolated sandbox for the WeChat QR detector.

Why this module exists
----------------------
``cv2.wechat_qrcode_WeChatQRCode().detectAndDecode(frame)`` ships with a
bundled ``zxing`` that has a long-standing, unfixed native bug — upstream
issue ``opencv_contrib#3570``.  On noisy camera-captured frames the
detector occasionally false-positives a Finder Pattern, walks module-size
estimation out of the ``BitMatrix`` bounds, and dereferences a wild
offset.  When that offset lands on an unmapped page the process dies with
``SIGSEGV`` / ``SIGTRAP`` before Python can raise an exception, taking
``qrs decode`` with it.

This module wraps the detector in a pool of short-lived subprocess
helpers so a native crash degrades to "this one frame is a no-detect"
instead of "the Python process dies".  LT fountain coding already
tolerates dropped frames by design; skipping one frame per helper crash
is invisible in the final decoded output as long as overhead is ≥ 1.5×.

Design notes
------------
- ``spawn`` start method is used unconditionally.  ``fork`` is unsafe on
  macOS 10.13+ (requires ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY`` and
  deadlocks with OpenCV in practice) and is unavailable on Windows.
  ``spawn`` is the lowest-common-denominator path that works everywhere.
- One shared ``mp.Queue`` for input (natural load-balancing across
  helpers) and one shared ``mp.Queue`` for output.  Per-request routing
  back to the waiting caller thread uses a single-slot ``queue.Queue``
  indexed by ``frame_idx`` inside the parent process.
- A dedicated daemon collector thread drains the output queue and
  reaps dead helpers.  Helpers that died unexpectedly (``exitcode != 0``)
  bump ``crash_count``, their presumed-in-flight frame is satisfied with
  ``None``, and a replacement helper is spawned.
- Crash-burst abort: if ``crash_count`` reaches ``crash_abort_threshold``
  within ``crash_abort_window`` seconds of the first crash, subsequent
  ``detect()`` calls raise ``RuntimeError``.  This stops a wedged input
  from consuming unbounded CPU respawning helpers.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import time
import uuid
from typing import Optional

import numpy as np


# Sentinel sent on ``_in_q`` to ask a helper to exit cleanly.  Must be
# picklable; a plain tuple is pickle-friendly across ``spawn``.
_STOP = ("__STOP__",)


def _helper_loop(in_q, out_q):
    """Subprocess entry point.

    Drains ``in_q`` forever, calls ``try_decode_qr`` on each frame, and
    pushes ``(frame_idx, decoded_or_None)`` onto ``out_q``.  Returns on
    ``_STOP`` or on unrecoverable IPC errors; a silent return is enough
    — the supervisor will observe the dead process via ``is_alive() ==
    False`` and respawn.

    Any Python-level exception in ``try_decode_qr`` is caught and mapped
    to ``None`` (no-detect), so callers uniformly treat any problem as
    "this frame contributed nothing".
    """
    # Import inside the child so the main process isn't forced to pay
    # the cv2 / WeChat-detector import cost when the sandbox is disabled.
    from .qr_utils import try_decode_qr

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
            # Malformed payload; drop silently and continue.
            continue

        try:
            frame = np.frombuffer(
                raw, dtype=np.dtype(dtype_str)
            ).reshape(shape)
            if not frame.flags.writeable:
                frame = frame.copy()
            decoded = try_decode_qr(frame)
        except Exception:
            decoded = None

        try:
            out_q.put((frame_idx, decoded))
        except (BrokenPipeError, OSError):
            return


class SandboxedDetector:
    """Pool-of-subprocess-helpers wrapper around ``try_decode_qr``.

    Instances are thread-safe; multiple worker threads in the parent
    process may call :meth:`detect` concurrently.
    """

    def __init__(
        self,
        pool_size: int = 3,
        *,
        start_method: str = "spawn",
        crash_abort_threshold: int = 3,
        crash_abort_window: float = 10.0,
    ):
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        if crash_abort_threshold < 1:
            raise ValueError(
                f"crash_abort_threshold must be >= 1, "
                f"got {crash_abort_threshold}"
            )
        if crash_abort_window <= 0:
            raise ValueError(
                f"crash_abort_window must be > 0, "
                f"got {crash_abort_window}"
            )

        self._pool_size = pool_size
        self._crash_abort_threshold = crash_abort_threshold
        self._crash_abort_window = crash_abort_window

        self._ctx = multiprocessing.get_context(start_method)
        # Bounded input queue for natural back-pressure.  ``pool_size *
        # 4`` keeps the helpers saturated without letting a slow
        # detector inflate memory use unboundedly.
        self._in_q = self._ctx.Queue(maxsize=max(pool_size * 4, 4))
        self._out_q = self._ctx.Queue()

        self._lock = threading.Lock()
        self._helpers: list = []
        self._results: dict[str, queue.Queue] = {}
        self._closed = False
        self._crash_count = 0
        self._first_crash_at: Optional[float] = None

        # Spawn the initial pool before starting the collector so the
        # very first ``detect()`` call finds at least one live helper.
        for _ in range(pool_size):
            self._helpers.append(self._spawn_helper())

        self._collector_stop = threading.Event()
        self._collector_thread = threading.Thread(
            target=self._collector_loop,
            name="SandboxedDetector-collector",
            daemon=True,
        )
        self._collector_thread.start()

    # ── subprocess-specific entry point override hook ────────────
    # Tests subclass ``SandboxedDetector`` and override this to inject
    # a crashing helper target while keeping all supervision logic
    # identical.  Production code never replaces it.
    def _helper_target(self):
        return _helper_loop

    # ── lifecycle helpers ────────────────────────────────────────
    def _spawn_helper(self):
        proc = self._ctx.Process(
            target=self._helper_target(),
            args=(self._in_q, self._out_q),
            daemon=True,
        )
        proc.start()
        return proc

    def _record_crash_locked(self) -> None:
        """Must be called with ``self._lock`` held."""
        self._crash_count += 1
        if self._first_crash_at is None:
            self._first_crash_at = time.monotonic()

    def _abort_triggered_locked(self) -> bool:
        """Return True if the crash-burst abort threshold has tripped.

        Must be called with ``self._lock`` held.  Resets the window when
        it has elapsed without hitting the threshold so a long-running
        decode can tolerate sparse, well-separated crashes.
        """
        if self._first_crash_at is None:
            return False
        now = time.monotonic()
        elapsed = now - self._first_crash_at
        if elapsed > self._crash_abort_window:
            # Reset the window.  A crash just outside the window
            # shouldn't carry over historical counts.
            self._first_crash_at = now
            # We keep ``_crash_count`` monotonic because the ``crashes
            # > 0`` summary line still wants the total for diagnostic
            # purposes; only the abort check rolls over.
            self._window_base = self._crash_count
            return False

        # _window_base is the crash count at the start of the current
        # window; only crashes inside the window count toward abort.
        base = getattr(self, "_window_base", 0)
        in_window = self._crash_count - base
        return in_window >= self._crash_abort_threshold

    def _reap_dead_helpers(self) -> None:
        """Detect helpers that exited unexpectedly and replace them.

        Called from the collector thread on every output-queue poll
        timeout.  For each helper with ``is_alive() == False`` and a
        non-normal exit we:

          1. record a crash,
          2. satisfy **one** pending waiter (approximating "the frame
             the dead helper was working on") with ``None``, and
          3. spawn a replacement.

        The waiter we pick is arbitrary — we can't know which frame the
        helper was actually processing.  LT fountain coding tolerates
        dropped frames, so picking any waiter is safe.
        """
        with self._lock:
            if self._closed:
                return

            new_helpers = []
            crashes_observed = 0
            for proc in self._helpers:
                if proc.is_alive():
                    new_helpers.append(proc)
                    continue
                # Dead helper. Join to reap the zombie.
                try:
                    proc.join(timeout=0.1)
                except Exception:
                    pass
                exitcode = proc.exitcode
                # exitcode 0 means clean exit (e.g. _STOP on close); any
                # other value (negative for signals, positive for
                # sys.exit) counts as a crash.
                if exitcode != 0:
                    self._record_crash_locked()
                    crashes_observed += 1

            # Satisfy one pending waiter per observed crash with None.
            for _ in range(crashes_observed):
                if not self._results:
                    break
                # Pop an arbitrary waiter — dict ordering gives us the
                # oldest in CPython 3.7+, which approximates FIFO.
                stale_key = next(iter(self._results))
                waiter = self._results.pop(stale_key)
                try:
                    waiter.put_nowait(None)
                except queue.Full:
                    pass

            # Respawn to keep the pool size stable.  If we are in the
            # middle of closing, the close() path will handle teardown;
            # don't respawn then.
            while len(new_helpers) < self._pool_size and not self._closed:
                new_helpers.append(self._spawn_helper())

            self._helpers = new_helpers

    def _collector_loop(self) -> None:
        """Drain ``_out_q`` and route replies to waiting threads.

        A short poll timeout also drives periodic helper-liveness
        checks, so a helper that dies mid-frame is noticed even when no
        new replies are flowing.
        """
        while not self._collector_stop.is_set():
            try:
                item = self._out_q.get(timeout=0.5)
            except queue.Empty:
                self._reap_dead_helpers()
                continue
            except (EOFError, OSError):
                # Queue machinery is going away (close()).
                return

            try:
                frame_key, payload = item
            except Exception:
                # Malformed reply; ignore.
                continue

            with self._lock:
                waiter = self._results.pop(frame_key, None)
            if waiter is not None:
                try:
                    waiter.put_nowait(payload)
                except queue.Full:
                    # Caller already gave up (timeout); drop the reply.
                    pass

    # ── public API ───────────────────────────────────────────────
    def detect(
        self,
        frame_idx: int,
        frame: np.ndarray,
        timeout: float = 30.0,
    ) -> Optional[str]:
        """Decode a QR code in ``frame`` via the helper pool.

        ``frame_idx`` is an opaque tag for logging / diagnostics; the
        sandbox disambiguates in-flight frames via an internal UUID, so
        two concurrent ``detect()`` calls with the same ``frame_idx``
        still route correctly.

        Returns the decoded string, or ``None`` on no-detect / helper
        crash / timeout.  Raises ``RuntimeError`` when the sandbox has
        been closed or the crash-burst abort threshold has tripped.
        Raises ``TimeoutError`` when the input queue is saturated long
        enough that a put times out (back-pressure signal).
        """
        del frame_idx  # informational only

        # Use an internal UUID to route replies.  Worker threads may
        # legitimately call detect() with the same frame_idx across
        # different phases (probe / main / recovery); we cannot rely on
        # it being unique.
        tag = uuid.uuid4().hex

        with self._lock:
            if self._closed:
                raise RuntimeError("SandboxedDetector is closed")
            if self._abort_triggered_locked():
                raise RuntimeError(
                    "Sandbox helpers repeatedly crashing; aborting"
                )
            waiter: queue.Queue = queue.Queue(maxsize=1)
            self._results[tag] = waiter

        # Serialise the ndarray outside of the lock — ``tobytes()`` can
        # be non-trivial on large frames and the lock is parent-process
        # mutation-only.
        arr = np.ascontiguousarray(frame)
        payload = (tag, arr.shape, arr.dtype.str, arr.tobytes())

        try:
            try:
                self._in_q.put(payload, timeout=timeout)
            except queue.Full as exc:
                # Clean up the dangling waiter registration.
                with self._lock:
                    self._results.pop(tag, None)
                raise TimeoutError(
                    "SandboxedDetector input queue full"
                ) from exc

            try:
                result = waiter.get(timeout=timeout)
            except queue.Empty:
                # Helper might still be working; remove the waiter so a
                # late reply is discarded instead of routed to a stale
                # single-slot queue.
                with self._lock:
                    self._results.pop(tag, None)
                return None
            return result
        except Exception:
            # Any other failure mode → best-effort cleanup + rethrow.
            with self._lock:
                self._results.pop(tag, None)
            raise

    def close(self, timeout: float = 5.0) -> None:
        """Stop all helpers and join them.  Idempotent.

        Workflow:

          1. Mark closed under the lock so new ``detect()`` calls error.
          2. Signal the collector thread to stop.
          3. Send one ``_STOP`` sentinel per helper and give them
             ``timeout`` seconds to exit.
          4. ``terminate()`` + short join for stragglers.
          5. Drain the output queue so ``mp.Queue`` internals can tear
             down cleanly.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            helpers = list(self._helpers)
            self._helpers = []

        self._collector_stop.set()

        # Tell every helper to exit cleanly.
        for _ in helpers:
            try:
                self._in_q.put(_STOP, timeout=0.5)
            except (queue.Full, OSError):
                break

        deadline = time.monotonic() + timeout
        for proc in helpers:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.join(timeout=remaining)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.join(timeout=1.0)
                except Exception:
                    pass
                if proc.is_alive():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.join(timeout=1.0)
                    except Exception:
                        pass

        # Collector thread: give it a moment to unblock from get().
        self._collector_thread.join(timeout=max(0.5, timeout / 2))

        # Satisfy any still-pending waiters with None so callers don't
        # hang forever if close() races with in-flight detect() calls.
        with self._lock:
            pending = list(self._results.items())
            self._results.clear()
        for _, waiter in pending:
            try:
                waiter.put_nowait(None)
            except queue.Full:
                pass

        # Drain whatever is left in the cross-process queues.
        for q in (self._in_q, self._out_q):
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass
            try:
                q.join_thread()
            except Exception:
                pass

    @property
    def crash_count(self) -> int:
        """Number of helper subprocess crashes observed so far."""
        with self._lock:
            return self._crash_count

    def __enter__(self) -> "SandboxedDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
