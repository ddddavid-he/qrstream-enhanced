# Implementation Handoff — WeChatQRCode Native Crash Sandbox

> **This document is a self-contained handoff.** The agent that
> implements this fix is not expected to have any prior context. Read
> this top-to-bottom, then read the three files listed in §2, then
> start implementing. Everything else in the repository is background
> you can consult as needed but are not required to read first.

---

## 0. TL;DR

The bug: `cv2.wechat_qrcode_WeChatQRCode().detectAndDecode(frame)`
occasionally SIGSEGVs on real phone-recorded QR videos, taking the
entire `qrs decode` process down with it. It is a long-standing,
unfixed native bug in `opencv_contrib`'s bundled `zxing` (upstream
issue `opencv_contrib#3570`).

The fix: run the detector in a pool of short-lived subprocess
helpers so that a native crash degrades to "this one frame is a
no-detect" instead of "the Python process dies". LT fountain
coding already tolerates dropped frames by design, so skipping
one frame per helper crash is invisible to the final decoded
output.

You are implementing this on branch `fix/wechat-native-crash`. The
branch already has two commits that you should NOT touch:

```
69eb1af docs(incident): report WeChatQRCode native crash root cause
0b2a36a docs(branching): record migration of legacy dev/* to archive/dev-*
```

When you finish, the final PR target is `dev` (not `main`).

---

## 1. Branching rules you must follow

Read `BRANCHING.md` in full before any git operation. Key rules for
this task:

- Working branch name is **`fix/wechat-native-crash`**. Do not rename.
- PR target is **`dev`** (integration branch), not `main`.
- Never push directly to `main` or `dev` for this feature work.
- Never force-push to `main` or `dev`.
- Never use `git commit --amend` on commits authored by someone
  else. Authorship check before amending: `git log -1 --format='%an %ae'`.
- Never update the git config.
- Never skip hooks (`--no-verify`, `--no-gpg-sign`).
- Do not touch `archive/*` branches — those are historical refs
  that have already been migrated from the old `dev/*` namespace.
- `.bench/` exists in the working tree but is untracked and must
  stay untracked. Do not `git add` it.

Commit message style used by this repo (see `git log --oneline` for
examples): imperative mood, scoped prefix (`feat(...)`, `fix(...)`,
`docs(...)`, `test(...)`, `refactor(...)`, `ci(...)`, `chore(...)`).
Body uses complete sentences and line-wraps around column 72.

---

## 2. Required reading before implementation

Read these three files end-to-end before writing code. Paths are
relative to repo root.

1. **`dev/INCIDENT-wechat-native-crash.md`** — full root cause analysis.
   Especially §3 (root cause), §4 (why re-runs sometimes succeed),
   §5 (why existing tests don't catch this). This tells you WHY
   this fix exists and the constraints it must honour.

2. **`src/qrstream/decoder.py`** — understand:
   - `_worker_detect_qr(frame_data)` (line ~263) — the function that
     currently calls `try_decode_qr` directly. We are redirecting
     its call to go through a module-level hook.
   - `_worker_detect_qr_clahe(frame_data)` (line ~313) — same
     redirection applies.
   - `extract_qr_from_video(...)` (line ~635) — where we install
     and tear down the sandbox.
   - `_stream_scan(...)` (line ~950) — you do NOT modify this
     function; just understand how it consumes worker results so
     you know what contract the sandboxed path must uphold.

3. **`src/qrstream/qr_utils.py`** — understand:
   - `try_decode_qr(frame, qr_detector=None) -> str | None` — the
     function we are wrapping. Note the per-thread
     `threading.local()` detector cache; in sandbox mode this cache
     lives inside helper subprocesses, one per helper.
   - `HAS_SEGNO` and `generate_qr_image(...)` — used by the
     sandbox test to synthesise a QR image without shipping a
     fixture file.

You do not need to read `encoder.py`, `lt_codec.py`, or `protocol.py`
for this task.

---

## 3. Decisions already made (do not re-litigate)

These were resolved in discussion and are not up for debate during
implementation:

| # | Decision |
|---|---|
| 1 | Default behaviour on ALL platforms: sandbox ON. |
| 2 | No `auto` mode. CLI choices are `on` and `off` only. |
| 3 | Pool size default: **3 helpers**. |
| 4 | When sandbox is active and detector crashes at least once, print a single summary line UNCONDITIONALLY (not gated on `--verbose`). |
| 5 | Sandbox test fixtures are SYNTHESISED via `qrstream.qr_utils.generate_qr_image`. Do not add any real video/image file to `tests/fixtures/`. |
| 6 | Minimal abstraction only. Do NOT introduce a `Detector` Protocol / base class / plugin registry. A single `DETECTOR_CAN_CRASH: bool` constant in `qr_utils.py` is the ONLY concession to "future MNN swap-in". |
| 7 | Sandbox scope: decoder detect path only. Do NOT touch encoder, LT codec, protocol, or any other subsystem. |
| 8 | Crash-burst abort policy: if `crash_count >= 3` within the first 10 seconds after the first crash, `detect()` starts raising `RuntimeError("Sandbox helpers repeatedly crashing; aborting")`. The 10s / 3 constants are constructor arguments on `SandboxedDetector` with these defaults. |
| 9 | No new dependencies. Everything uses Python stdlib `multiprocessing` + existing `numpy` + existing `cv2`. |

---

## 4. File-level implementation plan

### 4.1 NEW `src/qrstream/qr_sandbox.py`

One new module, approximately 200 lines. Full structure:

**Module docstring** explaining:
- Why the module exists (upstream opencv_contrib bug, link the
  issue number `#3570`),
- That it uses `spawn` start method (for macOS fork-safety and
  Windows compat),
- That it is crash-tolerant: helpers respawn on `exitcode != 0`,
- Design note: shared input queue (natural load-balancing across
  helpers), per-request single-slot response queue in the caller.

**Module-level constants / helpers:**

```python
_STOP = ("__STOP__",)   # sentinel object queued to ask a helper to exit cleanly
```

**`_helper_loop(in_q, out_q)`** — module-level function (pickle-safe;
required by `spawn`):

```python
def _helper_loop(in_q, out_q):
    """Subprocess entry point. Drains in_q forever, calls try_decode_qr,
    pushes (frame_idx, decoded_or_None) onto out_q. Returns on _STOP or
    on unrecoverable errors; a silent return is enough — the supervisor
    will see the dead process via is_alive()==False and respawn."""
    from .qr_utils import try_decode_qr

    while True:
        try:
            item = in_q.get()
        except (EOFError, KeyboardInterrupt):
            return
        if item == _STOP:
            return
        frame_idx, shape, dtype_str, raw = item
        try:
            frame = np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
            if not frame.flags.writeable:
                frame = frame.copy()
            decoded = try_decode_qr(frame)
        except Exception:   # Python-level errors → no-detect
            decoded = None
        try:
            out_q.put((frame_idx, decoded))
        except (BrokenPipeError, OSError):
            return
```

**`SandboxedDetector` class** — the main class. Full public surface:

```python
class SandboxedDetector:
    def __init__(self, pool_size: int = 3, *,
                 start_method: str = "spawn",
                 crash_abort_threshold: int = 3,
                 crash_abort_window: float = 10.0):
        """
        pool_size: number of helper subprocesses.
        start_method: 'spawn' | 'fork' | 'forkserver'. Default 'spawn'
            because it is the only method that works on macOS
            (OBJC_DISABLE_INITIALIZE_FORK_SAFETY) and Windows.
        crash_abort_threshold / crash_abort_window: if more than
            threshold crashes occur within window seconds of the
            first crash, detect() starts raising RuntimeError.
        """

    def detect(self, frame_idx: int, frame: np.ndarray,
               timeout: float = 30.0) -> str | None:
        """Thread-safe. Multiple worker threads may call concurrently.
        Returns decoded QR string, or None on no-detect / helper crash
        / timeout. Raises RuntimeError if the abort threshold is
        exceeded, or if the sandbox has been closed."""

    def close(self, timeout: float = 5.0) -> None:
        """Stop all helpers and join them. Idempotent."""

    @property
    def crash_count(self) -> int:
        """Number of helper subprocess crashes observed so far."""

    def __enter__(self): ...
    def __exit__(self, *exc): ...
```

Internal state:

- `self._ctx = multiprocessing.get_context(start_method)`
- `self._in_q: mp.Queue` — shared input queue (bounded:
  `maxsize=pool_size * 4` for backpressure).
- `self._out_q: mp.Queue` — shared output queue.
- `self._helpers: list[mp.Process]` — current helper pool.
- `self._results: dict[int, queue.Queue]` — maps `frame_idx` →
  single-slot `queue.Queue(maxsize=1)`; the thread-local waiter
  for each in-flight frame.
- `self._lock: threading.Lock` — protects `_results`, `_helpers`,
  `_closed`, `_crash_count`, `_first_crash_at`.
- `self._collector_thread: threading.Thread` — daemon thread that
  drains `_out_q` and polls helpers for liveness.
- `self._collector_stop: threading.Event`.

Key methods (internal):

- `_spawn_helper() -> mp.Process`: `self._ctx.Process(target=_helper_loop, args=(self._in_q, self._out_q), daemon=True)`; `start()`; return.
- `_collector_loop()`: in a loop while not stopped, do
  `(frame_idx, payload) = self._out_q.get(timeout=0.5)`. On timeout,
  call `_reap_dead_helpers()` and continue. On a real item, route
  to waiter in `self._results`.
- `_reap_dead_helpers()`: find any helpers with
  `is_alive() == False`. For each dead one, join it, bump
  `crash_count`, record `first_crash_at` if this is the first
  crash, pop one pending `frame_idx` from `_results` and put
  `None` into its waiter, and spawn a replacement helper.
- `detect()`: allocate a per-call waiter, register in `_results`,
  `put` to `_in_q`, then `get` from the waiter. On success return
  the payload; on timeout return `None`.

Thread-safety contract: **every mutation of `_results`,
`_helpers`, `_closed`, `_crash_count`, `_first_crash_at` must
happen under `self._lock`**. The monitor thread and user-facing
`detect()` / `close()` all respect this.

### 4.2 NEW `tests/test_qr_sandbox.py`

Tests for the sandbox module in isolation. Do NOT touch decoder or
real videos here. All QR frames are synthesised with
`qrstream.qr_utils.generate_qr_image`.

Required test cases:

1. `test_detect_roundtrip_returns_same_as_inprocess` — Build a QR
   image via `generate_qr_image(b"hello world", version=5)`; call
   `try_decode_qr(img)` in-process to get the expected string; open
   a `SandboxedDetector(pool_size=1)` and call `.detect(0, img)`;
   assert equal.

2. `test_multiple_frames_round_trip_concurrently` — Use a
   `ThreadPoolExecutor(max_workers=4)`; submit 20 frames
   concurrently to a shared `SandboxedDetector(pool_size=3)`; assert
   every submitted frame gets its own correct result back (no
   cross-contamination of `frame_idx` routing).

3. `test_worker_crash_is_recovered` — Monkey-patch
   `qrstream.qr_utils.try_decode_qr` (at the module level the
   helpers import from) with a function that does
   `os._exit(134)` when the frame content equals a specific
   sentinel byte pattern, and otherwise returns `"ok"`. Submit 10
   frames where exactly one has the sentinel pattern. Assert:
   - all 10 `.detect()` calls return (none hang);
   - the sentinel frame returns `None`;
   - the other 9 return `"ok"`;
   - `sandbox.crash_count >= 1`;
   - `sandbox.close()` completes within 5 s.

   **Implementation note:** monkey-patching across the process
   boundary requires either (a) putting the crash-trigger function
   in a dedicated helper module imported by the helper, or (b)
   using a subclass of `SandboxedDetector` that overrides
   `_helper_loop` to call a test-only target. Option (b) is
   simpler; pick whichever keeps the test readable.

4. `test_repeated_crashes_trigger_abort` — Make every call to
   `try_decode_qr` in the helper do `os._exit(134)`. Construct
   `SandboxedDetector(pool_size=1, crash_abort_threshold=3,
   crash_abort_window=10.0)`. Submit frames in a loop; assert that
   after at most ~5 submissions, `.detect()` raises
   `RuntimeError` whose message contains `"repeatedly crashing"`.
   Also assert `close()` still works after the abort.

5. `test_close_is_idempotent` — `close(); close()` does not raise.

6. `test_close_after_no_use` — Open a sandbox, immediately close,
   assert no zombie processes and no exception.

7. `test_detect_after_close_raises` — `.detect(0, frame)` after
   `.close()` raises `RuntimeError`.

8. `test_context_manager_closes_on_exit` — `with
   SandboxedDetector() as sb:` + exit cleans up.

9. `test_detect_timeout_returns_none_not_raises` — Use a
   helper implementation that sleeps forever on a specific frame;
   call `.detect(idx, frame, timeout=0.5)` and assert it returns
   `None` rather than raising (so callers uniformly treat IPC
   problems as no-detect).

10. `test_out_q_routing_under_many_in_flight` — Submit 100 frames
    concurrently via `ThreadPoolExecutor(max_workers=8)` to a
    `SandboxedDetector(pool_size=3)`. Assert every caller receives
    the result for its own `frame_idx` (test with content-encoded
    payloads: frame `i` embeds string `f"idx-{i}"`).

**Performance budget for the test file**: the whole file should run
in under ~15 seconds on a 4-core CI machine. Each `SandboxedDetector`
construction costs ~300–500 ms (spawn + cv2 import in child), so
factor out fixtures where possible.

### 4.3 EDIT `src/qrstream/qr_utils.py`

Add one module-level constant and its docstring. Place it near the
top, after the imports:

```python
# Future-facing flag. WeChatQRCode (opencv_contrib) has known
# unfixed native crashes in its bundled zxing code (issue
# opencv_contrib#3570). When someone swaps the detector out for
# a non-crashing implementation (e.g. an MNN-based QR pipeline),
# flip this to False and rely on callers to stop spawning
# sandboxes. Nothing in this module reads the flag; it is purely
# a signal consumed by `qrstream.decoder` / future code paths.
DETECTOR_CAN_CRASH: bool = True
```

No other changes in this file.

### 4.4 EDIT `src/qrstream/decoder.py`

Three edits, minimally invasive:

**Edit A — imports** (near top of file, after existing imports):

```python
from .qr_utils import try_decode_qr, DETECTOR_CAN_CRASH
from . import qr_sandbox
```

Add above the `_PROGRESS_BAR_THRESHOLD` line:

```python
# ── crash-isolation dispatch hook ────────────────────────────────
# Worker functions call _dispatch_detect instead of try_decode_qr
# directly. extract_qr_from_video swaps this to
# SandboxedDetector.detect when detect_isolation != 'off', and
# restores it on exit.

def _in_process_detect(_frame_idx: int, frame: "np.ndarray") -> str | None:
    return try_decode_qr(frame)

_dispatch_detect = _in_process_detect
```

**Edit B — replace `try_decode_qr(frame)` calls inside worker
functions**:

In `_worker_detect_qr`, change

```python
    qr_data = try_decode_qr(frame)
```

to

```python
    qr_data = _dispatch_detect(frame_idx, frame)
```

In `_worker_detect_qr_clahe`, change

```python
    qr_data = try_decode_qr(boosted)
```

to

```python
    qr_data = _dispatch_detect(frame_idx, boosted)
```

**Edit C — `extract_qr_from_video` signature and setup/teardown**:

Signature: add `detect_isolation: str = "on"` parameter, defaulting
to `"on"`. Keyword-only is fine.

Body wrap: put the existing function body inside a `try/finally`
that installs and uninstalls the sandbox. Pseudocode:

```python
def extract_qr_from_video(video_path, sample_rate=0, verbose=False,
                          workers=None,
                          detect_isolation: str = "on"):
    """...(update docstring to document detect_isolation param)..."""

    global _dispatch_detect
    _validate_isolation_mode(detect_isolation)

    sandbox = None
    original_dispatch = _dispatch_detect
    try:
        if detect_isolation == "on":
            try:
                sandbox = qr_sandbox.SandboxedDetector(pool_size=3)
                _dispatch_detect = sandbox.detect
            except Exception as exc:
                print(f"[sandbox] failed to initialise "
                      f"({exc}); falling back to in-process detection.")
                sandbox = None
        # else: 'off' → stay with _in_process_detect

        # <<< all the existing body goes here, UNCHANGED >>>
        result = ...
        return result
    finally:
        _dispatch_detect = original_dispatch
        if sandbox is not None:
            crashes = sandbox.crash_count
            sandbox.close()
            if crashes > 0:
                # Unconditional print (not gated on verbose) per
                # handoff decision #4.
                print(f"[sandbox] detector crashed {crashes} time(s) "
                      f"during decode; affected frames treated as "
                      f"no-detect. Decoding proceeded normally.")
```

Add a small module-level validator:

```python
def _validate_isolation_mode(mode: str) -> None:
    if mode not in ("on", "off"):
        raise ValueError(
            f"detect_isolation must be 'on' or 'off', got {mode!r}"
        )
```

Keep the existing workers/probe/scan/recovery code **completely
intact**. The only thing that changes for those code paths is that
`_dispatch_detect` now points into the sandbox.

### 4.5 EDIT `src/qrstream/cli.py`

In the `decode` subparser (the one with `dec.add_argument(...)`
calls for `video`, `output`, `sample-rate`, `workers`, `verbose`),
add:

```python
    dec.add_argument(
        '--detect-isolation', choices=['on', 'off'], default='on',
        help='Isolate the WeChat QR detector in subprocess helpers '
             'so a native crash (opencv_contrib#3570) degrades to '
             'a single dropped frame instead of killing the decode '
             'process. Default: on. Use "off" to trade safety for '
             '~20-30%% throughput when you know your input is safe.')
```

In `cmd_decode`, pass the flag through:

```python
    blocks = extract_qr_from_video(
        args.video, args.sample_rate, args.verbose, args.workers,
        detect_isolation=args.detect_isolation)
```

No other CLI changes.

### 4.6 NEW `tests/test_decoder_sandbox_integration.py`

Cover the decoder/sandbox wiring. Use the existing fixture
`tests/fixtures/real-phone-v4/v073-10kB.mp4` which is short and
stable. No new fixtures.

Required test cases:

1. `test_extract_with_isolation_on_matches_off` — Run
   `extract_qr_from_video(path, detect_isolation='off')` and
   `extract_qr_from_video(path, detect_isolation='on')` on the same
   fixture; both must return block lists that, when fed to
   `LTDecoder`, yield the same recovered bytes. Mark `@pytest.mark.slow`.

2. `test_dispatch_detect_is_restored_after_decode` — Record
   `decoder._dispatch_detect`, run `extract_qr_from_video(...,
   detect_isolation='on')` once, assert
   `decoder._dispatch_detect is decoder._in_process_detect` after
   the call. Also test with `detect_isolation='off'`.

3. `test_extract_rejects_invalid_isolation_mode` —
   `extract_qr_from_video(path, detect_isolation='auto')` raises
   `ValueError`. (We removed the `auto` mode; make sure nobody can
   sneak it in.)

4. `test_extract_falls_back_when_sandbox_init_fails` —
   Monkey-patch `qr_sandbox.SandboxedDetector` to raise on
   construction; assert `extract_qr_from_video(..., detect_isolation='on')`
   still completes (falls back to in-process) and emits the warning
   line. Mark `@pytest.mark.slow` if it actually decodes video,
   else keep it quick by using a tiny synthetic video.

### 4.7 NEW `tests/test_cli_detect_isolation.py`

Small test for the CLI flag. A few assertions:

1. `test_cli_accepts_detect_isolation_on` — parse
   `['decode', 'x.mp4', '-o', 'y', '--detect-isolation', 'on']`;
   assert `args.detect_isolation == 'on'`.

2. `test_cli_accepts_detect_isolation_off` — same for `'off'`.

3. `test_cli_default_is_on` — omit the flag; default is `'on'`.

4. `test_cli_rejects_invalid_value` — `--detect-isolation auto` →
   `SystemExit` (argparse behaviour).

Use `qrstream.cli.build_parser()` directly; no subprocess.

### 4.8 EDIT `README.md` and `README-zh.md`

Append a short "Troubleshooting: decoder native crashes" section at
the end of each. English version template (translate to Chinese for
the zh version with matching technical detail):

```markdown
### Decoder native crashes

If `qrs decode` exits with `trace trap` or a SIGSEGV/SIGTRAP
message, you are hitting a known unfixed crash in the WeChat QR
detector bundled with `opencv_contrib` (upstream issue
`opencv_contrib#3570`). Since v0.8.0, `qrs decode` runs detection
in subprocess helpers by default, so a single crashing frame is
caught and treated as a dropped frame — the decode continues and
completes as long as LT overhead (see `--overhead`) is ≥ 1.5.

To see whether the subprocess sandbox caught any crashes, look for
a summary line like `[sandbox] detector crashed N time(s) during
decode`. If you believe the sandbox overhead is unnecessary on your
input, you can disable it at your own risk with
`--detect-isolation off`.
```

### 4.9 EDIT `dev/INCIDENT-wechat-native-crash.md`

Append a new section at the very end:

```markdown
## 10. Resolution

Implemented in branch `fix/wechat-native-crash`, PR to `dev`.

Approach: section §7 option (A), subprocess-isolated detector,
default on, overridable with `--detect-isolation off`.

Key design deviations from §7 (A) as written:

- No `auto` mode; sandbox is ON on all platforms by default (see
  handoff decision #1). Rationale: the native crash is not
  platform-specific in principle — it's a content-dependent native
  OOB read that happens to fire more often on macOS arm64 due to
  ASLR but can fire anywhere.
- `pool_size = 3` (not 2) to provide more headroom during helper
  respawn without materially increasing memory footprint.
- Unconditional one-line summary when any crash was caught, not
  gated on `--verbose`.
- Minimal future-proofing: single `DETECTOR_CAN_CRASH: bool`
  constant in `qr_utils.py`. No `Detector` abstract base class, no
  plugin registry; those are postponed until an actual
  non-crashing detector (e.g. MNN) is ready to land.

The §7 option (B) `skip_crc` hardening and option (C) new fixture
video were explicitly out of scope for this PR.
```

---

## 5. IPC data format (one-page reference)

Worker thread → helper subprocess, via `_in_q`:

```
(frame_idx: int,
 shape:     tuple[int, ...],   # e.g. (720, 720, 3)
 dtype_str: str,                # e.g. '|u1' for uint8
 raw:       bytes)              # arr.tobytes()
```

Helper reconstructs via
`np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)` and
`.copy()`s if necessary to get a writable ndarray.

Helper subprocess → worker thread, via `_out_q`:

```
(frame_idx: int,
 decoded:   str | None)
```

---

## 6. Error / abnormal situations — required behaviour

| Situation | Required behaviour |
|---|---|
| Helper SIGSEGV / SIGTRAP / SIGBUS / SIGABRT during `detect` | `crash_count += 1`; pop one pending `frame_idx` from `_results` and put `None` into its waiter; spawn a replacement helper; decode continues. |
| Helper exits cleanly with `_STOP` (we asked it to) | Normal path in `close()`. Not a crash. Do not increment `crash_count`. |
| `crash_count` reaches `crash_abort_threshold` within `crash_abort_window` seconds of the first crash | `detect()` starts raising `RuntimeError("Sandbox helpers repeatedly crashing; aborting")`. `extract_qr_from_video` catches and re-raises, aborting the decode. |
| `detect()` timeout (waiter queue empty after `timeout`) | Return `None` (treat as no-detect). Do not raise. Clean up `_results[frame_idx]` entry. |
| `_in_q.put` timeout (queue full) | Raise `TimeoutError` out of `detect()`. The worker thread's try/except in `_stream_scan` will treat it as no-detect. |
| `SandboxedDetector` construction failure (e.g. `mp.get_context('spawn')` blows up) | `extract_qr_from_video` catches, prints a fallback warning, proceeds with in-process detection. |
| User `Ctrl+C` during decode | Finally block runs; sandbox.close() runs; helpers get `_STOP` sentinels (up to `pool_size` of them), are given 5 s to exit, then `terminate()` + 1 s join, then forced. |
| Normal decode completion | Finally block restores `_dispatch_detect`; sandbox.close() drains and joins helpers; one-line summary printed iff `crash_count > 0`. |

---

## 7. Not in scope

Do NOT do any of the following in this PR:

- Do not add a `Detector` abstract base class or Protocol.
- Do not add a plugin / entry-point / environment-variable-driven
  detector selector.
- Do not change the encoder side (encode uses ThreadPool and has no
  native crash; sandbox is decoder-only).
- Do not change `skip_crc=True/False` policy in `_stream_scan` or
  elsewhere. This is a separate defensive hardening tracked in
  `INCIDENT.md §7 (B)`.
- Do not add new fixture videos under `tests/fixtures/`. All
  sandbox tests synthesise QR images in-memory.
- Do not enable any `multiprocessing` start method other than
  `spawn` by default.
- Do not touch `archive/*` branches.
- Do not modify `BRANCHING.md`, `pyproject.toml`, or any
  `.github/workflows/*.yml`. (Dependencies are unchanged; new tests
  are picked up by existing `pytest tests/` invocation.)

---

## 8. Commit / PR plan

Two commits on the `fix/wechat-native-crash` branch, both authored
by the implementing agent. Do not squash them before PR.

### Commit 1 — core sandbox, no wiring

- New file: `src/qrstream/qr_sandbox.py`
- Edit: `src/qrstream/qr_utils.py` (add `DETECTOR_CAN_CRASH`)
- New file: `tests/test_qr_sandbox.py`

Suggested message:

```
feat(sandbox): add SandboxedDetector for WeChat native crash isolation

Introduces qr_sandbox.SandboxedDetector, a pool of subprocess
helpers that run try_decode_qr under crash-isolated conditions.
When a helper dies from a native signal, the supervisor reaps it,
satisfies one pending frame request with None (no-detect), and
spawns a replacement helper so throughput does not collapse.

Also adds qr_utils.DETECTOR_CAN_CRASH = True as a future-facing
flag for when the WeChat backend is swapped out.

Decoder wiring and CLI flag are added in a follow-up commit.
```

Verify Commit 1 passes: `uv run pytest tests/test_qr_sandbox.py -v`.

### Commit 2 — decoder and CLI wiring

- Edit: `src/qrstream/decoder.py`
- Edit: `src/qrstream/cli.py`
- New file: `tests/test_decoder_sandbox_integration.py`
- New file: `tests/test_cli_detect_isolation.py`
- Edit: `README.md`
- Edit: `README-zh.md`
- Edit: `dev/INCIDENT-wechat-native-crash.md` (append §10 Resolution)

Suggested message:

```
feat(decoder): route detect through SandboxedDetector by default

extract_qr_from_video now owns a SandboxedDetector for the whole
decode run when detect_isolation='on' (the default). Worker
threads dispatch through a module-level hook so the sandbox swap
is transparent to _worker_detect_qr / _worker_detect_qr_clahe.

The CLI adds --detect-isolation {on,off}, default on. Users on
known-safe inputs can opt out for ~20-30%% throughput.

A single post-decode summary line is printed whenever the sandbox
observed at least one helper crash, regardless of --verbose.
```

Verify Commit 2 passes: `uv run pytest tests/ -v` (default run;
slow tests are excluded by `addopts = "-m 'not slow and not e2e'"`)
and `uv run pytest tests/ -v -m slow` (needs real videos; expected
to pass).

### After both commits land

Do NOT push yet. Leave the branch local. The maintainer will
review and push / PR manually.

---

## 9. Verification checklist (run this after implementation)

```bash
cd /data/workspace/qrstream-enhanced
uv sync --no-progress

# Unit tests (fast path; excludes slow + e2e by default)
uv run pytest tests/ -v

# Sandbox-specific unit tests
uv run pytest tests/test_qr_sandbox.py -v
uv run pytest tests/test_cli_detect_isolation.py -v

# Integration tests that exercise sandbox against real fixtures
uv run pytest tests/test_decoder_sandbox_integration.py -v -m slow

# Full slow suite (phone-recording fixtures must still pass)
uv run pytest tests/ -v -m slow

# E2E encode-decode (make sure our hook didn't break the pipeline)
uv run pytest tests/ -v -m e2e
```

All green before declaring done.

Sanity-check the CLI interactively:

```bash
# Should print the new flag
uv run qrs decode --help | grep -A2 detect-isolation

# Default behaviour (sandbox on)
uv run qrs decode tests/fixtures/real-phone-v4/v073-10kB.mp4 \
    -o /tmp/out.bin -v

# Explicit off
uv run qrs decode tests/fixtures/real-phone-v4/v073-10kB.mp4 \
    -o /tmp/out2.bin --detect-isolation off -v

# Invalid value should error out
uv run qrs decode tests/fixtures/real-phone-v4/v073-10kB.mp4 \
    -o /tmp/out3.bin --detect-isolation auto
# expected: argparse error about invalid choice
```

---

## 10. Anticipated pitfalls

- **`os.fork()` on macOS**: do NOT use `fork` as the start method.
  macOS 10.13+ requires `OBJC_DISABLE_INITIALIZE_FORK_SAFETY` and is
  a known source of deadlocks with OpenCV. We use `spawn`
  unconditionally.
- **Pickling a lambda for `multiprocessing.Process(target=...)`**:
  `_helper_loop` must be a module-level function. Do not use a
  closure or lambda.
- **Shared mutable state across helpers**: `_in_q` and `_out_q` are
  the ONLY shared state. No globals, no file locks, no shared
  memory.
- **`queue.Queue` vs `mp.Queue`**: single-slot waiters for
  `_results[frame_idx]` are `queue.Queue` (thread-local inside the
  parent process). The cross-process queues are `mp.Queue`. Do not
  mix them up.
- **Detector import cost**: cv2 + WeChat detector construction
  takes several hundred milliseconds per helper. All tests that
  construct a `SandboxedDetector` pay this. Factor fixtures where
  possible.
- **Test pollution**: monkey-patching
  `qrstream.qr_utils.try_decode_qr` in a parent process does NOT
  propagate to `spawn`ed helpers — they re-import the module fresh.
  Use a helper-module-level injection (e.g. set an environment
  variable that `_helper_loop` reads) for tests that need to
  simulate crashes.
- **Test cleanup**: every test that opens a `SandboxedDetector`
  must close it (use `with` or `try/finally`). A leaked helper
  holds the test process open and turns a fast test into a
  CI hang.

---

## 11. Definition of done

- Both commits exist on `fix/wechat-native-crash` and are NOT
  pushed.
- `uv run pytest tests/` is green (default and `-m slow` and `-m e2e`).
- `uv run qrs decode --help` shows the new flag.
- `dev/INCIDENT-wechat-native-crash.md` has a §10 Resolution
  section written.
- Both READMEs have a Troubleshooting section added.
- No changes outside the files listed in §4.
- No new dependencies in `pyproject.toml`.
- No modifications to `BRANCHING.md`, `.github/workflows/*`, or any
  `archive/*` refs.

When all above are true, report back for review.
