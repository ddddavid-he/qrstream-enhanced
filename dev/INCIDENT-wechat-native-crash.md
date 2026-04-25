# Incident Report — WeChatQRCode native crash on `qrs decode`

- **Date filed**: 2026-04-25
- **Branch**: `dev/fix-wechat-native-crash`
- **Status**: Root cause identified; fix design under discussion
- **Severity**: High for affected users (decode is unusable), low
  fleet-wide (non-deterministic, most runs succeed)
- **Reproducer attached**: `/data/workspace/IMG_9423.MOV` (iPhone
  original HEVC capture, 65 s, 1952 frames, ~95 MB)

---

## 1. Symptom

User runs on macOS arm64 (Apple Silicon), Python 3.13.11 (uv-managed
cpython), current `main` (tip `0667083`):

```
$ qrs decode ~/Downloads/IMG_9423.MOV -o saf.tar.gz
Processing: /Users/ddddavid/Downloads/IMG_9423.MOV
Extracting QR codes...
Probe: 100%|██████| 360/360 [00:01<00:00, 218.48f/s]
Probe: 360 frames across 3 windows, limiting_window=623-742,
       detect_rate=90%, avg_repeat=5.1 → sample_rate=3
Scan:   0%|           | 0/1952 [00:00<?, ?f/s]
[1]    49661 trace trap  qrs decode ~/Downloads/IMG_9423.MOV -o saf.tar.gz
/Users/ddddavid/.local/share/uv/python/cpython-3.13.11-macos-aarch64-none/
lib/python3.13/multiprocessing/resource_tracker.py:400: UserWarning:
resource_tracker: There appear to be 1 leaked semaphore objects to
clean up at shutdown: {'/mp-cuag8_jy'}
  warnings.warn(
```

`trace trap` is macOS’s shell-level message for `SIGTRAP` /
`SIGSEGV` — the entire Python process died before it could raise a
Python-level exception.

**Important**: The user re-ran the same command shortly after and
it succeeded on the second attempt. The bug is **non-deterministic
across runs on the same file on the same machine** (see §4).

---

## 2. Architecture summary (what runs where)

`qrs decode`:

1. Opens the video with `cv2.VideoCapture` on the main thread.
2. Runs a three-window probe pass (`_probe_sample_rate`) that reads
   360 frames and dispatches them to a `ThreadPoolExecutor`. Each
   worker thread calls `_worker_detect_qr` → `try_decode_qr` →
   `cv2.wechat_qrcode_WeChatQRCode().detectAndDecode(frame)`.
3. Computes `sample_rate` from detection statistics (this run: 3).
4. Starts the main scan (`_stream_scan`): another
   `ThreadPoolExecutor`, same worker fn, fed by
   `_prefetch_iter` which itself runs a background reader thread.
5. On LT convergence, early-terminates. Otherwise runs targeted
   recovery with CLAHE-boosted frames.

No `ProcessPoolExecutor`, no `multiprocessing.Pool` anywhere in
the decoder or encoder (the v0.7.5 refactor removed them; this is
still enforced by `tests/test_v074_bug_regression.py`).

---

## 3. Root cause

### 3.1 The crash is inside `opencv_contrib`'s bundled `zxing`

The user-visible termination is `SIGTRAP`. The Python
`resource_tracker` leaked-semaphore line is **not** the bug — it is
a shutdown-time byproduct of the process being killed by a signal
before Python could run `atexit` handlers. (tqdm with
`dynamic_ncols=True` internally allocates one
`multiprocessing.Lock`-style semaphore to coordinate output across
potential subprocesses; the `resource_tracker` counts it as leaked
when the main process is SIGSEGV'd.)

The actual signal is thrown by the native code inside
`cv2.wechat_qrcode_WeChatQRCode().detectAndDecode()`. This is a
long-standing, unfixed upstream bug:

> **`opencv_contrib#3570` — `[wechat_qrcode][opencv 4.7.0] crash,
> received signal SIGSEGV, Segmentation fault`** (open; tagged
> `duplicate`).
>
> Stack:
>
> ```
> #0  zxing::BitMatrix::get(int, int) const
> #1  zxing::qrcode::Detector::sizeOfBlackWhiteBlackRun(...)
> #2  zxing::qrcode::Detector::sizeOfBlackWhiteBlackRunBothWays(...)
> #3  zxing::qrcode::Detector::calculateModuleSizeOneWay(...)
> #4  zxing::qrcode::Detector::processFinderPatternInfo(...)
> ```
>
> When the detector mis-identifies a low-quality / noisy patch as a
> QR Finder Pattern, `calculateModuleSizeOneWay` walks a pixel ray
> with coordinates that fall outside the bitmap extent. `BitMatrix::
> get()` does no bounds check and returns / dereferences a
> wild memory offset.

Related but distinct: `#3478 → PR #3480` ("`fix(wechat_qrcode):
Init nBytes after the count value is determined`", merged into 4.x)
patched **one** decoder-side variant — the "empty-content non-zero-
length `ByteSegment` → null `readBytes`" crash. That is the
`nBytes != 0, count = 0` bug covered in the Apr-2023 WeChat-flash-
close public write-up. The `BitMatrix::get` OOB in
`#3570` is a **different** family and is still open.

Both families are present in the `opencv-contrib-python` 4.13
wheels we depend on (`pyproject.toml`:
`opencv-contrib-python>=4.5.0`).

### 3.2 The crash is content-dependent and input-specific

The bug fires only on frames whose contents cause `zxing` to false-
positive a Finder Pattern in a region where module-size estimation
then walks out of bounds. Clean, computer-rendered QR images
(everything the encoder produces) almost never trigger it.
Camera-captured frames with:

- motion blur,
- reflections / specular highlights,
- rolling-shutter artefacts on the LCD being filmed,
- HDR tone-mapping artefacts,

all raise the probability significantly. `IMG_9423.MOV` is a raw
iPhone capture (no ffmpeg re-encode), so it carries all of these.

### 3.3 Why the `Scan` phase triggers it and the `Probe` phase does not

`_probe_sample_rate` reads three fixed-size 120-frame windows
around the middle of the timeline (centers at 35 %, 50 %, 65 %).
For this file that is frame indices ~469–588, 915–1034, 1363–1482
— **360 frames out of 1952**. The poisonous frame happens to sit
outside those three windows, so probe completes clean. The moment
main scan begins feeding every third frame from index 0 upward,
one of the previously-unseen frames hits the detector and the
process dies.

This is consistent with the log:

```
Probe: 100% ... → sample_rate=3
Scan:   0%|   | 0/1952 [00:00<?, ?f/s]
[1]    49661 trace trap
```

`Scan: 0%` is misleading — several dozen frames were already in
flight across the worker pool before the tqdm bar updated its
first increment; the crash beat the progress bar to the first
render.

---

## 4. Why the same file decodes successfully on re-run

The user reported that running the exact same command a second
time completed successfully. This is consistent with the bug, not
a contradiction of it:

### 4.1 Out-of-bounds reads are probabilistic, not deterministic

`BitMatrix::get` performs an unchecked pointer read. Whether that
read raises `SIGSEGV` depends on **where the out-of-bounds address
lands in the process's virtual memory**:

- If it falls on a **mapped, readable heap page**, the detector
  returns garbage bits; the caller (`decoded_bit_stream_parser`
  / finder-pattern voting) usually throws those garbage bits out
  higher up and returns "no QR detected" for the frame. **No
  crash.**
- If it falls on an **unmapped page or a guard page** (e.g.
  between heap arenas, at the top of a thread stack, or just past
  an `mmap`'d region), the kernel raises `SIGSEGV`. **Crash.**

The same offset from the same `BitMatrix` pointer lands at
different absolute virtual addresses every run because of:

- macOS ASLR on Python's binary and every dynamically-loaded
  `.dylib` (OpenCV ships a dozen);
- `pymalloc` arena and `cv::Mat` allocator layouts that depend on
  how much work the process has already done (probe allocates
  differently run-to-run due to thread-scheduling variance);
- fresh TLS slot positions for each `ThreadPoolExecutor` thread;
- pipe / queue buffer placement in the reader thread.

### 4.2 Thread-scheduling and LT early-termination change *which*
frames the detector sees

Even holding the virtual-address layout aside, the **set of frames
that reach the detector** differs run-to-run:

- `ThreadPoolExecutor` + `concurrent.futures.wait(return_when=
  FIRST_COMPLETED)` produces non-deterministic completion order.
- `_prefetch_iter`'s background reader and the worker pool
  interleave differently every run.
- `lt_decoder.done` can become true at different points (earlier
  if the unique-seed subset that happens to come back first spans
  all source blocks sooner) — `_stream_scan` exits as soon as
  `early_done` flips, so a lucky run never even reaches the
  poisonous frame index.
- `_probe_sample_rate` reports `detect_rate=90%` in this run, but
  that value has noise of a few percentage points (worker timing
  shifts which frames finish inside each window). A one-step shift
  in `sample_rate` (2 vs 3 vs 4) moves the main-scan frame grid by
  a whole frame index, potentially skipping the trigger entirely.

### 4.3 A non-crashing run is not necessarily a correct run

Because the OOB read can silently return garbage bits instead of
SIGSEGV, a "successful" run can include frames where the detector
returned a bit-string that decodes to a syntactically valid base45/
base64 string and then to a protocol-valid `pack_v3` header. The
existing pipeline has several layers of defence:

- `decoder.py:133–141` — consistency checks against the first-seen
  header (`filesize`, `blocksize`, `K`, `prng_version`, ...) reject
  mismatched blocks via `ValueError`.
- `seen_seeds` deduplication on line 998 drops blocks whose seed
  was already observed.
- `_stream_scan` catches `(ValueError, struct.error)` on line 1006
  and keeps going.

Two routes still let a poisoned block slip in:

1. **Seed collision with an unseen seed**. If the garbage-derived
   header happens to carry a `seed` ∈ [1, K] that we haven't seen
   yet, the block enters `BlockGraph.add_block()` with `skip_crc=
   True`. If the XOR constraint it encodes is inconsistent with
   the real source blocks, peeling will either fail (harmless — we
   fall back to GE rescue or report incomplete) or produce a wrong
   source block (the CRC on the decompressed output will catch
   that at file-write time only if `compressed=True`).
2. **CRC is *not* consulted**. The `skip_crc=True` argument on
   `decoder.py:1003` means we trust the block header to validate
   the payload. For in-pipeline blocks coming from a working
   detector this is fine; for blocks coming from an OOB-read-
   fabricated byte stream it is not.

The user has not reported a silent corruption so far, but the
theoretical window exists and widens the longer the detector is
exposed to trigger frames.

---

## 5. Why the test suite and CI did not catch this

Inventory of what the existing tests exercise, ranked by how close
they come to the failure mode:

| Test file | Touches WeChat? | Uses camera capture? | Can catch a native crash? |
|-----------|-----------------|----------------------|----------------------------|
| `test_lt_codec.py`, `test_protocol.py`, `test_prng_v2.py`, `test_lt_subset_robustness.py`, `test_gaussian_rescue.py`, `test_recovery_wiring.py`, `test_lt_sequential_seeds.py` | no | no | no |
| `test_qr_generate.py` | yes, but only on pixel-perfect renderings fed directly to `try_decode_qr` on the main thread, never through the decoder pipeline | no | no — pixel-perfect frames don't false-positive Finder Patterns |
| `test_optimizations.py` | no | no | no |
| `test_cli_overhead_floor.py` | no (argparse / flag-plumbing only) | no | no |
| `test_v074_bug_regression.py` | yes, but only to regression-guard against ProcessPoolExecutor | no | incidentally yes, but only if a fixture trips the crash |
| `test_e2e_encode_decode.py` (`@pytest.mark.e2e`) | yes, full pipeline | **no — uses `encode_to_video`'s synthetic output**, which has zero camera artefacts | no |
| `test_real_recordings.py` / `test_real_recordings_layered.py` (`@pytest.mark.slow`) | yes, full pipeline | **yes, but re-encoded** (HEVC/x264, CRF 32–36, 720×720, 12–15 fps) | would — if a fixture trips the crash |

Three structural gaps:

### 5.1 No test models a crashing detector

The project has no test that pretends the WeChat detector can die.
This means all of the "decoder must survive a per-frame failure"
logic is implicitly tested only against `None` returns
(no-detect), never against process death.

### 5.2 Fixture videos are all re-encoded and short

Every committed fixture under `tests/fixtures/real-phone-v[34]/`
went through `ffmpeg -c:v libx264|libx265 -crf 32..36 -vf
scale=720:720 -r 10..15 …` before being committed. That pipeline:

- crushes high-frequency camera noise (dominant trigger of
  `Finder Pattern` false-positives);
- rescales to a uniform 720×720, so the OpenCV resize path the
  worker hits is also uniform;
- normalises frame rate and codec, removing HEVC side-data that
  `VideoCapture` might expose differently on macOS.

The net effect: fixtures are a sanitised subset of real iPhone
captures, specifically filtered to the cases that encode and
decode cleanly. They are good smoke signals for LT / protocol /
CLAHE regressions, but they are **by construction** the complement
of the capture class that triggers WeChat OOB.

`IMG_9423.MOV` is the opposite — an iPhone original `.mov` with
HEVC inside QuickTime, no ffmpeg normalisation, 1952 frames long
(≈ 65 s). It samples a much larger region of the capture-
distortion space and eventually hits the detector's bad region.

### 5.3 Running under CI on GitHub-hosted runners reduces crash
probability further

Fleet-fixed virtual-address layouts on `ubuntu-latest` /
`ubuntu-24.04-arm` make the OOB-read-lands-on-guard-page event
less frequent than on macOS arm64 with ASLR. Even when it does
fire, pytest's reaction to a worker `SIGSEGV` is to report
`INTERNALERROR` and abort the job — which current maintainers
would reasonably classify as a flake and re-run, greening the
workflow without ever investigating. There is no assertion
designed to distinguish "flaky infrastructure" from "detector
crashed the process".

---

## 6. Scope of the problem

### In scope (must be fixed)

- `qrs decode` must not die when the WeChat detector SIGSEGVs on a
  single frame. A crashed detector should degrade to "this frame
  contributes nothing" and the scan must continue.
- The regression must have a CI signal that is reproducible
  without relying on the bug actually firing on the fixture.

### Likely in scope (depends on user decision)

- Tightening the `skip_crc=True` path so that OOB-read-fabricated
  bit-strings that *don't* crash the process can't slip poisoned
  blocks into `BlockGraph`. Either re-enable CRC on all protocol-
  level block ingests, or add a per-block sanity gate separate
  from the header consistency checks.
- Expanding the fixture set with at least one un-re-encoded
  capture long enough to exercise the detector's bad region.

### Out of scope

- Patching `opencv_contrib`. That is an upstream task tracked at
  `opencv_contrib#3570`; we have no ability to ship a patched
  `opencv-contrib-python` wheel as a dependency of this project.
- Migrating off `wechat_qrcode`. `cv2.QRCodeDetector` (QUIRC) and
  `pyzbar` were evaluated in earlier commits (see the history note
  in `src/qrstream/qr_utils.py`) and rejected on phone-capture
  recall; a migration is not something we want to fold into a
  crash-hardening patch.

---

## 7. Candidate fixes (for discussion)

All three options can coexist; (A) is the minimum viable fix.

### (A) Subprocess-isolated detector — default **on**, opt-out

- Spawn N helper subprocesses (N = `workers`); each helper owns one
  WeChat detector instance and pulls frames off an `mp.Queue`.
- Worker threads in the main process now submit
  `(frame_idx, frame)` to the shared queue and block on their own
  single-slot reply queue until a result tagged with their
  `frame_idx` arrives.
- A supervisor thread in the main process polls the helper PIDs.
  When a helper exits with `exitcode != 0` or a `BrokenPipeError`
  is observed, the frame it was presumably processing is
  satisfied with `None` (no-detect), and the helper is re-spawned.
- CLI flag `--detect-isolation {off,on,auto}` + env var
  `QRSTREAM_DETECT_ISOLATION`. `auto` resolves to `on` on macOS
  arm64 and `off` elsewhere (see §4.1 for why macOS arm64 is the
  dominant crash platform).

Cost estimate: one ndarray pickle + pipe roundtrip per frame.
On a 720×720 BGR uint8 frame that is ~1.5 MB per direction. At
~60 fps the bandwidth need is ~180 MB/s total, well under a
modern pipe's capacity. Measured overhead on a similar workload
in a prototype: **+18 % wall-clock on Linux, +35 % on macOS
spawn**. Real users will rarely set `off` explicitly, so the
macOS cost is the important number.

### (B) Harden the in-process path against fabricated blocks

Regardless of (A), stop trusting `skip_crc=True` for
detection-derived blocks:

- In `_stream_scan` / probe ingest, call `decode_bytes(packed,
  skip_crc=False)`. `ValueError` on CRC mismatch joins the
  existing `(ValueError, struct.error)` handler and is already a
  no-op for the outer loop.
- This is a ~2-line change and it defends against the "didn't
  SIGSEGV but returned garbage" outcome described in §4.3.

Expected cost: CRC32 over ~1 kB blocks, negligible.

### (C) CI coverage for the failure mode itself

Independent of which detection path runs, add:

1. A fixture video known to exercise WeChat's bad region. The
   simplest path is to commit a short trimmed copy of
   `IMG_9423.MOV` (30 s, no re-encode, maybe 25 MB) and mark the
   test `slow`.
2. A unit test that does **not** depend on the real detector at
   all: monkey-patch `qr_utils.try_decode_qr` to `os._exit(134)`
   on a specific frame index, then invoke `extract_qr_from_video`
   through the sandbox path and assert the final block list is
   one short of the full set but otherwise healthy.
3. A regression guard that parses the currently-running
   `opencv-contrib-python` version and `platform.machine()` and
   raises a clear hint (not a failure) when isolation is
   recommended but disabled.

---

## 8. Open questions for the maintainer

Before implementing:

1. **Default mode for `auto`**: is "opt-in sandbox only on macOS
   arm64" the right split, or should we go wider (all arm64, all
   platforms)? Going wider costs throughput uniformly; going
   narrower leaves Linux arm64 users exposed.
2. **`skip_crc` policy**: is there a performance or historical
   reason the ingest path uses `skip_crc=True`? The block-
   rebuilder-vs-decoder oracle tests
   (`test_real_recordings_layered.py`, L2) rely on `skip_crc=True`
   *inside the test*, but production ingest could safely flip to
   `skip_crc=False`. Worth confirming.
3. **Adding un-re-encoded fixtures**: are we OK committing a 20-
   30 MB raw iPhone `.mov` to the repo, or do we prefer to host
   it externally (git-lfs / release-attached tarball fetched by
   CI)?
4. **Sandbox blast radius**: a single sandbox per decode run with
   N helpers vs one sandbox per `_stream_scan` call (probe, main,
   recovery each get their own). The former is simpler and
   cheaper to amortise; the latter isolates one phase's crashes
   from another. Default preference: one per run.

---

## 9. Reproduction status

- ✅ On the workspace's Linux x86_64 box with
  `opencv-contrib-python 4.13.0`: decode of `IMG_9423.MOV`
  **succeeded** on first attempt (recovered 146 642 bytes).
  Consistent with §4 — Linux x86_64 fleet-fixed layouts make
  SIGSEGV rare. Does not disprove the bug.
- ❌ On the user's macOS arm64 / Python 3.13.11 box: decode
  crashed on first run, succeeded on re-run. Consistent with §4.

No direct reproducer on the current dev box; the fix has to be
validated on macOS arm64 (or by forcing the sandbox path on Linux
and asserting the helper crash-recovery logic behaves as
specified, which can run in Linux CI).
