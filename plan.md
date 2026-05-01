# qrstream UI/UX refactor — v2

Rich-driven CLI UX rework for `qrstream`.  Removes `tqdm` entirely,
unifies all progress/status output under a single `--output-mode` flag,
introduces a qBittorrent-style source-block location map for file
recovery, and splits decode into four single-word phases with
purpose-specific visualisations.

---

## 1. Output modes

Both `encode` and `decode` take a single new flag:

```
--output-mode auto|interactive|log|quiet|verbose
```

The legacy `-v / --verbose` flag is kept as a hidden alias that
upgrades `auto` to `verbose`, so existing scripts and tests keep
working.

| Mode          | Shape                                             | Typical use                     |
|---------------|---------------------------------------------------|---------------------------------|
| `auto`        | Rich interactive on TTY; `log` otherwise          | default                         |
| `interactive` | Rich animated UI (always)                         | human viewing                   |
| `log`         | Append-only `key=value` lines                     | CI / `tee` / issue repro        |
| `quiet`       | Only errors + one final success line              | scripted invocations            |
| `verbose`     | Rich + diagnostic events (TTY) / verbose-log      | debugging and performance runs  |

All progress/status text goes to `stderr`.  Errors/warnings that are
covered by `capsys`-based tests (`test_cli_overhead_floor.py`) still
use plain `print(...)` to `stdout` so the tests keep passing.

---

## 2. Decode UX (four phases, one word each)

### 2.1 `Probe`

Spinner while running; one-line summary on completion.

```
Probe  ⠋ detecting sample rate / crop / ppm
Probe  ✓  sample=2  detect=68%  repeat=1.9  crop=-64%
```

`crop_reduction = 1 − (crop_area / frame_area)` (displayed as
`crop=off` when no ROI is derived; logged as `crop_reduction=64%`).

### 2.2 `Scan`

Thin pip/rich-style coloured bar + sliding-window hit rate + wide
file-block map.  Neither the sample rate nor the raw `691/1827`
counts are displayed.

```
Scan   ━━━━━━━━━━━━━━━╸━━━━━━━━━━━━  52%  hit 68%
File   █████▓▓██░░░░███░░░▓▓████░░░░░░░░░░░░░░  37.8%
```

`hit` is a 128-sample sliding window over per-frame "QR decoded?"
(0/1) results.

### 2.3 `Recover`

Thin bar + hit window + **range strip** (segments projected onto the
video timeline) + file-block map.

```
Recover  ━━━━━╸━━━━━━━━━━━━━━━━━━  31%  hit 74%
Range    ···████·····███······█████··········
File     █████▓████░░░░███▓░░░░██████░░░░░░░░░  57.1%
```

Range strip colours: pending = yellow, current = bright cyan,
scanned = green, idle = grey.

### 2.4 `Save`

Default mode shows only a completion line; there is no per-block
write progress bar.  Verbose surfaces an extra `debug` diagnostic.

```
Save   ✓  report.pdf  12.4 MB
```

---

## 3. Encode UX

Before the run, always print a compact summary (no frame counts).

```
Encode  video=01:24  fps=10  qr=V25  mode=base45  overhead=2.0x
```

During encoding, show a thin coloured bar with `percentage`, live
`fps`, and `ETA` — never `342/635` frame counters.

```
Encode  ━━━━━━━━━━━━━╸━━━━━━━━━━  61%   18.7 fps   ETA 00:21
Done    output.mp4  18.4 MB
```

`K`, `blocksize`, `workers`, payload size go into `verbose`.

---

## 4. Log-mode format (`key=value`, scheme B)

Examples:

```
[14:21:03] phase=probe status=start
[14:21:03] phase=probe status=done sample=2 detect=68% repeat=1.9 crop_reduction=64%
[14:21:07] phase=scan video=23.0% file=37.8% hit_window=68%
[14:21:10] phase=recover level=L1-clahe progress=31.0% file=57.1% hit_window=74%
[14:21:12] phase=save status=done output=report.pdf bytes=13002344
[14:20:00] phase=encode status=start duration=84.0s fps=10 qr=v25 mode=base45 overhead=2.0x
[14:20:06] phase=encode progress=18.0% speed=18.2fps eta=00:38
[14:20:19] phase=encode status=done output=report.mp4 size=18.4 MB
```

### Throttling

`scan_update` / `recover_update` / `encode_update`:

- emit immediately on the first call after phase start;
- otherwise only when |Δpct| ≥ 5 % OR ≥ 2 s have elapsed since the
  last emission;
- `verbose` loosens the interval to 0.5 s and attaches `map=…`.

`phase=... status=start|done` events always emit.  Values containing
space / `=` / `"` / tab are wrapped in double quotes and escaped.

---

## 5. Visual language

- **Thin bars** (Scan / Recover / Encode): Rich `Progress` with
  `BarColumn(complete_style=…)` and a pip/rich look.
  - Scan: cyan family
  - Recover: yellow family
  - Encode: green family
- **Wide block map** (File): bucket-based colour map with density
  tiers `░ (grey) → ▒ (blue) → ▓ (cyan) → ▓ (green) → █ (bright
  green)`.  Bucket size = `ceil(K / width)`; width auto-fits the
  terminal, clamped to `[24, 80]`.
- **Range strip**: idle `·` (grey), pending `▁` (yellow),
  current `▶` (bright cyan), scanned `█` (green).

---

## 6. Architecture

```
CLI  ── --output-mode / -v ──►  resolve_output_mode()
                                     │
                                     ▼
              ┌────────────── ProgressReporter ─────────────┐
              │                                             │
        RichReporter           LogReporter           QuietReporter
     (Live + thin bars +     (throttled key=value   (errors + final
      block map + range)      lines on stderr)        success line)

Encoder/Decoder business functions emit semantic events:
  probe_start / probe_done
  scan_start / scan_update / scan_done
  recover_start / recover_update / recover_done
  save_done
  encode_start / encode_update / encode_done
  info / warn / error / debug
```

`SlidingHitWindow`, `compute_block_map_cells`, `compute_range_strip_cells`
live in `qrstream/ui.py` and are used by both business code (to build
the event payload) and the Rich renderer (to draw the colourised
version).

---

## 7. Code-change inventory

- `src/qrstream/ui.py` *(new)* — enums, protocol, three reporters,
  renderers, throttle, resolver.
- `src/qrstream/cli.py` — new `--output-mode`; legacy `-v` hidden;
  build reporter once in `cmd_encode` / `cmd_decode` and pass down.
- `src/qrstream/encoder.py` — no more `tqdm`; `reporter` kwarg;
  encode summary + thin bar + `Done` line; verbose-only diagnostics
  via `reporter.debug`.
- `src/qrstream/decoder.py` — no more `tqdm` / `tqdm.write`; reporter
  drives Probe/Scan/Recover/Save.  `_stream_scan` replaces its `pbar`
  argument with an `on_frame(fidx, hit)` callback so both main scan
  and recovery can share the pipeline.  The old LT-decode bar in
  `_decode_into_decoder` is deleted outright.  All business error
  text still goes through `print(...)` for `capsys` compatibility.
- `pyproject.toml` — drop `tqdm>=4.60.0`, add `rich>=13.0.0`.
- `README.md` / `README-zh.md` — dependency list updated.

---

## 8. Test and CI adaptation

- `tests/test_ui_reporter.py` *(new)* — unit tests for
  `SlidingHitWindow`, block-map/range-strip renderers, `QuietReporter`,
  `LogReporter` throttling + value escaping, `resolve_output_mode`.
- `tests/test_optimizations.py` — keeps the existing `-v ⇒ verbose`
  assertion (still valid because `-v` is the hidden alias), and adds
  `--output-mode` coverage (`auto` default + every enum value parses).
- `tests/test_cli_overhead_floor.py` — no change needed; overhead-floor
  business errors still `print` to `stdout`.
- `tests/test_cli_detect_isolation.py` — no change; shared argparse
  group keeps its shape.
- `tests/test_real_recordings*.py`, `tests/test_e2e_encode_decode.py`,
  `tests/test_v074_bug_regression.py`,
  `tests/test_decoder_sandbox_integration.py`, `tests/test_roundtrip.py` —
  no change.  `extract_qr_from_video(..., verbose=False, workers=…)` and
  `decode_blocks_to_file(blocks, path, verbose=False)` keep their
  positional / keyword signatures.  The new `reporter=` parameter is
  optional; callers who don't pass one get a `QuietReporter` and keep
  their existing observable behaviour.
- `tests/test_recovery_wiring.py` — `_stream_scan` still exposes
  `worker_fn` as before; `extract_qr_from_video` does not reintroduce
  the forbidden `and sample_rate > 1` gate.
- `tests/test_decoder.py::test_decode_blocks_to_file_writes_uncompressed_output`
  — `bytes_dump_to_file` still accepts (and ignores) `show_progress`.
- `.github/workflows/*.yml` — no business text assertions; no `-v`
  intended as the qrstream flag (the existing `-v` in `pytest` calls
  is pytest's own verbose switch).  No workflow edits needed.

---

## 9. Known follow-up

The existing "double LT decode" (once inside `extract_qr_from_video`
for early termination, once inside `_decode_into_decoder` for the
final byte output) is left untouched by this refactor.  Now that the
LT-decode bar is gone the duplication is no longer user-visible, but
it remains a worthwhile structural cleanup — likely to ship as a
single `DecodeSession` returning `(blocks, decoder)` in a later PR.
