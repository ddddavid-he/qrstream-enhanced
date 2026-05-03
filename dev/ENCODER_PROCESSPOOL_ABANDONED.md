# Encoder ProcessPool Experiment — Abandoned

Branch: `feature/encoder-processpool`
Status: **do not continue / do not merge into `dev`**
Date: 2026-05-03

## Question

The encoder previously auto-selected `workers = min(os.cpu_count(), 4)` for QR generation. We investigated whether this should become:

- GIL-enabled CPython: use `ProcessPoolExecutor` to bypass the GIL.
- Free-threaded CPython: use native `ThreadPoolExecutor`.

`segno` remains the QR generation backend throughout this experiment.

## Findings

### Isolated QR generation

Process pools do speed up the isolated pure-Python QR generation hot path:

| config | wall time | speedup |
|---|---:|---:|
| ThreadPool w=1 | 2.868s | 1.00x |
| ThreadPool w=4 | 2.377s | 1.21x |
| ProcessPool w=4 | 1.026s | 2.80x |
| ProcessPool w=6 | 0.980s | 2.93x |
| ProcessPool w=8 | 0.989s | 2.90x |
| ProcessPool w=14 | 1.052s | 2.73x |

### Full `encode_to_video()` pipeline

In the full encoder pipeline, the gain disappears. The total wall time is dominated by `VideoWriter.write()` / muxing / codec output rather than by QR generation alone.

Podman Linux smoke benchmark after the process-pool experiment:

| size | workers=1 | workers=4 | workers=6 |
|---|---:|---:|---:|
| 100KB | 2.36s | 2.55s | 2.45s |
| 500KB | 10.43s | 10.46s | 10.38s |
| 1MB | 20.55s | 20.96s | 20.87s |

`mjpeg` showed the same pattern: no meaningful full-pipeline acceleration.

## Conclusion

The ProcessPool design is not worth continuing for the default encoder path:

1. Isolated QR generation benefits, but full encode does not.
2. The added complexity (`spawn`, IPC, grayscale IPC optimization, old v0.7.4 fork-safety regression handling) is not justified by end-to-end results.
3. The safer product decision is to leave the default fixed-mask encoder path single-worker unless a future profiling effort removes or parallelizes the video writer bottleneck.

This branch is preserved only as an experiment record and should not be merged into `dev`.

## Verification performed

Using podman with Python 3.13:

- `pytest tests/test_qr_generate.py tests/test_v074_bug_regression.py -v`
  - `71 passed`
- `pytest tests/ -v -m "not slow and not e2e"`
  - `271 passed, 1 skipped, 37 deselected`
