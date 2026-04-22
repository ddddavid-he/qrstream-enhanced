# Test Fixtures — Real phone-recorded QR videos

These fixtures exercise the full encode → display → phone
recording → decode pipeline, i.e. exactly the scenario qrstream is
designed for.  The unit test suite (`tests/`) mocks out most of
the pipeline, so these end-to-end recordings are a separate,
slower smoke layer.

## Directory layout

Fixtures are split by the protocol path they exercise:

    tests/fixtures/
      real-phone-v3/   # captures decoded via the legacy
                       # prng_version=0 LT path (qrstream ≤ 0.7)
      real-phone-v4/   # captures decoded via the prng_version=1
                       # LT path (qrstream ≥ 0.8 default,
                       # SplitMix64 mixer + GE rescue)

Each sub-dir contains pairs of ``<case>.input.bin`` (the raw
payload that was fed into the encoder) and ``<case>.mp4`` (the
phone recording, re-encoded to a git-friendly size).  The case
stem encodes the qrstream CLI version used to produce the
original encoded video (e.g. ``v061`` means qrstream 0.6.1), so
you can tell from a filename alone which encoder path produced a
given fixture.

## Files

### real-phone-v3 (legacy LCG-warmup PRNG path)

| File | Input SHA-256 (first 8) | Input size | Encoded with | Recorded | Compressed |
|---|---|---|---|---|---|
| `v061.*` | `4a440b6d…` | 30 720 B | v0.6.0 defaults (V25, base45, ec=M, border=10%, lead-in=1.5 s) | iPhone @ 60 fps, 720×720 | x264 CRF 36, 30 fps |
| `v070.*` | `2a20b62e…` | 307 200 B | v0.7.0 defaults (V25, base45, ec=M, border=10%, lead-in=1.5 s) | iPhone @ 30 fps, 1054×1168 | x264 CRF 36, 10 fps |

`v070` is a known-marginal capture. It decodes on most hardware /
OpenCV builds but occasionally misses ~0.5 % of frames on bespoke
ffmpeg builds, which can push it below the LT convergence
threshold. The test harness marks it
`@pytest.mark.xfail(strict=False)` so a regression shows up in
the workflow summary without blocking a release — see
`tests/test_real_recordings.py` for the rationale.

### real-phone-v4 (SplitMix64 PRNG path + GE rescue)

| File | Input SHA-256 (first 8) | Input size | Encoded with | Recorded | Compressed |
|---|---|---|---|---|---|
| `v073-10kB.*` | `897d28b6…` | 10 240 B | v0.7.3 defaults + `--overhead 1.5 --fps 10 --lead-in-seconds 1.0` | iPhone @ 60 fps, 1036×1036 (HEVC) | libx265 CRF 32, 720×720, 15 fps |
| `v073-100kB.*` | `6fbf396b…` | 102 400 B | same | iPhone @ 60 fps, 1080×1080 (HEVC) | libx265 CRF 32, 720×720, 15 fps |
| `v073-300kB.*` | `115e32de…` | 307 200 B | same | iPhone @ 60 fps, ~1080×1080 (HEVC) | libx265 CRF 36, 720×720, 12 fps |

All three v4 cases are **gating** — a regression blocks the
real-world workflow.

## How the fixtures were generated

### v3 legacy cases

1. Generate a random input file of the documented size with
   `os.urandom`.
2. Encode it to a QR video using the CLI:

       qrstream encode v070.input.bin \
           -o v070.source.mp4 \
           --qr-version 25 --qr-mode alphanumeric \
           --ec-level 1 --overhead 2.0 --fps 10 \
           --border 10 --lead-in-seconds 1.5

3. Play the resulting `.mp4` full-screen on a monitor.
4. Record the screen with a phone camera.
5. Re-encode the phone recording with ffmpeg at CRF 36:

       ffmpeg -i phone-recording.mov \
           -c:v libx264 -crf 36 -preset slow -r 10 -an \
           v070.mp4

### v4 cases (qrstream 0.7.3+ default path)

1. Generate a random input with a small human-readable header
   (see `dev/make_test_fixtures.py` history) so the file is
   auditable in a hex viewer.
2. Encode:

       qrs encode v073-300kB.input.bin \
           -o source.mp4 \
           --overhead 1.5 --fps 10 \
           --lead-in-seconds 1.0

   `--overhead 1.5` is the minimum the CLI accepts (hard floor is
   1.20, recommended ≥1.50).  Fewer frames → shorter recording.
3. Play the ``.mp4`` full-screen, record the screen with a phone.
4. Re-encode with **HEVC / 720×720 / 12-15 fps / CRF 32-36**:

       # 10 kB, 100 kB (short, needs slightly better quality)
       ffmpeg -i phone.mov \
           -vf "scale=720:720:flags=lanczos,fps=15" \
           -c:v libx265 -crf 32 -preset slow -tag:v hvc1 -an \
           v073-100kB.mp4

       # 300 kB (longer, tolerates higher compression)
       ffmpeg -i phone.mov \
           -vf "scale=720:720:flags=lanczos,fps=12" \
           -c:v libx265 -crf 36 -preset slow -tag:v hvc1 -an \
           v073-300kB.mp4

   These parameters were chosen empirically: the next step below
   each CRF (CRF 34 / 38) starts failing to decode on the
   marginal 300 kB case.

## How the tests use them

See `tests/test_real_recordings.py`.  The tests are marked
`@pytest.mark.slow` and are skipped by default so the normal
`pytest tests/` run stays fast.

These slow tests run in a **dedicated** GitHub Actions workflow
(`.github/workflows/real-world-tests.yml`) rather than the per-
Python-version unit matrix — they exercise OpenCV / WeChatQRCode
rather than any Python-version-specific logic, so one run per
architecture on Python 3.13 is sufficient coverage.

Run locally with either::

    uv run pytest -m slow -v
    uv run pytest tests/test_real_recordings.py -v
