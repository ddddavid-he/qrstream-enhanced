# Test Fixtures — Real phone-recorded QR videos

These fixtures exercise the full encode → display → phone
recording → decode pipeline, i.e. exactly the scenario qrstream is
designed for.  The unit test suite (`tests/`) mocks out most of
the pipeline, so these end-to-end recordings are a separate,
slower smoke layer.

## Directory layout

Fixtures are split by the protocol path they exercise:

    tests/fixtures/
      real-phone-v073/     # captures decoded via the prng_version=1
                           # LT path (qrstream ≥ 0.8 default,
                           # SplitMix64 mixer + GE rescue)
      real-phone-v092/     # v0.9.2 LT and RaptorQ phone captures
                           # of a deterministic π payload
      zxing-1132/         # sanitized frame reproducing zxing-cpp 3.1.0 regression

Historical sub-dirs contain pairs of ``<case>.input.bin`` (the
raw payload that was fed into the encoder) and ``<case>.mp4`` (the
phone recording, re-encoded to a git-friendly size).  The v0.9.2
π-payload fixtures commit only ``.mp4`` files; the source is
identified by size plus SHA-256 so a 1 MB input file does not need
to be duplicated per codec.  Case stems encode the qrstream
CLI version used to produce the original encoded video (e.g.
``v073`` means qrstream 0.7.3), so you can tell from a filename
alone which encoder path produced a given fixture.

## Files

### zxing-1132 (QR detection regression)

`zxing-1132/frame142.png` is the byte-identical, sanitized reproducer
submitted by this project's author in
[zxing-cpp #1132](https://github.com/zxing-cpp/zxing-cpp/issues/1132).
It is derived from frame 142 of a real phone recording, cropped and
processed with CLAHE; unrelated background and a hand outside the QR
region were replaced with gray before publication.

- [Original PNG](https://github.com/user-attachments/assets/447b24e2-3de4-4496-a230-8e91d0c4930e): 840 × 842 RGB.
- PNG SHA-256: `e05152676b85972c1e93eeb475f4e018dbfad66dc4411f69c9ad1aed6ada9672`.
- Expected decoded payload: 3378 bytes, SHA-256 `bc235bbcc0907917fce0e1cbcb10a7a3c1134ebf5d7e0e5cee504a830f707316`.

`tests/test_zxing_regression.py` checks both the default ZXing reader
and QRStream's production reader in the normal unit suite. Both checks
fail with zxing-cpp 3.1.0 and pass with 3.1.1. The four complete
phone-recording fixtures also pass with 3.1.1 on macOS arm64 / Python
3.14.5 (verified 2026-10-02).

### real-phone-v073 (SplitMix64 PRNG path + GE rescue)

| File | Input SHA-256 (first 8) | Input size | Encoded with | Recorded | Compressed |
|---|---|---|---|---|---|
| `v073-100kB.*` | `6fbf396b…` | 102 400 B | v0.7.3 defaults + `--overhead 1.5 --fps 10 --lead-in-seconds 1.0` | iPhone @ 60 fps, 1080×1080 (HEVC) | libx265 CRF 32, 720×720, 15 fps |
| `v073-300kB.*` | `115e32de…` | 307 200 B | same | iPhone @ 60 fps, ~1080×1080 (HEVC) | libx265 CRF 36, 720×720, 12 fps |

Both v4 cases are **gating** — a regression blocks the
real-world workflow.

### real-phone-v092 (v0.9.2 LT + RaptorQ paths)

| File | Output SHA-256 (first 8) | Decoded size | Encoded with | Recorded | Compressed |
|---|---|---|---|---|---|
| `v092-lt-pi-1MB.mp4` | `7806ee47…` | 1 000 000 B | v0.9.2 LT, π decimal digits, `--no-compress --qr-version 40 --fps 25 --overhead 1.2` | iPhone 15 Pro | HEVC, 640×616, CRF 28 |
| `v092-raptorq-pi-1MB.mp4` | `7806ee47…` | 1 000 000 B | v0.9.2 RaptorQ, π decimal digits, `--no-compress --qr-version 40 --fps 25 --overhead 1.1` | iPhone 15 Pro | HEVC, 840×1002, CRF 28 (crop=1040:1240:15:270 from 1080×1920 portrait) |

The π source is deterministic and not committed.  The tests verify
its decoded byte stream by size plus SHA-256.

## How the fixtures were generated

### v4 cases (qrstream 0.7.3+ default path)

1. Generate a random input with a small human-readable header
   (see the historical `make_test_fixtures.py` helper) so the file is
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

### v0.9.2 π fixtures

1. Generate the first 1,000,000 digits after π's decimal point:

       uv run --with mpmath python -c \
         "from mpmath import mp; mp.dps=2000000; open('/tmp/pi_1mb.txt','w').write(str(mp.pi)[2:1000002])"

2. Encode with v0.9.2 codecs and no zlib compression:

       qrstream encode /tmp/pi_1mb.txt -o raptorq.source.mp4 \
           --qr-version 40 --fps 25 --overhead 1.1 \
           --fountain-codec raptorq --no-compress

       qrstream encode /tmp/pi_1mb.txt -o lt.source.mp4 \
           --qr-version 40 --fps 25 --overhead 1.2 \
           --fountain-codec lt --no-compress

3. Record the screen with a phone camera, then re-encode / resize
   empirically to the smallest files that still decode reliably:

       # RaptorQ — crop QR region from portrait recording, then scale to 840px wide.
       # The original CRF 30 / 700px encoding only yielded ~444/451 decodable blocks
       # on x86_64 (raptorq 2.0.0 has a convergence bug at near-minimum overhead);
       # 840px wide at CRF 28 consistently yields ≥452 blocks on all platforms.
       ffmpeg -i phone.mov \
           -vf "crop=1040:1240:15:270,scale=840:-2:flags=lanczos" \
           -c:v libx265 -crf 28 -preset medium -tag:v hvc1 -an \
           v092-raptorq-pi-1MB.mp4

       # LT
       ffmpeg -i phone.mov -vf "scale=640:-2:flags=lanczos" \
           -c:v libx265 -crf 28 -preset medium -tag:v hvc1 -an \
           v092-lt-pi-1MB.mp4

## How the tests use them

See `tests/test_real_recordings.py`.  The tests are marked
`@pytest.mark.slow` and are skipped by default so the normal
`pytest tests/` run stays fast.

These slow tests run in a **dedicated** GitHub Actions workflow
(`.github/workflows/real-world-tests.yml`) rather than the per-
Python-version unit matrix — they exercise native video/QR detection
(OpenCV frame handling plus zxing-cpp) rather than any Python-version-
specific logic, so one run per architecture on Python 3.13 is
sufficient coverage.

Run locally with either::

    uv run pytest -m slow -v
    uv run pytest tests/test_real_recordings.py -v
