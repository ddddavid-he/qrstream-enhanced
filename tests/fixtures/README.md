# Test Fixtures — Real phone-recorded QR videos

These fixtures exercise the full encode → display → phone recording →
decode pipeline, i.e. exactly the scenario qrstream is designed for.
The unit test suite (`tests/`) mocks out most of the pipeline, so
these end-to-end recordings are a separate, slower smoke layer.

## Files

| File | Input SHA-256 (first 8) | Input size | Encoded with | Recorded | Compressed |
|---|---|---|---|---|---|
| `testcase-v061.input.bin` + `testcase-v061.mp4` | `4a440b6d…` | 30 720 B | v0.6.0 defaults (V25, base45, ec=M, border=10%, lead-in=1.5 s) | iPhone @ 60 fps, 720×720 | x264 CRF 36, 30 fps |
| `testcase-v070.input.bin` + `testcase-v070.mp4` | `2a20b62e…` | 307 200 B | v0.7.0 defaults (V25, base45, ec=M, border=10%, lead-in=1.5 s) | iPhone @ 30 fps, 1054×1168 | x264 CRF 36, 10 fps |

Full SHA-256 values are embedded in `tests/test_real_recordings.py` so
the tests are self-contained; the `*.input.bin` files exist so the
input is auditable and the generator can be reproduced.

## How the fixtures were generated

1. Generate a random input file of the documented size with
   `os.urandom`.
2. Encode it to a QR video using the CLI with the documented
   parameters, e.g.

       qrstream encode testcase-v070.input.bin \
           -o testcase-v070.source.mp4 \
           --qr-version 25 --qr-mode alphanumeric \
           --ec-level 1 --overhead 2.0 --fps 10 \
           --border 10 --lead-in-seconds 1.5

3. Play the resulting `.mp4` full-screen on a monitor.
4. Record the screen with a phone camera.
5. Re-encode the phone recording with ffmpeg at CRF 36 to shrink it
   to a git-friendly size:

       ffmpeg -i phone-recording.mov \
           -c:v libx264 -crf 36 -preset slow -r 10 -an \
           testcase-v070.mp4

   CRF 36 is two steps below the empirical breaking point (CRF 40
   for v070, CRF 40 for v061), giving a safety margin against future
   opencv / WeChatQRCode changes without blowing up the repo.

## How the tests use them

See `tests/test_real_recordings.py`. The tests are marked
`@pytest.mark.slow` and are skipped by default so the normal
`pytest tests/` run stays fast. The CI workflow runs them explicitly
in a single job (Python 3.13 only) as a post-unit smoke layer.

Run locally with either of:

    uv run pytest -m slow -v
    uv run pytest tests/test_real_recordings.py -v
