#!/usr/bin/env python3
"""Informational probe: does the decoder still decode correctly if
``multiprocessing`` worker processes are spawned via ``fork`` instead
of the currently-enforced ``spawn`` start method?

This script is **not** part of the test gate.  Production decoders
continue to force ``spawn`` on every platform (see
:data:`qrstream.decoder._MP_SPAWN_CTX`).  The probe only exists so
CI can observe, on every push, whether the fork path *would* still
work — a useful forward-looking signal when OpenCV / Python /
dependency upgrades happen, without ever letting a fork-specific
regression block a release.

Why we keep ``spawn`` as the default
------------------------------------

``fork()`` on a multi-threaded parent is unsafe because a child
process inherits the parent's full memory image — including the
*state* of every mutex, malloc arena lock, OpenCV IPP/BLAS thread
pool lock, libc stdio lock — but **only the calling thread**.  Any
lock held by a *different* thread at the moment of ``fork()`` lives
on in the child as "held by a thread that no longer exists"; the
next code path that touches it deadlocks forever.

Our decoder runs a background reader thread (``_prefetch_iter``)
concurrently with ``ProcessPoolExecutor`` worker dispatch, so at
the moment of fork the Python interpreter has > 1 live thread.
Python 3.12 started emitting a ``DeprecationWarning`` for exactly
this pattern:

    DeprecationWarning: This process (pid=...) is multi-threaded,
    use of fork() may lead to deadlocks in the child.

See PEP 711 and https://github.com/python/cpython/issues/84559 for
the upstream plan.  Rough timeline: 3.12/3.13 warn, 3.14 changes
the default multiprocessing start method on Linux away from
``fork``, and some later release promotes the warning into an
error.  Staying on ``spawn`` makes the decoder forward-compatible
with that trajectory at the cost of ~1–3 s of per-scan cold start.

What this probe actually checks
-------------------------------

1. Monkey-patch :data:`qrstream.decoder._MP_SPAWN_CTX` to
   ``multiprocessing.get_context("fork")`` *before* calling any
   decoder entry point.
2. Run the full ``extract_qr_from_video`` → ``decode_blocks_to_file``
   pipeline on ``tests/fixtures/testcase-v070.mp4``.
3. Compare the decoded SHA-256 against the committed fixture oracle
   (``testcase-v070.input.bin``).

Exit codes
----------
    0  — fork mode decoded byte-exactly (informational).
    1  — fork mode decoded wrong / incomplete.
    2  — infrastructure error (fixture missing, worker crashed, etc.)

CI wires this in as a non-gating job so a failing run surfaces in
the Actions UI without blocking ``release.yml`` / ``publish.yml``.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import platform
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    video = fixtures / "testcase-v070.mp4"
    input_bin = fixtures / "testcase-v070.input.bin"
    expected_sha = _sha256_file(input_bin)

    if not video.exists() or not input_bin.exists():
        print(f"FAIL: fixture missing under {fixtures}", file=sys.stderr)
        return 2

    # Import *after* confirming fixtures exist so the import cost is
    # not paid on setup errors.
    import qrstream.decoder as dec_mod

    original_ctx = dec_mod._MP_SPAWN_CTX
    fork_ctx = multiprocessing.get_context("fork")
    dec_mod._MP_SPAWN_CTX = fork_ctx

    print(
        f"[probe-fork-safety] arch={platform.machine()} "
        f"python={platform.python_version()} "
        f"ctx={dec_mod._MP_SPAWN_CTX.get_start_method()} "
        f"(patched from {original_ctx.get_start_method()})"
    )
    print(f"[probe-fork-safety] video      = {video}")
    print(f"[probe-fork-safety] expected   = {expected_sha}")

    # Capture any DeprecationWarning the fork path emits.  Python
    # 3.12+ warns on fork() from a multi-threaded parent; this probe
    # lets CI see the running count per release without failing.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)

        try:
            blocks = dec_mod.extract_qr_from_video(
                str(video), sample_rate=0, verbose=False, workers=None,
            )
        except Exception as e:  # pragma: no cover - diagnostic only
            print(
                f"FAIL: extract_qr_from_video raised "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return 2

        if not blocks:
            print("FAIL: decoder returned no blocks", file=sys.stderr)
            return 1

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            out_path = Path(tmp.name)
        try:
            written = dec_mod.decode_blocks_to_file(
                blocks, str(out_path), verbose=False,
            )
            if written is None:
                print(
                    "FAIL: decode_blocks_to_file returned None (LT stuck)",
                    file=sys.stderr,
                )
                return 1
            actual_sha = _sha256_file(out_path)
        finally:
            if out_path.exists():
                out_path.unlink()

    print(f"[probe-fork-safety] uniq blocks = {len(blocks)}")
    print(f"[probe-fork-safety] written    = {written}")
    print(f"[probe-fork-safety] actual     = {actual_sha}")

    # Summarise warnings so CI logs carry a forward-looking signal.
    by_category: Counter[str] = Counter()
    for w in captured:
        by_category[w.category.__name__] += 1
    if by_category:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(by_category.items()))
        print(f"[probe-fork-safety] warnings   = {parts}")
        # Surface the first DeprecationWarning verbatim — usually the
        # "fork() in multi-threaded parent" one — so future readers
        # know which category is accumulating.
        for w in captured:
            if issubclass(w.category, DeprecationWarning):
                print(
                    f"[probe-fork-safety] first-dep  = "
                    f"{w.filename}:{w.lineno}: {w.message}"
                )
                break
    else:
        print("[probe-fork-safety] warnings   = none")

    if actual_sha != expected_sha:
        print("FAIL: sha256 mismatch", file=sys.stderr)
        return 1

    print("OK: fork mode decoded byte-exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
