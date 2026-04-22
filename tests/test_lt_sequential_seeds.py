"""Regression tests for LT decoding under the encoder's *sequential*
seed schedule (``seed = 1, 2, 3, … N``).

Why this file exists
--------------------
The existing :mod:`tests.test_lt_subset_robustness` asserts that LT
peeling converges for *random* subsets of the encoder's output pool.
Random sampling hides a real bug: because
:meth:`qrstream.encoder.LTEncoder.generate_blocks` feeds the PRNG with
*consecutive* seeds ``1 … N``, the real decode path consumes blocks in
exactly that deterministic order.  The LCG-based PRNG in
:mod:`qrstream.lt_codec` produces a degree sequence for consecutive
seeds whose peeling graph contains very few degree-1 check nodes at
the low end of ``N`` — LT belief-propagation then stalls at 3–12 % of
``K`` even though the information-theoretic bound is long satisfied.

Concrete regression symptom (qrstream ≤ 0.7):

    # K=1827, blocksize=938, overhead=1.5 → 2740 sequential seeds fed
    # LT recovers only 225/1827 source blocks and reports failure.

Default ``overhead=2.0`` happened to cross the threshold where enough
"easy" seeds appear so peeling could start, which is why nobody
noticed until a user ran ``qrs encode --overhead 1.5``.

What these tests guard
----------------------
1. **Sequential-seed decodability**: at a realistic production
   overhead (``1.5×``) the encoder's deterministic ``1..N`` seed
   stream must let LT peeling converge for *every* tested ``K`` —
   including the pathological K=1827 fixture.
2. **Cross-K coverage**: the bug depends on ``K`` (it showed up at
   K=1827 but not at K=1024 or K=2048), so the test sweeps a range.
3. **Exact-byte roundtrip**: on top of "all blocks recovered" we
   also check ``bytes_dump() == payload`` to catch any silent
   corruption introduced by a future refactor of the graph.

These tests are *cheap* — they bypass QR rendering and video I/O and
feed bytes straight from encoder to decoder.
"""

from __future__ import annotations

import random
import struct

import pytest

from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder


def _deterministic_payload(size: int) -> bytes:
    """Reproducible payload; seed is deliberately different from
    :mod:`tests.test_lt_subset_robustness` so test matrices can't
    alias each other's random state by accident."""
    rng = random.Random(0xC0FFEE42)
    return bytes(rng.randrange(256) for _ in range(size))


def _sequential_decode(payload: bytes, blocksize: int, overhead: float):
    """Mirror the production flow: generate blocks with seeds 1..N in
    order, feed them sequentially into a fresh :class:`LTDecoder`,
    return ``(decoder, num_blocks_fed)``.

    Matches exactly what :func:`qrstream.encoder.encode_to_video`
    plus :func:`qrstream.decoder.decode_blocks` do end-to-end when
    no frames are dropped by QR capture.
    """
    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=True,
    )
    K = enc.K
    num_blocks = int(K * overhead)
    enc._seq = 0  # deterministic block_seq field, match encoder.encode_to_video
    dec = LTDecoder()
    for packed, _seed, _seq in enc.generate_blocks(num_blocks):
        try:
            done, _ = dec.decode_bytes(packed)
            if done:
                break
        except (ValueError, struct.error):
            # A production corrupt-block path; should never trigger
            # on the pure-bytes pipeline, but be defensive so a
            # future protocol refactor can't make this test a
            # silent no-op.
            pass
    return dec, num_blocks


# ---------------------------------------------------------------------
# Guard 1: The user-reported regression at K=1827, overhead=1.5.
# ---------------------------------------------------------------------

def test_sequential_seeds_k1827_overhead_1p5_must_decode():
    """Exact reproduction of the failing configuration from the
    ``llm_km_report.pdf`` user report (qrstream 0.7 bug):

    * compressed payload size matching the real PDF (≈1.71 MB),
    * ``blocksize=938`` (what ``auto_blocksize`` picks for V25/M
      alphanumeric),
    * ``overhead=1.5`` (the user-provided CLI flag).

    Before the fix: LT recovered 225/1827 source blocks.
    After the fix: must recover all 1827 blocks and match bytes.
    """
    K = 1827
    blocksize = 938
    payload = _deterministic_payload(K * blocksize)

    dec, num_blocks = _sequential_decode(payload, blocksize, overhead=1.5)

    assert dec.is_done(), (
        f"LT failed to converge on sequential seeds 1..{num_blocks} "
        f"(recovered {dec.num_recovered}/{dec.K}); this is the "
        f"K=1827 overhead=1.5 regression."
    )
    assert dec.bytes_dump() == payload, (
        "LT reported success but output bytes don't match payload; "
        "silent peeling corruption."
    )


# ---------------------------------------------------------------------
# Guard 2: Broader K sweep at the production-default-ish overhead.
#
# 1.5× overhead is already above the Luby-soliton information-theoretic
# minimum (~1.05×). Any reasonable LT+PRNG combination should converge
# at 1.5× for every K in this range.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("K", [64, 128, 256, 328, 512, 1024, 1827, 2048])
def test_sequential_seeds_various_K_overhead_1p5(K):
    """At ``overhead=1.5`` the encoder's sequential seed stream must
    yield a decodable block set for a range of ``K`` values."""
    blocksize = 64
    payload = _deterministic_payload(K * blocksize)

    dec, num_blocks = _sequential_decode(payload, blocksize, overhead=1.5)

    assert dec.is_done(), (
        f"Sequential-seed decode failed at K={K}, overhead=1.5 "
        f"(fed {num_blocks} blocks, recovered {dec.num_recovered}/{dec.K}). "
        f"This is the class of regression that broke qrs encode "
        f"--overhead 1.5 in qrstream 0.7."
    )


# ---------------------------------------------------------------------
# Guard 3: Minimum-acceptable overhead ceiling for sequential seeds.
#
# We want to *allow* future encoder/PRNG refactors to bring the
# required overhead down, but we must not silently *raise* it above
# what the CLI advertises to users. qrs CLI exposes overhead as low
# as 1.2, so any configuration that needs more than 1.5 to decode
# a small K is a regression.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("K", [128, 328, 1827])
def test_sequential_seeds_overhead_ceiling_must_not_regress(K):
    """The required overhead for sequential-seed convergence on a
    deterministic payload must not exceed 1.5×. If this test starts
    failing, the encoder's seed/PRNG pipeline has regressed and the
    decoder can no longer recover from the CLI's advertised low-
    overhead settings."""
    blocksize = 64
    payload = _deterministic_payload(K * blocksize)

    # We scan the overhead up to the ceiling. If LT can't finish at
    # 1.5 even though it succeeds at 2.0, we've regressed back to
    # the qrstream 0.7 state.
    dec_15, _ = _sequential_decode(payload, blocksize, overhead=1.5)
    assert dec_15.is_done(), (
        f"Sequential-seed convergence at K={K} now requires more than "
        f"1.5× overhead; this is a regression of the v0.7 "
        f"sequential-seed LT bug."
    )
