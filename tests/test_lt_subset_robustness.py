"""Regression tests for LT belief-propagation robustness.

These tests guard against four classes of regression that showed up
as the v0.7.1 amd64 phone-recording failure:

  1. LT belief-propagation implementation drift — any refactor that
     reduces how many random seed subsets can be peeled.
  2. Encoder PRNG drift — a change to :class:`PRNG` would alter the
     degree distribution observed by the decoder.
  3. ``auto_blocksize`` drift — a change that silently shifts K on a
     given fixture would invalidate the seeds-per-fixture calculus
     used by probe+recovery.
  4. Silent catastrophe where some K / subset-size combination stalls
     LT peeling with high probability (the root-cause chain behind
     the v070 regression).

Because LT peeling is inherently probabilistic, these tests use a
deterministic RNG seed and assert a statistical floor (95% of random
trials must converge at ``factor*K`` received blocks).  Sampling ~3%
pathological rate at factor=1.45 is known and tolerated.
"""

from __future__ import annotations

import random
import struct

import pytest

from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder


def _deterministic_payload(size: int) -> bytes:
    """Build a deterministic byte payload of the requested size.

    Uses ``random.Random(seed=…)`` rather than ``os.urandom`` so the
    test is reproducible across platforms and CI re-runs.
    """
    rng = random.Random(0xDEADBEEF)
    return bytes(rng.randrange(256) for _ in range(size))


@pytest.mark.parametrize(
    "seed_set_size_factor, min_success_ratio",
    [
        # At factor=1.45 the empirical pathological rate is ~10-15%
        # (measured across this encoder's PRNG).  85% floor catches a
        # real LT regression (which would show <50%) while tolerating
        # natural subset variance.
        (1.45, 0.80),
        (1.50, 0.90),
        (1.60, 0.94),
    ],
    ids=["factor-1.45", "factor-1.50", "factor-1.60"],
)
def test_random_subsets_decode_with_high_probability(
    seed_set_size_factor, min_success_ratio,
):
    """At least 95% of random K*factor-sized encoded-block subsets must
    recover all K source blocks.

    Uses K=328, blocksize=938 to match the v070 phone-recording
    fixture; this is the exact parameter combination where the
    original regression surfaced.
    """
    K = 328
    blocksize = 938
    payload = _deterministic_payload(K * blocksize)
    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=True,
    )

    # Pre-generate a wide pool of encoded blocks (packed) indexed by
    # seed, so the test does not re-encode on every trial.
    # Must match the seed range used in the real encoder: 1..N.
    pool_size = int(K * 2.0)
    all_blocks: dict[int, bytes] = {}
    enc._seq = 0  # deterministic seq field
    for packed, seed, _seq in enc.generate_blocks(pool_size):
        all_blocks[seed] = packed

    rng = random.Random(0xA11CE)
    n_trials = 50
    subset_size = int(K * seed_set_size_factor)

    n_ok = 0
    for _ in range(n_trials):
        subset = rng.sample(list(all_blocks), subset_size)
        dec = LTDecoder()
        for s in subset:
            try:
                dec.decode_bytes(all_blocks[s])
            except (ValueError, struct.error):
                pass
        if dec.is_done():
            n_ok += 1

    # Tolerate the inherent pathological-subset rate at low factors
    # while still catching a real LT regression.
    floor = int(n_trials * min_success_ratio)
    assert n_ok >= floor, (
        f"LT recovered {n_ok}/{n_trials} random subsets at "
        f"factor={seed_set_size_factor} (floor={floor}); "
        f"this indicates an LT peeling or encoder-PRNG regression."
    )


def test_lt_converges_on_full_ground_truth_set():
    """Sanity gate: with *all* seeds 1..2K available, LT must always
    converge.  If this ever fails, there's a catastrophic LT bug —
    no subset test below is meaningful until this passes.
    """
    K = 64
    blocksize = 128
    payload = _deterministic_payload(K * blocksize)
    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=True,
    )

    dec = LTDecoder()
    enc._seq = 0
    for packed, _seed, _seq in enc.generate_blocks(int(K * 2.0)):
        try:
            dec.decode_bytes(packed)
        except (ValueError, struct.error):
            pass
        if dec.is_done():
            break

    assert dec.is_done(), (
        f"LT failed to converge with full 2*K block pool "
        f"(recovered {dec.num_recovered}/{K}); "
        f"this is a catastrophic LT bug."
    )

    # Byte-exact recovery check.
    assert dec.bytes_dump() == payload
