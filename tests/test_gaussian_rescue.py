"""Tests for the Gauss-Jordan rescue fallback in the LT decoder.

When belief-propagation (peeling) stalls — the graph has no
degree-1 check node but the surviving equations still collectively
span the unknown source blocks — :meth:`LTDecoder.try_gaussian_rescue`
must finish decoding without needing additional encoded frames.

We construct stalls by using very low overhead (1.05×) so that peeling
cannot converge on its own, then verify that the GE pass rescues the
decode.
"""

from __future__ import annotations

import random
import struct

from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder, decode_blocks


def _payload(size: int) -> bytes:
    rng = random.Random(0xFEEDBEEF)
    return bytes(rng.randrange(256) for _ in range(size))


def test_gaussian_rescue_finishes_stalled_stream():
    """K=512 at overhead=1.05 causes peeling to stall (not enough
    degree-1 check nodes).  GE rescue must finish it."""
    K = 512
    blocksize = 128
    payload = _payload(K * blocksize)

    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=True,
    )
    dec = LTDecoder()
    for packed, _seed, _seq in enc.generate_blocks(int(K * 1.05)):
        try:
            dec.decode_bytes(packed)
        except (ValueError, struct.error):
            pass

    # Sanity: peeling really did stall below K.
    assert dec.num_recovered < K, (
        f"Expected peeling to stall at K={K} overhead=1.05, but it "
        f"already recovered {dec.num_recovered}/{K} via peeling alone. "
        f"Test premise no longer holds — pick a lower overhead."
    )

    rescued = dec.try_gaussian_rescue()
    assert rescued, (
        f"GE rescue failed: {dec.num_recovered}/{K} after rescue. "
        f"Either the surviving check graph genuinely doesn't span "
        f"the unknowns (insufficient information), or the GE pass "
        f"has a bug."
    )
    assert dec.is_done()
    assert dec.bytes_dump() == payload


def test_gaussian_rescue_is_noop_when_peeling_already_done():
    """When peeling converged, rescue must short-circuit to True
    and not touch the recovered blocks."""
    K = 64
    blocksize = 64
    payload = _payload(K * blocksize)
    enc = LTEncoder(payload, blocksize=blocksize)
    dec = LTDecoder()
    for packed, _seed, _seq in enc.generate_blocks(int(K * 2.0)):
        done, _ = dec.decode_bytes(packed)
        if done:
            break
    assert dec.is_done()

    snapshot = {idx: bytes(block) for idx, block in dec.block_graph.eliminated.items()}
    assert dec.try_gaussian_rescue()  # idempotent
    # Same block contents after the no-op call.
    assert {idx: bytes(block) for idx, block in dec.block_graph.eliminated.items()} == snapshot


def test_gaussian_rescue_gives_up_cleanly_without_enough_info():
    """Feed the decoder fewer blocks than the information-theoretic
    minimum (K/2). GE cannot recover what was never transmitted —
    the rescue must return False, and is_done must stay False."""
    K = 64
    blocksize = 64
    payload = _payload(K * blocksize)
    enc = LTEncoder(payload, blocksize=blocksize)
    dec = LTDecoder()
    for packed, _seed, _seq in enc.generate_blocks(K // 2):
        try:
            dec.decode_bytes(packed)
        except (ValueError, struct.error):
            pass
    assert not dec.is_done()
    rescued = dec.try_gaussian_rescue()
    assert rescued is False
    assert not dec.is_done()


def test_decode_blocks_auto_triggers_rescue_on_stalled_stream():
    """The top-level :func:`decode_blocks` must auto-invoke GE
    rescue when peeling finishes without converging — that's how
    users benefit from the fallback without any flag flip."""
    K = 512
    blocksize = 128
    payload = _payload(K * blocksize)
    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=True,
    )
    # Collect 1.05×K encoded blocks — peeling alone can't finish this
    # at such low overhead.
    blocks = [packed for packed, _, _ in enc.generate_blocks(int(K * 1.05))]

    recovered = decode_blocks(blocks, verbose=False)
    assert recovered == payload, (
        "decode_blocks must finish a stalled stream via the "
        "built-in GE rescue path."
    )
