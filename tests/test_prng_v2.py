"""Tests for the prng_version=1 (SplitMix64) codec path and for
backward compatibility with prng_version=0 encoded blocks.

The PRNG schema is carried in V3 header flag bit 0x04:
  * cleared: legacy LCG with 5 warmup rounds (qrstream ≤ 0.7)
  * set:     SplitMix64 seed-mixer (qrstream ≥ 0.8, default)

These tests pin down the wire format of the flag bit and verify
that both codec paths round-trip correctly.
"""

from __future__ import annotations

import random
import struct

import pytest

from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder
from qrstream.lt_codec import PRNG, splitmix64_mix
from qrstream.protocol import V3_VERSION, pack_v3, unpack_v3


# ---------------------------------------------------------------------
# 1. SplitMix64 mixer — deterministic, in the expected output range,
# avoids the LCG fixed point at 0.
# ---------------------------------------------------------------------

def test_splitmix64_mix_is_deterministic():
    a = splitmix64_mix(1)
    b = splitmix64_mix(1)
    assert a == b


def test_splitmix64_mix_output_range():
    # PRNG_M = 2^31 - 1. Output must be in [1, PRNG_M - 1].
    for seed in [0, 1, 2, 42, 1827, 2**30, 2**40]:
        v = splitmix64_mix(seed)
        assert 1 <= v <= 2**31 - 2


def test_splitmix64_mix_decorrelates_small_seeds():
    """Avalanche sanity — consecutive small seeds must not produce
    near-identical mixed states. A Hamming-distance check catches
    accidental regressions to a linear mixer."""
    prev = splitmix64_mix(1)
    for s in range(2, 20):
        curr = splitmix64_mix(s)
        diff = bin(prev ^ curr).count('1')
        # For uncorrelated 31-bit values the expected Hamming distance
        # is ~15; 8 is a conservative floor that still catches the
        # "forgot to mix" regression (Hamming distance would be ~1-2).
        assert diff >= 8, (
            f"Seeds {s-1}→{s} mixed states differ by only {diff} bits; "
            f"mixer avalanche is broken."
        )
        prev = curr


# ---------------------------------------------------------------------
# 2. Flag bit 0x04 on the V3 wire format.
# ---------------------------------------------------------------------

def test_pack_v3_sets_flag_bit_for_prng_v1():
    data = b'\x00' * 32
    raw = pack_v3(
        filesize=128, blocksize=32, block_count=4,
        seed=7, block_seq=0, data=data,
        compressed=False, alphanumeric_qr=False,
        prng_version=1,
    )
    # Layout: version byte, then flags byte.
    assert raw[0] == V3_VERSION
    assert raw[1] & 0x04, "prng_version=1 must set flag bit 0x04"


def test_pack_v3_clears_flag_bit_for_prng_v0():
    data = b'\x00' * 32
    raw = pack_v3(
        filesize=128, blocksize=32, block_count=4,
        seed=7, block_seq=0, data=data,
        compressed=False, alphanumeric_qr=False,
        prng_version=0,
    )
    assert (raw[1] & 0x04) == 0, "prng_version=0 must clear flag bit 0x04"


def test_unpack_v3_reports_prng_version():
    for want in (0, 1):
        raw = pack_v3(
            filesize=128, blocksize=32, block_count=4,
            seed=7, block_seq=0, data=b'\x00' * 32,
            prng_version=want,
        )
        header, _ = unpack_v3(raw)
        assert header.prng_version == want


def test_pack_v3_rejects_unknown_prng_version():
    with pytest.raises(ValueError, match="prng_version"):
        pack_v3(
            filesize=128, blocksize=32, block_count=4,
            seed=7, block_seq=0, data=b'\x00' * 32,
            prng_version=2,
        )


# ---------------------------------------------------------------------
# 3. End-to-end roundtrip for both PRNG versions.
# ---------------------------------------------------------------------

def _payload(size: int) -> bytes:
    rng = random.Random(0xBADFACE)
    return bytes(rng.randrange(256) for _ in range(size))


@pytest.mark.parametrize("prng_version", [0, 1])
def test_encoder_decoder_roundtrip_for_each_prng_version(prng_version):
    K = 256
    blocksize = 64
    payload = _payload(K * blocksize)
    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=False,
        prng_version=prng_version,
    )
    dec = LTDecoder()
    for packed, _seed, _seq in enc.generate_blocks(int(K * 2.0)):
        try:
            done, _ = dec.decode_bytes(packed)
            if done:
                break
        except (ValueError, struct.error):
            pass
    assert dec.is_done()
    assert dec.prng_version == prng_version
    assert dec.bytes_dump() == payload


def test_mixing_prng_versions_in_same_session_raises():
    """Blocks with different prng_version flags in the same decode
    session are unsolvable; the decoder must reject the inconsistent
    second block loudly."""
    K = 64
    blocksize = 64
    payload = _payload(K * blocksize)

    enc_v1 = LTEncoder(payload, blocksize=blocksize, prng_version=1)
    enc_v0 = LTEncoder(payload, blocksize=blocksize, prng_version=0)

    first, _, _ = next(enc_v1.generate_blocks(1))
    second, _, _ = next(enc_v0.generate_blocks(1))

    dec = LTDecoder()
    dec.decode_bytes(first)
    with pytest.raises(ValueError, match="prng_version mismatch"):
        dec.decode_bytes(second)


# ---------------------------------------------------------------------
# 4. Backward compatibility — prng_version=0 must still decode
# even when the default encoder is prng_version=1.
# ---------------------------------------------------------------------

def test_legacy_prng_v0_blocks_still_decode():
    """Explicitly emit prng_version=0 blocks (as qrstream ≤ 0.7 would
    have) and verify the current decoder handles them via the
    LCG-warmup fallback path."""
    K = 128
    blocksize = 64
    payload = _payload(K * blocksize)
    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=True,
        prng_version=0,
    )
    dec = LTDecoder()
    for packed, _seed, _seq in enc.generate_blocks(int(K * 2.5)):
        try:
            done, _ = dec.decode_bytes(packed)
            if done:
                break
        except (ValueError, struct.error):
            pass
    assert dec.is_done()
    assert dec.prng_version == 0
    assert dec.bytes_dump() == payload


# ---------------------------------------------------------------------
# 5. PRNG class honours prng_version kwarg.
# ---------------------------------------------------------------------

def test_prng_rejects_unknown_version():
    with pytest.raises(ValueError, match="prng_version"):
        PRNG(K=32, prng_version=42)


def test_prng_versions_produce_different_src_blocks():
    """For the same seed, the two PRNG schemas must produce
    different (degree, src_blocks) tuples — otherwise the flag bit
    would be meaningless."""
    K = 256
    p0 = PRNG(K, prng_version=0)
    p1 = PRNG(K, prng_version=1)
    diffs = 0
    for seed in range(1, 30):
        _, d0, n0 = p0.get_src_blocks(seed=seed)
        _, d1, n1 = p1.get_src_blocks(seed=seed)
        if (d0, n0) != (d1, n1):
            diffs += 1
    assert diffs >= 20, (
        f"Only {diffs}/29 seeds differ between prng v0 and v1 — "
        f"the two schemas are suspiciously close."
    )
