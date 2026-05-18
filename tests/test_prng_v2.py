"""Tests for the SplitMix64 PRNG codec path (qrstream 0.10+).

As of v0.10, prng_version=0 (legacy LCG) has been removed.  Only
prng_version=1 (SplitMix64) is supported.  The V3 header flag bit 0x04
is now always set by pack_v3(), and unpack_v3() raises ValueError if
the bit is cleared (rejecting legacy prng_version=0 blocks).

These tests pin down the wire format of the flag bit and verify the
single supported codec path round-trips correctly.
"""

from __future__ import annotations

import random
import struct
import zlib

import pytest

from qrstream.encoder import LTEncoder
from qrstream.decoder import LTDecoder
from qrstream.lt_codec import splitmix64_mix
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

def test_pack_v3_sets_flag_bit():
    """pack_v3 always sets the 0x04 flag bit (SplitMix64)."""
    data = b'\x00' * 32
    raw = pack_v3(
        filesize=128, blocksize=32, block_count=4,
        seed=7, block_seq=0, data=data,
        compressed=False, alphanumeric_qr=False,
    )
    # Layout: version byte, then flags byte.
    assert raw[0] == V3_VERSION
    assert raw[1] & 0x04, "pack_v3 must always set flag bit 0x04"


def test_unpack_v3_reports_prng_version():
    """unpack_v3 always reports prng_version=1."""
    raw = pack_v3(
        filesize=128, blocksize=32, block_count=4,
        seed=7, block_seq=0, data=b'\x00' * 32,
    )
    header, _ = unpack_v3(raw)
    assert header.prng_version == 1


def test_unpack_v3_rejects_legacy_prng_v0_blocks():
    """Manually construct a V3 block with the 0x04 flag cleared and
    verify that unpack_v3 raises ValueError."""
    blocksize = 32
    data = b'\x00' * blocksize
    # Build a raw V3 block with flag bit 0x04 cleared (legacy prng_version=0).
    flags = 0x00  # no compression, no high-density, NO prng bit
    header = struct.pack(
        '>BBQHIIHH',
        V3_VERSION,
        flags,
        128,        # filesize
        blocksize,  # blocksize
        4,          # block_count
        7,          # seed
        0,          # block_seq
        0,          # reserved
    )
    payload = header + data
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    raw = payload + struct.pack('>I', crc)

    with pytest.raises(ValueError, match="prng_version=0 was removed"):
        unpack_v3(raw)


# ---------------------------------------------------------------------
# 3. End-to-end roundtrip for the SplitMix64 PRNG path.
# ---------------------------------------------------------------------

def _payload(size: int) -> bytes:
    rng = random.Random(0xBADFACE)
    return bytes(rng.randrange(256) for _ in range(size))


def test_encoder_decoder_roundtrip():
    """Round-trip using the default (and only) SplitMix64 PRNG path."""
    K = 256
    blocksize = 64
    payload = _payload(K * blocksize)
    enc = LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=False,
        alphanumeric_qr=False,
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
    assert dec.prng_version == 1
    assert dec.bytes_dump() == payload
