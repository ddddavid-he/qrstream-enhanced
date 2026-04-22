"""Layered end-to-end assertions against phone-recording fixtures.

Where :mod:`test_real_recordings` asserts a single "byte-exact
SHA-256" oracle (great for CI pass/fail, terrible for post-mortems),
this module tears the pipeline apart into four layers and asserts
each one independently.  When a regression hits in the future, the
failing layer tells you *where* without having to bisect into the
decoder internals.

Layers:
  L1 — video → unique-blocks extraction.
  L2 — each decoded block matches the ground-truth (seed, data).
  L3 — LT belief-propagation converges on the observed seed set.
  L4 — final byte stream equals the input fixture.

All tests are marked ``slow`` (they take several seconds each) and
opt-in via ``pytest -m slow`` so the default ``pytest tests/`` run
remains snappy.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from qrstream.decoder import LTDecoder, extract_qr_from_video
from qrstream.encoder import LTEncoder
from qrstream.protocol import auto_blocksize, unpack


_FIXTURES_DIR = Path(__file__).parent / "fixtures"


class _FixtureSpec:
    """Bundle fixture metadata used by the layered tests."""

    def __init__(
        self,
        video: str,
        input_bin: str,
        expected_sha: str,
        min_uniq_blocks: int,
        qr_version: int,
        ec_level: int,
    ):
        self.video = _FIXTURES_DIR / video
        self.input_bin = _FIXTURES_DIR / input_bin
        self.expected_sha = expected_sha
        self.min_uniq_blocks = min_uniq_blocks
        self.qr_version = qr_version
        self.ec_level = ec_level


# Keep this list in sync with tests/test_real_recordings.py::_CASES.
# ``min_uniq_blocks`` is a conservative lower bound on what
# extract_qr_from_video should yield *after* targeted recovery; it
# must be greater than K (number of source blocks in the fixture)
# and leave some margin for normal run-to-run variance.
_FIXTURES = {
    "v070": _FixtureSpec(
        video="testcase-v070.mp4",
        input_bin="testcase-v070.input.bin",
        expected_sha=(
            "2a20b62e35bf4b3a7f5fa4854397eeafea99d1efb8db38737cda4df55a4d5b8d"
        ),
        # v070 fixture: 307200 bytes, blocksize=938 → K=328.  Main
        # scan historically lands at ~488 unique blocks; recovery can
        # add a few more.  Assert we stay well above K.
        min_uniq_blocks=360,
        qr_version=25,
        ec_level=1,
    ),
    "v061": _FixtureSpec(
        video="testcase-v061.mp4",
        input_bin="testcase-v061.input.bin",
        expected_sha=(
            "4a440b6da851a9a2e35eacca95b7b2fe29e3560c169b0a57211fccc2f5469443"
        ),
        min_uniq_blocks=40,
        qr_version=25,
        ec_level=1,
    ),
}


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _build_ground_truth_encoder(
    input_bin: Path, reference_block: bytes,
) -> LTEncoder:
    """Rebuild the encoder that produced ``input_bin``'s QR video.

    The exact ``compressed`` / ``alphanumeric_qr`` flags used at
    encode time are recoverable from any block in the video — we
    unpack the reference block we got out of the decoder and clone
    its flags.  This keeps the test honest across both compressed
    and uncompressed fixtures without having to hard-code per-fixture
    encoder knobs.
    """
    header, _ = unpack(reference_block)
    raw = input_bin.read_bytes()
    if header.compressed:
        import zlib
        payload = zlib.compress(raw)
    else:
        payload = raw
    # blocksize/K are already pinned by the reference header —
    # rebuilding via auto_blocksize with the same knobs must produce
    # the same value (this is asserted implicitly by L3 convergence
    # later).  We still call it so that any drift in auto_blocksize
    # surfaces here rather than as a silent mismatch downstream.
    blocksize = auto_blocksize(
        len(payload),
        ec_level=1,
        qr_version=25,
        alphanumeric_qr=header.alphanumeric_qr,
    )
    assert blocksize == header.blocksize, (
        f"auto_blocksize drift: recomputed={blocksize}, "
        f"recorded={header.blocksize}; this would invalidate the "
        f"L2 ground-truth oracle."
    )
    return LTEncoder(
        payload,
        blocksize=blocksize,
        compressed=header.compressed,
        alphanumeric_qr=header.alphanumeric_qr,
    )


def _ground_truth_block_for_seed(enc: LTEncoder, seed: int) -> bytes:
    """Re-generate the encoded *data* payload for a given seed.

    We compare only the LT-data region, not the full packed block,
    because sequence numbers (``block_seq``) in the decoded frame
    depend on frame ordering and are not part of the LT math.
    """
    enc.prng.set_seed(seed)
    data, _seq = enc.generate_block(seed)
    return data


@pytest.fixture(scope="module", params=["v070", "v061"])
def fixture_spec(request) -> _FixtureSpec:
    spec = _FIXTURES[request.param]
    if not spec.video.exists() or not spec.input_bin.exists():
        pytest.skip(f"fixture {request.param} missing on this checkout")
    return spec


@pytest.mark.slow
def test_layer1_extract_yields_enough_unique_blocks(fixture_spec):
    """L1: main scan (+ recovery) must yield ≥ min_uniq_blocks seeds.

    If this fails in isolation (L2/L3/L4 not reached), the regression
    is in frame-read / WeChat detection / resize pipeline, not in
    LT or the byte-level protocol.
    """
    blocks = extract_qr_from_video(
        str(fixture_spec.video), sample_rate=0, verbose=False, workers=None,
    )
    assert blocks, "decoder returned no blocks"
    assert len(blocks) >= fixture_spec.min_uniq_blocks, (
        f"L1 (frame→block) returned {len(blocks)} unique blocks, "
        f"expected ≥ {fixture_spec.min_uniq_blocks}. "
        f"Likely regression: WeChatQRCode detection, cv2.resize, or "
        f"targeted-recovery wiring."
    )


@pytest.mark.slow
def test_layer2_each_block_matches_ground_truth(fixture_spec):
    """L2: every decoded (seed, data) must byte-equal the ground truth.

    If this fails: the decoder accepted a corrupt block whose CRC
    happened to match but whose payload bytes are wrong.  That
    points at a protocol / CRC / encoder-PRNG bug, not LT.
    """
    blocks = extract_qr_from_video(
        str(fixture_spec.video), sample_rate=0, verbose=False, workers=None,
    )
    assert blocks, "decoder returned no blocks"
    enc = _build_ground_truth_encoder(fixture_spec.input_bin, blocks[0])

    poison_seeds: list[int] = []
    for packed in blocks:
        try:
            header, data = unpack(packed)
        except (ValueError, struct.error):
            continue
        gt = _ground_truth_block_for_seed(enc, header.seed)
        if data[: len(gt)] != gt:
            poison_seeds.append(header.seed)

    assert not poison_seeds, (
        f"L2 (block byte-equality) found {len(poison_seeds)} poison "
        f"blocks, e.g. seeds {poison_seeds[:5]}. "
        f"Likely regression: protocol pack/unpack, CRC, or encoder PRNG."
    )


@pytest.mark.slow
def test_layer3_lt_converges_on_observed_seeds(fixture_spec):
    """L3: LT peeling must converge on the exact seed subset the
    decoder observed.

    If L1/L2 pass but L3 fails, we've landed in a pathological LT
    seed subset (the v070 regression class).  The fix is in the
    frame-recovery path (CLAHE worker, broader recovery gate, etc.),
    not in the LT implementation itself.
    """
    blocks = extract_qr_from_video(
        str(fixture_spec.video), sample_rate=0, verbose=False, workers=None,
    )
    dec = LTDecoder()
    for packed in blocks:
        try:
            dec.decode_bytes(packed, skip_crc=True)
        except (ValueError, struct.error):
            pass

    assert dec.is_done(), (
        f"L3 (LT peeling) stuck at {dec.num_recovered}/{dec.K} after "
        f"{len(blocks)} observed blocks. "
        f"Likely regression: targeted recovery path (e.g. CLAHE "
        f"worker removed, ``sample_rate > 1`` gate re-introduced)."
    )


@pytest.mark.slow
def test_layer4_decoded_bytes_match_input(fixture_spec):
    """L4: full-pipeline byte-equality.

    Redundant with the sibling ``test_real_recordings`` suite, but
    recorded here so the layered view remains self-contained.
    """
    blocks = extract_qr_from_video(
        str(fixture_spec.video), sample_rate=0, verbose=False, workers=None,
    )
    dec = LTDecoder()
    for packed in blocks:
        try:
            dec.decode_bytes(packed, skip_crc=True)
        except (ValueError, struct.error):
            pass
    assert dec.is_done(), "prerequisite: L3 must pass"

    recovered = dec.bytes_dump()
    assert _sha256_bytes(recovered) == fixture_spec.expected_sha, (
        "L4 (final bytes) sha256 drift; recovered payload does not "
        "match the committed fixture oracle."
    )
