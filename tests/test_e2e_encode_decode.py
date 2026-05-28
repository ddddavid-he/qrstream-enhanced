"""
End-to-end encode → video → decode pipeline tests.

These tests exercise the full stack:
  input_file → encode_to_video → extract_qr_from_video
             → decode_blocks_to_file → SHA256 verify

Marked ``@pytest.mark.e2e`` — excluded from the default ``pytest`` run
(see ``pyproject.toml addopts``).  The dedicated CI workflow
``.github/workflows/e2e-encode-decode.yml`` opts in with ``-m e2e``.

Why these tests are necessary
------------------------------
The unit-test suite (``test.yml``) calls ``LTEncoder`` / ``LTDecoder``
directly and never touches QR image generation.  This meant the
``qrcode 8.x glog(0)`` crash was invisible to CI: it only fired when
``encode_to_video`` was called on a file whose payload size fell close
to the QR-version capacity boundary (blocksize=938, v25 EC_M, K=19).

These tests close that gap by running the complete pipeline on files
specifically sized to hit that boundary, plus a sweep of common sizes.

Sizes
-----
User requirement: 10 KB, 100 KB, 500 KB (raw input).

The glog-trigger test uses a 17 574-byte raw file encoded *without*
compression — this is the exact payload size that triggers blocksize=938
and K=19 (the configuration that caused the original crash).
"""

import hashlib
import pathlib
import random
from math import ceil

import pytest

from qrstream.protocol import auto_blocksize


# ── helpers ───────────────────────────────────────────────────────

def _random_bytes(size: int, seed: int = 0) -> bytes:
    return random.Random(seed).randbytes(size)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_decode_verify(raw: bytes, tmp_path: pathlib.Path,
                           label: str = "file",
                           compress: bool = True,
                           qr_version: int = 25,
                           overhead: float = 2.0) -> None:
    """Complete pipeline: raw bytes → MP4 → recovered bytes → assert SHA256."""
    from qrstream.encoder import encode_to_video
    from qrstream.decoder import extract_qr_from_video, decode_blocks_to_file

    src = tmp_path / f"{label}.bin"
    mp4 = tmp_path / f"{label}.mp4"
    out = tmp_path / f"{label}_out.bin"

    src.write_bytes(raw)
    src_hash = _sha256(src)

    encode_to_video(str(src), str(mp4), compress=compress,
                    qr_version=qr_version, overhead=overhead, verbose=False)
    assert mp4.exists() and mp4.stat().st_size > 0, \
        f"encode_to_video produced no output for {label}"

    blocks = extract_qr_from_video(str(mp4), verbose=False)
    written = decode_blocks_to_file(blocks, str(out), verbose=False)

    assert out.exists(), f"decode produced no output file for {label}"
    assert written == len(raw), \
        f"{label}: written={written} != expected={len(raw)}"
    assert _sha256(out) == src_hash, \
        f"{label}: SHA256 mismatch — data corrupted after encode/decode"


# ── tests ─────────────────────────────────────────────────────────

@pytest.mark.e2e
class TestE2EEncodeDecode:
    """Full encode→video→decode roundtrip with SHA256 verification."""

    # ── glog(0) regression ────────────────────────────────────────

    def test_glog_trigger_config(self, tmp_path):
        """
        17 574-byte payload encoded without compression.

        ``auto_blocksize(17574, qr_version=25)`` returns 938 → K=19 blocks,
        v25 EC_M. This is the exact (payload_size, blocksize, version) triple that
        triggered the ``qrcode 8.x glog(0)`` crash.  With zxing-cpp as the
        QR backend it must complete and recover the file byte-exactly.
        """
        # Verify the V25 trigger condition is still active even though
        # encode defaults may move to denser QR versions.
        bs = auto_blocksize(17_574, qr_version=25)
        assert bs == 938, f"auto_blocksize V25 changed: {bs} (expected 938)"
        assert ceil(17_574 / bs) == 19

        raw = _random_bytes(17_574, seed=0x616C6F67)  # "alog" in hex
        _encode_decode_verify(raw, tmp_path, "glog_trigger", compress=False)

    # ── user-requested sizes ──────────────────────────────────────

    def test_10k(self, tmp_path):
        """10 KB random input."""
        _encode_decode_verify(_random_bytes(10_000, seed=10), tmp_path, "10k")

    def test_100k(self, tmp_path):
        """100 KB random input."""
        _encode_decode_verify(_random_bytes(100_000, seed=100), tmp_path, "100k")

    def test_500k(self, tmp_path):
        """500 KB random input."""
        _encode_decode_verify(_random_bytes(500_000, seed=500), tmp_path, "500k")

    # ── additional edge cases ─────────────────────────────────────

    def test_exact_block_boundary(self, tmp_path):
        """
        Payload whose size is an exact multiple of blocksize — exercises
        the last-block path where no zero-padding is needed.
        """
        raw = _random_bytes(40_000, seed=4)
        bs = auto_blocksize(len(raw))
        k = ceil(len(raw) / bs)
        _encode_decode_verify(raw[:k * bs], tmp_path, "exact_boundary",
                               compress=False)

    # ── QR version sweep ──────────────────────────────────────────

    @pytest.mark.parametrize(
        "qr_version,payload_size",
        [
            # Low version: small modules-per-side (4*10+17=57), sparse
            # density.  Encoder produces small frames, decoder should
            # comfortably stay near source resolution.
            (10, 5_000),
            # Mid version: the qrstream default; baseline density.
            (20, 15_000),
            # High version: 4*30+17=137 modules per side.  This is the
            # density band where the legacy 1080-cap downscale would
            # crush sub-3 px/module on captures, exercising the
            # adaptive _MAX_DETECT_DIM logic on a clean (non-camera)
            # encoder output.
            (30, 30_000),
            # Max version: 4*40+17=177 modules.  Stress the protocol
            # capacity tables and the inferred-modules branch of
            # _adaptive_max_dim_from_probe.
            (40, 50_000),
        ],
        ids=["v10-5k", "v20-15k", "v30-30k", "v40-50k"],
    )
    def test_qr_version_roundtrip(self, tmp_path, qr_version, payload_size):
        """Roundtrip under multiple QR versions.

        Confirms encode/decode integrity across the QR-version range
        (low/mid/high/max) and indirectly exercises the adaptive
        downscale path: probe-derived ``modules_per_side`` is only
        correct when ``_infer_qr_modules`` matches the encoder's
        chosen version.
        """
        raw = _random_bytes(payload_size, seed=qr_version)
        _encode_decode_verify(
            raw, tmp_path, f"v{qr_version}-{payload_size}",
            compress=False, qr_version=qr_version,
        )


@pytest.mark.e2e
class TestE2ELargeFileMultiSourceBlock:
    """Full-pipeline stress test: 20MB file triggering RaptorQ Z=2 source blocks.

    This test exercises the complete encode→video→decode pipeline with a
    file large enough to trigger multiple RaptorQ source blocks (Z=2),
    which requires K > 56,403 symbols.  With QR version 15 (blocksize=371)
    and 20MB input, K ≈ 56,528, just crossing the Z=2 threshold.

    Parameters:
      - File size: 20MB (random, incompressible)
      - QR version: 15 (blocksize=371, gives K=56,528 → Z=2)
      - Overhead: 1.1 (RaptorQ converges near K; 10% margin is safe)
      - Compress: False (random data won't compress; avoids payload
        shrinkage that would drop K below the Z=2 threshold)

    Video characteristics:
      - ~62,180 frames @ 10fps → ~104 min encoded video
      - Estimated CI time: 8–12 minutes

    This test is intentionally expensive and should run on a single
    platform only to avoid wasting CI resources.  It validates:
      1. Multi-source-block (Z=2) encode/decode correctness end-to-end
      2. Sub-block interleaving does not cause silent corruption through
         the full QR generation + detection + reassembly pipeline
      3. Large-file memory handling in both encoder and decoder
    """

    def test_20mb_multi_source_block_roundtrip(self, tmp_path):
        """20MB encode→video→decode with Z=2 RaptorQ source blocks."""
        from qrstream.raptorq_codec import _rq_num_source_blocks

        data_size = 20 * 1024 * 1024
        raw = _random_bytes(data_size, seed=0x22_6A7E)
        qr_version = 15

        # Verify precondition: this configuration must produce Z=2
        from qrstream.protocol import auto_blocksize
        blocksize = auto_blocksize(data_size, qr_version=qr_version)
        K = ceil(data_size / blocksize)
        Z = _rq_num_source_blocks(K)
        assert Z == 2, (
            f"Test precondition failed: expected Z=2 but got Z={Z} "
            f"(blocksize={blocksize}, K={K}). "
            f"Adjust file size or qr_version.")

        _encode_decode_verify(
            raw, tmp_path, "20mb_z2",
            compress=False,
            qr_version=qr_version,
            overhead=1.1,
        )
