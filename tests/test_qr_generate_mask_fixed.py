"""
Regression tests for perf/encoder-segno-mask.

Lock in:

1. generate_qr_image() always uses a deterministic mask pattern, so
   the same (payload, version, ec, alphanumeric) inputs yield
   byte-identical images across runs.

2. The vectorized _render_qr path produces the same pixels as the
   reference nested-loop implementation would.

If a future segno upgrade changes the mask=0 bit layout these tests
will flip red immediately; update the stored hashes only after
verifying the new layout still decodes cleanly via WeChatQRCode on
the real fixtures.
"""

import numpy as np
import pytest

from qrstream.qr_utils import generate_qr_image, try_decode_qr


# ── Determinism tests ────────────────────────────────────────────
# Each tuple: (payload, version, alphanumeric).
# Payloads are chosen so that (encoded size, version, ec=M) never
# overflows, and empty payload is only tested with alphanumeric=False
# (segno rejects alphanumeric mode on empty input).

_DETERMINISM_CASES = [
    # empty — byte mode only (alphanumeric rejects empty)
    (b"", 5, False),
    # tiny payloads — both modes, small version
    (b"A", 5, True),
    (b"A", 5, False),
    # medium payload — fits V5 in both modes
    (b"hello world", 5, True),
    (b"hello world", 5, False),
    # larger payloads — need bigger versions
    (b"hello world", 15, True),
    (b"hello world", 25, True),
    (bytes(range(256)) * 2, 25, False),       # 512 bytes → base64 ~684 chars, V25 M
    (bytes(range(256)) * 2, 25, True),        # 512 bytes → base45, V25 M
    (b"QRSTREAM-" * 40, 25, True),            # ~360 bytes → base45, V25 M
    (b"QRSTREAM-" * 40, 25, False),           # ~360 bytes → base64, V25 M
    (b"QRSTREAM-" * 10, 15, True),            # ~90 bytes → base45, V15 M
    (b"QRSTREAM-" * 10, 15, False),           # ~90 bytes → base64, V15 M
]


@pytest.mark.parametrize("payload,version,alphanumeric", _DETERMINISM_CASES)
def test_generate_qr_image_is_deterministic(payload, version, alphanumeric):
    """Same inputs → bitwise-identical image on every call."""
    a = generate_qr_image(
        payload, ec_level=1, version=version, alphanumeric=alphanumeric,
    )
    b = generate_qr_image(
        payload, ec_level=1, version=version, alphanumeric=alphanumeric,
    )
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert np.array_equal(a, b), (
        "generate_qr_image is not deterministic; did mask get unpinned?"
    )


@pytest.mark.parametrize("payload,version", [
    (b"hello world", 5),
    (b"QRSTREAM-" * 10, 10),
])
def test_generate_qr_image_round_trips_through_wechat(payload, version):
    """Sanity: whichever mask we pinned to still scans cleanly.

    Uses the default alphanumeric=True (base45) path — the production
    encoding.  The decoder sees base45-encoded ASCII; we verify
    against the same encoding.
    """
    from qrstream.protocol import base45_encode

    img = generate_qr_image(
        payload, ec_level=1, version=version, alphanumeric=True,
    )
    expected = base45_encode(payload).decode("ascii")
    got = try_decode_qr(img)
    assert got == expected


def test_vectorized_paint_matches_reference():
    """The vectorized paint must match the old nested-loop exactly.

    We rebuild the reference in-place here rather than importing a
    private helper so this test stays stable if the implementation
    file is reorganized.
    """
    # Pick a payload + version combination that hits every module
    # pattern (binary payload at V20 is a good mix of finder /
    # alignment / timing / data modules).
    # Uses the default alphanumeric=True (base45) production path.
    payload = bytes(range(256))
    img = generate_qr_image(
        payload, ec_level=1, version=20, alphanumeric=True,
    )

    # Reference: regenerate the same QR via segno directly and paint
    # with the nested loop. (This is what _render_qr did before the
    # vectorization.)
    import cv2
    import segno
    from qrstream.protocol import base45_encode

    b45 = base45_encode(payload).decode("ascii")
    qr = segno.make(b45, version=20, error="m", mode="alphanumeric",
                    boost_error=False, mask=0)
    mat = qr.matrix
    n = len(mat)
    bs = 10
    bd = 4
    side = (n + 2 * bd) * bs
    ref = np.full((side, side), 255, dtype=np.uint8)
    for r, row in enumerate(mat):
        for c, v in enumerate(row):
            if v & 1:
                y = (r + bd) * bs
                x = (c + bd) * bs
                ref[y:y + bs, x:x + bs] = 0
    ref_bgr = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    assert np.array_equal(img, ref_bgr), (
        "vectorized _render_qr diverged from the reference nested-loop "
        "paint; fix the vectorization (do not weaken this test)."
    )
