"""Real-camera QR detection regression reported in zxing-cpp #1132."""

import hashlib
from pathlib import Path

import cv2
import pytest
import zxingcpp

from qrstream.qr_utils import try_decode_qr


_IMAGE = Path(__file__).parent / "fixtures" / "zxing-1132" / "frame142.png"
_IMAGE_SHA = "e05152676b85972c1e93eeb475f4e018dbfad66dc4411f69c9ad1aed6ada9672"
_PAYLOAD_SHA = "bc235bbcc0907917fce0e1cbcb10a7a3c1134ebf5d7e0e5cee504a830f707316"


@pytest.fixture
def regression_frame():
    assert hashlib.sha256(_IMAGE.read_bytes()).hexdigest() == _IMAGE_SHA
    frame = cv2.imread(str(_IMAGE))
    assert frame is not None and frame.shape == (842, 840, 3)
    return frame


def test_zxing_default_reader_recovers_reported_frame(regression_frame):
    result = zxingcpp.read_barcode(regression_frame, formats=zxingcpp.QRCode)
    assert result is not None and result.valid
    assert len(result.bytes) == 3378
    assert hashlib.sha256(result.bytes).hexdigest() == _PAYLOAD_SHA


def test_production_reader_recovers_reported_frame(regression_frame):
    text = try_decode_qr(regression_frame)
    assert text is not None
    payload = text.encode("ascii")
    assert len(payload) == 3378
    assert hashlib.sha256(payload).hexdigest() == _PAYLOAD_SHA
