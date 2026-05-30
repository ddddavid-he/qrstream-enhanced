"""Optional tests for the Rust V4 core prototype."""

import base64

import pytest

from qrstream import DecodeSession
from qrstream.protocol import base45_encode
from qrstream.raptorq_codec import RaptorQEncoder

qrstream_rs = pytest.importorskip("qrstream_rs")


def _blocks(data: bytes, blocksize: int = 64, extra: int = 4):
    encoder = RaptorQEncoder(data, blocksize)
    return [packed for packed, _, _ in encoder.generate_blocks(encoder.K + extra)]


class TestPythonDecodeSessionRustBackend:
    def test_decode_session_uses_rust_v4_backend(self):
        data = b"python-rust-backend" * 20
        session = DecodeSession(use_rust_v4=True)

        result = None
        for block in _blocks(data):
            result = session.consume_block(block)
            if result.done:
                break

        assert result is not None
        assert result.accepted is True
        assert result.done is True
        assert result.protocol_version == 0x04
        assert session.snapshot().done is True
        assert session.result_bytes() == data

    def test_decode_session_rust_backend_qr_text(self):
        data = b"python-rust-qr-text" * 20
        block = _blocks(data)[0]
        session = DecodeSession(use_rust_v4=True)

        result = session.consume_qr_text(base45_encode(block).decode("ascii"))

        assert result.accepted is True
        assert result.protocol_version == 0x04

    def test_decode_session_rust_backend_duplicate(self):
        block = _blocks(b"python-rust-duplicate" * 20)[0]
        session = DecodeSession(use_rust_v4=True)

        first = session.consume_block(block)
        second = session.consume_block(block)

        assert first.accepted is True
        assert second.accepted is True
        assert second.duplicate is True
        assert second.progress == first.progress


class TestRustV4DecodeSessionPoc:
    def test_decodes_raw_v4_blocks(self):
        data = b"rust-v4-core" * 20
        session = qrstream_rs.V4DecodeSession()

        result = None
        for block in _blocks(data):
            result = session.consume_block(block)
            if result.done:
                break

        assert result is not None
        assert result.accepted is True
        assert result.done is True
        assert result.protocol_version == 0x04
        assert session.result_bytes() == data

    def test_decodes_base45_qr_text(self):
        data = b"rust-base45" * 20
        session = qrstream_rs.V4DecodeSession()
        block = _blocks(data)[0]

        result = session.consume_qr_text(base45_encode(block).decode("ascii"))

        assert result.accepted is True
        assert result.protocol_version == 0x04

    def test_decodes_base64_qr_text(self):
        data = b"rust-base64" * 20
        session = qrstream_rs.V4DecodeSession()
        block = _blocks(data)[0]

        result = session.consume_qr_text(base64.b64encode(block).decode("ascii"))

        assert result.accepted is True
        assert result.protocol_version == 0x04

    def test_reports_duplicates(self):
        data = b"rust-duplicate" * 20
        block = _blocks(data)[0]
        session = qrstream_rs.V4DecodeSession()

        first = session.consume_block(block)
        second = session.consume_block(block)

        assert first.accepted is True
        assert second.accepted is True
        assert second.duplicate is True
        assert second.progress == first.progress

    def test_rejects_non_v4_block(self):
        session = qrstream_rs.V4DecodeSession()

        result = session.consume_block(b"not a v4 block")

        assert result.accepted is False
        assert result.done is False
        assert result.error is not None

    def test_snapshot_before_and_after_decode(self):
        data = b"rust-snapshot" * 20
        session = qrstream_rs.V4DecodeSession()

        empty = session.snapshot()
        assert empty.initialized is False
        assert empty.progress == 0.0

        for block in _blocks(data):
            if session.consume_block(block).done:
                break

        snapshot = session.snapshot()
        assert snapshot.initialized is True
        assert snapshot.done is True
        assert snapshot.progress == 1.0
        assert snapshot.filesize == len(data)
        assert snapshot.protocol_version == 0x04
