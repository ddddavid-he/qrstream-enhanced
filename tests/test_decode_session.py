"""Tests for the platform-neutral DecodeSession API."""

import base64

import pytest

from qrstream import DecodeSession
from qrstream.decode_session import DecodeSessionSnapshot
from qrstream.encoder import LTEncoder
from qrstream.protocol import base45_encode, pack_v3
from qrstream.raptorq_codec import RaptorQEncoder


def _raptorq_blocks(data: bytes, blocksize: int = 64, extra: int = 4):
    encoder = RaptorQEncoder(data, blocksize)
    return [packed for packed, _, _ in encoder.generate_blocks(encoder.K + extra)]


def _lt_blocks(data: bytes, blocksize: int = 64, overhead: float = 3.0):
    encoder = LTEncoder(data, blocksize)
    return [packed for packed, _, _ in encoder.generate_blocks(int(encoder.K * overhead))]


class TestDecodeSessionRawBlocks:
    def test_v4_raptorq_blocks_decode(self):
        data = b"decode-session-v4" * 20
        session = DecodeSession()

        last = None
        for block in _raptorq_blocks(data):
            last = session.consume_block(block)
            if last.done:
                break

        assert last is not None
        assert last.accepted is True
        assert last.done is True
        assert last.protocol_version == 0x04
        assert session.result_bytes() == data

    def test_v3_lt_blocks_decode(self):
        data = b"decode-session-v3" * 20
        session = DecodeSession()

        last = None
        for block in _lt_blocks(data):
            last = session.consume_block(block)
            if last.done:
                break

        assert last is not None
        assert last.accepted is True
        assert last.done is True
        assert last.protocol_version == 0x03
        assert session.result_bytes() == data

    def test_duplicate_block_is_reported_without_progress_regression(self):
        data = b"duplicate-frame" * 20
        first_block = _raptorq_blocks(data)[0]
        session = DecodeSession()

        first = session.consume_block(first_block)
        duplicate = session.consume_block(first_block)

        assert first.accepted is True
        assert duplicate.accepted is True
        assert duplicate.duplicate is True
        assert duplicate.progress == first.progress
        assert duplicate.num_recovered == first.num_recovered

    def test_invalid_block_does_not_initialize_session(self):
        session = DecodeSession()

        result = session.consume_block(b"not a qrstream block")

        assert result.accepted is False
        assert result.error is not None
        assert session.snapshot() == DecodeSessionSnapshot(
            initialized=False,
            done=False,
            progress=0.0,
            num_recovered=0,
            symbol_count=None,
            filesize=None,
            protocol_version=None,
        )

    def test_mixed_protocol_blocks_are_rejected_after_initialization(self):
        session = DecodeSession()
        v4_block = _raptorq_blocks(b"mixed-protocol" * 20)[0]
        v3_block = pack_v3(
            filesize=128,
            blocksize=64,
            block_count=2,
            seed=1,
            block_seq=0,
            data=b"A" * 64,
        )

        first = session.consume_block(v4_block)
        mixed = session.consume_block(v3_block)

        assert first.accepted is True
        assert mixed.accepted is False
        assert "version mismatch" in mixed.error
        assert session.snapshot().protocol_version == 0x04

    def test_result_bytes_raises_before_completion(self):
        session = DecodeSession()
        first_block = _raptorq_blocks(b"incomplete" * 100, extra=0)[0]

        session.consume_block(first_block)

        with pytest.raises(RuntimeError, match="Decoding incomplete"):
            session.result_bytes()


class TestDecodeSessionQrText:
    def test_consume_base45_qr_text(self):
        data = b"base45-payload" * 20
        block = _raptorq_blocks(data)[0]
        qr_text = base45_encode(block).decode("ascii")
        session = DecodeSession()

        result = session.consume_qr_text(qr_text, frame_index=12, timestamp=1.25)

        assert result.accepted is True
        assert result.protocol_version == 0x04

    def test_consume_base64_qr_text(self):
        data = b"base64-payload" * 20
        block = _raptorq_blocks(data)[0]
        qr_text = base64.b64encode(block).decode("ascii")
        session = DecodeSession()

        result = session.consume_qr_text(qr_text)

        assert result.accepted is True
        assert result.protocol_version == 0x04

    def test_malformed_qr_text_is_non_fatal(self):
        session = DecodeSession()

        result = session.consume_qr_text("this is not a qrstream payload")

        assert result.accepted is False
        assert result.done is False
        assert result.error == "invalid QRStream payload"
        assert session.snapshot().initialized is False


class TestDecodeSessionSnapshot:
    def test_snapshot_transitions_from_empty_to_done(self):
        data = b"snapshot" * 20
        session = DecodeSession()

        empty = session.snapshot()
        assert empty.initialized is False
        assert empty.progress == 0.0

        done = None
        for block in _raptorq_blocks(data):
            done = session.consume_block(block)
            if done.done:
                break

        snapshot = session.snapshot()
        assert done is not None
        assert snapshot.initialized is True
        assert snapshot.done is True
        assert snapshot.progress == 1.0
        assert snapshot.filesize == len(data)
        assert snapshot.protocol_version == 0x04
