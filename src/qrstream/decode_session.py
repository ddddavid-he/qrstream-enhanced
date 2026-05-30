"""Platform-neutral stateful decode session for QRStream blocks.

This module intentionally avoids video, camera, QR-detector, and UI imports.
It is the Python API prototype for a future Rust core: callers feed decoded
QR text or raw protocol blocks and receive immutable progress snapshots.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass

from .lt_codec import DEFAULT_C, DEFAULT_DELTA
from .lt_decoder import LTDecoder
from .protocol import V4_VERSION, base45_decode, unpack
from .raptorq_codec import RaptorQDecoder


@dataclass(frozen=True)
class DecodeSessionResult:
    """Result of feeding one QRStream block candidate."""

    accepted: bool
    duplicate: bool
    done: bool
    progress: float
    num_recovered: int
    symbol_count: int | None
    filesize: int | None
    protocol_version: int | None
    error: str | None = None


@dataclass(frozen=True)
class DecodeSessionSnapshot:
    """Immutable session state for progress UI or FFI boundaries."""

    initialized: bool
    done: bool
    progress: float
    num_recovered: int
    symbol_count: int | None
    filesize: int | None
    protocol_version: int | None


class DecodeSession:
    """Platform-neutral stateful QRStream decoder.

    `consume_qr_text()` is the mobile-facing entry point: platform-native
    camera code scans a QR frame, passes the decoded text here, and receives
    progress/done state. `consume_block()` is the lower-level entry point for
    tests, CLI integrations, and future Rust FFI shims.
    """

    def __init__(self, c: float = DEFAULT_C, delta: float = DEFAULT_DELTA,
                 use_rust_v4: bool = False):
        self.c = c
        self.delta = delta
        self.use_rust_v4 = use_rust_v4
        self._decoder = None
        self._seen_blocks: set[tuple[int, int]] = set()
        self._last_frame_index: int | None = None
        self._last_timestamp: float | None = None

    def consume_qr_text(
        self,
        qr_text: str,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
    ) -> DecodeSessionResult:
        """Decode and consume one QR payload string.

        Returns a non-throwing result for malformed/non-QRStream payloads so a
        camera loop can keep feeding frames without exception handling in the
        hot path.
        """
        self._last_frame_index = frame_index
        self._last_timestamp = timestamp
        if isinstance(self._decoder, _RustV4Decoder):
            return self._decoder.consume_qr_text(qr_text)
        block_bytes = _decode_qr_payload(qr_text)
        if block_bytes is None:
            return self._result(False, False, "invalid QRStream payload")
        return self.consume_block(block_bytes)

    def consume_block(self, block_bytes: bytes) -> DecodeSessionResult:
        """Consume one raw V3/V4 protocol block."""
        if self.use_rust_v4 and self._decoder is None:
            self._decoder = _RustV4Decoder()
            result = self._decoder.consume_block(block_bytes)
            if result.accepted or not _could_be_v3_block(block_bytes):
                return result
            self._decoder = None

        try:
            header, data = unpack(block_bytes)
        except (ValueError, struct.error) as exc:
            return self._result(False, False, str(exc))

        if self._decoder is None:
            try:
                self._decoder = self._new_decoder(header.version)
            except RuntimeError as exc:
                return self._result(False, False, str(exc))

        if isinstance(self._decoder, _RustV4Decoder):
            return self._decoder.consume_block(block_bytes)

        block_id = (header.version, header.seed)
        if block_id in self._seen_blocks:
            return self._result(True, True, None)

        try:
            done, _ = self._decoder.consume_block(header, data)
        except (ValueError, struct.error) as exc:
            return self._result(False, False, str(exc))

        self._seen_blocks.add(block_id)
        return self._result(True, False, None, done=done)

    def snapshot(self) -> DecodeSessionSnapshot:
        """Return current immutable session state."""
        decoder = self._decoder
        if decoder is None:
            return DecodeSessionSnapshot(
                initialized=False,
                done=False,
                progress=0.0,
                num_recovered=0,
                symbol_count=None,
                filesize=None,
                protocol_version=None,
            )
        return DecodeSessionSnapshot(
            initialized=bool(decoder.initialized),
            done=bool(decoder.is_done()),
            progress=float(decoder.progress),
            num_recovered=int(decoder.num_recovered),
            symbol_count=int(decoder.K) if decoder.initialized else None,
            filesize=int(decoder.filesize) if decoder.initialized else None,
            protocol_version=decoder.protocol_version,
        )

    def result_bytes(self) -> bytes:
        """Return reconstructed bytes after completion."""
        if self._decoder is None or not self._decoder.is_done():
            raise RuntimeError("Decoding incomplete — no result available")
        return self._decoder.bytes_dump()

    def try_rescue(self) -> DecodeSessionResult:
        """Attempt decoder-specific rescue after all available blocks are fed."""
        if self._decoder is None:
            return self._result(False, False, "session is not initialized")
        try:
            done = self._decoder.try_gaussian_rescue()
        except Exception as exc:
            return self._result(False, False, str(exc))
        return self._result(done, False, None, done=done)

    def _new_decoder(self, version: int):
        if version == V4_VERSION:
            if self.use_rust_v4:
                return _RustV4Decoder()
            return RaptorQDecoder()
        return LTDecoder(c=self.c, delta=self.delta)

    def _result(
        self,
        accepted: bool,
        duplicate: bool,
        error: str | None,
        *,
        done: bool | None = None,
    ) -> DecodeSessionResult:
        snapshot = self.snapshot()
        is_done = snapshot.done if done is None else bool(done)
        return DecodeSessionResult(
            accepted=accepted,
            duplicate=duplicate,
            done=is_done,
            progress=snapshot.progress,
            num_recovered=snapshot.num_recovered,
            symbol_count=snapshot.symbol_count,
            filesize=snapshot.filesize,
            protocol_version=snapshot.protocol_version,
            error=error,
        )


class _RustV4Decoder:
    def __init__(self):
        try:
            import qrstream_rs
        except ImportError as exc:
            raise RuntimeError("Rust V4 backend is not available") from exc
        self._session = qrstream_rs.V4DecodeSession()

    @property
    def initialized(self) -> bool:
        return self._session.snapshot().initialized

    @property
    def K(self) -> int:
        return self._session.snapshot().symbol_count or 0

    @property
    def filesize(self) -> int:
        return self._session.snapshot().filesize or 0

    @property
    def protocol_version(self) -> int | None:
        return self._session.snapshot().protocol_version

    @property
    def progress(self) -> float:
        return self._session.snapshot().progress

    @property
    def num_recovered(self) -> int:
        return self._session.snapshot().num_recovered

    def is_done(self) -> bool:
        return self._session.snapshot().done

    def consume_block(self, block_bytes: bytes) -> DecodeSessionResult:
        return _convert_rust_result(self._session.consume_block(block_bytes))

    def consume_blocks(self, blocks) -> DecodeSessionResult:
        return _convert_rust_result(self._session.consume_blocks(blocks))

    def consume_qr_text(self, qr_text: str) -> DecodeSessionResult:
        return _convert_rust_result(self._session.consume_qr_text(qr_text))

    def try_gaussian_rescue(self) -> bool:
        return self.is_done()

    def bytes_dump(self) -> bytes:
        return self._session.result_bytes()


def _convert_rust_result(result) -> DecodeSessionResult:
    return DecodeSessionResult(
        accepted=result.accepted,
        duplicate=result.duplicate,
        done=result.done,
        progress=result.progress,
        num_recovered=result.num_recovered,
        symbol_count=result.symbol_count,
        filesize=result.filesize,
        protocol_version=result.protocol_version,
        error=result.error,
    )


def _could_be_v3_block(block_bytes: bytes) -> bool:
    return bool(block_bytes) and block_bytes[0] == 0x03


def _decode_qr_payload(qr_text: str) -> bytes | None:
    for decode_fn in (_try_base45, _try_base64):
        candidate = decode_fn(qr_text)
        if candidate is None:
            continue
        try:
            unpack(candidate)
        except (ValueError, struct.error):
            continue
        return candidate
    return None


def _try_base45(qr_text: str) -> bytes | None:
    try:
        return base45_decode(qr_text)
    except (ValueError, KeyError):
        return None


def _try_base64(qr_text: str) -> bytes | None:
    try:
        return base64.b64decode(qr_text)
    except (ValueError, base64.binascii.Error):
        return None
