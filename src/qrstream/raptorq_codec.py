"""
RaptorQ (RFC 6330) codec for QRStream.

Wraps the ``raptorq`` PyPI package (Rust implementation with PyO3
bindings) to provide an encoder/decoder interface compatible with the
existing LT pipeline.

Key differences from the LT codec:

* **Systematic code**: the first K packets carry the original source
  data unmodified.  Repair packets begin at ESI = K.
* **Near-optimal recovery**: decoding succeeds with high probability
  as soon as any K packets (source or repair) are received.
* **No PRNG / degree distribution**: block-to-source mapping is handled
  internally by the RaptorQ algebra; the wire identifier is a simple
  32-bit ESI (Encoding Symbol Identifier).
* **No Gauss-Jordan rescue**: the built-in decoder handles all
  recovery internally.
"""

import io
import struct
import zlib
from math import ceil

import raptorq as _raptorq

from .protocol import (
    V4Header,
    _resolve_alphanumeric_flag,
    pack_v4,
    unpack,
)

# 4-byte big-endian ESI header that the raptorq library prepends to
# each serialised packet.
_RQ_ESI_HEADER_SIZE = 4


class RaptorQEncoder:
    """Encodes a payload into RaptorQ-coded symbols for QR streaming.

    Interface mirrors :class:`qrstream.encoder.LTEncoder` so the
    encode pipeline can swap codecs transparently.
    """

    def __init__(self, data, blocksize: int,
                 compressed: bool = False,
                 binary_qr: bool = False,
                 alphanumeric_qr: bool | None = None):
        if isinstance(data, (bytes, bytearray)):
            self.data = bytes(data)
        else:
            # MmapDataSource or similar — materialise for raptorq which
            # requires a contiguous bytes object.
            self.data = bytes(data[:len(data)])
        self.filesize = len(self.data)
        self.compressed = compressed
        self.alphanumeric_qr = _resolve_alphanumeric_flag(
            binary_qr, alphanumeric_qr)

        # The raptorq library may adjust the symbol size for internal
        # alignment (e.g. rounding to a multiple of its sub-symbol size
        # Al).  We probe the actual symbol size from a test packet and
        # use that as the effective blocksize.
        padded = self.data
        remainder = self.filesize % blocksize
        if remainder != 0:
            padded = self.data + b'\x00' * (blocksize - remainder)
        self._encoder = _raptorq.Encoder.with_defaults(padded, blocksize)
        # Probe actual symbol size from first packet.
        probe_packets = self._encoder.get_encoded_packets(0)
        if probe_packets:
            actual_symbol_size = len(probe_packets[0]) - _RQ_ESI_HEADER_SIZE
        else:
            actual_symbol_size = blocksize
        self.blocksize = actual_symbol_size
        self.K = len(probe_packets) if probe_packets else (
            ceil(self.filesize / self.blocksize) if self.filesize > 0 else 0)
        self._seq = 0

    # Keep ``binary_qr`` as a read-only alias for symmetry with
    # LTEncoder.
    @property
    def binary_qr(self) -> bool:
        return self.alphanumeric_qr

    def generate_blocks(self, count: int):
        """Generate ``count`` encoded symbols as packed V4 byte strings.

        The first K packets are systematic (source data); the
        remaining ``count - K`` are repair symbols.

        Yields ``(packed_v4_bytes, esi, seq)`` triples.
        """
        repair_count = max(0, count - self.K)
        packets = self._encoder.get_encoded_packets(repair_count)

        # ``packets`` contains K source + repair_count repair.
        # Each packet is: 4-byte BE ESI + symbol_size payload.
        self._seq = 0
        for pkt in packets[:count]:
            esi = struct.unpack('>I', pkt[:_RQ_ESI_HEADER_SIZE])[0]
            symbol_data = pkt[_RQ_ESI_HEADER_SIZE:]
            seq = self._seq & 0xFFFF
            packed = pack_v4(
                filesize=self.filesize,
                symbol_size=self.blocksize,
                symbol_count=self.K,
                esi=esi,
                block_seq=seq,
                data=symbol_data,
                compressed=self.compressed,
                alphanumeric_qr=self.alphanumeric_qr,
            )
            yield packed, esi, seq
            self._seq += 1


class RaptorQDecoder:
    """Consumes RaptorQ V4 symbols and reconstructs the original data.

    Interface mirrors :class:`qrstream.decoder.LTDecoder`.
    """

    def __init__(self):
        self.K = 0
        self.filesize = 0
        self.blocksize = 0      # = symbol_size
        self.done = False
        self.compressed = False
        self.protocol_version = None
        self.initialized = False
        self._rq_decoder = None
        self._result: bytes | None = None
        self._fed_count = 0
        # Track which source blocks we can confirm as received.
        # For ESI < K (systematic source symbols) we know exactly
        # which block it is.  For ESI >= K (repair) we don't know
        # which blocks they help — the raptorq library handles that
        # internally.  When decoding completes, all K blocks are
        # marked at once.
        #
        # Keyed by block index → True, matching the dict-key-membership
        # protocol that ``compute_block_map_cells`` expects (same as
        # ``BlockGraph.eliminated``).
        self.eliminated: dict[int, bool] = {}

    @property
    def progress(self) -> float:
        if not self.initialized or self.K == 0:
            return 0.0
        if self.done:
            return 1.0
        # Approximate: fed / K, capped at 0.99 until actually done.
        return min(self._fed_count / self.K, 0.99)

    @property
    def num_recovered(self) -> int:
        if self.done:
            return self.K
        return min(self._fed_count, self.K)

    def is_done(self) -> bool:
        return self.done

    def consume_block(self, header, data: bytes) -> tuple[bool, bool]:
        """Feed a parsed V4 block (header + data) into the decoder.

        Returns ``(done, compressed)``.
        """
        filesize = header.filesize
        symbol_size = header.blocksize   # V4Header.blocksize property
        symbol_count = header.block_count  # V4Header.block_count property
        esi = header.seed                  # V4Header.seed property
        compressed = header.compressed

        if symbol_size <= 0:
            raise ValueError(f"Invalid symbol_size: {symbol_size}")

        if not self.initialized:
            self.protocol_version = header.version
            self.filesize = filesize
            self.blocksize = symbol_size
            self.K = symbol_count
            self.compressed = compressed
            # Padded length for raptorq decoder (K * symbol_size).
            padded_len = self.K * symbol_size
            self._rq_decoder = _raptorq.Decoder.with_defaults(
                padded_len, symbol_size)
            self.initialized = True
        else:
            if header.version != self.protocol_version:
                raise ValueError(
                    f"version mismatch: {header.version} != "
                    f"{self.protocol_version}")
            if filesize != self.filesize:
                raise ValueError(
                    f"filesize mismatch: {filesize} != {self.filesize}")
            if symbol_size != self.blocksize:
                raise ValueError(
                    f"symbol_size mismatch: {symbol_size} != {self.blocksize}")
            if symbol_count != self.K:
                raise ValueError(
                    f"symbol_count mismatch: {symbol_count} != {self.K}")
            if compressed != self.compressed:
                raise ValueError(
                    f"compressed flag mismatch: {compressed} != "
                    f"{self.compressed}")

        if self.done:
            return True, self.compressed

        # Reconstruct the raptorq packet: 4-byte BE ESI + symbol data.
        # Pad data to symbol_size if short.
        if len(data) < symbol_size:
            data = data + b'\x00' * (symbol_size - len(data))
        elif len(data) > symbol_size:
            data = data[:symbol_size]

        pkt = struct.pack('>I', esi) + data
        result = self._rq_decoder.decode(pkt)
        self._fed_count += 1

        # Track source-symbol reception for block map display.
        if esi < self.K:
            self.eliminated[esi] = True

        if result is not None:
            # Trim to original filesize (remove padding).
            self._result = result[:self.filesize]
            self.done = True
            # Fill all remaining blocks — decoding is complete.
            for i in range(self.K):
                if i not in self.eliminated:
                    self.eliminated[i] = True

        return self.done, self.compressed

    def try_gaussian_rescue(self) -> bool:
        """No-op for RaptorQ — recovery is handled internally.

        Returns ``True`` if already decoded, ``False`` otherwise.
        """
        return self.done

    def decode_bytes(self, block_bytes: bytes,
                     skip_crc: bool = False) -> tuple[bool, bool]:
        """Decode a raw V4 protocol block from bytes."""
        header, data = unpack(block_bytes, skip_crc=skip_crc)
        return self.consume_block(header, data)

    def _iter_recovered_chunks(self):
        """Yield recovered source data in blocksize chunks."""
        if self._result is None:
            raise RuntimeError("Decoding incomplete — no result available")
        for offset in range(0, len(self._result), self.blocksize):
            end = min(offset + self.blocksize, len(self._result))
            yield self._result[offset:end]

    def bytes_dump(self) -> bytes:
        """Reconstruct the original file data."""
        if self._result is None:
            raise RuntimeError("Decoding incomplete — no result available")
        raw = self._result
        if self.compressed:
            try:
                return zlib.decompress(raw)
            except zlib.error as e:
                raise RuntimeError(
                    f"Decompression failed: {e}. "
                    f"Decoded payload may be corrupted.") from e
        return raw

    def bytes_dump_to_file(self, output_path: str,
                           show_progress: bool = False) -> int:
        """Write the reconstructed output directly to a file."""
        del show_progress
        raw = self.bytes_dump()
        with open(output_path, 'wb') as f:
            f.write(raw)
        return len(raw)
