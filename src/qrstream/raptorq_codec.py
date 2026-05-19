"""
RaptorQ (RFC 6330) codec for QRStream.

Wraps the ``raptorq`` PyPI package (Rust implementation with PyO3
bindings) to provide an encoder/decoder interface compatible with the
existing LT pipeline.

Key differences from the LT codec:

* **Systematic code**: source packets carry original source symbols;
  repair packets carry parity symbols for the same RaptorQ source block.
* **Near-optimal recovery**: decoding succeeds with high probability
  as soon as any K packets (source or repair) are received.
* **RaptorQ PayloadId**: the wire identifier is ``SBN || ESI``
  (1-byte source block number + 24-bit local encoding symbol id), not a
  flat global ESI.  QRStream maps systematic PayloadIds back to global
  source-symbol indices for block-map rendering.
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

# 4-byte RaptorQ PayloadId header that the raptorq library prepends to
# each serialised packet: 1-byte source block number + 24-bit local ESI.
_RQ_ESI_HEADER_SIZE = 4
_RQ_ESI_MASK = 0x00FF_FFFF
_RQ_SBN_SHIFT = 24
_RQ_MAX_SOURCE_SYMBOLS_PER_BLOCK = 56_403


def _rq_payload_id_parts(payload_id: int) -> tuple[int, int]:
    """Return ``(source_block_number, local_esi)`` from PayloadId."""
    return (payload_id >> _RQ_SBN_SHIFT) & 0xFF, payload_id & _RQ_ESI_MASK


def _rq_payload_id(source_block_number: int, local_esi: int) -> int:
    """Pack RaptorQ ``source_block_number`` and local ESI into PayloadId."""
    return ((source_block_number & 0xFF) << _RQ_SBN_SHIFT) | (
        local_esi & _RQ_ESI_MASK)


def _rq_num_source_blocks(total_symbols: int) -> int:
    """Mirror ``raptorq`` defaults for source-block partition count."""
    if total_symbols <= 0:
        return 0
    return ceil(total_symbols / _RQ_MAX_SOURCE_SYMBOLS_PER_BLOCK)


def _rq_source_blocks_from_packets(packets: list[bytes]) -> int:
    """Return Z by counting unique SBNs in serialised RaptorQ packets."""
    sbns = set()
    for pkt in packets:
        if len(pkt) >= _RQ_ESI_HEADER_SIZE:
            payload_id = struct.unpack('>I', pkt[:_RQ_ESI_HEADER_SIZE])[0]
            sbn, _ = _rq_payload_id_parts(payload_id)
            sbns.add(sbn)
    return len(sbns)


def _rq_source_block_layout(total_symbols: int,
                            source_blocks: int | None = None
                            ) -> list[tuple[int, int]]:
    """Return ``[(global_start, symbol_count), ...]`` for RaptorQ SBNs.

    The upstream Rust implementation uses RFC 6330 ``Partition[Kt, Z]``:
    the first ``ZL`` source blocks are one symbol larger when ``Kt`` is not
    divisible by ``Z``.  If ``source_blocks`` is omitted, QRStream mirrors
    the default ``raptorq`` Z; decoded V4 frames pass header.reserved here.
    """
    if source_blocks is None:
        source_blocks = _rq_num_source_blocks(total_symbols)
    if total_symbols <= 0 or source_blocks <= 0:
        return []

    large = ceil(total_symbols / source_blocks)
    small = large - 1
    large_count = total_symbols - small * source_blocks

    layout: list[tuple[int, int]] = []
    offset = 0
    for sbn in range(source_blocks):
        count = large if sbn < large_count else small
        layout.append((offset, count))
        offset += count
    return layout


def _rq_source_index(payload_id: int, total_symbols: int,
                     source_blocks: int | None = None) -> int | None:
    """Map a systematic PayloadId to QRStream's global source index.

    Returns ``None`` for repair PayloadIds or out-of-range source block ids.
    """
    sbn, local_esi = _rq_payload_id_parts(payload_id)
    layout = _rq_source_block_layout(total_symbols, source_blocks)
    if sbn >= len(layout):
        return None
    offset, count = layout[sbn]
    if local_esi >= count:
        return None
    return offset + local_esi


def _rq_source_ordinal(payload_id: int, total_symbols: int,
                       source_blocks: int | None = None
                       ) -> tuple[int, int] | None:
    """Return ``(local_esi, sbn)`` for source-first round-robin ordering."""
    source_idx = _rq_source_index(payload_id, total_symbols, source_blocks)
    if source_idx is None:
        return None
    sbn, local_esi = _rq_payload_id_parts(payload_id)
    return local_esi, sbn


def _rq_repair_ordinal(payload_id: int, total_symbols: int,
                       source_blocks: int | None = None) -> tuple[int, int]:
    """Return ``(repair_index_within_sbn, sbn)`` for stable repair ordering."""
    sbn, local_esi = _rq_payload_id_parts(payload_id)
    layout = _rq_source_block_layout(total_symbols, source_blocks)
    if sbn < len(layout):
        _, count = layout[sbn]
        return max(0, local_esi - count), sbn
    return local_esi, sbn


def _rq_order_packets(packets: list[bytes], total_symbols: int,
                      source_blocks: int | None = None) -> list[bytes]:
    """Order RaptorQ packets as source round-robin, then repair round-robin."""
    source_packets: list[tuple[tuple[int, int], bytes]] = []
    repair_packets: list[tuple[tuple[int, int], bytes]] = []
    for pkt in packets:
        payload_id = struct.unpack('>I', pkt[:_RQ_ESI_HEADER_SIZE])[0]
        source_ordinal = _rq_source_ordinal(
            payload_id, total_symbols, source_blocks)
        if source_ordinal is None:
            repair_packets.append(
                (_rq_repair_ordinal(payload_id, total_symbols, source_blocks),
                 pkt))
        else:
            source_packets.append((source_ordinal, pkt))

    source_packets.sort(key=lambda item: item[0])
    repair_packets.sort(key=lambda item: item[0])
    ordered = [pkt for _, pkt in source_packets]
    ordered.extend(pkt for _, pkt in repair_packets)
    return ordered


class RaptorQEncoder:
    """Encodes a payload into RaptorQ-coded symbols for QR streaming.

    Interface mirrors :class:`qrstream.encoder.LTEncoder` so the
    encode pipeline can swap codecs transparently.
    """

    def __init__(self, data, blocksize: int,
                 compressed: bool = False,
                 binary_qr: bool = False,
                 alphanumeric_qr: bool | None = None):
        if isinstance(data, bytes):
            self.data = data
        elif isinstance(data, bytearray):
            self.data = bytes(data)
        else:
            # MmapDataSource or similar random-access input.  Keep it
            # file-backed so systematic source symbols can be emitted
            # without eagerly copying the whole file into memory.
            self.data = data
        self.filesize = len(self.data)
        self.compressed = compressed
        self.alphanumeric_qr = _resolve_alphanumeric_flag(
            binary_qr, alphanumeric_qr)
        self._requested_blocksize = blocksize
        self._encoder = None

        # The raptorq library may adjust the symbol size for internal
        # alignment (e.g. rounding to a multiple of its sub-symbol size
        # Al).  Probe with a tiny payload so mmap-backed inputs are not
        # materialised during construction.
        probe_encoder = _raptorq.Encoder.with_defaults(b'\x00', blocksize)
        probe_packets = probe_encoder.get_encoded_packets(0)
        if probe_packets:
            actual_symbol_size = len(probe_packets[0]) - _RQ_ESI_HEADER_SIZE
        else:
            actual_symbol_size = blocksize
        self.blocksize = actual_symbol_size

        remainder = self.filesize % blocksize
        self._padded_size = self.filesize
        if remainder != 0:
            self._padded_size += blocksize - remainder
        self.K = (
            ceil(self._padded_size / self.blocksize)
            if self._padded_size > 0 else 0
        )
        self.source_blocks = _rq_num_source_blocks(self.K)
        self._seq = 0

    def _materialize_padded_data(self) -> bytes:
        if isinstance(self.data, bytes):
            payload = self.data
        else:
            payload = bytes(self.data[:self.filesize])
        padding = self._padded_size - len(payload)
        if padding > 0:
            payload += b'\x00' * padding
        return payload

    def _ensure_encoder(self):
        if self._encoder is None:
            self._encoder = _raptorq.Encoder.with_defaults(
                self._materialize_padded_data(),
                self._requested_blocksize,
            )
        return self._encoder

    def _build_source_symbol_map(self, source_blocks: int
                                    ) -> dict[int, bytes]:
        """Obtain source symbols from the raptorq library.

        The raptorq Rust library may apply sub-block interleaving
        (RFC 6330 Section 5.6) when K is large, so the Nth source symbol
        is NOT necessarily ``data[N*T : (N+1)*T]``.  We must retrieve
        the actual symbol payloads from the library to guarantee
        encode/decode consistency.

        Returns a dict mapping PayloadId → symbol data (without the
        4-byte PayloadId header).
        """
        packets = self._ensure_encoder().get_encoded_packets(0)
        symbol_map: dict[int, bytes] = {}
        for pkt in packets:
            payload_id = struct.unpack('>I', pkt[:_RQ_ESI_HEADER_SIZE])[0]
            if _rq_source_index(payload_id, self.K, source_blocks) is not None:
                symbol_map[payload_id] = pkt[_RQ_ESI_HEADER_SIZE:]
        return symbol_map

    def _iter_source_packets(self, source_blocks: int):
        """Yield ``(payload_id, symbol_data)`` in source-block round-robin order.

        Source symbol data is obtained from the raptorq library (which
        handles sub-block interleaving correctly) rather than slicing
        the original data linearly.
        """
        symbol_map = self._build_source_symbol_map(source_blocks)
        layout = _rq_source_block_layout(self.K, source_blocks)
        max_symbols = max((count for _, count in layout), default=0)
        for local_esi in range(max_symbols):
            for sbn, (_offset, count) in enumerate(layout):
                if local_esi >= count:
                    continue
                payload_id = _rq_payload_id(sbn, local_esi)
                yield payload_id, symbol_map[payload_id]

    # Keep ``binary_qr`` as a read-only alias for symmetry with
    # LTEncoder.
    @property
    def binary_qr(self) -> bool:
        return self.alphanumeric_qr

    def generate_blocks(self, count: int):
        """Generate ``count`` encoded symbols as packed V4 byte strings.

        Systematic packets are emitted first in source-block round-robin
        order; repair packets follow in source-block round-robin order.  The
        upstream ``raptorq`` API returns packets grouped per source block
        (source + repair), so QRStream reorders them to keep early frames
        useful for block-map rendering and evenly distribute repair symbols.

        Yields ``(packed_v4_bytes, payload_id, seq)`` triples.
        """
        source_blocks = self.source_blocks or _rq_num_source_blocks(self.K)
        repair_count = max(0, count - self.K)
        repair_packets: list[bytes] = []
        if repair_count > 0:
            repair_per_block = (
                ceil(repair_count / source_blocks)
                if source_blocks > 0 else 0
            )
            packets = self._ensure_encoder().get_encoded_packets(repair_per_block)
            packet_source_blocks = _rq_source_blocks_from_packets(packets)
            if packet_source_blocks > 0:
                source_blocks = packet_source_blocks
                self.source_blocks = source_blocks

            ordered_packets = _rq_order_packets(packets, self.K, source_blocks)
            for pkt in ordered_packets:
                payload_id = struct.unpack('>I', pkt[:_RQ_ESI_HEADER_SIZE])[0]
                if _rq_source_index(payload_id, self.K, source_blocks) is None:
                    repair_packets.append(pkt)

        self._seq = 0
        emitted = 0
        for payload_id, symbol_data in self._iter_source_packets(source_blocks):
            if emitted >= count:
                return
            seq = self._seq & 0xFFFF
            packed = pack_v4(
                filesize=self.filesize,
                symbol_size=self.blocksize,
                symbol_count=self.K,
                esi=payload_id,
                block_seq=seq,
                data=symbol_data,
                compressed=self.compressed,
                alphanumeric_qr=self.alphanumeric_qr,
                reserved=source_blocks,
            )
            yield packed, payload_id, seq
            self._seq += 1
            emitted += 1

        for pkt in repair_packets[:max(0, count - emitted)]:
            payload_id = struct.unpack('>I', pkt[:_RQ_ESI_HEADER_SIZE])[0]
            symbol_data = pkt[_RQ_ESI_HEADER_SIZE:]
            seq = self._seq & 0xFFFF
            packed = pack_v4(
                filesize=self.filesize,
                symbol_size=self.blocksize,
                symbol_count=self.K,
                esi=payload_id,
                block_seq=seq,
                data=symbol_data,
                compressed=self.compressed,
                alphanumeric_qr=self.alphanumeric_qr,
                reserved=source_blocks,
            )
            yield packed, payload_id, seq
            self._seq += 1


class RaptorQDecoder:
    """Consumes RaptorQ V4 symbols and reconstructs the original data.

    Interface mirrors :class:`qrstream.decoder.LTDecoder`.
    """

    def __init__(self):
        self.K = 0
        self.filesize = 0
        self.blocksize = 0      # = symbol_size
        # V4 reserved; 0 on wire means legacy single-SB.
        self.source_blocks = 1
        self.done = False
        self.compressed = False
        self.protocol_version = None
        self.initialized = False
        self._rq_decoder = None
        self._result: bytes | None = None
        self._fed_count = 0
        # Track which source symbols we can confirm as available.
        # Systematic PayloadIds can be mapped to global source-symbol
        # indices.  Repair PayloadIds do not identify the specific source
        # symbols they help recover; the current upstream Python binding
        # only reveals completion, so remaining symbols are marked when the
        # full object decodes.
        #
        # Keyed by global source-symbol index → True, matching the
        # dict-key-membership protocol that ``compute_block_map_cells``
        # expects (same as ``BlockGraph.eliminated``).
        self.eliminated: dict[int, bool] = {}

    @property
    def progress(self) -> float:
        if not self.initialized or self.K == 0:
            return 0.0
        if self.done:
            return 1.0
        return min(len(self.eliminated) / self.K, 0.99)

    @property
    def num_recovered(self) -> int:
        if self.done:
            return self.K
        return len(self.eliminated)

    def is_done(self) -> bool:
        return self.done

    def consume_block(self, header, data: bytes) -> tuple[bool, bool]:
        """Feed a parsed V4 block (header + data) into the decoder.

        Returns ``(done, compressed)``.
        """
        filesize = header.filesize
        symbol_size = header.blocksize      # V4Header.blocksize property
        symbol_count = header.block_count   # V4Header.block_count property
        payload_id = header.seed            # V4Header.seed property
        compressed = header.compressed
        source_blocks = header.reserved if header.reserved > 0 else 1

        if symbol_size <= 0:
            raise ValueError(f"Invalid symbol_size: {symbol_size}")

        if not self.initialized:
            self.protocol_version = header.version
            self.filesize = filesize
            self.blocksize = symbol_size
            self.K = symbol_count
            self.source_blocks = source_blocks
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
            if source_blocks != self.source_blocks:
                raise ValueError(
                    f"source_blocks mismatch: {source_blocks} != "
                    f"{self.source_blocks}")
            if compressed != self.compressed:
                raise ValueError(
                    f"compressed flag mismatch: {compressed} != "
                    f"{self.compressed}")

        if self.done:
            return True, self.compressed

        # Reconstruct the raptorq packet: 4-byte PayloadId + symbol data.
        # Pad data to symbol_size if short.
        if len(data) < symbol_size:
            data = data + b'\x00' * (symbol_size - len(data))
        elif len(data) > symbol_size:
            data = data[:symbol_size]

        pkt = struct.pack('>I', payload_id) + data
        result = self._rq_decoder.decode(pkt)
        self._fed_count += 1

        # Track systematic source-symbol reception for block map display.
        source_idx = _rq_source_index(payload_id, self.K, self.source_blocks)
        if source_idx is not None:
            self.eliminated[source_idx] = True

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
