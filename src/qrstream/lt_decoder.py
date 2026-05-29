"""Platform-neutral LT decoder for QRStream V3 blocks."""

import io
import zlib
from math import ceil

import numpy as np

from .lt_codec import PRNG, BlockGraph, DEFAULT_C, DEFAULT_DELTA
from .protocol import unpack


class LTDecoder:
    """Consumes LT fountain-coded V3 blocks and reconstructs the original data.

    Accepts V3 blocks with CRC validation; corrupt blocks are silently
    discarded. For V4 (RaptorQ) blocks, use
    :class:`qrstream.raptorq_codec.RaptorQDecoder` instead.
    """

    def __init__(self, c: float = DEFAULT_C, delta: float = DEFAULT_DELTA):
        self.c = c
        self.delta = delta
        self.K = 0
        self.filesize = 0
        self.blocksize = 0
        self.done = False
        self.compressed = False
        self.protocol_version = None
        self.prng_version = None
        self.block_graph = None
        self.prng = None
        self.initialized = False

    @property
    def progress(self) -> float:
        """Return decoding progress as a fraction [0.0, 1.0]."""
        if not self.initialized or self.K == 0:
            return 0.0
        return min(len(self.block_graph.eliminated) / self.K, 1.0)

    @property
    def num_recovered(self) -> int:
        if self.block_graph is None:
            return 0
        return len(self.block_graph.eliminated)

    def is_done(self) -> bool:
        return self.done

    def consume_block(self, header, data: bytes) -> tuple[bool, bool]:
        """Feed a parsed block (header + data bytes) into the decoder.

        Returns (done, compressed).
        """
        filesize = header.filesize
        blocksize = header.blocksize
        block_count = header.block_count
        seed = header.seed
        compressed = header.compressed

        if blocksize <= 0:
            raise ValueError(f"Invalid blocksize: {blocksize}")

        expected_block_count = ceil(filesize / blocksize) if filesize > 0 else 0
        if block_count != expected_block_count:
            raise ValueError(
                f"block_count mismatch: header={block_count}, expected={expected_block_count}")

        if not self.initialized:
            self.protocol_version = header.version
            self.prng_version = header.prng_version
            self.filesize = filesize
            self.blocksize = blocksize
            self.K = block_count
            self.compressed = compressed
            self.block_graph = BlockGraph(self.K)
            self.prng = PRNG(self.K, delta=self.delta, c=self.c)
            self.initialized = True
        else:
            if header.version != self.protocol_version:
                raise ValueError(
                    f"version mismatch: {header.version} != {self.protocol_version}")
            if filesize != self.filesize:
                raise ValueError(f"filesize mismatch: {filesize} != {self.filesize}")
            if blocksize != self.blocksize:
                raise ValueError(f"blocksize mismatch: {blocksize} != {self.blocksize}")
            if block_count != self.K:
                raise ValueError(f"block_count mismatch: {block_count} != {self.K}")
            if compressed != self.compressed:
                raise ValueError(
                    f"compressed flag mismatch: {compressed} != {self.compressed}")
            if header.prng_version != self.prng_version:
                raise ValueError(
                    f"prng_version mismatch: {header.prng_version} "
                    f"!= {self.prng_version}")

        _, _, src_blocks = self.prng.get_src_blocks(seed=seed)

        if len(data) < self.blocksize:
            data = data + b'\x00' * (self.blocksize - len(data))
        elif len(data) > self.blocksize:
            data = data[:self.blocksize]

        self.done = self.block_graph.add_block(src_blocks, data)
        return self.done, self.compressed

    def try_gaussian_rescue(self) -> bool:
        """Attempt GF(2) Gauss-Jordan recovery after peeling stalls."""
        if not self.initialized or self.block_graph is None:
            return False
        if self.done:
            return True
        recovered = self.block_graph.try_gaussian_rescue()
        if recovered:
            self.done = True
        return recovered

    def decode_bytes(self, block_bytes: bytes, skip_crc: bool = False) -> tuple[bool, bool]:
        """Decode a raw protocol block from bytes."""
        header, data = unpack(block_bytes, skip_crc=skip_crc)
        return self.consume_block(header, data)

    def _iter_recovered_chunks(self):
        for ix in range(self.K):
            block = self.block_graph.eliminated.get(ix)
            if block is None:
                raise RuntimeError(
                    f"Missing block {ix}/{self.K} — decoding incomplete")
            if isinstance(block, np.ndarray):
                block = block.tobytes()
            if ix < self.K - 1 or self.filesize % self.blocksize == 0:
                yield block
            else:
                yield block[:self.filesize % self.blocksize]

    def bytes_dump(self) -> bytes:
        """Reconstruct the original file data from recovered blocks."""
        buf = io.BytesIO()
        for chunk in self._iter_recovered_chunks():
            buf.write(chunk)
        raw_data = buf.getvalue()
        if self.compressed:
            try:
                return zlib.decompress(raw_data)
            except zlib.error as e:
                raise RuntimeError(
                    f"Decompression failed: {e}. Decoded payload may be corrupted.") from e
        return raw_data

    def bytes_dump_to_file(self, output_path: str, show_progress: bool = False) -> int:
        """Write the reconstructed output directly to a file."""
        del show_progress
        written = 0
        with open(output_path, 'wb') as f:
            if self.compressed:
                decompressor = zlib.decompressobj()
                try:
                    for chunk in self._iter_recovered_chunks():
                        data = decompressor.decompress(chunk)
                        if data:
                            f.write(data)
                            written += len(data)
                    tail = decompressor.flush()
                except zlib.error as e:
                    raise RuntimeError(
                        f"Decompression failed: {e}. Decoded payload may be corrupted.") from e
                if tail:
                    f.write(tail)
                    written += len(tail)
            else:
                for chunk in self._iter_recovered_chunks():
                    f.write(chunk)
                    written += len(chunk)
        return written
