"""
LT Fountain Code Decoder: block consumption → file reconstruction.

The :class:`LTDecoder` consumes LT-coded blocks (V2/V3 protocol with
CRC32 validation) and reconstructs the original file via belief-
propagation peeling with optional GF(2) Gauss-Jordan rescue.

Split from ``decoder.py`` for readability; all public symbols are
re-exported by ``decoder.py`` so external imports are unchanged.
"""

import io
import zlib
from math import ceil

import numpy as np
from tqdm import tqdm

from .lt_codec import PRNG, BlockGraph, DEFAULT_C, DEFAULT_DELTA
from .protocol import unpack


_PROGRESS_BAR_THRESHOLD = 512


class LTDecoder:
    """Consumes LT fountain-coded blocks and reconstructs the original data.

    Accepts V2/V3 blocks with CRC validation; corrupt blocks are silently
    discarded.
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
        self.prng_version = None  # set from the first block's header
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
            self.prng = PRNG(self.K, delta=self.delta, c=self.c,
                             prng_version=self.prng_version)
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
        """Opt-in GF(2) Gauss-Jordan pass over the current check-node
        graph.

        Call this *after* all available blocks have been fed and
        :meth:`is_done` still returns False.  When the surviving
        check equations together span the missing source blocks,
        this recovers the whole file without needing any more
        encoded frames.  Safe no-op when peeling already converged.

        Returns True iff every source block is now recovered.
        """
        if not self.initialized or self.block_graph is None:
            return False
        if self.done:
            return True
        recovered = self.block_graph.try_gaussian_rescue()
        if recovered:
            self.done = True
        return recovered

    def decode_bytes(self, block_bytes: bytes, skip_crc: bool = False) -> tuple[bool, bool]:
        """Decode a raw protocol block from bytes.

        Validates CRC32 — raises ValueError on corrupt data,
        unless skip_crc=True (for pre-validated blocks).
        """
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
        written = 0
        pbar = None
        if show_progress:
            pbar = tqdm(total=self.K, desc="Write",
                        unit="blk", dynamic_ncols=True,
                        mininterval=0.1)

        try:
            with open(output_path, 'wb') as f:
                if self.compressed:
                    decompressor = zlib.decompressobj()
                    try:
                        for chunk in self._iter_recovered_chunks():
                            data = decompressor.decompress(chunk)
                            if data:
                                f.write(data)
                                written += len(data)
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(bytes=written)
                        tail = decompressor.flush()
                    except zlib.error as e:
                        raise RuntimeError(
                            f"Decompression failed: {e}. Decoded payload may be corrupted.") from e
                    if tail:
                        f.write(tail)
                        written += len(tail)
                        if pbar is not None:
                            pbar.set_postfix(bytes=written)
                else:
                    for chunk in self._iter_recovered_chunks():
                        f.write(chunk)
                        written += len(chunk)
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(bytes=written)
        finally:
            if pbar is not None:
                pbar.close()
        return written


# ── Block-level decode helpers ───────────────────────────────────


def _decode_into_decoder(blocks, verbose=False) -> LTDecoder | None:
    if not blocks:
        print("Error: No blocks to decode")
        return None

    decoder = LTDecoder()
    show_progress = verbose or len(blocks) >= _PROGRESS_BAR_THRESHOLD
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(blocks), desc="LT decode",
                    unit="blk", dynamic_ncols=True,
                    mininterval=0.1)

    try:
        for i, block_bytes in enumerate(blocks):
            try:
                done, compressed = decoder.decode_bytes(block_bytes)
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix_str(
                            f"got={decoder.num_recovered}/{decoder.K}")
                if done:
                    if verbose:
                        print(f"  Decoded after {i + 1}/{len(blocks)} blocks "
                              f"(filesize={decoder.filesize}, K={decoder.K}, "
                              f"compressed={compressed}, v={decoder.protocol_version})")
                    return decoder
            except ValueError as e:
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix_str(
                            f"got={decoder.num_recovered}/{decoder.K}")
                if verbose:
                    print(f"  Block {i} error, skipping: {e}")
            except Exception as e:
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix_str(
                            f"got={decoder.num_recovered}/{decoder.K}")
                if verbose:
                    print(f"  Block {i} error: {e}")
    finally:
        if pbar is not None:
            pbar.close()

    # Peeling (belief-propagation) exhausted all blocks without
    # converging. Attempt a GF(2) Gauss-Jordan rescue pass over the
    # accumulated check-node graph.
    #
    # TODO(v0.10.0): the main reason peeling fails on a post-0.8
    # stream is legacy prng_version=0 encoding. Once v0 support is
    # dropped (see ``protocol.py``), revisit whether the rescue is
    # still worth carrying.
    if decoder.initialized and not decoder.done:
        rescued = decoder.try_gaussian_rescue()
        if rescued:
            if verbose:
                print(f"  GE rescue recovered all "
                      f"{decoder.num_recovered}/{decoder.K} blocks "
                      f"after peeling stalled.")
            else:
                print(f"  GE rescue recovered "
                      f"{decoder.num_recovered}/{decoder.K} source blocks.")
            return decoder
        elif verbose:
            print(f"  GE rescue attempted, still "
                  f"{decoder.num_recovered}/{decoder.K} recovered.")

    n_recovered = decoder.num_recovered
    k = decoder.K if decoder.K else '?'
    print(f"\nDecoding incomplete: {n_recovered}/{k} source blocks recovered "
          f"from {len(blocks)} encoded blocks.")
    print("Try recording the QR stream longer to capture more unique frames.")
    return None


def decode_blocks(blocks, verbose=False) -> bytes | None:
    """Feed blocks into LT decoder to reconstruct the file."""
    decoder = _decode_into_decoder(blocks, verbose=verbose)
    if decoder is None:
        return None
    try:
        return decoder.bytes_dump()
    except RuntimeError as e:
        print(f"Error: {e}")
        return None


def decode_blocks_to_file(blocks, output_path: str, verbose=False) -> int | None:
    """Decode blocks and write the result directly to a file."""
    decoder = _decode_into_decoder(blocks, verbose=verbose)
    if decoder is None:
        return None
    try:
        show_progress = verbose or (decoder.K >= _PROGRESS_BAR_THRESHOLD)
        return decoder.bytes_dump_to_file(output_path, show_progress=show_progress)
    except RuntimeError as e:
        print(f"Error: {e}")
        return None
