"""Tests for decoder-specific behavior."""

from math import ceil
import zlib

import numpy as np
import pytest

from qrstream.decoder import (
    LTDecoder,
    _analyze_probe_window,
    _build_probe_ranges,
    decode_blocks_to_file,
)
from qrstream.encoder import LTEncoder
from qrstream.protocol import pack_v3


class TestDecoderValidation:
    def test_rejects_mismatched_headers_after_initialization(self):
        decoder = LTDecoder()

        first = pack_v3(
            filesize=128,
            blocksize=64,
            block_count=2,
            seed=1,
            block_seq=0,
            data=b'A' * 64,
        )
        decoder.decode_bytes(first)

        mismatched = pack_v3(
            filesize=192,
            blocksize=64,
            block_count=3,
            seed=2,
            block_seq=1,
            data=b'B' * 64,
        )

        with pytest.raises(ValueError, match="filesize mismatch"):
            decoder.decode_bytes(mismatched)


class TestProbeStrategy:
    def test_build_probe_ranges_spreads_three_windows_across_middle(self):
        ranges = _build_probe_ranges(1000, window_size=120, gap_ratio=0.15)

        assert ranges == [(290, 409), (440, 559), (589, 708)]
        assert sum(end - start + 1 for start, end in ranges) == 360

    def test_build_probe_ranges_uses_single_range_for_short_video(self):
        assert _build_probe_ranges(20, window_size=120, gap_ratio=0.15) == [(0, 19)]

    def test_probe_window_requires_multiple_seeds_for_repeat_estimate(self):
        stats = _analyze_probe_window([
            (100, b'data', 7),
            (101, b'data', 7),
            (102, b'data', 7),
        ])

        assert stats['distinct_seed_count'] == 1
        assert stats['sample_rate'] is None

    def test_probe_window_computes_sample_rate_for_multiple_seeds(self):
        stats = _analyze_probe_window([
            (100, b'data', 7),
            (101, b'data', 7),
            (102, b'data', 8),
            (103, b'data', 8),
        ])

        assert stats['distinct_seed_count'] == 2
        assert stats['avg_repeat'] == 2.0
        assert stats['sample_rate'] is not None


class TestDecoderOutputPaths:
    def test_decode_blocks_to_file_writes_uncompressed_output(self, tmp_path):
        data = b"streamed-output" * 50
        blocksize = 64
        encoder = LTEncoder(data, blocksize)
        blocks = [packed for packed, _, _ in encoder.generate_blocks(36)]

        output_path = tmp_path / "decoded.bin"
        written = decode_blocks_to_file(blocks, str(output_path))

        assert written == len(data)
        assert output_path.read_bytes() == data

    def test_bytes_dump_to_file_writes_compressed_output(self, tmp_path):
        data = b'A' * 1024
        payload = zlib.compress(data)
        blocksize = 64
        K = ceil(len(payload) / blocksize)
        encoder = LTEncoder(payload, blocksize, compressed=True)
        decoder = LTDecoder()

        for packed, _, _ in encoder.generate_blocks(K * 3):
            done, _ = decoder.decode_bytes(packed)
            if done:
                break

        output_path = tmp_path / "decoded.txt"
        written = decoder.bytes_dump_to_file(str(output_path))

        assert written == len(data)
        assert output_path.read_bytes() == data

    def test_bytes_dump_reports_decompression_errors_cleanly(self):
        data = b'B' * 1024
        payload = zlib.compress(data)
        blocksize = 64
        K = ceil(len(payload) / blocksize)
        encoder = LTEncoder(payload, blocksize, compressed=True)
        decoder = LTDecoder()

        for packed, _, _ in encoder.generate_blocks(K * 3):
            done, _ = decoder.decode_bytes(packed)
            if done:
                break

        decoder.block_graph.eliminated[0] = np.frombuffer(
            b'corrupted!' + payload[:blocksize - 10],
            dtype=np.uint8,
        ).copy()

        with pytest.raises(RuntimeError, match="Decompression failed"):
            decoder.bytes_dump()
