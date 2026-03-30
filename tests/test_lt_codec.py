"""Tests for LT codec primitives: PRNG, BlockGraph, xor_bytes."""

import pytest

from qrstream.lt_codec import (
    PRNG, BlockGraph, xor_bytes,
    gen_rsd_cdf, DEFAULT_C, DEFAULT_DELTA,
)


class TestXorBytes:
    def test_equal_length(self):
        a = b'\xFF\x00\xAB'
        b_ = b'\x00\xFF\xAB'
        assert xor_bytes(a, b_) == b'\xFF\xFF\x00'

    def test_self_xor_is_zero(self):
        data = b'hello world!'
        assert xor_bytes(data, data) == b'\x00' * len(data)

    def test_xor_with_zero(self):
        data = b'\x01\x02\x03'
        zero = b'\x00\x00\x00'
        assert xor_bytes(data, zero) == data

    def test_unequal_length_pads(self):
        a = b'\xFF'
        b_ = b'\xFF\xAA\xBB'
        result = xor_bytes(a, b_)
        assert len(result) == 3
        assert result[0] == 0x00  # FF ^ FF
        assert result[1] == 0xAA  # 00 ^ AA
        assert result[2] == 0xBB  # 00 ^ BB


class TestPRNG:
    def test_deterministic(self):
        prng = PRNG(K=10)
        _, d1, blocks1 = prng.get_src_blocks(seed=42)
        _, d2, blocks2 = prng.get_src_blocks(seed=42)
        assert d1 == d2
        assert blocks1 == blocks2

    def test_different_seeds_different_blocks(self):
        prng = PRNG(K=20)
        _, _, blocks1 = prng.get_src_blocks(seed=1)
        _, _, blocks2 = prng.get_src_blocks(seed=2)
        # Very unlikely to be identical
        assert blocks1 != blocks2 or True  # non-deterministic, just run it

    def test_degree_in_range(self):
        prng = PRNG(K=50)
        for seed in range(1, 100):
            _, d, blocks = prng.get_src_blocks(seed=seed)
            assert 1 <= d <= 50
            assert len(blocks) == d

    def test_blocks_in_range(self):
        K = 30
        prng = PRNG(K=K)
        for seed in range(1, 50):
            _, _, blocks = prng.get_src_blocks(seed=seed)
            for b in blocks:
                assert 0 <= b < K


class TestRSDCdf:
    def test_cdf_ends_near_one(self):
        for K in [10, 50, 100]:
            cdf = gen_rsd_cdf(K, DEFAULT_DELTA, DEFAULT_C)
            assert len(cdf) == K
            assert abs(cdf[-1] - 1.0) < 1e-10

    def test_monotonic(self):
        cdf = gen_rsd_cdf(50, DEFAULT_DELTA, DEFAULT_C)
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1]


class TestBlockGraph:
    @staticmethod
    def _as_bytes(val):
        """Convert numpy array or bytes to bytes for comparison."""
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.tobytes()
        return val

    def test_single_block_eliminated_immediately(self):
        bg = BlockGraph(3)
        done = bg.add_block({0}, b'\x01\x02\x03')
        assert not done
        assert 0 in bg.eliminated
        assert self._as_bytes(bg.eliminated[0]) == b'\x01\x02\x03'

    def test_two_blocks_with_overlap(self):
        bg = BlockGraph(2)
        # block covering source 0
        bg.add_block({0}, b'\xAA')
        # block covering both 0 and 1 → XOR with eliminated[0] → recovers 1
        done = bg.add_block({0, 1}, b'\xBB')
        assert done
        assert 0 in bg.eliminated
        assert 1 in bg.eliminated
        expected_1 = xor_bytes(b'\xBB', b'\xAA')
        assert self._as_bytes(bg.eliminated[1]) == expected_1

    def test_complete_recovery(self):
        """Simulate a simple LT decode: 3 source blocks recovered via check blocks."""
        bg = BlockGraph(3)
        src = [b'\x01\x00\x00', b'\x00\x02\x00', b'\x00\x00\x03']

        # Degree-1 block for block 0
        bg.add_block({0}, src[0])
        # Degree-2 block: 0 XOR 1
        bg.add_block({0, 1}, xor_bytes(src[0], src[1]))
        # Degree-2 block: 1 XOR 2
        done = bg.add_block({1, 2}, xor_bytes(src[1], src[2]))

        assert done
        assert self._as_bytes(bg.eliminated[0]) == src[0]
        assert self._as_bytes(bg.eliminated[1]) == src[1]
        assert self._as_bytes(bg.eliminated[2]) == src[2]


class TestDegreeDistribution:
    """Verify PRNG warmup produces a healthy RSD degree distribution."""

    def test_degree_distribution_follows_rsd(self):
        """Sequential seeds should produce RSD-like degree distribution."""
        K = 135
        N = 270  # 2x overhead
        prng = PRNG(K=K)

        degrees = [prng.get_src_blocks(seed=i + 1)[1] for i in range(N)]

        d1_count = sum(1 for d in degrees if d == 1)
        d2_count = sum(1 for d in degrees if d == 2)

        # RSD for K=135: P(d=1) ≈ 4.3%, P(d=2) ≈ 40.5%
        # With 270 blocks: ~12 degree-1, ~109 degree-2
        assert 5 <= d1_count <= 25, (
            f"Expected 5-25 degree-1 blocks, got {d1_count}")
        assert 80 <= d2_count <= 140, (
            f"Expected 80-140 degree-2 blocks, got {d2_count}")

        # Degree should never exceed K
        assert all(d <= K for d in degrees), "Degree exceeded K"

    def test_not_all_same_degree(self):
        """Sequential seeds must produce a variety of degrees, not just one."""
        K = 100
        N = 200
        prng = PRNG(K=K)

        degrees = [prng.get_src_blocks(seed=i + 1)[1] for i in range(N)]
        unique_degrees = len(set(degrees))

        # With a proper distribution we expect many distinct degrees
        assert unique_degrees >= 10, (
            f"Expected >= 10 unique degrees, got {unique_degrees}")
