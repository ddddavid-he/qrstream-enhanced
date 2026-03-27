"""
LT Fountain Code primitives: PRNG, Robust Soliton Distribution, BlockGraph.

Ported from the original decode.py with key improvement:
- Data stored as `bytes` instead of Python big-int
- xor_bytes() uses numpy for vectorized XOR (10-50x faster)
"""

from math import log, floor, sqrt
from collections import defaultdict

import numpy as np

# ── Constants ──────────────────────────────────────────────────────

DEFAULT_C = 0.1
DEFAULT_DELTA = 0.5

PRNG_A = 16807
PRNG_M = (1 << 31) - 1
PRNG_MAX_RAND = PRNG_M - 1

# Number of LCG iterations to run before sampling the degree.
# Sequential seeds (1, 2, 3, ...) produce tiny PRNG outputs that all map
# to the first CDF bucket. Warmup spreads the state across [0, M).
PRNG_WARMUP_ROUNDS = 5


# ── Robust Soliton Distribution ───────────────────────────────────

def gen_tau(s, k, delta):
    pivot = floor(k / s)
    return ([s / k * 1 / d for d in range(1, pivot)]
            + [s / k * log(s / delta)]
            + [0 for d in range(pivot, k)])


def gen_rho(k):
    return [1 / k] + [1 / (d * (d - 1)) for d in range(2, k + 1)]


def gen_mu(k, delta, c):
    S = c * log(k / delta) * sqrt(k)
    tau = gen_tau(S, k, delta)
    rho = gen_rho(k)
    normalizer = sum(rho) + sum(tau)
    return [(rho[d] + tau[d]) / normalizer for d in range(k)]


def gen_rsd_cdf(k, delta, c):
    mu = gen_mu(k, delta, c)
    return [sum(mu[:d + 1]) for d in range(k)]


# ── PRNG ──────────────────────────────────────────────────────────

class PRNG:
    """Linear congruential PRNG for deterministic block selection."""

    def __init__(self, K, delta=DEFAULT_DELTA, c=DEFAULT_C):
        self.state = None
        self.K = K
        self.cdf = gen_rsd_cdf(K, delta, c)

    def _get_next(self):
        self.state = PRNG_A * self.state % PRNG_M
        return self.state

    def _sample_d(self):
        p = self._get_next() / PRNG_MAX_RAND
        for ix, v in enumerate(self.cdf):
            if v > p:
                return ix + 1
        return ix + 1

    def set_seed(self, seed):
        self.state = seed

    def get_src_blocks(self, seed=None):
        """Return (blockseed, degree, src_block_indices) for a given seed."""
        if seed:
            self.state = seed
        blockseed = self.state
        # Warmup: spread the LCG state across the full [0, M) range so
        # that sequential seeds don't all land in the same CDF bucket.
        for _ in range(PRNG_WARMUP_ROUNDS):
            self._get_next()
        d = self._sample_d()
        have = 0
        nums = set()
        while have < d:
            num = self._get_next() % self.K
            if num not in nums:
                nums.add(num)
                have += 1
        return blockseed, d, nums


# ── Bytes XOR ─────────────────────────────────────────────────────

def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings using numpy vectorization.

    If lengths differ, the result is the length of the longer input
    (shorter one is zero-padded on the right).
    """
    la, lb = len(a), len(b)
    maxlen = max(la, lb)
    arr_a = np.frombuffer(a, dtype=np.uint8)
    arr_b = np.frombuffer(b, dtype=np.uint8)
    if la < maxlen:
        arr_a = np.pad(arr_a, (0, maxlen - la))
    if lb < maxlen:
        arr_b = np.pad(arr_b, (0, maxlen - lb))
    return bytes(np.bitwise_xor(arr_a, arr_b))


# ── Block Graph (bytes-based) ────────────────────────────────────

class CheckNode:
    """A check node in the bipartite LT graph."""
    __slots__ = ('src_nodes', 'check')

    def __init__(self, src_nodes, check: bytes):
        self.check = check
        self.src_nodes = src_nodes


class BlockGraph:
    """Bipartite graph for LT decoding using belief-propagation (peeling).

    Stores block data as `bytes` and uses numpy-vectorized XOR.
    """

    def __init__(self, num_blocks):
        self.checks = defaultdict(list)
        self.num_blocks = num_blocks
        self.eliminated = {}  # block_index -> bytes

    def add_block(self, nodes, data: bytes):
        """Add an encoded block. Returns True when all source blocks are recovered."""
        if len(nodes) == 1:
            to_eliminate = list(self.eliminate(next(iter(nodes)), data))
            while to_eliminate:
                other, check = to_eliminate.pop()
                to_eliminate.extend(self.eliminate(other, check))
        else:
            for node in list(nodes):
                if node in self.eliminated:
                    nodes.remove(node)
                    data = xor_bytes(data, self.eliminated[node])
            if len(nodes) == 1:
                return self.add_block(nodes, data)
            else:
                check = CheckNode(nodes, data)
                for node in nodes:
                    self.checks[node].append(check)
        return len(self.eliminated) >= self.num_blocks

    def eliminate(self, node, data: bytes):
        """Eliminate a source block node, propagating through the graph."""
        self.eliminated[node] = data
        others = self.checks[node]
        del self.checks[node]
        for check in others:
            check.check = xor_bytes(check.check, data)
            check.src_nodes.remove(node)
            if len(check.src_nodes) == 1:
                yield (next(iter(check.src_nodes)), check.check)
