"""
LT Fountain Code primitives: PRNG, Robust Soliton Distribution, BlockGraph.

Ported from the original decode.py with key improvements:
- Data stored as numpy uint8 arrays internally for zero-copy XOR
- xor_bytes() uses numpy for vectorized XOR (10-50x faster)
- BlockGraph uses in-place numpy XOR to eliminate allocation overhead
"""

import functools
from math import log, floor, sqrt
from collections import defaultdict
import bisect

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


@functools.lru_cache(maxsize=64)
def gen_rsd_cdf(k, delta, c):
    mu = gen_mu(k, delta, c)
    cdf = tuple(sum(mu[:d + 1]) for d in range(k))
    return cdf


# ── PRNG ──────────────────────────────────────────────────────────

class PRNG:
    """Linear congruential PRNG for deterministic block selection."""

    def __init__(self, K, delta=DEFAULT_DELTA, c=DEFAULT_C):
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        self.state = None
        self.K = K
        self.cdf = gen_rsd_cdf(K, delta, c)

    def _get_next(self):
        self.state = PRNG_A * self.state % PRNG_M
        return self.state

    def _sample_d(self):
        p = self._get_next() / PRNG_MAX_RAND
        # Use binary search instead of linear search for CDF sampling
        # Clamp to K so degree never exceeds the number of source blocks
        ix = bisect.bisect_right(self.cdf, p)
        return min(ix + 1, self.K)

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


def _to_np(data) -> np.ndarray:
    """Convert data to numpy uint8 array if it isn't already."""
    if isinstance(data, np.ndarray):
        return data
    return np.frombuffer(data, dtype=np.uint8).copy()


def _xor_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR two numpy arrays, padding shorter one if needed. Returns new array."""
    if len(a) == len(b):
        return np.bitwise_xor(a, b)
    maxlen = max(len(a), len(b))
    if len(a) < maxlen:
        a = np.pad(a, (0, maxlen - len(a)))
    if len(b) < maxlen:
        b = np.pad(b, (0, maxlen - len(b)))
    return np.bitwise_xor(a, b)


def _xor_np_inplace(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """In-place XOR: a ^= b. Arrays must be same length. Returns a."""
    np.bitwise_xor(a, b, out=a)
    return a


# ── Block Graph (numpy-optimized) ────────────────────────────────

class CheckNode:
    """A check node in the bipartite LT graph."""
    __slots__ = ('src_nodes', 'check')

    def __init__(self, src_nodes, check: np.ndarray):
        self.check = check
        self.src_nodes = src_nodes


class BlockGraph:
    """Bipartite graph for LT decoding using belief-propagation (peeling).

    Stores block data as numpy uint8 arrays for zero-copy in-place XOR.
    """

    def __init__(self, num_blocks):
        self.checks = defaultdict(list)
        self.num_blocks = num_blocks
        self.eliminated = {}  # block_index -> np.ndarray (uint8)

    def add_block(self, nodes, data):
        """Add an encoded block. Returns True when all source blocks are recovered.

        Args:
            nodes: set of source block indices
            data: block data as bytes or numpy array
        """
        data = _to_np(data)
        nodes = set(nodes)

        if len(nodes) == 1:
            to_eliminate = list(self.eliminate(next(iter(nodes)), data.copy()))
            while to_eliminate:
                other, check = to_eliminate.pop()
                to_eliminate.extend(self.eliminate(other, check))
        else:
            for node in list(nodes):
                if node in self.eliminated:
                    nodes.remove(node)
                    data = _xor_np(data, self.eliminated[node])
            if len(nodes) == 1:
                return self.add_block(nodes, data)
            else:
                check = CheckNode(nodes, data.copy())
                for node in nodes:
                    self.checks[node].append(check)
        return len(self.eliminated) >= self.num_blocks

    def eliminate(self, node, data: np.ndarray):
        """Eliminate a source block node, propagating through the graph."""
        self.eliminated[node] = data
        others = self.checks[node]
        del self.checks[node]
        for check in others:
            # In-place XOR avoids allocating a new array each time
            if len(check.check) == len(data):
                _xor_np_inplace(check.check, data)
            else:
                check.check = _xor_np(check.check, data)
            check.src_nodes.remove(node)
            if len(check.src_nodes) == 1:
                yield (next(iter(check.src_nodes)), check.check.copy())
