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

# Number of LCG iterations to run before sampling the degree, for
# the legacy ``prng_version=0`` code path.
# Sequential seeds (1, 2, 3, ...) produce tiny PRNG outputs that all
# map to the first CDF bucket. Warmup spreads the state across
# [0, M).  See ``splitmix64_mix`` for the qrstream ≥ 0.8 replacement
# that doesn't rely on warmup.
PRNG_WARMUP_ROUNDS = 5


# ── SplitMix64 seed mixer (prng_version=1, qrstream ≥ 0.8) ────────
#
# The legacy ``prng_version=0`` path feeds raw sequential seeds
# 1, 2, 3, … into a 31-bit LCG with 5 warmup rounds.  LCG state is
# linear in its input, so consecutive small seeds produce highly
# correlated first outputs — which under the LT degree distribution
# translates to a peeling graph with too few degree-1 check nodes
# at the front of the block stream.  Empirically this makes K=1827
# stall at ~12% recovery when fed ≤1.5×K sequential blocks (the
# user-facing ``qrs encode --overhead 1.5`` failure this fix
# targets).
#
# ``splitmix64_mix`` replaces the warmup loop with a single-shot
# non-linear scrambler (Steele/Lea 2014, used in JDK
# ``SplittableRandom``).  Benchmarked in dev/bench_mixing.py
# against Knuth / Murmur3 / wyhash variants; SplitMix64 matches or
# beats all of them on LT convergence across K ∈ {328, 1024, 1827,
# 2048, 4096} while costing only 3 multiplies + 3 xor-shifts.
_SPLITMIX_MASK = (1 << 64) - 1
_SPLITMIX_MUL0 = 0x9E3779B97F4A7C15  # golden-ratio odd constant
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2 = 0x94D049BB133111EB


def splitmix64_mix(seed: int) -> int:
    """Map ``seed`` (any non-negative int) to an LCG state in
    [1, PRNG_M − 1].

    Pure integer arithmetic — deterministic, platform independent,
    no floating-point rounding drift.
    """
    x = (seed * _SPLITMIX_MUL0) & _SPLITMIX_MASK
    x ^= (x >> 30)
    x = (x * _SPLITMIX_MUL1) & _SPLITMIX_MASK
    x ^= (x >> 27)
    x = (x * _SPLITMIX_MUL2) & _SPLITMIX_MASK
    x ^= (x >> 31)
    # Map into [1, PRNG_M - 1].  state=0 is a fixed point for the
    # LCG (multiplies stay zero), so we must avoid it.
    return (x % (PRNG_M - 1)) + 1


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
    """Linear congruential PRNG for deterministic block selection.

    ``prng_version`` selects how ``seed`` is mapped to the LCG initial
    state:

    * 0 (legacy, qrstream ≤ 0.7): set ``state = seed`` and run
      :data:`PRNG_WARMUP_ROUNDS` LCG iterations before degree
      sampling.  Kept so decoders can replay videos produced by
      older encoders (flag bit 0x04 cleared).

    * 1 (default, qrstream ≥ 0.8): apply :func:`splitmix64_mix` to
      ``seed`` and use the result directly as the LCG state.  No
      warmup — the mixer's avalanche already decorrelates
      consecutive seeds.
    """

    def __init__(self, K, delta=DEFAULT_DELTA, c=DEFAULT_C,
                 prng_version: int = 1):
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if prng_version not in (0, 1):
            raise ValueError(f"Unsupported prng_version: {prng_version}")
        self.state = None
        self.K = K
        self.prng_version = prng_version
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
        if seed is not None:
            self.state = seed
        blockseed = self.state
        if self.prng_version == 0:
            # Legacy warmup: spread the LCG state across [0, M) so that
            # sequential seeds don't all land in the same CDF bucket.
            # Insufficient to fully decorrelate; see the module-level
            # rationale for ``splitmix64_mix``.
            #
            # TODO(v0.10.0): drop this branch together with the
            # prng_version=0 support. See ``protocol.py`` for the
            # full removal checklist.
            for _ in range(PRNG_WARMUP_ROUNDS):
                self._get_next()
        else:
            # prng_version == 1: single-shot non-linear mix.
            self.state = splitmix64_mix(blockseed)
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

    # ── Gaussian-elimination rescue ───────────────────────────────
    #
    # Belief-propagation (peeling) is greedy and stalls the moment
    # no check node has degree-1. When that happens the graph may
    # still contain enough information to recover every source
    # block — we just need a non-greedy solver. Gauss-Jordan over
    # GF(2) is the textbook choice for LT: each check equation is a
    # row whose left-hand side is a binary indicator vector over
    # source blocks and whose right-hand side is the XOR of payloads.
    #
    # This path is opt-in (off by default) because peeling is
    # effectively free while GE scales as O(K²·M/64) bitwise ops +
    # O(K·blocksize) payload XORs. For K ~ 2000 and blocksize ~ 1 kB
    # it runs in well under a second on numpy, but we only pay that
    # cost when peeling has already failed.
    def try_gaussian_rescue(self) -> bool:
        """Attempt to finish decoding via Gauss-Jordan over GF(2).

        Returns True when every source block has been recovered
        (either already by peeling or newly by this call). The
        caller is responsible for deciding when to invoke this —
        typically after peeling has processed all available blocks
        and :attr:`eliminated` is still short of
        :attr:`num_blocks`.
        """
        K = self.num_blocks
        if len(self.eliminated) >= K:
            return True

        # Collect unique CheckNodes still in the graph. Each node is
        # referenced once per remaining source index, so pass through
        # a set keyed by id().
        unique_checks: dict[int, CheckNode] = {}
        for chk_list in self.checks.values():
            for chk in chk_list:
                if chk.src_nodes:
                    unique_checks[id(chk)] = chk
        checks = list(unique_checks.values())
        if not checks:
            return False

        # Build augmented system for the still-unknown source blocks.
        # Known blocks are XOR'd into each row's constant side so we
        # only solve over unknowns.
        unknown_indices = [i for i in range(K) if i not in self.eliminated]
        if not unknown_indices:
            return True
        col_of = {src: col for col, src in enumerate(unknown_indices)}
        n_cols = len(unknown_indices)

        # Coefficients as bit-packed uint8 matrix: one row per check,
        # n_cols bits per row. Numpy's ``packbits`` / ``unpackbits``
        # make XORing two rows a single ``np.bitwise_xor`` on O(K/8)
        # bytes, which is how we keep GE affordable at K≈2000.
        row_bytes = (n_cols + 7) // 8
        coef = np.zeros((len(checks), row_bytes), dtype=np.uint8)

        # RHS payload matrix (uint8). Normalize all rows to the same
        # length by picking the widest check payload seen.
        blocksize = max(len(chk.check) for chk in checks)
        rhs = np.zeros((len(checks), blocksize), dtype=np.uint8)

        for row_idx, chk in enumerate(checks):
            # Pack column bits for this row.
            col_bits = np.zeros(n_cols, dtype=np.uint8)
            for src in chk.src_nodes:
                col = col_of.get(src)
                if col is not None:
                    col_bits[col] = 1
            coef[row_idx] = np.packbits(col_bits, bitorder='big')[:row_bytes]

            data = chk.check
            if len(data) < blocksize:
                rhs[row_idx, :len(data)] = data
            else:
                rhs[row_idx] = data[:blocksize]

        # Gauss-Jordan elimination. Pivot column-by-column, partial
        # pivoting by the first row below the pivot with a 1 in the
        # current column. Operate on both coef (bit-packed) and rhs
        # (uint8 payload) so the invariant "coef @ unknowns = rhs"
        # holds at every step.
        n_rows = coef.shape[0]
        row_pivot_of_col: dict[int, int] = {}
        row = 0
        for col in range(n_cols):
            # Locate a pivot row at or below ``row`` with a 1 in ``col``.
            byte_idx = col >> 3
            bit_mask = np.uint8(1 << (7 - (col & 7)))
            pivot = -1
            for r in range(row, n_rows):
                if coef[r, byte_idx] & bit_mask:
                    pivot = r
                    break
            if pivot == -1:
                continue
            if pivot != row:
                coef[[row, pivot]] = coef[[pivot, row]]
                rhs[[row, pivot]] = rhs[[pivot, row]]

            # Eliminate the column from every other row with a 1 there.
            pivot_coef = coef[row]
            pivot_rhs = rhs[row]
            for r in range(n_rows):
                if r == row:
                    continue
                if coef[r, byte_idx] & bit_mask:
                    np.bitwise_xor(coef[r], pivot_coef, out=coef[r])
                    np.bitwise_xor(rhs[r], pivot_rhs, out=rhs[r])

            row_pivot_of_col[col] = row
            row += 1
            if row == n_rows:
                break

        # Every unknown column must have a pivot for the system to
        # be uniquely solvable.
        if len(row_pivot_of_col) < n_cols:
            return False

        for col, src in enumerate(unknown_indices):
            r = row_pivot_of_col[col]
            self.eliminated[src] = rhs[r].copy()

        return len(self.eliminated) >= K
