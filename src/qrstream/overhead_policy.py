"""Shared fountain-code overhead policy constants."""

# LT (legacy) needs a larger convergence floor than RaptorQ. The empirical
# worst case for the current SplitMix64 LT path is K=328 at 1.19x.
MIN_OVERHEAD_LT = 1.20
RECOMMENDED_OVERHEAD_LT = 1.50
DEFAULT_OVERHEAD_LT = 2.00

# RaptorQ (RFC 6330) is near-optimal. Local validation shows 1.05x is a
# practical lower bound for qrstream's default RaptorQ path.
MIN_OVERHEAD_RQ = 1.05
RECOMMENDED_OVERHEAD_RQ = 1.10
DEFAULT_OVERHEAD_RQ = 1.20
