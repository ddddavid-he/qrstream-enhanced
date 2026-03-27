"""
V1/V2 protocol header serialization and deserialization.

V1 (legacy, 13 bytes):
    Offset  Size  Field
     0      1     magic_byte   (bit0 = compressed)
     1      4     filesize     uint32 BE
     5      4     blocksize    uint32 BE
     9      4     seed         uint32 BE
    13      ...   data         (blocksize bytes, big-int encoded)

V2 (new, 20 bytes):
    Offset  Size  Field
     0      1     version      0x02
     1      1     flags        bit0=compressed(zlib), bit1-7=reserved
     2      4     filesize     uint32 BE (compressed size if compressed)
     6      2     blocksize    uint16 BE
     8      2     block_count  uint16 BE  K = ceil(filesize / blocksize)
    10      4     seed         uint32 BE  PRNG seed
    14      2     block_seq    uint16 BE  monotonic sequence number
    16      4     crc32        CRC32 of header[0:16] + data
    20      ...   data         blocksize bytes
"""

import struct
import zlib
from dataclasses import dataclass
from math import ceil

# ── Header sizes ──────────────────────────────────────────────────

V1_HEADER_SIZE = 13
V2_HEADER_SIZE = 20

# ── QR capacity table (byte mode, per version and ECC level) ─────
# Source: ISO/IEC 18004 — byte mode capacity for versions 1-40.
# Key: (version, ec_level) → max bytes.  ec_level: 0=L, 1=M, 2=Q, 3=H
_QR_CAPACITY_BY_VERSION = {
    (1, 0): 17,   (1, 1): 14,   (1, 2): 11,   (1, 3): 7,
    (2, 0): 34,   (2, 1): 26,   (2, 2): 20,   (2, 3): 14,
    (3, 0): 55,   (3, 1): 42,   (3, 2): 32,   (3, 3): 24,
    (4, 0): 80,   (4, 1): 62,   (4, 2): 46,   (4, 3): 34,
    (5, 0): 108,  (5, 1): 84,   (5, 2): 60,   (5, 3): 44,
    (6, 0): 136,  (6, 1): 106,  (6, 2): 74,   (6, 3): 58,
    (7, 0): 156,  (7, 1): 122,  (7, 2): 86,   (7, 3): 64,
    (8, 0): 194,  (8, 1): 152,  (8, 2): 108,  (8, 3): 84,
    (9, 0): 232,  (9, 1): 180,  (9, 2): 130,  (9, 3): 98,
    (10, 0): 271, (10, 1): 213, (10, 2): 151, (10, 3): 119,
    (11, 0): 321, (11, 1): 251, (11, 2): 177, (11, 3): 137,
    (12, 0): 367, (12, 1): 287, (12, 2): 203, (12, 3): 155,
    (13, 0): 425, (13, 1): 331, (13, 2): 241, (13, 3): 177,
    (14, 0): 458, (14, 1): 362, (14, 2): 258, (14, 3): 194,
    (15, 0): 520, (15, 1): 412, (15, 2): 292, (15, 3): 220,
    (16, 0): 586, (16, 1): 450, (16, 2): 322, (16, 3): 250,
    (17, 0): 644, (17, 1): 504, (17, 2): 364, (17, 3): 280,
    (18, 0): 718, (18, 1): 560, (18, 2): 394, (18, 3): 310,
    (19, 0): 792, (19, 1): 624, (19, 2): 442, (19, 3): 338,
    (20, 0): 858, (20, 1): 666, (20, 2): 482, (20, 3): 382,
    (21, 0): 929, (21, 1): 711, (21, 2): 509, (21, 3): 403,
    (22, 0): 1003,(22, 1): 779, (22, 2): 565, (22, 3): 439,
    (23, 0): 1091,(23, 1): 857, (23, 2): 611, (23, 3): 461,
    (24, 0): 1171,(24, 1): 911, (24, 2): 661, (24, 3): 511,
    (25, 0): 1273,(25, 1): 997, (25, 2): 715, (25, 3): 535,
    (26, 0): 1367,(26, 1): 1059,(26, 2): 751, (26, 3): 593,
    (27, 0): 1465,(27, 1): 1125,(27, 2): 805, (27, 3): 625,
    (28, 0): 1528,(28, 1): 1190,(28, 2): 868, (28, 3): 658,
    (29, 0): 1628,(29, 1): 1264,(29, 2): 908, (29, 3): 698,
    (30, 0): 1732,(30, 1): 1370,(30, 2): 982, (30, 3): 742,
    (31, 0): 1840,(31, 1): 1452,(31, 2): 1030,(31, 3): 790,
    (32, 0): 1952,(32, 1): 1538,(32, 2): 1112,(32, 3): 842,
    (33, 0): 2068,(33, 1): 1628,(33, 2): 1168,(33, 3): 898,
    (34, 0): 2188,(34, 1): 1722,(34, 2): 1228,(34, 3): 958,
    (35, 0): 2303,(35, 1): 1809,(35, 2): 1283,(35, 3): 983,
    (36, 0): 2431,(36, 1): 1911,(36, 2): 1351,(36, 3): 1051,
    (37, 0): 2563,(37, 1): 1989,(37, 2): 1423,(37, 3): 1093,
    (38, 0): 2699,(38, 1): 2099,(38, 2): 1499,(38, 3): 1139,
    (39, 0): 2809,(39, 1): 2213,(39, 2): 1579,(39, 3): 1219,
    (40, 0): 2953,(40, 1): 2331,(40, 2): 1663,(40, 3): 1273,
}

# Legacy shortcut: Version 40 capacities by ec_level
_QR_BINARY_CAPACITY = {
    0: 2953,   # L
    1: 2331,   # M
    2: 1663,   # Q
    3: 1273,   # H
}


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class V1Header:
    version: int  # 0 or 1
    compressed: bool
    filesize: int
    blocksize: int
    seed: int


@dataclass
class V2Header:
    version: int  # 2
    compressed: bool
    filesize: int
    blocksize: int
    block_count: int
    seed: int
    block_seq: int
    crc32: int


# ── V2 packing ───────────────────────────────────────────────────

def pack_v2(filesize: int, blocksize: int, block_count: int,
            seed: int, block_seq: int, data: bytes,
            compressed: bool = False) -> bytes:
    """Serialize a V2 block (header + data) to bytes."""
    flags = 0x01 if compressed else 0x00
    # Build the header without CRC first (bytes 0-15)
    header_no_crc = struct.pack('!BBIHHI',
                                0x02,           # version
                                flags,          # flags
                                filesize,       # uint32
                                blocksize,      # uint16
                                block_count,    # uint16
                                seed)           # uint32 (via I after HH)
    # Wait — the struct above doesn't match. Let me be explicit:
    header_no_crc = struct.pack('>BBIHHIH',
                                0x02,           # 1 byte  version
                                flags,          # 1 byte  flags
                                filesize,       # 4 bytes filesize
                                blocksize,      # 2 bytes blocksize
                                block_count,    # 2 bytes block_count
                                seed,           # 4 bytes seed
                                block_seq)      # 2 bytes block_seq
    # header_no_crc should be 1+1+4+2+2+4+2 = 16 bytes
    assert len(header_no_crc) == 16, f"header_no_crc is {len(header_no_crc)} bytes, expected 16"

    crc = zlib.crc32(header_no_crc + data) & 0xFFFFFFFF
    return header_no_crc + struct.pack('>I', crc) + data


# ── Unpacking (auto-detect V1/V2) ────────────────────────────────

def unpack(raw: bytes):
    """Unpack a raw block. Returns (header, data_bytes).

    Auto-detects V1 vs V2 based on the first byte.
    For V2, validates CRC32 and raises ValueError on mismatch.
    """
    if len(raw) < V1_HEADER_SIZE:
        raise ValueError(f"Block too short: {len(raw)} bytes")

    version_byte = raw[0]

    if version_byte == 0x02:
        return _unpack_v2(raw)
    else:
        return _unpack_v1(raw)


def _unpack_v1(raw: bytes):
    """Parse a V1 block."""
    magic, filesize, blocksize, seed = struct.unpack('!BIII', raw[:V1_HEADER_SIZE])
    compressed = bool(magic & 0x01)
    data = raw[V1_HEADER_SIZE:]
    header = V1Header(version=magic & 0xFE, compressed=compressed,
                      filesize=filesize, blocksize=blocksize, seed=seed)
    return header, data


def _unpack_v2(raw: bytes):
    """Parse a V2 block, checking CRC32."""
    if len(raw) < V2_HEADER_SIZE:
        raise ValueError(f"V2 block too short: {len(raw)} bytes")

    (version, flags, filesize, blocksize, block_count,
     seed, block_seq) = struct.unpack('>BBIHHIH', raw[:16])

    stored_crc = struct.unpack('>I', raw[16:20])[0]
    data = raw[V2_HEADER_SIZE:]

    # Validate CRC
    computed_crc = zlib.crc32(raw[:16] + data) & 0xFFFFFFFF
    if computed_crc != stored_crc:
        raise ValueError(
            f"CRC32 mismatch: stored=0x{stored_crc:08X}, "
            f"computed=0x{computed_crc:08X}")

    compressed = bool(flags & 0x01)
    header = V2Header(version=version, compressed=compressed,
                      filesize=filesize, blocksize=blocksize,
                      block_count=block_count, seed=seed,
                      block_seq=block_seq, crc32=stored_crc)
    return header, data


# ── Auto blocksize ────────────────────────────────────────────────

def auto_blocksize(filesize: int, ec_level: int = 1,
                   qr_version: int = 20) -> int:
    """Choose optimal blocksize given file size, QR error-correction level,
    and target QR version.

    Aims to fit each encoded block (header + data) into a single QR code
    after base64 encoding (4/3 expansion).

    Args:
        filesize:    Size of the (possibly compressed) data in bytes
        ec_level:    0=L, 1=M (default), 2=Q, 3=H
        qr_version:  QR code version 1-40 (default: 20)
    """
    qr_capacity = _QR_CAPACITY_BY_VERSION.get(
        (qr_version, ec_level),
        _QR_BINARY_CAPACITY.get(ec_level, 2331),  # fallback to v40
    )
    # base64 encodes every 3 bytes as 4 chars: ceil(N/3)*4 <= qr_capacity
    # So max raw bytes N = floor(qr_capacity / 4) * 3
    max_usable = (qr_capacity // 4) * 3
    max_blocksize = max_usable - V2_HEADER_SIZE
    max_blocksize = max(max_blocksize, 64)  # minimum sensible block size

    # Don't make blocks larger than the file itself
    blocksize = min(max_blocksize, filesize)
    blocksize = max(blocksize, 64)

    # Ensure block_count fits in uint16
    block_count = ceil(filesize / blocksize)
    if block_count > 65535:
        # Need larger blocks
        blocksize = ceil(filesize / 65535)

    return blocksize
