"""
Protocol serialization helpers for QRStream.

V2 block layout (20 bytes overhead):
    16-byte fixed header + 4-byte CRC32 + data

V3 block layout (28 bytes overhead):
    24-byte fixed header + data + 4-byte trailing CRC32
"""

import struct
import zlib
from dataclasses import dataclass
from math import ceil

# ── COBS (Consistent Overhead Byte Stuffing) ─────────────────────
# Eliminates all \x00 bytes from data with ~0.4% overhead.
# This allows binary data to survive QR decoders that use C strings.


def cobs_encode(data: bytes) -> bytes:
    """COBS-encode data: output contains no \x00 bytes.

    Overhead is at most 1 byte per 254 input bytes (~0.4%).
    """
    output = bytearray()
    idx = 0
    length = len(data)
    while idx <= length:
        group = bytearray()
        while idx < length and data[idx] != 0 and len(group) < 254:
            group.append(data[idx])
            idx += 1
        if idx < length and data[idx] == 0:
            output.append(len(group) + 1)
            output.extend(group)
            idx += 1
        else:
            if len(group) == 254:
                output.append(0xFF)
                output.extend(group)
            else:
                output.append(len(group) + 1)
                output.extend(group)
                break
    return bytes(output)


def cobs_decode(data: bytes) -> bytes:
    """Decode COBS-encoded data back to original bytes."""
    output = bytearray()
    idx = 0
    length = len(data)
    while idx < length:
        code = data[idx]
        if code == 0:
            raise ValueError("COBS decode error: unexpected zero byte")
        idx += 1
        for _ in range(code - 1):
            if idx >= length:
                raise ValueError("COBS decode error: truncated data")
            output.append(data[idx])
            idx += 1
        if code < 0xFF and idx < length:
            output.append(0)
    return bytes(output)


# ── Header constants ──────────────────────────────────────────────

V2_VERSION = 0x02
V3_VERSION = 0x03

V2_HEADER_NO_CRC_SIZE = 16
V2_HEADER_SIZE = 20
V2_BLOCK_OVERHEAD = 20

V3_HEADER_SIZE = 24
V3_TRAILING_CRC_SIZE = 4
V3_BLOCK_OVERHEAD = V3_HEADER_SIZE + V3_TRAILING_CRC_SIZE

# QR capacity table (byte mode, per version and ECC level)
# Source: ISO/IEC 18004
# Key: (version, ec_level) -> max bytes.  ec_level: 0=L, 1=M, 2=Q, 3=H
_QR_CAPACITY = {
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
    (22, 0): 1003, (22, 1): 779, (22, 2): 565, (22, 3): 439,
    (23, 0): 1091, (23, 1): 857, (23, 2): 611, (23, 3): 461,
    (24, 0): 1171, (24, 1): 911, (24, 2): 661, (24, 3): 511,
    (25, 0): 1273, (25, 1): 997, (25, 2): 715, (25, 3): 535,
    (26, 0): 1367, (26, 1): 1059, (26, 2): 751, (26, 3): 593,
    (27, 0): 1465, (27, 1): 1125, (27, 2): 805, (27, 3): 625,
    (28, 0): 1528, (28, 1): 1190, (28, 2): 868, (28, 3): 658,
    (29, 0): 1628, (29, 1): 1264, (29, 2): 908, (29, 3): 698,
    (30, 0): 1732, (30, 1): 1370, (30, 2): 982, (30, 3): 742,
    (31, 0): 1840, (31, 1): 1452, (31, 2): 1030, (31, 3): 790,
    (32, 0): 1952, (32, 1): 1538, (32, 2): 1112, (32, 3): 842,
    (33, 0): 2068, (33, 1): 1628, (33, 2): 1168, (33, 3): 898,
    (34, 0): 2188, (34, 1): 1722, (34, 2): 1228, (34, 3): 958,
    (35, 0): 2303, (35, 1): 1809, (35, 2): 1283, (35, 3): 983,
    (36, 0): 2431, (36, 1): 1911, (36, 2): 1351, (36, 3): 1051,
    (37, 0): 2563, (37, 1): 1989, (37, 2): 1423, (37, 3): 1093,
    (38, 0): 2699, (38, 1): 2099, (38, 2): 1499, (38, 3): 1139,
    (39, 0): 2809, (39, 1): 2213, (39, 2): 1579, (39, 3): 1219,
    (40, 0): 2953, (40, 1): 2331, (40, 2): 1663, (40, 3): 1273,
}

# Fallback: Version 40 capacities by ec_level
_QR_CAPACITY_V40 = {0: 2953, 1: 2331, 2: 1663, 3: 1273}


@dataclass
class V2Header:
    version: int
    compressed: bool
    filesize: int
    blocksize: int
    block_count: int
    seed: int
    block_seq: int
    crc32: int
    binary_qr: bool = False


@dataclass
class V3Header:
    version: int
    compressed: bool
    filesize: int
    blocksize: int
    block_count: int
    seed: int
    block_seq: int
    crc32: int
    binary_qr: bool = False
    reserved: int = 0


def pack_v2(filesize: int, blocksize: int, block_count: int,
            seed: int, block_seq: int, data: bytes,
            compressed: bool = False,
            binary_qr: bool = False) -> bytes:
    """Serialize a V2 block (header + data) to bytes."""
    if filesize > 0xFFFFFFFF:
        raise ValueError("V2 filesize exceeds uint32 limit; use V3")
    if block_count > 0xFFFF:
        raise ValueError("V2 block_count exceeds uint16 limit; use V3")
    if blocksize > 0xFFFF:
        raise ValueError("V2 blocksize exceeds uint16 limit")
    if len(data) > blocksize:
        raise ValueError("Block data longer than blocksize")

    flags = 0x00
    if compressed:
        flags |= 0x01
    if binary_qr:
        flags |= 0x02
    header_no_crc = struct.pack(
        '>BBIHHIH',
        V2_VERSION, flags, filesize,
        blocksize, block_count,
        seed, block_seq,
    )
    crc = zlib.crc32(header_no_crc + data) & 0xFFFFFFFF
    return header_no_crc + struct.pack('>I', crc) + data


def pack_v3(filesize: int, blocksize: int, block_count: int,
            seed: int, block_seq: int, data: bytes,
            compressed: bool = False,
            binary_qr: bool = False) -> bytes:
    """Serialize a V3 block (header + data + trailing CRC32) to bytes."""
    if filesize > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("V3 filesize exceeds uint64 limit")
    if block_count > 0xFFFFFFFF:
        raise ValueError("V3 block_count exceeds uint32 limit")
    if blocksize > 0xFFFF:
        raise ValueError("V3 blocksize exceeds uint16 limit")
    if len(data) > blocksize:
        raise ValueError("Block data longer than blocksize")

    flags = 0x00
    if compressed:
        flags |= 0x01
    if binary_qr:
        flags |= 0x02

    header = struct.pack(
        '>BBQHIIHH',
        V3_VERSION,
        flags,
        filesize,
        blocksize,
        block_count,
        seed,
        block_seq,
        0,
    )
    crc = zlib.crc32(header + data) & 0xFFFFFFFF
    return header + data + struct.pack('>I', crc)


def unpack_v2(raw: bytes, skip_crc: bool = False) -> tuple[V2Header, bytes]:
    """Unpack a V2 block."""
    if len(raw) < V2_HEADER_SIZE:
        raise ValueError(f"Block too short: {len(raw)} bytes")
    if raw[0] != V2_VERSION:
        raise ValueError(f"Not a V2 block: version byte 0x{raw[0]:02X}")

    (version, flags, filesize, blocksize, block_count,
     seed, block_seq) = struct.unpack('>BBIHHIH', raw[:V2_HEADER_NO_CRC_SIZE])

    stored_crc = struct.unpack('>I', raw[16:20])[0]
    data = raw[V2_HEADER_SIZE:]

    if len(data) != blocksize:
        raise ValueError(
            f"V2 data length mismatch: expected {blocksize}, got {len(data)}")

    if not skip_crc:
        computed_crc = zlib.crc32(raw[:V2_HEADER_NO_CRC_SIZE] + data) & 0xFFFFFFFF
        if computed_crc != stored_crc:
            raise ValueError(
                f"CRC32 mismatch: stored=0x{stored_crc:08X}, "
                f"computed=0x{computed_crc:08X}")

    header = V2Header(
        version=version,
        compressed=bool(flags & 0x01),
        filesize=filesize,
        blocksize=blocksize,
        block_count=block_count,
        seed=seed,
        block_seq=block_seq,
        crc32=stored_crc,
        binary_qr=bool(flags & 0x02),
    )
    return header, data


def unpack_v3(raw: bytes, skip_crc: bool = False) -> tuple[V3Header, bytes]:
    """Unpack a V3 block."""
    if len(raw) < V3_BLOCK_OVERHEAD:
        raise ValueError(f"Block too short: {len(raw)} bytes")
    if raw[0] != V3_VERSION:
        raise ValueError(f"Not a V3 block: version byte 0x{raw[0]:02X}")

    (version, flags, filesize, blocksize, block_count,
     seed, block_seq, reserved) = struct.unpack('>BBQHIIHH', raw[:V3_HEADER_SIZE])

    data = raw[V3_HEADER_SIZE:-V3_TRAILING_CRC_SIZE]
    stored_crc = struct.unpack('>I', raw[-V3_TRAILING_CRC_SIZE:])[0]

    if len(data) != blocksize:
        raise ValueError(
            f"V3 data length mismatch: expected {blocksize}, got {len(data)}")

    if not skip_crc:
        computed_crc = zlib.crc32(raw[:-V3_TRAILING_CRC_SIZE]) & 0xFFFFFFFF
        if computed_crc != stored_crc:
            raise ValueError(
                f"CRC32 mismatch: stored=0x{stored_crc:08X}, "
                f"computed=0x{computed_crc:08X}")

    header = V3Header(
        version=version,
        compressed=bool(flags & 0x01),
        filesize=filesize,
        blocksize=blocksize,
        block_count=block_count,
        seed=seed,
        block_seq=block_seq,
        crc32=stored_crc,
        binary_qr=bool(flags & 0x02),
        reserved=reserved,
    )
    return header, data


def unpack(raw: bytes, skip_crc: bool = False):
    """Unpack a V2 or V3 block based on the version byte."""
    if not raw:
        raise ValueError("Block too short: 0 bytes")
    if raw[0] == V2_VERSION:
        return unpack_v2(raw, skip_crc=skip_crc)
    if raw[0] == V3_VERSION:
        return unpack_v3(raw, skip_crc=skip_crc)
    raise ValueError(f"Unsupported block version: 0x{raw[0]:02X}")


def _block_overhead(protocol_version: int) -> int:
    if protocol_version == V2_VERSION:
        return V2_BLOCK_OVERHEAD
    if protocol_version == V3_VERSION:
        return V3_BLOCK_OVERHEAD
    raise ValueError(f"Unsupported protocol version: {protocol_version}")


def auto_blocksize(filesize: int, ec_level: int = 1,
                   qr_version: int = 20,
                   binary_qr: bool = True,
                   protocol_version: int = V3_VERSION) -> int:
    """Choose an optimal blocksize for the given QR parameters.

    When binary_qr=True, accounts for COBS overhead.
    When binary_qr=False, accounts for base64 overhead.
    The returned blocksize respects the selected protocol overhead.
    """
    if ec_level not in (0, 1, 2, 3):
        raise ValueError(f"ec_level must be 0-3, got {ec_level}")
    if not 1 <= qr_version <= 40:
        raise ValueError(f"qr_version must be 1-40, got {qr_version}")

    qr_capacity = _QR_CAPACITY.get(
        (qr_version, ec_level),
        _QR_CAPACITY_V40.get(ec_level, 2331),
    )

    if binary_qr:
        # COBS: N + ceil(N/254) <= qr_capacity => N <= qr_capacity * 254/255
        max_usable = (qr_capacity * 254) // 255
    else:
        # base64: ceil(N/3)*4 <= qr_capacity => N = floor(qr_capacity/4)*3
        max_usable = (qr_capacity // 4) * 3

    overhead = _block_overhead(protocol_version)
    max_blocksize = max(max_usable - overhead, 64)
    blocksize = max(min(max_blocksize, filesize), 64)

    if protocol_version == V2_VERSION and ceil(filesize / blocksize) > 65535:
        required = ceil(filesize / 65535)
        if required > max_blocksize:
            raise ValueError(
                "File too large for V2 with current QR settings; use V3 or a larger QR version")
        blocksize = required

    return blocksize
