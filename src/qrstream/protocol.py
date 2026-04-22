"""
Protocol serialization helpers for QRStream.

Block layout (V3, 28 bytes overhead):
    24-byte fixed header + data + 4-byte trailing CRC32

QR-encoding flag (flag bit 0x02 in the header):
    - 0: base64 (standard mode, ASCII, pure byte-mode QR)
    - 1: high-density mode. Currently implemented with base45
      (QR alphanumeric mode). Older videos produced by qrstream <= 0.5
      used COBS+latin-1 here; the decoder's multi-strategy try chain
      keeps those playable.

The flag only tells the decoder which decoders to try first for the
on-wire QR payload. It does NOT change the LT/CRC layout at all.

Compatibility:
    The V2 protocol (0x02 version byte) was only ever produced by the
    pre-v0.4.0 internal releases. From v0.4.0 onwards every published
    qrstream built V3 blocks by default. V2 support has been dropped.
"""

import struct
import zlib
from dataclasses import dataclass
from math import ceil


# ── COBS (legacy decoder support only) ───────────────────────────
# COBS was the pre-0.6 high-density encoding. It is retained so that
# videos produced by older versions can still be decoded.  New
# encoders no longer emit COBS payloads.

def cobs_encode(data: bytes) -> bytes:
    """COBS-encode data: output contains no \x00 bytes.

    Overhead is at most 1 byte per 254 input bytes (~0.4%).
    Still exported for backward-compatible test fixtures; new code
    should use :func:`base45_encode` instead.
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
    """Decode COBS-encoded data back to original bytes.

    Used by the decoder when the incoming video was produced by a
    pre-0.6 qrstream version.
    """
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


# ── Base45 (RFC 9285, QR alphanumeric mode) ──────────────────────
# 2 raw bytes -> 3 ASCII chars from the 45-char QR alphanumeric
# alphabet.  The output string fits natively in QR's alphanumeric
# mode which packs every 2 chars into 11 bits, so the physical
# capacity at V20/M jumps from 499 B (base64) to 646 B.

_B45_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
assert len(_B45_ALPHABET) == 45
_B45_BYTES = _B45_ALPHABET.encode("ascii")
_B45_INDEX = {c: i for i, c in enumerate(_B45_ALPHABET)}


def base45_encode(data: bytes) -> bytes:
    """Encode bytes as a base45 ASCII string (RFC 9285).

    Returns bytes (ASCII-safe) for API consistency with
    :func:`base64.b64encode`.
    """
    out = bytearray()
    i = 0
    length = len(data)
    while i + 2 <= length:
        n = (data[i] << 8) | data[i + 1]
        c = n // 2025
        n -= c * 2025
        b = n // 45
        a = n - b * 45
        out.append(_B45_BYTES[a])
        out.append(_B45_BYTES[b])
        out.append(_B45_BYTES[c])
        i += 2
    if i < length:
        n = data[i]
        b = n // 45
        a = n - b * 45
        out.append(_B45_BYTES[a])
        out.append(_B45_BYTES[b])
    return bytes(out)


def base45_decode(data) -> bytes:
    """Decode a base45 string (bytes or str) back to raw bytes."""
    if isinstance(data, bytes):
        try:
            s = data.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("base45 input is not ASCII") from exc
    else:
        s = data
    out = bytearray()
    length = len(s)
    i = 0
    while i + 3 <= length:
        try:
            a = _B45_INDEX[s[i]]
            b = _B45_INDEX[s[i + 1]]
            c = _B45_INDEX[s[i + 2]]
        except KeyError as exc:
            raise ValueError(f"invalid base45 character: {exc}") from exc
        n = a + b * 45 + c * 2025
        if n > 0xFFFF:
            raise ValueError("invalid base45 triplet")
        out.append((n >> 8) & 0xFF)
        out.append(n & 0xFF)
        i += 3
    remaining = length - i
    if remaining == 2:
        try:
            a = _B45_INDEX[s[i]]
            b = _B45_INDEX[s[i + 1]]
        except KeyError as exc:
            raise ValueError(f"invalid base45 character: {exc}") from exc
        n = a + b * 45
        if n > 0xFF:
            raise ValueError("invalid base45 tail")
        out.append(n)
    elif remaining != 0:
        raise ValueError(f"invalid base45 length (remainder {remaining})")
    return bytes(out)


# ── V3 block layout ──────────────────────────────────────────────

V3_VERSION = 0x03

V3_HEADER_SIZE = 24
V3_TRAILING_CRC_SIZE = 4
V3_BLOCK_OVERHEAD = V3_HEADER_SIZE + V3_TRAILING_CRC_SIZE

# QR byte-mode capacity (ISO/IEC 18004), keyed by (version, ec_level).
# ec_level: 0=L, 1=M, 2=Q, 3=H.
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

# Fallback: Version 40 capacities by ec_level.
_QR_CAPACITY_V40 = {0: 2953, 1: 2331, 2: 1663, 3: 1273}

# QR alphanumeric-mode capacity (ISO/IEC 18004), keyed by (version,
# ec_level).  Each char occupies 5.5 bits (11 bits per 2-char group),
# so the alphanumeric capacity is about 1.37x the byte-mode capacity.
_QR_CAPACITY_ALPHANUMERIC = {
    (1, 0): 25,    (1, 1): 20,    (1, 2): 16,    (1, 3): 10,
    (2, 0): 47,    (2, 1): 38,    (2, 2): 29,    (2, 3): 20,
    (3, 0): 77,    (3, 1): 61,    (3, 2): 47,    (3, 3): 35,
    (4, 0): 114,   (4, 1): 90,    (4, 2): 67,    (4, 3): 50,
    (5, 0): 154,   (5, 1): 122,   (5, 2): 87,    (5, 3): 64,
    (6, 0): 195,   (6, 1): 154,   (6, 2): 108,   (6, 3): 84,
    (7, 0): 224,   (7, 1): 178,   (7, 2): 125,   (7, 3): 93,
    (8, 0): 279,   (8, 1): 221,   (8, 2): 157,   (8, 3): 122,
    (9, 0): 335,   (9, 1): 262,   (9, 2): 189,   (9, 3): 143,
    (10, 0): 395,  (10, 1): 311,  (10, 2): 221,  (10, 3): 174,
    (11, 0): 468,  (11, 1): 366,  (11, 2): 259,  (11, 3): 200,
    (12, 0): 535,  (12, 1): 419,  (12, 2): 296,  (12, 3): 227,
    (13, 0): 619,  (13, 1): 483,  (13, 2): 352,  (13, 3): 259,
    (14, 0): 667,  (14, 1): 528,  (14, 2): 376,  (14, 3): 283,
    (15, 0): 758,  (15, 1): 600,  (15, 2): 426,  (15, 3): 321,
    (16, 0): 854,  (16, 1): 656,  (16, 2): 470,  (16, 3): 365,
    (17, 0): 938,  (17, 1): 734,  (17, 2): 531,  (17, 3): 408,
    (18, 0): 1046, (18, 1): 816,  (18, 2): 574,  (18, 3): 452,
    (19, 0): 1153, (19, 1): 909,  (19, 2): 644,  (19, 3): 493,
    (20, 0): 1249, (20, 1): 970,  (20, 2): 702,  (20, 3): 557,
    (21, 0): 1352, (21, 1): 1035, (21, 2): 742,  (21, 3): 587,
    (22, 0): 1460, (22, 1): 1134, (22, 2): 823,  (22, 3): 640,
    (23, 0): 1588, (23, 1): 1248, (23, 2): 890,  (23, 3): 672,
    (24, 0): 1704, (24, 1): 1326, (24, 2): 963,  (24, 3): 744,
    (25, 0): 1853, (25, 1): 1451, (25, 2): 1041, (25, 3): 779,
    (26, 0): 1990, (26, 1): 1542, (26, 2): 1094, (26, 3): 864,
    (27, 0): 2132, (27, 1): 1637, (27, 2): 1172, (27, 3): 910,
    (28, 0): 2223, (28, 1): 1732, (28, 2): 1263, (28, 3): 958,
    (29, 0): 2369, (29, 1): 1839, (29, 2): 1322, (29, 3): 1016,
    (30, 0): 2520, (30, 1): 1994, (30, 2): 1429, (30, 3): 1080,
    (31, 0): 2677, (31, 1): 2113, (31, 2): 1499, (31, 3): 1150,
    (32, 0): 2840, (32, 1): 2238, (32, 2): 1618, (32, 3): 1226,
    (33, 0): 3009, (33, 1): 2369, (33, 2): 1700, (33, 3): 1307,
    (34, 0): 3183, (34, 1): 2506, (34, 2): 1787, (34, 3): 1394,
    (35, 0): 3351, (35, 1): 2632, (35, 2): 1867, (35, 3): 1431,
    (36, 0): 3537, (36, 1): 2780, (36, 2): 1966, (36, 3): 1530,
    (37, 0): 3729, (37, 1): 2894, (37, 2): 2071, (37, 3): 1591,
    (38, 0): 3927, (38, 1): 3054, (38, 2): 2181, (38, 3): 1658,
    (39, 0): 4087, (39, 1): 3220, (39, 2): 2298, (39, 3): 1774,
    (40, 0): 4296, (40, 1): 3391, (40, 2): 2420, (40, 3): 1852,
}

_QR_CAPACITY_ALPHANUMERIC_V40 = {0: 4296, 1: 3391, 2: 2420, 3: 1852}


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
    # Flag bit 0x02: set when the on-wire QR payload is encoded in a
    # high-density mode (base45 today, historically COBS).  Kept under
    # the legacy ``binary_qr`` name so existing API consumers don't
    # break; use the ``alphanumeric_qr`` alias for new code.
    binary_qr: bool = False
    reserved: int = 0

    @property
    def alphanumeric_qr(self) -> bool:
        """Alias for the high-density flag (base45 / legacy COBS)."""
        return self.binary_qr


def _resolve_alphanumeric_flag(binary_qr: bool,
                                alphanumeric_qr: bool | None) -> bool:
    """Reconcile the legacy ``binary_qr`` kw with the new alias."""
    if alphanumeric_qr is None:
        return binary_qr
    return alphanumeric_qr


def pack_v3(filesize: int, blocksize: int, block_count: int,
            seed: int, block_seq: int, data: bytes,
            compressed: bool = False,
            binary_qr: bool = False,
            alphanumeric_qr: bool | None = None) -> bytes:
    """Serialize a V3 block (header + data + trailing CRC32) to bytes.

    ``binary_qr`` and ``alphanumeric_qr`` are aliases for the
    high-density flag bit (0x02). Prefer ``alphanumeric_qr`` in new code.
    """
    high_density = _resolve_alphanumeric_flag(binary_qr, alphanumeric_qr)
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
    if high_density:
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
    """Unpack a V3 block based on the version byte.

    V2 support was dropped in qrstream 0.6 — the V2 layout was only
    ever produced by pre-v0.4.0 internal builds (no public release
    shipped V2 as default).
    """
    if not raw:
        raise ValueError("Block too short: 0 bytes")
    if raw[0] == V3_VERSION:
        return unpack_v3(raw, skip_crc=skip_crc)
    raise ValueError(f"Unsupported block version: 0x{raw[0]:02X}")


def _alphanumeric_byte_capacity(qr_version: int, ec_level: int) -> int:
    """Max raw bytes that fit when encoded via base45 in the given QR.

    base45 output size = ceil(N/2) * 3 characters when N is even; for
    odd N the tail is 2 chars (1 byte).  Given C alphanumeric chars of
    capacity, each full triplet carries 2 bytes and a 2-char tail can
    hold 1 byte.
    """
    cap = _QR_CAPACITY_ALPHANUMERIC.get(
        (qr_version, ec_level),
        _QR_CAPACITY_ALPHANUMERIC_V40.get(ec_level, 3391),
    )
    triplets = cap // 3
    usable = triplets * 2
    if cap % 3 >= 2:
        usable += 1
    return usable


def auto_blocksize(filesize: int, ec_level: int = 1,
                   qr_version: int = 25,
                   binary_qr: bool = True,
                   alphanumeric_qr: bool | None = None) -> int:
    """Choose an optimal blocksize for the given QR parameters.

    When the high-density flag is set, blocksize accounts for base45
    inflation and QR alphanumeric-mode capacity.  When it is cleared,
    blocksize accounts for base64 inflation and QR byte-mode capacity.

    ``binary_qr`` and ``alphanumeric_qr`` are aliases; prefer the
    latter in new code.
    """
    if ec_level not in (0, 1, 2, 3):
        raise ValueError(f"ec_level must be 0-3, got {ec_level}")
    if not 1 <= qr_version <= 40:
        raise ValueError(f"qr_version must be 1-40, got {qr_version}")

    high_density = _resolve_alphanumeric_flag(binary_qr, alphanumeric_qr)

    if high_density:
        max_usable = _alphanumeric_byte_capacity(qr_version, ec_level)
    else:
        qr_capacity = _QR_CAPACITY.get(
            (qr_version, ec_level),
            _QR_CAPACITY_V40.get(ec_level, 2331),
        )
        # base64: ceil(N/3)*4 <= qr_capacity  =>  N = floor(qr_capacity/4)*3
        max_usable = (qr_capacity // 4) * 3

    # Leave a 1-byte margin; qrcode occasionally refuses payloads
    # exactly at the alphanumeric boundary for some byte values.
    max_blocksize = max(max_usable - V3_BLOCK_OVERHEAD - 1, 64)
    return max(min(max_blocksize, filesize), 64)


__all__ = [
    "V3_VERSION",
    "V3_HEADER_SIZE", "V3_TRAILING_CRC_SIZE", "V3_BLOCK_OVERHEAD",
    "V3Header",
    "pack_v3", "unpack", "unpack_v3",
    "auto_blocksize",
    "cobs_encode", "cobs_decode",
    "base45_encode", "base45_decode",
]
