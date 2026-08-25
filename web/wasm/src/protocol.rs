//! V4 (RaptorQ) protocol block parsing.
//!
//! Block layout: 24-byte big-endian header + symbol data + 4-byte trailing CRC32.
//! Wire format struct: `>BBQHIIHH` (version, flags, filesize, symbol_size,
//! symbol_count, esi, block_seq, reserved).

use crc32fast::Hasher;

pub const V4_VERSION: u8 = 0x04;
pub const HEADER_SIZE: usize = 24;
pub const TRAILING_CRC_SIZE: usize = 4;
pub const BLOCK_OVERHEAD: usize = HEADER_SIZE + TRAILING_CRC_SIZE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V4Header {
    pub version: u8,
    pub compressed: bool,
    pub filesize: u64,
    pub symbol_size: u16,
    pub symbol_count: u32,
    pub esi: u32,
    pub block_seq: u16,
    pub binary_qr: bool,
    pub reserved: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V4Block {
    pub header: V4Header,
    pub data: Vec<u8>,
}

fn crc32(data: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(data);
    h.finalize()
}

/// Unpack and validate a raw V4 block (header + data + trailing CRC32).
pub fn unpack(raw: &[u8]) -> Result<V4Block, String> {
    if raw.len() < BLOCK_OVERHEAD {
        return Err(format!("Block too short: {} bytes", raw.len()));
    }
    if raw[0] != V4_VERSION {
        return Err(format!("Unsupported block version: 0x{:02X}", raw[0]));
    }

    let version = raw[0];
    let flags = raw[1];
    let filesize = u64::from_be_bytes(raw[2..10].try_into().unwrap());
    let symbol_size = u16::from_be_bytes(raw[10..12].try_into().unwrap());
    let symbol_count = u32::from_be_bytes(raw[12..16].try_into().unwrap());
    let esi = u32::from_be_bytes(raw[16..20].try_into().unwrap());
    let block_seq = u16::from_be_bytes(raw[20..22].try_into().unwrap());
    let reserved = u16::from_be_bytes(raw[22..24].try_into().unwrap());

    let data = &raw[HEADER_SIZE..raw.len() - TRAILING_CRC_SIZE];
    let stored_crc = u32::from_be_bytes(raw[raw.len() - TRAILING_CRC_SIZE..].try_into().unwrap());

    if data.len() != symbol_size as usize {
        return Err(format!(
            "V4 data length mismatch: expected {}, got {}",
            symbol_size,
            data.len()
        ));
    }

    let computed_crc = crc32(&raw[..raw.len() - TRAILING_CRC_SIZE]);
    if computed_crc != stored_crc {
        return Err(format!(
            "CRC32 mismatch: stored=0x{stored_crc:08X}, computed=0x{computed_crc:08X}"
        ));
    }

    Ok(V4Block {
        header: V4Header {
            version,
            compressed: flags & 0x01 != 0,
            filesize,
            symbol_size,
            symbol_count,
            esi,
            block_seq,
            binary_qr: flags & 0x02 != 0,
            reserved,
        },
        data: data.to_vec(),
    })
}

/// RaptorQ PayloadId helpers (SBN || local ESI).
pub const RQ_SBN_SHIFT: u32 = 24;
pub const RQ_ESI_MASK: u32 = 0x00FF_FFFF;

#[inline]
pub fn payload_id_parts(payload_id: u32) -> (u32, u32) {
    (
        (payload_id >> RQ_SBN_SHIFT) & 0xFF,
        payload_id & RQ_ESI_MASK,
    )
}

#[inline]
pub fn make_payload_id(sbn: u32, local_esi: u32) -> u32 {
    ((sbn & 0xFF) << RQ_SBN_SHIFT) | (local_esi & RQ_ESI_MASK)
}

/// RFC 6330-style source block layout for `total_symbols` across `source_blocks`.
/// Returns `[(global_start, symbol_count)]` per SBN.
pub fn source_block_layout(total_symbols: u32, source_blocks: u32) -> Vec<(u32, u32)> {
    let mut layout = Vec::new();
    if total_symbols == 0 || source_blocks == 0 {
        return layout;
    }
    let large = total_symbols.div_ceil(source_blocks);
    let small = large - 1;
    let large_count = total_symbols - small * source_blocks;
    let mut offset = 0u32;
    for sbn in 0..source_blocks {
        let count = if sbn < large_count { large } else { small };
        layout.push((offset, count));
        offset += count;
    }
    layout
}

/// Map a systematic PayloadId to a global source-symbol index.
/// Returns `None` for repair PayloadIds or out-of-range SBNs.
pub fn source_index(payload_id: u32, total_symbols: u32, source_blocks: u32) -> Option<u32> {
    let (sbn, local_esi) = payload_id_parts(payload_id);
    let layout = source_block_layout(total_symbols, source_blocks);
    let &(offset, count) = layout.get(sbn as usize)?;
    if local_esi >= count {
        return None;
    }
    Some(offset + local_esi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_v4(
        filesize: u64,
        symbol_size: u16,
        symbol_count: u32,
        esi: u32,
        block_seq: u16,
        data: &[u8],
        compressed: bool,
        reserved: u16,
    ) -> Vec<u8> {
        let mut flags = 0u8;
        if compressed {
            flags |= 0x01;
        }
        let mut raw = Vec::with_capacity(BLOCK_OVERHEAD + data.len());
        raw.push(V4_VERSION);
        raw.push(flags);
        raw.extend_from_slice(&filesize.to_be_bytes());
        raw.extend_from_slice(&symbol_size.to_be_bytes());
        raw.extend_from_slice(&symbol_count.to_be_bytes());
        raw.extend_from_slice(&esi.to_be_bytes());
        raw.extend_from_slice(&block_seq.to_be_bytes());
        raw.extend_from_slice(&reserved.to_be_bytes());
        raw.extend_from_slice(data);
        let crc = crc32(&raw);
        raw.extend_from_slice(&crc.to_be_bytes());
        raw
    }

    #[test]
    fn unpack_roundtrip() {
        let data = b"hello world symbol data";
        let raw = pack_v4(1234, data.len() as u16, 7, 0x0102_03FF, 3, data, true, 2);
        let block = unpack(&raw).unwrap();
        assert_eq!(block.header.filesize, 1234);
        assert_eq!(block.header.symbol_size, data.len() as u16);
        assert_eq!(block.header.symbol_count, 7);
        assert_eq!(block.header.esi, 0x0102_03FF);
        assert_eq!(block.header.block_seq, 3);
        assert!(block.header.compressed);
        assert_eq!(block.header.reserved, 2);
        assert_eq!(block.data, data);
    }

    #[test]
    fn rejects_short_block() {
        assert!(unpack(&[0x04, 0x00]).is_err());
    }

    #[test]
    fn rejects_wrong_version() {
        let mut raw = vec![0u8; BLOCK_OVERHEAD + 4];
        raw[0] = 0x03;
        assert!(unpack(&raw).is_err());
    }

    #[test]
    fn rejects_bad_crc() {
        let data = b"payload";
        let mut raw = pack_v4(10, data.len() as u16, 1, 0, 0, data, false, 0);
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        assert!(unpack(&raw).is_err());
    }

    #[test]
    fn payload_id_roundtrip() {
        assert_eq!(payload_id_parts(0x0102_03FF), (1, 0x0203FF));
        assert_eq!(make_payload_id(1, 0x0203FF), 0x0102_03FF);
    }

    #[test]
    fn source_layout_matches_python() {
        // Mirror of _rq_source_block_layout tests: K=10, Z=3
        // large=4, small=3, large_count=1 -> [(0,4),(4,3),(7,3)]
        let layout = source_block_layout(10, 3);
        assert_eq!(layout, vec![(0, 4), (4, 3), (7, 3)]);
        // K=100, Z=1 -> [(0,100)]
        assert_eq!(source_block_layout(100, 1), vec![(0, 100)]);
    }

    #[test]
    fn source_index_mapping() {
        // SBN=0, esi=0 -> 0; SBN=0, esi=3 -> 3; SBN=1, esi=0 -> 4
        assert_eq!(source_index(make_payload_id(0, 0), 10, 3), Some(0));
        assert_eq!(source_index(make_payload_id(0, 3), 10, 3), Some(3));
        assert_eq!(source_index(make_payload_id(1, 0), 10, 3), Some(4));
        // esi >= count within SBN -> repair symbol
        assert_eq!(source_index(make_payload_id(0, 4), 10, 3), None);
        assert_eq!(source_index(make_payload_id(2, 2), 10, 3), Some(9));
        assert_eq!(source_index(make_payload_id(2, 3), 10, 3), None);
        assert_eq!(source_index(make_payload_id(3, 0), 10, 3), None);
    }
}
