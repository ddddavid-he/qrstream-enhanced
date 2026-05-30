pub(crate) const V4_VERSION: u8 = 0x04;

const V4_HEADER_SIZE: usize = 24;
const V4_TRAILING_CRC_SIZE: usize = 4;
const V4_BLOCK_OVERHEAD: usize = V4_HEADER_SIZE + V4_TRAILING_CRC_SIZE;
const RQ_ESI_MASK: u32 = 0x00ff_ffff;
const RQ_SBN_SHIFT: u32 = 24;

#[derive(Debug, Clone)]
pub(crate) struct V4Block<'a> {
    pub(crate) compressed: bool,
    pub(crate) filesize: u64,
    pub(crate) symbol_size: u16,
    pub(crate) symbol_count: u32,
    pub(crate) esi: u32,
    pub(crate) source_blocks: u16,
    pub(crate) data: &'a [u8],
}

pub(crate) fn parse_v4_block(raw: &[u8]) -> Result<V4Block<'_>, String> {
    if raw.len() < V4_BLOCK_OVERHEAD {
        return Err(format!("Block too short: {} bytes", raw.len()));
    }
    if raw[0] != V4_VERSION {
        return Err(format!("Not a V4 block: version byte 0x{:02X}", raw[0]));
    }

    let stored_crc = u32::from_be_bytes(raw[raw.len() - 4..].try_into().unwrap());
    let computed_crc = crc32fast::hash(&raw[..raw.len() - V4_TRAILING_CRC_SIZE]);
    if computed_crc != stored_crc {
        return Err(format!(
            "CRC32 mismatch: stored=0x{:08X}, computed=0x{:08X}",
            stored_crc, computed_crc
        ));
    }

    let flags = raw[1];
    let filesize = u64::from_be_bytes(raw[2..10].try_into().unwrap());
    let symbol_size = u16::from_be_bytes(raw[10..12].try_into().unwrap());
    let symbol_count = u32::from_be_bytes(raw[12..16].try_into().unwrap());
    let esi = u32::from_be_bytes(raw[16..20].try_into().unwrap());
    let source_blocks = u16::from_be_bytes(raw[22..24].try_into().unwrap());
    let data = &raw[V4_HEADER_SIZE..raw.len() - V4_TRAILING_CRC_SIZE];

    if data.len() != symbol_size as usize {
        return Err(format!(
            "V4 data length mismatch: expected {}, got {}",
            symbol_size,
            data.len()
        ));
    }

    Ok(V4Block {
        compressed: flags & 0x01 != 0,
        filesize,
        symbol_size,
        symbol_count,
        esi,
        source_blocks,
        data,
    })
}

pub(crate) fn source_index(payload_id: u32, total_symbols: u32, source_blocks: u16) -> Option<u32> {
    let sbn = ((payload_id >> RQ_SBN_SHIFT) & 0xff) as u16;
    let local_esi = payload_id & RQ_ESI_MASK;
    if total_symbols == 0 || source_blocks == 0 || sbn >= source_blocks {
        return None;
    }

    let source_blocks_u32 = source_blocks as u32;
    let large = total_symbols.div_ceil(source_blocks_u32);
    let small = large.saturating_sub(1);
    let large_count = total_symbols.saturating_sub(small * source_blocks_u32);

    let mut offset = 0;
    for current_sbn in 0..source_blocks_u32 {
        let count = if current_sbn < large_count { large } else { small };
        if current_sbn == sbn as u32 {
            if local_esi < count {
                return Some(offset + local_esi);
            }
            return None;
        }
        offset += count;
    }
    None
}
