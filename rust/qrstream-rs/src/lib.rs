use std::collections::HashSet;
use std::io::Read;

use base64::Engine;
use flate2::read::ZlibDecoder;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use raptorq::{Decoder, EncodingPacket, ObjectTransmissionInformation, PayloadId};

const V4_VERSION: u8 = 0x04;
const V4_HEADER_SIZE: usize = 24;
const V4_TRAILING_CRC_SIZE: usize = 4;
const V4_BLOCK_OVERHEAD: usize = V4_HEADER_SIZE + V4_TRAILING_CRC_SIZE;
const RQ_ESI_MASK: u32 = 0x00ff_ffff;
const RQ_SBN_SHIFT: u32 = 24;

const B45_ALPHABET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

#[derive(Debug, Clone)]
struct V4Block<'a> {
    compressed: bool,
    filesize: u64,
    symbol_size: u16,
    symbol_count: u32,
    esi: u32,
    source_blocks: u16,
    data: &'a [u8],
}

#[pyclass]
#[derive(Clone)]
struct DecodeResult {
    #[pyo3(get)]
    accepted: bool,
    #[pyo3(get)]
    duplicate: bool,
    #[pyo3(get)]
    done: bool,
    #[pyo3(get)]
    progress: f64,
    #[pyo3(get)]
    num_recovered: usize,
    #[pyo3(get)]
    symbol_count: Option<u32>,
    #[pyo3(get)]
    filesize: Option<u64>,
    #[pyo3(get)]
    protocol_version: Option<u8>,
    #[pyo3(get)]
    error: Option<String>,
}

#[pymethods]
impl DecodeResult {
    fn __repr__(&self) -> String {
        format!(
            "DecodeResult(accepted={}, duplicate={}, done={}, progress={}, num_recovered={}, symbol_count={:?}, filesize={:?}, protocol_version={:?}, error={:?})",
            self.accepted,
            self.duplicate,
            self.done,
            self.progress,
            self.num_recovered,
            self.symbol_count,
            self.filesize,
            self.protocol_version,
            self.error,
        )
    }
}

#[pyclass]
#[derive(Clone)]
struct DecodeSnapshot {
    #[pyo3(get)]
    initialized: bool,
    #[pyo3(get)]
    done: bool,
    #[pyo3(get)]
    progress: f64,
    #[pyo3(get)]
    num_recovered: usize,
    #[pyo3(get)]
    symbol_count: Option<u32>,
    #[pyo3(get)]
    filesize: Option<u64>,
    #[pyo3(get)]
    protocol_version: Option<u8>,
}

#[pymethods]
impl DecodeSnapshot {
    fn __repr__(&self) -> String {
        format!(
            "DecodeSnapshot(initialized={}, done={}, progress={}, num_recovered={}, symbol_count={:?}, filesize={:?}, protocol_version={:?})",
            self.initialized,
            self.done,
            self.progress,
            self.num_recovered,
            self.symbol_count,
            self.filesize,
            self.protocol_version,
        )
    }
}

#[pyclass]
struct V4DecodeSession {
    decoder: Option<Decoder>,
    initialized: bool,
    done: bool,
    compressed: bool,
    filesize: u64,
    symbol_size: u16,
    symbol_count: u32,
    source_blocks: u16,
    result: Option<Vec<u8>>,
    seen_payloads: HashSet<u32>,
    recovered_sources: HashSet<u32>,
}

#[pymethods]
impl V4DecodeSession {
    #[new]
    fn new() -> Self {
        Self {
            decoder: None,
            initialized: false,
            done: false,
            compressed: false,
            filesize: 0,
            symbol_size: 0,
            symbol_count: 0,
            source_blocks: 0,
            result: None,
            seen_payloads: HashSet::new(),
            recovered_sources: HashSet::new(),
        }
    }

    fn consume_qr_text(&mut self, qr_text: &str) -> DecodeResult {
        match decode_qr_payload(qr_text) {
            Ok(block) => self.consume_block(block.as_slice()),
            Err(err) => self.result(false, false, Some(err)),
        }
    }

    fn consume_block(&mut self, block_bytes: &[u8]) -> DecodeResult {
        match self.consume_block_inner(block_bytes) {
            Ok(result) => result,
            Err(err) => self.result(false, false, Some(err)),
        }
    }

    fn consume_blocks(&mut self, blocks: Vec<Vec<u8>>) -> DecodeResult {
        let mut last = self.result(false, false, None);
        for block in blocks {
            last = match self.consume_block_inner(block.as_slice()) {
                Ok(result) => result,
                Err(err) => return self.result(false, false, Some(err)),
            };
            if last.done {
                return last;
            }
        }
        last
    }

    fn snapshot(&self) -> DecodeSnapshot {
        DecodeSnapshot {
            initialized: self.initialized,
            done: self.done,
            progress: self.progress(),
            num_recovered: self.num_recovered(),
            symbol_count: self.initialized.then_some(self.symbol_count),
            filesize: self.initialized.then_some(self.filesize),
            protocol_version: self.initialized.then_some(V4_VERSION),
        }
    }

    fn result_bytes(&self) -> PyResult<Vec<u8>> {
        if !self.done {
            return Err(PyRuntimeError::new_err("Decoding incomplete — no result available"));
        }
        Ok(self.result.clone().unwrap_or_default())
    }
}

impl V4DecodeSession {
    fn consume_block_inner(&mut self, block_bytes: &[u8]) -> Result<DecodeResult, String> {
        let block = parse_v4_block(block_bytes)?;

        if self.seen_payloads.contains(&block.esi) {
            return Ok(self.result(true, true, None));
        }

        self.ensure_initialized(&block)?;
        self.validate_consistent(&block)?;

        if self.done {
            self.seen_payloads.insert(block.esi);
            return Ok(self.result(true, false, None));
        }

        let payload_id = PayloadId::deserialize(&block.esi.to_be_bytes());
        let packet = EncodingPacket::new(payload_id, block.data.to_vec());
        let decoded = self.decoder.as_mut().unwrap().decode(packet);
        self.seen_payloads.insert(block.esi);

        if source_index(block.esi, self.symbol_count, self.source_blocks).is_some() {
            self.recovered_sources.insert(block.esi);
        }

        if let Some(bytes) = decoded {
            let trimmed_len = self.filesize as usize;
            let raw = bytes[..trimmed_len.min(bytes.len())].to_vec();
            let result = maybe_decompress(raw, self.compressed)?;
            self.result = Some(result);
            self.done = true;
            for idx in 0..self.symbol_count {
                self.recovered_sources.insert(idx);
            }
        }

        Ok(self.result(true, false, None))
    }

    fn ensure_initialized(&mut self, block: &V4Block<'_>) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }
        let padded_len = block.symbol_count as u64 * block.symbol_size as u64;
        let oti = ObjectTransmissionInformation::with_defaults(padded_len, block.symbol_size);
        self.decoder = Some(Decoder::new(oti));
        self.initialized = true;
        self.compressed = block.compressed;
        self.filesize = block.filesize;
        self.symbol_size = block.symbol_size;
        self.symbol_count = block.symbol_count;
        self.source_blocks = if block.source_blocks > 0 { block.source_blocks } else { 1 };
        Ok(())
    }

    fn validate_consistent(&self, block: &V4Block<'_>) -> Result<(), String> {
        let source_blocks = if block.source_blocks > 0 { block.source_blocks } else { 1 };
        if block.filesize != self.filesize {
            return Err(format!("filesize mismatch: {} != {}", block.filesize, self.filesize));
        }
        if block.symbol_size != self.symbol_size {
            return Err(format!("symbol_size mismatch: {} != {}", block.symbol_size, self.symbol_size));
        }
        if block.symbol_count != self.symbol_count {
            return Err(format!("symbol_count mismatch: {} != {}", block.symbol_count, self.symbol_count));
        }
        if source_blocks != self.source_blocks {
            return Err(format!("source_blocks mismatch: {} != {}", source_blocks, self.source_blocks));
        }
        if block.compressed != self.compressed {
            return Err(format!("compressed flag mismatch: {} != {}", block.compressed, self.compressed));
        }
        Ok(())
    }

    fn result(&self, accepted: bool, duplicate: bool, error: Option<String>) -> DecodeResult {
        DecodeResult {
            accepted,
            duplicate,
            done: self.done,
            progress: self.progress(),
            num_recovered: self.num_recovered(),
            symbol_count: self.initialized.then_some(self.symbol_count),
            filesize: self.initialized.then_some(self.filesize),
            protocol_version: self.initialized.then_some(V4_VERSION),
            error,
        }
    }

    fn progress(&self) -> f64 {
        if !self.initialized || self.symbol_count == 0 {
            return 0.0;
        }
        if self.done {
            return 1.0;
        }
        (self.recovered_sources.len() as f64 / self.symbol_count as f64).min(0.99)
    }

    fn num_recovered(&self) -> usize {
        if self.done {
            self.symbol_count as usize
        } else {
            self.recovered_sources.len()
        }
    }
}

fn parse_v4_block(raw: &[u8]) -> Result<V4Block<'_>, String> {
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

fn decode_qr_payload(qr_text: &str) -> Result<Vec<u8>, String> {
    if let Ok(bytes) = base45_decode(qr_text) {
        if parse_v4_block(&bytes).is_ok() {
            return Ok(bytes);
        }
    }
    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(qr_text) {
        if parse_v4_block(&bytes).is_ok() {
            return Ok(bytes);
        }
    }
    Err("invalid QRStream V4 payload".to_string())
}

fn base45_decode(input: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i + 3 <= chars.len() {
        let a = b45_value(chars[i])?;
        let b = b45_value(chars[i + 1])?;
        let c = b45_value(chars[i + 2])?;
        let n = a + b * 45 + c * 2025;
        if n > 0xffff {
            return Err("invalid base45 triplet".to_string());
        }
        out.push(((n >> 8) & 0xff) as u8);
        out.push((n & 0xff) as u8);
        i += 3;
    }

    match chars.len() - i {
        0 => Ok(out),
        2 => {
            let a = b45_value(chars[i])?;
            let b = b45_value(chars[i + 1])?;
            let n = a + b * 45;
            if n > 0xff {
                return Err("invalid base45 tail".to_string());
            }
            out.push(n as u8);
            Ok(out)
        }
        rem => Err(format!("invalid base45 length (remainder {})", rem)),
    }
}

fn b45_value(ch: char) -> Result<u32, String> {
    B45_ALPHABET
        .find(ch)
        .map(|idx| idx as u32)
        .ok_or_else(|| format!("invalid base45 character: {}", ch))
}

fn maybe_decompress(raw: Vec<u8>, compressed: bool) -> Result<Vec<u8>, String> {
    if !compressed {
        return Ok(raw);
    }
    let mut decoder = ZlibDecoder::new(raw.as_slice());
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|err| format!("Decompression failed: {}. Decoded payload may be corrupted.", err))?;
    Ok(out)
}

fn source_index(payload_id: u32, total_symbols: u32, source_blocks: u16) -> Option<u32> {
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

#[pymodule]
fn qrstream_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<V4DecodeSession>()?;
    m.add_class::<DecodeResult>()?;
    m.add_class::<DecodeSnapshot>()?;
    Ok(())
}
