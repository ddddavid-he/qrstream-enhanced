use std::collections::HashSet;
use std::io::Read;

use flate2::read::ZlibDecoder;
use raptorq::{Decoder, EncodingPacket, ObjectTransmissionInformation, PayloadId};

use crate::payload::decode_qr_payload;
use crate::protocol_v4::{parse_v4_block, source_index, V4Block, V4_VERSION};

#[derive(Clone, Debug)]
pub(crate) struct CoreDecodeResult {
    pub(crate) accepted: bool,
    pub(crate) duplicate: bool,
    pub(crate) done: bool,
    pub(crate) progress: f64,
    pub(crate) num_recovered: u64,
    pub(crate) symbol_count: Option<u64>,
    pub(crate) filesize: Option<u64>,
    pub(crate) protocol_version: Option<u8>,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct CoreDecodeSnapshot {
    pub(crate) initialized: bool,
    pub(crate) done: bool,
    pub(crate) progress: f64,
    pub(crate) num_recovered: u64,
    pub(crate) symbol_count: Option<u64>,
    pub(crate) filesize: Option<u64>,
    pub(crate) protocol_version: Option<u8>,
}

pub(crate) struct CoreV4DecodeSession {
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

impl CoreV4DecodeSession {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn consume_qr_text(&mut self, qr_text: &str) -> CoreDecodeResult {
        match decode_qr_payload(qr_text) {
            Ok(block) => self.consume_block(block.as_slice()),
            Err(err) => self.result(false, false, Some(err)),
        }
    }

    pub(crate) fn consume_block(&mut self, block_bytes: &[u8]) -> CoreDecodeResult {
        match self.consume_block_inner(block_bytes) {
            Ok(result) => result,
            Err(err) => self.result(false, false, Some(err)),
        }
    }

    pub(crate) fn consume_blocks(&mut self, blocks: Vec<Vec<u8>>) -> CoreDecodeResult {
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

    pub(crate) fn snapshot(&self) -> CoreDecodeSnapshot {
        CoreDecodeSnapshot {
            initialized: self.initialized,
            done: self.done,
            progress: self.progress(),
            num_recovered: self.num_recovered() as u64,
            symbol_count: self.initialized.then_some(self.symbol_count as u64),
            filesize: self.initialized.then_some(self.filesize),
            protocol_version: self.initialized.then_some(V4_VERSION),
        }
    }

    pub(crate) fn result_bytes(&self) -> Result<Vec<u8>, String> {
        if !self.done {
            return Err("Decoding incomplete — no result available".to_string());
        }
        Ok(self.result.clone().unwrap_or_default())
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::new();
    }

    fn consume_block_inner(&mut self, block_bytes: &[u8]) -> Result<CoreDecodeResult, String> {
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

    fn result(&self, accepted: bool, duplicate: bool, error: Option<String>) -> CoreDecodeResult {
        CoreDecodeResult {
            accepted,
            duplicate,
            done: self.done,
            progress: self.progress(),
            num_recovered: self.num_recovered() as u64,
            symbol_count: self.initialized.then_some(self.symbol_count as u64),
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
