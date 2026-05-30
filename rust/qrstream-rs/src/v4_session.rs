use std::collections::HashSet;
use std::io::Read;

use flate2::read::ZlibDecoder;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use raptorq::{Decoder, EncodingPacket, ObjectTransmissionInformation, PayloadId};

use crate::payload::decode_qr_payload;
use crate::protocol_v4::{parse_v4_block, source_index, V4Block, V4_VERSION};

#[pyclass]
#[derive(Clone)]
pub(crate) struct DecodeResult {
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
pub(crate) struct DecodeSnapshot {
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
pub(crate) struct V4DecodeSession {
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
