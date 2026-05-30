use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::v4_core::{CoreDecodeResult, CoreDecodeSnapshot, CoreV4DecodeSession};

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
    num_recovered: u64,
    #[pyo3(get)]
    symbol_count: Option<u64>,
    #[pyo3(get)]
    filesize: Option<u64>,
    #[pyo3(get)]
    protocol_version: Option<u8>,
    #[pyo3(get)]
    error: Option<String>,
}

impl From<CoreDecodeResult> for DecodeResult {
    fn from(result: CoreDecodeResult) -> Self {
        Self {
            accepted: result.accepted,
            duplicate: result.duplicate,
            done: result.done,
            progress: result.progress,
            num_recovered: result.num_recovered,
            symbol_count: result.symbol_count,
            filesize: result.filesize,
            protocol_version: result.protocol_version,
            error: result.error,
        }
    }
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
    num_recovered: u64,
    #[pyo3(get)]
    symbol_count: Option<u64>,
    #[pyo3(get)]
    filesize: Option<u64>,
    #[pyo3(get)]
    protocol_version: Option<u8>,
}

impl From<CoreDecodeSnapshot> for DecodeSnapshot {
    fn from(snapshot: CoreDecodeSnapshot) -> Self {
        Self {
            initialized: snapshot.initialized,
            done: snapshot.done,
            progress: snapshot.progress,
            num_recovered: snapshot.num_recovered,
            symbol_count: snapshot.symbol_count,
            filesize: snapshot.filesize,
            protocol_version: snapshot.protocol_version,
        }
    }
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
    core: CoreV4DecodeSession,
}

#[pymethods]
impl V4DecodeSession {
    #[new]
    fn new() -> Self {
        Self {
            core: CoreV4DecodeSession::new(),
        }
    }

    fn consume_qr_text(&mut self, qr_text: &str) -> DecodeResult {
        self.core.consume_qr_text(qr_text).into()
    }

    fn consume_block(&mut self, block_bytes: &[u8]) -> DecodeResult {
        self.core.consume_block(block_bytes).into()
    }

    fn consume_blocks(&mut self, blocks: Vec<Vec<u8>>) -> DecodeResult {
        self.core.consume_blocks(blocks).into()
    }

    fn snapshot(&self) -> DecodeSnapshot {
        self.core.snapshot().into()
    }

    fn result_bytes(&self) -> PyResult<Vec<u8>> {
        self.core
            .result_bytes()
            .map_err(|_| PyRuntimeError::new_err("Decoding incomplete — no result available"))
    }
}
