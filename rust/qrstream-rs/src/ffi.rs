use std::sync::Mutex;

use crate::v4_core::{CoreDecodeResult, CoreDecodeSnapshot, CoreV4DecodeSession};

pub struct FfiDecodeResult {
    pub accepted: bool,
    pub duplicate: bool,
    pub done: bool,
    pub progress: f64,
    pub num_recovered: u64,
    pub symbol_count: u64,
    pub filesize: u64,
    pub error_message: Option<String>,
}

impl From<CoreDecodeResult> for FfiDecodeResult {
    fn from(result: CoreDecodeResult) -> Self {
        Self {
            accepted: result.accepted,
            duplicate: result.duplicate,
            done: result.done,
            progress: result.progress,
            num_recovered: result.num_recovered,
            symbol_count: result.symbol_count.unwrap_or(0),
            filesize: result.filesize.unwrap_or(0),
            error_message: result.error,
        }
    }
}

pub struct FfiDecodeSnapshot {
    pub initialized: bool,
    pub done: bool,
    pub progress: f64,
    pub num_recovered: u64,
    pub symbol_count: u64,
    pub filesize: u64,
}

impl From<CoreDecodeSnapshot> for FfiDecodeSnapshot {
    fn from(snapshot: CoreDecodeSnapshot) -> Self {
        Self {
            initialized: snapshot.initialized,
            done: snapshot.done,
            progress: snapshot.progress,
            num_recovered: snapshot.num_recovered,
            symbol_count: snapshot.symbol_count.unwrap_or(0),
            filesize: snapshot.filesize.unwrap_or(0),
        }
    }
}

pub struct FfiV4DecodeSession {
    core: Mutex<CoreV4DecodeSession>,
}

impl FfiV4DecodeSession {
    pub fn new() -> Self {
        Self {
            core: Mutex::new(CoreV4DecodeSession::new()),
        }
    }

    pub fn consume_qr_text(&self, qr_text: String) -> FfiDecodeResult {
        self.core
            .lock()
            .unwrap()
            .consume_qr_text(&qr_text)
            .into()
    }

    pub fn consume_block(&self, block: Vec<u8>) -> FfiDecodeResult {
        self.core.lock().unwrap().consume_block(&block).into()
    }

    pub fn snapshot(&self) -> FfiDecodeSnapshot {
        self.core.lock().unwrap().snapshot().into()
    }

    pub fn result_bytes(&self) -> Vec<u8> {
        self.core
            .lock()
            .unwrap()
            .result_bytes()
            .unwrap_or_default()
    }

    pub fn reset(&self) {
        self.core.lock().unwrap().reset();
    }
}
