//! WASM entry point for the QRStream V4 (RaptorQ) decode core.
//!
//! Mirrors the Python `DecodeSession` API: feed decoded QR text or raw
//! protocol blocks, poll `snapshot()` for progress, and fetch
//! `result_bytes()` when done.

use wasm_bindgen::prelude::*;

mod base45;
mod protocol;
mod session;

pub use session::{DecodeSession, SessionResult, SessionSnapshot};

#[wasm_bindgen]
pub struct WasmDecodeSession {
    inner: DecodeSession,
}

#[wasm_bindgen]
impl WasmDecodeSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: DecodeSession::new(),
        }
    }

    /// Feed one decoded QR payload string (base45 or base64).
    /// Returns a JSON object describing acceptance/duplication/progress.
    pub fn consume_qr_text(&mut self, qr_text: &str) -> String {
        let result = self.inner.consume_qr_text(qr_text);
        session_result_to_json(&result)
    }

    /// Feed one raw binary V4 protocol block.
    pub fn consume_block(&mut self, block_bytes: &[u8]) -> String {
        let result = self.inner.consume_block(block_bytes);
        session_result_to_json(&result)
    }

    /// Current session state as a JSON object.
    pub fn snapshot(&self) -> String {
        let snap = self.inner.snapshot();
        serde_json::json!({
            "initialized": snap.initialized,
            "done": snap.done,
            "progress": snap.progress,
            "num_recovered": snap.num_recovered,
            "symbol_count": snap.symbol_count,
            "filesize": snap.filesize,
            "protocol_version": snap.protocol_version,
        })
        .to_string()
    }

    /// Reconstructed bytes after completion. Errors if not done yet.
    pub fn result_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.inner.result_bytes().map_err(|e| JsError::new(&e))
    }

    /// Reconstructed bytes, base64-encoded (convenient for JS tests).
    pub fn result_bytes_base64(&self) -> Result<String, JsError> {
        let bytes = self.inner.result_bytes().map_err(|e| JsError::new(&e))?;
        Ok(base64_encode(&bytes))
    }
}

impl Default for WasmDecodeSession {
    fn default() -> Self {
        Self::new()
    }
}

fn session_result_to_json(result: &SessionResult) -> String {
    serde_json::json!({
        "accepted": result.accepted,
        "duplicate": result.duplicate,
        "done": result.done,
        "progress": result.progress,
        "num_recovered": result.num_recovered,
        "symbol_count": result.symbol_count,
        "filesize": result.filesize,
        "protocol_version": result.protocol_version,
        "error": result.error,
    })
    .to_string()
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 0x3F] as char);
        out.push(TABLE[(n >> 12) as usize & 0x3F] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 0x3F] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 0x3F] as char
        } else {
            '='
        });
    }
    out
}
