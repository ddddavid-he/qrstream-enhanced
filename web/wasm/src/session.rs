//! Platform-neutral stateful decode session (WASM port of
//! `qrstream.decode_session.DecodeSession`, V4/RaptorQ only).

use std::collections::HashSet;

use raptorq::{Decoder as RqDecoder, EncodingPacket};

use crate::protocol::{self, V4Block, V4Header};

#[derive(Debug, Clone, PartialEq)]
pub struct SessionResult {
    pub accepted: bool,
    pub duplicate: bool,
    pub done: bool,
    pub progress: f64,
    pub num_recovered: u32,
    pub symbol_count: Option<u32>,
    pub filesize: Option<u64>,
    pub protocol_version: Option<u8>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionSnapshot {
    pub initialized: bool,
    pub done: bool,
    pub progress: f64,
    pub num_recovered: u32,
    pub symbol_count: Option<u32>,
    pub filesize: Option<u64>,
    pub protocol_version: Option<u8>,
}

#[derive(Debug)]
struct RaptorQState {
    filesize: u64,
    symbol_size: u16,
    k: u32,
    source_blocks: u32,
    compressed: bool,
    protocol_version: u8,
    rq_decoder: RqDecoder,
    result: Option<Vec<u8>>,
    /// Global source-symbol indices confirmed received (systematic only).
    eliminated: HashSet<u32>,
}

impl RaptorQState {
    fn new(header: &V4Header) -> Self {
        let padded_len = (header.symbol_count as u64) * (header.symbol_size as u64);
        Self {
            filesize: header.filesize,
            symbol_size: header.symbol_size,
            k: header.symbol_count,
            source_blocks: if header.reserved > 0 {
                header.reserved as u32
            } else {
                1
            },
            compressed: header.compressed,
            protocol_version: header.version,
            rq_decoder: RqDecoder::new(raptorq::ObjectTransmissionInformation::with_defaults(
                padded_len,
                header.symbol_size,
            )),
            result: None,
            eliminated: HashSet::new(),
        }
    }

    fn done(&self) -> bool {
        self.result.is_some()
    }

    fn progress(&self) -> f64 {
        if self.k == 0 {
            return 0.0;
        }
        if self.done() {
            return 1.0;
        }
        (self.eliminated.len() as f64 / self.k as f64).min(0.99)
    }

    fn num_recovered(&self) -> u32 {
        if self.done() {
            self.k
        } else {
            self.eliminated.len() as u32
        }
    }

    fn ensure_header_match(&self, header: &V4Header) -> Result<(), String> {
        if header.version != self.protocol_version {
            return Err(format!(
                "version mismatch: {} != {}",
                header.version, self.protocol_version
            ));
        }
        if header.filesize != self.filesize {
            return Err(format!(
                "filesize mismatch: {} != {}",
                header.filesize, self.filesize
            ));
        }
        if header.symbol_size != self.symbol_size {
            return Err(format!(
                "symbol_size mismatch: {} != {}",
                header.symbol_size, self.symbol_size
            ));
        }
        if header.symbol_count != self.k {
            return Err(format!(
                "symbol_count mismatch: {} != {}",
                header.symbol_count, self.k
            ));
        }
        let source_blocks = if header.reserved > 0 {
            header.reserved as u32
        } else {
            1
        };
        if source_blocks != self.source_blocks {
            return Err(format!(
                "source_blocks mismatch: {source_blocks} != {}",
                self.source_blocks
            ));
        }
        if header.compressed != self.compressed {
            return Err(format!(
                "compressed flag mismatch: {} != {}",
                header.compressed, self.compressed
            ));
        }
        Ok(())
    }

    /// Feed one parsed V4 block. Returns `done`.
    fn consume_block(&mut self, block: &V4Block) -> bool {
        if self.done() {
            return true;
        }
        let symbol_size = self.symbol_size as usize;
        let mut data: &[u8] = &block.data;
        if data.len() > symbol_size {
            data = &data[..symbol_size];
        }
        // Reconstruct the raptorq packet: 4-byte PayloadId + symbol data
        // (padded to symbol_size when short, mirroring the Python decoder).
        let mut pkt = block.header.esi.to_be_bytes().to_vec();
        pkt.extend_from_slice(data);
        pkt.resize(4 + symbol_size, 0);
        self.feed_packet(pkt)
    }

    fn feed_packet(&mut self, pkt: Vec<u8>) -> bool {
        if pkt.len() < 4 {
            return self.done();
        }
        let packet = EncodingPacket::deserialize(&pkt);
        let payload_id = u32::from_be_bytes(pkt[..4].try_into().unwrap());
        // Track systematic source-symbol reception.
        if let Some(idx) = protocol::source_index(payload_id, self.k, self.source_blocks) {
            self.eliminated.insert(idx);
        }
        if let Some(result) = self.rq_decoder.decode(packet) {
            let result = &result[..self.filesize as usize];
            self.result = Some(result.to_vec());
            true
        } else {
            self.done()
        }
    }

    /// Raw reconstructed (still compressed) bytes, trimmed to filesize.
    fn raw_bytes(&self) -> Option<&[u8]> {
        self.result.as_deref()
    }
}

#[derive(Debug, Default)]
pub struct DecodeSession {
    state: Option<RaptorQState>,
    seen_blocks: HashSet<(u8, u32)>,
}

impl DecodeSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume one raw V4 protocol block (binary).
    pub fn consume_block(&mut self, block_bytes: &[u8]) -> SessionResult {
        let block = match protocol::unpack(block_bytes) {
            Ok(b) => b,
            Err(e) => return self.result(false, false, Some(e), None),
        };

        let block_id = (block.header.version, block.header.esi);
        if self.seen_blocks.contains(&block_id) {
            return self.result(true, true, None, None);
        }

        if self.state.is_none() {
            self.state = Some(RaptorQState::new(&block.header));
        }

        let done = match self.state.as_mut() {
            Some(state) => match state.ensure_header_match(&block.header) {
                Ok(()) => state.consume_block(&block),
                Err(e) => return self.result(false, false, Some(e), None),
            },
            None => unreachable!(),
        };

        self.seen_blocks.insert(block_id);
        self.result(true, false, None, Some(done))
    }

    /// Consume one decoded QR payload string (base45 or base64).
    pub fn consume_qr_text(&mut self, qr_text: &str) -> SessionResult {
        let block_bytes = match decode_qr_payload(qr_text) {
            Some(b) => b,
            None => {
                return self.result(false, false, Some("invalid QRStream payload".into()), None)
            }
        };
        self.consume_block(&block_bytes)
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        match &self.state {
            None => SessionSnapshot {
                initialized: false,
                done: false,
                progress: 0.0,
                num_recovered: 0,
                symbol_count: None,
                filesize: None,
                protocol_version: None,
            },
            Some(state) => SessionSnapshot {
                initialized: true,
                done: state.done(),
                progress: state.progress(),
                num_recovered: state.num_recovered(),
                symbol_count: Some(state.k),
                filesize: Some(state.filesize),
                protocol_version: Some(state.protocol_version),
            },
        }
    }

    /// Reconstructed bytes after completion (decompressing if flagged).
    pub fn result_bytes(&self) -> Result<Vec<u8>, String> {
        let state = self
            .state
            .as_ref()
            .ok_or("Decoding incomplete — no result available")?;
        let raw = state
            .raw_bytes()
            .ok_or("Decoding incomplete — no result available")?;
        if state.compressed {
            decompress_zlib(raw).map_err(|e| {
                format!("Decompression failed: {e}. Decoded payload may be corrupted.")
            })
        } else {
            Ok(raw.to_vec())
        }
    }

    fn result(
        &self,
        accepted: bool,
        duplicate: bool,
        error: Option<String>,
        done_override: Option<bool>,
    ) -> SessionResult {
        let snapshot = self.snapshot();
        let done = done_override.unwrap_or(snapshot.done);
        SessionResult {
            accepted,
            duplicate,
            done,
            progress: snapshot.progress,
            num_recovered: snapshot.num_recovered,
            symbol_count: snapshot.symbol_count,
            filesize: snapshot.filesize,
            protocol_version: snapshot.protocol_version,
            error,
        }
    }
}

fn decode_qr_payload(qr_text: &str) -> Option<Vec<u8>> {
    // base45 first (high-density mode), then base64 fallback.
    if let Ok(candidate) = crate::base45::decode(qr_text.as_bytes()) {
        if protocol::unpack(&candidate).is_ok() {
            return Some(candidate);
        }
    }
    if let Ok(candidate) = base64_decode(qr_text) {
        if protocol::unpack(&candidate).is_ok() {
            return Some(candidate);
        }
    }
    None
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rev = [255u8; 256];
    for (i, &c) in TABLE.iter().enumerate() {
        rev[c as usize] = i as u8;
    }
    let input: Vec<u8> = input
        .bytes()
        .filter(|&b| b != b'=' && b != b'\n' && b != b'\r')
        .collect();
    if input.len() % 4 == 1 {
        return Err("invalid base64 length".into());
    }
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut chunk = [0u8; 4];
    let mut n: usize = 0;
    for &c in &input {
        let d = rev[c as usize];
        if d == 255 {
            return Err(format!("invalid base64 character: {c:?}"));
        }
        chunk[n] = d;
        n += 1;
        if n == 4 {
            n = 0;
            let bits = ((chunk[0] as u32) << 18)
                | ((chunk[1] as u32) << 12)
                | ((chunk[2] as u32) << 6)
                | chunk[3] as u32;
            out.push((bits >> 16) as u8);
            out.push((bits >> 8) as u8);
            out.push(bits as u8);
        }
    }
    match n {
        0 => {}
        2 => {
            let bits = ((chunk[0] as u32) << 18) | ((chunk[1] as u32) << 12);
            out.push((bits >> 16) as u8);
        }
        3 => {
            let bits =
                ((chunk[0] as u32) << 18) | ((chunk[1] as u32) << 12) | ((chunk[2] as u32) << 6);
            out.push((bits >> 16) as u8);
            out.push((bits >> 8) as u8);
        }
        _ => unreachable!(),
    }
    Ok(out)
}

fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decode_works() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello".to_vec());
        assert_eq!(base64_decode("aGVsbG8h").unwrap(), b"hello!".to_vec());
        assert_eq!(base64_decode("aGVsbG8hIQ==").unwrap(), b"hello!!".to_vec());
    }

    #[test]
    fn fresh_session_snapshot() {
        let session = DecodeSession::new();
        let snap = session.snapshot();
        assert!(!snap.initialized);
        assert_eq!(snap.progress, 0.0);
    }

    #[test]
    fn invalid_payload_rejected() {
        let mut session = DecodeSession::new();
        let result = session.consume_qr_text("not-a-qrstream-payload!!");
        assert!(!result.accepted);
        assert_eq!(result.error.as_deref(), Some("invalid QRStream payload"));
    }

    #[test]
    fn result_bytes_before_done_errors() {
        let session = DecodeSession::new();
        assert!(session.result_bytes().is_err());
    }
}
