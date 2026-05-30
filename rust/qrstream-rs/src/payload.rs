use base64::Engine;

use crate::protocol_v4::parse_v4_block;

const B45_ALPHABET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

pub(crate) fn decode_qr_payload(qr_text: &str) -> Result<Vec<u8>, String> {
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
