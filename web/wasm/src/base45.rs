//! Base45 (RFC 9285) decoding for QR alphanumeric-mode payloads.
//!
//! 2 raw bytes -> 3 ASCII chars from the 45-char QR alphanumeric alphabet.

const ALPHABET: &[u8; 45] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

/// Reverse lookup table: ASCII byte value (0-127) -> digit value, or 0xFF for invalid.
static DECODE_TABLE: [u8; 128] = {
    let mut table = [0xFFu8; 128];
    let mut i = 0;
    while i < 45 {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

#[inline]
fn digit_of(c: u8) -> Option<u8> {
    if c < 128 {
        let d = DECODE_TABLE[c as usize];
        if d != 0xFF {
            return Some(d);
        }
    }
    None
}

/// Decode a base45 string (as raw ASCII/UTF-8 bytes) back to raw bytes.
pub fn decode(input: &[u8]) -> Result<Vec<u8>, String> {
    // Validate ASCII first so multi-byte UTF-8 sequences are rejected cleanly.
    if !input.is_ascii() {
        return Err("base45 input is not ASCII".to_string());
    }
    let mut out = Vec::with_capacity(input.len() * 2 / 3 + 1);
    let len = input.len();
    let mut i = 0;
    while i + 3 <= len {
        let a = digit_of(input[i]).ok_or_else(|| invalid_char(input[i]))?;
        let b = digit_of(input[i + 1]).ok_or_else(|| invalid_char(input[i + 1]))?;
        let c = digit_of(input[i + 2]).ok_or_else(|| invalid_char(input[i + 2]))?;
        let n = a as u32 + (b as u32) * 45 + (c as u32) * 2025;
        if n > 0xFFFF {
            return Err("invalid base45 triplet".to_string());
        }
        out.push((n >> 8) as u8);
        out.push((n & 0xFF) as u8);
        i += 3;
    }
    let remaining = len - i;
    match remaining {
        0 => {}
        2 => {
            let a = digit_of(input[i]).ok_or_else(|| invalid_char(input[i]))?;
            let b = digit_of(input[i + 1]).ok_or_else(|| invalid_char(input[i + 1]))?;
            let n = a as u32 + (b as u32) * 45;
            if n > 0xFF {
                return Err("invalid base45 tail".to_string());
            }
            out.push(n as u8);
        }
        r => return Err(format!("invalid base45 length (remainder {r})")),
    }
    Ok(out)
}

fn invalid_char(c: u8) -> String {
    format!("invalid base45 character: {c:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(data: &[u8]) -> String {
        let mut out = String::new();
        let mut i = 0;
        while i + 2 <= data.len() {
            let n = ((data[i] as u32) << 8) | data[i + 1] as u32;
            let c = n / 2025;
            let n = n - c * 2025;
            let b = n / 45;
            let a = n - b * 45;
            out.push(ALPHABET[a as usize] as char);
            out.push(ALPHABET[b as usize] as char);
            out.push(ALPHABET[c as usize] as char);
            i += 2;
        }
        if i < data.len() {
            let n = data[i] as u32;
            let b = n / 45;
            let a = n - b * 45;
            out.push(ALPHABET[a as usize] as char);
            out.push(ALPHABET[b as usize] as char);
        }
        out
    }

    #[test]
    fn rfc9285_examples() {
        // "AB" -> "BB8", "Hello!!" -> "%69 VD92EX0"
        assert_eq!(decode(b"BB8").unwrap(), b"AB".to_vec());
        assert_eq!(decode(b"%69 VD92EX0").unwrap(), b"Hello!!".to_vec());
        // RFC: "qi%ta" is invalid
        assert!(decode(b"qi%ta").is_err());
    }

    #[test]
    fn roundtrip_various_lengths() {
        for len in [0usize, 1, 2, 3, 10, 31, 32, 33, 100, 255] {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            let s = encode(&data);
            assert_eq!(decode(s.as_bytes()).unwrap(), data, "len={len}");
        }
    }

    #[test]
    fn rejects_invalid() {
        assert!(decode(b"A").is_err()); // remainder 1
        assert!(decode(b"AAAAB").is_err()); // remainder 2 but... actually remainder 2 is ok
        assert!(decode(&[0xFF]).is_err()); // non-ASCII
    }
}
