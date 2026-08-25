//! End-to-end tests against vectors produced by the Python encoder
//! (`tests/generate_vectors.py` → `tests/vectors.json`).
//!
//! Covers: base45 and base64 QR text paths, raw block paths, compressed
//! streams, shuffled delivery, and lossy (drop-half) delivery.

use base64::Engine;
use qrstream_decode::DecodeSession;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Case {
    name: String,
    compressed: bool,
    data_b64: String,
    payload_b64: String,
    filesize: usize,
    #[serde(rename = "K")]
    k: u32,
    num_frames: usize,
    frames: Vec<String>,
    qr_texts_base45: Vec<String>,
    qr_texts_base64: Vec<String>,
}

fn load_cases() -> Vec<Case> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors.json");
    let raw = std::fs::read_to_string(path).expect("vectors.json missing; run generate_vectors.py");
    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    serde_json::from_value(doc["cases"].clone()).unwrap()
}

fn expected_data(case: &Case) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(&case.data_b64)
        .unwrap()
}

fn decode_frame_b64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}

#[test]
fn all_vectors_in_order_base45() {
    for case in load_cases() {
        let mut session = DecodeSession::new();
        let mut done = false;
        for text in &case.qr_texts_base45 {
            let result = session.consume_qr_text(text);
            assert!(
                result.accepted,
                "{}: frame rejected: {:?}",
                case.name, result.error
            );
            done = result.done;
            if done {
                break;
            }
        }
        assert!(done, "{}: not done after all frames", case.name);
        let bytes = session.result_bytes().unwrap();
        assert_eq!(bytes, expected_data(&case), "{}: data mismatch", case.name);
        let snap = session.snapshot();
        assert!(snap.done);
        assert_eq!(snap.progress, 1.0);
        assert_eq!(snap.symbol_count, Some(case.k));
        // Header filesize = streamed payload size (pre-decompression).
        let payload_len = decode_frame_b64(&case.payload_b64).len();
        assert_eq!(snap.filesize, Some(payload_len as u64));
        assert_eq!(snap.protocol_version, Some(4));
    }
}

#[test]
fn all_vectors_in_order_base64() {
    for case in load_cases() {
        let mut session = DecodeSession::new();
        let mut done = false;
        for text in &case.qr_texts_base64 {
            let result = session.consume_qr_text(text);
            assert!(result.accepted, "{}: frame rejected", case.name);
            done = result.done;
            if done {
                break;
            }
        }
        assert!(done, "{}: not done after all frames", case.name);
        assert_eq!(session.result_bytes().unwrap(), expected_data(&case));
    }
}

#[test]
fn all_vectors_raw_blocks_shuffled() {
    fn pseudo_shuffle<T>(v: &mut Vec<T>) {
        let mut rng_state: u64 = 0x9E3779B97F4A7C15;
        let mut i = v.len();
        while i > 1 {
            i -= 1;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            v.swap(i, j);
        }
    }

    for case in load_cases() {
        let mut frames: Vec<Vec<u8>> = case.frames.iter().map(|f| decode_frame_b64(f)).collect();
        pseudo_shuffle(&mut frames);
        let mut session = DecodeSession::new();
        let mut done = false;
        for frame in &frames {
            let result = session.consume_block(frame);
            assert!(result.accepted, "{}: shuffled frame rejected", case.name);
            done = result.done;
            if done {
                break;
            }
        }
        assert!(done, "{}: not done after shuffled frames", case.name);
        assert_eq!(session.result_bytes().unwrap(), expected_data(&case));
    }
}

#[test]
fn all_vectors_drop_half_frames() {
    for case in load_cases() {
        // Simulate heavy frame loss: keep only every other frame.  Skip
        // cases where the kept count would fall below K (physically
        // undecodable regardless of codec quality).
        let kept: Vec<&String> = case
            .qr_texts_base45
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, s)| s)
            .collect();
        if kept.len() < case.k as usize {
            continue;
        }
        let mut session = DecodeSession::new();
        let mut done = false;
        for text in kept {
            let result = session.consume_qr_text(text);
            assert!(result.accepted);
            done = done || result.done;
        }
        assert!(
            done,
            "{}: not done after dropping half the frames",
            case.name
        );
        assert_eq!(session.result_bytes().unwrap(), expected_data(&case));
    }
}

#[test]
fn duplicate_frames_are_reported() {
    let cases = load_cases();
    let case = cases.iter().find(|c| c.name == "tiny").expect("tiny case");
    let mut session = DecodeSession::new();
    let first = session.consume_qr_text(&case.qr_texts_base45[0]);
    assert!(first.accepted && !first.duplicate);
    let second = session.consume_qr_text(&case.qr_texts_base45[0]);
    assert!(second.accepted && second.duplicate);
    assert!(second.done);
}

#[test]
fn mixed_base45_and_base64_frames() {
    let cases = load_cases();
    let case = cases
        .iter()
        .find(|c| c.name == "medium_50k")
        .expect("medium case");
    let mut session = DecodeSession::new();
    let mut done = false;
    for (i, b45) in case.qr_texts_base45.iter().enumerate() {
        if i % 2 == 0 {
            let r = session.consume_qr_text(b45);
            assert!(r.accepted);
            done = done || r.done;
        } else {
            let r = session.consume_qr_text(&case.qr_texts_base64[i]);
            assert!(r.accepted);
            done = done || r.done;
        }
        if done {
            break;
        }
    }
    assert!(done, "medium: not done with mixed encodings");
    assert_eq!(session.result_bytes().unwrap(), expected_data(case));
}

#[test]
fn corrupt_frame_rejected_not_fatal() {
    let cases = load_cases();
    let case = cases
        .iter()
        .find(|c| c.name == "medium_50k")
        .expect("medium case");
    let mut session = DecodeSession::new();
    // Feed first valid frame.
    let r = session.consume_qr_text(&case.qr_texts_base45[0]);
    assert!(r.accepted);
    // Feed a corrupted copy (flip a data byte inside the base45 text).
    let mut corrupted = case.qr_texts_base45[1].clone();
    corrupted.replace_range(
        30..31,
        if corrupted.as_bytes()[30] == b'A' {
            "B"
        } else {
            "A"
        },
    );
    let r = session.consume_qr_text(&corrupted);
    // CRC should catch it -> rejected with error, session still usable.
    assert!(!r.accepted, "corrupted frame must be rejected");
    assert!(r.error.is_some());
    // Feed the real frame 1 and continue to completion.
    for text in case.qr_texts_base45.iter().skip(1) {
        let r = session.consume_qr_text(text);
        assert!(r.accepted);
        if r.done {
            assert_eq!(session.result_bytes().unwrap(), expected_data(case));
            return;
        }
    }
    panic!("not done after valid frames");
}
