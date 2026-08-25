/**
 * Node-based smoke test of the WASM decode core using the Python
 * encoder's test vectors. Run: node tests/wasm-smoke.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const here = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.join(here, '..', 'wasm', 'pkg', 'qrstream_decode_bg.wasm');
const vectorsPath = path.join(here, '..', 'wasm', 'tests', 'vectors.json');

const { default: init, WasmDecodeSession } = await import(
  path.join(here, '..', 'wasm', 'pkg', 'qrstream_decode.js')
);
await init({ module_or_path: readFileSync(wasmPath) });

const { cases } = JSON.parse(readFileSync(vectorsPath, 'utf8'));

function b64decode(s) {
  return Buffer.from(s, 'base64');
}

let passed = 0;
for (const c of cases) {
  // 1. In-order base45 delivery
  {
    const s = new WasmDecodeSession();
    let done = false;
    for (const text of c.qr_texts_base45) {
      const r = JSON.parse(s.consume_qr_text(text));
      if (!r.accepted) throw new Error(`${c.name}: frame rejected: ${r.error}`);
      done = done || r.done;
    }
    assert.ok(done, `${c.name}: not done (base45)`);
    const got = s.result_bytes();
    assert.deepEqual(Buffer.from(got), b64decode(c.data_b64), `${c.name}: data mismatch (base45)`);
    const snap = JSON.parse(s.snapshot());
    assert.equal(snap.done, true);
    assert.equal(snap.progress, 1);
    passed++;
  }
  // 2. Shuffled raw-block delivery
  {
    const frames = c.frames.map(b64decode);
    for (let i = frames.length - 1; i > 0; i--) {
      const j = (i * 7 + 3) % (i + 1);
      [frames[i], frames[j]] = [frames[j], frames[i]];
    }
    const s = new WasmDecodeSession();
    let done = false;
    for (const f of frames) {
      const r = JSON.parse(s.consume_block(f));
      if (!r.accepted) throw new Error(`${c.name}: shuffled frame rejected`);
      done = done || r.done;
      if (done) break;
    }
    assert.ok(done, `${c.name}: not done (shuffled)`);
    assert.deepEqual(Buffer.from(s.result_bytes()), b64decode(c.data_b64));
    passed++;
  }
  // 3. Duplicate frame reporting
  {
    const s = new WasmDecodeSession();
    const r1 = JSON.parse(s.consume_qr_text(c.qr_texts_base45[0]));
    const r2 = JSON.parse(s.consume_qr_text(c.qr_texts_base45[0]));
    assert.ok(r1.accepted && !r1.duplicate, `${c.name}: first frame not fresh`);
    assert.ok(r2.accepted && r2.duplicate, `${c.name}: dup not flagged`);
    passed++;
  }
  console.log(`case "${c.name}" OK (filesize=${c.filesize}, K=${c.K}, frames=${c.num_frames})`);
}
console.log(`\nAll ${passed} assertions passed across ${cases.length} cases.`);
