/**
 * WASM decode-core benchmark using the bench vectors produced by the
 * Python encoder. Run: node tests/wasm-bench.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const here = path.dirname(fileURLToPath(import.meta.url));
const wasmPkgDir = path.join(here, '..', 'wasm', 'pkg');
const wasmPath = path.join(wasmPkgDir, 'qrstream_decode_bg.wasm');
const vectorsPath = path.join(here, '..', 'wasm', 'tests', 'bench_vectors.json');

const { default: init, WasmDecodeSession } = await import(
  path.join(wasmPkgDir, 'qrstream_decode.js')
);
await init({ module_or_path: readFileSync(wasmPath) });

const { cases } = JSON.parse(readFileSync(vectorsPath, 'utf8'));
const expected = new Map(cases.map((c) => [c.name, Buffer.from(c.data_b64, 'base64')]));

function fmt(n, digits = 1) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(digits)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(digits)}K`;
  return n.toFixed(digits);
}

function benchCase(c, { warmup = false } = {}) {
  const frames = c.qr_texts_base45;
  // Simulate a realistic camera feed: shuffled, duplicates sprinkled in.
  const feed = frames.slice();
  for (let i = feed.length - 1; i > 0; i--) {
    const j = (i * 7 + 3) % (i + 1);
    [feed[i], feed[j]] = [feed[j], feed[i]];
  }
  for (let i = 0; i < feed.length; i += 7) feed.splice(i, 0, feed[i]); // ~14% dupes

  const s = new WasmDecodeSession();
  const t0 = performance.now();
  let framesFed = 0;
  let done = false;
  for (const text of feed) {
    const r = JSON.parse(s.consume_qr_text(text));
    framesFed++;
    if (r.done) {
      done = true;
      break;
    }
  }
  const t1 = performance.now();
  const ms = t1 - t0;
  const got = Buffer.from(s.result_bytes());
  assert.ok(done, `${c.name}: not done`);
  assert.deepEqual(got, expected.get(c.name), `${c.name}: data mismatch`);
  s.free();

  if (warmup) return { ms, framesFed, payload: got.length };

  // Per-frame cost at a realistic 25fps scan rate (with JSON parse).
  const perFrameUs = (ms / framesFed) * 1000;
  const throughput = (got.length / 1024 / 1024) / (ms / 1000);
  console.log(
    `${c.name.padEnd(12)} payload=${(got.length / 1024).toFixed(0).padStart(5)}KB ` +
    `K=${String(c.K).padStart(4)} frames=${String(framesFed).padStart(4)} ` +
    `total=${ms.toFixed(0).padStart(6)}ms ` +
    `perFrame=${perFrameUs.toFixed(0).padStart(5)}µs ` +
    `throughput=${fmt(throughput, 2).padStart(7)}MB/s`,
  );
  return { ms, framesFed, payload: got.length };
}

// Warmup (JIT + page faulting on the wasm heap).
for (const c of cases) if (c.name === 'bench_warmup') benchCase(c, { warmup: true });

console.log('case          payload     K frames   total  perFrame  throughput');
const results = cases.filter((c) => c.name !== 'bench_warmup').map((c) => benchCase(c));

console.log('\nRealtime headroom check (25 fps camera feed):');
for (const { payload, perFrameUs } of results.map((r, i) => ({
  ...r,
  perFrameUs: (r.ms / r.framesFed) * 1000,
}))) {
  const budget = 40_000; // 40ms = 25fps
  console.log(
    `  ${(payload / 1024).toFixed(0).padStart(5)}KB: ${((perFrameUs / budget) * 100).toFixed(2)}% of frame budget`,
  );
}
