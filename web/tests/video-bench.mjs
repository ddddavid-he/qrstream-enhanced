/**
 * End-to-end QR video benchmark: ffmpeg decode/resize -> zxing-wasm QR
 * detection -> QRStream WASM session. The baseline and balance fixtures must
 * both complete above the configured processing FPS gate.
 *
 * Usage:
 *   node tests/video-bench.mjs /path/to/baseline.MOV /path/to/balance.MOV
 *   node tests/video-bench.mjs --min-fps 30 --width 1280 video1.MOV ...
 */
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const wasmPkgDir = path.join(here, '..', 'wasm', 'pkg');
const wasmPath = path.join(wasmPkgDir, 'qrstream_decode_bg.wasm');
const zxingWasmPath = fileURLToPath(
  import.meta.resolve('zxing-wasm/reader/zxing_reader.wasm'),
);

// Generated independently with the Python video decoder. This makes the test
// check reconstructed content, not merely RaptorQ's completion flag.
const EXPECTED_SHA256 = new Map([
  ['baseline.MOV', 'f2eaccaba1e9486b0c2d254f2a2287bfbd2154363f3a83f3ff8ee1ab10a24b9c'],
  ['balance.MOV', 'a81b74ae441b9f8ef5b7f1539a26889d66266c7e6d16e455b2406ec712d40d1b'],
  ['dense.MOV', '92bdfb16a6c09d3e09d4f58800de74c0d50d559c3c8cda894839a298cfa61181'],
  ['throughput.MOV', 'addbc33a94277c548d5b42fbcf3c7d31f6275a4024efe16724e6d58331706a6b'],
]);

let minFps = 30;
let width = 1080;
const videos = [];
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === '--min-fps') minFps = Number(process.argv[++i]);
  else if (process.argv[i] === '--width') width = Number(process.argv[++i]);
  else videos.push(path.resolve(process.argv[i]));
}
if (videos.length === 0) {
  for (const name of ['baseline', 'balance', 'dense', 'throughput']) {
    videos.push(path.join(homedir(), 'Downloads', `${name}.MOV`));
  }
}
for (const video of videos) {
  if (!existsSync(video)) throw new Error(`Video not found: ${video}`);
}

globalThis.ImageData = class ImageData {
  constructor(data, frameWidth, frameHeight) {
    this.data = data;
    this.width = frameWidth;
    this.height = frameHeight;
    this.colorSpace = 'srgb';
  }
};

const [{ default: init, WasmDecodeSession }, zxing] = await Promise.all([
  import(path.join(wasmPkgDir, 'qrstream_decode.js')),
  import('zxing-wasm/reader'),
]);
await init({ module_or_path: readFileSync(wasmPath) });
await zxing.prepareZXingModule({
  overrides: { wasmBinary: readFileSync(zxingWasmPath) },
  fireImmediately: true,
});

function dimensions(video) {
  const probe = spawnSync(
    'ffprobe',
    [
      '-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration:stream_side_data=rotation',
      '-of', 'json', video,
    ],
    { encoding: 'utf8' },
  );
  if (probe.status !== 0) throw new Error(probe.stderr || 'ffprobe failed');
  const stream = JSON.parse(probe.stdout).streams[0];
  const rotation = stream.side_data_list?.find((item) => item.rotation != null)?.rotation ?? 0;
  const rotated = Math.abs(rotation) % 180 === 90;
  const displayWidth = rotated ? Number(stream.height) : Number(stream.width);
  const displayHeight = rotated ? Number(stream.width) : Number(stream.height);
  const height = Math.round((displayHeight * width) / displayWidth / 2) * 2;
  const [num, den] = stream.r_frame_rate.split('/').map(Number);
  return {
    width,
    height,
    sourceFps: num / den,
    expectedFrames: Number(stream.nb_frames),
    duration: Number(stream.duration),
  };
}

async function benchmark(video) {
  const info = dimensions(video);
  const frameBytes = info.width * info.height * 4;
  const ffmpeg = spawn(
    'ffmpeg',
    [
      '-v', 'error', '-i', video,
      '-vf', `scale=${info.width}:${info.height}:flags=fast_bilinear`,
      '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1',
    ],
    { stdio: ['ignore', 'pipe', 'pipe'] },
  );

  let pending = Buffer.alloc(0);
  let stderr = '';
  let frames = 0;
  let detected = 0;
  let accepted = 0;
  let duplicates = 0;
  let invalid = 0;
  let doneAtFrame = null;
  let detectMs = 0;
  let feedMs = 0;
  const session = new WasmDecodeSession();
  ffmpeg.stderr.setEncoding('utf8');
  ffmpeg.stderr.on('data', (chunk) => { stderr += chunk; });

  const startedAt = performance.now();
  for await (const chunk of ffmpeg.stdout) {
    pending = pending.length === 0 ? chunk : Buffer.concat([pending, chunk]);
    while (pending.length >= frameBytes) {
      const raw = pending.subarray(0, frameBytes);
      pending = pending.subarray(frameBytes);
      const pixels = new Uint8ClampedArray(raw.buffer, raw.byteOffset, frameBytes);
      const image = new ImageData(pixels, info.width, info.height);

      const detectStart = performance.now();
      const results = await zxing.readBarcodesFromImageData(image, {
        formats: ['QRCode'],
        tryHarder: true,
      });
      detectMs += performance.now() - detectStart;
      frames++;

      if (results.length > 0) detected++;
      for (const result of results) {
        const feedStart = performance.now();
        const state = JSON.parse(session.consume_qr_text(result.text));
        feedMs += performance.now() - feedStart;
        if (state.accepted && !state.duplicate) accepted++;
        if (state.duplicate) duplicates++;
        if (!state.accepted) invalid++;
        if (state.done && doneAtFrame === null) doneAtFrame = frames;
      }
    }
  }
  const endedAt = performance.now();
  const exitCode = await new Promise((resolve) => ffmpeg.on('close', resolve));
  if (exitCode !== 0) throw new Error(`ffmpeg failed (${exitCode}): ${stderr}`);

  const snapshot = JSON.parse(session.snapshot());
  const elapsedMs = endedAt - startedAt;
  const processingFps = frames / (elapsedMs / 1000);
  const detectorFps = frames / (detectMs / 1000);
  const hitRate = frames === 0 ? 0 : detected / frames;
  const output = snapshot.done ? Buffer.from(session.result_bytes()) : Buffer.alloc(0);
  const resultBytes = output.length;
  const sha256 = createHash('sha256').update(output).digest('hex');
  session.free();

  return {
    name: path.basename(video),
    ...info,
    frames,
    detected,
    accepted,
    duplicates,
    invalid,
    done: snapshot.done,
    doneAtFrame,
    progress: snapshot.progress,
    resultBytes,
    sha256,
    elapsedMs,
    detectMs,
    feedMs,
    processingFps,
    detectorFps,
    hitRate,
  };
}

const results = [];
for (const video of videos) {
  process.stdout.write(`Benchmarking ${path.basename(video)} at ${width}px... `);
  const result = await benchmark(video);
  results.push(result);
  console.log(
    `${result.processingFps.toFixed(1)} FPS, ` +
      `QR hit ${(result.hitRate * 100).toFixed(1)}%, ` +
      `accepted ${result.accepted}, done=${result.done}`,
  );
}

console.table(
  results.map((r) => ({
    video: r.name,
    frames: r.frames,
    source_fps: r.sourceFps.toFixed(2),
    processing_fps: r.processingFps.toFixed(2),
    detector_fps: r.detectorFps.toFixed(2),
    qr_hit_rate: `${(r.hitRate * 100).toFixed(1)}%`,
    accepted: r.accepted,
    duplicates: r.duplicates,
    invalid: r.invalid,
    done_at_frame: r.doneAtFrame ?? '-',
    progress: `${(r.progress * 100).toFixed(1)}%`,
    result_bytes: r.resultBytes,
    integrity: EXPECTED_SHA256.has(r.name) ? 'verified' : 'unchecked',
  })),
);

for (const result of results) {
  assert.equal(result.done, true, `${result.name}: QRStream decode incomplete`);
  const expectedHash = EXPECTED_SHA256.get(result.name);
  if (expectedHash) {
    assert.equal(result.sha256, expectedHash, `${result.name}: reconstructed bytes differ`);
  } else {
    console.warn(`WARN: ${result.name} has no expected hash; integrity was not verified.`);
  }
  assert.ok(
    result.processingFps > minFps,
    `${result.name}: ${result.processingFps.toFixed(2)} FPS <= ${minFps} FPS requirement`,
  );
}
console.log(`PASS: ${results.length} videos decoded above ${minFps} FPS.`);
