/** Exercise the real App lifecycle with camera/worker doubles, without camera hardware. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const source = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8')
  .split('const app = new App();')[0] + '\nexports.App = App;';
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;
const deferred = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
};

for (const state of ['idle', 'scanning', 'paused']) {
  const stream = {};
  const video = { srcObject: stream, cancelVideoFrameCallback() {} };
  const root = { querySelector: () => video };
  let cameraStops = 0, cameraOpens = 0, resets = 0, progressUpdates = 0, metricsUpdates = 0;
  const reset = deferred(), consume = deferred(), snapshot = deferred();
  class Ui {
    setControlState(value) { this.state = value; }
    setStatus(value) { this.status = value; }
    resetSession() { resets++; }
    updateProgress() { progressUpdates++; }
    updateRuntimeMetrics() { metricsUpdates++; }
    markReady() { throw new Error('stop must preserve camera readiness'); }
    setCameraInactive() { throw new Error('stop must preserve camera controls'); }
  }
  class MetricsTracker { reset() {} }
  const context = {
    exports: {}, performance, document: { querySelector: () => root },
    window: { setInterval: () => 1 }, clearInterval() {},
    require(name) {
      if (name === './ui') return { Ui };
      if (name === './metrics') return { MetricsTracker };
      if (name === './camera') return { startCamera: () => { cameraOpens++; } };
      return {};
    },
  };
  vm.runInNewContext(compiled, context);
  const app = new context.exports.App();
  app.camera = { stop: () => { cameraStops++; }, videoTrack: {} };
  const camera = app.camera;
  app.decoder = { reset: () => reset.promise, consume: () => consume.promise, snapshot: () => snapshot.promise };
  app.detector = { name: 'fixture' };
  app.setState('scanning');
  const pendingFeed = app.feed('old block');
  const pendingMetrics = app.publishMetrics();
  app.setState(state);
  const stopping = app.stopScanning();
  assert.equal(app.state, 'starting', 'restart must wait for reset acknowledgement');
  await app.startScanning();
  assert.equal(cameraOpens, 0);
  consume.resolve({ accepted: true, duplicate: false, done: false });
  snapshot.resolve({ num_received: 123 });
  await Promise.all([pendingFeed, pendingMetrics]);
  assert.equal(progressUpdates, 0, 'old decode results cannot repopulate progress');
  assert.equal(metricsUpdates, 0, 'old snapshots cannot repopulate counters');
  reset.resolve(); await stopping;
  assert.equal(resets, 1, 'stop clears the session immediately');
  assert.equal(app.pendingTexts.size, 0);
  assert.equal(app.state, 'idle');
  assert.equal(app.camera, camera);
  assert.equal(video.srcObject, stream);
  assert.equal(cameraStops, 0, 'camera tracks must remain live');
  let scheduled = 0;
  app.scheduleNextFrame = () => { scheduled++; };
  await app.startScanning();
  assert.equal(app.state, 'scanning');
  assert.equal(scheduled, 1);
  assert.equal(cameraOpens, 0, 'restart reuses the preview and resolution');
  console.log(`PASS: stop from ${state} clears session, retains preview, rejects stale results and restarts`);
}
