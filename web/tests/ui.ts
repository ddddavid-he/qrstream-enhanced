/** Browser regression harness: /tests/ui.html on the Vite server. No camera permission needed. */
import { Ui } from '../src/ui';
import '../src/styles.css';
import type { RuntimeMetrics } from '../src/metrics';

const markup = await fetch(new URL('../index.html', import.meta.url)).then(r => r.text());
const root = new DOMParser().parseFromString(markup, 'text/html').querySelector<HTMLElement>('#app')!;
document.body.append(root);
const node = <T extends HTMLElement = HTMLElement>(id: string) => root.querySelector<T>(`#${id}`)!;
const click = (id: string) => node(id).click();
const tick = () => new Promise(resolve => setTimeout(resolve, 30));
const assert = (condition: unknown, description: string) => { if (!condition) throw new Error(description); };
const sheet = node<HTMLDialogElement>('sheet');
let toggles = 0, stops = 0;
const downloads: { filename: string; href: string }[] = [];
const interceptDownload = (event: MouseEvent) => {
  if (event.target instanceof HTMLAnchorElement && event.target.hasAttribute('download')) {
    event.preventDefault();
    downloads.push({ filename: event.target.download, href: event.target.href });
  }
};
document.addEventListener('click', interceptDownload, true);
const options = [
  { id: '720p' as const, label: '720p', width: 1280, height: 720 },
  { id: '1080p' as const, label: '1080p', width: 1920, height: 1080 },
];
const ui = new Ui(root, {
  onToggle: () => { toggles++; ui.setControlState(root.dataset.phase === 'scanning' ? 'paused' : 'scanning'); },
  onStop: () => { stops++; ui.resetSession(); ui.setControlState('idle'); },
  onDownload: () => ui.download(),
  onResolutionChange: id => {
    ui.setResolutionBusy(true);
    const selected = options.find(option => option.id === id)!;
    ui.setCameraSettings(selected);
    ui.setResolutionOptions(options, selected);
    ui.setResolutionBusy(false);
  },
});
const progress = { initialized: true, done: false, progress: .64, num_recovered: 303, num_received: 303, symbol_count: 473, filesize: 1258291, protocol_version: 4 };
const metrics: RuntimeMetrics = { detector: 'Test detector', elapsedSeconds: 18, cameraFps: 60, scanFps: 38, qrFps: 35, acceptedFps: 24, payloadBytesPerSecond: 60000, hitRate: .92, callbackFrames: 500, scans: 480, qrFrames: 440, qrTexts: 440, accepted: 303, duplicates: 42, invalid: 3, detectAverageMs: 17, detectP95Ms: 21.6, decodeAverageMs: 1.8, videoWidth: 1920, videoHeight: 1080, trackFps: 60 };
try {
  ui.setControlState('idle');
  ui.setResolutionOptions(options, { width: 1280, height: 720 });
  assert(node('resolution-value').textContent === '720p', 'must show negotiated resolution, not requested default');
  click('start-btn'); ui.updateProgress(progress); ui.updateRuntimeMetrics(metrics);
  assert(toggles === 1 && node('live-hit').textContent === '92%', 'start and live metrics');
  click('details-btn');
  assert(sheet.open && root.dataset.phase === 'scanning', 'details must not pause scanning');
  assert(node('file-blocks').textContent === '303 / 473', 'file blocks belong to session');
  click('sheet-close'); await tick();
  assert(document.activeElement === node('details-btn'), 'closing must restore focus');
  click('stop-btn'); assert(sheet.open && stops === 0, 'stop with progress requires confirmation');
  click('confirm-cancel'); await tick();
  assert(root.dataset.phase === 'scanning' && stops === 0, 'cancel stop must preserve reception');
  click('start-btn'); ui.updateRuntimeMetrics(metrics);
  assert(node('live-hit').textContent === '—', 'paused metrics cannot look live');
  const pausedTime = node('elapsed').textContent;
  await new Promise(resolve => setTimeout(resolve, 1050));
  assert(node('elapsed').textContent === pausedTime, 'pause must freeze elapsed timer');
  click('resolution-btn'); await tick();
  assert(!sheet.open, 'top resolution switch must not open a sheet');
  assert(node('resolution-value').textContent === '1080p', 'resolution change');
  assert(node('progress-label').textContent === '64%', 'resolution must preserve file progress');
  ui.setResolutionBusy(true);
  assert(node<HTMLButtonElement>('start-btn').disabled && node<HTMLButtonElement>('resolution-btn').disabled, 'switch in flight must gate controls');
  ui.setResolutionBusy(false);
  click('stop-btn'); click('confirm-action'); await tick();
  assert(stops === 1 && !node<HTMLButtonElement>('resolution-btn').disabled, 'confirmed stop preserves camera controls');
  assert(node('progress-label').textContent === '0%' && node('file-blocks').textContent === '等待接收', 'stop clears collected blocks and progress');
  assert(node('elapsed').textContent === '00:00' && root.dataset.phase === 'idle', 'stop resets timer and returns to preview');
  ui.resetSession(); ui.setControlState('scanning');
  ui.updateProgress({ ...progress, done: true, progress: 1 });
  assert(node('progress-bar').getAttribute('aria-valuenow') === '99', '100% must wait for actual file');
  click('stop-btn'); ui.setControlState('finishing'); await tick();
  assert(!sheet.open && stops === 1, 'completion must dismiss stale stop confirmation');
  ui.setControlState('done'); ui.showDone(URL.createObjectURL(new Blob(['fixture'])), 'test.bin', 7);
  assert(node('progress-bar').getAttribute('aria-valuenow') === '100' && !node('done-panel').hidden, 'completion accessibility and download');
  const filename = node<HTMLInputElement>('save-filename');
  click('download-btn');
  assert(sheet.open && filename.value === 'test.bin' && downloads.length === 0, 'save opens filename editor before download');
  assert(document.activeElement === filename && filename.selectionEnd === 4, 'select basename while keeping extension visible');
  filename.value = 'cancelled.txt'; click('save-cancel'); await tick();
  assert(!sheet.open && downloads.length === 0 && node('done-info').textContent === 'test.bin · 7 B', 'cancel preserves recovered result and filename');
  assert(document.activeElement === node('download-btn'), 'cancel save restores focus');
  click('download-btn');
  assert(filename.value === 'test.bin', 'cancelled edit is discarded');
  for (const invalid of ['', '   ', '../test.bin', 'folder\\test.bin', '..', 'test?.bin']) {
    filename.value = invalid; filename.dispatchEvent(new Event('input')); click('save-action');
    assert(sheet.open && !filename.validity.valid && downloads.length === 0, `reject invalid filename: ${invalid}`);
  }
  filename.value = '  接收文件.txt  '; filename.dispatchEvent(new Event('input')); click('save-action'); await tick();
  assert(!sheet.open && downloads.length === 1 && downloads[0].filename === '接收文件.txt', 'custom filename reaches actual download link');
  assert(await fetch(downloads[0].href).then(r => r.text()) === 'fixture', 'renaming preserves downloaded bytes');
  assert(node('done-info').textContent === '接收文件.txt · 7 B', 'result displays confirmed filename');
  click('download-btn'); assert(filename.value === '接收文件.txt', 'repeat save remembers confirmed filename');
  click('save-action'); await tick();
  assert(downloads.length === 2 && downloads[1].href === downloads[0].href, 'repeat save reuses recovered file');
  click('start-btn'); assert(sheet.open, 'restart must protect unsaved result');
  click('confirm-cancel'); await tick();
  assert(!node('done-panel').hidden, 'cancel restart preserves download');
  ui.resetSession(); ui.setControlState('idle');
  ui.showDone(URL.createObjectURL(new Blob(['new'])), 'next.bin', 3);
  click('download-btn'); assert(filename.value === 'next.bin', 'new file uses its own default filename');
  ui.resetSession(); await tick();
  assert(!sheet.open && node('done-panel').hidden && downloads.length === 2, 'reset dismisses stale save editor without downloading');
  click('stop-btn'); assert(stops === 2 && !sheet.open, 'empty session stops directly');
  ui.setResolutionOptions([options[1]], { width: 1920, height: 1080 });
  assert(node<HTMLButtonElement>('resolution-btn').disabled, 'single active mode is not switchable');
  ui.setResolutionOptions([], {}); ui.setResolutionBusy(false);
  assert(node<HTMLButtonElement>('resolution-btn').disabled, 'empty hardware options stay disabled');
  ui.setControlState('error'); assert(!node<HTMLButtonElement>('start-btn').disabled, 'fatal init failure has retry');
  document.documentElement.dataset.testResult = 'passed';
  console.info('PASS: QRStream UI lifecycle, save naming, modal, progress, timer and resolution regression checks');
} catch (error) {
  document.documentElement.dataset.testResult = 'failed';
  document.documentElement.dataset.testError = String(error);
  console.error(error);
}
document.removeEventListener('click', interceptDownload, true);
// Leave a representative live UI available for visual inspection. These are fixture values.
ui.resetSession(); ui.setControlState('scanning'); ui.setResolutionOptions(options, { width: 1920, height: 1080 });
ui.setCameraSettings({ width: 1920, height: 1080, frameRate: 60 });
for (const part of ['camera', 'core', 'detector'] as const) ui.markReady(part, true);
ui.setDetectorName('ZXing · regression fixture'); ui.setStatus('正在接收数据…');
ui.updateProgress(progress); ui.updateRuntimeMetrics(metrics);
if (new URLSearchParams(location.search).get('preview') === 'save') {
  ui.setControlState('done');
  ui.showDone(URL.createObjectURL(new Blob(['fixture'])), 'qrstream-output.bin', 7);
}
window.addEventListener('pagehide', () => ui.dispose(), { once: true });
