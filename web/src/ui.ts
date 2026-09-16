/** Progress, diagnostics, and interaction UI. */

import type { SessionResult, SessionSnapshot } from './decode';
import type { RuntimeMetrics } from './metrics';

export interface UiCallbacks {
  onReset(): void;
  onDownload(): void;
}

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fmtFps(value: number): string {
  return value < 10 ? value.toFixed(1) : Math.round(value).toString();
}

export class Ui {
  private readonly root: HTMLElement;
  private readonly status: HTMLElement;
  private readonly statusDot: HTMLElement;
  private readonly bar: HTMLElement;
  private readonly barContainer: HTMLElement;
  private readonly progressLabel: HTMLElement;
  private readonly stats: HTMLElement;
  private readonly donePanel: HTMLElement;
  private readonly doneInfo: HTMLElement;
  private readonly downloadBtn: HTMLButtonElement;
  private readonly resetBtn: HTMLButtonElement;
  private readonly errorBox: HTMLElement;
  private readonly videoWrap: HTMLElement;
  private readonly detectorName: HTMLElement;
  private downloadUrl: string | null = null;
  private pulseTimer: number | null = null;

  constructor(root: HTMLElement, callbacks: UiCallbacks) {
    this.root = root;
    this.status = root.querySelector<HTMLElement>('#status')!;
    this.statusDot = root.querySelector<HTMLElement>('#status-dot')!;
    this.bar = root.querySelector<HTMLElement>('#progress-fill')!;
    this.barContainer = root.querySelector<HTMLElement>('#progress-bar')!;
    this.progressLabel = root.querySelector<HTMLElement>('#progress-label')!;
    this.stats = root.querySelector<HTMLElement>('#stats')!;
    this.donePanel = root.querySelector<HTMLElement>('#done-panel')!;
    this.doneInfo = root.querySelector<HTMLElement>('#done-info')!;
    this.downloadBtn = root.querySelector<HTMLButtonElement>('#download-btn')!;
    this.resetBtn = root.querySelector<HTMLButtonElement>('#reset-btn')!;
    this.errorBox = root.querySelector<HTMLElement>('#error-box')!;
    this.videoWrap = root.querySelector<HTMLElement>('#video-wrap')!;
    this.detectorName = root.querySelector<HTMLElement>('#detector-name')!;

    this.resetBtn.addEventListener('click', callbacks.onReset);
    this.downloadBtn.addEventListener('click', callbacks.onDownload);
  }

  setPhase(phase: 'loading' | 'scanning' | 'done'): void {
    this.root.dataset.phase = phase;
    this.statusDot.dataset.phase = phase;
  }

  setStatus(text: string): void {
    this.status.textContent = text;
  }

  markReady(part: 'core' | 'detector' | 'camera'): void {
    const node = this.root.querySelector<HTMLElement>(`#ready-${part}`);
    if (node) node.dataset.ready = 'true';
  }

  setDetectorName(name: string): void {
    this.detectorName.textContent = name;
  }

  setCameraSettings(settings: MediaTrackSettings): void {
    const width = settings.width ?? 0;
    const height = settings.height ?? 0;
    const fps = settings.frameRate ? ` @ ${Math.round(settings.frameRate)} FPS` : '';
    const node = this.root.querySelector<HTMLElement>('#camera-settings')!;
    node.textContent = width && height ? `${width} × ${height}${fps}` : '自动协商';
  }

  showError(message: string): void {
    this.errorBox.textContent = message;
    this.errorBox.hidden = false;
    this.statusDot.dataset.phase = 'error';
  }

  clearError(): void {
    this.errorBox.hidden = true;
  }

  pulseScan(hit: boolean): void {
    if (!hit) return;
    this.videoWrap.classList.add('has-hit');
    if (this.pulseTimer != null) window.clearTimeout(this.pulseTimer);
    this.pulseTimer = window.setTimeout(() => {
      this.videoWrap.classList.remove('has-hit');
    }, 160);
  }

  updateProgress(result: SessionResult | SessionSnapshot): void {
    const percent = Math.max(0, Math.min(100, Math.round(result.progress * 100)));
    this.bar.style.width = `${percent}%`;
    this.barContainer.setAttribute('aria-valuenow', String(percent));
    this.progressLabel.textContent = `${percent}%`;

    const parts: string[] = [];
    if (result.symbol_count != null) {
      parts.push(`${result.num_recovered} / ${result.symbol_count} 数据块`);
    }
    if (result.filesize != null) parts.push(fmtBytes(result.filesize));
    if (result.protocol_version != null) parts.push(`协议 V${result.protocol_version}`);
    this.stats.textContent = parts.join(' · ');
  }

  updateRuntimeMetrics(metrics: RuntimeMetrics): void {
    this.setMetric('metric-camera', `${fmtFps(metrics.cameraFps)} FPS`);
    this.setMetric('metric-scan', `${fmtFps(metrics.scanFps)} FPS`);
    this.setMetric('metric-latency', `${metrics.detectP95Ms.toFixed(1)} ms`);
    this.setMetric('metric-hit', `${Math.round(metrics.hitRate * 100)}%`);
    this.setMetric('metric-accepted', String(metrics.accepted));
    this.setMetric('metric-resolution', `${metrics.videoWidth} × ${metrics.videoHeight}`);
  }

  showDone(downloadUrl: string, filename: string, size: number): void {
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    this.downloadUrl = downloadUrl;
    this.bar.style.width = '100%';
    this.progressLabel.textContent = '100%';
    this.doneInfo.textContent = `${filename} · ${fmtBytes(size)}`;
    this.downloadBtn.dataset.url = downloadUrl;
    this.downloadBtn.dataset.filename = filename;
    this.donePanel.hidden = false;
    this.setPhase('done');
    this.setStatus('文件已完整恢复');
  }

  download(): void {
    const url = this.downloadBtn.dataset.url;
    const filename = this.downloadBtn.dataset.filename;
    if (!url || !filename) return;
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  }

  reset(): void {
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    this.downloadUrl = null;
    this.bar.style.width = '0%';
    this.barContainer.setAttribute('aria-valuenow', '0');
    this.progressLabel.textContent = '0%';
    this.stats.textContent = '等待第一个有效数据块';
    this.donePanel.hidden = true;
    this.downloadBtn.dataset.url = '';
    this.downloadBtn.dataset.filename = '';
    this.errorBox.hidden = true;
    this.setPhase('scanning');
    this.setStatus('将二维码保持在取景框内');
  }

  private setMetric(id: string, value: string): void {
    const node = this.root.querySelector<HTMLElement>(`#${id}`);
    if (node) node.textContent = value;
  }
}
