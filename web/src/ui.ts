/** Camera controls and a modal sheet; opening details never changes reception. */
import type { SessionResult, SessionSnapshot } from './decode';
import type { RuntimeMetrics } from './metrics';
import type { ResolutionId, ResolutionOption } from './camera';

export interface UiCallbacks {
  onToggle(): void;
  onStop(): void;
  onResolutionChange(id: ResolutionId): void;
  onDownload(): void;
}
export type ControlState = 'loading' | 'idle' | 'starting' | 'scanning' | 'paused' | 'finishing' | 'stopped' | 'done' | 'error';
function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function fmtFps(value: number): string { return value < 10 ? value.toFixed(1) : Math.round(value).toString(); }

export class Ui {
  private state: ControlState = 'loading';
  private readonly sheet: HTMLDialogElement;
  private opener: HTMLButtonElement | null = null;
  private confirmation: (() => void) | null = null;
  private downloadUrl: string | null = null;
  private filename = '';
  private received = 0;
  private pulseTimer: number | null = null;
  private readonly clockTimer: number;
  private activeSince: number | null = null;
  private elapsedMs = 0;
  private resolutionBusy = false;
  private resolutions: ResolutionOption[] = [];
  private selectedResolution: ResolutionId | null = null;

  constructor(private readonly root: HTMLElement, private readonly callbacks: UiCallbacks) {
    this.sheet = this.node<HTMLDialogElement>('sheet');
    this.node('start-btn').addEventListener('click', () => {
      if (this.state === 'done') {
        this.confirm('开始新的接收？', '请先保存已恢复的文件。开始新任务会清除当前结果。', '重新开始', callbacks.onToggle);
      } else if (this.state === 'error') {
        location.reload();
      } else callbacks.onToggle();
    });
    this.node('stop-btn').addEventListener('click', () => {
      if (this.received > 0) {
        this.confirm('停止接收？', '摄像头将关闭。重新开始会清空当前文件的接收进度。', '停止接收', callbacks.onStop);
      } else callbacks.onStop();
    });
    this.node('details-btn').addEventListener('click', () => this.openSheet('details', '接收详情', 'details-btn'));
    this.node('resolution-btn').addEventListener('click', () => {
      const next = this.nextResolution();
      if (next && !this.resolutionBusy) this.callbacks.onResolutionChange(next.id);
    });
    this.node('download-btn').addEventListener('click', callbacks.onDownload);
    this.node('sheet-close').addEventListener('click', () => this.sheet.close());
    this.node('confirm-cancel').addEventListener('click', () => this.sheet.close());
    this.node('confirm-action').addEventListener('click', () => {
      const action = this.confirmation;
      this.sheet.close();
      this.confirmation = null;
      action?.();
    });
    // Native dialog provides focus containment, Escape and an inert background.
    this.sheet.addEventListener('click', (event) => {
      const rect = this.sheet.getBoundingClientRect();
      if (event.target === this.sheet && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) this.sheet.close();
    });
    this.sheet.addEventListener('close', () => {
      this.confirmation = null;
      this.opener?.setAttribute('aria-expanded', 'false');
      if (this.opener && !this.opener.disabled) this.opener.focus();
    });
    const handle = this.node('sheet-handle');
    let startY: number | null = null;
    handle.addEventListener('pointerdown', (event) => {
      startY = event.clientY;
      handle.setPointerCapture(event.pointerId);
    });
    handle.addEventListener('pointerup', (event) => {
      if (startY != null && event.clientY - startY > 65) this.sheet.close();
      startY = null;
    });
    handle.addEventListener('pointercancel', () => { startY = null; });
    this.clockTimer = window.setInterval(() => this.renderClock(), 500);
  }

  private node<T extends HTMLElement = HTMLElement>(id: string): T {
    return this.root.querySelector<T>(`#${id}`)!;
  }
  private text(id: string, value: string): void { this.node(id).textContent = value; }
  private openSheet(body: 'details' | 'confirm', title: string, opener: string): void {
    for (const id of ['details', 'confirm']) this.node(`${id}-body`).hidden = id !== body;
    this.text('sheet-title', title);
    this.opener = this.node<HTMLButtonElement>(opener);
    this.opener.setAttribute('aria-expanded', 'true');
    this.sheet.showModal();
    this.node(`${body}-body`).scrollTop = 0;
  }
  private confirm(title: string, copy: string, actionLabel: string, action: () => void): void {
    this.confirmation = action;
    this.text('confirm-copy', copy);
    this.text('confirm-action', actionLabel);
    this.openSheet('confirm', title, this.state === 'done' ? 'start-btn' : 'stop-btn');
  }
  setControlState(state: ControlState): void {
    if (this.activeSince != null) this.elapsedMs += performance.now() - this.activeSince;
    this.activeSince = state === 'scanning' ? performance.now() : null;
    this.state = state;
    this.root.dataset.phase = state;
    // A file may finish while the stop confirmation is open.
    if (this.confirmation && ['finishing', 'done', 'error'].includes(state)) this.sheet.close();
    const label = state === 'scanning' ? '暂停接收' : state === 'paused' ? '继续接收' : state === 'done' || state === 'stopped' ? '重新开始' : state === 'error' ? '刷新重试' : '开始接收';
    this.node('start-btn').setAttribute('aria-label', label);
    this.text('start-label', label);
    this.updateButtons();
    if (state !== 'scanning') {
      this.text('live-hit', '—'); this.text('live-fps', '—');
    }
    this.renderClock();
  }
  private updateButtons(): void {
    this.node<HTMLButtonElement>('start-btn').disabled = this.resolutionBusy || !['idle', 'scanning', 'paused', 'stopped', 'done', 'error'].includes(this.state);
    this.node<HTMLButtonElement>('stop-btn').disabled = this.resolutionBusy || !['idle', 'scanning', 'paused'].includes(this.state);
    this.node<HTMLButtonElement>('resolution-btn').disabled = this.resolutionBusy || !this.nextResolution();
  }
  private renderClock(): void {
    const seconds = Math.floor((this.elapsedMs + (this.activeSince == null ? 0 : performance.now() - this.activeSince)) / 1000);
    this.text('elapsed', `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`);
  }
  setStatus(text: string): void { this.text('status', text); }
  markReady(part: 'core' | 'detector' | 'camera', ready: boolean): void { this.text(`ready-${part}`, ready ? '就绪' : '未就绪'); }
  setDetectorName(name: string): void { this.text('detector-name', name); }
  setCameraSettings(settings: MediaTrackSettings): void {
    this.text('camera-settings', settings.width && settings.height ? `${settings.width} × ${settings.height}${settings.frameRate ? ` · ${Math.round(settings.frameRate)} FPS` : ''}` : '自动协商');
  }
  setResolutionOptions(options: ResolutionOption[], settings: MediaTrackSettings): void {
    this.resolutions = options;
    const actual = options.find(option => (option.width === settings.width && option.height === settings.height) || (option.width === settings.height && option.height === settings.width));
    this.selectedResolution = actual?.id ?? null;
    this.text('resolution-value', actual?.label ?? (settings.width && settings.height ? `${Math.min(settings.width, settings.height)}p` : '自动'));
    const next = this.nextResolution();
    const description = `采集分辨率 ${this.node('resolution-value').textContent}${next ? `，点击切换为 ${next.label}` : '，无其他可选档位'}`;
    this.node('resolution-btn').setAttribute('aria-label', description);
    this.node('resolution-btn').title = description;
    this.updateButtons();
  }
  private nextResolution(): ResolutionOption | undefined {
    if (this.resolutions.length === 0) return undefined;
    const index = this.resolutions.findIndex(option => option.id === this.selectedResolution);
    if (index >= 0 && this.resolutions.length === 1) return undefined;
    return this.resolutions[(index + 1) % this.resolutions.length];
  }
  setResolutionBusy(busy: boolean): void {
    this.resolutionBusy = busy;
    this.node('resolution-btn').setAttribute('aria-busy', String(busy));
    this.updateButtons();
  }
  setCameraInactive(): void {
    this.text('camera-settings', '摄像头已关闭');
    this.resolutions = [];
    this.updateButtons();
  }
  showError(message: string): void { this.text('error-box', message); this.node('error-box').hidden = false; }
  clearError(): void { this.node('error-box').hidden = true; }
  pulseScan(hit: boolean): void {
    if (!hit) return;
    this.node('video-wrap').classList.add('has-hit');
    if (this.pulseTimer != null) clearTimeout(this.pulseTimer);
    this.pulseTimer = window.setTimeout(() => this.node('video-wrap').classList.remove('has-hit'), 160);
  }
  private setProgress(percent: number): void {
    this.node('progress-bar').style.setProperty('--progress', `${percent}%`);
    this.node('progress-bar').setAttribute('aria-valuenow', String(percent));
    this.text('progress-label', `${percent}%`);
  }
  updateProgress(result: SessionResult | SessionSnapshot): void {
    this.received = result.num_received;
    // Completion belongs to resultBytes()/showDone(), not merely enough symbols.
    this.setProgress(Math.max(0, Math.min(99, Math.round(result.progress * 100))));
    this.text('file-blocks', result.symbol_count == null ? String(result.num_received) : `${result.num_received} / ${result.symbol_count}`);
    if (result.filesize != null) this.text('file-size', fmtBytes(result.filesize));
    if (result.protocol_version != null) this.text('file-protocol', `V${result.protocol_version} / RaptorQ`);
  }
  updateRuntimeMetrics(metrics: RuntimeMetrics): void {
    const running = this.state === 'scanning' && !this.resolutionBusy;
    const fps = running ? fmtFps(metrics.scanFps) : '—';
    const hit = running && metrics.scanFps > 0 ? `${Math.round(metrics.hitRate * 100)}%` : '—';
    this.text('live-fps', fps); this.text('live-hit', hit);
    this.text('metric-camera', `${fmtFps(metrics.cameraFps)} FPS`);
    this.text('metric-scan', running ? `${fps} FPS` : '—');
    this.text('metric-hit', hit);
    this.text('metric-latency', `${metrics.detectP95Ms.toFixed(1)} ms`);
    this.text('metric-decode', `${metrics.decodeAverageMs.toFixed(1)} ms`);
    this.text('metric-throughput', running ? `${fmtFps(metrics.acceptedFps)} 块/秒` : '—');
    this.text('metric-rejected', `${metrics.duplicates} / ${metrics.invalid}`);
  }
  showDone(downloadUrl: string, filename: string, size: number): void {
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    this.downloadUrl = downloadUrl;
    this.filename = filename;
    this.setProgress(100);
    this.text('done-info', `${filename} · ${fmtBytes(size)}`);
    this.node('done-panel').hidden = false;
    this.setStatus('文件已完整恢复');
  }
  download(): void {
    if (!this.downloadUrl) return;
    const anchor = document.createElement('a');
    anchor.href = this.downloadUrl; anchor.download = this.filename;
    document.body.append(anchor); anchor.click(); anchor.remove();
  }
  resetSession(): void {
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    this.downloadUrl = null; this.filename = ''; this.received = 0;
    this.elapsedMs = 0; this.activeSince = null; this.renderClock();
    this.setProgress(0);
    this.text('file-size', '—'); this.text('file-blocks', '等待接收'); this.text('file-protocol', 'V4 / RaptorQ');
    for (const id of ['camera', 'scan', 'hit', 'latency', 'decode', 'throughput', 'rejected']) this.text(`metric-${id}`, '—');
    this.node('done-panel').hidden = true; this.clearError();
    this.setStatus('准备新的扫描任务…');
  }
  dispose(): void {
    clearInterval(this.clockTimer);
    if (this.pulseTimer != null) clearTimeout(this.pulseTimer);
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
  }
}
