/** Progress and interaction UI. */

import type { SessionResult, SessionSnapshot } from './decode';

export interface UiCallbacks {
  onReset(): void;
  onDownload(): void;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export class Ui {
  private readonly status: HTMLElement;
  private readonly bar: HTMLElement;
  private readonly barContainer: HTMLElement;
  private readonly stats: HTMLElement;
  private readonly donePanel: HTMLElement;
  private readonly doneInfo: HTMLElement;
  private readonly downloadBtn: HTMLButtonElement;
  private readonly resetBtn: HTMLButtonElement;
  private readonly errorBox: HTMLElement;

  constructor(root: HTMLElement, callbacks: UiCallbacks) {
    this.status = root.querySelector<HTMLElement>('#status')!;
    this.bar = root.querySelector<HTMLElement>('#progress-fill')!;
    this.barContainer = root.querySelector<HTMLElement>('#progress-bar')!;
    this.stats = root.querySelector<HTMLElement>('#stats')!;
    this.donePanel = root.querySelector<HTMLElement>('#done-panel')!;
    this.doneInfo = root.querySelector<HTMLElement>('#done-info')!;
    this.downloadBtn = root.querySelector<HTMLButtonElement>('#download-btn')!;
    this.errorBox = root.querySelector<HTMLElement>('#error-box')!;

    this.resetBtn = root.querySelector<HTMLButtonElement>('#reset-btn')!;
    this.resetBtn.addEventListener('click', callbacks.onReset);
    this.downloadBtn.addEventListener('click', callbacks.onDownload);
  }

  setStatus(text: string): void {
    this.status.textContent = text;
  }

  showError(message: string): void {
    this.errorBox.textContent = message;
    this.errorBox.hidden = false;
  }

  clearError(): void {
    this.errorBox.hidden = true;
  }

  updateProgress(result: SessionResult | SessionSnapshot): void {
    const pct = Math.round(result.progress * 100);
    this.bar.style.width = `${pct}%`;
    this.barContainer.hidden = false;

    const parts: string[] = [`${pct}%`];
    if (result.symbol_count != null) {
      parts.push(`${result.num_recovered}/${result.symbol_count} symbols`);
    }
    if (result.filesize != null) {
      parts.push(fmtBytes(result.filesize));
    }
    if (result.protocol_version != null) {
      parts.push(`V${result.protocol_version}`);
    }
    this.stats.textContent = parts.join(' · ');
    this.stats.hidden = false;
  }

  showDone(downloadUrl: string, filename: string, size: number): void {
    this.bar.style.width = '100%';
    this.doneInfo.textContent = `${filename} · ${fmtBytes(size)}`;
    this.downloadBtn.dataset.url = downloadUrl;
    this.downloadBtn.dataset.filename = filename;
    this.donePanel.hidden = false;
    this.setStatus('Decoding complete');
  }

  download(): void {
    const url = this.downloadBtn.dataset.url;
    const filename = this.downloadBtn.dataset.filename;
    if (!url || !filename) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  }

  reset(): void {
    this.bar.style.width = '0%';
    this.barContainer.hidden = true;
    this.stats.hidden = true;
    this.stats.textContent = '';
    this.donePanel.hidden = true;
    this.downloadBtn.dataset.url = '';
    this.downloadBtn.dataset.filename = '';
    this.errorBox.hidden = true;
    this.setStatus('Point the camera at a QRStream display');
  }
}
