/** Entry point: wires camera → detector → WASM decode session → UI. */

import { startCamera, CameraError, type CameraHandle } from './camera';
import { createQrDetector } from './detector';
import { loadWasm, createSession, parseResult } from './decode';
import { Ui } from './ui';
import type { WasmDecodeSession } from 'qrstream-decode';

const SCAN_INTERVAL_MS = 40; // ~25 fps detection rate

class App {
  private readonly video: HTMLVideoElement;
  private readonly ui: Ui;
  private camera: CameraHandle | null = null;
  private session: WasmDecodeSession | null = null;
  private scanTimer: number | null = null;
  private scanBusy = false;
  private lastSeenText: string | null = null;
  private lastSeenAt = 0;
  private finished = false;

  constructor() {
    const root = document.querySelector<HTMLElement>('#app')!;
    this.video = root.querySelector<HTMLVideoElement>('#video')!;
    this.ui = new Ui(root, {
      onReset: () => this.reset(),
      onDownload: () => this.ui.download(),
    });
  }

  async start(): Promise<void> {
    this.ui.setStatus('Loading decoder…');
    try {
      await loadWasm();
    } catch {
      this.ui.showError('Failed to load the WASM decoder core.');
      return;
    }

    if (!('BarcodeDetector' in globalThis)) {
      this.ui.setStatus('BarcodeDetector unavailable — using zxing-wasm fallback…');
    }

    this.ui.setStatus('Starting camera…');
    try {
      this.camera = await startCamera(this.video);
    } catch (err) {
      if (err instanceof CameraError) {
        this.ui.showError(err.message);
        this.ui.setStatus('Camera unavailable');
      } else {
        this.ui.showError(`Unexpected camera error: ${String(err)}`);
      }
      return;
    }

    const detector = await createQrDetector();
    this.session = createSession();
    this.ui.clearError();
    this.ui.setStatus('Point the camera at a QRStream display');

    this.scanTimer = window.setInterval(() => {
      void this.scan(detector);
    }, SCAN_INTERVAL_MS);
  }

  private async scan(
    detector: (video: HTMLVideoElement) => Promise<string[]>,
  ): Promise<void> {
    if (this.scanBusy || this.finished || !this.session) return;
    const video = this.camera?.video;
    if (!video || video.readyState < 2) return;
    this.scanBusy = true;
    try {
      const texts = await detector(video);
      const now = performance.now();
      for (const text of texts) {
        // Skip repeated identical payloads within 500 ms (same QR on screen).
        if (text === this.lastSeenText && now - this.lastSeenAt < 500) continue;
        this.lastSeenText = text;
        this.lastSeenAt = now;
        this.feed(text);
        if (this.finished) break;
      }
    } catch {
      // Transient detection errors are non-fatal; keep scanning.
    } finally {
      this.scanBusy = false;
    }
  }

  private feed(text: string): void {
    if (!this.session) return;
    let result;
    try {
      result = parseResult(this.session.consume_qr_text(text));
    } catch {
      return; // WASM boundary failure — treat as a bad frame.
    }
    if (result.error && !result.accepted) {
      // Malformed/CRC failure: ignore but surface once per stream.
      this.ui.setStatus(`Skipping invalid frame (${result.error})`);
      return;
    }
    if (result.duplicate) {
      return;
    }
    if (result.accepted) {
      this.ui.updateProgress(result);
      if (!result.done) {
        this.ui.setStatus('Receiving…');
      }
    }
    if (result.done && !this.finished) {
      this.finish();
    }
  }

  private finish(): void {
    if (!this.session) return;
    this.finished = true;
    let bytes: Uint8Array;
    try {
      bytes = this.session.result_bytes();
    } catch (e) {
      this.finished = false;
      this.ui.showError(`Decode failed at finalize: ${String(e)}`);
      return;
    }
    const blob = new Blob([bytes.slice().buffer as ArrayBuffer], {
      type: 'application/octet-stream',
    });
    const url = URL.createObjectURL(blob);
    this.ui.showDone(url, 'qrstream-output.bin', bytes.length);
  }

  private reset(): void {
    this.finished = false;
    this.lastSeenText = null;
    this.session?.free();
    this.session = createSession();
    this.ui.reset();
  }

  stop(): void {
    if (this.scanTimer != null) {
      clearInterval(this.scanTimer);
      this.scanTimer = null;
    }
    this.camera?.stop();
    this.session?.free();
  }
}

const app = new App();
void app.start();
