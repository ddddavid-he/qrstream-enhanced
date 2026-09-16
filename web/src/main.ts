/** Entry point: camera -> frame-paced detector pool -> WASM session -> UI. */

import { startCamera, CameraError, type CameraHandle } from './camera';
import { createQrDetector, type QrDetector } from './detector';
import { loadWasm, createSession, parseResult, parseSnapshot } from './decode';
import { MetricsTracker } from './metrics';
import { Ui } from './ui';
import type { WasmDecodeSession } from 'qrstream-decode';

class App {
  private readonly video: HTMLVideoElement;
  private readonly ui: Ui;
  private readonly metrics = new MetricsTracker();
  private camera: CameraHandle | null = null;
  private detector: QrDetector | null = null;
  private session: WasmDecodeSession | null = null;
  private frameCallbackId: number | null = null;
  private animationFrameId: number | null = null;
  private metricsTimer: number | null = null;
  private finished = false;
  private generation = 0;

  constructor() {
    const root = document.querySelector<HTMLElement>('#app')!;
    this.video = root.querySelector<HTMLVideoElement>('#video')!;
    this.ui = new Ui(root, {
      onReset: () => this.reset(),
      onDownload: () => this.ui.download(),
    });
  }

  async start(): Promise<void> {
    this.ui.setPhase('loading');
    this.ui.setStatus('正在加载解码核心…');
    try {
      await loadWasm();
      this.ui.markReady('core');
    } catch {
      this.ui.showError('WASM 解码核心加载失败，请刷新页面重试。');
      return;
    }

    this.ui.setStatus('正在初始化高速 QR 检测器…');
    try {
      this.detector = await createQrDetector();
      this.ui.markReady('detector');
      this.ui.setDetectorName(this.detector.name);
    } catch (error) {
      this.ui.showError(`QR 检测器加载失败：${String(error)}`);
      return;
    }

    this.ui.setStatus('正在请求摄像头权限…');
    try {
      this.camera = await startCamera(this.video);
      this.ui.markReady('camera');
      this.ui.setCameraSettings(this.camera.videoTrack.getSettings());
    } catch (error) {
      if (error instanceof CameraError) {
        this.ui.showError(error.message);
        this.ui.setStatus('摄像头不可用');
      } else {
        this.ui.showError(`无法启动摄像头：${String(error)}`);
      }
      return;
    }

    this.session = createSession();
    this.metrics.reset();
    this.ui.clearError();
    this.ui.setPhase('scanning');
    this.ui.setStatus('将二维码保持在取景框内');
    this.scheduleNextFrame();
    this.metricsTimer = window.setInterval(() => this.publishMetrics(), 500);
  }

  private scheduleNextFrame(): void {
    if (this.finished || !this.detector) return;
    if ('requestVideoFrameCallback' in this.video) {
      this.frameCallbackId = this.video.requestVideoFrameCallback((now, metadata) => {
        this.frameCallbackId = null;
        this.onVideoFrame(now, metadata.presentedFrames);
      });
    } else {
      this.animationFrameId = requestAnimationFrame((now) => {
        this.animationFrameId = null;
        this.onVideoFrame(now);
      });
    }
  }

  private onVideoFrame(now: number, presentedFrames?: number): void {
    this.metrics.recordVideoFrame(now, presentedFrames);
    this.scheduleNextFrame();
    if (this.finished || !this.detector || this.video.readyState < 2) return;
    const generation = this.generation;
    void this.detector.detect(this.video)
      .then((batch) => {
        if (!batch || generation !== this.generation || this.finished) return;
        const completedAt = performance.now();
        this.metrics.recordScan(completedAt, batch.detectMs, batch.texts.length);
        this.ui.pulseScan(batch.texts.length > 0);
        for (const text of batch.texts) {
          this.feed(text);
          if (this.finished) break;
        }
      })
      .catch(() => {
        // A transient capture failure must not break the frame callback chain.
      });
  }

  private feed(text: string): void {
    if (!this.session) return;
    const startedAt = performance.now();
    let result;
    try {
      result = parseResult(this.session.consume_qr_text(text));
    } catch {
      return;
    }
    const finishedAt = performance.now();
    this.metrics.recordDecode(finishedAt, finishedAt - startedAt, result);
    if (result.error && !result.accepted) {
      this.ui.setStatus('已忽略一个损坏帧，继续扫描…');
      return;
    }
    if (result.accepted && !result.duplicate) {
      this.ui.updateProgress(result);
      if (!result.done) this.ui.setStatus('正在接收数据…');
    }
    if (result.done && !this.finished) this.finish();
  }

  private publishMetrics(): void {
    if (!this.session || !this.detector || !this.camera) return;
    try {
      const snapshot = parseSnapshot(this.session.snapshot());
      const runtime = this.metrics.snapshot(
        this.detector.name,
        this.video,
        this.camera.videoTrack,
        snapshot,
      );
      this.ui.updateRuntimeMetrics(runtime);
    } catch {
      // Metrics are diagnostic only and must never stop decoding.
    }
  }

  private finish(): void {
    if (!this.session) return;
    this.finished = true;
    this.cancelFrameCallback();
    let bytes: Uint8Array;
    try {
      bytes = this.session.result_bytes();
    } catch (error) {
      this.finished = false;
      this.ui.showError(`文件恢复失败：${String(error)}`);
      this.scheduleNextFrame();
      return;
    }
    const blob = new Blob([bytes.slice().buffer as ArrayBuffer], {
      type: 'application/octet-stream',
    });
    this.ui.showDone(URL.createObjectURL(blob), 'qrstream-output.bin', bytes.length);
  }

  private reset(): void {
    this.generation++;
    this.finished = false;
    this.session?.free();
    this.session = createSession();
    this.metrics.reset();
    this.ui.reset();
    this.cancelFrameCallback();
    this.scheduleNextFrame();
  }

  private cancelFrameCallback(): void {
    if (this.frameCallbackId != null) {
      this.video.cancelVideoFrameCallback(this.frameCallbackId);
      this.frameCallbackId = null;
    }
    if (this.animationFrameId != null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  stop(): void {
    this.cancelFrameCallback();
    if (this.metricsTimer != null) clearInterval(this.metricsTimer);
    this.detector?.stop();
    this.camera?.stop();
    this.session?.free();
  }
}

const app = new App();
window.addEventListener('pagehide', () => app.stop(), { once: true });
void app.start();
