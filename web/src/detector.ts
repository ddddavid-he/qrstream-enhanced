/** QR detection with a native BarcodeDetector fast path and a bounded worker pool. */

export interface DetectionBatch {
  texts: string[];
  detectMs: number;
  width: number;
  height: number;
}

export interface QrDetector {
  readonly name: string;
  readonly maxConcurrency: number;
  detect(video: HTMLVideoElement): Promise<DetectionBatch | null>;
  stop(): void;
}

interface BarcodeDetectorLike {
  detect(source: CanvasImageSource): Promise<{ rawValue: string }[]>;
}

interface BarcodeDetectorCtor {
  new (options?: { formats?: string[] }): BarcodeDetectorLike;
  getSupportedFormats(): Promise<string[]>;
}

interface WorkerResult {
  type: 'result';
  id: number;
  texts: string[];
  detectMs: number;
  width: number;
  height: number;
  error?: string;
}

interface WorkerSlot {
  worker: Worker;
  busy: boolean;
  generation: number;
}

const WATCHDOG_MS = 1_500;
const DEFAULT_DETECT_DIM = 1_280;
const PROBE_DETECT_DIM = 1_600;

function getBarcodeDetectorCtor(): BarcodeDetectorCtor | null {
  const ctor = (globalThis as Record<string, unknown>).BarcodeDetector;
  return typeof ctor === 'function' ? (ctor as BarcodeDetectorCtor) : null;
}

export function isBarcodeDetectorSupported(): boolean {
  return getBarcodeDetectorCtor() !== null;
}

export async function createQrDetector(): Promise<QrDetector> {
  const ctor = getBarcodeDetectorCtor();
  if (ctor) {
    try {
      let formats: string[] | undefined;
      try {
        const supported = await ctor.getSupportedFormats();
        if (supported.includes('qr_code')) formats = ['qr_code'];
      } catch {
        // Some implementations expose the constructor but not format probing.
      }
      return new NativeDetector(new ctor(formats ? { formats } : {}));
    } catch {
      // Construction can fail behind browser flags; use the portable path.
    }
  }

  if (
    typeof Worker !== 'undefined' &&
    typeof OffscreenCanvas !== 'undefined' &&
    typeof createImageBitmap !== 'undefined'
  ) {
    const hardwareThreads = navigator.hardwareConcurrency || 4;
    const workerCount = Math.max(2, Math.min(4, hardwareThreads - 2));
    return WorkerPoolDetector.create(workerCount);
  }

  return MainThreadZxingDetector.create();
}

class NativeDetector implements QrDetector {
  readonly name = 'BarcodeDetector';
  readonly maxConcurrency = 1;
  private busy = false;

  constructor(private readonly detector: BarcodeDetectorLike) {}

  async detect(video: HTMLVideoElement): Promise<DetectionBatch | null> {
    if (this.busy) return null;
    this.busy = true;
    const startedAt = performance.now();
    try {
      const codes = await this.detector.detect(video);
      return {
        texts: codes.map((code) => code.rawValue),
        detectMs: performance.now() - startedAt,
        width: video.videoWidth,
        height: video.videoHeight,
      };
    } finally {
      this.busy = false;
    }
  }

  stop(): void {}
}

class WorkerPoolDetector implements QrDetector {
  readonly name: string;
  readonly maxConcurrency: number;
  private readonly slots: WorkerSlot[] = [];
  private requestId = 0;
  private stopped = false;
  private consecutiveMisses = 0;

  private constructor(workerCount: number) {
    this.maxConcurrency = workerCount;
    this.name = `ZXing worker pool ×${workerCount}`;
  }

  static async create(workerCount: number): Promise<WorkerPoolDetector> {
    const pool = new WorkerPoolDetector(workerCount);
    await Promise.all(
      Array.from({ length: workerCount }, (_, index) => pool.spawn(index)),
    );
    return pool;
  }

  async detect(video: HTMLVideoElement): Promise<DetectionBatch | null> {
    if (this.stopped) return null;
    const slot = this.slots.find((candidate) => !candidate.busy);
    if (!slot) return null; // Never queue stale camera frames.

    slot.busy = true;
    const generation = slot.generation;
    const maxDim = this.consecutiveMisses >= 45 ? PROBE_DETECT_DIM : DEFAULT_DETECT_DIM;
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const scale = Math.min(1, maxDim / Math.max(sourceWidth, sourceHeight));
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));

    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(video, {
        resizeWidth: width,
        resizeHeight: height,
        resizeQuality: 'medium',
      });
    } catch (error) {
      slot.busy = false;
      throw error;
    }

    const id = ++this.requestId;
    return new Promise<DetectionBatch | null>((resolve) => {
      const timer = window.setTimeout(() => {
        if (slot.generation !== generation) return;
        this.recycle(slot);
        resolve(null);
      }, WATCHDOG_MS);

      slot.worker.onmessage = (event: MessageEvent<WorkerResult>) => {
        if (event.data.type !== 'result' || event.data.id !== id) return;
        window.clearTimeout(timer);
        if (slot.generation !== generation) return;
        slot.busy = false;
        const result = event.data;
        if (result.error) {
          this.recycle(slot);
          resolve(null);
          return;
        }
        this.consecutiveMisses = result.texts.length > 0
          ? 0
          : Math.min(90, this.consecutiveMisses + 1);
        resolve(result);
      };

      slot.worker.postMessage(
        { type: 'detect', id, bitmap, width, height },
        [bitmap],
      );
    });
  }

  stop(): void {
    this.stopped = true;
    for (const slot of this.slots) slot.worker.terminate();
    this.slots.length = 0;
  }

  private async spawn(index: number): Promise<void> {
    const worker = new Worker(new URL('./detect-worker.ts', import.meta.url), {
      type: 'module',
      name: `qrstream-detector-${index}`,
    });
    const slot: WorkerSlot = { worker, busy: true, generation: 0 };
    this.slots[index] = slot;
    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(
        () => reject(new Error('QR detector worker initialization timed out')),
        WATCHDOG_MS * 4,
      );
      worker.onmessage = (event: MessageEvent<{ type: string; error?: string }>) => {
        if (event.data.type === 'ready') {
          window.clearTimeout(timer);
          slot.busy = false;
          resolve();
        } else if (event.data.type === 'error') {
          window.clearTimeout(timer);
          reject(new Error(event.data.error ?? 'QR detector worker failed'));
        }
      };
      worker.onerror = (event) => {
        window.clearTimeout(timer);
        reject(new Error(event.message));
      };
    });
  }

  private recycle(slot: WorkerSlot): void {
    const index = this.slots.indexOf(slot);
    if (index < 0 || this.stopped) return;
    slot.worker.terminate();
    slot.generation++;
    slot.busy = true;
    void this.spawn(index).catch(() => {
      // A later scan can continue with the remaining healthy workers.
    });
  }
}

class MainThreadZxingDetector implements QrDetector {
  readonly name = 'ZXing compatibility mode';
  readonly maxConcurrency = 1;
  private busy = false;

  private constructor(
    private readonly read: (
      image: ImageData,
      options: { formats: ['QRCode']; tryHarder: boolean },
    ) => Promise<{ text: string }[]>,
    private readonly canvas: HTMLCanvasElement,
    private readonly context: CanvasRenderingContext2D,
  ) {}

  static async create(): Promise<MainThreadZxingDetector> {
    const { readBarcodesFromImageData, prepareZXingModule } = await import(
      'zxing-wasm/reader'
    );
    const { default: zxingWasmUrl } = await import(
      'zxing-wasm/reader/zxing_reader.wasm?url'
    );
    await Promise.resolve(prepareZXingModule({
      overrides: { locateFile: () => zxingWasmUrl },
      fireImmediately: true,
    }));
    const canvas = document.createElement('canvas');
    // Deliberately omit willReadFrequently: it can force expensive CPU-backed draws.
    const context = canvas.getContext('2d');
    if (!context) throw new Error('2D canvas context unavailable');
    return new MainThreadZxingDetector(readBarcodesFromImageData, canvas, context);
  }

  async detect(video: HTMLVideoElement): Promise<DetectionBatch | null> {
    if (this.busy) return null;
    this.busy = true;
    try {
      const scale = Math.min(1, DEFAULT_DETECT_DIM / video.videoWidth);
      const width = Math.max(1, Math.round(video.videoWidth * scale));
      const height = Math.max(1, Math.round(video.videoHeight * scale));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
      }
      const startedAt = performance.now();
      this.context.drawImage(video, 0, 0, width, height);
      const image = this.context.getImageData(0, 0, width, height);
      const codes = await this.read(image, { formats: ['QRCode'], tryHarder: true });
      return {
        texts: codes.map((code) => code.text),
        detectMs: performance.now() - startedAt,
        width,
        height,
      };
    } finally {
      this.busy = false;
    }
  }

  stop(): void {}
}
