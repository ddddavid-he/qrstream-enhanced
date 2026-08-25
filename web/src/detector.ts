/**
 * QR detection: BarcodeDetector API when available, zxing-wasm fallback
 * for Safari/Firefox. Detector functions receive the live <video> and
 * return decoded string payloads.
 */

export type QrDetector = (video: HTMLVideoElement) => Promise<string[]>;

import type { ReaderOptions } from 'zxing-wasm/reader';

interface BarcodeDetectorLike {
  detect(source: CanvasImageSource): Promise<{ rawValue: string }[]>;
}

interface BarcodeDetectorCtor {
  new (options?: { formats?: string[] }): BarcodeDetectorLike;
  getSupportedFormats(): Promise<string[]>;
}

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
      // Prefer QR format only if supported; otherwise let the detector do all 1D/2D.
      let formats: string[] | undefined;
      try {
        const supported = await ctor.getSupportedFormats();
        if (supported.includes('qr_code')) {
          formats = ['qr_code'];
        }
      } catch {
        // getSupportedFormats may be unavailable; fall back to default formats.
      }
      const detector = new ctor(formats ? { formats } : {});
      return async (video: HTMLVideoElement) => {
        const codes = await detector.detect(video);
        return codes.map((c) => c.rawValue);
      };
    } catch {
      // Construction failed; fall through to zxing-wasm below.
    }
  }
  return createZxingDetector();
}

/** Frame capture helper: draw the video into a canvas at reduced scale. */
function createFrameGrabber(maxWidth: number) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas context unavailable');
  return (video: HTMLVideoElement): ImageData | null => {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;
    const scale = Math.min(1, maxWidth / vw);
    const w = Math.max(1, Math.round(vw * scale));
    const h = Math.max(1, Math.round(vh * scale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.drawImage(video, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  };
}

/** zxing-wasm based detector for browsers without BarcodeDetector (Safari, Firefox). */
async function createZxingDetector(): Promise<QrDetector> {
  const { readBarcodesFromImageData, prepareZXingModule } = await import(
    'zxing-wasm/reader'
  );
  // Eagerly compile the wasm module so the first scan doesn't stall.
  void Promise.resolve(prepareZXingModule()).catch(() => {});

  const grabFrame = createFrameGrabber(1280);
  const options: ReaderOptions = {
    formats: ['QRCode'],
    tryHarder: true,
  };
  return async (video: HTMLVideoElement) => {
    const image = grabFrame(video);
    if (!image) return [];
    const results = await readBarcodesFromImageData(image, options);
    return results.map((r) => r.text);
  };
}
