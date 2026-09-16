/// <reference lib="webworker" />

import { prepareZXingModule, readBarcodesFromImageData } from 'zxing-wasm/reader';
import zxingWasmUrl from 'zxing-wasm/reader/zxing_reader.wasm?url';

const scope = self as DedicatedWorkerGlobalScope;
let canvas: OffscreenCanvas | null = null;
let context: OffscreenCanvasRenderingContext2D | null = null;

async function initialize(): Promise<void> {
  try {
    await Promise.resolve(prepareZXingModule({
      overrides: { locateFile: () => zxingWasmUrl },
      fireImmediately: true,
    }));
    scope.postMessage({ type: 'ready' });
  } catch (error) {
    scope.postMessage({ type: 'error', error: String(error) });
  }
}

scope.onmessage = (event: MessageEvent<{
  type: 'detect';
  id: number;
  bitmap: ImageBitmap;
  width: number;
  height: number;
}>) => {
  if (event.data.type !== 'detect') return;
  const { id, bitmap, width, height } = event.data;
  void detect(id, bitmap, width, height);
};

async function detect(
  id: number,
  bitmap: ImageBitmap,
  width: number,
  height: number,
): Promise<void> {
  const startedAt = performance.now();
  try {
    if (!canvas || canvas.width !== width || canvas.height !== height) {
      canvas = new OffscreenCanvas(width, height);
      // Do not request willReadFrequently; GPU video draws regress badly on Chromium.
      context = canvas.getContext('2d');
    }
    if (!context) throw new Error('OffscreenCanvas 2D context unavailable');
    context.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    const image = context.getImageData(0, 0, width, height);
    const results = await readBarcodesFromImageData(image, {
      formats: ['QRCode'],
      tryHarder: false,
    });
    scope.postMessage({
      type: 'result',
      id,
      texts: results.map((result) => result.text),
      detectMs: performance.now() - startedAt,
      width,
      height,
    });
  } catch (error) {
    try { bitmap.close(); } catch { /* already released after draw */ }
    scope.postMessage({
      type: 'result',
      id,
      texts: [],
      detectMs: performance.now() - startedAt,
      width,
      height,
      error: String(error),
    });
  }
}

void initialize();
