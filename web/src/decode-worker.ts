/// <reference lib="webworker" />

import init, { WasmDecodeSession } from 'qrstream-decode';
import type { SessionResult, SessionSnapshot } from './decode';

const scope = self as DedicatedWorkerGlobalScope;
let session: WasmDecodeSession | null = null;

type WorkerRequest =
  | { type: 'consume'; id: number; text: string }
  | { type: 'snapshot'; id: number }
  | { type: 'bytes'; id: number }
  | { type: 'reset'; id: number };

function requireSession(): WasmDecodeSession {
  if (!session) throw new Error('RaptorQ decoder is not initialized');
  return session;
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === 'consume') {
      const result = JSON.parse(
        requireSession().consume_qr_text(message.text),
      ) as SessionResult;
      scope.postMessage({ type: 'result', id: message.id, result });
    } else if (message.type === 'snapshot') {
      const snapshot = JSON.parse(requireSession().snapshot()) as SessionSnapshot;
      scope.postMessage({ type: 'snapshot', id: message.id, snapshot });
    } else if (message.type === 'bytes') {
      const bytes = requireSession().result_bytes();
      scope.postMessage({ type: 'bytes', id: message.id, bytes }, [bytes.buffer]);
    } else if (message.type === 'reset') {
      session?.free();
      session = new WasmDecodeSession();
      scope.postMessage({ type: 'reset', id: message.id });
    }
  } catch (error) {
    scope.postMessage({ type: 'error', id: message.id, error: String(error) });
  }
};

async function initialize(): Promise<void> {
  try {
    await init();
    session = new WasmDecodeSession();
    scope.postMessage({ type: 'ready' });
  } catch (error) {
    scope.postMessage({ type: 'error', error: String(error) });
  }
}

void initialize();
