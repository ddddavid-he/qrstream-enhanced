/** Dedicated-worker client for the stateful RaptorQ WASM decoder. */

export interface SessionResult {
  accepted: boolean;
  duplicate: boolean;
  done: boolean;
  progress: number;
  num_recovered: number;
  num_received: number;
  symbol_count: number | null;
  filesize: number | null;
  protocol_version: number | null;
  error: string | null;
}

export interface SessionSnapshot {
  initialized: boolean;
  done: boolean;
  progress: number;
  num_recovered: number;
  num_received: number;
  symbol_count: number | null;
  filesize: number | null;
  protocol_version: number | null;
}

type RequestKind = 'consume' | 'snapshot' | 'bytes' | 'reset';

interface PendingRequest<T> {
  resolve(value: T): void;
  reject(error: Error): void;
}

type WorkerResponse =
  | { type: 'ready' }
  | { type: 'result'; id: number; result: SessionResult }
  | { type: 'snapshot'; id: number; snapshot: SessionSnapshot }
  | { type: 'bytes'; id: number; bytes: Uint8Array }
  | { type: 'reset'; id: number }
  | { type: 'error'; id?: number; error: string };

const READY_TIMEOUT_MS = 10_000;
const MAX_PENDING_SYMBOLS = 32;

export class DecodeWorkerClient {
  private readonly worker: Worker;
  private readonly pending = new Map<number, PendingRequest<unknown>>();
  private nextId = 0;
  private pendingSymbols = 0;
  private stopped = false;
  private readyResolve!: () => void;
  private readyReject!: (error: Error) => void;
  private readonly readyPromise: Promise<void>;

  private constructor() {
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.worker = new Worker(new URL('./decode-worker.ts', import.meta.url), {
      type: 'module',
      name: 'qrstream-raptorq-decoder',
    });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      this.handleMessage(event.data);
    };
    this.worker.onerror = (event) => {
      this.failAll(new Error(event.message || 'RaptorQ decoder worker failed'));
    };
  }

  static async create(): Promise<DecodeWorkerClient> {
    const client = new DecodeWorkerClient();
    const timer = window.setTimeout(() => {
      client.readyReject(new Error('RaptorQ decoder worker initialization timed out'));
    }, READY_TIMEOUT_MS);
    try {
      await client.readyPromise;
      return client;
    } catch (error) {
      client.stop();
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  consume(text: string): Promise<SessionResult | null> {
    if (this.pendingSymbols >= MAX_PENDING_SYMBOLS) return Promise.resolve(null);
    this.pendingSymbols++;
    return this.request<SessionResult>('consume', { text }).finally(() => {
      this.pendingSymbols--;
    });
  }

  snapshot(): Promise<SessionSnapshot> {
    return this.request<SessionSnapshot>('snapshot');
  }

  resultBytes(): Promise<Uint8Array> {
    return this.request<Uint8Array>('bytes');
  }

  reset(): Promise<void> {
    return this.request<void>('reset');
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.worker.terminate();
    this.failAll(new Error('RaptorQ decoder worker stopped'));
  }

  private request<T>(kind: RequestKind, payload: Record<string, unknown> = {}): Promise<T> {
    if (this.stopped) return Promise.reject(new Error('RaptorQ decoder worker stopped'));
    const id = ++this.nextId;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      this.worker.postMessage({ type: kind, id, ...payload });
    });
  }

  private handleMessage(message: WorkerResponse): void {
    if (message.type === 'ready') {
      this.readyResolve();
      return;
    }
    if (message.type === 'error' && message.id == null) {
      this.readyReject(new Error(message.error));
      return;
    }
    if (!('id' in message) || message.id == null) return;
    const id = message.id;
    const pending = this.pending.get(id);
    if (!pending) return;
    this.pending.delete(id);
    if (message.type === 'error') {
      pending.reject(new Error(message.error));
    } else if (message.type === 'result') {
      pending.resolve(message.result);
    } else if (message.type === 'snapshot') {
      pending.resolve(message.snapshot);
    } else if (message.type === 'bytes') {
      pending.resolve(message.bytes);
    } else {
      pending.resolve(undefined);
    }
  }

  private failAll(error: Error): void {
    this.readyReject(error);
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }
}

export function createDecodeClient(): Promise<DecodeWorkerClient> {
  return DecodeWorkerClient.create();
}
