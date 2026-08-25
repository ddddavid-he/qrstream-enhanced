import init, { WasmDecodeSession } from 'qrstream-decode';

export interface SessionResult {
  accepted: boolean;
  duplicate: boolean;
  done: boolean;
  progress: number;
  num_recovered: number;
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
  symbol_count: number | null;
  filesize: number | null;
  protocol_version: number | null;
}

let wasmReady: Promise<unknown> | null = null;

/** Load the WASM module once; subsequent calls share the promise. */
export function loadWasm(): Promise<unknown> {
  if (!wasmReady) {
    wasmReady = init();
  }
  return wasmReady;
}

/** Create a new decode session (WASM must be loaded first). */
export function createSession(): WasmDecodeSession {
  return new WasmDecodeSession();
}

/** Parse a session result JSON string from the WASM boundary. */
export function parseResult(json: string): SessionResult {
  return JSON.parse(json) as SessionResult;
}

/** Parse a snapshot JSON string from the WASM boundary. */
export function parseSnapshot(json: string): SessionSnapshot {
  return JSON.parse(json) as SessionSnapshot;
}
