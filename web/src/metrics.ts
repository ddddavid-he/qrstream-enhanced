import type { SessionResult, SessionSnapshot } from './decode';

const WINDOW_MS = 3_000;
const MAX_TIMINGS = 180;

export interface RuntimeMetrics {
  detector: string;
  elapsedSeconds: number;
  cameraFps: number;
  scanFps: number;
  qrFps: number;
  acceptedFps: number;
  payloadBytesPerSecond: number;
  hitRate: number;
  callbackFrames: number;
  scans: number;
  qrFrames: number;
  qrTexts: number;
  accepted: number;
  duplicates: number;
  invalid: number;
  detectAverageMs: number;
  detectP95Ms: number;
  decodeAverageMs: number;
  videoWidth: number;
  videoHeight: number;
  trackFps: number | null;
}

export class MetricsTracker {
  private startedAt = performance.now();
  private callbackTimes: number[] = [];
  private scanTimes: number[] = [];
  private qrTimes: number[] = [];
  private acceptedTimes: number[] = [];
  private detectDurations: number[] = [];
  private decodeDurations: number[] = [];
  private callbackFrames = 0;
  private scans = 0;
  private qrFrames = 0;
  private qrTexts = 0;
  private accepted = 0;
  private duplicates = 0;
  private invalid = 0;
  private presentedSamples: { time: number; frames: number }[] = [];

  reset(now = performance.now()): void {
    this.startedAt = now;
    this.callbackTimes = [];
    this.scanTimes = [];
    this.qrTimes = [];
    this.acceptedTimes = [];
    this.detectDurations = [];
    this.decodeDurations = [];
    this.callbackFrames = 0;
    this.scans = 0;
    this.qrFrames = 0;
    this.qrTexts = 0;
    this.accepted = 0;
    this.duplicates = 0;
    this.invalid = 0;
    this.presentedSamples = [];
  }

  recordVideoFrame(now: number, presentedFrames?: number): void {
    this.callbackFrames++;
    this.pushTime(this.callbackTimes, now);
    if (presentedFrames != null) {
      this.presentedSamples.push({ time: now, frames: presentedFrames });
      const cutoff = now - WINDOW_MS;
      while (this.presentedSamples.length > 0 && this.presentedSamples[0].time < cutoff) {
        this.presentedSamples.shift();
      }
    }
  }

  recordScan(now: number, durationMs: number, qrTexts: number): void {
    this.scans++;
    this.pushTime(this.scanTimes, now);
    this.pushTiming(this.detectDurations, durationMs);
    this.qrTexts += qrTexts;
    if (qrTexts > 0) {
      this.qrFrames++;
      this.pushTime(this.qrTimes, now);
    }
  }

  recordDecode(now: number, durationMs: number, result: SessionResult): void {
    this.pushTiming(this.decodeDurations, durationMs);
    if (result.accepted && !result.duplicate) {
      this.accepted++;
      this.pushTime(this.acceptedTimes, now);
    } else if (result.duplicate) {
      this.duplicates++;
    } else {
      this.invalid++;
    }
  }

  snapshot(
    detector: string,
    video: HTMLVideoElement,
    track: MediaStreamTrack | undefined,
    session: SessionSnapshot,
    now = performance.now(),
  ): RuntimeMetrics {
    this.prune(this.callbackTimes, now);
    this.prune(this.scanTimes, now);
    this.prune(this.qrTimes, now);
    this.prune(this.acceptedTimes, now);
    const elapsedMs = Math.max(1, now - this.startedAt);
    const trackFps = track?.getSettings().frameRate ?? null;
    const firstPresented = this.presentedSamples[0];
    const lastPresented = this.presentedSamples[this.presentedSamples.length - 1];
    const observedCameraFps =
      firstPresented && lastPresented && lastPresented.time > firstPresented.time
        ? ((lastPresented.frames - firstPresented.frames) * 1000) /
          (lastPresented.time - firstPresented.time)
        : (trackFps ?? this.rate(this.callbackTimes, elapsedMs));
    const acceptedFps = this.rate(this.acceptedTimes, elapsedMs);
    const symbolBytes =
      session.filesize != null && session.symbol_count
        ? session.filesize / session.symbol_count
        : 0;

    return {
      detector,
      elapsedSeconds: elapsedMs / 1000,
      cameraFps: observedCameraFps,
      scanFps: this.rate(this.scanTimes, elapsedMs),
      qrFps: this.rate(this.qrTimes, elapsedMs),
      acceptedFps,
      payloadBytesPerSecond: acceptedFps * symbolBytes,
      hitRate: this.scanTimes.length === 0 ? 0 : this.qrTimes.length / this.scanTimes.length,
      callbackFrames: this.callbackFrames,
      scans: this.scans,
      qrFrames: this.qrFrames,
      qrTexts: this.qrTexts,
      accepted: this.accepted,
      duplicates: this.duplicates,
      invalid: this.invalid,
      detectAverageMs: average(this.detectDurations),
      detectP95Ms: percentile(this.detectDurations, 0.95),
      decodeAverageMs: average(this.decodeDurations),
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      trackFps,
    };
  }

  private rate(times: number[], elapsedMs: number): number {
    const seconds = Math.min(WINDOW_MS, elapsedMs) / 1000;
    return seconds > 0 ? times.length / seconds : 0;
  }

  private pushTime(times: number[], now: number): void {
    times.push(now);
    this.prune(times, now);
  }

  private prune(times: number[], now: number): void {
    const cutoff = now - WINDOW_MS;
    while (times.length > 0 && times[0] < cutoff) times.shift();
  }

  private pushTiming(values: number[], value: number): void {
    values.push(value);
    if (values.length > MAX_TIMINGS) values.shift();
  }
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}
