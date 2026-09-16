/** Camera acquisition: getUserMedia with rear-camera preference. */

export interface CameraHandle {
  stream: MediaStream;
  video: HTMLVideoElement;
  videoTrack: MediaStreamTrack;
  resolutionOptions: ResolutionOption[];
  setResolution(id: ResolutionId): Promise<MediaTrackSettings>;
  stop(): void;
}

export type ResolutionId = '720p' | '1080p' | '1440p' | '4k';

export interface ResolutionOption {
  id: ResolutionId;
  label: string;
  width: number;
  height: number;
}

const RESOLUTION_PRESETS: readonly ResolutionOption[] = [
  { id: '720p', label: '720p', width: 1280, height: 720 },
  { id: '1080p', label: '1080p', width: 1920, height: 1080 },
  { id: '1440p', label: '1440p', width: 2560, height: 1440 },
  { id: '4k', label: '4K', width: 3840, height: 2160 },
];

export class CameraError extends Error {
  constructor(
    message: string,
    public readonly kind:
      | 'insecure-context'
      | 'not-supported'
      | 'permission-denied'
      | 'not-found'
      | 'unknown',
  ) {
    super(message);
    this.name = 'CameraError';
  }
}

function classifyError(err: DOMException): CameraError {
  switch (err.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return new CameraError(
        'Camera permission denied. Allow camera access and retry.',
        'permission-denied',
      );
    case 'NotFoundError':
    case 'OverconstrainedError':
      return new CameraError('No suitable camera found on this device.', 'not-found');
    default:
      return new CameraError(`Camera error: ${err.message}`, 'unknown');
  }
}

function inRange(value: number, range: ULongRange | undefined): boolean {
  return !range || (value >= (range.min ?? 0) && value <= (range.max ?? Infinity));
}

function supportsPreset(
  preset: ResolutionOption,
  capabilities: MediaTrackCapabilities,
): boolean {
  const landscape = inRange(preset.width, capabilities.width) &&
    inRange(preset.height, capabilities.height);
  const portrait = inRange(preset.height, capabilities.width) &&
    inRange(preset.width, capabilities.height);
  return landscape || portrait;
}

function availableResolutions(track: MediaStreamTrack): ResolutionOption[] {
  try {
    const capabilities = track.getCapabilities();
    if (capabilities.width && capabilities.height) {
      return RESOLUTION_PRESETS.filter((preset) => supportsPreset(preset, capabilities));
    }
  } catch {
    // Older Safari builds may expose getCapabilities but still throw.
  }

  // Conservative fallback: only advertise standard modes no larger than the
  // currently negotiated stream. Exact applyConstraints remains the authority.
  const settings = track.getSettings();
  const longEdge = Math.max(settings.width ?? 0, settings.height ?? 0);
  const shortEdge = Math.min(settings.width ?? 0, settings.height ?? 0);
  return RESOLUTION_PRESETS.filter(
    (preset) => preset.width <= longEdge && preset.height <= shortEdge,
  );
}

/** Start the camera and attach the stream to a video element. */
export async function startCamera(video: HTMLVideoElement): Promise<CameraHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    if (!window.isSecureContext) {
      throw new CameraError(
        'Camera requires HTTPS (or localhost). Open the page over a secure context.',
        'insecure-context',
      );
    }
    throw new CameraError('getUserMedia is not supported in this browser.', 'not-supported');
  }

  // Prefer a high-frame-rate 1080p mode. Asking iOS for 4K and 60 fps with
  // only "ideal" hints can make Safari select a 4K/24 capture preset, while
  // every frame is downscaled before QR detection anyway.
  const performanceConstraints: MediaStreamConstraints = {
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1920, max: 1920 },
      height: { ideal: 1080, max: 1080 },
      frameRate: { ideal: 60, min: 30, max: 60 },
    },
  };

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(performanceConstraints);
  } catch (err) {
    // Some browsers reject a minimum frame rate even when the camera can run
    // faster. Keep the rear-camera and 1080p preferences on the first retry.
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1920, max: 1920 },
          height: { ideal: 1080, max: 1080 },
          frameRate: { ideal: 60, max: 60 },
        },
      });
    } catch (err2) {
      // Final compatibility fallback for older desktop browsers.
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      } catch (err3) {
        const primary = err instanceof DOMException
          ? err
          : err2 instanceof DOMException
            ? err2
            : (err3 as DOMException);
        throw classifyError(primary);
      }
    }
  }

  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  await video.play();

  const videoTrack = stream.getVideoTracks()[0];
  if (!videoTrack) {
    for (const track of stream.getTracks()) track.stop();
    throw new CameraError('Camera stream has no video track.', 'not-found');
  }

  const resolutionOptions = availableResolutions(videoTrack);

  return {
    stream,
    video,
    videoTrack,
    resolutionOptions,
    async setResolution(id: ResolutionId) {
      const preset = resolutionOptions.find((candidate) => candidate.id === id);
      if (!preset) throw new Error(`Resolution ${id} is not supported by this camera`);
      const current = videoTrack.getSettings();
      const portrait = (current.width ?? 0) < (current.height ?? 0);
      await videoTrack.applyConstraints({
        width: { exact: portrait ? preset.height : preset.width },
        height: { exact: portrait ? preset.width : preset.height },
        frameRate: { ideal: 60, max: 60 },
      });
      return videoTrack.getSettings();
    },
    stop() {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      video.srcObject = null;
    },
  };
}
