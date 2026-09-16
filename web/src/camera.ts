/** Camera acquisition: getUserMedia with rear-camera preference. */

export interface CameraHandle {
  stream: MediaStream;
  video: HTMLVideoElement;
  videoTrack: MediaStreamTrack;
  stop(): void;
}

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

  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 3840 },
      height: { ideal: 2160 },
      frameRate: { ideal: 60, max: 60 },
    },
  };

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    // Retry without the facingMode constraint (some desktops reject it).
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
    } catch (err2) {
      const primary = err instanceof DOMException ? err : (err2 as DOMException);
      throw classifyError(primary);
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

  return {
    stream,
    video,
    videoTrack,
    stop() {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      video.srcObject = null;
    },
  };
}
