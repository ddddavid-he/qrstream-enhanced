"""Adaptive channel calibration for QRStream.

Generates a calibration video with stepped QR versions and frame rates,
then analyzes the captured result to recommend optimal encoding parameters.

Encoder side:
    ``generate_calibration()`` produces a ~25-90s calibration video/display
    with metadata, QR version ladder, FPS ladder, and end marker segments.

Decoder side:
    ``analyze_calibration()`` reads a captured calibration video, computes
    per-step detect rates, and outputs three-tier recommendations
    (safe / balanced / aggressive).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from ._compat import suppress_native_stderr

with suppress_native_stderr():
    import cv2
    import numpy as np
    import av

from .protocol import (
    _alphanumeric_byte_capacity,
    base45_decode,
)
from .qr_utils import generate_qr_image, generate_qr_module_image, try_decode_qr
from .ui import ProgressReporter, QuietReporter

# ── Calibration protocol constants ──────────────────────────────────

CAL_MAGIC = b"QRSCAL"
CAL_VERSION = 1

#: ``struct`` format for the 12-byte calibration frame payload.
#: Fields: magic(6s) cal_version(B) segment_id(B) param(B)
#:         step_index(B) total_steps(B) frame_seq(B)
CAL_STRUCT = ">6sBBBBBB"
CAL_STRUCT_SIZE = struct.calcsize(CAL_STRUCT)  # 12

# Segment IDs
SEG_META = 1
SEG_VERSION = 2
SEG_FPS = 3
SEG_END = 4

# Precision preset IDs (written into meta segment ``param`` field).
# ``quick``/``thorough`` remain accepted aliases for older callers.
PRESET_LOW = 0
PRESET_FAST = 1
PRESET_QUICK = PRESET_FAST
PRESET_STANDARD = 2
PRESET_FULL = 3
PRESET_THOROUGH = PRESET_FULL
PRESET_HIGH = 4

PRESET_NAMES = {
    PRESET_LOW: "low",
    PRESET_FAST: "fast",
    PRESET_STANDARD: "standard",
    PRESET_FULL: "full",
    PRESET_HIGH: "high",
}
_PRESET_ALIASES = {
    "quick": "fast",
    "thorough": "full",
}
PRESET_IDS = {v: k for k, v in PRESET_NAMES.items()}
PRESET_IDS.update({alias: PRESET_IDS[name]
                   for alias, name in _PRESET_ALIASES.items()})

# ── Calibration frame rates used for fixed-fps segments ─────────────

#: Meta/end segments always play at this rate (low, reliable).
_META_FPS = 10

#: Version-ladder segment always plays at this rate.
_VERSION_LADDER_FPS = 10

# ── Preset ladder configurations ────────────────────────────────────

# Version ladders: list of QR version numbers to test.
_VERSION_LADDER_LOW = [5, 8, 10, 12, 15, 17, 20, 22, 25, 28]
_VERSION_LADDER_FAST = [15, 20, 25, 28, 30, 33, 35, 40]
_VERSION_LADDER_QUICK = _VERSION_LADDER_FAST
_VERSION_LADDER_STANDARD = [15, 20, 22, 25, 27, 28, 30, 32, 33, 35, 38, 40]
_VERSION_LADDER_FULL = [
    15, 17, 20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 35, 38, 40,
]
_VERSION_LADDER_THOROUGH = _VERSION_LADDER_FULL
_VERSION_LADDER_HIGH = [25, 28, 30, 32, 33, 35, 36, 38, 39, 40]

# FPS ladders: list of target frame rates to test.
_FPS_LADDER_LOW = [5, 6, 8, 10, 12, 15, 18, 20]
_FPS_LADDER_FAST = [8, 10, 15, 20, 25, 30]
_FPS_LADDER_QUICK = _FPS_LADDER_FAST
_FPS_LADDER_STANDARD = [8, 10, 12, 15, 18, 20, 25, 30]
_FPS_LADDER_FULL = [5, 8, 9, 10, 12, 14, 15, 18, 22, 30]
_FPS_LADDER_THOROUGH = _FPS_LADDER_FULL
# ``high`` uses a candidate pool filtered by display refresh rate.
_FPS_CANDIDATES_HIGH = [15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 100, 120]

# FPS anchor versions (used to encode QR frames in the FPS segment).
_FPS_ANCHOR_LOW = 15
_FPS_ANCHOR_FAST = 25
_FPS_ANCHOR_QUICK = _FPS_ANCHOR_FAST
_FPS_ANCHOR_STANDARD = 25
_FPS_ANCHOR_FULL = 25
_FPS_ANCHOR_THOROUGH = _FPS_ANCHOR_FULL
_FPS_ANCHOR_HIGH = 35

# Target total durations in seconds.  Legacy low/high keep approximate
# targets for API compatibility; the public presets are fast/standard/full.
_PRESET_TARGET_SECONDS = {
    PRESET_LOW: 15.0,
    PRESET_FAST: 15.0,
    PRESET_STANDARD: 30.0,
    PRESET_FULL: 60.0,
    PRESET_HIGH: 60.0,
}

# Kept for older tests/callers that import the constant directly.  Actual
# frame counts are computed from _PRESET_TARGET_SECONDS in resolve_preset().
_FRAMES_PER_STEP = {
    PRESET_LOW: (6, 7),
    PRESET_FAST: (8, 14),
    PRESET_STANDARD: (11, 24),
    PRESET_FULL: (18, 32),
    PRESET_HIGH: (28, 80),
}

_CALIBRATION_EC_LEVEL = 1
_CALIBRATION_BORDER_MODULES = 4.0
_CALIBRATION_BOX_SIZE = 10
_CALIBRATION_PAYLOAD_SAFETY_BYTES = 1

# Meta/end segment timing.
_META_SECONDS = 2.0   # 2s of meta frames
_END_SECONDS = 1.0    # 1s of end frames


@dataclass
class PresetConfig:
    """Resolved ladder configuration for a single preset."""

    preset_id: int
    preset_name: str
    version_ladder: list[int]
    fps_ladder: list[int]
    fps_anchor_version: int
    frames_per_version_step: int
    frames_per_fps_step: int

    @property
    def meta_frames(self) -> int:
        return max(1, round(_META_SECONDS * _META_FPS))

    @property
    def end_frames(self) -> int:
        return max(1, round(_END_SECONDS * _META_FPS))


def _get_display_refresh_rate() -> int:
    """Detect primary monitor refresh rate.  Returns Hz, fallback 60."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        screen = app.primaryScreen()
        if screen is not None:
            rate = round(screen.refreshRate())
            if rate > 0:
                return rate
    except Exception:
        pass
    return 60


def _canonical_preset_name(preset_name: str) -> str:
    return _PRESET_ALIASES.get(preset_name, preset_name)


def _frames_for_target_duration(version_ladder: list[int],
                                fps_ladder: list[int],
                                target_seconds: float) -> tuple[int, int]:
    fixed_seconds = _META_SECONDS + _END_SECONDS
    remaining = max(1.0, target_seconds - fixed_seconds)
    version_seconds = remaining * 0.5
    fps_seconds = remaining - version_seconds

    frames_per_version_step = max(
        2,
        round(version_seconds * _VERSION_LADDER_FPS / len(version_ladder)),
    )
    fps_weight = sum(1.0 / fps for fps in fps_ladder if fps > 0)
    frames_per_fps_step = max(2, round(fps_seconds / fps_weight))
    return frames_per_version_step, frames_per_fps_step


def _estimate_sequence_duration(config: PresetConfig) -> float:
    version_seconds = (
        len(config.version_ladder)
        * config.frames_per_version_step
        / _VERSION_LADDER_FPS
    )
    fps_seconds = sum(
        config.frames_per_fps_step / fps for fps in config.fps_ladder
    )
    return _META_SECONDS + version_seconds + fps_seconds + _END_SECONDS


def _presentation_repeat_count(frame_seq: int, target_fps: int,
                               presentation_fps: int) -> int:
    if target_fps <= 0 or presentation_fps <= 0:
        return 1
    ratio = presentation_fps / target_fps
    start = int(frame_seq * ratio + 0.5)
    end = int((frame_seq + 1) * ratio + 0.5)
    return max(1, end - start)


def _presentation_frame_count(
    frame_seq: list[tuple[CalibrationFrame, int, int]],
    presentation_fps: int,
) -> int:
    return sum(
        _presentation_repeat_count(cf.frame_seq, target_fps, presentation_fps)
        for cf, _qr_ver, target_fps in frame_seq
    )


def resolve_preset(preset_name: str,
                   display_hz: int | None = None) -> PresetConfig:
    """Build a fully resolved :class:`PresetConfig` for *preset_name*.

    Parameters
    ----------
    preset_name:
        One of ``"fast"``, ``"standard"``, ``"full"``.  Legacy aliases
        ``"quick"`` and ``"thorough"`` are accepted.
    display_hz:
        Display refresh rate in Hz (used only by the ``"high"`` preset to
        cap the FPS ladder).  ``None`` means auto-detect.
    """
    canonical_name = _canonical_preset_name(preset_name)
    if canonical_name not in PRESET_IDS:
        raise ValueError(f"Unknown preset: {preset_name!r}")
    pid = PRESET_IDS[canonical_name]

    if canonical_name == "low":
        ver = list(_VERSION_LADDER_LOW)
        fps = list(_FPS_LADDER_LOW)
        anchor = _FPS_ANCHOR_LOW
    elif canonical_name == "fast":
        ver = list(_VERSION_LADDER_FAST)
        fps = list(_FPS_LADDER_FAST)
        anchor = _FPS_ANCHOR_FAST
    elif canonical_name == "standard":
        ver = list(_VERSION_LADDER_STANDARD)
        fps = list(_FPS_LADDER_STANDARD)
        anchor = _FPS_ANCHOR_STANDARD
    elif canonical_name == "full":
        ver = list(_VERSION_LADDER_FULL)
        fps = list(_FPS_LADDER_FULL)
        anchor = _FPS_ANCHOR_FULL
    elif canonical_name == "high":
        ver = list(_VERSION_LADDER_HIGH)
        if display_hz is None:
            display_hz = _get_display_refresh_rate()
        fps = [f for f in _FPS_CANDIDATES_HIGH if f <= display_hz]
        if not fps:
            fps = [_FPS_CANDIDATES_HIGH[0]]  # at least one entry
        anchor = _FPS_ANCHOR_HIGH
    else:
        raise ValueError(f"Unknown preset: {preset_name!r}")

    fpv, fpf = _frames_for_target_duration(
        ver, fps, _PRESET_TARGET_SECONDS[pid])

    return PresetConfig(
        preset_id=pid,
        preset_name=canonical_name,
        version_ladder=ver,
        fps_ladder=fps,
        fps_anchor_version=anchor,
        frames_per_version_step=fpv,
        frames_per_fps_step=fpf,
    )


# ── CalibrationFrame ────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationFrame:
    """A single 12-byte calibration frame payload.

    Attributes
    ----------
    segment_id : int
        ``SEG_META`` (1), ``SEG_VERSION`` (2), ``SEG_FPS`` (3), or
        ``SEG_END`` (4).
    param : int
        Segment-dependent: preset ID for meta, QR version for version
        ladder, target FPS for fps ladder, 0 for end.
    step_index : int
        0-based step number within the current segment.
    total_steps : int
        Total number of steps in the current segment.
    frame_seq : int
        0-based frame number within the current step.
    """

    segment_id: int
    param: int
    step_index: int
    total_steps: int
    frame_seq: int

    def pack(self) -> bytes:
        """Serialize to 12-byte payload."""
        return struct.pack(
            CAL_STRUCT,
            CAL_MAGIC,
            CAL_VERSION,
            self.segment_id,
            self.param,
            self.step_index,
            self.total_steps,
            self.frame_seq,
        )

    @classmethod
    def unpack(cls, data: bytes) -> CalibrationFrame:
        """Deserialize 12 bytes.

        Raises
        ------
        ValueError
            On bad magic or incompatible ``cal_version``.
        """
        if len(data) < CAL_STRUCT_SIZE:
            raise ValueError(
                f"Calibration frame too short: {len(data)} bytes "
                f"(need {CAL_STRUCT_SIZE})"
            )
        magic, ver, seg, param, si, ts, fs = struct.unpack(
            CAL_STRUCT, data[:CAL_STRUCT_SIZE]
        )
        if magic != CAL_MAGIC:
            raise ValueError("Not a calibration frame")
        if ver != CAL_VERSION:
            raise ValueError(
                f"Calibration format v{ver} not supported by this version. "
                "Regenerate calibration video with current qrstream."
            )
        return cls(
            segment_id=seg,
            param=param,
            step_index=si,
            total_steps=ts,
            frame_seq=fs,
        )


# ── Recommendation dataclasses ──────────────────────────────────────

#: Minimum overhead for RaptorQ — never recommend less than this.
_MIN_OVERHEAD_RQ = 1.02

# Recommendation tier definitions.
_TIERS = {
    "safe": {"threshold": 0.98, "safety_margin": 1.05},
    "balanced": {"threshold": 0.85, "safety_margin": 1.15},
    "aggressive": {"threshold": 0.70, "safety_margin": 1.30},
}

#: Detect rate at or above which the boundary is considered excellent.
_EXCELLENT_THRESHOLD = 0.90

#: Detect rate below which the boundary is considered very poor.
_POOR_THRESHOLD = 0.70

#: If FPS anchor version has detect rate below this, FPS data is unreliable.
_FPS_ANCHOR_RELIABILITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class TierRecommendation:
    """One tier of the calibration recommendation."""

    tier: str  # "safe" / "balanced" / "aggressive"
    available: bool
    qr_version: int | None = None
    fps: int | None = None
    overhead: float | None = None
    throughput_bps: float | None = None  # bytes/sec estimate


@dataclass
class CalibrationResult:
    """Full calibration analysis output."""

    preset: str
    channel_quality: str  # "poor" / "fair" / "good" / "excellent"
    version_detect_rates: dict[int, float]  # qr_version -> rate
    fps_detect_rates: dict[int, float]      # target_fps -> rate
    fps_data_reliable: bool
    recommendations: list[TierRecommendation] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _estimate_throughput(qr_version: int, fps: int,
                         overhead: float) -> float:
    """Estimate throughput in bytes/sec for a given parameter set."""
    if not 1 <= qr_version <= 40:
        return 0.0
    capacity = _alphanumeric_byte_capacity(
        qr_version, _CALIBRATION_EC_LEVEL)
    if capacity == 0 or overhead <= 0:
        return 0.0
    return capacity * fps / overhead


def compute_recommendations(
    version_detect_rates: dict[int, float],
    fps_detect_rates: dict[int, float],
    fps_data_reliable: bool,
    preset_name: str,
) -> CalibrationResult:
    """Compute three-tier recommendations from raw detect rates.

    Parameters
    ----------
    version_detect_rates:
        Mapping of QR version -> detect rate (0.0 to 1.0).
    fps_detect_rates:
        Mapping of target FPS -> detect rate (0.0 to 1.0).
    fps_data_reliable:
        False if the FPS anchor version had poor detect rate.
    preset_name:
        The calibration preset used (for display purposes).
    """
    messages: list[str] = []
    recommendations: list[TierRecommendation] = []

    # ── Boundary messages ───────────────────────────────────────────

    if version_detect_rates:
        sorted_versions = sorted(version_detect_rates.keys())
        lowest_ver = sorted_versions[0]
        highest_ver = sorted_versions[-1]
        lowest_ver_dr = version_detect_rates[lowest_ver]
        highest_ver_dr = version_detect_rates[highest_ver]

        if lowest_ver_dr < _POOR_THRESHOLD:
            messages.append(
                f"⚠ Channel quality very poor: V{lowest_ver} detect "
                f"rate only {lowest_ver_dr:.0%}. Consider reducing distance, "
                f"improving lighting, or using higher camera resolution."
            )
        if highest_ver_dr >= _EXCELLENT_THRESHOLD:
            messages.append(
                "ℹ Channel quality excellent: all tested versions "
                "usable. Recommend using max version for best throughput."
            )

    if fps_detect_rates and fps_data_reliable:
        sorted_fps = sorted(fps_detect_rates.keys())
        lowest_fps = sorted_fps[0]
        highest_fps = sorted_fps[-1]
        lowest_fps_dr = fps_detect_rates[lowest_fps]
        highest_fps_dr = fps_detect_rates[highest_fps]

        if lowest_fps_dr < _POOR_THRESHOLD:
            messages.append(
                "⚠ Frame capture rate very low. Check camera focus, "
                "reduce motion blur, or stabilize device."
            )
        if highest_fps_dr >= _EXCELLENT_THRESHOLD:
            messages.append(
                "ℹ Frame capture rate excellent at all tested levels."
            )

    if not fps_data_reliable:
        messages.append(
            "⚠ FPS ladder data unreliable due to low anchor version "
            "detect rate. Defaulting to conservative 10fps estimate."
        )

    # ── Channel quality classification ──────────────────────────────

    if version_detect_rates:
        avg_dr = sum(version_detect_rates.values()) / len(version_detect_rates)
        if avg_dr >= 0.90:
            channel_quality = "excellent"
        elif avg_dr >= 0.70:
            channel_quality = "good"
        elif avg_dr >= 0.50:
            channel_quality = "fair"
        else:
            channel_quality = "poor"
    else:
        channel_quality = "poor"

    # ── Per-tier selection ──────────────────────────────────────────

    for tier_name, cfg in _TIERS.items():
        threshold = cfg["threshold"]
        safety = cfg["safety_margin"]

        # Select max version meeting threshold
        max_ver = None
        for v in sorted(version_detect_rates.keys(), reverse=True):
            if version_detect_rates[v] >= threshold:
                max_ver = v
                break

        # Select max fps meeting threshold
        max_fps = None
        if fps_data_reliable:
            for f in sorted(fps_detect_rates.keys(), reverse=True):
                if fps_detect_rates[f] >= threshold:
                    max_fps = f
                    break
        else:
            max_fps = 10  # conservative fallback

        if max_ver is None or max_fps is None:
            recommendations.append(TierRecommendation(
                tier=tier_name, available=False,
            ))
            continue

        # Compute overhead
        ver_dr = version_detect_rates[max_ver]
        fps_dr = (fps_detect_rates.get(max_fps, 0.90)
                  if fps_data_reliable else 0.90)
        combined_dr = ver_dr * fps_dr
        raw_overhead = 1.0 / combined_dr if combined_dr > 0 else 10.0
        overhead = max(raw_overhead * safety, _MIN_OVERHEAD_RQ)
        # Round to 2 decimal places for display
        overhead = round(overhead, 2)

        throughput = _estimate_throughput(max_ver, max_fps, overhead)

        recommendations.append(TierRecommendation(
            tier=tier_name,
            available=True,
            qr_version=max_ver,
            fps=max_fps,
            overhead=overhead,
            throughput_bps=throughput,
        ))

    # Check if safe tier is unavailable
    if recommendations and not recommendations[0].available:
        messages.append(
            "⚠ Cannot produce reliable recommendation. "
            "Improve capture conditions and retry."
        )

    return CalibrationResult(
        preset=preset_name,
        channel_quality=channel_quality,
        version_detect_rates=version_detect_rates,
        fps_detect_rates=fps_detect_rates,
        fps_data_reliable=fps_data_reliable,
        recommendations=recommendations,
        messages=messages,
    )


# ── Calibration video generation (encoder side) ─────────────────────

def _build_frame_sequence(
    config: PresetConfig,
) -> list[tuple[CalibrationFrame, int, int]]:
    """Build the complete ordered list of calibration frames.

    Returns a list of ``(CalibrationFrame, qr_version_for_encoding,
    target_fps)`` tuples.  ``target_fps`` is only meaningful for
    SEG_FPS frames (determines inter-frame timing); for other segments
    the caller should use the segment's fixed fps.
    """
    frames: list[tuple[CalibrationFrame, int, int]] = []
    meta_version = min(config.version_ladder)  # lowest version = most reliable

    # Seg 1: Meta
    n_meta = config.meta_frames
    for i in range(n_meta):
        cf = CalibrationFrame(
            segment_id=SEG_META,
            param=config.preset_id,
            step_index=0,
            total_steps=1,
            frame_seq=i,
        )
        frames.append((cf, meta_version, _META_FPS))

    # Seg 2: Version ladder
    n_ver_steps = len(config.version_ladder)
    for step_idx, ver in enumerate(config.version_ladder):
        for fseq in range(config.frames_per_version_step):
            cf = CalibrationFrame(
                segment_id=SEG_VERSION,
                param=ver,
                step_index=step_idx,
                total_steps=n_ver_steps,
                frame_seq=fseq,
            )
            frames.append((cf, ver, _VERSION_LADDER_FPS))

    # Seg 3: FPS ladder
    n_fps_steps = len(config.fps_ladder)
    anchor = config.fps_anchor_version
    for step_idx, target_fps in enumerate(config.fps_ladder):
        for fseq in range(config.frames_per_fps_step):
            cf = CalibrationFrame(
                segment_id=SEG_FPS,
                param=target_fps,
                step_index=step_idx,
                total_steps=n_fps_steps,
                frame_seq=fseq,
            )
            frames.append((cf, anchor, target_fps))

    # Seg 4: End
    n_end = config.end_frames
    for i in range(n_end):
        cf = CalibrationFrame(
            segment_id=SEG_END,
            param=0,
            step_index=0,
            total_steps=1,
            frame_seq=i,
        )
        frames.append((cf, meta_version, _META_FPS))

    return frames


def _container_fps(fps_ladder: list[int]) -> int:
    """Choose a container frame rate for MP4 output.

    The container fps must be >= the highest target FPS in the FPS
    ladder.  We use 2x the max target fps (with a floor of 60) so that
    spacer frames can simulate variable frame rates.
    """
    max_target = max(fps_ladder) if fps_ladder else 30
    return max(max_target * 2, 60)


def generate_calibration(
    preset_name: str = "standard",
    output_path: str | None = None,
    display: bool = False,
    display_hz: int | None = None,
    codec: str = "h264",
    reporter: ProgressReporter | None = None,
) -> PresetConfig:
    """Generate a calibration video or display sequence.

    Parameters
    ----------
    preset_name:
        Calibration precision preset.
    output_path:
        Path to write the calibration MP4.  Mutually exclusive with
        *display*.
    display:
        If True, play calibration sequence on screen via Qt player.
    display_hz:
        Override display refresh rate (Hz).  Used by ``high`` preset in
        video mode.  Ignored in display mode (auto-detected).
    codec:
        Video codec for file output.
    reporter:
        Progress reporter.

    Returns
    -------
    PresetConfig
        The resolved preset configuration used.
    """
    if reporter is None:
        reporter = QuietReporter()

    if display:
        # In display mode, auto-detect Hz for high preset
        config = resolve_preset(preset_name, display_hz=None)
    else:
        config = resolve_preset(preset_name, display_hz=display_hz)

    frame_seq = _build_frame_sequence(config)
    total_logical = len(frame_seq)

    duration = _estimate_sequence_duration(config)
    reporter.info(
        f"Calibration: preset={config.preset_name}, "
        f"version_steps={len(config.version_ladder)}, "
        f"fps_steps={len(config.fps_ladder)}, "
        f"duration≈{duration:.1f}s, "
        f"logical_frames={total_logical}"
    )

    if display:
        _generate_display(config, frame_seq, reporter)
    elif output_path:
        _generate_video(config, frame_seq, output_path, codec, reporter)
    else:
        raise ValueError("Either output_path or display must be specified")

    return config


def _calibration_payload(cal_frame: CalibrationFrame,
                         qr_version: int) -> bytes:
    """Return a dense, deterministic calibration payload for one QR."""
    header = cal_frame.pack()
    capacity = _alphanumeric_byte_capacity(qr_version, _CALIBRATION_EC_LEVEL)
    target_size = max(
        len(header),
        capacity - _CALIBRATION_PAYLOAD_SAFETY_BYTES,
    )
    if target_size <= len(header):
        return header

    seed = header + bytes([qr_version])
    payload = bytearray(header)
    counter = 0
    while len(payload) < target_size:
        payload.extend(hashlib.blake2s(
            seed + counter.to_bytes(4, "big"),
            digest_size=32,
        ).digest())
        counter += 1
    return bytes(payload[:target_size])


def _scale_to_fit_square(img: np.ndarray, side: int) -> np.ndarray:
    """Integer-scale *img* as large as possible, then center-pad to side."""
    if img.shape[0] > side or img.shape[1] > side:
        raise ValueError(
            f"image {img.shape[1]}x{img.shape[0]} exceeds canvas {side}x{side}"
        )
    scale = max(1, min(side // img.shape[0], side // img.shape[1]))
    if scale > 1:
        img = cv2.resize(
            img,
            (img.shape[1] * scale, img.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    if img.shape[0] == side and img.shape[1] == side:
        return img
    if img.ndim == 2:
        canvas = np.full((side, side), 255, dtype=img.dtype)
        y_off = (side - img.shape[0]) // 2
        x_off = (side - img.shape[1]) // 2
        canvas[y_off:y_off + img.shape[0], x_off:x_off + img.shape[1]] = img
    else:
        canvas = np.full((side, side, img.shape[2]), 255, dtype=img.dtype)
        y_off = (side - img.shape[0]) // 2
        x_off = (side - img.shape[1]) // 2
        canvas[y_off:y_off + img.shape[0], x_off:x_off + img.shape[1], :] = img
    return canvas


def _encode_cal_frame_to_qr(
    cal_frame: CalibrationFrame,
    qr_version: int,
    border: float = _CALIBRATION_BORDER_MODULES,
) -> np.ndarray:
    """Encode a CalibrationFrame into a BGR QR image."""
    payload_bytes = _calibration_payload(cal_frame, qr_version)
    return generate_qr_image(
        payload_bytes,
        ec_level=_CALIBRATION_EC_LEVEL,
        box_size=_CALIBRATION_BOX_SIZE,
        border=border,
        version=qr_version,
        alphanumeric=True,  # base45 high-density
    )


def _generate_video(
    config: PresetConfig,
    frame_seq: list[tuple[CalibrationFrame, int, int]],
    output_path: str,
    codec: str,
    reporter: ProgressReporter,
) -> None:
    """Write calibration frames to an MP4 file.

    The video uses a fixed square frame.  Lower target FPS values are
    simulated by holding the current QR frame for multiple container frames.
    """
    from .encoder import _PYAV_CODEC_MAP, _PYAV_CONTAINER_FORMAT

    container_rate = _container_fps(config.fps_ladder)

    codec_info = _PYAV_CODEC_MAP.get(codec)
    if codec_info is None:
        raise ValueError(
            f"Unsupported codec: {codec!r}. "
            f"Choose from: {list(_PYAV_CODEC_MAP)}"
        )
    pyav_codec, pix_fmt, _default_ext, stream_opts = codec_info
    container_format = _PYAV_CONTAINER_FORMAT[codec]

    output = av.open(output_path, "w", format=container_format)
    out_stream = output.add_stream(pyav_codec, rate=container_rate)
    frame_side = (
        (4 * max(config.version_ladder) + 17)
        + 2 * int(_CALIBRATION_BORDER_MODULES)
    ) * _CALIBRATION_BOX_SIZE
    out_stream.width = frame_side
    out_stream.height = frame_side
    out_stream.pix_fmt = pix_fmt
    if stream_opts:
        out_stream.options = stream_opts

    written = 0

    try:
        for idx, (cf, qr_ver, target_fps) in enumerate(frame_seq):
            payload_bytes = _calibration_payload(cf, qr_ver)
            mod_img = generate_qr_module_image(
                payload_bytes,
                ec_level=_CALIBRATION_EC_LEVEL,
                border=_CALIBRATION_BORDER_MODULES,
                version=qr_ver,
                alphanumeric=True,
            )
            qr_img = _scale_to_fit_square(mod_img, frame_side)
            qr_img = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)
            repeats = _presentation_repeat_count(
                cf.frame_seq, target_fps, container_rate)
            for _ in range(repeats):
                frame_av = av.VideoFrame.from_ndarray(qr_img, format="bgr24")
                for packet in out_stream.encode(frame_av):
                    output.mux(packet)
                written += 1

            if idx % 50 == 0 or idx == len(frame_seq) - 1:
                pct = (idx + 1) / len(frame_seq) * 100
                reporter.info(
                    f"Generating calibration video: {pct:.0f}% "
                    f"({idx + 1}/{len(frame_seq)} logical frames)"
                )

        # Flush
        for packet in out_stream.encode():
            output.mux(packet)
    finally:
        output.close()

    reporter.info(
        f"Calibration video written: {output_path} "
        f"({written} container frames @ {container_rate}fps)"
    )


def _generate_display(
    config: PresetConfig,
    frame_seq: list[tuple[CalibrationFrame, int, int]],
    reporter: ProgressReporter,
) -> None:
    """Play calibration frames via Qt display player.

    Frames are expanded to the display refresh rate by holding each QR
    update for the appropriate number of refresh ticks.
    """
    from .display_player import DisplayProducerState
    from .display_player_qt import (
        DisplayPlayerQtConfig,
        play_display_qt,
        require_pyside6,
    )
    from .display_cache import (
        ModuleFrameCache,
        plan_module_cache,
        pack_module_image,
    )
    require_pyside6()

    # Determine module-frame cache sizing from the largest version.
    max_ver = max(config.version_ladder)
    border = _CALIBRATION_BORDER_MODULES
    modules_side = 4 * max_ver + 17 + 2 * int(border)

    display_fps = _get_display_refresh_rate()
    total_frames = _presentation_frame_count(frame_seq, display_fps)
    budget = plan_module_cache(total_frames, modules_side, display_fps)
    cache = ModuleFrameCache.from_plan(total_frames, modules_side, budget)
    state = DisplayProducerState(total_frames)

    # Pre-generate all presentation frames into the cache.  Smaller QR
    # versions are resized, not padded, so every version fills the window.
    out_idx = 0
    for cf, qr_ver, target_fps in frame_seq:
        payload_bytes = _calibration_payload(cf, qr_ver)
        mod_img = generate_qr_module_image(
            payload_bytes,
            ec_level=_CALIBRATION_EC_LEVEL,
            border=border,
            version=qr_ver,
            alphanumeric=True,
        )
        mod_img = _scale_to_fit_square(mod_img, modules_side)
        packed = pack_module_image(mod_img)
        repeats = _presentation_repeat_count(
            cf.frame_seq, target_fps, display_fps)
        for _ in range(repeats):
            cache.put_packed(out_idx, packed)
            state.mark_produced()
            out_idx += 1
    cache.mark_done()
    state.mark_done()

    player_config = DisplayPlayerQtConfig(
        title="QRStream Calibration",
        lock_window_size=True,
    )
    play_display_qt(cache, state, display_fps, config=player_config)
    reporter.info("Calibration display complete.")


# ── Calibration video analysis (decoder side) ───────────────────────

def analyze_calibration(
    video_path: str,
    workers: int | None = None,
    reporter: ProgressReporter | None = None,
) -> CalibrationResult:
    """Analyze a captured calibration video and produce recommendations.

    Parameters
    ----------
    video_path:
        Path to the captured calibration video.
    workers:
        Number of parallel decode workers (unused for now, reserved).
    reporter:
        Progress reporter.
    """
    if reporter is None:
        reporter = QuietReporter()

    reporter.info(f"Analyzing calibration video: {video_path}")

    # ── Phase 1: Read all frames and attempt QR decode ──────────────

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames or 0
    # If total_frames is 0 (some containers don't report it), count later.

    decoded_frames: list[CalibrationFrame] = []
    frame_count = 0
    preset_name = "standard"  # default, may be overridden by meta segment

    for av_frame in container.decode(video=0):
        frame_count += 1
        img = av_frame.to_ndarray(format="bgr24")
        text = try_decode_qr(img)
        if text is None:
            continue

        # Decode the QR payload
        try:
            # CalibrationFrame is encoded via base45 into QR alphanumeric
            # mode; the detected text is the base45-encoded payload.
            raw = base45_decode(text)
            cf = CalibrationFrame.unpack(raw)
            decoded_frames.append(cf)

            # Extract preset from meta segment
            if cf.segment_id == SEG_META:
                pid = cf.param
                if pid in PRESET_NAMES:
                    preset_name = PRESET_NAMES[pid]

        except (ValueError, struct.error):
            # Not a calibration frame — skip.
            continue

        if frame_count % 200 == 0:
            reporter.info(
                f"Analyzing: {frame_count} video frames scanned, "
                f"{len(decoded_frames)} calibration frames decoded"
            )

    container.close()

    reporter.info(
        f"Scan complete: {frame_count} video frames, "
        f"{len(decoded_frames)} calibration frames decoded"
    )

    if not decoded_frames:
        return CalibrationResult(
            preset=preset_name,
            channel_quality="poor",
            version_detect_rates={},
            fps_detect_rates={},
            fps_data_reliable=False,
            messages=[
                "⚠ No calibration frames detected. Ensure the video "
                "contains a valid QRStream calibration sequence."
            ],
        )

    # ── Phase 2: Group by segment and step, compute detect rates ────

    # Version segment: group by param (= qr_version).  Held presentation
    # frames repeat the same logical QR, so count unique step/frame IDs.
    version_seen: dict[int, set[tuple[int, int]]] = {}
    version_expected: dict[int, int] = {}     # version -> expected count

    # FPS segment: group by param (= target_fps)
    fps_seen: dict[int, set[tuple[int, int]]] = {}
    fps_expected: dict[int, int] = {}

    for cf in decoded_frames:
        key = (cf.step_index, cf.frame_seq)
        if cf.segment_id == SEG_VERSION:
            version_seen.setdefault(cf.param, set()).add(key)
        elif cf.segment_id == SEG_FPS:
            fps_seen.setdefault(cf.param, set()).add(key)

    # To determine expected counts, we resolve the preset config and
    # use its frames_per_step values.  However, the decode side may not
    # know the exact config.  We derive it from the decoded frames:
    # for each step, total_steps and the largest frame_seq seen give us
    # the expected count.

    # Rebuild expected counts from decoded frames' metadata.
    # For version: group by step_index, take max frame_seq + 1 or
    # use total_steps and the known frames-per-step.
    # Simpler approach: use the preset config.
    try:
        config = resolve_preset(preset_name, display_hz=60)
    except ValueError:
        config = resolve_preset("standard", display_hz=60)

    for ver in config.version_ladder:
        version_expected[ver] = config.frames_per_version_step
    for fps in config.fps_ladder:
        fps_expected[fps] = config.frames_per_fps_step

    # Compute detect rates
    version_detect_rates: dict[int, float] = {}
    for ver in sorted(version_expected.keys()):
        expected = version_expected[ver]
        detected = len(version_seen.get(ver, set()))
        version_detect_rates[ver] = min(detected / expected, 1.0) if expected > 0 else 0.0

    fps_detect_rates: dict[int, float] = {}
    for fps in sorted(fps_expected.keys()):
        expected = fps_expected[fps]
        detected = len(fps_seen.get(fps, set()))
        fps_detect_rates[fps] = min(detected / expected, 1.0) if expected > 0 else 0.0

    # ── Phase 3: Check FPS anchor reliability ───────────────────────

    anchor_ver = config.fps_anchor_version
    anchor_dr = version_detect_rates.get(anchor_ver, 0.0)
    fps_data_reliable = anchor_dr >= _FPS_ANCHOR_RELIABILITY_THRESHOLD

    if not fps_data_reliable and anchor_ver in version_detect_rates:
        reporter.warn(
            f"FPS ladder recorded at V{anchor_ver} which has only "
            f"{anchor_dr:.0%} detect rate on this channel. "
            "FPS data unreliable — defaulting to conservative "
            "10fps estimate."
        )

    # ── Phase 4: Compute recommendations ────────────────────────────

    result = compute_recommendations(
        version_detect_rates=version_detect_rates,
        fps_detect_rates=fps_detect_rates,
        fps_data_reliable=fps_data_reliable,
        preset_name=preset_name,
    )

    return result


# ── Pretty-printing results (Rich) ─────────────────────────────────

def format_results(result: CalibrationResult) -> str:
    """Format calibration results as a human-readable string.

    Uses plain text formatting (no Rich markup) for maximum
    compatibility.  The CLI layer may further wrap this in Rich panels.
    """
    lines: list[str] = []
    lines.append("QRStream Calibration Results")
    lines.append("=" * 50)
    lines.append(f"  Channel quality : {result.channel_quality.capitalize()}")
    lines.append(f"  Precision       : {result.preset}")
    lines.append("")

    # Messages / warnings
    for msg in result.messages:
        lines.append(f"  {msg}")
    if result.messages:
        lines.append("")

    # Recommendations table
    if any(r.available for r in result.recommendations):
        lines.append(
            f"  {'Tier':<12} {'Version':>8} {'FPS':>6} "
            f"{'Overhead':>9} {'Throughput':>12}"
        )
        lines.append(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*9} {'-'*12}")
        for rec in result.recommendations:
            if rec.available:
                tp = _format_throughput(rec.throughput_bps or 0)
                lines.append(
                    f"  {rec.tier.capitalize():<12} "
                    f"{'V' + str(rec.qr_version):>8} "
                    f"{rec.fps:>6} "
                    f"{rec.overhead:>9.2f} "
                    f"{tp:>12}"
                )
            else:
                lines.append(
                    f"  {rec.tier.capitalize():<12}    "
                    f"{'-- unavailable --':>38}"
                )
        lines.append("")

        # Recommended command
        balanced = next(
            (r for r in result.recommendations
             if r.tier == "balanced" and r.available),
            None,
        )
        if balanced is None:
            # Fall back to first available tier
            balanced = next(
                (r for r in result.recommendations if r.available), None)
        if balanced:
            lines.append("  Recommended encode command:")
            lines.append(
                f"  qrstream encode FILE "
                f"-v {balanced.qr_version} "
                f"--fps {balanced.fps} "
                f"--overhead {balanced.overhead}"
            )
    else:
        lines.append("  No recommendations available.")

    lines.append("")
    return "\n".join(lines)


def _format_throughput(bps: float) -> str:
    """Format throughput as human-readable string."""
    if bps >= 1024 * 1024:
        return f"~{bps / (1024 * 1024):.1f} MB/s"
    elif bps >= 1024:
        return f"~{bps / 1024:.1f} KB/s"
    else:
        return f"~{bps:.0f} B/s"
