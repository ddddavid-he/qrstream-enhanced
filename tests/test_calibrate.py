"""Tests for the calibrate module (channel calibration)."""

import struct

import pytest

from qrstream.calibrate import (
    CAL_MAGIC,
    CAL_STRUCT_SIZE,
    CAL_VERSION,
    CalibrationFrame,
    CalibrationResult,
    PresetConfig,
    PRESET_IDS,
    PRESET_NAMES,
    SEG_END,
    SEG_FPS,
    SEG_META,
    SEG_VERSION,
    TierRecommendation,
    VideoMetadata,
    _EXCELLENT_THRESHOLD,
    _FPS_ANCHOR_RELIABILITY_THRESHOLD,
    _MIN_OVERHEAD_RQ,
    _POOR_THRESHOLD,
    _TIERS,
    _CALIBRATION_EC_LEVEL,
    _build_frame_sequence,
    _calibration_payload,
    _container_fps,
    _estimate_sequence_duration,
    _estimate_throughput,
    compute_recommendations,
    estimate_target_k,
    format_results,
    resolve_preset,
)
from qrstream.overhead_policy import MIN_OVERHEAD_RQ
from qrstream.protocol import _alphanumeric_byte_capacity


CANONICAL_PRESETS = ["low", "fast", "standard", "full", "high"]


# ── CalibrationFrame pack/unpack ────────────────────────────────────

class TestCalibrationFrame:
    """Unit tests for 12-byte CalibrationFrame serialization."""

    @pytest.mark.parametrize("segment_id,param,step_idx,total,fseq", [
        (SEG_META, 2, 0, 1, 0),
        (SEG_VERSION, 25, 3, 12, 10),
        (SEG_FPS, 15, 5, 8, 29),
        (SEG_END, 0, 0, 1, 0),
        # Edge: max values within uint8
        (SEG_VERSION, 40, 15, 16, 44),
        (SEG_FPS, 120, 13, 14, 39),
    ], ids=[
        "meta-preset2",
        "version-V25-step3",
        "fps-15-step5",
        "end-marker",
        "version-max-steps",
        "fps-120hz",
    ])
    def test_pack_unpack_roundtrip(self, segment_id, param,
                                   step_idx, total, fseq):
        cf = CalibrationFrame(
            segment_id=segment_id,
            param=param,
            step_index=step_idx,
            total_steps=total,
            frame_seq=fseq,
        )
        packed = cf.pack()
        assert len(packed) == CAL_STRUCT_SIZE == 12
        assert packed[:6] == CAL_MAGIC

        unpacked = CalibrationFrame.unpack(packed)
        assert unpacked == cf

    def test_pack_starts_with_magic(self):
        cf = CalibrationFrame(SEG_META, 0, 0, 1, 0)
        assert cf.pack()[:6] == b"QRSCAL"

    def test_pack_version_byte(self):
        cf = CalibrationFrame(SEG_META, 0, 0, 1, 0)
        packed = cf.pack()
        assert packed[6] == CAL_VERSION

    def test_unpack_bad_magic(self):
        bad = b"BADMAG" + b"\x01\x01\x00\x00\x01\x00"
        with pytest.raises(ValueError, match="Not a calibration frame"):
            CalibrationFrame.unpack(bad)

    def test_unpack_wrong_version(self):
        # Build a valid frame then tweak the version byte.
        cf = CalibrationFrame(SEG_META, 0, 0, 1, 0)
        packed = bytearray(cf.pack())
        packed[6] = 99  # bad version
        with pytest.raises(ValueError, match="v99 not supported"):
            CalibrationFrame.unpack(bytes(packed))

    def test_unpack_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            CalibrationFrame.unpack(b"QRSCAL\x01")

    def test_unpack_ignores_trailing_bytes(self):
        """Extra bytes beyond 12 are silently ignored."""
        cf = CalibrationFrame(SEG_VERSION, 30, 5, 12, 0)
        packed = cf.pack() + b"\xff\xff\xff"
        unpacked = CalibrationFrame.unpack(packed)
        assert unpacked == cf

    def test_dynamic_payload_keeps_header_unpackable(self):
        cf = CalibrationFrame(SEG_VERSION, 30, 5, 12, 0)
        payload = _calibration_payload(cf, 30)
        assert len(payload) > CAL_STRUCT_SIZE
        assert CalibrationFrame.unpack(payload) == cf

    def test_dynamic_payload_changes_by_frame(self):
        cf1 = CalibrationFrame(SEG_VERSION, 30, 5, 12, 0)
        cf2 = CalibrationFrame(SEG_VERSION, 30, 5, 12, 1)
        assert _calibration_payload(cf1, 30) != _calibration_payload(cf2, 30)


# ── Preset configuration ────────────────────────────────────────────

class TestPresetConfig:
    """Unit tests for preset ladder configurations."""

    @pytest.mark.parametrize("name", CANONICAL_PRESETS)
    def test_resolve_preset(self, name):
        cfg = resolve_preset(name, display_hz=60)
        assert cfg.preset_name == name
        assert cfg.preset_id == PRESET_IDS[name]
        assert len(cfg.version_ladder) > 0
        assert len(cfg.fps_ladder) > 0

    @pytest.mark.parametrize("name", CANONICAL_PRESETS)
    def test_version_ladder_monotonic(self, name):
        cfg = resolve_preset(name, display_hz=60)
        for i in range(1, len(cfg.version_ladder)):
            assert cfg.version_ladder[i] > cfg.version_ladder[i - 1], (
                f"{name}: version ladder not monotonically increasing: "
                f"{cfg.version_ladder}"
            )

    @pytest.mark.parametrize("name", CANONICAL_PRESETS)
    def test_fps_ladder_monotonic(self, name):
        cfg = resolve_preset(name, display_hz=60)
        for i in range(1, len(cfg.fps_ladder)):
            assert cfg.fps_ladder[i] > cfg.fps_ladder[i - 1], (
                f"{name}: fps ladder not monotonically increasing: "
                f"{cfg.fps_ladder}"
            )

    @pytest.mark.parametrize("name", CANONICAL_PRESETS)
    def test_version_ladder_within_qr_range(self, name):
        cfg = resolve_preset(name, display_hz=60)
        for v in cfg.version_ladder:
            assert 1 <= v <= 40, f"{name}: version {v} out of QR range"

    def test_non_low_preset_fps_caps_at_min_display_hz_60(self):
        cfg_50 = resolve_preset("standard", display_hz=50)
        cfg_120 = resolve_preset("standard", display_hz=120)
        assert max(cfg_50.fps_ladder) == 50
        assert max(cfg_120.fps_ladder) == 60

    def test_high_preset_fps_extends_to_display_hz(self):
        cfg_60 = resolve_preset("high", display_hz=60)
        cfg_144 = resolve_preset("high", display_hz=144)
        assert max(cfg_60.fps_ladder) == 60
        assert max(cfg_144.fps_ladder) == 144
        assert len(cfg_144.fps_ladder) >= len(cfg_60.fps_ladder)

    def test_high_preset_fps_at_least_one_entry(self):
        # Even with a very low Hz, should have at least one entry.
        cfg = resolve_preset("high", display_hz=10)
        assert len(cfg.fps_ladder) >= 1

    def test_preset_names_roundtrip(self):
        for pid, name in PRESET_NAMES.items():
            assert PRESET_IDS[name] == pid

    @pytest.mark.parametrize("alias,canonical", [
        ("quick", "fast"),
        ("thorough", "full"),
    ])
    def test_legacy_preset_aliases(self, alias, canonical):
        cfg = resolve_preset(alias, display_hz=60)
        assert cfg.preset_name == canonical
        assert cfg.preset_id == PRESET_IDS[canonical]

    @pytest.mark.parametrize("name,target", [
        ("fast", 15.0),
        ("standard", 30.0),
        ("full", 60.0),
    ])
    def test_public_preset_duration_targets(self, name, target):
        cfg = resolve_preset(name, display_hz=60)
        assert _estimate_sequence_duration(cfg) == pytest.approx(target, abs=1.0)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_preset("nonexistent", display_hz=60)


# ── Frame sequence building ─────────────────────────────────────────

class TestFrameSequence:
    """Tests for the calibration frame sequence builder."""

    def test_standard_frame_count(self):
        cfg = resolve_preset("standard", display_hz=60)
        frames = _build_frame_sequence(cfg)
        # meta + version + fps + end
        expected = (cfg.meta_frames
                    + len(cfg.version_ladder) * cfg.frames_per_version_step
                    + len(cfg.fps_ladder) * cfg.frames_per_fps_step
                    + cfg.end_frames)
        assert len(frames) == expected

    @pytest.mark.parametrize("name", CANONICAL_PRESETS)
    def test_frame_sequence_segment_order(self, name):
        cfg = resolve_preset(name, display_hz=60)
        frames = _build_frame_sequence(cfg)
        # Verify segments appear in order: META -> VERSION -> FPS -> END
        seen_segments = []
        for cf, _ver, _fps in frames:
            if not seen_segments or seen_segments[-1] != cf.segment_id:
                seen_segments.append(cf.segment_id)
        assert seen_segments == [SEG_META, SEG_VERSION, SEG_FPS, SEG_END]

    def test_version_segment_params_match_ladder(self):
        cfg = resolve_preset("standard", display_hz=60)
        frames = _build_frame_sequence(cfg)
        ver_params = []
        for cf, _ver, _fps in frames:
            if cf.segment_id == SEG_VERSION and cf.frame_seq == 0:
                ver_params.append(cf.param)
        assert ver_params == cfg.version_ladder

    def test_fps_segment_params_match_ladder(self):
        cfg = resolve_preset("standard", display_hz=60)
        frames = _build_frame_sequence(cfg)
        fps_params = []
        for cf, _ver, _fps in frames:
            if cf.segment_id == SEG_FPS and cf.frame_seq == 0:
                fps_params.append(cf.param)
        assert fps_params == cfg.fps_ladder


# ── Recommendation algorithm ────────────────────────────────────────

class TestRecommendations:
    """Unit tests for the three-tier recommendation engine."""

    def _make_rates(self, versions, rate):
        """Helper: uniform detect rate for all versions."""
        return {v: rate for v in versions}

    def test_perfect_channel_all_tiers_available(self):
        """100% detect rate can be recommended for every risk tier."""
        ver_rates = self._make_rates([15, 20, 25, 30, 35, 40], 1.0)
        fps_rates = self._make_rates([8, 10, 15, 20, 25, 30], 1.0)

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        by_tier = {r.tier: r for r in result.recommendations}
        assert result.channel_quality == "excellent"
        assert by_tier["safe"].available
        assert by_tier["safe"].qr_version == 40
        assert by_tier["safe"].fps == 30
        assert by_tier["safe"].overhead >= _MIN_OVERHEAD_RQ
        assert by_tier["balanced"].available
        assert by_tier["balanced"].qr_version == 40
        assert by_tier["balanced"].fps == 30
        assert by_tier["aggressive"].available
        assert by_tier["aggressive"].qr_version == 40
        assert by_tier["aggressive"].fps == 30

    def test_tier_targets_have_probability_semantics(self):
        """Tiers are gated by estimated decode success probability."""
        ver_rates = {40: 1.0}
        fps_rates = {10: 0.95, 15: 0.85, 20: 0.75, 25: 0.65}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        by_tier = {r.tier: r for r in result.recommendations}

        assert by_tier["safe"].available
        assert by_tier["safe"].estimated_success >= 0.99
        assert by_tier["balanced"].available
        assert by_tier["balanced"].estimated_success >= 0.95
        assert by_tier["aggressive"].available
        assert by_tier["aggressive"].estimated_success >= 0.90

    def test_high_quality_channel_has_all_tiers(self):
        ver_rates = {40: 1.0}
        fps_rates = {12: 0.96, 30: 0.83}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        by_tier = {r.tier: r for r in result.recommendations}
        assert by_tier["safe"].available
        assert by_tier["balanced"].available
        assert by_tier["aggressive"].available

    def test_poor_channel_safe_unavailable(self):
        """Low detect rates -> safe tier unavailable."""
        ver_rates = {15: 0.60, 20: 0.40, 25: 0.20}
        fps_rates = {8: 0.80, 10: 0.50, 15: 0.30}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        safe = result.recommendations[0]
        assert safe.tier == "safe"
        assert not safe.available
        assert any("Safe tier unavailable" in m for m in result.messages)

    def test_safe_unavailable_but_balanced_available_is_not_fatal(self):
        ver_rates = {40: 1.0}
        fps_rates = {15: 0.64}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        by_tier = {r.tier: r for r in result.recommendations}
        assert not by_tier["safe"].available
        assert by_tier["balanced"].available
        assert not any(
            "Cannot produce reliable" in m for m in result.messages)
        assert any("Safe tier unavailable" in m for m in result.messages)

    def test_excellent_boundary_message(self):
        """All versions >= 90% -> concise headroom message."""
        ver_rates = self._make_rates([15, 20, 25, 30, 35, 40], 0.95)
        fps_rates = self._make_rates([8, 10, 15, 20, 25, 30], 0.95)

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        assert any("headroom" in m.lower() for m in result.messages)

    def test_poor_boundary_message(self):
        """Lowest version < 70% -> suggests low precision preset."""
        ver_rates = {15: 0.50, 20: 0.90, 25: 0.95}
        fps_rates = {8: 0.95, 10: 0.90}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        assert any("--precision low" in m for m in result.messages)

    def test_fps_unreliable_fallback(self):
        """When FPS data unreliable, all tiers use fps=10."""
        ver_rates = self._make_rates([15, 20, 25, 30], 0.99)
        fps_rates = self._make_rates([8, 10, 15, 20], 0.99)

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=False,
            preset_name="standard",
        )
        for rec in result.recommendations:
            if rec.available:
                assert rec.fps == 10

    def test_tier_order(self):
        """Recommendations always in order: safe, balanced, aggressive."""
        ver_rates = self._make_rates([15, 20, 25, 30], 0.90)
        fps_rates = self._make_rates([8, 10, 15, 20], 0.90)

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        tier_names = [r.tier for r in result.recommendations]
        assert tier_names == ["safe", "balanced", "aggressive"]

    def test_overhead_never_below_minimum(self):
        """Even with 100% detect rate, overhead >= shared RaptorQ floor."""
        assert _MIN_OVERHEAD_RQ == MIN_OVERHEAD_RQ == 1.05
        ver_rates = self._make_rates([25], 1.0)
        fps_rates = self._make_rates([10], 1.0)

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        for rec in result.recommendations:
            if rec.available:
                assert rec.overhead >= MIN_OVERHEAD_RQ

    def test_graduated_channel_selects_different_versions(self):
        """Graduated detect rates -> tiers pick different versions."""
        ver_rates = {15: 1.0, 20: 0.95, 25: 0.88, 30: 0.75, 35: 0.60}
        fps_rates = {8: 0.99, 10: 0.95, 15: 0.85, 20: 0.70}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
        )
        safe = next(r for r in result.recommendations if r.tier == "safe")
        balanced = next(r for r in result.recommendations if r.tier == "balanced")
        aggressive = next(r for r in result.recommendations if r.tier == "aggressive")

        assert safe.available
        assert balanced.available
        assert aggressive.available
        assert safe.qr_version <= balanced.qr_version <= aggressive.qr_version

    def test_video_fps_caps_recommended_fps(self):
        """Captured video FPS limits recommendable calibration FPS."""
        ver_rates = {40: 1.0}
        fps_rates = {30: 0.95, 45: 0.95, 60: 0.95}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
            video_metadata=VideoMetadata(width=1920, height=1080, fps=29.97),
        )
        safe = next(r for r in result.recommendations if r.tier == "safe")
        assert safe.available
        assert safe.fps == 30
        assert any("ignoring calibration FPS above 30fps" in m
                   for m in result.messages)

    def test_cadence_gain_message_and_balanced_overhead(self):
        ver_rates = {40: 1.0}
        fps_rates = {25: 0.65, 30: 0.83}

        result = compute_recommendations(
            ver_rates, fps_rates,
            fps_data_reliable=True,
            preset_name="standard",
            video_metadata=VideoMetadata(width=3840, height=2160, fps=59.96),
        )
        balanced = next(r for r in result.recommendations
                        if r.tier == "balanced")
        assert balanced.available
        assert balanced.fps == 30
        assert balanced.overhead < 2.0
        assert any("30fps outperformed 25fps" in m for m in result.messages)

    def test_format_includes_video_metadata(self):
        result = CalibrationResult(
            preset="standard",
            channel_quality="excellent",
            version_detect_rates={40: 1.0},
            fps_detect_rates={30: 0.95},
            fps_data_reliable=True,
            recommendations=[
                TierRecommendation("safe", True, 40, 30, 1.30, 4000.0),
            ],
            video_metadata=VideoMetadata(width=1920, height=1080, fps=29.97),
        )
        text = format_results(result)
        assert "Video" in text
        assert "1920x1080" in text
        assert "29.97fps" in text

    def test_format_includes_estimated_success(self):
        result = CalibrationResult(
            preset="standard",
            channel_quality="excellent",
            version_detect_rates={40: 1.0},
            fps_detect_rates={30: 0.95},
            fps_data_reliable=True,
            recommendations=[
                TierRecommendation(
                    "safe", True, 40, 30, 1.30, 4000.0,
                    estimated_success=0.991,
                ),
            ],
        )
        text = format_results(result)
        assert "Success" in text
        assert "99.1%" in text

    def test_compute_recommendations_records_target_k(self):
        result = compute_recommendations(
            {40: 1.0}, {30: 1.0}, True, "standard", target_k=2500)

        assert result.target_k == 2500
        assert any("K≈2500" in m for m in result.messages)

    def test_estimate_target_k_defaults_to_long_file_scale(self):
        assert estimate_target_k(None) == 1000
        assert estimate_target_k(100_000_000) > 1000


# ── Throughput estimate ─────────────────────────────────────────────

class TestThroughput:
    """Tests for throughput estimation."""

    def test_throughput_formula(self):
        """Throughput = capacity * fps / overhead."""
        cap = _alphanumeric_byte_capacity(25, _CALIBRATION_EC_LEVEL)
        assert cap > 0
        tp = _estimate_throughput(25, 10, 1.2)
        expected = cap * 10 / 1.2
        assert abs(tp - expected) < 0.01

    def test_throughput_zero_overhead(self):
        assert _estimate_throughput(25, 10, 0) == 0.0

    def test_throughput_invalid_version(self):
        assert _estimate_throughput(99, 10, 1.0) == 0.0


# ── Container FPS ───────────────────────────────────────────────────

class TestContainerFps:

    def test_minimum_60(self):
        assert _container_fps([8, 10, 15]) == 60

    def test_doubles_max_target(self):
        assert _container_fps([15, 20, 25, 30]) == 60  # max(30*2, 60) = 60
        assert _container_fps([30, 60]) == 120  # max(60*2, 60) = 120

    def test_empty_list(self):
        assert _container_fps([]) == 60


# ── Format results ──────────────────────────────────────────────────

class TestFormatResults:

    def test_format_includes_channel_quality(self):
        result = CalibrationResult(
            preset="standard",
            channel_quality="good",
            version_detect_rates={25: 0.90},
            fps_detect_rates={10: 0.95},
            fps_data_reliable=True,
            recommendations=[
                TierRecommendation("safe", True, 25, 10, 1.30, 4000.0),
                TierRecommendation("balanced", True, 25, 10, 1.15, 4500.0),
                TierRecommendation("aggressive", True, 25, 10, 1.05, 5000.0),
            ],
        )
        text = format_results(result)
        assert "Good" in text
        assert "standard" in text
        assert "V25" in text

    def test_format_unavailable_tier(self):
        result = CalibrationResult(
            preset="standard",
            channel_quality="poor",
            version_detect_rates={15: 0.60},
            fps_detect_rates={8: 0.50},
            fps_data_reliable=True,
            recommendations=[
                TierRecommendation("safe", False),
                TierRecommendation("balanced", False),
                TierRecommendation("aggressive", False),
            ],
        )
        text = format_results(result)
        assert "No recommendations available" in text


# ── E2E tests ───────────────────────────────────────────────────────

@pytest.mark.e2e
class TestCalibrateE2E:
    """End-to-end tests: generate calibration video -> analyze."""

    def test_roundtrip_fast(self, tmp_path):
        """Generate fast preset MP4 -> analyze -> 100% detect rates."""
        from qrstream.calibrate import generate_calibration, analyze_calibration

        out = str(tmp_path / "cal.mp4")
        config = generate_calibration(
            preset_name="fast",
            output_path=out,
            display_hz=60,
        )

        result = analyze_calibration(video_path=out)

        # In a clean encode->decode (no channel loss), all rates should
        # be 100% (or very close due to video codec compression).
        for ver, rate in result.version_detect_rates.items():
            assert rate >= 0.95, (
                f"V{ver} detect rate {rate:.2%} below 95% in lossless test"
            )
        for fps, rate in result.fps_detect_rates.items():
            assert rate >= 0.95, (
                f"{fps}fps detect rate {rate:.2%} below 95% in lossless test"
            )
        assert result.fps_data_reliable
        assert all(r.available for r in result.recommendations)

    def test_roundtrip_standard(self, tmp_path):
        """Standard preset generates correct frame count."""
        from qrstream.calibrate import generate_calibration, analyze_calibration

        out = str(tmp_path / "cal_std.mp4")
        config = generate_calibration(
            preset_name="standard",
            output_path=out,
            display_hz=60,
        )

        result = analyze_calibration(video_path=out)
        # Verify all expected versions appear in results
        for v in config.version_ladder:
            assert v in result.version_detect_rates, (
                f"V{v} missing from analysis results"
            )
        for f in config.fps_ladder:
            assert f in result.fps_detect_rates, (
                f"{f}fps missing from analysis results"
            )

    @pytest.mark.parametrize("preset", ["fast", "standard", "full"])
    def test_preset_generates_valid_video(self, tmp_path, preset):
        """All presets generate a video that can be opened and analyzed."""
        from qrstream.calibrate import generate_calibration, analyze_calibration

        out = str(tmp_path / f"cal_{preset}.mp4")
        config = generate_calibration(
            preset_name=preset,
            output_path=out,
            display_hz=60,
        )

        # Just verify it doesn't crash and produces results
        result = analyze_calibration(video_path=out)
        assert len(result.version_detect_rates) > 0
        assert result.preset == preset
