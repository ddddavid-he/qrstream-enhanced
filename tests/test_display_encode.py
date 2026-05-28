"""Tests for display-only encode orchestration."""

from qrstream.display_cache import ModuleCachePlan
from qrstream.encoder import encode_to_display


class _CaptureReporter:
    def __init__(self):
        self.starts = []
        self.debugs = []

    def info(self, message):
        pass

    def warn(self, message):
        pass

    def error(self, message):
        pass

    def debug(self, message):
        self.debugs.append(message)

    def encode_start(self, **kwargs):
        self.starts.append(kwargs)

    def encode_update(self, **kwargs):
        pass

    def encode_done(self, **kwargs):
        pass

    def close(self):
        pass


class _FakePacket:
    pass


class _FakeStream:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.pix_fmt = None
        self.options = None

    def encode(self, frame=None):
        return [_FakePacket()] if frame is not None else []


class _FakeOutput:
    def __init__(self, output_path):
        self.output_path = output_path
        self.stream = _FakeStream()

    def add_stream(self, codec, rate):
        return self.stream

    def mux(self, packet):
        with open(self.output_path, "ab") as f:
            f.write(b"packet\n")

    def close(self):
        return None


class _FakeVideoFrame:
    @staticmethod
    def from_ndarray(frame, format):
        return frame


def test_encode_to_display_populates_module_cache_with_fake_player(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"display mode smoke test")

    seen = {}

    def fake_player(cache, state, fps):
        assert fps == 10
        assert state.wait_done(timeout=10)
        seen["cache"] = cache
        seen["state"] = state

    cache = encode_to_display(
        input_path=str(src),
        overhead=2.0,
        fps=10,
        qr_version=10,
        lead_in_seconds=0.0,
        compress=False,
        player=fake_player,
    )

    assert seen["cache"] is cache
    assert seen["state"].is_done()
    assert cache.is_done()
    assert cache.total_frames == 2
    assert cache.valid_count == 2
    assert cache.get_module_image(0) is not None


def test_encode_to_display_reports_input_size(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"display reporter metadata")
    reporter = _CaptureReporter()

    def fake_player(cache, state, fps):
        assert state.wait_done(timeout=10)

    encode_to_display(
        input_path=str(src),
        overhead=2.0,
        fps=10,
        qr_version=10,
        lead_in_seconds=0.0,
        compress=False,
        player=fake_player,
        reporter=reporter,
    )

    assert reporter.starts[-1]["input_path"] == str(src)
    assert reporter.starts[-1]["file_size"] == src.stat().st_size
    assert reporter.starts[-1]["anonymous"] is False


def test_encode_to_display_anonymous_hides_input_path_from_reporter(tmp_path):
    src = tmp_path / "secret.bin"
    src.write_bytes(b"display reporter metadata")
    reporter = _CaptureReporter()

    def fake_player(cache, state, fps):
        assert state.wait_done(timeout=10)

    encode_to_display(
        input_path=str(src),
        overhead=2.0,
        fps=10,
        qr_version=10,
        lead_in_seconds=0.0,
        compress=False,
        player=fake_player,
        reporter=reporter,
        anonymous=True,
    )

    assert reporter.starts[-1]["input_path"] == ""
    assert reporter.starts[-1]["anonymous"] is True


def test_encode_to_display_anonymous_verbose_hides_source_metadata(tmp_path):
    src = tmp_path / "secret.bin"
    src.write_bytes(b"display reporter metadata")
    reporter = _CaptureReporter()

    def fake_player(cache, state, fps):
        assert state.wait_done(timeout=10)

    encode_to_display(
        input_path=str(src),
        overhead=2.0,
        fps=10,
        qr_version=10,
        lead_in_seconds=0.0,
        compress=False,
        verbose=True,
        player=fake_player,
        reporter=reporter,
        anonymous=True,
    )

    debug_output = "\n".join(reporter.debugs)
    assert "secret.bin" not in debug_output
    assert str(src) not in debug_output
    assert "anonymous" in debug_output


def test_encode_to_display_streams_video_to_output(monkeypatch, tmp_path):
    import qrstream.encoder as enc

    src = tmp_path / "src.bin"
    out = tmp_path / "out.mp4"
    src.write_bytes(b"display mode video output")

    monkeypatch.setattr(
        enc.av,
        "open",
        lambda output_path, mode, format=None: _FakeOutput(output_path),
    )
    monkeypatch.setattr(enc.av, "VideoFrame", _FakeVideoFrame)

    def fake_player(cache, state, fps):
        assert state.wait_done(timeout=10)

    encode_to_display(
        input_path=str(src),
        output_path=str(out),
        overhead=2.0,
        fps=10,
        qr_version=10,
        lead_in_seconds=0.0,
        compress=False,
        player=fake_player,
    )

    assert out.exists()
    assert out.read_bytes().startswith(b"packet\n")


def test_display_output_does_not_force_full_cache(monkeypatch, tmp_path):
    import qrstream.encoder as enc

    src = tmp_path / "src.bin"
    out = tmp_path / "out.mp4"
    src.write_bytes(b"display mode output cache policy")

    class FakeSink:
        instances = []

        def __init__(self, *args, **kwargs):
            self.finalized = False
            FakeSink.instances.append(self)

        def offer(self, frame_index, packed_frame):
            return True

        def finalize(self, total_frames, module_frame_at):
            self.finalized = True
            self.total_frames = total_frames

        def discard(self):
            self.discarded = True

    monkeypatch.setattr(enc, "_DisplayVideoSink", FakeSink)
    monkeypatch.setattr(
        enc,
        "plan_module_cache",
        lambda total_frames, module_side, fps: ModuleCachePlan(
            "window", total_frames, 1, 1),
    )

    def fake_player(cache, state, fps):
        assert state.wait_done(timeout=10)

    cache = encode_to_display(
        input_path=str(src),
        output_path=str(out),
        overhead=2.0,
        fps=10,
        qr_version=10,
        lead_in_seconds=0.0,
        compress=False,
        player=fake_player,
    )

    assert cache.mode == "window"
    assert FakeSink.instances[-1].finalized


def test_display_output_finalizes_after_early_close(monkeypatch, tmp_path):
    import qrstream.encoder as enc

    src = tmp_path / "src.bin"
    out = tmp_path / "out.mp4"
    src.write_bytes(b"display closes before encode completes")

    class FakeSink:
        instances = []

        def __init__(self, *args, **kwargs):
            self.generated = []
            FakeSink.instances.append(self)

        def offer(self, frame_index, packed_frame):
            return False

        def finalize(self, total_frames, module_frame_at):
            for frame_index in range(total_frames):
                assert module_frame_at(frame_index) is not None
                self.generated.append(frame_index)

        def discard(self):
            self.discarded = True

    monkeypatch.setattr(enc, "_DisplayVideoSink", FakeSink)

    def fake_player(cache, state, fps):
        state.request_cancel()

    cache = encode_to_display(
        input_path=str(src),
        output_path=str(out),
        overhead=2.0,
        fps=10,
        qr_version=10,
        lead_in_seconds=0.0,
        compress=False,
        player=fake_player,
    )

    sink = FakeSink.instances[-1]
    assert sink.generated == list(range(cache.total_frames))
