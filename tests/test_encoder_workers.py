import numpy as np
import pytest

from qrstream.encoder import encode_to_video


class _CaptureReporter:
    def __init__(self):
        self.warnings = []
        self.debugs = []
        self.done = []

    def info(self, message):
        pass

    def warn(self, message):
        self.warnings.append(message)

    def error(self, message):
        pass

    def debug(self, message):
        self.debugs.append(message)

    def encode_start(self, **kwargs):
        pass

    def encode_update(self, **kwargs):
        pass

    def encode_done(self, **kwargs):
        self.done.append(kwargs)

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


def _patch_fast_encode(monkeypatch):
    import qrstream.encoder as enc

    monkeypatch.setattr(
        enc,
        "generate_qr_image",
        lambda *args, **kwargs: np.full((16, 16, 3), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        enc.av,
        "open",
        lambda output_path, mode: _FakeOutput(output_path),
    )
    monkeypatch.setattr(
        enc.av,
        "VideoFrame",
        _FakeVideoFrame,
    )


def test_encode_defaults_to_one_worker_without_warning(monkeypatch, tmp_path):
    _patch_fast_encode(monkeypatch)
    reporter = _CaptureReporter()
    src = tmp_path / "input.bin"
    out = tmp_path / "out.mp4"
    src.write_bytes(b"hello encoder workers")

    encode_to_video(
        str(src),
        str(out),
        workers=None,
        verbose=True,
        reporter=reporter,
    )

    assert reporter.warnings == []
    assert any("workers: 1" in msg for msg in reporter.debugs)
    assert out.exists() and out.stat().st_size > 0


def test_encode_warns_when_manual_workers_exceeds_one(monkeypatch, tmp_path):
    _patch_fast_encode(monkeypatch)
    reporter = _CaptureReporter()
    src = tmp_path / "input.bin"
    out = tmp_path / "out.mp4"
    src.write_bytes(b"hello encoder workers")

    encode_to_video(
        str(src),
        str(out),
        workers=2,
        verbose=False,
        reporter=reporter,
    )

    assert len(reporter.warnings) == 1
    assert "--workers > 1" in reporter.warnings[0]
    assert "may not improve performance" in reporter.warnings[0]
    assert out.exists() and out.stat().st_size > 0


def test_encode_reports_rewritten_output_path(monkeypatch, tmp_path):
    _patch_fast_encode(monkeypatch)
    reporter = _CaptureReporter()
    src = tmp_path / "input.bin"
    requested = tmp_path / "out.mov"
    expected = tmp_path / "out.mp4"
    src.write_bytes(b"hello encoder workers")

    encode_to_video(
        str(src),
        str(requested),
        codec="h264",
        reporter=reporter,
    )

    assert expected.exists() and expected.stat().st_size > 0
    assert reporter.done
    assert reporter.done[-1]["output_path"] == str(expected)


def test_encode_surfaces_writer_thread_failures(monkeypatch, tmp_path):
    import qrstream.encoder as enc

    _patch_fast_encode(monkeypatch)
    src = tmp_path / "input.bin"
    out = tmp_path / "out.mp4"
    src.write_bytes(b"hello encoder workers")

    def _boom(packet):
        raise RuntimeError("mux failed")

    fake_output = _FakeOutput(str(out))
    monkeypatch.setattr(enc.av, "open", lambda output_path, mode: fake_output)
    fake_output.mux = _boom

    with pytest.raises(RuntimeError, match="video writer thread failed"):
        encode_to_video(str(src), str(out), reporter=_CaptureReporter())
