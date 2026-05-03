import numpy as np

from qrstream.encoder import encode_to_video


class _CaptureReporter:
    def __init__(self):
        self.warnings = []
        self.debugs = []

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
        pass

    def close(self):
        pass


class _FakeVideoWriter:
    def __init__(self, output_path, *args):
        self.output_path = output_path
        self._opened = True

    def isOpened(self):
        return self._opened

    def write(self, frame):
        with open(self.output_path, "ab") as f:
            f.write(b"frame\n")

    def release(self):
        self._opened = False


def _patch_fast_encode(monkeypatch):
    import qrstream.encoder as enc

    monkeypatch.setattr(
        enc,
        "generate_qr_image",
        lambda *args, **kwargs: np.full((16, 16, 3), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(enc.cv2, "VideoWriter", _FakeVideoWriter)
    monkeypatch.setattr(enc.cv2, "VideoWriter_fourcc", lambda *args: 0)


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
