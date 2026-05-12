"""Tests for display-only encode orchestration."""

from qrstream.encoder import encode_to_display


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
