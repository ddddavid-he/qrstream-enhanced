#!/usr/bin/env python3
"""Generate deterministic QRStream videos for iOS replay fixture recording."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PAYLOAD_DOMAIN = b"qrstream-ios-replay-fixture-v1\x00"


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    size_bytes: int
    qr_version: int
    fps: int
    overhead: float
    purpose: str

    @property
    def stem(self) -> str:
        return f"ios-replay-{self.case_id}"


CASES = (
    Case(
        case_id="baseline-v20-15fps",
        size_bytes=128 * 1024,
        qr_version=20,
        fps=15,
        overhead=1.30,
        purpose="Low-pressure correctness and latency baseline.",
    ),
    Case(
        case_id="balanced-v30-25fps",
        size_bytes=512 * 1024,
        qr_version=30,
        fps=25,
        overhead=1.35,
        purpose="Representative high-throughput operating point.",
    ),
    Case(
        case_id="dense-v40-30fps",
        size_bytes=1024 * 1024,
        qr_version=40,
        fps=30,
        overhead=1.50,
        purpose="Dense-symbol detection latency and hit-rate stress.",
    ),
    Case(
        case_id="throughput-v40-45fps",
        size_bytes=1024 * 1024,
        qr_version=40,
        fps=45,
        overhead=2.00,
        purpose="Backpressure and late-frame-drop stress.",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_payload(case: Case) -> bytes:
    header = (
        "QRSTREAM-IOS-REPLAY-FIXTURE\n"
        f"schema={SCHEMA_VERSION}\n"
        f"case={case.case_id}\n"
        f"size={case.size_bytes}\n"
        "generator=sha256-counter-v1\n"
        "\n"
    ).encode("ascii")
    if len(header) > case.size_bytes:
        raise ValueError(f"header exceeds payload size for {case.case_id}")

    payload = bytearray(header)
    counter = 0
    case_domain = PAYLOAD_DOMAIN + case.case_id.encode("ascii") + b"\x00"
    while len(payload) < case.size_bytes:
        payload.extend(
            hashlib.sha256(case_domain + counter.to_bytes(8, "big")).digest()
        )
        counter += 1
    return bytes(payload[: case.size_bytes])


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "format=duration,size:"
            "stream=codec_name,width,height,pix_fmt,"
            "avg_frame_rate,r_frame_rate,nb_frames"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def encode_case(case: Case, source: Path, video: Path) -> list[str]:
    args = [
        "encode",
        str(source),
        "-o",
        str(video),
        "--fountain-codec",
        "raptorq",
        "--qr-mode",
        "alphanumeric",
        "--codec",
        "h264",
        "--no-compress",
        "--qr-version",
        str(case.qr_version),
        "--fps",
        str(case.fps),
        "--overhead",
        str(case.overhead),
        "--border",
        "10",
        "--lead-in-seconds",
        "2.0",
        "--anonymous",
        "--output-mode",
        "log",
    ]
    run([sys.executable, "-m", "qrstream", *args])
    return args


def verify_case(source: Path, video: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qrstream-ios-replay-verify-") as temp:
        decoded = Path(temp) / "decoded.bin"
        run(
            [
                sys.executable,
                "-m",
                "qrstream",
                "decode",
                str(video),
                "-o",
                str(decoded),
                "--output-mode",
                "log",
            ]
        )
        source_sha = sha256_file(source)
        decoded_sha = sha256_file(decoded)
        return {
            "complete": True,
            "decoded_size_bytes": decoded.stat().st_size,
            "decoded_sha256": decoded_sha,
            "matches_source": (
                decoded.stat().st_size == source.stat().st_size
                and decoded_sha == source_sha
            ),
        }


def recording_readme() -> str:
    rows = "\n".join(
        (
            f"| `{case.case_id}` | {case.size_bytes} | {case.qr_version} | "
            f"{case.fps} | {case.overhead:.2f} |"
        )
        for case in CASES
    )
    return f"""# QRStream iOS replay fixture recording kit

This directory is generated. `manifest.json` contains the source and video
SHA-256 values plus the exact encode arguments.

| Case | Payload bytes | QR version | FPS | RaptorQ overhead |
|---|---:|---:|---:|---:|
{rows}

## Recording procedure

Record each video into a separate original camera file:

1. Disable notifications and auto-lock on the playback device, set a fixed
   brightness, and play the video full-screen.
2. Use a fixed camera mode for the complete set; 4K60 is preferred for the
   performance baseline.
3. Keep the full QR quiet zone visible. Aim for the QR content to occupy
   60%-80% of the recording's short edge.
4. Start recording at least two seconds before playback. Stop at least two
   seconds after playback finishes.
5. Do not pause, seek, or concatenate multiple cases into one recording.
6. Use a stable mount for the first take. Optional handheld takes must be
   separate files.
7. Keep the original MOV/MP4 files. Do not send them through software that
   recompresses video.

Suggested filename:

    <case>__<device>__4k60__fixed__take01.mov

Return the recordings together with the device model, OS version, camera mode,
and whether the take was fixed or handheld.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to create. Existing paths are rejected.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Offline-decode every generated video and compare payload SHA-256.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required")

    sources_dir = output_dir / "sources"
    videos_dir = output_dir / "videos"
    sources_dir.mkdir(parents=True)
    videos_dir.mkdir()

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kit_id": "qrstream-ios-replay-recording-kit-v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "generator": "scripts/make_ios_replay_recording_kit.py",
        "cases": [],
    }

    for case in CASES:
        print(f"\n==> Generating {case.case_id}", flush=True)
        source = sources_dir / f"{case.stem}.bin"
        video = videos_dir / f"{case.stem}.mp4"
        source.write_bytes(deterministic_payload(case))
        encode_args = encode_case(case, source, video)
        verification = verify_case(source, video) if args.verify else None
        if verification is not None and not verification["matches_source"]:
            raise RuntimeError(f"verification failed for {case.case_id}")

        manifest["cases"].append(
            {
                "case_id": case.case_id,
                "purpose": case.purpose,
                "source": {
                    "path": str(source.relative_to(output_dir)),
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                    "generator": "sha256-counter-v1",
                },
                "encode": {
                    "args": encode_args,
                    "fountain_codec": "raptorq",
                    "qr_mode": "alphanumeric",
                    "codec": "h264",
                    "qr_version": case.qr_version,
                    "fps": case.fps,
                    "overhead": case.overhead,
                    "border_percent": 10,
                    "lead_in_seconds": 2.0,
                    "compression": "disabled",
                },
                "video": {
                    "path": str(video.relative_to(output_dir)),
                    "size_bytes": video.stat().st_size,
                    "sha256": sha256_file(video),
                    "ffprobe": ffprobe(video),
                },
                "offline_verification": verification,
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README-recording.md").write_text(
        recording_readme(),
        encoding="utf-8",
    )

    print(f"\nRecording kit ready: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
