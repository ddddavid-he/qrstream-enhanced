# QRStream

[中文](README-zh.md) · [Web receiver](https://ddddavid-he.github.io/qrstream-enhanced/) · [User guide](docs/USAGE.md)

Transfer files through QR code streams: display them on a computer, then recover the file with a phone camera or from a recording. RaptorQ is the default codec; legacy LT streams are also supported by the CLI.

## Quick start

Requires Python 3.10 or newer. Install with pip (or `uv tool install qrstream`):

```bash
pip install qrstream
```

Display a file as QR codes on your computer:

```bash
qrstream encode report.pdf
```

On your phone, open the **[Web receiver](https://ddddavid-he.github.io/qrstream-enhanced/)**, allow camera access, point at the screen and press the red shutter. Save the file when recovery finishes.

The browser supports **V4 / RaptorQ** streams. It processes camera frames and file contents on your device, without uploading them. Downloads are named `qrstream-output.bin`; rename the file to its original extension.

## Camera controls

- **Red shutter:** start or pause reception. Pause retains collected blocks.
- **Stop:** clear collected blocks and progress while keeping the preview running. Confirm first if data has been collected.
- **Resolution:** tap the upper-left label to cycle through supported modes; file progress is retained.
- **Details / save:** open receiver information or save the recovered file. The ring reaches 100% only after recovery.

## Video files

Create a QR video, or recover a file from a recorded video:

```bash
qrstream encode report.pdf -o report.mp4
qrstream decode recording.mp4 -o recovered.pdf
```

Use `qrs` as a short alias. Run `qrstream encode --help` or `qrstream decode --help` for options.

## Documentation and development

- [User guide](docs/USAGE.md): full options, calibration, Python API, Web development and troubleshooting context.
- [Architecture](docs/ARCH.md): modules, protocols and test tooling.
- [Contributing](docs/CONTRIBUTING.md): development, branches, CI and releases.

To work from this repository:

```bash
git clone https://github.com/ddddavid-he/qrstream-enhanced.git
cd qrstream-enhanced
uv sync --dev
```

Web changes merged into `main` are built and deployed by the [Pages workflow](.github/workflows/pages.yml).

## License

MIT
