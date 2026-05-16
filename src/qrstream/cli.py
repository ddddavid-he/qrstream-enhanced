"""
Unified CLI for QRStream.

Usage:
    qrstream -v | --version
    qrstream encode <file> [-o output.mp4] [--display] [--overhead RATIO]
                          [--fps 10] [--output-mode MODE]
    qrstream decode <video> -o output_file [-s sample_rate]
                          [--output-mode MODE]

``--output-mode`` selects how progress/status is rendered:

* ``auto``        — Rich interactive on TTY, ``log`` otherwise (default).
* ``interactive`` — Force Rich animated UI (thin bars + file block map).
* ``log``         — Append-only ``key=value`` lines; CI / ``tee`` safe.
* ``quiet``       — Only errors and the final success line.
* ``verbose``     — Verbose diagnostic output (Rich on TTY, log-verbose
                    otherwise).

The hidden ``-V / --verbose`` flag is accepted on subcommands as an alias
for ``--output-mode verbose``.
"""

import sys
import os
import re
import argparse

from .__init__ import __version__
from .overhead_policy import (
    DEFAULT_OVERHEAD_LT as _DEFAULT_OVERHEAD_LT,
    DEFAULT_OVERHEAD_RQ as _DEFAULT_OVERHEAD_RQ,
    MIN_OVERHEAD_LT as _MIN_OVERHEAD_LT,
    MIN_OVERHEAD_RQ as _MIN_OVERHEAD_RQ,
    RECOMMENDED_OVERHEAD_LT as _RECOMMENDED_OVERHEAD_LT,
    RECOMMENDED_OVERHEAD_RQ as _RECOMMENDED_OVERHEAD_RQ,
)
from .ui import OutputMode, resolve_output_mode


def cmd_colors():
    """Display all colour palettes used by the qrstream UI."""
    try:
        from rich.console import Console
        from rich.text import Text
    except ImportError:
        print("Error: 'rich' is required for colour display. "
              "Install with: pip install rich")
        sys.exit(1)

    from .ui import (
        _DENSITY_CHAR_AND_STYLE,
        _density_cell,
        _density_cell_truecolor,
        _detect_rate_style,
        _detect_rate_markup,
        _DET_GRADIENT_ANCHORS,
    )

    console = Console(stderr=False, highlight=False, force_terminal=True)
    truecolor = (console.color_system == "truecolor")

    console.print(
        f"[bold]qrstream {__version__}[/bold] — UI colour palette\n"
    )
    console.print(f"  Terminal colour system: [bold]{console.color_system}[/bold]\n")

    # ── Detect-rate gradient ──────────────────────────────────────
    console.print("[bold underline]Detect-rate gradient[/bold underline]")
    console.print(
        "  Shown in the Scan/Recover stats column during decode.\n"
        "  Anchors: <50% red → 60% orange → 70% yellow → 80%+ green\n"
    )

    # Show a continuous band — 50 cells mapping 0%–100%.
    bar_width = 50
    band = Text("  ")
    for i in range(bar_width):
        hit = i / (bar_width - 1)
        style = _detect_rate_style(hit, truecolor=truecolor)
        band.append("█", style=style)
    console.print(band)
    # Scale labels aligned to the bar (2-space indent + positions).
    # Each cell = 2% → 50%=cell25, 60%=cell30, 70%=cell35, 80%=cell40
    scale = Text("  ")
    markers = [(0, "0%"), (25, "50%"), (30, "60%"), (35, "70%"),
               (40, "80%"), (49, "100%")]
    pos = 0
    for col, label in markers:
        if col > pos:
            scale.append(" " * (col - pos))
        scale.append(label)
        pos = col + len(label)
    console.print(scale)
    console.print()

    # Show discrete sample values
    console.print("  Sample values:")
    samples = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    line = Text("  ")
    for pct in samples:
        hit = pct / 100.0
        style = _detect_rate_style(hit, truecolor=truecolor)
        line.append(f" {pct:>3d}%", style=style)
    console.print(line)
    console.print()

    # Fallback (discrete) mode comparison
    if truecolor:
        console.print("  [dim]Discrete fallback (non-truecolor terminals):[/dim]")
        line_fb = Text("  ")
        for pct in samples:
            hit = pct / 100.0
            style = _detect_rate_style(hit, truecolor=False)
            line_fb.append(f" {pct:>3d}%", style=style)
        console.print(line_fb)
        console.print()

    # ── Block-map (File row) ──────────────────────────────────────
    console.print("[bold underline]Block-map density[/bold underline]")
    console.print(
        "  Shown in the File row during decode — each cell represents\n"
        "  a region of the output file coloured by recovery density.\n"
        "  Gamma curve (0.55) stretches early colours for visibility.\n"
    )

    # Show a simulated block-map bar filling from 0% to 100%
    console.print("  Truecolor gradient (0% → 100%):")
    bar = Text("  ")
    for i in range(50):
        density = i / 49.0
        ch, st = _density_cell_truecolor(density)
        bar.append(ch, style=st)
    console.print(bar)
    console.print()

    # Discrete fallback
    console.print("  [dim]Discrete fallback (non-truecolor terminals):[/dim]")
    bar_fb = Text("  ")
    for i in range(50):
        density = i / 49.0
        ch, st = _density_cell(density)
        bar_fb.append(ch, style=st)
    console.print(bar_fb)
    console.print()

    # Tier table
    console.print("  Discrete tiers:")
    console.print("  Density    Glyph   Style")
    prev_thresh = 0.0
    for thresh, char, style_name in _DENSITY_CHAR_AND_STYLE:
        label = f"  {prev_thresh*100:>5.1f}%–{thresh*100:>5.1f}%"
        sample = Text(f"   {char} {char} {char}", style=style_name)
        line = Text(f"{label}  ")
        line.append_text(sample)
        line.append(f"   {style_name}")
        console.print(line)
        prev_thresh = thresh
    console.print()

    # ── Phase label colours ───────────────────────────────────────
    console.print("[bold underline]Phase labels[/bold underline]\n")
    console.print("  [bold cyan]Probe[/bold cyan]   — probing video for QR detection rate")
    console.print("  [bold cyan]Scan[/bold cyan]    — main decode pass")
    console.print("  [bold yellow]Recover[/bold yellow] — targeted recovery of missing blocks")
    console.print("  [bold cyan]File[/bold cyan]    — block-map / output file progress")
    console.print("  [bold cyan]Plan[/bold cyan]    — decode plan parameters")
    console.print()

    # ── Status indicators ─────────────────────────────────────────
    console.print("[bold underline]Status indicators[/bold underline]\n")
    console.print("  [green]✓[/green] Success     [yellow]⚠[/yellow] Warning     [bold red]✗[/bold red] Error")
    console.print()


# Legacy aliases (used by some tests).
_MIN_OVERHEAD = _MIN_OVERHEAD_LT
_RECOMMENDED_OVERHEAD = _RECOMMENDED_OVERHEAD_LT


def _resolve_mode(args) -> OutputMode:
    """Reconcile ``--verbose`` with the new ``--output-mode`` flag.

    ``-V / --verbose`` is a hidden alias that upgrades ``auto`` to
    ``verbose``; users who explicitly pass ``--output-mode`` get
    their choice honoured verbatim.
    """
    raw = getattr(args, 'output_mode', None) or 'auto'
    mode = OutputMode(raw)
    if getattr(args, 'verbose', False) and mode is OutputMode.AUTO:
        return OutputMode.VERBOSE
    return mode


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|bytes?)?\s*$", re.I)
_SIZE_UNITS = {
    None: 1,
    "": 1,
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000 ** 2,
    "mb": 1000 ** 2,
    "g": 1000 ** 3,
    "gb": 1000 ** 3,
    "t": 1000 ** 4,
    "tb": 1000 ** 4,
    "kib": 1024,
    "mib": 1024 ** 2,
    "gib": 1024 ** 3,
    "tib": 1024 ** 4,
}


def _parse_size_bytes(value: str) -> int:
    match = _SIZE_RE.match(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "size must look like 100M, 1.5GiB, or a byte count")
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit not in _SIZE_UNITS:
        raise argparse.ArgumentTypeError(f"unsupported size unit: {unit}")
    size = int(number * _SIZE_UNITS[unit])
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return size


def _check_output_path_writable(output: str) -> str | None:
    """Validate an ``-o / --output`` path *before* the heavy job.

    encode / decode are long-running (often minutes for real
    videos).  If the destination turns out to be unreachable —
    parent directory missing, not writable, existing file
    read-only — the user should hear about it in the first second,
    not after they've waited through a probe + scan.

    Contract:
    * Parent directory must exist.  We deliberately do **not**
      ``mkdir -p`` here: a typo like ``/tmp/typo/out.bin``
      silently creating ``/tmp/typo/`` is worse than failing loud.
    * Parent directory must be writable by the current process.
    * If the output path already exists, it must be a regular
      file (not a directory) and writable — so we can truncate
      and overwrite it.

    Returns ``None`` on success or a user-facing error message on
    failure.  The caller is responsible for printing the message
    and exiting with the appropriate status code.
    """
    if not output:
        return "output path is empty"

    # Resolve the parent directory.  An empty ``dirname`` (e.g.
    # plain ``out.bin``) means "current working directory".
    parent = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(parent):
        return (
            f"output directory does not exist: {parent}\n"
            f"Create it first (e.g. `mkdir -p {parent}`) or pick "
            f"a different path with -o."
        )
    if not os.access(parent, os.W_OK):
        return (
            f"output directory is not writable: {parent}\n"
            f"Check permissions or pick a different path with -o."
        )

    # If the file already exists, make sure we'll actually be
    # able to truncate and overwrite it.
    if os.path.exists(output):
        if os.path.isdir(output):
            return (
                f"output path is an existing directory: {output}\n"
                f"Specify a file path with -o, not a directory."
            )
        if not os.access(output, os.W_OK):
            return (
                f"output file exists but is not writable: {output}\n"
                f"Remove it or adjust its permissions first."
            )
    return None


def _build_reporter(args) -> tuple[OutputMode, object]:
    """Resolve CLI output mode and build the matching reporter."""
    mode = _resolve_mode(args)
    try:
        reporter = resolve_output_mode(
            mode,
            explicit=(getattr(args, 'output_mode', 'auto') != 'auto'),
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    return mode, reporter


def _close_reporter(reporter) -> None:
    try:
        reporter.close()
    except Exception:
        pass


def cmd_encode(args):
    """Handle the 'encode' subcommand."""
    from .encoder import encode_to_video

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    output = args.output
    output_requested = output is not None
    display = bool(getattr(args, 'display', False)) or not output_requested

    if output_requested:
        # Fail fast on unreachable output paths so the user doesn't
        # wait through a minutes-long encode before learning the
        # destination directory doesn't exist or isn't writable.
        err = _check_output_path_writable(output)
        if err is not None:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)

    fountain_codec = getattr(args, 'fountain_codec', 'raptorq')

    if fountain_codec == 'raptorq':
        min_oh = _MIN_OVERHEAD_RQ
        rec_oh = _RECOMMENDED_OVERHEAD_RQ
        default_oh = _DEFAULT_OVERHEAD_RQ
        codec_name = 'RaptorQ'
    else:
        min_oh = _MIN_OVERHEAD_LT
        rec_oh = _RECOMMENDED_OVERHEAD_LT
        default_oh = _DEFAULT_OVERHEAD_LT
        codec_name = 'LT'

    if args.overhead is None:
        args.overhead = default_oh

    if args.overhead < min_oh:
        print(
            f"Error: --overhead {args.overhead} is below the {codec_name} codec's "
            f"convergence floor ({min_oh}x). Decoding would fail "
            f"even on a perfect capture. Use --overhead {rec_oh} "
            f"or higher for reliable real-world recording."
        )
        sys.exit(2)
    if args.overhead < rec_oh:
        print(
            f"Warning: --overhead {args.overhead} is near the {codec_name} convergence "
            f"floor. Recommended: >={rec_oh} so camera frame "
            f"loss and QR detector misses don't push decoding below the "
            f"threshold."
        )

    if output_requested and os.path.abspath(args.file) == os.path.abspath(output):
        print(
            f"Error: output path is the same as the input file '{args.file}'.\n"
            f"Specify a different path with -o."
        )
        sys.exit(1)

    alphanumeric_qr = (args.qr_mode == 'alphanumeric')

    mode, reporter = _build_reporter(args)
    verbose = mode is OutputMode.VERBOSE

    try:
        common_kwargs = dict(
            input_path=args.file,
            overhead=args.overhead,
            fps=args.fps,
            ec_level=args.ec_level,
            qr_version=args.qr_version,
            border=args.border,
            lead_in_seconds=args.lead_in_seconds,
            compress=not args.no_compress,
            verbose=verbose,
            workers=args.workers,
            use_legacy_qr=args.legacy_qr,
            alphanumeric_qr=alphanumeric_qr,
            force_compress=args.force_compress,
            auto_mask=args.auto_mask,
            reporter=reporter,
            fountain_codec=fountain_codec,
        )
        if display:
            from .encoder import encode_to_display
            encode_to_display(
                output_path=output if output_requested else None,
                codec=args.codec,
                report_display_done=not output_requested,
                **common_kwargs,
            )
        else:
            encode_to_video(
                output_path=output,
                codec=args.codec,
                **common_kwargs,
            )
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(3)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        # Remove partial/corrupt output file on interrupt.
        try:
            if output and os.path.exists(output):
                os.unlink(output)
        except OSError:
            pass
        raise
    finally:
        _close_reporter(reporter)


def cmd_decode(args):
    """Handle the 'decode' subcommand."""
    from .decoder import extract_qr_from_video, decode_blocks_to_file

    if not os.path.exists(args.video):
        print(f"Error: File not found: {args.video}")
        sys.exit(1)

    # Fail fast on unreachable output paths so users don't wait
    # through a probe + scan before finding out the destination
    # directory doesn't exist or isn't writable.
    err = _check_output_path_writable(args.output)
    if err is not None:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    mode, reporter = _build_reporter(args)
    verbose = mode is OutputMode.VERBOSE

    output_path = args.output
    try:
        blocks, completed_decoder = extract_qr_from_video(
            args.video, args.sample_rate, verbose, args.workers,
            reporter=reporter,
            return_decoder=True,
        )

        if not blocks:
            print("No QR codes detected. Check that the video clearly shows QR codes.")
            sys.exit(1)

        written = decode_blocks_to_file(
            blocks, output_path, verbose, reporter=reporter,
            decoder=completed_decoder,
        )

        if written is None:
            sys.exit(1)
    except KeyboardInterrupt:
        # Remove partial output file on interrupt.
        try:
            if os.path.exists(output_path):
                os.unlink(output_path)
        except OSError:
            pass
        raise
    finally:
        _close_reporter(reporter)


def cmd_calibrate(args):
    """Handle the 'calibrate' subcommand."""
    from .calibrate import (
        estimate_target_k,
        generate_calibration,
        analyze_calibration,
        format_results,
        render_results,
    )

    mode, reporter = _build_reporter(args)

    try:
        if (getattr(args, 'display', False)
                or (not args.output and not args.input)):
            # Encoder side: display mode (default when no mode is specified)
            generate_calibration(
                preset_name=args.precision,
                display=True,
                codec=args.codec,
                reporter=reporter,
            )
        elif args.output:
            # Encoder side: video output mode
            err = _check_output_path_writable(args.output)
            if err is not None:
                print(f"Error: {err}", file=sys.stderr)
                sys.exit(1)
            generate_calibration(
                preset_name=args.precision,
                output_path=args.output,
                display_hz=args.display_hz,
                codec=args.codec,
                reporter=reporter,
            )
        elif args.input:
            # Decoder side: analyze captured video
            if not os.path.exists(args.input):
                print(f"Error: File not found: {args.input}",
                      file=sys.stderr)
                sys.exit(1)
            target_size = args.target_size
            if args.target_file:
                if not os.path.exists(args.target_file):
                    print(f"Error: File not found: {args.target_file}",
                          file=sys.stderr)
                    sys.exit(1)
                target_size = os.path.getsize(args.target_file)
            result = analyze_calibration(
                video_path=args.input,
                workers=args.workers,
                reporter=reporter,
                target_k=estimate_target_k(target_size),
                fountain_codec=args.fountain_codec,
            )
            console = getattr(reporter, '_console', None)
            if console is not None:
                console.print(render_results(result))
            else:
                print(format_results(result))
        else:
            print("Error: Specify --display, -o, or -i.",
                  file=sys.stderr)
            sys.exit(1)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(3)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        raise
    finally:
        _close_reporter(reporter)


def _add_output_mode_group(sub: argparse.ArgumentParser) -> None:
    """Attach the shared ``--output-mode`` option plus hidden ``-V``."""
    sub.add_argument(
        '--output-mode',
        dest='output_mode',
        choices=[m.value for m in OutputMode],
        default=OutputMode.AUTO.value,
        help='Control progress/status rendering. '
             '"auto" picks Rich interactive on TTY, "log" otherwise. '
             '"log" emits append-only key=value lines for CI. '
             '"quiet" prints only errors and the final path. '
             '"verbose" enables full diagnostic output.',
    )
    # Hidden alias: ``-V`` → upgrade ``auto`` to ``verbose``.
    sub.add_argument('-V', '--verbose', action='store_true',
                     help=argparse.SUPPRESS)


def build_parser(prog: str = 'qrstream') -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='QRStream: Encode and decode files via QR code video streams')
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ── encode ────────────────────────────────────────────────────
    enc = subparsers.add_parser(
        'encode', help='Encode a file into a QR code video')
    enc.add_argument('file', help='Path to the input file')
    enc.add_argument('-o', '--output', required=False,
                     help='Output video path (e.g. output.mp4). If omitted, '
                          'encode displays frames on screen.')
    enc.add_argument('--display', action='store_true',
                     help='Display encoded QR frames in the built-in GUI player. '
                          'When used with -o, the video is saved after display '
                          'rendering completes if needed.')
    enc.add_argument('--overhead', type=float, default=None,
                     help='Ratio of encoded blocks to source blocks '
                          f'(default: {_DEFAULT_OVERHEAD_RQ} for raptorq, '
                          f'{_DEFAULT_OVERHEAD_LT} for lt; minimum: '
                          f'{_MIN_OVERHEAD_RQ} for raptorq, '
                          f'{_MIN_OVERHEAD_LT} for lt)')
    enc.add_argument('--fps', type=int, default=10,
                     help='Frames per second in output video (default: 10)')
    # TODO(v0.10.0): remove ``--ec-level`` entirely.  QR-level error
    # correction is redundant in qrstream's pipeline because LT fountain
    # ``--overhead`` already handles frame-level loss (which is the
    # dominant failure mode on phone captures), and WeChatQRCode either
    # decodes a QR into its payload or fails outright — EC rarely
    # rescues a borderline frame that the detector would otherwise
    # return ``None`` for.  The option is kept hidden in v0.8/0.9 so
    # scripts built around the previous CLI continue to work, but users
    # should stop setting it.  See ``encoder.encode_to_video`` for the
    # corresponding API-level parameter which is retained for the same
    # deprecation window.
    enc.add_argument('--ec-level', type=int, default=1, choices=[0, 1, 2, 3],
                     help=argparse.SUPPRESS)
    enc.add_argument('--qr-version', type=int, default=25,
                     choices=range(1, 41), metavar='N',
                     help='QR code version 1-40, controls density (default: 25)')
    enc.add_argument('--border', type=float, default=None,
                     help='Quiet-zone width as a percentage of QR content width (default: standard 4-module quiet zone; use 0 to disable)')
    enc.add_argument('--lead-in-seconds', type=float, default=0.0,
                     dest='lead_in_seconds',
                     help='White lead-in duration before the first QR frame')
    enc.add_argument('--no-compress', action='store_true',
                     help='Disable zlib compression')
    enc.add_argument('--force-compress', action='store_true',
                     help='Force compression even for large V3 inputs (uses more memory)')
    enc.add_argument('--legacy-qr', action='store_true',
                     help='Accepted for backward compatibility; ignored.')
    enc.add_argument('--qr-mode', choices=['alphanumeric', 'base64'],
                     default='alphanumeric',
                     help='QR payload encoding: alphanumeric (default, base45 '
                          'into QR alphanumeric mode, ~29%% more capacity) '
                          'or base64 (standard, QR byte mode).')
    enc.add_argument('--codec', choices=['h264', 'mp4v', 'mjpeg'], default='h264',
                     help='Video codec: h264 (default), mp4v, or mjpeg (faster, larger)')
    enc.add_argument('--fountain-codec', dest='fountain_codec',
                     choices=['raptorq', 'lt'], default='raptorq',
                     help='Fountain code: raptorq (default, RFC 6330, near-optimal) '
                          'or lt (legacy LT codes)')
    enc.add_argument('-w', '--workers', type=int, default=None,
                     help='Parallel workers for QR generation (default: 1; higher values may not improve performance)')
    enc.add_argument('--auto-mask', action='store_true',
                     help='Accepted for backward compatibility; ignored. '
                          'zxing-cpp always evaluates all 8 ISO 18004 mask '
                          'patterns in native C++ at negligible cost.')
    _add_output_mode_group(enc)

    # ── decode ────────────────────────────────────────────────────
    dec = subparsers.add_parser(
        'decode', help='Decode a QR code video back to the original file')
    dec.add_argument('video', help='Path to the video file (MOV, MP4, etc.)')
    dec.add_argument('-o', '--output', required=True,
                     help='Output file path')
    dec.add_argument('-s', '--sample-rate', type=int, default=0,
                     help='Process every Nth frame (default: 0=auto-detect)')
    dec.add_argument('-w', '--workers', type=int, default=None,
                     help='Parallel workers (default: all CPU cores)')
    _add_output_mode_group(dec)

    # ── colors ───────────────────────────────────────────────────
    subparsers.add_parser(
        'colors',
        help='Display the colour palette used by the UI '
             '(detect-rate gradient, block-map, etc.)')

    # ── calibrate ────────────────────────────────────────────────
    cal = subparsers.add_parser(
        'calibrate',
        help='Auto-calibrate channel parameters for optimal encode settings')
    cal_mode = cal.add_mutually_exclusive_group()
    cal_mode.add_argument(
        '--display', action='store_true',
        help='Play calibration sequence on screen via Qt player (default)')
    cal_mode.add_argument(
        '-o', '--output', metavar='PATH',
        help='Write calibration video to file (encoder side)')
    cal_mode.add_argument(
        '-i', '--input', metavar='PATH',
        help='Analyze a captured calibration video (decoder side)')
    cal.add_argument(
        '--precision',
        metavar='{low,fast,standard,full,high}',
        default='standard',
        help='Calibration preset: low for weak channels; fast (~15s), '
             'standard (~30s), full (~60s), or high (~60s). '
             'Default: standard')
    cal.add_argument(
        '--display-hz', type=int, default=None,
        help='Override display refresh rate in Hz for video output mode '
             '(default: auto-detect in display mode, 60 in video mode)')
    cal.add_argument(
        '--codec', default='h264',
        choices=['h264', 'mp4v', 'mjpeg'],
        help='Video codec for calibration output (default: h264)')
    target_group = cal.add_mutually_exclusive_group()
    target_group.add_argument(
        '--target-size', type=_parse_size_bytes, default=None,
        help='Target payload size for file-specific overhead estimates '
             '(analysis mode; e.g. 100M, 1.5GiB)')
    target_group.add_argument(
        '--target-file', metavar='PATH', default=None,
        help='Target payload file for file-specific overhead estimates '
             '(analysis mode)')
    cal.add_argument(
        '--fountain-codec', dest='fountain_codec',
        choices=['raptorq', 'lt'], default='raptorq',
        help='Fountain code model for overhead estimates (default: raptorq)')
    cal.add_argument(
        '-w', '--workers', type=int, default=None,
        help='Parallel workers for analysis (default: auto)')
    _add_output_mode_group(cal)

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == 'encode':
            cmd_encode(args)
        elif args.command == 'decode':
            cmd_decode(args)
        elif args.command == 'calibrate':
            cmd_calibrate(args)
        elif args.command == 'colors':
            cmd_colors()
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        # Clean single-line message instead of a Python traceback.
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == '__main__':
    main()
