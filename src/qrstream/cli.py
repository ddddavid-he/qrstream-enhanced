"""
Unified CLI for QRStream.

Usage:
    qrstream -V | --version
    qrstream encode <file> -o output.mp4 [--overhead 2.0] [--fps 10]
                          [--output-mode MODE]
    qrstream decode <video> -o output_file [-s sample_rate]
                          [--output-mode MODE]

``--output-mode`` selects how progress/status is rendered:

* ``auto``        — Rich interactive on TTY, ``log`` otherwise (default).
* ``interactive`` — Force Rich animated UI (thin bars + file block map).
* ``log``         — Append-only ``key=value`` lines; CI / ``tee`` safe.
* ``quiet``       — Only errors and the final success line.
* ``verbose``     — Verbose diagnostic output (Rich on TTY, log-verbose
                    otherwise).

The legacy ``-v / --verbose`` flag is accepted as a hidden alias for
``--output-mode verbose`` to keep existing scripts working.
"""

import sys
import os
import argparse

from .__init__ import __version__
from .ui import OutputMode, resolve_output_mode


# Minimum overhead the default LT codec (SplitMix64 PRNG mixer,
# qrstream ≥ 0.8) needs to converge on sequential seeds across all
# K we've benchmarked (328..4096).  The empirical worst case is
# K=328 at 1.19×; we round up to 1.20× as the hard floor and
# recommend ≥1.50× for real captures where frame loss / detector
# misses eat into the margin.
#
# Anything below the floor indicates either a misunderstanding of
# the codec (LT can't converge below its PRNG-dependent threshold,
# period) or a test/benchmark use case — those can bypass via the
# LTEncoder API directly.
_MIN_OVERHEAD = 1.20
_RECOMMENDED_OVERHEAD = 1.50


def _resolve_mode(args) -> OutputMode:
    """Reconcile legacy ``-v`` with the new ``--output-mode`` flag.

    ``-v`` is kept as a hidden alias that upgrades ``auto`` to
    ``verbose``; users who explicitly pass ``--output-mode`` get
    their choice honoured verbatim.
    """
    raw = getattr(args, 'output_mode', None) or 'auto'
    mode = OutputMode(raw)
    if getattr(args, 'verbose', False) and mode is OutputMode.AUTO:
        return OutputMode.VERBOSE
    return mode


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


def cmd_encode(args):
    """Handle the 'encode' subcommand."""
    from .encoder import encode_to_video

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    # Fail fast on unreachable output paths so the user doesn't
    # wait through a minutes-long encode before learning the
    # destination directory doesn't exist or isn't writable.
    err = _check_output_path_writable(args.output)
    if err is not None:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    if args.overhead < _MIN_OVERHEAD:
        print(
            f"Error: --overhead {args.overhead} is below the LT codec's "
            f"convergence floor ({_MIN_OVERHEAD}×). Decoding would fail "
            f"even on a perfect capture. Use --overhead {_RECOMMENDED_OVERHEAD} "
            f"or higher for reliable real-world recording."
        )
        sys.exit(2)
    if args.overhead < _RECOMMENDED_OVERHEAD:
        print(
            f"Warning: --overhead {args.overhead} is near the LT convergence "
            f"floor. Recommended: ≥{_RECOMMENDED_OVERHEAD} so camera frame "
            f"loss and QR detector misses don't push decoding below the "
            f"threshold."
        )

    output = args.output

    if os.path.abspath(args.file) == os.path.abspath(output):
        print(
            f"Error: output path is the same as the input file '{args.file}'.\n"
            f"Specify a different path with -o."
        )
        sys.exit(1)

    alphanumeric_qr = (args.qr_mode == 'alphanumeric')

    mode = _resolve_mode(args)
    try:
        reporter = resolve_output_mode(
            mode,
            explicit=(getattr(args, 'output_mode', 'auto') != 'auto'),
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    verbose = mode is OutputMode.VERBOSE

    try:
        encode_to_video(
            input_path=args.file,
            output_path=output,
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
            codec=args.codec,
            alphanumeric_qr=alphanumeric_qr,
            force_compress=args.force_compress,
            auto_mask=args.auto_mask,
            reporter=reporter,
        )
    finally:
        try:
            reporter.close()
        except Exception:
            pass


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

    mode = _resolve_mode(args)
    try:
        reporter = resolve_output_mode(
            mode,
            explicit=(getattr(args, 'output_mode', 'auto') != 'auto'),
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    verbose = mode is OutputMode.VERBOSE

    try:
        blocks = extract_qr_from_video(
            args.video, args.sample_rate, verbose, args.workers,
            detect_isolation=args.detect_isolation,
            reporter=reporter,
        )

        if not blocks:
            print("No QR codes detected. Check that the video clearly shows QR codes.")
            sys.exit(1)

        output_path = args.output
        written = decode_blocks_to_file(
            blocks, output_path, verbose, reporter=reporter,
        )

        if written is None:
            sys.exit(1)
    finally:
        try:
            reporter.close()
        except Exception:
            pass


def _add_output_mode_group(sub: argparse.ArgumentParser) -> None:
    """Attach the shared ``--output-mode`` option plus hidden ``-v``."""
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
    # Legacy alias kept hidden: ``-v`` → upgrade ``auto`` to ``verbose``.
    sub.add_argument('-v', '--verbose', action='store_true',
                     help=argparse.SUPPRESS)


def build_parser(prog: str = 'qrstream') -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='QRStream: Encode and decode files via QR code video streams')
    parser.add_argument('-V', '--version', action='version',
                        version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ── encode ────────────────────────────────────────────────────
    enc = subparsers.add_parser(
        'encode', help='Encode a file into a QR code video')
    enc.add_argument('file', help='Path to the input file')
    enc.add_argument('-o', '--output', required=True,
                     help='Output video path (e.g. output.mp4)')
    enc.add_argument('--overhead', type=float, default=2.0,
                     help=f'Ratio of encoded blocks to source blocks '
                          f'(default: 2.0, minimum: {_MIN_OVERHEAD}, '
                          f'recommended: ≥{_RECOMMENDED_OVERHEAD})')
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
    dec.add_argument(
        '--detect-isolation', choices=['on', 'off'], default='on',
        help='[Deprecated] Previously isolated the WeChatQRCode detector '
             'in subprocess helpers to survive native crashes '
             '(opencv_contrib#3570). The backend is now zxing-cpp, which '
             'does not crash. This flag is accepted for backward '
             'compatibility but is ignored. Will be removed in a future '
             'release.')
    _add_output_mode_group(dec)

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'encode':
        cmd_encode(args)
    elif args.command == 'decode':
        cmd_decode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
