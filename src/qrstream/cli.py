"""
Unified CLI for QRStream.

Usage:
    qrstream -V | --version
    qrstream encode <file> -o output.mp4 [--overhead 2.0] [--fps 10] [-v]
    qrstream decode <video> -o output_file [-s sample_rate] [-v]
"""

import sys
import os
import argparse

from .__init__ import __version__


def cmd_encode(args):
    """Handle the 'encode' subcommand."""
    from .encoder import encode_to_video

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.file))[0]
        ext = '.avi' if args.codec == 'mjpeg' else '.mp4'
        output = f"{base}{ext}"

    alphanumeric_qr = (args.qr_mode == 'alphanumeric')

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
        verbose=args.verbose,
        workers=args.workers,
        use_legacy_qr=args.legacy_qr,
        codec=args.codec,
        alphanumeric_qr=alphanumeric_qr,
        protocol_version=2 if args.protocol == 'v2' else 3,
        force_compress=args.force_compress,
    )


def cmd_decode(args):
    """Handle the 'decode' subcommand."""
    from .decoder import extract_qr_from_video, decode_blocks_to_file

    if not os.path.exists(args.video):
        print(f"Error: File not found: {args.video}")
        sys.exit(1)

    print(f"Processing: {args.video}")
    print("Extracting QR codes...")

    blocks = extract_qr_from_video(
        args.video, args.sample_rate, args.verbose, args.workers)

    if not blocks:
        print("No QR codes detected. Check that the video clearly shows QR codes.")
        sys.exit(1)

    print(f"Found {len(blocks)} unique blocks. Decoding...")

    output_path = args.output or "decoded_output"
    written = decode_blocks_to_file(blocks, output_path, args.verbose)

    if written is None:
        sys.exit(1)

    print(f"\nSuccess! Saved to: {output_path} ({written} bytes)")


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
    enc.add_argument('-o', '--output', default=None,
                     help='Output video path (default: <filename>.mp4)')
    enc.add_argument('--overhead', type=float, default=2.0,
                     help='Ratio of encoded blocks to source blocks (default: 2.0)')
    enc.add_argument('--fps', type=int, default=10,
                     help='Frames per second in output video (default: 10)')
    enc.add_argument('--ec-level', type=int, default=1, choices=[0, 1, 2, 3],
                     help='QR error correction: 0=L, 1=M (default), 2=Q, 3=H')
    enc.add_argument('--qr-version', type=int, default=20,
                     choices=range(1, 41), metavar='N',
                     help='QR code version 1-40, controls density (default: 20)')
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
    enc.add_argument('--codec', choices=['mp4v', 'mjpeg'], default='mp4v',
                     help='Video codec: mp4v (default) or mjpeg (faster, larger)')
    enc.add_argument('--protocol', choices=['v2', 'v3'], default='v3',
                     help='Protocol version for encoding (default: v3)')
    enc.add_argument('-w', '--workers', type=int, default=None,
                     help='Parallel workers for QR generation (default: CPU count)')
    enc.add_argument('-v', '--verbose', action='store_true',
                     help='Print extra detail (block stats, compression ratio, etc.)')

    # ── decode ────────────────────────────────────────────────────
    dec = subparsers.add_parser(
        'decode', help='Decode a QR code video back to the original file')
    dec.add_argument('video', help='Path to the video file (MOV, MP4, etc.)')
    dec.add_argument('-o', '--output', default=None,
                     help='Output file path (default: decoded_output)')
    dec.add_argument('-s', '--sample-rate', type=int, default=0,
                     help='Process every Nth frame (default: 0=auto-detect)')
    dec.add_argument('-w', '--workers', type=int, default=None,
                     help='Parallel workers (default: all CPU cores)')
    dec.add_argument('-v', '--verbose', action='store_true',
                     help='Print detailed progress')

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
