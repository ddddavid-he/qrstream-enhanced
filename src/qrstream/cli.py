"""
Unified CLI for QRStream: encode and decode subcommands.

Usage:
    qrstream encode <file> -o output.mp4 [--overhead 2.0] [--fps 10] [--ec-level 1] [--qr-version 20] [-w 8] [-v]
    qrstream decode <video> -o output_file [-s sample_rate] [-w workers] [-v]
"""

import sys
import os
import argparse

from .encoder import encode_to_video
from .decoder import extract_qr_from_video, decode_blocks


def cmd_encode(args):
    """Handle the 'encode' subcommand."""
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.file))[0]
        ext = '.avi' if args.codec == 'mjpeg' else '.mp4'
        output = f"{base}{ext}"

    encode_to_video(
        input_path=args.file,
        output_path=output,
        overhead=args.overhead,
        fps=args.fps,
        ec_level=args.ec_level,
        qr_version=args.qr_version,
        compress=not args.no_compress,
        verbose=args.verbose,
        workers=args.workers,
        use_legacy_qr=args.legacy_qr,
        codec=args.codec,
        binary_qr=args.binary_qr,
    )


def cmd_decode(args):
    """Handle the 'decode' subcommand."""
    if not os.path.exists(args.video):
        print(f"Error: File not found: {args.video}")
        sys.exit(1)

    print(f"Processing: {args.video}")
    print("Extracting QR codes...")

    used_sample_rate = args.sample_rate
    blocks = extract_qr_from_video(
        args.video, used_sample_rate, args.verbose, args.workers)

    if not blocks:
        print("No QR codes detected. Check that the video clearly shows QR codes.")
        sys.exit(1)

    print(f"Found {len(blocks)} unique blocks. Decoding...")

    result = decode_blocks(blocks, args.verbose)

    if result is None:
        sys.exit(1)

    output_path = args.output or "decoded_output"
    with open(output_path, 'wb') as f:
        f.write(result)

    print(f"\nSuccess! Saved to: {output_path} ({len(result)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        prog='qrstream',
        description='QRStream: Encode and decode files via QR code video streams')

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
    enc.add_argument('--no-compress', action='store_true',
                     help='Disable zlib compression')
    enc.add_argument('--legacy-qr', action='store_true',
                     help='Use qrcode library instead of OpenCV for QR generation (slower, more control)')
    enc.add_argument('--codec', choices=['mp4v', 'mjpeg'], default='mp4v',
                     help='Video codec: mp4v (default) or mjpeg (faster encoding, larger files)')
    enc.add_argument('--binary-qr', action='store_true',
                     help='Embed raw bytes in QR (skip base64, 33%% more capacity per frame, experimental)')
    enc.add_argument('-w', '--workers', type=int, default=None,
                     help='Number of parallel workers for QR generation (default: CPU count, max 8)')
    enc.add_argument('-v', '--verbose', action='store_true',
                     help='Print detailed progress')

    # ── decode ────────────────────────────────────────────────────
    dec = subparsers.add_parser(
        'decode', help='Decode a QR code video back to the original file')
    dec.add_argument('video', help='Path to the video file (MOV, MP4, etc.)')
    dec.add_argument('-o', '--output', default=None,
                     help='Output file path (default: decoded_output)')
    dec.add_argument('-s', '--sample-rate', type=int, default=0,
                     help='Process every Nth frame (default: 0=auto-detect)')
    dec.add_argument('-w', '--workers', type=int, default=None,
                     help='Number of parallel workers (default: CPU count, max 8)')
    dec.add_argument('-v', '--verbose', action='store_true',
                     help='Print detailed progress')

    args = parser.parse_args()

    if args.command == 'encode':
        cmd_encode(args)
    elif args.command == 'decode':
        cmd_decode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
