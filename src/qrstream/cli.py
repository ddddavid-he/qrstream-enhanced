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


def cmd_encode(args):
    """Handle the 'encode' subcommand."""
    from .encoder import encode_to_video

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
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
        force_compress=args.force_compress,
        auto_mask=args.auto_mask,
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
        args.video, args.sample_rate, args.verbose, args.workers,
        use_mnn=args.use_mnn,
        detect_isolation=args.detect_isolation,
        decode_attempts=args.decode_attempts)

    if not blocks:
        print("No QR codes detected. Check that the video clearly shows QR codes.")
        sys.exit(1)

    print(f"Found {len(blocks)} unique blocks. Decoding...")

    output_path = args.output
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
    enc.add_argument('-o', '--output', required=True,
                     help='Output video path (e.g. output.mp4)')
    enc.add_argument('--overhead', type=float, default=2.0,
                     help=f'Ratio of encoded blocks to source blocks '
                          f'(default: 2.0, minimum: {_MIN_OVERHEAD}, '
                          f'recommended: ≥{_RECOMMENDED_OVERHEAD})')
    enc.add_argument('--fps', type=int, default=10,
                     help='Frames per second in output video (default: 10)')
    enc.add_argument('--ec-level', type=int, default=1, choices=[0, 1, 2, 3],
                     help='QR error correction: 0=L, 1=M (default), 2=Q, 3=H')
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
    enc.add_argument('--codec', choices=['mp4v', 'mjpeg'], default='mp4v',
                     help='Video codec: mp4v (default) or mjpeg (faster, larger)')
    enc.add_argument('-w', '--workers', type=int, default=None,
                     help='Parallel workers for QR generation (default: CPU count)')
    enc.add_argument('--auto-mask', action='store_true',
                     help='Let segno evaluate all 8 ISO 18004 mask patterns '
                          'instead of using the fixed mask=0 fast path. '
                          'Slower (~5× per frame) but may improve scan '
                          'quality under adverse capture conditions.')
    enc.add_argument('-v', '--verbose', action='store_true',
                     help='Print extra detail (block stats, compression ratio, etc.)')

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
    dec.add_argument('--mnn', action='store_true', default=False,
                     dest='use_mnn',
                     help='Enable MNN-accelerated QR detection '
                          '(auto-selects Metal on Apple, CPU elsewhere; '
                          'falls back to OpenCV WeChatQRCode on failure)')
    dec.add_argument(
        '--decode-attempts', type=int, choices=[1, 2, 3], default=1,
        help='Number of zxing-cpp binarizer strategies to try per crop '
             'on the MNN path (ignored without --mnn).  '
             '1 (default): LocalAverage only — fastest; zxing-cpp already '
             'applies try_invert / try_rotate / try_downscale internally, '
             'so this single call covers most cases.  '
             '2: add GlobalHistogram fallback.  '
             '3: add OpenCV adaptive-threshold fallback (for crops with '
             'min(h,w) >= 80 px).  '
             'See dev/wechatqrcode-mnn-poc/results/m3_report.md for the '
             'data behind the single-attempt default.')
    dec.add_argument(
        '--mnn-confidence-threshold', type=float, default=0.0,
        metavar='T',
        help='Drop SSD detections with confidence < T on the MNN path '
             '(ignored without --mnn).  Range [0.0, 1.0); default 0.0 '
             'keeps every detection.  Setting 0.95 typically saves '
             '3-5%% wall clock on real-phone captures by skipping '
             'zxing-cpp work on bboxes that empirically never decode. '
             'Override via QRSTREAM_MNN_CONFIDENCE_THRESHOLD env var. '
             'See dev/wechatqrcode-mnn-poc/results/m3_confidence_report.md.')
    dec.add_argument('-v', '--verbose', action='store_true',
                     help='Print detailed progress')
    dec.add_argument(
        '--detect-isolation', choices=['on', 'off'], default='on',
        help='Isolate the WeChat QR detector in subprocess helpers so a '
             'native crash (opencv_contrib#3570) degrades to a single '
             'dropped frame instead of killing the decode process. '
             'Default: on. Use "off" to trade safety for ~20-30%% '
             'throughput when you know your input is safe.')

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
