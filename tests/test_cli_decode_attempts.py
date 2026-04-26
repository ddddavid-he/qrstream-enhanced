"""Tests for the CLI's ``--decode-attempts`` argument (M3 §3.2)."""

from __future__ import annotations

import pytest

from qrstream.cli import build_parser


class TestDecodeAttemptsCLI:
    def test_default_is_one(self):
        parser = build_parser()
        args = parser.parse_args(
            ['decode', 'in.mp4', '-o', 'out.bin'],
        )
        assert args.decode_attempts == 1

    def test_accepts_valid_values(self):
        parser = build_parser()
        for n in (1, 2, 3):
            args = parser.parse_args(
                ['decode', 'in.mp4', '-o', 'out.bin',
                 '--decode-attempts', str(n)],
            )
            assert args.decode_attempts == n

    def test_rejects_invalid_values(self):
        parser = build_parser()
        for bad in ('0', '4', '-1', 'foo'):
            with pytest.raises(SystemExit):
                parser.parse_args(
                    ['decode', 'in.mp4', '-o', 'out.bin',
                     '--decode-attempts', bad],
                )

    def test_works_with_mnn_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            ['decode', 'in.mp4', '-o', 'out.bin',
             '--mnn', '--decode-attempts', '2'],
        )
        assert args.use_mnn is True
        assert args.decode_attempts == 2


class TestMnnConfidenceThresholdCLI:
    """``--mnn-confidence-threshold`` CLI flag (M3 §3.3 / C1)."""

    def test_default_is_zero(self):
        parser = build_parser()
        args = parser.parse_args(['decode', 'in.mp4', '-o', 'out.bin'])
        assert args.mnn_confidence_threshold == 0.0

    def test_accepts_in_range_values(self):
        parser = build_parser()
        for v in ('0.0', '0.3', '0.5', '0.95', '0.999'):
            args = parser.parse_args(
                ['decode', 'in.mp4', '-o', 'out.bin',
                 '--mnn-confidence-threshold', v],
            )
            assert args.mnn_confidence_threshold == pytest.approx(float(v))

    def test_works_with_mnn_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            ['decode', 'in.mp4', '-o', 'out.bin',
             '--mnn', '--mnn-confidence-threshold', '0.95'],
        )
        assert args.use_mnn is True
        assert args.mnn_confidence_threshold == pytest.approx(0.95)

    def test_combines_with_decode_attempts(self):
        parser = build_parser()
        args = parser.parse_args(
            ['decode', 'in.mp4', '-o', 'out.bin',
             '--mnn', '--decode-attempts', '2',
             '--mnn-confidence-threshold', '0.5'],
        )
        assert args.decode_attempts == 2
        assert args.mnn_confidence_threshold == pytest.approx(0.5)
