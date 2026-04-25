"""CLI flag plumbing for --detect-isolation."""

import pytest

from qrstream.cli import build_parser


def _parse(argv):
    parser = build_parser()
    return parser.parse_args(argv)


def test_cli_accepts_detect_isolation_on():
    args = _parse([
        'decode', 'x.mp4', '-o', 'y',
        '--detect-isolation', 'on',
    ])
    assert args.detect_isolation == 'on'


def test_cli_accepts_detect_isolation_off():
    args = _parse([
        'decode', 'x.mp4', '-o', 'y',
        '--detect-isolation', 'off',
    ])
    assert args.detect_isolation == 'off'


def test_cli_default_is_on():
    args = _parse(['decode', 'x.mp4', '-o', 'y'])
    assert args.detect_isolation == 'on'


def test_cli_rejects_invalid_value():
    with pytest.raises(SystemExit):
        _parse([
            'decode', 'x.mp4', '-o', 'y',
            '--detect-isolation', 'auto',
        ])
