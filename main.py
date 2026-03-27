#!/usr/bin/env python3
"""
QRStream main entry point.

Supports two calling modes:
  - uv run main.py [args]
  - uv run qrstream [args]

The second form is configured via pyproject.toml console_scripts.
"""

from qrstream.cli import main

if __name__ == "__main__":
    main()
