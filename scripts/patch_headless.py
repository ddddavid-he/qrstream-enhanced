#!/usr/bin/env python3
"""Patch pyproject.toml to produce the ``qrstream-headless`` variant.

This script is run in CI before ``uv build`` to produce a second
PyPI package that omits PySide6 (the only GUI dependency).  Everything
else (source, version, entry points) stays identical.

Usage::

    python scripts/patch_headless.py
    uv build
    # → dist/qrstream_headless-*.whl
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def patch(toml_path: Path = Path("pyproject.toml")) -> None:
    content = toml_path.read_text(encoding="utf-8")
    original = content

    # 1. Rename the package.
    content = content.replace(
        'name = "qrstream"',
        'name = "qrstream-headless"',
        1,
    )

    # 2. Strip PySide6-Essentials from the dependencies list.
    content = re.sub(
        r'^\s*"PySide6-Essentials[^"]*",?\n',
        "",
        content,
        flags=re.MULTILINE,
    )

    if content == original:
        print("Warning: no changes made — already patched?", file=sys.stderr)
        sys.exit(1)

    toml_path.write_text(content, encoding="utf-8")
    print(f"Patched {toml_path} for headless build")


if __name__ == "__main__":
    patch()
