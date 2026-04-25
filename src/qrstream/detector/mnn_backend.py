"""
MNN backend detection and selection.

Probes the runtime environment for available MNN backends and selects
the best one according to a platform-aware priority order.

Priority (first match wins):
- Apple (macOS / iOS): Metal → CPU
- NVIDIA GPU present:  CUDA  → CPU       (future, Milestone 3)
- OpenCL capable:      OpenCL → CPU      (future, Milestone 3)
- Fallback:            CPU

All selection is done at **runtime**, not install time.
"""

from __future__ import annotations

import logging
import platform
import sys
from enum import Enum

logger = logging.getLogger(__name__)


class MNNBackend(Enum):
    """Supported MNN backend identifiers.

    Values must match the strings accepted by MNN 3.5+
    ``Interpreter.createSession({'backend': ...})``.
    """
    CPU = "CPU"
    METAL = "METAL"
    CUDA = "CUDA"
    OPENCL = "OpenCL"


def is_mnn_available() -> bool:
    """Check whether MNN Python bindings are importable."""
    try:
        import MNN  # noqa: F401
        return True
    except ImportError:
        return False


def _is_apple_platform() -> bool:
    return sys.platform == "darwin"


def _probe_metal() -> bool:
    """Try to create an MNN Metal session to confirm runtime support."""
    if not _is_apple_platform():
        return False
    try:
        import MNN
        # MNN.nn.backend_type for Metal is typically 1 or "Metal".
        # We do a lightweight probe: create an interpreter with a
        # trivial model and request Metal backend.  If MNN raises,
        # Metal is not available.
        # For now, we just check if the import succeeds and we're on Apple.
        # Full probe will be done when we have real .mnn models.
        return True
    except Exception:
        return False


def select_backend(preferred: str | None = None) -> MNNBackend:
    """Select the best available MNN backend.

    Args:
        preferred: Optional explicit backend name (``"metal"``,
            ``"cpu"``, etc.).  If given and available, it is used
            directly; otherwise falls back to the priority order.

    Returns:
        The selected :class:`MNNBackend`.

    Raises:
        RuntimeError: If MNN is not installed at all.
    """
    if not is_mnn_available():
        raise RuntimeError(
            "MNN Python bindings not found. "
            "Install MNN or run without --mnn."
        )

    # Explicit override
    if preferred:
        try:
            backend = MNNBackend(preferred.lower())
        except ValueError:
            logger.warning("Unknown MNN backend %r, falling back to auto", preferred)
        else:
            logger.info("MNN backend: using explicit %s", backend.value)
            return backend

    # Auto-select by platform priority
    if _is_apple_platform() and _probe_metal():
        logger.info("MNN backend: auto-selected Metal (Apple platform)")
        return MNNBackend.METAL

    logger.info("MNN backend: auto-selected CPU")
    return MNNBackend.CPU
