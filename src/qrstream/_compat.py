"""Platform compatibility helpers."""

import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_native_stderr():
    """Temporarily redirect fd 2 to /dev/null for native-lib imports.

    Silences ObjC duplicate-class warnings emitted by the macOS runtime
    when cv2 and av bundle separate FFmpeg dylibs.  The fd is always
    restored via finally, even if the wrapped code raises.

    On non-Darwin platforms this is a no-op.
    """
    if sys.platform != "darwin":
        yield
        return

    old_fd = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
        finally:
            os.close(devnull)
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
