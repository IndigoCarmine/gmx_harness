"""sha256 of files, read in chunks so that multi-GB trajectories never sit in memory at once."""

import hashlib
import os

__all__ = ["sha256_file"]


def sha256_file(path: str | os.PathLike[str], bufsize: int = 1 << 20) -> str:
    """Hex sha256 of a file, read ``bufsize`` bytes at a time (same value as ``sha256sum``)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()
