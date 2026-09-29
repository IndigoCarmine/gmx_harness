"""Guards shared by everything that writes files or shell scripts.

Everything a user (or an AI agent) passes in that ends up in a path or in a
generated bash script goes through one of these checks. They are deliberately
strict: a name that fails here can always be renamed, while a name that slips
through can break out of the working directory or inject shell code.
"""

import re
from pathlib import Path


class UnsafeNameError(ValueError):
    """A name or token would be unsafe in a path or a generated shell script."""


class UnsafeOperationError(RuntimeError):
    """An operation needs explicit opt-in (``allow_unsafe=True`` / ``confirm=True``)."""


# Step / directory names: letters, digits, "_", "-", "." -- no leading dot or dash.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
# A single file name inside a step directory (same alphabet, no separators).
_SAFE_FILENAME = _SAFE_NAME
# Preprocessor defines for grompp: NAME or NAME=VALUE without whitespace/quotes.
_SAFE_DEFINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(=[A-Za-z0-9_.+\-]*)?$")
# GROMACS residue names are at most 5 characters in a .gro file.
_SAFE_RESNAME = re.compile(r"^[A-Za-z0-9_+\-]{1,5}$")
# Environment variable names.
_SAFE_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_name(name: str, what: str = "calculation_name") -> str:
    """Return ``name`` if it is safe as a directory name and inside bash, else raise."""
    if not isinstance(name, str) or not _SAFE_NAME.match(name) or name in (".", ".."):
        raise UnsafeNameError(
            f"{what}={name!r} is not allowed: use only letters, digits, '_', '-' and '.', "
            "and do not start with '.' or '-'"
        )
    if len(name) > 100:
        raise UnsafeNameError(f"{what}={name!r} is longer than 100 characters")
    return name


def validate_filename(name: str, what: str = "file name") -> str:
    """A bare file name (no directories) that is safe in a generated script."""
    if not isinstance(name, str) or not _SAFE_FILENAME.match(name) or name in (".", ".."):
        raise UnsafeNameError(
            f"{what}={name!r} is not allowed: it must be a plain file name made of "
            "letters, digits, '_', '-' and '.'"
        )
    return name


def validate_define(define: str) -> str:
    if not isinstance(define, str) or not _SAFE_DEFINE.match(define):
        raise UnsafeNameError(
            f"define={define!r} is not allowed: use NAME or NAME=VALUE "
            "(letters, digits, '_', no spaces). Do not include the leading -D."
        )
    return define


def validate_resname(resname: str) -> str:
    if not isinstance(resname, str) or not _SAFE_RESNAME.match(resname):
        raise UnsafeNameError(f"residue name {resname!r} must be 1-5 characters of [A-Za-z0-9_+-]")
    return resname


def validate_env_name(name: str) -> str:
    if not isinstance(name, str) or not _SAFE_ENV.match(name):
        raise UnsafeNameError(f"environment variable name {name!r} is not valid")
    return name


def ensure_within(base: Path, target: Path) -> Path:
    """Resolve ``target`` and make sure it lies inside ``base``."""
    base_r = base.resolve()
    target_r = target.resolve()
    if target_r != base_r and base_r not in target_r.parents:
        raise UnsafeOperationError(f"{target} resolves outside of the working directory {base}")
    return target_r


def ensure_deletable_root(path: Path) -> None:
    """Refuse to recursively delete obviously wrong places (/, $HOME, cwd, their parents)."""
    p = path.resolve()
    forbidden = {Path(p.anchor), Path.home().resolve(), Path.cwd().resolve()}
    if p in forbidden or p in Path.cwd().resolve().parents or p in Path.home().resolve().parents:
        raise UnsafeOperationError(f"refusing to delete {p}")
