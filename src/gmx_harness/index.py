"""Build GROMACS index (.ndx) files from molecule ranges.

For systems made of identical molecules in index order (e.g. an assembly built
with ``gmx_harness.build``), groups are most easily given as molecule numbers:

    fiber = molecule_atoms(169, range(216))                       # all atoms of molecules 0..215
    cores = molecule_atoms(169, range(216), local=range(3, 52))    # atoms 3..51 of each molecule
    write_ndx("MOL.ndx", {"Fiber1": fiber, "fiberA": fiber})

The text format (15 right-aligned indices per line) is the one of the usage
workspace's ``make_ndx.py``, so its files are reproduced exactly.
"""

import os
import re
from collections.abc import Iterable, Mapping, Sequence

from .safety import UnsafeNameError

_GROUP = re.compile(r"^[A-Za-z0-9_+\-.]+$")


def molecule_atoms(
    natoms_per_mol: int,
    molecules: Iterable[int],
    local: Iterable[int] | None = None,
    offset: int = 0,
) -> list[int]:
    """
    1-based atom indices of the given molecules (0-based molecule numbers).

    ``local`` restricts each molecule to these 1-based atom numbers within it;
    ``offset`` is the number of atoms before the first molecule.
    """
    if natoms_per_mol <= 0:
        raise ValueError("natoms_per_mol must be > 0")
    loc = list(range(1, natoms_per_mol + 1)) if local is None else sorted(set(local))
    if loc and (loc[0] < 1 or loc[-1] > natoms_per_mol):
        raise ValueError(f"local atom numbers must be within 1..{natoms_per_mol}")
    out: list[int] = []
    for m in molecules:
        if m < 0:
            raise ValueError("molecule numbers are 0-based and must be >= 0")
        base = offset + m * natoms_per_mol
        out += [base + a for a in loc]
    return out


def format_ndx(groups: Mapping[str, Sequence[int]], values_per_line: int = 15) -> str:
    """Index-file text: ``[ name ]`` followed by the indices, ``values_per_line`` per line."""
    if values_per_line <= 0:
        raise ValueError("values_per_line must be > 0")
    blocks: list[str] = []
    for name, values in groups.items():
        if not _GROUP.match(name):
            raise UnsafeNameError(f"index group name {name!r} may only use [A-Za-z0-9_+-.]")
        lines = [f"[ {name} ]"]
        vals = list(values)
        for i in range(0, len(vals), values_per_line):
            lines.append(" ".join(f"{v:4d}" for v in vals[i : i + values_per_line]))
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


def write_ndx(path: str | os.PathLike[str], groups: Mapping[str, Sequence[int]], values_per_line: int = 15) -> None:
    """Write ``format_ndx(groups)`` to ``path``."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(format_ndx(groups, values_per_line))
