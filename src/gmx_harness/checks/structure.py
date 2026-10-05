"""Checks on structures and topologies (gro / top / itp / ndx), pure Python."""

import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from ..io.gro import GroFile
from .report import Report

_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"')


def _sections(text: str) -> Iterable[tuple[str, str]]:
    """(section, data line) pairs, comments stripped; preprocessor lines are skipped."""
    section = ""
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[] ").strip().lower()
            continue
        yield section, line


def topology_molecules(text: str) -> list[tuple[str, int]]:
    """``[ molecules ]`` entries (name, count) in order."""
    out = []
    for section, line in _sections(text):
        parts = line.split()
        if section == "molecules" and len(parts) >= 2 and parts[1].isdigit():
            out.append((parts[0], int(parts[1])))
    return out


def moleculetype_atoms(text: str) -> dict[str, int]:
    """Atom count of every ``[ moleculetype ]`` defined in ``text``."""
    out: dict[str, int] = {}
    current: str | None = None
    for section, line in _sections(text):
        if section == "moleculetype":  # one data line: name nrexcl
            current = line.split()[0]
            out[current] = 0
        elif section == "atoms" and current is not None:
            out[current] += 1
    return out


def _with_includes(text: str, base: Path | None, files: Mapping[str, str] | None, depth: int = 0) -> tuple[str, list[str]]:
    """Text with local ``#include`` files appended (from ``files`` or next to ``base``); unresolved names."""
    parts: list[str] = [text]
    missing: list[str] = []
    if depth > 5:
        return text, missing
    for raw in text.splitlines():
        m = _INCLUDE.match(raw)
        if not m:
            continue
        name = m.group(1)
        sub: str | None = None
        if files is not None and name in files:
            sub = files[name]
        elif base is not None and (base / name).is_file():
            sub = (base / name).read_text(encoding="utf-8", errors="replace")
        if sub is None:
            missing.append(name)
            continue
        t, miss = _with_includes(sub, (base / name).parent if base is not None else None, files, depth + 1)
        parts.append(t)
        missing += miss
    return "\n".join(parts), missing


def topology_atom_count(top_text: str, *, base: str | os.PathLike[str] | None = None,
                        files: Mapping[str, str] | None = None, where: str = "topo.top") -> tuple[int | None, Report]:
    """Total atoms implied by ``[ molecules ]``; S011 if a molecule type cannot be found."""
    rep = Report()
    full, _missing = _with_includes(top_text, Path(base) if base is not None else None, files)
    natoms = moleculetype_atoms(full)
    total = 0
    for name, count in topology_molecules(top_text):
        if name not in natoms:
            rep.error("S011", f"moleculetype {name} not found in the topology or its local includes, so the atom "
                              "count cannot be checked (include it locally, or waive S011 with the reason)", where)
            return None, rep
        total += natoms[name] * count
    return total, rep


def check_topology(gro: GroFile | int, top_text: str, *, base: str | os.PathLike[str] | None = None,
                   files: Mapping[str, str] | None = None, where: str = "",
                   single_type: bool = False) -> Report:
    """S001: gro atoms == topology atoms; S003 (with ``single_type``): only one molecule type."""
    n_gro = gro if isinstance(gro, int) else len(gro)
    total, rep = topology_atom_count(top_text, base=base, files=files, where=where or "topology")
    mols = topology_molecules(top_text)
    if not mols:
        rep.error("S001", "no [ molecules ] entries", where)
    if single_type and len({n for n, _ in mols}) > 1:
        rep.error("S003", f"several molecule types {[n for n, _ in mols]}; this step expects one", where)
    if total is not None and total != n_gro:
        desc = " + ".join(f"{c} x {n}" for n, c in mols)
        rep.error("S001", f"structure has {n_gro} atoms, topology [ molecules ] ({desc}) has {total}", where)
    return rep


def check_whole_molecules(natoms_total: int, natoms_per_mol: int, where: str = "") -> Report:
    """S002: ``natoms_total`` is a whole number of ``natoms_per_mol`` molecules."""
    rep = Report()
    if natoms_per_mol <= 0 or natoms_total % natoms_per_mol:
        rep.error("S002", f"{natoms_total} atoms is not a whole number of {natoms_per_mol}-atom molecules "
                          f"(remainder {natoms_total % natoms_per_mol if natoms_per_mol > 0 else '?'})", where)
    return rep


def check_box(gro: GroFile, min_edge: float, where: str = "") -> Report:
    """S004: every box edge >= ``min_edge`` nm."""
    rep = Report()
    small = [f"{ax}={e:.2f}" for ax, e in zip("xyz", (gro.box_x, gro.box_y, gro.box_z)) if e < min_edge]
    if small:
        rep.error("S004", f"box edge(s) {', '.join(small)} nm < {min_edge} nm: the assembly may see its periodic "
                          "image within the cut-off", where)
    return rep


def check_periodic_twist(ndisk: int, rot: float, nros: int, where: str = "", tol: float = 1e-6) -> Report:
    """S005: a fiber closed on its periodic image needs ndisk*rot to be a multiple of 360/nros."""
    rep = Report()
    period = 360.0 / nros
    total = ndisk * rot
    k = round(total / period)
    if not math.isclose(total, k * period, abs_tol=tol):
        rep.error("S005", f"ndisk*rot = {total:g} deg is not a multiple of 360/nros = {period:g} deg: the last disk "
                          "does not match the first disk's periodic image", where)
    return rep


def check_bond_pairs(bonds: Sequence[tuple[int, int]], natoms: int, where: str = "") -> Report:
    """S006: (a, b) with a in the monomer and b in the monomer pair (1..2*natoms)."""
    rep = Report()
    for a, b in bonds:
        if not 1 <= a <= natoms or not 1 <= b <= 2 * natoms:
            rep.error("S006", f"bond ({a}, {b}) is outside a {natoms}-atom monomer / its pair "
                              f"(a in 1..{natoms}, b in 1..{2 * natoms})", where)
    return rep


def check_bond_lengths(gro: GroFile, bonds: Sequence[tuple[int, int]], lo: float, hi: float,
                       where: str = "") -> Report:
    """S007: every built bond (global 1-based atom pairs) is between ``lo`` and ``hi`` nm."""
    rep = Report()
    if not bonds:
        return rep
    xyz = np.array([a.coordinate for a in gro.atoms])
    pairs = np.array(bonds) - 1
    if pairs.max() >= len(xyz) or pairs.min() < 0:
        rep.error("S006", f"bond atom numbers exceed the structure's {len(xyz)} atoms", where)
        return rep
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    bad = np.flatnonzero((d < lo) | (d > hi))
    if bad.size:
        ex = ", ".join(f"{bonds[i][0]}-{bonds[i][1]}: {d[i]:.3f}" for i in bad[:5])
        rep.error("S007", f"{bad.size} of {len(bonds)} built bonds outside {lo:g}-{hi:g} nm ({ex}"
                          f"{', ...' if bad.size > 5 else ''})", where)
    return rep


def check_contacts(gro: GroFile, natoms_per_mol: int, threshold: float = 0.1, where: str = "") -> Report:
    """S008 (warn): atoms of different molecules closer than ``threshold`` nm (no PBC; cell list)."""
    rep = Report()
    xyz = np.array([a.coordinate for a in gro.atoms])
    if len(xyz) == 0 or natoms_per_mol <= 0:
        return rep
    mol = np.arange(len(xyz)) // natoms_per_mol
    cell = np.floor(xyz / threshold).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, c in enumerate(map(tuple, cell)):
        buckets.setdefault((int(c[0]), int(c[1]), int(c[2])), []).append(i)
    close = 0
    example = ""
    for (cx, cy, cz), members in buckets.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = buckets.get((cx + dx, cy + dy, cz + dz))
                    if not other:
                        continue
                    a = np.array(members)
                    b = np.array(other)
                    d = np.linalg.norm(xyz[a][:, None, :] - xyz[b][None, :, :], axis=2)
                    mask = (d < threshold) & (mol[a][:, None] < mol[b][None, :])
                    n = int(mask.sum())
                    if n:
                        close += n
                        if not example:
                            i, j = np.argwhere(mask)[0]
                            example = f"atoms {a[i] + 1} and {b[j] + 1}: {d[i, j]:.3f} nm"
    if close:
        rep.warn("S008", f"{close} atom pairs of different molecules closer than {threshold} nm (e.g. {example}); "
                         "relax the structure before EM", where)
    return rep


def check_labels(gro: GroFile, label_names: Sequence[str], nmol: int, offset: int = 0, where: str = "") -> Report:
    """S009: atoms offset.. of the system repeat the labeled monomer's atom names ``nmol`` times."""
    rep = Report()
    n = len(label_names)
    if offset + nmol * n > len(gro):
        rep.error("S009", f"{nmol} x {n}-atom monomers (+{offset}) do not fit in {len(gro)} atoms", where)
        return rep
    for j in range(nmol):
        names = [gro.atoms[offset + j * n + i].atom_name for i in range(n)]
        if names != list(label_names):
            k = next(i for i in range(n) if names[i] != label_names[i])
            rep.error("S009", f"molecule {j}: atom {offset + j * n + k + 1} is {names[k]!r}, the labeled monomer has "
                              f"{label_names[k]!r} there (different atom order or monomer)", where)
            break
    return rep


def parse_ndx(text: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            current = s.strip("[] ").strip()
            groups[current] = []
        elif s and current is not None:
            groups[current] += [int(v) for v in s.split()]
    return groups


def check_ndx(ndx: str | os.PathLike[str], natoms: int, required: Iterable[str] = (), where: str = "") -> Report:
    """S010: every group non-empty and within 1..natoms; ``required`` groups exist."""
    rep = Report()
    groups = parse_ndx(Path(ndx).read_text(encoding="utf-8"))
    for name in required:
        if name not in groups:
            rep.error("S010", f"index group {name} is missing (have {sorted(groups)})", where)
    for name, idx in groups.items():
        if not idx:
            rep.error("S010", f"index group {name} is empty", where)
        elif min(idx) < 1 or max(idx) > natoms:
            rep.error("S010", f"index group {name} has atoms outside 1..{natoms}", where)
    return rep
