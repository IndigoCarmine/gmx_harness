"""Measurements on built structures: distances between molecules, contacts, planes.

Every function takes the structure itself (a ``GroFile`` of identical molecules of
``natoms_per_mol`` atoms, in order) and a selection, never a coordinate array, so a
script can check a build without handling coordinates:

    heavy = [i + 1 for i, a in enumerate(mono.atoms) if not a.atom_name.startswith("H")]
    d = min_distance(sheet, natoms, range(16), range(16, 32), atoms=heavy)   # layer 1 vs layer 2
    pairs = find_intermolecular_pairs(sheet, natoms, donors=(10, 12), acceptors=(16, 14), cutoff=0.35)

Selections: molecules are 0-based molecule numbers (as in ``molecule_atoms``); ``atoms``
are 1-based atom numbers *within* a molecule (None = all). ``pbc`` names the axes along
which the minimum-image convention applies, with the box of the ``GroFile``
(rectangular boxes only). The default ``"xyz"`` is what GROMACS sees; pass ``""`` for a
structure whose box is only for viewing.
"""

from collections.abc import Iterable, Iterator

import numpy as np
import numpy.typing as npt

from .io.gro import GroFile

Array = npt.NDArray[np.float64]

_CHUNK = 2_000_000  # distance-matrix elements per block


def _box(gro: GroFile, pbc: str) -> tuple[list[int], Array]:
    axes = sorted({"xyz".index(c) for c in pbc.lower()}) if pbc else []
    if any(c not in "xyz" for c in pbc.lower()):
        raise ValueError(f"pbc must be made of x, y, z, got {pbc!r}")
    if axes and gro.box_triclinic is not None and any(gro.box_triclinic):
        raise NotImplementedError("minimum image for a triclinic box is not supported")
    box = np.array([gro.box_x, gro.box_y, gro.box_z], dtype=float)
    if any(box[a] <= 0 for a in axes):
        raise ValueError(f"box edge along a periodic axis ({pbc}) is not positive: {box.tolist()}")
    return axes, box


def _min_image(dv: Array, axes: list[int], box: Array) -> Array:
    if len(axes) == 3:
        dv -= box * np.round(dv / box)
    else:
        for a in axes:
            dv[..., a] -= box[a] * np.round(dv[..., a] / box[a])
    return dv


def _atoms(gro: GroFile, natoms_per_mol: int, mols: Iterable[int], atoms: Iterable[int] | None) -> tuple[list[int], list[int]]:
    """(molecule numbers, local 0-based atom offsets), validated."""
    if natoms_per_mol <= 0 or len(gro) % natoms_per_mol:
        raise ValueError(f"{len(gro)} atoms is not a whole number of {natoms_per_mol}-atom molecules")
    nmol = len(gro) // natoms_per_mol
    ms = list(mols)
    if any(m < 0 or m >= nmol for m in ms):
        raise ValueError(f"molecule numbers must be within 0..{nmol - 1}")
    loc = list(range(natoms_per_mol)) if atoms is None else [a - 1 for a in atoms]
    if any(a < 0 or a >= natoms_per_mol for a in loc):
        raise ValueError(f"atom numbers must be within 1..{natoms_per_mol}")
    return ms, loc


def _blocks(gro: GroFile, natoms_per_mol: int, mols_a: Iterable[int], mols_b: Iterable[int] | None,
            atoms: Iterable[int] | None, pbc: str, exclude_neighbours: int) -> Iterator[Array]:
    """Distance blocks (rows: atoms of one A molecule, columns: B atoms); excluded pairs are inf."""
    if exclude_neighbours < 0:
        raise ValueError("exclude_neighbours must be >= 0")
    a_mols, loc = _atoms(gro, natoms_per_mol, mols_a, atoms)
    within = mols_b is None
    b_mols = a_mols if mols_b is None else _atoms(gro, natoms_per_mol, mols_b, atoms)[0]
    axes, box = _box(gro, pbc)
    xyz = gro.positions()
    local = np.array(loc, dtype=int)
    b_idx = (np.array(b_mols, dtype=int)[:, None] * natoms_per_mol + local[None]).ravel()
    b_mol = np.repeat(np.array(b_mols, dtype=int), len(local))
    xb = xyz[b_idx]
    for i in a_mols:
        keep = np.abs(b_mol - i) > exclude_neighbours
        if within:
            keep &= b_mol > i  # each unordered pair once
        if not keep.any():
            continue
        xa = xyz[i * natoms_per_mol + local]
        dv = _min_image(xb[keep][None] - xa[:, None], axes, box)
        yield np.linalg.norm(dv, axis=2)


def min_distance(gro: GroFile, natoms_per_mol: int, mols_a: Iterable[int], mols_b: Iterable[int] | None = None, *,
                 atoms: Iterable[int] | None = None, pbc: str = "xyz", exclude_neighbours: int = 0) -> float:
    """
    Shortest distance (nm) between ``atoms`` of a molecule in ``mols_a`` and ``atoms`` of a different
    molecule in ``mols_b`` (``mols_b`` = None: between two molecules of ``mols_a``). Pairs of molecules
    whose numbers differ by ``exclude_neighbours`` or less are skipped (1 = skip direct neighbours of a
    chain). inf if no pair is left.
    """
    best = np.inf
    for d in _blocks(gro, natoms_per_mol, mols_a, mols_b, atoms, pbc, exclude_neighbours):
        best = min(best, float(d.min()))
    return float(best)


def count_contacts(gro: GroFile, natoms_per_mol: int, mols_a: Iterable[int], mols_b: Iterable[int] | None = None, *,
                   cutoff: float, atoms: Iterable[int] | None = None, pbc: str = "xyz",
                   exclude_neighbours: int = 0) -> int:
    """
    Number of atom pairs closer than ``cutoff`` nm, selected as in ``min_distance``. With ``mols_b`` = None
    each pair is counted once; with ``mols_b`` every (A atom, B atom) pair is counted, so keep A and B apart.
    """
    return sum(int((d < cutoff).sum()) for d in _blocks(gro, natoms_per_mol, mols_a, mols_b, atoms, pbc,
                                                        exclude_neighbours))


def find_intermolecular_pairs(gro: GroFile, natoms_per_mol: int, donors: Iterable[int], acceptors: Iterable[int],
                              cutoff: float, *, pbc: str = "xyz") -> list[tuple[int, int, float]]:
    """
    Every (donor, acceptor) atom pair of *different* molecules closer than ``cutoff`` nm, as
    ``(donor, acceptor, distance)`` with 1-based atom numbers of the whole system. ``donors`` and
    ``acceptors`` are atom numbers within a molecule. Ordered by molecule, then by the order of
    ``donors``, then by acceptor. Use it to find the H-bonds of a built assembly for
    ``write_intermolecular_bonds`` instead of listing them by hand.
    """
    mols, d_loc = _atoms(gro, natoms_per_mol, range(len(gro) // max(natoms_per_mol, 1)), donors)
    _, a_loc = _atoms(gro, natoms_per_mol, mols, acceptors)
    axes, box = _box(gro, pbc)
    xyz = gro.positions()
    don = np.array([k * natoms_per_mol + i for k in mols for i in d_loc], dtype=int)  # 0-based
    acc = np.array([k * natoms_per_mol + i for k in mols for i in a_loc], dtype=int)
    out: list[tuple[int, int, float]] = []
    step = max(1, _CHUNK // max(len(acc), 1))
    for s in range(0, len(don), step):
        rows = don[s:s + step]
        dv = _min_image(xyz[acc][None] - xyz[rows][:, None], axes, box)
        d = np.linalg.norm(dv, axis=2)
        d[rows[:, None] // natoms_per_mol == acc[None] // natoms_per_mol] = np.inf  # not within a molecule
        for i, j in zip(*np.nonzero(d < cutoff)):
            out.append((int(rows[i]) + 1, int(acc[j]) + 1, float(d[i, j])))
    return out


def centroid(gro: GroFile, atoms: Iterable[int] | None = None) -> Array:
    """Mean position (nm) of ``atoms`` (1-based positions in ``gro``; None = all)."""
    out: Array = gro.positions(atoms).mean(0)
    return out


def plane_normal(gro: GroFile, atoms: Iterable[int]) -> Array:
    """
    Unit normal of the best plane through ``atoms`` (1-based positions in ``gro``, at least 3):
    the direction of least spread. Its sign is not chosen by any rule; orient it yourself if it matters.
    """
    p = gro.positions(atoms)
    if len(p) < 3:
        raise ValueError("a plane needs at least 3 atoms")
    out: Array = np.linalg.svd(p - p.mean(0))[2][2]
    return out
