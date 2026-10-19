"""Build molecular assemblies from .gro structures (ported from mylibs' ``gromacs.pre_coordinator`` / ``rosette_maker``).

Everything works on ``GroFile`` objects and plain numpy rotation matrices:

    mono = GroFile.from_gro_file("MOL.gro")
    precoordinate2(mono, top=1, aromatic_nh=2, aromatic_o=3)      # orient the monomer
    rosette = make_rosette2(mono, n=6, size=0.35).to_gro()      # one ring
    fiber = make_oligorosette(rosette, n=10, length=0.35, angle=10)  # a stack of rings
    fiber.to_gro(box=(10, 10, 3.5), renumber=True).save_gro("fiber.gro")

Atom arguments of the pre-coordination functions are GROMACS atom numbers
(the index column of the .gro file, usually 1-based).

Newer, non-mutating API: molecules move only as rigid bodies (``RigidTransform``), every call
returns a new object, and no coordinate is set by hand:

    tape = Assembly.replicate(mono, [step.power(k) for k in range(8)])     # one copy per transform
    sheet = tape.repeat([lx, 0, 0], 2).repeat([0, 0.35, 0], 16)            # periodic copies
    sheet = sheet.centered_in_box(box).number_residues()
    save_assembly("MOL_sheet", sheet.to_gro(box=box, renumber=True), len(mono), {"Sheet": range(len(sheet))})

``Assembly.translate/rotate`` and ``GroFile.translate/rotate`` (in place) stay for existing scripts.
"""

import copy
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from .index import molecule_atoms, write_ndx
from .io.gro import GroAtom, GroFile
from .transform import RigidTransform, align_matrix, rotation_matrix

Matrix = npt.NDArray[np.float64]
Axis = Literal["x", "y", "z"]


# 3x3 matrices, kept here for existing callers; ``RigidTransform`` (transform.py) is the newer API.
rotation = rotation_matrix
align = align_matrix


class Assembly:
    """An ordered group of molecules (``GroFile``) moved together (mylibs' ``Substructure``)."""

    def __init__(self, molecules: list[GroFile]):
        self.molecules = molecules

    def __iter__(self) -> Iterator[GroFile]:
        return iter(self.molecules)

    def __len__(self) -> int:
        return len(self.molecules)

    def translate(self, vector: npt.ArrayLike) -> None:
        for m in self.molecules:
            m.translate(vector)

    def rotate(self, matrix: Matrix) -> None:
        for m in self.molecules:
            m.rotate(matrix)

    def atoms(self) -> list[GroAtom]:
        return [a for m in self.molecules for a in m.atoms]

    def to_gro(
        self,
        title: str = "MOL",
        box: tuple[float, float, float] = (0.0, 0.0, 0.0),
        renumber: bool = False,
    ) -> GroFile:
        """All atoms as one ``GroFile`` (copies). ``renumber`` renumbers atoms from 1."""
        gro = GroFile(title, copy.deepcopy(self.atoms()), *box)
        if renumber:
            gro.renumber()
        return gro

    # ----------------------------------------------------------- non-mutating API
    # Everything below returns a new object and leaves this one unchanged; molecules are
    # only moved as rigid bodies (``RigidTransform``), so no coordinate is ever set by hand.

    @classmethod
    def replicate(cls, monomer: GroFile, transforms: Iterable[RigidTransform]) -> "Assembly":
        """One copy of ``monomer`` per transform, in order; residues numbered 1, 2, ... (``number_residues``)."""
        return cls([monomer.transformed(t) for t in transforms]).number_residues()

    @classmethod
    def concat(cls, *assemblies: "Assembly") -> "Assembly":
        """Copies of the molecules of ``assemblies``, one after the other (residue numbers kept)."""
        return cls([copy.deepcopy(m) for a in assemblies for m in a])

    def __getitem__(self, key: slice) -> "Assembly":
        """``assembly[a:b]``: a copy of molecules a..b-1 as a new Assembly."""
        if not isinstance(key, slice):
            raise TypeError("index an Assembly with a slice (assembly[i:i + 1] for one molecule)")
        return Assembly(copy.deepcopy(self.molecules[key]))

    def transformed(self, transform: RigidTransform) -> "Assembly":
        """A copy with every molecule moved by ``transform``."""
        return Assembly([m.transformed(transform) for m in self.molecules])

    def repeat(self, vector: npt.ArrayLike, n: int) -> "Assembly":
        """
        ``n`` copies of the whole assembly, copy k shifted by k * ``vector`` (nm), copy by copy
        (all molecules of copy 0 first). Residue numbers are kept; call ``number_residues`` after.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        v = np.asarray(vector, dtype=float)
        return Assembly([m.transformed(RigidTransform.translation(k * v)) for k in range(n) for m in self.molecules])

    def number_residues(self, start: int = 1) -> "Assembly":
        """
        A copy with residues numbered consecutively from ``start`` in atom order. A new residue
        begins with every molecule and wherever the residue number changes inside one, so a
        molecule that is itself a ring of several residues (a rosette) keeps them apart.
        """
        new = Assembly(copy.deepcopy(self.molecules))
        number = start - 1
        for mol in new.molecules:
            previous: int | None = None
            for atom in mol.atoms:
                if atom.residue_number != previous:
                    previous = atom.residue_number
                    number += 1
                atom.residue_number = number
        return new

    def positions(self) -> npt.NDArray[np.float64]:
        """Read-only (N, 3) copy of all coordinates (nm), molecule by molecule."""
        out = np.vstack([m.positions() for m in self.molecules]) if self.molecules else np.zeros((0, 3))
        out.flags.writeable = False
        return out

    def bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(lowest, highest) corner of the atoms' bounding box (nm)."""
        p = self.positions()
        if len(p) == 0:
            raise ValueError("empty assembly")
        return p.min(0), p.max(0)

    def extent(self) -> npt.NDArray[np.float64]:
        """Size of the atoms' bounding box along x, y, z (nm)."""
        lo, hi = self.bounds()
        out: npt.NDArray[np.float64] = hi - lo
        return out

    def fitting_box(self, *, factor: float = 1.0, pad: float = 0.0, min_edge: float = 0.0) -> tuple[float, float, float]:
        """Box edges ``max(factor * extent + 2 * pad, min_edge)`` along x, y, z (nm)."""
        e = factor * self.extent() + 2 * pad
        return (float(max(e[0], min_edge)), float(max(e[1], min_edge)), float(max(e[2], min_edge)))

    def centered_in_box(self, box: tuple[float, float, float], about: npt.ArrayLike | None = None) -> "Assembly":
        """
        A copy moved so that ``about`` lands on the centre of ``box`` (nm). ``about`` = None uses
        the centre of the bounding box; pass e.g. ``(0, 0, 0)`` when the assembly was built
        around the origin (the axis of a fiber).
        """
        if about is None:
            lo, hi = self.bounds()
            centre = (lo + hi) / 2
        else:
            centre = np.asarray(about, dtype=float)
        return self.transformed(RigidTransform.translation(np.asarray(box, dtype=float) / 2 - centre))


def save_assembly(
    base: str | os.PathLike[str],
    gro: GroFile,
    natoms_per_mol: int,
    groups: Mapping[str, Iterable[int]],
) -> list[Path]:
    """
    Write ``<base>.gro``, ``<base>.pdb`` (for viewers) and ``<base>.ndx`` for a system of identical
    molecules of ``natoms_per_mol`` atoms. ``groups`` maps index-group names to 0-based molecule
    numbers (``molecule_atoms``). ``base`` has no extension (one is appended, never replaced, so a
    name like ``fiber_rot_+0.5`` is safe). Returns the written paths.
    """
    if natoms_per_mol <= 0 or len(gro) % natoms_per_mol:
        raise ValueError(f"{len(gro)} atoms is not a whole number of {natoms_per_mol}-atom molecules")
    nmol = len(gro) // natoms_per_mol
    ndx = {}
    for name, mols in groups.items():
        ms = list(mols)
        if any(m >= nmol for m in ms):
            raise ValueError(f"group {name!r}: molecule numbers must be < {nmol}")
        ndx[name] = molecule_atoms(natoms_per_mol, ms)
    paths = [Path(f"{os.fspath(base)}{ext}") for ext in (".gro", ".pdb", ".ndx")]
    gro.save_gro(str(paths[0]))
    gro.save_pdb(str(paths[1]))
    write_ndx(paths[2], ndx)
    return paths


def pre_coordinate(molecule: GroFile, top: int, side_a: int, side_b: int) -> GroFile:
    """
    Put atom ``top`` at the origin, turn the sum vector of atoms ``side_a`` and
    ``side_b`` onto +x, then roll about x so that ``side_a`` lies in the xz plane.
    Modifies and returns ``molecule``.
    """
    molecule.translate(-molecule.get_child(top).coordinate)
    vector = molecule.get_child(side_a).coordinate + molecule.get_child(side_b).coordinate
    molecule.rotate(align(vector, [1.0, 0.0, 0.0]))
    v = molecule.get_child(side_a).coordinate
    molecule.rotate(rotation("x", np.arctan2(v[1], v[2])))
    return molecule


def precoordinate2(molecule: GroFile, top: int, aromatic_nh: int, aromatic_o: int) -> GroFile:
    """
    Put atom ``top`` at the origin, turn atom ``aromatic_nh`` onto the +x axis, then
    roll about x so that ``aromatic_o`` lies in the xy plane on the -y side (z = 0, y < 0).
    (mylibs' docstring said "xz plane"; this is what its code does and what the
    rosette builders expect.) Modifies and returns ``molecule``.
    """
    molecule.translate(-molecule.get_child(top).coordinate)
    v = molecule.get_child(aromatic_nh).coordinate
    molecule.rotate(rotation("z", -np.arctan2(v[1], v[0])))
    v = molecule.get_child(aromatic_nh).coordinate
    molecule.rotate(rotation("y", np.arctan2(v[2], v[0])))
    v = molecule.get_child(aromatic_o).coordinate
    molecule.rotate(rotation("x", np.arctan2(v[2], -v[1])))
    return molecule


def make_rosette(monomer: GroFile, n: int, size: float, angle: float, degree: bool = True) -> Assembly:
    """
    ``n`` copies of ``monomer`` on a circle of diameter ``size`` (nm) in the xz plane,
    each tilted by ``angle`` about y and turned radially about y.
    """
    rosette = Assembly([copy.deepcopy(monomer) for _ in range(n)])
    for i, m in enumerate(rosette):
        m.rotate(rotation("y", -angle, degrees=degree))
        m.translate([size / 2, 0.0, 0.0])
        m.rotate(rotation("y", 360 / n * i, degrees=True))
    return rosette


def make_rosette2(monomer: GroFile, n: int, size: float) -> Assembly:
    """``n`` copies of ``monomer`` moved out by ``size`` (nm) along x and turned radially about z."""
    rosette = Assembly([copy.deepcopy(monomer) for _ in range(n)])
    for i, m in enumerate(rosette):
        m.translate([size, 0.0, 0.0])
        m.rotate(rotation("z", 360 / n * i, degrees=True))
    return rosette


def make_half_rosette2(monomer: GroFile, n: int, size: float) -> Assembly:
    """Like ``make_rosette2`` but spread over a half circle (180 degrees)."""
    rosette = Assembly([copy.deepcopy(monomer) for _ in range(n)])
    for i, m in enumerate(rosette):
        m.translate([size, 0.0, 0.0])
        m.rotate(rotation("z", 180 / n * i, degrees=True))
    return rosette


def make_oligorosette(
    rosette: GroFile, n: int, length: float, angle: float, slip: float = 0, degree: bool = True
) -> Assembly:
    """
    Stack ``n`` copies of ``rosette`` along z every ``length`` nm, each turned by
    ``angle`` more about z; a non-zero ``slip`` (nm) offsets the stack into a helix.
    """
    stack = Assembly([copy.deepcopy(rosette) for _ in range(n)])
    # distance between the supramolecular polymer axis and the rosette centre
    radius = (slip / 2) / np.sin((angle / 2) * np.pi / 180) if slip != 0 else 0.0
    for i, r in enumerate(stack):
        r.translate([0.0, 0.0, length * i])
        r.rotate(rotation("z", angle * i, degrees=degree))
        r.translate([radius * np.sin(i * angle * np.pi / 180), radius * np.cos(i * angle * np.pi / 180), 0.0])
    return stack
