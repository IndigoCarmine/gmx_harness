"""Build molecular assemblies from .gro structures (ported from mylibs' ``gromacs.pre_coordinator`` / ``rosette_maker``).

Everything works on ``GroFile`` objects and plain numpy rotation matrices:

    mono = GroFile.from_gro_file("MOL.gro")
    precoordinate2(mono, top=1, aromatic_nh=2, aromatic_o=3)      # orient the monomer
    rosette = make_rosette2(mono, n=6, size=0.35).to_gro()      # one ring
    fiber = make_oligorosette(rosette, n=10, length=0.35, angle=10)  # a stack of rings
    fiber.to_gro(box=(10, 10, 3.5), renumber=True).save_gro("fiber.gro")

Atom arguments of the pre-coordination functions are GROMACS atom numbers
(the index column of the .gro file, usually 1-based).
"""

import copy
from collections.abc import Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt

from .io.gro import GroAtom, GroFile

Matrix = npt.NDArray[np.float64]
Axis = Literal["x", "y", "z"]


def rotation(axis: Axis, angle: float, degrees: bool = False) -> Matrix:
    """3x3 matrix of a right-handed rotation by ``angle`` about ``axis``."""
    a = np.deg2rad(angle) if degrees else float(angle)
    c, s = np.cos(a), np.sin(a)
    match axis.lower():
        case "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        case "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        case "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"axis must be x, y or z, got {axis!r}")


def align(vector: npt.ArrayLike, target: npt.ArrayLike) -> Matrix:
    """Smallest rotation that turns the direction of ``vector`` into that of ``target``."""
    v = np.asarray(vector, dtype=float)
    t = np.asarray(target, dtype=float)
    v, t = v / np.linalg.norm(v), t / np.linalg.norm(t)
    axis = np.cross(v, t)
    s, c = np.linalg.norm(axis), float(np.dot(v, t))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: turn 180 degrees about any axis perpendicular to v
        perp = np.cross(v, [1.0, 0.0, 0.0] if abs(v[0]) < 0.9 else [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    result: Matrix = np.eye(3) + s * kx + (1 - c) * (kx @ kx)
    return result


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
