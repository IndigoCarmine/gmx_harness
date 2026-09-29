"""Read, edit and write GROMACS .gro files and residue-based index (.ndx) files.

Ported from mylibs' ``mole.gro`` without the dependency on ``mole.molecules``.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class _Rotation(Protocol):
    def apply(self, vectors: Any) -> Any: ...


class GroAtom:
    """One atom line of a .gro file. ``index`` is the atom number, ``coordinate`` is in nm."""

    def __init__(
        self,
        atom_number: int,
        atom_name: str,
        residue_name: str,
        residue_number: int,
        coordinate: npt.NDArray[np.float64],
    ) -> None:
        self.index = atom_number
        self.atom_name = atom_name
        self.residue_name = residue_name
        self.residue_number = residue_number
        self.coordinate = np.asarray(coordinate, dtype=float)

    @property
    def atom_symbol(self) -> str:
        """First character of the atom name (crude element guess)."""
        return self.atom_name[0]

    @property
    def symbol(self) -> str:
        return self.atom_symbol

    def __str__(self) -> str:
        # (i5,2a5,i5,3f8.3)
        c = self.coordinate
        return (
            f"{self.residue_number % 100000:>5}{self.residue_name:<5}{self.atom_name:>5}{self.index % 100000:>5}"
            f"{c[0]:>8.3f}{c[1]:>8.3f}{c[2]:>8.3f}"
        )

    def __repr__(self) -> str:
        return f"GroAtom({self.index}, {self.atom_name!r}, {self.residue_name!r}, {self.residue_number}, {self.coordinate!r})"

    @classmethod
    def from_gro_line(cls, line: str) -> "GroAtom":
        return cls(
            int(line[15:20]),
            line[10:15].strip(),
            line[5:10].strip(),
            int(line[:5]),
            np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])]),
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "GroAtom":
        return GroAtom(self.index, self.atom_name, self.residue_name, self.residue_number, self.coordinate.copy())


@dataclass
class GroFile:
    """A .gro structure: title, atoms and a rectangular box (nm)."""

    title: str
    atoms: list[GroAtom]
    box_x: float
    box_y: float
    box_z: float
    box_angle_x: float = 90
    box_angle_y: float = 90
    box_angle_z: float = 90

    def __len__(self) -> int:
        return len(self.atoms)

    def find_atom(self, index: int) -> list[GroAtom]:
        return [atom for atom in self.atoms if atom.index == index]

    def get_child(self, index: int) -> GroAtom:
        atoms = self.find_atom(index)
        if len(atoms) != 1:
            raise ValueError(f"expected exactly one atom with index {index}, found {len(atoms)}")
        return atoms[0]

    def get_children(self) -> list[GroAtom]:
        return self.atoms

    def generate_gro_text(self) -> list[str]:
        return [self.title, f"   {len(self.atoms)}", *(str(a) for a in self.atoms),
                f"{self.box_x} {self.box_y} {self.box_z}"]

    @classmethod
    def from_gro_text(cls, lines: list[str]) -> "GroFile":
        lines = [ln.rstrip("\r\n") for ln in lines]
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) < 3:
            raise ValueError("not a .gro file: fewer than 3 lines")
        natoms = int(lines[1].split()[0])
        atom_lines = lines[2:-1]
        if len(atom_lines) != natoms:
            raise ValueError(f"header says {natoms} atoms but {len(atom_lines)} atom lines were found")
        box = lines[-1].split()
        return cls(lines[0].strip(), [GroAtom.from_gro_line(ln) for ln in atom_lines],
                   float(box[0]), float(box[1]), float(box[2]))

    @classmethod
    def from_gro_file(cls, file_path: str) -> "GroFile":
        with open(file_path, "r") as f:
            return cls.from_gro_text(f.readlines())

    def save_gro(self, file_path: str) -> None:
        with open(file_path, "w", newline="\n") as f:
            f.write("\n".join(self.generate_gro_text()))

    def renumber(self, start: int = 1) -> None:
        for i, atom in enumerate(self.atoms):
            atom.index = i + start

    def set_residue_number(self, number: int) -> None:
        for atom in self.atoms:
            atom.residue_number = number

    def translate(self, vector: npt.ArrayLike) -> None:
        v = np.asarray(vector, dtype=float)
        for atom in self.atoms:
            atom.coordinate = atom.coordinate + v

    def rotate(self, rotation: "_Rotation | npt.NDArray[np.float64]") -> None:
        """Rotate about the origin by a scipy ``Rotation`` (anything with ``.apply``) or a 3x3 matrix."""
        if hasattr(rotation, "apply"):
            for atom in self.atoms:
                atom.coordinate = np.asarray(rotation.apply(atom.coordinate), dtype=float)
        else:
            m = np.asarray(rotation, dtype=float)
            for atom in self.atoms:
                atom.coordinate = m @ atom.coordinate

    # --------------------------------------------------------------- xyz I/O

    def generate_xyz_text(self) -> list[str]:
        nm_to_angstrom = 10
        out = [f"{len(self.atoms)}", self.title]
        for a in self.atoms:
            x, y, z = a.coordinate * nm_to_angstrom
            out.append(f"{a.atom_symbol} {x:12.6f} {y:12.6f} {z:12.6f}")
        return out

    def save_xyz(self, file_path: str) -> None:
        with open(file_path, "w", newline="\n") as f:
            f.write("\n".join(self.generate_xyz_text()))

    def load_xyz_text(self, data: list[str], multiple_molecules: bool = False) -> None:
        """Take coordinates from xyz text (Angstrom). With ``multiple_molecules`` the atoms are replicated."""
        angstrom_to_nm = 0.1
        atomnum = int(data[0].strip()) if data[0].strip().isnumeric() else 0

        def coord(line: str) -> npt.NDArray[np.float64]:
            parts = line.split()
            return np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * angstrom_to_nm

        if not multiple_molecules:
            if atomnum != len(self.atoms):
                raise ValueError("number of atoms in gro file and xyz file are not equal")
            for i in range(atomnum):
                self.atoms[i].coordinate = coord(data[i + 2])
            return
        n = len(self.atoms)
        if atomnum % n != 0:
            raise ValueError("number of atoms in xyz file is not a multiple of the number of atoms in the gro file")
        atoms: list[GroAtom] = []
        for m in range(atomnum // n):
            for i in range(n):
                atom = deepcopy(self.atoms[i])
                atom.index = atom.index + m * n
                atom.coordinate = coord(data[m * n + i + 2])
                atoms.append(atom)
        self.atoms = atoms

    def load_xyz_file(self, file_path: str, multiple_molecules: bool = False) -> None:
        with open(file_path, "r") as f:
            self.load_xyz_text(f.readlines(), multiple_molecules)

    # --------------------------------------------------------------- index

    def generate_ndx_text(self) -> str:
        """One ``[ Mol<residue number> ]`` group per residue, 10 indices per line (same text as mylibs)."""
        out: list[str] = []
        last_residue = 0
        line_length = 0
        for atom in self.atoms:
            if atom.residue_number != last_residue:
                out.append(f"\n\n[ Mol{atom.residue_number} ]\n")
                last_residue = atom.residue_number
                line_length = 0
            out.append(f"{atom.index} ")
            line_length += 1
            if line_length >= 10:  # split lines after 10 indices
                out.append("\n")
                line_length = 0
        return "".join(out)

    def generate_ndx(self, path: str) -> None:
        """Write ``generate_ndx_text()`` to ``path`` (``.ndx`` is appended if missing)."""
        if not path.endswith(".ndx"):
            path = path + ".ndx"
        with open(path, "w", newline="\n") as f:
            f.write(self.generate_ndx_text())
