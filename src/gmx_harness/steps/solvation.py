"""Solvent insertion steps (MCH via insert-molecules / solvate, SPC216 water)."""

import warnings
from typing import override

import numpy as np
import numpy.typing as npt

from ..safety import validate_name
from ..scripts import NOOP_GROMPP, PY, body_script, gmx_command
from .base import Calculation, default_file_content, logger

# Solvents that ship with a .gro/.itp in gmx_harness/data, with (mass g/mol, density g/cm^3).
_SOLVENTS: dict[str, tuple[float, float]] = {"MCH": (98.186, 0.77)}
_AVOGADRO_SCALED = 602.2  # N_A / 1e21: converts nm^3 / (cm^3/mol) into a molecule count


def _check_solvent(solvent: str) -> str:
    if solvent == "H2O":
        raise ValueError("for water use SolvationSCP216 (gmx solvate with spc216.gro)")
    if solvent not in _SOLVENTS:
        raise ValueError(f"unsupported solvent {solvent!r}; available: {sorted(_SOLVENTS)}")
    return solvent


def molecules_to_fill(cell_size: npt.ArrayLike, solvent: str = "MCH", rate: float = 1.0) -> int:
    """Number of solvent molecules that fill a box of ``cell_size`` (nm) at ``rate`` x bulk density."""
    mass, density = _SOLVENTS[_check_solvent(solvent)]
    volume = float(np.prod(np.asarray(cell_size, dtype=float)))
    return int(volume / (mass / density) * rate * _AVOGADRO_SCALED)


def _tool(*args: str) -> str:
    return " ".join([PY, "top_tool.py", *args])


class Solvation(Calculation):
    """
    Insert ``nmol`` solvent molecules with ``gmx insert-molecules`` (``ntry`` attempts each),
    then add the solvent's #include and [ molecules ] line to topo.top.
    """

    def __init__(self, solvent: str = "MCH", calculation_name: str = "solvation", nmol: int = 100, ntry: int = 300):
        self.solvent = _check_solvent(solvent)
        self.calculation_name = validate_name(calculation_name)
        if nmol <= 0 or ntry <= 0:
            raise ValueError("nmol and ntry must be > 0")
        self.nmol = int(nmol)
        self.ntry = int(ntry)

    @classmethod
    def from_cell_size(
        cls, cell_size: npt.ArrayLike, name: str = "solvation", solvent: str = "MCH", rate: float = 1.0
    ) -> "Solvation":
        """Choose ``nmol`` so that a box of ``cell_size`` (nm) is filled at ``rate`` x bulk density."""
        nmol = molecules_to_fill(cell_size, solvent, rate)
        logger.info("%s: filling the cell with %d %s molecules", name, nmol, solvent)
        return cls(solvent, calculation_name=name, nmol=nmol, ntry=300)

    @override
    def generate(self) -> dict[str, str]:
        s = self.solvent
        body = "\n".join(
            [
                gmx_command(
                    "insert-molecules",
                    ["-f", "input.gro", "-ci", s, "-nmol", str(self.nmol), "-try", str(self.ntry),
                     "-o", "output.gro"],
                )
                + " 2>&1 | tee insert.log",
                _tool("add-molecules", "--from-log", "insert.log", "--resname", s, "--include-itp", f"{s}.itp"),
            ]
        )
        return {
            "mdrun.sh": body_script(body),
            f"{s}.itp": default_file_content(f"{s}.itp"),
            f"{s}.gro": default_file_content(f"{s}.gro"),
            "top_tool.py": default_file_content("top_tool.py"),
            "grommp.sh": NOOP_GROMPP,
        }


class RuntimeSolvation(Calculation):
    """
    Deprecated: like ``Solvation`` but the molecule count is computed at run time
    from the box of input.gro (``rate`` x bulk density). Prefer ``SolvationMCH``.
    """

    def __init__(self, *, solvent: str = "MCH", calculation_name: str = "solvation", rate: float = 1.0,
                 ntry: int = 300):
        warnings.warn("RuntimeSolvation is deprecated; use SolvationMCH", DeprecationWarning, stacklevel=2)
        self.solvent = _check_solvent(solvent)
        self.calculation_name = validate_name(calculation_name)
        if rate <= 0 or ntry <= 0:
            raise ValueError("rate and ntry must be > 0")
        self.rate = float(rate)
        self.ntry = int(ntry)

    @override
    def generate(self) -> dict[str, str]:
        s = self.solvent
        mass, density = _SOLVENTS[s]
        body = "\n".join(
            [
                "NMOL=$(" + _tool("solvent-count", "input.gro", "--rate", str(self.rate),
                                  "--mass", str(mass), "--density", str(density)) + ")",
                'echo "inserting $NMOL molecules"',
                f'"$GMX" insert-molecules -f input.gro -ci {s}.gro -nmol "$NMOL" -try {self.ntry}'
                " -o output.gro 2>&1 | tee insert.log",
                _tool("add-molecules", "--from-log", "insert.log", "--resname", s, "--include-itp", f"{s}.itp"),
            ]
        )
        return {
            "mdrun.sh": body_script(body),
            f"{s}.itp": default_file_content(f"{s}.itp"),
            f"{s}.gro": default_file_content(f"{s}.gro"),
            "top_tool.py": default_file_content("top_tool.py"),
            "grommp.sh": NOOP_GROMPP,
        }

    def check(self, cell_size: npt.ArrayLike) -> "RuntimeSolvation":
        n = molecules_to_fill(cell_size, self.solvent, self.rate)
        logger.info("%s: will insert about %d molecules (%d atoms)", self.name, n, n * 20)
        return self


class SolvationSCP216(Calculation):
    """Fill the box with SPC216 water (``gmx solvate -cs spc216.gro``). topo.top must already include a water model."""

    def __init__(self, calculation_name: str = "solvation"):
        self.calculation_name = validate_name(calculation_name)

    @override
    def generate(self) -> dict[str, str]:
        body = "\n".join(
            [
                gmx_command("solvate", ["-cp", "input.gro", "-cs", "spc216.gro", "-o", "output.gro", "-p", "dummy.top"]),
                _tool("add-molecules", "--from-dummy", "dummy.top"),
            ]
        )
        return {
            "dummy.top": "",
            "grommp.sh": NOOP_GROMPP,
            "top_tool.py": default_file_content("top_tool.py"),
            "mdrun.sh": body_script(body),
        }


class SolvationMCH(Calculation):
    """
    Fill the box with methylcyclohexane from a pre-equilibrated solvent box
    (``gmx solvate -cs MCH_solventbox.gro -scale``); updates topo.top.

    Attributes:
        scale: van der Waals radius scaling for gmx solvate (-scale). Smaller packs more tightly.
    """

    def __init__(self, calculation_name: str = "solvation", scale: float = 0.57):
        self.calculation_name = validate_name(calculation_name)
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.scale = float(scale)

    @override
    def generate(self) -> dict[str, str]:
        body = "\n".join(
            [
                gmx_command(
                    "solvate",
                    ["-cp", "input.gro", "-cs", "MCH_solventbox.gro", "-o", "output.gro", "-p", "dummy.top",
                     "-scale", str(self.scale)],
                ),
                _tool("add-molecules", "--from-dummy", "dummy.top"),
                _tool("include-itp", "MCH.itp"),
            ]
        )
        return {
            "dummy.top": "",
            "grommp.sh": NOOP_GROMPP,
            "top_tool.py": default_file_content("top_tool.py"),
            "mdrun.sh": body_script(body),
            "MCH.itp": default_file_content("MCH.itp"),
            "MCH_solventbox.gro": default_file_content("MCH_solventbox.gro"),
        }

    def check(self, cell_size: npt.ArrayLike) -> "SolvationMCH":
        n = molecules_to_fill(cell_size, "MCH")
        logger.info("%s: the cell holds about %d MCH molecules (%d atoms)", self.name, n, n * 20)
        return self
