"""Steps that edit structure/topology files instead of running a simulation.

The typed steps (``RemoveResidue``, ``ResizeBox``, ``AddFiles``) cover the
usual needs with validated arguments. ``RawShellStep`` is the escape hatch
for anything else; it has to be enabled explicitly with ``allow_unsafe=True``
because its command is written into the script unchecked.
"""

import warnings
from typing import override

from ..safety import UnsafeOperationError, validate_filename, validate_name, validate_resname
from .. import topology
from ..scripts import NOOP_GROMPP, body_script, gmx_command
from .base import Calculation


def _remove_residue_lines(resname: str, source: str = "input.gro") -> list[str]:
    ndx = f"without{resname}.ndx"
    return [
        gmx_command("make_ndx", ["-f", source, "-o", ndx], stdin=f"!r{resname}\nq"),
        gmx_command("trjconv", ["-f", source, "-s", source, "-o", "output.gro", "-n", ndx], stdin=f"!{resname}"),
        topology.remove_molecule(resname),
    ]


class RemoveResidue(Calculation):
    """
    Remove every residue named ``resname`` from the structure and from topo.top
    (its [ molecules ] entry and ``#include "<resname>.itp"``). Nothing else in
    topo.top is touched.
    """

    def __init__(self, calculation_name: str, resname: str = "MCH"):
        self.calculation_name = validate_name(calculation_name)
        self.resname = validate_resname(resname)

    @override
    def generate(self) -> dict[str, str]:
        return {
            "grommp.sh": NOOP_GROMPP,
            "mdrun.sh": body_script("\n".join(_remove_residue_lines(self.resname))),
        }


class ResizeBox(Calculation):
    """
    Set the box to ``x`` x ``y`` x ``z`` nm (``gmx editconf -box``). If
    ``remove_resname`` is given (default "MCH", like mylibs' cell_resizing),
    that residue is removed first.
    """

    def __init__(self, calculation_name: str, x: float, y: float, z: float, remove_resname: str | None = "MCH"):
        self.calculation_name = validate_name(calculation_name)
        for v in (x, y, z):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(f"box lengths must be positive numbers, got {v!r}")
        # kept as given (5 stays "5" on the command line, like mylibs)
        self.x, self.y, self.z = x, y, z
        self.remove_resname = validate_resname(remove_resname) if remove_resname is not None else None

    @override
    def generate(self) -> dict[str, str]:
        box = [str(self.x), str(self.y), str(self.z)]
        files = {"grommp.sh": NOOP_GROMPP}
        if self.remove_resname is not None:
            lines = _remove_residue_lines(self.remove_resname)
            lines.append(gmx_command("editconf", ["-f", "output.gro", "-o", "output.gro", "-box", *box], stdin="1"))
        else:
            lines = [gmx_command("editconf", ["-f", "input.gro", "-o", "output.gro", "-box", *box], stdin="1")]
        files["mdrun.sh"] = body_script("\n".join(lines))
        return files


class AddFiles(Calculation):
    """
    Put extra files (e.g. an .itp or an index file) into the pipeline without
    changing the structure: ``input.gro`` is passed through as ``output.gro``.
    .top/.itp files are carried on to the following steps.
    """

    def __init__(self, calculation_name: str, files: dict[str, str]):
        self.calculation_name = validate_name(calculation_name)
        for fname in files:
            validate_filename(fname)
            if fname in ("run.sh", "copy.sh", "grommp.sh", "mdrun.sh", "input.gro", "output.gro"):
                raise ValueError(f"{fname} is reserved")
        self.files = dict(files)

    @override
    def generate(self) -> dict[str, str]:
        return {
            **self.files,
            "grommp.sh": NOOP_GROMPP,
            "mdrun.sh": "#!/bin/bash\nset -e\ncp input.gro output.gro\n",
        }


class RawShellStep(Calculation):
    """
    Escape hatch: run an arbitrary bash ``command`` (``"$GMX"`` is the GROMACS command).
    The command is NOT validated, so it must be enabled with ``allow_unsafe=True``.
    It is responsible for writing output.gro. ``allow_unsafe`` is never saved
    to JSON; ``load_json`` needs ``allow_unsafe=True`` again.
    """

    def __init__(self, calculation_name: str, command: str, *, allow_unsafe: bool = False):
        if not allow_unsafe:
            raise UnsafeOperationError(
                f"{type(self).__name__}({calculation_name!r}) runs an unchecked shell command; "
                "pass allow_unsafe=True if that is intended, or use RemoveResidue / ResizeBox / AddFiles."
            )
        self.calculation_name = validate_name(calculation_name)
        self.command = command

    @override
    def generate(self) -> dict[str, str]:
        body = "# WARNING: user-supplied command below is not validated by gmx_harness\n" + self.command
        return {"grommp.sh": NOOP_GROMPP, "mdrun.sh": body_script(body)}


class FileControl(RawShellStep):
    """Deprecated name of ``RawShellStep`` (mylibs). Its helpers return the typed steps."""

    def __init__(self, calculation_name: str, command: str, *, allow_unsafe: bool = False):
        warnings.warn("FileControl is deprecated; use RawShellStep or a typed step", DeprecationWarning, stacklevel=2)
        super().__init__(calculation_name, command, allow_unsafe=allow_unsafe)

    @classmethod
    def remove_MCH(cls, name: str) -> RemoveResidue:
        return RemoveResidue(name, "MCH")

    @classmethod
    def cell_resizing(cls, name: str, x: float, y: float, z: float) -> ResizeBox:
        return ResizeBox(name, x, y, z, remove_resname="MCH")
