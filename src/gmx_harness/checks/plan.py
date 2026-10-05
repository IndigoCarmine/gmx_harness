"""Checks of a planned pipeline (``Plan``), run by ``build_plan(checks=True)``."""

import shlex
from typing import TYPE_CHECKING

from ..io.gro import GroFile
from .plumed import check_plumed, parse_plumed
from .report import Report
from .structure import check_topology

if TYPE_CHECKING:
    from ..pipeline import Plan


def check_plan(plan: "Plan") -> Report:
    """L001-L006 plus S001/P-checks on the planned files (see ``gmx_harness.checks.CODES``)."""
    rep = Report()
    files = {f.relpath: f.content for f in plan.files}
    first = plan.steps[0].dirname
    step0 = {rel.split("/", 1)[1]: c.decode("utf-8", errors="replace") for rel, c in files.items()
             if rel.startswith(first + "/")}

    natoms: int | None = None
    if "input.gro" in step0:
        try:
            gro = GroFile.from_gro_text(step0["input.gro"].splitlines())
            natoms = len(gro)
        except ValueError as e:
            rep.error("L003", f"input.gro cannot be read: {e}", first)
        if natoms is not None and "topo.top" in step0:
            sub = check_topology(natoms, step0["topo.top"], files=step0, where=f"{first}/topo.top")
            for i in sub.issues:
                code = "L003" if i.code == "S001" else i.code
                rep.add(code, i.level, i.message, i.where)

    carried_ndx = "index.ndx" in step0
    for s in plan.steps:
        c = s.calculation
        name = s.dirname
        maxwarn = int(getattr(c, "maxwarn", 0) or 0)
        if maxwarn > 0:
            rep.error("L004", f"maxwarn={maxwarn} (grompp warnings are ignored); fix the warning or waive L004 "
                              "with the warning text as reason", name)
        if getattr(c, "strict_mdp", True) is False:
            rep.error("L005", "strict_mdp=False: mdp validation errors are only logged", name)
        if type(c).__name__ in ("RawShellStep", "FileControl"):
            rep.error("L006", "unchecked shell command step (allow_unsafe=True)", name)
        grompp = files.get(f"{name}/grommp.sh", b"").decode()
        argv = next((shlex.split(ln) for ln in grompp.splitlines() if " grompp " in ln), [])
        if "-n" in argv[:-1]:
            idx = argv[argv.index("-n") + 1]
            if f"{name}/{idx}" not in files and not (carried_ndx and idx == "index.ndx"):
                rep.error("L002", f"grompp uses -n {idx} but no {idx} is planned for this step", name)

        plumed = getattr(c, "plumed", None) or (getattr(c, "extra_files", {}) or {}).get("plumed.dat")
        if plumed:
            rep.extend(check_plumed(plumed, natoms=None, where=f"{name}/plumed.dat"))
            nsteps = int(getattr(c, "nsteps", 0) or 0)
            nstout = int(getattr(c, "nstout", 0) or 0)
            for act in parse_plumed(plumed):
                stride_key = "PACE" if act.name in ("METAD", "PBMETAD") else "STRIDE" if act.name == "PRINT" else None
                if stride_key is None or stride_key not in act.keys or not act.keys[stride_key].isdigit():
                    continue
                stride = int(act.keys[stride_key])
                if stride <= 0 or (nsteps and nsteps % stride):
                    rep.warn("L001", f"{act.name} {stride_key}={stride} does not divide nsteps={nsteps}", name)
                if act.name == "PRINT" and nstout and nstout % stride:
                    rep.warn("L001", f"PRINT STRIDE={stride} does not divide nstout={nstout}: COLVAR rows and "
                                     "trajectory frames do not line up", name)
    return rep
