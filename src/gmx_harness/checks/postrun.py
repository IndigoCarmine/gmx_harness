"""Checks after a run: read output.log / run.out / job output / COLVAR of a step directory."""

import math
import os
import re
from dataclasses import replace
from pathlib import Path

from .report import Report

_LINCS = re.compile(r"LINCS WARNING|SETTLE warning|Step \d+, time .* LINCS", re.I)
_FATAL = re.compile(r"^Fatal error", re.M)
_GPU = re.compile(r"\bXid\b|CUDA error|\bcudaError\w+|GPU has fallen off the bus")
_NAN = re.compile(r"(?<![A-Za-z])(nan|-nan|inf)(?![A-Za-z])", re.I)
# an mdp/tpr parameter line of the log header ("   epsilon-rf   = inf", "   ref-t:  300"): its value is
# a setting, not a computed number (energies are tables of numbers; "Potential Energy  = ..." has a space)
_PARAM = re.compile(r"^\s*[A-Za-z][\w\-\[\]().]*\s*[=:]")
# job output in a system directory: Gromacs.sbatch (sbatch_%x.log), slurmapi (<name>-<jobid>.out/.err)
_JOB_OUTPUT = ("sbatch_*.log", "slurm-*.out", "*-[0-9]*.out", "*-[0-9]*.err")


def read_colvar(path: str | os.PathLike[str]) -> tuple[list[str], list[list[float]]]:
    """(FIELDS, rows) of a PLUMED COLVAR file."""
    fields: list[str] = []
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#! FIELDS"):
            fields = line.split()[2:]
        elif line and not line.startswith("#"):
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                continue
    return fields, rows


def _job_errors(name: str, text: str, rep: Report, where: str) -> None:
    """R005 fatal error, R003 GPU failure in one log / job output."""
    if _FATAL.search(text):
        rep.error("R005", f"{name}: GROMACS fatal error", where)
    if _GPU.search(text):
        rep.error("R003", f"{name}: GPU error (Xid/CUDA; the GPU failed, results after it are suspect)", where)


def check_step(step_dir: str | os.PathLike[str], *, cv: str | None = "theta", min_span: float = 0.05,
               tail: int = 2000) -> Report:
    """
    R001 NaN, R002 LINCS warnings, R003 GPU error (Xid/CUDA), R005 fatal error, R006 started
    but not finished (output.log without output.gro or "Finished mdrun"), R004 the biased CV
    ``cv`` spans less than ``min_span`` over the last ``tail`` COLVAR rows (stalled).
    """
    d = Path(step_dir)
    rep = Report()
    w = d.name
    for name in ("output.log", "run.out", *sorted(p.name for p in d.glob("slurm-*.out"))):
        p = d / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        _job_errors(name, text, rep, w)
        n = len(_LINCS.findall(text))
        if n:
            rep.warn("R002", f"{name}: {n} LINCS/SETTLE warnings", w)
        if name != "output.log":
            continue
        bad = [ln for ln in text.splitlines() if _NAN.search(ln) and not _PARAM.match(ln)]
        if bad:
            rep.error("R001", f"{name}: NaN/inf ({bad[0].strip()[:80]})", w)
        if not (d / "output.gro").exists() and "Finished mdrun" not in text:
            rep.error("R006", "output.log exists but no output.gro and no 'Finished mdrun' in the log "
                              "(timed out, crashed, or still running)", w)
    colvar = d / "COLVAR"
    if colvar.is_file():
        fields, rows = read_colvar(colvar)
        if any(any(math.isnan(v) or math.isinf(v) for v in r) for r in rows):
            rep.error("R001", "COLVAR has NaN/inf", w)
        if cv and cv in fields and rows:
            k = fields.index(cv)
            vals = [r[k] for r in rows[-tail:] if len(r) > k]
            if len(vals) >= 10 and max(vals) - min(vals) < min_span:
                rep.warn("R004", f"{cv} spans only {max(vals) - min(vals):.4f} over the last {len(vals)} rows "
                                 "(stalled? check HEIGHT/SIGMA and the CV)", w)
    return rep


def check_tree(root: str | os.PathLike[str], *, cv: str | None = "theta", min_span: float = 0.05,
               tail: int = 2000) -> Report:
    """
    ``check_step`` for every ``<system>/<i>_<step>`` below ``root`` that has run, plus R003/R005
    in the job output of each ``<system>`` directory (``sbatch_*.log``, ``slurm-*.out``,
    ``<name>-<jobid>.out`` / ``.err``).
    """
    rep = Report()
    for system in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        outs = sorted({p for pat in _JOB_OUTPUT for p in system.glob(pat) if p.is_file()})
        for p in outs:
            _job_errors(p.name, p.read_text(encoding="utf-8", errors="replace"), rep, system.name)
    for step in sorted(Path(root).glob("*/*_*")):
        if step.is_dir() and step.name.split("_")[0].isdigit() and any(step.glob("output.*")):
            sub = check_step(step, cv=cv, min_span=min_span, tail=tail)
            rep.extend(replace(i, where=f"{step.parent.name}/{i.where}") for i in sub.issues)
    return rep
