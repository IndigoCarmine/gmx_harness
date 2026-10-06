"""Static checks of a PLUMED input (pure Python; PLUMED itself is not needed).

``check_plumed(text, natoms=..., symmetry=...)`` parses the expanded plumed.dat and
reports duplicate labels, references to undefined labels, a METAD grid that does
not match the periodicity of its CV, a SIGMA that is small for the grid, n-fold
symmetry factors that differ from the rosette size, single-atom COMs and biases on
raw single atoms, atom indices beyond the system, and missing UNITS.

``expand_template(path, layout, defines)`` expands a template with
``gmx_harness.plumed`` and runs ``check_plumed`` on it, adding P009 for defines the
template does not use (a typo in a define name otherwise changes nothing silently).

A real ``plumed driver`` run on the structure catches what static checks cannot;
that is done by a script in the workspace, not by this library.
"""

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..plumed import Layout, preprocess_file_tracked
from .report import Report

BIASES = {"METAD", "PBMETAD", "OPES_METAD", "OPES_METAD_EXPLORE", "RESTRAINT", "UPPER_WALLS", "LOWER_WALLS",
          "MOVINGRESTRAINT", "ABMD", "EXTERNAL", "BIASVALUE"}
VIRTUAL_ATOMS = {"COM", "CENTER", "GHOST", "FIXEDATOM", "CENTER_OF_MULTICOLVAR"}
GEOMETRY = {"DISTANCE", "TORSION", "ANGLE", "POSITION", "DIPOLE"}
_ATOM_KEYS = re.compile(r"^(ATOMS\d*|ATOM|GROUPA|GROUPB|GROUP|ENTITY\d+)$")
_ARG_KEYS = re.compile(r"^ARG\d*$")
_NUM_EXPR = re.compile(r"^[0-9.eE+\-*/() ]*(pi[0-9.eE+\-*/() ]*)*$")


@dataclass
class Action:
    name: str
    label: str | None
    keys: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    line: int = 0


def _num(expr: str) -> float | None:
    expr = expr.strip()
    if not expr or not _NUM_EXPR.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi}))  # noqa: S307 - digits, operators and pi only
    except Exception:
        return None


def parse_plumed(text: str) -> list[Action]:
    """Actions of a PLUMED input (comments removed, ``...`` continuation blocks joined)."""
    actions: list[Action] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        start = i + 1
        line = lines[i].split("#")[0].strip()
        i += 1
        if not line:
            continue
        if line.endswith("..."):
            body = [line[:-3]]
            while i < len(lines):
                nxt = lines[i].split("#")[0].strip()
                i += 1
                if nxt.startswith("..."):
                    break
                body.append(nxt)
            line = " ".join(body)
        tokens = line.split()
        label = None
        if tokens[0].endswith(":"):
            label = tokens.pop(0)[:-1]
        if not tokens:
            continue
        act = Action(tokens[0].upper(), label, line=start)
        for tok in tokens[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                act.keys[k.upper()] = v
            else:
                act.flags.append(tok.upper())
        if act.label is None and "LABEL" in act.keys:
            act.label = act.keys["LABEL"]
        actions.append(act)
    return actions


def _atom_items(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


def _expand_atoms(items: list[str]) -> list[int] | None:
    """Raw atom numbers of ``items`` (None if any item is not a number/range)."""
    out: list[int] = []
    for it in items:
        if re.fullmatch(r"\d+", it):
            out.append(int(it))
        elif m := re.fullmatch(r"(\d+)-(\d+)", it):
            out += range(int(m.group(1)), int(m.group(2)) + 1)
        else:
            return None
    return out


def _periodic(act: Action) -> tuple[float, float] | None | str:
    """(lo, hi) for a periodic CV, None for non-periodic, "?" if unknown."""
    if act.name == "TORSION":
        return (-math.pi, math.pi)
    p = act.keys.get("PERIODIC")
    if p is None:
        return "?" if act.name not in GEOMETRY else None
    if p.upper() == "NO":
        return None
    parts = p.split(",")
    lo, hi = (_num(parts[0]), _num(parts[1])) if len(parts) == 2 else (None, None)
    return (lo, hi) if lo is not None and hi is not None else "?"


def check_plumed(text: str, *, natoms: int | None = None, symmetry: int | None = None, where: str = "plumed.dat") -> Report:
    """Static checks P001-P008, P010 (see module docstring)."""
    rep = Report()
    actions = parse_plumed(text)
    defined: dict[str, Action] = {}
    deps: dict[str, set[str]] = {}
    raw_atoms: dict[str, list[int] | None] = {}
    for act in actions:
        w = f"{where}:{act.line}"
        refs: set[str] = set()
        for k, v in act.keys.items():
            if _ARG_KEYS.match(k):
                for item in v.split(","):
                    if not item or item.startswith("(") or "*" in item:
                        continue
                    base = item.split(".")[0]
                    if base not in defined:
                        code = "P005" if act.name in BIASES else "P002"
                        rep.error(code, f"{act.name} {k}={item}: label {base} is not defined (before this line)", w)
                    refs.add(base)
            elif _ATOM_KEYS.match(k):
                items = _atom_items(v)
                for it in items:
                    if it.startswith("@") or re.fullmatch(r"\d+(-\d+)?", it):
                        continue
                    if it not in defined:
                        rep.error("P002", f"{act.name} {k}: {it} is neither an atom number nor a defined label", w)
                    refs.add(it)
                nums = _expand_atoms(items)
                if nums and natoms is not None and max(nums) > natoms:
                    rep.error("P008", f"{act.name} {k}: atom {max(nums)} > {natoms} atoms of the system", w)
                if act.name in VIRTUAL_ATOMS and nums is not None and len(set(nums)) < 2:
                    rep.error("P007", f"{act.name} of a single atom ({v}): put the force on a group (COM of a "
                                      "rigid fragment), not on one atom", w)
                if act.label:
                    raw_atoms[act.label] = nums
        if act.name == "CUSTOM" and symmetry is not None:
            func = act.keys.get("FUNC", "")
            factors = {int(m) for m in re.findall(r"atan2\([^)]*\)/(\d+)", func)}
            factors |= {abs(int(m)) for m in re.findall(r"(?:sin|cos)\((-?\d+)\*", func)}
            for f in sorted(factors - {symmetry}):
                rep.error("P006", f"FUNC uses a {f}-fold factor but nros={symmetry}", w)
        if act.label:
            if act.label in defined:
                rep.error("P001", f"label {act.label} already defined at line {defined[act.label].line}", w)
            defined[act.label] = act
            deps[act.label] = refs

        if act.name in ("METAD", "PBMETAD"):
            _check_grid(act, defined, rep, w)

    units = [a for a in actions if a.name == "UNITS"]
    if not units:
        rep.warn("P010", "no UNITS line: values are read as nm / kJ/mol; write UNITS LENGTH=nm ENERGY=kj/mol "
                         "so the intent is explicit", where)
    else:
        u = units[0].keys
        if u.get("LENGTH", "nm").lower() != "nm" or u.get("ENERGY", "kj/mol").lower() not in ("kj/mol", "kjmol"):
            rep.warn("P010", f"UNITS {u}: SIGMA/HEIGHT/walls are then not in nm / kJ/mol", where)

    # biases acting (through any chain of CVs) on geometry of raw single atoms
    def reach(label: str, seen: set[str]) -> set[str]:
        if label in seen:
            return set()
        seen.add(label)
        out = {label}
        for d in deps.get(label, ()):
            out |= reach(d, seen)
        return out

    for act in actions:
        if act.name not in BIASES:
            continue
        roots = {it.split(".")[0] for k, v in act.keys.items() if _ARG_KEYS.match(k) for it in v.split(",") if it}
        for lab in sorted(set().union(*(reach(r, set()) for r in roots)) if roots else set()):
            a = defined.get(lab)
            if a is not None and a.name in GEOMETRY and raw_atoms.get(lab):
                rep.warn("P007", f"{act.name} biases {a.name} {lab} of raw atoms {a.keys.get('ATOMS')}: a bias on single "
                                 "atoms concentrates the force on them; prefer COMs of rigid fragments",
                         f"{where}:{act.line}")
    return rep


def _check_grid(act: Action, defined: dict[str, Action], rep: Report, where: str) -> None:
    args = [a.split(".")[0] for a in act.keys.get("ARG", "").split(",") if a]
    gmin = act.keys.get("GRID_MIN", "").split(",")
    gmax = act.keys.get("GRID_MAX", "").split(",")
    gbin = act.keys.get("GRID_BIN", "").split(",")
    gsp = act.keys.get("GRID_SPACING", "").split(",")
    sig = act.keys.get("SIGMA", "").split(",")
    for n, arg in enumerate(args):
        src = defined.get(arg)
        lo = _num(gmin[n]) if n < len(gmin) else None
        hi = _num(gmax[n]) if n < len(gmax) else None
        if src is not None and lo is not None and hi is not None:
            per = _periodic(src)
            if isinstance(per, tuple) and not (math.isclose(per[0], lo, abs_tol=1e-6)
                                               and math.isclose(per[1], hi, abs_tol=1e-6)):
                rep.error("P003", f"{arg} is periodic on [{per[0]:.4f}, {per[1]:.4f}) but the grid is "
                                  f"[{lo:.4f}, {hi:.4f}]", where)
        width = None
        if n < len(gsp) and _num(gsp[n]) is not None:
            width = _num(gsp[n])
        elif n < len(gbin) and lo is not None and hi is not None and (b := _num(gbin[n])):
            width = (hi - lo) / b
        s = _num(sig[n]) if n < len(sig) else None
        if width and s is not None and s < 2 * width:
            rep.warn("P004", f"SIGMA {s:g} of {arg} is < 2 grid bins ({width:.4g}); refine the grid", where)


def expand_template(path: str | os.PathLike[str], layout: Layout, defines: dict[str, Any] | None = None, *,
                    natoms: int | None = None, symmetry: int | None = None,
                    helpers: dict[str, Callable[..., Any]] | None = None) -> tuple[str, Report]:
    """Expand a template (``gmx_harness.plumed``) and check the result: (plumed.dat text, Report)."""
    text, unused = preprocess_file_tracked(path, layout, defines, helpers)
    rep = Report()
    for name in unused:
        rep.error("P009", f"define {name} is not used by the template (typo, or a template without that setting)",
                  os.path.basename(os.fspath(path)))
    rep.extend(check_plumed(text, natoms=natoms if natoms is not None else layout.offset + layout.nmol * layout.mol.natoms,
                            symmetry=symmetry, where=os.path.basename(os.fspath(path)) + " (expanded)"))
    return text, rep
