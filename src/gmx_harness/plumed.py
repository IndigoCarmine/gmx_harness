"""Preprocess a plumed.dat template with label-based atom selections.

A template is ordinary PLUMED input plus a C-preprocessor-like layer:

  #define NAME expr          named constant (Python expression); overridable from Python
  #for v in a..b             inclusive loop (bounds are expressions), nestable
  #endfor
  #include "file"            path relative to the including template (must stay below it)
  {expr}                     inline expression (defines, loop variables, helpers)
  @sel(...)                  sugar for {sel(...)}; any helper can be called as @name(...)
  {{ / }}                    literal braces

Other lines starting with ``#`` are kept as PLUMED comments.

Atoms are selected by the fragment labels of a labeled monomer .gro (the
residue-name column, e.g. LNK, PNP, CHL, BAR, TDP, ALK). The system is ``nmol``
copies of that monomer in index order, grouped into disks of ``nros``:

  sel(disk=None, mol=None, res=None, name=None, heavy=False) -> "4401-4449,4570-..."

``disk`` / ``mol`` (position in the disk) take an int, a list, or None (= all);
``res`` / ``name`` take "BAR,CHL,PNP" or a list. ``join(fmt, n, sep=",")`` formats
``fmt`` with ``{i}`` for i in range(n) (or over an iterable) and joins the results.

Expressions are evaluated with a small set of builtins and are checked before
evaluation: dunder names/attributes, lambdas and imports are rejected, so a
template cannot reach Python internals.

Example::

    layout = Layout(MoleculeLabels.from_gro("MOL_labeled.gro"), nmol=60, nros=6)
    text = preprocess_file("metad_twist.plumed.in", layout, {"NDISK": 10, "BIAS_IFACE": 4})
    step = MD(..., mdrun_args=["-plumed", "plumed.dat"], extra_files={"plumed.dat": text})

Ported from the ``plumed_pp.py`` of the usage workspace; the output text is unchanged.
"""

import ast
import builtins
import dataclasses
import math
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .io.gro import GroFile

_DIRECTIVE = re.compile(r"#(define|for|endfor|include)\b(.*)$")  # no space: "# for ..." stays a comment
_FOR = re.compile(r"(\w+)\s+in\s+(.+?)\.\.(.+)$")


class PreprocessError(ValueError):
    """A template could not be expanded (the message names the template line)."""


@dataclasses.dataclass(frozen=True)
class MoleculeLabels:
    """Per-atom fragment label (residue-name column) and atom name of one monomer, in gro order."""

    labels: tuple[str, ...]
    names: tuple[str, ...]
    masses: tuple[float, ...] | None = None  # from the topology; makes ``heavy`` exact (else: name starts with H)

    @classmethod
    def from_gro(cls, path: str | os.PathLike[str]) -> "MoleculeLabels":
        with open(path) as f:
            lines = f.read().splitlines()
        natoms = int(lines[1].split()[0])
        atoms = lines[2 : 2 + natoms]
        return cls(tuple(a[5:10].strip() for a in atoms), tuple(a[10:15].strip() for a in atoms))

    @classmethod
    def from_grofile(cls, gro: GroFile) -> "MoleculeLabels":
        return cls(tuple(a.residue_name for a in gro.atoms), tuple(a.atom_name for a in gro.atoms))

    @property
    def natoms(self) -> int:
        return len(self.labels)

    def heavy(self, i: int) -> bool:
        if self.masses is not None:
            return self.masses[i] > 1.5
        return not self.names[i].upper().startswith("H")

    def with_masses_from_top(self, top_text: str) -> "MoleculeLabels":
        """Copy with the masses of the first ``[ atoms ]`` block of a topology (same atom order)."""
        masses: list[float] = []
        section = ""
        for raw in top_text.splitlines():
            line = raw.split(";")[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                if section == "atoms" and masses:
                    break
                section = line.strip("[] ").strip().lower()
                continue
            if section == "atoms":
                parts = line.split()
                try:
                    masses.append(float(parts[7]))
                except (IndexError, ValueError):
                    raise PreprocessError(f"topology [ atoms ] line has no mass (8th column): {raw.strip()!r}") from None
        if len(masses) != self.natoms:
            raise PreprocessError(f"topology [ atoms ] has {len(masses)} atoms, the labeled monomer {self.natoms}")
        return dataclasses.replace(self, masses=tuple(masses))


def _as_set(value: Any, kind: str, allowed: Iterable[Any]) -> set[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    elif isinstance(value, int):
        value = [value]
    chosen = set(value)
    unknown = chosen - set(allowed)
    if unknown:
        raise PreprocessError(f"unknown {kind}: {sorted(unknown, key=str)} (have {sorted(set(allowed), key=str)})")
    return chosen


def compress(indices: list[int]) -> str:
    """[1,2,3,7,9,10] -> '1-3,7,9-10' (PLUMED atom ranges)."""
    out: list[str] = []
    start: int | None = None
    prev = 0
    for a in indices:
        if start is None:
            start = prev = a
        elif a == prev + 1:
            prev = a
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = a
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(out)


@dataclasses.dataclass
class Layout:
    """``nmol`` identical monomers in index order; molecule j is in disk j // nros."""

    mol: MoleculeLabels
    nmol: int
    nros: int
    offset: int = 0  # atoms before the first monomer

    def indices(self, disk: Any = None, mol: Any = None, res: Any = None, name: Any = None,
                heavy: bool = False) -> list[int]:
        """1-based atom indices of the selection (see module docstring)."""
        ndisk = math.ceil(self.nmol / self.nros)
        disks = _as_set(disk, "disk", range(ndisk))
        pos = _as_set(mol, "mol", range(self.nros))
        labels = _as_set(res, "label", self.mol.labels)
        names = _as_set(name, "atom name", self.mol.names)
        local = [i for i in range(self.mol.natoms)
                 if (labels is None or self.mol.labels[i] in labels)
                 and (names is None or self.mol.names[i] in names)
                 and (not heavy or self.mol.heavy(i))]
        if not local:
            raise PreprocessError(f"empty selection res={res!r} name={name!r} heavy={heavy}")
        out: list[int] = []
        for j in range(self.nmol):
            if (disks is None or j // self.nros in disks) and (pos is None or j % self.nros in pos):
                base = self.offset + j * self.mol.natoms + 1
                out += [base + i for i in local]
        return out

    def sel(self, **kw: Any) -> str:
        """``indices(**kw)`` in PLUMED range notation."""
        return compress(self.indices(**kw))


def join(fmt: str, n: int | Iterable[Any], sep: str = ",") -> str:
    """``fmt.format(i=i)`` for i in range(n) (or over an iterable), joined with ``sep``."""
    items = range(n) if isinstance(n, int) else n
    return sep.join(fmt.format(i=i) for i in items)


_SAFE_BUILTINS = {k: getattr(builtins, k)
                  for k in ("range", "len", "str", "int", "float", "abs", "min", "max", "sum", "round")}
_FORBIDDEN_NODES = (ast.Lambda, ast.Import, ast.ImportFrom, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)


def _check(expr: str, where: str) -> None:
    """Reject expressions that could reach Python internals (dunders, lambdas, imports)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise PreprocessError(f"{where}: cannot parse {expr!r}: {e.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise PreprocessError(f"{where}: {type(node).__name__} is not allowed in {expr!r}")
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name is not None and name.startswith("_"):
            raise PreprocessError(f"{where}: names starting with '_' are not allowed in {expr!r}")


def _eval(expr: str, env: dict[str, Any], where: str) -> Any:
    _check(expr, where)
    try:
        return eval(expr, {"__builtins__": _SAFE_BUILTINS, "pi": math.pi}, env)  # noqa: S307 - checked above
    except Exception as e:  # report with the template position
        raise PreprocessError(f"{where}: cannot evaluate {expr!r}: {e}") from e


def _scan_to(text: str, i: int, open_: str, close: str) -> int:
    """Index of the ``close`` matching ``text[i] == open_``, skipping string literals."""
    depth, quote = 0, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise PreprocessError(f"unbalanced {open_!r}")


def _expand_line(line: str, env: dict[str, Any], where: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if line.startswith("{{", i) or line.startswith("}}", i):
            out.append(c)
            i += 2
        elif c == "{":
            try:
                j = _scan_to(line, i, "{", "}")
            except PreprocessError as e:
                raise PreprocessError(f"{where}: {e}") from None
            out.append(str(_eval(line[i + 1 : j], env, where)))
            i = j + 1
        elif c == "}":
            raise PreprocessError(f"{where}: stray '}}' (use '}}}}' for a literal brace)")
        elif c == "@" and (m := re.match(r"@([A-Za-z_]\w*)\(", line[i:])):
            try:
                j = _scan_to(line, i + m.end() - 1, "(", ")")
            except PreprocessError as e:
                raise PreprocessError(f"{where}: {e}") from None
            out.append(str(_eval(line[i + 1 : j + 1], env, where)))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _read(path: str) -> list[tuple[str, str]]:
    with open(path) as f:
        return [(f"{path}:{n}", ln) for n, ln in enumerate(f.read().splitlines(), 1)]


_MISSING = object()


class _Env(dict[str, Any]):
    """
    Evaluation namespace that remembers which names expressions looked up (for unused defines).
    Only a lookup counts: a ``#define`` of the same name in the template does not, so a define
    passed from Python that the template declares but never uses is still reported.
    """

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.used: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.used.add(key)
        return super().__getitem__(key)


def _run(lines: list[tuple[str, str]], env: dict[str, Any], fixed: set[str], base: str, root: str,
         out: list[str]) -> None:
    i = 0
    while i < len(lines):
        where, raw = lines[i]
        m = _DIRECTIVE.match(raw.strip())
        if not m:
            out.append(_expand_line(raw, env, where))
            i += 1
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind == "define":
            dm = re.match(r"([A-Za-z_]\w*)\s+(.+)$", rest)
            if not dm:
                raise PreprocessError(f"{where}: bad #define {rest!r}")
            if dm.group(1) not in fixed:
                env[dm.group(1)] = _eval(dm.group(2), env, where)
            i += 1
        elif kind == "include":
            path = os.path.normpath(os.path.join(base, str(_eval(rest, env, where))))
            root_abs, path_abs = Path(root).resolve(), Path(path).resolve()
            if path_abs != root_abs and root_abs not in path_abs.parents:
                raise PreprocessError(f"{where}: #include {rest} leaves the template directory")
            _run(_read(path), env, fixed, os.path.dirname(path), root, out)
            i += 1
        elif kind == "endfor":
            raise PreprocessError(f"{where}: #endfor without #for")
        else:  # for
            fm = _FOR.match(rest)
            if not fm:
                raise PreprocessError(f"{where}: bad #for {rest!r} (want: #for v in a..b)")
            depth, j = 1, i + 1
            while j < len(lines):
                dj = _DIRECTIVE.match(lines[j][1].strip())
                if dj and dj.group(1) == "for":
                    depth += 1
                elif dj and dj.group(1) == "endfor":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            else:
                raise PreprocessError(f"{where}: #for without #endfor")
            var = fm.group(1)
            lo, hi = (int(_eval(fm.group(g), env, where)) for g in (2, 3))
            saved = env.get(var, _MISSING)
            for v in range(lo, hi + 1):
                env[var] = v
                _run(lines[i + 1 : j], env, fixed, base, root, out)
            if saved is _MISSING:
                env.pop(var, None)  # absent when the loop ran zero times
            else:
                env[var] = saved
            i = j + 1


def preprocess(text: str, layout: Layout, defines: dict[str, Any] | None = None, source: str = "<template>",
               base: str = ".", helpers: dict[str, Callable[..., Any]] | None = None) -> str:
    """
    Expand a template (see module docstring) into plain PLUMED input.

    ``defines`` override the template's ``#define`` values; ``helpers`` add callables
    usable as ``{name(...)}`` / ``@name(...)`` (``sel`` and ``join`` are always there).
    ``#include`` paths must stay inside ``base``.
    """
    return preprocess_tracked(text, layout, defines, source, base, helpers)[0]


def preprocess_tracked(text: str, layout: Layout, defines: dict[str, Any] | None = None,
                       source: str = "<template>", base: str = ".",
                       helpers: dict[str, Callable[..., Any]] | None = None) -> tuple[str, list[str]]:
    """``preprocess``, also returning the ``defines`` that no expression of the template looks up."""
    defines = dict(defines or {})
    for key in [*defines, *(helpers or {})]:
        if key.startswith("_"):
            raise PreprocessError(f"define/helper names may not start with '_': {key!r}")
    env = _Env({"sel": layout.sel, "join": join, **(helpers or {}), **defines})
    lines = [(f"{source}:{n}", ln) for n, ln in enumerate(text.splitlines(), 1)]
    out = [f"# generated by plumed_pp from {os.path.basename(source)} -- edit the template, not this file"]
    _run(lines, env, set(defines), base, base, out)
    return "\n".join(out) + "\n", sorted(set(defines) - env.used)


def preprocess_file(path: str | os.PathLike[str], layout: Layout, defines: dict[str, Any] | None = None,
                    helpers: dict[str, Callable[..., Any]] | None = None) -> str:
    """``preprocess`` a template file; ``#include`` is resolved relative to (and confined to) its directory."""
    return preprocess_file_tracked(path, layout, defines, helpers)[0]


def preprocess_file_tracked(path: str | os.PathLike[str], layout: Layout, defines: dict[str, Any] | None = None,
                            helpers: dict[str, Callable[..., Any]] | None = None) -> tuple[str, list[str]]:
    """``preprocess_file``, also returning the unused ``defines`` (see ``preprocess_tracked``)."""
    path = os.fspath(path)
    with open(path) as f:
        text = f.read()
    return preprocess_tracked(text, layout, defines, source=path, base=os.path.dirname(os.path.abspath(path)),
                              helpers=helpers)
