"""Issues, reports and waivers shared by every check.

A check never raises on a problem it finds; it adds an ``Issue`` with a stable
code (``S001``, ``P002`` ...) and a ``where`` (the file, step or variant it is
about) to a ``Report``. ``Report.enforce(waive)`` then stops the script unless
every error is waived. A waiver is one entry of ``waive``:

- ``{"CODE@pattern": "why"}`` waives the ``CODE`` issues whose ``where`` matches
  ``pattern`` (``fnmatch``, case-sensitive, ``\\`` read as ``/``), e.g.
  ``{"S004@*fiber_rot_+10*": "periodic along z by design"}``; the same code at
  another place still stops the script. A ``where`` ending in ``:<line>``
  (PLUMED checks) is also matched without the line number, so
  ``"P007@6_md_metad/plumed.dat"`` covers the whole file. An issue with an empty
  ``where`` is never matched by a pattern.
- ``{"CODE": "why"}`` waives every ``CODE`` issue of the report (code-wide); it is
  shown as such (``for every S004``) so a broad waiver stays visible.

A waiver needs a known code, a non-empty pattern after ``@`` and a non-empty
reason. The reason is printed and written to ``checks.json`` together with
the waiver key and the ``where`` of every issue it waived, so every accepted
exception stays visible next to the files it concerns. A waiver that matches
no issue is reported as ``W001`` (warn).
"""

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal

Level = Literal["error", "warn", "info"]
Waivers = Mapping[str, str]

# Every code a check may emit, with a one-line meaning (also the table in the docs).
CODES: dict[str, str] = {
    # facts: values handed from one stage script to the next
    "F001": "a value this script assumes differs from the value the producing script recorded",
    "F002": "a value this script assumes was never recorded upstream",
    "F003": "an input changed after the facts of a file were recorded (stale result)",
    "F004": "the file has no facts (it was not made by a stage script that records them)",
    "F005": "the file changed after its facts were recorded (edited by hand?)",
    # structure: gro / top / itp / ndx
    "S001": "gro atom count differs from the molecules of the topology",
    "S002": "atom count is not a whole number of molecules",
    "S003": "[ molecules ] has several molecule types where one was expected",
    "S004": "a box edge is shorter than the minimum",
    "S005": "a periodic fiber does not close on its image (ndisk*rot not a multiple of 360/nros)",
    "S006": "a bond atom number is outside the monomer (or monomer pair)",
    "S007": "a built bond length is outside the expected range",
    "S008": "atoms of different molecules are closer than the threshold",
    "S009": "atom names/order of the system differ from the labeled monomer",
    "S010": "an index group is empty or refers to atoms outside the structure",
    "S011": "a molecule's atom count could not be determined from the topology",
    # PLUMED input
    "P001": "a PLUMED label is defined twice",
    "P002": "ARG/ATOMS refers to a label that is not defined",
    "P003": "the METAD grid does not match the periodicity of its CV",
    "P004": "METAD SIGMA is small compared with the grid spacing",
    "P005": "a bias acts on a label that is not defined",
    "P006": "the n-fold symmetry in the CV (sin/cos/atan2) differs from nros",
    "P007": "a COM/CENTER uses a single atom, or a bias acts on raw single atoms",
    "P008": "an atom index exceeds the number of atoms of the system",
    "P009": "a define passed from Python is not used by the template",
    "P010": "PLUMED UNITS missing or not nm / kJ/mol",
    # plan (pipeline)
    "L001": "PLUMED PACE/PRINT stride does not divide nsteps/nstout",
    "L002": "a step needs an index file that is not planned",
    "L003": "the first step's input.gro does not match its topo.top",
    "L004": "maxwarn > 0 (grompp warnings are ignored)",
    "L005": "strict_mdp=False (mdp validation errors are ignored)",
    "L006": "a raw shell step (unchecked command) is in the pipeline",
    # preflight (run by a person in the workspace, with GROMACS/PLUMED)
    "G001": "grompp failed in the preflight dry run (with the step's maxwarn)",
    "G002": "plumed driver failed (or wrote no COLVAR row) on the structure the step receives in the preflight dry run",
    "G003": "a CV on the structure a step receives in the preflight dry run differs from the value it was built with",
    "G004": "the preflight CV check could not run (rot/nros not recorded upstream)",
    "G005": "a step without setting.mdp (solvation etc.) or its hand-over (copy.sh) failed, or would run mdrun, "
            "in the preflight dry run",
    # after a run
    "R001": "NaN in COLVAR or the log",
    "R002": "LINCS warnings in the log",
    "R003": "GPU error (Xid / CUDA error) in the job output",
    "R004": "the biased CV barely moves (stalled)",
    "R005": "GROMACS fatal error in the log",
    "R006": "a step started but did not finish (no output.gro, no 'Finished mdrun' in the log)",
    # waivers themselves
    "W001": "a waiver matches no issue (its code did not occur, or not at the waiver's place)",
}

_CODE = re.compile(r"^[A-Z]\d{3}$")


class HarnessCheckError(RuntimeError):
    """A check found errors that are not waived (``report`` lists everything)."""

    def __init__(self, report: "Report", waive: Waivers):
        self.report = report
        super().__init__("checks failed:\n" + report.format(waive))


@dataclass(frozen=True)
class Issue:
    code: str
    level: Level
    message: str
    where: str = ""

    def __str__(self) -> str:
        return f"{self.level.upper():5} {self.code} {self.where + ': ' if self.where else ''}{self.message}"


def split_waiver(key: str) -> tuple[str, str | None]:
    """``"CODE@pattern"`` -> (``CODE``, pattern with ``/`` separators); ``"CODE"`` -> (``CODE``, None)."""
    code, at, pattern = key.partition("@")
    return code, (pattern.strip().replace("\\", "/") if at else None)


def validate_waivers(waive: Waivers | None) -> dict[str, str]:
    """
    Check the waivers and return them as ``{key: reason}`` (keys as given). A key is

    - ``"CODE@pattern"``: waives the ``CODE`` issues whose ``where`` matches ``pattern``
      (``fnmatch``, case-sensitive, ``\\`` read as ``/``; a ``where`` ``<file>:<line>`` also
      matches as ``<file>``). An issue with an empty ``where`` never matches. A waiver for a
      place is credited before a code-wide one.
    - ``"CODE"``: waives every ``CODE`` issue (code-wide; shown as ``for every CODE``).

    ``ValueError`` for an unknown code, nothing after ``@``, or an empty reason. A key that
    matches no issue is W001 (``Report.unused_waivers``).
    """
    out: dict[str, str] = {}
    for key, reason in (waive or {}).items():
        if not isinstance(key, str):
            raise ValueError(f"waiver {key!r}: the key must be a string, \"CODE\" or \"CODE@pattern\"")
        code, pattern = split_waiver(key)
        if not _CODE.match(code) or code not in CODES:
            raise ValueError(f"waiver {key!r}: unknown code {code!r} (see gmx_harness.checks.CODES; "
                             "write \"CODE\" or \"CODE@pattern\")")
        if pattern is not None and not pattern:
            raise ValueError(f"waiver {key!r}: nothing after '@' (write \"{code}@<pattern of the issue's "
                             f"place>\", or \"{code}\" for every {code})")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"waiver {key}: write why the problem is acceptable (empty reason)")
        out[key] = reason.strip()
    return out


def _where_forms(where: str) -> tuple[str, ...]:
    w = where.replace("\\", "/")
    m = re.fullmatch(r"(.+):\d+", w)  # "<file>:<line>" of the PLUMED checks: the file matches too
    return (w, m.group(1)) if m else (w,)


def waiver_matches(key: str, issue: "Issue") -> bool:
    """True if the waiver ``key`` (``"CODE"`` or ``"CODE@pattern"``) covers ``issue``."""
    code, pattern = split_waiver(key)
    if code != issue.code:
        return False
    if pattern is None:
        return True
    return bool(issue.where) and any(fnmatchcase(w, pattern) for w in _where_forms(issue.where))


def waiver_for(issue: "Issue", waive: Mapping[str, str]) -> str | None:
    """The key of ``waive`` that waives ``issue``: the first matching ``CODE@pattern``, else ``CODE``, else None."""
    placed = [k for k in waive if "@" in k and waiver_matches(k, issue)]
    if placed:
        return placed[0]
    return issue.code if issue.code in waive else None


@dataclass
class Report:
    """Issues found by one or more checks."""

    issues: list[Issue] = field(default_factory=list)

    def add(self, code: str, level: Level, message: str, where: str = "") -> None:
        if code not in CODES:
            raise ValueError(f"unknown check code {code!r}")
        self.issues.append(Issue(code, level, message, where))

    def error(self, code: str, message: str, where: str = "") -> None:
        self.add(code, "error", message, where)

    def warn(self, code: str, message: str, where: str = "") -> None:
        self.add(code, "warn", message, where)

    def info(self, code: str, message: str, where: str = "") -> None:
        self.add(code, "info", message, where)

    def extend(self, other: "Report | Iterable[Issue]") -> "Report":
        """Add issues (an identical issue already present is not repeated)."""
        for issue in other.issues if isinstance(other, Report) else other:
            if issue not in self.issues:
                self.issues.append(issue)
        return self

    def errors(self, waive: Waivers | None = None) -> list[Issue]:
        """Errors that are not waived (``waive`` is checked with ``validate_waivers``)."""
        w = validate_waivers(waive)
        return [i for i in self.issues if i.level == "error" and waiver_for(i, w) is None]

    def ok(self, waive: Waivers | None = None) -> bool:
        return not self.errors(waive)

    def unused_waivers(self, waive: Waivers | None = None) -> list[str]:
        """W001: the waiver keys that match no issue of this report (of any level)."""
        w = validate_waivers(waive)
        return [k for k in w if not any(waiver_matches(k, i) for i in self.issues)]

    def format(self, waive: Waivers | None = None) -> str:
        w = validate_waivers(waive)
        lines = []
        for i in self.issues:
            key = waiver_for(i, w)
            if key is None:
                lines.append(str(i))
                continue
            by = f'"{key}"' if "@" in key else f'"{key}" (for every {key})'
            lines.append(f"WAIVED {i.code} {i.where + ': ' if i.where else ''}{i.message}  -- waived by {by}, "
                         f"reason: {w[key]}")
        for key in self.unused_waivers(w):
            code, pattern = split_waiver(key)
            places = sorted({i.where for i in self.issues if i.code == code})
            if pattern is None or not places:
                lines.append(f"WARN  W001 waiver {key} is not needed (no {code} issue); remove it")
            else:
                at = ", ".join(p or "(no place)" for p in places)
                lines.append(f"WARN  W001 waiver {key} matches no issue ({code} occurs at: {at}); "
                             "remove it or fix the pattern")
        return "\n".join(lines) if lines else "all checks passed"

    def to_dict(self, waive: Waivers | None = None) -> dict[str, object]:
        """
        The ``checks.json`` record: issues, waivers and which issue each waiver covered.

        Keys: ``ok``; ``issues`` (code, level, message, where); ``waived`` (the waivers as given,
        ``{key: reason}``); ``waived_issues`` (one entry per waived issue: ``waiver`` key,
        ``scope`` ``"where"`` for ``CODE@pattern`` or ``"code"`` for a code-wide waiver, ``code``,
        ``where``, ``level``, ``message``, ``reason``); ``unused_waivers`` (the W001 keys).
        """
        w = validate_waivers(waive)
        applied: list[dict[str, str]] = []
        for i in self.issues:
            key = waiver_for(i, w)
            if key is not None:
                applied.append({"waiver": key, "scope": "where" if "@" in key else "code", "code": i.code,
                                "where": i.where, "level": i.level, "message": i.message, "reason": w[key]})
        return {"ok": self.ok(w), "issues": [asdict(i) for i in self.issues], "waived": w,
                "waived_issues": applied, "unused_waivers": self.unused_waivers(w)}

    def enforce(self, waive: Waivers | None = None, *, out_dir: str | os.PathLike[str] | None = None,
                name: str = "checks.json", quiet: bool = False) -> "Report":
        """
        Print the report, write it to ``out_dir/name`` if given, and raise
        ``HarnessCheckError`` if any error is not waived. Returns ``self``.
        """
        w = validate_waivers(waive)
        if not quiet:
            print(self.format(w))
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            Path(out_dir, name).write_text(json.dumps(self.to_dict(w), indent=1) + "\n", encoding="utf-8",
                                           newline="\n")
        if self.errors(w):
            raise HarnessCheckError(self, w)
        return self
