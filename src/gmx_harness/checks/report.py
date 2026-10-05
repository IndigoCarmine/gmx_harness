"""Issues, reports and waivers shared by every check.

A check never raises on a problem it finds; it adds an ``Issue`` with a stable
code (``S001``, ``P002`` ...) to a ``Report``. ``Report.enforce(waive)`` then
stops the script unless every error is waived, i.e. listed in ``waive`` as
``{"CODE": "why this is acceptable here"}``. A waiver needs a known code and a
non-empty reason, and the reason is printed and written to ``checks.json``,
so every accepted exception stays visible next to the files it concerns.
"""

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
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
    "G001": "grompp failed in the preflight (with the step's maxwarn)",
    "G002": "plumed driver failed (or wrote no COLVAR row) on the step-0 structure",
    "G003": "a CV on the step-0 (relaxed) structure differs from the value it was built with",
    "G004": "the step-0 structure CV check could not run (rot/nros not recorded upstream)",
    # after a run
    "R001": "NaN in COLVAR or the log",
    "R002": "LINCS warnings in the log",
    "R003": "GPU error (Xid / CUDA error) in the job output",
    "R004": "the biased CV barely moves (stalled)",
    "R005": "GROMACS fatal error in the log",
    "R006": "a step started but did not finish (no output.gro, no 'Finished mdrun' in the log)",
    # waivers themselves
    "W001": "a waiver was given for a code that did not occur",
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


def validate_waivers(waive: Waivers | None) -> dict[str, str]:
    """Known codes with a non-empty reason only."""
    out: dict[str, str] = {}
    for code, reason in (waive or {}).items():
        if not _CODE.match(code) or code not in CODES:
            raise ValueError(f"waiver {code!r}: unknown code (see gmx_harness.checks.CODES)")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"waiver {code}: write why the problem is acceptable (empty reason)")
        out[code] = reason.strip()
    return out


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
        return [i for i in self.issues if i.level == "error" and i.code not in w]

    def ok(self, waive: Waivers | None = None) -> bool:
        return not self.errors(waive)

    def format(self, waive: Waivers | None = None) -> str:
        w = validate_waivers(waive)
        lines = []
        for i in self.issues:
            if i.code in w:
                lines.append(f"WAIVED {i.code} {i.where + ': ' if i.where else ''}{i.message}  -- reason: {w[i.code]}")
            else:
                lines.append(str(i))
        for code in sorted(set(w) - {i.code for i in self.issues}):
            lines.append(f"WARN  W001 waiver {code} is not needed (no {code} issue); remove it")
        return "\n".join(lines) if lines else "all checks passed"

    def to_dict(self, waive: Waivers | None = None) -> dict[str, object]:
        w = validate_waivers(waive)
        return {"ok": self.ok(w), "issues": [asdict(i) for i in self.issues], "waived": w}

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
