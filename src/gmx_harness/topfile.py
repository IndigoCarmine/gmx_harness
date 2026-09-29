"""Prepare topology (.top) files in Python before a pipeline is built.

(``topology.py`` holds the bash/awk edits that run *during* a pipeline; this
module is for the input topology you hand to ``build_plan``.)

Typical use for an assembly of N copies of one molecule held together by
intermolecular bonds that should only act while ``-DINTER`` is defined:

    text = open("MOL.top").read()
    text = set_molecule_count(text, 216)
    text = add_conditional_include(text, "MOL_hbond.itp", define="INTER")
    open("MOL_fixed.top", "w").write(text)

``prepare_topology`` does exactly that and reproduces the usage workspace's
``fixed_top_for_interaction`` output byte for byte.
"""

import os
import re

from .safety import UnsafeNameError, validate_filename

_DEFINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AUTO_COMMENT = "; following sections are added automatically"


def _is_molecule_entry(line: str) -> bool:
    parts = line.split(";")[0].split()
    return len(parts) == 2 and parts[1].isdigit()


def set_molecule_count(text: str, count: int, names: list[str] | None = None) -> str:
    """
    Set the count of ``[ molecules ]`` entries (all of them, or only ``names``) to ``count``.
    Rewritten entries are formatted as `` NAME<pad to 16> COUNT``.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    out: list[str] = []
    section = None
    seen = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[] ").strip()
            seen = seen or section == "molecules"
        elif section == "molecules" and _is_molecule_entry(line):
            name = stripped.split()[0]
            if names is None or name in names:
                line = f" {name:<16} {count}\n"
        out.append(line)
    if not seen:
        raise ValueError("no [ molecules ] section found")
    return "".join(out)


def add_conditional_include(text: str, itp: str, define: str = "INTER") -> str:
    """Append ``#ifdef DEFINE / #include "itp" / #endif`` at the end of the topology."""
    validate_filename(itp, "itp")
    if not _DEFINE.match(define):
        raise UnsafeNameError(f"define {define!r} must be an identifier")
    if text and not text.endswith("\n"):
        text += "\n"
    return text + f'{_AUTO_COMMENT}\n#ifdef {define}\n#include "{itp}"\n#endif\n'


def prepare_topology(
    top_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    nmols: int,
    itp_name: str,
    define: str = "INTER",
) -> None:
    """Write ``top_path`` with its molecule count set to ``nmols`` and ``itp_name`` included under ``#ifdef define``."""
    with open(top_path, "r", encoding="utf8") as f:
        text = f.read()
    text = add_conditional_include(set_molecule_count(text, nmols), itp_name, define)
    with open(out_path, "w", encoding="utf8", newline="\n") as f:
        f.write(text)
