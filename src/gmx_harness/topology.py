"""Bash + POSIX awk snippets that edit ``topo.top`` while a generated script runs.

Solvation steps only learn how many molecules were added after GROMACS has
run, so the topology has to be updated at run time on the machine that runs
the pipeline. These snippets do that with nothing but bash and awk (no Python
needed there). The edits match what mylibs' helper scripts did:

- a new ``RES N`` line goes just before the first line after ``[ molecules ]``
  that is not blank, a comment or a molecule entry (usually the next section,
  e.g. ``[ intermolecular_interactions ]``), or at the end of the file;
- ``#include "RES.itp"`` goes before ``[ system ]`` (skipped if already there);
- the previous topology is kept as ``topo_old.top``.

Values are passed to awk through the environment, never spliced into the
awk program, and resnames are validated (``safety.validate_resname``).
"""

import shlex

from .safety import validate_filename, validate_resname

_FUNCS = r"""function clean(s,   t) { t = s; sub(/;.*/, "", t); gsub(/\r/, "", t); return t }
function ismol(s,   n, a) { n = split(clean(s), a); return (n == 2 && a[2] ~ /^[0-9]+$/) }
function incname(s,   t) { t = clean(s); if (t !~ /^[ \t]*#include[ \t]/) return ""; sub(/^[^"]*"/, "", t); sub(/".*$/, "", t); return t }
"""

_EDIT = (
    _FUNCS
    + r"""BEGIN { inc = ENVIRON["TOP_INCLUDE"]; line = ENVIRON["TOP_LINE"] }
NR == FNR { if (inc != "" && incname($0) == inc) have = 1; next }
{
    if (inc != "" && !have && !incdone && $0 ~ /^[ \t]*\[[ \t]*system[ \t]*\]/) { print "#include \"" inc "\""; incdone = 1 }
    if (line != "" && inmol && !done) { t = clean($0); gsub(/[ \t]/, "", t); if (t != "" && !ismol($0)) { print line; done = 1 } }
    if ($0 ~ /^[ \t]*\[[ \t]*molecules[ \t]*\]/) inmol = 1
    print
}
END {
    if (inc != "" && !have && !incdone) { print "gmx_harness: no [ system ] section in the topology" > "/dev/stderr"; exit 1 }
    if (line != "" && !inmol) { print "gmx_harness: no [ molecules ] section in the topology" > "/dev/stderr"; exit 1 }
    if (line != "" && !done) print line
}"""
)

_REMOVE = (
    _FUNCS
    + r"""BEGIN { res = ENVIRON["TOP_RESNAME"] }
{ t = clean($0) }
t ~ /^[ \t]*\[/ { inmol = (t ~ /^[ \t]*\[[ \t]*molecules[ \t]*\]/) }
{ f = incname($0); b = f; sub(/.*\//, "", b); if (f != "" && b == res ".itp") next }
inmol { n = split(t, a); if (n == 2 && a[2] ~ /^[0-9]+$/ && a[1] == res) next }
{ print }"""
)

_FIRST_MOLECULE = _FUNCS + r"""ismol($0) { sub(/\r$/, ""); print; exit }"""

_ADDED = r"""match($0, /Added [0-9]+ molecules/) { split(substr($0, RSTART, RLENGTH), a, " "); print a[2]; exit }"""


def _edit(line_expr: str, include_itp: str | None) -> str:
    inc = shlex.quote(validate_filename(include_itp, "include_itp")) if include_itp else "''"
    return (
        "cp topo.top topo_old.top\n"
        f"TOP_LINE={line_expr} TOP_INCLUDE={inc} awk '{_EDIT}' topo_old.top topo_old.top > topo.top"
    )


def add_molecules_from_log(resname: str, log: str, include_itp: str | None = None) -> str:
    """Add ``RES N`` with N read from gmx insert-molecules output ("Added N molecules")."""
    validate_resname(resname)
    validate_filename(log, "log")
    return (
        f"TOP_N=$(awk '{_ADDED}' {log})\n"
        f"""[ -n "$TOP_N" ] || {{ echo "could not find 'Added N molecules' in {log}"; exit 1; }}\n"""
        + _edit(f"\"$(printf '%-15s %s' {resname} \"$TOP_N\")\"", include_itp)
    )


def add_molecules_from_dummy(dummy: str, include_itp: str | None = None) -> str:
    """Add the ``RES N`` line that gmx solvate wrote into ``dummy`` (copied verbatim)."""
    validate_filename(dummy, "dummy")
    return (
        f"TOP_LINE=$(awk '{_FIRST_MOLECULE}' {dummy})\n"
        f"""[ -n "$TOP_LINE" ] || {{ echo "no 'RES N' line in {dummy}"; exit 1; }}\n"""
        + _edit('"$TOP_LINE"', include_itp)
    )


def include_itp(itp: str) -> str:
    """Add ``#include "itp"`` before ``[ system ]`` (no-op if already included)."""
    return _edit("''", itp)


def remove_molecule(resname: str) -> str:
    """Drop RES from ``[ molecules ]`` and its ``#include "RES.itp"``; nothing else changes. No backup (like mylibs' sed)."""
    validate_resname(resname)
    return f"TOP_RESNAME={resname} awk '{_REMOVE}' topo.top > topo.top.tmp && mv topo.top.tmp topo.top"


def solvent_count(gro: str, mass: float, density: float, rate: float) -> str:
    """Shell expression: molecules of (mass g/mol, density g/cm^3) filling the box of ``gro`` at ``rate`` x density."""
    validate_filename(gro, "gro")
    prog = r"""NF { last = $0 } END { split(last, b); printf "%d\n", b[1] * b[2] * b[3] / (m / d) * r * 602.2 }"""
    return f"$(awk -v m={float(mass)!r} -v d={float(density)!r} -v r={float(rate)!r} '{prog}' {gro})"
