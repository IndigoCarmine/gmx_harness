"""Bash snippets for the generated step directories.

These functions only build script *text*; the scripts are executed later
by a human or a job scheduler, usually on another machine. Nothing about the
generating machine is written into them: GROMACS is looked up when the
script runs, and can be chosen with environment variables:

    GMX         GROMACS command (default: first of gmx_d, gmx_mpi, gmx on PATH)
    MDRUN_ARGS  extra arguments appended to every ``mdrun`` (e.g. "-ntomp 8 -gpu_id 0")

Besides GROMACS the scripts only need bash and awk.

Every value generated into a script is either validated (``safety``) or
quoted with ``shlex.quote``.
"""

import re
import shlex

from .safety import UnsafeNameError, validate_filename

FREEZE_GUARD = """if [ -f "freeze" ]; then
    echo "File \\"freeze\\" exists"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!! it is stopped for protection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi
"""

GMX_SETUP = """GMX="${GMX:-$(command -v gmx_d || command -v gmx_mpi || command -v gmx || true)}"
[ -n "$GMX" ] || { echo "GROMACS not found (set GMX)"; exit 1; }
"""

def _header(guard: bool = True) -> str:
    return "#!/bin/bash\nset -eo pipefail\n\n" + (FREEZE_GUARD + "\n" if guard else "") + GMX_SETUP + "\n"


def grompp_script(
    *,
    maxwarn: int = 0,
    restraint: bool = False,
    index_file: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """``grompp.sh``: setting.mdp + topo.top + input.gro -> output.tpr."""
    args = ["-f", "setting.mdp", "-p", "topo.top", "-c", "input.gro", "-o", "output.tpr", "-po", "output.mdp"]
    args += ["-maxwarn", str(int(maxwarn))]
    if restraint:
        args += ["-r", "input.gro"]
    if index_file:
        args += ["-n", validate_filename(index_file, "index_file")]
    args += list(extra_args or [])
    return _header() + '"$GMX" grompp ' + " ".join(shlex.quote(a) for a in args) + "\n"


def mdrun_script(*, extra_args: list[str] | None = None) -> str:
    """
    ``mdrun.sh``: runs output.tpr, resuming from output.cpt if it exists.
    ``extra_args`` (step-specific, e.g. ``-ntmpi 1``) go right after ``mdrun``;
    ``$MDRUN_ARGS`` from the running environment is appended at the end.
    stdout is copied to run.out.
    """
    pre = "".join(" " + shlex.quote(a) for a in extra_args or [])
    return (
        _header()
        + 'if [ -f "output.cpt" ]; then\n'
        + f'    "$GMX" mdrun{pre} -deffnm output -v -cpi output.cpt $MDRUN_ARGS | tee run.out\n'
        + "else\n"
        + f'    "$GMX" mdrun{pre} -deffnm output -v $MDRUN_ARGS | tee run.out\n'
        + "fi\n"
    )


def generate_xtc_script() -> str:
    """``generate_xtc.sh``: output.trr -> output.xtc with molecules made whole. Non-interactive."""
    return (
        _header(guard=False)
        + 'echo 0 | "$GMX" trjconv -f output.trr -s output.tpr -o output.xtc -pbc mol\n'
        + 'if [ ! -f output.xtc ]; then\n    echo "Failed to generate xtc file."\n    exit 1\nfi\n'
    )


def body_script(body: str, *, guard: bool = True) -> str:
    """A script with the standard header followed by ``body`` (already safe bash)."""
    return _header(guard) + body.rstrip("\n") + "\n"


NOOP_GROMPP = '#!/bin/bash\necho "this is a dummy file for automation"\n'


def gmx_command(subcommand: str, args: list[str], stdin: str | None = None) -> str:
    """
    One quoted ``"$GMX" <subcommand>`` command line, optionally fed ``stdin`` (e.g. group selections).

    Example: ``gmx_command("hbond", ["-f", "output.xtc", "-s", "output.tpr", "-num", "hbond.xvg"], stdin="0 0")``
    """
    if not re.match(r"^[a-z][a-z0-9_\-]*$", subcommand):
        raise UnsafeNameError(f"gmx subcommand {subcommand!r} is not valid")
    line = '"$GMX" ' + subcommand + "".join(" " + shlex.quote(a) for a in args)
    if stdin is not None:
        answers = " ".join(shlex.quote(s) for s in stdin.split("\n"))
        line = f"printf '%s\\n' {answers} | " + line
    return line


def step_run_script(is_last: bool) -> str:
    """``run.sh`` inside a step directory."""
    done = 'if [ -f "output.gro" ]; then\n    echo "output.gro already exists. This calculation is finished."\n'
    done += "    . ./copy.sh\n" if not is_last else ""
    done += "    exit 0\nfi\n"
    return (
        "#!/bin/bash\nset -e\n\n"
        + FREEZE_GUARD
        + "\n"
        + done
        + "\n"
        + "if ls ./*.pdb >/dev/null 2>&1; then\n"
        + '    echo "this calculation has problems and this is already calculated."\n'
        + "    exit 1\nfi\n\n"
        + "bash grommp.sh\n"
        + "bash mdrun.sh\n"
        + ("" if is_last else ". ./copy.sh\n")
    )


def copy_script(this_name: str, next_dir: str, next_name: str) -> str:
    """``copy.sh``: hand topology files and the output structure to the next step."""
    nd = shlex.quote("../" + next_dir)
    return (
        "for f in *.top *.itp; do\n"
        f'    if [ -e "$f" ]; then cp "$f" {nd}/; fi\n'
        "done\n"
        f"cp output.gro {nd}/input.gro\n"
        f"echo {shlex.quote(this_name)} is done\n"
        f"echo Next calculation is {shlex.quote(next_name)}\n"
    )


def pipeline_run_script(step_dirs: list[str]) -> str:
    """Top-level ``run.sh`` running every step in order and stopping at the first failure."""
    out = ["#!/bin/bash", "set -e", 'cd "$(dirname "$0")"', ""]
    for d in step_dirs:
        out += [f"(cd {shlex.quote(d)} && bash run.sh)", ""]
    out.append('echo "All calculations are done"')
    return "\n".join(out) + "\n"
