"""Bash snippets for the generated step directories.

These functions only build script *text*; the scripts are executed later
by a human or a job scheduler, usually on another machine. Nothing about the
generating machine is written into them: GROMACS is looked up when the
script runs, and can be chosen with environment variables:

    GMX         GROMACS command (default: first of gmx_d, gmx_mpi, gmx on PATH)
    MDRUN_ARGS  extra arguments appended to every ``mdrun`` (e.g. "-ntomp 8 -gpu_id 0")

Besides GROMACS the scripts only need bash and awk (and sha256sum when
``require_preflight`` is used).

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


def mdrun_script(*, extra_args: list[str] | None = None, plumed: bool = False) -> str:
    """
    ``mdrun.sh``: runs output.tpr, resuming from output.cpt if it exists.
    ``extra_args`` (step-specific, e.g. ``-ntmpi 1``) go right after ``mdrun``;
    ``$MDRUN_ARGS`` from the running environment is appended at the end.
    stdout is copied to run.out.

    With ``plumed`` the run uses ``-plumed plumed.dat``. When it continues from
    output.cpt, PLUMED must append to HILLS/COLVAR instead of starting over, so
    a copy with ``RESTART`` on top (plumed_restart.dat) is used unless plumed.dat
    already says RESTART.
    """
    pre = "".join(" " + shlex.quote(a) for a in extra_args or [])
    fresh = " -plumed plumed.dat" if plumed else ""
    resume = ' -plumed "$PLUMED_IN"' if plumed else ""
    restart_setup = (
        "    if grep -qE '^[[:space:]]*RESTART([[:space:]]|$)' plumed.dat; then\n"
        "        PLUMED_IN=plumed.dat\n"
        "    else\n"
        "        { echo RESTART; cat plumed.dat; } > plumed_restart.dat\n"
        "        PLUMED_IN=plumed_restart.dat\n"
        "    fi\n"
    ) if plumed else ""
    return (
        _header()
        + 'if [ -f "output.cpt" ]; then\n'
        + restart_setup
        + f'    "$GMX" mdrun{pre} -deffnm output -v -cpi output.cpt{resume} $MDRUN_ARGS | tee run.out\n'
        + "else\n"
        + f'    "$GMX" mdrun{pre} -deffnm output -v{fresh} $MDRUN_ARGS | tee run.out\n'
        + "fi\n"
    )


def extend_script() -> str:
    """
    ``extend.sh <ps>``: lengthen a finished or interrupted MD step by ``<ps>``
    (``gmx convert-tpr -extend``), then ``bash run.sh`` continues from output.cpt
    without re-running grompp. The previous final structure is kept as
    output_before_extend.gro.
    """
    return (
        _header()
        + 'case "${1:-}" in\n'
        + "    ''|*[!0-9.]*) echo \"usage: bash extend.sh <ps to add>\"; exit 2 ;;\n"
        + "esac\n"
        + 'if [ ! -f output.tpr ] || [ ! -f output.cpt ]; then\n'
        + '    echo "extend.sh needs output.tpr and output.cpt from an earlier run of this step"\n'
        + "    exit 1\n"
        + "fi\n"
        + '"$GMX" convert-tpr -s output.tpr -extend "$1" -o output.tpr\n'
        + "if [ -f output.gro ]; then mv output.gro output_before_extend.gro; fi\n"
        + 'echo "extended by $1 ps; run \'bash run.sh\' to continue from output.cpt"\n'
    )


def generate_xtc_script() -> str:
    """
    ``generate_xtc.sh [GROUP] [OUTPUT]``: output.trr -> OUTPUT (default output.xtc) with
    molecules made whole, writing group GROUP (default 0 = System), e.g.
    ``bash generate_xtc.sh MOL mol_whole.xtc``. Non-interactive.
    """
    return (
        _header(guard=False)
        + 'GROUP="${1:-0}"\n'
        + 'OUT="${2:-output.xtc}"\n'
        + 'echo "$GROUP" | "$GMX" trjconv -f output.trr -s output.tpr -o "$OUT" -pbc mol\n'
        + 'if [ ! -f "$OUT" ]; then\n    echo "Failed to generate $OUT."\n    exit 1\nfi\n'
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


PREFLIGHT = "preflight.ok"


def preflight_guard(where: str = ".") -> str:
    """
    Refuse to start unless ``<where>/preflight.ok`` exists and its ``sha256sum`` lines
    still match (the inputs were not changed after the preflight run). Needs sha256sum
    (GNU coreutils) besides bash; without it the script stops with its own message.
    """
    w = shlex.quote(where)
    return (
        "if ! command -v sha256sum >/dev/null 2>&1; then\n"
        f'    echo "ERROR: sha256sum (GNU coreutils) is not available, so {PREFLIGHT} cannot be verified" >&2\n'
        "    exit 1\nfi\n"
        f'if [ ! -f {w}/{PREFLIGHT} ] || ! (cd {w} && sha256sum --quiet -c {PREFLIGHT}); then\n'
        f'    echo "ERROR: {PREFLIGHT} is missing or the inputs changed since the preflight check;'
        " run the preflight (grompp with each step's maxwarn / plumed driver) again\" >&2\n"
        "    exit 1\nfi\n"
    )


def step_run_script(is_last: bool, require_preflight: bool = False) -> str:
    """``run.sh`` inside a step directory (with ``require_preflight``: checks ``../preflight.ok`` first)."""
    done = 'if [ -f "output.gro" ]; then\n    echo "output.gro already exists. This calculation is finished."\n'
    done += "    . ./copy.sh\n" if not is_last else ""
    done += "    exit 0\nfi\n"
    return (
        "#!/bin/bash\nset -e\n\n"
        + FREEZE_GUARD
        + "\n"
        + done
        + ("\n" + preflight_guard("..") if require_preflight else "")
        + "\n"
        + "if ls ./*.pdb >/dev/null 2>&1; then\n"
        + '    echo "this calculation has problems and this is already calculated."\n'
        + "    exit 1\nfi\n\n"
        + 'if [ -f "output.cpt" ] && [ -f "output.tpr" ]; then\n'
        + '    echo "output.cpt found: continuing the run (grompp skipped, output.tpr kept)"\n'
        + "else\n"
        + "    bash grommp.sh\n"
        + "fi\n"
        + "bash mdrun.sh\n"
        + ("" if is_last else ". ./copy.sh\n")
    )


CARRY_DEFAULT = ("*.top", "*.itp")
_CARRY = re.compile(r"^[A-Za-z0-9_.*?\-]+$")


def validate_carry(patterns: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """File patterns copy.sh hands to the next step (plain globs, no paths)."""
    for pat in patterns:
        if not isinstance(pat, str) or not _CARRY.match(pat) or pat in (".", ".."):
            raise UnsafeNameError(f"carry pattern {pat!r} may only use [A-Za-z0-9_.-] and * ?")
    return tuple(patterns)


def copy_script(this_name: str, next_dir: str, next_name: str, carry: tuple[str, ...] = CARRY_DEFAULT) -> str:
    """
    ``copy.sh``: hand ``carry`` files (topology by default) and the output structure
    to the next step -- unless the next step already finished (it has output.gro).
    Re-running a finished pipeline (or a clone of it) then cannot overwrite a later
    step's topology with an earlier one.
    """
    nd = shlex.quote("../" + next_dir)
    return (
        f"if [ -f {nd}/output.gro ]; then\n"
        f"    echo {shlex.quote(next_name)} already finished: its inputs are left as they are\n"
        "else\n"
        f"    for f in {' '.join(validate_carry(carry))}; do\n"
        f'        if [ -e "$f" ]; then cp "$f" {nd}/; fi\n'
        "    done\n"
        f"    cp output.gro {nd}/input.gro\n"
        "fi\n"
        f"echo {shlex.quote(this_name)} is done\n"
        f"echo Next calculation is {shlex.quote(next_name)}\n"
    )


def pipeline_run_script(step_dirs: list[str], require_preflight: bool = False) -> str:
    """Top-level ``run.sh`` running every step in order and stopping at the first failure."""
    out = ["#!/bin/bash", "set -e", 'cd "$(dirname "$0")"', ""]
    if require_preflight:
        out += [preflight_guard().rstrip("\n"), ""]
    for d in step_dirs:
        q = shlex.quote(d)
        out += [f"(cd {q} && bash run.sh)",
                f'[ -f {q}/output.gro ] || {{ echo "ERROR: {d} finished without producing output.gro"; exit 1; }}',
                ""]
    out.append('echo "All calculations are done"')
    return "\n".join(out) + "\n"
