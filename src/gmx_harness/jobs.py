"""Batch-job files for pipelines (SLURM ``sbatch`` by default).

gmx_harness only writes these files; submitting them is up to the human.
The job template is the user's own (cluster settings, module loads, scratch
staging ...) with ``str.format`` fields, e.g. ``{JOB_NAME}`` and ``{SCRIPT}``;
literal braces in the template are written doubled (``${{SLURM_JOB_ID}}``):

    write_job_scripts("calc", ["MOL_fiber_rot_+10", "MOL_fiber_rot_-10"],
                      open("resource/Gromacs.sbatch").read())

writes ``calc/<system>/Gromacs.sbatch`` for each system plus ``calc/submit.sh``
and ``calc/submit_restart.sh`` (resubmission that waits for the previous job of
the same name, ``sbatch -d singleton``).
"""

import os
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from .safety import UnsafeNameError, validate_filename, validate_name

_COMMAND = re.compile(r"^[A-Za-z0-9_./\-]+$")


def render_job_script(template: str, **fields: str) -> str:
    """``template.format(**fields)``, with a clear error for a missing field."""
    try:
        return template.format(**fields)
    except KeyError as e:
        raise ValueError(f"job template needs field {e.args[0]!r} (given: {sorted(fields)})") from None
    except (IndexError, ValueError) as e:
        raise ValueError(f"job template is not a valid format string ({e}); write literal braces as {{{{ }}}}") from None


def _write(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"{path} exists with different content (overwrite=True to replace)")
    path.write_text(text, encoding="utf-8", newline="\n")


def write_job_scripts(
    calc_path: str | os.PathLike[str],
    systems: Sequence[str],
    template: str,
    *,
    job_file: str = "Gromacs.sbatch",
    script: str = "run.sh",
    prefix: str = "",
    submit_command: str = "sbatch",
    restart_args: Sequence[str] = ("-d", "singleton"),
    fields: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Render ``template`` into ``calc_path/<system>/<job_file>`` for every system
    (fields ``JOB_NAME`` = prefix + system, ``SCRIPT`` = ``script``, plus ``fields``),
    and write ``submit.sh`` / ``submit_restart.sh`` next to the systems.
    Files that exist with different content are refused unless ``overwrite``.
    Returns the written paths.
    """
    root = Path(calc_path)
    validate_filename(job_file, "job_file")
    validate_filename(script, "script")
    if prefix:
        validate_name(prefix, "prefix")
    if not _COMMAND.match(submit_command):
        raise UnsafeNameError(f"submit_command {submit_command!r} must be a plain command name or path")
    written: list[Path] = []
    for name in systems:
        validate_name(name, "system")
        target = root / name
        if not target.is_dir():
            raise FileNotFoundError(f"{target} does not exist (build the pipeline first)")
        text = render_job_script(template, JOB_NAME=prefix + name, SCRIPT=script, **dict(fields or {}))
        _write(target / job_file, text, overwrite)
        written.append(target / job_file)

    def submit(extra: Sequence[str], done: str) -> str:
        cmd = " ".join([shlex.quote(submit_command), *(shlex.quote(a) for a in extra), shlex.quote(job_file)])
        lines = ["#!/bin/bash", "set -e", 'cd "$(dirname "$0")"', ""]
        lines += [f"(cd {shlex.quote(n)} && {cmd})" for n in systems]
        return "\n".join(lines + ["", f'echo "{done}"']) + "\n"

    for fname, text in (("submit.sh", submit((), "All jobs are submitted")),
                        ("submit_restart.sh", submit(list(restart_args), "All jobs are resubmitted"))):
        _write(root / fname, text, overwrite)
        written.append(root / fname)
    return written
