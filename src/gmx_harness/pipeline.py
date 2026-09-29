"""Turn a list of steps into a directory tree of inputs and bash scripts.

Flow::

    plan = build_plan([EM(), MD(...)], "start.gro", "work", extra_inputs=["topo.top"])
    print(plan.preview())            # what would be created / overwritten / refused
    plan.write()                     # OverwritePolicy.ERROR by default

Nothing is ever executed. ``write`` checks every conflict *before* touching
the disk, never deletes anything it did not generate itself (tracked in
``.gmx_harness_manifest.json`` with content hashes), and never asks for
interactive input, so it is safe to call from scripts and AI agents.
"""

import enum
import hashlib
import json
import os
import shlex
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .safety import UnsafeOperationError, ensure_deletable_root, ensure_within, validate_filename, validate_name
from .scripts import CARRY_DEFAULT, NOOP_GROMPP, copy_script, validate_carry, pipeline_run_script, step_run_script
from .steps.base import Calculation

MANIFEST = ".gmx_harness_manifest.json"
# per-step scripts that connect a step to its neighbours
_WIRING = {"run.sh", "copy.sh"}
_RESERVED = {"run.sh", "copy.sh", MANIFEST}


class OverwritePolicy(enum.Enum):
    """
    What ``Plan.write`` does when target files already exist.

    ERROR: refuse if any step directory or file already exists (default).
    SKIP_EXISTING: keep existing step directories, add only new steps. Only their
        run.sh / copy.sh (the hand-over to the next step) are updated, and only
        if gmx_harness generated them and nobody edited them.
    REPLACE_GENERATED: overwrite files that gmx_harness generated earlier and
        nobody edited since (checked by hash); refuse for anything else.
    """

    ERROR = "error"
    SKIP_EXISTING = "skip_existing"
    REPLACE_GENERATED = "replace_generated"


class PlanConflictError(RuntimeError):
    """``Plan.write`` refused; ``.preview`` lists every conflict."""

    def __init__(self, preview: "PlanPreview"):
        self.preview = preview
        super().__init__("refusing to write:\n" + "\n".join(f"  - {c}" for c in preview.conflicts))


@dataclass(frozen=True)
class PlannedFile:
    relpath: str  # posix path relative to the working directory
    content: bytes
    executable: bool = False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True)
class StepPlan:
    index: int
    name: str
    dirname: str
    calculation: Calculation


@dataclass
class PlanPreview:
    """Result of ``Plan.preview`` / ``Plan.write``. ``ok`` is False when there are conflicts."""

    working_dir: str
    policy: str
    create: list[str] = field(default_factory=list)
    overwrite: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    stale_outputs: list[str] = field(default_factory=list)
    written: bool = False

    @property
    def ok(self) -> bool:
        return not self.conflicts

    def __str__(self) -> str:
        out = [f"working_dir: {self.working_dir}  (policy: {self.policy})"]
        for title, items in (
            ("create", self.create),
            ("overwrite", self.overwrite),
            ("remove (generated earlier, no longer planned)", self.remove),
            ("unchanged", self.unchanged),
            ("skipped steps (already exist)", self.skipped_steps),
            ("existing run outputs (may be stale)", self.stale_outputs),
            ("CONFLICTS", self.conflicts),
        ):
            if items:
                out.append(f"{title}: {len(items)}")
                out += [f"    {i}" for i in items]
        out.append("status: " + ("written" if self.written else ("OK to write" if self.ok else "REFUSED")))
        return "\n".join(out)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(working_dir: Path) -> dict[str, str]:
    p = working_dir / MANIFEST
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    files = data.get("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"{p} is corrupt")
    return {str(k): str(v) for k, v in files.items()}


class Plan:
    """Everything ``build_plan`` decided to write. Inspect with ``summary()`` / ``preview()``, then ``write()``."""

    def __init__(self, working_dir: Path, steps: list[StepPlan], files: list[PlannedFile]):
        self.working_dir = working_dir
        self.steps = steps
        self.files = files

    def summary(self) -> str:
        """Step list and the files of each step (no disk access)."""
        out = [f"{self.working_dir}/"]
        by_dir: dict[str, list[str]] = {}
        for f in self.files:
            head, _, tail = f.relpath.partition("/")
            by_dir.setdefault(head if tail else "", []).append(tail or head)
        for name in by_dir.get("", []):
            out.append(f"  {name}")
        for s in self.steps:
            out.append(f"  {s.dirname}/  ({type(s.calculation).__name__})")
            out += [f"    {n}" for n in by_dir.get(s.dirname, [])]
        return "\n".join(out)

    def file(self, relpath: str) -> str:
        """Content of one planned file, e.g. ``plan.file("1_md/setting.mdp")``."""
        for f in self.files:
            if f.relpath == relpath:
                return f.content.decode("utf-8")
        raise KeyError(relpath)

    def preview(self, overwrite: OverwritePolicy = OverwritePolicy.ERROR, *, force_modified: bool = False) -> PlanPreview:
        """Compare the plan with the disk. Pure read; nothing is changed."""
        wd = self.working_dir
        pv = PlanPreview(str(wd), overwrite.value)
        manifest = _read_manifest(wd)
        skipped_dirs: set[str] = set()

        for s in self.steps:
            d = wd / s.dirname
            if not d.exists():
                continue
            if not d.is_dir():
                pv.conflicts.append(f"{s.dirname} exists and is not a directory")
            elif overwrite is OverwritePolicy.ERROR:
                pv.conflicts.append(f"{s.dirname}/ already exists (use SKIP_EXISTING or REPLACE_GENERATED)")
            elif overwrite is OverwritePolicy.SKIP_EXISTING:
                skipped_dirs.add(s.dirname)
                pv.skipped_steps.append(s.dirname)

        planned = {f.relpath for f in self.files}
        for f in self.files:
            head, _, name = f.relpath.partition("/")
            # A skipped step keeps its content, but its hand-over scripts must follow the
            # new step list (e.g. the old last step now needs a copy.sh to the new step).
            if head in skipped_dirs and name not in _WIRING:
                continue
            p = wd / f.relpath
            if not p.exists():
                pv.create.append(f.relpath)
                continue
            if p.is_dir():
                pv.conflicts.append(f"{f.relpath} is a directory")
                continue
            current = _sha(p)
            if current == f.sha256:
                pv.unchanged.append(f.relpath)
            elif overwrite is OverwritePolicy.ERROR:
                pv.conflicts.append(f"{f.relpath} already exists")
            elif f.relpath not in manifest:
                pv.conflicts.append(f"{f.relpath} exists but was not generated by gmx_harness")
            elif manifest[f.relpath] != current and not force_modified:
                pv.conflicts.append(f"{f.relpath} was modified after generation (force_modified=True to replace)")
            else:
                pv.overwrite.append(f.relpath)

        if overwrite is OverwritePolicy.REPLACE_GENERATED:
            for rel, digest in manifest.items():
                if rel in planned or rel.split("/")[0] in skipped_dirs:
                    continue
                p = wd / rel
                if p.is_file():
                    if _sha(p) == digest or force_modified:
                        pv.remove.append(rel)
                    else:
                        pv.conflicts.append(f"{rel} is no longer planned but was modified; remove it by hand")
            for s in self.steps:
                d = wd / s.dirname
                if d.is_dir() and s.dirname not in skipped_dirs:
                    for child in sorted(d.iterdir()):
                        rel = f"{s.dirname}/{child.name}"
                        if rel not in planned and rel not in manifest:
                            pv.stale_outputs.append(rel)
        return pv

    def write(self, overwrite: OverwritePolicy = OverwritePolicy.ERROR, *, force_modified: bool = False) -> PlanPreview:
        """Write the plan. Raises ``PlanConflictError`` (without writing anything) on conflicts."""
        pv = self.preview(overwrite, force_modified=force_modified)
        if not pv.ok:
            raise PlanConflictError(pv)
        wd = self.working_dir
        wd.mkdir(parents=True, exist_ok=True)
        manifest = _read_manifest(wd)
        todo = set(pv.create) | set(pv.overwrite)
        for f in self.files:
            if f.relpath not in todo:
                continue
            p = ensure_within(wd, wd / f.relpath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f.content)
            if f.executable:
                p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            manifest[f.relpath] = f.sha256
        for f in self.files:
            if f.relpath in pv.unchanged:
                manifest.setdefault(f.relpath, f.sha256)
        for rel in pv.remove:
            ensure_within(wd, wd / rel).unlink()
            manifest.pop(rel, None)
        (wd / MANIFEST).write_text(
            json.dumps({"generator": "gmx_harness", "version": __version__, "files": dict(sorted(manifest.items()))},
                       indent=2),
            encoding="utf-8",
        )
        pv.written = True
        return pv


InputFiles = Sequence[str | os.PathLike[str]] | Mapping[str, str | os.PathLike[str]]


def _input_items(inputs: InputFiles | None) -> list[tuple[str, Path]]:
    """(target file name, source path) pairs from a list (keep names) or a {target: source} dict."""
    if inputs is None:
        return []
    if isinstance(inputs, Mapping):
        items = [(str(t), Path(src)) for t, src in inputs.items()]
    else:
        items = [(Path(src).name, Path(src)) for src in inputs]
    return [(validate_filename(t, "input file name"), src) for t, src in items]


def _as_bytes(text: str) -> bytes:
    return text.replace("\r\n", "\n").encode("utf-8")


def build_plan(
    calculations: list[Calculation],
    input_gro: str | os.PathLike[str],
    working_dir: str | os.PathLike[str],
    *,
    extra_inputs: InputFiles | None = None,
    step_inputs: Mapping[str, InputFiles] | None = None,
    carry: Sequence[str] = CARRY_DEFAULT,
) -> Plan:
    """
    Plan a pipeline: ``working_dir/0_<name>/``, ``1_<name>/``, ... plus a top-level ``run.sh``.

    Args:
        calculations: steps in order; names must be unique.
        input_gro: starting structure, copied to ``0_<name>/input.gro``.
        working_dir: output directory (created on write).
        extra_inputs: files copied into the first step, typically the topology and ``*.itp``.
            A list keeps each file's name; a dict ``{target name: source path}`` renames,
            e.g. ``{"topo.top": "MOL_fixed.top"}``. .top/.itp files are handed on to every
            following step at run time.
        step_inputs: files for other steps, ``{calculation_name: files}`` with ``files`` as in
            ``extra_inputs``, e.g. ``{"metad": {"index.ndx": "sp/MOL.ndx"}}``.
        carry: file patterns each step hands to the next at run time (default ``*.top``
            and ``*.itp``); add ``"*.ndx"`` to pass an index file down the whole pipeline.
    Returns:
        Plan (nothing is written yet).
    """
    if not calculations:
        raise ValueError("no calculations given")
    names = [validate_name(c.name) for c in calculations]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate calculation names: {dupes}")

    carry_t = validate_carry(tuple(carry))
    wd = Path(working_dir)
    steps = [StepPlan(i, c.name, f"{i}_{c.name}", c) for i, c in enumerate(calculations)]
    files: list[PlannedFile] = []

    for i, s in enumerate(steps):
        generated = s.calculation.generate()
        if "mdrun.sh" not in generated:
            raise ValueError(f"{s.dirname}: generate() must return an mdrun.sh")
        generated.setdefault("grommp.sh", NOOP_GROMPP)
        for fname, content in generated.items():
            validate_filename(fname, f"{s.dirname} file name")
            if fname in _RESERVED:
                raise ValueError(f"{s.dirname}: {fname} is reserved for gmx_harness")
            if not isinstance(content, str):
                raise TypeError(f"{s.dirname}/{fname}: content must be str")
            files.append(PlannedFile(f"{s.dirname}/{fname}", _as_bytes(content), fname.endswith(".sh")))
        is_last = i == len(steps) - 1
        files.append(PlannedFile(f"{s.dirname}/run.sh", _as_bytes(step_run_script(is_last)), True))
        if not is_last:
            nxt = steps[i + 1]
            files.append(PlannedFile(f"{s.dirname}/copy.sh", _as_bytes(copy_script(s.name, nxt.dirname, nxt.name, carry_t)), True))

    by_name = {s.name: s.dirname for s in steps}
    unknown = sorted(set(step_inputs or {}) - set(by_name))
    if unknown:
        raise ValueError(f"step_inputs: no step named {unknown} (steps: {sorted(by_name)})")
    placements: list[tuple[str, str, Path]] = [(steps[0].dirname, "input.gro", Path(input_gro))]
    placements += [(steps[0].dirname, t, src) for t, src in _input_items(extra_inputs)]
    for step_name, inputs in (step_inputs or {}).items():
        placements += [(by_name[step_name], t, src) for t, src in _input_items(inputs)]

    taken = {f.relpath for f in files}
    for dirname, target, srcp in placements:
        rel = f"{dirname}/{target}"
        if rel in taken or target in _RESERVED or target == "output.gro":
            raise ValueError(f"{rel} is planned twice or is managed by gmx_harness")
        taken.add(rel)
        files.append(PlannedFile(rel, srcp.read_bytes()))

    files.append(PlannedFile("run.sh", _as_bytes(pipeline_run_script([s.dirname for s in steps])), True))
    for f in files:
        ensure_within(wd, wd / f.relpath)
    return Plan(wd, steps, files)


# ----------------------------------------------------------------------------- mylibs-compatible API


class OverwriteType(enum.Enum):
    """mylibs' overwrite modes for ``launch``: no -> ERROR, add_calculation -> SKIP_EXISTING."""

    no = 0
    full_overwrite = 1
    add_calculation = 2


def launch(
    calculations: list[Calculation],
    input_gro: str,
    working_dir: str,
    overwrite: OverwriteType = OverwriteType.no,
    *,
    confirm: bool = False,
    extra_inputs: InputFiles | None = None,
    step_inputs: Mapping[str, InputFiles] | None = None,
    carry: Sequence[str] = CARRY_DEFAULT,
) -> PlanPreview:
    """
    mylibs-compatible one-shot ``build_plan(...).write(...)``.

    ``full_overwrite`` deletes ``working_dir`` first. It never prompts: it needs
    ``confirm=True`` and only works on a directory that gmx_harness created
    (it has a manifest). Prefer ``build_plan`` + ``OverwritePolicy.REPLACE_GENERATED``.
    """
    plan = build_plan(calculations, input_gro, working_dir, extra_inputs=extra_inputs, step_inputs=step_inputs,
                      carry=carry)
    wd = Path(working_dir)
    if overwrite is OverwriteType.full_overwrite:
        if wd.exists():
            if not confirm:
                raise UnsafeOperationError(
                    f"full_overwrite would delete {wd}; pass confirm=True "
                    "(or use build_plan(...).write(OverwritePolicy.REPLACE_GENERATED))"
                )
            if not (wd / MANIFEST).exists():
                raise UnsafeOperationError(f"{wd} was not created by gmx_harness (no {MANIFEST}); delete it by hand")
            ensure_deletable_root(wd)
            shutil.rmtree(wd)
        return plan.write(OverwritePolicy.ERROR)
    policy = OverwritePolicy.SKIP_EXISTING if overwrite is OverwriteType.add_calculation else OverwritePolicy.ERROR
    return plan.write(policy)


def generate_stepbystep_runfile(
    init_structures: list[str],
    calculation_name_and_isparaleljob: list[tuple[str, bool]],
    calc_path: str,
) -> None:
    """
    Write ``<step dir>.sh`` per step plus ``stebystep.sh`` into ``calc_path``, running one
    step for every structure directory before moving to the next step (optionally in parallel).
    """
    for s in init_structures:
        validate_name(s, "init structure directory")
    for calculation_name, do_parallel in calculation_name_and_isparaleljob:
        validate_name(calculation_name, "step directory")
        lines = ["#!/bin/bash"]
        if do_parallel:
            lines.append("echo 'Starting parallel job'")
        for s in init_structures:
            target = shlex.quote(f"{s}/{calculation_name}")
            lines.append(f"(cd {target} && bash run.sh)" + (" &" if do_parallel else ""))
        if do_parallel:
            lines.append("wait")
        Path(calc_path, f"{calculation_name}.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with open(os.path.join(calc_path, "stebystep.sh"), "w", newline="\n") as f:
        f.write("#!/bin/bash\nset -e\n")
        for calculation_name, _ in calculation_name_and_isparaleljob:
            f.write(f"bash {shlex.quote(calculation_name + '.sh')}\n")
        f.write('echo "All calculations are done"\n')


def generate_batch_execution_script(
    init_name: list[str],
    base_dir: str,
    calc_path: str | None = None,
    command: str = "bash run.sh",
    *,
    allow_unsafe: bool = False,
) -> None:
    """
    Write ``base_dir/run.sh`` that runs ``command`` in every ``<init>/<calc_path>`` directory.
    A ``command`` other than "bash run.sh" is unchecked shell code and needs ``allow_unsafe=True``.
    """
    if command != "bash run.sh" and not allow_unsafe:
        raise UnsafeOperationError("a custom command is written unchecked; pass allow_unsafe=True")
    if calc_path is not None:
        validate_name(calc_path, "calc_path")
    lines = ["#!/bin/bash"]
    for file in init_name:
        d = validate_name(os.path.basename(file).split(".")[0], "init structure directory")
        target = shlex.quote(d if calc_path is None else f"{d}/{calc_path}")
        lines += [f"(cd {target} && {command})", ""]
    lines.append('echo "All calculations are done"')
    Path(base_dir, "run.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
