"""Archive calculation directories as plain, dated, checksummed copies (store / export).

::

    python -m gmx_harness.archive --root D:/md_archive store calc_metad/MOL_x           # preview only
    python -m gmx_harness.archive --root D:/md_archive store calc_metad/MOL_x --write   # copy
    python -m gmx_harness.archive --root D:/md_archive tree          # lineage
    python -m gmx_harness.archive --root D:/md_archive verify        # re-hash everything
    python -m gmx_harness.archive --root D:/md_archive export 20261012-002 restore --write

The archive is meant to be read without any tool. Files are copied unchanged, under their own
names and in their own places; nothing is packed, split or renamed::

    ARCHIVE_ROOT/
      tree.json                          overview of every run (rebuilt from the run.json files on each store)
      20261006-001_calc_metad_MOL_x/     one store = one run directory: <date>-<seq>_<tree>_<system>
        run.json                         parent, steps and their state, date, source, note, git commit, job ids
        SHA256SUMS                       "sha256sum -c SHA256SUMS" (GNU coreutils) checks every file
        run.sh  DESCRIPTION.md  ...      the files of the system directory (<tree>/<system>/), as they were
        0_em_vac/ ... 5_npt_unlock/      the step directories, as they were
        _tree/                           the files directly in the tree directory (submit.sh, ...)
      20261012-002_calc_metad_MOL_x/     a branch: it holds only its new steps
        run.json                         "parent": "20261006-001_calc_metad_MOL_x/5_npt_unlock"
        5_npt_unlock/copy.sh             run.sh / copy.sh of an inherited step that changed ("rewired")
        6_md_prod/

Steps are the ``<i>_<name>`` directories of a system, in numeric order. ``store`` compares them, from
step 0 on, with the step sequence every archived run stands for (its ancestors' steps plus its own),
by the names and sha256 of all files except each step's ``run.sh`` / ``copy.sh``. The deepest matching
step becomes the parent and only the steps after it are copied; a step that changed (an extension, a
rerun) therefore starts a new run that branches off just before it and holds that step whole. When the
steps, their run.sh / copy.sh and the system's own files (except ``postcheck.json``) all match a run
of the same ``<tree>/<system>``, nothing is written ("already archived"). Runs of other systems can be
parents too (a variant that shares its first steps), so shared steps are stored once.

A store takes ``.lock`` (never taken over automatically), copies into ``.incoming-<id>/`` while
hashing, stops if a source file changes meanwhile (a job may still be running), writes SHA256SUMS
and run.json, renames the directory into place and makes every file read-only. Entries of the root
whose names start with ``.`` are ignored when reading. Real protection against deletion needs
snapshots or a separate account; read-only files only stop accidents.

``export RUN DEST`` rebuilds ``DEST/<tree>/<system>/`` (``RUN/STEP``: up to that step, without the
records of later runs such as checks.json, postcheck.json, preflight.ok and job output), checking
every file against SHA256SUMS. It never writes into an existing system directory. A prefix export
keeps ``.gmx_harness_manifest.json`` as stored; it may still list files of later steps, which is
fine because the next ``plan_*.py`` / ``build_plan(...).write`` rewrites it.

Everything here is pure Python (no gmx, git or sha256sum is run); every write is previewed first.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import sys
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from . import __version__
from .checks.postrun import _JOB_OUTPUT, step_state
from .hashing import sha256_file
from .safety import UnsafeNameError, validate_relpath

__all__ = [
    "FORMAT",
    "ArchiveError",
    "ArchiveConflictError",
    "ArchiveLockedError",
    "RunInfo",
    "StorePlan",
    "StorePreview",
    "ExportPlan",
    "ExportPreview",
    "VerifyReport",
    "plan_store",
    "plan_export",
    "runs",
    "tree",
    "verify",
    "main",
]

FORMAT = "gmx_harness.archive/1"
RUN_JSON = "run.json"
SUMS = "SHA256SUMS"
TREE_FILES = "_tree"
TREE_JSON = "tree.json"
LOCK = ".lock"
INCOMING = ".incoming-"
_RESERVED = (RUN_JSON, SUMS, TREE_FILES)
_WIRING = ("copy.sh", "run.sh")  # per-step hand-over scripts; rewritten when a step is added after them
_STEP_DIR = re.compile(r"^(\d+)_(.+)$")
_RUN_ID = re.compile(r"^(\d{8})-(\d{3,})")
_JOB_ID = re.compile(r"-(\d+)\.(?:out|err)$")
_SUM_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
# records that are rewritten after every run; a difference in them alone is not a new state
_COMPARE_EXCLUDE = ("postcheck.json",)
# system files that describe the whole run, left out of a RUN/STEP export
_PREFIX_EXCLUDE = ("preflight.ok", "preflight_checks.json", "checks.json", "postcheck.json")
_BUF = 1 << 20
_WINDOWS_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(\..*)?$", re.I)
_WINDOWS_BAD_CHARS = set('<>:"|?*')

# Test hook: called with each source path right after it was copied, before it is checked for changes.
_after_copy: Callable[[Path], None] | None = None


class ArchiveError(RuntimeError):
    """A store / export / verify could not be done (unknown run, broken archive, source changed, ...)."""


class ArchiveConflictError(ArchiveError):
    """``write`` refused; ``.preview`` lists every conflict."""

    def __init__(self, preview: "StorePreview | ExportPreview"):
        self.preview = preview
        super().__init__("refusing to write:\n" + "\n".join(f"  - {c}" for c in preview.conflicts))


class ArchiveLockedError(ArchiveError):
    """Another store holds ``.lock`` (or a crashed one left it behind; a person removes it)."""


# ----------------------------------------------------------------------------- small helpers


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _dump(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _step_key(name: str) -> tuple[int, str]:
    m = _STEP_DIR.match(name)
    return (int(m.group(1)) if m else -1, name)


def _id_key(run_id: str) -> tuple[int, int]:
    m = _RUN_ID.match(run_id)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _size(n: int) -> str:
    x = float(n)
    for unit in ("B", "kB", "MB", "GB"):
        if x < 1000 or unit == "GB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1000
    return f"{n} B"


def _span(names: Sequence[str]) -> str:
    if not names:
        return "(no steps)"
    return names[0] if len(names) == 1 else f"{names[0]} .. {names[-1]}"


def _kind(rel: str) -> str:
    """Where a path inside a run directory (or a source system) belongs: meta / tree / step / system."""
    if rel in (RUN_JSON, SUMS):
        return "meta"
    top, sep, _ = rel.partition("/")
    if sep and top == TREE_FILES:
        return "tree"
    if sep and _STEP_DIR.match(top):
        return "step"
    return "system"


def _dir_kind(rel: str) -> str:
    """Same as ``_kind`` for a directory path (an empty step directory has no '/')."""
    return "step" if _STEP_DIR.match(rel.split("/")[0]) else "system"


def _copy_hashed(src: Path, dst: Path) -> str:
    """Copy ``src`` to the new file ``dst`` (never replaces one) keeping its mtime; return the sha256 copied."""
    h = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "xb") as fo:
        while chunk := fi.read(_BUF):
            h.update(chunk)
            fo.write(chunk)
    st = os.stat(src)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    return h.hexdigest()


def _abs(path: str | os.PathLike[str], base: Path | None) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (base if base is not None else Path.cwd()) / p
    return Path(os.path.abspath(p))


def _inside(child: Path, parent: Path) -> bool:
    c, p = child.resolve(), parent.resolve()
    return c == p or p in c.parents


# ----------------------------------------------------------------------------- reading the archive


@dataclass
class RunInfo:
    """One run directory, as recorded in its run.json (``runs(root)`` lists them, oldest first)."""

    name: str
    id: str
    stored_at: str
    tree: str
    system: str
    parent: str | None
    steps: dict[str, str]
    rewired: list[str] = field(default_factory=list)
    rewired_removed: list[str] = field(default_factory=list)
    empty_dirs: list[str] = field(default_factory=list)
    note: str = ""
    source: str = ""
    host: str = ""
    git: str | None = None
    job_ids: list[str] = field(default_factory=list)
    path: str = ""

    @property
    def parent_run(self) -> str | None:
        """Run directory name of the parent (``None`` for a root run)."""
        return None if self.parent is None else self.parent.split("/", 1)[0]

    @property
    def parent_step(self) -> str | None:
        """Step directory of the parent this run branches off after."""
        return None if self.parent is None else self.parent.partition("/")[2]

    @property
    def date(self) -> str:
        """``YYYY-MM-DD`` of the store."""
        return self.stored_at[:10]


def _str_list(data: dict[str, object], key: str) -> list[str]:
    v = data.get(key, [])
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ArchiveError(f"{RUN_JSON}: '{key}' must be a list of strings")
    return [str(x) for x in v]


def _opt_str(data: dict[str, object], key: str) -> str | None:
    v = data.get(key)
    if v is not None and not isinstance(v, str):
        raise ArchiveError(f"{RUN_JSON}: '{key}' must be a string or null")
    return v


def _load_info(run_dir: Path) -> RunInfo:
    data = json.loads((run_dir / RUN_JSON).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ArchiveError(f"{RUN_JSON} is not a {FORMAT} record")
    steps = data.get("steps", {})
    if not isinstance(steps, dict) or not all(isinstance(v, str) for v in steps.values()):
        raise ArchiveError(f"{RUN_JSON}: 'steps' must map step directories to states")
    name = str(data.get("name", ""))
    if name != run_dir.name:
        raise ArchiveError(f"{RUN_JSON} names the run {name!r} but the directory is {run_dir.name!r} (renamed?)")
    for key in ("rewired", "rewired_removed", "empty_dirs"):
        for rel in _str_list(data, key):
            validate_relpath(rel, f"{RUN_JSON} {key}")
    for s in steps:
        validate_relpath(s, f"{RUN_JSON} step")
    return RunInfo(
        name=name, id=str(data.get("id", "")), stored_at=str(data.get("stored_at", "")),
        tree=str(data.get("tree", "")), system=str(data.get("system", "")), parent=_opt_str(data, "parent"),
        steps={str(k): str(v) for k, v in sorted(steps.items(), key=lambda kv: _step_key(kv[0]))},
        rewired=_str_list(data, "rewired"), rewired_removed=_str_list(data, "rewired_removed"),
        empty_dirs=_str_list(data, "empty_dirs"), note=str(data.get("note", "")),
        source=str(data.get("source", "")), host=str(data.get("host", "")), git=_opt_str(data, "git"),
        job_ids=_str_list(data, "job_ids"), path=str(run_dir),
    )


def _parse_sums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for n, line in enumerate(path.read_bytes().decode("utf-8").split("\n"), 1):
        if not line:
            continue
        m = _SUM_LINE.match(line)
        if not m:
            raise ArchiveError(f"{SUMS} line {n} is not '<sha256>  <path>'")
        rel = validate_relpath(m.group(2), f"{SUMS} line {n}")
        if rel in out:
            raise ArchiveError(f"{SUMS} lists {rel} twice")
        out[rel] = m.group(1)
    return out


@dataclass
class _Step:
    """One step of an effective sequence: where its files live and what they hash to."""

    dirname: str
    owner: str  # run directory holding the step's files ("" for the source being stored)
    files: dict[str, str]  # path inside the step -> sha256, without the step's own run.sh / copy.sh
    dirs: frozenset[str]  # empty directories inside the step
    wiring: dict[str, tuple[str, str]]  # "run.sh" / "copy.sh" -> (run directory holding it, sha256)
    state: str = ""

    def key(self) -> tuple[frozenset[tuple[str, str]], frozenset[str]]:
        return frozenset(self.files.items()), self.dirs

    def wiring_sha(self) -> dict[str, str]:
        return {n: sha for n, (_, sha) in self.wiring.items()}


@dataclass
class _Run:
    info: RunInfo
    sums: dict[str, str]
    path: Path


class _Archive:
    """Every run directory below a root, plus what does not belong there."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs: dict[str, _Run] = {}
        self.problems: list[str] = []
        self.unknown: list[str] = []
        self.incoming: list[str] = []
        self.lock: str | None = None
        self._eff: dict[str, list[_Step]] = {}
        self.order: list[str] = []
        if not root.is_dir():
            return
        for e in sorted(os.scandir(root), key=lambda e: e.name):
            if e.name == TREE_JSON:
                continue
            if e.name == LOCK:
                try:
                    self.lock = Path(e.path).read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    self.lock = "(unreadable)"
                continue
            if e.name.startswith(INCOMING):
                self.incoming.append(e.name)
                continue
            if e.name.startswith("."):
                continue
            p = Path(e.path)
            if not e.is_dir(follow_symlinks=False) or not (p / RUN_JSON).is_file():
                self.unknown.append(e.name)
                continue
            try:
                info = _load_info(p)
                sums = _parse_sums(p / SUMS)
            except (ArchiveError, UnsafeNameError, OSError, ValueError) as err:
                self.problems.append(f"{e.name}: {err}")
                continue
            self.runs[e.name] = _Run(info, sums, p)
        self.order = sorted(self.runs, key=lambda n: _id_key(self.runs[n].info.id))

    def effective(self, name: str, _chain: tuple[str, ...] = ()) -> list[_Step]:
        """The step sequence a run stands for: its ancestors' steps (with its rewiring) plus its own."""
        if name in self._eff:
            return self._eff[name]
        if name in _chain:
            raise ArchiveError(f"parent cycle: {' -> '.join((*_chain, name))}")
        run = self.runs.get(name)
        if run is None:
            raise ArchiveError(f"run {name} not found")
        info = run.info
        steps: list[_Step] = []
        if info.parent is not None:
            prun, pstep = info.parent_run or "", info.parent_step or ""
            if prun not in self.runs:
                raise ArchiveError(f"{name}: parent run {prun} is missing")
            peff = self.effective(prun, (*_chain, name))
            idx = next((i for i, s in enumerate(peff) if s.dirname == pstep), None)
            if idx is None:
                raise ArchiveError(f"{name}: parent step {info.parent} does not exist")
            steps = [replace(s, wiring=dict(s.wiring)) for s in peff[: idx + 1]]
        by_dir = {s.dirname: s for s in steps}
        for key, removing in (("rewired", False), ("rewired_removed", True)):
            for rel in info.rewired_removed if removing else info.rewired:
                d, _, n = rel.partition("/")
                if d not in by_dir or n not in _WIRING or (not removing and rel not in run.sums):
                    raise ArchiveError(f"{name}: {key} entry {rel} does not name an inherited run.sh/copy.sh")
                if removing:
                    by_dir[d].wiring.pop(n, None)
                else:
                    by_dir[d].wiring[n] = (name, run.sums[rel])
        for d, state in info.steps.items():
            files: dict[str, str] = {}
            wiring: dict[str, tuple[str, str]] = {}
            prefix = d + "/"
            for rel, sha in run.sums.items():
                if rel.startswith(prefix):
                    inner = rel[len(prefix):]
                    if inner in _WIRING:
                        wiring[inner] = (name, sha)
                    else:
                        files[inner] = sha
            dirs = frozenset(e[len(prefix):] for e in info.empty_dirs if e.startswith(prefix))
            steps.append(_Step(d, name, files, dirs, wiring, state))
        self._eff[name] = steps
        return steps

    def system_set(self, name: str) -> frozenset[tuple[str, str]]:
        """The run's own system files (and empty directories) as compared by store."""
        run = self.runs[name]
        out = {(rel, sha) for rel, sha in run.sums.items()
               if _kind(rel) == "system" and rel not in _COMPARE_EXCLUDE}
        out |= {(d + "/", "") for d in run.info.empty_dirs if _dir_kind(d) == "system"}
        return frozenset(out)

    def resolve(self, ref: str) -> tuple[str, str | None]:
        """``RUN`` or ``RUN/STEP`` (RUN: directory name, id, or a unique prefix of either) -> (run, step)."""
        ref = ref.replace("\\", "/").strip("/")
        head, _, step = ref.partition("/")
        if head in self.runs:
            name = head
        else:
            matches = [n for n in self.order if n.startswith(head) or self.runs[n].info.id == head]
            if not head or not matches:
                known = ", ".join(self.order[-5:]) or "(archive is empty)"
                raise ArchiveError(f"no run matches {head!r} (latest: {known})")
            if len(matches) > 1:
                raise ArchiveError(f"{head!r} is ambiguous: {', '.join(matches)}")
            name = matches[0]
        if not step:
            return name, None
        names = [s.dirname for s in self.effective(name)]
        if step not in names:
            raise ArchiveError(f"{name} has no step {step!r} (steps: {', '.join(names)})")
        return name, step

    def quick_hashes(self) -> dict[tuple[str, int, int], str]:
        """(path, size, mtime_ns) -> sha256 of the archived files (copies keep the source's mtime)."""
        out: dict[tuple[str, int, int], str] = {}
        for run in self.runs.values():
            for rel, sha in run.sums.items():
                try:
                    st = os.stat(run.path / rel)
                except OSError:
                    continue
                out[(rel, st.st_size, st.st_mtime_ns)] = sha
        return out


def runs(root: str | os.PathLike[str]) -> list[RunInfo]:
    """Every run in the archive, oldest first (unreadable run directories are left out; ``verify`` names them)."""
    a = _Archive(Path(root))
    return [a.runs[n].info for n in a.order]


# ----------------------------------------------------------------------------- the source being stored


@dataclass
class _SrcFile:
    rel: str  # path inside the run directory; tree-level files are "_tree/<name>"
    path: Path
    size: int
    mtime_ns: int


@dataclass
class _Source:
    system_dir: Path
    tree: str
    system: str
    files: dict[str, _SrcFile] = field(default_factory=dict)
    empty_dirs: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def under(self, d: str) -> list[_SrcFile]:
        prefix = d + "/"
        return [f for rel, f in self.files.items() if rel.startswith(prefix)]

    def of_kind(self, kind: str) -> list[_SrcFile]:
        return [f for rel, f in self.files.items() if _kind(rel) == kind]

    def signature(self) -> tuple[dict[str, tuple[int, int]], list[str], list[str]]:
        return {r: (f.size, f.mtime_ns) for r, f in self.files.items()}, self.empty_dirs, self.steps


def _scan(system_dir: Path) -> _Source:
    src = _Source(system_dir, system_dir.parent.name, system_dir.name)
    if not system_dir.is_dir():
        src.conflicts.append(f"{system_dir} is not a directory")
        return src
    for what, name in (("tree", src.tree), ("system", src.system)):
        try:
            validate_relpath(name, f"{what} directory")
        except UnsafeNameError as err:
            src.conflicts.append(str(err))

    def walk(directory: Path, prefix: str) -> None:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
        if not entries and prefix:
            src.empty_dirs.append(prefix)
        for e in entries:
            rel = f"{prefix}/{e.name}" if prefix else e.name
            try:
                validate_relpath(rel, "file name")
            except UnsafeNameError as err:
                src.conflicts.append(str(err))
                continue
            if e.is_symlink() or e.is_junction():
                src.conflicts.append(f"{rel} is a symbolic link (replace it with the real file or directory)")
            elif e.is_dir(follow_symlinks=False):
                if not prefix and _STEP_DIR.match(e.name):
                    src.steps.append(e.name)
                walk(Path(e.path), rel)
            elif e.is_file(follow_symlinks=False):
                st = os.stat(e.path, follow_symlinks=False)  # not DirEntry.stat: on Windows that can lag a writer
                src.files[rel] = _SrcFile(rel, Path(e.path), st.st_size, st.st_mtime_ns)
            else:
                src.conflicts.append(f"{rel} is not a regular file or directory")

    walk(system_dir, "")
    for name in _RESERVED:
        if (system_dir / name).exists() or (system_dir / name).is_symlink():
            src.conflicts.append(f"{name} in the system directory clashes with the archive's own {name}")
    src.steps.sort(key=_step_key)
    if not src.steps:
        src.conflicts.append(f"{system_dir} has no step directories (<i>_<name>)")
    for t in sorted(os.scandir(system_dir.parent), key=lambda e: e.name):
        if t.is_dir():  # systems (a linked directory too): only the files of the tree are kept
            continue
        rel = f"{TREE_FILES}/{t.name}"
        try:
            validate_relpath(rel, "tree file name")
        except UnsafeNameError as err:
            src.conflicts.append(str(err))
            continue
        if t.is_symlink():
            src.conflicts.append(f"{src.tree}/{t.name} is a symbolic link (replace it with the real file)")
        elif t.is_file(follow_symlinks=False):
            st = os.stat(t.path, follow_symlinks=False)
            src.files[rel] = _SrcFile(rel, Path(t.path), st.st_size, st.st_mtime_ns)
    return src


class _Hasher:
    """sha256 of source files, computed once per (path, size, mtime); optionally trusting archived copies."""

    def __init__(self) -> None:
        self.cache: dict[tuple[str, int, int], str] = {}
        self.quick: dict[tuple[str, int, int], str] = {}

    def sha(self, f: _SrcFile) -> str:
        k = (str(f.path), f.size, f.mtime_ns)
        if k not in self.cache:
            q = self.quick.get((f.rel, f.size, f.mtime_ns))
            self.cache[k] = q if q is not None else sha256_file(f.path, _BUF)
        return self.cache[k]

    def known(self, f: _SrcFile) -> str | None:
        return self.cache.get((str(f.path), f.size, f.mtime_ns))


def _src_names(src: _Source, d: str) -> tuple[set[str], frozenset[str]]:
    prefix = d + "/"
    names = {f.rel[len(prefix):] for f in src.under(d)} - set(_WIRING)
    dirs = frozenset(e[len(prefix):] for e in src.empty_dirs if e.startswith(prefix))
    return names, dirs


def _src_step(src: _Source, d: str, hasher: _Hasher) -> _Step:
    prefix = d + "/"
    files: dict[str, str] = {}
    wiring: dict[str, tuple[str, str]] = {}
    for f in src.under(d):
        inner = f.rel[len(prefix):]
        if inner in _WIRING:
            wiring[inner] = ("", hasher.sha(f))
        else:
            files[inner] = hasher.sha(f)
    return _Step(d, "", files, _src_names(src, d)[1], wiring, step_state(src.system_dir / d))


def _src_system_set(src: _Source, hasher: _Hasher) -> frozenset[tuple[str, str]]:
    out = {(f.rel, hasher.sha(f)) for f in src.of_kind("system") if f.rel not in _COMPARE_EXCLUDE}
    out |= {(d + "/", "") for d in src.empty_dirs if _dir_kind(d) == "system"}
    return frozenset(out)


@dataclass
class _Analysis:
    src: _Source
    depth: int = 0  # number of leading steps already in the archive
    best: str | None = None  # newest run that matches those steps
    parent: tuple[str, str] | None = None  # (run holding the deepest matching step, step)
    already: str | None = None
    rewired: list[str] = field(default_factory=list)
    rewired_removed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _analyze(a: _Archive, src: _Source, hasher: _Hasher) -> _Analysis:
    an = _Analysis(src, warnings=list(a.problems))
    effs: dict[str, list[_Step]] = {}
    for name in a.order:
        try:
            effs[name] = a.effective(name)
        except ArchiveError as e:
            an.warnings.append(f"{name} is skipped: {e}")
    if src.conflicts:
        return an
    alive = list(effs)
    for i, d in enumerate(src.steps):
        names, dirs = _src_names(src, d)
        cands = [n for n in alive if len(effs[n]) > i and effs[n][i].dirname == d
                 and set(effs[n][i].files) == names and effs[n][i].dirs == dirs]
        if not cands:
            break
        key = _src_step(src, d, hasher).key()  # hashes only steps that may match
        cands = [n for n in cands if effs[n][i].key() == key]
        if not cands:
            break
        alive, an.depth = cands, i + 1
    if an.depth == 0:
        return an
    an.best = max(alive, key=lambda n: _id_key(a.runs[n].info.id))
    owner = effs[an.best][an.depth - 1].owner
    an.parent = (owner, effs[an.best][an.depth - 1].dirname)
    base = effs[owner][: an.depth]
    src_steps = [_src_step(src, d, hasher) for d in src.steps[: an.depth]]
    for s, b in zip(src_steps, base):
        mine, theirs = s.wiring_sha(), b.wiring_sha()
        for n in _WIRING:
            if mine.get(n) == theirs.get(n):
                continue
            (an.rewired if n in mine else an.rewired_removed).append(f"{s.dirname}/{n}")
    if an.depth == len(src.steps):
        # same system only: an identical replica in another directory is still a state of its own
        same = [n for n in sorted(alive, key=lambda n: _id_key(a.runs[n].info.id), reverse=True)
                if len(effs[n]) == an.depth and (a.runs[n].info.tree, a.runs[n].info.system) == (src.tree, src.system)
                and all(s.wiring_sha() == e.wiring_sha() for s, e in zip(src_steps, effs[n]))]
        if same:
            mine_sys = _src_system_set(src, hasher)
            an.already = next((n for n in same if a.system_set(n) == mine_sys), None)
    return an


# ----------------------------------------------------------------------------- store


@dataclass
class StorePreview:
    """
    Result of ``StorePlan.preview`` / ``write``. ``ok`` is False when there are ``conflicts``; ``already``
    names the run that holds exactly this state (then nothing is written). ``run`` is the planned run
    directory; its id is final only after ``write`` (``written``).
    """

    source: str
    root: str
    run: str = ""
    parent: str | None = None
    inherited: list[str] = field(default_factory=list)
    steps: dict[str, str] = field(default_factory=dict)
    rewired: list[str] = field(default_factory=list)
    rewired_removed: list[str] = field(default_factory=list)
    system_files: list[str] = field(default_factory=list)
    tree_files: list[str] = field(default_factory=list)
    files: int = 0
    bytes: int = 0
    already: str | None = None
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    written: bool = False

    @property
    def ok(self) -> bool:
        """True when there are no conflicts (``write`` would go ahead)."""
        return not self.conflicts

    def __str__(self) -> str:
        out = [f"source: {self.source}", f"archive: {self.root}"]
        if self.already is not None and self.ok:
            out.append(f"already archived as {self.already}")
        else:
            out.append(f"run: {self.run}" + ("" if self.written else "  (the id is assigned on write)"))
            out.append(f"parent: {self.parent or '(none, a new root)'}")
            if self.inherited:
                out.append(f"inherited steps: {_span(self.inherited)}")
            out.append(f"new steps: {len(self.steps)}")
            out += [f"    {d}  {s}" for d, s in self.steps.items()]
            for title, items in (("rewired", self.rewired), ("rewired (removed)", self.rewired_removed),
                                 ("system files", self.system_files), ("_tree files", self.tree_files)):
                if items:
                    out.append(f"{title}: {len(items)}")
                    out += [f"    {i}" for i in items]
            out.append(f"copy: {self.files} files, {_size(self.bytes)}")
        for title, items in (("warnings", self.warnings), ("CONFLICTS", self.conflicts)):
            if items:
                out.append(f"{title}: {len(items)}")
                out += [f"    {i}" for i in items]
        if not self.ok:
            status = "REFUSED"
        elif self.already is not None:
            status = "nothing to write"
        else:
            status = f"written as {self.run}" if self.written else "OK to write"
        out.append("status: " + status)
        return "\n".join(out)


def _next_id(root: Path) -> str:
    today = datetime.now().astimezone().strftime("%Y%m%d")
    seq = 0
    if root.is_dir():
        for e in os.scandir(root):
            m = _RUN_ID.match(e.name.removeprefix(INCOMING))
            if m and m.group(1) == today:
                seq = max(seq, int(m.group(2)))
    return f"{today}-{seq + 1:03d}"


def _root_conflicts(root: Path, system_dirs: Iterable[Path], base: Path | None) -> list[str]:
    out: list[str] = []
    trees: set[Path] = set()
    for s in system_dirs:
        if _inside(root, s):
            out.append(f"the archive root {root} is inside the system directory {s}")
        elif _inside(s, root):
            out.append(f"{s} is inside the archive root {root}")
        trees.add(s.parent)
    if base is not None and base.is_dir():
        trees |= {p for p in base.glob("calc*") if p.is_dir()}
    for t in sorted(trees):
        if _inside(root, t):
            out.append(f"the archive root {root} is inside the calculation tree {t}")
    if root.exists():
        if not root.is_dir():
            out.append(f"the archive root {root} is not a directory")
        else:
            names = [n for n in os.listdir(root) if n != LOCK and not n.startswith(INCOMING)]
            is_archive = (root / TREE_JSON).is_file() or any((root / n / RUN_JSON).is_file() for n in names)
            if names and not is_archive:
                out.append(f"{root} is not empty and is not an archive (no {TREE_JSON}, no run directories)")
    return out


def _git_commit(start: Path) -> str | None:
    """Commit checked out in the git work tree containing ``start``, read from .git files (git is not run)."""
    sha = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    for d in (start, *start.parents):
        g = d / ".git"
        if g.is_dir():
            gitdir = g
        elif g.is_file():
            text = g.read_text(encoding="utf-8", errors="replace").strip()
            if not text.startswith("gitdir:"):
                return None
            gitdir = Path(text[len("gitdir:"):].strip())
            if not gitdir.is_absolute():
                gitdir = d / gitdir
        else:
            continue
        try:
            head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
            if not head.startswith("ref:"):
                return head if sha.match(head) else None
            ref = head[len("ref:"):].strip()
            if ".." in ref.split("/"):
                return None
            common = gitdir
            if (gitdir / "commondir").is_file():
                c = Path((gitdir / "commondir").read_text(encoding="utf-8").strip())
                common = c if c.is_absolute() else gitdir / c
            for g2 in (gitdir, common):
                if (g2 / ref).is_file():
                    v = (g2 / ref).read_text(encoding="utf-8").strip()
                    return v if sha.match(v) else None
            packed = common / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref and sha.match(parts[0]):
                        return parts[0]
        except OSError:
            return None
        return None
    return None


class _Lock:
    """``.lock`` in the archive root, created exclusively; never taken over automatically."""

    def __init__(self, root: Path) -> None:
        self.path = root / LOCK

    def __enter__(self) -> "_Lock":
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                held = self.path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                held = "(unreadable)"
            raise ArchiveLockedError(
                f"{self.path} exists, so another store is running ({held}). If none is running (it was "
                f"interrupted), make sure of that and delete {self.path} by hand, then retry."
            ) from None
        with os.fdopen(fd, "wb") as f:
            f.write(_dump({"host": socket.gethostname(), "pid": os.getpid(), "started": _now()}))
        return self

    def __exit__(self, *exc: object) -> None:
        # Removes only the lock file this process created in __enter__.
        self.path.unlink(missing_ok=True)


def _discard_incoming(root: Path, incoming: Path) -> None:
    # Deleting is allowed here: .incoming-<id> is the work directory this store created moments ago and
    # nothing in it was archived yet (the library deletes only what it made itself).
    if incoming.parent == root and incoming.name.startswith(INCOMING) and incoming.is_dir():
        shutil.rmtree(incoming, ignore_errors=True)


def _set_readonly(top: Path) -> None:
    """Files read-only for everyone (the read-only attribute on Windows); directories too on POSIX."""
    ro = stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH
    nowrite = ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    for dirpath, _dirs, files in os.walk(top, topdown=False):
        for f in files:
            os.chmod(os.path.join(dirpath, f), ro)
        if os.name != "nt":  # a directory's read-only attribute means nothing on Windows
            os.chmod(dirpath, stat.S_IMODE(os.stat(dirpath).st_mode) & nowrite)


def _tree_node(a: _Archive, name: str, children: dict[str, list[str]]) -> dict[str, object]:
    info = a.runs[name].info
    kids = children.get(name, [])
    steps: list[dict[str, object]] = []
    for d, state in info.steps.items():
        steps.append({"step": d, "state": state,
                      "branches": [_tree_node(a, c, children) for c in kids if a.runs[c].info.parent_step == d]})
    for d in sorted({a.runs[c].info.parent_step or "" for c in kids} - set(info.steps), key=_step_key):
        steps.append({"step": d, "state": None, "inherited": True,
                      "branches": [_tree_node(a, c, children) for c in kids if a.runs[c].info.parent_step == d]})
    return {"run": name, "id": info.id, "stored_at": info.stored_at, "tree": info.tree, "system": info.system,
            "parent": info.parent, "note": info.note, "rewired": info.rewired,
            "rewired_removed": info.rewired_removed, "steps": steps}


def _family(a: _Archive) -> tuple[list[str], dict[str, list[str]]]:
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for n in a.order:
        p = a.runs[n].info.parent_run
        if p is not None and p in a.runs and p != n:
            children.setdefault(p, []).append(n)
        else:
            roots.append(n)
    return roots, children


def _write_tree_json(root: Path) -> None:
    a = _Archive(root)
    roots, children = _family(a)
    data = {"format": FORMAT, "generated_at": _now(),
            "about": "overview rebuilt from every run.json on each store; the run.json files are authoritative",
            "runs": [_tree_node(a, r, children) for r in roots]}
    tmp = root / f".{TREE_JSON}.{uuid.uuid4().hex[:12]}.tmp"
    tmp.write_bytes(_dump(data))
    os.replace(tmp, root / TREE_JSON)


class StorePlan:
    """What ``plan_store`` decided to copy. ``preview()`` reads only; ``write()`` copies into the archive."""

    def __init__(self, root: Path, system_dir: Path, note: str, base: Path | None):
        self.root = root
        self.system_dir = system_dir
        self.note = note
        self.base = base
        self._hasher = _Hasher()

    def _run(self) -> tuple[_Analysis, StorePreview, list[_SrcFile]]:
        src = _scan(self.system_dir)
        a = _Archive(self.root)
        an = _analyze(a, src, self._hasher)
        pv = StorePreview(str(self.system_dir), str(self.root), warnings=an.warnings,
                          conflicts=[*_root_conflicts(self.root, [self.system_dir], self.base), *src.conflicts])
        pv.run = f"{_next_id(self.root)}_{src.tree}_{src.system}"
        if an.parent is not None:
            pv.parent = f"{an.parent[0]}/{an.parent[1]}"
        pv.inherited = src.steps[: an.depth]
        pv.already = an.already
        pv.rewired, pv.rewired_removed = an.rewired, an.rewired_removed
        own = src.steps[an.depth:]
        pv.steps = {d: step_state(self.system_dir / d) for d in own}
        copy = [f for d in own for f in src.under(d)]
        copy += [src.files[r] for r in an.rewired]
        system = src.of_kind("system")
        trees = src.of_kind("tree")
        copy += system + trees
        pv.system_files = [f.rel for f in system]
        pv.tree_files = [f.rel for f in trees]
        pv.files, pv.bytes = len(copy), sum(f.size for f in copy)
        return an, pv, sorted(copy, key=lambda f: f.rel)

    def preview(self) -> StorePreview:
        """Compare the system directory with the archive. Reads (and hashes) only; nothing is written."""
        return self._run()[1]

    def write(self) -> StorePreview:
        """
        Store the run (after a fresh preview under the lock). Raises ``ArchiveConflictError`` on
        conflicts, ``ArchiveLockedError`` when ``.lock`` exists and ``ArchiveError`` when a source file
        changed while it was copied; nothing is left behind in those cases. Returns the preview with
        ``written`` (False when it was already archived).
        """
        pv = self.preview()
        if not pv.ok:
            raise ArchiveConflictError(pv)
        self.root.mkdir(parents=True, exist_ok=True)
        with _Lock(self.root):
            an, pv, copy = self._run()  # the archive may have changed since the preview
            if not pv.ok:
                raise ArchiveConflictError(pv)
            if pv.already is not None:
                return pv
            src = an.src
            run_id = _next_id(self.root)
            name = f"{run_id}_{src.tree}_{src.system}"
            final = self.root / name
            incoming = self.root / f"{INCOMING}{run_id}"
            incoming.mkdir()
            try:
                sums: dict[str, str] = {}
                for f in copy:
                    digest = _copy_hashed(f.path, incoming / f.rel)
                    if _after_copy is not None:
                        _after_copy(f.path)
                    st = os.stat(f.path)
                    known = self._hasher.known(f)
                    if (st.st_size, st.st_mtime_ns) != (f.size, f.mtime_ns) or known not in (None, digest):
                        raise ArchiveError(f"{f.path} changed while it was being stored (is a job still "
                                           "running?); nothing was stored")
                    sums[f.rel] = digest
                own = list(pv.steps)
                empty = [d for d in src.empty_dirs if _dir_kind(d) == "system"
                         or d.split("/")[0] in own]
                for d in empty:
                    (incoming / d).mkdir(parents=True, exist_ok=True)
                if _scan(self.system_dir).signature() != src.signature():
                    raise ArchiveError(f"{self.system_dir} changed while it was being stored (is a job still "
                                       "running?); nothing was stored")
                system_names = {f.rel.split("/")[0] for f in src.of_kind("system")}
                job_ids = sorted({m.group(1) for n in system_names
                                  if any(fnmatch.fnmatch(n, p) for p in _JOB_OUTPUT) and (m := _JOB_ID.search(n))},
                                 key=int)
                record = {
                    "format": FORMAT, "id": run_id, "name": name, "stored_at": _now(),
                    "source": self.system_dir.as_posix(), "tree": src.tree, "system": src.system,
                    "host": socket.gethostname(), "parent": pv.parent, "steps": pv.steps,
                    "rewired": pv.rewired, "rewired_removed": pv.rewired_removed, "empty_dirs": empty,
                    "note": self.note, "git": _git_commit(self.base if self.base is not None else self.system_dir),
                    "job_ids": job_ids, "tool": f"gmx_harness {__version__}",
                }
                data = _dump(record)
                (incoming / RUN_JSON).write_bytes(data)
                sums[RUN_JSON] = hashlib.sha256(data).hexdigest()
                (incoming / SUMS).write_bytes(
                    "".join(f"{sums[r]}  {r}\n" for r in sorted(sums)).encode("utf-8"))
                if final.exists():
                    raise ArchiveError(f"{final} already exists")
                os.rename(incoming, final)
            except BaseException:
                _discard_incoming(self.root, incoming)
                raise
            _set_readonly(final)
            _write_tree_json(self.root)
        pv.run, pv.written = name, True
        return pv


def plan_store(root: str | os.PathLike[str], system_dir: str | os.PathLike[str], *, note: str = "",
               base: str | os.PathLike[str] | None = None) -> StorePlan:
    """
    Plan storing one system directory (``<tree>/<system>``, holding ``<i>_<name>`` step directories)
    into the archive at ``root`` (created on write). ``note`` goes into run.json. ``base`` (the
    workspace root) resolves relative paths and adds its ``calc*`` trees to the places the root must
    not be in. Nothing is read until ``preview()``; ``write()`` copies.
    """
    b = _abs(base, None) if base is not None else None
    return StorePlan(_abs(root, b), _abs(system_dir, b), str(note), b)


# ----------------------------------------------------------------------------- export


@dataclass
class ExportPreview:
    """
    Result of ``ExportPlan.preview`` / ``write``: ``dest`` (``DEST/<tree>/<system>``) gets ``files``;
    ``tree_files`` go to ``DEST/<tree>/`` (``tree_unchanged`` are there already with the same content);
    ``excluded`` are left out of a ``RUN/STEP`` export. ``ok`` is False when there are ``conflicts``.
    """

    run: str
    step: str | None
    dest: str
    steps: list[str] = field(default_factory=list)
    files: int = 0
    bytes: int = 0
    tree_files: list[str] = field(default_factory=list)
    tree_unchanged: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    written: bool = False

    @property
    def ok(self) -> bool:
        """True when there are no conflicts (``write`` would go ahead)."""
        return not self.conflicts

    def __str__(self) -> str:
        out = [f"run: {self.run}" + (f"  (up to {self.step})" if self.step else ""), f"dest: {self.dest}",
               f"steps: {_span(self.steps)}", f"copy: {self.files} files, {_size(self.bytes)}"]
        for title, items in (("tree files (to create)", self.tree_files),
                             ("tree files (already there, same content)", self.tree_unchanged),
                             ("excluded (records of later steps)", self.excluded), ("CONFLICTS", self.conflicts)):
            if items:
                out.append(f"{title}: {len(items)}")
                out += [f"    {i}" for i in items]
        out.append("status: " + ("written" if self.written else ("OK to write" if self.ok else "REFUSED")))
        return "\n".join(out)


@dataclass
class _Item:
    src: Path
    rel: str
    sha: str


def _portability(rels: Iterable[str], longest_base: Path, windows: bool) -> list[str]:
    """Names this platform cannot create: Windows-reserved names / characters, case clashes, long paths."""
    out: list[str] = []
    seen: dict[str, str] = {}
    for rel in rels:
        if windows:
            for part in rel.split("/"):
                if _WINDOWS_RESERVED.match(part) or part.endswith((".", " ")) or _WINDOWS_BAD_CHARS & set(part):
                    out.append(f"{rel}: {part!r} cannot be created on Windows")
                    break
            if len(str(longest_base)) + 1 + len(rel) >= 260:
                out.append(f"{rel}: the path would be 260 characters or longer (Windows)")
        if windows or sys.platform == "darwin":
            low = rel.lower()
            if low in seen and seen[low] != rel:
                out.append(f"{rel} and {seen[low]} differ only in case")
            seen.setdefault(low, rel)
    return out


class ExportPlan:
    """What ``plan_export`` decided to copy out. ``preview()`` reads only; ``write()`` creates the copy."""

    def __init__(self, root: Path, ref: str, dest: Path):
        self.root = root
        self.dest = dest
        a = _Archive(root)
        self.run, self.step = a.resolve(ref)
        self._tmp_name = ""

    def _plan(self) -> tuple[ExportPreview, list[_Item], list[str], list[_Item], Path]:
        a = _Archive(self.root)
        if self.run not in a.runs:
            raise ArchiveError(f"run {self.run} disappeared from {self.root}")
        run = a.runs[self.run]
        eff = a.effective(self.run)
        if self.step is not None:
            eff = eff[: [s.dirname for s in eff].index(self.step) + 1]
        info = run.info
        treedir = self.dest / info.tree
        target = treedir / info.system
        pv = ExportPreview(self.run, self.step, str(target), steps=[s.dirname for s in eff])
        items: list[_Item] = []
        dirs: list[str] = []
        for s in eff:
            dirs.append(s.dirname)
            dirs += [f"{s.dirname}/{d}" for d in sorted(s.dirs)]
            for inner, sha in sorted(s.files.items()):
                items.append(_Item(self.root / s.owner / s.dirname / inner, f"{s.dirname}/{inner}", sha))
            for n, (holder, sha) in sorted(s.wiring.items()):
                items.append(_Item(self.root / holder / s.dirname / n, f"{s.dirname}/{n}", sha))
        tree_items: list[_Item] = []
        for rel, sha in sorted(run.sums.items()):
            kind = _kind(rel)
            if kind == "tree":
                tree_items.append(_Item(run.path / rel, rel[len(TREE_FILES) + 1:], sha))
            elif kind == "system":
                top = rel.split("/")[0]
                if self.step is not None and "/" not in rel and (
                        top in _PREFIX_EXCLUDE or any(fnmatch.fnmatch(top, p) for p in _JOB_OUTPUT)):
                    pv.excluded.append(rel)
                    continue
                items.append(_Item(run.path / rel, rel, sha))
        dirs += [d for d in info.empty_dirs if _dir_kind(d) == "system"]
        pv.files = len(items)
        for it in items:
            try:
                pv.bytes += os.stat(it.src).st_size
            except OSError:
                pv.conflicts.append(f"{it.src} is missing from the archive (run verify)")
        if _inside(self.dest, self.root) or _inside(self.root, self.dest):
            pv.conflicts.append(f"{self.dest} and the archive {self.root} overlap")
        if target.exists() or target.is_symlink():
            pv.conflicts.append(f"{target} already exists (export never writes into an existing system directory)")
        for it in tree_items:
            p = treedir / it.rel
            if not p.exists():
                pv.tree_files.append(it.rel)
            elif p.is_file() and sha256_file(p, _BUF) == it.sha:
                pv.tree_unchanged.append(it.rel)
            else:
                pv.conflicts.append(f"{p} already exists with different content")
        if not self._tmp_name:
            self._tmp_name = f".{info.system}.export-{uuid.uuid4().hex[:12]}"
        tmp = treedir / self._tmp_name
        rels = [it.rel for it in items] + dirs
        pv.conflicts += _portability(rels, tmp, os.name == "nt")
        pv.conflicts += _portability([it.rel for it in tree_items], treedir, os.name == "nt")
        tree_new = [it for it in tree_items if it.rel in pv.tree_files]
        return pv, items, dirs, tree_new, tmp

    def preview(self) -> ExportPreview:
        """What would be created. Reads only."""
        return self._plan()[0]

    def write(self) -> ExportPreview:
        """
        Create ``DEST/<tree>/<system>`` (via a temporary sibling directory, renamed at the end) and the
        missing tree files, checking every file against SHA256SUMS. Copies are writable, keep their
        mtime, and on POSIX ``*.sh`` and ``#!`` files get the exec bit. Raises ``ArchiveConflictError``
        without writing on conflicts, ``ArchiveError`` when an archived file does not match its hash.
        """
        pv, items, dirs, tree_new, tmp = self._plan()
        if not pv.ok:
            raise ArchiveConflictError(pv)
        treedir, target = tmp.parent, Path(pv.dest)
        treedir.mkdir(parents=True, exist_ok=True)
        tmp.mkdir()
        try:
            for d in dirs:
                (tmp / d).mkdir(parents=True, exist_ok=True)
            for it in items:
                self._copy(it, tmp / it.rel)
            if target.exists():
                raise ArchiveError(f"{target} appeared while exporting")
            os.rename(tmp, target)
        except BaseException:
            # The temporary directory was created above by this call; nothing else is in it.
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        for it in tree_new:
            self._copy(it, treedir / it.rel)
        pv.written = True
        return pv

    def _copy(self, it: _Item, dst: Path) -> None:
        digest = _copy_hashed(it.src, dst)
        if digest != it.sha:
            dst.unlink()  # the copy this call just created
            raise ArchiveError(f"{it.src} does not match its SHA256SUMS entry (run verify)")
        mode = stat.S_IMODE(os.stat(dst).st_mode) | stat.S_IWUSR
        if os.name != "nt":
            with open(dst, "rb") as f:
                shebang = f.read(2) == b"#!"
            if shebang or dst.name.endswith(".sh"):
                mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(dst, mode)


def plan_export(root: str | os.PathLike[str], ref: str, dest: str | os.PathLike[str], *,
                base: str | os.PathLike[str] | None = None) -> ExportPlan:
    """
    Plan rebuilding ``DEST/<tree>/<system>/`` from a run: ``ref`` is ``RUN`` (directory name, id
    ``YYYYMMDD-NNN`` or a unique prefix) or ``RUN/STEP`` (only up to that step, for starting a branch;
    checks.json, postcheck.json, preflight files and job output are then left out, and
    ``.gmx_harness_manifest.json`` is kept as stored even if it lists later steps: the next plan
    rewrites it). Raises ``ArchiveError`` for an unknown run or step. ``base`` resolves relative paths.
    """
    b = _abs(base, None) if base is not None else None
    return ExportPlan(_abs(root, b), ref, _abs(dest, b))


# ----------------------------------------------------------------------------- tree / verify / status


def _state_summary(steps: dict[str, str]) -> str:
    if not steps:
        return ""
    states = list(steps.values())
    unfinished = [d for d, s in list(steps.items())[:-1] if s != "finished"]
    return states[-1] + (f" (not finished: {', '.join(unfinished)})" if unfinished else "")


def tree(root: str | os.PathLike[str]) -> str:
    """The lineage of every run as indented text (roots, then the runs branching off each step)."""
    a = _Archive(Path(root))
    if not a.runs:
        return "(archive is empty)"
    roots, children = _family(a)
    lines: list[str] = []

    def note(info: RunInfo) -> str:
        return f"  {json.dumps(info.note, ensure_ascii=False)}" if info.note else ""

    def kids(name: str, prefix: str) -> None:
        cs = children.get(name, [])
        for i, c in enumerate(cs):
            last = i == len(cs) - 1
            info = a.runs[c].info
            own = list(info.steps)
            what = f"+{_span(own)} {_state_summary(info.steps)}" if own else "(no new steps: rewiring / system files)"
            lines.append(f"{prefix}{'└─' if last else '├─'} {info.parent_step} → {c}  {info.date}  {what}{note(info)}")
            kids(c, prefix + ("   " if last else "│  "))

    for r in roots:
        info = a.runs[r].info
        missing = f"  (parent {info.parent} missing)" if info.parent is not None else ""
        lines.append(f"{r}  {info.date}  {_span(list(info.steps))}  {_state_summary(info.steps)}{note(info)}{missing}")
        kids(r, "")
    return "\n".join(lines)


@dataclass
class VerifyReport:
    """
    Result of ``verify``: files whose hash differs (``modified``), listed but absent (``missing``),
    present but not listed (``extra``), and broken records (``problems``: unreadable run.json or
    SHA256SUMS, missing parent, unknown entries). ``warnings`` (left-over ``.incoming-*``, ``.lock``)
    do not make ``ok`` False.
    """

    root: str
    runs: list[str] = field(default_factory=list)
    files: int = 0
    modified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing is modified, missing or extra and every record is readable."""
        return not (self.modified or self.missing or self.extra or self.problems)

    def __str__(self) -> str:
        out = [f"archive: {self.root}", f"checked: {len(self.runs)} runs, {self.files} files"]
        for title, items in (("MODIFIED", self.modified), ("MISSING", self.missing), ("EXTRA", self.extra),
                             ("PROBLEMS", self.problems), ("warnings", self.warnings)):
            if items:
                out.append(f"{title}: {len(items)}")
                out += [f"    {i}" for i in items]
        out.append("status: " + ("OK" if self.ok else "FAILED"))
        return "\n".join(out)


def verify(root: str | os.PathLike[str], run: str | None = None) -> VerifyReport:
    """Re-hash every file of every run (or of ``run``) and compare with SHA256SUMS. Reads only."""
    r = Path(root)
    a = _Archive(r)
    rep = VerifyReport(str(r))
    if not r.is_dir():
        rep.problems.append(f"{r} does not exist")
        return rep
    if run is None:
        names = a.order
        rep.problems += a.problems
        rep.problems += [f"{n}: not a run directory (no {RUN_JSON})" for n in a.unknown]
        rep.warnings += [f"{n}: left over from an interrupted store (check, then delete by hand)"
                         for n in a.incoming]
        if a.lock is not None:
            rep.warnings.append(f"{LOCK} exists ({a.lock}): a store is running, or one was interrupted")
    else:
        names = [a.resolve(run)[0]]
    for name in names:
        rep.runs.append(name)
        rr = a.runs[name]
        try:
            a.effective(name)
        except ArchiveError as e:
            rep.problems.append(str(e))
        implied: set[str] = set()
        for rel in [*rr.sums, *rr.info.empty_dirs]:
            parts = rel.split("/")
            implied |= {"/".join(parts[:i]) for i in range(1, len(parts))}
        implied |= set(rr.info.empty_dirs)
        for dirpath, dirnames, filenames in os.walk(rr.path):
            here = Path(dirpath).relative_to(rr.path).as_posix()
            pre = "" if here == "." else here + "/"
            for d in dirnames:
                if pre + d not in implied or Path(dirpath, d).is_symlink():
                    rep.extra.append(f"{name}/{pre}{d}/")
            for f in filenames:
                rel = pre + f
                if rel != SUMS and rel not in rr.sums:
                    rep.extra.append(f"{name}/{rel}")
        for d in rr.info.empty_dirs:
            if not (rr.path / d).is_dir():
                rep.missing.append(f"{name}/{d}/")
        for rel, sha in rr.sums.items():
            p = rr.path / rel
            if not p.is_file():
                rep.missing.append(f"{name}/{rel}")
                continue
            rep.files += 1
            if sha256_file(p, _BUF) != sha:
                rep.modified.append(f"{name}/{rel}")
    return rep


def _systems(base: Path, trees: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for t in trees:
        tdir = base / t
        if not tdir.is_dir():
            continue
        for p in sorted(tdir.iterdir()):
            if p.is_dir() and not p.name.startswith(".") and any(
                    c.is_dir() and _STEP_DIR.match(c.name) for c in p.iterdir()):
                out.append(p)
    return out


def _status(root: Path, base: Path, trees: Sequence[str]) -> list[str]:
    """One line per system of ``trees``. Unchanged files are recognised by size and mtime (copies keep it)."""
    a = _Archive(root)
    hasher = _Hasher()
    hasher.quick = a.quick_hashes()
    out: list[str] = []
    for sdir in _systems(base, trees):
        label = sdir.relative_to(base).as_posix()
        src = _scan(sdir)
        if src.conflicts:
            out.append(f"{label}: cannot be stored ({src.conflicts[0]})")
            continue
        an = _analyze(a, src, hasher)
        if an.already is not None:
            out.append(f"{label}: archived as {an.already}")
        elif an.best is not None:
            new = src.steps[an.depth:]
            what = f"new: {_span(new)}" if new else "rewiring / system files"
            out.append(f"{label}: changed since {an.best} (same up to {src.steps[an.depth - 1]}; {what})")
        else:
            out.append(f"{label}: not archived")
    return out


# ----------------------------------------------------------------------------- command line


def main(argv: list[str], *, root: str | os.PathLike[str] | None = None,
         base: str | os.PathLike[str] | None = None, trees: Sequence[str] = ()) -> int:
    """
    Command line: ``[--root R] [status | tree | verify [RUN] | store DIR... [--all] [--note T] [--write]
    | export RUN[/STEP] DEST [--write]]`` (default ``status``). ``root`` / ``base`` / ``trees`` are the
    defaults a workspace script passes (``--root``, ``--base``, ``--tree`` override them); relative
    paths resolve against ``base`` (else the current directory). ``status`` prints the tree and, for
    every system of ``trees``, "not archived", "archived as RUN" or "changed since RUN" (fast: a file with
the size and mtime of an archived copy counts as unchanged; ``store`` hashes what it compares). Without
    ``--write`` store and export only preview. Returns 0, or 1 on conflicts / verify failures / errors.
    """
    p = argparse.ArgumentParser(prog="python -m gmx_harness.archive",
                                description="Store calculation directories in a plain, checksummed archive.")
    p.add_argument("--root", help="archive root directory")
    p.add_argument("--base", help="workspace root (relative paths, --all, status)")
    p.add_argument("--tree", action="append", dest="trees", help="calculation tree to scan (repeatable)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("status", help="tree plus the archive state of every system")
    sub.add_parser("tree", help="lineage of the runs")
    v = sub.add_parser("verify", help="re-hash and compare with SHA256SUMS")
    v.add_argument("run", nargs="?")
    s = sub.add_parser("store", help="store system directories (preview unless --write)")
    s.add_argument("dirs", nargs="*")
    s.add_argument("--all", action="store_true", help="every system of the trees")
    s.add_argument("--note", default="")
    s.add_argument("--write", action="store_true")
    e = sub.add_parser("export", help="rebuild <tree>/<system> under DEST (preview unless --write)")
    e.add_argument("ref", help="RUN or RUN/STEP")
    e.add_argument("dest")
    e.add_argument("--write", action="store_true")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 2)
    b = _abs(args.base, None) if args.base else (_abs(base, None) if base is not None else None)
    root_arg = args.root if args.root else root
    if root_arg is None:
        print("error: no archive root (--root)", file=sys.stderr)
        return 2
    r = _abs(root_arg, b)
    tree_names: Sequence[str] = args.trees or trees
    cmd = args.cmd or "status"
    try:
        if cmd == "status":
            print(tree(r))
            lines = _status(r, b if b is not None else Path.cwd(), tree_names)
            if lines:
                print()
                print("\n".join(lines))
            return 0
        if cmd == "tree":
            print(tree(r))
            return 0
        if cmd == "verify":
            rep = verify(r, args.run)
            print(rep)
            return 0 if rep.ok else 1
        if cmd == "store":
            dirs = [_abs(d, b) for d in args.dirs]
            if args.all:
                dirs += [d for d in _systems(b if b is not None else Path.cwd(), tree_names) if d not in dirs]
            if not dirs:
                print("error: store needs DIR... or --all (with trees)", file=sys.stderr)
                return 2
            plans = [plan_store(r, d, note=args.note, base=b) for d in dirs]
            previews = [pl.preview() for pl in plans]
            if not args.write or not all(pv.ok for pv in previews):
                print("\n\n".join(str(pv) for pv in previews))
                return 0 if all(pv.ok for pv in previews) else 1
            for i, pl in enumerate(plans):  # each write previews again under the lock
                print(("\n" if i else "") + str(pl.write()), flush=True)
            return 0
        plan = plan_export(r, args.ref, _abs(args.dest, b))
        epv = plan.preview()
        if args.write and epv.ok:
            epv = plan.write()
        print(epv)
        return 0 if epv.ok else 1
    except (ArchiveError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
