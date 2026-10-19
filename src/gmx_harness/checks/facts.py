"""Facts handed from one stage script to the next.

Each stage script keeps its own settings (readable on its own, easy to tweak and
rerun). Instead of one central spec, the script that *makes* a file records what
it knows about it next to it, and every script that *uses* the file states what
it assumes; a mismatch stops the second script and names the first.

    # 2builder/build.py (producer)
    record_facts(sp_gro, script=__file__, inputs=[rosette_gro], ndisk=10, nros=6, rot=10.0)

    # 3md_planning/plan_metad.py (consumer)
    expect(relaxed_gro, ndisk=NDISK, nros=NROS).enforce(WAIVE)

``<file>.facts.json`` holds free key/value facts (any JSON value, no schema), the
producing script's name, and the sha256 of the file and of every input. Facts are
inherited along ``inputs``: a relaxed structure that only records its inputs still
answers ``ndisk`` from the structure it was relaxed from. When an input changes
after the facts were written (e.g. ``sp/`` was rebuilt but an old relaxed file was
kept), ``expect`` / ``check_fresh`` report ``F003``.

``uv run python -m gmx_harness.checks <dir>`` lists every facts file below ``dir`` with its
producer and staleness (read only).
"""

import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..hashing import sha256_file
from .report import Report

SUFFIX = ".facts.json"


def facts_path(path: str | os.PathLike[str]) -> Path:
    """``<file>.facts.json`` next to ``path``."""
    p = Path(path)
    return p.with_name(p.name + SUFFIX)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return _jsonable(value.tolist())
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, range)):
        return [_jsonable(v) for v in value]
    raise TypeError(f"fact value {value!r} is not JSON-like (use numbers, strings, lists, dicts)")


def record_facts(path: str | os.PathLike[str], *, script: str | os.PathLike[str],
                 inputs: Iterable[str | os.PathLike[str]] = (), **facts: Any) -> Path:
    """
    Write ``<path>.facts.json``: ``facts`` (free keys), the producing ``script`` (pass
    ``__file__``) and the sha256 of ``path`` and of each of ``inputs``. Call it right
    after writing ``path``. Returns the facts file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"record_facts: {p} does not exist (write it first)")
    base = p.resolve().parent
    ins: dict[str, str] = {}
    for i in inputs:
        ip = Path(i).resolve()
        if not ip.is_file():
            raise FileNotFoundError(f"record_facts: input {ip} does not exist")
        ins[os.path.relpath(ip, base).replace(os.sep, "/")] = sha256_file(ip)
    data = {"generator": "gmx_harness", "script": Path(script).name, "sha256": sha256_file(p), "inputs": ins,
            "facts": _jsonable(facts)}
    out = facts_path(p)
    out.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    return out


def read_facts(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The raw facts record of ``path`` (None if there is none)."""
    fp = facts_path(path)
    if not fp.is_file():
        return None
    data = json.loads(fp.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("facts"), dict):
        raise ValueError(f"{fp} is not a facts file")
    return data


def _inputs(path: Path, rec: dict[str, Any]) -> list[Path]:
    return [(path.resolve().parent / rel).resolve() for rel in rec.get("inputs", {})]


def _short(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


def check_fresh(path: str | os.PathLike[str], *, recursive: bool = True) -> Report:
    """F004/F005/F003: the facts of ``path`` exist and neither it nor its inputs changed since."""
    rep = Report()
    seen: set[Path] = set()

    def visit(p: Path, top: bool) -> None:
        if p in seen:
            return
        seen.add(p)
        rec = read_facts(p)
        if rec is None:
            if top:
                rep.error("F004", "no facts recorded; make it with the stage script that records them", _short(p))
            return
        if not p.is_file():
            rep.error("F003", "file is gone but its facts remain", _short(p))
            return
        if sha256_file(p) != rec.get("sha256"):
            rep.error("F005", f"changed after {rec.get('script')} recorded its facts; rerun {rec.get('script')}",
                      _short(p))
        for rel, digest in rec.get("inputs", {}).items():
            ip = (p.resolve().parent / rel).resolve()
            if not ip.is_file():
                rep.error("F003", f"input {rel} of {rec.get('script')} no longer exists", _short(p))
            elif sha256_file(ip) != digest:
                rep.error("F003", f"input {rel} changed after {rec.get('script')} made this file; "
                                  f"rerun {rec.get('script')} (an old result was kept?)", _short(p))
            elif recursive:
                visit(ip, False)

    visit(Path(path).resolve(), True)
    return rep


def lookup(path: str | os.PathLike[str], key: str) -> tuple[Any, str, Path] | None:
    """(value, producing script, file) of ``key`` for ``path``, searching its inputs breadth first."""
    queue = [Path(path).resolve()]
    seen: set[Path] = set()
    while queue:
        p = queue.pop(0)
        if p in seen:
            continue
        seen.add(p)
        rec = read_facts(p)
        if rec is None:
            continue
        if key in rec["facts"]:
            return rec["facts"][key], str(rec.get("script")), p
        queue += _inputs(p, rec)
    return None


def _same(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, str) and not isinstance(b, str):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    return bool(a == b)


def expect(path: str | os.PathLike[str], *, tol: float = 1e-6, fresh: bool = True, **expected: Any) -> Report:
    """
    Compare what this script assumes (``expected``) with the facts recorded for ``path``
    (or inherited from its inputs). F001 mismatch, F002 never recorded; with ``fresh``
    also ``check_fresh`` (F003-F005). Returns a Report; call ``.enforce(WAIVE)``.
    """
    p = Path(path)
    rep = check_fresh(p) if fresh else Report()
    if read_facts(p) is None:
        if not fresh:
            rep.error("F004", "no facts recorded; make it with the stage script that records them", _short(p))
        return rep
    for key, want in expected.items():
        found = lookup(p, key)
        if found is None:
            rep.error("F002", f"{key}={want!r} is assumed here but no stage script recorded {key}", p.name)
            continue
        have, script, src = found
        if not _same(_jsonable(want), have, tol):
            rep.error("F001", f"{key}: {script} recorded {have!r} (for {src.name}), this script assumes {want!r}. "
                              f"Change the setting here, or change {script} and rerun it", p.name)
    return rep


def scan(root: str | os.PathLike[str]) -> str:
    """Table of every facts file below ``root``: file, producer, facts keys, status."""
    lines = []
    for fp in sorted(Path(root).rglob("*" + SUFFIX)):
        target = fp.with_name(fp.name[: -len(SUFFIX)])
        rec = read_facts(target) or {}
        rep = check_fresh(target, recursive=False)
        status = "ok" if rep.ok() else "; ".join(f"{i.code} {i.message}" for i in rep.issues)
        keys = ", ".join(f"{k}={v!r}" for k, v in rec.get("facts", {}).items() if not isinstance(v, (list, dict)))
        lines.append(f"{os.path.relpath(target, root)}  [{rec.get('script')}]  {keys}  -> {status}")
    return "\n".join(lines) if lines else f"no {SUFFIX} files below {root}"
