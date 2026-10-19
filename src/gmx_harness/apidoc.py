"""Generate the LLM-facing API reference (Markdown) from ``__all__`` and docstrings.

No third-party dependency: signatures come from ``inspect``, prose from the
docstrings, so keeping docstrings accurate keeps the agent docs accurate.
"""

import dataclasses
import enum
import importlib
import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Any

# (module, names or None for the module's __all__, heading)
_SECTIONS: list[tuple[str, list[str] | None, str]] = [
    ("gmx_harness", None, "gmx_harness"),
    ("gmx_harness.checks", None, "gmx_harness.checks"),
    ("gmx_harness.archive", None, "gmx_harness.archive"),
    ("gmx_harness.relax", ["relax", "parse_inter_bonds", "find_inter_itp", "write_gro"],
     "gmx_harness.relax"),
    ("gmx_harness.analysis", ["Recorder", "analyze_trj", "process_files", "generate_excel", "mixing_recorders",
                              "get_calcname", "make_analyze_command_script"], "gmx_harness.analysis"),
]


# "collections.abc.Sequence" -> "Sequence", "pathlib._local.Path" -> "Path": shorter, and the
# same text on every Python version (the module path of a class is an implementation detail).
_QUALIFIED = re.compile(r"(?<![\w.'\"])(?:[a-z_][a-z0-9_]*\.)+(?=[A-Za-z_]\w*)")


class _Default:
    """Shows a dataclass ``default_factory`` value in a signature instead of ``<factory>``."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


def _doc(obj: Any) -> str:
    return inspect.getdoc(obj) or ""


def _own_doc(obj: Any) -> str:
    return inspect.cleandoc(obj.__doc__) if getattr(obj, "__doc__", None) else ""


def _sig(obj: Any, *, drop_self: bool = False) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    params = list(sig.parameters.values())
    if drop_self and params and params[0].name in ("self", "cls"):
        params = params[1:]
    if inspect.isclass(obj):
        if dataclasses.is_dataclass(obj):
            factories = {f.name: f.default_factory for f in dataclasses.fields(obj)
                         if f.default_factory is not dataclasses.MISSING}
            params = [p.replace(default=_Default(factories[p.name]())) if p.name in factories else p
                      for p in params]
        sig = sig.replace(parameters=params, return_annotation=inspect.Signature.empty)
    else:
        sig = sig.replace(parameters=params)
    return _QUALIFIED.sub("", str(sig))


def _overrides(cls: type, name: str) -> bool:
    return any(name in base.__dict__ for base in cls.__mro__[1:])


def _describe(name: str, obj: Any) -> list[str]:
    out: list[str] = []
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        out.append(f"### enum `{name}`")
        out.append(", ".join(f"`{m.name}`" for m in obj))
    elif inspect.isclass(obj):
        base = ", ".join(b.__name__ for b in obj.__bases__ if b is not object)
        out.append(f"### class `{name}{_sig(obj)}`" + (f"  (bases: {base})" if base else ""))
        if _doc(obj):
            out.append(_doc(obj))
        methods = []
        for mname, member in obj.__dict__.items():
            if mname.startswith("_"):
                continue
            fn = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
            if isinstance(member, property):
                first = (_doc(member).splitlines() or [""])[0]
                methods.append(f"- property `{mname}`" + (f" - {first}" if first else ""))
            elif inspect.isfunction(fn):
                first = (_own_doc(fn).splitlines() or [""])[0]
                if not first and _overrides(obj, mname):
                    continue  # an undocumented override: the base class lists it
                kind = "classmethod " if isinstance(member, classmethod) else ""
                methods.append(f"- {kind}`{mname}{_sig(fn, drop_self=True)}`" + (f" - {first}" if first else ""))
        if methods:
            out += ["", "Methods:", *methods]
    elif callable(obj):
        out.append(f"### `{name}{_sig(obj)}`")
        if _doc(obj):
            out.append(_doc(obj))
    elif isinstance(obj, dict) and obj and all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
        out += [f"### `{name}`", "", "| key | value |", "|---|---|"]
        out += ["| `%s` | %s |" % (k, v.replace("|", r"\|")) for k, v in obj.items()]
    else:
        out.append(f"### `{name}` = `{obj!r}`")
    return out


def _owners() -> dict[int, str]:
    """Object id -> heading of the section that documents it: the most specific module exporting it."""
    best: dict[int, tuple[int, str]] = {}
    for module_name, names, heading in _SECTIONS:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in names if names is not None else list(getattr(mod, "__all__", [])):
            obj = getattr(mod, name)
            if id(obj) not in best or len(module_name) > best[id(obj)][0]:
                best[id(obj)] = (len(module_name), heading)
    return {k: heading for k, (_, heading) in best.items()}


def generate_api_markdown() -> str:
    lines = [
        "# gmx_harness API reference (for AI agents)",
        "",
        "Generated by `gmx_harness.apidoc.write_api_docs()` from the docstrings. Do not edit by hand.",
        "",
        "Ground rules: gmx_harness only *writes* files and bash scripts; it never runs GROMACS. "
        "Build a plan with `build_plan`, show `plan.preview()`, then `plan.write()`. "
        "Anything that needs `allow_unsafe=True` or `confirm=True` must be approved by the human.",
    ]
    owners = _owners()
    for module_name, names, heading in _SECTIONS:
        try:
            mod: ModuleType = importlib.import_module(module_name)
        except ImportError as e:
            lines += ["", f"## {heading}", "", f"_(not documented: {e})_"]
            continue
        lines += ["", f"## {heading}"]
        if mod.__doc__ and names is None:
            lines += ["", inspect.cleandoc(mod.__doc__)]
        elsewhere: dict[str, list[str]] = {}
        for name in names if names is not None else list(getattr(mod, "__all__", [])):
            if name.startswith("__"):
                continue
            obj = getattr(mod, name)
            if owners.get(id(obj), heading) != heading:
                elsewhere.setdefault(owners[id(obj)], []).append(f"`{name}`")
                continue
            lines += [""] + _describe(name, obj)
        for other, exported in elsewhere.items():
            lines += ["", f"Also exported here, documented under `{other}`: {', '.join(exported)}."]
    return "\n".join(lines) + "\n"


def bundled_api_docs_path() -> Path:
    return Path(__file__).parent / "harness" / "skills" / "gmx-pipeline" / "api.md"


def write_api_docs(path: str | Path | None = None) -> Path:
    """Regenerate the API reference (default: the bundled ``harness/skills/gmx-pipeline/api.md``). Returns the path."""
    out = Path(path) if path is not None else bundled_api_docs_path()
    out.write_text(generate_api_markdown(), encoding="utf-8", newline="\n")
    return out
