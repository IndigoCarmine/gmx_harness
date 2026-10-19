"""Save/load pipelines as JSON.

Format::

    {
      "format": "gmx_harness.pipeline",
      "version": 1,
      "steps": [
        {"type": "EM", "params": {"nsteps": 3000, ...}},
        {"type": "MD", "params": {"type": "v_rescale_c_rescale", ...}}
      ]
    }

``type`` is the step class name, ``params`` its constructor arguments
(``Calculation.params()``) except those left at their default; enums are
stored by member name. The children of a ``Nested`` step are stored the same
way, as ``"children": [[sub-folder, {"type": ..., "params": ...}], ...]``.
"""

import enum
import inspect
import json
import os
import typing
from collections.abc import Iterator
from typing import Any

from .steps import Calculation, Nested, RawShellStep

FORMAT = "gmx_harness.pipeline"
VERSION = 1


def _all_subclasses(cls: type) -> Iterator[type]:
    sub: type
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def calculation_registry() -> dict[str, type[Calculation]]:
    """Class name -> class for every ``Calculation`` subclass imported so far (including user-defined ones)."""
    return {c.__name__: c for c in _all_subclasses(Calculation) if not inspect.isabstract(c)}


def _enum_params(cls: type) -> dict[str, type[enum.Enum]]:
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}
    params = dict(inspect.signature(cls).parameters)
    out: dict[str, type[enum.Enum]] = {}
    for key in params:
        t = hints.get(key, params[key].annotation)
        if inspect.isclass(t) and issubclass(t, enum.Enum):
            out[key] = t
    return out


def _non_default_params(calc: Calculation) -> dict[str, Any]:
    """
    ``calc.params()`` without the values that are exactly the constructor default.

    Defaults are not type-converted by pydantic (``emtol: float = 300`` stays the
    int 300 and is written as ``300``) but loaded values are (``300.0``), so
    storing defaults would make a reloaded pipeline generate different text.
    """
    defaults = {k: p.default for k, p in inspect.signature(type(calc)).parameters.items()
                if p.default is not inspect.Parameter.empty}
    return {
        k: v for k, v in calc.params().items()
        if not (k in defaults and type(v) is type(defaults[k]) and v == defaults[k])
    }


def _entry(calc: Calculation) -> dict[str, Any]:
    """``{"type", "params"}`` of one step (recursing into the children of a ``Nested`` step)."""
    params = _non_default_params(calc)
    if isinstance(calc, Nested):
        params["children"] = [[sub, _entry(child)] for sub, child in calc.children]
    return {"type": type(calc).__name__, "params": params}


def to_json(calculations: list[Calculation]) -> str:
    """Serialize steps to a JSON string (parameters left at their default are omitted)."""
    doc = {"format": FORMAT, "version": VERSION, "steps": [_entry(c) for c in calculations]}
    return json.dumps(doc, indent=2)


def from_json(text: str, *, allow_unsafe: bool = False) -> list[Calculation]:
    """
    Rebuild steps from ``to_json`` output. ``RawShellStep`` entries contain
    unchecked shell code and need ``allow_unsafe=True``; that flag is never stored.
    """
    doc = json.loads(text)
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} document")
    if doc.get("version") != VERSION:
        raise ValueError(f"unsupported {FORMAT} version {doc.get('version')!r} (supported: {VERSION})")
    registry = calculation_registry()
    return [_build(entry, f"steps[{i}]", registry, allow_unsafe) for i, entry in enumerate(doc.get("steps", []))]


def _build(entry: Any, where: str, registry: dict[str, type[Calculation]], allow_unsafe: bool) -> Calculation:
    """Rebuild one step from its ``{"type", "params"}`` entry."""
    type_name = entry.get("type")
    cls = registry.get(str(type_name))
    if cls is None:
        raise ValueError(f"{where}: unknown step type {type_name!r} (known: {sorted(registry)})")
    params: dict[str, Any] = dict(entry.get("params", {}))
    # pydantic dataclasses silently drop unknown keyword arguments; a typo must not vanish
    accepted = set(inspect.signature(cls).parameters)
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ValueError(f"{where} ({type_name}): unknown params {unknown}; accepted: {sorted(accepted)}")
    for key, enum_type in _enum_params(cls).items():
        if key in params and not isinstance(params[key], enum_type):
            try:
                params[key] = enum_type[params[key]]
            except KeyError:
                raise ValueError(
                    f"{where}.{key}: {params[key]!r} is not one of {[m.name for m in enum_type]}"
                ) from None
    if issubclass(cls, RawShellStep):
        params["allow_unsafe"] = allow_unsafe
    if issubclass(cls, Nested) and "children" in params:
        params["children"] = [
            (str(sub), _build(child, f"{where}.children[{j}]", registry, allow_unsafe))
            for j, (sub, child) in enumerate(params["children"])
        ]
    try:
        return cls(**params)
    except TypeError as e:
        raise ValueError(f"{where} ({type_name}): {e}") from e


def save_json(calculations: list[Calculation], filepath: str | os.PathLike[str]) -> None:
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_json(calculations))


def load_json(filepath: str | os.PathLike[str], *, allow_unsafe: bool = False) -> list[Calculation]:
    with open(filepath, "r", encoding="utf-8") as f:
        return from_json(f.read(), allow_unsafe=allow_unsafe)
