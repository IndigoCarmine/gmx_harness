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
(``Calculation.params()``); enums are stored by member name.
"""

import enum
import inspect
import json
import os
import typing
from collections.abc import Iterator
from typing import Any

from .steps import Calculation, RawShellStep

FORMAT = "gmx_harness.pipeline"
VERSION = 1


def _all_subclasses(cls: type) -> Iterator[type]:
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


def to_json(calculations: list[Calculation]) -> str:
    """Serialize steps to a JSON string."""
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "steps": [{"type": type(c).__name__, "params": c.params()} for c in calculations],
    }
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
    result: list[Calculation] = []
    for i, entry in enumerate(doc.get("steps", [])):
        type_name = entry.get("type")
        cls = registry.get(str(type_name))
        if cls is None:
            raise ValueError(f"steps[{i}]: unknown step type {type_name!r} (known: {sorted(registry)})")
        params: dict[str, Any] = dict(entry.get("params", {}))
        # pydantic dataclasses silently drop unknown keyword arguments; a typo must not vanish
        accepted = set(inspect.signature(cls).parameters)
        unknown = sorted(set(params) - accepted)
        if unknown:
            raise ValueError(f"steps[{i}] ({type_name}): unknown params {unknown}; accepted: {sorted(accepted)}")
        for key, enum_type in _enum_params(cls).items():
            if key in params and not isinstance(params[key], enum_type):
                try:
                    params[key] = enum_type[params[key]]
                except KeyError:
                    raise ValueError(
                        f"steps[{i}].{key}: {params[key]!r} is not one of {[m.name for m in enum_type]}"
                    ) from None
        if issubclass(cls, RawShellStep):
            params["allow_unsafe"] = allow_unsafe
        try:
            result.append(cls(**params))
        except TypeError as e:
            raise ValueError(f"steps[{i}] ({type_name}): {e}") from e
    return result


def save_json(calculations: list[Calculation], filepath: str | os.PathLike[str]) -> None:
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_json(calculations))


def load_json(filepath: str | os.PathLike[str], *, allow_unsafe: bool = False) -> list[Calculation]:
    with open(filepath, "r", encoding="utf-8") as f:
        return from_json(f.read(), allow_unsafe=allow_unsafe)
