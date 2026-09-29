"""The ``Calculation`` interface every pipeline step implements."""

import dataclasses
import enum
import logging
from abc import ABC, abstractmethod
from importlib import resources
from typing import Any

from ..mdp import MDParameters
from ..safety import validate_define, validate_name

logger = logging.getLogger("gmx_harness")


def default_file_content(name: str) -> str:
    """Text of a bundled data file (``gmx_harness/data/<name>``)."""
    return resources.files("gmx_harness").joinpath("data", name).read_text(encoding="utf-8")


class Calculation(ABC):
    """
    One step of a GROMACS pipeline (one numbered directory).

    ``generate()`` returns ``{file name: content}``. It must contain
    ``mdrun.sh``; ``grommp.sh`` is optional (a no-op is written if missing).
    Each step reads ``input.gro`` (+ ``topo.top``/``*.itp`` handed over by
    the previous step) and must produce ``output.gro``.

    To add your own step type, subclass this, give it a ``calculation_name``
    attribute and implement ``generate``; validation and writing are handled
    by ``build_plan``.
    """

    calculation_name: str

    @abstractmethod
    def generate(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.calculation_name

    def params(self) -> dict[str, Any]:
        """
        Constructor arguments that rebuild this step (used by ``save_json``).
        Enums are stored by member name. Override if ``__init__`` arguments
        differ from the stored attributes.
        """
        if dataclasses.is_dataclass(self):
            d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        else:
            d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return {k: (v.name if isinstance(v, enum.Enum) else v) for k, v in d.items()}


def check_common(name: str, defines: list[str] | None = None, maxwarn: int = 0, temperature: float | None = None,
                 gen_vel: str | None = None) -> None:
    """Validation shared by the MD-like steps."""
    validate_name(name)
    for d in defines or []:
        validate_define(d)
    if maxwarn < 0:
        raise ValueError("maxwarn must be >= 0")
    if maxwarn != 0:
        logger.warning("%s: maxwarn=%d. grompp warnings will be ignored - make sure you know which ones.", name, maxwarn)
    if gen_vel is not None and gen_vel not in ("yes", "no"):
        raise ValueError("gen_vel must be 'yes' or 'no'")
    if temperature is not None and temperature < 0:
        raise ValueError("temperature must be >= 0")


def apply_common_mdp(
    mdp_file: MDParameters,
    *,
    defines: list[str],
    additional: dict[str, Any],
    restraint: bool = False,
) -> MDParameters:
    for key, value in additional.items():
        mdp_file.add_or_update(key, str(value))
    if restraint:
        mdp_file.add_or_update("refcoord_scaling", "all")
    if defines:
        mdp_file.add_or_update("define", " ".join("-D" + d for d in defines))
    return mdp_file


def log_time_span(name: str, nsteps: int, dt: float, nstout: int | None = None) -> None:
    span_ps = nsteps * dt
    frames = f", {nsteps // nstout} frames" if nstout else ""
    logger.info("%s: %g ps (%g ns) with dt=%g ps%s", name, span_ps, span_ps / 1000, dt, frames)
