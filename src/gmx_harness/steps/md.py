"""Energy minimization and MD steps (atomistic, Martini, AWH, BAR)."""

import dataclasses
import enum
from typing import override

from pydantic.dataclasses import dataclass

from .. import mdp
from ..scripts import generate_xtc_script, grompp_script, mdrun_script
from .base import Calculation, apply_common_mdp, check_common, default_file_content, log_time_span, logger

MDPExtra = dict[str, str | int | float]


def _set_continuation(mdp_file: mdp.MDParameters, continuation: bool | None) -> None:
    """``continuation = yes`` means: do not re-apply constraints to the starting structure."""
    if continuation is not None:
        mdp_file.add_or_update("continuation", "yes" if continuation else "no")


def _md_files(
    mdp_file: mdp.MDParameters,
    *,
    strict: bool,
    maxwarn: int,
    restraint: bool,
    index_file: str | None = None,
    single_domain: bool = False,
    xtc: bool = True,
) -> dict[str, str]:
    mdp_file.ensure_valid(strict=strict)
    files = {
        "setting.mdp": mdp_file.export(),
        "grommp.sh": grompp_script(maxwarn=maxwarn, restraint=restraint, index_file=index_file),
        # Coarse-grained boxes are small; a single domain avoids DD "box smaller
        # than 2*cell" errors while OpenMP still uses every core.
        "mdrun.sh": mdrun_script(extra_args=["-ntmpi", "1"] if single_domain else None),
    }
    if xtc:
        files["generate_xtc.sh"] = generate_xtc_script()
    return files


@dataclass(kw_only=True)
class EM(Calculation):
    """
    Steepest-descent energy minimization (atomistic template ``mdp.EM_MDP``).

    Attributes:
        nsteps: maximum number of minimization steps.
        emtol: stop when the maximum force is below this (kJ/mol/nm).
        defines: preprocessor defines without "-D" (e.g. ["POSRES"]).
        maxwarn: grompp -maxwarn. Keep 0 unless you know which warning you accept.
        useRestraint: pass input.gro as the position-restraint reference (-r).
        additional_mdp_parameters: any extra/overriding mdp options.
        strict_mdp: raise on mdp validation errors (False: only warn).
    """

    nsteps: int = 3000
    emtol: float = 300
    calculation_name: str = "em"
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = 0
    useRestraint: bool = False
    additional_mdp_parameters: MDPExtra = dataclasses.field(default_factory=dict)
    strict_mdp: bool = True

    def __post_init__(self) -> None:
        check_common(self.calculation_name, self.defines, self.maxwarn)

    @override
    def generate(self) -> dict[str, str]:
        mdp_file = mdp.MDParameters(mdp.EM_MDP).add_or_update("nsteps", self.nsteps).add_or_update("emtol", self.emtol)
        if self.defines:
            mdp_file.add_or_update("define", " ".join("-D" + d for d in self.defines))
        for key, value in self.additional_mdp_parameters.items():
            mdp_file.add_or_update(key, str(value))
        return _md_files(
            mdp_file, strict=self.strict_mdp, maxwarn=self.maxwarn, restraint=self.useRestraint, xtc=False
        )


class MDType(enum.Enum):
    """Thermostat/barostat preset for ``MD``."""

    v_rescale_c_rescale = 1  # NPT: v-rescale + c-rescale (default)
    v_rescale_only_nvt = 2  # NVT: v-rescale, no pressure coupling
    nose_hoover_parinello_rahman = 3  # NPT production: Nose-Hoover + Parrinello-Rahman (template: continuation=yes)
    berendsen = 4  # not implemented


_MD_TEMPLATES: dict[MDType, dict[str, mdp.MDPValue]] = {
    MDType.v_rescale_c_rescale: mdp.V_RESCALE_C_RESCALE_MDP,
    MDType.v_rescale_only_nvt: mdp.V_RESCALE_ONLY_NVT_MDP,
    MDType.nose_hoover_parinello_rahman: mdp.NOSE_HOOVER_PARINELLO_RAHMAN_MDP,
}


@dataclass(kw_only=True)
class MD(Calculation):
    """
    Atomistic MD run from one of the ``MDType`` templates (dt = 0.002 ps).

    Attributes:
        type: ensemble preset (MDType).
        calculation_name: directory/step name ([A-Za-z0-9_.-]).
        nsteps: number of steps (time = nsteps * dt).
        nstout: output interval for trr/edr (nstxout/nstvout/nstfout/nstenergy).
        gen_vel: "yes" to generate velocities at ``temperature``, else "no".
        temperature: ref_t / gen_temp in K.
        defines: preprocessor defines without "-D".
        maxwarn: grompp -maxwarn.
        useRestraint: -r input.gro and refcoord_scaling=all.
        useSemiisotropic: semiisotropic pressure coupling (NPT types only).
        additional_mdp_parameters: any extra/overriding mdp options (applied last-but-one; flexible escape hatch).
        continuation: mdp ``continuation`` (True/False -> yes/no). None keeps the template value.
        strict_mdp: raise on mdp validation errors (False: only warn).
    """

    type: MDType
    calculation_name: str
    nsteps: int = 10000
    nstout: int = 1000
    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = 0
    useRestraint: bool = False
    useSemiisotropic: bool = False
    additional_mdp_parameters: MDPExtra = dataclasses.field(default_factory=dict)
    continuation: bool | None = None
    strict_mdp: bool = True

    def __post_init__(self) -> None:
        check_common(self.calculation_name, self.defines, self.maxwarn, self.temperature, self.gen_vel)
        if self.type is MDType.berendsen:
            raise NotImplementedError("MDType.berendsen is not implemented")
        if self.nstout <= 0:
            raise ValueError("nstout must be > 0")
        dt = mdp.MDParameters(_MD_TEMPLATES[self.type])
        for k, v in self.additional_mdp_parameters.items():
            dt.add_or_update(k, v)
        log_time_span(self.calculation_name, self.nsteps, dt.dt(), self.nstout)

    @override
    def generate(self) -> dict[str, str]:
        mdp_file = (
            mdp.MDParameters(_MD_TEMPLATES[self.type])
            .add_or_update("nsteps", self.nsteps)
            .add_or_update("nstxout", self.nstout)
            .add_or_update("nstvout", self.nstout)
            .add_or_update("nstfout", self.nstout)
            .add_or_update("nstenergy", self.nstout)
            .add_or_update("gen_vel", self.gen_vel)
            .add_or_update("ref_t", self.temperature)
            .add_or_update("gen_temp", self.temperature)
        )
        _set_continuation(mdp_file, self.continuation)
        apply_common_mdp(
            mdp_file, defines=self.defines, additional=self.additional_mdp_parameters, restraint=self.useRestraint
        )
        if self.useSemiisotropic:
            if self.type is MDType.v_rescale_only_nvt:
                raise ValueError("useSemiisotropic makes no sense for an NVT run (no pressure coupling)")
            mdp_file.add_or_update("pcoupltype", "semiisotropic")
            mdp_file.add_or_update("ref_p", " ".join([str(mdp_file.get("ref_p"))] * 2))
            mdp_file.add_or_update("compressibility", " ".join([str(mdp_file.get("compressibility"))] * 2))
        return _md_files(
            mdp_file, strict=self.strict_mdp, maxwarn=self.maxwarn, restraint=self.useRestraint)


@dataclass(kw_only=True)
class MartiniEM(Calculation):
    """Energy minimization for a Martini 3 coarse-grained system (``mdp.MARTINI_MIN_MDP``, -ntmpi 1)."""

    nsteps: int = 5000
    emtol: float = 100.0
    calculation_name: str = "em"
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = 10  # CG runs routinely emit benign grompp notes
    useRestraint: bool = False
    additional_mdp_parameters: MDPExtra = dataclasses.field(default_factory=dict)
    strict_mdp: bool = True

    def __post_init__(self) -> None:
        check_common(self.calculation_name, self.defines, 0)

    @override
    def generate(self) -> dict[str, str]:
        mdp_file = (
            mdp.MDParameters(mdp.MARTINI_MIN_MDP)
            .add_or_update("nsteps", self.nsteps)
            .add_or_update("emtol", self.emtol)
        )
        if self.defines:
            mdp_file.add_or_update("define", " ".join("-D" + d for d in self.defines))
        for key, value in self.additional_mdp_parameters.items():
            mdp_file.add_or_update(key, str(value))
        return _md_files(
            mdp_file,
            strict=self.strict_mdp,
            maxwarn=self.maxwarn,
            restraint=self.useRestraint,
            single_domain=True,
            xtc=False,
        )


@dataclass(kw_only=True)
class MartiniMD(Calculation):
    """
    Martini 3 MD (v-rescale / c-rescale NPT, dt = 0.02 ps, -ntmpi 1).
    ``useSemiisotropic`` switches to semiisotropic pressure coupling (membranes/interfaces).
    """

    calculation_name: str
    nsteps: int = 500000
    nstout: int = 5000
    gen_vel: str = "yes"
    temperature: float = 300
    dt: float = 0.02
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = 10
    useRestraint: bool = False
    useSemiisotropic: bool = False
    additional_mdp_parameters: MDPExtra = dataclasses.field(default_factory=dict)
    continuation: bool | None = None
    strict_mdp: bool = True

    def __post_init__(self) -> None:
        check_common(self.calculation_name, self.defines, 0, self.temperature, self.gen_vel)
        if self.nstout <= 0:
            raise ValueError("nstout must be > 0")
        log_time_span(self.calculation_name, self.nsteps, self.dt, self.nstout)

    @override
    def generate(self) -> dict[str, str]:
        mdp_file = (
            mdp.MDParameters(mdp.MARTINI_MD_MDP)
            .add_or_update("dt", self.dt)
            .add_or_update("nsteps", self.nsteps)
            .add_or_update("nstxout", self.nstout)
            .add_or_update("nstvout", self.nstout)
            .add_or_update("nstfout", self.nstout)
            .add_or_update("nstenergy", self.nstout)
            .add_or_update("nstlog", self.nstout)
            .add_or_update("nstxout-compressed", self.nstout)
            .add_or_update("gen_vel", self.gen_vel)
            .add_or_update("ref_t", self.temperature)
            .add_or_update("gen_temp", self.temperature)
        )
        _set_continuation(mdp_file, self.continuation)
        apply_common_mdp(
            mdp_file, defines=self.defines, additional=self.additional_mdp_parameters, restraint=self.useRestraint
        )
        if self.useSemiisotropic:
            mdp_file.add_or_update("Pcoupltype", "semiisotropic")
            mdp_file.add_or_update("ref_p", " ".join([str(mdp_file.get("ref_p"))] * 2))
            mdp_file.add_or_update("compressibility", " ".join([str(mdp_file.get("compressibility"))] * 2))
        return _md_files(
            mdp_file,
            strict=self.strict_mdp,
            maxwarn=self.maxwarn,
            restraint=self.useRestraint,
            single_domain=True,
        )


@dataclass(kw_only=True)
class AWH(Calculation):
    """
    Accelerated weight histogram run. Base mdp: bundled ``awh_setting.mdp`` or
    ``settingfile_abs_path``; nsteps/output/temperature fields are overwritten
    by the attributes. Needs an index file (``index_file``) with the pull groups.
    """

    calculation_name: str = "awh"
    settingfile_abs_path: str | None = None
    nsteps: int = 500000
    nstout: int = 5000
    nstout_energy: int | None = None
    gen_vel: str = "no"
    temperature: float = 300.0
    defines: list[str] = dataclasses.field(default_factory=list)
    additional_mdp_parameters: MDPExtra = dataclasses.field(default_factory=dict)
    maxwarn: int = 0
    useRestraint: bool = False
    index_file: str = "index.ndx"
    continuation: bool | None = None
    strict_mdp: bool = True

    def __post_init__(self) -> None:
        check_common(self.calculation_name, self.defines, self.maxwarn, self.temperature, self.gen_vel)
        if self.settingfile_abs_path is not None:
            logger.warning(
                "%s: using %s as base mdp; nsteps, nstxout, ref_t, ... are overwritten by the class attributes",
                self.calculation_name,
                self.settingfile_abs_path,
            )
        log_time_span(self.calculation_name, self.nsteps, 0.002, self.nstout)

    @override
    def generate(self) -> dict[str, str]:
        if self.settingfile_abs_path is not None:
            base = mdp.MDParameters.from_file(self.settingfile_abs_path)
        else:
            base = mdp.MDParameters.from_text(default_file_content("awh_setting.mdp"))
        mdp_file = (
            base.add_or_update("nsteps", self.nsteps)
            .add_or_update("nstxout", self.nstout)
            .add_or_update("nstvout", self.nstout)
            .add_or_update("nstfout", self.nstout)
            .add_or_update("nstenergy", self.nstout_energy if self.nstout_energy is not None else self.nstout)
            .add_or_update("gen_vel", self.gen_vel)
            .add_or_update("ref_t", self.temperature)
            .add_or_update("gen_temp", self.temperature)
        )
        if self.defines:
            mdp_file.add_or_update("define", " ".join("-D" + d for d in self.defines))
        _set_continuation(mdp_file, self.continuation)
        for key, value in self.additional_mdp_parameters.items():
            mdp_file.add_or_update(key, str(value))
        return _md_files(
            mdp_file,
            strict=self.strict_mdp,
            maxwarn=self.maxwarn,
            restraint=self.useRestraint,
            index_file=self.index_file or None,
        )


@dataclass(kw_only=True)
class BarMethod(Calculation):
    """
    One lambda window series for free-energy perturbation (BAR), base ``barmethod.mdp``.
    All *_lambdas lists must have the same length.
    """

    type: MDType
    calculation_name: str
    nsteps: int = 10000
    nstout: int = 1000
    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = 0
    useRestraint: bool = False
    useSemiisotropic: bool = False
    additional_mdp_parameters: MDPExtra = dataclasses.field(default_factory=dict)

    vdw_lambdas: list[float]
    coul_lambdas: list[float]
    bonded_lambdas: list[float]
    restraint_lambdas: list[float]
    mass_lambdas: list[float]
    temperature_lambdas: list[float]

    couple_moltype: str = "System"
    couple_lamda0: str = "vdw"
    couple_lamda1: str = "none"
    nstdhdl: int = 100
    continuation: bool | None = None
    strict_mdp: bool = True

    def __post_init__(self) -> None:
        check_common(self.calculation_name, self.defines, self.maxwarn, self.temperature, self.gen_vel)
        n = len(self.vdw_lambdas)
        if n == 0:
            raise ValueError("vdw_lambdas is empty")
        for attr in ("coul_lambdas", "bonded_lambdas", "restraint_lambdas", "mass_lambdas", "temperature_lambdas"):
            if len(getattr(self, attr)) != n:
                raise ValueError(f"{attr} is not the same length as vdw_lambdas")
        log_time_span(self.calculation_name, self.nsteps, 0.002)

    @override
    def generate(self) -> dict[str, str]:
        def join(values: list[float]) -> str:
            return " ".join(str(v) for v in values)

        mdp_file = (
            mdp.MDParameters.from_text(default_file_content("barmethod.mdp"))
            .add_or_update("nsteps", self.nsteps)
            .add_or_update("nstxout", self.nstout)
            .add_or_update("nstvout", self.nstout)
            .add_or_update("nstfout", self.nstout)
            .add_or_update("nstenergy", self.nstout)
            .add_or_update("gen_vel", self.gen_vel)
            .add_or_update("ref_t", self.temperature)
            .add_or_update("gen_temp", self.temperature)
            .add_or_update("vdw_lambdas", join(self.vdw_lambdas))
            .add_or_update("coul_lambdas", join(self.coul_lambdas))
            .add_or_update("bonded_lambdas", join(self.bonded_lambdas))
            .add_or_update("restraint_lambdas", join(self.restraint_lambdas))
            .add_or_update("mass_lambdas", join(self.mass_lambdas))
            .add_or_update("temperature_lambdas", join(self.temperature_lambdas))
            .add_or_update("couple-moltype", self.couple_moltype)
            .add_or_update("couple-lambda0", self.couple_lamda0)
            .add_or_update("couple-lambda1", self.couple_lamda1)
            .add_or_update("nstdhdl", self.nstdhdl)
        )
        if self.defines:
            mdp_file.add_or_update("define", " ".join("-D" + d for d in self.defines))
        if self.useRestraint:
            mdp_file.add_or_update("refcoord_scaling", "all")
        _set_continuation(mdp_file, self.continuation)
        for key, value in self.additional_mdp_parameters.items():
            mdp_file.add_or_update(key, str(value))
        return _md_files(
            mdp_file, strict=self.strict_mdp, maxwarn=self.maxwarn, restraint=self.useRestraint)
