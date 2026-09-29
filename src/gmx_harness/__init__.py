"""
gmx_harness: build GROMACS simulation pipelines as files + bash scripts, safely.

The library never runs GROMACS. It writes numbered step directories with an
``.mdp``, ``grommp.sh``/``mdrun.sh`` and a top-level ``run.sh`` that a human
(or a job scheduler) executes on a machine with GROMACS.

Quick start::

    from gmx_harness import EM, MD, MDType, build_plan

    plan = build_plan(
        [EM(), MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=50000),
         MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", nsteps=500000, gen_vel="no")],
        input_gro="start.gro", working_dir="work", extra_inputs=["topo.top"],
    )
    print(plan.preview())
    plan.write()

Not imported here: ``gmx_harness.relax`` (OpenMM soft-core pre-relaxation; heavy import)
and ``gmx_harness.analysis`` (MDAnalysis trajectory analysis).
"""

__version__ = "0.1.0"

from .agent_harness import install_skills
from .io import GroAtom, GroFile, XvgData, load_xvg, parse_xvg
from .itp import generate_inermolecular_interactions
from .mdp import MDParameters, MDPValidationError, ValidationIssue, validate_mdp_text
from .pipeline import (
    OverwritePolicy,
    OverwriteType,
    Plan,
    PlanConflictError,
    PlanPreview,
    build_plan,
    generate_batch_execution_script,
    generate_stepbystep_runfile,
    launch,
)
from .safety import UnsafeNameError, UnsafeOperationError
from .scripts import DEFAULT_CONFIG, GmxConfig, gmx_command
from .serialization import from_json, load_json, save_json, to_json
from .steps import (
    AWH,
    EM,
    MD,
    AddFiles,
    BarMethod,
    Calculation,
    FileControl,
    MartiniEM,
    MartiniMD,
    MDType,
    RawShellStep,
    RemoveResidue,
    ResizeBox,
    RuntimeSolvation,
    Solvation,
    SolvationMCH,
    SolvationSCP216,
    molecules_to_fill,
)

__all__ = [
    "__version__",
    # steps
    "Calculation",
    "EM",
    "MD",
    "MDType",
    "MartiniEM",
    "MartiniMD",
    "AWH",
    "BarMethod",
    "Solvation",
    "RuntimeSolvation",
    "SolvationSCP216",
    "SolvationMCH",
    "molecules_to_fill",
    "RemoveResidue",
    "ResizeBox",
    "AddFiles",
    "RawShellStep",
    "FileControl",
    # pipeline
    "build_plan",
    "Plan",
    "PlanPreview",
    "PlanConflictError",
    "OverwritePolicy",
    "launch",
    "OverwriteType",
    "generate_stepbystep_runfile",
    "generate_batch_execution_script",
    "GmxConfig",
    "DEFAULT_CONFIG",
    "gmx_command",
    # serialization
    "save_json",
    "load_json",
    "to_json",
    "from_json",
    # mdp
    "MDParameters",
    "MDPValidationError",
    "ValidationIssue",
    "validate_mdp_text",
    # files
    "GroAtom",
    "GroFile",
    "XvgData",
    "load_xvg",
    "parse_xvg",
    "generate_inermolecular_interactions",
    # AI harness
    "install_skills",
    # safety
    "UnsafeNameError",
    "UnsafeOperationError",
]
