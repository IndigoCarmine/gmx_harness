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

Design checks (``gmx_harness.checks``): ``build_plan`` checks the planned pipeline, and the
stage scripts of a workspace hand values on with ``record_facts`` / ``expect``; errors
stop the script unless waived with a reason, per code or per code and place
(``report.enforce(WAIVE)``, ``WAIVE = {"S004@*fiber*": "why"}``).

Not imported here: ``gmx_harness.relax`` (OpenMM soft-core pre-relaxation; heavy import)
and ``gmx_harness.analysis`` (MDAnalysis trajectory analysis).
"""

__version__ = "0.1.1"

from .agent_harness import install_skills
from .build import (
    Assembly,
    make_half_rosette2,
    make_oligorosette,
    make_rosette,
    make_rosette2,
    pre_coordinate,
    precoordinate2,
    save_assembly,
)
from .geometry import centroid, count_contacts, find_intermolecular_pairs, min_distance, plane_normal
from .io import GroAtom, GroFile, XvgData, load_xvg, parse_xvg
from .io.mol import gro_from_mol, save_mol
from .itp import format_intermolecular_bonds, generate_inermolecular_interactions, write_intermolecular_bonds
from .index import format_ndx, molecule_atoms, write_ndx
from .jobs import render_job_script, write_job_scripts
from .mdp import MDParameters, MDPValidationError, ValidationIssue, validate_mdp_text
from .checks import CODES, HarnessCheckError, Report, check_fresh, expect, expand_template, record_facts
from .plumed import Layout, MoleculeLabels, PreprocessError, preprocess, preprocess_file
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
    step_dir_names,
    write_tree,
)
from .safety import UnsafeNameError, UnsafeOperationError
from .scripts import gmx_command
from .serialization import from_json, load_json, save_json, to_json
from .topfile import add_conditional_include, atom_types, prepare_topology, set_molecule_count
from .transform import RigidTransform
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
    "step_dir_names",
    "write_tree",
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
    # structure building
    "Assembly",
    "pre_coordinate",
    "precoordinate2",
    "make_rosette",
    "make_rosette2",
    "make_half_rosette2",
    "make_oligorosette",
    "RigidTransform",
    "save_assembly",
    # measuring built structures
    "min_distance",
    "count_contacts",
    "find_intermolecular_pairs",
    "centroid",
    "plane_normal",
    # files
    "GroAtom",
    "GroFile",
    "XvgData",
    "load_xvg",
    "parse_xvg",
    "generate_inermolecular_interactions",
    "format_intermolecular_bonds",
    "write_intermolecular_bonds",
    "save_mol",
    "gro_from_mol",
    # AI harness
    "install_skills",
    # index / topology preparation / batch jobs
    "molecule_atoms",
    "format_ndx",
    "write_ndx",
    "set_molecule_count",
    "atom_types",
    "add_conditional_include",
    "prepare_topology",
    "render_job_script",
    "write_job_scripts",
    # PLUMED templates
    "Layout",
    "MoleculeLabels",
    "PreprocessError",
    "preprocess",
    "preprocess_file",
    # design checks (more in gmx_harness.checks)
    "CODES",
    "HarnessCheckError",
    "Report",
    "record_facts",
    "expect",
    "check_fresh",
    "expand_template",
    # safety
    "UnsafeNameError",
    "UnsafeOperationError",
]
