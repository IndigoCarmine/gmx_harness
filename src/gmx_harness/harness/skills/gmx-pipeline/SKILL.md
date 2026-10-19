---
name: gmx-pipeline
description: Use gmx_harness to design a GROMACS MD pipeline (EM → NVT → NPT → production, solvation, residue removal, box resizing, Martini, AWH, BAR) and generate the step directories and bash scripts safely. Keywords: GROMACS, gmx, MD pipeline, equilibration, solvation, mdp, run.sh, build_plan
---

# Generating a GROMACS pipeline (gmx_harness)

## Principles
- gmx_harness **only generates files**. Never run gmx or run.sh (see AGENTS.md).
- Order: gather requirements → design → `save_json` → `build_plan` → show `summary()`/`preview()` → approval → `write()`.
- Exact signatures, defaults and docstrings of the public API, and the table of check codes: `api.md` next to this
  file (generated from the code). Grep it for the name you need instead of reading it whole, and use it rather than
  guessing an argument or importing something that is not listed there.

## 1. Ask for these (do not guess them)
- Starting structure (.gro), topology (topo.top), and the .itp files it includes
- Atomistic or Martini (CG)
- Temperature, pressure, whether restraints are needed (the name used in `#ifdef POSRES`), production length (ns)
- Solvent (MCH / SPC216 water / none)
- Where it will run (the scripts are machine-independent; GROMACS is chosen there with `GMX`, threads/GPU with `MDRUN_ARGS`)

## 2. Step reference
| Purpose | Class | Notes |
|---|---|---|
| Energy minimization | `EM(nsteps, emtol, defines)` | First step |
| NVT equilibration | `MD(type=MDType.v_rescale_only_nvt, ...)` | `gen_vel="yes"` |
| NPT equilibration / production | `MD(type=MDType.v_rescale_c_rescale, gen_vel="no")` | c-rescale is safe even early in equilibration |
| NPT production (NH+PR) | `MD(type=MDType.nose_hoover_parinello_rahman, gen_vel="no")` | Only for already equilibrated systems. The template has continuation=yes; change it with `continuation=False` |
| Martini | `MartiniEM`, `MartiniMD(dt=0.02)` | Uses -ntmpi 1 |
| MCH solvation | `SolvationMCH(scale=0.57)` / `Solvation.from_cell_size(...)` | Edits topo.top automatically |
| Water | `SolvationSCP216()` | topo.top must already include a water model |
| Residue removal | `RemoveResidue(name, resname)` | Removes only that residue from topo.top |
| Box resizing | `ResizeBox(name, x, y, z, remove_resname=None)` | nm |
| Adding files | `AddFiles(name, {"x.itp": text})` | .top/.itp files are carried to later steps |
| Anything else | `RawShellStep(..., allow_unsafe=True)` | **Requires human approval** |

- Time = `nsteps × dt` (atomistic dt = 0.002 ps → 1 ns = 500,000 steps; Martini uses 0.02 ps).
- `calculation_name` may only use `[A-Za-z0-9_.+-]` and must be unique.
- Any mdp option can be added or overridden with `additional_mdp_parameters={"key": value}`. It is checked when the files are generated.
- Write `defines` without `-D` (`["POSRES"]`).
- PLUMED and other mdrun options: `MD(..., mdrun_args=["-plumed", "plumed.dat"], extra_files={"plumed.dat": text})`.
- PLUMED input from a template with atom selections by fragment label: `gmx_harness.preprocess_file("metad.plumed.in", Layout(MoleculeLabels.from_gro(labeled_gro), nmol, nros), defines)` (the template's file path) (`#define/#for/#include`, `{expr}`, `@sel(disk=, mol=, res=, name=, heavy=)`). Show the expanded text to the human before using it.
- Inputs: `extra_inputs={"topo.top": "MOL_fixed.top", "MOL_hbond.itp": "..."}` (first step, renaming allowed); `step_inputs={"metad": {"index.ndx": "MOL.ndx"}}` for a later step.

## 3. Generate
```python
from gmx_harness import *
plan = build_plan(steps, "start.gro", "work", extra_inputs=["topo.top", "mol.itp"])
print(plan.summary())
print(plan.file("1_nvt/setting.mdp"))   # show the key mdp files to the human
print(plan.preview())
```
- `print(plan.preview())` also lists the design checks (`checks:`, each as `CODE <where>: message`). For an error,
  fix the design; if the human says the issue is acceptable, pass `waive={"CODE@<where pattern>": "their reason"}`
  (only the issues at that place, e.g. `"L004@1_em"`: an fnmatch pattern on the step directory or file printed after
  the code) or `waive={"CODE": "their reason"}` (every issue of that code). Never on your own, never `checks=False`.
  A waiver that matches nothing is `W001` (warn). Format and the `where` of each code: gmx-troubleshoot skill.
  Errors (block the write): `L003` input.gro vs topo.top, `L002` a step needs an index file that is not planned,
  `L004` maxwarn > 0, `L005` strict_mdp=False, `L006` raw shell step, `S011` a molecule type of topo.top not found
  locally, most `P0xx` of the PLUMED input. Warnings (shown, do not block): `L001` PLUMED PACE/PRINT strides,
  `P004` SIGMA small for the grid, `P010` UNITS, `P007` for a bias on raw single atoms (a single-atom COM is an error)
  (see `gmx_harness.checks.CODES`).
- PLUMED from a template: prefer `text, report = gmx_harness.expand_template("metad.plumed.in", layout, defines, symmetry=nros)`
  (the first argument is the template's file path, not its text); it also reports defines the template never uses
  (`P009`, typos). `report.enforce(WAIVE)` before using `text`.
- `require_preflight=True` makes run.sh refuse to start until a human has run the preflight (grompp with each step's
  maxwarn / plumed driver) and it has written `preflight.ok` (sha256sum of the planned files). The library does not
  write `preflight.ok`; a preflight script of the workspace does (gmx_template: `3md_planning/preflight.py`, a dry
  run of every step: grompp without mdrun, the solvation steps really run, plumed driver on the structure each step
  receives). The run.sh files then also need `sha256sum`.
- If `PlanPreview.conflicts` is not empty, report it and ask the human how to proceed.
  - To add only new steps: `plan.write(OverwritePolicy.SKIP_EXISTING)`
  - To regenerate after changing parameters: `plan.write(OverwritePolicy.REPLACE_GENERATED)`
    (files a human has edited are refused. Do not force them with `force_modified`.)
- Once approved, call `plan.write()`. Tell the human how to run it: `cd work && bash run.sh`
  (optionally with `GMX=gmx_mpi MDRUN_ARGS="-ntomp 8"` on the running machine).
  Creating a `freeze` file in a step directory stops that step.
  Re-running `run.sh` resumes (finished steps are skipped, an interrupted MD step continues from output.cpt).
  Extending a finished run is `bash extend.sh <ps>` in that step (the human runs it, like everything that calls gmx).
- Useful options: `carry=("*.top", "*.itp", "*.ndx")` to hand an index file down the pipeline;
  `MD(..., plumed=text)` for PLUMED (restarts add RESTART automatically);
  `molecule_atoms` / `write_ndx` for index groups, `prepare_topology` for the input topology,
  `write_job_scripts` for sbatch files + submit.sh (never submit them yourself).

## 4. Many structures in bulk
Run `build_plan` per structure into `work/<structure>`, then create a per-step batch runner with
`generate_stepbystep_runfile(structures, [(step_dir, parallel?)...], "work")`.
