# gmx_harness

English | [日本語](README.ja.md)

A library that **generates** GROMACS simulation pipelines (EM → NVT → NPT → production, solvation, box operations, Martini, AWH, BAR) as a set of files and bash scripts. It comes bundled with the harness an AI agent (and also human) needs to handle it safely.
It was split out of the `gromacs` package in mylibs (`yagaiG-libs`).

- **It never runs anything**: the library does not call `gmx`. A person or the job scheduler runs the generated `run.sh`.
  It has no dependency on GROMACS or PLUMED; dependencies are pip packages only (pydantic, numpy, OpenMM, MDAnalysis, pandas, openpyxl).
- **Flexibility**: any mdp option can be set via `additional_mdp_parameters`. Custom steps are possible by subclassing `Calculation`,
  and arbitrary commands can be run with `RawShellStep` (explicit opt-in required).
- **Safety**:
  - `plan → preview → write` flow: all conflicts are checked before anything is written
    (the only files written without a preview are the small check records `<file>.facts.json`
    from `checks.record_facts` and `checks.json` from `Report.enforce(out_dir=...)`)
  - Design checks (`gmx_harness.checks`) run by default; an error that is not waived with a reason refuses the write
  - Only files gmx_harness generated itself (hashes recorded in `.gmx_harness_manifest.json`) are overwritten or deleted; files a person has edited are refused
  - No interactive `input()` anywhere (it cannot hang an agent)
  - Step names, file names, defines and residue names are validated; anything that goes into a script is `shlex.quote`d
  - mdp validation (leftover placeholders, numeric ranges, group counts, line injection, ...)
  - Destructive or unchecked operations require `allow_unsafe=True` / `confirm=True` / `force_modified=True`

## Installation

```bash
pip install "gmx-harness @ git+<URL of this repository>"
```

## Usage

```python
from gmx_harness import EM, MD, MDType, OverwritePolicy, build_plan, save_json

steps = [
    EM(),
    MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=50_000,
       useRestraint=True, defines=["POSRES"]),
    MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", nsteps=500_000, gen_vel="no"),
]
save_json(steps, "pipeline.json")

plan = build_plan(steps, "start.gro", "work", extra_inputs=["topo.top", "mol.itp"])
# renaming / files for later steps:
# build_plan(..., extra_inputs={"topo.top": "MOL_fixed.top"}, step_inputs={"npt": {"index.ndx": "MOL.ndx"}})
print(plan.summary())
print(plan.preview())
plan.write()                                     # OverwritePolicy.ERROR (default)
# plan.write(OverwritePolicy.REPLACE_GENERATED)  # regenerate after changing parameters
```

Generated layout:

```
work/
  run.sh                     # runs every step in order, stops on the first failure
  .gmx_harness_manifest.json
  0_em/   setting.mdp grommp.sh mdrun.sh run.sh copy.sh input.gro topo.top mol.itp
  1_nvt/  ...
  2_npt/  ...
```

Run it: `cd work && bash run.sh`. Placing a `freeze` file in a step directory stops that step.

Re-running and continuing:
- Running `run.sh` again skips finished steps. A finished step never overwrites the inputs of a
  later step that has also finished (safe for resubmission and copied/cloned trees).
- An interrupted MD step continues from `output.cpt` without re-running grompp.
- Extending a run: `cd work/6_md_prod && bash extend.sh 200000` (ps, via `gmx convert-tpr -extend`),
  then `bash run.sh` again.
- `generate_xtc.sh [GROUP] [OUTPUT]`, e.g. `bash generate_xtc.sh MOL mol_whole.xtc`.
- Files handed from step to step: `build_plan(..., carry=("*.top", "*.itp", "*.ndx"))` (default: top/itp).
- PLUMED: `MD(..., plumed=text)` writes plumed.dat and runs `-plumed plumed.dat`; when continuing from a
  checkpoint it adds `RESTART` automatically (HILLS/COLVAR are appended to).
- A step that ends without `output.gro` stops the top-level `run.sh` with an error.

The scripts contain nothing specific to the machine that generated them; GROMACS is looked up when they run,
so the directory can be copied to a cluster as is. Environment variables on the running machine:

| Variable | Meaning | Default |
|---|---|---|
| `GMX` | GROMACS command | first of `gmx_d`, `gmx_mpi`, `gmx` on PATH |
| `MDRUN_ARGS` | extra arguments for every `mdrun` (e.g. `"-ntomp 8 -gpu_id 0"`) | none |

Besides GROMACS, the scripts only need bash and awk (topology edits after solvation are done with awk);
with `require_preflight=True` they also need `sha256sum` (GNU coreutils).
Load GROMACS the usual way before running (`module load gromacs`, `source .../GMXRC`, job script, ...).

### Steps

| Class | Purpose |
|---|---|
| `EM`, `MartiniEM` | Energy minimization |
| `MD(type=MDType.*)`, `MartiniMD` | MD (NVT / NPT / NH+PR) |
| `AWH`, `BarMethod` | Free energy |
| `Solvation`, `SolvationMCH`, `SolvationSCP216`, `RuntimeSolvation` (deprecated) | Solvation |
| `RemoveResidue`, `ResizeBox`, `AddFiles` | Typed file operations |
| `RawShellStep` (`allow_unsafe=True` required) | Arbitrary bash |

Other: `MDParameters` (mdp editing and validation), `GroFile` (.gro/.ndx/.xyz/.pdb), `load_xvg`,
`generate_inermolecular_interactions`, `gmx_harness.relax.relax` (OpenMM), `gmx_harness.analysis` (MDAnalysis).

### Index files, topology preparation and batch jobs

```python
from gmx_harness import molecule_atoms, write_ndx, prepare_topology, write_job_scripts

fiber = molecule_atoms(169, range(216))                      # 1-based atoms of molecules 0..215
write_ndx("MOL_fiber.ndx", {"Fiber1": fiber, "fiberA": fiber})
prepare_topology("MOL.top", "MOL_fixed.top", nmols=216, itp_name="MOL_hbond.itp")   # count + #ifdef INTER include
write_job_scripts("calc", ["MOL_fiber_rot_+10"], open("Gromacs.sbatch").read())    # {JOB_NAME}/{SCRIPT} template
```

`write_job_scripts` renders your own job template into every system directory and writes `submit.sh` /
`submit_restart.sh` (`sbatch -d singleton`); it only writes files and refuses to replace edited ones.

### PLUMED input from templates

```python
from gmx_harness import MD, MDType, Layout, MoleculeLabels, preprocess_file

layout = Layout(MoleculeLabels.from_gro("MOL_labeled.gro"), nmol=60, nros=6)  # residue column = fragment labels
text = preprocess_file("metad_twist.plumed.in", layout, {"NDISK": 10, "BIAS_IFACE": 4})
metad = MD(type=MDType.v_rescale_c_rescale, calculation_name="metad", gen_vel="no", plumed=text)
```

Templates are PLUMED input plus `#define` / `#for v in a..b` / `#endfor` / `#include`, `{expr}` and
`@sel(disk=, mol=, res=, name=, heavy=)` (atom ranges of the selected fragments). Expressions run in a
sandbox (no dunders, lambdas or imports) and `#include` cannot leave the template's directory.

### Design checks

`build_plan` runs `gmx_harness.checks` by default (pure Python, nothing is executed): input.gro vs topo.top,
maxwarn, `strict_mdp=False`, raw shell steps, PLUMED input and strides, and so on. Every issue has a stable
code (`L003`, `P002`, ...). An error makes `preview().ok` False and `write()` refuses, unless it is waived
with a reason:

```python
plan = build_plan(steps, "start.gro", "work", extra_inputs=["topo.top"],
                  waive={"L004": "grompp warning about ... is expected here"})
```

A waiver needs a known code and a non-empty reason. `expand_template` checks an expanded PLUMED template
(including defines the template ignores), `check_tree` reads logs and COLVAR after a run, and
`require_preflight=True` makes run.sh refuse to start until `preflight.ok` (written by the workspace's own
preflight script after running grompp/plumed) exists and still matches. All codes: `gmx_harness.checks.CODES`
(also listed in `harness/llm_docs/api.md`).

### Building structures from .gro

```python
from gmx_harness import Assembly, GroFile, make_oligorosette, make_rosette2, precoordinate2

mono = precoordinate2(GroFile.from_gro_file("MOL.gro"), 1, 2, 3)   # atom numbers: top, NH, O
ring = make_rosette2(mono, n=6, size=0.35)                          # Assembly of 6 monomers
for i, m in enumerate(ring):
    m.set_residue_number(i + 1)
ring_gro = ring.to_gro(renumber=True)
fiber = make_oligorosette(ring_gro, n=10, length=0.35, angle=10.0)  # stacked rings (slip= for a helix)
fiber.to_gro(box=(10, 10, 3.5), renumber=True).save_gro("fiber.gro")
```

Also `pre_coordinate`, `make_rosette`, `make_half_rosette2`, `Assembly.translate/rotate`, and
`gmx_harness.build.rotation` / `align` (3x3 matrices; `GroFile.rotate` also accepts scipy `Rotation`).
Ported from mylibs' `pre_coordinator` / `rosette_maker`; coordinates agree with them to ~1e-16 nm.



## Migrating from mylibs

| mylibs | gmx_harness |
|---|---|
| `gromacs.calculation.*` | `gmx_harness.*` (same class names) |
| `gromacs.mdp` | `gmx_harness.mdp` (`check(key)` still works, `validate()` / `ensure_valid()` added) |
| `gromacs.itp`, `gromacs.relax`, `gromacs.analyzing` | `gmx_harness.itp`, `gmx_harness.relax`, `gmx_harness.analysis` |
| `gromacs.pre_coordinator`, `gromacs.rosette_maker` | `gmx_harness.build` (`precoordinate2`, `make_rosette2`, `make_oligorosette`, `Assembly`, ...) |
| `mole.gro.GroFile` | `gmx_harness.GroFile` (no dependency on `mole`) |
| `base_utils.plotlib.load_xvgdata` | `gmx_harness.load_xvg` (returns `XvgData`; the first row is no longer dropped) |
| `launch(..., full_overwrite)` | Requires `confirm=True`, and only for directories gmx_harness created |
| `save_json` / `load_json` (list of `__class__` dicts) | Own versioned format `{"format": "gmx_harness.pipeline", "version": 1, "steps": [{"type", "params"}]}`; mylibs JSON cannot be loaded |
| `FileControl(name, cmd)` | `RawShellStep(name, cmd, allow_unsafe=True)`; `FileControl.remove_MCH` → `RemoveResidue` |
| `sed -i '/MCH/d' topol.top` | `RemoveResidue` / `ResizeBox`: an awk edit in the generated script that touches only [ molecules ] and the #include (also fixes the topol.top typo) |

Intentional behaviour changes: the per-step scripts use `set -eo pipefail` and stop on failure; `freeze` exits with status 1;
`generate_xtc.sh` no longer prompts or launches ovito; `print` output goes to `logging` (`gmx_harness` logger);
re-running `run.sh` never overwrites the inputs of finished steps and skips grompp when a checkpoint exists.

## Development

```bash
uv sync
uv run python -m unittest discover -s tests -t .
uv run mypy
uv run python -c "from gmx_harness.apidoc import write_api_docs; write_api_docs()"   # after changing docstrings
```
