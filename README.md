# gmx_harness

A library that **generates** GROMACS simulation pipelines (EM → NVT → NPT → production, solvation, box operations, Martini, AWH, BAR) as a set of files and bash scripts. It comes bundled with the harness an AI agent (and also human) needs to handle it safely.
It was split out of the `gromacs` package in mylibs (`yagaiG-libs`).

- **It never runs anything**: the library does not call `gmx`. A person or the job scheduler runs the generated `run.sh`.
  It has no dependency on GROMACS or PLUMED; dependencies are pip packages only (pydantic, numpy, OpenMM, MDAnalysis, pandas, openpyxl).
- **Flexibility**: any mdp option can be set via `additional_mdp_parameters`. Custom steps are possible by subclassing `Calculation`,
  and arbitrary commands can be run with `RawShellStep` (explicit opt-in required).
- **Safety**:
  - `plan → preview → write` flow: all conflicts are checked before anything is written
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
from gmx_harness import EM, MD, MDType, GmxConfig, OverwritePolicy, build_plan, save_json

steps = [
    EM(),
    MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=50_000,
       useRestraint=True, defines=["POSRES"]),
    MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", nsteps=500_000, gen_vel="no"),
]
save_json(steps, "pipeline.json")

plan = build_plan(steps, "start.gro", "work",
                  extra_inputs=["topo.top", "mol.itp"],
                  config=GmxConfig(binary="gmx_mpi", env={"OMP_NUM_THREADS": "8"}))
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

### Steps

| Class | Purpose |
|---|---|
| `EM`, `MartiniEM` | Energy minimization |
| `MD(type=MDType.*)`, `MartiniMD` | MD (NVT / NPT / NH+PR) |
| `AWH`, `BarMethod` | Free energy |
| `Solvation`, `SolvationMCH`, `SolvationSCP216`, `RuntimeSolvation` (deprecated) | Solvation |
| `RemoveResidue`, `ResizeBox`, `AddFiles` | Typed file operations |
| `RawShellStep` (`allow_unsafe=True` required) | Arbitrary bash |

Other: `MDParameters` (mdp editing and validation), `GroFile` (.gro/.ndx), `load_xvg`, `generate_inermolecular_interactions`,
`gmx_harness.relax.relax` (OpenMM), `gmx_harness.analysis` (MDAnalysis).



## Migrating from mylibs

| mylibs | gmx_harness |
|---|---|
| `gromacs.calculation.*` | `gmx_harness.*` (same class names; `generate()` takes an optional `GmxConfig`) |
| `gromacs.mdp` | `gmx_harness.mdp` (`check(key)` still works, `validate()` / `ensure_valid()` added) |
| `gromacs.itp`, `gromacs.relax`, `gromacs.analyzing` | `gmx_harness.itp`, `gmx_harness.relax`, `gmx_harness.analysis` |
| `mole.gro.GroFile` | `gmx_harness.GroFile` (no dependency on `mole`) |
| `base_utils.plotlib.load_xvgdata` | `gmx_harness.load_xvg` (returns `XvgData`; the first row is no longer dropped) |
| `launch(..., full_overwrite)` | Requires `confirm=True`, and only for directories gmx_harness created |
| `FileControl(name, cmd)` | `RawShellStep(name, cmd, allow_unsafe=True)`; `FileControl.remove_MCH` → `RemoveResidue` |
| `sed -i '/MCH/d' topol.top` | `top_tool.py remove-molecule MCH` (touches only [ molecules ] and the #include; also fixes the topol.top typo) |

Intentional behaviour changes: the per-step scripts use `set -eo pipefail` and stop on failure; `freeze` exits with status 1;
`generate_xtc.sh` no longer prompts or launches ovito; `print` output goes to `logging` (`gmx_harness` logger);
trailing comments in the mdp templates are stripped.

## Development

```bash
uv sync
uv run python -m unittest discover -s tests -t .
uv run mypy
uv run python -c "from gmx_harness.apidoc import write_api_docs; write_api_docs()"   # after changing docstrings
```
