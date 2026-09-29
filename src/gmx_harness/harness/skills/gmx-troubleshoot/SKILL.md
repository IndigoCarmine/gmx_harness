---
name: gmx-troubleshoot
description: Diagnose failures of a gmx_harness-generated pipeline from grompp/mdrun error text or run.out pasted by the human, and propose fixes. Keywords: GROMACS error, grompp warning, LINCS, segmentation fault, blowing up, domain decomposition, error, crashed
---

# Troubleshooting GROMACS failures

Assumption: **the agent does not rerun anything.** Read the logs the human provides (run.out, output.log, grompp output),
fix the step parameters, and regenerate (`OverwritePolicy.REPLACE_GENERATED`).

## Ask for
- Which step directory failed (`N_name/`), and whether it was grompp or mdrun
- The last 50 lines of `run.out` / `output.log`, and whether any `step*.pdb` files exist
- Whether a `freeze` file exists (if it does, the stop was intentional)

## Common causes and fixes
| Symptom | Cause | Fix (step parameters) |
|---|---|---|
| `Too many warnings` (grompp) | The warning text itself describes the problem | Fix the cause. Only if the human agrees after you explain it, raise `maxwarn` to the minimum needed |
| `No such moleculetype` / atom count mismatch | topo.top [ molecules ] does not match the gro | Check topo.top. After solvation, compare with `topo_old.top` (the topology before the solvent line was added) |
| `LINCS WARNING`, `step*.pdb` produced | Bad starting structure, dt too large | Add `EM`, lower `emtol`, pre-relax with `gmx_harness.relax`, shorten the first NVT with `{"dt": 0.001}` |
| `There is no domain decomposition` | Box too small for the number of ranks | Run with `MDRUN_ARGS="-ntmpi 1"`, or use a Martini* step |
| `Pressure scaling more than 1%` | Unequilibrated system with strong pressure coupling | Run NVT first; use c-rescale with a larger tau_p |
| Job hit the time limit / GPU crash, `output.cpt` exists | Interrupted run | The human resubmits; `run.sh` continues from output.cpt (no grompp). For PLUMED runs RESTART is added automatically, but HILLS/COLVAR entries written after the checkpoint time must be trimmed by hand |
| Need a longer run | nsteps reached | The human runs `bash extend.sh <ps>` in that step, then `run.sh` |
| `ERROR: N_x finished without producing output.gro` | mdrun ended early (crash, walltime) | Look at output.log / the job log of that step |
| Reports `this calculation has problems` | `*.pdb` files from an earlier failure remain | After fixing the cause, the **human** deletes the pdb files and reruns |
| `GROMACS not found (set GMX)` | gmx is not on PATH on the running machine | Load GROMACS (`module load` / `source GMXRC`) or set `GMX=/path/to/gmx_mpi` before running |

## After fixing
Change the parameters → `build_plan` → `preview(OverwritePolicy.REPLACE_GENERATED)`.
If `stale_outputs` lists old outputs (output.gro, etc.), tell the human they must be deleted before rerunning.
