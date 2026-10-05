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
| `check_tree` reports `R006` (output.log, but no output.gro and no `Finished mdrun`) | The step timed out, crashed or is still running | Read the end of output.log and the job output; if it was interrupted, the human resubmits and it continues from output.cpt |
| Need a longer run | nsteps reached | The human runs `bash extend.sh <ps>` in that step, then `run.sh` |
| `ERROR: N_x finished without producing output.gro` | mdrun ended early (crash, walltime) | Look at output.log / the job log of that step |
| Reports `this calculation has problems` | `*.pdb` files from an earlier failure remain | After fixing the cause, the **human** deletes the pdb files and reruns |
| `GROMACS not found (set GMX)` | gmx is not on PATH on the running machine | Load GROMACS (`module load` / `source GMXRC`) or set `GMX=/path/to/gmx_mpi` before running |
| `sha256sum (GNU coreutils) is not available` / `preflight.ok is missing or the inputs changed` | `require_preflight=True`: run.sh checks `preflight.ok` | Install coreutils there, or have the human rerun the workspace's preflight script (it writes `preflight.ok`; the library does not) |

## Design-check errors (before anything runs)
`HarnessCheckError` / `checks:` in a preview lists codes. Read the code, fix the setting it names, rerun the script.

| Code | Meaning | Usual fix |
|---|---|---|
| `F001` | a value differs from what the producing script recorded | change it here, or change the producer and rerun it (ask the human which) |
| `F003` | an input changed after a file was made (stale result) | rerun the producing script (the human deletes a kept result if needed) |
| `S001`/`L003` | gro atoms differ from the topology | molecule count in the topology, or the wrong structure/topology pair |
| `S004` | box edge too small | larger box, or a waiver with the human's reason |
| `P002`/`P005` | a label is not defined (e.g. a bias on an interface index beyond the stack) | fix the define (e.g. BIAS_IFACE) |
| `P009` | a define is ignored by the template | typo in the define name |
| `L004` | maxwarn > 0 | fix the grompp warning; a waiver must quote the accepted warning |
| `S011` | a molecule type of `[ molecules ]` is not in the topology or its local includes (atom count unchecked) | include the .itp locally; if it comes from a force-field directory, a waiver with the human's reason |
| `G004` | the preflight could not check the CV of the step-0 (relaxed) structure (rot/nros not recorded upstream) | rerun the producing stage script so it records them |

After a run, `gmx_harness.checks.check_tree(calc_dir)` reports NaN (`R001`), LINCS (`R002`), GPU errors such as
Xid / CUDA error (`R003`), a stalled CV (`R004`), fatal errors (`R005`) and steps that started but did not finish
(`R006`). It expects the layout `<calc_dir>/<system>/<i>_<step>/` (pass the tree that holds the system directories,
not a system or step directory); job output in a system directory (`sbatch_*.log`, `slurm-*.out`,
`<name>-<jobid>.out/.err`) is read for `R003`/`R005`. For one step directory use `check_step(step_dir)`.

## After fixing
Change the parameters → `build_plan` → `preview(OverwritePolicy.REPLACE_GENERATED)`.
If `stale_outputs` lists old outputs (output.gro, etc.), tell the human they must be deleted before rerunning.
