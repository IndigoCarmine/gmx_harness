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
Each line is `LEVEL CODE <where>: message`; `<where>` is the place (file, step, variant) the issue is about.

| Code | Meaning | Usual fix |
|---|---|---|
| `F001` | a value differs from what the producing script recorded | change it here, or change the producer and rerun it (ask the human which) |
| `F003` | an input changed after a file was made (stale result) | rerun the producing script (the human deletes a kept result if needed) |
| `S001`/`L003` | gro atoms differ from the topology | molecule count in the topology, or the wrong structure/topology pair |
| `S004` | box too small: an edge below `min_edge`, or along a `periodic_axes` axis (an assembly continued into its own image) below `min_periodic_edge` (about 2 x the cut-off + margin); a triclinic box is judged by its widths (`box_heights`) | larger box (for a periodic fiber: more disks), or a waiver with the human's reason |
| `P002`/`P005` | a label is not defined (e.g. a bias on an interface index beyond the stack) | fix the define (e.g. BIAS_IFACE) |
| `P009` | a define is ignored by the template | typo in the define name |
| `L004` | maxwarn > 0 | fix the grompp warning; a waiver must quote the accepted warning |
| `S011` | a molecule type of `[ molecules ]` is not in the topology or its local includes (atom count unchecked) | include the .itp locally; if it comes from a force-field directory, a waiver with the human's reason |
| `G001` | grompp of a step failed in the workspace's preflight dry run (with that step's maxwarn) | read the grompp error the human pastes; fix the step parameters, topology or index, regenerate |
| `G002`/`G003` | plumed driver failed on the structure a step receives in the dry run / a twist CV there differs from the built twist | fix the template or its defines; a template expanded with `SIGN=-1` measures `-rot`, so G003 is then waived with that reason |
| `G004` | the preflight could not compare the CV (rot/nros not recorded upstream) | rerun the producing stage script so it records them |
| `G005` | in the dry run a step without setting.mdp (solvation, make_ndx, RawShellStep ...) or its copy.sh failed, or the step calls mdrun and was not run | read its output (the human can keep the temporary copy, e.g. `--keep`); the later steps were **not checked**, and stay unchecked if G005 is waived |
| `W001` | (warn) a waiver matches no issue | remove it, or fix its pattern (the message lists where that code did occur) |

### Waivers
A waiver is the human's decision with the human's reason; never add or widen one yourself. It is one entry of
`waive=` (`build_plan`), of a script's `WAIVE`, or of `Report.enforce(waive)`:
- `{"CODE@pattern": "reason"}` waives only the `CODE` issues whose `where` matches `pattern` (`fnmatch`: `*`, `?`,
  `[...]`; case-sensitive; `\` is read as `/`; `*` also matches `/`). A `where` of the form `<file>:<line>`
  (PLUMED checks) also matches as `<file>`, so `"P007@6_md_metad/plumed.dat"` covers the whole file. An issue
  with an empty `where` is never matched by a pattern. Prefer this form: the same code at another place still stops.
- `{"CODE": "reason"}` waives every `CODE` issue (code-wide; printed as `(for every CODE)`).

If both match an issue, the place waiver is credited. An unknown code, nothing after `@` or an empty reason is a
`ValueError`. A waiver that matches no issue is `W001` (warn), which also happens when one `WAIVE` serves several
systems and a waiver names only one of them. The output shows which waiver applied
(`WAIVED S004 fiber_a: ... -- waived by "S004@fiber_*", reason: ...`), and `checks.json` records `waived` (the
waivers as given), `waived_issues` (per issue: `waiver`, `scope` `where`/`code`, `code`, `where`, `level`,
`message`, `reason`) and `unused_waivers`.

| Code | `where` |
|---|---|
| `F001`/`F002` | the file passed to `expect` (its name, e.g. `MOL_fiber_rot_+10.gro`) |
| `F003`-`F005` | `<parent dir>/<file>` (e.g. `top_fixed/MOL_fixed.top`) |
| `S001`/`S003` | the caller's `where` (gmx_template: `MOL<tag>_fixed.top` in interaction.py, `MOL_<variant> vs MOL<tag>_fixed.top` in relax.py) |
| `S002`, `S004`, `S005`, `S008` | the caller's `where` (gmx_template: the super-particle gro name for S002, the variant name for S004/S005) |
| `S006` | the caller's `where` (gmx_template: `BONDS` for the pair check, `MOL<tag>` for the built bonds) |
| `S007` | the caller's `where` (gmx_template: `MOL<tag>`) |
| `S009` | the caller's `where` (gmx_template: the gro name) |
| `S010` | the ndx file name (default) |
| `S011` | the caller's `where`, default `topology` |
| `P001`-`P008`, `P010` | `build_plan`: `<step>/plumed.dat:<line>`; `expand_template`: `<template> (expanded):<line>` (`P010` without `:<line>`) |
| `P009` | the template file name |
| `L001`, `L002`, `L004`-`L006` | the step directory (e.g. `0_em_vac`), without the system: such a waiver applies to every system planned with it |
| `L003` (and `S003`/`S011` of step 0) | the first step directory, or `<step 0>/topo.top` |
| `G001`-`G005` | `<tree>/<system>/<step>` (gmx_template preflight.py) |
| `R001`-`R006` | `<system>/<step>` (`check_tree`; `<step>` from `check_step` alone); `R003`/`R005` from the job output of a system: `<system>` |

A PLUMED issue of a template is reported twice when a script checks both the expanded template
(`expand_template`) and the plan (`build_plan`; gmx_template plan_metad.py does): a place waiver must then match
both places (two keys, e.g. `"P007@6_md_metad/plumed.dat"` and `"P007@metad_twist.plumed.in*"`), and each key shows
`W001` in the other report.

After a run, `gmx_harness.checks.check_tree(calc_dir)` reports NaN (`R001`), LINCS (`R002`), GPU errors such as
Xid / CUDA error (`R003`), a stalled CV (`R004`), fatal errors (`R005`) and steps that started but did not finish
(`R006`). It expects the layout `<calc_dir>/<system>/<i>_<step>/` (pass the tree that holds the system directories,
not a system or step directory); job output in a system directory (`sbatch_*.log`, `slurm-*.out`,
`<name>-<jobid>.out/.err`) is read for `R003`/`R005`. For one step directory use `check_step(step_dir)`.

## After fixing
Change the parameters → `build_plan` → `preview(OverwritePolicy.REPLACE_GENERATED)`.
If `stale_outputs` lists old outputs (output.gro, etc.), tell the human they must be deleted before rerunning.
