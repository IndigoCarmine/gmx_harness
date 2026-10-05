# GROMACS pipelines with gmx_harness: rules for AI agents

This project builds GROMACS simulation inputs with the `gmx_harness` Python library.
Agents working on it follow these rules.

## Hard rules

1. **Never run GROMACS yourself.** Do not execute `gmx`, `gmx_mpi`, `gmx_d`, `mdrun`, `bash run.sh`, `bash extend.sh`, `bash generate_xtc.sh`, `submit.sh`, `sbatch`, `qsub` and similar commands.
   The job ends when the files have been generated. A human runs them.
2. **Always preview before writing.** Show the human `plan.summary()` and `plan.preview()`, get their approval, then call `plan.write()`.
   Only when `preview().ok` is True.
3. **Never enable destructive or unchecked operations on your own.** Never add these unless the human explicitly asks for them:
   - `allow_unsafe=True` (RawShellStep, raw command strings)
   - `confirm=True` (deleting a directory with `launch(..., full_overwrite)`)
   - `force_modified=True` (overwriting files the human has edited)
   - `strict_mdp=False` (ignoring mdp validation errors)
   - `maxwarn > 0` (ignoring grompp warnings)
4. **Do not delete or overwrite existing working directories with `rm` or `shutil`.** To regenerate, use `OverwritePolicy.REPLACE_GENERATED`.
   It only overwrites files that gmx_harness generated and that nobody has edited since.
5. Do not hand-edit generated `.sh` files. If a change is needed, change the step parameters and regenerate.
6. **Design checks are not optional.** `build_plan` runs `gmx_harness.checks` (codes such as `L003`, `P002`, `L004`);
   an error makes `preview().ok` False. Fix the design. A waiver (`waive={...}`, or a script's `WAIVE`) is the human's
   decision: explain the issue, ask, and write their reason. `{"CODE@pattern": "reason"}` waives only the issues
   whose place (the `where` printed after the code: file, step or variant; fnmatch pattern) matches, e.g.
   `{"S004@fiber_rot_+10": "..."}`; `{"CODE": "reason"}` waives every issue of that code. Prefer the place form.
   Never add or widen a waiver on your own, and never turn checks off (`checks=False`).
7. **Do not edit generated files or check records** (`calc*/`, `resource/{gro,top,fixed_gro,preco,rosette,sp,sp_relaxed,top_fixed}/`,
   `*.facts.json`, `checks.json`, `preflight_checks.json`, `postcheck.json`, `preflight.ok`, `.gmx_harness_manifest.json`).
   Rerun the stage script that makes them.
8. **The archive (`gmx_harness.archive`) is append-only for you.** You may run `store` and `export` (they only preview
   without `--write`; show the human the result). Never change, delete, move, rename or change the permissions of
   anything inside the archive root (run directories, `run.json`, `SHA256SUMS`, `tree.json`, `.lock`, `.incoming-*`),
   by any means. A left-over `.lock` or `.incoming-*`, or a failing `verify`, is reported to the human, who decides.

## Workflow

```python
from gmx_harness import EM, MD, MDType, build_plan, save_json

steps = [
    EM(),
    MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=50_000, useRestraint=True, defines=["POSRES"]),
    MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", nsteps=500_000, gen_vel="no"),
]
save_json(steps, "pipeline.json")          # save the design (reproducibility / review)
plan = build_plan(steps, "start.gro", "work", extra_inputs=["topo.top", "mol.itp"])
print(plan.summary()); print(plan.preview())
# after the human approves:
plan.write()
```

A saved design can be reloaded with `load_json("pipeline.json")` and passed to `build_plan` the same way.

## References
- API: `harness/llm_docs/api.md` inside the package (`gmx_harness.agent_harness.harness_dir()`)
- Skills: `gmx-pipeline`, `gmx-mdp-tuning`, `gmx-troubleshoot`, `gmx-relax`
- Check codes: `gmx_harness.checks.CODES` (also in api.md)
