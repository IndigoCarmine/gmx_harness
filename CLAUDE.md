# CLAUDE.md: gmx_harness development rules

## Design principles (do not break these)
- **The library never runs external programs.** Do not add `subprocess`, `os.system`, or `shutil.which("gmx")`.
  Generate script text instead; running it is the user's job.
- Do not add non-Python programs (GROMACS, PLUMED, ...) as dependencies. Dependencies are only what pip can install.
- Any value that ends up in a path or a script goes through `safety.validate_*` or `shlex.quote`.
- Nothing is written to disk without `Plan.preview()` first. Deletion is limited to files recorded in the manifest.
  Exception: the checks write small record files directly, `checks.record_facts` (`<file>.facts.json` next to a file
  the calling script just wrote) and `Report.enforce(out_dir=...)` (`checks.json`); they never overwrite inputs.
  Exception: `gmx_harness.archive` writes only inside the archive root and only after a preview (`StorePlan.write`
  previews again under the lock). Each run directory is written once (copied into `.incoming-<id>/`, renamed into
  place, made read-only) and never changed afterwards; besides that it only rewrites `tree.json`, creates and removes
  its `.lock`, and removes its own `.incoming-<id>/` when a store fails. `export` only creates a new directory
  (`DEST/<tree>/<system>`, via a temporary sibling it renames) and tree files that do not exist yet.
- Never call `input()` or prompt interactively. Destructive or unchecked operations need an explicit flag (`allow_unsafe` / `confirm` / `force_modified`).
- Do not lose flexibility: keep escape hatches such as `additional_mdp_parameters`, `strict_mdp=False`, `RawShellStep`, and subclassing `Calculation`.
- Generated scripts must run with only GROMACS, bash and POSIX awk on the target machine (no Python there). Runtime topology edits live in `topology.py` as bash/awk snippets; pass values via environment variables, never splice them into awk code.
  Exception: with `require_preflight=True` the run.sh files also need `sha256sum` (GNU coreutils) to verify `preflight.ok`.

## Checks
```bash
uv run python -m unittest discover -s tests -t .
uv run mypy                     # strict
uv run python -c "from gmx_harness.apidoc import write_api_docs; write_api_docs()"
```
- Whenever public API or docstrings change, regenerate api.md (`tests/test_harness.py` checks that it is not stale).
- When adding a new step class, add it to `steps/__init__.py` and `gmx_harness/__init__.py` `__all__`, add a round-trip case to `test_serialization` (`params()` must be able to rebuild it), and update the skill (`harness/skills/gmx-pipeline`) table.
- The pipeline JSON (`serialization.py`) is gmx_harness's own versioned format. When changing it incompatibly, bump `VERSION`.
