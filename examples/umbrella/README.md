# Example: umbrella sampling of a rotation CV (steered MD → windows)

A complete job tree built from gmx_harness steps plus two custom steps. It computes the free energy
F(θ) of a periodic CV θ that rotates one block of atoms (`upper`) against another (`lower`).

```
OUT/
  run.sh
  0_em  1_solv  2_em2  3_equil_lock  4_equil_unlock   relaxation; the system is solvated ONCE
  5_umbrella/                                         Nested step
      smd/        steered MD (NVT): lower block frozen, upper block rigid, pose CVs held,
                  θ → θ + one period (MOVINGRESTRAINT in segments shorter than half a period)
      extract/    one frame per window out of smd/ → ../wNN/input.gro, windows.json
      w00 … wNN/  Nested(chain): 00_relax (NPT) → 01_umb (production), RESTRAINT on θ
```

All windows share N, box, topology and atom numbers, as WHAM/MBAR needs. `bash run.sh` in any folder
finishes that folder; a re-run skips finished steps and continues MD from `output.cpt`.

## Files

| file | role |
|---|---|
| `umbrella.py` | `UmbrellaSettings`, PLUMED text (`prune`, `window_plumed`), the custom steps `SteeredMD` and `ExtractWindows`, and `build_umbrella` / `build_windows_only` |
| `make_umbrella_jobs.py` | CLI: config JSON → `build_plan` → `preview` → `write` |
| `node/umbrella_node.py` | runs on the node: `smd-prep` (CV values of the equilibrated frame via `plumed driver` → steering `plumed.dat`, `index.ndx`) and `extract` (window frames via `gmx trjconv`) |
| `node/make_ref_from_gro.py` | runs on the node: refreshes the RMSD reference `ref_block.pdb` from the equilibrated frame |
| `example_config.json` | a config for an imaginary 2400-atom system |

## Usage

```bash
uv run python examples/umbrella/make_umbrella_jobs.py config.json jobs/umbrella             # full tree
uv run python examples/umbrella/make_umbrella_jobs.py config.json jobs/umbrella_test --test # every MD step 0.1 ns
uv run python examples/umbrella/make_umbrella_jobs.py config.json jobs/umbrella_wall \
    --windows-only results/umbrella/5_umbrella   # windows only, from existing wNN/input.gro (e.g. with "wall" set)
```

Nothing is run or submitted. Put `run.sh` into your own job script (e.g. with `gmx_harness.write_job_scripts`);
set `OMP_NUM_THREADS` there: mdrun runs with `-ntmpi 1 -bonded gpu -cpt 5` (`mdrun_args`), and with
`"parallel": P` the windows run P at a time and share the threads.

### Config

See the docstring of `make_umbrella_jobs.py`. `settings` takes any `UmbrellaSettings` field (`umbrella.py`);
`lower` / `upper` are the 1-based atom ranges of the frozen and the steered block, `rigid_atoms` the
upper-block atoms held rigid. The defaults are only a starting point: choose `kappa_umb` and the window
spacing so that neighbouring windows overlap, and the sweep speed (`smd_ns`, `kappa_smd`) so that θ follows
the steering centre closely (`extract` warns when a window frame is more than 1° off its target).

### `plumed_defs.dat`

Your CV definitions (UNITS, WHOLEMOLECULES, the CVs and what they depend on). It must define:

- `theta` — the biased CV, periodic with `period_deg`;
- every label in `print_args` (printed in all runs), in `pose` (held at their start values during the
  steered MD) and the CV of `wall`.

Actions that none of these need are pruned away, separately for the steered MD and for the windows.

## Requirements on the node

GROMACS patched with PLUMED, the `plumed` command (for `plumed driver` in `smd-prep`), bash, and
**python3** (standard library only, 3.6+) for the two node scripts. This is the one place that goes
beyond the library's "GROMACS, bash and awk only" rule, which is why it is an example and not a library step.

## Not included

The MBAR/WHAM analysis of the windows: combine `wNN/01_umb/COLVAR` with the `rst: RESTRAINT ... AT= KAPPA=`
of each window's `plumed.dat` (e.g. with pymbar), taking the period of θ into account.
