---
name: gmx-relax
description: Pre-relax geometrically built structures with bad atom overlaps using gmx_harness.relax (OpenMM soft-core LJ) before GROMACS EM, and generate intermolecular_interactions bonds with gmx_harness.itp. Keywords: soft-core, overlap, relax, OpenMM, rosette, intermolecular bond
---

# Pre-relaxing with soft-core LJ

Use this for structures built by geometric placement (rosette assemblies, etc.) where atoms overlap so badly that plain EM blows up.
OpenMM is a regular dependency of gmx_harness. This is Python computation, not GROMACS, so the agent may run it,
but it takes a long time on large systems, so check with the human first.

```python
from gmx_harness.relax import relax
out = relax("built.gro", "topo.top", out_dir="relaxed")   # returns relaxed/built_relaxed.gro
```
- It reads the .itp referenced under `#ifdef INTER` ([ intermolecular_interactions ], funct 6) automatically,
  and pins those atoms in place with position restraints (see the relax.py docstring for the trade-offs).
- Pass the result as `input_gro` to `build_plan`, and still start with `EM`.
- Several systems: `relax_systems(systems, src_dir, top_dir, out_dir)` relaxes `<src_dir>/<s>.gro` with
  `<top_dir>/<s>_fixed.top` into `<out_dir>/<s>.gro` (+ copies `<s>.ndx`), keeps outputs that already
  exist (they may feed submitted jobs) and checks every input before the first minimization.

## Building the assembly first
`gmx_harness.build` makes the geometric structures that usually need this relaxation:
`precoordinate2(mono, top, nh, o)` → `make_rosette2(mono, n, size)` → `.to_gro(renumber=True)` →
`make_oligorosette(ring, n, length, angle, slip=0)` → `.to_gro(box=..., renumber=True)`.
These are pure Python, so the agent may run them; show the human the resulting .gro/.pdb (`save_pdb`) before continuing.
For other shapes (tapes, sheets) do not set coordinates by hand: place copies with `RigidTransform`
(`Assembly.replicate`, `.repeat`, `.transformed`, `.centered_in_box`, `.number_residues`) and write
`.gro/.pdb/.ndx` with `save_assembly`. Check overlaps with `gmx_harness.min_distance` /
`count_contacts` or `checks.check_contacts(gro, natoms, pbc="xy")`.

## Intermolecular bonds
```python
from gmx_harness import generate_inermolecular_interactions
generate_inermolecular_interactions(natoms=120, nmols=12, bonds=[(5, 130)], nmols_in_rosette=6,
                                    outfile_path="inter.itp")
```
For an assembly that is not a ring of rosettes, find the pairs in the built structure instead of listing them:
`pairs = find_intermolecular_pairs(gro, natoms, donors, acceptors, cutoff)` (minimum image), then
`write_intermolecular_bonds("inter.itp", pairs, length, k)`.
Put `#ifdef INTER` / `#include "inter.itp"` / `#endif` in topo.top and use `defines=["INTER"]` for the MD steps.
Pass inter.itp to `build_plan(..., extra_inputs=[...])`.
