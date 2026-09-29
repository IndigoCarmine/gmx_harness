---
name: gmx-mdp-tuning
description: Adjust and validate GROMACS .mdp parameters with gmx_harness (MDParameters, additional_mdp_parameters, validate). Use when changing temperature/pressure coupling, cutoffs, output intervals, or free-energy settings, or when reviewing an existing mdp. Keywords: mdp, tcoupl, pcoupl, semiisotropic, nstxout, validate
---

# Adjusting and validating mdp files

## Rules
- Do not hand-edit the generated `setting.mdp`. Change the step class's `additional_mdp_parameters` and regenerate.
- Unknown keys only produce a warning, so every GROMACS option can be used.
  If a warning appears, check for typos (`_` and `-` are equivalent, and case does not matter).
- Do not set `strict_mdp=False` on your own. If it is needed, explain why and ask the human.

## Checking
```python
from gmx_harness import MDParameters
m = MDParameters.from_file("setting.mdp")
for issue in m.validate(): print(issue)       # error / warning
m.ensure_valid()                               # raises MDPValidationError on error
```
For text: `validate_mdp_text(open("setting.mdp").read())`

Reported as errors:
- A template placeholder left unfilled (e.g. `nsteps = nsteps`)
- Non-numeric or out-of-range values (dt <= 0, nsteps < -1, ...)
- The number of `tc-grps` does not match the number of `tau_t`/`ref_t` values
- The number of `ref_p`/`compressibility` values does not match `pcoupltype` (isotropic 1, semiisotropic 2, anisotropic 6)
- Free-energy lambda arrays of different lengths
- A newline inside a value, or duplicate keys under equivalent spellings

## Common adjustments
| Goal | Example |
|---|---|
| Couple temperature per group | `{"tc_grps": "MOL SOL", "tau_t": "0.2 0.2", "ref_t": "300 300"}` |
| Compressed trajectory output | `{"nstxout-compressed": 5000, "nstxout": 0, "nstvout": 0, "nstfout": 0}` |
| Membranes / interfaces | `useSemiisotropic=True` (ref_p and compressibility are doubled automatically) |
| Change the time step | `{"dt": 0.001}` (nsteps is not recalculated, so recompute the run length yourself) |
| Continuation run | `gen_vel="no", continuation=True` (step argument; `None` keeps the template value) |

The simulated time is logged when the step is created (`logging.getLogger("gmx_harness")`, INFO).
