"""Parse, edit, validate and export GROMACS .mdp parameter files.

``MDParameters`` accepts any key/value (so every GROMACS option stays
available), and ``validate()`` catches the mistakes that are easy to make when
filling templates by hand or by an agent: unfilled placeholders, values that
are not numbers, mismatched group counts, or text that would inject extra
lines into the file. Unknown keys only produce warnings.
"""

import copy
import warnings
from dataclasses import dataclass
from typing import Literal

MDPValue = str | int | float


@dataclass(frozen=True)
class ValidationIssue:
    """One finding of ``MDParameters.validate``. ``level`` is "error" or "warning"."""

    level: Literal["error", "warning"]
    key: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.key}: {self.message}"


class MDPValidationError(ValueError):
    """Raised by ``MDParameters.ensure_valid`` when validation finds errors."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("invalid mdp parameters:\n" + "\n".join(f"  {i}" for i in issues))


def normalize_key(key: str) -> str:
    """GROMACS treats ``_`` and ``-`` as the same and ignores case."""
    return key.strip().lower().replace("_", "-")


# Template placeholders. A value equal to one of these was never filled in.
PLACEHOLDERS = frozenset(
    {"nsteps", "nstxout", "nstvout", "nstfout", "nstenergy", "ref_t", "gen_vel", "gen_temp", "emtol"}
)

_INT_KEYS = {
    "nsteps": -1,  # -1 means "run forever"
    "nstcalcenergy": 0,
    "nstenergy": 0,
    "nstlog": 0,
    "nstxout": 0,
    "nstvout": 0,
    "nstfout": 0,
    "nstxout-compressed": 0,
    "nstlist": 0,
    "nstcomm": 0,
    "nstdhdl": 0,
    "pme-order": 3,
    "init-lambda-state": -1,
    "nsttcouple": -1,
    "nstpcouple": -1,
    "lincs-order": 1,
    "niter": 1,
    "ld-seed": -1,
    "gen-seed": -1,
    "nstcgsteep": 0,
    "nbfgscorr": 0,
}
_POSITIVE_FLOAT_KEYS = {
    "dt",
    "emtol",
    "emstep",
    "rlist",
    "rcoulomb",
    "rvdw",
    "fourierspacing",
    "tau-p",
    "verlet-buffer-tolerance",
    "ewald-rtol",
}
_NONNEG_FLOAT_KEYS = {"tinit", "gen-temp", "rvdw-switch", "rcoulomb-switch", "epsilon-rf", "epsilon-r", "sc-alpha"}

_CHOICES: dict[str, set[str]] = {
    "integrator": {"md", "md-vv", "md-vv-avek", "sd", "bd", "steep", "cg", "l-bfgs", "nm", "tpi", "tpic", "mimic"},
    "gen-vel": {"yes", "no"},
    "continuation": {"yes", "no"},
    "cutoff-scheme": {"verlet", "group"},
    "pbc": {"xyz", "no", "xy", "screw"},
    "tcoupl": {"no", "berendsen", "nose-hoover", "andersen", "andersen-massive", "v-rescale"},
    "pcoupl": {"no", "berendsen", "c-rescale", "parrinello-rahman", "mttk"},
    "pcoupltype": {"isotropic", "semiisotropic", "anisotropic", "surface-tension"},
    "constraints": {"none", "h-bonds", "all-bonds", "h-angles", "all-angles"},
    "constraint-algorithm": {"lincs", "shake"},
    "free-energy": {"yes", "no"},
    "refcoord-scaling": {"no", "all", "com"},
}
# Choices whose wrong value is certainly a typo -> error instead of warning.
_STRICT_CHOICES = {"gen-vel", "continuation", "free-energy", "integrator", "pbc"}

# Number of values expected in ref_p / compressibility per pcoupltype.
_PCOUPL_NVALUES = {"isotropic": 1, "semiisotropic": 2, "anisotropic": 6, "surface-tension": 2}

# Keys GROMACS (2020-2025) knows, normalized. Anything else only warns.
_KNOWN_KEYS = {
    "include", "define", "integrator", "tinit", "dt", "nsteps", "init-step", "simulation-part",
    "mts", "mts-levels", "mts-level2-forces", "mts-level2-factor", "comm-mode", "nstcomm", "comm-grps",
    "bd-fric", "ld-seed", "emtol", "emstep", "niter", "fcstep", "nstcgsteep", "nbfgscorr",
    "rtpi", "nstxout", "nstvout", "nstfout", "nstlog", "nstcalcenergy", "nstenergy",
    "nstxout-compressed", "compressed-x-precision", "compressed-x-grps", "energygrps",
    "cutoff-scheme", "nstlist", "pbc", "periodic-molecules", "verlet-buffer-tolerance",
    "verlet-buffer-pressure-tolerance", "rlist", "coulombtype", "coulomb-modifier", "rcoulomb-switch",
    "rcoulomb", "epsilon-r", "epsilon-rf", "vdwtype", "vdw-type", "vdw-modifier", "rvdw-switch", "rvdw",
    "dispcorr", "table-extension", "energygrp-table", "fourierspacing", "fourier-nx", "fourier-ny",
    "fourier-nz", "pme-order", "ewald-rtol", "ewald-rtol-lj", "lj-pme-comb-rule", "ewald-geometry",
    "epsilon-surface", "implicit-solvent", "ensemble-temperature-setting", "ensemble-temperature",
    "tcoupl", "nsttcouple", "nh-chain-length", "print-nose-hoover-chain-variables", "tc-grps", "tau-t",
    "ref-t", "pcoupl", "pcoupltype", "nstpcouple", "tau-p", "compressibility", "ref-p",
    "refcoord-scaling", "annealing", "annealing-npoints", "annealing-time", "annealing-temp",
    "gen-vel", "gen-temp", "gen-seed", "constraints", "constraint-algorithm", "continuation",
    "shake-tol", "lincs-order", "lincs-iter", "lincs-warnangle", "morse", "energygrp-excl",
    "nwall", "wall-type", "wall-r-linpot", "wall-atomtype", "wall-density", "wall-ewald-zfac",
    "pull", "awh", "rotation", "imd-group", "disre", "disre-weighting", "disre-mixed", "disre-fc",
    "disre-tau", "nstdisreout", "orire", "orire-fc", "orire-tau", "orire-fitgrp", "nstorireout",
    "free-energy", "couple-moltype", "couple-lambda0", "couple-lambda1", "couple-intramol",
    "init-lambda", "init-lambda-state", "delta-lambda", "nstdhdl", "fep-lambdas", "mass-lambdas",
    "coul-lambdas", "vdw-lambdas", "bonded-lambdas", "restraint-lambdas", "temperature-lambdas",
    "calc-lambda-neighbors", "init-lambda-weights", "dhdl-print-energy", "sc-alpha", "sc-r-power",
    "sc-coul", "sc-power", "sc-sigma", "sc-function", "sc-gapsys-scale-linpoint-lj",
    "sc-gapsys-scale-linpoint-q", "sc-gapsys-sigma-lj", "separate-dhdl-file", "dhdl-derivatives",
    "dh-hist-size", "dh-hist-spacing", "nstexpanded", "lmc-stats", "lmc-move", "lmc-weights-equil",
    "acc-grps", "accelerate", "freezegrps", "freezedim", "cos-acceleration", "deform",
    "electric-field-x", "electric-field-y", "electric-field-z", "qmmm-cp2k-active",
    "qmmm-cp2k-qmgroup", "qmmm-cp2k-qmmethod", "qmmm-cp2k-qmcharge", "qmmm-cp2k-qmmultiplicity",
    "swapcoords", "density-guided-simulation-active", "user1-grps", "user2-grps", "userint1",
    "userint2", "userint3", "userint4", "userreal1", "userreal2", "userreal3", "userreal4",
    "nstcheckpoint",  # removed in 2018, kept for old files
    "ns-type",  # obsolete (ignored since 2020) but present in the bundled templates
}
_KNOWN_PREFIXES = ("awh", "pull-", "rot-", "imd-", "swap-", "density-guided-simulation-", "qmmm-", "simulated-")


class MDParameters:
    """
    Ordered set of GROMACS .mdp parameters.

    Keys are kept as written (GROMACS itself treats ``_``/``-`` and case as
    equivalent). Chain ``add_or_update`` calls to fill templates, call
    ``validate()`` (or ``ensure_valid()``) before ``export()``.
    """

    def __init__(self, data: dict[str, MDPValue] | None = None, ignore_deepcopy: bool = False):
        self.data: dict[str, MDPValue]
        if data is None:
            self.data = {}
        elif ignore_deepcopy:
            self.data = data
        else:
            self.data = copy.deepcopy(data)

    def export(self) -> str:
        """Return the parameters in .mdp syntax (``key = value`` lines)."""
        if not self.data:
            return ""
        max_length = max(len(key) for key in self.data.keys())
        return "\n".join(f"{key.ljust(max_length)} = {value}" for key, value in self.data.items())

    def __str__(self) -> str:
        return "; GROMACS MDP parameters file preview\n" + self.export() + "\n; End of MDP parameters file preview"

    @classmethod
    def from_text(cls, text: str) -> "MDParameters":
        """
        Parse .mdp text. Comment-only lines are dropped; a trailing ``; comment``
        stays part of the value (and is written back by ``export``), as in mylibs.
        """
        mdp = cls()
        for lineno, raw in enumerate(text.split("\n"), start=1):
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if "=" not in line:
                raise ValueError(f"line {lineno}: expected 'key = value', got {raw!r}")
            key, value = line.split("=", 1)
            mdp.data[key.strip()] = value.strip()
        return mdp

    @classmethod
    def from_file(cls, path: str) -> "MDParameters":
        with open(path, "r") as f:
            return cls.from_text(f.read())

    def _find_key(self, key: str) -> str | None:
        """The stored spelling of ``key`` (matching GROMACS' normalization), if present."""
        if key in self.data:
            return key
        n = normalize_key(key)
        for k in self.data:
            if normalize_key(k) == n:
                return k
        return None

    def has(self, key: str) -> bool:
        """True if ``key`` (in any GROMACS-equivalent spelling) is set."""
        return self._find_key(key) is not None

    def check(self, key: str) -> bool:
        """Same as ``has``; the name mylibs used."""
        return self.has(key)

    def add_or_update(self, key: str, value: MDPValue) -> "MDParameters":
        """Set ``key``. An existing equivalent spelling (``tc_grps`` vs ``tc-grps``) is replaced in place."""
        existing = self._find_key(key)
        if existing is not None and existing != key:
            # keep the original position, but use the new spelling
            self.data = {(key if k == existing else k): (value if k == existing else v) for k, v in self.data.items()}
        else:
            self.data[key] = value
        return self

    def remove(self, key: str) -> "MDParameters":
        existing = self._find_key(key)
        if existing is not None:
            del self.data[existing]
        return self

    def get(self, key: str) -> MDPValue | None:
        existing = self._find_key(key)
        return None if existing is None else self.data[existing]

    # ------------------------------------------------------------------ validation

    def validate(self) -> list[ValidationIssue]:
        """Return all problems found. Errors make grompp fail or silently do the wrong thing."""
        issues: list[ValidationIssue] = []

        def err(key: str, msg: str) -> None:
            issues.append(ValidationIssue("error", key, msg))

        def warn(key: str, msg: str) -> None:
            issues.append(ValidationIssue("warning", key, msg))

        seen: dict[str, str] = {}
        norm: dict[str, str] = {}
        for key, value in self.data.items():
            text = str(value)
            if "\n" in key or "\r" in key or "=" in key or ";" in key or not key.strip():
                err(key, "key contains a newline, '=', ';' or is empty")
                continue
            if "\n" in text or "\r" in text:
                err(key, "value contains a newline (would inject extra mdp lines)")
            text = text.split(";", 1)[0]  # a trailing comment is not part of the value for GROMACS
            n = normalize_key(key)
            if n in seen:
                err(key, f"duplicate of {seen[n]!r} (GROMACS treats them as the same option)")
            seen[n] = key
            norm[n] = text.strip()
            if text.strip() in PLACEHOLDERS:
                err(key, f"placeholder {text!r} was never filled in")
                continue
            if n not in _KNOWN_KEYS and not n.startswith(_KNOWN_PREFIXES):
                warn(key, "unknown mdp option (typo?) - grompp will warn or ignore it")

        for n, minimum in _INT_KEYS.items():
            if n in norm and norm[n] not in PLACEHOLDERS:
                try:
                    if int(norm[n]) < minimum:
                        err(seen[n], f"must be >= {minimum}, got {norm[n]}")
                except ValueError:
                    err(seen[n], f"must be an integer, got {norm[n]!r}")
        for n in _POSITIVE_FLOAT_KEYS | _NONNEG_FLOAT_KEYS:
            if n in norm and norm[n] not in PLACEHOLDERS:
                try:
                    v = float(norm[n])
                except ValueError:
                    err(seen[n], f"must be a number, got {norm[n]!r}")
                    continue
                if n in _POSITIVE_FLOAT_KEYS and v <= 0:
                    err(seen[n], f"must be > 0, got {norm[n]}")
                elif v < 0:
                    err(seen[n], f"must be >= 0, got {norm[n]}")
        for n, choices in _CHOICES.items():
            if n in norm and norm[n] not in PLACEHOLDERS and norm[n].lower() not in choices:
                msg = f"{norm[n]!r} is not one of {sorted(choices)}"
                (err if n in _STRICT_CHOICES else warn)(seen[n], msg)

        # temperature coupling groups
        tcoupl = norm.get("tcoupl", "no").lower()
        integrator = norm.get("integrator", "md").lower()
        if (tcoupl != "no" or integrator in ("sd", "bd")) and "tc-grps" in norm:
            ngroups = len(norm["tc-grps"].split())
            for n in ("tau-t", "ref-t"):
                if n in norm and norm[n] not in PLACEHOLDERS:
                    values = norm[n].split()
                    if len(values) != ngroups:
                        err(seen[n], f"has {len(values)} value(s) but tc-grps has {ngroups} group(s)")
                    for token in values:
                        try:
                            if float(token) < 0 and n == "ref-t":
                                err(seen[n], f"negative temperature {token}")
                        except ValueError:
                            err(seen[n], f"{token!r} is not a number")

        # pressure coupling value counts
        pcoupl = norm.get("pcoupl", "no").lower()
        if pcoupl != "no":
            ptype = norm.get("pcoupltype", "isotropic").lower()
            expected = _PCOUPL_NVALUES.get(ptype)
            if expected is not None:
                for n in ("ref-p", "compressibility"):
                    if n in norm:
                        got = len(norm[n].split())
                        if got != expected:
                            err(seen[n], f"pcoupltype={ptype} needs {expected} value(s), got {got}")

        # free-energy lambda arrays must have the same length
        lambda_keys = [
            k for k in ("fep-lambdas", "mass-lambdas", "coul-lambdas", "vdw-lambdas", "bonded-lambdas",
                        "restraint-lambdas", "temperature-lambdas") if k in norm
        ]
        lengths = {k: len(norm[k].split()) for k in lambda_keys}
        if len(set(lengths.values())) > 1:
            err(lambda_keys[0], f"lambda arrays differ in length: {lengths}")

        if norm.get("continuation", "no").lower() == "yes" and norm.get("gen-vel", "no").lower() == "yes":
            warn(seen["continuation"], "continuation=yes with gen_vel=yes: velocities are regenerated although "
                 "the run claims to continue; use gen_vel=no (or continuation=no)")
        if integrator in ("steep", "cg", "l-bfgs") and norm.get("gen-vel", "no").lower() == "yes":
            warn(seen.get("gen-vel", "gen_vel"), "gen_vel=yes has no effect for energy minimization")
        return issues

    def ensure_valid(self, *, strict: bool = True) -> "MDParameters":
        """
        Validate; raise ``MDPValidationError`` on errors, send warnings to ``warnings.warn``.

        With ``strict=False`` errors are downgraded to warnings (escape hatch
        for setups the validator does not understand).
        """
        issues = self.validate()
        errors = [i for i in issues if i.level == "error"]
        for i in issues:
            if i.level == "warning" or not strict:
                warnings.warn(str(i), stacklevel=2)
        if errors and strict:
            raise MDPValidationError(errors)
        return self

    def dt(self, default: float = 0.002) -> float:
        """The time step in ps (``default`` if missing or not a number)."""
        value = self.get("dt")
        try:
            return float(str(value).split(";", 1)[0]) if value is not None else default
        except ValueError:
            return default


def validate_mdp_text(text: str) -> list[ValidationIssue]:
    """Parse .mdp text and return validation issues (convenience for agents)."""
    return MDParameters.from_text(text).validate()


EM_MDP: dict[str, MDPValue] = {
    "integrator": "steep",
    "nsteps": "100000000",
    "emtol": "100",
    "emstep": "0.1",
    "ns_type": "grid",
    "rlist": "1",
    "rcoulomb": "1",
    "rvdw": "1",
    "pbc": "xyz",
}

# --- Martini 3 (coarse-grained) MDP templates ---------------------------------
# Values follow the official Martini 3 "new-RF" recommendation
# (de Jong et al. 2016; Martini 3, Souza et al. Nat. Methods 2021):
#   Verlet scheme, reaction-field electrostatics with epsilon_rf=0 (conductor)
#   and epsilon_r=15, potential-shift-Verlet vdw, rvdw=rcoulomb=1.1 nm.
# Placeholder tokens ("nsteps", "ref_t", ...) are filled by the step classes;
# validate() reports any that are left over.
MARTINI_MIN_MDP: dict[str, MDPValue] = {
    "integrator": "steep",
    "nsteps": "nsteps",
    "emtol": "emtol",
    "emstep": "0.01",
    "cutoff-scheme": "Verlet",
    "nstlist": "10",
    "verlet-buffer-tolerance": "0.005",
    "coulombtype": "reaction-field",
    "rcoulomb": "1.1",
    "epsilon_r": "15",
    "epsilon_rf": "0",
    "vdw-type": "cutoff",
    "vdw-modifier": "Potential-shift-verlet",
    "rvdw": "1.1",
    "pbc": "xyz",
}

MARTINI_MD_MDP: dict[str, MDPValue] = {
    "integrator": "md",
    "dt": "0.02",
    "nsteps": "nsteps",
    "nstxout": "nstxout",
    "nstvout": "nstvout",
    "nstfout": "nstfout",
    "nstlog": "nstenergy",
    "nstenergy": "nstenergy",
    "nstxout-compressed": "nstxout",
    "compressed-x-precision": "1000",
    "cutoff-scheme": "Verlet",
    "nstlist": "20",
    "verlet-buffer-tolerance": "0.005",
    "ns_type": "grid",
    "coulombtype": "reaction-field",
    "rcoulomb": "1.1",
    "epsilon_r": "15",
    "epsilon_rf": "0",
    "vdw-type": "cutoff",
    "vdw-modifier": "Potential-shift-verlet",
    "rvdw": "1.1",
    "tcoupl": "v-rescale",
    "tc_grps": "system",
    "tau_t": "1.0",
    "ref_t": "ref_t",
    "Pcoupl": "c-rescale",
    "Pcoupltype": "isotropic",
    "tau_p": "12.0",
    "compressibility": "3e-4",
    "ref_p": "1.0",
    "gen_vel": "gen_vel",
    "gen_temp": "gen_temp",
    "constraints": "none",
    "constraint_algorithm": "Lincs",
    "pbc": "xyz",
    "refcoord_scaling": "all",
}

V_RESCALE_C_RESCALE_MDP: dict[str, MDPValue] = {
    "integrator": "md",
    "dt": "0.002",
    "nsteps": "nsteps",
    "nstxout": "nstxout",
    "nstvout": "nstvout",
    "nstfout": "nstfout",
    "nstenergy": "nstenergy",
    "cutoff-scheme": "verlet",
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "nstlist": "10",
    "ns_type": "grid",
    "tcoupl": "v-rescale",
    "tc_grps": "system",
    "tau_t": "0.2",
    "ref_t": "ref_t",
    "rlist": "1.4",
    "coulombtype": "PME",
    "rcoulomb": "1.4",
    "fourierspacing": "0.30",
    "pme_order": "4",
    "vdwtype": "Cut-off",
    "rvdw": "1.4",
    "Pcoupl": "c-rescale",
    "tau_p": "2.0",
    "ref_p": "1",
    "compressibility": "4.5e-05",
    "gen_vel": "gen_vel",
    "gen_temp": "gen_temp",
    "pbc": "xyz",
}

V_RESCALE_ONLY_NVT_MDP: dict[str, MDPValue] = {
    "integrator": "md",
    "dt": "0.002",
    "nsteps": "nsteps",
    "nstxout": "nstxout",
    "nstvout": "nstvout",
    "nstfout": "nstfout",
    "nstenergy": "nstenergy",
    "cutoff-scheme": "verlet",
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "nstlist": "10",
    "ns_type": "grid",
    "tcoupl": "v-rescale",
    "tc_grps": "system",
    "tau_t": "0.2",
    "ref_t": "ref_t",
    "rlist": "1.4",
    "coulombtype": "PME",
    "rcoulomb": "1.4",
    "fourierspacing": "0.30",
    "pme_order": "4",
    "vdwtype": "Cut-off",
    "rvdw": "1.4",
    "Pcoupl": "no",
    "gen_vel": "gen_vel",
    "gen_temp": "gen_temp",
    "pbc": "xyz",
}

NOSE_HOOVER_PARINELLO_RAHMAN_MDP: dict[str, MDPValue] = {
    "integrator": "md",
    "dt": "0.002",
    "nsteps": "10000",
    "nstxout": "5000",
    "nstvout": "1000",
    "nstfout": "1000",
    "nstenergy": "1000",
    "cutoff-scheme": "verlet",
    "continuation": "yes",
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "nstlist": "10",
    "ns_type": "grid",
    "tcoupl": "nose-hoover",
    "tc_grps": "system",
    "tau_t": "1",
    "ref_t": "300",
    "rlist": "1.4",
    "coulombtype": "PME",
    "rcoulomb": "1.4",
    "fourierspacing": "0.30",
    "pme_order": "4",
    "vdwtype": "Cut-off",
    "rvdw": "1.4",
    "Pcoupl": "Parrinello-Rahman",
    "tau_p": "5.0",
    "ref_p": "1",
    "compressibility": "2.0e-05",
    "gen_vel": "no",
    "pbc": "xyz",
}

__all__ = [
    "MDParameters",
    "MDPValidationError",
    "ValidationIssue",
    "validate_mdp_text",
    "normalize_key",
    "EM_MDP",
    "MARTINI_MIN_MDP",
    "MARTINI_MD_MDP",
    "V_RESCALE_C_RESCALE_MDP",
    "V_RESCALE_ONLY_NVT_MDP",
    "NOSE_HOOVER_PARINELLO_RAHMAN_MDP",
]
