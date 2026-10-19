"""Umbrella sampling of a periodic CV ``theta``, from steered-MD start frames, built with gmx_harness.

``theta`` is a rotation of one block of atoms (``upper``) against another (``lower``), e.g. a torsion
between two parts of an assembly. The protocol::

    0_em  1_solv  2_em2  3_equil_lock  4_equil_unlock      the usual build + relaxation, solvated ONCE
    5_umbrella/                                            a Nested step
        smd/        steered MD (NVT): lower block frozen (freezegrps), upper block rigid (PLUMED RMSD),
                    its pose CVs held, theta -> theta + one period
        extract/    cuts one frame per window out of smd/ (theta = start + i * step)
        w00 .. wNN/ Nested(chain): 00_relax (NPT) -> 01_umb (production), RESTRAINT on theta only

All windows share N, box, topology and atom numbers (solvent is added once), as WHAM/MBAR needs.
``build_windows_only`` gives the windows alone, for start frames that already exist (e.g. to rerun
them with an extra wall on another CV).

The steered MD needs values that only exist on the node (theta, pose CVs of the equilibrated frame):
its grommp.sh runs ``node/make_ref_from_gro.py`` and ``node/umbrella_node.py smd-prep`` (python3,
standard library only) before grompp, and the frame extraction runs ``umbrella_node.py extract``.
Everything else is plain gmx_harness.
"""

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any, override

from pydantic.dataclasses import dataclass

from gmx_harness import EM, MD, Calculation, GroFile, MDType, Nested, SolvationMCH
from gmx_harness.plumed import compress
from gmx_harness.safety import validate_name
from gmx_harness.scripts import NOOP_GROMPP, body_script

HERE = Path(__file__).resolve().parent
NODE = HERE / "node"


@dataclasses.dataclass
class UmbrellaSettings:
    """Every number of the protocol. The defaults are only a starting point; tune them for your system."""

    dt: float = 0.002                       # ps (the MD templates' dt)
    # ---- system ----
    lower: tuple[int, int] = (1, 1)         # 1-based atom range frozen in the steered MD
    upper: tuple[int, int] = (2, 2)         # 1-based atom range of the steered block
    rigid_atoms: tuple[int, ...] = ()       # atoms of the upper block held rigid by an RMSD restraint
    defines: tuple[str, ...] = ("POSRES",)  # grompp defines of the restrained steps (em, equil_lock)
    # ---- the CVs (labels of plumed_defs.dat) ----
    period_deg: float = 360.0               # period of theta (smaller for a symmetric block, e.g. 120 for 3-fold)
    print_args: tuple[str, ...] = ("theta",)            # printed in every run
    pose: tuple[tuple[str, float], ...] = ()            # (CV, kappa): held at the start value in the steered MD
    # ---- windows ----
    n_win: int = 36                         # theta = win_start + i*win_step
    win_start_deg: float = -180.0
    win_step_deg: float = 10.0
    kappa_umb: float = 200.0                # kJ/mol/rad^2: sigma = sqrt(kT/k) ~ 6 deg, so neighbours overlap
    relax_ns: float = 2.0
    umb_ns: float = 10.0
    umb_nstout: int = 50000                 # trr frames of the production (x only)
    umb_print_stride: int = 250             # COLVAR rows (0.5 ps)
    wall: tuple[str, float, float] | None = None   # (CV, AT, KAPPA): UPPER_WALLS in the windows, e.g. to keep
                                                   # the blocks from drifting apart while theta is restrained
    parallel: int = 1                       # windows run at a time (they share the node's cores)
    # ---- steered MD ----
    smd_ns: float = 24.0                    # length of the sweep over one period
    smd_hold_steps: int = 50000             # at the start angle before the sweep
    smd_end_hold_steps: int = 1_000_000     # after the sweep, so theta can catch up
    kappa_smd: float = 5000.0               # kJ/mol/rad^2
    per_atom_k: float = 1000.0              # kJ/mol/nm^2 per rigid atom (RMSD restraint kappa = k*N)
    smd_segments: int = 6                   # the sweep in pieces shorter than half a period
    smd_frame_steps: int = 5000             # trr frame spacing (10 ps)
    smd_print_stride: int = 500             # COLVAR spacing (1 ps)
    # ---- relaxation before the umbrella ----
    em_nsteps: int = 2000
    em2_nsteps: int = 3000
    equil_nsteps: int = 500_000             # 1 ns each, restrained then free
    equil_nstout: int = 25_000
    mdrun_args: tuple[str, ...] = ("-ntmpi", "1", "-bonded", "gpu", "-cpt", "5")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UmbrellaSettings":
        def tup(v: Any) -> Any:
            return tuple(tup(x) for x in v) if isinstance(v, list) else v
        unknown = sorted(set(d) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ValueError(f"unknown settings {unknown}")
        return cls(**{k: tup(v) for k, v in d.items()})

    def test_mode(self) -> "UmbrellaSettings":
        """Every MD step 0.1 ns (50,000 steps), to see whether the whole tree runs through."""
        return dataclasses.replace(
            self, equil_nsteps=50_000, smd_hold_steps=5000, smd_ns=0.08, smd_end_hold_steps=5000,
            smd_frame_steps=1000, smd_print_stride=100, relax_ns=0.1, umb_ns=0.1, umb_nstout=25000)

    def steps(self, ns: float) -> int:
        return int(round(ns * 1000 / self.dt))

    def node_config(self) -> dict[str, Any]:
        """umbrella_config.json: what node/umbrella_node.py reads."""
        return {
            "dt": self.dt, "lower": list(self.lower), "upper": list(self.upper), "n_rigid": len(self.rigid_atoms),
            "per_atom_k": self.per_atom_k, "kappa_smd": self.kappa_smd, "pose": [list(p) for p in self.pose],
            "print_args": list(self.print_args), "period_deg": self.period_deg,
            "smd_hold_steps": self.smd_hold_steps, "smd_ramp_steps": self.steps(self.smd_ns),
            "smd_end_hold_steps": self.smd_end_hold_steps, "smd_print_stride": self.smd_print_stride,
            "smd_segments": self.smd_segments, "smd_frame_steps": self.smd_frame_steps,
            "n_win": self.n_win, "win_start_deg": self.win_start_deg, "win_step_deg": self.win_step_deg,
        }


# ---------------------------------------------------------------- PLUMED text
ROOT_ACTIONS = {"UNITS", "WHOLEMOLECULES", "PRINT", "RESTRAINT", "MOVINGRESTRAINT", "UPPER_WALLS", "LOWER_WALLS",
                "METAD"}


def prune(text: str, keep: list[str]) -> str:
    """Drop every labelled PLUMED action that nothing consumes: only the roots (UNITS, WHOLEMOLECULES, PRINT,
    restraints) and the labels in `keep` survive, together with everything they depend on. Comments stay
    attached to the action that follows them. A label counts as referenced by any identifier in the argument
    text, which can only keep too much, never too little."""
    acts: list[tuple[str | None, str, set[str], str]] = []    # (label or None, action name, identifiers, line)
    for line in text.splitlines():
        m = re.match(r"^\s*(\w+)\s*:\s*(\w+)\b(.*)$", line)
        if m:
            acts.append((m.group(1), m.group(2), set(re.findall(r"[A-Za-z_]\w*", m.group(3))), line))
        elif line.strip() and not line.lstrip().startswith("#"):
            first = line.split()[0]
            acts.append((None, first, set(re.findall(r"[A-Za-z_]\w*", line[len(first):])), line))
    labels = {a[0]: a for a in acts if a[0]}
    missing = sorted(set(keep) - set(labels))
    if missing:
        raise ValueError(f"plumed_defs.dat does not define {missing}")
    need = {a[0] for a in acts if a[0] and a[1] in ROOT_ACTIONS} | set(keep)
    todo = list(need)
    while todo:
        for ident in labels[todo.pop()][2]:
            if ident in labels and ident not in need:
                need.add(ident)
                todo.append(ident)
    kept = {a[3] for a in acts if a[0] is None or a[0] in need}   # lines are unique in a generated file
    out: list[str] = []
    pending: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            pending.append(line)
            continue
        if line in kept:
            out += [p for p in pending if p.strip() or (out and out[-1].strip())] + [line]
        pending = []
    return "\n".join(out) + "\n"


def window_plumed(defs: str, s: UmbrellaSettings, theta_rad: float) -> str:
    wall = ""
    if s.wall is not None:
        cv, at, kappa = s.wall
        wall = f"uwall: UPPER_WALLS ARG={cv} AT={at:g} KAPPA={kappa:g}\n"
    args = ",".join([*s.print_args, "rst.bias"] + (["uwall.bias"] if wall else []))
    return (defs + f"\nrst: RESTRAINT ARG=theta AT={theta_rad:.6f} KAPPA={s.kappa_umb:g}\n" + wall + "\n"
            f"PRINT ARG={args} STRIDE={s.umb_print_stride} FILE=COLVAR\n")


def ref_pdb_text(solute: GroFile, atoms: tuple[int, ...]) -> str:
    """PLUMED RMSD reference: the selected atoms only, serial = global index, occupancy (alignment weight) =
    beta (displacement weight) = 1, Angstrom. The coordinates are replaced from the equilibrated frame on the
    node (make_ref_from_gro.py), so any structure with the same atom order will do here."""
    out = ["REMARK  rigid block, reference for PLUMED RMSD TYPE=OPTIMAL"]
    for a in atoms:
        atom = solute.atoms[a - 1]  # by position: the atom-number column of a .gro wraps at 100000
        x, y, z = atom.coordinate * 10
        out.append(f"ATOM  {a:5d} {atom.atom_name:<4s}{atom.residue_name:>4s} A{atom.residue_number % 10000:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  1.00")
    return "\n".join(out + ["END", ""])


# ---------------------------------------------------------------- custom steps
@dataclass(kw_only=True)
class SteeredMD(MD):
    """MD whose plumed.dat is written on the node: grommp.sh first refreshes ref_block.pdb from input.gro
    and runs ``umbrella_node.py smd-prep`` (plumed.dat, index.ndx for freezegrps), then grompp."""

    def __post_init__(self) -> None:
        if self.plumed is None:  # mdrun must run with -plumed; the real input is written before grompp
            self.plumed = "# replaced by umbrella_node.py smd-prep before grompp\n"
        super().__post_init__()

    @override
    def generate(self) -> dict[str, str]:
        files = super().generate()
        prep = ""
        if "ref_block.pdb" in files:
            prep += "python3 make_ref_from_gro.py input.gro ref_block.pdb\n"
        prep += "python3 umbrella_node.py smd-prep --dir . --gro input.gro\n"
        files["grommp.sh"] = files["grommp.sh"].replace('"$GMX" grompp', prep + '"$GMX" grompp', 1)
        return files


class ExtractWindows(Calculation):
    """Cut one start frame per umbrella window out of ../smd (into ../wNN/input.gro); done = ../smd/output.gro."""

    def __init__(self, calculation_name: str, config_json: str):
        self.calculation_name = validate_name(calculation_name)
        self.config_json = config_json

    @override
    def generate(self) -> dict[str, str]:
        return {
            "grommp.sh": NOOP_GROMPP,
            "mdrun.sh": body_script("export GMX\n"
                                    "python3 umbrella_node.py extract --dir ../smd --out ..\n"
                                    "cp ../smd/output.gro output.gro"),
            "umbrella_config.json": self.config_json,
            "umbrella_node.py": (NODE / "umbrella_node.py").read_text(),
        }


# ---------------------------------------------------------------- the tree
def window_nodes(defs: str, s: UmbrellaSettings) -> list[tuple[str, Calculation]]:
    """[(wNN, Nested 00_relax -> 01_umb)] of all windows; each wNN needs input.gro, topo.top and *.itp."""
    relax_steps, umb_steps = s.steps(s.relax_ns), s.steps(s.umb_ns)
    windows: list[tuple[str, Calculation]] = []
    for i in range(s.n_win):
        plumed = window_plumed(defs, s, math.radians(s.win_start_deg + i * s.win_step_deg))
        relax = MD(type=MDType.v_rescale_c_rescale, calculation_name="relax", nsteps=relax_steps,
                   nstout=relax_steps // 10, gen_vel="yes", maxwarn=1, plumed=plumed, mdrun_args=list(s.mdrun_args))
        umb = MD(type=MDType.v_rescale_c_rescale, calculation_name="umb", nsteps=umb_steps, nstout=s.umb_nstout,
                 gen_vel="no", maxwarn=1, plumed=plumed, mdrun_args=list(s.mdrun_args),
                 additional_mdp_parameters={"nstvout": "0", "nstfout": "0"})
        windows.append((f"w{i:02d}", Nested(f"w{i:02d}", [("00_relax", relax), ("01_umb", umb)], chain=True)))
    return windows


def window_cvs(s: UmbrellaSettings) -> list[str]:
    """CV labels the windows need from plumed_defs.dat."""
    return list(dict.fromkeys([*s.print_args, "theta", *([s.wall[0]] if s.wall else [])]))


def build_umbrella(s: UmbrellaSettings, plumed_defs: str, solute: GroFile) -> list[Calculation]:
    """The whole pipeline: relaxation steps, then the Nested umbrella step (smd, extract, windows)."""
    cfg = json.dumps(s.node_config(), indent=1) + "\n"
    smd_keep = list(dict.fromkeys([*s.print_args, *(p for p, _ in s.pose), "theta"]))
    smd_defs = prune(plumed_defs, smd_keep)
    win_defs = prune(plumed_defs, window_cvs(s))
    extra = {"plumed_defs.dat": smd_defs, "umbrella_config.json": cfg,
             "umbrella_node.py": (NODE / "umbrella_node.py").read_text()}
    if s.rigid_atoms:
        if max(s.rigid_atoms) > len(solute):
            raise ValueError(f"rigid_atoms go up to {max(s.rigid_atoms)}, the solute has {len(solute)} atoms")
        extra["ref_block.pdb"] = ref_pdb_text(solute, s.rigid_atoms)
        extra["make_ref_from_gro.py"] = (NODE / "make_ref_from_gro.py").read_text()
    smd = SteeredMD(
        type=MDType.v_rescale_only_nvt, calculation_name="smd",
        nsteps=s.smd_hold_steps + s.steps(s.smd_ns) + s.smd_end_hold_steps, nstout=s.smd_frame_steps,
        gen_vel="no", maxwarn=1, index_file="index.ndx", extra_files=extra, mdrun_args=list(s.mdrun_args),
        additional_mdp_parameters={"tc_grps": "System", "freezegrps": "lower", "freezedim": "Y Y Y",
                                   "comm_mode": "None", "nstvout": "0", "nstfout": "0"})
    windows = window_nodes(win_defs, s)
    umbrella = Nested("umbrella", [("smd", smd), ("extract", ExtractWindows("extract", cfg)), *windows],
                      broadcast=True, parallel=s.parallel, sequential_first=2)
    defines = list(s.defines)
    return [
        EM(calculation_name="em", nsteps=s.em_nsteps, defines=defines, maxwarn=1),
        SolvationMCH("solv"),
        EM(calculation_name="em2", nsteps=s.em2_nsteps, defines=defines, maxwarn=1),
        MD(type=MDType.v_rescale_c_rescale, calculation_name="equil_lock", nsteps=s.equil_nsteps,
           nstout=s.equil_nstout, gen_vel="yes", useRestraint=True, defines=defines, maxwarn=1),
        MD(type=MDType.v_rescale_c_rescale, calculation_name="equil_unlock", nsteps=s.equil_nsteps,
           nstout=s.equil_nstout, gen_vel="no", maxwarn=1),
        umbrella,
    ]


def build_windows_only(s: UmbrellaSettings, plumed_defs: str) -> list[Calculation]:
    """The windows alone (one Nested step); every ``wNN/input.gro`` has to be placed separately."""
    win_defs = prune(plumed_defs, window_cvs(s))
    windows = window_nodes(win_defs, s)
    return [Nested("windows", windows, broadcast=True, input_to="none", parallel=s.parallel)]


def parse_ranges(text: str) -> tuple[int, ...]:
    """'1-3,7' -> (1, 2, 3, 7) (PLUMED atom-range notation, the inverse of gmx_harness.plumed.compress)."""
    out: list[int] = []
    for part in text.replace(" ", "").split(","):
        if part:
            lo, _, hi = part.partition("-")
            out += range(int(lo), int(hi or lo) + 1)
    if compress(sorted(set(out))) != compress(out):
        raise ValueError("atom ranges must be ascending and must not overlap")
    return tuple(out)
