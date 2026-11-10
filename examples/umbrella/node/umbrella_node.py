"""Node-side helpers of the umbrella-sampling sample (standard library only, Python 3.6).

Called by the generated scripts of the umbrella step (SteeredMD's grommp.sh, ExtractWindows' mdrun.sh;
see ../umbrella.py). Both read umbrella_config.json from the current folder.

  smd-prep  --dir . --gro input.gro                (run inside <umbrella>/smd)
      Reads the CVs of the frame with `plumed driver`, then writes the steering input
      plumed.dat, index.ndx and smd_params.json next to it (plumed_defs.dat, ref_block.pdb are already there).
  extract   --dir ../smd --out ..                  (run inside <umbrella>/extract)
      After the steered MD: for every umbrella window theta_i picks the SMD frame whose (unwrapped) theta
      is closest to it, dumps it with `gmx trjconv` to <out>/wNN/input.gro and writes windows.json here.

All numbers come from umbrella_config.json, never from this file. The biased CV is always the label
`theta` of plumed_defs.dat; its period (360 degrees, or less for a symmetric block) is `period_deg`.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def gmx():
    """The GROMACS binary: $GMX (as the gmx_harness scripts set it), else gmx_d, gmx_mpi or gmx."""
    if os.environ.get("GMX"):
        return os.environ["GMX"]
    for name in ("gmx_d", "gmx_mpi", "gmx"):
        if shutil.which(name):
            return name
    sys.exit("no gromacs installation found (set GMX)")


def read_gro(path):
    lines = Path(path).read_text().splitlines()
    return int(lines[1])


def read_colvar(path):
    """-> (field names, rows of floats) of a PLUMED COLVAR."""
    names, rows = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#! FIELDS"):
            names = line.split()[2:]
        elif line and not line.startswith("#"):
            rows.append([float(v) for v in line.split()])
    return names, rows


def period(cfg):
    return math.radians(cfg["period_deg"])


def wrap_half(x, per):
    """x wrapped into [-per/2, per/2)."""
    return x - per * math.floor(x / per + 0.5)


def steer_line(th0, hold, ramp, kappa, nseg, per):
    """MOVINGRESTRAINT: hold at th0 for `hold` steps, then nseg equal segments up to th0 + one period."""
    pts = [(0, th0), (hold, th0)] + [(hold + int(round(ramp * j / nseg)), th0 + per * j / nseg)
                                     for j in range(1, nseg + 1)]
    return "steer: MOVINGRESTRAINT ARG=theta " + " ".join(
        "STEP%d=%d AT%d=%.6f KAPPA%d=%g" % (i, st, i, at, i, kappa) for i, (st, at) in enumerate(pts))


def smd_args(cfg):
    """CVs printed in the steered MD: the window CVs plus the pose CVs not already among them."""
    args = list(cfg["print_args"])
    return args + [a for a, _ in cfg["pose"] if a not in args]


def smd_prep(a):
    cfg = json.loads(Path(a.config).read_text())
    d = Path(a.dir)
    defs = (d / "plumed_defs.dat").read_text()
    args = smd_args(cfg)

    # ---- values of the equilibrated frame ----
    (d / "init.dat").write_text(defs + "\nPRINT ARG=%s STRIDE=1 FILE=init.COLVAR\n" % ",".join(args))
    r = subprocess.run(["plumed", "driver", "--plumed", "init.dat", "--igro", str(Path(a.gro).resolve()),
                        "--length-units", "nm"], cwd=str(d), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    if r.returncode != 0:
        sys.exit("plumed driver failed:\n" + r.stdout[-3000:])
    names, rows = read_colvar(d / "init.COLVAR")
    v = dict(zip(names, rows[0]))
    th0 = v["theta"]
    print("start frame: " + "  ".join("%s = %.4f" % (k, v[k]) for k in args))

    hold = cfg["smd_hold_steps"]
    ramp = cfg["smd_ramp_steps"]
    pose = cfg["pose"]
    L = [defs.rstrip(), ""]
    biases = []
    if cfg["n_rigid"]:
        L += ["# ---- upper block rigid: internal RMSD of its %d reference atoms against the start frame ----"
              % cfg["n_rigid"],
              "# (the lower block is frozen in GROMACS; per-atom spring %g kJ/mol/nm^2 = kappa/N)" % cfg["per_atom_k"],
              "rmsdBlk: RMSD REFERENCE=ref_block.pdb TYPE=OPTIMAL",
              "rigid: RESTRAINT ARG=rmsdBlk AT=0 KAPPA=%g" % (cfg["per_atom_k"] * cfg["n_rigid"])]
        args = args + ["rmsdBlk"]
        biases.append("rigid.bias")
    if pose:
        L += ["# ---- pose of the upper block: only theta moves ----",
              "pose: RESTRAINT ARG=%s AT=%s KAPPA=%s" % (
                  ",".join(p for p, _ in pose), ",".join("%.6f" % v[p] for p, _ in pose),
                  ",".join("%g" % k for _, k in pose))]
        biases.append("pose.bias")
    L += ["# ---- steering: hold, then theta0 -> theta0 + one period ----",
          "# PLUMED interpolates a periodic CV along the SHORTEST path, so the sweep is cut into segments that",
          "# are each shorter than half a period (a single full-period segment is a zero path and would not move)",
          steer_line(th0, hold, ramp, cfg["kappa_smd"], cfg["smd_segments"], period(cfg)),
          "",
          "PRINT ARG=%s STRIDE=%d FILE=COLVAR" % (
              ",".join(args + ["steer.theta_cntr", "steer.theta_work"] + biases), cfg["smd_print_stride"]), ""]
    (d / "plumed.dat").write_text("\n".join(L))

    # ---- index groups for freezegrps ----
    n = read_gro(a.gro)

    def grp(name, lo, hi):
        ids = list(range(lo, hi + 1))
        rows = [" ".join("%d" % i for i in ids[j:j + 15]) for j in range(0, len(ids), 15)]
        return "[ %s ]\n%s\n" % (name, "\n".join(rows))
    (d / "index.ndx").write_text(grp("System", 1, n) + grp("lower", *cfg["lower"]) + grp("upper", *cfg["upper"]))

    params = {"theta0": th0}
    params.update((p + "0", v[p]) for p, _ in pose)
    params.update({"hold_steps": hold, "ramp_steps": ramp, "end_hold_steps": cfg["smd_end_hold_steps"],
                   "dt_ps": cfg["dt"]})
    (d / "smd_params.json").write_text(json.dumps(params, indent=1) + "\n")
    print("wrote %s/{plumed.dat,index.ndx,smd_params.json}" % d)


def extract(a):
    cfg = json.loads(Path(a.config).read_text())
    per = period(cfg)
    d = Path(a.dir)
    p = json.loads((d / "smd_params.json").read_text())
    names, rows = read_colvar(d / "COLVAR")
    it, ith, ic = names.index("time"), names.index("theta"), names.index("steer.theta_cntr")
    dt_frame = cfg["smd_frame_steps"] * cfg["dt"]               # ps between trr frames
    t_start = p["hold_steps"] * cfg["dt"]
    t_end = (p["hold_steps"] + p["ramp_steps"] + p["end_hold_steps"]) * cfg["dt"]   # up to the end of the final hold

    # unwrapped theta of the system: the steering centre plus the wrapped lag
    traj = []
    for r in rows:
        t = r[it]
        if t < t_start - 1e-6 or t > t_end + 1e-6:
            continue
        k = t / dt_frame
        if abs(k - round(k)) > 1e-4:                              # only times that have a trr frame
            continue
        traj.append((t, r[ic] + wrap_half(r[ith] - r[ic], per)))
    if not traj:
        sys.exit("no steering frames found in %s/COLVAR" % d)

    out = Path(a.out)
    wins = []
    for i in range(cfg["n_win"]):
        target = math.radians(cfg["win_start_deg"] + i * cfg["win_step_deg"])
        # the same state recurs every period: take the copy inside the swept range
        lo = p["theta0"]
        base = target + per * math.ceil((lo - target) / per - 1e-9)
        t, thu = min(traj, key=lambda x: abs(x[1] - base))
        err = math.degrees(thu - base)
        wd = out / ("w%02d" % i)
        wd.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([gmx(), "trjconv", "-s", str(d / "output.tpr"), "-f", str(d / "output.trr"),
                            "-dump", "%g" % t, "-o", str(wd / "input.gro")], input="0\n",
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if r.returncode != 0 or not (wd / "input.gro").exists():
            sys.exit("trjconv failed for window %d:\n%s" % (i, r.stdout[-2000:]))
        wins.append({"index": i, "theta_deg": math.degrees(target), "theta_rad": wrap_half(target, per),
                     "frame_ps": t, "frame_theta_deg": math.degrees(thu), "miss_deg": err})
        print("w%02d  target %+6.1f deg  frame %.0f ps  (%.2f deg off)" % (i, math.degrees(target), t, err))
    Path("windows.json").write_text(json.dumps(wins, indent=1) + "\n")
    worst = max(abs(w["miss_deg"]) for w in wins)
    print("window frames written to %s/wNN/input.gro, worst miss %.2f deg" % (out, worst))
    if worst > 1.0:
        print("WARNING: a window frame is more than 1 degree away from its target", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True                                           # Python 3.6: no required= in add_subparsers
    for name, fn in (("smd-prep", smd_prep), ("extract", extract)):
        s = sub.add_parser(name)
        s.add_argument("--dir", required=True)
        s.add_argument("--config", default="umbrella_config.json")
        if name == "smd-prep":
            s.add_argument("--gro", required=True)
        else:
            s.add_argument("--out", required=True)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
