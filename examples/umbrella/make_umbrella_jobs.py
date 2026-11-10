"""Write an umbrella-sampling job tree (see umbrella.py and README.md) from a JSON config.

Usage:
    uv run python examples/umbrella/make_umbrella_jobs.py CONFIG.json OUT_DIR [--test] [--replace]
    uv run python examples/umbrella/make_umbrella_jobs.py CONFIG.json OUT_DIR --windows-only START_DIR

CONFIG.json (paths relative to the config file)::

    {
      "solute_gro": "system/solute.gro",          start structure (solute only, before solvation)
      "topology": "system/topo.top",              copied as topo.top
      "itps": ["system/MOL.itp", ...],            included by the topology
      "plumed_defs": "plumed_defs.dat",           CV definitions; must define `theta` and every CV named below
      "rigid_atoms": "1201-1230,1251-1280",       upper-block atoms held rigid in the steered MD ("" = none)
      "settings": {"lower": [1, 1200], "upper": [1201, 2400], ...},  UmbrellaSettings fields
      "waive": {"L004": "the grompp warning ... is expected"}        design-check waivers (optional)
    }

The MD steps run grompp with maxwarn=1, which the design checks report as L004 errors: put the
grompp warning you expect into "waive" (or a narrower "L004@<step>" key) after checking it.

--windows-only START_DIR: only the windows, each from START_DIR/wNN/input.gro (e.g. the windows of an earlier
umbrella job), with the topology and *.itp of START_DIR/w00 (the solvated system).
--test: every MD step 0.1 ns. Nothing is run or submitted; the job's run.sh does everything.
"""

import argparse
import json
from pathlib import Path

from gmx_harness import GroFile, OverwritePolicy, build_plan
from gmx_harness.pipeline import PlannedFile

import umbrella


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--windows-only", type=Path, metavar="START_DIR")
    ap.add_argument("--test", action="store_true", help="every MD step 0.1 ns")
    ap.add_argument("--replace", action="store_true", help="regenerate files gmx_harness wrote earlier")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    base = a.config.resolve().parent
    s = umbrella.UmbrellaSettings.from_dict(cfg.get("settings", {}))
    if cfg.get("rigid_atoms"):
        s = umbrella.UmbrellaSettings(**{**s.__dict__, "rigid_atoms": umbrella.parse_ranges(cfg["rigid_atoms"])})
    if a.test:
        s = s.test_mode()
    defs = (base / cfg["plumed_defs"]).read_text()
    waive = {**umbrella.WAIVE, **cfg.get("waive", {})}

    if a.windows_only:
        src = a.windows_only
        frames = [src / f"w{i:02d}" / "input.gro" for i in range(s.n_win)]
        plan = build_plan(umbrella.build_windows_only(s, defs), frames[0], a.out,
                          extra_inputs={"topo.top": src / "w00" / "topo.top",
                                        **{p.name: p for p in sorted((src / "w00").glob("*.itp"))}},
                          waive=waive)
        step = plan.steps[0].dirname
        plan.files += [PlannedFile(f"{step}/w{i:02d}/input.gro", f.read_bytes()) for i, f in enumerate(frames)]
    else:
        gro = base / cfg["solute_gro"]
        plan = build_plan(umbrella.build_umbrella(s, defs, GroFile.from_gro_file(str(gro))), gro, a.out,
                          extra_inputs={"topo.top": base / cfg["topology"],
                                        **{Path(p).name: base / p for p in cfg.get("itps", [])}},
                          waive=waive)
    policy = OverwritePolicy.REPLACE_GENERATED if a.replace else OverwritePolicy.ERROR
    pv = plan.preview(policy)
    if not pv.ok:
        print(pv)
    pv = plan.write(policy)
    print(f"{a.out}: {len(pv.create)} files created, {len(pv.overwrite)} overwritten, "
          f"{len(pv.started_steps)} steps kept because they already ran")


if __name__ == "__main__":
    main()
