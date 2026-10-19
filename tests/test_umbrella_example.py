"""examples/umbrella: the job tree it builds and its node-side helpers (run with fake plumed / gmx)."""

import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from gmx_harness import GroFile, Nested, build_plan

from .test_topology import BASH

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "umbrella"
sys.path.insert(0, str(EXAMPLE))
import umbrella  # noqa: E402


def _node() -> ModuleType:
    spec = importlib.util.spec_from_file_location("umbrella_node", EXAMPLE / "node" / "umbrella_node.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


node = _node()

GRO = "test\n 4\n" + "".join(
    f"{i // 2 + 1:5d}MOL  {'C' + str(i):>5s}{i + 1:5d}{0.1 * i:8.3f}{0.2:8.3f}{0.3:8.3f}\n" for i in range(4)
) + "   3.00000   3.00000   3.00000\n"

DEFS = """UNITS LENGTH=nm ENERGY=kj/mol
WHOLEMOLECULES ENTITY0=1-2 ENTITY1=3-4
# centres
cLo: CENTER ATOMS=1,2
cUp: CENTER ATOMS=3,4
unused: DISTANCE ATOMS=1,4
d: DISTANCE ATOMS=cLo,cUp COMPONENTS
theta: TORSION ATOMS=1,2,3,4
order: CUSTOM ARG=theta FUNC=cos(x) PERIODIC=NO
lat: CUSTOM ARG=d.x,d.y VAR=x,y FUNC=sqrt(x*x+y*y) PERIODIC=NO
sep: COMBINE ARG=d.z PERIODIC=NO
tilt: COMBINE ARG=d.x PERIODIC=NO
"""

SOLUTE = GroFile.from_gro_text(GRO.splitlines())
SETTINGS = umbrella.UmbrellaSettings(
    lower=(1, 2), upper=(3, 4), rigid_atoms=(3, 4), period_deg=60.0, n_win=4, win_start_deg=-30.0, win_step_deg=3.0,
    kappa_umb=3000.0, print_args=("theta", "order", "lat", "sep"), pose=(("lat", 10000.0), ("sep", 10000.0), ("tilt", 20000.0)),
).test_mode()


class TestTree(unittest.TestCase):
    def test_pipeline(self) -> None:
        calcs = umbrella.build_umbrella(SETTINGS, DEFS, SOLUTE)
        self.assertEqual([c.name for c in calcs], ["em", "solv", "em2", "equil_lock", "equil_unlock", "umbrella"])
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.gro").write_text(GRO)
            plan = build_plan(calcs, Path(d) / "s.gro", Path(d) / "w")
            smd_grompp = plan.file("5_umbrella/smd/grommp.sh")
            self.assertLess(smd_grompp.index("umbrella_node.py smd-prep"), smd_grompp.index('"$GMX" grompp'))
            self.assertIn("-n index.ndx", smd_grompp)
            self.assertIn("-plumed plumed.dat", plan.file("5_umbrella/smd/mdrun.sh"))
            self.assertIn("freezegrps", plan.file("5_umbrella/smd/setting.mdp"))
            self.assertIn("ATOM      3 C2", plan.file("5_umbrella/smd/ref_block.pdb"))
            defs = plan.file("5_umbrella/smd/plumed_defs.dat")
            self.assertIn("tilt:", defs)  # pose CV
            self.assertNotIn("unused:", defs)  # pruned
            for i in range(4):
                text = plan.file(f"5_umbrella/w{i:02d}/01_umb/plumed.dat")
                self.assertIn(f"RESTRAINT ARG=theta AT={math.radians(-30 + 3 * i):.6f} KAPPA=3000", text)
                self.assertNotIn("tilt:", text)
                self.assertNotIn("UPPER_WALLS", text)
            self.assertIn("python3 umbrella_node.py extract", plan.file("5_umbrella/extract/mdrun.sh"))
            # the checks look inside 5_umbrella; with the example's own waiver only the maxwarn=1 errors are left
            waived = build_plan(calcs, Path(d) / "s.gro", Path(d) / "x", waive=umbrella.WAIVE)
            errors = {i.code for i in waived.report.errors(umbrella.WAIVE)}
            self.assertEqual(errors, {"L004"})
            self.assertIn("5_umbrella/w00/01_umb", {i.where for i in waived.report.issues if i.code == "L004"})

    def test_windows_only_with_wall(self) -> None:
        import dataclasses

        s = dataclasses.replace(SETTINGS, wall=("lat", 0.4, 50000.0), parallel=2)
        (windows,) = umbrella.build_windows_only(s, DEFS)
        assert isinstance(windows, Nested)
        self.assertEqual((windows.input_to, windows.parallel), ("none", 2))
        text = windows.generate()["w01/00_relax/plumed.dat"]
        self.assertIn("uwall: UPPER_WALLS ARG=lat AT=0.4 KAPPA=50000", text)
        self.assertIn("uwall.bias", text)

    def test_rigid_atoms_must_exist(self) -> None:
        import dataclasses

        with self.assertRaises(ValueError):
            umbrella.build_umbrella(dataclasses.replace(SETTINGS, rigid_atoms=(3, 5)), DEFS, SOLUTE)

    def test_generic_defaults(self) -> None:
        s = umbrella.UmbrellaSettings(lower=(1, 2), upper=(3, 4))
        calcs = umbrella.build_umbrella(s, DEFS, SOLUTE)
        files = calcs[-1].generate()
        self.assertEqual(len([k for k in files if k.endswith("01_umb/plumed.dat")]), 36)
        self.assertIn("RESTRAINT ARG=theta AT=-3.141593", files["w00/01_umb/plumed.dat"])
        self.assertNotIn("ref_block.pdb", "".join(files))  # no rigid atoms

    def test_prune_reports_missing_cvs(self) -> None:
        with self.assertRaises(ValueError):
            umbrella.prune(DEFS, ["theta", "nope"])

    def test_settings_from_dict(self) -> None:
        s = umbrella.UmbrellaSettings.from_dict({"lower": [1, 10], "pose": [["sep", 1.0]]})
        self.assertEqual((s.lower, s.pose), ((1, 10), (("sep", 1.0),)))
        with self.assertRaises(ValueError):
            umbrella.UmbrellaSettings.from_dict({"kapa_umb": 1})
        self.assertEqual(umbrella.parse_ranges("1-3,7, 9-10"), (1, 2, 3, 7, 9, 10))


class TestNodeHelpers(unittest.TestCase):
    def test_steer_line(self) -> None:
        per = math.pi / 3
        line = node.steer_line(0.1, 100, 600, 60000.0, 6, per)
        self.assertIn("STEP0=0 AT0=0.100000 KAPPA0=60000", line)
        self.assertIn("STEP1=100 AT1=0.100000", line)
        self.assertIn(f"STEP7=700 AT7={0.1 + per:.6f}", line)

    def test_wrap_half(self) -> None:
        per = math.pi / 3
        self.assertAlmostEqual(node.wrap_half(per * 0.6, per), -per * 0.4)
        self.assertAlmostEqual(node.wrap_half(-per * 0.5, per), -per * 0.5)


@unittest.skipIf(BASH is None, "bash not available")
class TestNodeRuns(unittest.TestCase):
    """smd-prep and extract in a scratch folder, with stand-ins for `plumed driver` and `gmx trjconv`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        bindir = self.d / "bin"
        bindir.mkdir()
        tools = {
            "plumed": "#!/bin/bash\nprintf '#! FIELDS time theta order lat sep tilt\\n0 0.2 0.9 0.05 0.3 0.01\\n' > init.COLVAR\n",
            "fakegmx": '#!/bin/bash\nprev=""\nfor a in "$@"; do\n'
                       '  if [ "$prev" = "-o" ]; then echo frame > "$a"; fi\n'
                       '  if [ "$prev" = "-dump" ]; then echo "$a" >> "$DUMPS"; fi\n  prev="$a"\ndone\n',
        }
        for name, text in tools.items():
            (bindir / name).write_text(text)
            (bindir / name).chmod((bindir / name).stat().st_mode | stat.S_IXUSR)
        self.env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}", GMX=str(bindir / "fakegmx"),
                        DUMPS=str(self.d / "dumps.txt"))
        files = umbrella.build_umbrella(SETTINGS, DEFS, SOLUTE)[-1].generate()
        for rel, text in files.items():
            if rel.split("/")[0] in ("smd", "extract"):
                (self.d / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.d / rel).write_text(text)
        (self.d / "smd" / "input.gro").write_text(GRO)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def node(self, cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "umbrella_node.py", *args], cwd=self.d / cwd, env=self.env,
                              capture_output=True, text=True)

    def test_smd_prep_then_extract(self) -> None:
        r = self.node("smd", "smd-prep", "--dir", ".", "--gro", "input.gro")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        smd = self.d / "smd"
        plumed = (smd / "plumed.dat").read_text()
        self.assertIn("pose: RESTRAINT ARG=lat,sep,tilt AT=0.050000,0.300000,0.010000 KAPPA=10000,10000,20000",
                      plumed)
        self.assertIn("rigid: RESTRAINT ARG=rmsdBlk AT=0 KAPPA=2000", plumed)  # 2 atoms x 1000
        self.assertIn("steer: MOVINGRESTRAINT ARG=theta STEP0=0 AT0=0.200000", plumed)
        self.assertIn("PRINT ARG=theta,order,lat,sep,tilt,rmsdBlk,steer.theta_cntr,steer.theta_work,rigid.bias,pose.bias",
                      plumed)
        self.assertEqual((smd / "index.ndx").read_text(), "[ System ]\n1 2 3 4\n[ lower ]\n1 2\n[ upper ]\n3 4\n")
        params = json.loads((smd / "smd_params.json").read_text())
        self.assertEqual(params["theta0"], 0.2)

        # a steered run that tracks its centre exactly: frame k * dt_frame sits at theta0 + rate * (t - t_hold)
        cfg = SETTINGS.node_config()
        dt, hold, ramp = cfg["dt"], params["hold_steps"], params["ramp_steps"]
        per = math.pi / 3
        rows = ["#! FIELDS time theta steer.theta_cntr"]
        for step in range(0, hold + ramp + params["end_hold_steps"] + 1, cfg["smd_print_stride"]):
            f = min(max(step - hold, 0) / ramp, 1.0)
            c = 0.2 + per * f
            rows.append(f"{step * dt:g} {node.wrap_half(c, per):.6f} {c:.6f}")
        (smd / "COLVAR").write_text("\n".join(rows) + "\n")
        r = self.node("extract", "extract", "--dir", "../smd", "--out", "..")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        wins = json.loads((self.d / "extract" / "windows.json").read_text())
        self.assertEqual(len(wins), 4)
        for w in wins:
            self.assertLessEqual(abs(w["miss_deg"]), 0.75 + 1e-6)  # frames 1.5 degrees apart in test mode
            self.assertEqual((self.d / f"w{w['index']:02d}" / "input.gro").read_text(), "frame\n")
        self.assertEqual(len((self.d / "dumps.txt").read_text().split()), 4)
