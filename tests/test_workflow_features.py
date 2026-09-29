"""Restart/extension, hand-over, PLUMED, xtc, index, topology preparation and job scripts."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from gmx_harness import (
    MD,
    MDType,
    UnsafeNameError,
    add_conditional_include,
    build_plan,
    format_ndx,
    molecule_atoms,
    prepare_topology,
    set_molecule_count,
    write_job_scripts,
)
from gmx_harness.scripts import copy_script, extend_script, generate_xtc_script, step_run_script

from .test_topology import BASH

USAGE = Path(__file__).resolve().parents[2] / "usage"
NAME = "BarPNPOMeTDPChiral_gaff2"
GRO = "t\n 1\n    1MOL      C    1   0.000   0.000   0.000\n 3 3 3\n"

# a stand-in for gmx: records its arguments (and stdin) and creates the -o file
FAKE_GMX = """#!/bin/bash
stdin=""
if [ ! -t 0 ]; then stdin=$(cat); fi
echo "$* | $stdin" >> gmx_calls.txt
prev=""
for a in "$@"; do
    if [ "$prev" = "-o" ]; then touch "$a"; fi
    prev="$a"
done
"""


@unittest.skipIf(BASH is None, "bash not available")
class ScriptCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        fake = self.d / "fakegmx"
        fake.write_bytes(FAKE_GMX.encode())
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        self.env = dict(os.environ, GMX=fake.as_posix())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def bash(self, script: str, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        (cwd / "_s.sh").write_bytes(script.encode())
        assert BASH is not None
        return subprocess.run([BASH, "_s.sh", *args], cwd=cwd, env=self.env, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL)

    def calls(self, cwd: Path) -> list[str]:
        f = cwd / "gmx_calls.txt"
        return f.read_text().splitlines() if f.exists() else []


class TestRestartAndHandOver(ScriptCase):
    def step_dir(self, name: str) -> Path:
        d = self.d / name
        d.mkdir()
        (d / "grommp.sh").write_bytes(b"echo GROMPP >> trace.txt\n")
        (d / "mdrun.sh").write_bytes(b"echo MDRUN >> trace.txt\ntouch output.gro\n")
        return d

    def test_grompp_is_skipped_when_continuing(self) -> None:
        d = self.step_dir("1_md")
        self.bash(step_run_script(is_last=True), d)
        self.assertEqual((d / "trace.txt").read_text().split(), ["GROMPP", "MDRUN"])
        (d / "output.gro").unlink()
        (d / "output.tpr").write_text("extended tpr")
        (d / "output.cpt").write_text("cpt")
        (d / "trace.txt").unlink()
        r = self.bash(step_run_script(is_last=True), d)
        self.assertEqual((d / "trace.txt").read_text().split(), ["MDRUN"], r.stdout)
        self.assertEqual((d / "output.tpr").read_text(), "extended tpr")

    def test_finished_next_step_is_not_overwritten(self) -> None:
        a, b = self.step_dir("0_a"), self.step_dir("1_b")
        for f, text in ((a / "topo.top", "vacuum"), (a / "output.gro", GRO), (b / "topo.top", "solvated"),
                        (b / "output.gro", GRO)):
            f.write_text(text)
        (a / "copy.sh").write_bytes(copy_script("a", "1_b", "b").encode())
        r = self.bash(step_run_script(is_last=False), a)  # re-running a finished step
        self.assertEqual((b / "topo.top").read_text(), "solvated", r.stdout)
        self.assertIn("already finished", r.stdout)
        (b / "output.gro").unlink()
        self.bash(step_run_script(is_last=False), a)
        self.assertEqual((b / "topo.top").read_text(), "vacuum")  # an unfinished next step does get the files

    def test_extend(self) -> None:
        d = self.d / "6_md"
        d.mkdir()
        self.assertEqual(self.bash(extend_script(), d, "abc").returncode, 2)
        self.assertEqual(self.bash(extend_script(), d, "1000").returncode, 1)  # nothing to extend yet
        for f in ("output.tpr", "output.cpt", "output.gro"):
            (d / f).write_text("x")
        r = self.bash(extend_script(), d, "200000")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.calls(d), ["convert-tpr -s output.tpr -extend 200000 -o output.tpr | "])
        self.assertFalse((d / "output.gro").exists())
        self.assertTrue((d / "output_before_extend.gro").exists())

    def test_pipeline_stops_on_a_step_without_output(self) -> None:
        plan = build_plan([MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt")],
                          self._gro(), self.d / "w")
        self.assertIn('[ -f 0_nvt/output.gro ] || { echo "ERROR: 0_nvt finished without producing output.gro"',
                      plan.file("run.sh"))

    def _gro(self) -> Path:
        p = self.d / "start.gro"
        p.write_bytes(GRO.encode())
        return p

    def test_carry_patterns(self) -> None:
        steps = [MD(type=MDType.v_rescale_only_nvt, calculation_name="a"),
                 MD(type=MDType.v_rescale_c_rescale, calculation_name="b", gen_vel="no")]
        plan = build_plan(steps, self._gro(), self.d / "w", carry=("*.top", "*.itp", "*.ndx"))
        self.assertIn("for f in *.top *.itp *.ndx; do", plan.file("0_a/copy.sh"))
        for bad in (["../x"], ["*.top; rm -rf ~"], ["a b"]):
            with self.assertRaises(UnsafeNameError):
                build_plan(steps, self._gro(), self.d / "w", carry=bad)


class TestPlumedAndXtc(ScriptCase):
    def test_plumed_restart(self) -> None:
        md = MD(type=MDType.v_rescale_c_rescale, calculation_name="metad", gen_vel="no",
                plumed="d: DISTANCE ATOMS=1,2\nPRINT ARG=d FILE=COLVAR\n")
        files = md.generate()
        self.assertEqual(files["plumed.dat"], "d: DISTANCE ATOMS=1,2\nPRINT ARG=d FILE=COLVAR\n")
        d = self.d / "metad"
        d.mkdir()
        (d / "plumed.dat").write_text(files["plumed.dat"])
        self.bash(files["mdrun.sh"], d)
        (d / "output.cpt").write_text("cpt")
        self.bash(files["mdrun.sh"], d)
        calls = self.calls(d)
        self.assertTrue(calls[0].startswith("mdrun -deffnm output -v -plumed plumed.dat"), calls)
        self.assertTrue(calls[1].startswith("mdrun -deffnm output -v -cpi output.cpt -plumed plumed_restart.dat"), calls)
        self.assertEqual((d / "plumed_restart.dat").read_text().splitlines()[0], "RESTART")
        with self.assertRaises(ValueError):
            MD(type=MDType.v_rescale_c_rescale, calculation_name="x", plumed="x",
               mdrun_args=["-plumed", "p.dat"]).generate()

    def test_generate_xtc_group_and_name(self) -> None:
        d = self.d / "md"
        d.mkdir()
        self.assertEqual(self.bash(generate_xtc_script(), d).returncode, 0)
        r = self.bash(generate_xtc_script(), d, "MOL", "mol_whole.xtc")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.calls(d), [
            "trjconv -f output.trr -s output.tpr -o output.xtc -pbc mol | 0",
            "trjconv -f output.trr -s output.tpr -o mol_whole.xtc -pbc mol | MOL",
        ])

    def test_md_steps_get_extend_sh(self) -> None:
        self.assertIn("extend.sh", MD(type=MDType.v_rescale_only_nvt, calculation_name="x").generate())


class TestIndexAndTopology(unittest.TestCase):
    def test_molecule_atoms(self) -> None:
        self.assertEqual(molecule_atoms(3, [0, 2]), [1, 2, 3, 7, 8, 9])
        self.assertEqual(molecule_atoms(3, range(2), local=[2]), [2, 5])
        self.assertEqual(molecule_atoms(3, [0], offset=10), [11, 12, 13])
        with self.assertRaises(ValueError):
            molecule_atoms(3, [0], local=[4])
        text = format_ndx({"A": list(range(1, 17))})
        self.assertEqual(text.splitlines(), ["[ A ]", " ".join(f"{i:4d}" for i in range(1, 16)), "  16"])
        with self.assertRaises(UnsafeNameError):
            format_ndx({"bad ]\n[ x": [1]})

    def test_topology_preparation(self) -> None:
        top = "[ system ]\nx\n\n[ molecules ]\n; Compound  nmols\n MOL              1\n"
        out = add_conditional_include(set_molecule_count(top, 216), "MOL_hbond.itp")
        self.assertEqual(out.splitlines()[-5:], [" MOL              216", "; following sections are added automatically",
                                                  "#ifdef INTER", '#include "MOL_hbond.itp"', "#endif"])
        with self.assertRaises(ValueError):
            set_molecule_count("[ system ]\nx\n", 2)
        with self.assertRaises(UnsafeNameError):
            add_conditional_include(top, "x.itp", define="A; B")


@unittest.skipUnless((USAGE / "resource").exists(), "usage workspace not present")
class TestAgainstUsageWorkspace(unittest.TestCase):
    """The same inputs as the usage workspace reproduce its files byte for byte."""

    def test_index_files(self) -> None:
        for sp, groups in ((f"{NAME}_fiber_rot_+10", {"Fiber1": range(216), "fiberA": range(216)}),
                           (f"{NAME}_stack10_rot_+10", {"Fiber1": range(60), "fiberA": range(60)}),
                           (f"{NAME}_bundle_ccw", {"Fiber1": range(216), "Fiber2": range(216, 432),
                                                   "fiberA": range(216), "fiberB": range(216, 432)})):
            text = format_ndx({k: molecule_atoms(169, v) for k, v in groups.items()})
            real = (USAGE / "resource/sp" / f"{sp}.ndx").read_bytes().decode().replace("\r\n", "\n")
            self.assertEqual(text, real, sp)

    def test_fixed_topologies(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for tag, n in (("", 216), ("_bundle", 432), ("_stack10", 60)):
                out = Path(d, f"{NAME}{tag}_fixed.top")
                prepare_topology(USAGE / "resource/top" / f"{NAME}.top", out, n, f"{NAME}{tag}_hbond.itp")
                real = (USAGE / "resource/top_fixed" / f"{NAME}{tag}_fixed.top").read_bytes()
                self.assertEqual(out.read_bytes(), real.replace(b"\r\n", b"\n"), tag)

    def test_job_scripts(self) -> None:
        template = (USAGE / "resource/Gromacs.sbatch").read_text()
        with tempfile.TemporaryDirectory() as d:
            systems = [f"{NAME}_fiber_rot_+10", f"{NAME}_fiber_rot_-10"]
            for s in systems:
                Path(d, s).mkdir()
            write_job_scripts(d, systems, template)
            for s in systems:
                real = (USAGE / "20260919_first_time_PBC" / s / "Gromacs.sbatch").read_bytes()
                self.assertEqual(Path(d, s, "Gromacs.sbatch").read_bytes(), real.replace(b"\r\n", b"\n"), s)
            submit = Path(d, "submit.sh").read_text()
            self.assertIn(f"(cd {NAME}_fiber_rot_+10 && sbatch Gromacs.sbatch)", submit)
            self.assertIn("sbatch -d singleton Gromacs.sbatch", Path(d, "submit_restart.sh").read_text())
            Path(d, systems[0], "Gromacs.sbatch").write_text("edited")
            with self.assertRaises(FileExistsError):
                write_job_scripts(d, systems, template)
            with self.assertRaises(ValueError):
                write_job_scripts(d, systems, "{MISSING}", overwrite=True)


if __name__ == "__main__":
    unittest.main()
