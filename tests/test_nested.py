"""Nested (composite) steps: layout, validation, JSON, and the generated scripts run for real with a fake gmx."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from gmx_harness import (
    MD,
    AddFiles,
    Calculation,
    MDType,
    Nested,
    OverwritePolicy,
    RawShellStep,
    UnsafeNameError,
    UnsafeOperationError,
    build_plan,
    from_json,
    to_json,
)

from .test_workflow_features import GRO, ScriptCase


def child(name: str, extra: str = "") -> RawShellStep:
    """A step that records that it ran (in ../../ran.txt) and its thread count, then passes input.gro on."""
    cmd = (f'echo {name} "$OMP_NUM_THREADS" >> ../ran.txt\n{extra}\n'
           "cp input.gro output.gro\n")
    return RawShellStep(name, cmd, allow_unsafe=True)


class TestNestedLayout(unittest.TestCase):
    def test_files(self) -> None:
        inner = Nested("w00", [("00_relax", child("relax")), ("01_umb", child("umb"))], chain=True)
        outer = Nested("umbrella", [child("smd"), inner])
        files = outer.generate()
        for rel in ("mdrun.sh", "grommp.sh", "smd/run.sh", "smd/mdrun.sh", "w00/run.sh", "w00/mdrun.sh",
                    "w00/00_relax/run.sh", "w00/00_relax/copy.sh", "w00/01_umb/run.sh"):
            self.assertIn(rel, files)
        self.assertNotIn("w00/01_umb/copy.sh", files)  # last of the chain
        self.assertNotIn("smd/copy.sh", files)  # not chained
        self.assertIn("cp output.gro ../01_umb/input.gro", files["w00/00_relax/copy.sh"])
        self.assertIn("cp 01_umb/output.gro output.gro", files["w00/mdrun.sh"])
        self.assertIn("cp input.gro output.gro", files["mdrun.sh"])

    def test_bare_steps_use_their_name(self) -> None:
        self.assertEqual([s for s, _ in Nested("n", [child("a"), child("b")]).children], ["a", "b"])

    def test_validation(self) -> None:
        with self.assertRaises(UnsafeNameError):
            Nested("n", [("../x", child("a"))])
        with self.assertRaises(ValueError):
            Nested("n", [child("a"), ("a", child("b"))])  # duplicate folder
        with self.assertRaises(ValueError):
            Nested("n", [])
        with self.assertRaises(ValueError):
            Nested("n", [child("a"), child("b")], chain=True, parallel=2)
        with self.assertRaises(ValueError):
            Nested("n", [child("a"), child("b")], chain=True, broadcast=True)
        with self.assertRaises(ValueError):
            Nested("n", [child("a")], input_to="some")
        with self.assertRaises(ValueError):
            Nested("n", [child("a")], parallel=0)
        with self.assertRaises(ValueError):
            Nested("n", [child("a")], parallel=2, sequential_first=2)
        with self.assertRaises(UnsafeNameError):
            Nested("n", [child("a")], carry=["*.top; rm -rf ~"])

    def test_child_cannot_write_wiring(self) -> None:
        class Bad(Calculation):
            calculation_name = "bad"

            def generate(self) -> dict[str, str]:
                return {"mdrun.sh": "", "run.sh": ""}

        with self.assertRaises(ValueError):
            Nested("n", [Bad()]).generate()

    def test_json_round_trip(self) -> None:
        tree = Nested("umbrella", [
            ("smd", MD(type=MDType.v_rescale_only_nvt, calculation_name="smd", index_file="index.ndx")),
            ("w00", Nested("w00", [("00_relax", child("relax")), ("01_umb", child("umb"))], chain=True)),
            ("w01", AddFiles("w01", {"x.itp": "; x"})),
        ], broadcast=True, parallel=2, sequential_first=1, carry=("*.top", "*.itp", "*.ndx"))
        text = to_json([tree])
        self.assertEqual(json.loads(text)["steps"][0]["params"]["children"][0][0], "smd")
        with self.assertRaises(UnsafeOperationError):  # the RawShellStep children need the opt-in again
            from_json(text)
        back = from_json(text, allow_unsafe=True)[0]
        self.assertIsInstance(back, Nested)
        self.assertEqual(back.generate(), tree.generate())
        self.assertEqual(to_json([back]), text)


class TestNestedRuns(ScriptCase):
    def plan(self, root: Calculation, *more: Calculation) -> Path:
        (self.d / "start.gro").write_text(GRO)
        (self.d / "topo.top").write_text("; top")
        # these tests are about the generated scripts; the design checks are covered in test_checks
        build_plan([root, *more], self.d / "start.gro", self.d / "w", extra_inputs=[self.d / "topo.top"],
                   checks=False).write()
        return self.d / "w"

    def run_all(self, wd: Path) -> tuple[int, str]:
        r = self.bash((wd / "run.sh").read_text(), wd)
        return r.returncode, r.stdout + r.stderr

    def test_chain_hands_over(self) -> None:
        wd = self.plan(Nested("job", [child("a", "echo from-a > handed.itp"), child("b")], chain=True))
        code, out = self.run_all(wd)
        self.assertEqual(code, 0, out)
        job = wd / "0_job"
        self.assertEqual((job / "a/topo.top").read_text(), "; top")  # input_to="first"
        self.assertEqual((job / "b/handed.itp").read_text().strip(), "from-a")  # copy.sh of a
        self.assertTrue((job / "b/input.gro").exists())
        self.assertTrue((job / "output.gro").exists())
        self.assertEqual([l.split()[0] for l in (job / "timing.log").read_text().splitlines()], ["a", "b", "TOTAL"])

    def test_broadcast_and_input_to(self) -> None:
        wd = self.plan(Nested("job", [child("a"), RawShellStep("b", "cp ../a/output.gro output.gro",
                                                                allow_unsafe=True)], broadcast=True))
        code, out = self.run_all(wd)
        self.assertEqual(code, 0, out)
        job = wd / "0_job"
        self.assertTrue((job / "b/topo.top").exists())  # broadcast
        self.assertFalse((job / "b/input.gro").exists())  # input.gro only to the first child

    def test_parallel(self) -> None:
        self.env["OMP_NUM_THREADS"] = "8"
        kids = [child("pre")] + [child(f"w{i}", "sleep 0.2") for i in range(4)]
        wd = self.plan(Nested("job", kids, input_to="all", parallel=2, sequential_first=1))
        code, out = self.run_all(wd)
        self.assertEqual(code, 0, out)
        job = wd / "0_job"
        ran = dict(l.split() for l in (job / "ran.txt").read_text().splitlines())
        self.assertEqual(ran, {"pre": "8", "w0": "4", "w1": "4", "w2": "4", "w3": "4"})
        timing = (job / "timing.log").read_text()
        self.assertIn("parallel block: P=2, 4 threads each", timing)
        self.assertFalse((job / "failed.txt").exists() and (job / "failed.txt").read_text())

    def test_parallel_failure_is_reported(self) -> None:
        kids = [child("w0"), child("w1", "exit 3"), child("w2")]
        wd = self.plan(Nested("job", kids, input_to="all", parallel=2))
        code, out = self.run_all(wd)
        self.assertNotEqual(code, 0)
        job = wd / "0_job"
        self.assertEqual((job / "failed.txt").read_text().split(), ["w1"])
        self.assertTrue((job / "w0/output.gro").exists() and (job / "w2/output.gro").exists())
        self.assertFalse((job / "output.gro").exists())
        self.assertIn("failed children: w1", out)

    def test_rerun_skips_finished_children(self) -> None:
        wd = self.plan(Nested("job", [child("a"), child("b", "exit 1")], input_to="all"))
        code, _ = self.run_all(wd)
        self.assertNotEqual(code, 0)
        job = wd / "0_job"
        (job / "b/mdrun.sh").write_text("#!/bin/bash\ncp input.gro output.gro\n")  # "fix" b
        code, out = self.run_all(wd)
        self.assertEqual(code, 0, out)
        ran = [line.split()[0] for line in (job / "ran.txt").read_text().splitlines()]
        self.assertEqual(ran, ["a", "b"])  # a ran once; the fixed b does not log

    def test_nested_in_nested_as_middle_step(self) -> None:
        windows = Nested("umbrella", [
            child("smd", "cp input.gro ../w00/input.gro"),  # as a frame extraction would
            Nested("w00", [("00_relax", child("relax")), ("01_umb", child("umb"))], chain=True),
        ], broadcast=True)
        wd = self.plan(child("em"), windows, child("after"))
        code, out = self.run_all(wd)
        self.assertEqual(code, 0, out)
        self.assertTrue((wd / "1_umbrella/w00/01_umb/output.gro").exists())
        self.assertTrue((wd / "2_after/input.gro").exists())

    def test_replace_generated_keeps_partly_run_nested(self) -> None:
        tree = Nested("job", [child("a"), child("b")], input_to="all")
        wd = self.plan(tree)
        (wd / "0_job/a/output.gro").write_text(GRO)  # child a has run
        plan = build_plan([Nested("job", [child("a", "echo changed"), child("b")], input_to="all")],
                          self.d / "start.gro", wd, extra_inputs=[self.d / "topo.top"])
        pv = plan.preview(OverwritePolicy.REPLACE_GENERATED)
        self.assertEqual(pv.started_steps, ["0_job"])
        self.assertNotIn("0_job/a/mdrun.sh", pv.overwrite)


@unittest.skipIf(shutil.which("bash") is None, "bash not available")
class TestNestedPaths(unittest.TestCase):
    def test_plan_accepts_sub_folders_but_not_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            gro = Path(d) / "s.gro"
            gro.write_text(GRO)
            plan = build_plan([Nested("n", [child("a")])], gro, Path(d) / "w")
            self.assertIn("0_n/a/run.sh", {f.relpath for f in plan.files})
            self.assertTrue(next(f for f in plan.files if f.relpath == "0_n/a/run.sh").executable)
            self.assertEqual(os.path.basename(plan.working_dir), "w")
