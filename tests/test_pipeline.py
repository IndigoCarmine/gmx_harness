import builtins
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from gmx_harness import (
    EM,
    MD,
    MDParameters,
    MDType,
    OverwritePolicy,
    OverwriteType,
    PlanConflictError,
    RawShellStep,
    RemoveResidue,
    UnsafeNameError,
    UnsafeOperationError,
    build_plan,
    launch,
)
from gmx_harness.pipeline import MANIFEST

GRO = "t\n 1\n    1MOL      C    1   0.000   0.000   0.000\n 3 3 3\n"


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        (self.tmp / "start.gro").write_text(GRO)
        (self.tmp / "topo.top").write_text("[ system ]\nx\n[ molecules ]\nMOL 1\n")
        self.wd = self.tmp / "work"
        # input() must never be called by the library
        patcher = mock.patch.object(builtins, "input", side_effect=AssertionError("input() called"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def steps(self, nsteps: int = 100) -> list[Any]:
        return [EM(), MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=nsteps)]

    def plan(self, nsteps: int = 100) -> Any:
        return build_plan(self.steps(nsteps), self.tmp / "start.gro", self.wd, extra_inputs=[self.tmp / "topo.top"])


class TestBuildAndWrite(PipelineTestCase):
    def test_layout(self) -> None:
        pv = self.plan().write()
        self.assertTrue(pv.written)
        for rel in ["run.sh", "0_em/input.gro", "0_em/topo.top", "0_em/setting.mdp", "0_em/copy.sh",
                    "1_nvt/grommp.sh", "1_nvt/mdrun.sh", "1_nvt/run.sh", MANIFEST]:
            self.assertTrue((self.wd / rel).exists(), rel)
        self.assertFalse((self.wd / "1_nvt/copy.sh").exists())
        self.assertIn("../1_nvt", (self.wd / "0_em/copy.sh").read_text())
        manifest = json.loads((self.wd / MANIFEST).read_text())["files"]
        self.assertIn("0_em/setting.mdp", manifest)
        # LF only, even on Windows
        self.assertNotIn(b"\r\n", (self.wd / "run.sh").read_bytes())

    def test_preview_writes_nothing(self) -> None:
        pv = self.plan().preview()
        self.assertTrue(pv.ok)
        self.assertIn("0_em/setting.mdp", pv.create)
        self.assertFalse(self.wd.exists())

    def test_error_policy_is_atomic(self) -> None:
        (self.wd / "1_nvt").mkdir(parents=True)
        with self.assertRaises(PlanConflictError) as cm:
            self.plan().write()
        self.assertTrue(any("1_nvt" in c for c in cm.exception.preview.conflicts))
        self.assertFalse((self.wd / "0_em").exists())  # nothing was written

    def test_skip_existing(self) -> None:
        self.plan().write()
        (self.wd / "0_em/setting.mdp").write_text("edited by a human")
        (self.wd / "1_nvt/output.gro").write_text(GRO)  # nvt already ran
        steps = self.steps() + [MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", gen_vel="no")]
        plan = build_plan(steps, self.tmp / "start.gro", self.wd, extra_inputs=[self.tmp / "topo.top"])
        pv = plan.write(OverwritePolicy.SKIP_EXISTING)
        self.assertEqual(pv.skipped_steps, ["0_em", "1_nvt"])
        # step content is left alone ...
        self.assertEqual((self.wd / "0_em/setting.mdp").read_text(), "edited by a human")
        self.assertTrue((self.wd / "1_nvt/output.gro").exists())
        self.assertTrue((self.wd / "2_npt/mdrun.sh").exists())
        self.assertIn("run.sh", pv.overwrite)
        # ... but the old last step must now hand its output to the new step
        self.assertIn("1_nvt/copy.sh", pv.create)
        self.assertIn("1_nvt/run.sh", pv.overwrite)
        self.assertIn("../2_npt", (self.wd / "1_nvt/copy.sh").read_text())
        self.assertIn(". ./copy.sh", (self.wd / "1_nvt/run.sh").read_text())

    def test_skip_existing_does_not_touch_edited_wiring(self) -> None:
        self.plan().write()
        (self.wd / "1_nvt/run.sh").write_text("# my own run script\n")
        steps = self.steps() + [MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", gen_vel="no")]
        plan = build_plan(steps, self.tmp / "start.gro", self.wd, extra_inputs=[self.tmp / "topo.top"])
        with self.assertRaises(PlanConflictError):
            plan.write(OverwritePolicy.SKIP_EXISTING)
        self.assertEqual((self.wd / "1_nvt/run.sh").read_text(), "# my own run script\n")

    def test_replace_generated(self) -> None:
        self.plan(100).write()
        (self.wd / "1_nvt/run.out").write_text("")  # a foreign file, but the step has not run yet
        pv = self.plan(200).write(OverwritePolicy.REPLACE_GENERATED)
        self.assertIn("1_nvt/setting.mdp", pv.overwrite)
        self.assertIn("1_nvt/run.out", pv.stale_outputs)
        mdp = MDParameters.from_file(str(self.wd / "1_nvt/setting.mdp"))
        self.assertEqual(mdp.get("nsteps"), "200")  # the new value really was written

    def test_replace_generated_keeps_steps_that_already_ran(self) -> None:
        self.plan(100).write()
        (self.wd / "1_nvt/output.gro").write_text(GRO)  # nvt has run
        (self.wd / "1_nvt/grommp.sh").write_text("# rewritten by the run environment\n")
        pv = self.plan(200).write(OverwritePolicy.REPLACE_GENERATED)
        self.assertEqual(pv.started_steps, ["1_nvt"])
        self.assertEqual(pv.conflicts, [])
        # its inputs keep matching its results
        self.assertEqual(MDParameters.from_file(str(self.wd / "1_nvt/setting.mdp")).get("nsteps"), "100")
        self.assertTrue((self.wd / "1_nvt/output.gro").exists())

    def test_solvation_rerun_after_gmx_rewrote_dummy_top(self) -> None:
        from gmx_harness import SolvationMCH

        steps = [EM(), SolvationMCH(calculation_name="solv")]
        build_plan(steps, self.tmp / "start.gro", self.wd).write()
        self.assertFalse((self.wd / "1_solv/dummy.top").exists())  # made by mdrun.sh, not tracked
        self.assertIn(": > dummy.top", (self.wd / "1_solv/mdrun.sh").read_text())
        (self.wd / "1_solv/dummy.top").write_text("MCH              8041\n")  # what gmx solvate writes
        pv = build_plan(steps, self.tmp / "start.gro", self.wd).write(OverwritePolicy.REPLACE_GENERATED)
        self.assertEqual(pv.conflicts, [])

    def test_replace_generated_refuses_human_edits(self) -> None:
        self.plan(100).write()
        (self.wd / "1_nvt/setting.mdp").write_text("edited")
        with self.assertRaises(PlanConflictError):
            self.plan(200).write(OverwritePolicy.REPLACE_GENERATED)
        self.assertEqual((self.wd / "1_nvt/setting.mdp").read_text(), "edited")
        self.plan(200).write(OverwritePolicy.REPLACE_GENERATED, force_modified=True)
        self.assertNotEqual((self.wd / "1_nvt/setting.mdp").read_text(), "edited")

    def test_replace_generated_refuses_foreign_files(self) -> None:
        (self.wd / "0_em").mkdir(parents=True)
        (self.wd / "0_em/setting.mdp").write_text("not ours")
        with self.assertRaises(PlanConflictError):
            self.plan().write(OverwritePolicy.REPLACE_GENERATED)

    def test_removed_step_files_are_cleaned_only_if_generated(self) -> None:
        self.plan().write()
        plan = build_plan([EM()], self.tmp / "start.gro", self.wd, extra_inputs=[self.tmp / "topo.top"])
        pv = plan.write(OverwritePolicy.REPLACE_GENERATED)
        self.assertIn("1_nvt/setting.mdp", pv.remove)
        self.assertFalse((self.wd / "1_nvt/setting.mdp").exists())

    def test_named_and_per_step_inputs(self) -> None:
        (self.tmp / "MOL_fixed.top").write_bytes(b"[ system ]\n")
        (self.tmp / "MOL.ndx").write_bytes(b"[ A ]\n1\n")
        plan = build_plan(
            self.steps(), self.tmp / "start.gro", self.wd,
            extra_inputs={"topo.top": self.tmp / "MOL_fixed.top"},
            step_inputs={"nvt": {"index.ndx": self.tmp / "MOL.ndx"}},
        )
        self.assertEqual(plan.file("0_em/topo.top"), "[ system ]\n")
        self.assertEqual(plan.file("1_nvt/index.ndx"), "[ A ]\n1\n")
        with self.assertRaises(ValueError):
            build_plan(self.steps(), self.tmp / "start.gro", self.wd, step_inputs={"nope": [self.tmp / "MOL.ndx"]})
        with self.assertRaises(ValueError):  # would clobber a generated file
            build_plan(self.steps(), self.tmp / "start.gro", self.wd,
                       step_inputs={"nvt": {"setting.mdp": self.tmp / "MOL.ndx"}})
        with self.assertRaises(UnsafeNameError):
            build_plan(self.steps(), self.tmp / "start.gro", self.wd,
                       extra_inputs={"../x.top": self.tmp / "MOL_fixed.top"})

    def test_mdrun_args_and_extra_files(self) -> None:
        md = MD(type=MDType.v_rescale_c_rescale, calculation_name="metad", gen_vel="no",
                mdrun_args=["-plumed", "plumed.dat"], extra_files={"plumed.dat": "d: DISTANCE ATOMS=1,2\n"})
        files = md.generate()
        self.assertIn('"$GMX" mdrun -plumed plumed.dat -deffnm output -v $MDRUN_ARGS', files["mdrun.sh"])
        self.assertEqual(files["plumed.dat"], "d: DISTANCE ATOMS=1,2\n")
        with self.assertRaises(ValueError):
            MD(type=MDType.v_rescale_c_rescale, calculation_name="x", extra_files={"setting.mdp": ""}).generate()

    def test_scripts_are_machine_independent(self) -> None:
        plan = self.plan()
        for f in plan.files:
            if f.relpath.endswith(".sh"):
                text = f.content.decode()
                # nothing about the generating machine ends up in a script
                self.assertNotIn(str(self.tmp), text, f.relpath)
                self.assertNotIn(self.tmp.as_posix(), text, f.relpath)
        mdrun = plan.file("1_nvt/mdrun.sh")
        self.assertIn('GMX="${GMX:-$(command -v gmx_d || command -v gmx_mpi || command -v gmx || true)}"', mdrun)
        self.assertIn('"$GMX" mdrun -deffnm output -v $MDRUN_ARGS', mdrun)
        self.assertIn('"$GMX" grompp -f setting.mdp', plan.file("1_nvt/grommp.sh"))


class TestSafety(PipelineTestCase):
    def test_bad_names(self) -> None:
        for bad in ["../x", "a;rm -rf ~", "a b", "", ".hidden", "-x", "a/b", "$(id)"]:
            # pydantic-dataclass steps wrap UnsafeNameError in a ValidationError; both are ValueErrors
            with self.assertRaises(ValueError, msg=bad):
                EM(calculation_name=bad)
            with self.assertRaises(UnsafeNameError, msg=bad):
                RemoveResidue(bad)
        with self.assertRaises(ValueError):
            EM(defines=["X; rm"])

    def test_duplicate_names(self) -> None:
        with self.assertRaises(ValueError):
            build_plan([EM(), EM()], self.tmp / "start.gro", self.wd)

    def test_raw_shell_step_needs_opt_in(self) -> None:
        with self.assertRaises(UnsafeOperationError):
            RawShellStep("raw", "rm -rf /")
        step = RawShellStep("raw", "cp input.gro output.gro", allow_unsafe=True)
        self.assertIn("WARNING", step.generate()["mdrun.sh"])

    def test_custom_step_cannot_escape(self) -> None:
        from gmx_harness import Calculation

        class Evil(Calculation):
            calculation_name = "evil"

            def generate(self) -> dict[str, str]:
                return {"mdrun.sh": "", "../../escape.txt": "x"}

        with self.assertRaises(UnsafeNameError):
            build_plan([Evil()], self.tmp / "start.gro", self.wd)


class TestLegacyLaunch(PipelineTestCase):
    def test_no_and_add(self) -> None:
        launch(self.steps(), str(self.tmp / "start.gro"), str(self.wd))
        with self.assertRaises(PlanConflictError):
            launch(self.steps(), str(self.tmp / "start.gro"), str(self.wd))
        pv = launch(self.steps(), str(self.tmp / "start.gro"), str(self.wd), OverwriteType.add_calculation)
        self.assertEqual(pv.skipped_steps, ["0_em", "1_nvt"])

    def test_full_overwrite_needs_confirm_and_manifest(self) -> None:
        launch(self.steps(), str(self.tmp / "start.gro"), str(self.wd))
        with self.assertRaises(UnsafeOperationError):
            launch(self.steps(), str(self.tmp / "start.gro"), str(self.wd), OverwriteType.full_overwrite)
        (self.wd / "1_nvt/output.gro").write_text(GRO)
        launch(self.steps(), str(self.tmp / "start.gro"), str(self.wd), OverwriteType.full_overwrite, confirm=True)
        self.assertFalse((self.wd / "1_nvt/output.gro").exists())

        foreign = self.tmp / "foreign"
        foreign.mkdir()
        (foreign / "important.txt").write_text("keep")
        with self.assertRaises(UnsafeOperationError):
            launch(self.steps(), str(self.tmp / "start.gro"), str(foreign), OverwriteType.full_overwrite, confirm=True)
        self.assertTrue((foreign / "important.txt").exists())


if __name__ == "__main__":
    unittest.main()
