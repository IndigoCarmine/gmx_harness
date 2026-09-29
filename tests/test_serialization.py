import json
import os
import tempfile
import unittest
import warnings

from gmx_harness import (
    EM,
    MD,
    AddFiles,
    Calculation,
    MDType,
    RawShellStep,
    RemoveResidue,
    ResizeBox,
    Solvation,
    SolvationMCH,
    SolvationSCP216,
    UnsafeOperationError,
    from_json,
    load_json,
    save_json,
    to_json,
)
from gmx_harness.steps import RuntimeSolvation


class TestCalculationJson(unittest.TestCase):
    def test_round_trip(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            calculations: list[Calculation] = [
                EM(calculation_name="em_1"),
                MD(calculation_name="md_1", type=MDType.v_rescale_c_rescale, nsteps=50000, temperature=298.15),
                RuntimeSolvation(calculation_name="rs_1"),
                Solvation(calculation_name="sol_1"),
                SolvationSCP216(calculation_name="scp_1"),
                SolvationMCH(calculation_name="mch_1", scale=0.5),
                RemoveResidue("rm_1", "MCH"),
                ResizeBox("box_1", 3, 4, 5, remove_resname=None),
                AddFiles("add_1", {"x.itp": "; itp"}),
            ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            save_json(calculations, path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                loaded = load_json(path)
        self.assertEqual(len(calculations), len(loaded))
        for original, back in zip(calculations, loaded):
            self.assertIs(type(original), type(back))
            self.assertEqual(original.params(), back.params())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(original.generate(), back.generate(), type(original).__name__)

    def test_defaults_are_not_stored(self) -> None:
        # EM.emtol defaults to the int 300 ("emtol = 300" in the mdp); storing it would come
        # back as 300.0 and change the generated file, so defaults are left out.
        doc = json.loads(to_json([EM(), EM(calculation_name="em2", emtol=300.0)]))
        self.assertNotIn("emtol", doc["steps"][0]["params"])
        self.assertEqual(doc["steps"][1]["params"]["emtol"], 300.0)  # explicitly given -> kept
        reloaded = from_json(json.dumps(doc))
        self.assertEqual(reloaded[0].generate(), EM().generate())
        self.assertEqual(reloaded[1].generate(), EM(calculation_name="em2", emtol=300.0).generate())

    def test_raw_shell_needs_opt_in_again(self) -> None:
        text = to_json([RawShellStep("raw", "echo hi", allow_unsafe=True)])
        self.assertNotIn("allow_unsafe", text)
        with self.assertRaises(UnsafeOperationError):
            from_json(text)
        self.assertEqual(from_json(text, allow_unsafe=True)[0].command, "echo hi")  # type: ignore[attr-defined]

    def test_format(self) -> None:
        doc = json.loads(to_json([MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt")]))
        self.assertEqual(doc["format"], "gmx_harness.pipeline")
        self.assertEqual(doc["version"], 1)
        self.assertEqual(doc["steps"][0]["type"], "MD")
        self.assertEqual(doc["steps"][0]["params"]["type"], "v_rescale_only_nvt")  # enum by name

    def test_rejects_bad_documents(self) -> None:
        good = json.loads(to_json([EM()]))
        for mutate in (
            lambda d: d.update(format="other"),
            lambda d: d.update(version=99),
            lambda d: d["steps"][0].update(type="Nope"),
            lambda d: d["steps"][0]["params"].update(unknown_param=1),
        ):
            bad = json.loads(json.dumps(good))
            mutate(bad)
            with self.assertRaises(ValueError):
                from_json(json.dumps(bad))
        bad = json.loads(to_json([MD(type=MDType.v_rescale_only_nvt, calculation_name="x")]))
        bad["steps"][0]["params"]["type"] = "no_such_type"
        with self.assertRaises(ValueError):
            from_json(json.dumps(bad))
        with self.assertRaises(ValueError):
            from_json("[]")  # the old mylibs list format is not accepted

if __name__ == "__main__":
    unittest.main()
