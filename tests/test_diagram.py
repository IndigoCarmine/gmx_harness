import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from gmx_harness import EM, MD, MDType, Nested, RawShellStep, SolvationMCH, save_drawio, to_drawio
from gmx_harness.steps import AWH


def _steps() -> list:
    return [
        EM(nsteps=30000, emtol=100, defines=["INTER"]),
        SolvationMCH(),
        MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=10_000_000, defines=["INTER"]),
        MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", nsteps=5_000_000, gen_vel="no",
           temperature=310),
    ]


def _labels(text: str) -> dict[str, str]:
    root = ET.fromstring(text)
    return {c.get("id", ""): c.get("value", "") for c in root.iter("mxCell") if c.get("vertex") == "1"}


class TestDrawio(unittest.TestCase):
    def test_cards(self) -> None:
        text = to_drawio(_steps(), define_labels={"INTER": "Intermolecular Interaction"}, none_label="なし")
        labels = _labels(text)
        self.assertEqual(labels["title"], "MD Calculation Workflow")
        self.assertEqual(
            labels["step0"],
            "Energy Minimization<hr>nsteps: 30,000<br>emtol: 100.0 kJ/mol/nm<br>defines: Intermolecular Interaction",
        )
        self.assertTrue(labels["step1"].startswith("Solvation (MCH) gmx solvate"))
        self.assertIn("nsteps: 10,000,000 (20.0 ns)", labels["step2"])
        self.assertIn("temperature: 310.0 K", labels["step3"])
        self.assertIn("defines: なし", labels["step3"])

    def test_edges_and_layout(self) -> None:
        root = ET.fromstring(to_drawio(_steps(), per_column=3, title=None))
        edges = [(c.get("source"), c.get("target")) for c in root.iter("mxCell") if c.get("edge") == "1"]
        self.assertEqual(edges, [("step0", "step1"), ("step1", "step2"), ("step2", "step3")])
        geo = {c.get("id"): c.find("mxGeometry") for c in root.iter("mxCell") if c.get("vertex") == "1"}
        self.assertNotIn("title", geo)
        g0, g3 = geo["step0"], geo["step3"]
        assert g0 is not None and g3 is not None
        self.assertEqual(g0.get("y"), g3.get("y"))  # 4th card starts the second column
        self.assertLess(float(g0.get("x", 0)), float(g3.get("x", 0)))

    def test_escaping_and_other_steps(self) -> None:
        raw = RawShellStep("raw", "cat a.gro > output.gro\necho done", allow_unsafe=True)
        labels = _labels(to_drawio([raw, AWH(nsteps=1000)], show_names=True))
        self.assertEqual(labels["step0"], "Shell Step [raw]<hr>cat a.gro &gt; output.gro")
        self.assertTrue(labels["step1"].startswith("AWH Simulation [awh]"))

    def test_additional_mdp_parameters_are_shown(self) -> None:
        md = MD(type=MDType.v_rescale_c_rescale, calculation_name="md", nsteps=1000,
                additional_mdp_parameters={"dt": 0.001, "define": "-DA -DB"})
        self.assertIn("nsteps: 1,000 (0.0 ns)", _labels(to_drawio([md]))["step0"])

    def test_save_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.drawio")
            save_drawio(_steps(), path)
            with self.assertRaises(FileExistsError):
                save_drawio(_steps(), path)
            save_drawio(_steps()[:1], path, overwrite=True)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(len(_labels(f.read())), 2)

    def test_bad_per_column(self) -> None:
        with self.assertRaises(ValueError):
            to_drawio(_steps(), per_column=0)


def _md(name: str, nsteps: int) -> MD:
    return MD(type=MDType.v_rescale_c_rescale, calculation_name=name, nsteps=nsteps, gen_vel="no")


def _umbrella(n_windows: int = 4, parallel: int = 1, vary: bool = False) -> Nested:
    """smd -> extract -> windows (each relax -> umb), like an umbrella-sampling tree."""
    windows = [Nested(f"w{i:02d}", [("relax", _md("relax", 1000)), ("umb", _md("umb", 2000 + (i if vary else 0)))],
                      chain=True) for i in range(n_windows)]
    extract = RawShellStep("extract", "cp ../smd/output.gro output.gro", allow_unsafe=True)
    return Nested("umbrella", [("smd", _md("smd", 500)), ("extract", extract), *windows], broadcast=True,
                  parallel=parallel, sequential_first=2 if parallel > 1 else 0)


def _cells(text: str) -> dict[str, ET.Element]:
    return {c.get("id", ""): c for c in ET.fromstring(text).iter("mxCell")}


def _xy(c: ET.Element) -> tuple[float, float]:
    g = c.find("mxGeometry")
    assert g is not None
    return float(g.get("x", 0)), float(g.get("y", 0))


class TestDrawioNested(unittest.TestCase):
    def test_frame_holds_children_and_repeats_collapse(self) -> None:
        text = to_drawio([EM(nsteps=10), _umbrella()], show_names=True, title=None)
        cells, labels = _cells(text), _labels(text)
        frame = cells["step1"]
        self.assertIn("swimlane", frame.get("style", ""))
        self.assertEqual(labels["step1"], "Nested (in order) [umbrella]")
        self.assertTrue(labels["step1_0"].startswith("MD Simulation [smd]<hr>"))
        self.assertTrue(labels["step1_1"].startswith("Shell Step [extract]"))
        self.assertEqual(labels["step1_2"], "Nested (chain) [w00 … w03] ×4")
        self.assertNotIn("step1_3", cells)  # the four identical windows are one box
        self.assertTrue(labels["step1_2_0"].startswith("MD Simulation [relax]"))
        self.assertIn("nsteps: 2,000", labels["step1_2_1"])
        for cid, parent in [("step1_0", "step1"), ("step1_2", "step1"), ("step1_2_1", "step1_2")]:
            self.assertEqual(cells[cid].get("parent"), parent)
        edges = {c.get("id"): c for c in cells.values() if c.get("edge") == "1"}
        self.assertEqual((edges["edge1"].get("source"), edges["edge1"].get("target")), ("step0", "step1"))
        self.assertIn("dashed=1", edges["step1_edge1"].get("style", ""))      # in order, no hand-over
        self.assertNotIn("dashed=1", edges["step1_2_edge1"].get("style", ""))  # chain: relax -> umb
        # children sit inside the frame, below its header
        g = frame.find("mxGeometry")
        assert g is not None
        x, y = _xy(cells["step1_2"])
        inner = cells["step1_2"].find("mxGeometry")
        assert inner is not None
        self.assertLessEqual(x + float(inner.get("width", 0)), float(g.get("width", 0)))
        self.assertLessEqual(y + float(inner.get("height", 0)), float(g.get("height", 0)))

    def test_no_collapse_and_different_windows(self) -> None:
        labels = _labels(to_drawio([_umbrella()], collapse_repeats=False, show_names=True))
        self.assertEqual([labels[f"step0_{j}"] for j in range(2, 6)],
                         [f"Nested (chain) [w{i:02d}]" for i in range(4)])
        labels = _labels(to_drawio([_umbrella(vary=True)]))  # windows differ in nsteps: nothing merged
        self.assertIn("step0_5", labels)
        self.assertNotIn("×", labels["step0_2"])

    def test_parallel_block_side_by_side(self) -> None:
        text = to_drawio([_umbrella(n_windows=3, parallel=3, vary=True)], show_names=True)
        cells, labels = _cells(text), _labels(text)
        self.assertEqual(labels["step0"], "Nested (parallel ×3) [umbrella]")
        (x0, y0), (x1, y1) = _xy(cells["step0_2"]), _xy(cells["step0_3"])
        self.assertEqual(y0, y1)
        self.assertLess(x0, x1)
        self.assertGreater(y0, _xy(cells["step0_1"])[1])
        sources = {c.get("source") for c in cells.values() if c.get("edge") == "1" and c.get("target") in
                   {"step0_2", "step0_3", "step0_4"}}
        self.assertEqual(sources, {"step0_1"})  # every parallel child starts after the sequential ones


if __name__ == "__main__":
    unittest.main()
