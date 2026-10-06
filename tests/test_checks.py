"""Design checks: known mistakes are stopped, correct designs pass, waivers need a code and a reason."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from gmx_harness import EM, MD, GroAtom, GroFile, Layout, MDType, MoleculeLabels, build_plan, set_molecule_count
from gmx_harness.pipeline import Plan
from gmx_harness.plumed import PreprocessError, preprocess
from gmx_harness.checks import (
    HarnessCheckError,
    check_tree,
    Report,
    check_bond_lengths,
    check_bond_pairs,
    box_heights,
    check_box,
    check_fresh,
    check_labels,
    check_ndx,
    check_periodic_twist,
    check_plumed,
    check_step,
    check_topology,
    check_whole_molecules,
    expand_template,
    expect,
    record_facts,
    step_state,
)

NAMES = ("C1", "H1", "N1", "H2")
LABELS = ("A", "A", "B", "B")
TOP = """[ moleculetype ]
MOL 3
[ atoms ]
1 c 1 MOL C1 1 0 12.0
2 h 1 MOL H1 2 0 1.0
3 n 1 MOL N1 3 0 14.0
4 h 1 MOL H2 4 0 1.0
[ system ]
t
[ molecules ]
MOL {n}
"""

TEMPLATE = """#define NDISK 2
#define NROS 2
#define BIAS_IFACE 0
UNITS LENGTH=nm ENERGY=kj/mol
#for k in 0..NDISK-1
#for i in 0..NROS-1
m{k}_{i}: COM ATOMS=@sel(disk=k, mol=i, res="A,B", heavy=True)
#endfor
cen{k}: COM ATOMS=@sel(disk=k, res="A,B")
#endfor
#for k in 0..NDISK-2
#for i in 0..NROS-1
t{k}_{i}: TORSION ATOMS=m{k}_{i},cen{k},cen{k+1},m{k+1}_{i}
#endfor
s{k}: CUSTOM ARG={join("t%d_{i}" % k, NROS)} VAR={join("a{i}", NROS)} FUNC={join("sin(2*a{i})", NROS, "+")} PERIODIC=NO
c{k}: CUSTOM ARG={join("t%d_{i}" % k, NROS)} VAR={join("a{i}", NROS)} FUNC={join("cos(2*a{i})", NROS, "+")} PERIODIC=NO
phi{k}: CUSTOM ARG=s{k},c{k} VAR=y,x FUNC=atan2(y,x)/2 PERIODIC=-pi/2,pi/2
#endfor
theta: COMBINE ARG=phi{BIAS_IFACE} PERIODIC=-pi/2,pi/2
metad: METAD ARG=theta SIGMA=0.05 HEIGHT=1 PACE=500 BIASFACTOR=10 TEMP=300 GRID_BIN=200 GRID_MIN=-pi/2 GRID_MAX=pi/2
PRINT STRIDE=500 ARG=phi0,theta,metad.bias FILE=COLVAR
"""


# fragments of a real GROMACS output.log: the parameter dump, an energy table, the end
LOG_HEADER = """Input Parameters:
   coulombtype                    = PME
   epsilon-r                      = 1
   epsilon-rf                     = inf
   ref-t:         300
Started mdrun on rank 0 Wed Sep 16 22:36:33 2026

"""
LOG_ENERGIES = """           Step           Time
              0        0.00000

   Energies (kJ/mol)
          LJ-14     Coulomb-14        LJ (SR)   Coulomb (SR)      Potential
    2.60669e+04   -3.55917e+05   -5.97472e+04    7.79982e+04{pot}
"""
LOG_END = "Finished mdrun on rank 0 Wed Sep 16 22:36:37 2026\n"


def gro(nmol: int, box: float = 12.0) -> GroFile:
    atoms = [GroAtom(m * 4 + i + 1, NAMES[i], LABELS[i], m + 1, [0.5 * m, 0.1 * i, 0.0])
             for m in range(nmol) for i in range(4)]
    return GroFile("t", atoms, box, box, box)


def codes(rep: Report) -> set[str]:
    return {i.code for i in rep.issues if i.level == "error"}


class TestWaivers(unittest.TestCase):
    def test_enforce(self) -> None:
        rep = Report()
        rep.error("S004", "small box")
        with self.assertRaises(HarnessCheckError):
            rep.enforce(quiet=True)
        rep.enforce({"S004": "tested before"}, quiet=True)
        for bad in ({"S004": " "}, {"X123": "why"}, {"S04": "why"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                rep.enforce(bad, quiet=True)
        self.assertIn("WARN  W001", rep.format({"S004": "x", "S001": "unneeded"}))
        for bad in ({"S004": ""}, {"X123": "why"}):          # errors()/ok() check waivers too
            with self.assertRaises(ValueError, msg=str(bad)):
                rep.errors(bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                rep.ok(bad)

    def test_report_written(self) -> None:
        rep = Report()
        rep.error("S004", "small box")
        with tempfile.TemporaryDirectory() as d:
            rep.enforce({"S004": "tested"}, out_dir=d, quiet=True)
            data = json.loads(Path(d, "checks.json").read_text(encoding="utf-8"))
            self.assertTrue(data["ok"])
            self.assertEqual(data["waived"], {"S004": "tested"})
            self.assertEqual([(w["waiver"], w["scope"]) for w in data["waived_issues"]], [("S004", "code")])
            self.assertEqual(data["unused_waivers"], [])

    @staticmethod
    def two_boxes() -> Report:
        rep = Report()
        rep.error("S004", "small box", "fiber_rot_+10")
        rep.error("S004", "small box", "finite_rot_+10")
        rep.error("S005", "no closure", "fiber_rot_+10")
        return rep

    def test_waiver_at_a_place(self) -> None:
        rep = self.two_boxes()
        w = {"S004@fiber_*": "periodic along z by design"}
        self.assertEqual([(i.code, i.where) for i in rep.errors(w)],     # the same code elsewhere still stops
                         [("S004", "finite_rot_+10"), ("S005", "fiber_rot_+10")])
        with self.assertRaises(HarnessCheckError):
            rep.enforce(w, quiet=True)
        rep.enforce({**w, "S004@finite_rot_+10": "tested", "S005@*fiber_rot_+10*": "ok"}, quiet=True)
        self.assertEqual(len(rep.errors({"S004@FIBER_*": "case-sensitive"})), 3)
        text = rep.format(w)
        self.assertIn('WAIVED S004 fiber_rot_+10: small box  -- waived by "S004@fiber_*", reason: periodic', text)
        self.assertIn("ERROR S004 finite_rot_+10: small box", text)

    def test_code_wide_waiver_is_marked(self) -> None:
        rep = self.two_boxes()
        self.assertEqual([i.code for i in rep.errors({"S004": "tested"})], ["S005"])     # backward compatible
        self.assertEqual(rep.format({"S004": "tested"}).count('waived by "S004" (for every S004)'), 2)
        # a place waiver is credited before a code-wide one
        d = rep.to_dict({"S004": "tested", "S004@finite*": "finite: tested"})
        self.assertEqual([(w["waiver"], w["scope"], w["where"]) for w in d["waived_issues"]],
                         [("S004", "code", "fiber_rot_+10"), ("S004@finite*", "where", "finite_rot_+10")])

    def test_place_waiver_needs_a_place(self) -> None:
        rep = Report()
        rep.error("S004", "small box")                                    # no where
        self.assertEqual(len(rep.errors({"S004@*": "anything"})), 1)
        self.assertEqual(rep.unused_waivers({"S004@*": "anything"}), ["S004@*"])
        rep = Report()
        rep.error("P007", "single atom COM", "6_md/plumed.dat:12")
        rep.error("P007", "single atom COM", "7_md/plumed.dat:3")
        self.assertEqual([i.where for i in rep.errors({"P007@6_md/plumed.dat": "x"})], ["7_md/plumed.dat:3"])
        self.assertEqual(rep.errors({"P007@6_md/plumed.dat:12": "x", "P007@7_md\\plumed.dat": "y"}), [])

    def test_w001_per_key(self) -> None:
        rep = self.two_boxes()
        w = {"S004@fiber_*": "a", "S004@bundle_*": "b", "S001": "c", "S011@x": "d"}
        self.assertEqual(rep.unused_waivers(w), ["S004@bundle_*", "S001", "S011@x"])
        text = rep.format(w)
        self.assertIn("WARN  W001 waiver S004@bundle_* matches no issue (S004 occurs at: fiber_rot_+10, "
                      "finite_rot_+10)", text)
        self.assertIn("WARN  W001 waiver S001 is not needed (no S001 issue)", text)
        self.assertIn("WARN  W001 waiver S011@x is not needed (no S011 issue)", text)
        self.assertEqual(rep.to_dict(w)["unused_waivers"], ["S004@bundle_*", "S001", "S011@x"])

    def test_invalid_place_waivers(self) -> None:
        rep = self.two_boxes()
        for bad in ({"S004@": "why"}, {"S004@  ": "why"}, {"X123@fiber": "why"}, {"S04@fiber": "why"},
                    {"@fiber": "why"}, {"S004@fiber": " "}, {"S004 @fiber": "why"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                rep.errors(bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                Plan(Path("w"), [], [], waive=bad)                    # build_plan(waive=...) too

    def test_place_waiver_written(self) -> None:
        rep = self.two_boxes()
        with tempfile.TemporaryDirectory() as d:
            w = {"S004@*rot_+10": "both tested", "S005@fiber_rot_+10": "closure checked by hand", "S001": "x"}
            rep.enforce(w, out_dir=d, quiet=True)
            data = json.loads(Path(d, "checks.json").read_text(encoding="utf-8"))
        self.assertTrue(data["ok"])
        self.assertEqual(data["waived"], w)
        self.assertEqual(data["waived_issues"][0], {"waiver": "S004@*rot_+10", "scope": "where", "code": "S004",
                                                    "where": "fiber_rot_+10", "level": "error",
                                                    "message": "small box", "reason": "both tested"})
        self.assertEqual([(x["waiver"], x["where"]) for x in data["waived_issues"]],
                         [("S004@*rot_+10", "fiber_rot_+10"), ("S004@*rot_+10", "finite_rot_+10"),
                          ("S005@fiber_rot_+10", "fiber_rot_+10")])
        self.assertEqual(data["unused_waivers"], ["S001"])


class TestStructure(unittest.TestCase):
    def test_topology(self) -> None:
        self.assertTrue(check_topology(gro(3), TOP.format(n=3)).ok())
        self.assertEqual(codes(check_topology(gro(4), TOP.format(n=3))), {"S001"})    # nmol + 1
        two = TOP.format(n=3) + "SOL 5\n"
        self.assertIn("S003", codes(check_topology(gro(3), two, single_type=True)))
        rep = check_topology(gro(3), "[ molecules ]\nXYZ 3\n")
        self.assertEqual([(i.code, i.level) for i in rep.issues], [("S011", "error")])
        rep.enforce({"S011": "XYZ comes from the force field directory"}, quiet=True)   # waivable

    def test_set_molecule_count_needs_names_for_several_types(self) -> None:
        with self.assertRaises(ValueError):
            set_molecule_count(TOP.format(n=3) + "SOL 5\n", 10)
        self.assertIn("SOL              10", set_molecule_count(TOP.format(n=3) + "SOL 5\n", 10, names=["SOL"]))

    def test_simple(self) -> None:
        self.assertEqual(codes(check_whole_molecules(10, 4)), {"S002"})
        self.assertTrue(check_whole_molecules(12, 4).ok())
        self.assertEqual(codes(check_box(gro(1, box=5.0), 12.0)), {"S004"})
        self.assertEqual(check_box(gro(1), 12.0, where="v").issues, [])
        self.assertTrue(check_periodic_twist(36, 10.0, 6).ok())
        self.assertEqual(codes(check_periodic_twist(10, 10.0, 6)), {"S005"})
        self.assertEqual(codes(check_bond_pairs([(3, 9)], 4)), {"S006"})
        self.assertTrue(check_bond_pairs([(3, 6)], 4).ok())
        self.assertEqual(codes(check_bond_lengths(gro(2), [(1, 5)], 0.2, 0.4)), {"S007"})  # 0.5 nm apart
        self.assertTrue(check_bond_lengths(gro(2), [(1, 5)], 0.4, 0.6).ok())

    def test_box_periodic_axis(self) -> None:
        g = gro(1)
        g.box_z = 3.5                                      # a 10-disk fiber closed on its image along z
        rep = check_box(g, 12.0, "fiber")
        self.assertEqual(codes(rep), {"S004"})
        self.assertIn("z=3.50", rep.issues[0].message)
        self.assertTrue(check_box(g, 12.0, periodic_axes="z", min_periodic_edge=3.2).ok())
        rep = check_box(g, 12.0, periodic_axes="z", min_periodic_edge=4.0)   # too short even when periodic
        self.assertEqual(codes(rep), {"S004"})
        self.assertIn("periodic axis z=3.50 nm < 4.0 nm", rep.issues[0].message)
        g.box_x = 5.0                                      # a non-periodic axis keeps min_edge
        rep = check_box(g, 12.0, periodic_axes="z", min_periodic_edge=3.2)
        self.assertEqual([i.message.split(" nm")[0] for i in rep.issues], ["box edge(s) x=5.00"])
        self.assertTrue(check_box(g, 12.0, periodic_axes="xz", min_periodic_edge=3.2).ok())
        for bad in ("w", "zz"):
            with self.assertRaises(ValueError):
                check_box(g, 12.0, periodic_axes=bad, min_periodic_edge=3.2)
        with self.assertRaises(ValueError):
            check_box(g, 12.0, periodic_axes="z")           # no built-in cut-off: the caller gives it

    def test_box_triclinic(self) -> None:
        text = "\n".join(gro(1).generate_gro_text()[:-1]) + "\n   12.0 12.0 12.0 0 0 0 0 0 11.0\n"
        g = GroFile.from_gro_text(text.splitlines())
        self.assertEqual(g.box_triclinic, (0.0, 0.0, 0.0, 0.0, 0.0, 11.0))
        h = box_heights(g)                                 # v3 leans along y: the y width shrinks
        self.assertAlmostEqual(h[0], 12.0)
        self.assertAlmostEqual(h[1], 12.0 * 12.0 / math.hypot(12.0, 11.0))
        self.assertAlmostEqual(h[2], 12.0)
        rep = check_box(g, 12.0, "tric")
        self.assertEqual(codes(rep), {"S004"})              # the diagonal alone (12, 12, 12) would pass
        self.assertIn("y=8.85", rep.issues[0].message)
        self.assertIn("triclinic", rep.issues[0].message)
        self.assertTrue(check_box(g, 8.0).ok())
        self.assertEqual(box_heights(gro(1)), (12.0, 12.0, 12.0))

    def test_labels_and_ndx(self) -> None:
        g = gro(2)
        self.assertTrue(check_labels(g, NAMES, 2).ok())
        g.atoms[5].atom_name = "X"
        self.assertEqual(codes(check_labels(g, NAMES, 2)), {"S009"})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "a.ndx")
            p.write_text("[ Fiber1 ]\n1 2 3\n[ bad ]\n9\n", encoding="utf-8")
            self.assertEqual(codes(check_ndx(p, 8, required=["Fiber1", "fiberA"])), {"S010"})
            self.assertEqual({i.where for i in check_ndx(p, 8).issues}, {"a.ndx"})     # the file when not given


class TestFacts(unittest.TestCase):
    def test_chain(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src, sp, relaxed = Path(d, "mono.gro"), Path(d, "sp.gro"), Path(d, "relaxed.gro")
            src.write_text("mono", encoding="utf-8")
            sp.write_text("sp", encoding="utf-8")
            record_facts(sp, script="build.py", inputs=[src], ndisk=10, nros=6, rot=10.0)
            relaxed.write_text("relaxed", encoding="utf-8")
            record_facts(relaxed, script="relax.py", inputs=[sp])

            self.assertTrue(expect(relaxed, ndisk=10, nros=6, rot=10.0 + 1e-9).ok())   # inherited from sp
            rep = expect(relaxed, ndisk=36)
            self.assertEqual(codes(rep), {"F001"})
            self.assertIn("build.py", rep.issues[0].message)
            self.assertEqual(codes(expect(relaxed, box=12.0)), {"F002"})

            sp.write_text("sp rebuilt", encoding="utf-8")        # an old relaxed file was kept
            self.assertIn("F003", codes(check_fresh(relaxed)))
            relaxed.write_text("edited", encoding="utf-8")
            self.assertIn("F005", codes(check_fresh(relaxed)))
            self.assertEqual(codes(expect(Path(d, "mono.gro"))), {"F004"})


class TestPlumed(unittest.TestCase):
    layout = Layout(MoleculeLabels(LABELS, NAMES), nmol=4, nros=2)

    def expand(self, **defines: int) -> Report:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "t.plumed.in")
            p.write_text(TEMPLATE, encoding="utf-8")
            return expand_template(p, self.layout, defines, symmetry=2)[1]

    def test_correct_template_passes(self) -> None:
        rep = self.expand(NDISK=2, NROS=2)
        self.assertTrue(rep.ok(), rep.format())

    def test_define_declared_but_unused(self) -> None:
        # a #define in the template does not count as use: the body must look the name up
        text = TEMPLATE.replace("#define BIAS_IFACE 0\n", "#define BIAS_IFACE 0\n#define WALL 3\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "t.plumed.in")
            p.write_text(text, encoding="utf-8")
            rep = expand_template(p, self.layout, {"NDISK": 2, "NROS": 2, "WALL": 5}, symmetry=2)[1]
            self.assertEqual(codes(rep), {"P009"})
            self.assertIn("WALL", rep.issues[0].message)
            rep = expand_template(p, self.layout, {"NDISK": 2, "NROS": 2, "BIAS_IFACE": 0}, symmetry=2)[1]
            self.assertTrue(rep.ok(), rep.format())

    def test_empty_loop(self) -> None:
        out = preprocess("#for i in 1..0\nx{i}\n#endfor\ny\n", self.layout)
        self.assertEqual(out.splitlines()[1:], ["y"])

    def test_masses_need_mass_column(self) -> None:
        with self.assertRaises(PreprocessError) as cm:
            MoleculeLabels(LABELS, NAMES).with_masses_from_top("[ atoms ]\n1 c 1 MOL C1 1 0\n")
        self.assertIn("1 c 1 MOL C1 1 0", str(cm.exception))
        self.assertEqual(MoleculeLabels(LABELS, NAMES).with_masses_from_top(TOP.format(n=1)).masses,
                         (12.0, 1.0, 14.0, 1.0))

    def test_injected_mistakes(self) -> None:
        self.assertIn("P002", codes(self.expand(NDISK=2, NROS=2, BIAS_IFACE=9)))     # phi9 does not exist
        self.assertIn("P009", codes(self.expand(NDISK=2, NROS=2, BIASFACTR=15)))     # typo in a define
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "t.plumed.in")
            p.write_text(TEMPLATE, encoding="utf-8")
            _, rep = expand_template(p, self.layout, {}, symmetry=6)               # nros differs from the 2-fold CV
            self.assertIn("P006", codes(rep))

    def test_static(self) -> None:
        text = ("UNITS LENGTH=nm ENERGY=kj/mol\n"
                "c: COM ATOMS=5\n"
                "d: DISTANCE ATOMS=1,20\n"
                "d: DISTANCE ATOMS=1,2\n"
                "t: TORSION ATOMS=1,2,3,4\n"
                "metad: METAD ARG=t,x SIGMA=0.001 HEIGHT=1 PACE=10 GRID_MIN=-pi/6 GRID_MAX=pi/6 GRID_BIN=100\n")
        rep = check_plumed(text, natoms=10)
        self.assertEqual(codes(rep), {"P001", "P003", "P005", "P007", "P008"})
        self.assertIn("P004", {i.code for i in rep.issues})
        self.assertIn("P010", {i.code for i in check_plumed("d: DISTANCE ATOMS=1,2\n").issues})
        warn = check_plumed("UNITS LENGTH=nm ENERGY=kj/mol\nd: DISTANCE ATOMS=1,2\nr: RESTRAINT ARG=d AT=1 KAPPA=1\n")
        self.assertEqual([i.code for i in warn.issues], ["P007"])     # bias on raw single atoms


class TestPlanChecks(unittest.TestCase):
    def test_plan(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            g, top = Path(d, "in.gro"), Path(d, "topo.top")
            gro(3).save_gro(str(g))
            top.write_text(TOP.format(n=3), encoding="utf-8")
            plumed = "UNITS LENGTH=nm ENERGY=kj/mol\nPRINT STRIDE=300 ARG=x FILE=COLVAR\n"
            steps = [EM(), MD(type=MDType.v_rescale_c_rescale, calculation_name="md", nsteps=1000, nstout=500,
                              gen_vel="no", plumed=plumed)]
            plan = build_plan(steps, g, Path(d, "w"), extra_inputs=[top])
            got = {i.code for i in plan.report.issues}
            self.assertEqual(got, {"P002", "L001"})
            self.assertFalse(plan.preview().ok)
            plan = build_plan(steps, g, Path(d, "w"), extra_inputs=[top], waive={"P002": "x is defined elsewhere"})
            self.assertTrue(plan.preview().ok)
            self.assertEqual([i.where for i in plan.report.issues if i.code == "P002"], ["1_md/plumed.dat:2"])
            plan = build_plan(steps, g, Path(d, "w"), extra_inputs=[top], waive={"P002@1_md/plumed.dat": "x"})
            self.assertTrue(plan.preview().ok)
            self.assertIn('waived by "P002@1_md/plumed.dat"', str(plan.preview()))
            plan = build_plan(steps, g, Path(d, "w"), extra_inputs=[top], waive={"P002@0_em/*": "x"})
            self.assertFalse(plan.preview().ok)
            self.assertIn("W001 waiver P002@0_em/* matches no issue", str(plan.preview()))

            top.write_text(TOP.format(n=4), encoding="utf-8")
            plan = build_plan([EM(maxwarn=1)], g, Path(d, "w2"), extra_inputs=[top])
            self.assertEqual({i.code for i in plan.report.errors()}, {"L003", "L004"})
            with self.assertRaises(Exception):
                plan.write()
            self.assertFalse(Path(d, "w2").exists())

    def test_preflight_guard(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            g, top = Path(d, "in.gro"), Path(d, "topo.top")
            gro(3).save_gro(str(g))
            top.write_text(TOP.format(n=3), encoding="utf-8")
            plan = build_plan([EM(), EM(calculation_name="em2")], g, Path(d, "w"), extra_inputs=[top],
                              require_preflight=True)
            self.assertIn("sha256sum --quiet -c preflight.ok", plan.file("run.sh"))
            self.assertIn("cd .. && sha256sum", plan.file("0_em/run.sh"))
            self.assertIn("command -v sha256sum", plan.file("0_em/run.sh"))   # no sha256sum: own message
            self.assertNotIn("preflight", build_plan([EM()], g, Path(d, "x"), extra_inputs=[top]).file("run.sh"))


class TestPostrun(unittest.TestCase):
    def test_step(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            step = Path(d, "6_md")
            step.mkdir()
            Path(step, "output.log").write_text("Step 10\n   Potential   nan\nLINCS WARNING\n", encoding="utf-8")
            Path(step, "output.gro").write_text("", encoding="utf-8")
            Path(step, "slurm-1.out").write_text("NVRM: Xid 79\n", encoding="utf-8")
            rows = "\n".join(f"{t} {0.1 + 1e-4 * math.sin(t)}" for t in range(50))
            Path(step, "COLVAR").write_text("#! FIELDS time theta\n" + rows + "\n", encoding="utf-8")
            rep = check_step(step)
            self.assertEqual({i.code for i in rep.issues}, {"R001", "R002", "R003", "R004"})

    def test_nan_only_in_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            step = Path(d, "3_nvt")
            step.mkdir()
            Path(step, "output.gro").write_text("", encoding="utf-8")
            Path(step, "output.log").write_text(LOG_HEADER + LOG_ENERGIES.format(pot="   -2.51473e+05") + LOG_END,
                                                encoding="utf-8")
            self.assertEqual(check_step(step).issues, [])          # epsilon-rf = inf is a setting
            Path(step, "output.log").write_text(LOG_HEADER + LOG_ENERGIES.format(pot="             nan") + LOG_END,
                                                encoding="utf-8")
            self.assertEqual({i.code for i in check_step(step).issues}, {"R001"})

    def test_unfinished_step(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            step = Path(d, "sys", "6_md")
            step.mkdir(parents=True)
            Path(step, "output.log").write_text(LOG_HEADER + LOG_ENERGIES.format(pot="   -2.5e+05"), encoding="utf-8")
            self.assertEqual({i.code for i in check_step(step).issues}, {"R006"})     # timed out / crashed
            Path(step, "output.log").write_text(LOG_HEADER + LOG_END, encoding="utf-8")
            self.assertEqual(check_step(step).issues, [])          # finished, output.gro not yet copied
            Path(d, "sys", "7_next").mkdir()                       # not started (no output.log): not checked
            Path(d, "sys", "7_next", "setting.mdp").write_text("", encoding="utf-8")
            self.assertEqual(check_tree(d).issues, [])

    def test_step_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            step = Path(d, "6_md")
            step.mkdir()
            Path(step, "setting.mdp").write_text("", encoding="utf-8")
            self.assertEqual(step_state(step), "not_started")
            Path(step, "run.out").write_text("", encoding="utf-8")
            self.assertEqual(step_state(step), "started")
            Path(step, "run.out").unlink()
            Path(step, "output_before_extend.gro").write_text("", encoding="utf-8")
            self.assertEqual(step_state(step), "started")             # extended, the extension not finished
            Path(step, "output_before_extend.gro").unlink()
            Path(step, "output.tpr").write_text("", encoding="utf-8")
            self.assertEqual(step_state(step), "started")
            Path(step, "output.gro").write_text("", encoding="utf-8")
            self.assertEqual(step_state(step), "finished")

    def test_job_output_of_system(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            sysdir = Path(d, "MOL_rot_+10")
            step = Path(sysdir, "0_em")
            step.mkdir(parents=True)
            Path(step, "output.gro").write_text("", encoding="utf-8")
            Path(step, "output.log").write_text(LOG_END, encoding="utf-8")
            Path(sysdir, "sbatch_MOL.log").write_text("Fatal error:\nThere is no domain decomposition\n",
                                                      encoding="utf-8")
            Path(sysdir, "MOL-1234.err").write_text("CUDA error #700 (cudaErrorIllegalAddress)\n", encoding="utf-8")
            rep = check_tree(d)
            self.assertEqual({(i.code, i.where) for i in rep.issues},
                             {("R005", "MOL_rot_+10"), ("R003", "MOL_rot_+10")})


if __name__ == "__main__":
    unittest.main()
