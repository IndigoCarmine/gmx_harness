"""Helpers moved out of the usage workspace: each test rebuilds the workspace script's own
procedure and checks that the library function gives the same result."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import gmx_harness as gh
from gmx_harness import (
    EM,
    MD,
    GroAtom,
    GroFile,
    MDType,
    SolvationMCH,
    atom_types,
    build_plan,
    format_intermolecular_bonds,
    generate_batch_execution_script,
    generate_stepbystep_runfile,
    step_dir_names,
    write_intermolecular_bonds,
    write_job_scripts,
    write_tree,
)

TOP = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
 ca  ca  0.0 0.0 A 3.3e-01 3.6e-01

[ moleculetype ]
;name            nrexcl
 QNP              3

[ atoms ]
;   nr  type  resi  res  atom  cgnr     charge      mass
     1   ca     1   QNP   C1    1    -0.1   12.01  ; qtot -0.1
     2   cp     1   QNP   C2    2     0.1   12.01
#ifdef X
#endif
     3   hn     1   QNP   H1    3     0.0    1.008

[ bonds ]
     1      2    1    1.3870e-01    4.0e+05

[ system ]
 QNP

[ molecules ]
; Compound        nmols
 QNP              1
"""


def workspace_types(path: Path) -> dict[int, str]:
    """2builder/screen_tapes.py / build_sheet.py, verbatim."""
    section, types = "", {}
    for line in path.read_text().splitlines():
        f = line.split(";")[0].split()
        if line.strip().startswith("["):
            section = line.strip("[] \n").split()[0]
        elif f and f[0].isdigit() and section == "atoms":
            types[int(f[0])] = f[1]
    return types


class TestAtomTypes(unittest.TestCase):
    def test_matches_workspace_parser(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "QNP.top")
            p.write_text(TOP)
            ref = workspace_types(p)
            self.assertEqual(atom_types(p), [ref[i + 1] for i in range(len(ref))])

    def test_several_moleculetypes_need_a_name(self) -> None:
        two = TOP.replace("[ system ]", "[ moleculetype ]\n SOL 2\n\n[ atoms ]\n 1 ow 1 SOL OW 1 0 16\n\n[ system ]")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "two.top")
            p.write_text(two)
            with self.assertRaises(ValueError):
                atom_types(p)
            self.assertEqual(atom_types(p, "SOL"), ["ow"])
            with self.assertRaises(ValueError):
                atom_types(p, "NOPE")


class TestIntermolecularBonds(unittest.TestCase):
    def test_matches_interaction_py(self) -> None:
        bonds, length, k = [(10, 186), (12, 300)], 0.29, 1000
        ref = "[ intermolecular_interactions ]\n[ bonds ]\n;  ai    aj funct   length    k\n"
        ref += "".join(f"{i:6d} {j:6d}   6    {length:6.3f}   {k:6.0f}\n" for i, j in bonds)
        self.assertEqual(format_intermolecular_bonds(bonds, length, k), ref)
        with tempfile.TemporaryDirectory() as d:
            # straight from find_intermolecular_pairs: (donor, acceptor, distance)
            write_intermolecular_bonds(Path(d, "x.itp"), [(10, 186, 0.29), (12, 300, 0.3)], length, k)
            self.assertEqual(Path(d, "x.itp").read_bytes(), ref.encode())

    def test_rejects_bad_atoms(self) -> None:
        with self.assertRaises(ValueError):
            format_intermolecular_bonds([(0, 3)], 0.3, 1000)
        with self.assertRaises(ValueError):
            format_intermolecular_bonds([(3, 3)], 0.3, 1000)


def steps() -> list[gh.Calculation]:
    npt = dict(type=MDType.v_rescale_c_rescale, gen_vel="no", temperature=300)
    return [
        EM(nsteps=200, emtol=100, calculation_name="em_vac", defines=["INTER"]),
        SolvationMCH(calculation_name="solv"),
        EM(nsteps=200, emtol=100, calculation_name="em_solv", defines=["INTER"]),
        MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt_lock", nsteps=100, gen_vel="yes", temperature=300),
        MD(calculation_name="md_prod", nsteps=100, **npt),
    ]


def tree_files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class TestTree(unittest.TestCase):
    systems = ["A_sheet_AA-CC", "A_fiber_rot_+10"]
    job = "#!/bin/bash\n#SBATCH -J {JOB_NAME}\nbash {SCRIPT}\n"

    def test_step_dir_names_match_build_plan(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            gro = Path(d, "in.gro")
            gro.write_text("t\n    1\n    1MOL     C1    1   0.000   0.000   0.000\n   1.0 1.0 1.0\n")
            plan = build_plan(steps(), gro, Path(d, "calc", "X"))
            self.assertEqual([s.dirname for s in plan.steps], step_dir_names(steps()))

    def test_write_tree_matches_plan_md(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d, "old"), Path(d, "new")
            for root in (old, new):
                for s in self.systems:
                    (root / s).mkdir(parents=True)
            s = steps()
            # 3md_planning/plan_md.py
            generate_batch_execution_script(self.systems, str(old))
            generate_stepbystep_runfile(self.systems, [(f"{i}_{c.name}", isinstance(c, SolvationMCH))
                                                       for i, c in enumerate(s)], str(old))
            write_job_scripts(old, self.systems, self.job, overwrite=True)
            written = write_tree(new, self.systems, s, self.job, overwrite_jobs=True)
            self.assertEqual(tree_files(old), tree_files(new))
            self.assertEqual(sorted(written), sorted(tree_files(new)))

    def test_stepbystep_rejects_a_mix(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(TypeError):
            generate_stepbystep_runfile(["A"], [EM(), ("1_x", False)], d)  # type: ignore[list-item]

    def test_report(self) -> None:
        p = gh.PlanPreview("wd", "replace_generated", create=["a", "b"], overwrite=["c"], stale_outputs=["1_md/output.gro"])
        self.assertEqual(p.report("calc/A"), "calc/A: 2 created, 1 updated\n    earlier result kept: 1_md/output.gro")
        self.assertTrue(p.report().startswith("wd: "))


class TestAnalyzeScripts(unittest.TestCase):
    def test_matches_make_analysis_scripts_py(self) -> None:
        from gmx_harness.analysis import make_analyze_command_script, make_analyze_command_scripts

        commands = [
            ("6_md_prod", ["hbond-legacy", "-f", "output.xtc", "-s", "output.tpr", "-num", "hbond.xvg"], "0 0", "hbond_prod"),
            ("6_md_prod", ["energy", "-f", "output.edr", "-o", "energy.xvg"], "Potential", "energy_prod"),
        ]
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d, "old"), Path(d, "new")
            for root in (old, new):
                (root / "A" / "6_md_prod").mkdir(parents=True)
            for step, argv, stdin, script in commands:  # 4analysis/make_analysis_scripts.py
                make_analyze_command_script(str(old), step, argv, script, stdin=stdin)
            (old / "run_analyze_all.sh").write_text("\n".join(f"bash {s}.sh" for *_, s in commands) + "\n", newline="\n")
            paths = make_analyze_command_scripts(str(new), commands)
            self.assertEqual(tree_files(old), tree_files(new))
            self.assertEqual(os.path.basename(paths[-1]), "run_analyze_all.sh")


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "rdkit is not installed")
class TestMol(unittest.TestCase):
    def ethanol(self) -> GroFile:
        xyz = [("C1", [-0.0748, 0.0015, -0.0008]), ("C2", [0.0766, 0.0012, 0.0001]), ("O1", [0.1229, -0.1318, 0.0004]),
               ("H1", [-0.1123, 0.1037, -0.0013]), ("H2", [-0.1110, -0.0502, 0.0886]),
               ("H3", [-0.1103, -0.0507, -0.0898]), ("H4", [0.1133, 0.0533, 0.0890]),
               ("H5", [0.1124, 0.0534, -0.0884]), ("H6", [0.2196, -0.1312, 0.0010])]
        return GroFile("EOH", [GroAtom(i + 1, n, "EOH", 1, np.array(p)) for i, (n, p) in enumerate(xyz)], 1.0, 1.0, 1.0)

    def test_round_trip_keeps_names_and_coordinates(self) -> None:
        from gmx_harness import gro_from_mol, save_mol

        g = self.ethanol()
        with tempfile.TemporaryDirectory() as d:
            save_mol(g, Path(d, "e.mol"))
            back = gro_from_mol(Path(d, "e.mol"), g)
        self.assertEqual([a.atom_name for a in back.atoms], [a.atom_name for a in g.atoms])
        np.testing.assert_allclose(back.positions(), g.positions(), atol=1e-4)

    def test_reordered_atoms_are_refused(self) -> None:
        from gmx_harness import gro_from_mol, save_mol

        g = self.ethanol()
        swapped = GroFile("EOH", [g.atoms[2], g.atoms[1], g.atoms[0], *g.atoms[3:]], 1.0, 1.0, 1.0)
        with tempfile.TemporaryDirectory() as d:
            save_mol(swapped, Path(d, "e.mol"))
            with self.assertRaises(ValueError):
                gro_from_mol(Path(d, "e.mol"), g)


if __name__ == "__main__":
    unittest.main()
