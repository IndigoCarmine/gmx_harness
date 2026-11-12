import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gmx_harness import Assembly, GroAtom, GroFile, RigidTransform, make_oligorosette, make_rosette2, save_assembly
from gmx_harness.index import molecule_atoms, write_ndx


def monomer(n: int = 6, seed: int = 0) -> GroFile:
    rng = np.random.default_rng(seed)
    names = ["N1", "H1", "C1", "O1", "C2", "H2", "C3", "O2"][:n]
    atoms = [GroAtom(i + 1, nm, "MOL", 1, rng.normal(scale=0.3, size=3)) for i, nm in enumerate(names)]
    return GroFile("MOL", atoms, 0.0, 0.0, 0.0)


def text(g: GroFile) -> str:
    return "\n".join(g.generate_gro_text())


def kabsch_screen_tapes(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2builder/screen_tapes.py's kabsch, verbatim."""
    sc, dc = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - sc).T @ (dst - dc))
    r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, dc - r @ sc


class TestRigidTransform(unittest.TestCase):
    def test_rejects_mirror_and_scaling(self) -> None:
        with self.assertRaises(ValueError):
            RigidTransform.from_matrix(np.diag([1.0, 1.0, -1.0]))
        with self.assertRaises(ValueError):
            RigidTransform.from_matrix(2 * np.eye(3))

    def test_immutable(self) -> None:
        t = RigidTransform.rotation("z", 30, degrees=True)
        with self.assertRaises(AttributeError):
            t._r = np.eye(3)  # type: ignore[misc]
        with self.assertRaises(ValueError):
            t.matrix[0, 0] = 2.0

    def test_compose_inverse_power(self) -> None:
        a = RigidTransform.rotation("x", 0.4) @ RigidTransform.translation([1.0, 2.0, 3.0])
        p = np.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.5]])
        np.testing.assert_allclose(a.inverse().apply(a.apply(p)), p, atol=1e-12)
        r = RigidTransform.rotation("z", 10, degrees=True)
        self.assertAlmostEqual(r.power(36).angle, 0.0, places=5)
        np.testing.assert_allclose((a @ r).apply(p), a.apply(r.apply(p)), atol=1e-12)

    def test_rotation_about_center(self) -> None:
        t = RigidTransform.rotation("z", 90, degrees=True, center=[1.0, 0.0, 0.0])
        np.testing.assert_allclose(t.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(t.apply([2.0, 0.0, 0.0]), [1.0, 1.0, 0.0], atol=1e-12)

    def test_frame(self) -> None:
        t = RigidTransform.frame([0.0, 1.0, 0.0], [1.0, 0.5, 0.0])
        np.testing.assert_allclose(t.apply([0.0, 2.0, 0.0]), [2.0, 0.0, 0.0], atol=1e-12)
        out = t.apply([1.0, 0.0, 0.0])
        self.assertAlmostEqual(out[1], 0.0)
        self.assertGreater(out[2], 0.0)

    def test_fit_matches_screen_tapes_kabsch_and_never_mirrors(self) -> None:
        rng = np.random.default_rng(1)
        src = rng.normal(size=(5, 3))
        true = RigidTransform.rotation("y", 0.7) @ RigidTransform.translation([0.3, -0.2, 1.0])
        dst = true.apply(src) + rng.normal(scale=0.01, size=(5, 3))
        fit = RigidTransform.fit(src, dst)
        r, t = kabsch_screen_tapes(src, dst)
        np.testing.assert_array_equal(fit.matrix, r)
        np.testing.assert_array_equal(fit.shift, t)
        mirrored = src * [1.0, 1.0, -1.0]
        self.assertGreater(np.linalg.det(RigidTransform.fit(src, mirrored).matrix), 0.0)


class TestGroFileReadOnly(unittest.TestCase):
    def test_positions_are_a_read_only_copy(self) -> None:
        g = monomer()
        p = g.positions()
        with self.assertRaises(ValueError):
            p[0, 0] = 9.0
        np.testing.assert_array_equal(g.positions([2, 1]), p[[1, 0]])
        with self.assertRaises(IndexError):
            g.positions([0])

    def test_transformed_leaves_the_original(self) -> None:
        g = monomer()
        before = text(g)
        moved = g.transformed(RigidTransform.translation([1.0, 0.0, 0.0]))
        self.assertEqual(text(g), before)
        np.testing.assert_allclose(moved.positions() - g.positions(), [[1.0, 0.0, 0.0]] * len(g))


class TestAssembly(unittest.TestCase):
    def test_replicate_numbers_residues(self) -> None:
        a = Assembly.replicate(monomer(), [RigidTransform.translation([k, 0.0, 0.0]) for k in range(3)])
        self.assertEqual([m.atoms[0].residue_number for m in a], [1, 2, 3])

    def test_number_residues_keeps_the_residues_of_a_rosette(self) -> None:
        """build.py: ring residues 1..6, then disk i residues + i*6, all sequential."""
        mono = monomer()
        ring = make_rosette2(mono, 6, 0.44)
        for i, m in enumerate(ring):
            m.set_residue_number(i + 1)
        rosette = ring.to_gro(box=(1, 1, 1), renumber=True)
        old = make_oligorosette(rosette, 4, 0.35, 10.0)
        old.translate([0.0, 0.0, -(4 - 1) * 0.35 / 2])
        for i, disk in enumerate(old):
            for atom in disk.atoms:
                atom.residue_number += i * 6
        new = make_oligorosette(rosette, 4, 0.35, 10.0).transformed(
            RigidTransform.translation([0.0, 0.0, -(4 - 1) * 0.35 / 2])).number_residues()
        self.assertEqual(text(old.to_gro(renumber=True)), text(new.to_gro(renumber=True)))

    def test_slice_copy_and_concat(self) -> None:
        a = Assembly.replicate(monomer(), [RigidTransform.identity()] * 3)
        b = a[1:2]
        self.assertEqual(len(b), 1)
        self.assertIsNot(b.molecules[0], a.molecules[1])
        self.assertEqual(len(Assembly.concat(a, b)), 4)
        with self.assertRaises(TypeError):
            a[0]  # type: ignore[index]

    def test_bounds_box_and_centre(self) -> None:
        a = Assembly.replicate(monomer(), [RigidTransform.translation([k, 0.0, 0.0]) for k in range(2)])
        lo, hi = a.bounds()
        np.testing.assert_allclose(a.extent(), hi - lo)
        box = a.fitting_box(factor=2.0, pad=0.5, min_edge=3.0)
        np.testing.assert_allclose(box, np.maximum(2.0 * (hi - lo) + 1.0, 3.0))
        c = a.centered_in_box(box)
        lo2, hi2 = c.bounds()
        np.testing.assert_allclose((lo2 + hi2) / 2, np.array(box) / 2, atol=1e-12)
        o = a.centered_in_box(box, about=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(o.positions() - a.positions(), [np.array(box) / 2] * len(a.positions()))

    def test_sheet_matches_build_sheet_procedure(self) -> None:
        """2builder/build_sheet.py, rebuilt with replicate/repeat/centered_in_box: same .gro text."""
        mono = monomer(8, seed=3)
        natoms, period, n_tape, n_layer, stack, gap = len(mono), 2, 5, 3, 0.35, 2.0
        rep_vec = np.array([0.9, 0.0, 0.0])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # +90 deg about x
        piece = Assembly.replicate(mono, [RigidTransform.from_matrix(np.eye(3), rx.T @ rep_vec * k)
                                          for k in range(4)]).to_gro(renumber=True)  # a tape piece along z

        # --- the script's procedure (verbatim apart from names)
        xyz = np.array([a.coordinate for a in piece.atoms]).reshape(-1, natoms, 3)
        xyz = xyz @ rx.T
        repeat = (xyz[period] - xyz[0]).mean(0)
        nrep = -(-n_tape // period)
        lx = nrep * repeat[0]
        tape = [xyz[k] + [r * repeat[0], 0.0, 0.0] for r in range(nrep) for k in range(period)]
        coords = [m + [0.0, layer * stack, 0.0] for layer in range(n_layer) for m in tape]
        allx = np.vstack(coords)
        box = (lx, n_layer * stack, np.ptp(allx[:, 2]) + 2 * gap)
        centre = np.array(box) / 2 - (allx.max(0) + allx.min(0)) / 2
        mols = []
        for i, c in enumerate(coords):
            mol = copy.deepcopy(mono)
            for atom, p in zip(mol.atoms, c + centre):
                atom.coordinate = p
            mol.set_residue_number(i + 1)
            mols.append(mol)
        old = Assembly(mols).to_gro(title="sheet", box=box, renumber=True)

        # --- the same with the non-mutating API
        tape_piece = Assembly([GroFile("MOL", piece.atoms[k * natoms:(k + 1) * natoms], 0, 0, 0)
                               for k in range(len(piece) // natoms)])
        turned = tape_piece.transformed(RigidTransform.from_matrix(rx))
        unit = turned[0:period]
        sheet = unit.repeat([repeat[0], 0.0, 0.0], nrep).repeat([0.0, stack, 0.0], n_layer)
        box2 = (lx, n_layer * stack, float(sheet.extent()[2]) + 2 * gap)
        new = sheet.centered_in_box(box2).number_residues().to_gro(title="sheet", box=box2, renumber=True)
        self.assertEqual(text(old), text(new))


class TestSaveAssembly(unittest.TestCase):
    def test_writes_gro_pdb_ndx_like_build_py(self) -> None:
        a = Assembly.replicate(monomer(), [RigidTransform.translation([k, 0.0, 0.0]) for k in range(4)])
        gro = a.to_gro(box=(5.0, 5.0, 5.0), renumber=True)
        n = len(monomer())
        groups = {"Fiber1": range(4), "Half": range(2)}
        with tempfile.TemporaryDirectory() as d:
            paths = save_assembly(Path(d, "MOL_fiber_rot_+0.5"), gro, n, groups)
            self.assertEqual([p.name for p in paths], ["MOL_fiber_rot_+0.5.gro", "MOL_fiber_rot_+0.5.pdb",
                                                        "MOL_fiber_rot_+0.5.ndx"])
            write_ndx(Path(d, "ref.ndx"), {g: molecule_atoms(n, m) for g, m in groups.items()})
            self.assertEqual(paths[2].read_text(), Path(d, "ref.ndx").read_text())
            self.assertEqual(paths[1].read_text(), gro.generate_pdb_text())
            with self.assertRaises(ValueError):
                save_assembly(Path(d, "x"), gro, n, {"Bad": [4]})
            with self.assertRaises(ValueError):
                save_assembly(Path(d, "x"), gro, n + 1, groups)


if __name__ == "__main__":
    unittest.main()
