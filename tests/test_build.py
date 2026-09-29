import unittest

import numpy as np

from gmx_harness import (
    Assembly,
    GroAtom,
    GroFile,
    make_half_rosette2,
    make_oligorosette,
    make_rosette,
    make_rosette2,
    pre_coordinate,
    precoordinate2,
)
from gmx_harness.build import align, rotation


def monomer() -> GroFile:
    rng = np.random.default_rng(0)
    atoms = [GroAtom(i + 1, n, "MOL", 1, rng.normal(size=3)) for i, n in enumerate(["O1", "N1", "O2", "C1", "C2"])]
    return GroFile("MOL", atoms, 1.0, 1.0, 1.0)


def coords(g: GroFile | Assembly) -> np.ndarray:
    atoms = g.atoms() if isinstance(g, Assembly) else g.atoms
    return np.array([a.coordinate for a in atoms])


def distances(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


class TestRotations(unittest.TestCase):
    def test_rotation_is_right_handed(self) -> None:
        np.testing.assert_allclose(rotation("z", 90, degrees=True) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(rotation("x", 90, degrees=True) @ [0, 1, 0], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(rotation("y", 90, degrees=True) @ [0, 0, 1], [1, 0, 0], atol=1e-12)

    def test_align(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            v, t = rng.normal(size=3), rng.normal(size=3)
            r = align(v, t)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1.0)
            np.testing.assert_allclose(r @ (v / np.linalg.norm(v)), t / np.linalg.norm(t), atol=1e-12)
        np.testing.assert_allclose(align([1, 0, 0], [-1, 0, 0]) @ [1, 0, 0], [-1, 0, 0], atol=1e-12)


class TestPreCoordinate(unittest.TestCase):
    def test_precoordinate2(self) -> None:
        m = monomer()
        before = distances(coords(m))
        precoordinate2(m, top=1, aromatic_nh=2, aromatic_o=3)
        np.testing.assert_allclose(distances(coords(m)), before, atol=1e-12)  # rigid motion
        np.testing.assert_allclose(m.get_child(1).coordinate, 0, atol=1e-12)
        nh = m.get_child(2).coordinate
        self.assertAlmostEqual(nh[1], 0.0)
        self.assertAlmostEqual(nh[2], 0.0)
        self.assertGreater(nh[0], 0.0)
        o = m.get_child(3).coordinate  # O in the xy plane, -y side (same as mylibs' code)
        self.assertAlmostEqual(o[2], 0.0)
        self.assertLess(o[1], 0.0)

    def test_pre_coordinate(self) -> None:
        m = monomer()
        pre_coordinate(m, 1, 2, 3)
        np.testing.assert_allclose(m.get_child(1).coordinate, 0, atol=1e-12)
        s = m.get_child(2).coordinate + m.get_child(3).coordinate
        np.testing.assert_allclose(s / np.linalg.norm(s), [1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(m.get_child(2).coordinate[1], 0.0)


class TestAssemblies(unittest.TestCase):
    def test_rosette2(self) -> None:
        m = precoordinate2(monomer(), 1, 2, 3)
        r = make_rosette2(m, 6, 0.35)
        self.assertEqual(len(r), 6)
        tops = np.array([mol.get_child(1).coordinate for mol in r])
        np.testing.assert_allclose(np.linalg.norm(tops, axis=1), 0.35, atol=1e-12)
        np.testing.assert_allclose(tops[:, 2], 0.0, atol=1e-12)
        angles = np.degrees(np.arctan2(tops[:, 1], tops[:, 0])) % 360
        np.testing.assert_allclose(angles, np.arange(6) * 60.0, atol=1e-9)
        # the input monomer is not modified
        np.testing.assert_allclose(coords(m), coords(precoordinate2(monomer(), 1, 2, 3)))

    def test_half_rosette(self) -> None:
        m = precoordinate2(monomer(), 1, 2, 3)
        r = make_half_rosette2(m, 4, 1.0)
        tops = np.array([mol.get_child(1).coordinate for mol in r])
        np.testing.assert_allclose(np.linalg.norm(tops, axis=1), 1.0, atol=1e-12)
        angles = np.degrees(np.arctan2(tops[:, 1], tops[:, 0])) % 360
        np.testing.assert_allclose(angles, [0, 45, 90, 135], atol=1e-9)  # spread over 180 degrees

    def test_rosette_in_xz_plane(self) -> None:
        m = precoordinate2(monomer(), 1, 2, 3)
        r = make_rosette(m, 4, 2.0, 30)  # size is a diameter
        self.assertEqual(len(r.atoms()), 4 * len(m))
        tops = np.array([mol.get_child(1).coordinate for mol in r])
        np.testing.assert_allclose(tops[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.hypot(tops[:, 0], tops[:, 2]), 1.0, atol=1e-12)

    def test_oligorosette_and_to_gro(self) -> None:
        ring = make_rosette2(precoordinate2(monomer(), 1, 2, 3), 6, 0.35).to_gro(renumber=True)
        stack = make_oligorosette(ring, 4, 0.35, 10.0)
        self.assertEqual(len(stack), 4)
        z = [np.mean(coords(g)[:, 2]) - np.mean(coords(ring)[:, 2]) for g in stack]
        np.testing.assert_allclose(z, [0, 0.35, 0.70, 1.05], atol=1e-12)
        gro = stack.to_gro(box=(5, 5, 1.4), renumber=True)
        self.assertEqual([a.index for a in gro.atoms], list(range(1, 4 * 30 + 1)))
        self.assertEqual((gro.box_x, gro.box_z), (5, 1.4))
        # to_gro copies: moving the result leaves the assembly alone
        gro.translate([1, 0, 0])
        self.assertFalse(np.allclose(coords(gro), coords(stack)))

    def test_helical_slip(self) -> None:
        ring = make_rosette2(monomer(), 3, 0.5).to_gro()
        plain = make_oligorosette(ring, 4, 0.35, 20.0)
        slipped = make_oligorosette(ring, 4, 0.35, 20.0, slip=0.1)
        # slip only adds an in-plane offset per layer, and neighbouring layers are `slip` apart
        offsets = np.array([coords(b).mean(axis=0) - coords(a).mean(axis=0) for a, b in zip(plain, slipped)])
        np.testing.assert_allclose(offsets[:, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(np.diff(offsets[:, :2], axis=0), axis=1), 0.1, atol=1e-12)


class TestPdb(unittest.TestCase):
    def test_pdb_text(self) -> None:
        text = monomer().generate_pdb_text()
        lines = text.splitlines()
        self.assertTrue(lines[1].startswith("CRYST1   10.000   10.000   10.000"))
        atom = lines[2]
        self.assertEqual(atom[:6], "ATOM  ")
        self.assertEqual(atom[12:16], " O1 ")
        self.assertEqual(atom[17:20], "MOL")
        self.assertAlmostEqual(float(atom[30:38]), monomer().atoms[0].coordinate[0] * 10, places=2)
        self.assertEqual(lines[-1], "END")


if __name__ == "__main__":
    unittest.main()
