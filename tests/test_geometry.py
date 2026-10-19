import itertools
import unittest

import numpy as np

from gmx_harness import (
    Assembly,
    GroAtom,
    GroFile,
    RigidTransform,
    centroid,
    count_contacts,
    find_intermolecular_pairs,
    min_distance,
    plane_normal,
)
from gmx_harness.checks import check_contacts

NATOMS = 5  # N, H, C, O, X per molecule; donor N = 1, acceptor O = 4


def monomer() -> GroFile:
    pos = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.1, 0.0], [-0.1, 0.22, 0.0], [-0.2, -0.1, 0.05]]
    return GroFile("M", [GroAtom(i + 1, n, "MOL", 1, np.array(p)) for i, (n, p) in
                         enumerate(zip(["N1", "H1", "C1", "O1", "C2"], pos))], 0, 0, 0)


def sheet(nx: int = 4, ny: int = 3, ax: float = 0.4, sy: float = 0.35) -> GroFile:
    """Molecules on a grid, periodic along x and y; box z leaves a gap."""
    a = Assembly.replicate(monomer(), [RigidTransform.translation([i * ax, j * sy, 0.0])
                                       for j in range(ny) for i in range(nx)])
    return a.to_gro(box=(nx * ax, ny * sy, 3.0), renumber=True)


def interaction_py(sp: GroFile, natoms: int, donors: tuple[int, ...], acceptors: tuple[int, ...], detect: float):
    """3md_planning/interaction.py's detection, verbatim apart from names."""
    nmols = len(sp) // natoms
    xyz = np.array([a.coordinate for a in sp.atoms])
    box = np.array([sp.box_x, sp.box_y, sp.box_z])
    don = np.array([k * natoms + i for k in range(nmols) for i in donors])
    acc = np.array([k * natoms + i for k in range(nmols) for i in acceptors])
    dv = xyz[acc - 1][None] - xyz[don - 1][:, None]
    dv -= box * np.round(dv / box)
    d = np.linalg.norm(dv, axis=2)
    d[(don - 1)[:, None] // natoms == (acc - 1)[None] // natoms] = np.inf
    hit = d < detect
    return [(int(don[i]), int(acc[j])) for i, j in zip(*np.nonzero(hit))], d[hit]


class TestIntermolecularPairs(unittest.TestCase):
    def test_matches_interaction_py_including_the_periodic_boundary(self) -> None:
        sp = sheet()
        bonds, dist = interaction_py(sp, NATOMS, (1, 3), (4, 2), 0.45)
        got = find_intermolecular_pairs(sp, NATOMS, (1, 3), (4, 2), 0.45)
        self.assertEqual([(i, j) for i, j, _ in got], bonds)
        np.testing.assert_array_equal([d for *_, d in got], dist)
        # the last column of molecules bonds to the first through the boundary along x
        self.assertTrue(any(j < i for i, j, _ in got))

    def test_without_pbc_misses_the_boundary(self) -> None:
        sp = sheet()
        with_pbc = find_intermolecular_pairs(sp, NATOMS, (1,), (4,), 0.45)
        without = find_intermolecular_pairs(sp, NATOMS, (1,), (4,), 0.45, pbc="")
        self.assertLess(len(without), len(with_pbc))

    def test_bad_selection(self) -> None:
        with self.assertRaises(ValueError):
            find_intermolecular_pairs(sheet(), NATOMS, (6,), (4,), 0.3)
        with self.assertRaises(ValueError):
            find_intermolecular_pairs(sheet(), 7, (1,), (4,), 0.3)
        with self.assertRaises(ValueError):
            find_intermolecular_pairs(sheet(), NATOMS, (1,), (4,), 0.3, pbc="xw")


class TestMinDistanceAndContacts(unittest.TestCase):
    def random_chain(self) -> GroFile:
        rng = np.random.default_rng(4)
        a = Assembly.replicate(monomer(), [RigidTransform.rotation("z", rng.uniform(0, 6.28)) @
                                           RigidTransform.translation(rng.normal(scale=0.4, size=3))
                                           for _ in range(7)])
        return a.to_gro(box=(5.0, 5.0, 5.0), renumber=True)

    def test_matches_screen_tapes_non_neighbour_loop(self) -> None:
        """2builder/screen_tapes.py: pairs j - i >= 2, heavy atoms, min / clash count / Ca subset."""
        g = self.random_chain()
        xs = [np.array([a.coordinate for a in g.atoms[k * NATOMS:(k + 1) * NATOMS]]) for k in range(7)]
        heavy = np.array([not a.atom_name.startswith("H") for a in monomer().atoms])
        ca = np.array([n in ("C1", "C2") for n in [a.atom_name for a in monomer().atoms]])
        d_min, n_clash, a_min = np.inf, 0, np.inf
        for i, j in itertools.combinations(range(7), 2):
            if j - i < 2:
                continue
            d = np.linalg.norm(xs[i][heavy][:, None] - xs[j][heavy][None], axis=2)
            d_min, n_clash = min(d_min, d.min()), n_clash + int((d < 0.3).sum())
            a_min = min(a_min, d[np.ix_(ca[heavy], ca[heavy])].min())

        heavy_atoms = [i + 1 for i, h in enumerate(heavy) if h]
        ca_atoms = [i + 1 for i, c in enumerate(ca) if c]
        kw = dict(pbc="", exclude_neighbours=1)
        self.assertEqual(min_distance(g, NATOMS, range(7), atoms=heavy_atoms, **kw), d_min)
        self.assertEqual(count_contacts(g, NATOMS, range(7), cutoff=0.3, atoms=heavy_atoms, **kw), n_clash)
        self.assertEqual(min_distance(g, NATOMS, range(7), atoms=ca_atoms, **kw), a_min)

    def test_between_two_groups_with_pbc(self) -> None:
        sp = sheet(nx=4, ny=3)
        layer1, layer3 = range(0, 4), range(8, 12)
        # layer 3 sits next to layer 1 through the boundary along y
        self.assertAlmostEqual(min_distance(sp, NATOMS, layer1, layer3, atoms=[1]), 0.35, places=9)  # N on the grid
        self.assertAlmostEqual(min_distance(sp, NATOMS, layer1, layer3, atoms=[1], pbc="x"), 0.70, places=9)
        self.assertEqual(min_distance(sp, NATOMS, [0], [1], exclude_neighbours=1), np.inf)

    def test_centroid_and_plane(self) -> None:
        g = monomer()
        np.testing.assert_allclose(centroid(g, [1, 2]), [0.05, 0.0, 0.0])
        n = plane_normal(g, [1, 2, 3, 4])
        np.testing.assert_allclose(abs(n), [0.0, 0.0, 1.0], atol=1e-12)


class TestCheckContactsPbc(unittest.TestCase):
    def test_finds_contact_through_the_boundary_only_with_pbc(self) -> None:
        atoms = [GroAtom(1, "C1", "A", 1, np.array([0.02, 1.0, 1.0])), GroAtom(2, "C1", "B", 2, np.array([1.97, 1.0, 1.0]))]
        g = GroFile("t", atoms, 2.0, 2.0, 2.0)
        self.assertEqual(len(check_contacts(g, 1, 0.1).issues), 0)
        self.assertEqual(len(check_contacts(g, 1, 0.1, pbc="x").issues), 1)
        self.assertEqual(len(check_contacts(g, 1, 0.1, pbc="yz").issues), 0)

    def test_same_count_as_without_pbc_inside_the_box(self) -> None:
        rng = np.random.default_rng(2)
        atoms = [GroAtom(i + 1, "C", "M", i // 3 + 1, rng.uniform(1.0, 2.0, size=3)) for i in range(60)]
        g = GroFile("t", atoms, 6.0, 6.0, 6.0)
        self.assertEqual(str(check_contacts(g, 3, 0.15).issues), str(check_contacts(g, 3, 0.15, pbc="xyz").issues))


if __name__ == "__main__":
    unittest.main()
