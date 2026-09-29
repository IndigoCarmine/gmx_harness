import os
import tempfile
import unittest

import numpy as np

from gmx_harness import GroFile, parse_xvg

GRO = """title
    3
    1MOL     C1    1   0.100   0.200   0.300
    1MOL     H1    2   0.110   0.210   0.310
    2SOL     OW    3   1.000   1.000   1.000
   2.50000   2.50000   2.50000
"""

XVG = """# comment
@    title "Energies"
@    xaxis  label "Time (ps)"
@    yaxis  label "(kJ/mol)"
@ s0 legend "Potential"
@ s1 legend "Kinetic"
    0.000000  -100.0  50.0
    1.000000  -110.0  51.0
"""


class TestGro(unittest.TestCase):
    def test_round_trip(self) -> None:
        g = GroFile.from_gro_text(GRO.splitlines(True))
        self.assertEqual(len(g), 3)
        self.assertEqual(g.atoms[2].residue_name, "SOL")
        self.assertAlmostEqual(g.box_x, 2.5)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.gro")
            g.save_gro(p)
            g2 = GroFile.from_gro_file(p)
        self.assertEqual([str(a) for a in g.atoms], [str(a) for a in g2.atoms])

    def test_atom_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            GroFile.from_gro_text(GRO.replace("    3\n", "    4\n", 1).splitlines(True))

    def test_translate_rotate(self) -> None:
        g = GroFile.from_gro_text(GRO.splitlines(True))
        g.translate([1, 0, 0])
        np.testing.assert_allclose(g.atoms[0].coordinate, [1.1, 0.2, 0.3])
        g.rotate(np.diag([-1.0, 1.0, 1.0]))
        np.testing.assert_allclose(g.atoms[0].coordinate, [-1.1, 0.2, 0.3])

    def test_ndx(self) -> None:
        g = GroFile.from_gro_text(GRO.splitlines(True))
        self.assertEqual(g.generate_ndx_text(), "\n\n[ Mol1 ]\n1 2 \n\n[ Mol2 ]\n3 ")  # same text as mylibs


class TestXvg(unittest.TestCase):
    def test_parse(self) -> None:
        x = parse_xvg(XVG)
        self.assertEqual(x.title, "Energies")
        self.assertEqual(x.xlabel, "Time (ps)")
        self.assertEqual(x.legends, ["Potential", "Kinetic"])
        np.testing.assert_allclose(x.x, [0, 1])  # first row is kept
        np.testing.assert_allclose(x.column(1), [50, 51])


if __name__ == "__main__":
    unittest.main()
