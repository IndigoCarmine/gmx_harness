import unittest
import os
import tempfile

from gmx_harness.itp import generate_inermolecular_interactions


class TestItp(unittest.TestCase):

    def test_generate_inermolecular_interactions(self):
        """
        Tests the generate_inermolecular_interactions function.
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
            outfile_path = tmpfile.name

        generate_inermolecular_interactions(
            natoms=10, nmols=6, bonds=[(1, 12)], nmols_in_rosette=6, outfile_path=outfile_path
        )

        with open(outfile_path, "r") as f:
            lines = f.readlines()

        expected_lines = [
            "[ intermolecular_interactions ]\n",
            "[ bonds ]\n",
            ";  ai    aj funct   length    k\n",
            "     1     12   6     0.300     5000\n",
            "    11     22   6     0.300     5000\n",
            "    21     32   6     0.300     5000\n",
            "    31     42   6     0.300     5000\n",
            "    41     52   6     0.300     5000\n",
            "    51      2   6     0.300     5000\n",
        ]

        self.assertEqual(lines, expected_lines)

        os.remove(outfile_path)

    def test_generate_inermolecular_interactions_multiple_rosettes(self):
        """
        Bonds in the second rosette must reference that rosette's own atoms,
        not collapse back onto the first rosette.
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
            outfile_path = tmpfile.name

        bonds = generate_inermolecular_interactions(
            natoms=10,
            nmols=12,
            bonds=[(1, 12)],
            nmols_in_rosette=6,
            outfile_path=outfile_path,
        )

        expected_bonds = [
            # rosette 0 (molecules 0-5, atoms 1-60), ring closure on the last molecule
            (1, 12),
            (11, 22),
            (21, 32),
            (31, 42),
            (41, 52),
            (51, 2),
            # rosette 1 (molecules 6-11), offset to its own atoms (61-120).
            # The closure must stay inside this rosette: it used to wrap to atom
            # 60, which belongs to rosette 0.
            (61, 72),
            (71, 82),
            (81, 92),
            (91, 102),
            (101, 112),
            (111, 62),
        ]
        self.assertEqual(bonds, expected_bonds)

        os.remove(outfile_path)


if __name__ == "__main__":
    unittest.main()
