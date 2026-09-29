import os
import tempfile
import unittest

from openmm import CustomNonbondedForce, HarmonicBondForce, NonbondedForce, System, unit
import openmm.app as app

from gmx_harness.relax import (
    add_fixed_atoms,
    add_intermolecular_bonds,
    add_softcore_lj,
    find_inter_itp,
    parse_inter_bonds,
    relax,
    write_gro,
)

nanometer = unit.nanometer


def _find_force(system, name):
    """Look a force up by name rather than by position in the force list."""
    return next(f for f in system.getForces() if f.getName() == name)


class TestFindInterItp(unittest.TestCase):
    def test_finds_include_inside_ifdef_inter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = os.path.join(tmpdir, "system.top")
            with open(top_path, "w") as f:
                f.write(
                    "#include \"forcefield.itp\"\n"
                    "#ifdef INTER\n"
                    "#include \"inter.itp\"\n"
                    "#endif\n"
                )
            with open(os.path.join(tmpdir, "inter.itp"), "w") as f:
                f.write("[ intermolecular_interactions ]\n")

            result = find_inter_itp(top_path)
            self.assertEqual(result, os.path.join(tmpdir, "inter.itp"))

    def test_returns_none_without_ifdef_inter_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = os.path.join(tmpdir, "system.top")
            with open(top_path, "w") as f:
                f.write("#include \"forcefield.itp\"\n")

            self.assertIsNone(find_inter_itp(top_path))


class TestAddFixedAtoms(unittest.TestCase):
    def _make_system(self, n):
        system = System()
        for _ in range(n):
            system.addParticle(1.0)
        return system

    def test_pins_atoms_at_their_positions(self):
        system = self._make_system(3)
        positions = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)] * nanometer

        count = add_fixed_atoms(system, positions, [0, 2])

        self.assertEqual(count, 2)
        force = _find_force(system, "FixedAtoms")
        self.assertEqual(force.getNumParticles(), 2)
        index, params = force.getParticleParameters(0)
        self.assertEqual(index, 0)
        for actual, expected in zip(params, (0.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected)
        index, params = force.getParticleParameters(1)
        self.assertEqual(index, 2)
        for actual, expected in zip(params, (4.0, 5.0, 6.0)):
            self.assertAlmostEqual(actual, expected)

    def test_deduplicates_repeated_indices(self):
        system = self._make_system(3)
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)] * nanometer

        count = add_fixed_atoms(system, positions, [1, 1, 1])

        self.assertEqual(count, 1)

    def test_empty_indices_adds_no_force(self):
        system = self._make_system(2)
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] * nanometer

        count = add_fixed_atoms(system, positions, [])

        self.assertEqual(count, 0)
        self.assertEqual(system.getNumForces(), 0)

    def test_out_of_range_index_raises(self):
        system = self._make_system(2)
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] * nanometer

        with self.assertRaises(ValueError):
            add_fixed_atoms(system, positions, [5])

    def test_negative_index_raises(self):
        """The lower bound is the one that fires in production.

        A 0 in the .itp becomes -1 here, and an upper-bound-only check would
        let it through and silently pin the *last* atom via negative indexing.
        """
        system = self._make_system(2)
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] * nanometer

        with self.assertRaises(ValueError):
            add_fixed_atoms(system, positions, [-1])

        self.assertEqual(system.getNumForces(), 0)


def _write_itp(tmpdir, body):
    itp_path = os.path.join(tmpdir, "inter.itp")
    with open(itp_path, "w") as f:
        f.write(body)
    return itp_path


class TestParseInterBonds(unittest.TestCase):
    """The single .itp parser, shared by add_intermolecular_bonds and relax().

    relax() used to carry its own looser copy of this logic; these cases pin the
    behaviour that copy got wrong.
    """

    def test_parses_bond_to_zero_based_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = _write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                ";  ai    aj funct   length    k\n"
                "     1     10   6     0.300     5000\n",
            )

            self.assertEqual(parse_inter_bonds(itp_path), [(0, 9, 0.300, 5000.0)])

    def test_zero_atom_index_raises(self):
        """Regression: generate_inermolecular_interactions used to emit a 0 here.

        A 0 becomes -1 after the 1-based conversion, which silently referred to
        the last atom instead of failing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = _write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "    51      0   6     0.300     5000\n",
            )

            with self.assertRaises(ValueError) as ctx:
                parse_inter_bonds(itp_path)

            message = str(ctx.exception)
            self.assertIn("inter.itp", message)
            self.assertIn("1-based", message)

    def test_ignores_sections_other_than_bonds(self):
        """The old inline parser had no section tracking and read these as bonds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = _write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "     1     10   6     0.300     5000\n"
                "[ angles ]\n"
                "     1      2      3   1   120.0   500\n",
            )

            self.assertEqual(parse_inter_bonds(itp_path), [(0, 9, 0.300, 5000.0)])

    def test_skips_preprocessor_directives(self):
        """The old inline parser died on these with int('#include')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = _write_itp(
                tmpdir,
                "#include \"other.itp\"\n"
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "#ifdef STRONG\n"
                "     1     10   6     0.300     5000\n"
                "#endif\n",
            )

            self.assertEqual(parse_inter_bonds(itp_path), [(0, 9, 0.300, 5000.0)])

    def test_short_line_raises_with_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = _write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "     1\n",
            )

            with self.assertRaises(ValueError) as ctx:
                parse_inter_bonds(itp_path)

            self.assertIn("inter.itp:3", str(ctx.exception))

    def test_output_of_the_real_generator_parses(self):
        """End-to-end contract with the producer, gmx_harness.itp."""
        from gmx_harness.itp import generate_inermolecular_interactions

        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = os.path.join(tmpdir, "inter.itp")
            generate_inermolecular_interactions(
                natoms=10,
                nmols=6,
                bonds=[(1, 10)],
                nmols_in_rosette=6,
                outfile_path=itp_path,
            )

            parsed = parse_inter_bonds(itp_path)

            self.assertEqual(len(parsed), 6)
            # Every index must be a valid 0-based atom of the 60-atom rosette.
            for i, j, _length, _k in parsed:
                self.assertGreaterEqual(i, 0)
                self.assertGreaterEqual(j, 0)
                self.assertLess(i, 60)
                self.assertLess(j, 60)


class TestAddIntermolecularBonds(unittest.TestCase):
    def _write_itp(self, tmpdir, body):
        return _write_itp(tmpdir, body)

    def test_adds_type6_bond(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                ";  ai    aj funct   length    k\n"
                "     1     10   6     0.300     5000\n",
            )
            system = System()
            for _ in range(10):
                system.addParticle(1.0)

            count = add_intermolecular_bonds(system, itp_path)

            self.assertEqual(count, 1)
            force = next(f for f in system.getForces() if isinstance(f, HarmonicBondForce))
            p1, p2, length, k = force.getBondParameters(0)
            self.assertEqual((p1, p2), (0, 9))
            self.assertAlmostEqual(length.value_in_unit(nanometer), 0.300)
            self.assertAlmostEqual(k.value_in_unit(unit.kilojoule_per_mole / nanometer**2), 5000)

    def test_rejects_unsupported_funct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "     1     10   1     0.300     5000\n",
            )
            system = System()
            for _ in range(10):
                system.addParticle(1.0)

            with self.assertRaises(ValueError):
                add_intermolecular_bonds(system, itp_path)

    def test_no_bonds_section_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(tmpdir, "[ intermolecular_interactions ]\n")
            system = System()
            system.addParticle(1.0)

            with self.assertRaises(ValueError):
                add_intermolecular_bonds(system, itp_path)

    def test_atom_outside_system_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "     1     10   6     0.300     5000\n",
            )
            system = System()
            for _ in range(5):  # too few particles for atom 10
                system.addParticle(1.0)

            with self.assertRaises(ValueError):
                add_intermolecular_bonds(system, itp_path)


class TestAddSoftcoreLj(unittest.TestCase):
    def test_moves_lj_into_softcore_force_and_zeroes_original(self):
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(0.5, 0.3, 1.0)  # charged particle
        nb.addParticle(0.0, 0.3, 1.0)  # neutral particle
        system.addForce(nb)

        softcore = add_softcore_lj(system)

        self.assertIsInstance(softcore, CustomNonbondedForce)
        self.assertEqual(softcore.getNumParticles(), 2)

        # LJ and charge zeroed out of the original NonbondedForce ...
        charge0, sigma0, epsilon0 = nb.getParticleParameters(0)
        self.assertEqual(epsilon0.value_in_unit(unit.kilojoule_per_mole), 0.0)
        self.assertEqual(charge0.value_in_unit(unit.elementary_charge), 0.0)
        # ... but restored via a lambda_q parameter offset instead.
        _, index, charge, _, _ = nb.getParticleParameterOffset(0)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(charge, 0.5)

        # ... and moved into the softcore force's per-particle parameters.
        sigma, epsilon = softcore.getParticleParameters(0)
        self.assertAlmostEqual(sigma, 0.3)
        self.assertAlmostEqual(epsilon, 1.0)

    def test_only_charged_particles_get_lambda_q_offset(self):
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(0.5, 0.3, 1.0)
        nb.addParticle(0.0, 0.3, 1.0)
        system.addForce(nb)

        add_softcore_lj(system)

        self.assertEqual(nb.getNumParticleParameterOffsets(), 1)

    def test_refuses_topology_whose_lj_is_in_a_custom_force(self):
        """Regression: this used to silently soften nothing.

        For combination rule 1/3 or [ nonbond_params ] (MARTINI), GromacsTopFile
        puts LJ in its own CustomNonbondedForce named 'LennardJonesForce' and
        leaves placeholder sigma/epsilon on the NonbondedForce. Softening the
        latter is a no-op that leaves the real 12-6 LJ at full strength.
        """
        system = System()
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(0.0, 1.0, 0.0)  # placeholder params, as GromacsTopFile leaves them
        system.addForce(nb)
        lj = CustomNonbondedForce("A1*A2/r^12-C1*C2/r^6")
        lj.addPerParticleParameter("C")
        lj.addPerParticleParameter("A")
        lj.addParticle([0.0, 0.0])
        lj.setName("LennardJonesForce")
        system.addForce(lj)

        with self.assertRaises(ValueError) as ctx:
            add_softcore_lj(system)

        self.assertIn("LennardJonesForce", str(ctx.exception))

    def test_unrelated_custom_force_does_not_block_softening(self):
        """The guard matches on the force name, not merely on its type.

        A CustomNonbondedForce added for some other purpose must not be mistaken
        for GROMACS' LJ replacement.
        """
        system = System()
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(0.0, 0.3, 1.0)
        system.addForce(nb)
        other = CustomNonbondedForce("0")
        other.setName("SomethingElse")
        other.addParticle([])
        system.addForce(other)

        softcore = add_softcore_lj(system)

        self.assertIsInstance(softcore, CustomNonbondedForce)
        self.assertEqual(softcore.getNumParticles(), 1)

    def test_missing_nonbonded_force_raises_with_a_message(self):
        system = System()
        system.addParticle(1.0)

        with self.assertRaises(ValueError) as ctx:
            add_softcore_lj(system)

        self.assertIn("NonbondedForce", str(ctx.exception))

    def test_mirrors_the_source_nonbonded_method(self):
        """The soft-core force must not hard-code periodicity."""
        system = System()
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.setNonbondedMethod(NonbondedForce.CutoffNonPeriodic)
        nb.addParticle(0.0, 0.3, 1.0)
        system.addForce(nb)

        softcore = add_softcore_lj(system)

        self.assertEqual(
            softcore.getNonbondedMethod(), CustomNonbondedForce.CutoffNonPeriodic
        )


class TestWriteGro(unittest.TestCase):
    def test_writes_expected_fields(self):
        top = app.Topology()
        chain = top.addChain()
        residue = top.addResidue("MOL", chain, id="1")
        top.addAtom("C1", app.element.carbon, residue)
        top.addAtom("C2", app.element.carbon, residue)

        positions = [(0.0, 0.0, 0.0), (1.234, 2.345, 3.456)] * nanometer
        box_vectors = [(2, 0, 0), (0, 2, 0), (0, 0, 2)] * nanometer

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.gro")
            write_gro(top, positions, box_vectors, out_path, "test title")

            with open(out_path, "r") as f:
                lines = f.readlines()

        self.assertEqual(lines[0].rstrip("\n"), "test title")
        self.assertEqual(lines[1].strip(), "2")
        # Pin both atom lines: the residue-name field is left-justified and the
        # atom-name field right-justified, which is easy to regress.
        self.assertEqual(
            lines[2].rstrip("\n"),
            "    1MOL     C1    1   0.000   0.000   0.000",
        )
        self.assertEqual(
            lines[3].rstrip("\n"),
            "    1MOL     C2    2   1.234   2.345   3.456",
        )
        self.assertEqual(lines[4].split(), ["2.00000", "2.00000", "2.00000"])

    def test_round_trips_through_openmms_own_reader(self):
        """Validate the format against a real parser, not a hand-typed string."""
        top = app.Topology()
        chain = top.addChain()
        residue = top.addResidue("MOL", chain, id="1")
        top.addAtom("C1", app.element.carbon, residue)
        top.addAtom("C2", app.element.carbon, residue)

        positions = [(0.0, 0.5, 1.0), (1.234, 2.345, 3.456)] * nanometer
        box_vectors = [(4, 0, 0), (0, 5, 0), (0, 0, 6)] * nanometer

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.gro")
            write_gro(top, positions, box_vectors, out_path, "round trip")

            reread = app.GromacsGroFile(out_path)

        expected = positions.value_in_unit(nanometer)
        actual = reread.getPositions().value_in_unit(nanometer)
        self.assertEqual(len(actual), 2)
        for got, want in zip(actual, expected):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=3)

        box = reread.getPeriodicBoxVectors().value_in_unit(nanometer)
        self.assertAlmostEqual(box[0][0], 4.0, places=3)
        self.assertAlmostEqual(box[1][1], 5.0, places=3)
        self.assertAlmostEqual(box[2][2], 6.0, places=3)

    def test_triclinic_box_raises(self):
        """Silently flattening a triclinic cell would change the periodicity."""
        top = app.Topology()
        chain = top.addChain()
        residue = top.addResidue("MOL", chain, id="1")
        top.addAtom("C1", app.element.carbon, residue)

        positions = [(0.0, 0.0, 0.0)] * nanometer
        box_vectors = [(3, 0, 0), (1.5, 2.6, 0), (1.5, 0.87, 2.45)] * nanometer

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.gro")
            with self.assertRaises(NotImplementedError):
                write_gro(top, positions, box_vectors, out_path, "triclinic")

    def test_multiple_residues_keep_distinct_numbers(self):
        top = app.Topology()
        chain = top.addChain()
        first = top.addResidue("AAA", chain, id="1")
        second = top.addResidue("BBB", chain, id="2")
        top.addAtom("C1", app.element.carbon, first)
        top.addAtom("C2", app.element.carbon, second)

        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] * nanometer
        box_vectors = [(2, 0, 0), (0, 2, 0), (0, 0, 2)] * nanometer

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.gro")
            write_gro(top, positions, box_vectors, out_path, "two residues")
            with open(out_path, "r") as f:
                lines = f.readlines()

        self.assertTrue(lines[2].startswith("    1AAA"))
        self.assertTrue(lines[3].startswith("    2BBB"))

    def test_non_numeric_residue_id_raises_clearly(self):
        top = app.Topology()
        chain = top.addChain()
        residue = top.addResidue("MOL", chain, id="1A")
        top.addAtom("C1", app.element.carbon, residue)

        positions = [(0.0, 0.0, 0.0)] * nanometer
        box_vectors = [(2, 0, 0), (0, 2, 0), (0, 0, 2)] * nanometer

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.gro")
            with self.assertRaises(ValueError) as ctx:
                write_gro(top, positions, box_vectors, out_path, "bad id")

            self.assertIn("1A", str(ctx.exception))


# A minimal two-particle system: enough to drive relax() end to end without
# depending on any force field on disk. Kept as a string rather than a committed
# fixture file so there is nothing to keep in sync.
_TOP_BODY = """\
[ defaults ]
; nbfunc  comb-rule  gen-pairs
1         2          no

[ atomtypes ]
; name  mass    charge  ptype  sigma    epsilon
 CA     12.011  0.000   A      0.35     0.30

[ moleculetype ]
; name  nrexcl
MOL     1

[ atoms ]
;  nr  type  resnr  residue  atom  cgnr  charge  mass
    1  CA    1      MOL      C1    1     0.000   12.011
    2  CA    1      MOL      C2    1     0.000   12.011

[ system ]
relax test

[ molecules ]
MOL 2
"""

# Two molecules, four atoms. The first pair is deliberately overlapping-ish so
# the minimizer has something to do.
_GRO_BODY = """\
relax test
    4
    1MOL     C1    1   0.500   0.500   0.500
    1MOL     C2    2   0.560   0.500   0.500
    2MOL     C1    3   1.500   1.500   1.500
    2MOL     C2    4   1.560   1.500   1.500
   3.00000   3.00000   3.00000
"""


class TestRelax(unittest.TestCase):
    """End-to-end coverage of the module's only public entry point."""

    def _write_inputs(self, tmpdir, inter_body=None):
        gro_path = os.path.join(tmpdir, "structure.gro")
        top_path = os.path.join(tmpdir, "structure.top")

        top_text = _TOP_BODY
        if inter_body is not None:
            with open(os.path.join(tmpdir, "inter.itp"), "w", newline="\n") as f:
                f.write(inter_body)
            top_text += '\n#ifdef INTER\n#include "inter.itp"\n#endif\n'

        with open(gro_path, "w", newline="\n") as f:
            f.write(_GRO_BODY)
        with open(top_path, "w", newline="\n") as f:
            f.write(top_text)
        return gro_path, top_path

    def test_relaxes_without_inter_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gro_path, top_path = self._write_inputs(tmpdir)

            out_path = relax(gro_path, top_path, tmpdir)

            self.assertEqual(out_path, os.path.join(tmpdir, "structure_relaxed.gro"))
            self.assertTrue(os.path.exists(out_path))
            # The output must be readable by the tool that consumes it next.
            reread = app.GromacsGroFile(out_path)
            self.assertEqual(len(reread.getPositions()), 4)

    def test_relaxes_with_inter_restraints(self):
        inter = (
            "[ intermolecular_interactions ]\n"
            "[ bonds ]\n"
            ";  ai    aj funct   length    k\n"
            "     1      3   6     0.300     5000\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gro_path, top_path = self._write_inputs(tmpdir, inter_body=inter)
            start = app.GromacsGroFile(gro_path).getPositions().value_in_unit(nanometer)

            out_path = relax(gro_path, top_path, tmpdir)

            end = app.GromacsGroFile(out_path).getPositions().value_in_unit(nanometer)
            # Atoms 1 and 3 are pinned, so they should barely move; the
            # unrestrained atoms are free to relax.
            for pinned in (0, 2):
                for a, b in zip(start[pinned], end[pinned]):
                    self.assertAlmostEqual(a, b, places=2)

    def test_rejects_itp_with_zero_atom_index(self):
        """Regression for the itp.py ring-closure wrap.

        This used to surface as an opaque OpenMMException about particle -1,
        raised several frames away from the file that caused it.
        """
        inter = (
            "[ intermolecular_interactions ]\n"
            "[ bonds ]\n"
            "     3      0   6     0.300     5000\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gro_path, top_path = self._write_inputs(tmpdir, inter_body=inter)

            with self.assertRaises(ValueError) as ctx:
                relax(gro_path, top_path, tmpdir)

            self.assertIn("inter.itp", str(ctx.exception))

    def test_missing_out_dir_fails_before_minimizing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gro_path, top_path = self._write_inputs(tmpdir)

            with self.assertRaises(NotADirectoryError):
                relax(gro_path, top_path, os.path.join(tmpdir, "does_not_exist"))

    def test_triclinic_input_is_rejected_before_minimizing(self):
        """A box write_gro cannot express should fail up front, not after the run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gro_path, top_path = self._write_inputs(tmpdir)
            # Replace the box line with a triclinic one (9 fields).
            with open(gro_path, "r") as f:
                lines = f.readlines()
            lines[-1] = "   3.00000   3.00000   3.00000   0.00000   0.00000   1.50000   0.00000   0.00000   0.00000\n"
            with open(gro_path, "w", newline="\n") as f:
                f.writelines(lines)

            with self.assertRaises(NotImplementedError):
                relax(gro_path, top_path, tmpdir)

            # Nothing should have been written.
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "structure_relaxed.gro")))

    def test_stem_is_derived_with_splitext(self):
        """A basename of four characters or fewer used to produce '_relaxed.gro'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gro_path, top_path = self._write_inputs(tmpdir)
            short_path = os.path.join(tmpdir, "ab.gro")
            os.replace(gro_path, short_path)

            out_path = relax(short_path, top_path, tmpdir)

            self.assertEqual(os.path.basename(out_path), "ab_relaxed.gro")


if __name__ == "__main__":
    unittest.main()
