"""Run the generated bash/awk topology edits for real (needs bash + awk)."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from gmx_harness import UnsafeNameError, topology


def _bash() -> str | None:
    if os.name == "nt":  # System32\bash.exe is the WSL launcher; prefer Git for Windows
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        return str(git_bash) if git_bash.exists() else None
    return shutil.which("bash")


BASH = _bash()

TOP = """#include "forcefield.itp"
#include "MOL.itp"

[ system ]
test with MCH solvent

[ molecules ]
; Compound  mols
MOL         10

[ intermolecular_interactions ]
[ bonds ]
1 2 6 0.3 5000
"""


@unittest.skipIf(BASH is None, "bash not available")
class TestTopologySnippets(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        (self.d / "topo.top").write_bytes(TOP.encode())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_sh(self, script: str) -> subprocess.CompletedProcess[str]:
        assert BASH is not None
        return subprocess.run([BASH, "-c", "set -eo pipefail\n" + script], cwd=self.d, capture_output=True, text=True)

    def top(self) -> str:
        return (self.d / "topo.top").read_bytes().decode()

    def test_add_from_log_with_include(self) -> None:
        (self.d / "insert.log").write_text("...\nAdded 432 molecules (out of 500 requested)\n")
        r = self.run_sh(topology.add_molecules_from_log("MCH", "insert.log", include_itp="MCH.itp"))
        self.assertEqual(r.returncode, 0, r.stderr)
        expected = TOP.replace("[ system ]", '#include "MCH.itp"\n[ system ]').replace(
            "MOL         10\n\n", "MOL         10\n\nMCH             432\n")
        self.assertEqual(self.top(), expected)
        self.assertEqual((self.d / "topo_old.top").read_bytes().decode(), TOP)

    def test_add_from_dummy_is_verbatim_and_at_end_if_last_section(self) -> None:
        (self.d / "topo.top").write_bytes(b"[ system ]\nx\n[ molecules ]\nMOL 1\n")
        (self.d / "dummy.top").write_text("\nSOL               2155\n")
        r = self.run_sh(topology.add_molecules_from_dummy("dummy.top"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.top(), "[ system ]\nx\n[ molecules ]\nMOL 1\nSOL               2155\n")

    def test_include_is_not_duplicated(self) -> None:
        r = self.run_sh(topology.include_itp("MOL.itp"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.top(), TOP)

    def test_remove_only_that_molecule(self) -> None:
        (self.d / "insert.log").write_text("Added 3 molecules\n")
        self.run_sh(topology.add_molecules_from_log("MCH", "insert.log", include_itp="MCH.itp"))
        r = self.run_sh(topology.remove_molecule("MCH"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.top(), TOP)  # 'test with MCH solvent' is kept (sed '/MCH/d' removed it)

    def test_errors_stop_the_script(self) -> None:
        (self.d / "insert.log").write_text("nothing useful\n")
        r = self.run_sh(topology.add_molecules_from_log("MCH", "insert.log") + "\necho SHOULD-NOT-RUN")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("SHOULD-NOT-RUN", r.stdout)
        (self.d / "topo.top").write_bytes(b"[ system ]\nx\n")
        (self.d / "insert.log").write_text("Added 3 molecules\n")
        r = self.run_sh(topology.add_molecules_from_log("MCH", "insert.log"))
        self.assertNotEqual(r.returncode, 0)

    def test_crlf_topology(self) -> None:
        (self.d / "topo.top").write_bytes(TOP.replace("\n", "\r\n").encode())
        (self.d / "insert.log").write_text("Added 5 molecules\n")
        r = self.run_sh(topology.add_molecules_from_log("MCH", "insert.log"))
        self.assertEqual(r.returncode, 0, r.stderr)
        # awk keeps the CRs on Linux; Git-for-Windows awk drops them on input. Either is fine for GROMACS.
        lines = self.top().replace("\r", "").split("\n")
        self.assertEqual(lines[lines.index("[ intermolecular_interactions ]") - 1], "MCH             5")

    def test_solvent_count(self) -> None:
        (self.d / "input.gro").write_text("t\n0\n 2.0 2.0 2.0\n")
        r = self.run_sh("echo " + topology.solvent_count("input.gro", 98.186, 0.77, 1.0))
        self.assertEqual(r.stdout.strip(), str(int(8.0 / (98.186 / 0.77) * 1.0 * 602.2)))


class TestValidation(unittest.TestCase):
    def test_bad_names(self) -> None:
        with self.assertRaises(UnsafeNameError):
            topology.remove_molecule("MCH;rm")
        with self.assertRaises(UnsafeNameError):
            topology.add_molecules_from_log("MCH", "../log")


if __name__ == "__main__":
    unittest.main()
