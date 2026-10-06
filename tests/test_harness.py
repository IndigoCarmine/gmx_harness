import tempfile
import unittest
from pathlib import Path

from gmx_harness import install_skills
from gmx_harness.apidoc import bundled_api_docs_path, generate_api_markdown, write_api_docs

SKILLS = ["gmx-mdp-tuning", "gmx-pipeline", "gmx-relax", "gmx-troubleshoot"]


class TestInstallSkills(unittest.TestCase):
    def test_install_and_keep_user_edits(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = Path(d, ".claude", "skills")
            report = install_skills(target, agents_md=Path(d, "AGENTS.md"))
            self.assertFalse([r for r in report if r.startswith("SKIPPED")], report)
            for name in SKILLS:
                text = (target / name / "SKILL.md").read_text(encoding="utf-8")
                self.assertTrue(text.startswith(f"---\nname: {name}\n"), name)
            self.assertTrue(Path(d, "AGENTS.md").exists())
            self.assertIn("# gmx_harness API reference", (target / "gmx-pipeline" / "api.md").read_text(encoding="utf-8"))

            self.assertTrue(all(r.startswith("unchanged") for r in install_skills(target)))

            (target / "gmx-pipeline" / "SKILL.md").write_text("mine", encoding="utf-8")
            report = install_skills(target)
            self.assertTrue(any(r.startswith("SKIPPED") for r in report))
            self.assertEqual((target / "gmx-pipeline" / "SKILL.md").read_text(encoding="utf-8"), "mine")

            install_skills(target, force=True)
            self.assertNotEqual((target / "gmx-pipeline" / "SKILL.md").read_text(encoding="utf-8"), "mine")


class TestApiDocs(unittest.TestCase):
    def test_covers_public_api(self) -> None:
        text = generate_api_markdown()
        for name in ["build_plan", "OverwritePolicy", "MDParameters", "gmx_command", "RawShellStep", "install_skills"]:
            self.assertIn(name, text)

    def test_compact(self) -> None:
        text = generate_api_markdown()
        self.assertEqual(text.count("\n### `CODES`"), 1)  # re-exports are documented once
        self.assertIn("| `L003` |", text)
        for noise in ["<factory>", "collections.abc.", "pathlib.", "(self", " - \n"]:
            self.assertNotIn(noise, text)

    def test_write_to_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = write_api_docs(Path(d, "api.md"))
            self.assertIn("# gmx_harness API reference", out.read_text(encoding="utf-8"))

    def test_bundled_docs_are_current(self) -> None:
        self.assertEqual(
            bundled_api_docs_path().read_text(encoding="utf-8"),
            generate_api_markdown(),
            'regenerate: python -c "from gmx_harness.apidoc import write_api_docs; write_api_docs()"',
        )


if __name__ == "__main__":
    unittest.main()
