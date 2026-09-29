"""Install the bundled AI-agent harness (Claude Code skills, AGENTS.md) into a project."""

import filecmp
import shutil
from importlib import resources
from pathlib import Path


def harness_dir() -> Path:
    """Directory of the bundled harness files (skills/, AGENTS.md, llm_docs/)."""
    return Path(str(resources.files("gmx_harness").joinpath("harness")))


def _same_tree(a: Path, b: Path) -> bool:
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    return all(_same_tree(a / d, b / d) for d in cmp.common_dirs)


def install_skills(
    target: str | Path = ".claude/skills",
    *,
    agents_md: str | Path | None = None,
    force: bool = False,
) -> list[str]:
    """
    Copy the bundled Claude Code skills into ``target`` (and AGENTS.md to ``agents_md`` if given).

    Existing skills/files that differ from the bundled ones are left untouched
    (reported as "SKIPPED") unless ``force=True``. Returns one report line per item.

    Example: ``install_skills(".claude/skills", agents_md="AGENTS.md")``
    """
    report: list[str] = []
    target = Path(target)
    src_root = harness_dir() / "skills"
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dst = target / src.name
        if dst.exists():
            if _same_tree(src, dst):
                report.append(f"unchanged  {dst}")
                continue
            if not force:
                report.append(f"SKIPPED    {dst} (exists and differs; force=True to replace)")
                continue
            shutil.rmtree(dst)
            report.append(f"replaced   {dst}")
        else:
            report.append(f"installed  {dst}")
        shutil.copytree(src, dst)
    if agents_md is not None:
        agents_md = Path(agents_md)
        src_md = harness_dir() / "AGENTS.md"
        if agents_md.exists() and not force and agents_md.read_bytes() != src_md.read_bytes():
            report.append(f"SKIPPED    {agents_md} (exists; merge by hand or force=True)")
        else:
            agents_md.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_md, agents_md)
            report.append(f"wrote      {agents_md}")
    return report
