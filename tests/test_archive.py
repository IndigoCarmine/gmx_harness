"""Archive: plain copies that round-trip byte for byte, automatic branching, hashes, and the safety rails."""

import contextlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from gmx_harness import archive
from gmx_harness.archive import (
    ArchiveConflictError,
    ArchiveError,
    ArchiveLockedError,
    main,
    plan_export,
    plan_store,
    runs,
    tree,
    verify,
)
from gmx_harness.safety import UnsafeNameError, validate_relpath

TREE = "calc_metad"
SYSTEM = "MOL_rot_+10"
STEPS = ["em_vac", "solv", "em_solv", "nvt_lock", "npt_lock", "npt_unlock"]


def put(p: Path, data: str | bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def force_rmtree(path: Path) -> None:
    """Remove a tree that holds read-only archive files (Windows refuses to delete those otherwise)."""
    if not path.exists():
        return
    for dirpath, dirs, files in os.walk(path):
        for n in dirs + files:
            p = os.path.join(dirpath, n)
            if not os.path.islink(p):
                os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR
                         if os.path.isdir(p) else stat.S_IMODE(os.lstat(p).st_mode) | stat.S_IWUSR)
    os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) | stat.S_IWUSR)
    shutil.rmtree(path)


def writable(p: Path) -> None:
    os.chmod(p, stat.S_IMODE(os.stat(p).st_mode) | stat.S_IWUSR)
    if os.name != "nt":
        os.chmod(p.parent, stat.S_IMODE(os.stat(p.parent).st_mode) | stat.S_IWUSR)


def wiring(sd: Path) -> None:
    """Rewrite run.sh / copy.sh of every step the way build_plan does (last step: no copy.sh)."""
    steps = sorted((d for d in sd.iterdir() if d.is_dir() and d.name[0].isdigit()),
                   key=lambda d: int(d.name.split("_")[0]))
    for i, d in enumerate(steps):
        last = i == len(steps) - 1
        put(d / "run.sh", "#!/bin/bash\nbash mdrun.sh\n" + ("" if last else "bash copy.sh\n"))
        if last:
            (d / "copy.sh").unlink(missing_ok=True)
        else:
            put(d / "copy.sh", f"#!/bin/bash\ncp output.gro ../{steps[i + 1].name}/input.gro\n")


def add_step(sd: Path, index: int, name: str, content: str = "x", state: str = "finished") -> Path:
    d = sd / f"{index}_{name}"
    put(d / "setting.mdp", f"nsteps = {content}\n")
    put(d / "mdrun.sh", "#!/bin/bash\ngmx mdrun\n")
    if state == "finished":
        put(d / "output.gro", f"out {name} {content}\n")
        put(d / "output.log", "Finished mdrun\n")
    elif state == "started":
        put(d / "output.log", "step 100\n")
    wiring(sd)
    return d


def make_system(ws: Path, system: str = SYSTEM, nsteps: int = 6) -> Path:
    sd = ws / TREE / system
    for i, name in enumerate(STEPS[:nsteps]):
        add_step(sd, i, name, f"base{i}")
    put(sd / "0_em_vac" / "input.gro", "start\n")
    put(sd / "0_em_vac" / "#output.log.1#", "backup\n")
    put(sd / "0_em_vac" / ".hidden", "dot\n")
    put(sd / "0_em_vac" / "a+b.itp", "plus\n")
    (sd / "1_solv" / "empty").mkdir()
    put(sd / "DESCRIPTION.md", "# 説明\n日本語のメモ\n")
    put(sd / "notes" / "sub" / "a.txt", "nested\n")
    (sd / "notes" / "emptydir").mkdir()
    put(sd / ".gmx_harness_manifest.json", "{}\n")
    put(sd / "MOL_metad-249.out", "job out\n")
    put(sd / "checks.json", "{}\n")
    put(ws / TREE / "submit.sh", "#!/bin/bash\nsbatch x\n")
    return sd


def snapshot(top: Path) -> tuple[dict[str, bytes], set[str]]:
    files: dict[str, bytes] = {}
    empty: set[str] = set()
    for dirpath, dirs, names in os.walk(top):
        rel = Path(dirpath).relative_to(top).as_posix()
        if not dirs and not names and rel != ".":
            empty.add(rel)
        for n in names:
            files[(Path(dirpath) / n).relative_to(top).as_posix()] = (Path(dirpath) / n).read_bytes()
    return files, empty


def copy_system(sd: Path, ws: Path, system: str) -> Path:
    dst = ws / TREE / system
    shutil.copytree(sd, dst)
    return dst


class ArchiveTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(force_rmtree, self.tmp)
        self.ws = self.tmp / "ws"
        self.root = self.tmp / "archive"

    def store(self, sd: Path, note: str = "") -> archive.StorePreview:
        pv = plan_store(self.root, sd, note=note).write()
        self.assertTrue(pv.written, str(pv))
        return pv

    def run_json(self, name: str) -> dict[str, object]:
        data = json.loads((self.root / name / "run.json").read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        return data


class TestStoreExport(ArchiveTestCase):
    def test_roundtrip_is_byte_identical(self) -> None:
        sd = make_system(self.ws)
        before = snapshot(sd)
        mtimes = {p: p.stat().st_mtime_ns for p in sd.rglob("*") if p.is_file()}
        pv = self.store(sd, note="first")
        self.assertIsNone(pv.parent)
        run = self.root / pv.run
        self.assertTrue(re.fullmatch(r"\d{8}-001_calc_metad_MOL_rot_\+10", pv.run))
        # stored in place, unchanged: a person can open the files directly
        self.assertEqual((run / "DESCRIPTION.md").read_bytes(), (sd / "DESCRIPTION.md").read_bytes())
        self.assertTrue((run / "0_em_vac" / "#output.log.1#").is_file())
        self.assertTrue((run / "_tree" / "submit.sh").is_file())
        self.assertEqual(snapshot(sd), before)  # the source is untouched
        dest = self.tmp / "restore"
        epv = plan_export(self.root, pv.run, dest).write()
        self.assertTrue(epv.written)
        self.assertEqual(snapshot(dest / TREE / SYSTEM), before)
        self.assertEqual((dest / TREE / "submit.sh").read_bytes(), (self.ws / TREE / "submit.sh").read_bytes())
        for p, m in mtimes.items():
            self.assertEqual((dest / TREE / SYSTEM / p.relative_to(sd)).stat().st_mtime_ns, m)
        # copies are writable again, exec bits on POSIX
        out = dest / TREE / SYSTEM / "0_em_vac" / "run.sh"
        self.assertTrue(os.access(out, os.W_OK))
        if os.name != "nt":
            self.assertTrue(out.stat().st_mode & stat.S_IXUSR)
        info = self.run_json(pv.run)
        self.assertEqual(info["job_ids"], ["249"])
        self.assertEqual(info["note"], "first")
        self.assertIn("1_solv/empty", info["empty_dirs"])  # type: ignore[operator]
        self.assertIn("notes/emptydir", info["empty_dirs"])  # type: ignore[operator]
        self.assertTrue(verify(self.root).ok)

    def test_branching_is_detected_with_ancestors(self) -> None:
        sd = make_system(self.ws)
        a = self.store(sd).run
        # B: 0..5 unchanged, a new 6 (adding it rewires 5's run.sh and copy.sh)
        add_step(sd, 6, "md_prod", "350K")
        pv = plan_store(self.root, sd, note="T=350K").preview()
        self.assertEqual(pv.parent, f"{a}/5_npt_unlock")
        self.assertEqual(list(pv.steps), ["6_md_prod"])
        self.assertEqual(pv.rewired, ["5_npt_unlock/copy.sh", "5_npt_unlock/run.sh"])
        b = self.store(sd, note="T=350K").run
        stored = {p.relative_to(self.root / b).as_posix() for p in (self.root / b).rglob("*") if p.is_file()}
        self.assertNotIn("0_em_vac/input.gro", stored)
        self.assertIn("6_md_prod/output.gro", stored)
        self.assertIn("5_npt_unlock/copy.sh", stored)
        self.assertNotIn("5_npt_unlock/setting.mdp", stored)
        # C: from B's state, a step 7 -> parent B/6, grand-parent A
        add_step(sd, 7, "md_long", "long", state="started")
        c = self.store(sd).run
        self.assertEqual(self.run_json(c)["parent"], f"{b}/6_md_prod")
        self.assertEqual(self.run_json(c)["steps"], {"7_md_long": "started"})
        before = snapshot(sd)
        dest = self.tmp / "out"
        plan_export(self.root, c, dest).write()
        self.assertEqual(snapshot(dest / TREE / SYSTEM), before)
        # exporting B rebuilds B's state (6 is last again: no copy.sh)
        plan_export(self.root, b, self.tmp / "outb").write()
        self.assertFalse((self.tmp / "outb" / TREE / SYSTEM / "6_md_prod" / "copy.sh").exists())
        self.assertFalse((self.tmp / "outb" / TREE / SYSTEM / "7_md_long").exists())
        text = tree(self.root)
        self.assertIn(f"└─ 5_npt_unlock → {b}", text)
        self.assertIn(f"   └─ 6_md_prod → {c}", text)
        self.assertIn('+6_md_prod finished  "T=350K"', text)
        self.assertIn("0_em_vac .. 5_npt_unlock  finished", text)
        # an identical replica in another directory is not "already archived": it branches off at its last step
        replica = copy_system(sd, self.ws, "MOL_replica")
        pv = plan_store(self.root, replica).preview()
        self.assertIsNone(pv.already)
        self.assertEqual((pv.parent, pv.steps), (f"{c}/7_md_long", {}))
        # a second system whose steps differ from step 0 on is a new root
        other = make_system(self.ws, "MOL_other")
        put(other / "0_em_vac" / "input.gro", "another start\n")
        self.assertIsNone(plan_store(self.root, other).preview().parent)

    def test_rewired_only(self) -> None:
        sd = make_system(self.ws)
        a = self.store(sd).run
        # 4's copy.sh disappears, 5 gets a copy.sh that it did not have
        (sd / "4_npt_lock" / "copy.sh").unlink()
        put(sd / "5_npt_unlock" / "copy.sh", "#!/bin/bash\n# hand over\n")
        pv = plan_store(self.root, sd).preview()
        self.assertEqual(pv.parent, f"{a}/5_npt_unlock")
        self.assertEqual(pv.steps, {})
        self.assertEqual(pv.rewired, ["5_npt_unlock/copy.sh"])
        self.assertEqual(pv.rewired_removed, ["4_npt_lock/copy.sh"])
        b = self.store(sd).run
        self.assertEqual(self.run_json(b)["rewired_removed"], ["4_npt_lock/copy.sh"])
        before = snapshot(sd)
        plan_export(self.root, b, self.tmp / "out").write()
        self.assertEqual(snapshot(self.tmp / "out" / TREE / SYSTEM), before)
        self.assertIsNotNone(plan_store(self.root, sd).preview().already)

    def test_extension_branches_before_the_changed_step(self) -> None:
        sd = make_system(self.ws)
        add_step(sd, 6, "md_prod", "short")
        a = self.store(sd).run
        put(sd / "6_md_prod" / "output.gro", "extended\n")
        put(sd / "6_md_prod" / "output_before_extend.gro", "short\n")
        pv = plan_store(self.root, sd).preview()
        self.assertEqual(pv.parent, f"{a}/5_npt_unlock")
        self.assertEqual(list(pv.steps), ["6_md_prod"])
        self.assertEqual(pv.rewired, [])
        b = self.store(sd).run
        self.assertTrue((self.root / b / "6_md_prod" / "setting.mdp").is_file())  # the whole step
        self.assertIn(f"├─ 5_npt_unlock → {b}", tree(self.root).replace("└─", "├─"))

    def test_step_states(self) -> None:
        sd = make_system(self.ws, nsteps=4)
        add_step(sd, 4, "npt_lock", state="started")
        add_step(sd, 5, "npt_unlock", state="not_started")
        name = self.store(sd).run
        steps = self.run_json(name)["steps"]
        assert isinstance(steps, dict)
        self.assertEqual(steps["3_nvt_lock"], "finished")
        self.assertEqual(steps["4_npt_lock"], "started")
        self.assertEqual(steps["5_npt_unlock"], "not_started")
        self.assertIn("started (not finished: 4_npt_lock)", tree(self.root).replace("not_started", "started"))

    def test_store_again_writes_nothing(self) -> None:
        sd = make_system(self.ws)
        a = self.store(sd).run
        put(sd / "postcheck.json", '{"when": "now"}\n')  # rewritten after each run: not a new state
        pv = plan_store(self.root, sd).preview()
        self.assertEqual(pv.already, a)
        self.assertIn(f"already archived as {a}", str(pv))
        listing = sorted(os.listdir(self.root))
        pv = plan_store(self.root, sd).write()
        self.assertFalse(pv.written)
        self.assertEqual(sorted(os.listdir(self.root)), listing)
        put(sd / "DESCRIPTION.md", "changed\n")  # a system file did change: new run, no new steps
        pv = plan_store(self.root, sd).preview()
        self.assertIsNone(pv.already)
        self.assertEqual(pv.parent, f"{a}/5_npt_unlock")
        self.assertEqual(pv.steps, {})

    def test_export_refuses_existing_paths(self) -> None:
        sd = make_system(self.ws)
        a = self.store(sd).run
        dest = self.tmp / "out"
        (dest / TREE / SYSTEM).mkdir(parents=True)
        pv = plan_export(self.root, a, dest).preview()
        self.assertFalse(pv.ok)
        with self.assertRaises(ArchiveConflictError):
            plan_export(self.root, a, dest).write()
        (dest / TREE / SYSTEM).rmdir()
        put(dest / TREE / "submit.sh", "something else\n")
        pv = plan_export(self.root, a, dest).preview()
        self.assertTrue(any("different content" in c for c in pv.conflicts), pv.conflicts)
        put(dest / TREE / "submit.sh", "#!/bin/bash\nsbatch x\n")
        pv = plan_export(self.root, a, dest).write()
        self.assertEqual(pv.tree_unchanged, ["submit.sh"])
        self.assertFalse([p for p in (dest / TREE).iterdir() if ".export-" in p.name])
        with self.assertRaises(ArchiveError):
            plan_export(self.root, "nope", dest)
        with self.assertRaises(ArchiveError):
            plan_export(self.root, f"{a}/9_none", dest)
        windows = archive._portability(["a/CON.txt", "b/x?", "c/A.gro", "c/a.gro"], Path("x"), True)
        self.assertEqual(len(windows), 3, windows)
        self.assertEqual(archive._portability(["a/" + "x" * 300], Path("y"), True)[0][-9:], "(Windows)")
        self.assertEqual(archive._portability(["a/CON.txt", "a/" + "x" * 300], Path("x"), False), [])

    def test_export_up_to_a_step(self) -> None:
        sd = make_system(self.ws)
        add_step(sd, 6, "md_prod")
        put(sd / "postcheck.json", "{}\n")
        put(sd / "preflight.ok", "x\n")
        a = self.store(sd).run
        dest = self.tmp / "out"
        pv = plan_export(self.root, f"{a[:12]}/3_nvt_lock", dest).write()
        out = dest / TREE / SYSTEM
        self.assertEqual(pv.steps[-1], "3_nvt_lock")
        self.assertTrue((out / "3_nvt_lock" / "output.gro").is_file())
        self.assertFalse((out / "4_npt_lock").exists())
        self.assertEqual(sorted(pv.excluded), ["MOL_metad-249.out", "checks.json", "postcheck.json", "preflight.ok"])
        self.assertTrue((out / ".gmx_harness_manifest.json").is_file())  # kept; the next plan rewrites it
        self.assertTrue((out / "DESCRIPTION.md").is_file())


class TestIntegrity(ArchiveTestCase):
    def test_sha256sums_format(self) -> None:
        sd = make_system(self.ws)
        name = self.store(sd).run
        raw = (self.root / name / "SHA256SUMS").read_bytes()
        self.assertNotIn(b"\r", raw)
        lines = raw.decode("utf-8").split("\n")
        self.assertEqual(lines[-1], "")
        rels = []
        for line in lines[:-1]:
            m = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
            assert m is not None, line
            rels.append(m.group(2))
            self.assertNotIn("\\", m.group(2))
        self.assertEqual(rels, sorted(rels))
        self.assertIn("run.json", rels)
        self.assertIn("DESCRIPTION.md", rels)
        self.assertIn("_tree/submit.sh", rels)
        self.assertNotIn("SHA256SUMS", rels)

    def test_files_are_read_only(self) -> None:
        sd = make_system(self.ws)
        run = self.root / self.store(sd).run
        for p in run.rglob("*"):
            if p.is_file():
                self.assertFalse(os.access(p, os.W_OK), p)
                self.assertFalse(p.stat().st_mode & stat.S_IWUSR, p)
        if os.name != "nt":
            self.assertFalse(run.stat().st_mode & stat.S_IWUSR)
        self.assertTrue(os.access(self.root / "tree.json", os.W_OK))
        data = json.loads((self.root / "tree.json").read_text(encoding="utf-8"))
        self.assertEqual(data["runs"][0]["run"], run.name)

    def test_verify_reports_modified_missing_extra(self) -> None:
        sd = make_system(self.ws)
        name = self.store(sd).run
        run = self.root / name
        writable(run / "0_em_vac" / "input.gro")
        put(run / "0_em_vac" / "input.gro", "tampered\n")
        writable(run / "DESCRIPTION.md")
        (run / "DESCRIPTION.md").unlink()
        put(run / "notes" / "extra.txt", "extra\n")
        (self.root / ".incoming-20000101-001").mkdir()
        rep = verify(self.root)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.modified, [f"{name}/0_em_vac/input.gro"])
        self.assertEqual(rep.missing, [f"{name}/DESCRIPTION.md"])
        self.assertEqual(rep.extra, [f"{name}/notes/extra.txt"])
        self.assertTrue(any(".incoming-" in w for w in rep.warnings))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--root", str(self.root), "verify", name[:12]]), 1)
        self.assertIn("MODIFIED: 1", out.getvalue())

    def test_verify_reports_missing_parent(self) -> None:
        sd = make_system(self.ws)
        a = self.store(sd).run
        add_step(sd, 6, "md_prod")
        self.store(sd)
        force_rmtree(self.root / a)
        rep = verify(self.root)
        self.assertTrue(any("parent run" in p for p in rep.problems), rep.problems)
        self.assertIn("missing", tree(self.root))

    def test_change_during_copy_aborts(self) -> None:
        sd = make_system(self.ws)
        target = sd / "DESCRIPTION.md"

        def touch(p: Path) -> None:
            if p == target:
                put(p, "written by a running job\n")

        archive._after_copy = touch
        try:
            with self.assertRaisesRegex(ArchiveError, "changed while"):
                plan_store(self.root, sd).write()
        finally:
            archive._after_copy = None
        self.assertEqual(sorted(os.listdir(self.root)), [])  # no run, no .incoming-*, no .lock
        self.assertTrue(plan_store(self.root, sd).write().written)

    def test_lock_is_not_taken_over(self) -> None:
        sd = make_system(self.ws)
        self.root.mkdir()
        put(self.root / ".lock", '{"host": "elsewhere", "pid": 1}\n')
        with self.assertRaisesRegex(ArchiveLockedError, "delete .* by hand"):
            plan_store(self.root, sd).write()
        self.assertTrue((self.root / ".lock").exists())
        (self.root / ".lock").unlink()
        self.store(sd)
        self.assertFalse((self.root / ".lock").exists())

    def test_root_safety(self) -> None:
        sd = make_system(self.ws)
        for bad in (sd / "arch", self.ws / TREE / "arch"):
            pv = plan_store(bad, sd).preview()
            self.assertFalse(pv.ok, bad)
        pv = plan_store(self.ws / "calc_other" / "arch", sd, base=self.ws).preview()
        self.assertTrue(pv.ok)  # calc_other does not exist yet
        (self.ws / "calc_other").mkdir()
        self.assertFalse(plan_store(self.ws / "calc_other" / "arch", sd, base=self.ws).preview().ok)
        put(self.tmp / "busy" / "file.txt", "not an archive\n")
        pv = plan_store(self.tmp / "busy", sd).preview()
        self.assertTrue(any("not an archive" in c for c in pv.conflicts), pv.conflicts)
        with self.assertRaises(ArchiveConflictError):
            plan_store(self.tmp / "busy", sd).write()
        self.assertFalse(self.root.exists())
        self.store(sd)  # created on write
        self.assertTrue((self.root / "tree.json").is_file())

    def test_reserved_names_and_links(self) -> None:
        sd = make_system(self.ws)
        put(sd / "run.json", "{}\n")
        self.assertFalse(plan_store(self.root, sd).preview().ok)
        (sd / "run.json").unlink()
        try:
            os.symlink(sd / "DESCRIPTION.md", sd / "link.md")
        except OSError:
            self.skipTest("symbolic links are not permitted here")
        pv = plan_store(self.root, sd).preview()
        self.assertTrue(any("symbolic link" in c for c in pv.conflicts), pv.conflicts)

    def test_git_commit_is_read_from_files(self) -> None:
        sha1, sha2, sha3 = "a" * 40, "b" * 40, "c" * 40
        repo = self.tmp / "repo"
        put(repo / ".git" / "HEAD", "ref: refs/heads/main\n")
        put(repo / ".git" / "refs" / "heads" / "main", sha1 + "\n")
        self.assertEqual(archive._git_commit(repo / "sub" / "dir"), sha1)
        (repo / ".git" / "refs" / "heads" / "main").unlink()
        put(repo / ".git" / "packed-refs", f"# pack-refs with: peeled\n{sha2} refs/heads/main\n")
        self.assertEqual(archive._git_commit(repo), sha2)
        wt = self.tmp / "wt"
        put(repo / ".git" / "worktrees" / "wt" / "HEAD", "ref: refs/heads/feature\n")
        put(repo / ".git" / "worktrees" / "wt" / "commondir", "../..\n")
        put(repo / ".git" / "refs" / "heads" / "feature", sha3 + "\n")
        put(wt / ".git", f"gitdir: {(repo / '.git' / 'worktrees' / 'wt').as_posix()}\n")
        self.assertEqual(archive._git_commit(wt), sha3)
        put(repo / ".git" / "HEAD", sha1 + "\n")
        self.assertEqual(archive._git_commit(repo), sha1)
        # recorded in run.json, from the workspace (base)
        sd = make_system(repo)
        pv = plan_store(self.root, sd, base=repo).write()
        self.assertEqual(self.run_json(pv.run)["git"], sha1)

    def test_validate_relpath(self) -> None:
        for ok in ("a/#x.1#", ".hidden", "rot_+10/日本語.md", "a b/c"):
            self.assertEqual(validate_relpath(ok), ok)
        for bad in ("", "a\\b", "/abs", "C:x", "a//b", "a/./b", "../x", "a/..", "a\x00b", "a\nb", "a/"):
            with self.assertRaises(UnsafeNameError, msg=repr(bad)):
                validate_relpath(bad)


class TestCommandLine(ArchiveTestCase):
    def cli(self, *argv: str) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv), root=self.root, base=self.ws, trees=[TREE, "calc_missing"])
        return code, out.getvalue() + err.getvalue()

    def test_store_status_tree_export_verify(self) -> None:
        sd = make_system(self.ws)
        code, out = self.cli()
        self.assertEqual(code, 0)
        self.assertIn("(archive is empty)", out)
        self.assertIn(f"{TREE}/{SYSTEM}: not archived", out)
        code, out = self.cli("store", f"{TREE}/{SYSTEM}", "--note", "first")
        self.assertEqual(code, 0, out)
        self.assertIn("status: OK to write", out)
        self.assertFalse(self.root.exists())  # preview only
        code, out = self.cli("store", "--all", "--write")
        self.assertEqual(code, 0, out)
        name = runs(self.root)[0].name
        self.assertIn(f"written as {name}", out)
        code, out = self.cli("status")
        self.assertIn(f"{TREE}/{SYSTEM}: archived as {name}", out)
        add_step(sd, 6, "md_prod")
        code, out = self.cli("status")
        self.assertIn(f"changed since {name} (same up to 5_npt_unlock; new: 6_md_prod)", out)
        code, out = self.cli("store", "--all", "--write")
        self.assertEqual(code, 0, out)
        code, out = self.cli("tree")
        self.assertIn("└─ 5_npt_unlock → ", out)
        code, out = self.cli("export", name, "restore")
        self.assertEqual(code, 0, out)
        self.assertFalse((self.ws / "restore").exists())
        code, out = self.cli("export", name, "restore", "--write")
        self.assertEqual(code, 0, out)
        self.assertTrue((self.ws / "restore" / TREE / SYSTEM / "5_npt_unlock" / "output.gro").is_file())
        code, out = self.cli("export", name, "restore")
        self.assertEqual(code, 1)
        self.assertIn("REFUSED", out)
        code, out = self.cli("verify")
        self.assertEqual(code, 0, out)
        self.assertIn("status: OK", out)
        code, out = self.cli("export", "2099", "x")
        self.assertEqual(code, 1)
        self.assertIn("no run matches", out)
        code, out = self.cli("store")
        self.assertEqual(code, 2)
        code, _ = self.cli("--help")
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
