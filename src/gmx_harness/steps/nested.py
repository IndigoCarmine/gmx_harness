"""A composite step: several steps in sub-folders of one step directory.

Layout of a ``Nested`` step (here as step ``5_umbrella`` of a pipeline)::

    5_umbrella/
        mdrun.sh          runs the children (in order, or up to ``parallel`` at a time)
        smd/              child 0: a normal step directory (grommp.sh, mdrun.sh, run.sh, ...)
        extract/          child 1
        w00/ w01/ ...     more children; a child may itself be a Nested step

Every child directory gets the usual ``run.sh`` (and ``copy.sh`` with ``chain=True``),
so ``bash run.sh`` inside any folder of the tree finishes that folder. The step is
done (has ``output.gro``) when all of its children are.
"""

import shlex
from collections.abc import Sequence
from typing import override

from ..safety import validate_name
from ..scripts import CARRY_DEFAULT, NOOP_GROMPP, body_script, copy_script, step_run_script, validate_carry
from .base import Calculation

INPUT_TO = ("first", "all", "none")
# names the pipeline writes into every step directory
_RESERVED = {"run.sh", "copy.sh"}

Child = Calculation | tuple[str, Calculation]


class Nested(Calculation):
    """
    A step made of child steps, each in its own sub-folder (see the module docstring).

    Attributes:
        children: the child steps, as ``(sub-folder, step)`` pairs or bare steps (sub-folder = step name).
        chain: run the children in order and hand ``output.gro`` (plus ``carry`` files) of each to the next,
            like the top-level pipeline does.
        carry: file patterns handed over between chained children and copied from this folder into
            the children (default ``*.top`` and ``*.itp``).
        broadcast: copy the ``carry`` files of this folder into every child before it runs
            (not with ``chain``: it would overwrite the topology a previous child handed over).
        input_to: which children get this folder's ``input.gro`` and ``carry`` files before they run:
            "first" (default), "all" or "none" (children whose input is placed by other means).
        parallel: run the children from index ``sequential_first`` on up to ``parallel`` at a time
            (background jobs). Each gets ``OMP_NUM_THREADS`` = (``$OMP_NUM_THREADS`` or ``nproc``) /
            ``parallel``, so the runs share the cores of the node; do not also put ``-ntomp`` in
            ``MDRUN_ARGS``. Failed children are listed in ``failed.txt`` and the step fails.
        sequential_first: with ``parallel`` > 1, the number of leading children run one after the other
            before the parallel block (e.g. a steered MD and the frame extraction before the windows).

    Elapsed seconds per child go to ``timing.log``. Children that already have ``output.gro`` are
    skipped, so re-running the step resumes it. When all children are done, ``output.gro`` is the last
    child's ``output.gro`` with ``chain``, otherwise a copy of ``input.gro``.
    """

    def __init__(
        self,
        calculation_name: str,
        children: Sequence[Child],
        *,
        chain: bool = False,
        carry: Sequence[str] = CARRY_DEFAULT,
        broadcast: bool = False,
        input_to: str = "first",
        parallel: int = 1,
        sequential_first: int = 0,
    ):
        self.calculation_name = validate_name(calculation_name)
        pairs: list[tuple[str, Calculation]] = []
        for c in children:
            sub, step = (c.name, c) if isinstance(c, Calculation) else c
            if not isinstance(step, Calculation):
                raise TypeError(f"{calculation_name}: child {sub!r} is not a Calculation")
            pairs.append((validate_name(sub, "child folder"), step))
        if not pairs:
            raise ValueError(f"{calculation_name}: no children")
        names = [s for s, _ in pairs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"{calculation_name}: duplicate child folders {dupes}")
        if input_to not in INPUT_TO:
            raise ValueError(f"input_to must be one of {INPUT_TO}, got {input_to!r}")
        if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
            raise ValueError("parallel must be an int >= 1")
        if parallel > 1 and chain:
            raise ValueError("chained children cannot run in parallel")
        if broadcast and chain:
            raise ValueError("broadcast would overwrite the files chained children hand over; use one of them")
        if not 0 <= sequential_first <= len(pairs):
            raise ValueError(f"sequential_first must be within 0..{len(pairs)}")
        self.children = pairs
        self.chain = chain
        self.carry = validate_carry(tuple(carry))
        self.broadcast = broadcast
        self.input_to = input_to
        self.parallel = parallel
        self.sequential_first = sequential_first

    def _child_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        n = len(self.children)
        for i, (sub, child) in enumerate(self.children):
            generated = child.generate()
            if "mdrun.sh" not in generated:
                raise ValueError(f"{self.calculation_name}/{sub}: generate() must return an mdrun.sh")
            generated.setdefault("grommp.sh", NOOP_GROMPP)
            for fname, content in generated.items():
                if fname in _RESERVED:
                    raise ValueError(f"{self.calculation_name}/{sub}: {fname} is reserved for gmx_harness")
                files[f"{sub}/{fname}"] = content
            nxt = self.children[i + 1][0] if self.chain and i + 1 < n else None
            files[f"{sub}/run.sh"] = step_run_script(is_last=nxt is None)
            if nxt is not None:
                files[f"{sub}/copy.sh"] = copy_script(sub, nxt, nxt, self.carry)
        return files

    def _deliver(self, i: int, sub: str) -> list[str]:
        """Lines that copy this folder's input.gro / carry files into child ``sub`` unless it is finished."""
        give_input = self.input_to == "all" or (self.input_to == "first" and i == 0)
        if not (give_input or self.broadcast):
            return []
        q = shlex.quote(sub)
        lines = [f"if [ ! -f {q}/output.gro ]; then",
                 f"    for f in {' '.join(self.carry)}; do",
                 f'        if [ -e "$f" ]; then cp "$f" {q}/; fi',
                 "    done"]
        if give_input:
            lines.append(f"    cp input.gro {q}/input.gro")
        return lines + ["fi"]

    @override
    def generate(self) -> dict[str, str]:
        files = self._child_files()
        nseq = self.sequential_first if self.parallel > 1 else len(self.children)
        me = shlex.quote(self.calculation_name)
        lines = ["T_ALL=$(date +%s)"]
        for i, (sub, _) in enumerate(self.children[:nseq]):
            q = shlex.quote(sub)
            lines += self._deliver(i, sub) + [
                "t0=$(date +%s)",
                f"(cd {q} && bash run.sh)",
                # names are validated ([A-Za-z0-9_.+-]), so they are safe inside double quotes
                f'[ -f {q}/output.gro ] || {{ echo "ERROR: {self.calculation_name}/{sub} finished without producing '
                'output.gro"; exit 1; }',
                f'echo {q} "$(( $(date +%s) - t0 )) s" >> timing.log',
            ]
        if nseq < len(self.children):
            lines += [
                f"# ---- the remaining children, up to {self.parallel} at a time, sharing the cores of the node ----",
                f"P={self.parallel}",
                "NT=${OMP_NUM_THREADS:-$(nproc)}",
                'PER=$(( NT / P )); [ "$PER" -ge 1 ] || PER=1',
                'FAILED="$PWD/failed.txt"; rm -f "$FAILED"',
                "tp0=$(date +%s)",
            ]
            for i, (sub, _) in enumerate(self.children[nseq:], nseq):
                q = shlex.quote(sub)
                lines += self._deliver(i, sub) + [
                    'while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do wait -n || true; done',
                    f'{{ t1=$(date +%s); if (cd {q} && OMP_NUM_THREADS="$PER" bash run.sh) && [ -f {q}/output.gro ]; then '
                    f'echo {q} "$(( $(date +%s) - t1 )) s" >> timing.log; else echo {q} >> "$FAILED"; fi; }} &',
                ]
            lines += [
                "wait",
                'echo "parallel block: P=$P, $PER threads each, $(( $(date +%s) - tp0 )) s" >> timing.log',
                'if [ -s "$FAILED" ]; then echo "ERROR: failed children: $(tr "\\n" " " < "$FAILED")"; exit 1; fi',
            ]
        last = shlex.quote(self.children[-1][0])
        lines += [
            f"cp {last}/output.gro output.gro" if self.chain else "cp input.gro output.gro",
            'echo TOTAL "$(( $(date +%s) - T_ALL )) s" >> timing.log',
            f"echo {me} is done",
        ]
        files["mdrun.sh"] = body_script("\n".join(lines))
        files["grommp.sh"] = NOOP_GROMPP
        return files
