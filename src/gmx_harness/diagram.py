"""Draw a pipeline as a draw.io (diagrams.net) flowchart: one card per step.

The cards are read from what each step would write (``generate()``): steps that
write a ``setting.mdp`` show nsteps, emtol or simulated time and temperature, and
the ``define`` flags, so subclasses and ``additional_mdp_parameters`` are shown as
they will run. A ``Nested`` step is drawn as a frame around the cards of its
children (recursively); runs of identical children, such as umbrella windows that
differ only in their PLUMED input, are drawn once with a count. Nothing is executed
and nothing is written unless ``save_drawio`` is called.
"""

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html import escape

from .mdp import MDParameters
from .steps import (
    AWH,
    AddFiles,
    BarMethod,
    Calculation,
    Nested,
    RawShellStep,
    RemoveResidue,
    ResizeBox,
    RuntimeSolvation,
    Solvation,
    SolvationMCH,
    SolvationSCP216,
)

# draw.io's built-in palette: (fill, stroke)
_BLUE = ("#dae8fc", "#6c8ebf")
_GREEN = ("#d5e8d4", "#82b366")
_YELLOW = ("#fff2cc", "#d6b656")
_PURPLE = ("#e1d5e7", "#9673a6")
_GRAY = ("#f5f5f5", "#666666")

_EM_INTEGRATORS = {"steep", "cg", "l-bfgs"}

# card and frame geometry (px)
_W, _H, _GAP_X, _GAP_Y, _MARGIN = 330, 120, 150, 90, 20
_PAD, _HEAD, _INNER_GAP = 20, 40, 50  # inside a Nested frame: padding, header height, gap between children


def _mdp_of(calc: Calculation) -> MDParameters | None:
    text = calc.generate().get("setting.mdp")
    return MDParameters.from_text(text) if text is not None else None


def _value(mdp: MDParameters, key: str) -> str | None:
    v = mdp.get(key)
    if v is None:
        return None
    s = str(v).split(";", 1)[0].strip()
    return s or None


def _number(s: str | None) -> float | None:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def _defines(mdp: MDParameters, labels: Mapping[str, str], none_label: str) -> str:
    flags = [f[2:] if f.startswith("-D") else f for f in (_value(mdp, "define") or "").split()]
    names = [labels.get(f.split("=", 1)[0], f) for f in flags]
    return ", ".join(names) if names else none_label


def _card(calc: Calculation, labels: Mapping[str, str], none_label: str) -> tuple[str, list[str], tuple[str, str]]:
    """(heading, body lines, colors) of one step."""
    if isinstance(calc, SolvationMCH):
        return "Solvation (MCH) gmx solvate", [f"scale: {calc.scale:g}"], _GREEN
    if isinstance(calc, SolvationSCP216):
        return "Solvation (SPC216) gmx solvate", [], _GREEN
    if isinstance(calc, Solvation):
        return f"Solvation ({calc.solvent}) gmx insert-molecules", [f"nmol: {calc.nmol:,}"], _GREEN
    if isinstance(calc, RuntimeSolvation):
        return f"Solvation ({calc.solvent}) gmx insert-molecules", [f"rate: {calc.rate:g} x bulk"], _GREEN
    if isinstance(calc, ResizeBox):
        lines = [f"box: {calc.x} x {calc.y} x {calc.z} nm"]
        if calc.remove_resname:
            lines.append(f"remove: {calc.remove_resname}")
        return "Resize Box", lines, _GRAY
    if isinstance(calc, RemoveResidue):
        return "Remove Residue", [f"resname: {calc.resname}"], _GRAY
    if isinstance(calc, AddFiles):
        return "Add Files", [", ".join(sorted(calc.files)) or none_label], _GRAY
    if isinstance(calc, RawShellStep):
        return "Shell Step", [calc.command.splitlines()[0] if calc.command else ""], _GRAY

    mdp = _mdp_of(calc)
    if mdp is None:
        return type(calc).__name__, [], _GRAY
    integrator = (_value(mdp, "integrator") or "md").lower()
    nsteps = _number(_value(mdp, "nsteps"))
    steps_text = f"{int(nsteps):,}" if nsteps is not None else "?"
    defines = f"defines: {_defines(mdp, labels, none_label)}"
    if integrator in _EM_INTEGRATORS:
        emtol = _number(_value(mdp, "emtol"))
        lines = [f"nsteps: {steps_text}"]
        if emtol is not None:
            lines.append(f"emtol: {emtol:.1f} kJ/mol/nm")
        return "Energy Minimization", lines + [defines], _BLUE
    if nsteps is not None and nsteps >= 0:
        steps_text += f" ({nsteps * mdp.dt() / 1000:.1f} ns)"
    lines = [f"nsteps: {steps_text}"]
    ref_t = (_value(mdp, "ref_t") or _value(mdp, "ref-t") or "").split()
    temperature = _number(ref_t[0]) if ref_t else None
    if temperature is not None:
        lines.append(f"temperature: {temperature:.1f} K")
    lines.append(defines)
    if isinstance(calc, AWH):
        return "AWH Simulation", lines, _PURPLE
    if isinstance(calc, BarMethod):
        return "Free Energy (BAR)", lines, _PURPLE
    return "MD Simulation", lines, _YELLOW


@dataclass
class _Node:
    """One drawn box: a card, or a frame (``children`` non-empty) for a Nested step."""

    heading: str
    lines: list[str]
    colors: tuple[str, str]
    names: list[str]  # step / sub-folder names; more than one when identical children were merged
    children: list["_Node"] = field(default_factory=list)
    row_from: int = 0  # children[row_from:] stand side by side (parallel block); the others are stacked
    chain: bool = True  # chain-style arrows (solid) between children; dashed when they only run in order
    w: float = _W
    h: float = _H

    def key(self) -> tuple[object, ...]:
        """What the box shows, apart from names: boxes with equal keys look the same."""
        return (self.heading, tuple(self.lines), self.colors, self.row_from, self.chain,
                tuple((c.key(), len(c.names)) for c in self.children))

    def title(self, show_names: bool) -> str:
        text = self.heading
        if show_names:
            text += f" [{self.names[0]}]" if len(self.names) == 1 else f" [{self.names[0]} … {self.names[-1]}]"
        if len(self.names) > 1:
            text += f" ×{len(self.names)}"
        return text


def _merge_runs(nodes: list[_Node]) -> list[_Node]:
    """Merge consecutive boxes that look the same into one (its names list all of them)."""
    out: list[_Node] = []
    for n in nodes:
        if out and out[-1].key() == n.key():
            out[-1].names.extend(n.names)
        else:
            out.append(n)
    return out


def _node(calc: Calculation, name: str, labels: Mapping[str, str], none_label: str, collapse: bool) -> _Node:
    if not isinstance(calc, Nested):
        heading, lines, colors = _card(calc, labels, none_label)
        return _Node(heading, lines, colors, [name])
    kids = [_node(c, sub, labels, none_label, collapse) for sub, c in calc.children]
    if calc.parallel > 1:
        first, rest = kids[:calc.sequential_first], kids[calc.sequential_first:]
        heading = f"Nested (parallel ×{calc.parallel})"
    else:
        first, rest = kids, []
        heading = "Nested (chain)" if calc.chain else "Nested (in order)"
    if collapse:
        first, rest = _merge_runs(first), _merge_runs(rest)
    node = _Node(heading, [], _GRAY, [name], first + rest, row_from=len(first) if rest else len(first + rest),
                 chain=calc.chain)
    stacked, row = node.children[:node.row_from], node.children[node.row_from:]
    row_w = sum(c.w for c in row) + _INNER_GAP * max(len(row) - 1, 0)
    node.w = 2 * _PAD + max([c.w for c in stacked] + [row_w])
    node.h = (_HEAD + 2 * _PAD + sum(c.h for c in stacked) + _INNER_GAP * max(len(stacked) - 1, 0)
              + ((_INNER_GAP if stacked else 0) + max(c.h for c in row) if row else 0))
    return node


def to_drawio(
    calculations: Sequence[Calculation],
    *,
    title: str | None = "MD Calculation Workflow",
    per_column: int = 3,
    define_labels: Mapping[str, str] | None = None,
    none_label: str = "none",
    show_names: bool = False,
    collapse_repeats: bool = True,
) -> str:
    """
    A draw.io file (uncompressed ``<mxfile>`` XML) drawing the pipeline as a flowchart.

    Cards: energy minimization blue, MD yellow, AWH/BAR purple, solvation green, file steps gray. They run
    top to bottom in columns of ``per_column`` and are joined by arrows. A ``Nested`` step is a gray frame
    holding its children: stacked top to bottom (solid arrows with ``chain``, dashed when they only run in
    order), and with ``parallel`` > 1 the children after ``sequential_first`` side by side. Open the result
    in draw.io / diagrams.net to edit it.

    Args:
        calculations: the pipeline steps, as given to ``build_plan``.
        title: text above the chart (None: no title).
        per_column: top-level boxes per column before wrapping to the next column.
        define_labels: readable names for ``define`` flags,
            e.g. ``{"INTERMOLECULAR": "Intermolecular Interaction"}``.
        none_label: shown when a step has no defines (e.g. ``"なし"``).
        show_names: append each step's ``calculation_name`` (sub-folder name inside a Nested step)
            to its heading.
        collapse_repeats: draw consecutive children of a Nested step that would look the same
            (e.g. umbrella windows ``w00`` … ``w19``) as one box marked ``×N``.
    """
    if per_column < 1:
        raise ValueError("per_column must be >= 1")
    labels = dict(define_labels or {})
    top = _MARGIN + (50 if title else 0)

    mxfile = ET.Element("mxfile", host="gmx_harness")
    diagram = ET.SubElement(mxfile, "diagram", id="workflow", name="Workflow")
    model = ET.SubElement(diagram, "mxGraphModel", grid="1", gridSize="10", guides="1", tooltips="1", connect="1",
                          arrows="1", fold="1", page="1", pageScale="1", math="0", shadow="0")
    root = ET.SubElement(model, "root")

    def cell(**attrs: str) -> ET.Element:
        return ET.SubElement(root, "mxCell", attrs)

    cell(id="0")
    cell(id="1", parent="0")

    def geometry(cell: ET.Element, x: float, y: float, w: float, h: float) -> None:
        ET.SubElement(cell, "mxGeometry", x=f"{x:g}", y=f"{y:g}", width=f"{w:g}", height=f"{h:g}").set("as", "geometry")

    def arrow(eid: str, parent: str, source: str, target: str, dashed: bool = False) -> None:
        edge = cell(
            id=eid, edge="1", parent=parent, source=source, target=target,
            style=("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;exitX=0.5;exitY=1;exitDx=0;"
                   "exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" + ("dashed=1;" if dashed else "")),
        )
        ET.SubElement(edge, "mxGeometry", relative="1").set("as", "geometry")

    def draw(node: _Node, cid: str, parent: str, x: float, y: float) -> None:
        fill, stroke = node.colors
        if not node.children:
            value = escape(node.title(show_names)) + "<hr>" + "<br>".join(escape(line) for line in node.lines)
            box = cell(
                id=cid, value=value, vertex="1", parent=parent,
                style=(f"rounded=1;whiteSpace=wrap;html=1;align=left;verticalAlign=top;spacingLeft=10;"
                       f"spacingRight=10;fontSize=14;fillColor={fill};strokeColor={stroke};"),
            )
            geometry(box, x, y, node.w, node.h)
            return
        frame = cell(
            id=cid, value=escape(node.title(show_names)), vertex="1", parent=parent,
            style=(f"swimlane;container=1;collapsible=0;rounded=1;html=1;whiteSpace=wrap;align=left;spacingLeft=10;"
                   f"startSize={_HEAD};fontSize=14;fontStyle=1;fillColor={fill};strokeColor={stroke};"
                   f"swimlaneFillColor=none;"),
        )
        geometry(frame, x, y, node.w, node.h)
        cy: float = _HEAD + _PAD
        stacked, row = node.children[:node.row_from], node.children[node.row_from:]
        for j, child in enumerate(stacked):
            draw(child, f"{cid}_{j}", cid, _PAD, cy)
            if j:
                arrow(f"{cid}_edge{j}", cid, f"{cid}_{j - 1}", f"{cid}_{j}", dashed=not node.chain)
            cy += child.h + _INNER_GAP
        cx: float = _PAD
        for j, child in enumerate(row, start=len(stacked)):
            draw(child, f"{cid}_{j}", cid, cx, cy)
            if stacked:
                arrow(f"{cid}_edge{j}", cid, f"{cid}_{len(stacked) - 1}", f"{cid}_{j}", dashed=True)
            cx += child.w + _INNER_GAP

    if title:
        title_cell = cell(id="title", value=escape(title), vertex="1", parent="1",
                          style="text;html=1;fontSize=20;fontStyle=1;align=left;verticalAlign=middle;")
        geometry(title_cell, _MARGIN, _MARGIN, 400, 40)

    nodes = [_node(calc, calc.name, labels, none_label, collapse_repeats) for calc in calculations]
    x: float = _MARGIN
    for start in range(0, len(nodes), per_column):
        column = nodes[start:start + per_column]
        y: float = top
        for i, node in enumerate(column, start=start):
            draw(node, f"step{i}", "1", x, y)
            if i:
                arrow(f"edge{i}", "1", f"step{i - 1}", f"step{i}")
            y += node.h + _GAP_Y
        x += max(n.w for n in column) + _GAP_X
    ET.indent(mxfile)
    return ET.tostring(mxfile, encoding="unicode") + "\n"


def save_drawio(
    calculations: Sequence[Calculation],
    filepath: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    title: str | None = "MD Calculation Workflow",
    per_column: int = 3,
    define_labels: Mapping[str, str] | None = None,
    none_label: str = "none",
    show_names: bool = False,
    collapse_repeats: bool = True,
) -> None:
    """Write ``to_drawio(...)`` to ``filepath`` (``.drawio``). An existing file is kept unless ``overwrite=True``."""
    text = to_drawio(calculations, title=title, per_column=per_column, define_labels=define_labels,
                     none_label=none_label, show_names=show_names, collapse_repeats=collapse_repeats)
    with open(filepath, "w" if overwrite else "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
