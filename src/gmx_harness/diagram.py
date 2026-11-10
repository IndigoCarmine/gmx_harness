"""Draw a pipeline as a draw.io (diagrams.net) flowchart: one card per step.

The cards are read from what each step would write (``generate()``): steps that
write a ``setting.mdp`` show nsteps, emtol or simulated time and temperature, and
the ``define`` flags, so subclasses and ``additional_mdp_parameters`` are shown as
they will run. Nothing is executed and nothing is written unless ``save_drawio``
is called.
"""

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from html import escape

from .mdp import MDParameters
from .steps import (
    AWH,
    AddFiles,
    BarMethod,
    Calculation,
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


def to_drawio(
    calculations: Sequence[Calculation],
    *,
    title: str | None = "MD Calculation Workflow",
    per_column: int = 3,
    define_labels: Mapping[str, str] | None = None,
    none_label: str = "none",
    show_names: bool = False,
) -> str:
    """
    A draw.io file (uncompressed ``<mxfile>`` XML) drawing the pipeline as a flowchart.

    Cards: energy minimization blue, MD yellow, AWH/BAR purple, solvation green,
    file steps gray. They run top to bottom in columns of ``per_column`` and are
    joined by arrows. Open the result in draw.io / diagrams.net to edit it.

    Args:
        calculations: the pipeline steps, as given to ``build_plan``.
        title: text above the chart (None: no title).
        per_column: cards per column before wrapping to the next column.
        define_labels: readable names for ``define`` flags,
            e.g. ``{"INTERMOLECULAR": "Intermolecular Interaction"}``.
        none_label: shown when a step has no defines (e.g. ``"なし"``).
        show_names: append each step's ``calculation_name`` to its heading.
    """
    if per_column < 1:
        raise ValueError("per_column must be >= 1")
    labels = dict(define_labels or {})
    width, height, gap_x, gap_y, margin = 330, 120, 150, 90, 20
    top = margin + (50 if title else 0)

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
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h)).set("as", "geometry")

    if title:
        title_cell = cell(id="title", value=escape(title), vertex="1", parent="1",
                          style="text;html=1;fontSize=20;fontStyle=1;align=left;verticalAlign=middle;")
        geometry(title_cell, margin, margin, 400, 40)

    for i, calc in enumerate(calculations):
        heading, lines, (fill, stroke) = _card(calc, labels, none_label)
        if show_names:
            heading += f" [{calc.name}]"
        value = escape(heading) + "<hr>" + "<br>".join(escape(line) for line in lines)
        col, row = divmod(i, per_column)
        card = cell(
            id=f"step{i}", value=value, vertex="1", parent="1",
            style=(f"rounded=1;whiteSpace=wrap;html=1;align=left;verticalAlign=top;spacingLeft=10;spacingRight=10;"
                   f"fontSize=14;fillColor={fill};strokeColor={stroke};"),
        )
        geometry(card, margin + col * (width + gap_x), top + row * (height + gap_y), width, height)
        if i:
            edge = cell(
                id=f"edge{i}", edge="1", parent="1", source=f"step{i - 1}", target=f"step{i}",
                style=("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;exitX=0.5;exitY=1;exitDx=0;"
                       "exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"),
            )
            ET.SubElement(edge, "mxGeometry", relative="1").set("as", "geometry")
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
) -> None:
    """Write ``to_drawio(...)`` to ``filepath`` (``.drawio``). An existing file is kept unless ``overwrite=True``."""
    text = to_drawio(calculations, title=title, per_column=per_column, define_labels=define_labels,
                     none_label=none_label, show_names=show_names)
    with open(filepath, "w" if overwrite else "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
