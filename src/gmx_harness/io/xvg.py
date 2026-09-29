"""Read GROMACS .xvg files (output of gmx energy, rms, hbond, ...)."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class XvgData:
    """
    Parsed .xvg data.

    Attributes:
        data: all numeric columns, shape (rows, columns). Column 0 is x.
        title, xlabel, ylabel: from the ``@`` header lines.
        legends: ``s0``, ``s1``, ... legends (one per y column, may be empty).
    """

    data: npt.NDArray[np.float64]
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    legends: list[str] = field(default_factory=list)

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self.data[:, 0]

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """The first y column."""
        return self.data[:, 1]

    def column(self, i: int) -> npt.NDArray[np.float64]:
        """y column ``i`` (0-based over the y columns, i.e. data column i+1)."""
        return self.data[:, i + 1]


def _quoted(line: str) -> str:
    parts = line.split('"')
    return parts[1] if len(parts) >= 2 else ""


def parse_xvg(text: str) -> XvgData:
    """Parse .xvg text. ``#`` comments are skipped, ``@`` lines give labels."""
    rows: list[list[float]] = []
    title = xlabel = ylabel = ""
    legends: dict[int, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("@"):
            body = s[1:].strip()
            if body.startswith("title"):
                title = _quoted(s)
            elif body.startswith("xaxis") and "label" in body:
                xlabel = _quoted(s)
            elif body.startswith("yaxis") and "label" in body:
                ylabel = _quoted(s)
            elif body.startswith("s") and " legend " in f" {body} ":
                head = body.split()[0][1:]
                if head.isdigit():
                    legends[int(head)] = _quoted(s)
            continue
        if s.startswith("&"):  # end of a data set in multi-set files
            continue
        rows.append([float(v) for v in s.split()])
    if not rows:
        raise ValueError("no numeric data in xvg")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("rows have different numbers of columns")
    ncols = width - 1
    return XvgData(np.array(rows, dtype=float), title, xlabel, ylabel, [legends.get(i, "") for i in range(ncols)])


def load_xvg(path: str) -> XvgData:
    """Read an .xvg file. (mylibs' ``load_xvgdata`` dropped the first data row; this does not.)"""
    with open(path, "r") as f:
        return parse_xvg(f.read())
