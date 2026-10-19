"""Rigid-body transforms: the only way the structure-building API moves molecules.

A ``RigidTransform`` is ``x -> R @ x + t`` with ``R`` a proper rotation (det = +1),
so applying one never changes a bond length, an angle or the handedness of a molecule.
It is immutable; combine transforms with ``@`` (``a @ b`` applies ``b`` first):

    T = RigidTransform.translation([0.0, 0.0, 0.35]) @ RigidTransform.rotation("z", 10, degrees=True)
    moved = mono.transformed(T)          # a copy; ``mono`` is unchanged

Coordinates are never written directly: positions are read with
``GroFile.positions()`` (a read-only copy) to *measure* (``gmx_harness.geometry``) or
to *fit* a transform (``RigidTransform.fit``), and structures change only through
``transformed`` / ``Assembly.replicate`` / ``Assembly.repeat``.
"""

from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]
Axis = Literal["x", "y", "z"]

_TOL = 1e-6


def _frozen(a: npt.ArrayLike, shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    out = np.array(a, dtype=float)
    if out.shape != shape:
        raise ValueError(f"expected shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("transform contains a non-finite value")
    out.flags.writeable = False
    return out


def rotation_matrix(axis: Axis, angle: float, degrees: bool = False) -> Matrix:
    """3x3 matrix of a right-handed rotation by ``angle`` about ``axis``."""
    a = np.deg2rad(angle) if degrees else float(angle)
    c, s = np.cos(a), np.sin(a)
    match axis.lower():
        case "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        case "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        case "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"axis must be x, y or z, got {axis!r}")


def align_matrix(vector: npt.ArrayLike, target: npt.ArrayLike) -> Matrix:
    """Smallest rotation that turns the direction of ``vector`` into that of ``target``."""
    v = np.asarray(vector, dtype=float)
    t = np.asarray(target, dtype=float)
    v, t = v / np.linalg.norm(v), t / np.linalg.norm(t)
    axis = np.cross(v, t)
    s, c = np.linalg.norm(axis), float(np.dot(v, t))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: turn 180 degrees about any axis perpendicular to v
        perp = np.cross(v, [1.0, 0.0, 0.0] if abs(v[0]) < 0.9 else [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    result: Matrix = np.eye(3) + s * kx + (1 - c) * (kx @ kx)
    return result


class RigidTransform:
    """
    ``x -> R @ x + t`` with ``R`` a proper rotation. Immutable; ``a @ b`` applies ``b`` first.

    Build one with ``identity``, ``rotation``, ``translation``, ``align``, ``frame`` or
    ``fit``; ``from_matrix`` refuses anything that is not a proper rotation (a mirror image
    or a scaling), so a transform can never turn a molecule into its enantiomer.
    """

    __slots__ = ("_r", "_t")
    _r: Matrix
    _t: Vector

    def __init__(self, r: npt.ArrayLike, t: npt.ArrayLike) -> None:
        r_ = _frozen(r, (3, 3))
        if not np.allclose(r_ @ r_.T, np.eye(3), atol=_TOL) or abs(np.linalg.det(r_) - 1.0) > _TOL:
            raise ValueError("not a proper rotation (R @ R.T != 1 or det(R) != +1): mirror images and "
                             "scalings are not rigid transforms")
        object.__setattr__(self, "_r", r_)
        object.__setattr__(self, "_t", _frozen(t, (3,)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RigidTransform is immutable")

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_matrix(cls, r: npt.ArrayLike, t: npt.ArrayLike = (0.0, 0.0, 0.0)) -> "RigidTransform":
        """From a 3x3 rotation matrix and a shift (nm). Raises ValueError unless ``r`` is a proper rotation."""
        return cls(r, t)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def translation(cls, vector: npt.ArrayLike) -> "RigidTransform":
        """Shift by ``vector`` (nm)."""
        return cls(np.eye(3), vector)

    @classmethod
    def rotation(cls, axis: Axis, angle: float, degrees: bool = False,
                 center: npt.ArrayLike = (0.0, 0.0, 0.0)) -> "RigidTransform":
        """Right-handed rotation by ``angle`` about the line through ``center`` along ``axis``."""
        r = rotation_matrix(axis, angle, degrees)
        c = np.asarray(center, dtype=float)
        return cls(r, c - r @ c)

    @classmethod
    def align(cls, vector: npt.ArrayLike, target: npt.ArrayLike) -> "RigidTransform":
        """Smallest rotation about the origin turning the direction of ``vector`` into that of ``target``."""
        return cls(align_matrix(vector, target), np.zeros(3))

    @classmethod
    def frame(cls, x_dir: npt.ArrayLike, z_dir: npt.ArrayLike) -> "RigidTransform":
        """
        Rotation about the origin that turns ``x_dir`` onto +x and ``z_dir`` into the xz plane
        on the +z side (its part perpendicular to ``x_dir`` onto +z).
        """
        ex = np.asarray(x_dir, dtype=float)
        ex = ex / np.linalg.norm(ex)
        ez = np.asarray(z_dir, dtype=float)
        ez = ez - np.dot(ez, ex) * ex
        n = np.linalg.norm(ez)
        if n < 1e-9:
            raise ValueError("z_dir is parallel to x_dir")
        ez /= n
        return cls(np.array([ex, np.cross(ez, ex), ez]), np.zeros(3))

    @classmethod
    def fit(cls, mobile: npt.ArrayLike, target: npt.ArrayLike) -> "RigidTransform":
        """
        Proper rotation + shift that brings the points ``mobile`` closest to ``target``
        (least squares, Kabsch). Both are (N, 3), N >= 3, in the same order. The result never
        mirrors: if the best fit would need a reflection, the best proper rotation is returned.
        """
        src = np.asarray(mobile, dtype=float)
        dst = np.asarray(target, dtype=float)
        if src.ndim != 2 or src.shape[1] != 3 or src.shape != dst.shape:
            raise ValueError(f"mobile and target must both be (N, 3), got {src.shape} and {dst.shape}")
        if len(src) < 3:
            raise ValueError("fit needs at least 3 points")
        sc, dc = src.mean(0), dst.mean(0)
        u, _, vt = np.linalg.svd((src - sc).T @ (dst - dc))
        r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
        return cls(r, dc - r @ sc)

    # ----------------------------------------------------------- use

    @property
    def matrix(self) -> Matrix:
        """The rotation matrix ``R`` (read-only)."""
        return self._r

    @property
    def shift(self) -> Vector:
        """The shift ``t`` (nm, read-only)."""
        return self._t

    @property
    def angle(self) -> float:
        """Rotation angle of ``R`` in degrees (0..180)."""
        return float(np.degrees(np.arccos(np.clip((np.trace(self._r) - 1) / 2, -1, 1))))

    def apply(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """New array of the transformed points ((3,) or (N, 3)); the input is not changed."""
        p = np.asarray(points, dtype=float)
        out: npt.NDArray[np.float64] = p @ self._r.T + self._t
        return out

    def inverse(self) -> "RigidTransform":
        return RigidTransform(self._r.T, -(self._r.T @ self._t))

    def __matmul__(self, other: "RigidTransform") -> "RigidTransform":
        if not isinstance(other, RigidTransform):
            return NotImplemented
        return RigidTransform(self._r @ other._r, self._r @ other._t + self._t)

    def power(self, n: int) -> "RigidTransform":
        """This transform applied ``n`` times (``n`` >= 0)."""
        if n < 0:
            raise ValueError("n must be >= 0")
        out = RigidTransform.identity()
        for _ in range(n):
            out = self @ out
        return out

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, RigidTransform) and np.array_equal(self._r, other._r)
                and np.array_equal(self._t, other._t))

    def __hash__(self) -> int:
        return hash((self._r.tobytes(), self._t.tobytes()))

    def __repr__(self) -> str:
        return f"RigidTransform(angle={self.angle:.3f} deg, shift={self._t.tolist()})"


def chain(transforms: Iterable[RigidTransform]) -> RigidTransform:
    """Composition of ``transforms`` applied in the given order (first one first)."""
    out = RigidTransform.identity()
    for t in transforms:
        out = t @ out
    return out
