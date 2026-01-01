"""Lightweight algebra backend for finite-field computations.

The goal of this module is to expose a minimal, deterministic API that mimics
the subset of ``sage.all`` used by ``algebra_patch.py``.  When SageMath is
available, the backend simply re-exports the native implementations.  In
lighter environments we fall back to ``galois`` (if installed) or a small set
of pure-Python shims so callers can still construct finite fields, vectors, and
matrices with a consistent interface.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

try:  # Prefer Sage when available for full FunctionField support
    from sage.all import (  # type: ignore
        FunctionField as _sage_FunctionField,
        GF as _sage_GF,
        PolynomialRing as _sage_PolynomialRing,
        matrix as _sage_matrix,
        vector as _sage_vector,
        ZZ as _sage_ZZ,
    )

    HAS_SAGE = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SAGE = False

try:
    import galois  # type: ignore

    HAS_GALOIS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_GALOIS = False

import numpy as _np


# ---------------------------------------------------------------------------
# Helpers for a non-Sage environment
# ---------------------------------------------------------------------------


def _field_zero(F):
    if hasattr(F, "zero") and callable(F.zero):
        return F.zero()
    if hasattr(F, "Zero") and callable(F.Zero):
        return F.Zero()
    try:
        return F(0)
    except Exception:
        return 0


def _field_one(F):
    if hasattr(F, "one") and callable(F.one):
        return F.one()
    if hasattr(F, "One") and callable(F.One):
        return F.One()
    try:
        return F(1)
    except Exception:
        return 1


def parent(x):  # pragma: no cover - thin compatibility shim
    if hasattr(x, "parent"):
        return x.parent()
    if hasattr(x, "field"):
        return x.field
    return None


def frobenius_power(x, q):
    if hasattr(x, "frobenius"):
        try:
            return x.frobenius(q)
        except Exception:
            pass
    return x ** q


class Matrix:
    def __init__(self, data, field=None):
        self.data = _np.array(data, dtype=object)
        self.field = field

    def nrows(self):
        return self.data.shape[0]

    def ncols(self):
        return self.data.shape[1]

    def transpose(self):
        return Matrix(self.data.T, field=self.field)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def _solve_linear_system(self, A, b):
        A = [[A[i, j] for j in range(self.ncols())] for i in range(self.nrows())]
        b = [b[i] for i in range(len(b))]
        n_rows, n_cols = self.nrows(), self.ncols()
        row = 0
        col = 0
        while row < n_rows and col < n_cols:
            pivot = None
            for r in range(row, n_rows):
                if A[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                col += 1
                continue
            if pivot != row:
                A[row], A[pivot] = A[pivot], A[row]
                b[row], b[pivot] = b[pivot], b[row]
            inv = 1 / A[row][col]
            A[row] = [inv * v for v in A[row]]
            b[row] = inv * b[row]
            for r in range(n_rows):
                if r == row:
                    continue
                factor = A[r][col]
                if factor == 0:
                    continue
                A[r] = [A[r][c] - factor * A[row][c] for c in range(n_cols)]
                b[r] = b[r] - factor * b[row]
            row += 1
            col += 1
        solution = [0 for _ in range(n_cols)]
        for r in range(n_rows):
            leading = None
            for c in range(n_cols):
                if A[r][c] != 0:
                    leading = c
                    break
            if leading is None:
                if b[r] != 0:
                    raise ValueError("Linear system is inconsistent")
                continue
            solution[leading] = b[r]
        return solution

    def solve_right(self, b: "Matrix") -> "Matrix":
        if HAS_SAGE:
            return self.data.solve_right(b.data)  # pragma: no cover - passthrough
        if HAS_GALOIS and isinstance(self.data, galois.FieldArray):
            solved = galois.linalg.solve(self.data, b.data)
            return Matrix(solved, field=self.field)
        b_col = [b[i, 0] for i in range(b.nrows())]
        sol = self._solve_linear_system(self.data, b_col)
        return Matrix([[x] for x in sol], field=self.field)

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self.data)

    def __repr__(self):  # pragma: no cover - debug
        return f"Matrix({self.data})"


class Vector:
    def __init__(self, entries, field=None):
        self.data = _np.array(entries, dtype=object)
        self.field = field

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __len__(self):  # pragma: no cover - convenience
        return len(self.data)

    def __repr__(self):  # pragma: no cover - debug
        return f"Vector({self.data})"


def matrix(*args):
    if HAS_SAGE:
        return _sage_matrix(*args)
    if len(args) == 1:
        return Matrix(args[0])
    if len(args) == 3:
        F, r, c = args
        zero = _field_zero(F)
        data = [[zero for _ in range(c)] for _ in range(r)]
        return Matrix(data, field=F)
    raise TypeError("matrix expects data or (field, nrows, ncols)")


def vector(*args):
    if HAS_SAGE:
        return _sage_vector(*args)
    if len(args) == 1:
        return Vector(args[0])
    if len(args) == 2:
        F, entries = args
        coerced = [F(e) if callable(F) else e for e in entries]
        return Vector(coerced, field=F)
    raise TypeError("vector expects entries or (field, entries)")


def PolynomialRing(field, names):  # pragma: no cover - small wrapper
    if HAS_SAGE:
        return _sage_PolynomialRing(field, names=names)
    if HAS_GALOIS:
        return galois.PolynomialRing(field, names=names)
    raise NotImplementedError("PolynomialRing requires SageMath or galois")


def FunctionField(base):  # pragma: no cover - small wrapper
    if HAS_SAGE:
        return _sage_FunctionField(base)
    raise NotImplementedError("Function fields are only supported when SageMath is installed.")


def GF(order, name=None):
    if HAS_SAGE:
        return _sage_GF(order, name=name)
    if HAS_GALOIS:
        return galois.GF(order, name=name)
    raise NotImplementedError("Finite fields require SageMath or galois to be installed.")


ZZ = _sage_ZZ if HAS_SAGE else int


def supports_function_fields():  # pragma: no cover - trivial
    return HAS_SAGE

