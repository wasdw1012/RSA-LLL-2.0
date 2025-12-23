#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP-backed lattice reduction for RSA experiments.

This file exists because the RSA attack pipeline expects a dedicated module
to expose a strict LLL reduction interface. The backend is the deterministic
LLL implementation shipped in `mvp19_signature_cvp.py` (LatticeBackend.lll_reduce).

Red-lines respected:
  - Deterministic: no randomness, no "try-some-magic" loops.
  - No silent fallback: parameters are either used or explicitly logged.
  - No interactive prompts.
  - Import / invariant errors are hard failures.

Notes on "Witt truncation":
  We interpret the pair (prime p, precision k) as defining the truncation ring
  Z / p^k Z. For an integer lattice basis matrix B, we:
    1) reduce each entry modulo p^k
    2) lift it to the canonical symmetric representative in (-p^k/2, p^k/2]
  This is a deterministic canonical lift; it is not a heuristic threshold.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class RSALLLError(RuntimeError):
    """Hard failure in MVP LLL layer."""


@dataclass
class LLLResult:
    success: bool
    reduced_basis: List[List[int]] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _is_prime(p: int) -> bool:
    # Deterministic for small-ish p; callers use small p by design.
    if p <= 1:
        return False
    if p <= 3:
        return True
    if p % 2 == 0:
        return False
    r = int(math.isqrt(p))
    f = 3
    while f <= r:
        if p % f == 0:
            return False
        f += 2
    return True


def _canonical_mod_lift(x: int, modulus: int) -> int:
    """
    Reduce x modulo modulus, then lift to the canonical symmetric representative.

    Output r satisfies:
      r ≡ x (mod modulus)
      -modulus/2 < r <= modulus/2
    """
    if modulus <= 0:
        raise RSALLLError(f"modulus must be positive, got {modulus}")
    r = x % modulus
    half = modulus // 2
    if r > half:
        r -= modulus
    return int(r)


def _witt_truncate_matrix(matrix: List[List[int]], *, prime: int, precision: int) -> List[List[int]]:
    if precision <= 0:
        raise RSALLLError(f"precision must be positive, got {precision}")
    if not _is_prime(int(prime)):
        raise RSALLLError(f"prime must be a prime integer, got p={prime}")
    modulus = int(prime) ** int(precision)
    return [[_canonical_mod_lift(int(x), modulus) for x in row] for row in matrix]


def lll_reduce_enhanced(
    basis: List[List[int]],
    *,
    prime: int,
    precision: int,
    noise_tolerance: float = 0.0,
    lll_delta: float = 3.0 / 4.0,
    apply_witt_truncation: bool = True,
) -> LLLResult:
    """
    Deterministic LLL reduction with optional Witt truncation.

    Args:
      basis: integer row-basis matrix.
      prime: p for truncation ring Z/p^kZ.
      precision: k for truncation ring Z/p^kZ.
      noise_tolerance: accepted for compatibility; **not used** by deterministic LLL.
                      We record it in diagnostics (explicit, not silent).
      lll_delta: δ parameter in (1/4,1); default is canonical 3/4.
      apply_witt_truncation: whether to apply the deterministic truncation step.
    """
    t0 = time.perf_counter()
    diag: Dict[str, Any] = {
        "prime": int(prime),
        "precision": int(precision),
        "noise_tolerance": float(noise_tolerance),
        "apply_witt_truncation": bool(apply_witt_truncation),
        "lll_delta": float(lll_delta),
    }

    if not basis:
        return LLLResult(success=False, reduced_basis=[], total_elapsed_ms=0.0, diagnostics=diag, error="empty basis")
    m = len(basis[0])
    if m == 0 or any(len(row) != m for row in basis):
        raise RSALLLError("invalid basis: dimension mismatch or zero width")
    if any(not isinstance(x, int) for row in basis for x in row):
        raise RSALLLError("basis must be integers (no floats)")

    # Import backend strictly
    try:
        # MVP19 lattice backend implementation lives here.
        from mvp19_signature_cvp import NativeLatticeBackend
    except Exception as ex:  # pragma: no cover
        raise RSALLLError(f"failed to import MVP19 NativeLatticeBackend: {ex}") from ex

    B = [row[:] for row in basis]
    if apply_witt_truncation:
        B = _witt_truncate_matrix(B, prime=int(prime), precision=int(precision))

    backend = NativeLatticeBackend()

    try:
        reduced = backend.lll_reduce(B, delta=float(lll_delta))
    except Exception as ex:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        diag["elapsed_ms"] = elapsed_ms
        return LLLResult(
            success=False,
            reduced_basis=[],
            total_elapsed_ms=elapsed_ms,
            diagnostics=diag,
            error=str(ex),
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    diag["elapsed_ms"] = elapsed_ms
    return LLLResult(
        success=True,
        reduced_basis=reduced,
        total_elapsed_ms=elapsed_ms,
        diagnostics=diag,
        error=None,
    )

