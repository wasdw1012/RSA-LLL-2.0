#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA direct attack (deterministic, no heuristics).

This module implements the **Wiener small-d attack**:

Given an RSA public key (n, e), if the private exponent d is unusually small,
then the fraction k/d appears among the convergents of the continued fraction
expansion of e/n (Wiener 1990). For each convergent (k, d), we can derive a
candidate φ(n) from:

    e*d - 1 = k*φ(n)

Then solve for p and q via:

    φ(n) = (p-1)(q-1) = n - (p+q) + 1  =>  p+q = n - φ(n) + 1

and check whether the resulting quadratic has integer roots.

Engineering red-lines enforced:
  - No interactive prompts.
  - No silent fallback: if the key is not Wiener-vulnerable, we return a
    structured failure (or exit non-zero in CLI).
  - No "magic numbers": only theorem-defined constants and exact arithmetic.
  - Errors are explicit and stop execution.
  - Health logs (stage timings + diagnostics).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


class RSADirectAttackError(RuntimeError):
    """Hard failure (deployment / invariants / invalid input)."""


@dataclass(frozen=True)
class WienerCandidate:
    k: int
    d: int


@dataclass
class AttackDiagnostics:
    n_bits: int
    e_bits: int
    convergents_tested: int = 0
    candidates_divisible: int = 0
    candidates_square_discriminant: int = 0
    elapsed_ms: float = 0.0
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class AttackResult:
    success: bool
    n: int
    e: int
    d: Optional[int] = None
    p: Optional[int] = None
    q: Optional[int] = None
    phi: Optional[int] = None
    candidate: Optional[WienerCandidate] = None
    diagnostics: Optional[AttackDiagnostics] = None
    error: Optional[str] = None


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended gcd: returns (g, x, y) s.t. a*x + b*y = g."""
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


def modinv(a: int, m: int) -> int:
    """Modular inverse a^{-1} mod m, fails hard if gcd(a,m) != 1."""
    if m <= 0:
        raise RSADirectAttackError(f"modulus must be positive, got m={m}")
    g, x, _y = _egcd(a % m, m)
    if g != 1:
        raise RSADirectAttackError(f"no modular inverse: gcd({a}, {m}) = {g} != 1")
    return x % m


def continued_fraction(numer: int, denom: int) -> List[int]:
    """Exact continued fraction expansion of numer/denom."""
    if denom == 0:
        raise RSADirectAttackError("continued_fraction: denom=0")
    if numer < 0 or denom < 0:
        raise RSADirectAttackError("continued_fraction expects non-negative inputs")
    a: List[int] = []
    while denom:
        q = numer // denom
        a.append(q)
        numer, denom = denom, numer - q * denom
    return a


def convergents_from_cf(cf: List[int]) -> Iterator[WienerCandidate]:
    """
    Yield convergents k/d for a continued fraction (as integers).

    Standard recurrence:
      p[-2]=0,p[-1]=1 ; q[-2]=1,q[-1]=0
      p[i]=a[i]*p[i-1]+p[i-2]
      q[i]=a[i]*q[i-1]+q[i-2]
    """
    p_nm2, p_nm1 = 0, 1
    q_nm2, q_nm1 = 1, 0
    for a_i in cf:
        p_i = a_i * p_nm1 + p_nm2
        q_i = a_i * q_nm1 + q_nm2
        p_nm2, p_nm1 = p_nm1, p_i
        q_nm2, q_nm1 = q_nm1, q_i
        yield WienerCandidate(k=p_i, d=q_i)


def _is_perfect_square(x: int) -> Tuple[bool, int]:
    if x < 0:
        return False, 0
    r = math.isqrt(x)
    return (r * r == x), r


def wiener_attack(n: int, e: int, *, verbose: bool = True) -> AttackResult:
    """
    Deterministic Wiener small-d attack.

    Returns:
      AttackResult with p,q,d if vulnerable; otherwise success=False with error.
    """
    t0 = time.perf_counter()

    if n <= 0:
        raise RSADirectAttackError(f"invalid n (must be positive), got n={n}")
    if e <= 1:
        raise RSADirectAttackError(f"invalid e (must be >1), got e={e}")
    if math.gcd(e, n) != 1:
        raise RSADirectAttackError("invalid public key: gcd(e, n) != 1")

    diag = AttackDiagnostics(n_bits=n.bit_length(), e_bits=e.bit_length())

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("=" * 70)
    log("RSA direct attack (Wiener small-d) — deterministic")
    log(f"[Input] n_bits={diag.n_bits}, e={e} (e_bits={diag.e_bits})")
    log("=" * 70)

    # Wiener precondition (informational, not a gating heuristic):
    # A sufficient condition from the classic theorem is:
    #   d < (1/3) * n^{1/4}
    # We do NOT assume it; we simply attempt the exact convergent check.
    diag.notes["theorem"] = "Wiener 1990: small d implies d appears in convergents of e/n"

    cf = continued_fraction(e, n)
    log(f"[Stage1] continued_fraction length={len(cf)}")

    # Enumerate convergents and validate candidates exactly.
    for cand in convergents_from_cf(cf):
        diag.convergents_tested += 1
        k, d = cand.k, cand.d
        if k == 0 or d == 0:
            continue

        ed_minus_1 = e * d - 1
        if ed_minus_1 % k != 0:
            continue
        diag.candidates_divisible += 1
        phi = ed_minus_1 // k
        if phi <= 0:
            continue

        # Solve x^2 - s*x + n = 0 where s = p+q = n - phi + 1
        s = n - phi + 1
        disc = s * s - 4 * n
        is_sq, sqrt_disc = _is_perfect_square(disc)
        if not is_sq:
            continue
        diag.candidates_square_discriminant += 1

        # Roots must be integers.
        if (s + sqrt_disc) % 2 != 0:
            continue
        p = (s + sqrt_disc) // 2
        q = (s - sqrt_disc) // 2
        if p <= 1 or q <= 1:
            continue
        if p * q != n:
            continue

        # Canonicalize ordering for stable output
        if p < q:
            p, q = q, p

        phi_exact = (p - 1) * (q - 1)
        # d must be the modular inverse of e mod phi(n)
        d_exact = modinv(e, phi_exact)
        if d != d_exact:
            # This should not happen if the derivation is consistent, but we refuse silently
            raise RSADirectAttackError("inconsistent candidate: derived d != invmod(e, phi(n))")

        diag.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log("[Stage2] candidate validated (p,q recovered)")
        log(f"[OK] p_bits={p.bit_length()}, q_bits={q.bit_length()}, d_bits={d.bit_length()}")
        log(f"[OK] tested_convergents={diag.convergents_tested}")
        log("=" * 70)

        return AttackResult(
            success=True,
            n=n,
            e=e,
            d=d,
            p=p,
            q=q,
            phi=phi_exact,
            candidate=cand,
            diagnostics=diag,
            error=None,
        )

    diag.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log("[FAIL] not Wiener-vulnerable (no convergent produced valid p,q)")
    log(f"[Diag] tested_convergents={diag.convergents_tested} | "
        f"divisible={diag.candidates_divisible} | "
        f"square_discriminant={diag.candidates_square_discriminant} | "
        f"time={diag.elapsed_ms:.1f}ms")
    log("=" * 70)

    return AttackResult(
        success=False,
        n=n,
        e=e,
        d=None,
        p=None,
        q=None,
        phi=None,
        candidate=None,
        diagnostics=diag,
        error="key does not satisfy Wiener small-d condition (attack found no valid convergent)",
    )


def _parse_int_auto(s: str) -> int:
    s2 = s.strip().lower()
    if s2.startswith("0x"):
        return int(s2, 16)
    # allow raw hex without 0x
    if all(c in "0123456789abcdef" for c in s2) and len(s2) > 0:
        return int(s2, 16)
    return int(s2, 10)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RSA direct attack (Wiener small-d, deterministic)")
    parser.add_argument("--n", required=True, help="RSA modulus n (hex with/without 0x, or decimal)")
    parser.add_argument("--e", type=int, default=65537, help="public exponent e (default: 65537)")
    parser.add_argument("--quiet", action="store_true", help="suppress logs")
    args = parser.parse_args(argv)

    try:
        n = _parse_int_auto(args.n)
        e = int(args.e)
        res = wiener_attack(n, e, verbose=(not args.quiet))
        if res.success:
            # machine-friendly one-liner (stable, no truncation)
            print(f"[RESULT] success=1 d={res.d} p={res.p} q={res.q}")
            return 0
        print(f"[RESULT] success=0 error={res.error}")
        return 2
    except RSADirectAttackError as ex:
        print(f"[FATAL] {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

