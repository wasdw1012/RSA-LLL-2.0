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
    signals: Optional[Dict[str, object]] = None
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


def mvp_lll_wiener_attack(
    n: int,
    e: int,
    *,
    prime: int,
    precision: int,
    noise_tolerance: float = 0.0,
    anchor_k: Optional[int] = None,
    # MVP16 LogShell signal parameters (explicit by redline)
    logshell_faltings_height: "object" = 0,
    logshell_conductor: Optional[int] = None,
    logshell_arakelov_height: Optional[int] = None,
    verbose: bool = True,
) -> AttackResult:
    """
    MVP-style pipeline wrapper around the same *strict* Wiener verification.

    What this does (deterministically):
      - Stage A: build the canonical 2D approximation lattice for e/n:
            L = span{ (n,0), (e,1) }
        Any lattice vector is (u*n + v*e, v), giving a rational approximation
        -u/v ≈ e/n when |u*n + v*e| is small.
      - Stage B: reduce the basis using MVP19 LLL backend via `rsa_lll.lll_reduce_enhanced`
        with Witt truncation controlled by (prime, precision).
      - Stage C: extract candidate (k,d) pairs from reduced rows where u is integral
        and validate them with the exact RSA algebra (same validator as Wiener).

    Important:
      - This does NOT assume "k can be any anchor". We only accept a candidate if
        it yields integer (p,q) such that p*q=n. If nothing validates -> hard fail.
    """
    if n <= 0:
        raise RSADirectAttackError(f"invalid n (must be positive), got n={n}")
    if e <= 1:
        raise RSADirectAttackError(f"invalid e (must be >1), got e={e}")
    if math.gcd(e, n) != 1:
        raise RSADirectAttackError("invalid public key: gcd(e, n) != 1")

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("=" * 70)
    log("RSA direct attack — MVP19 LLL assisted Wiener (deterministic)")
    log(f"[Input] n_bits={n.bit_length()}, e={e}, p={prime}, witt_k={precision}, noise_tol={noise_tolerance}")
    if anchor_k is not None:
        if int(anchor_k) == 0:
            raise RSADirectAttackError("anchor_k must be non-zero (MSB cannot be empty)")
        log(f"[Input] anchor_k={int(anchor_k)} (explicit; no auto-injection)")
    log("=" * 70)

    # Stage A: lattice construction (canonical 2D)
    basis = [[int(n), 0], [int(e), 1]]

    # Stage B: MVP19 LLL reduction + Witt truncation
    try:
        from rsa_lll import lll_reduce_enhanced
    except Exception as ex:
        raise RSADirectAttackError(f"failed to import rsa_lll: {ex}") from ex

    lll_res = lll_reduce_enhanced(
        basis,
        prime=int(prime),
        precision=int(precision),
        noise_tolerance=float(noise_tolerance),
        apply_witt_truncation=True,
    )
    if not lll_res.success:
        return AttackResult(
            success=False,
            n=n,
            e=e,
            error=f"LLL reduction failed: {lll_res.error}",
            diagnostics=AttackDiagnostics(
                n_bits=n.bit_length(),
                e_bits=e.bit_length(),
                convergents_tested=0,
                candidates_divisible=0,
                candidates_square_discriminant=0,
                elapsed_ms=float(lll_res.total_elapsed_ms),
                notes={"lll_error": str(lll_res.error), "lll_diag": str(lll_res.diagnostics)},
            ),
        )

    reduced = lll_res.reduced_basis
    log(f"[StageB] LLL ok | reduced_rows={len(reduced)} | {lll_res.total_elapsed_ms:.1f}ms")
    log(f"[StageB] LLL diag: {lll_res.diagnostics}")

    # Stage C: derive (k,d) candidates from reduced basis rows.
    # Each row is (x, v) with x = u*n + v*e. If (x - v*e) divisible by n then u integral.
    # We'll collect either (k,d) pairs or just d values (if anchor_k is provided).
    candidates: List[WienerCandidate] = []
    d_only: List[int] = []
    for row in reduced:
        if len(row) != 2:
            continue
        x, v = int(row[0]), int(row[1])
        if v == 0:
            continue
        num = x - v * e
        if num % n != 0:
            continue
        u = num // n
        k = abs(int(-u))
        d = abs(int(v))
        if k == 0 or d == 0:
            continue
        if anchor_k is None:
            candidates.append(WienerCandidate(k=k, d=d))
        else:
            d_only.append(int(d))

    # Also include the full convergent list (deterministic and complete).
    # This ensures we don't "depend on luck" of a single reduced row.
    cf = continued_fraction(e, n)
    if anchor_k is None:
        candidates.extend(list(convergents_from_cf(cf)))
    else:
        for cand in convergents_from_cf(cf):
            if cand.d != 0:
                d_only.append(int(cand.d))

    if anchor_k is None:
        # Deduplicate pairs while preserving order
        seen = set()
        uniq: List[WienerCandidate] = []
        for c in candidates:
            key = (c.k, c.d)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        log(f"[StageC] candidate_pairs={len(uniq)} (lll_rows+convergents)")
    else:
        # Deduplicate d list while preserving order
        seen_d = set()
        uniq_d: List[int] = []
        for d in d_only:
            if d in seen_d:
                continue
            seen_d.add(d)
            uniq_d.append(d)
        log(f"[StageC] candidate_d_values={len(uniq_d)} (lll_rows+convergents) with fixed k={int(anchor_k)}")

    # Validate candidates exactly (same algebra as wiener_attack)
    t0 = time.perf_counter()
    diag = AttackDiagnostics(n_bits=n.bit_length(), e_bits=e.bit_length())
    diag.notes["mvp_lll"] = "used MVP19 LLL on 2D approximation lattice"
    diag.notes["lll_diag"] = str(lll_res.diagnostics)

    if anchor_k is None:
        iterable_pairs = uniq
    else:
        iterable_pairs = [WienerCandidate(k=int(anchor_k), d=int(dv)) for dv in uniq_d]

    for cand in iterable_pairs:
        diag.convergents_tested += 1
        k, d = int(cand.k), int(cand.d)
        if k == 0 or d == 0:
            continue
        ed_minus_1 = e * d - 1
        if ed_minus_1 % k != 0:
            continue
        diag.candidates_divisible += 1
        phi = ed_minus_1 // k
        if phi <= 0:
            continue
        s = n - phi + 1
        disc = s * s - 4 * n
        is_sq, sqrt_disc = _is_perfect_square(disc)
        if not is_sq:
            continue
        diag.candidates_square_discriminant += 1
        if (s + sqrt_disc) % 2 != 0:
            continue
        p = (s + sqrt_disc) // 2
        q = (s - sqrt_disc) // 2
        if p <= 1 or q <= 1:
            continue
        if p * q != n:
            continue
        if p < q:
            p, q = q, p
        phi_exact = (p - 1) * (q - 1)
        d_exact = modinv(e, phi_exact)
        if d_exact != d:
            raise RSADirectAttackError("inconsistent candidate: d != invmod(e, phi(n))")

        diag.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log("[OK] candidate validated (p,q,d recovered)")
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
            signals={"mode": "STRICT_RSA"},
            error=None,
        )

    diag.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log("[FAIL] no candidate validated under strict algebraic checks")
    log(f"[Diag] tested={diag.convergents_tested} divisible={diag.candidates_divisible} square_disc={diag.candidates_square_discriminant}")

    # Emit MVP16-style signals for the first non-zero candidate (for observability),
    # even if strict RSA validation fails.
    sig: Optional[Dict[str, object]] = None
    first = None
    for cand in iterable_pairs:
        if int(cand.k) != 0 and int(cand.d) != 0:
            first = cand
            break
    if first is not None:
        try:
            sig = rsa_logshell_signals(
                n=n,
                e=e,
                k=int(first.k),
                d=int(first.d),
                prime=int(prime),
                precision=int(precision),
                arakelov_height=logshell_arakelov_height,
                faltings_height=logshell_faltings_height,
                conductor=logshell_conductor,
                curvature=None,
            )
        except Exception as ex:
            sig = {"mode": "LOGSHELL_SIGNALS_ONLY", "error": str(ex)}

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
        signals=sig,
        error="MVP-LLL-assisted Wiener validation found no (k,d) yielding integer factorization",
    )


def _parse_int_auto(s: str) -> int:
    s2 = s.strip().lower()
    if s2.startswith("0x"):
        return int(s2, 16)
    # allow raw hex without 0x
    if all(c in "0123456789abcdef" for c in s2) and len(s2) > 0:
        return int(s2, 16)
    return int(s2, 10)


def rsa_logshell_signals(
    *,
    n: int,
    e: int,
    k: int,
    d: int,
    prime: int,
    precision: int,
    arakelov_height: Optional[int] = None,
    faltings_height: object = None,
    conductor: Optional[int] = None,
    curvature: Optional[int] = None,
) -> Dict[str, object]:
    """
    MVP16-style Log-Shell/Kummer signal output for an RSA candidate (k,d).

    This mirrors the ABC smoke's split semantics: `logshell` may be True while
    `kummer` is False. It does NOT claim the RSA private key is recovered.

    Mapping:
      center := k * n
      target := e*d - 1
    """
    if k == 0 or d == 0:
        raise RSADirectAttackError("k and d must be non-zero (MSB cannot be empty)")
    if faltings_height is None:
        raise RSADirectAttackError("faltings_height is required (pass 0 explicitly if desired)")

    try:
        from fractions import Fraction
        from frobenioid_base import LogShell, PrimeSpec
    except Exception as ex:
        raise RSADirectAttackError(f"failed to import MVP16 LogShell/PrimeSpec: {ex}") from ex

    prime_spec = PrimeSpec(p=int(prime), k=int(precision))
    hA = int(arakelov_height) if arakelov_height is not None else int(max(0, n.bit_length()))
    N_cond = int(conductor) if conductor is not None else int(prime_spec.modulus)

    shell = LogShell(
        prime_spec=prime_spec,
        arakelov_height=hA,
        faltings_height=Fraction(faltings_height),
        conductor=N_cond,
        epsilon_scheduler=None,
    )

    center = Fraction(int(k) * int(n))
    target = Fraction(int(e) * int(d) - 1)

    context = None
    if curvature is not None:
        context = {"curvature": int(curvature)}

    eps_eff, eps_sched = shell.epsilon_effective_with_certificate(center=center, context=context)
    vol_min, vol_max = shell.volume_interval(center, context=context)
    logshell_ok = (vol_min <= target <= vol_max)

    kummer_cert = shell.kummer_equivalence_certificate(
        source_value=int(center),
        target_value=int(target),
        include_float_approx=False,
        context=context,
    )

    return {
        "mode": "LOGSHELL_SIGNALS_ONLY",
        "center": str(center),
        "target": str(target),
        "epsilon_base": kummer_cert.get("epsilon_base_exact"),
        "epsilon_effective": kummer_cert.get("epsilon_effective_exact"),
        "epsilon_schedule": eps_sched,
        "log_shell_min": str(vol_min),
        "log_shell_max": str(vol_max),
        "logshell": bool(logshell_ok),
        "kummer": bool(kummer_cert.get("kummer_degree") is not None),
        "kummer_degree": kummer_cert.get("kummer_degree"),
        "derivation": {
            "prime": int(prime),
            "precision_k": int(precision),
            "arakelov_height": int(hA),
            "faltings_height": str(Fraction(faltings_height)),
            "conductor": int(N_cond),
            "mapping": "center=k*n, target=e*d-1",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RSA direct attack (Wiener small-d, deterministic)")
    parser.add_argument("--n", required=True, help="RSA modulus n (hex with/without 0x, or decimal)")
    parser.add_argument("--e", type=int, default=65537, help="public exponent e (default: 65537)")
    parser.add_argument("--mvp", action="store_true", help="use MVP19 LLL-assisted pipeline (still strict validation)")
    parser.add_argument("--prime", type=int, help="(MVP) truncation prime p (required with --mvp)")
    parser.add_argument("--precision", type=int, help="(MVP) truncation precision k (required with --mvp)")
    parser.add_argument("--noise", type=float, default=0.0, help="(MVP) noise tolerance (recorded; deterministic backend)")
    parser.add_argument("--anchor-k", type=int, help="(MVP) explicit anchor k (non-zero). No auto-injection.")
    parser.add_argument("--logshell-faltings", type=str, default="0", help="(MVP16) explicit faltings_height (default: 0). Never implicit.")
    parser.add_argument("--logshell-conductor", type=int, help="(MVP16) explicit conductor (default: p^k).")
    parser.add_argument("--logshell-height", type=int, help="(MVP16) explicit arakelov_height (default: n_bits).")
    parser.add_argument("--quiet", action="store_true", help="suppress logs")
    args = parser.parse_args(argv)

    try:
        n = _parse_int_auto(args.n)
        e = int(args.e)
        if args.mvp:
            if args.prime is None or args.precision is None:
                raise RSADirectAttackError("--mvp requires --prime and --precision")
            try:
                from fractions import Fraction
                faltings_h = Fraction(str(args.logshell_faltings).strip())
            except Exception as ex:
                raise RSADirectAttackError(f"invalid --logshell-faltings: {ex}") from ex
            res = mvp_lll_wiener_attack(
                n,
                e,
                prime=int(args.prime),
                precision=int(args.precision),
                noise_tolerance=float(args.noise),
                anchor_k=(int(args.anchor_k) if args.anchor_k is not None else None),
                logshell_faltings_height=faltings_h,
                logshell_conductor=(int(args.logshell_conductor) if args.logshell_conductor is not None else None),
                logshell_arakelov_height=(int(args.logshell_height) if args.logshell_height is not None else None),
                verbose=(not args.quiet),
            )
        else:
            res = wiener_attack(n, e, verbose=(not args.quiet))
        if res.success:
            # machine-friendly one-liner (stable, no truncation)
            print(f"[RESULT] success=1 d={res.d} p={res.p} q={res.q}")
            return 0
        print(f"[RESULT] success=0 error={res.error}")
        if res.signals is not None:
            print(f"[SIGNALS] {res.signals}")
        return 2
    except RSADirectAttackError as ex:
        print(f"[FATAL] {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

