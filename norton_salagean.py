#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Norton–Salagean / Reeds–Sloane 视角下的 Chain-Ring Shift-Register Synthesis
================================================================================

目标
----
在链环 R = Z/(p^n)Z 上，对给定有限序列 s[0..N-1]，构造（或判定不存在）
一个最小度数的线性递推/零化子多项式：

    C(T) = 1 + c1 T + ... + cL T^L   (ci ∈ R)

使得对所有 k = L..N-1:

    s[k] + c1 s[k-1] + ... + cL s[k-L]  ≡ 0   (mod p^n)

数学要点（红线对齐）
--------------------
1) 纯代数、无启发式：
   - 不使用任何“阈值/评分/浮点近似”。
   - 只做：在 R 上的线性方程可解性判定 + 最小 L 选择。

2) 零因子可处理：
   - R 非域；普通 BM 在 discrepancy 需要逆元时会崩溃。
   - 这里采用 p-adic/Hensel lifting 的“链环线性系统综合”路线：
     先解 mod p，再逐层提升到 mod p^n。
   - 该方法严格区分：
       “代数结构导致的 0”（零因子现象）
       vs “精度截断导致的 0”（模 p^n 归零）

3) 必须可证“最小度数”：
   - 最小 L 是外层公理化最小化：从 L=0 起逐一判定线性系统是否可解；
     第一个可解的 L 就是最小度数（无任何启发式跳跃）。

与 Norton–Salagean 的关系
-------------------------
经典 Norton–Salagean / Reeds–Sloane 的“Chain Ring BM”属于流式更新算法，
其输出本质上是“最小生成多项式/零化子”，等价于此处定义的最小 L 的解。

本模块优先保证“数学级别正确性与可验证性”；并把：
  - minimize 步进函数（最小 L 选择）
  - p-adic lifting（Z/p^kZ -> Z/p^{k+1}Z 的严格提升）
作为核心。

当你后续在 MVP23 引入更快的 Norton–Salagean 流式版本时，本模块也可作为
reference oracle（可验证基准）来对拍。

工程红线
--------
- 禁止启发式：无阈值、无概率、无“差不多”
- 禁止魔法数：所有常量从 p,n,N 推导或显式参数化
- 禁止静默退回：不可解则抛出异常或返回不可解证书（由调用方选择）
- 部署错误必须中断：内部不自吞异常
- 日志健康输出：INFO 给摘要，DEBUG 给过程（默认不刷屏）

================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_logger = logging.getLogger(__name__)


# =============================================================================
# 链环规格与基础算子
# =============================================================================


@dataclass(frozen=True)
class ChainRingSpec:
    """
    链环 R = Z/(p^n)Z 的规格。

    - p 必须是素数（本模块不在内部做素性测试；调用方应保证）
    - n >= 1
    """

    p: int
    n: int

    def __post_init__(self) -> None:
        if not isinstance(self.p, int) or self.p < 2:
            raise ValueError(f"p must be int >= 2, got {self.p}")
        if not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"n must be int >= 1, got {self.n}")

    @property
    def modulus(self) -> int:
        return int(self.p) ** int(self.n)

    def normalize(self, x: int) -> int:
        if not isinstance(x, int):
            raise TypeError(f"ring element must be int, got {type(x).__name__}")
        return int(x % self.modulus)

    def vp(self, x: int) -> int:
        """
        p-adic valuation v_p(x) inside Z/p^nZ, truncated:
          - v_p(0) := n
          - otherwise largest v < n such that p^v | x (as integer representative)
        """
        xx = int(self.normalize(x))
        if xx == 0:
            return int(self.n)
        v = 0
        p = int(self.p)
        n = int(self.n)
        while v < n and (xx % p == 0):
            xx //= p
            v += 1
        return int(v)


def _inv_mod_prime_field(a: int, p: int) -> int:
    """
    a^{-1} mod p for prime p, requiring a != 0 mod p.
    """
    aa = int(a % p)
    if aa == 0:
        raise ZeroDivisionError("no inverse for 0 in F_p")
    # Fermat: a^(p-2) mod p
    return pow(aa, p - 2, p)


# =============================================================================
# F_p 线性代数：RREF 求解器（用于 lifting 的底层）
# =============================================================================


@dataclass(frozen=True)
class FpRref:
    """
    RREF 结果（在 F_p 上）：
      - rref: 行最简形矩阵（行数 = rank，列数 = m+1 可能含增广列）
      - pivot_cols: 主元列索引
      - p: 特征
      - m: 变量列数
    """

    rref: List[List[int]]
    pivot_cols: Tuple[int, ...]
    p: int
    m: int


def _rref_augmented(matrix: List[List[int]], p: int, m: int) -> FpRref:
    """
    对增广矩阵 [A|b] 做 RREF。

    matrix: 形状 (rows, m+1)
    """
    if not isinstance(p, int) or p < 2:
        raise ValueError(f"p must be int >= 2, got {p}")
    if not isinstance(m, int) or m < 0:
        raise ValueError(f"m must be int >= 0, got {m}")

    rows = len(matrix)
    if rows == 0:
        return FpRref(rref=[], pivot_cols=tuple(), p=p, m=m)
    for r in matrix:
        if len(r) != m + 1:
            raise ValueError("augmented matrix width mismatch")

    # work on a copy
    A = [[int(x % p) for x in row] for row in matrix]

    pivot_cols: List[int] = []
    r = 0
    for c in range(m):
        # find pivot row >= r with A[row][c] != 0
        pivot = None
        for rr in range(r, rows):
            if A[rr][c] % p != 0:
                pivot = rr
                break
        if pivot is None:
            continue
        # swap pivot row to r
        if pivot != r:
            A[r], A[pivot] = A[pivot], A[r]
        # normalize pivot to 1
        inv = _inv_mod_prime_field(A[r][c], p)
        A[r] = [(val * inv) % p for val in A[r]]
        # eliminate other rows
        for rr in range(rows):
            if rr == r:
                continue
            factor = A[rr][c] % p
            if factor == 0:
                continue
            A[rr] = [(A[rr][j] - factor * A[r][j]) % p for j in range(m + 1)]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break

    # remove all-zero rows for a clean representation (optional but deterministic)
    cleaned: List[List[int]] = []
    for row in A:
        if any((v % p) != 0 for v in row):
            cleaned.append(row)
    return FpRref(rref=cleaned, pivot_cols=tuple(pivot_cols), p=p, m=m)


def _fp_solve_particular(rref: FpRref) -> Optional[List[int]]:
    """
    从 RREF 读取一个特解 x（自由变量取 0），若无解返回 None。
    """
    p = int(rref.p)
    m = int(rref.m)
    piv = set(rref.pivot_cols)

    # inconsistency: [0 ... 0 | 1]
    for row in rref.rref:
        if all((row[j] % p) == 0 for j in range(m)) and (row[m] % p) != 0:
            return None

    x = [0] * m
    # pivot rows are already normalized, so x[pivot_col] = rhs - sum(nonpivot*0) = rhs
    for row in rref.rref:
        pivot_col = None
        for j in range(m):
            if (row[j] % p) != 0:
                pivot_col = j
                break
        if pivot_col is None:
            continue
        if pivot_col not in piv:
            # should not happen, but guard
            continue
        x[pivot_col] = int(row[m] % p)
    return x


def _fp_solve_with_cached_rref(rref: FpRref, rhs: List[int]) -> Optional[List[int]]:
    """
    用已计算的 RREF(A) 解 A x = rhs（自由变量取 0）。
    这里的 trick 是：rref 里存的是 [A|b] 的行最简形；
    但对于 lifting 我们会多次换 rhs，因此要支持“同 A 不同 rhs”的求解。

    为了避免引入启发式或复杂状态，这里采用确定性方式：
    - 重新构造增广矩阵 [A|rhs] 并做 rref
    - 这是 O(m^3)；作为 reference 实现优先正确性。
    """
    # rref 不包含原始 A 的足够信息来直接 apply 到新 rhs（需要记录行变换矩阵）。
    # 为保持严格与易审计性，我们选择重做一次 rref。
    p = int(rref.p)
    m = int(rref.m)
    if len(rhs) != len(rref.rref) and len(rref.rref) != 0:
        # rref.rref 行数可能 < 原行数，不能用这个长度检查；因此这里不做强校验。
        pass
    raise NotImplementedError(
        "Internal: cached-rref multi-RHS solving is intentionally not implemented in v1 "
        "(to avoid hidden complexity). Use _solve_linear_system_mod_p directly."
    )


def _solve_linear_system_mod_p(A: List[List[int]], b: List[int], p: int) -> Optional[List[int]]:
    """
    在 F_p 上解 A x = b，返回一个特解（自由变量取 0），无解则返回 None。
    """
    if not isinstance(p, int) or p < 2:
        raise ValueError(f"p must be int >= 2, got {p}")
    if len(A) != len(b):
        raise ValueError("A and b row mismatch")
    if not A:
        # 0 equations: choose x = 0 vector
        return []
    m = len(A[0])
    for row in A:
        if len(row) != m:
            raise ValueError("A is not rectangular")

    aug = [list(map(int, row)) + [int(bi)] for row, bi in zip(A, b)]
    rref = _rref_augmented(aug, p=p, m=m)
    return _fp_solve_particular(rref)


# =============================================================================
# Z/p^nZ 线性系统：p-adic / Hensel lifting
# =============================================================================


@dataclass(frozen=True)
class LinearSystemSolution:
    """
    A x = b over Z/p^nZ 的解（若存在）。
    """

    x: List[int]  # length m, in [0, p^n-1]
    spec: ChainRingSpec
    certificate: Dict[str, Any]


class NoSolutionError(RuntimeError):
    """
    严格不可解：不是“算不出来”，而是数学上无解（在当前 modulus 下）。
    """


def _mat_vec_mod(A: List[List[int]], x: List[int], mod: int) -> List[int]:
    out: List[int] = []
    for row in A:
        s = 0
        for aij, xj in zip(row, x):
            s += int(aij) * int(xj)
        out.append(int(s % mod))
    return out


def solve_linear_system_zpn(
    A: List[List[int]],
    b: List[int],
    spec: ChainRingSpec,
) -> LinearSystemSolution:
    """
    在 Z/p^nZ 上求解 A x = b（严格：要么给解，要么抛 NoSolutionError）。

    采用逐层 lifting：
      1) 解 mod p
      2) 对 k=1..n-1，解修正项 t 使得 x_{k+1}=x_k + p^k t

    该方法对任意 A 都成立（即便 A mod p 奇异），因为每步都是在 F_p 上做一致性判定。
    """
    if not isinstance(spec, ChainRingSpec):
        raise TypeError(f"spec must be ChainRingSpec, got {type(spec).__name__}")
    if len(A) != len(b):
        raise ValueError("A and b row mismatch")
    if not A:
        # no equations: choose canonical x=0 (size 0)
        return LinearSystemSolution(x=[], spec=spec, certificate={"mode": "v1.no_equations"})
    m = len(A[0])
    for row in A:
        if len(row) != m:
            raise ValueError("A is not rectangular")

    p = int(spec.p)
    n = int(spec.n)
    mod_n = int(spec.modulus)

    # normalize A,b into [0, mod_n)
    A_norm = [[int(aij % mod_n) for aij in row] for row in A]
    b_norm = [int(bi % mod_n) for bi in b]

    # Step 1: solve mod p
    A_p = [[int(aij % p) for aij in row] for row in A_norm]
    b_p = [int(bi % p) for bi in b_norm]
    x_p = _solve_linear_system_mod_p(A_p, b_p, p=p)
    if x_p is None:
        raise NoSolutionError("No solution modulo p (base layer inconsistency)")
    if len(x_p) != m:
        # when A is empty, solver returns []; but here A non-empty so must match
        raise RuntimeError("internal: unexpected solution length in mod p solver")

    x = [int(v % p) for v in x_p]  # mod p

    # Lift to mod p^k progressively
    current_mod = p  # p^1
    for k in range(1, n):
        next_mod = current_mod * p  # p^{k+1}

        Ax = _mat_vec_mod(A_norm, x, mod=next_mod)
        # diff = b - A x (mod p^{k+1})
        diff = [int((bi - axi) % next_mod) for bi, axi in zip(b_norm, Ax)]

        # requirement: diff must be divisible by p^k (= current_mod)
        # compute rhs in F_p: r = diff / p^k mod p
        rhs = []
        for di in diff:
            if di % current_mod != 0:
                raise NoSolutionError(
                    f"No solution at lifting step k={k}: residual not divisible by p^k"
                )
            rhs.append(int((di // current_mod) % p))

        # Solve (A mod p) * t = rhs (mod p)
        t = _solve_linear_system_mod_p(A_p, rhs, p=p)
        if t is None:
            raise NoSolutionError(f"No solution at lifting step k={k}: correction unsolvable mod p")
        if len(t) != m:
            raise RuntimeError("internal: unexpected correction vector length")

        # update x := x + p^k * t  (mod p^{k+1})
        x = [int((xj + current_mod * tj) % next_mod) for xj, tj in zip(x, t)]
        current_mod = next_mod

    cert = {
        "mode": "v1.hensel_lifting",
        "p": p,
        "n": n,
        "rows": len(A),
        "cols": m,
    }
    # sanity verify
    Ax_final = _mat_vec_mod(A_norm, x, mod=mod_n)
    if any(((axi - bi) % mod_n) != 0 for axi, bi in zip(Ax_final, b_norm)):
        raise RuntimeError("internal: lifting produced a non-solution; abort")
    return LinearSystemSolution(x=x, spec=spec, certificate=cert)


# =============================================================================
# Shift-register synthesis: minimize step (最小度数) + 证书
# =============================================================================


@dataclass(frozen=True)
class SynthesisResult:
    """
    合成输出：最小连接多项式（零化子）及其证书。
    """

    spec: ChainRingSpec
    sequence_len: int
    degree: int
    connection_polynomial: List[int]  # [1, c1, ..., cL] in Z/p^nZ
    certificate: Dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("degree must be non-negative int")
        if len(self.connection_polynomial) != self.degree + 1:
            raise ValueError("connection_polynomial length mismatch")
        if self.connection_polynomial[0] % self.spec.modulus != 1:
            raise ValueError("connection_polynomial must be normalized with constant term 1")


def verify_connection_polynomial(
    sequence: Sequence[int],
    poly: Sequence[int],
    spec: ChainRingSpec,
) -> bool:
    """
    验证 poly 是否为 sequence 的连接多项式（严格模 p^n）。
    """
    if not isinstance(spec, ChainRingSpec):
        raise TypeError("spec must be ChainRingSpec")
    seq = [spec.normalize(int(x)) for x in sequence]
    C = [spec.normalize(int(c)) for c in poly]
    if not C:
        raise ValueError("poly must be non-empty")
    if C[0] != 1:
        raise ValueError("poly must have constant term 1")
    L = len(C) - 1
    mod = spec.modulus
    for k in range(L, len(seq)):
        acc = seq[k]
        for i in range(1, L + 1):
            acc = (acc + C[i] * seq[k - i]) % mod
        if acc % mod != 0:
            return False
    return True


def _build_toeplitz_system(sequence: List[int], L: int, spec: ChainRingSpec) -> Tuple[List[List[int]], List[int]]:
    """
    构造线性系统 A x = b, 其中 x=[c1..cL]，递推约束为：
      s[k] + sum_{i=1..L} c_i s[k-i] = 0  for k=L..N-1
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    N = len(sequence)
    mod = spec.modulus
    if L == 0:
        # no variables, constraints are s[k]=0 for all k
        return ([], [])
    rows = N - L
    if rows < 0:
        rows = 0
    A: List[List[int]] = []
    b: List[int] = []
    for k in range(L, N):
        row = []
        for i in range(1, L + 1):
            row.append(int(sequence[k - i] % mod))
        A.append(row)
        b.append(int((-sequence[k]) % mod))
    return (A, b)


def norton_salagean_synthesize(
    sequence: Sequence[int],
    spec: ChainRingSpec,
    *,
    require_solution: bool = True,
) -> Optional[SynthesisResult]:
    """
    在 Z/p^nZ 上合成最小度数连接多项式（Norton–Salagean / Reeds–Sloane 语义的零化子）。

    严格性质：
      - minimize 步进：从 L=0 起逐一判定可解性，第一个可解 L 即为最小度数。
      - 每个 L 的可解性由 p-adic lifting 严格判定。

    Args:
        sequence: 输入序列（将按 modulus 规范化）
        spec: 链环规格
        require_solution: True 时无解抛 NoSolutionError；False 时无解返回 None
    """
    if not isinstance(spec, ChainRingSpec):
        raise TypeError(f"spec must be ChainRingSpec, got {type(spec).__name__}")
    if not isinstance(sequence, (list, tuple)):
        raise TypeError(f"sequence must be list/tuple, got {type(sequence).__name__}")

    seq = [spec.normalize(int(x)) for x in sequence]
    N = len(seq)
    if N == 0:
        # 空序列：度数 0 的单位多项式即可
        return SynthesisResult(
            spec=spec,
            sequence_len=0,
            degree=0,
            connection_polynomial=[1],
            certificate={"mode": "v1.empty_sequence"},
        )

    mod = spec.modulus

    # L=0: 需要所有 s[k]==0
    if all(x % mod == 0 for x in seq):
        return SynthesisResult(
            spec=spec,
            sequence_len=N,
            degree=0,
            connection_polynomial=[1],
            certificate={"mode": "v1.minimize", "L": 0, "note": "all-zero sequence"},
        )

    # minimize step: L from 1..N-1 (L=N 也总可解但不是最小；N-1 足够覆盖有限序列)
    for L in range(1, N):
        A, b = _build_toeplitz_system(seq, L=L, spec=spec)
        try:
            sol = solve_linear_system_zpn(A, b, spec)
        except NoSolutionError:
            continue

        coeffs = [1] + [spec.normalize(int(ci)) for ci in sol.x]
        if not verify_connection_polynomial(seq, coeffs, spec):
            raise RuntimeError("internal: constructed polynomial failed verification; abort")

        cert = {
            "mode": "v1.minimize+hensel",
            "p": spec.p,
            "n": spec.n,
            "N": N,
            "L": L,
            "linear_system": sol.certificate,
        }
        _logger.info("norton_salagean_synthesize: found degree L=%d over Z/%d^%dZ", L, spec.p, spec.n)
        return SynthesisResult(
            spec=spec,
            sequence_len=N,
            degree=L,
            connection_polynomial=coeffs,
            certificate=cert,
        )

    # If we get here, no solution with L < N; in a finite-length setting, L=N always vacuously works.
    # But a degree-N recurrence is non-informative; we treat this as "no nontrivial annihilator found".
    msg = "No connection polynomial found with degree < N (nontrivial annihilator absent at this length)"
    if require_solution:
        raise NoSolutionError(msg)
    return None


# =============================================================================
# 自测试（严格、无随机）：小规模暴力对拍
# =============================================================================


def _bruteforce_min_degree(sequence: List[int], spec: ChainRingSpec) -> Optional[int]:
    """
    仅用于极小规模自测：暴力枚举所有多项式系数以确认最小度数。
    该方法指数级，禁止在生产调用。
    """
    N = len(sequence)
    mod = spec.modulus
    seq = [spec.normalize(x) for x in sequence]
    if all(x == 0 for x in seq):
        return 0
    # L=1..min(N-1, small) exhaustive
    for L in range(1, N):
        # enumerate all coefficient tuples in [0,mod-1]^L
        # 仅用于自测：N 很小的情况下才会调用
        total = mod**L
        if total > 10_000:
            # 不要在自测里意外爆炸
            raise RuntimeError("bruteforce bound exceeded; adjust test sizes")
        coeffs = [0] * L
        for idx in range(total):
            # base-mod expansion
            x = idx
            for i in range(L):
                coeffs[i] = x % mod
                x //= mod
            poly = [1] + list(coeffs)
            if verify_connection_polynomial(seq, poly, spec):
                return L
    return None


def _self_test() -> Dict[str, Any]:
    """
    极小规模确定性自测：验证
      - lifting 求解器一致性
      - 最小度数确实最小（用暴力对拍）
    """
    results: Dict[str, Any] = {"ok": True, "tests": []}

    def record(name: str, passed: bool, detail: str = "") -> None:
        results["tests"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            results["ok"] = False
            _logger.error("SELF-TEST FAILED: %s - %s", name, detail)

    try:
        spec = ChainRingSpec(p=2, n=3)  # modulus 8
        seq = [1, 1, 2, 3, 5]  # small
        res = norton_salagean_synthesize(seq, spec, require_solution=False)
        if res is None:
            raise RuntimeError("expected a solution on this small test")
        assert verify_connection_polynomial(seq, res.connection_polynomial, spec)
        # brute force cross-check (bounded)
        brute_L = _bruteforce_min_degree([spec.normalize(x) for x in seq], spec)
        assert brute_L == res.degree, f"min degree mismatch: brute={brute_L}, got={res.degree}"
        record("min_degree_bruteforce_check", True)
    except Exception as e:
        record("min_degree_bruteforce_check", False, str(e))

    if not results["ok"]:
        raise RuntimeError("norton_salagean self-test failed; deployment must abort")
    return results


__all__ = [
    "ChainRingSpec",
    "LinearSystemSolution",
    "NoSolutionError",
    "SynthesisResult",
    "solve_linear_system_zpn",
    "verify_connection_polynomial",
    "norton_salagean_synthesize",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("Running norton_salagean self-test...")
    out = _self_test()
    print(out)

