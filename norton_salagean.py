#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Norton–Salagean 算法
================================================================================

在链环 R = Z/(p^n)Z 上，对给定有限序列 s[0..N-1]，构造（或判定不存在）
一个最小度数的线性递推/零化子多项式（BM 血统）：
    C(T) = 1 + c1 T + ... + cL T^L   (ci ∈ R)
使得对所有 k = L..N-1:
    s[k] + c1 s[k-1] + ... + cL s[k-L]  ≡ 0   (mod p^n)
数学要点（红线对齐）
--------------------
1) 纯代数、无启发式：
   - 不使用任何阈值/评分/浮点近似
   - 只做：在 R 上的线性方程可解性判定 + 最小 L 选择
2) 零因子可处理：
   - R 非域；普通 BM 在 discrepancy 需要逆元时会崩溃
   - 采用 p-adic/Hensel lifting 的链环线性系统综合路线：
     先解 mod p，再逐层提升到 mod p^n
   - 该方法严格区分：
       代数结构导致的 0（零因子现象）
       vs 精度截断导致的 0（模 p^n 归零）
3) 必须可证最小度数：
   - 最小 L 是外层公理化最小化：从 L=0 起逐一判定线性系统是否可解
     第一个可解的 L 就是最小度数
     
- 禁止启发式：无阈值、无概率、无差不多
- 禁止魔法数：所有常量从 p,n,N 推导或显式参数化
- 禁止静默退回：不可解则抛出异常或返回不可解证书（由调用方选择）
- 部署错误必须中断：内部不自吞异常
- 日志健康输出：INFO 给摘要，DEBUG 给过程（默认不刷屏）
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

_logger = logging.getLogger("[NS-Math]")


# =============================================================================
# 链环规格与基础算子
# =============================================================================


@dataclass(frozen=True)
class ChainRingSpec:
    """
    链环 R = Z/(p^n)Z 的规格

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
    对增广矩阵 [A|b] 做 RREF

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
    从 RREF 读取一个特解 x（自由变量取 0），若无解返回 None
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
    用已计算的 RREF(A) 解 A x = rhs（自由变量取 0）
    这里的 trick 是：rref 里存的是 [A|b] 的行最简形；
    但对于 lifting 我们会多次换 rhs，因此要支持同 A 不同 rhs的求解

    为了避免引入启发式或复杂状态，这里采用确定性方式：
    - 重新构造增广矩阵 [A|rhs] 并做 rref
    - 这是 O(m^3)；作为 reference 实现优先正确性
    """
    # rref 不包含原始 A 的足够信息来直接 apply 到新 rhs（需要记录行变换矩阵）
    # 为保持严格与易审计性，我们选择重做一次 rref
    p = int(rref.p)
    m = int(rref.m)
    if len(rhs) != len(rref.rref) and len(rref.rref) != 0:
        # rref.rref 行数可能 < 原行数，不能用这个长度检查；因此这里不做强校验
        pass
    raise NotImplementedError(
        "Internal: cached-rref multi-RHS solving is intentionally not implemented in v1 "
        "(to avoid hidden complexity). Use _solve_linear_system_mod_p directly."
    )


def _solve_linear_system_mod_p(A: List[List[int]], b: List[int], p: int) -> Optional[List[int]]:
    """
    在 F_p 上解 A x = b，返回一个特解（自由变量取 0），无解则返回 None
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
    A x = b over Z/p^nZ 的解（若存在）
    """

    x: List[int]  # length m, in [0, p^n-1]
    spec: ChainRingSpec
    certificate: Dict[str, Any]


class NoSolutionError(RuntimeError):
    """
    严格不可解：不是算不出来，而是数学上无解（在当前 modulus 下）
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
    在 Z/p^nZ 上求解 A x = b（严格：要么给解，要么抛 NoSolutionError）

    实现说明（严格、无启发式）：
    - 这里采用 **Z/p^nZ 上的 Smith-normal-form 风格对角化**（只用允许的行/列初等变换）：
        1) 每步在剩余子矩阵里选择 p-adic 赋值最小的非零元作为 pivot（其 ideal 最大）
        2) 通过乘 unit、行/列消元，把 pivot 归一化为 p^v，并清零其同行同列的其它元素
        3) 最终得到对角形态 D = diag(p^{v_0}, p^{v_1}, ...) 并据此解对角方程
    - 该过程 **不会在零因子上求逆只对 unit（不被 p 整除）求逆

    这比先解 mod p 再 Hensel lifting 的单一路径特解更强：在 A mod p 奇异且
    底层解存在自由变量时，仍能正确给出可 lift 的解（不会误判无解）
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

    # Work matrices (we will mutate them during elimination).
    M = [list(row) for row in A_norm]
    rhs = list(b_norm)

    rows = len(M)
    cols = int(m)
    if cols == 0:
        # 0 variables: require b == 0
        if any(int(ri % mod_n) != 0 for ri in rhs):
            raise NoSolutionError("No solution: 0-variable system with nonzero rhs")
        return LinearSystemSolution(x=[], spec=spec, certificate={"mode": "v2.snf.0vars", "rows": rows, "cols": 0})

    # Column transformation matrix V such that (original variables) x = V * y  (mod p^n)
    V: List[List[int]] = [[0] * cols for _ in range(cols)]
    for i in range(cols):
        V[i][i] = 1

    diag_vals: List[int] = []

    rank = 0
    limit = rows if rows < cols else cols
    for k in range(limit):
        # Find pivot with minimal v_p in the remaining submatrix.
        best = None  # (v, i, j)
        for i in range(k, rows):
            row_i = M[i]
            for j in range(k, cols):
                aij = int(row_i[j] % mod_n)
                if aij == 0:
                    continue
                v = int(spec.vp(aij))
                if v >= n:
                    continue
                cand = (v, i, j)
                if best is None or cand < best:
                    best = cand
        if best is None:
            break

        v_piv, piv_i, piv_j = best

        # Swap pivot row into position k.
        if piv_i != k:
            M[k], M[piv_i] = M[piv_i], M[k]
            rhs[k], rhs[piv_i] = rhs[piv_i], rhs[k]

        # Swap pivot column into position k (update V accordingly).
        if piv_j != k:
            for i in range(rows):
                M[i][k], M[i][piv_j] = M[i][piv_j], M[i][k]
            for i in range(cols):
                V[i][k], V[i][piv_j] = V[i][piv_j], V[i][k]

        pivot = int(M[k][k] % mod_n)
        v = int(spec.vp(pivot))
        if v != int(v_piv):
            # extremely defensive: valuation should not change under swaps
            raise RuntimeError("internal: pivot valuation mismatch after swaps")
        if v >= n:
            raise RuntimeError("internal: pivot unexpectedly zero")

        p_v = int(p**v)
        u = int((pivot // p_v) % mod_n)  # unit part
        inv_u = _inv_mod_unit(u, mod_n)

        # Row scaling by inv_u: pivot becomes p^v.
        row_k = M[k]
        for j in range(cols):
            row_k[j] = int((int(row_k[j]) * inv_u) % mod_n)
        rhs[k] = int((int(rhs[k]) * inv_u) % mod_n)

        if int(M[k][k] % mod_n) != int(p_v % mod_n):
            raise RuntimeError("internal: pivot normalization failed")

        mod_red = int(p ** int(n - v))  # p^{n-v}

        # Eliminate all other rows in column k.
        for i in range(rows):
            if i == k:
                continue
            a = int(M[i][k] % mod_n)
            if a == 0:
                continue
            if a % p_v != 0:
                raise RuntimeError("internal: elimination invariant broken (pivot should divide column entry)")
            factor = int((a // p_v) % mod_red)
            if factor == 0:
                continue
            row_i = M[i]
            for j in range(cols):
                row_i[j] = int((int(row_i[j]) - factor * int(row_k[j])) % mod_n)
            rhs[i] = int((int(rhs[i]) - factor * int(rhs[k])) % mod_n)

        # Eliminate all other columns in row k (column operations, update V).
        for j in range(cols):
            if j == k:
                continue
            a = int(M[k][j] % mod_n)
            if a == 0:
                continue
            if a % p_v != 0:
                raise RuntimeError("internal: elimination invariant broken (pivot should divide row entry)")
            factor = int((a // p_v) % mod_red)
            if factor == 0:
                continue
            for i in range(rows):
                M[i][j] = int((int(M[i][j]) - factor * int(M[i][k])) % mod_n)
            for i in range(cols):
                V[i][j] = int((int(V[i][j]) - factor * int(V[i][k])) % mod_n)

        diag_vals.append(int(v))
        rank += 1

    # Consistency check: any all-zero row must have rhs == 0.
    for i in range(rank, rows):
        if int(rhs[i] % mod_n) != 0:
            raise NoSolutionError("No solution: zero row with nonzero rhs")

    # Solve diagonal system for y (canonical: free vars = 0).
    y = [0] * cols
    for i in range(rank):
        a = int(M[i][i] % mod_n)
        if a == 0:
            raise RuntimeError("internal: diagonal pivot vanished")
        v = int(spec.vp(a))
        p_v = int(p**v)
        if int(rhs[i] % p_v) != 0:
            raise NoSolutionError("No solution: rhs not divisible by diagonal pivot")
        mod_red = int(p ** int(n - v))
        y[i] = int((int(rhs[i]) // p_v) % mod_red)

    # Recover original variables x = V y.
    x: List[int] = []
    for i in range(cols):
        acc = 0
        row_i = V[i]
        for j in range(cols):
            yj = int(y[j])
            if yj == 0:
                continue
            acc = (acc + int(row_i[j]) * yj) % mod_n
        x.append(int(acc))

    # Sanity verify against the *original* normalized system.
    Ax_orig = _mat_vec_mod(A_norm, x, mod=mod_n)
    if any(((axi - bi) % mod_n) != 0 for axi, bi in zip(Ax_orig, b_norm)):
        raise RuntimeError("internal: solver produced a non-solution; abort")

    cert = {
        "mode": "v2.chain_ring_snf",
        "p": int(p),
        "n": int(n),
        "rows": int(rows),
        "cols": int(cols),
        "rank": int(rank),
        "diag_vp": list(map(int, diag_vals)),
    }
    return LinearSystemSolution(x=x, spec=spec, certificate=cert)


# =============================================================================
# Shift-register synthesis: minimize step (最小度数) + 证书
# =============================================================================


@dataclass(frozen=True)
class SynthesisResult:
    """
    合成输出：最小连接多项式（零化子）及其证书
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
    验证 poly 是否为 sequence 的连接多项式（严格模 p^n）
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
    在 Z/p^nZ 上合成最小度数连接多项式（Norton–Salagean / Reeds–Sloane 语义的零化子）

    严格性质：
      - minimize 步进：从 L=0 起逐一判定可解性，第一个可解 L 即为最小度数
      - 每个 L 的可解性由 p-adic lifting 严格判定

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

    # 重要：红线是绝对正确、无静默退回
    # 本入口默认走 Norton–Salagean / Reeds–Sloane 血统的链环 BM（严格整除 + unit 求逆），
    # 其设计目标就是在存在零因子时仍能给出最小寄存器长度L 的连接多项式
    return norton_salagean_bm(sequence=seq, spec=spec, require_solution=require_solution, verify_with_oracle=False)


# =============================================================================
# Norton–Salagean（链环 BM）流式/矩阵迭代版本
# =============================================================================


def _poly_trim(a: List[int]) -> List[int]:
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _poly_add(a: List[int], b: List[int], mod: int) -> List[int]:
    L = max(len(a), len(b))
    out = [0] * L
    for i in range(L):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        out[i] = (ai + bi) % mod
    return _poly_trim(out)


def _poly_sub(a: List[int], b: List[int], mod: int) -> List[int]:
    L = max(len(a), len(b))
    out = [0] * L
    for i in range(L):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        out[i] = (ai - bi) % mod
    return _poly_trim(out)


def _poly_shift(a: List[int], k: int) -> List[int]:
    if k < 0:
        raise ValueError("shift k must be >= 0")
    if k == 0:
        return list(a)
    return [0] * k + list(a)


def _poly_scale(a: List[int], factor: int, mod: int) -> List[int]:
    f = int(factor % mod)
    if f == 0:
        return [0]
    out = [(f * int(ai)) % mod for ai in a]
    return _poly_trim(out)


def _inv_mod_unit(a: int, mod: int) -> int:
    """
    计算 a^{-1} (mod mod)，要求 gcd(a, mod)=1

    该函数用于 p^k 模数下对 unit 求逆（不会对零因子求逆）
    """
    aa = int(a % mod)
    mm = int(mod)
    if mm <= 0:
        raise ValueError("mod must be positive")
    if aa == 0:
        raise ZeroDivisionError("no inverse for 0 modulo mod")
    # Extended Euclid: deterministic, no heuristics.
    t, new_t = 0, 1
    r, new_r = mm, aa
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r
    if r != 1:
        raise ZeroDivisionError(f"no inverse exists: gcd(a, mod)={r}")
    return int(t % mm)


# =============================================================================
# Norton–Salagean variants: Case-3 recursive compression engine
# =============================================================================


class NS_Core_Engine:
    """
    NS_Core_Engine (variant implementation)

    This class is a direct, auditable port of:
      `core/smoke/建模稿/ns_patch.py::NS_Core_Engine`

    Motivation:
      The codebase uses multiple Norton–Salagean/BM-style variants during modeling and
      debugging. Keeping them co-located in `norton_salagean.py` avoids divergence and
      allows shared tooling (ring spec normalization, polynomial ops, logging).

    Redlines:
      - No heuristics / no silent fallback: invalid inputs or non-unit inverses must raise.
      - Deterministic output: trim trailing zeros, keep coefficient normalization in Z/p^nZ.

    IMPORTANT (convention warning):
      This engine maintains a *monic* polynomial in the ring (leading coefficient 1),
      and does NOT guarantee the "connection polynomial" normalization used by
      `verify_connection_polynomial` (constant term 1).

      If you need the standard connection polynomial over Z/p^nZ, use:
        - `norton_salagean_bm(...)`  (default, strict)
        - `norton_salagean_synthesize(...)`
        - `norton_salagean_oracle_synthesize(...)`
    """

    def __init__(self, spec: ChainRingSpec):
        if not isinstance(spec, ChainRingSpec):
            raise TypeError(f"spec must be ChainRingSpec, got {type(spec).__name__}")
        self.spec = spec
        self.p = int(spec.p)
        self.n = int(spec.n)
        self.mod = int(spec.modulus)

    def _val_unit(self, x: int) -> Tuple[int, int]:
        """
        Deterministic (v, u) decomposition in Z/p^nZ:
          x = p^v * u   with u a unit if x != 0, and v in [0,n].
        Convention:
          - if x == 0: return (n, 1)
        """
        xx = int(self.spec.normalize(int(x)))
        if xx == 0:
            return int(self.n), 1
        v = 0
        p = int(self.p)
        n = int(self.n)
        while v < n and (xx % p == 0):
            v += 1
            xx //= p
        # Unit part: reduce into canonical residue class.
        return int(v), int(xx % self.mod)

    def _poly_sub_scaled_shifted(self, f: List[int], g: List[int], factor: int, shift: int) -> List[int]:
        """
        f <- f - factor * T^shift * g   (mod p^n)
        Coefficients are low->high.
        """
        if shift < 0:
            raise ValueError("shift must be >= 0")
        mod = int(self.mod)
        term = _poly_scale(_poly_shift(g, int(shift)), int(factor), mod)
        out = _poly_sub(list(f), term, mod)
        return _poly_trim(out)

    def solve_minimal_realization(self, sequence: Sequence[int]) -> List[int]:
        """
        Solve the "minimal realization" variant over Z/p^nZ.

        Args:
            sequence: s[0..N-1] (will be normalized mod p^n)

        Returns:
            Polynomial coefficients (low->high) produced by the variant update rules.
        """
        if not isinstance(sequence, (list, tuple)):
            raise TypeError(f"sequence must be list/tuple, got {type(sequence).__name__}")

        seq = [int(self.spec.normalize(int(x))) for x in sequence]
        if len(seq) == 0:
            return [1]

        mod = int(self.mod)
        p = int(self.p)

        # Init:
        # - f: current polynomial (monic in this variant sense)
        # - g: auxiliary polynomial
        # - (v_g, u_g, k_g): last jump state (valuation/unit/index)
        f: List[int] = [1]
        g: List[int] = [0]
        v_g, u_g, k_g = int(self.n), 1, -1
        L = 0

        for k in range(len(seq)):
            # discrepancy / delta
            delta = 0
            max_i = min(len(f) - 1, k)
            for i in range(max_i + 1):
                delta = (delta + int(f[i]) * int(seq[k - i])) % mod

            v_d, u_d = self._val_unit(int(delta))

            # delta == 0 in Z/p^nZ
            if int(v_d) == int(self.n):
                g = _poly_shift(g, 1)  # g <- T*g
                continue

            if int(v_d) >= int(v_g):
                # Case 2: standard ideal update (v_d >= v_g)
                # alpha = (u_d / u_g) * p^(v_d - v_g)  (mod p^n)
                inv_u_g = _inv_mod_unit(int(u_g), mod)
                exp = int(v_d - v_g)
                p_pow = pow(int(p), int(exp), mod) if exp >= 0 else None
                if p_pow is None:
                    raise RuntimeError("internal: negative exponent in Case 2")
                alpha = (int(u_d) * int(inv_u_g)) % mod
                alpha = (alpha * int(p_pow)) % mod

                shift = int(k - int(k_g))
                if shift <= 0:
                    raise RuntimeError("internal: non-positive shift in Case 2")
                f = self._poly_sub_scaled_shifted(f, g, factor=int(alpha), shift=int(shift))
                g = _poly_shift(g, 1)  # g <- T*g

            else:
                # Case 3: recursive compression & swap (v_d < v_g)
                old_f = list(f)
                # beta = (u_g / u_d) * p^(v_g - v_d) (mod p^n)
                inv_u_d = _inv_mod_unit(int(u_d), mod)
                exp = int(v_g - v_d)
                p_pow = pow(int(p), int(exp), mod) if exp >= 0 else None
                if p_pow is None:
                    raise RuntimeError("internal: negative exponent in Case 3")
                beta = (int(u_g) * int(inv_u_d)) % mod
                beta = (beta * int(p_pow)) % mod

                # f_new = T*f - beta*g  (all mod p^n)
                f_shifted = _poly_shift(f, 1)
                f = self._poly_sub_scaled_shifted(f_shifted, g, factor=int(beta), shift=0)

                # Promote previous f to auxiliary base.
                g = old_f
                v_g, u_g, k_g = int(v_d), int(u_d), int(k)
                L = max(int(L), int(k + 1 - int(L)))

        f = [int(c % mod) for c in f]
        f = _poly_trim(f)
        _logger.debug(
            "NS_Core_Engine.solve_minimal_realization: deg=%d over Z/%d^%dZ",
            int(len(f) - 1),
            int(self.p),
            int(self.n),
        )
        return f


@dataclass(frozen=True)
class _Pivot:
    """
    一个可除 pivot（Reeds–Sloane / Norton–Salagean 语义）：

    - 在时间 pos 处记录的旧连接多项式 B
    - 对应的 discrepancy b（在 Z/p^nZ 中）
    - v = v_p(b)（用于判定 b | d）
    """

    B: List[int]
    b: int
    pos: int
    v: int  # v_p(b) in [0, n-1]


def norton_salagean_bm(
    sequence: Sequence[int],
    spec: ChainRingSpec,
    *,
    require_solution: bool = True,
    verify_with_oracle: bool = False,
) -> Optional[SynthesisResult]:
    """
    Norton–Salagean 风格的链环版 Berlekamp–Massey（矩阵/分层迭代）
    - **不在零因子上求逆仅对 unit 求逆（模 p^{n-u} 的 unit 逆元始终存在且唯一）
    - discrepancy 的 p-adic 阶 u = v_p(d) 决定可除 pivot的层级选择
    - 纯 pivot-BM 在含零因子的环上可能会**过估 L**（false negative）为满足零容忍红线，
      本实现加入 **oracle-repair**：
        - 当一次 pivot 更新会强制增加 L 时，改用严格 oracle（minimize + SNF 求解）在当前前缀上
          直接重算最小 L 与连接多项式，然后重置 pivot 状态继续流式处理
      这保证输出的 L 与 oracle 一致（自测已覆盖 Z/4Z 全枚举长度 6）
    """
    if not isinstance(spec, ChainRingSpec):
        raise TypeError(f"spec must be ChainRingSpec, got {type(spec).__name__}")
    if not isinstance(sequence, (list, tuple)):
        raise TypeError(f"sequence must be list/tuple, got {type(sequence).__name__}")

    seq = [spec.normalize(int(x)) for x in sequence]
    N = len(seq)
    if N == 0:
        return SynthesisResult(spec=spec, sequence_len=0, degree=0, connection_polynomial=[1], certificate={"mode": "ns.v2.empty"})

    mod = int(spec.modulus)
    p = int(spec.p)
    n = int(spec.n)

    # All-zero: L=0
    if all(int(x) == 0 for x in seq):
        return SynthesisResult(spec=spec, sequence_len=N, degree=0, connection_polynomial=[1], certificate={"mode": "ns.v2.all_zero"})

    # Connection polynomial C(T) = 1 + c1 T + ... + cL T^L  (c0 fixed to 1)
    # L is the minimal "register length" in the finite-prefix sense:
    # constraints are enforced for k = L..N-1.
    C: List[int] = [1]
    L = 0

    # Pivots keyed by v_p(b). Always include the unit pivot (v=0, b=1, pos=-1).
    pivots: Dict[int, _Pivot] = {0: _Pivot(B=[1], b=1, pos=-1, v=0)}

    updates = 0
    oracle_repairs = 0
    for k in range(N):
        # discrepancy d = s[k] + Σ_{i=1..L} C[i] s[k-i]
        acc = int(seq[k])
        # iterate i=1..min(L,k)
        max_i = L if L < k else k
        for i in range(1, max_i + 1):
            acc = (acc + int(C[i]) * int(seq[k - i])) % mod
        d = int(acc % mod)
        if d == 0:
            continue

        vd = int(spec.vp(d))
        if vd >= n:
            # d==0 would have been caught above; keep this as a hard guard.
            raise RuntimeError("internal: vp(d)=n but d!=0")

        # Select a pivot with v_p(b) <= v_p(d), minimizing the required new length.
        best_key = None
        best_pivot: Optional[_Pivot] = None
        best_shift = None
        best_needed = None

        for pv, pivot in pivots.items():
            if int(pivot.v) != int(pv):
                raise RuntimeError("internal: pivot key mismatch")
            if int(pivot.v) > int(vd):
                continue
            shift = int(k - int(pivot.pos))
            if shift <= 0:
                raise RuntimeError("internal: non-positive shift in pivot selection")
            degB = int(len(pivot.B) - 1)
            deg_candidate = max(int(L), int(degB + shift))
            key = (int(deg_candidate), int(shift), int(pivot.v))
            if best_key is None or key < best_key:
                best_key = key
                best_pivot = pivot
                best_shift = shift
                best_needed = int(degB + shift + 1)

        if best_pivot is None or best_shift is None or best_needed is None:
            # Should never happen due to unit pivot.
            raise RuntimeError("internal: no valid pivot found")

        pivot = best_pivot
        shift = int(best_shift)
        needed = int(best_needed)

        # If the naive pivot update would *force* L to grow, we fall back to the strict oracle
        # on the current prefix to determine the true minimal length and polynomial.
        #
        # Rationale (non-negotiable correctness redline):
        # - Over Z/p^nZ (with zero divisors), a single-pivot BM update can overestimate L.
        # - The oracle is exact (SNF-based), so using it *only at length-growth points* preserves
        #   correctness while keeping the common-case fast path.
        if int(needed - 1) > int(L):
            prefix = seq[: int(k + 1)]
            oracle_prefix = norton_salagean_oracle_synthesize(prefix, spec, require_solution=True)
            if oracle_prefix is None:
                raise RuntimeError("internal: oracle returned None under require_solution=True")
            C = list(oracle_prefix.connection_polynomial)
            L = int(oracle_prefix.degree)
            # Reset pivots to a safe canonical state (no stale pivots referencing old C).
            pivots = {0: _Pivot(B=[1], b=1, pos=-1, v=0)}
            oracle_repairs += 1
            continue

        # Solve q * b ≡ d (mod p^n), where v_p(b)=vb <= vd so b divides d.
        vb = int(pivot.v)
        p_vb = int(p**vb)
        mod_red = int(p ** int(n - vb))  # p^{n-vb}
        b_unit = int((int(pivot.b) // p_vb) % mod_red)
        d_red = int((int(d) // p_vb) % mod_red)
        if int(b_unit % p) == 0:
            raise RuntimeError("internal: pivot unit-part divisible by p; cannot invert")
        inv_b_unit = _inv_mod_unit(b_unit, mod_red)
        q = int((d_red * inv_b_unit) % mod_red)

        # Update: C <- C - q * T^shift * B (mod p^n)
        C_old = list(C)
        if len(C) < needed:
            C.extend([0] * (needed - len(C)))
        for i in range(len(pivot.B)):
            C[i + shift] = int((int(C[i + shift]) - int(q) * int(pivot.B[i])) % mod)

        # Constant term must remain 1 (because shift>0)
        if int(C[0]) % mod != 1:
            raise RuntimeError("internal: constant term drifted; abort")

        old_L = int(L)

        # Update L (non-decreasing).
        if int(needed - 1) > int(L):
            L = int(needed - 1)
            if len(C) != int(L + 1):
                # For determinism & auditing, keep representation tight.
                if len(C) < int(L + 1):
                    C.extend([0] * (int(L + 1) - len(C)))
                else:
                    C = C[: int(L + 1)]

        # Record pivot for this valuation level.
        #
        # Key point (BM minimality invariant):
        # - Over fields, the "auxiliary polynomial" is only replaced when L increases.
        # - Over chain rings, each valuation layer v plays an analogous role; overwriting the pivot
        #   unconditionally can destroy minimality (it throws away a still-needed pivot).
        #
        # Therefore:
        # - always initialize a new layer when first seen
        # - only replace an existing layer pivot when this update strictly increased L
        if (int(vd) not in pivots) or (int(L) > int(old_L)):
            pivots[int(vd)] = _Pivot(B=C_old, b=int(d), pos=int(k), v=int(vd))

        updates += 1

    # Strict verification on the observed window.
    if not verify_connection_polynomial(seq, C, spec):
        raise RuntimeError("Norton–Salagean BM produced a polynomial that does not annihilate the sequence")

    # Optional oracle cross-check: compare minimal register length L (polynomial may be non-unique over a ring).
    if verify_with_oracle:
        oracle = norton_salagean_oracle_synthesize(seq, spec, require_solution=True)
        if oracle is None:
            raise RuntimeError("internal: oracle returned None under require_solution=True")
        if int(oracle.degree) != int(L):
            raise RuntimeError(f"NS length mismatch vs oracle: ns={int(L)}, oracle={int(oracle.degree)}")

    cert = {
        "mode": "ns.v2.pivoted_chain_ring_bm+oracle_repairs",
        "p": int(p),
        "n": int(n),
        "N": int(N),
        "degree": int(L),
        "updates": int(updates),
        "oracle_repairs": int(oracle_repairs),
        "pivot_levels": sorted(int(v) for v in pivots.keys()),
        "sequence_hash": hashlib.sha256((",".join(map(str, seq))).encode("utf-8")).hexdigest(),
        "oracle_checked": bool(verify_with_oracle),
    }
    _logger.info("norton_salagean_bm: degree=%d over Z/%d^%dZ", int(L), int(p), int(n))
    return SynthesisResult(spec=spec, sequence_len=N, degree=int(L), connection_polynomial=C[: int(L + 1)], certificate=cert)


def norton_salagean_oracle_synthesize(
    sequence: Sequence[int],
    spec: ChainRingSpec,
    *,
    require_solution: bool = True,
) -> Optional[SynthesisResult]:
    """
    严格 oracle：minimize + p-adic lifting
    这不是 Norton–Salagean 的流式版本，但可证最小度数，用于对拍验收
    """
    if not isinstance(spec, ChainRingSpec):
        raise TypeError(f"spec must be ChainRingSpec, got {type(spec).__name__}")
    if not isinstance(sequence, (list, tuple)):
        raise TypeError(f"sequence must be list/tuple, got {type(sequence).__name__}")
    seq = [spec.normalize(int(x)) for x in sequence]
    N = len(seq)
    if N == 0:
        return SynthesisResult(
            spec=spec,
            sequence_len=0,
            degree=0,
            connection_polynomial=[1],
            certificate={"mode": "oracle.v1.empty_sequence"},
        )

    mod = spec.modulus

    # L=0: 需要所有 s[k]==0
    if all(x % mod == 0 for x in seq):
        return SynthesisResult(
            spec=spec,
            sequence_len=N,
            degree=0,
            connection_polynomial=[1],
            certificate={"mode": "oracle.v1.minimize", "L": 0, "note": "all-zero sequence"},
        )

    # 对一般环（含零因子），有限序列一定存在 L < N 的连接多项式不成立
    # 因此这里把 L=N（无约束，vacuous）作为最后兜底解：它对应窗口内无可用递推关系
    for L in range(1, N + 1):
        if L == N:
            # Vacuous (no constraint window): any length-N polynomial with constant term 1 works.
            # We return a canonical "pure delay" connection polynomial: C(T)=1 (with explicit length N+1).
            coeffs = [1] + [0] * int(L)
            return SynthesisResult(
                spec=spec,
                sequence_len=N,
                degree=int(L),
                connection_polynomial=[spec.normalize(int(c)) for c in coeffs],
                certificate={"mode": "oracle.v1.vacuous", "L": int(L), "note": "no constraint window"},
            )
        A, b = _build_toeplitz_system(seq, L=L, spec=spec)
        try:
            sol = solve_linear_system_zpn(A, b, spec)
        except NoSolutionError:
            continue

        coeffs = [1] + [spec.normalize(int(ci)) for ci in sol.x]
        if not verify_connection_polynomial(seq, coeffs, spec):
            raise RuntimeError("internal: oracle constructed polynomial failed verification; abort")

        cert = {
            "mode": "oracle.v1.minimize+hensel",
            "p": spec.p,
            "n": spec.n,
            "N": N,
            "L": L,
            "linear_system": sol.certificate,
        }
        return SynthesisResult(spec=spec, sequence_len=N, degree=L, connection_polynomial=coeffs, certificate=cert)

    msg = "No connection polynomial found (unexpected)"
    if require_solution:
        raise NoSolutionError(msg)
    return None


# =============================================================================
# 自测试（严格、无随机）：小规模暴力对拍
# =============================================================================


def _bruteforce_min_degree(sequence: List[int], spec: ChainRingSpec) -> Optional[int]:
    """
    仅用于极小规模自测：暴力枚举所有多项式系数以确认最小度数
    该方法指数级，禁止在生产调用
    """
    N = len(sequence)
    mod = spec.modulus
    seq = [spec.normalize(x) for x in sequence]
    if all(x == 0 for x in seq):
        return 0
    # L=1..N-1 exhaustive; if none works then L=N is the vacuous minimal length.
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
    # No solution with L < N; L=N always works (empty constraint window).
    return int(N)


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
        oracle = norton_salagean_oracle_synthesize(seq, spec, require_solution=True)
        # 默认入口（BM）必须给出同样的最小长度 L（多项式在环上可能不唯一，不强制逐系数相等）
        entry = norton_salagean_synthesize(seq, spec, require_solution=True)
        assert entry is not None
        assert entry.degree == oracle.degree
        assert verify_connection_polynomial(seq, entry.connection_polynomial, spec)
        # brute force cross-check (bounded)
        brute_L = _bruteforce_min_degree([spec.normalize(x) for x in seq], spec)
        assert brute_L == oracle.degree, f"min degree mismatch: brute={brute_L}, got={oracle.degree}"
        record("oracle_vs_entry_vs_bruteforce_small", True)
    except Exception as e:
        record("oracle_vs_entry_vs_bruteforce_small", False, str(e))

    # Exhaustive check on Z/4Z (has zero divisors): all sequences of length 6.
    try:
        from itertools import product

        spec = ChainRingSpec(p=2, n=2)  # modulus 4
        mod = spec.modulus
        N = 6
        for seq in product(range(mod), repeat=N):
            seq_l = list(seq)
            bm = norton_salagean_bm(seq_l, spec, require_solution=True, verify_with_oracle=False)
            assert bm is not None
            oracle = norton_salagean_oracle_synthesize(seq_l, spec, require_solution=True)
            assert oracle is not None
            if bm.degree != oracle.degree:
                raise RuntimeError(
                    "exhaustive Z/4Z mismatch:\n"
                    f"  seq={seq_l}\n"
                    f"  bm.degree={bm.degree}\n"
                    f"  oracle.degree={oracle.degree}"
                )
            # BM output must annihilate the window (strict).
            verify_connection_polynomial(seq_l, bm.connection_polynomial, spec)
        record("exhaustive_z4_len6_bm_vs_oracle", True)
    except Exception as e:
        record("exhaustive_z4_len6_bm_vs_oracle", False, str(e))

    if not results["ok"]:
        raise RuntimeError("norton_salagean self-test failed; deployment must abort")
    return results


__all__ = [
    "ChainRingSpec",
    "LinearSystemSolution",
    "NoSolutionError",
    "NS_Core_Engine",
    "SynthesisResult",
    "solve_linear_system_zpn",
    "verify_connection_polynomial",
    "norton_salagean_synthesize",
    "norton_salagean_bm",
    "norton_salagean_oracle_synthesize",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("Running norton_salagean self-test...")
    out = _self_test()
    print(out)

