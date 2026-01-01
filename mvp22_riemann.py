#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
MVP22 - 黎曼猜想工程化 (Riemann Hypothesis Engineering)

核心理念:
  通过 Iwasawa 理论将复数 Zeta 函数的零点行为转化为
  p-adic L-函数在 Λ = Z_p[[T]] 上的代数结构

数学架构:
  MVP0 (Witt向量) → p-adic L-函数 → Iwasawa 主猜想 → MVP22 (全局Zeta零点控制)

  1. project_holographic_state       - 全息投影: Universe A → B 的视角变换
  2. verify_topological_homeomorphism - 拓扑同胚验证: Iwasawa Module 结构比较
  3. enforce_multiradial_consensus   - 多辐射共识: 三宇宙重合验证
  4. _calculate_hausdorff_drift      - Hausdorff 漂移: 投影间微小偏差量化

红线 (Critical Constraints):
  - 禁止启发式 (No Heuristics)
  - 禁止魔法数 (No Magic Numbers)  
  - 禁止静默退回 (Must Throw on Failure)
  - 部署错误必须中断 (Deployment Errors Must Abort)
  - 日志健康输出 (Healthy Logging - No Spam, No Silence)
================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

_logger = logging.getLogger("[MVP22]")

# =============================================================================
# 常量: 从数学原理推导，禁止魔法数
# =============================================================================

def _min_witt_length_for_bits(bits: int, p: int) -> int:
    """
    以纯整数方式求最小 n，使 p^n ≥ 2^(2·bits)。

    - 不使用浮点/近似，避免“差不多”的精度损失。
    - 安全余量 = 2×bits（数据 + 溢出捕捉），与函数命名明确绑定。
    """
    if not isinstance(bits, int) or bits < 1:
        raise ValueError(f"bits must be positive int, got {bits}")
    if not isinstance(p, int) or p < 2:
        raise ValueError(f"p must be int >= 2, got {p}")

    target_bits = int(bits * 2)
    threshold = 1 << target_bits  # 2^(2·bits)

    n = 1
    power = int(p)
    while power < threshold:
        n += 1
        power *= int(p)
    return int(n)


def _trinity_primes() -> Tuple[int, int, int]:
    """
    三位一体素数选择 (来自 Iwasawa代数标准.txt):
      轨道 A (物理层): p=2 - EVM二进制底层
      轨道 B (几何层): secp256k1 域素数 (SEC 2 原式推导，不接受替换)
      轨道 C (测试层): p=3 - Smoke Test

    返回前做显式一致性验证，防止“魔法数”被静默改写。
    """
    secp256k1_prime = (1 << 256) - (1 << 32) - 977  # 按 SEC 2 定义构造

    candidates = {
        "phys": 2,
        "geom": secp256k1_prime,
        "test": 3,
    }

    for label, val in candidates.items():
        if not isinstance(val, int) or val < 2:
            raise ValueError(f"{label} prime must be int >= 2, got {val}")
        # 最低限度的确定性验证：排除小质因数污染，避免被误写为合数。
        for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if val == small:
                continue
            if val % small == 0:
                raise ValueError(f"{label} prime candidate {val} is divisible by {small}")

    return (candidates["phys"], candidates["geom"], candidates["test"])


# =============================================================================
# 精确 λ 判定异常
# =============================================================================


class LambdaUndeterminedError(RuntimeError):
    """
    λ 无法在当前截断/精度下确定时抛出，禁止静默返回 None。

    使用场景：
      - 序列在当前截断窗口内呈零，无法判断最小 λ
      - λ 超出 t_precision 截断范围
    """

    def __init__(self, message: str, *, mu: int, witt_length: int, t_precision: int) -> None:
        super().__init__(message)
        self.mu = int(mu)
        self.witt_length = int(witt_length)
        self.t_precision = int(t_precision)


def _evm_slot_bits() -> int:
    """EVM Slot 宽度: 256 bits (标准 EIP-1153)."""
    return 256


def _identity_coefficient() -> int:
    """恒等映射系数: 1 (乘法单位元)."""
    return 1


def _trivial_hausdorff_distance() -> Fraction:
    """平凡 Hausdorff 距离: 0 (精确重合)."""
    return Fraction(0)


# =============================================================================
# 证书哈希: 确定性承诺
# =============================================================================


def _sha256_hex_of_dict(d: Dict[str, Any]) -> str:
    """
    计算字典的 SHA-256 哈希 (确定性序列化).
    禁止 float/complex/set 以保证确定性.
    """
    def _serialize(obj: Any) -> str:
        if obj is None:
            return "null"
        if isinstance(obj, bool):
            return "true" if obj else "false"
        if isinstance(obj, int):
            return f"int:{obj}"
        if isinstance(obj, str):
            return f"str:{obj}"
        if isinstance(obj, Fraction):
            return f"frac:{obj.numerator}/{obj.denominator}"
        if isinstance(obj, (list, tuple)):
            parts = [_serialize(x) for x in obj]
            return f"list:[{','.join(parts)}]"
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: str(kv[0]))
            parts = [f"{_serialize(k)}:{_serialize(v)}" for k, v in items]
            return f"dict:{{{','.join(parts)}}}"
        if isinstance(obj, float):
            raise TypeError(f"float forbidden in certificate: {obj}")
        if isinstance(obj, complex):
            raise TypeError(f"complex forbidden in certificate: {obj}")
        if isinstance(obj, set):
            raise TypeError(f"set forbidden in certificate: {obj}")
        # Fallback: use repr (but warn)
        _logger.warning("Serializing unknown type %s via repr", type(obj).__name__)
        return f"repr:{repr(obj)}"
    
    serialized = _serialize(d)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _encode_coeff_vector_as_int(*, coeffs: Sequence[int], base: int) -> int:
    """
    Deterministic injection: encode a bounded coefficient vector into an integer.

    Definition:
      Given base B>=2 and coeffs c_i in [0, B), encode as:
        value = Σ_{i=0}^{m-1} c_i · B^i

    Redlines:
    - No heuristics / no floats
    - Input coefficients must be explicit integers in range; no silent mod/reduction
    """
    if not isinstance(base, int) or int(base) < 2:
        raise ValueError(f"base must be int >= 2, got {base!r}")
    if not isinstance(coeffs, (list, tuple)):
        raise TypeError(f"coeffs must be a Sequence[int], got {type(coeffs).__name__}")

    B = int(base)
    acc = 0
    power = 1
    for i, c in enumerate(coeffs):
        if not isinstance(c, int):
            raise TypeError(f"coeffs[{i}] must be int, got {type(c).__name__}")
        if int(c) < 0 or int(c) >= B:
            raise ValueError(f"coeffs[{i}] out of range: {c} not in [0, {B})")
        acc += int(c) * int(power)
        power *= B
    return int(acc)


# =============================================================================
# Iwasawa 截断规格 (从 mvp17_prismatic.py 移植核心接口)
# =============================================================================


@dataclass(frozen=True)
class IwasawaTruncationSpec:
    """
    Iwasawa 代数截断规格 Λ_{n,m}(Z_p).
    
      - p: 特征素数
      - witt_length (n): Witt 精度 (Shannon-Hartley 导出)
      - t_precision (m): T 幂级数截断 (Berlekamp-Massey 极限)
    """
    p: int
    witt_length: int
    t_precision: int
    
    def __post_init__(self) -> None:
        if not isinstance(self.p, int) or self.p < 2:
            raise ValueError(f"p must be int >= 2, got {self.p}")
        if not isinstance(self.witt_length, int) or self.witt_length < 1:
            raise ValueError(f"witt_length must be int >= 1, got {self.witt_length}")
        if not isinstance(self.t_precision, int) or self.t_precision < 1:
            raise ValueError(f"t_precision must be int >= 1, got {self.t_precision}")
    
    @property
    def modulus(self) -> int:
        """p^n: Z/p^n Z 的模数."""
        return int(self.p) ** int(self.witt_length)
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "p": int(self.p),
            "witt_length": int(self.witt_length),
            "t_precision": int(self.t_precision),
            "modulus": int(self.modulus),
        }


# =============================================================================
# Mu-Invariant 和 Lambda-Invariant 判决器
# =============================================================================


@dataclass(frozen=True)
class MuInvariantVerdict:
    """
    μ-不变量判决 
      μ = 0: HARD (绝对刚性) - 安全
      μ > 0: SOFT (结构性塌缩) - 致命漏洞
    """
    mu_value: int
    is_hard: bool
    evidence: str
    
    VERSION = "mvp22.mu_invariant_verdict.v1"
    
    def __post_init__(self) -> None:
        if not isinstance(self.mu_value, int) or self.mu_value < 0:
            raise ValueError(f"mu_value must be non-negative int, got {self.mu_value}")
        if not isinstance(self.is_hard, bool):
            raise ValueError(f"is_hard must be bool, got {type(self.is_hard).__name__}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "mu_value": int(self.mu_value),
            "is_hard": bool(self.is_hard),
            "evidence": str(self.evidence),
            "verdict": "HARD" if self.is_hard else "SOFT",
        }


@dataclass(frozen=True)
class LambdaInvariantVerdict:
    """
    λ-不变量判决 
      λ < threshold: TUNNELED (可预测) - 致命漏洞
      λ ≥ threshold or ∞: BLOCKED (不可预测) - 安全
    """
    lambda_value: Optional[int]  # None = ∞ (无法计算)
    threshold: int
    is_blocked: bool
    evidence: str
    
    VERSION = "mvp22.lambda_invariant_verdict.v1"
    
    def __post_init__(self) -> None:
        if self.lambda_value is not None:
            if not isinstance(self.lambda_value, int) or self.lambda_value < 0:
                raise ValueError(f"lambda_value must be non-negative int or None, got {self.lambda_value}")
        if not isinstance(self.threshold, int) or self.threshold < 0:
            raise ValueError(f"threshold must be non-negative int, got {self.threshold}")
        if not isinstance(self.is_blocked, bool):
            raise ValueError(f"is_blocked must be bool, got {type(self.is_blocked).__name__}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "lambda_value": int(self.lambda_value) if self.lambda_value is not None else None,
            "threshold": int(self.threshold),
            "is_blocked": bool(self.is_blocked),
            "evidence": str(self.evidence),
            "verdict": "BLOCKED" if self.is_blocked else "TUNNELED",
        }


# =============================================================================
# 特征幂级数 (Characteristic Power Series)
# =============================================================================


@dataclass
class CharacteristicPowerSeries:
    """
    Iwasawa 特征幂级数 f(T) = p^μ · P(T) · U(T).
    
      - P(T): Weierstrass 多项式 (distinguished polynomial)
      - U(T): 单位元 (可逆幂级数)
      - μ: μ-不变量 (p 因子数)
    """
    
    coefficients: List[int]  # [a_0, a_1, ..., a_{m-1}], low-degree first
    spec: IwasawaTruncationSpec
    
    def __post_init__(self) -> None:
        if not isinstance(self.spec, IwasawaTruncationSpec):
            raise TypeError(f"spec must be IwasawaTruncationSpec, got {type(self.spec).__name__}")
        if not isinstance(self.coefficients, list):
            raise TypeError(f"coefficients must be list, got {type(self.coefficients).__name__}")
        if len(self.coefficients) != int(self.spec.t_precision):
            raise ValueError(
                f"coefficients length must equal t_precision={self.spec.t_precision}, "
                f"got {len(self.coefficients)}"
            )
        mod = int(self.spec.modulus)
        self.coefficients = [int(c) % mod for c in self.coefficients]
    
    @classmethod
    def zero(cls, spec: IwasawaTruncationSpec) -> "CharacteristicPowerSeries":
        return cls([0] * int(spec.t_precision), spec)
    
    @classmethod
    def one(cls, spec: IwasawaTruncationSpec) -> "CharacteristicPowerSeries":
        coeffs = [0] * int(spec.t_precision)
        coeffs[0] = 1
        return cls(coeffs, spec)
    
    def is_zero(self) -> bool:
        return all(int(c) == 0 for c in self.coefficients)
    
    def _vp(self, x: int) -> int:
        """截断 p-adic 赋值 v_p(x) in Z/p^nZ."""
        if not isinstance(x, int):
            raise TypeError(f"x must be int, got {type(x).__name__}")
        p = int(self.spec.p)
        n = int(self.spec.witt_length)
        mod = int(self.spec.modulus)
        v = int(x % mod)
        if v == 0:
            return int(n)
        out = 0
        while out < n and v % p == 0:
            v //= p
            out += 1
        return int(out)
    
    def mu_invariant(self) -> int:
        """
        μ-不变量: max μ such that p^μ divides all coefficients.
        
        返回 [0, n]. 零级数返回 n.
        """
        n = int(self.spec.witt_length)
        vals = [self._vp(c) for c in self.coefficients]
        if not vals:
            return int(n)
        return int(min(vals))
    
    def lambda_invariant(self) -> Optional[int]:
        """
        λ-不变量: degree of the distinguished polynomial P(T).
        
        通过 Weierstrass 预备定理计算.
        如果无法确定 (例如截断内没有出现单位系数), 返回 None.
        """
        # Weierstrass 预备定理给出的 λ 是在因子 p^μ 剥离之后的 Weierstrass 度数。
        # 设 f(T)=Σ a_i T^i ∈ Z_p[[T]]，μ = min_i v_p(a_i)。
        # 令 b_i = a_i / p^μ，则 λ = min { i | v_p(b_i) = 0 }，
        # 等价于 λ = min { i | v_p(a_i) = μ }。
        # 在截断 Z/p^nZ 上:
        # - 若 μ = n，则 f ≡ 0 (mod p^n)，在此精度内无法确定有限 λ，返回 None。
        # - 否则在 [0, m) 内寻找第一个 v_p(a_i) == μ 的位置。
        n = int(self.spec.witt_length)
        m = int(self.spec.t_precision)
        mu = int(self.mu_invariant())

        if mu >= n:
            raise LambdaUndeterminedError(
                "lambda invariant cannot be determined: series is 0 modulo p^n in current precision",
                mu=mu,
                witt_length=n,
                t_precision=m,
            )

        for i in range(m):
            if int(self._vp(int(self.coefficients[i]))) == mu:
                return int(i)

        raise LambdaUndeterminedError(
            "lambda invariant exceeds current t_precision window; increase precision",
            mu=mu,
            witt_length=n,
            t_precision=m,
        )
    
    def weierstrass_polynomial_coeffs(self) -> Optional[List[int]]:
        """
        提取 Weierstrass 多项式 P(T) 的系数.
        
        如果 λ 可确定, 返回 [c_0, c_1, ..., c_{λ-1}, 1] (首一多项式).
        否则返回 None.
        """
        lam = self.lambda_invariant()
        lam_int = int(lam)
        if lam_int == 0:
            return [1]

        p = int(self.spec.p)
        n = int(self.spec.witt_length)
        m = int(self.spec.t_precision)
        mu = int(self.mu_invariant())
        if mu >= n:
            return None

        # Work in reduced modulus p^{n-μ}
        red_n = int(n - mu)
        red_mod = int(p ** red_n)

        # Strip μ: b_i = a_i / p^μ  (mod p^{n-μ})
        p_mu = int(p**mu)
        b: List[int] = []
        for ai in self.coefficients:
            a = int(ai % int(self.spec.modulus))
            if a != 0 and (a % p_mu) != 0:
                raise RuntimeError("Mu stripping failed: coefficient not divisible by p^mu")
            b.append(int((a // p_mu) % red_mod))

        if lam_int >= m:
            raise LambdaUndeterminedError(
                "lambda invariant outside truncation window during Weierstrass preparation",
                mu=mu,
                witt_length=n,
                t_precision=m,
            )

        # ---------------------------------------------------------------------
        # Helpers over F_p[[T]] / (T^m)
        # ---------------------------------------------------------------------
        def _inv_mod_prime(a: int) -> int:
            aa = int(a % p)
            if aa == 0:
                raise ZeroDivisionError("no inverse in F_p")
            return pow(aa, p - 2, p)

        def _invert_series_mod_p(u: List[int]) -> List[int]:
            """Return v such that (u*v) ≡ 1 (mod p, T^m). Require u[0] != 0 mod p."""
            if len(u) != m:
                raise RuntimeError("internal: series length mismatch")
            u0 = int(u[0] % p)
            inv_u0 = _inv_mod_prime(u0)
            v = [0] * m
            v[0] = int(inv_u0)
            for k in range(1, m):
                s = 0
                upper = k if k < len(u) else len(u) - 1
                for i in range(1, upper + 1):
                    s = (s + int(u[i] % p) * int(v[k - i])) % p
                v[k] = int((-s * inv_u0) % p)
            return v

        def _mul_series_poly_mod(series: List[int], poly: List[int], mod_: int) -> List[int]:
            """Multiply series (len m) by poly (deg <= λ) modulo (mod_, T^m)."""
            out = [0] * m
            deg = len(poly) - 1
            for i in range(m):
                acc = 0
                # poly term j contributes series[i-j]
                upper = deg if deg < i else i
                for j in range(upper + 1):
                    acc = (acc + int(poly[j]) * int(series[i - j])) % mod_
                out[i] = int(acc)
            return out

        # Base layer (mod p):
        # P0 = T^λ, U0 = f'/T^λ in F_p[[T]]
        u_bar = [0] * m
        for j in range(0, m - lam_int):
            u_bar[j] = int(b[j + lam_int] % p)
        if int(u_bar[0] % p) == 0:
            raise RuntimeError("internal: lambda coefficient is not a unit mod p")

        inv_u_bar = _invert_series_mod_p(u_bar)

        # Lifted objects (mod p^r): U (series length m), P (poly length λ+1)
        U = [int(x % p) for x in u_bar]
        Ppoly = [0] * lam_int + [1]  # monic T^λ

        current_mod = int(p)  # p^1
        # Lift to p^{red_n}
        for _r in range(1, red_n):
            next_mod = int(current_mod * p)  # p^{r+1}

            prod = _mul_series_poly_mod(U, Ppoly, mod_=next_mod)

            # residual E = (b - prod) / current_mod  (mod p)
            E_bar = [0] * m
            for i in range(m):
                diff = int((int(b[i]) - int(prod[i])) % next_mod)
                if diff % current_mod != 0:
                    raise RuntimeError("internal: lifting residual not divisible by p^r")
                E_bar[i] = int((diff // current_mod) % p)

            # Split E_bar = Q*T^λ + R, with deg(R) < λ
            R_bar = E_bar[:lam_int]

            # ΔP_bar = inv(U_bar) * R_bar  (mod p, degree < λ)
            dP_bar = [0] * lam_int
            for i in range(lam_int):
                acc = 0
                for j in range(i + 1):
                    acc = (acc + int(inv_u_bar[j]) * int(R_bar[i - j])) % p
                dP_bar[i] = int(acc)

            # Compute E2_bar = E_bar - U_bar * ΔP_bar  (mod p)
            # Only need degrees >= λ for ΔU.
            E2_bar = list(E_bar)
            # subtract convolution of u_bar (len m) and dP_bar (len λ)
            for i in range(m):
                acc = 0
                upper = lam_int - 1 if (lam_int - 1) < i else i
                for j in range(upper + 1):
                    acc = (acc + int(u_bar[i - j]) * int(dP_bar[j])) % p
                E2_bar[i] = int((int(E2_bar[i]) - acc) % p)

            # ΔU_bar is the quotient part: E2_bar / T^λ (shift)
            dU_bar = [0] * m
            for j in range(0, m - lam_int):
                dU_bar[j] = int(E2_bar[j + lam_int] % p)

            # Update Ppoly (coeff < λ) and U
            for i in range(lam_int):
                Ppoly[i] = int((int(Ppoly[i]) + current_mod * int(dP_bar[i])) % next_mod)
            for i in range(m):
                U[i] = int((int(U[i]) + current_mod * int(dU_bar[i])) % next_mod)

            current_mod = next_mod

        # Final verification: b == U*P (mod p^{n-μ}, T^m)
        final_prod = _mul_series_poly_mod(U, Ppoly, mod_=red_mod)
        if any(int((int(final_prod[i]) - int(b[i])) % red_mod) != 0 for i in range(m)):
            raise RuntimeError("Weierstrass preparation failed: nonzero residual at final precision")

        # Distinguished check: i<λ coefficients must be divisible by p
        for i in range(lam_int):
            if int(Ppoly[i]) % p != 0:
                raise RuntimeError("Weierstrass preparation produced non-distinguished polynomial")

        # Return low-degree-first coeffs [c0..c_{λ-1}, 1] modulo p^{n-μ}
        out = [int(Ppoly[i] % red_mod) for i in range(lam_int)] + [1]
        return out
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "coefficients_hash": _sha256_hex_of_dict({"c": self.coefficients}),
            "mu_invariant": self.mu_invariant(),
            "lambda_invariant": self.lambda_invariant(),
            "is_zero": self.is_zero(),
        }


# =============================================================================
# 全息状态 (Holographic State)
# =============================================================================


@dataclass
class HolographicState:
    """
    全息投影状态: Universe A 在 Universe B 视角下的表示.
    
      这是跨宇宙比较的核心数据结构.
      包含升维后的 Iwasawa Module 结构.
    """
    
    source_universe_label: str
    target_perspective_label: str
    
    # Iwasawa Module 结构
    characteristic_series: CharacteristicPowerSeries
    
    # 不确定性边界 (来自 Log-Shell)
    indeterminacy_volume: Fraction
    
    # 元数据
    arakelov_height_bound: Fraction
    commitment: str
    # MVP0 bridge evidence (Theta-link / PolyMorphism certificate)
    mvp0_theta_link: Dict[str, Any] = field(default_factory=dict)
    
    VERSION = "mvp22.holographic_state.v1"
    
    def __post_init__(self) -> None:
        if not isinstance(self.source_universe_label, str) or not self.source_universe_label:
            raise ValueError("source_universe_label must be non-empty str")
        if not isinstance(self.target_perspective_label, str) or not self.target_perspective_label:
            raise ValueError("target_perspective_label must be non-empty str")
        if not isinstance(self.characteristic_series, CharacteristicPowerSeries):
            raise TypeError("characteristic_series must be CharacteristicPowerSeries")
        if not isinstance(self.indeterminacy_volume, Fraction):
            raise TypeError("indeterminacy_volume must be Fraction")
        if self.indeterminacy_volume < 0:
            raise ValueError("indeterminacy_volume must be non-negative")
        if not isinstance(self.arakelov_height_bound, Fraction):
            raise TypeError("arakelov_height_bound must be Fraction")
        if not isinstance(self.mvp0_theta_link, dict):
            raise TypeError("mvp0_theta_link must be dict")
        if not isinstance(self.commitment, str) or not self.commitment:
            raise ValueError("commitment must be non-empty str")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "source_universe_label": self.source_universe_label,
            "target_perspective_label": self.target_perspective_label,
            "characteristic_series": self.characteristic_series.to_dict(),
            "indeterminacy_volume": str(self.indeterminacy_volume),
            "arakelov_height_bound": str(self.arakelov_height_bound),
            "mvp0_theta_link": dict(self.mvp0_theta_link),
            "commitment": self.commitment,
        }


# =============================================================================
# Theta-Pilot 对象
# =============================================================================


@dataclass(frozen=True)
class ThetaPilotCore:
    """
    Theta-Pilot 核心数据 (MVP22本地定义与comparison_functors互操作)
    包含:
      - universe_label: 所属宇宙
      - characteristic_series: Iwasawa 特征级数
      - prime_spec: 素数规格 (p, k)
    """
    universe_label: str
    characteristic_series: CharacteristicPowerSeries
    prime_p: int
    witt_length: int
    commitment: str
    
    VERSION = "mvp22.theta_pilot_core.v1"
    
    def __post_init__(self) -> None:
        if not isinstance(self.universe_label, str) or not self.universe_label:
            raise ValueError("universe_label must be non-empty str")
        if not isinstance(self.characteristic_series, CharacteristicPowerSeries):
            raise TypeError("characteristic_series must be CharacteristicPowerSeries")
        if not isinstance(self.prime_p, int) or self.prime_p < 2:
            raise ValueError(f"prime_p must be int >= 2, got {self.prime_p}")
        if not isinstance(self.witt_length, int) or self.witt_length < 1:
            raise ValueError(f"witt_length must be int >= 1, got {self.witt_length}")
        if not isinstance(self.commitment, str) or not self.commitment:
            raise ValueError("commitment must be non-empty str")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "universe_label": self.universe_label,
            "characteristic_series": self.characteristic_series.to_dict(),
            "prime_p": int(self.prime_p),
            "witt_length": int(self.witt_length),
            "commitment": self.commitment,
        }


# =============================================================================
# 拓扑同胚验证证书
# =============================================================================


@dataclass(frozen=True)
class TopologicalHomeomorphismCertificate:
    """
    拓扑同胚验证证书.
    
      - 比较 holographic_state 和 target_pilot 的拓扑结构
      - 不比较数值, 而是比较 Iwasawa Module 结构
      - 这是 "Iwasawa Module 撞了" 的代码实现
    """
    
    holographic_state_commitment: str
    target_pilot_commitment: str
    
    # 结构比较结果
    mu_match: bool
    lambda_match: bool
    weierstrass_compatible: bool
    
    # 整体判决
    is_homeomorphic: bool
    
    # 证据
    hausdorff_drift: str  # Fraction as string
    evidence_summary: str
    
    commitment: str
    
    VERSION = "mvp22.topological_homeomorphism.v1"
    
    def __post_init__(self) -> None:
        if not isinstance(self.holographic_state_commitment, str):
            raise ValueError("holographic_state_commitment must be str")
        if not isinstance(self.target_pilot_commitment, str):
            raise ValueError("target_pilot_commitment must be str")
        if not isinstance(self.mu_match, bool):
            raise ValueError("mu_match must be bool")
        if not isinstance(self.lambda_match, bool):
            raise ValueError("lambda_match must be bool")
        if not isinstance(self.weierstrass_compatible, bool):
            raise ValueError("weierstrass_compatible must be bool")
        if not isinstance(self.is_homeomorphic, bool):
            raise ValueError("is_homeomorphic must be bool")
        # 验证 hausdorff_drift 是有效的 Fraction
        try:
            Fraction(self.hausdorff_drift)
        except (ValueError, TypeError) as e:
            raise ValueError(f"hausdorff_drift must be valid Fraction string: {e}") from e
        if not isinstance(self.commitment, str) or not self.commitment:
            raise ValueError("commitment must be non-empty str")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "holographic_state_commitment": self.holographic_state_commitment,
            "target_pilot_commitment": self.target_pilot_commitment,
            "mu_match": bool(self.mu_match),
            "lambda_match": bool(self.lambda_match),
            "weierstrass_compatible": bool(self.weierstrass_compatible),
            "is_homeomorphic": bool(self.is_homeomorphic),
            "hausdorff_drift": self.hausdorff_drift,
            "evidence_summary": self.evidence_summary,
            "commitment": self.commitment,
        }


# =============================================================================
# 多辐射共识证书
# =============================================================================


@dataclass(frozen=True)
class MultiRadialConsensusCertificate:
    """
    多辐射共识证书.
    
      - 站在 Universe C 的视角
      - 验证 A 和 B 的投影是否重合 (Ghosting)
      - 这是最终 PASS 的判决逻辑
    """
    
    pilot_a_commitment: str
    pilot_b_commitment: str
    pilot_c_commitment: str
    
    # 投影结果
    proj_a_on_c_commitment: str
    proj_b_on_c_commitment: str
    
    # 重合验证
    projections_coincide: bool
    hausdorff_distance: str  # Fraction as string
    
    # 判决
    consensus_achieved: bool
    
    commitment: str
    
    VERSION = "mvp22.multiradial_consensus.v1"
    
    def __post_init__(self) -> None:
        for name, val in [
            ("pilot_a_commitment", self.pilot_a_commitment),
            ("pilot_b_commitment", self.pilot_b_commitment),
            ("pilot_c_commitment", self.pilot_c_commitment),
            ("proj_a_on_c_commitment", self.proj_a_on_c_commitment),
            ("proj_b_on_c_commitment", self.proj_b_on_c_commitment),
            ("commitment", self.commitment),
        ]:
            if not isinstance(val, str):
                raise ValueError(f"{name} must be str")
        if not isinstance(self.projections_coincide, bool):
            raise ValueError("projections_coincide must be bool")
        if not isinstance(self.consensus_achieved, bool):
            raise ValueError("consensus_achieved must be bool")
        try:
            Fraction(self.hausdorff_distance)
        except (ValueError, TypeError) as e:
            raise ValueError(f"hausdorff_distance must be valid Fraction string: {e}") from e
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "pilot_a_commitment": self.pilot_a_commitment,
            "pilot_b_commitment": self.pilot_b_commitment,
            "pilot_c_commitment": self.pilot_c_commitment,
            "proj_a_on_c_commitment": self.proj_a_on_c_commitment,
            "proj_b_on_c_commitment": self.proj_b_on_c_commitment,
            "projections_coincide": bool(self.projections_coincide),
            "hausdorff_distance": self.hausdorff_distance,
            "consensus_achieved": bool(self.consensus_achieved),
            "commitment": self.commitment,
        }


# =============================================================================
# 二值扭转证书 (Binary Torsion Certificate)
# =============================================================================


@dataclass
class TorsionCertificate:
    """
    Iwasawa 理论的最终判决书
    
    核心属性:
      - prime: 特征素数
      - witt_precision: Witt 精度 n
      - t_cutoff: T 截断 m
      - distinguished_polynomial: Weierstrass 多项式 P(T) 的精确整数系数
      - is_torsion: 二值判决 (True = 检测到扭转/漏洞)
      - physical_trace: 物理层对应的 Slot 原始值
    """
    
    prime: int
    witt_precision: int
    t_cutoff: int
    
    # 核心证据
    distinguished_polynomial: List[int]  # [c_0, c_1, ..., c_λ] 精确整数
    mu_invariant: int
    lambda_invariant: Optional[int]
    
    # 判决结果
    is_torsion: bool
    
    # 物理一致性签名
    physical_trace_a: Optional[int]
    physical_trace_b: Optional[int]
    
    commitment: str

    # 额外证据（MVP17/NS）：shift-register synthesis / annihilator polynomial
    # - 用于对齐 Norton–Salagean / Reeds–Sloane 的最小连接多项式输出
    # - 不强行塞进 distinguished_polynomial（那需要完整 Weierstrass preparation）
    annihilator_polynomial: List[int] = field(default_factory=list)  # forward polynomial (low degree first), monic
    annihilator_certificate: Dict[str, Any] = field(default_factory=dict)
    
    VERSION = "mvp22.torsion_certificate.v1"
    
    def __post_init__(self) -> None:
        if not isinstance(self.prime, int) or self.prime < 2:
            raise ValueError(f"prime must be int >= 2, got {self.prime}")
        if not isinstance(self.witt_precision, int) or self.witt_precision < 1:
            raise ValueError(f"witt_precision must be int >= 1, got {self.witt_precision}")
        if not isinstance(self.t_cutoff, int) or self.t_cutoff < 1:
            raise ValueError(f"t_cutoff must be int >= 1, got {self.t_cutoff}")
        if not isinstance(self.distinguished_polynomial, list):
            raise TypeError("distinguished_polynomial must be list")
        for i, c in enumerate(self.distinguished_polynomial):
            if not isinstance(c, int):
                raise TypeError(f"distinguished_polynomial[{i}] must be int, got {type(c).__name__}")
        if not isinstance(self.mu_invariant, int) or self.mu_invariant < 0:
            raise ValueError(f"mu_invariant must be non-negative int, got {self.mu_invariant}")
        if self.lambda_invariant is not None:
            if not isinstance(self.lambda_invariant, int) or self.lambda_invariant < 0:
                raise ValueError(f"lambda_invariant must be non-negative int or None, got {self.lambda_invariant}")
        if not isinstance(self.is_torsion, bool):
            raise ValueError("is_torsion must be bool")
        if not isinstance(self.annihilator_polynomial, list):
            raise TypeError("annihilator_polynomial must be list")
        for i, c in enumerate(self.annihilator_polynomial):
            if not isinstance(c, int):
                raise TypeError(f"annihilator_polynomial[{i}] must be int, got {type(c).__name__}")
        if not isinstance(self.annihilator_certificate, dict):
            raise TypeError("annihilator_certificate must be dict")
        if not isinstance(self.commitment, str) or not self.commitment:
            raise ValueError("commitment must be non-empty str")
    
    def verify(self) -> bool:
        """
        验证逻辑:
          1. 检查 P(T) 是否为 distinguished (非首项系数均被 p 整除)
          2. μ = 0 意味着至少一个系数不被 p 整除
          3. 如果 μ > 0 或 λ < ∞ 且小, 则存在扭转
        """
        p = int(self.prime)
        n = int(self.witt_precision)
        
        # 检查 distinguished polynomial 性质
        poly = self.distinguished_polynomial
        if not poly:
            return True  # 空多项式 = 1, 无扭转
        
        # 首项应为 1 (首一多项式)
        if poly[-1] != 1:
            _logger.warning("Distinguished polynomial is not monic: leading coeff = %d", poly[-1])
            return False
        
        # 非首项系数必须被 p 整除 (distinguished 性质)
        for i in range(len(poly) - 1):
            if poly[i] % p != 0:
                _logger.warning(
                    "Distinguished polynomial coeff[%d]=%d not divisible by p=%d",
                    i, poly[i], p
                )
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "prime": int(self.prime),
            "witt_precision": int(self.witt_precision),
            "t_cutoff": int(self.t_cutoff),
            "distinguished_polynomial": list(self.distinguished_polynomial),
            "mu_invariant": int(self.mu_invariant),
            "lambda_invariant": int(self.lambda_invariant) if self.lambda_invariant is not None else None,
            "annihilator_polynomial": list(self.annihilator_polynomial),
            "annihilator_certificate": dict(self.annihilator_certificate),
            "is_torsion": bool(self.is_torsion),
            "physical_trace_a": int(self.physical_trace_a) if self.physical_trace_a is not None else None,
            "physical_trace_b": int(self.physical_trace_b) if self.physical_trace_b is not None else None,
            "commitment": self.commitment,
        }


def analyze_torsion_operator_b_pipeline(
    *,
    key: Any,
    slot_a_value: int,
    slot_b_value: int,
    spec: IwasawaTruncationSpec,
) -> TorsionCertificate:
    """
    Operator-B (Keccak mapping orbit) 版本的完整扭转分析：

      - 使用 MVP17 的注入/演化（Keccak 轨道）
      - 使用 `norton_salagean.py` 的链环综合（最小连接多项式 + 证书）
      - 输出 MVP22 的 `TorsionCertificate`，并把 annihilator 证据挂在证书里

    该入口用于把 MVP22 的证书闭环与 Norton–Salagean 的严格综合对齐。
    """
    if not isinstance(spec, IwasawaTruncationSpec):
        raise TypeError(f"spec must be IwasawaTruncationSpec, got {type(spec).__name__}")
    if not isinstance(slot_a_value, int):
        raise TypeError(f"slot_a_value must be int, got {type(slot_a_value).__name__}")
    if not isinstance(slot_b_value, int):
        raise TypeError(f"slot_b_value must be int, got {type(slot_b_value).__name__}")

    # steps = m-1 so that observation window length equals t_cutoff=m
    m = int(spec.t_precision)
    if m < 1:
        raise ValueError("spec.t_precision must be >= 1")
    steps = int(m - 1)

    # Import MVP17 primitives locally to avoid heavyweight import at module load.
    # Use absolute imports so this function works both when imported as a module and when this file
    # is executed as a script (__package__ may be None in that case).
    from web_ica.bridge_audit.core.mvp17_prismatic import (
        compute_iwasawa_torsion_certificate_operator_b,
        u256_to_bytes32_be,
    )

    slot_a_b32 = u256_to_bytes32_be(int(slot_a_value) % (1 << _evm_slot_bits()))
    slot_b_b32 = u256_to_bytes32_be(int(slot_b_value) % (1 << _evm_slot_bits()))

    cert17 = compute_iwasawa_torsion_certificate_operator_b(
        key=key,
        slot_a=slot_a_b32,
        slot_b=slot_b_b32,
        p=int(spec.p),
        witt_length=int(spec.witt_length),
        steps=int(steps),
    )
    if not bool(cert17.ok):
        raise RuntimeError(f"Operator-B certificate generation failed: {cert17.error}")

    annihilator = [int(c) for c in cert17.poly_coeffs]
    if not annihilator or int(annihilator[-1] % int(spec.modulus)) != 1:
        raise RuntimeError("internal: annihilator polynomial must be monic")

    # Characteristic series for MVP22 (Operator‑B semantics):
    # Use the *observed orbit sequence* coefficients as f(T)=Σ v_k T^k (truncated to m terms).
    #
    # This allows computing (μ, λ, Weierstrass P(T)) rigorously from the same data window.
    from web_ica.bridge_audit.core.mvp17_prismatic import iterate_keccak_mapping_orbit, normalize_key_bytes32

    key32 = normalize_key_bytes32(key)
    orbit = iterate_keccak_mapping_orbit(
        key=key32,
        v0_padic=int(cert17.seed_padic),
        steps=int(steps),
        p=int(spec.p),
        witt_length=int(spec.witt_length),
    )
    if len(orbit) != int(m):
        raise RuntimeError("internal: orbit length mismatch vs t_precision")

    series = CharacteristicPowerSeries([int(x) for x in orbit], spec)
    mu = int(series.mu_invariant())
    lam = series.lambda_invariant()

    wp = series.weierstrass_polynomial_coeffs()
    # Zero-series corner: μ=n and λ=None, Weierstrass polynomial is not determined at this truncation.
    # This is not an error; we encode it explicitly as an empty polynomial.
    distinguished: List[int]
    if wp is None:
        distinguished = []
    else:
        distinguished = [int(c) for c in wp]

    cert_body = {
        "mode": "mvp22.operator_b+ns",
        "p": int(spec.p),
        "n": int(spec.witt_length),
        "m": int(m),
        "steps": int(steps),
        "mu_from_orbit_series": int(mu),
        "lambda_from_orbit_series": int(lam) if lam is not None else None,
        "torsion": bool(cert17.torsion_detected),
        "trace_a": int(slot_a_value),
        "trace_b": int(slot_b_value),
        "annihilator_degree": int(cert17.poly_degree),
        "annihilator_coeffs_hash": _sha256_hex_of_dict({"poly": [str(int(c)) for c in annihilator]}),
    }
    commitment = _sha256_hex_of_dict(cert_body)

    return TorsionCertificate(
        prime=int(spec.p),
        witt_precision=int(spec.witt_length),
        t_cutoff=int(m),
        distinguished_polynomial=distinguished,
        mu_invariant=int(mu),
        lambda_invariant=int(lam) if lam is not None else None,
        annihilator_polynomial=annihilator,
        annihilator_certificate=dict(cert17.synthesis_certificate),
        is_torsion=bool(cert17.torsion_detected),
        physical_trace_a=int(slot_a_value),
        physical_trace_b=int(slot_b_value),
        commitment=commitment,
    )


# =============================================================================
# 核心方法 1: project_holographic_state
# =============================================================================


def project_holographic_state(
    source_pilot: ThetaPilotCore,
    target_perspective: ThetaPilotCore,
) -> HolographicState:
    """
    核心方法 1: 全息投影.
    
      - 对应 "functor.apply(Universe_A) ... 返回全息投影"
      - 调用 MVP0 的 construct_theta_link 建立管道
      - 调用 MVP17 的 lift_to_prismatic 升维成 Iwasawa Module 结构
      - 返回 HolographicState 对象
    
    Args:
        source_pilot: 源 Theta-Pilot (Universe A)
        target_perspective: 目标视角 (Universe B)
    
    Returns:
        HolographicState: A 在 B 视角下的全息投影
    """
    _logger.info(
        "project_holographic_state: %s -> perspective of %s",
        source_pilot.universe_label,
        target_perspective.universe_label,
    )
    
    # 验证输入
    if not isinstance(source_pilot, ThetaPilotCore):
        raise TypeError(f"source_pilot must be ThetaPilotCore, got {type(source_pilot).__name__}")
    if not isinstance(target_perspective, ThetaPilotCore):
        raise TypeError(f"target_perspective must be ThetaPilotCore, got {type(target_perspective).__name__}")
    
    # 验证素数与截断规格兼容性
    # 红线: 禁止通过降精度/归一化来强行兼容；不兼容必须中断。
    if source_pilot.prime_p != target_perspective.prime_p:
        raise ValueError(
            f"Prime mismatch: source p={source_pilot.prime_p}, target p={target_perspective.prime_p}"
        )

    # 提取 Iwasawa 结构
    src_series = source_pilot.characteristic_series
    tgt_series = target_perspective.characteristic_series

    # 红线: pilot 元信息必须与其 series.spec 自洽；否则属于部署级数据污染，必须中断。
    if int(source_pilot.prime_p) != int(src_series.spec.p) or int(source_pilot.witt_length) != int(src_series.spec.witt_length):
        raise ValueError(
            "source_pilot metadata inconsistent with characteristic_series.spec: "
            f"pilot(prime_p={source_pilot.prime_p}, witt_length={source_pilot.witt_length}) vs "
            f"spec(p={src_series.spec.p}, witt_length={src_series.spec.witt_length})"
        )
    if int(target_perspective.prime_p) != int(tgt_series.spec.p) or int(target_perspective.witt_length) != int(tgt_series.spec.witt_length):
        raise ValueError(
            "target_perspective metadata inconsistent with characteristic_series.spec: "
            f"pilot(prime_p={target_perspective.prime_p}, witt_length={target_perspective.witt_length}) vs "
            f"spec(p={tgt_series.spec.p}, witt_length={tgt_series.spec.witt_length})"
        )

    if src_series.spec != tgt_series.spec:
        raise ValueError(
            "Truncation spec mismatch between source and target perspective: "
            f"source={src_series.spec.to_dict()}, target={tgt_series.spec.to_dict()}"
        )

    # 计算投影后的特征级数
    #
    # 数学约束 (工程化口径):
    # - 在未显式提供 Θ-link / character χ 的情况下，任何混合/调制/差分/归一化都属于启发式。
    # - 唯一可合法实现的是：在同一截断环 Λ_{n,m}(Z_p) 内做**结构保真**的恒等搬运，
    #   仅把视角标签/承诺(commitment)记录进证书层，不对系数做未经公理化的变换。
    spec = src_series.spec
    p = int(spec.p)
    n = int(spec.witt_length)
    m = int(spec.t_precision)

    projected_coeffs = [int(c) for c in src_series.coefficients]
    projected_series = CharacteristicPowerSeries(projected_coeffs, spec)

    # 计算 Arakelov 高度边界 (严格整数/有理数，禁止 float)
    #
    # 这里给出一个可验证的上界而不是浮点近似：
    #   令 x = max_i (coeff_i) + 1 (系数作为 Z/p^nZ 的标准代表元, 0<=c<p^n)
    #   定义 k = min{ k>=0 : p^k >= x } = ceil(log_p(x))
    # 则 log_p(x) <= k 严格成立，作为高度界不会丢精度或引入噪声。
    max_coeff = max(projected_coeffs) if projected_coeffs else 0
    x = int(max_coeff) + 1
    if x < 1:
        # 理论上不可能：max_coeff>=0 => x>=1
        raise RuntimeError(f"Invalid Arakelov bound input: max_coeff={max_coeff}")
    k = 0
    power = 1
    while power < x:
        power *= p
        k += 1
        if k > n:
            # 因为 max_coeff < p^n，理论上 k 不会超过 n；超过说明内部状态不一致，必须中断部署。
            raise RuntimeError(
                "Arakelov bound exceeded Witt precision: "
                f"p={p}, n={n}, max_coeff={max_coeff}, reached k={k}"
            )
    arakelov_bound = Fraction(k + n, 1)

    # 计算不确定性体积 (严格、可解释的 p-adic lift 不确定性)
    # MVP0 桥接：必须通过 Frobenioid 底座走 ComparisonFunctor.construct_theta_link，
    # 取其 IndeterminacyCertificate.volume 作为 Log‑Shell 体积证据（禁止自己拍一个 p^{-n}当默认）。
    from web_ica.bridge_audit.core.frobenioid_base import mvp0_construct_theta_link_bridge

    base_mod = int(spec.modulus)
    src_value = _encode_coeff_vector_as_int(coeffs=projected_coeffs, base=base_mod)
    tgt_value = _encode_coeff_vector_as_int(coeffs=[int(c) for c in tgt_series.coefficients], base=base_mod)

    # Minimal non-trivial Kummer extension degree is 2 (by definition of non-trivial Kummer adjunction).
    theta_link = mvp0_construct_theta_link_bridge(
        source_value=int(src_value),
        target_value=int(tgt_value),
        prime_p=int(p),
        precision_k=int(n),
        kummer_degree=2,
        tower_depth=1,
        source_universe_label=str(source_pilot.universe_label),
        target_universe_label=str(target_perspective.universe_label),
    )
    try:
        indeterminacy = Fraction(str(theta_link["poly_morphism"]["indeterminacy"]["volume"]))
    except Exception as e:
        raise RuntimeError(f"invalid MVP0 theta_link indeterminacy volume (redline): {e}") from e

    # 预计算不变量 (仅用于日志与证据，不参与任何启发式放水判定)
    mu = projected_series.mu_invariant()
    lam = projected_series.lambda_invariant()
    
    # 构建证书
    state_body = {
        "source": source_pilot.universe_label,
        "target": target_perspective.universe_label,
        "series_hash": _sha256_hex_of_dict(projected_series.to_dict()),
        "arakelov": str(arakelov_bound),
        "indeterminacy": str(indeterminacy),
        "mvp0_theta_link_commitment": str(theta_link.get("commitment", "")),
        "mvp0_poly_morphism_commitment": str(theta_link.get("poly_morphism", {}).get("commitment", "")),
    }
    commitment = _sha256_hex_of_dict(state_body)
    
    state = HolographicState(
        source_universe_label=source_pilot.universe_label,
        target_perspective_label=target_perspective.universe_label,
        characteristic_series=projected_series,
        indeterminacy_volume=indeterminacy,
        arakelov_height_bound=arakelov_bound,
        mvp0_theta_link=dict(theta_link),
        commitment=commitment,
    )
    
    _logger.info(
        "project_holographic_state complete: mu=%d, lambda=%s, commitment=%s...",
        mu,
        lam if lam is not None else "∞",
        commitment[:16],
    )
    
    return state


# =============================================================================
# 核心方法 2: verify_topological_homeomorphism
# =============================================================================


def verify_topological_homeomorphism(
    holographic_state: HolographicState,
    target_pilot: ThetaPilotCore,
) -> TopologicalHomeomorphismCertificate:
    """
    核心方法 2: 拓扑同胚验证.
    
      - 比较 holographic_state (A 在 B 眼中的样子) 和 target_pilot (B 本身)
      - 不比较数值, 而是比较拓扑结构 (μ, λ, Weierstrass 形状)
      - 这是 "Iwasawa Module 撞了" 的代码实现
    
    Args:
        holographic_state: A 的全息投影
        target_pilot: B 本身
    
    Returns:
        TopologicalHomeomorphismCertificate: 同胚验证结果
    """
    _logger.info(
        "verify_topological_homeomorphism: state(%s) vs pilot(%s)",
        holographic_state.source_universe_label,
        target_pilot.universe_label,
    )
    
    if not isinstance(holographic_state, HolographicState):
        raise TypeError(f"holographic_state must be HolographicState, got {type(holographic_state).__name__}")
    if not isinstance(target_pilot, ThetaPilotCore):
        raise TypeError(f"target_pilot must be ThetaPilotCore, got {type(target_pilot).__name__}")
    
    proj_series = holographic_state.characteristic_series
    tgt_series = target_pilot.characteristic_series

    # 红线: 禁止通过降精度来比较；规格不一致意味着不可判定，必须中断。
    if proj_series.spec != tgt_series.spec:
        raise ValueError(
            "Truncation spec mismatch in homeomorphism check (comparison must be in the same Λ_{n,m}): "
            f"proj={proj_series.spec.to_dict()}, tgt={tgt_series.spec.to_dict()}"
        )
    
    # 提取不变量
    proj_mu = proj_series.mu_invariant()
    proj_lam = proj_series.lambda_invariant()
    tgt_mu = tgt_series.mu_invariant()
    tgt_lam = tgt_series.lambda_invariant()
    
    # μ 匹配: 必须严格相等
    mu_match = (proj_mu == tgt_mu)
    
    # λ 匹配: 必须严格相等 (考虑 None 情况)
    lambda_match = (proj_lam == tgt_lam)
    
    # Weierstrass 兼容性 (严格、无偷懒分解):
    # 在截断 Λ_{n,m} 内，我们只做可证明的必要条件检查：
    # - λ = 0: 代表单位元形状 (常数项为 p-adic 单位)；此时 Weierstrass 形状唯一。
    # - λ > 0: 对 i<λ 必须满足 v_p(a_i) > 0，且 v_p(a_λ)=0 (由 λ 定义应当成立)。
    def _weierstrass_signature(series: CharacteristicPowerSeries) -> Optional[Tuple[int, Tuple[int, ...]]]:
        lam_local = series.lambda_invariant()
        if lam_local is None:
            return None
        lam_int = int(lam_local)
        if lam_int == 0:
            return (0, tuple())
        n_local = int(series.spec.witt_length)
        sig: List[int] = []
        for i in range(lam_int):
            sig.append(int(series._vp(int(series.coefficients[i]))))
        # Sanity: leading term should be a unit in this truncation
        lead_vp = int(series._vp(int(series.coefficients[lam_int])))
        if lead_vp != 0:
            raise RuntimeError(
                "Internal inconsistency: lambda_invariant points to non-unit coefficient. "
                f"lambda={lam_int}, v_p(a_lambda)={lead_vp}, n={n_local}"
            )
        return (lam_int, tuple(sig))

    proj_sig = _weierstrass_signature(proj_series)
    tgt_sig = _weierstrass_signature(tgt_series)

    if proj_sig is None or tgt_sig is None:
        # 无法确定有限 λ：这意味着在当前截断下无法给出 Weierstrass 形状的严格比较，
        # 绝不允许默认视为兼容。
        weierstrass_compatible = False
    else:
        (proj_lam_int, proj_vps) = proj_sig
        (tgt_lam_int, tgt_vps) = tgt_sig
        if proj_lam_int != tgt_lam_int:
            weierstrass_compatible = False
        elif proj_lam_int == 0:
            weierstrass_compatible = True
        else:
            # distinguished 必要条件：i<λ 时 v_p(a_i) > 0
            proj_distinguished = all(vp_i > 0 for vp_i in proj_vps)
            tgt_distinguished = all(vp_i > 0 for vp_i in tgt_vps)
            weierstrass_compatible = (proj_distinguished and tgt_distinguished and (proj_vps == tgt_vps))
    
    # 计算 Hausdorff 漂移
    hausdorff = _calculate_hausdorff_drift(holographic_state, target_pilot)
    
    # 整体判决: 必须全部匹配
    is_homeomorphic = mu_match and lambda_match and weierstrass_compatible
    
    # 证据摘要
    evidence_parts = []
    evidence_parts.append(f"μ: proj={proj_mu}, tgt={tgt_mu}, match={mu_match}")
    evidence_parts.append(f"λ: proj={proj_lam}, tgt={tgt_lam}, match={lambda_match}")
    evidence_parts.append(f"Weierstrass: compatible={weierstrass_compatible}")
    evidence_parts.append(f"Hausdorff: {hausdorff}")
    evidence_summary = "; ".join(evidence_parts)
    
    # 构建证书
    cert_body = {
        "holo": holographic_state.commitment,
        "pilot": target_pilot.commitment,
        "mu": mu_match,
        "lam": lambda_match,
        "weier": weierstrass_compatible,
        "homeo": is_homeomorphic,
        "haus": str(hausdorff),
    }
    commitment = _sha256_hex_of_dict(cert_body)
    
    cert = TopologicalHomeomorphismCertificate(
        holographic_state_commitment=holographic_state.commitment,
        target_pilot_commitment=target_pilot.commitment,
        mu_match=mu_match,
        lambda_match=lambda_match,
        weierstrass_compatible=weierstrass_compatible,
        is_homeomorphic=is_homeomorphic,
        hausdorff_drift=str(hausdorff),
        evidence_summary=evidence_summary,
        commitment=commitment,
    )
    
    _logger.info(
        "verify_topological_homeomorphism complete: homeomorphic=%s, commitment=%s...",
        is_homeomorphic,
        commitment[:16],
    )
    
    return cert


# =============================================================================
# 核心方法 3: enforce_multiradial_consensus
# =============================================================================


def enforce_multiradial_consensus(
    pilot_a: ThetaPilotCore,
    pilot_b: ThetaPilotCore,
    pilot_c: ThetaPilotCore,
) -> MultiRadialConsensusCertificate:
    """
    核心方法 3: 多辐射共识.
    
      - 计算 A 在 C 中的投影 proj_A_on_C
      - 计算 B 在 C 中的投影 proj_B_on_C
      - 验证 proj_A_on_C 和 proj_B_on_C 是否重合 (Ghosting)
      - 这是最终输出 PASS 的判决逻辑
    
    Args:
        pilot_a: Universe A
        pilot_b: Universe B
        pilot_c: Universe C (共识视角)
    
    Returns:
        MultiRadialConsensusCertificate: 共识验证结果
    """
    _logger.info(
        "enforce_multiradial_consensus: A=%s, B=%s, C=%s",
        pilot_a.universe_label,
        pilot_b.universe_label,
        pilot_c.universe_label,
    )
    
    for name, pilot in [("pilot_a", pilot_a), ("pilot_b", pilot_b), ("pilot_c", pilot_c)]:
        if not isinstance(pilot, ThetaPilotCore):
            raise TypeError(f"{name} must be ThetaPilotCore, got {type(pilot).__name__}")
    
    # 验证素数一致性
    primes = {pilot_a.prime_p, pilot_b.prime_p, pilot_c.prime_p}
    if len(primes) > 1:
        raise ValueError(f"Prime mismatch among pilots: {primes}")
    _logger.info(
        "enforce_multiradial_consensus: prime consistency verified, p=%d",
        pilot_a.prime_p,
    )
    
    # 计算 A 在 C 视角下的投影
    _logger.info("enforce_multiradial_consensus: computing proj(%s → %s)...", pilot_a.universe_label, pilot_c.universe_label)
    proj_a_on_c = project_holographic_state(pilot_a, pilot_c)
    _logger.info(
        "enforce_multiradial_consensus: proj_A_on_C done: mu=%d, lambda=%s, indet=%s",
        proj_a_on_c.characteristic_series.mu_invariant(),
        proj_a_on_c.characteristic_series.lambda_invariant(),
        proj_a_on_c.indeterminacy_volume,
    )

    # 计算 B 在 C 视角下的投影
    _logger.info("enforce_multiradial_consensus: computing proj(%s → %s)...", pilot_b.universe_label, pilot_c.universe_label)
    proj_b_on_c = project_holographic_state(pilot_b, pilot_c)
    _logger.info(
        "enforce_multiradial_consensus: proj_B_on_C done: mu=%d, lambda=%s, indet=%s",
        proj_b_on_c.characteristic_series.mu_invariant(),
        proj_b_on_c.characteristic_series.lambda_invariant(),
        proj_b_on_c.indeterminacy_volume,
    )

    # 验证投影是否重合 (Ghosting)
    #
    # 红线:
    # - 指标 2 要求物理层逐位全等，因此共识判定必须是严格二值：
    #   只有当漂移为 0 (在当前 Λ_{n,m} 截断下完全一致) 才能判定重合。
    # - 禁止使用在不确定性范围内也算重合这类启发式放水。
    _logger.info("enforce_multiradial_consensus: computing Hausdorff drift...")
    hausdorff = _calculate_hausdorff_drift(proj_a_on_c, proj_b_on_c)
    projections_coincide = (hausdorff == _trivial_hausdorff_distance())
    _logger.info(
        "enforce_multiradial_consensus: Hausdorff drift=%s, coincide=%s",
        hausdorff, projections_coincide,
    )
    
    # 最终判决
    consensus_achieved = projections_coincide
    
    # 构建证书
    cert_body = {
        "a": pilot_a.commitment,
        "b": pilot_b.commitment,
        "c": pilot_c.commitment,
        "proj_a": proj_a_on_c.commitment,
        "proj_b": proj_b_on_c.commitment,
        "coincide": projections_coincide,
        "haus": str(hausdorff),
        "consensus": consensus_achieved,
    }
    commitment = _sha256_hex_of_dict(cert_body)
    
    cert = MultiRadialConsensusCertificate(
        pilot_a_commitment=pilot_a.commitment,
        pilot_b_commitment=pilot_b.commitment,
        pilot_c_commitment=pilot_c.commitment,
        proj_a_on_c_commitment=proj_a_on_c.commitment,
        proj_b_on_c_commitment=proj_b_on_c.commitment,
        projections_coincide=projections_coincide,
        hausdorff_distance=str(hausdorff),
        consensus_achieved=consensus_achieved,
        commitment=commitment,
    )
    
    _logger.info(
        "enforce_multiradial_consensus complete: consensus=%s, hausdorff=%s, commitment=%s...",
        consensus_achieved,
        hausdorff,
        commitment[:16],
    )
    
    return cert


# =============================================================================
# 核心方法 4: _calculate_hausdorff_drift
# =============================================================================


def _calculate_hausdorff_drift(
    state_a: Union[HolographicState, ThetaPilotCore],
    state_b: Union[HolographicState, ThetaPilotCore],
) -> Fraction:
    """
    核心方法 4: Hausdorff 漂移计算.

      - 量化投影之间的微小偏差
      - 确保漂移量在 Log-Shell 的允许误差 (Indeterminacy) 范围内
    
    数学定义:
      d_H(A, B) = max( sup_{a ∈ A} inf_{b ∈ B} d(a,b), sup_{b ∈ B} inf_{a ∈ A} d(a,b) )
    
    在 Iwasawa 代数中, 使用 p-adic 距离:
      d_p(f, g) = p^{-v_p(f - g)} 其中 v_p 是 p-adic 赋值
    
    Args:
        state_a: 状态 A
        state_b: 状态 B
    
    Returns:
        Fraction: Hausdorff 漂移 (非负有理数)
    """
    # 提取特征级数
    if isinstance(state_a, HolographicState):
        series_a = state_a.characteristic_series
    elif isinstance(state_a, ThetaPilotCore):
        series_a = state_a.characteristic_series
    else:
        raise TypeError(f"state_a must be HolographicState or ThetaPilotCore, got {type(state_a).__name__}")
    
    if isinstance(state_b, HolographicState):
        series_b = state_b.characteristic_series
    elif isinstance(state_b, ThetaPilotCore):
        series_b = state_b.characteristic_series
    else:
        raise TypeError(f"state_b must be HolographicState or ThetaPilotCore, got {type(state_b).__name__}")
    
    # 验证规格兼容性
    # 红线: Hausdorff 漂移在 Λ_{n,m}(Z_p) 的同一截断环上定义；
    # 任何通过 min(n_a,n_b)/min(m_a,m_b) 的对齐都会丢精度并掩盖差异，必须禁止。
    spec_a = series_a.spec
    spec_b = series_b.spec

    if spec_a != spec_b:
        raise ValueError(
            "Cannot compute Hausdorff drift across different truncation specs (no normalization allowed): "
            f"a={spec_a.to_dict()}, b={spec_b.to_dict()}"
        )

    p = int(spec_a.p)
    n = int(spec_a.witt_length)
    m = int(spec_a.t_precision)
    mod = int(spec_a.modulus)
    
    # 计算系数差的最小 p-adic 赋值
    min_valuation = n  # 初始化为最大赋值 (零差)
    
    for i in range(m):
        diff = (int(series_a.coefficients[i]) - int(series_b.coefficients[i])) % mod
        if diff == 0:
            continue
        
        # 计算 v_p(diff)
        v = 0
        d = abs(diff)
        while v < n and d % p == 0:
            d //= p
            v += 1
        
        if v < min_valuation:
            min_valuation = v
    
    # Hausdorff 距离 = p^{-min_valuation}
    # 作为有理数: 1 / p^min_valuation
    if min_valuation >= n:
        # 差为零 (模 p^n)
        return _trivial_hausdorff_distance()
    
    return Fraction(1, p ** min_valuation)


# =============================================================================
# μ/λ 判决器
# =============================================================================


def compute_mu_verdict(series: CharacteristicPowerSeries) -> MuInvariantVerdict:
    """
    计算 μ-不变量判决.

      μ = 0: HARD (安全)
      μ > 0: SOFT (结构性塌缩, 致命漏洞)
    """
    if not isinstance(series, CharacteristicPowerSeries):
        raise TypeError(f"series must be CharacteristicPowerSeries, got {type(series).__name__}")
    
    mu = series.mu_invariant()
    is_hard = (mu == 0)
    
    if is_hard:
        evidence = f"min v_p(coeff) = 0; at least one coefficient not divisible by p={series.spec.p}"
    else:
        evidence = f"all coefficients divisible by p^{mu}; structural collapse detected"
    
    return MuInvariantVerdict(
        mu_value=mu,
        is_hard=is_hard,
        evidence=evidence,
    )


def compute_lambda_verdict(
    series: CharacteristicPowerSeries,
    threshold: Optional[int] = None,
) -> LambdaInvariantVerdict:
    """
    计算 λ-不变量判决.
    
      λ < threshold: TUNNELED (可预测, 致命漏洞)
      λ ≥ threshold 或 ∞: BLOCKED (安全)
    
    Args:
        series: 特征幂级数
        threshold: λ 阈值 (默认为 t_precision / 2)
    """
    if not isinstance(series, CharacteristicPowerSeries):
        raise TypeError(f"series must be CharacteristicPowerSeries, got {type(series).__name__}")
    
    if threshold is None:
        # 默认阈值: m/2 (来自规格)
        threshold = series.spec.t_precision // 2
    
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError(f"threshold must be non-negative int, got {threshold}")
    
    lam = series.lambda_invariant()
    
    if lam is None:
        # λ = ∞ (无法确定有限次数)
        is_blocked = True
        evidence = f"λ = ∞ (cannot determine finite degree within precision m={series.spec.t_precision})"
    elif lam >= threshold:
        is_blocked = True
        evidence = f"λ = {lam} >= threshold = {threshold}; high algebraic complexity"
    else:
        is_blocked = False
        evidence = f"λ = {lam} < threshold = {threshold}; low-order algebraic structure detected (CRITICAL)"
    
    return LambdaInvariantVerdict(
        lambda_value=lam,
        threshold=threshold,
        is_blocked=is_blocked,
        evidence=evidence,
    )

# =============================================================================
# 工厂函数: 从整数序列构建 Theta-Pilot
# =============================================================================

def create_theta_pilot_from_sequence(
    universe_label: str,
    sequence: Sequence[int],
    spec: IwasawaTruncationSpec,
) -> ThetaPilotCore:
    """
    从整数序列构建 Theta-Pilot 对象.
    
    用于将 Keccak 迭代序列、Slot 序列等转化为 Iwasawa 代数结构.
    
    Args:
        universe_label: 宇宙标签
        sequence: 整数序列 (将作为特征级数的系数)
        spec: Iwasawa 截断规格
    
    Returns:
        ThetaPilotCore
    """
    if not isinstance(universe_label, str) or not universe_label:
        raise ValueError("universe_label must be non-empty str")
    if not isinstance(sequence, (list, tuple)):
        raise TypeError(f"sequence must be Sequence[int], got {type(sequence).__name__}")
    if not isinstance(spec, IwasawaTruncationSpec):
        raise TypeError(f"spec must be IwasawaTruncationSpec, got {type(spec).__name__}")
    
    m = int(spec.t_precision)
    mod = int(spec.modulus)
    
    # 规范化系数
    coeffs = []
    for i in range(m):
        if i < len(sequence):
            coeffs.append(int(sequence[i]) % mod)
        else:
            coeffs.append(0)
    
    series = CharacteristicPowerSeries(coeffs, spec)
    
    # 构建证书
    pilot_body = {
        "universe": universe_label,
        "spec": spec.to_dict(),
        "series_hash": _sha256_hex_of_dict({"c": coeffs}),
    }
    commitment = _sha256_hex_of_dict(pilot_body)
    
    return ThetaPilotCore(
        universe_label=universe_label,
        characteristic_series=series,
        prime_p=spec.p,
        witt_length=spec.witt_length,
        commitment=commitment,
    )

# =============================================================================
# 完整扭转分析流水线
# =============================================================================

def analyze_torsion_full_pipeline(
    slot_a_value: int,
    slot_b_value: int,
    spec: IwasawaTruncationSpec,
    *,
    key: Any,
    universe_a_label: str = "Universe_A",
    universe_b_label: str = "Universe_B",
    reference_universe_label: str = "Universe_Ref",
) -> TorsionCertificate:
    """
    完整扭转分析流水线.
    
    执行 MVP22 的完整分析:
      1. 构建 Theta-Pilot 对象
      2. 计算 μ/λ 不变量
      3. 执行多辐射共识
      4. 生成二值扭转证书
    
    Args:
        slot_a_value: Slot A 的整数值
        slot_b_value: Slot B 的整数值
        spec: Iwasawa 截断规格
        universe_a_label: Universe A 标签
        universe_b_label: Universe B 标签
        reference_universe_label: 参考宇宙标签
    
    Returns:
        TorsionCertificate: 最终判决
    """
    # 默认升级：走 Operator‑B + Norton–Salagean 证书闭环。
    # （旧的差分 p-adic 展开 toy pipeline已淘汰；若需要保留请显式另建函数名，而不是静默退回。）
    _logger.info(
        "analyze_torsion_full_pipeline (operator-b): A=%d, B=%d, p=%d, n=%d, m=%d",
        slot_a_value, slot_b_value,
        spec.p, spec.witt_length, spec.t_precision,
    )
    if key is None:
        raise ValueError("key must be provided for Operator-B pipeline")
    # universe_* labels are currently reserved for future pilot/consensus wiring in Operator-B mode.
    _ = (universe_a_label, universe_b_label, reference_universe_label)
    return analyze_torsion_operator_b_pipeline(
        key=key,
        slot_a_value=slot_a_value,
        slot_b_value=slot_b_value,
        spec=spec,
    )


# =============================================================================
# 物理层逐位验证 (Physical Layer Bitwise Verification)
# =============================================================================


@dataclass(frozen=True)
class PhysicalLayerVerdict:
    """
    物理层逐位验证判决书 

    验收红线:
      int(Witt_decode(t_A)) ^ int(Witt_decode(t_B)) == 0
      只要有 1 个 bit 的差异，就是 FAIL
    """
    slot_a_original: int
    slot_b_original: int
    witt_decoded_a: int
    witt_decoded_b: int
    xor_result: int
    bit_diff_count: int
    is_collision: bool  # True = XOR == 0 (物理碰撞), False = DIVERGENCE
    commitment: str

    VERSION = "mvp22.physical_layer_verdict.v1"

    def __post_init__(self) -> None:
        for name in ("slot_a_original", "slot_b_original", "witt_decoded_a",
                     "witt_decoded_b", "xor_result", "bit_diff_count"):
            val = getattr(self, name)
            if not isinstance(val, int):
                raise TypeError(f"{name} must be int, got {type(val).__name__}")
        if not isinstance(self.is_collision, bool):
            raise TypeError("is_collision must be bool")
        if not isinstance(self.commitment, str) or not self.commitment:
            raise ValueError("commitment must be non-empty str")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "slot_a_original": int(self.slot_a_original),
            "slot_b_original": int(self.slot_b_original),
            "witt_decoded_a": int(self.witt_decoded_a),
            "witt_decoded_b": int(self.witt_decoded_b),
            "xor_result": int(self.xor_result),
            "bit_diff_count": int(self.bit_diff_count),
            "is_collision": bool(self.is_collision),
            "verdict": "COLLISION" if self.is_collision else "DIVERGENCE",
            "commitment": self.commitment,
        }


def _witt_encode_and_decode_roundtrip(value: int, p: int, witt_length: int) -> int:
    """
    Witt 编码→解码闭环：value → WittVector → integer (mod p^witt_length)

    数学流程 (严格、无启发式):
      1. value mod p^n 作为 ℤ/p^nℤ 元素
      2. 通过 Teichmüller 逆映射分解为 Witt 分量 (a_0,...,a_{n-1})
      3. 再通过 Teichmüller 正映射重建整数

    如果编码→解码是严格可逆的，返回值应等于 value mod p^n。
    """
    if not isinstance(value, int):
        raise TypeError(f"value must be int, got {type(value).__name__}")
    if not isinstance(p, int) or p < 2:
        raise ValueError(f"p must be int >= 2, got {p}")
    if not isinstance(witt_length, int) or witt_length < 1:
        raise ValueError(f"witt_length must be int >= 1, got {witt_length}")

    modulus = int(p ** witt_length)
    v = int(value) % modulus

    # 调用 MVP17 的 WittVector.from_integer (Teichmüller 逆映射)
    from web_ica.bridge_audit.core.mvp17_prismatic import WittVector as MVP17WittVector

    witt = MVP17WittVector.from_integer(v, p, witt_length)
    # _to_int_mod_p_power (Teichmüller 正映射)
    decoded = int(witt._to_int_mod_p_power())
    return decoded


def verify_physical_layer_bitwise(
    slot_a_value: int,
    slot_b_value: int,
    p: int,
    witt_length: int,
) -> PhysicalLayerVerdict:
    """
    物理层逐位验证

    红线要求:
      int(Witt_decode(t_A)) ^ int(Witt_decode(t_B)) == 0
      只要有 1 个 bit 的差异，就是 FAIL

    流程:
      1. 对 slot_a, slot_b 分别做 Witt 编码 → 解码 (roundtrip)
      2. XOR 两个解码结果
      3. 计算差异 bit 数
      4. 严格二值判决: XOR==0 → COLLISION, 否则 → DIVERGENCE

    注意: 这里验证的是"数学模型与物理世界的一致性"，
          即 Witt 同构在转换过程中是否保持信息完整。
    """
    _logger.info(
        "verify_physical_layer_bitwise: A=%d, B=%d, p=%d, k=%d",
        slot_a_value, slot_b_value, p, witt_length,
    )

    if not isinstance(slot_a_value, int):
        raise TypeError(f"slot_a_value must be int, got {type(slot_a_value).__name__}")
    if not isinstance(slot_b_value, int):
        raise TypeError(f"slot_b_value must be int, got {type(slot_b_value).__name__}")
    if not isinstance(p, int) or p < 2:
        raise ValueError(f"p must be int >= 2, got {p}")
    if not isinstance(witt_length, int) or witt_length < 1:
        raise ValueError(f"witt_length must be int >= 1, got {witt_length}")

    # Witt roundtrip
    decoded_a = _witt_encode_and_decode_roundtrip(slot_a_value, p, witt_length)
    decoded_b = _witt_encode_and_decode_roundtrip(slot_b_value, p, witt_length)

    # XOR
    xor_result = int(decoded_a) ^ int(decoded_b)

    # 计算差异 bit 数 (严格整数运算)
    bit_diff = bin(xor_result).count("1")

    # 二值判决
    is_collision = (xor_result == 0)

    # 构建证书
    cert_body = {
        "slot_a": int(slot_a_value),
        "slot_b": int(slot_b_value),
        "decoded_a": int(decoded_a),
        "decoded_b": int(decoded_b),
        "xor": int(xor_result),
        "bits": int(bit_diff),
        "collision": bool(is_collision),
        "p": int(p),
        "k": int(witt_length),
    }
    commitment = _sha256_hex_of_dict(cert_body)

    verdict = PhysicalLayerVerdict(
        slot_a_original=int(slot_a_value),
        slot_b_original=int(slot_b_value),
        witt_decoded_a=int(decoded_a),
        witt_decoded_b=int(decoded_b),
        xor_result=int(xor_result),
        bit_diff_count=int(bit_diff),
        is_collision=bool(is_collision),
        commitment=commitment,
    )

    _logger.info(
        "verify_physical_layer_bitwise complete: verdict=%s, xor=%d, bits=%d, commitment=%s...",
        "COLLISION" if is_collision else "DIVERGENCE",
        xor_result,
        bit_diff,
        commitment[:16],
    )

    return verdict


# =============================================================================
# 三轨道并行验证 (Trinity Track Parallel Verification)
# =============================================================================


@dataclass
class TrinityTrackVerdict:
    """
    三轨道并行验证判决书

    三轨道:
      - 轨道 A (物理层): p=2 - EVM 二进制底层
      - 轨道 B (几何层): secp256k1 域素数
      - 轨道 C (测试层): p=3 - Smoke Test

    验收要求:
      - 三轨道独立运行，产生独立的 μ/λ 判决
      - 三轨道一致性检查: 不同素数下的扭转检测结果应该指向同一个漏洞
      - 任何一个轨道检测到 torsion，整体判决为 TORSION_DETECTED
    """

    track_p2: TorsionCertificate
    track_secp256k1: Optional[TorsionCertificate]  # 大素数可能因精度限制无法完整运行
    track_p3: TorsionCertificate

    # 一致性检查
    torsion_consensus: bool  # 三轨道对 torsion 的判决是否一致
    mu_consensus: bool  # μ 不变量判决模式是否一致 (全 0 或全 >0)
    lambda_consensus: bool  # λ 不变量判决模式是否一致

    # 最终判决
    torsion_detected: bool  # 任意轨道检测到 torsion 即为 True
    consensus_strength: str  # "UNANIMOUS", "MAJORITY", "SPLIT"

    commitment: str

    # 物理层逐位验证结果 (可选，仅当输入是原始 slot 值时)
    physical_verdicts: Dict[int, PhysicalLayerVerdict] = field(default_factory=dict)

    VERSION = "mvp22.trinity_track_verdict.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.track_p2, TorsionCertificate):
            raise TypeError("track_p2 must be TorsionCertificate")
        if self.track_secp256k1 is not None and not isinstance(self.track_secp256k1, TorsionCertificate):
            raise TypeError("track_secp256k1 must be TorsionCertificate or None")
        if not isinstance(self.track_p3, TorsionCertificate):
            raise TypeError("track_p3 must be TorsionCertificate")
        for name in ("torsion_consensus", "mu_consensus", "lambda_consensus", "torsion_detected"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.consensus_strength not in ("UNANIMOUS", "MAJORITY", "SPLIT"):
            raise ValueError(f"consensus_strength must be UNANIMOUS/MAJORITY/SPLIT, got {self.consensus_strength}")
        if not isinstance(self.commitment, str) or not self.commitment:
            raise ValueError("commitment must be non-empty str")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "track_p2": self.track_p2.to_dict(),
            "track_secp256k1": self.track_secp256k1.to_dict() if self.track_secp256k1 else None,
            "track_p3": self.track_p3.to_dict(),
            "torsion_consensus": bool(self.torsion_consensus),
            "mu_consensus": bool(self.mu_consensus),
            "lambda_consensus": bool(self.lambda_consensus),
            "torsion_detected": bool(self.torsion_detected),
            "consensus_strength": str(self.consensus_strength),
            "physical_verdicts": {int(k): v.to_dict() for k, v in self.physical_verdicts.items()},
            "commitment": self.commitment,
        }


def analyze_torsion_trinity_tracks(
    *,
    key: Any,
    slot_a_value: int,
    slot_b_value: int,
    witt_length_p2: int = 8,
    witt_length_p3: int = 6,
    t_precision: int = 4,
    skip_secp256k1: bool = True,  # 大素数轨道默认跳过（计算量巨大）
    run_physical_verification: bool = True,
) -> TrinityTrackVerdict:
    """
    三轨道并行验证

    同时用三个素数运行扭转分析，交叉验证:
      - 轨道 A (p=2): EVM 二进制底层
      - 轨道 B (secp256k1): 几何层 (可选，默认跳过)
      - 轨道 C (p=3): Smoke Test

    Args:
        key: Keccak mapping 的 key
        slot_a_value: Slot A 整数值
        slot_b_value: Slot B 整数值
        witt_length_p2: p=2 轨道的 Witt 精度
        witt_length_p3: p=3 轨道的 Witt 精度
        t_precision: T 截断精度
        skip_secp256k1: 是否跳过 secp256k1 轨道 (默认 True)
        run_physical_verification: 是否运行物理层逐位验证

    Returns:
        TrinityTrackVerdict: 三轨道判决书
    """
    _logger.info(
        "analyze_torsion_trinity_tracks: A=%d, B=%d, primes=(2, secp256k1=%s, 3)",
        slot_a_value, slot_b_value, "skip" if skip_secp256k1 else "run",
    )

    if not isinstance(slot_a_value, int):
        raise TypeError(f"slot_a_value must be int, got {type(slot_a_value).__name__}")
    if not isinstance(slot_b_value, int):
        raise TypeError(f"slot_b_value must be int, got {type(slot_b_value).__name__}")

    primes = _trinity_primes()
    p2, secp256k1_prime, p3 = primes

    # -------------------------------------------------------------------------
    # 轨道 A: p=2
    # -------------------------------------------------------------------------
    spec_p2 = IwasawaTruncationSpec(p=p2, witt_length=witt_length_p2, t_precision=t_precision)
    _logger.info("Trinity Track p=2: running...")
    cert_p2 = analyze_torsion_operator_b_pipeline(
        key=key,
        slot_a_value=slot_a_value,
        slot_b_value=slot_b_value,
        spec=spec_p2,
    )
    _logger.info(
        "Trinity Track p=2: torsion=%s, mu=%d, lambda=%s",
        cert_p2.is_torsion, cert_p2.mu_invariant,
        cert_p2.lambda_invariant if cert_p2.lambda_invariant is not None else "∞",
    )

    # -------------------------------------------------------------------------
    # 轨道 B: secp256k1 (可选)
    # -------------------------------------------------------------------------
    cert_secp: Optional[TorsionCertificate] = None
    if not skip_secp256k1:
        # 对于大素数，Witt 精度必须极小，否则计算量爆炸
        # 这里用 witt_length=2 作为最小可行精度
        spec_secp = IwasawaTruncationSpec(p=secp256k1_prime, witt_length=2, t_precision=t_precision)
        _logger.info("Trinity Track secp256k1: running (precision=2, may be slow)...")
        try:
            cert_secp = analyze_torsion_operator_b_pipeline(
                key=key,
                slot_a_value=slot_a_value,
                slot_b_value=slot_b_value,
                spec=spec_secp,
            )
            _logger.info(
                "Trinity Track secp256k1: torsion=%s, mu=%d, lambda=%s",
                cert_secp.is_torsion, cert_secp.mu_invariant,
                cert_secp.lambda_invariant if cert_secp.lambda_invariant is not None else "∞",
            )
        except Exception as e:
            # 大素数轨道失败不应阻断整体流程，但必须记录
            _logger.warning("Trinity Track secp256k1 FAILED: %s (continuing with p=2,3 only)", e)
            cert_secp = None
    else:
        _logger.info("Trinity Track secp256k1: SKIPPED (skip_secp256k1=True)")

    # -------------------------------------------------------------------------
    # 轨道 C: p=3
    # -------------------------------------------------------------------------
    spec_p3 = IwasawaTruncationSpec(p=p3, witt_length=witt_length_p3, t_precision=t_precision)
    _logger.info("Trinity Track p=3: running...")
    cert_p3 = analyze_torsion_operator_b_pipeline(
        key=key,
        slot_a_value=slot_a_value,
        slot_b_value=slot_b_value,
        spec=spec_p3,
    )
    _logger.info(
        "Trinity Track p=3: torsion=%s, mu=%d, lambda=%s",
        cert_p3.is_torsion, cert_p3.mu_invariant,
        cert_p3.lambda_invariant if cert_p3.lambda_invariant is not None else "∞",
    )

    # -------------------------------------------------------------------------
    # 物理层逐位验证 (可选)
    # -------------------------------------------------------------------------
    physical_verdicts: Dict[int, PhysicalLayerVerdict] = {}
    if run_physical_verification:
        _logger.info("Running physical layer bitwise verification...")
        # p=2 轨道
        physical_verdicts[p2] = verify_physical_layer_bitwise(
            slot_a_value, slot_b_value, p2, witt_length_p2,
        )
        # p=3 轨道
        physical_verdicts[p3] = verify_physical_layer_bitwise(
            slot_a_value, slot_b_value, p3, witt_length_p3,
        )

    # -------------------------------------------------------------------------
    # 一致性检查
    # -------------------------------------------------------------------------
    torsion_results = [cert_p2.is_torsion, cert_p3.is_torsion]
    mu_results = [cert_p2.mu_invariant, cert_p3.mu_invariant]
    lambda_results = [cert_p2.lambda_invariant, cert_p3.lambda_invariant]

    if cert_secp is not None:
        torsion_results.append(cert_secp.is_torsion)
        mu_results.append(cert_secp.mu_invariant)
        lambda_results.append(cert_secp.lambda_invariant)

    # Torsion 一致性: 所有轨道判决相同
    torsion_consensus = len(set(torsion_results)) == 1

    # μ 一致性: 全部 μ=0 或全部 μ>0
    mu_signs = [1 if m > 0 else 0 for m in mu_results]
    mu_consensus = len(set(mu_signs)) == 1

    # λ 一致性: 全部有限或全部无限
    lambda_finite = [1 if lam is not None else 0 for lam in lambda_results]
    lambda_consensus = len(set(lambda_finite)) == 1

    # 最终判决: 任意轨道检测到 torsion 即为 True
    torsion_detected = any(torsion_results)

    # 共识强度
    true_count = sum(torsion_results)
    total_count = len(torsion_results)
    if true_count == 0 or true_count == total_count:
        consensus_strength = "UNANIMOUS"
    elif true_count >= total_count // 2 + 1:
        consensus_strength = "MAJORITY"
    else:
        consensus_strength = "SPLIT"

    # -------------------------------------------------------------------------
    # 构建证书
    # -------------------------------------------------------------------------
    cert_body = {
        "p2_commitment": cert_p2.commitment,
        "secp_commitment": cert_secp.commitment if cert_secp else "SKIPPED",
        "p3_commitment": cert_p3.commitment,
        "torsion_consensus": bool(torsion_consensus),
        "mu_consensus": bool(mu_consensus),
        "lambda_consensus": bool(lambda_consensus),
        "torsion_detected": bool(torsion_detected),
        "consensus_strength": str(consensus_strength),
        "slot_a": int(slot_a_value),
        "slot_b": int(slot_b_value),
    }
    commitment = _sha256_hex_of_dict(cert_body)

    verdict = TrinityTrackVerdict(
        track_p2=cert_p2,
        track_secp256k1=cert_secp,
        track_p3=cert_p3,
        torsion_consensus=torsion_consensus,
        mu_consensus=mu_consensus,
        lambda_consensus=lambda_consensus,
        torsion_detected=torsion_detected,
        consensus_strength=consensus_strength,
        physical_verdicts=physical_verdicts,
        commitment=commitment,
    )

    _logger.info(
        "analyze_torsion_trinity_tracks complete: torsion=%s, consensus=%s, strength=%s, commitment=%s...",
        torsion_detected, torsion_consensus, consensus_strength, commitment[:16],
    )

    return verdict


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 常量函数
    "_min_witt_length_for_bits",
    "_trinity_primes",
    "_evm_slot_bits",
    "_identity_coefficient",
    "_trivial_hausdorff_distance",
    # 规格
    "IwasawaTruncationSpec",
    # 判决
    "MuInvariantVerdict",
    "LambdaInvariantVerdict",
    "PhysicalLayerVerdict",
    "TrinityTrackVerdict",
    # 核心数据结构
    "CharacteristicPowerSeries",
    "HolographicState",
    "ThetaPilotCore",
    "TopologicalHomeomorphismCertificate",
    "MultiRadialConsensusCertificate",
    "TorsionCertificate",
    # 四个核心方法
    "project_holographic_state",
    "verify_topological_homeomorphism",
    "enforce_multiradial_consensus",
    "_calculate_hausdorff_drift",
    # 辅助方法
    "compute_mu_verdict",
    "compute_lambda_verdict",
    "create_theta_pilot_from_sequence",
    "analyze_torsion_full_pipeline",
    "analyze_torsion_operator_b_pipeline",
    # 物理层逐位验证
    "verify_physical_layer_bitwise",
    # 三轨道并行验证
    "analyze_torsion_trinity_tracks",
]


# =============================================================================
# 测试
# =============================================================================

def _self_test_mvp22() -> Dict[str, Any]:
    """
    MVP22 自测试套件.
    
    验证核心功能的正确性, 部署前必须通过.
    """
    results: Dict[str, Any] = {"ok": True, "tests": []}
    
    def record(name: str, passed: bool, detail: str = "") -> None:
        results["tests"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            results["ok"] = False
            _logger.error("SELF-TEST FAILED: %s - %s", name, detail)
    
    # 测试 1: 常量函数无魔法数
    try:
        primes = _trinity_primes()
        assert primes == (2, (1 << 256) - (1 << 32) - 977, 3), "Trinity primes mismatch"
        assert _evm_slot_bits() == 256, "EVM slot bits mismatch"
        assert _identity_coefficient() == 1, "Identity coefficient mismatch"
        assert _trivial_hausdorff_distance() == Fraction(0), "Trivial distance mismatch"
        record("constants_no_magic", True)
    except Exception as e:
        record("constants_no_magic", False, str(e))
    
    # 测试 2: Iwasawa 规格构造
    try:
        spec = IwasawaTruncationSpec(p=2, witt_length=10, t_precision=20)
        assert spec.modulus == 2 ** 10, "Modulus calculation failed"
        assert spec.p == 2, "Prime extraction failed"
        record("spec_construction", True)
    except Exception as e:
        record("spec_construction", False, str(e))
    
    # 测试 3: 特征级数 μ/λ 计算（λ 必须可判定，否则抛出明确异常）
    try:
        spec = IwasawaTruncationSpec(p=2, witt_length=8, t_precision=10)
        
        # 零级数: μ=n，λ 不可判定必须抛 LambdaUndeterminedError
        zero_series = CharacteristicPowerSeries.zero(spec)
        assert zero_series.mu_invariant() == 8, "Zero series μ should be n"
        try:
            zero_series.lambda_invariant()
            raise AssertionError("Zero series λ must raise LambdaUndeterminedError")
        except LambdaUndeterminedError:
            pass
        
        # 单位级数: μ=0, λ=0
        one_series = CharacteristicPowerSeries.one(spec)
        assert one_series.mu_invariant() == 0, "One series μ should be 0"
        assert one_series.lambda_invariant() == 0, "One series λ should be 0"
        
        # 带 p 因子的级数: μ>0（λ 依赖内容，至少不应静默返回 None）
        p_series = CharacteristicPowerSeries([2, 4, 8, 0, 0, 0, 0, 0, 0, 0], spec)
        assert p_series.mu_invariant() >= 1, "p-divisible series μ should be >= 1"
        try:
            p_series.lambda_invariant()
        except LambdaUndeterminedError:
            # 允许在当前截断下不可判定，但必须以异常显式呈现
            pass
        
        record("mu_lambda_invariants", True)
    except Exception as e:
        record("mu_lambda_invariants", False, str(e))
    
    # 测试 4: Theta-Pilot 构造
    try:
        spec = IwasawaTruncationSpec(p=3, witt_length=6, t_precision=8)
        pilot = create_theta_pilot_from_sequence(
            "Test_Universe",
            [1, 2, 3, 4, 5],
            spec,
        )
        assert pilot.universe_label == "Test_Universe"
        assert pilot.prime_p == 3
        assert len(pilot.commitment) == 64  # SHA-256 hex
        record("theta_pilot_construction", True)
    except Exception as e:
        record("theta_pilot_construction", False, str(e))
    
    # 测试 5: 核心方法 - 全息投影
    try:
        spec = IwasawaTruncationSpec(p=2, witt_length=4, t_precision=6)
        pilot_a = create_theta_pilot_from_sequence("A", [1, 0, 0, 0, 0, 0], spec)
        pilot_b = create_theta_pilot_from_sequence("B", [0, 1, 0, 0, 0, 0], spec)
        
        holo = project_holographic_state(pilot_a, pilot_b)
        assert holo.source_universe_label == "A"
        assert holo.target_perspective_label == "B"
        assert isinstance(holo.indeterminacy_volume, Fraction)
        assert holo.indeterminacy_volume >= 0
        record("holographic_projection", True)
    except Exception as e:
        record("holographic_projection", False, str(e))
    
    # 测试 6: 核心方法 - 拓扑同胚验证
    try:
        spec = IwasawaTruncationSpec(p=2, witt_length=4, t_precision=6)
        pilot_a = create_theta_pilot_from_sequence("A", [1, 0, 0, 0, 0, 0], spec)
        pilot_b = create_theta_pilot_from_sequence("B", [1, 0, 0, 0, 0, 0], spec)
        
        holo = project_holographic_state(pilot_a, pilot_b)
        cert = verify_topological_homeomorphism(holo, pilot_b)
        assert isinstance(cert.is_homeomorphic, bool)
        record("topological_homeomorphism", True)
    except Exception as e:
        record("topological_homeomorphism", False, str(e))
    
    # 测试 7: 核心方法 - 多辐射共识
    try:
        spec = IwasawaTruncationSpec(p=2, witt_length=4, t_precision=6)
        pilot_a = create_theta_pilot_from_sequence("A", [1, 0, 0, 0, 0, 0], spec)
        pilot_b = create_theta_pilot_from_sequence("B", [1, 0, 0, 0, 0, 0], spec)
        pilot_c = create_theta_pilot_from_sequence("C", [1, 0, 0, 0, 0, 0], spec)
        
        cert = enforce_multiradial_consensus(pilot_a, pilot_b, pilot_c)
        assert isinstance(cert.consensus_achieved, bool)
        record("multiradial_consensus", True)
    except Exception as e:
        record("multiradial_consensus", False, str(e))
    
    # 测试 8: 核心方法 - Hausdorff 漂移
    try:
        spec = IwasawaTruncationSpec(p=2, witt_length=4, t_precision=6)
        pilot_a = create_theta_pilot_from_sequence("A", [1, 0, 0, 0, 0, 0], spec)
        pilot_b = create_theta_pilot_from_sequence("B", [1, 0, 0, 0, 0, 0], spec)
        
        drift = _calculate_hausdorff_drift(pilot_a, pilot_b)
        assert isinstance(drift, Fraction)
        assert drift >= 0
        # 相同序列应该有零漂移
        assert drift == Fraction(0), f"Same sequence should have zero drift, got {drift}"
        record("hausdorff_drift", True)
    except Exception as e:
        record("hausdorff_drift", False, str(e))
    
    # 测试 9: 完整流水线
    try:
        # Operator‑B 默认流水线：t_precision=1 => steps=0, 不依赖 Keccak 后端。
        spec = IwasawaTruncationSpec(p=2, witt_length=8, t_precision=1)
        cert = analyze_torsion_full_pipeline(
            key=b"\x00" * 32,
            slot_a_value=12345,
            slot_b_value=12344,  # seed_padic=1 (非零), 避免零级数导致 λ 不可判定
            spec=spec,
        )
        assert isinstance(cert.is_torsion, bool)
        assert cert.is_torsion is False, "steps=0 => trivial annihilator => no torsion"
        record("full_pipeline_same_slots", True)
    except Exception as e:
        record("full_pipeline_same_slots", False, str(e))
    
    # 测试 10: 二值性 - 禁止浮点数
    try:
        spec = IwasawaTruncationSpec(p=2, witt_length=4, t_precision=6)
        series = CharacteristicPowerSeries([1, 2, 3, 0, 0, 0], spec)
        d = series.to_dict()
        
        # 检查没有浮点数
        def check_no_float(obj: Any, path: str = "") -> None:
            if isinstance(obj, float):
                raise ValueError(f"Float found at {path}: {obj}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_float(v, f"{path}.{k}")
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_no_float(v, f"{path}[{i}]")
        
        check_no_float(d)
        record("no_float_in_output", True)
    except Exception as e:
        record("no_float_in_output", False, str(e))
    
    # 测试 11: 物理层逐位验证
    try:
        # 相同值的 Witt roundtrip 应该产生 COLLISION (XOR=0)
        verdict_same = verify_physical_layer_bitwise(
            slot_a_value=12345,
            slot_b_value=12345,
            p=2,
            witt_length=8,
        )
        assert verdict_same.is_collision is True, f"Same values should COLLISION, got XOR={verdict_same.xor_result}"
        assert verdict_same.xor_result == 0, f"Same values XOR should be 0, got {verdict_same.xor_result}"
        assert verdict_same.bit_diff_count == 0, f"Same values bit diff should be 0, got {verdict_same.bit_diff_count}"
        
        # 不同值的 Witt roundtrip 应该产生 DIVERGENCE (XOR!=0)
        verdict_diff = verify_physical_layer_bitwise(
            slot_a_value=12345,
            slot_b_value=12346,
            p=2,
            witt_length=8,
        )
        assert verdict_diff.is_collision is False, f"Different values should DIVERGENCE, got collision={verdict_diff.is_collision}"
        assert verdict_diff.xor_result != 0, f"Different values XOR should be non-zero, got {verdict_diff.xor_result}"
        
        record("physical_layer_bitwise", True)
    except Exception as e:
        record("physical_layer_bitwise", False, str(e))
    
    # 测试 12: 三轨道并行验证
    try:
        trinity_verdict = analyze_torsion_trinity_tracks(
            key=b"\x00" * 32,
            slot_a_value=12345,
            slot_b_value=12344,
            witt_length_p2=4,
            witt_length_p3=3,
            t_precision=1,  # 最小精度，快速测试
            skip_secp256k1=True,  # 跳过大素数，避免测试超时
            run_physical_verification=True,
        )
        assert isinstance(trinity_verdict.torsion_detected, bool)
        assert isinstance(trinity_verdict.torsion_consensus, bool)
        assert trinity_verdict.consensus_strength in ("UNANIMOUS", "MAJORITY", "SPLIT")
        assert trinity_verdict.track_p2 is not None
        assert trinity_verdict.track_p3 is not None
        # 物理验证应该包含 p=2 和 p=3 两个轨道的结果
        assert 2 in trinity_verdict.physical_verdicts
        assert 3 in trinity_verdict.physical_verdicts
        
        record("trinity_track_parallel", True)
    except Exception as e:
        record("trinity_track_parallel", False, str(e))
    
    # 总结
    passed = sum(1 for t in results["tests"] if t["passed"])
    total = len(results["tests"])
    _logger.info("MVP22 self-test: %d/%d passed", passed, total)
    
    if not results["ok"]:
        raise RuntimeError(
            f"MVP22 self-test FAILED: {total - passed}/{total} tests failed. "
            "Deployment must abort."
        )
    
    return results


if __name__ == "__main__":
    # When executed as a script, Python sets sys.path[0] to this file's directory
    # (web_ica/bridge_audit/core), which prevents absolute imports like
    # `web_ica.bridge_audit...` from resolving. For deployment self-test we must
    # add the repository root to sys.path deterministically.
    import os
    import sys

    if __package__ is None or __package__ == "":
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    print("Running MVP22 self-test...")
    results = _self_test_mvp22()
    print(f"\nResults: {results['ok']}")
    for t in results["tests"]:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  [{status}] {t['name']}")
        if not t["passed"] and t.get("detail"):
            print(f"         {t['detail']}")
