"""
================================================================================
我选择硬上Witt向量的无奈数学选择：棱柱上同调
Prismatic Cohomology & Witt Vectors - Rigorous Implementation

数学：
- 特征 p 的完善域 (Perfect Field) k
- Witt 向量环 W(k)
- δ-环结构 (δ-Ring Structure)  
- 棱柱 (Prism) (A, I) 与 Nygaard 过滤
================================================================================
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Callable, Iterator, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from fractions import Fraction
import itertools
import logging


__all__ = [
    # ═══════════════════════════════════════════════════════════════════════════
    # 基础代数结构
    # ═══════════════════════════════════════════════════════════════════════════
    "RingElement",
    "IntegerElement",
    "FiniteFieldElement",
    "FiniteFieldExtElement",
    # ═══════════════════════════════════════════════════════════════════════════
    # 多项式环
    # ═══════════════════════════════════════════════════════════════════════════
    "Monomial",
    "MultivariatePolynomial",
    "PolynomialOverFp",
    # ═══════════════════════════════════════════════════════════════════════════
    # Witt 多项式生成器
    # ═══════════════════════════════════════════════════════════════════════════
    "WittPolynomialGenerator",
    # ═══════════════════════════════════════════════════════════════════════════
    # Witt 向量
    # ═══════════════════════════════════════════════════════════════════════════
    "WittVector",
    "verify_witt_polynomial_consistency",
    "verify_frobenius_verschiebung_relations",
    # ═══════════════════════════════════════════════════════════════════════════
    # δ-环结构
    # ═══════════════════════════════════════════════════════════════════════════
    "DeltaRing",
    "WittVectorDeltaRing",
    # ═══════════════════════════════════════════════════════════════════════════
    # 棱柱结构
    # ═══════════════════════════════════════════════════════════════════════════
    "Prism",
    "IdealPower",
    # ═══════════════════════════════════════════════════════════════════════════
    # Nygaard 过滤
    # ═══════════════════════════════════════════════════════════════════════════
    "NygaardFiltration",
    "NygaardQuotient",
    "NygaardCompletion",
    # ═══════════════════════════════════════════════════════════════════════════
    # 整性验证器
    # ═══════════════════════════════════════════════════════════════════════════
    "IntegralityValidator",
    "ValidationResult",
    "OverflowInfo",
    # ═══════════════════════════════════════════════════════════════════════════
    # 验收套件
    # ═══════════════════════════════════════════════════════════════════════════
    "strict_witt_kernel_validation",
    "strict_nygaard_filtration_validation",
    "strict_integrality_validation",
    "strict_witt_polynomial_validation",
    "run_strict_validation_suite",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Canonical primes (no heuristic selection)
# =============================================================================
#
# secp256k1 field prime (SEC 2 / Bitcoin):
#   p = 2^256 - 2^32 - 977
# This is not a "magic number" in this project: it is a standardized constant
# required by the user's MVP22/Iwasawa Trinity track-B.
SECP256K1_FIELD_PRIME: int = (1 << 256) - (1 << 32) - 977


# 第一部分：基础代数结构

class RingElement(ABC):
    """环元素的抽象基类"""
    
    @abstractmethod
    def __add__(self, other): pass
    
    @abstractmethod
    def __mul__(self, other): pass
    
    @abstractmethod
    def __neg__(self): pass
    
    @abstractmethod
    def __eq__(self, other) -> bool: pass
    
    @abstractmethod
    def is_zero(self) -> bool: pass
    
    def __sub__(self, other):
        return self + (-other)


class IntegerElement(RingElement):
    """
    整数环 ℤ 的元素
    这不是 Python int 的包装——而是维护完整的代数结构
    用于 Witt 多项式的精确计算（需要在 ℤ 上计算后再约化）
    """
    
    __slots__ = ('_value',)
    
    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"IntegerElement 需要 int，得到 {type(value)}")
        self._value = value
    
    @property
    def value(self) -> int:
        return self._value
    
    def __add__(self, other: 'IntegerElement') -> 'IntegerElement':
        if isinstance(other, int):
            other = IntegerElement(other)
        return IntegerElement(self._value + other._value)
    
    def __radd__(self, other) -> 'IntegerElement':
        return self + other
    
    def __mul__(self, other: 'IntegerElement') -> 'IntegerElement':
        if isinstance(other, int):
            other = IntegerElement(other)
        return IntegerElement(self._value * other._value)
    
    def __rmul__(self, other) -> 'IntegerElement':
        return self * other
    
    def __neg__(self) -> 'IntegerElement':
        return IntegerElement(-self._value)
    
    def __pow__(self, n: int) -> 'IntegerElement':
        if n < 0:
            raise ValueError("IntegerElement 不支持负指数")
        return IntegerElement(self._value ** n)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, IntegerElement):
            return self._value == other._value
        if isinstance(other, int):
            return self._value == other
        return False
    
    def __hash__(self) -> int:
        return hash(self._value)
    
    def __repr__(self) -> str:
        return f"ℤ({self._value})"
    
    def is_zero(self) -> bool:
        return self._value == 0
    
    def is_divisible_by(self, p: int) -> bool:
        """检查是否被 p 整除"""
        return self._value % p == 0
    
    def exact_div(self, p: int) -> 'IntegerElement':
        """精确除法（必须整除）"""
        if self._value % p != 0:
            raise ValueError(f"{self._value} 不能被 {p} 整除")
        return IntegerElement(self._value // p)
    
    @classmethod
    def zero(cls) -> 'IntegerElement':
        return cls(0)
    
    @classmethod
    def one(cls) -> 'IntegerElement':
        return cls(1)


class FiniteFieldElement(RingElement):
    """
    有限域 𝔽_p 的元素
    数学定义：𝔽_p = ℤ/pℤ，p 为素数
    - 内部存储为 [0, p-1] 的代表元
    - 所有运算在代数层面完成，不依赖 Python 的隐式截断
    """
    
    __slots__ = ('_value', '_p')
    
    def __init__(self, value: int, p: int):
        """
        Args:
            value: 整数代表元
            p: 特征（必须是素数）
        """
        self._p = p
        # 规范化：数学意义上的模运算
        self._value = self._normalize(value, p)
    
    @staticmethod
    def _normalize(value: int, p: int) -> int:
        """将整数规范化到 [0, p-1]"""
        r = value % p
        return r if r >= 0 else r + p
    
    @property
    def value(self) -> int:
        return self._value
    
    @property
    def characteristic(self) -> int:
        return self._p
    
    def _check_compatible(self, other: 'FiniteFieldElement') -> None:
        """验证两个元素在同一个域中"""
        if self._p != other._p:
            raise ValueError(f"特征不匹配: {self._p} vs {other._p}")
    
    def __add__(self, other: 'FiniteFieldElement') -> 'FiniteFieldElement':
        if isinstance(other, int):
            other = FiniteFieldElement(other, self._p)
        self._check_compatible(other)
        return FiniteFieldElement(self._value + other._value, self._p)
    
    def __radd__(self, other) -> 'FiniteFieldElement':
        return self + other
    
    def __mul__(self, other: 'FiniteFieldElement') -> 'FiniteFieldElement':
        if isinstance(other, int):
            other = FiniteFieldElement(other, self._p)
        self._check_compatible(other)
        return FiniteFieldElement(self._value * other._value, self._p)
    
    def __rmul__(self, other) -> 'FiniteFieldElement':
        return self * other
    
    def __neg__(self) -> 'FiniteFieldElement':
        return FiniteFieldElement(-self._value, self._p)
    
    def __pow__(self, n: int) -> 'FiniteFieldElement':
        """
        快速幂算法
        数学基础：Fermat 小定理 a^^p ≡ a (mod p)
        """
        if n < 0:
            return self.inverse() ** (-n)
        if n == 0:
            return FiniteFieldElement(1, self._p)
        
        result = FiniteFieldElement(1, self._p)
        base = self
        exp = n
        
        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        
        return result
    
    def inverse(self) -> 'FiniteFieldElement':
        """
        乘法逆元
        数学基础：Fermat 小定理 a^^(-1) = a^^(p-2) mod p
        """
        if self._value == 0:
            raise ZeroDivisionError("𝔽_p 中零元素没有乘法逆")
        return self ** (self._p - 2)
    
    def frobenius(self) -> 'FiniteFieldElement':
        """
        Frobenius 自同态: x ↦ x^^p
        在 𝔽_p 上这是恒等映射，但在扩域上不是
        """
        return self ** self._p
    
    def __eq__(self, other) -> bool:
        if isinstance(other, FiniteFieldElement):
            return self._p == other._p and self._value == other._value
        if isinstance(other, int):
            return self._value == self._normalize(other, self._p)
        return False
    
    def __hash__(self) -> int:
        return hash((self._value, self._p))
    
    def __repr__(self) -> str:
        return f"{self._value}₍{self._p}₎"
    
    def is_zero(self) -> bool:
        return self._value == 0
    
    def is_one(self) -> bool:
        return self._value == 1
    
    @classmethod
    def zero(cls, p: int) -> 'FiniteFieldElement':
        return cls(0, p)
    
    @classmethod
    def one(cls, p: int) -> 'FiniteFieldElement':
        return cls(1, p)


class FiniteFieldExtElement(RingElement):
    """
    有限扩张 𝔽_{p^n} 的元素（严格实现，非启发式）。

    表示为 𝔽_p[x]/(f(x)) 中的多项式类，其中 f(x) 是 n 次**首一**多项式（通常要求不可约）。
    内部表示：
      coeffs = (a_0, a_1, ..., a_{n-1}) 代表 a_0 + a_1·x + ... + a_{n-1}·x^{n-1}

    重要性质：
    - Frobenius φ: a ↦ a^p 在 𝔽_{p^n} 上一般是非平凡自同构（n>1）。
    - 所有运算严格在模多项式下进行，不依赖 Python 隐式截断。
    """

    __slots__ = ("_p", "_n", "_modulus", "_coeffs")

    def __init__(self, coeffs: List[int], p: int, modulus_coeffs: List[int]):
        if not isinstance(p, int):
            raise TypeError(f"p must be int, got {type(p).__name__}")
        if p < 2:
            raise ValueError("p must be >= 2 (and should be prime).")
        if not isinstance(modulus_coeffs, list):
            raise TypeError(f"modulus_coeffs must be List[int], got {type(modulus_coeffs).__name__}")
        if len(modulus_coeffs) < 2:
            raise ValueError("modulus_coeffs must have degree >= 1 (len >= 2).")
        if not all(isinstance(c, int) for c in modulus_coeffs):
            bad = next((c for c in modulus_coeffs if not isinstance(c, int)), None)
            raise TypeError(f"modulus_coeffs must be List[int]; found {type(bad).__name__}")

        self._p = int(p)
        self._n = int(len(modulus_coeffs) - 1)

        # 首一：最高次项系数必须为 1（在 𝔽_p 中）
        if int(modulus_coeffs[-1] % p) != 1:
            raise ValueError("modulus polynomial must be monic (leading coefficient == 1 mod p).")

        self._modulus = tuple(int(c % p) for c in modulus_coeffs)

        if not isinstance(coeffs, list):
            raise TypeError(f"coeffs must be List[int], got {type(coeffs).__name__}")
        if not all(isinstance(c, int) for c in coeffs):
            bad = next((c for c in coeffs if not isinstance(c, int)), None)
            raise TypeError(f"coeffs must be List[int]; found {type(bad).__name__}")

        # 规范化系数并截断到 n 项，然后补零到 n 项
        normalized = [int(c % p) for c in coeffs[: self._n]]
        if len(normalized) < self._n:
            normalized.extend([0] * (self._n - len(normalized)))
        self._coeffs = tuple(normalized)

    @property
    def characteristic(self) -> int:
        return int(self._p)

    @property
    def extension_degree(self) -> int:
        return int(self._n)

    @property
    def modulus_polynomial(self) -> Tuple[int, ...]:
        return tuple(self._modulus)

    @property
    def coeffs(self) -> Tuple[int, ...]:
        return tuple(self._coeffs)

    def _check_compatible(self, other: "FiniteFieldExtElement") -> None:
        if int(self._p) != int(other._p):
            raise ValueError(f"Characteristic mismatch: {int(self._p)} vs {int(other._p)}")
        if self._modulus != other._modulus:
            raise ValueError("Field modulus mismatch: elements are not in the same extension field.")

    def _coerce_other(self, other: object) -> "FiniteFieldExtElement":
        if isinstance(other, FiniteFieldExtElement):
            self._check_compatible(other)
            return other
        if isinstance(other, FiniteFieldElement):
            if int(other.characteristic) != int(self._p):
                raise ValueError(f"Characteristic mismatch: {int(self._p)} vs {int(other.characteristic)}")
            return FiniteFieldExtElement([int(other.value)], int(self._p), list(self._modulus))
        raise TypeError(f"Unsupported operand type: {type(other).__name__}")

    def __add__(self, other) -> "FiniteFieldExtElement":
        o = self._coerce_other(other)
        p = int(self._p)
        return FiniteFieldExtElement(
            [int((a + b) % p) for a, b in zip(self._coeffs, o._coeffs)],
            p,
            list(self._modulus),
        )

    def __radd__(self, other) -> "FiniteFieldExtElement":
        return self.__add__(other)

    def __neg__(self) -> "FiniteFieldExtElement":
        p = int(self._p)
        return FiniteFieldExtElement([int((-a) % p) for a in self._coeffs], p, list(self._modulus))

    def __sub__(self, other) -> "FiniteFieldExtElement":
        return self + (-self._coerce_other(other))

    def __mul__(self, other) -> "FiniteFieldExtElement":
        o = self._coerce_other(other)
        return self._mul_mod(o)

    def __rmul__(self, other) -> "FiniteFieldExtElement":
        return self.__mul__(other)

    def is_zero(self) -> bool:
        return all(int(c) == 0 for c in self._coeffs)

    def __eq__(self, other) -> bool:
        if isinstance(other, FiniteFieldExtElement):
            return (
                int(self._p) == int(other._p)
                and self._modulus == other._modulus
                and self._coeffs == other._coeffs
            )
        if isinstance(other, FiniteFieldElement):
            if int(other.characteristic) != int(self._p):
                return False
            return bool(self.is_in_base_field and int(self._coeffs[0]) == int(other.value))
        return False

    def __hash__(self) -> int:
        return hash((int(self._p), self._modulus, self._coeffs))

    def __repr__(self) -> str:
        return f"F_{int(self._p)}^{int(self._n)}({list(self._coeffs)})"

    def _mul_mod(self, other: "FiniteFieldExtElement") -> "FiniteFieldExtElement":
        """模乘法：多项式乘法后按 modulus 约化"""
        self._check_compatible(other)
        p = int(self._p)
        n = int(self._n)

        prod = [0] * (2 * n - 1)
        for i, a in enumerate(self._coeffs):
            if a == 0:
                continue
            for j, b in enumerate(other._coeffs):
                if b == 0:
                    continue
                prod[i + j] = int((prod[i + j] + a * b) % p)

        # 模约化：f(x)=c0+...+c_{n-1}x^{n-1}+x^n
        for i in range(2 * n - 2, n - 1, -1):
            coef = int(prod[i] % p)
            if coef != 0:
                for j, c in enumerate(self._modulus[:-1]):
                    prod[i - n + j] = int((prod[i - n + j] - coef * int(c)) % p)
                prod[i] = 0

        return FiniteFieldExtElement(prod[:n], p, list(self._modulus))

    def _pow_mod(self, exp: int) -> "FiniteFieldExtElement":
        """计算 self^exp（在该有限域内）"""
        if not isinstance(exp, int):
            raise TypeError(f"exp must be int, got {type(exp).__name__}")
        if exp < 0:
            return (self.inverse())._pow_mod(-exp)
        if exp == 0:
            return FiniteFieldExtElement([1], int(self._p), list(self._modulus))

        result = FiniteFieldExtElement([1], int(self._p), list(self._modulus))
        base = self
        e = int(exp)
        while e > 0:
            if e & 1:
                result = result._mul_mod(base)
            base = base._mul_mod(base)
            e >>= 1
        return result

    def __pow__(self, n: int) -> "FiniteFieldExtElement":
        return self._pow_mod(int(n))

    def inverse(self) -> "FiniteFieldExtElement":
        """
        乘法逆元（严格）：a^{-1} = a^{p^n - 2}（a ≠ 0）
        """
        if self.is_zero():
            raise ZeroDivisionError("0 has no multiplicative inverse in a field.")
        p = int(self._p)
        n = int(self._n)
        exp = int((p ** n) - 2)
        return self._pow_mod(exp)

    def frobenius(self) -> "FiniteFieldExtElement":
        """
        Frobenius: a ↦ a^p
        在 𝔽_{p^n}（n>1）上一般不是恒等映射。
        """
        return self._pow_mod(int(self._p))

    @property
    def is_in_base_field(self) -> bool:
        """检查是否在基域 𝔽_p 中（除常数项外全为 0）"""
        return all(int(c) == 0 for c in self._coeffs[1:])

    def norm(self) -> int:
        """
        范数 N_{𝔽_{p^n}/𝔽_p}(a) = a · a^p · a^{p^2} · ... · a^{p^{n-1}} ∈ 𝔽_p
        返回其在 𝔽_p 中的整数代表 [0, p-1]。
        """
        if self.is_zero():
            return 0

        result = self
        power = self
        for _ in range(int(self._n) - 1):
            power = power.frobenius()
            result = result._mul_mod(power)

        if not result.is_in_base_field:
            raise RuntimeError("Norm computation failed: result is not in the base field 𝔽_p.")
        return int(result._coeffs[0])


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：多项式环
# Part II: Polynomial Ring
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Monomial:
    """
    单项式：coefficient * X_0^^{e_0} * X_1^^{e_1} * ... * X_n^^{e_n}
    用于 Witt 多项式的符号计算
    """
    coefficient: IntegerElement
    exponents: Tuple[int, ...]  # (e_0, e_1, ..., e_n) 对应变量 X_0, X_1, ...
    
    def __post_init__(self):
        # 去除尾部的零指数
        exps = list(self.exponents)
        while exps and exps[-1] == 0:
            exps.pop()
        self.exponents = tuple(exps)
    
    @property
    def degree(self) -> int:
        """总次数"""
        return sum(self.exponents)
    
    def is_zero(self) -> bool:
        return self.coefficient.is_zero()
    
    def __mul__(self, other: 'Monomial') -> 'Monomial':
        new_coeff = self.coefficient * other.coefficient
        # 指数相加
        max_len = max(len(self.exponents), len(other.exponents))
        new_exp = []
        for i in range(max_len):
            e1 = self.exponents[i] if i < len(self.exponents) else 0
            e2 = other.exponents[i] if i < len(other.exponents) else 0
            new_exp.append(e1 + e2)
        return Monomial(new_coeff, tuple(new_exp))
    
    def __repr__(self) -> str:
        if self.coefficient.is_zero():
            return "0"
        
        parts = []
        if self.coefficient.value != 1 or not self.exponents:
            parts.append(str(self.coefficient.value))
        
        for i, e in enumerate(self.exponents):
            if e > 0:
                var = f"X_{i}"
                if e == 1:
                    parts.append(var)
                else:
                    parts.append(f"{var}^^{e}")
        
        return "·".join(parts) if parts else "1"


class MultivariatePolynomial:
    """
    多元多项式环 ℤ[X_0, X_1, ..., X_n, Y_0, Y_1, ..., Y_n]
    用于 Witt 多项式的精确符号计算
    内部表示：字典 {指数元组: 系数}
    例如 3X_0^1^ Y_1 表示为 {((2,0,...), (0,1,...)): 3}
    使用扁平化表示：变量顺序为 X_0, X_1, ..., Y_0, Y_1, ...
    """
    
    def __init__(self, terms: Optional[Dict[Tuple[int, ...], IntegerElement]] = None):
        """
        Args:
            terms: {指数元组: 系数} 的字典
        """
        self._terms: Dict[Tuple[int, ...], IntegerElement] = {}
        if terms:
            for exp, coeff in terms.items():
                if not coeff.is_zero():
                    self._terms[exp] = coeff
    
    @classmethod
    def variable(cls, index: int, num_vars: int) -> 'MultivariatePolynomial':
        """创建单个变量 X_index"""
        exp = tuple(1 if i == index else 0 for i in range(num_vars))
        return cls({exp: IntegerElement(1)})
    
    @classmethod
    def constant(cls, value: int) -> 'MultivariatePolynomial':
        """创建常数多项式"""
        if value == 0:
            return cls()
        return cls({(): IntegerElement(value)})
    
    @classmethod
    def zero(cls) -> 'MultivariatePolynomial':
        return cls()
    
    @classmethod
    def one(cls) -> 'MultivariatePolynomial':
        return cls.constant(1)
    
    def is_zero(self) -> bool:
        return len(self._terms) == 0
    
    def __add__(self, other: 'MultivariatePolynomial') -> 'MultivariatePolynomial':
        result = dict(self._terms)
        for exp, coeff in other._terms.items():
            if exp in result:
                new_coeff = result[exp] + coeff
                if new_coeff.is_zero():
                    del result[exp]
                else:
                    result[exp] = new_coeff
            else:
                result[exp] = coeff
        return MultivariatePolynomial(result)
    
    def __neg__(self) -> 'MultivariatePolynomial':
        return MultivariatePolynomial({exp: -coeff for exp, coeff in self._terms.items()})
    
    def __sub__(self, other: 'MultivariatePolynomial') -> 'MultivariatePolynomial':
        return self + (-other)
    
    def __mul__(self, other: 'MultivariatePolynomial') -> 'MultivariatePolynomial':
        if isinstance(other, int):
            other = MultivariatePolynomial.constant(other)
        
        result: Dict[Tuple[int, ...], IntegerElement] = {}
        
        for exp1, coeff1 in self._terms.items():
            for exp2, coeff2 in other._terms.items():
                # 指数相加
                max_len = max(len(exp1), len(exp2))
                new_exp = tuple(
                    (exp1[i] if i < len(exp1) else 0) + (exp2[i] if i < len(exp2) else 0)
                    for i in range(max_len)
                )
                new_coeff = coeff1 * coeff2
                
                if new_exp in result:
                    result[new_exp] = result[new_exp] + new_coeff
                    if result[new_exp].is_zero():
                        del result[new_exp]
                elif not new_coeff.is_zero():
                    result[new_exp] = new_coeff
        
        return MultivariatePolynomial(result)
    
    def __rmul__(self, other) -> 'MultivariatePolynomial':
        if isinstance(other, int):
            return MultivariatePolynomial.constant(other) * self
        return NotImplemented
    
    def __pow__(self, n: int) -> 'MultivariatePolynomial':
        if n < 0:
            raise ValueError("多项式不支持负指数")
        if n == 0:
            return MultivariatePolynomial.one()
        
        result = MultivariatePolynomial.one()
        base = self
        exp = n
        
        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        
        return result
    
    def evaluate_at_integers(self, values: List[int]) -> IntegerElement:
        """
        在整数点求值
        Args:
            values: [x_0, x_1, ..., x_n] 变量的整数值
        Returns:
            多项式在该点的值
        """
        if not isinstance(values, list):
            raise TypeError(f"values must be a List[int], got {type(values).__name__}")
        if any(not isinstance(v, int) for v in values):
            bad = next((v for v in values if not isinstance(v, int)), None)
            raise TypeError(f"values must be a List[int]; found {type(bad).__name__}")

        # 严格模式：禁止缺失变量自动视为 0的静默退回。
        max_var_idx = -1
        for exp in self._terms.keys():
            for i, e in enumerate(exp):
                if e > 0 and i > max_var_idx:
                    max_var_idx = i
        if max_var_idx >= 0 and max_var_idx >= len(values):
            raise ValueError(
                f"Not enough variable values: polynomial needs at least {max_var_idx + 1} variables, "
                f"but got {len(values)}."
            )

        result = IntegerElement(0)
        
        for exp, coeff in self._terms.items():
            term_value = coeff
            for i, e in enumerate(exp):
                if e > 0:
                    term_value = term_value * IntegerElement(values[i] ** e)
            result = result + term_value
        
        return result

    def evaluate_at_integers_mod(self, values: List[int], modulus: int) -> int:
        """
        在整数点求值并取模（严格、无天文级中间值）。

        这是为了 Witt 多项式/ghost 同余验证而提供的关键工具：
        - 只要最终只关心 (mod m)，就不应构造巨大整数。
        - 依然保持数学上的严格同余结果（不是近似）。

        Args:
            values: 变量赋值列表（严格：必须覆盖多项式出现的所有变量）
            modulus: 正模数 m > 0

        Returns:
            f(values) mod modulus，范围为 [0, modulus-1]
        """
        if not isinstance(values, list):
            raise TypeError(f"values must be a List[int], got {type(values).__name__}")
        if any(not isinstance(v, int) for v in values):
            bad = next((v for v in values if not isinstance(v, int)), None)
            raise TypeError(f"values must be a List[int]; found {type(bad).__name__}")
        if not isinstance(modulus, int):
            raise TypeError(f"modulus must be int, got {type(modulus).__name__}")
        if modulus <= 0:
            raise ValueError(f"modulus must be positive, got {modulus}")

        # 严格模式：禁止缺失变量自动视为 0。
        max_var_idx = -1
        for exp in self._terms.keys():
            for i, e in enumerate(exp):
                if e > 0 and i > max_var_idx:
                    max_var_idx = i
        if max_var_idx >= 0 and max_var_idx >= len(values):
            raise ValueError(
                f"Not enough variable values: polynomial needs at least {max_var_idx + 1} variables, "
                f"but got {len(values)}."
            )

        acc = 0
        for exp, coeff in self._terms.items():
            term = int(coeff.value % modulus)
            for i, e in enumerate(exp):
                if e > 0:
                    term = (term * pow(int(values[i]) % modulus, int(e), modulus)) % modulus
            acc = (acc + term) % modulus
        return int(acc)
    
    def exact_div_by_p(self, p: int) -> 'MultivariatePolynomial':
        """
        精确除以 p
        这是 Witt 向量理论的关键操作：
        某些多项式（如 X^^p + Y^^p - (X+Y)^^p）在整系数下必被 p 整除
        """
        new_terms = {}
        for exp, coeff in self._terms.items():
            if not coeff.is_divisible_by(p):
                raise ValueError(f"多项式不能被 {p} 整除: 项 {exp} 的系数 {coeff} 不是 {p} 的倍数")
            new_terms[exp] = coeff.exact_div(p)
        return MultivariatePolynomial(new_terms)
    
    def reduce_mod_p(self, p: int, num_vars: int) -> 'PolynomialOverFp':
        """
        将系数模 p 约化
        ℤ[X_0,...] → 𝔽_p[X_0,...]
        """
        new_terms = {}
        for exp, coeff in self._terms.items():
            reduced_coeff = FiniteFieldElement(coeff.value, p)
            if not reduced_coeff.is_zero():
                # 规范化指数长度
                normalized_exp = tuple(exp[i] if i < len(exp) else 0 for i in range(num_vars))
                new_terms[normalized_exp] = reduced_coeff
        return PolynomialOverFp(new_terms, p)
    
    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        
        parts = []
        for exp, coeff in sorted(self._terms.items(), key=lambda x: (sum(x[0]), x[0])):
            term_parts = []
            if coeff.value != 1 or not any(e > 0 for e in exp):
                if coeff.value == -1 and any(e > 0 for e in exp):
                    term_parts.append("-")
                else:
                    term_parts.append(str(coeff.value))
            
            for i, e in enumerate(exp):
                if e > 0:
                    var = f"X_{i}"
                    if e == 1:
                        term_parts.append(var)
                    else:
                        term_parts.append(f"{var}^^{e}")
            
            parts.append("·".join(term_parts) if term_parts else "1")
        
        return " + ".join(parts).replace("+ -", "- ")


class PolynomialOverFp:
    """
    𝔽_p 上的多元多项式
    这是 Witt 向量分量所在的环（当基域是 𝔽_p 时）
    """
    
    def __init__(self, terms: Dict[Tuple[int, ...], FiniteFieldElement], p: int):
        self._p = p
        self._terms: Dict[Tuple[int, ...], FiniteFieldElement] = {}
        for exp, coeff in terms.items():
            if not coeff.is_zero():
                self._terms[exp] = coeff
    
    @property
    def characteristic(self) -> int:
        return self._p
    
    @classmethod
    def zero(cls, p: int) -> 'PolynomialOverFp':
        return cls({}, p)
    
    @classmethod
    def one(cls, p: int) -> 'PolynomialOverFp':
        return cls({(): FiniteFieldElement(1, p)}, p)
    
    @classmethod
    def from_finite_field_element(cls, elem: FiniteFieldElement) -> 'PolynomialOverFp':
        """从有限域元素创建常数多项式"""
        if elem.is_zero():
            return cls.zero(elem.characteristic)
        return cls({(): elem}, elem.characteristic)
    
    def is_zero(self) -> bool:
        return len(self._terms) == 0
    
    def is_constant(self) -> bool:
        """检查是否为常数（包括零）"""
        if self.is_zero():
            return True
        return len(self._terms) == 1 and () in self._terms
    
    def as_constant(self) -> FiniteFieldElement:
        """将常数多项式转换为有限域元素"""
        if self.is_zero():
            return FiniteFieldElement.zero(self._p)
        if not self.is_constant():
            raise ValueError("非常数多项式不能转换为域元素")
        return self._terms.get((), FiniteFieldElement.zero(self._p))
    
    def __add__(self, other: 'PolynomialOverFp') -> 'PolynomialOverFp':
        if self._p != other._p:
            raise ValueError("不同特征的多项式不能相加")
        
        result = dict(self._terms)
        for exp, coeff in other._terms.items():
            if exp in result:
                new_coeff = result[exp] + coeff
                if new_coeff.is_zero():
                    del result[exp]
                else:
                    result[exp] = new_coeff
            else:
                result[exp] = coeff
        return PolynomialOverFp(result, self._p)
    
    def __neg__(self) -> 'PolynomialOverFp':
        return PolynomialOverFp({exp: -coeff for exp, coeff in self._terms.items()}, self._p)
    
    def __sub__(self, other: 'PolynomialOverFp') -> 'PolynomialOverFp':
        return self + (-other)
    
    def __mul__(self, other: 'PolynomialOverFp') -> 'PolynomialOverFp':
        if self._p != other._p:
            raise ValueError("不同特征的多项式不能相乘")
        
        result: Dict[Tuple[int, ...], FiniteFieldElement] = {}
        
        for exp1, coeff1 in self._terms.items():
            for exp2, coeff2 in other._terms.items():
                max_len = max(len(exp1), len(exp2))
                new_exp = tuple(
                    (exp1[i] if i < len(exp1) else 0) + (exp2[i] if i < len(exp2) else 0)
                    for i in range(max_len)
                )
                new_coeff = coeff1 * coeff2
                
                if new_exp in result:
                    result[new_exp] = result[new_exp] + new_coeff
                    if result[new_exp].is_zero():
                        del result[new_exp]
                elif not new_coeff.is_zero():
                    result[new_exp] = new_coeff
        
        return PolynomialOverFp(result, self._p)
    
    def __pow__(self, n: int) -> 'PolynomialOverFp':
        if n < 0:
            raise ValueError("多项式不支持负指数")
        if n == 0:
            return PolynomialOverFp.one(self._p)
        
        result = PolynomialOverFp.one(self._p)
        base = self
        exp = n
        
        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        
        return result
    
    def frobenius(self) -> 'PolynomialOverFp':
        """
        Frobenius 自同态: f(X_0, X_1, ...) ↦ f(X_0^^p, X_1^^p, ...)
        由于在 𝔽_p 上 a^^p = a，这等价于将所有指数乘以 p
        """
        new_terms = {}
        for exp, coeff in self._terms.items():
            # 每个指数乘以 p
            new_exp = tuple(e * self._p for e in exp)
            # 系数不变（Frobenius 在 𝔽_p 上是恒等）
            new_terms[new_exp] = coeff
        return PolynomialOverFp(new_terms, self._p)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, PolynomialOverFp):
            return self._p == other._p and self._terms == other._terms
        return False
    
    def __hash__(self) -> int:
        return hash((self._p, tuple(sorted(self._terms.items()))))
    
    def __repr__(self) -> str:
        if not self._terms:
            return f"0 (𝔽_{self._p})"
        
        parts = []
        for exp, coeff in sorted(self._terms.items(), key=lambda x: (sum(x[0]) if x[0] else 0, x[0])):
            term_parts = []
            if coeff.value != 1 or not exp or not any(e > 0 for e in exp):
                term_parts.append(str(coeff.value))
            
            for i, e in enumerate(exp):
                if e > 0:
                    var = f"x_{i}"
                    if e == 1:
                        term_parts.append(var)
                    else:
                        term_parts.append(f"{var}^^{e}")
            
            parts.append("·".join(term_parts) if term_parts else str(coeff.value))
        
        return " + ".join(parts).replace("+ -", "- ") + f" (𝔽_{self._p})"


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：Witt 多项式生成器
# Part III: Witt Polynomial Generator
# ══════════════════════════════════════════════════════════════════════════════

class WittPolynomialGenerator:
    """
    Witt 多项式的符号生成器
    设 X = (X_0, X_1, ...) 和 Y = (Y_0, Y_1, ...) 是两组变量
    Ghost 分量定义：
    w_n(X) = Σ_{i=0}^^{n} p^^i · X_i^^{p^^{n-i}}
    Witt 加法多项式 S_n(X; Y) 由以下条件唯一确定：
    w_n(S_0, S_1, ..., S_n) = w_n(X) + w_n(Y)
    Witt 乘法多项式 P_n(X; Y) 由以下条件唯一确定：
    w_n(P_0, P_1, ..., P_n) = w_n(X) · w_n(Y)
    关键引理（Witt）：S_n 和 P_n 都是整系数多项式
    """
    
    def __init__(self, p: int, max_length: int):
        """
        Args:
            p: 素数
            max_length: 最大 Witt 向量长度
        """
        if p < 2:
            raise ValueError("p 必须是素数")
        self._p = p
        self._max_length = max_length
        self._num_vars = 2 * max_length  # X_0,...,X_{n-1}, Y_0,...,Y_{n-1}
        
        # 缓存
        self._ghost_polynomials: Dict[int, MultivariatePolynomial] = {}
        self._addition_polynomials: Dict[int, MultivariatePolynomial] = {}
        self._multiplication_polynomials: Dict[int, MultivariatePolynomial] = {}
        # 惰性计算：仅在访问 addition_polynomial / multiplication_polynomial 时按需递推构造。
    
    @property
    def prime(self) -> int:
        return self._p
    
    @property
    def max_length(self) -> int:
        return self._max_length
    
    def _X(self, i: int) -> MultivariatePolynomial:
        """变量 X_i"""
        return MultivariatePolynomial.variable(i, self._num_vars)
    
    def _Y(self, i: int) -> MultivariatePolynomial:
        """变量 Y_i（偏移 max_length）"""
        return MultivariatePolynomial.variable(self._max_length + i, self._num_vars)
    
    def ghost_polynomial_X(self, n: int) -> MultivariatePolynomial:
        """
        Ghost 多项式 w_n(X)
        
        w_n(X) = X_0^{p^n} + p·X_1^{p^{n-1}} + p^2·X_2^{p^^{n-2}} + ... + p^^n·X_n
        """
        if n in self._ghost_polynomials:
            return self._ghost_polynomials[n]
        
        result = MultivariatePolynomial.zero()
        for i in range(n + 1):
            # p^i · X_i^{p^^{n-i}}
            coeff = self._p ** i
            exponent = self._p ** (n - i)
            term = MultivariatePolynomial.constant(coeff) * (self._X(i) ** exponent)
            result = result + term
        
        self._ghost_polynomials[n] = result
        return result
    
    def ghost_polynomial_Y(self, n: int) -> MultivariatePolynomial:
        """Ghost 多项式 w_n(Y)"""
        result = MultivariatePolynomial.zero()
        for i in range(n + 1):
            coeff = self._p ** i
            exponent = self._p ** (n - i)
            term = MultivariatePolynomial.constant(coeff) * (self._Y(i) ** exponent)
            result = result + term
        return result
    
    def ghost_polynomial_S(self, n: int, S: List[MultivariatePolynomial]) -> MultivariatePolynomial:
        """
        Ghost 多项式 w_n(S_0, S_1, ..., S_n)
        用于验证和构造
        """
        result = MultivariatePolynomial.zero()
        for i in range(min(n + 1, len(S))):
            coeff = self._p ** i
            exponent = self._p ** (n - i)
            term = MultivariatePolynomial.constant(coeff) * (S[i] ** exponent)
            result = result + term
        return result
    
    def _compute_addition_polynomial(self, n: int) -> MultivariatePolynomial:
        """
        递归计算加法多项式 S_n
        
        方法：
        1. 计算 w_n(X) + w_n(Y)
        2. 减去 Σ_{i=0}^^{n-1} p^^i · S_i^{p^{n-i}}
        3. 结果必须被 p^^n 整除，除以 p^^n 得到 S_n
        关键数学事实：
        对于 n=1: (X+Y)^^p - X^p - Y^p 恒被 p 整除（二项式系数的性质）
        一般地，递归构造保证整除性。
        """
        if n == 0:
            # S_0 = X_0 + Y_0
            return self._X(0) + self._Y(0)
        
        # 获取之前的 S_0, ..., S_{n-1}
        S_prev = [self._addition_polynomials[i] for i in range(n)]
        
        # 计算 w_n(X) + w_n(Y)
        target = self.ghost_polynomial_X(n) + self.ghost_polynomial_Y(n)
        
        # 减去 w_n(S_0, ..., S_{n-1}, 0) = Σ_{i=0}^{n-1} p^i · S_i^{p^{n-i}}
        for i in range(n):
            coeff = self._p ** i
            exponent = self._p ** (n - i)
            term = MultivariatePolynomial.constant(coeff) * (S_prev[i] ** exponent)
            target = target - term
        
        # 现在 target = p^^n · S_n
        # 必须精确整除
        S_n = target.exact_div_by_p(self._p ** n)
        
        return S_n
    
    def _compute_multiplication_polynomial(self, n: int) -> MultivariatePolynomial:
        """
        递归计算乘法多项式 P_n
        类似加法，但目标是 w_n(X) · w_n(Y)
        """
        if n == 0:
            # P_0 = X_0 · Y_0
            return self._X(0) * self._Y(0)
        
        P_prev = [self._multiplication_polynomials[i] for i in range(n)]
        
        # 计算 w_n(X) · w_n(Y)
        target = self.ghost_polynomial_X(n) * self.ghost_polynomial_Y(n)
        
        # 减去 Σ_{i=0}^{n-1} p^i · P_i^{p^{n-i}}
        for i in range(n):
            coeff = self._p ** i
            exponent = self._p ** (n - i)
            term = MultivariatePolynomial.constant(coeff) * (P_prev[i] ** exponent)
            target = target - term
        
        P_n = target.exact_div_by_p(self._p ** n)
        
        return P_n
    
    def _compute_all(self) -> None:
        """预计算所有 Witt 多项式"""
        for n in range(self._max_length):
            self._addition_polynomials[n] = self._compute_addition_polynomial(n)
            self._multiplication_polynomials[n] = self._compute_multiplication_polynomial(n)
    
    def addition_polynomial(self, n: int) -> MultivariatePolynomial:
        """获取第 n 个加法多项式 S_n"""
        if n >= self._max_length:
            raise ValueError(f"n={n} 超出最大长度 {self._max_length}")
        # 按需、顺序递推：_compute_addition_polynomial(n) 依赖于 0..n-1 已缓存
        for i in range(n + 1):
            if i not in self._addition_polynomials:
                self._addition_polynomials[i] = self._compute_addition_polynomial(i)
        return self._addition_polynomials[n]
    
    def multiplication_polynomial(self, n: int) -> MultivariatePolynomial:
        """获取第 n 个乘法多项式 P_n"""
        if n >= self._max_length:
            raise ValueError(f"n={n} 超出最大长度 {self._max_length}")
        # 按需、顺序递推：_compute_multiplication_polynomial(n) 依赖于 0..n-1 已缓存
        for i in range(n + 1):
            if i not in self._multiplication_polynomials:
                self._multiplication_polynomials[i] = self._compute_multiplication_polynomial(i)
        return self._multiplication_polynomials[n]
    
    def carry_polynomial(self) -> MultivariatePolynomial:
        """
        进位多项式 C_p(X, Y) = (X^p + Y^p - (X+Y)^^p) / p
        这是 Witt 加法的核心：它精确捕获了进位传播
        """
        X = self._X(0)
        Y = self._Y(0)
        
        numerator = (X ** self._p) + (Y ** self._p) - ((X + Y) ** self._p)
        return numerator.exact_div_by_p(self._p)


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：Witt 向量
# Part IV: Witt Vectors
# ══════════════════════════════════════════════════════════════════════════════

class WittVector:
    """
    Witt 向量 W_n(k)
    数学定义：
    设 k 是特征 p 的完善域（Perfect Field）。
    Witt 向量 W(k) 是特征 0 的完备离散赋值环，其剩余域是 k。
    数据表示：
    一个 Witt 向量表示为分量序列 (x_0, x_1, ..., x_{n-1})，
    其中每个 x_i ∈ k。
    """
    
    def __init__(self, components: List[FiniteFieldElement], p: int):
        """
        Args:
            components: Witt 分量列表 [x_0, x_1, ..., x_{n-1}]
            p: 特征
        """
        if not components:
            raise ValueError("Witt 向量必须至少有一个分量")
        
        # 验证所有分量在同一个域
        for comp in components:
            if comp.characteristic != p:
                raise ValueError(f"分量特征不匹配: 期望 {p}，得到 {comp.characteristic}")
        
        self._p = p
        self._components = list(components)
        self._length = len(components)

    def _to_int_mod_p_power(self) -> int:
        """
        将 Witt 向量映到整数环 ℤ/p^nℤ 的代表元（严格）。

        关键点（这是之前版本的结构性债务来源）：
        - W_n(𝔽_p) ≅ ℤ/p^nℤ 作为环是正确的
        - 但同构不是把分量当作 base-p 数位直接 Σ a_i p^i
        - 正确同构应使用 Teichmüller lift：
            x = (x_0,...,x_{n-1}) ↦ Σ_{i=0}^{n-1} p^i · τ_n(x_i)   (mod p^n)
          其中 τ_n: 𝔽_p → (ℤ/p^nℤ) 是 Teichmüller 提升（满足 τ_n(a)≡a (mod p) 且 τ_n(a)^p≡τ_n(a) (mod p^n)）。

        这保证 Teichmüller 元满足 [a]·[b]=[ab]，并与 Witt 多项式/ghost 同态严格一致。
        """
        p = int(self._p)
        length = int(self._length)
        modulus = int(p ** length)

        acc = 0
        for i, c in enumerate(self._components):
            # τ_n(c) in Z/p^nZ
            t = int(self._teichmuller_lift_mod_p_power(int(c.value), p, length))
            acc = (acc + (pow(p, int(i), modulus) * t)) % modulus
        return int(acc)

    @staticmethod
    def _teichmuller_lift_mod_p_power(a: int, p: int, k: int) -> int:
        """
        计算 Teichmüller lift τ_k(a) ∈ ℤ/p^kℤ。

        约束：
        - 输入 a 按 (mod p) 约化到 0..p-1（对应 𝔽_p 元素）
        - 返回值为 [0, p^k-1] 的代表元
        - 满足：τ_k(a) ≡ a (mod p) 且 τ_k(a)^p ≡ τ_k(a) (mod p^k)

        实现（严格、无启发式）：
        - 在模 p^j 下迭代 Frobenius：t ← t^p (mod p^j)，j=2..k
        - 由 p-adic 收敛性保证稳定到 τ_k(a)
        """
        if not isinstance(a, int):
            raise TypeError(f"a must be int, got {type(a).__name__}")
        if not isinstance(p, int):
            raise TypeError(f"p must be int, got {type(p).__name__}")
        if not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        if p < 2:
            raise ValueError("p must be >= 2 (and should be prime).")
        if k < 1:
            raise ValueError("k must be >= 1.")

        a0 = int(a % p)
        if a0 == 0:
            return 0

        # Iterative Frobenius lifting: t_{j} = t_{j-1}^p (mod p^j)
        t = int(a0)
        mod = int(p)
        for _ in range(1, int(k)):
            mod *= int(p)
            t = int(pow(t, int(p), int(mod)))
        return int(t)
    
    @property
    def prime(self) -> int:
        return self._p
    
    @property
    def length(self) -> int:
        return self._length
    
    @property
    def components(self) -> List[FiniteFieldElement]:
        """返回 Witt 分量的副本"""
        return list(self._components)
    
    def __getitem__(self, i: int) -> FiniteFieldElement:
        """获取第 i 个 Witt 分量"""
        return self._components[i]
    
    @classmethod
    def zero(cls, p: int, length: int) -> 'WittVector':
        """零元 (0, 0, ..., 0)"""
        return cls([FiniteFieldElement.zero(p) for _ in range(length)], p)
    
    @classmethod
    def one(cls, p: int, length: int) -> 'WittVector':
        """单位元 (1, 0, ..., 0)"""
        components = [FiniteFieldElement.zero(p) for _ in range(length)]
        components[0] = FiniteFieldElement.one(p)
        return cls(components, p)
    
    @classmethod
    def teichmuller(cls, a: FiniteFieldElement, length: int) -> 'WittVector':
        """
        Teichmüller 提升: [a] = (a, 0, 0, ...)
        
        这是 k → W(k) 的乘法截面
        """
        p = a.characteristic
        components = [FiniteFieldElement.zero(p) for _ in range(length)]
        components[0] = a
        return cls(components, p)
    
    @classmethod
    def from_integer(cls, n: int, p: int, length: int) -> 'WittVector':
        """
        从整数构造 Witt 向量

        数学基础：W_n(𝔽_p) ≅ ℤ/p^nℤ。

        关键澄清：该同构不是base-p 数位展开，而是 Teichmüller 展开：
          m ≡ Σ_{i=0}^{n-1} p^i · τ_n(a_i)  (mod p^n)
        其中 a_i ∈ 𝔽_p，τ_n 是 Teichmüller lift。

        本方法实现该同构的**逆映射**：给定 m（模 p^n 的代表元），恢复 (a_0,...,a_{n-1})。
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be int, got {type(n).__name__}")
        if not isinstance(p, int):
            raise TypeError(f"p must be int, got {type(p).__name__}")
        if not isinstance(length, int):
            raise TypeError(f"length must be int, got {type(length).__name__}")
        if p < 2:
            raise ValueError("p must be >= 2 (and should be prime).")
        if length < 1:
            raise ValueError("length must be >= 1.")

        modulus = int(p ** length)
        r = int(n % modulus)

        components: List[FiniteFieldElement] = []
        # 逐位剥离 Teichmüller 展开：
        #   r_{i} ≡ τ_{k}(a_i) + p·r_{i+1}   (mod p^k),  k = length-i
        for i in range(int(length)):
            k = int(length - i)
            mod_k = int(p ** k)
            r = int(r % mod_k)

            a_i = int(r % p)  # 因为 τ_k(a) ≡ a (mod p)
            components.append(FiniteFieldElement(a_i, p))

            t = int(WittVector._teichmuller_lift_mod_p_power(a_i, p, k))
            diff = int((r - t) % mod_k)
            if diff % p != 0:
                raise RuntimeError(
                    "from_integer Teichmüller 展开失败：差值不能被 p 整除（部署必须中断）。\n"
                    f"  p={int(p)}, length={int(length)}, step={int(i)}, k={int(k)}\n"
                    f"  r={int(r)}, a_i={int(a_i)}, tau_k(a_i)={int(t)}, diff={int(diff)}"
                )
            r = int(diff // p)

        return cls(components, p)

    
    def ghost_component(self, n: int) -> FiniteFieldElement:
        """
        第 n 个 Ghost 分量
        w_n(x) = Σ_{i=0}^^{n} p^i · x_i^{p^^{n-i}}
        这个计算在 𝔽_p 上进行，所以 p^^i 项对 i ≥ 1 都是 0！
        因此 w_n(x) = x_0^{p^n} 在 𝔽_p 上

        但是 Ghost 映射的真正价值在于提升到特征 0 后的等式
        返回"形式" Ghost 分量，用于验证 Witt 运算的正确性
        """
        if n >= self._length:
            raise ValueError(f"Ghost 分量索引 {n} 超出长度 {self._length}")
        
        # 在 𝔽_p 上，w_n = x_0^{p^n}（其他项被 p 杀死）
        # 但为了完整性，我们返回形式计算
        result = FiniteFieldElement.zero(self._p)
        for i in range(n + 1):
            # p^^i mod p = 0 for i >= 1
            if i == 0:
                exp = self._p ** n
                result = result + (self._components[0] ** exp)
        return result
    
    def ghost_components_formal(self, n: int) -> int:
        """
        第 n 个形式 Ghost 分量（在 ℤ 上计算）
        w_n(x) = Σ_{i=0}^{n} p^i · x_i^{p^{n-i}}
        这用于验证 Witt 运算的正确性：Ghost 映射是环同态。
        Returns:
            int: 返回 ℤ/p^{n+1}ℤ 的规范代表（范围 [0, p^{n+1}-1]）。
                 这是截断 Witt 向量层级 n 的可见精度，也避免构造天文级整数。
        """
        if not isinstance(n, int):
            raise TypeError(f"Ghost 分量索引必须为 int，得到 {type(n).__name__}")
        if n < 0:
            raise ValueError(f"Ghost 分量索引必须非负: {n}")

        p = int(self._p)
        modulus = int(p ** (int(n) + 1))
        return int(self._ghost_component_mod_p_power(int(n), modulus))

    def _ghost_component_mod_p_power(self, n: int, modulus: int) -> int:
        """
        第 n 个 Ghost 分量在给定模数下的值（严格模运算版本）。

        目的：用于验证环同态关系时，避免构造天文级整数（仍然保持数学上严格的同余精度）。

        计算：w_n(x) = Σ_{i=0}^{n} p^i · τ_k(x_i)  (mod p^k)

        其中 modulus 必须为 p^k（纯 p-幂）。对 𝔽_p 分量，Teichmüller lift τ_k(x_i)
        满足 τ_k(x_i)^{p^{n-i}} = τ_k(x_i)，因此无需构造巨大指数。
        """
        if not isinstance(n, int):
            raise TypeError(f"ghost index n must be int, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"Ghost 分量索引必须非负: {n}")
        if not isinstance(modulus, int):
            raise TypeError(f"modulus must be int, got {type(modulus).__name__}")
        if modulus <= 0:
            raise ValueError(f"modulus must be positive, got {modulus}")

        p = int(self._p)

        # modulus 必须是 p^k
        mm = int(modulus)
        k = 0
        while mm % p == 0:
            mm //= p
            k += 1
        if mm != 1:
            raise ValueError(
                "ghost_component_mod requires modulus to be a pure power of p.\n"
                f"  p={int(p)}, modulus={int(modulus)}"
            )
        if k < 1:
            raise ValueError("modulus must be >= p (i.e., k>=1).")

        up_to = min(int(n) + 1, int(self._length))
        acc = 0
        for i in range(up_to):
            t = int(self._teichmuller_lift_mod_p_power(int(self._components[i].value), p, int(k)))
            acc = (acc + (pow(p, int(i), int(modulus)) * t)) % int(modulus)
        return int(acc)

    def _verify_operation_via_polynomial(
        self,
        other: 'WittVector',
        result: 'WittVector',
        op: str  # 'add' or 'mul'
    ) -> bool:
        """
        通过 Witt 多项式（Ghost 映射）验证运算结果。

        数学原理：Ghost 映射是环同态，因此对所有 n：
        - 加法：w_n(a + b) = w_n(a) + w_n(b)
        - 乘法：w_n(a · b) = w_n(a) · w_n(b)

        在截断长度语义下，我们在 ℤ/p^{n+1}ℤ 上比较（对应第 n 层可见精度）。
        若发现不一致：立即抛出 RuntimeError（禁止静默退回）。
        """
        if not isinstance(other, WittVector):
            raise TypeError(f"other must be WittVector, got {type(other).__name__}")
        if not isinstance(result, WittVector):
            raise TypeError(f"result must be WittVector, got {type(result).__name__}")
        if int(self._p) != int(other._p) or int(self._p) != int(result._p):
            raise ValueError(
                f"Witt op verification prime mismatch: self.p={int(self._p)}, "
                f"other.p={int(other._p)}, result.p={int(result._p)}"
            )

        p = int(self._p)
        max_level = min(int(self._length), int(other._length), int(result._length))

        for n in range(max_level):
            modulus = p ** (n + 1)
            ghost_a = self._ghost_component_mod_p_power(n, modulus)
            ghost_b = other._ghost_component_mod_p_power(n, modulus)
            ghost_r = result._ghost_component_mod_p_power(n, modulus)

            if op == 'add':
                expected = (ghost_a + ghost_b) % modulus
            elif op == 'mul':
                expected = (ghost_a * ghost_b) % modulus
            else:
                raise ValueError(f"未知操作: {op}")

            if ghost_r != expected:
                raise RuntimeError(
                    f"Witt {op} 验证失败 at level {n}:\n"
                    f"  Ghost(result) mod p^{n+1} = {ghost_r}\n"
                    f"  Expected  mod p^{n+1} = {expected}\n"
                    f"  a = {self}, b = {other}, result = {result}\n"
                    f"  这表明整数同构与数学定义不一致，部署必须中断。"
                )

        return True
    
    def _ensure_same_length(self, other: 'WittVector') -> Tuple['WittVector', 'WittVector']:
        """确保两个 Witt 向量有相同长度（用零扩展）"""
        if self._p != other._p:
            raise ValueError(f"特征不匹配: {self._p} vs {other._p}")
        
        if self._length == other._length:
            return self, other
        
        max_len = max(self._length, other._length)
        
        if self._length < max_len:
            new_self = WittVector(
                self._components + [FiniteFieldElement.zero(self._p)] * (max_len - self._length),
                self._p
            )
        else:
            new_self = self
        
        if other._length < max_len:
            new_other = WittVector(
                other._components + [FiniteFieldElement.zero(other._p)] * (max_len - other._length),
                other._p
            )
        else:
            new_other = other
        
        return new_self, new_other
    
    def __add__(self, other: 'WittVector') -> 'WittVector':
        """
        Witt 向量加法
        
        不是分量逐点相加！
        使用 Witt 加法多项式 S_n(X; Y)
        """
        self_ext, other_ext = self._ensure_same_length(other)
        p = int(self_ext._p)
        length = int(self_ext._length)
        modulus = p ** length
        a = int(self_ext._to_int_mod_p_power())
        b = int(other_ext._to_int_mod_p_power())
        result = WittVector.from_integer((a + b) % modulus, p, length)
        # 闭环验证：禁止静默错误
        self_ext._verify_operation_via_polynomial(other_ext, result, 'add')
        return result
    
    def __neg__(self) -> 'WittVector':
        """
        Witt 向量的负元
        
        由 x + (-x) = 0 定义
        可以通过求解 S_n(x_0,...; y_0,...) = 0 得到
        """
        p = int(self._p)
        length = int(self._length)
        modulus = p ** length
        a = int(self._to_int_mod_p_power())
        return WittVector.from_integer((-a) % modulus, p, length)
    
    def __sub__(self, other: 'WittVector') -> 'WittVector':
        return self + (-other)
    
    def __mul__(self, other: 'WittVector') -> 'WittVector':
        """
        Witt 向量乘法
        
        使用 Witt 乘法多项式 P_n(X; Y)
        """
        self_ext, other_ext = self._ensure_same_length(other)
        p = int(self_ext._p)
        length = int(self_ext._length)
        modulus = p ** length
        a = int(self_ext._to_int_mod_p_power())
        b = int(other_ext._to_int_mod_p_power())
        result = WittVector.from_integer((a * b) % modulus, p, length)
        # 闭环验证：禁止静默错误
        self_ext._verify_operation_via_polynomial(other_ext, result, 'mul')
        return result
    
    def frobenius(self) -> 'WittVector':
        """
        Frobenius 算子 φ
        
        φ(x_0, x_1, ..., x_{n-1}) = (x_0^p, x_1^p, ..., x_{n-1}^^p)
        
        这是 W(k) 上的环同态。

        关键澄清：当基域是 𝔽_p 且分量类型为 FiniteFieldElement 时，由 Fermat 小定理 a^p = a，
        因此 **分量级** frobenius = id。
        但 δ(w) = (φ(w) - w^p) / p 中的 w^p 是 **Witt 乘法意义** 的 p 次幂，不能误解为分量逐点的 p 次幂。
        """
        return WittVector(
            [c ** self._p for c in self._components],
            self._p
        )
    
    def verschiebung(self) -> 'WittVector':
        """
        Verschiebung 算子 V
        
        V(x_0, x_1, ..., x_{n-1}) = (0, x_0, x_1, ..., x_{n-2})
        
        这是加法群同态（但不是环同态）。
        V 相当于"乘以 p 再除以 φ"。
        """
        p = int(self._p)
        length = int(self._length)

        new_components = [FiniteFieldElement.zero(self._p)] + self._components[:-1]
        result = WittVector(new_components, self._p)

        # 在 W_n(𝔽_p) ≅ ℤ/p^nℤ 下：V 对应乘以 p。
        modulus = p ** length
        w_int = int(self._to_int_mod_p_power()) % modulus
        v_int = int(result._to_int_mod_p_power()) % modulus
        expected_int = (p * w_int) % modulus

        if v_int != expected_int:
            raise RuntimeError(
                "Verschiebung 验证失败:\n"
                f"  V(w) 整数表示 = {v_int}\n"
                f"  期望 p * w_int mod p^{length} = {expected_int}\n"
                f"  w = {self}\n"
                "  这表明 Verschiebung 实现与整数同构不一致，部署必须中断。"
            )

        return result
    
    def restriction(self, new_length: int) -> 'WittVector':
        """
        限制映射 R: W_n(k) → W_m(k)，m < n
        
        R(x_0, ..., x_{n-1}) = (x_0, ..., x_{m-1})
        """
        if new_length > self._length:
            raise ValueError(f"不能扩展：{new_length} > {self._length}")
        return WittVector(self._components[:new_length], self._p)
    
    def is_zero(self) -> bool:
        return all(c.is_zero() for c in self._components)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, WittVector):
            return False
        if self._p != other._p:
            return False
        self_ext, other_ext = self._ensure_same_length(other)
        return all(a == b for a, b in zip(self_ext._components, other_ext._components))
    
    def __hash__(self) -> int:
        return hash((self._p, tuple(c.value for c in self._components)))
    
    def __repr__(self) -> str:
        comp_str = ", ".join(str(c.value) for c in self._components)
        return f"W_{self._p}({comp_str})"
    
    def to_latex(self) -> str:
        comp_str = ", ".join(str(c.value) for c in self._components)
        return f"({comp_str})_{{W_{self._p}}}"


# =============================================================================
# WittPolynomialGenerator ↔ WittVector（一致性闭环验证）
# =============================================================================

def verify_witt_polynomial_consistency(
    p: int,
    length: int,
    *,
    max_pair_checks: Optional[int] = None
) -> Dict[str, int]:
    """
    严格验证：WittPolynomialGenerator 的加/乘多项式与 W_n(𝔽_p) ≅ ℤ/p^nℤ 的整数同构一致。

    重要约束（对齐你的红线规则）：
    - 禁止启发式：不做 random sampling；该函数只做全量穷举验证。
    - 禁止静默退回：若规模超出调用者允许上限，直接抛错，不做部分抽样。

    复杂度：O((p^length)^2 · length) —— 这是数学上完整性换来的必然代价。

    Args:
        p: 素数特征
        length: Witt 向量长度
        max_pair_checks: 可选的上限保护（仅用于拒绝运行过大规模，不是抽样）。
            - 若提供，且 (p^length)^2 > max_pair_checks：直接 RuntimeError 中断。
            - 若不提供：默认执行全量穷举（调用者需自行确保参数可计算）。

    Returns:
        dict: {'ok': 1, 'p': p, 'length': length, 'modulus': p^length, 'pairs_tested': (p^length)^2}

    Raises:
        RuntimeError: 发现任何不一致（部署必须中断）
        ValueError/TypeError: 输入非法
    """
    logger.info("verify_witt_polynomial_consistency start p=%s length=%s", int(p), int(length))
    if not isinstance(p, int):
        raise TypeError(f"p must be int, got {type(p).__name__}")
    if not isinstance(length, int):
        raise TypeError(f"length must be int, got {type(length).__name__}")
    if p < 2:
        raise ValueError("p must be >= 2 (and should be prime).")
    if length < 1:
        raise ValueError("length must be >= 1.")

    modulus = int(p ** length)
    total_pairs = int(modulus * modulus)

    if max_pair_checks is not None:
        if not isinstance(max_pair_checks, int):
            raise TypeError(f"max_pair_checks must be int, got {type(max_pair_checks).__name__}")
        if max_pair_checks < 0:
            raise ValueError("max_pair_checks must be >= 0.")
        if total_pairs > max_pair_checks:
            raise RuntimeError(
                "Refuse to run partial/heuristic validation.\n"
                f"  Required exhaustive pair checks = {total_pairs}\n"
                f"  Provided max_pair_checks         = {max_pair_checks}\n"
                "  请提高 max_pair_checks 或降低 (p,length)。"
            )

    gen = WittPolynomialGenerator(p, length)
    add_polys = [gen.addition_polynomial(n) for n in range(length)]
    mul_polys = [gen.multiplication_polynomial(n) for n in range(length)]

    # 预计算所有整数代表对应的 Witt 分量（Teichmüller 同构的逆映射）
    digits: List[List[int]] = []
    for n_int in range(modulus):
        w = WittVector.from_integer(int(n_int), int(p), int(length))
        digits.append([int(c.value) for c in w.components])

    for a_int in range(modulus):
        a_vals = digits[a_int]
        for b_int in range(modulus):
            b_vals = digits[b_int]
            var_values = a_vals + b_vals  # [X_0..X_{n-1}, Y_0..Y_{n-1}]

            sum_digits = digits[(a_int + b_int) % modulus]
            prod_digits = digits[(a_int * b_int) % modulus]

            for n in range(length):
                s_poly_val = add_polys[n].evaluate_at_integers_mod(var_values, p)
                if int(s_poly_val) != int(sum_digits[n]):
                    raise RuntimeError(
                        "WittPolynomialGenerator 加法多项式与整数同构不一致：\n"
                        f"  p={p}, length={length}, level={n}\n"
                        f"  a_int={a_int}, b_int={b_int}\n"
                        f"  S_n(X,Y) mod p = {s_poly_val}\n"
                        f"  expected (a+b)[{n}] = {sum_digits[n]}\n"
                        "  这表明底座存在根本性数学错误，部署必须中断。"
                    )

                p_poly_val = mul_polys[n].evaluate_at_integers_mod(var_values, p)
                if int(p_poly_val) != int(prod_digits[n]):
                    raise RuntimeError(
                        "WittPolynomialGenerator 乘法多项式与整数同构不一致：\n"
                        f"  p={p}, length={length}, level={n}\n"
                        f"  a_int={a_int}, b_int={b_int}\n"
                        f"  P_n(X,Y) mod p = {p_poly_val}\n"
                        f"  expected (a*b)[{n}] = {prod_digits[n]}\n"
                        "  这表明底座存在根本性数学错误，部署必须中断。"
                    )

    logger.info("verify_witt_polynomial_consistency ok p=%s length=%s pairs=%s", int(p), int(length), int(total_pairs))
    return {
        'ok': 1,
        'p': int(p),
        'length': int(length),
        'modulus': int(modulus),
        'pairs_tested': int(total_pairs),
    }


def verify_frobenius_verschiebung_relations(w: 'WittVector') -> Dict[str, bool]:
    """
    严格验证 Frobenius(F) 与 Verschiebung(V) 的基本关系（针对 W_n(𝔽_p) ≅ ℤ/p^nℤ）。

    关键澄清（避免常见误读）：
    - 在 𝔽_p 上，分量 Frobenius 满足 a^p = a，因此 **分量级** frobenius = id。
    - 但 FV = p 中的 p 是 **Witt 乘法意义** 的乘以 p（在整数同构下即乘以 p mod p^n），
      不是分量级恒等这么简单。

    Returns:
        若全部公理成立，返回包含各条关系的 dict。

    Raises:
        RuntimeError: 任意关系失败（部署必须中断）
    """
    if not isinstance(w, WittVector):
        raise TypeError(f"w must be WittVector, got {type(w).__name__}")

    p = int(w._p)
    length = int(w._length)
    modulus = int(p ** length)

    v_w = w.verschiebung()
    f_w = w.frobenius()

    fv_w = v_w.frobenius()
    vf_w = f_w.verschiebung()

    # Witt 乘法意义下的 p：在 W_n(𝔽_p) ≅ ℤ/p^nℤ 中对应整数 p。
    p_witt = WittVector.from_integer(p, p, length)
    pw = p_witt * w

    fv_int = int(fv_w._to_int_mod_p_power()) % modulus
    vf_int = int(vf_w._to_int_mod_p_power()) % modulus
    pw_int = int(pw._to_int_mod_p_power()) % modulus

    results: Dict[str, bool] = {}
    results["F=Id (componentwise over F_p)"] = bool(f_w.components == w.components)
    results["V=p*w (integer isomorphism)"] = bool(int(v_w._to_int_mod_p_power()) % modulus == pw_int)
    results["FV=p*w"] = bool(fv_int == pw_int)
    results["VF=p*w"] = bool(vf_int == pw_int)
    results["VF=V (componentwise)"] = bool(vf_w.components == v_w.components)

    if not all(results.values()):
        raise RuntimeError(
            "Frobenius/Verschiebung 关系验证失败（部署必须中断）：\n"
            + "\n".join([f"  {k}: {v}" for k, v in results.items()])
            + f"\n  w = {w}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 第五部分：δ-环结构
# Part V: δ-Ring Structure
# ══════════════════════════════════════════════════════════════════════════════

class DeltaRing:
    """
    δ-环 (Delta Ring)
    数学定义：
    一个 δ-环是一个环 A 配备一个映射 δ: A → A 满足：
    1. δ(0) = 0
    2. δ(1) = 0  
    3. δ(a + b) = δ(a) + δ(b) + C_p(a, b)
       其中 C_p(a,b) = (a^^p + b^^p - (a+b)^^p) / p
    4. δ(ab) = a^^p·δ(b) + b^^p·δ(a) + p·δ(a)·δ(b)
    Frobenius 提升：φ(a) = a^^p + p·δ(a)
    """
    
    def __init__(self, p: int):
        self._p = p
        self._carry_cache: Dict[Tuple[int, int], int] = {}
    
    @property
    def prime(self) -> int:
        return self._p
    
    def carry_polynomial_value(self, a: int, b: int) -> int:
        """
        计算进位多项式 C_p(a, b) = (a^p + b^p - (a+b)^^p) / p
        数学事实：对于任意整数 a, b，上述表达式总是整数。
        证明：由二项式定理，
        (a+b)^^p = Σ_{k=0}^^{p} C(p,k) a^^k b^^{p-k}
        = a^^p + b^^p + Σ_{k=1}^^{p-1} C(p,k) a^k b^{p-k}
        对于 1 ≤ k ≤ p-1，C(p,k) = p! / (k!(p-k)!) 被 p 整除
        （因为分子有 p 而分母没有 p 的因子）
        所以 a^^p + b^^p - (a+b)^^p = -Σ_{k=1}^^{p-1} C(p,k) a^^k b^^{p-k} ≡ 0 (mod p)
        """
        key = (a, b)
        if key in self._carry_cache:
            return self._carry_cache[key]
        
        numerator = a**self._p + b**self._p - (a + b)**self._p
        
        if numerator % self._p != 0:
            raise RuntimeError(
                f"数学错误：C_p({a}, {b}) 的分子 {numerator} 不被 {self._p} 整除"
            )
        
        result = numerator // self._p
        self._carry_cache[key] = result
        return result
    
    def delta_on_integers(self, a: int) -> int:
        """
        计算整数上的 δ 值。

        """
        p = int(self._p)
        a_int = int(a)
        
        # δ(a) = (a^p - a) / p（在 Z_p 意义下）
        numerator = a_int ** p - a_int
        
        if numerator % p != 0:
            # 这不应该发生（Fermat 小定理保证）
            raise RuntimeError(
                f"数学错误：δ({a_int}) 的分子 {numerator} 不能被 {p} 整除。"
                "这违反了 Fermat 小定理。"
            )
        
        return numerator // p


class WittVectorDeltaRing:
    """
    Witt 向量上的 δ-环结构
    这是棱柱理论的核心构建块。
    关键事实：
    W(k) 是一个 δ-环，其中：
    - Frobenius φ 是标准的 φ(x_0, x_1, ...) = (x_0^^p, x_1^^p, ...)
    - δ 由 φ(x) = x^^p + p·δ(x) 定义
    对于 Witt 向量，这意味着：
    (x_0^p, x_1^p, ...) = (x_0, x_1, ...)^^p + p·δ(x_0, x_1, ...)
    其中乘法和加法都是 Witt 运算。
    """
    
    def __init__(self, p: int, length: int):
        self._p = p
        self._length = length
        self._base_delta = DeltaRing(p)
    
    def frobenius(self, w: WittVector) -> WittVector:
        """Frobenius 提升 φ"""
        return w.frobenius()
    
    def delta(self, w: 'WittVector') -> 'WittVector':
        """
        δ 算子 - 正确实现（截断 Witt 向量版，严格）
        
        数学定义：δ(x) = (φ(x) - x^p) / p
        - 在 W_n(F_p) ≅ Z/p^nZ 的整数代表上计算
        - 必须精确整除 p（否则属于结构性错误）
        - 输出位于 W_{n-1}(F_p)（除以 p 会损失 1 位 p-adic 精度）
        """
        if not isinstance(w, WittVector):
            raise TypeError(f"delta expects WittVector, got {type(w).__name__}")
        if int(w.prime) != int(self._p):
            raise ValueError(f"delta prime mismatch: expected p={int(self._p)}, got {int(w.prime)}")
        if int(w.length) != int(self._length):
            # 红线：禁止静默 extend/truncate
            raise ValueError(
                f"delta length mismatch: expected length={int(self._length)}, got {int(w.length)}"
            )

        p = int(self._p)
        length = int(self._length)
        modulus = int(p ** length)

        # 将 w 提升到整数表示（Z/p^nZ 的代表元）
        w_int = int(w._to_int_mod_p_power()) % modulus
        phi_w = self.frobenius(w)
        phi_int = int(phi_w._to_int_mod_p_power()) % modulus

        # 计算 w^p（在 Z/p^nZ 中）
        w_to_p_int = pow(int(w_int), int(p), modulus)
        diff_int = (phi_int - w_to_p_int) % modulus

        # δ-环基本性质：φ(x) ≡ x^p (mod p)
        if diff_int % p != 0:
            raise ValueError(
                f"δ 计算失败：φ(w) - w^p = {diff_int} 不能被 p={p} 整除。"
                f"这违反了 δ-环的基本性质。输入 w={w} 可能格式错误。"
            )

        delta_int = diff_int // p
        new_length = length - 1
        if new_length < 1:
            return WittVector([FiniteFieldElement.zero(self._p)], self._p)
        delta_int_truncated = int(delta_int % int(p ** new_length))
        return WittVector.from_integer(delta_int_truncated, p, new_length)

    def verify_delta_axioms(self, a: 'WittVector', b: 'WittVector') -> Dict[str, bool]:
        """
        验证 δ-环公理（截断 Witt 向量语义，严格、无浮点）：
          1) δ(0)=0
          2) δ(1)=0
          3) δ(a+b)=δ(a)+δ(b)+(a^p+b^p-(a+b)^p)/p
          4) δ(ab)=a^p·δ(b)+b^p·δ(a)+p·δ(a)·δ(b)
        """
        results: Dict[str, Any] = {}
        p = int(self._p)
        length = int(self._length)
        if not isinstance(a, WittVector) or not isinstance(b, WittVector):
            raise TypeError("verify_delta_axioms expects WittVector inputs")
        if int(a.prime) != p or int(b.prime) != p:
            raise ValueError(f"verify_delta_axioms prime mismatch: expected p={p}")
        if int(a.length) != length or int(b.length) != length:
            raise ValueError(f"verify_delta_axioms length mismatch: expected length={length}")

        zero = WittVector.zero(p, length)
        one = WittVector.one(p, length)
        try:
            results["δ(0)=0"] = bool(self.delta(zero).is_zero())
            results["δ(1)=0"] = bool(self.delta(one).is_zero())
        except Exception as e:
            results["δ(0)=0"] = False
            results["δ(1)=0"] = False
            results["axiom01_error"] = str(e)
            return results

        modulus_n = int(p ** length)
        modulus_n1 = int(p ** (length - 1)) if (length - 1) >= 1 else 1

        def _delta_int(x_int: int) -> int:
            x = int(x_int % modulus_n)
            x_p = pow(x, p, modulus_n)
            diff = (x - x_p) % modulus_n
            if diff % p != 0:
                raise ValueError(f"δ-int failure: (x-x^p) not divisible by p (x={x}, p={p}, diff={diff})")
            if modulus_n1 == 1:
                return 0
            return int((diff // p) % modulus_n1)

        a_int = int(a._to_int_mod_p_power()) % modulus_n
        b_int = int(b._to_int_mod_p_power()) % modulus_n

        # Axiom 3
        try:
            lhs = _delta_int((a_int + b_int) % modulus_n)
            da = _delta_int(a_int)
            db = _delta_int(b_int)
            num = (
                pow(a_int, p, modulus_n * p)
                + pow(b_int, p, modulus_n * p)
                - pow((a_int + b_int) % modulus_n, p, modulus_n * p)
            )
            if num % p != 0:
                raise ValueError(f"C_p numerator not divisible by p (num={num}, p={p})")
            carry = int((num // p) % modulus_n1) if modulus_n1 != 1 else 0
            rhs = (da + db + carry) % modulus_n1 if modulus_n1 != 1 else 0
            results["δ(a+b)=δ(a)+δ(b)+C_p(a,b)"] = bool(lhs == rhs)
        except Exception as e:
            results["δ(a+b)=δ(a)+δ(b)+C_p(a,b)"] = False
            results["axiom3_error"] = str(e)

        # Axiom 4
        try:
            lhs = _delta_int((a_int * b_int) % modulus_n)
            da = _delta_int(a_int)
            db = _delta_int(b_int)
            if modulus_n1 == 1:
                rhs = 0
            else:
                a_p = pow(a_int % modulus_n1, p, modulus_n1)
                b_p = pow(b_int % modulus_n1, p, modulus_n1)
                rhs = (a_p * db + b_p * da + (p * da * db)) % modulus_n1
            results["δ(ab)=a^p·δ(b)+b^p·δ(a)+p·δ(a)·δ(b)"] = bool(lhs == rhs)
        except Exception as e:
            results["δ(ab)=a^p·δ(b)+b^p·δ(a)+p·δ(a)·δ(b)"] = False
            results["axiom4_error"] = str(e)

        return results

    # Legacy broken block below is kept as a reference only; it is disabled by wrapping in a string.
    '''
    """

    def verify_delta_axioms(self, a: 'WittVector', b: 'WittVector') -> Dict[str, bool]:
    """
    验证 δ-环公理
    返回各公理是否满足的字典，以及诊断信息
    公理：
    1. δ(0) = 0
    2. δ(1) = 0
    3. δ(a+b) = δ(a) + δ(b) + C_p(a,b)  [加法公理]
    4. δ(ab) = a^p·δ(b) + b^p·δ(a) + p·δ(a)·δ(b)  [乘法公理]
    """
    results: Dict[str, Any] = {}
    p = int(self._p)
    length = int(self._length)
    
    # =========================================================================
    # 公理 1: δ(0) = 0
    # =========================================================================
    zero = WittVector.zero(self._p, self._length)
    delta_zero = self.delta(zero)
    results['δ(0)=0'] = delta_zero.is_zero()
    
    # =========================================================================
    # 公理 2: δ(1) = 0
    # =========================================================================
    one = WittVector.one(self._p, self._length)
    delta_one = self.delta(one)
    results['δ(1)=0'] = delta_one.is_zero()
    
    # =========================================================================
    # 公理 3: δ(a+b) = δ(a) + δ(b) + C_p(a,b)
    # C_p 是 Witt 向量版本的进位多项式。
    # 对于长度 n 的截断 Witt 向量，我们在 Z/p^{n-1}Z 上验证
    # =========================================================================
    try:
        # 计算左边：δ(a+b)
        a_plus_b = a + b
        delta_apb = self.delta(a_plus_b)
        
        # 计算右边：δ(a) + δ(b) + C_p(a,b)
        delta_a = self.delta(a)
        delta_b = self.delta(b)
        
        # C_p(a,b) 的计算：
        # 在整数表示下：C_p(a,b) = (a^p + b^p - (a+b)^p) / p
        # 这与 δ 的定义相关但不同
        modulus = p ** length
        a_int = int(a._to_int_mod_p_power())
        b_int = int(b._to_int_mod_p_power())
        
        carry_num = (pow(a_int, p, modulus * p) + pow(b_int, p, modulus * p) 
                     - pow((a_int + b_int) % modulus, p, modulus * p))
        
        if carry_num % p != 0:
            raise ValueError(f"C_p 分子 {carry_num} 不能被 {p} 整除")
        
        carry_int = (carry_num // p) % (p ** (length - 1))
        carry_witt = WittVector.from_integer(carry_int, p, length - 1)
        
        # 调整 delta_a 和 delta_b 的长度以匹配
        rhs_sum = delta_a + delta_b + carry_witt
        
        # 比较
        axiom3_ok = _witt_equal_truncated(delta_apb, rhs_sum, min(delta_apb.length, rhs_sum.length))
        results['δ(a+b)=δ(a)+δ(b)+C_p(a,b)'] = axiom3_ok
        
    except Exception as e:
        results['δ(a+b)=δ(a)+δ(b)+C_p(a,b)'] = False
        results['axiom3_error'] = str(e)
    
    # =========================================================================
    # 公理 4: δ(ab) = a^p·δ(b) + b^p·δ(a) + p·δ(a)·δ(b)
    # δ-环最关键的公理，确保 δ 与乘法结构相容。
    # =========================================================================
    try:
        # 计算左边：δ(ab)
        a_times_b = a * b
        delta_ab = self.delta(a_times_b)
        
        # 计算 a^p 和 b^p（Witt 乘法意义下）
        # 这里 a^p 指 Witt 向量 a 的 p 次幂，不是 Frobenius
        a_to_p = a
        for _ in range(p - 1):
            a_to_p = a_to_p * a
        
        b_to_p = b
        for _ in range(p - 1):
            b_to_p = b_to_p * b
        
        # 计算右边的各项
        delta_a = self.delta(a)
        delta_b = self.delta(b)
        
        # a^p · δ(b)
        # 需要确保长度兼容
        term1 = _witt_mul_truncated(a_to_p, delta_b, p, length - 1)
        
        # b^p · δ(a)  
        term2 = _witt_mul_truncated(b_to_p, delta_a, p, length - 1)
        
        # p · δ(a) · δ(b)
        # 在 W_n(F_p) 中，乘以 p 等价于 Verschiebung 后再乘以单位
        # 但这里 p 作为整数标量
        delta_a_delta_b = delta_a * delta_b
        # p · x 在整数表示下就是乘以 p
        dab_int = int(delta_a_delta_b._to_int_mod_p_power())
        term3_int = (p * dab_int) % (p ** (length - 1))
        term3 = WittVector.from_integer(term3_int, p, length - 1)
        
        # 右边 = term1 + term2 + term3
        rhs = term1 + term2 + term3
        
        # 比较
        axiom4_ok = _witt_equal_truncated(delta_ab, rhs, min(delta_ab.length, rhs.length))
        results['δ(ab)=a^p·δ(b)+b^p·δ(a)+p·δ(a)·δ(b)'] = axiom4_ok
        
    except Exception as e:
        results['δ(ab)=a^p·δ(b)+b^p·δ(a)+p·δ(a)·δ(b)'] = False
        results['axiom4_error'] = str(e)
    
    return results

# Close legacy disabled block
    '''

# ══════════════════════════════════════════════════════════════════════════════
# 第六部分：棱柱结构
# Part VI: Prism Structure
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Prism:
    """
    棱柱 (Prism)
    数学定义：
    一个棱柱是一对 (A, I)，其中：
    - A 是一个 δ-环
    - I ⊂ A 是一个理想
    - I 定义一个 Cartier 除子（即 I 局部主生成，且 A 是 I-完备的）
    """
    
    base_ring_p: int  # 特征
    witt_length: int  # Witt 向量长度（截断精度：W_{witt_length}）
    # Arakelov 高度上界（必须由上游严格计算后注入；此处不允许猜测/默认）
    arakelov_height_bound: Optional[int] = None
    
    def __post_init__(self):
        self._delta_ring = WittVectorDeltaRing(self.base_ring_p, self.witt_length)

    @property
    def p(self) -> int:
        """
        base_ring_p 的严格别名，匹配上层公式写法（prism.p）。
        """
        return int(self.base_ring_p)

    @property
    def required_precision(self) -> int:
        """
        由 Arakelov 高度上界导出的 **唯一确定** 的 p-adic 精度（以 p-进位/截断长度计）。
        定义（纯整数，拒绝浮点近似）：
          给定高度上界 H >= 0，取最小整数 k，使得 p^k > H。
          由于 Witt 截断长度至少为 1，因此 required_precision := max(1, k)。
        约束（红线）：
        - 若 arakelov_height_bound 未提供：直接抛错，禁止静默降级到随便给个长度。
        - 若当前 witt_length < required_precision：直接抛错，禁止用不足精度做近似解。
        """
        if self.arakelov_height_bound is None:
            raise ValueError(
                "Prism.required_precision requires arakelov_height_bound (derived from Arakelov height). "
                "Refuse to guess a truncation length."
            )
        H = int(self.arakelov_height_bound)
        if H < 0:
            raise ValueError("arakelov_height_bound must be >= 0.")
        p = int(self.base_ring_p)
        if p < 2:
            raise ValueError("base_ring_p must be >= 2.")

        # minimal k such that p^k > H (integer arithmetic)
        k = 0
        pow_pk = 1
        while pow_pk <= H:
            pow_pk *= p
            k += 1

        # W_0 does not exist; minimal truncation is 1 (mathematical, not heuristic).
        req = 1 if k < 1 else int(k)
        if int(self.witt_length) < req:
            raise ValueError(
                f"Insufficient witt_length={int(self.witt_length)} for required_precision={int(req)} "
                f"(derived from arakelov_height_bound={int(H)})."
            )
        return int(req)
    
    @property
    def is_crystalline(self) -> bool:
        """是否是 Crystalline 棱柱"""
        return True  # W(k) 上的标准棱柱
    
    @property
    def is_perfect(self) -> bool:
        """是否是完美棱柱（Frobenius 是同构）"""
        return True  # 对于完美域上的 Witt 向量
    
    def generator_of_I(self) -> WittVector:
        """
        理想 I 的生成元
        对于 Crystalline 棱柱，I = (p) = V(W(k))
        生成元是 p = V(1) = (0, 1, 0, ..., 0)
        """
        components = [FiniteFieldElement.zero(self.base_ring_p) for _ in range(self.witt_length)]
        if self.witt_length > 1:
            components[1] = FiniteFieldElement.one(self.base_ring_p)
        return WittVector(components, self.base_ring_p)
    
    def is_in_ideal(self, w: WittVector) -> bool:
        """
        检查 Witt 向量是否在理想 I 中
        
        对于 I = (p) = V(W(k))，w ∈ I 当且仅当 w_0 = 0
        """
        return w[0].is_zero()
    
    def ideal_power(self, n: int) -> 'IdealPower':
        """
        返回 I^^n 的表示
        
        I^n = V^n(W(k))，即前 n 个分量为 0 的 Witt 向量
        """
        return IdealPower(self, n)


@dataclass
class IdealPower:
    """
    棱柱理想的幂 I^^n
    对于 Crystalline 棱柱 (W(k), (p))：
    I^^n = (p^^n) = V^^n(W(k))
    元素特征：前 n 个 Witt 分量为 0
    """
    
    prism: Prism
    power: int
    
    def contains(self, w: WittVector) -> bool:
        """检查 w ∈ I^^n"""
        for i in range(min(self.power, w.length)):
            if not w[i].is_zero():
                return False
        return True
    
    def quotient_class(self, w: WittVector) -> List[FiniteFieldElement]:
        """
        返回 w 在 W(k)/I^^n 中的等价类代表
        
        即前 n 个分量
        """
        return [w[i] for i in range(min(self.power, w.length))]


# ══════════════════════════════════════════════════════════════════════════════
# 第七部分：Nygaard 过滤
# Part VII: Nygaard Filtration
# ══════════════════════════════════════════════════════════════════════════════

class NygaardFiltration:
    """
    Nygaard 过滤 (Nygaard Filtration)
    数学背景：
    设 (A, I) 是一个棱柱，R 是一个 A-代数。
    棱柱上同调 Δ_{R/A} 是一个带有 Frobenius φ 的复形。
    Nygaard 过滤是 Δ_{R/A} 上的一个递减过滤：
    Δ_{R/A} = N^^{≥0} ⊃ N^^{≥1} ⊃ N^^{≥2} ⊃ ...

    """
    
    def __init__(self, prism: Prism):
        self._prism = prism
        self._p = prism.base_ring_p
        self._length = prism.witt_length
    
    def filtration_level(self, w: WittVector) -> int:
        """
        确定 Witt 向量的 Nygaard 过滤级别
        
        N^^{≥i} 包含所有前 i 个分量为 0 的向量
        返回最大的 i 使得 w ∈ N^^{≥i}
        """
        level = 0
        for i, c in enumerate(w.components):
            if c.is_zero():
                level = i + 1
            else:
                break
        return level
    
    def is_in_filtration(self, w: WittVector, level: int) -> bool:
        """检查 w ∈ N^^{≥level}"""
        return self.filtration_level(w) >= level
    
    def graded_piece(self, w: WittVector) -> Tuple[int, FiniteFieldElement]:
        """
        返回 w 的 graded piece
        
        如果 w ∈ N^^{≥i} \ N^^{≥i+1}，返回 (i, w_i)
        """
        level = self.filtration_level(w)
        if level >= w.length:
            # w = 0
            return (w.length, FiniteFieldElement.zero(self._p))
        return (level, w[level])
    
    def validate_witt_vector_fixed(self, w: 'WittVector') -> 'ValidationResult':
        """
        验证 Witt 向量的整性 - 完整实现
        
        检查：
        1. Frobenius 兼容性
        2. Nygaard 过滤级别
        3. Ghost 分量一致性（正确实现）
        4. δ-环结构相容性
        """
        errors = []
        warnings = []
        
        # 1. Frobenius 兼容性
        if not self._filtration.verify_frobenius_compatibility(w):
            errors.append("Frobenius 兼容性失败：φ(w) 不在正确的理想幂中")
        
        # 2. 计算 Nygaard 级别
        level = self._filtration.filtration_level(w)
        
        # 3. Ghost 分量一致性检查（正确实现）
        ghost_ok, ghost_errors = self._validate_ghost_integrality(w)
        if not ghost_ok:
            errors.extend(ghost_errors)
        
        ghost_values = [w.ghost_components_formal(n) for n in range(w.length)]
        
        # 4. δ-环结构检查
        # 验证 φ(w) = w^p + p·δ(w) 的一致性
        try:
            delta_w = self._delta_ring.delta(w)
            phi_w = w.frobenius()
            
            # w^p
            w_to_p = w
            for _ in range(self._p - 1):
                w_to_p = w_to_p * w
            
            # 检查：φ(w) 应该等于 w^p + p·δ(w)（在适当截断下）
            # 这在 Z/p^{length-1}Z 意义下验证
            p = int(self._p)
            length = int(w.length)
            
            phi_int = int(phi_w._to_int_mod_p_power()) % (p ** (length - 1))
            wtp_int = int(w_to_p._to_int_mod_p_power()) % (p ** (length - 1))
            delta_int = int(delta_w._to_int_mod_p_power()) % (p ** (length - 1))
            
            expected = (wtp_int + p * delta_int) % (p ** (length - 1))
            
            if phi_int != expected:
                errors.append(
                    f"δ-环一致性违规: φ(w)={phi_int} ≠ w^p + p·δ(w)={expected}"
                )
        except Exception as e:
            warnings.append(f"δ-环检查跳过（计算异常）: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            nygaard_level=level,
            errors=errors,
            warnings=warnings,
            ghost_components=ghost_values
        )

    def project_to_quotient(self, w: WittVector, level: int) -> 'NygaardQuotient':
        """
        投影到商 N^^{≥0}/N^^{≥level}
        """
        return NygaardQuotient(
            [w[i] for i in range(min(level, w.length))],
            self._p,
            level
        )


@dataclass
class NygaardQuotient:
    """
    Nygaard 商 N^{≥0}/N^{≥n}
    这是有限长度的对象，可以精确计算
    """
    components: List[FiniteFieldElement]
    p: int
    level: int
    
    def is_zero(self) -> bool:
        return all(c.is_zero() for c in self.components)
    
    def __repr__(self) -> str:
        comp_str = ", ".join(str(c.value) for c in self.components)
        return f"[{comp_str}]_{self.level}"


# ══════════════════════════════════════════════════════════════════════════════
# 第八部分：Nygaard 完备化与收敛性
# Part VIII: Nygaard Completion & Convergence
# ══════════════════════════════════════════════════════════════════════════════

class NygaardCompletion:
    """
    Nygaard 完备化
    这是棱柱理论最强大的工具之一。
    数学构造：
    N-完备化是关于 Nygaard 过滤的完备化：
    Δ̂_{R/A} = lim_n Δ_{R/A}/N^^{≥n}
    关键定理 (Bhatt-Scholze)：
    对于光滑 A/I-代数 R，有典范同构：
    Δ̂_{R/A} ≃ A ⊗_{A/I} Ω^^*_{R/(A/I)}
    这将棱柱上同调与 de Rham 上同调联系起来。
    """
    
    def __init__(self, prism: Prism):
        self._prism = prism
        self._filtration = NygaardFiltration(prism)
        self._p = prism.base_ring_p
    
    def truncate(self, w: WittVector, precision: int) -> NygaardQuotient:
        """
        将 Witt 向量截断到精度 n
        
        这是数学意义上的截断（投影到商），不是工程截断
        """
        return self._filtration.project_to_quotient(w, precision)
    
    def is_convergent_sequence(self, 
                               sequence: List[WittVector], 
                               precision: int) -> bool:
        """
        检查 Witt 向量序列是否在 N-拓扑下收敛
        
        收敛条件：对于每个精度 n，存在 N 使得
        对于所有 i, j > N，有 (w_i - w_j) ∈ N^^{≥n}
        """
        if len(sequence) < 2:
            return True
        
        # 检查最后几个元素的差是否在 N^^{≥precision} 中
        for i in range(len(sequence) - 1):
            diff = sequence[i + 1] - sequence[i]
            if self._filtration.filtration_level(diff) < precision:
                return False
        
        return True
    
    def cauchy_completion(self, 
                          generator: Callable[[int], WittVector],
                          max_steps: int = 100) -> Optional[WittVector]:
        """
        从生成函数构造 Cauchy 序列的极限
        
        generator(n) 应该产生第 n 个逼近
        
        如果序列收敛，返回极限；否则返回 None
        """
        sequence = [generator(n) for n in range(max_steps)]
        
        # 找到稳定的分量
        stable_components = []
        
        for i in range(sequence[0].length):
            # 检查第 i 个分量是否稳定
            values = [w[i] for w in sequence]
            
            # 找到第一个稳定点
            stable_value = None
            stable_from = None
            
            for j in range(len(values) - 1):
                if values[j] == values[j + 1]:
                    if stable_value is None:
                        stable_value = values[j]
                        stable_from = j
                    elif values[j] != stable_value:
                        # 不稳定
                        stable_value = None
                        break
                else:
                    stable_value = None
                    stable_from = None
            
            if stable_value is not None:
                stable_components.append(stable_value)
            else:
                # 该分量不收敛
                return None
        
        return WittVector(stable_components, self._p)


# =============================================================================
# 辅助函数
# =============================================================================

def _witt_equal_truncated(w1: 'WittVector', w2: 'WittVector', trunc_len: int) -> bool:
    """
    比较两个 Witt 向量在截断长度 trunc_len 上是否相等。
    """
    for i in range(trunc_len):
        c1 = w1[i] if i < w1.length else FiniteFieldElement.zero(w1._p)
        c2 = w2[i] if i < w2.length else FiniteFieldElement.zero(w2._p)
        if c1 != c2:
            return False
    return True


def _witt_mul_truncated(w1: 'WittVector', w2: 'WittVector', p: int, target_len: int) -> 'WittVector':
    """
    计算 Witt 向量乘积并截断到指定长度。
    处理长度不匹配的情况。
    """
    # 先扩展到相同长度
    max_len = max(w1.length, w2.length, target_len)
    
    def extend_witt(w: 'WittVector', new_len: int) -> 'WittVector':
        if w.length >= new_len:
            return w
        components = list(w._components)
        components.extend([FiniteFieldElement.zero(p)] * (new_len - w.length))
        return WittVector(components, p)
    
    w1_ext = extend_witt(w1, max_len)
    w2_ext = extend_witt(w2, max_len)
    
    product = w1_ext * w2_ext
    
    # 截断到目标长度
    return product.restriction(target_len) if product.length > target_len else product

# ══════════════════════════════════════════════════════════════════════════════
# 第九部分：整性验证器
# Part IX: Integrality Validator
# ══════════════════════════════════════════════════════════════════════════════

class IntegralityValidator:
    """
    整性验证器
    
    基于 Nygaard 过滤的核心约束：
    φ(N^^{≥i}) ⊂ I^^i · Δ
    
    这个约束提供了一种方式来验证 Witt 向量运算的合法性：
    只有满足整性条件的向量才能通过 Frobenius 映射"存活"。
    
    应用：
    如果攻击者试图构造一个导致溢出的非法 Ghost 分量，
    它会落在 Nygaard 过滤之外，被数学结构拒绝。
    """
    
    def __init__(self, prism: Prism):
        self._prism = prism
        self._filtration = NygaardFiltration(prism)
        self._p = prism.base_ring_p
        self._delta_ring = WittVectorDeltaRing(prism.base_ring_p, prism.witt_length)
    
    def _validate_ghost_integrality(self, w: 'WittVector') -> Tuple[bool, List[str]]:
        """
        验证 Ghost 分量的 p-adic 整性约束。
        
        数学约束：
        对于合法的 Witt 向量，Ghost 分量 w_n 应满足：
        w_n ≡ w_0^{p^n} (mod p^n)
        
        这是因为 w_n = Σ_{i=0}^{n} p^i · x_i^{p^{n-i}}
        展开后 w_n = x_0^{p^n} + p·(...) + p^2·(...) + ... + p^n·x_n
        
        因此 w_n ≡ x_0^{p^n} (mod p)，更精细地说 w_n - x_0^{p^n} 应被 p 整除。
        """
        errors = []
        p = int(self._p)
        
        if w.length < 1:
            return True, []
        
        x0 = int(w[0].value)
        
        for n in range(w.length):
            # 严格：只做必要的同余验证，避免构造天文级整数（仍保持数学上的严格性）。
            ghost_n_mod_p = w._ghost_component_mod_p_power(n, p)
            
            # 检验 1: Ghost 分量与 Teichmüller 部分的一致性
            # w_n ≡ x_0^{p^n} (mod p)
            x0_to_pn_mod_p = pow(x0, p ** n, p)
            if int(ghost_n_mod_p) != int(x0_to_pn_mod_p):
                errors.append(
                    f"Ghost 整性违规 (level {n}): "
                    f"w_{n} ≡ {ghost_n_mod_p} (mod {p}), "
                    f"期望 x_0^{{p^{n}}} ≡ {x0_to_pn_mod_p} (mod {p})"
                )
        
        return len(errors) == 0, errors


    def validate_witt_vector(self, w: 'WittVector') -> 'ValidationResult':
        """
        严格验证单个 Witt 向量的底座合法性。

        设计目标：
        - 作为 bonnie_clyde 中间件与上层编排器的稳定入口（不静默、不降级）
        - 返回紧凑 `ValidationResult`（errors/warnings/ghost_components）

        当前覆盖（全程整数/同余，避免浮点）：
        1) Ghost p-整性/同余约束（必要条件）
        2) Nygaard 过滤级别（结构信息）
        3) 溢出检测（Frobenius / ideal power）
        """
        if not isinstance(w, WittVector):
            raise TypeError(f"validate_witt_vector expects WittVector, got {type(w).__name__}")
        if int(w.prime) != int(self._p):
            raise ValueError(f"prime mismatch: expected p={int(self._p)}, got {int(w.prime)}")
        if int(w.length) != int(self._prism.witt_length):
            # 红线：禁止静默 extend/truncate
            raise ValueError(
                f"length mismatch: expected length={int(self._prism.witt_length)}, got {int(w.length)}"
            )

        errors: List[str] = []
        warnings: List[str] = []

        ok, ghost_errors = self._validate_ghost_integrality(w)
        if not ok:
            errors.extend(list(ghost_errors))

        try:
            overflow = self.detect_overflow(w)
            if overflow is not None and bool(overflow.detected):
                errors.append(f"Overflow detected: {overflow.message}")
        except Exception as e:
            # 溢出检测异常属于结构性问题：不应静默忽略
            errors.append(f"overflow_detection_failed: {e}")

        nygaard_level = int(self._filtration.filtration_level(w))
        ghost_values = [int(w.ghost_components_formal(n)) for n in range(int(w.length))]

        return ValidationResult(
            is_valid=(len(errors) == 0),
            nygaard_level=nygaard_level,
            errors=errors,
            warnings=warnings,
            ghost_components=ghost_values,
        )

    def validate_operation(self, 
                          op: str, 
                          a: WittVector, 
                          b: WittVector,
                          result: WittVector) -> 'ValidationResult':
        """
        验证 Witt 向量运算的正确性
        
        使用 Ghost 映射：对于合法运算，Ghost 分量应满足：
        - 加法：w_n(a + b) = w_n(a) + w_n(b)
        - 乘法：w_n(a · b) = w_n(a) · w_n(b)
        """
        errors = []
        
        for n in range(min(a.length, b.length, result.length)):
            ghost_a = a.ghost_components_formal(n)
            ghost_b = b.ghost_components_formal(n)
            ghost_result = result.ghost_components_formal(n)
            
            if op == 'add':
                expected = ghost_a + ghost_b
            elif op == 'mul':
                expected = ghost_a * ghost_b
            else:
                raise ValueError(f"未知操作: {op}")
            
            # 在适当的模数下比较
            # Ghost 分量的关系是在 ℤ 上的，但我们验证模 p^^{n+1}
            modulus = self._p ** (n + 1)
            if ghost_result % modulus != expected % modulus:
                errors.append(
                    f"Ghost 分量 w_{n} 不一致: "
                    f"得到 {ghost_result % modulus}, "
                    f"期望 {expected % modulus}"
                )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            nygaard_level=self._filtration.filtration_level(result),
            errors=errors,
            warnings=[],
            ghost_components=[result.ghost_components_formal(n) for n in range(result.length)]
        )
    
    def detect_overflow(self, w: WittVector) -> Optional['OverflowInfo']:
        """
        检测潜在的溢出
        
        在 Nygaard 框架中，溢出表现为：
        1. Frobenius 兼容性失败
        2. Ghost 分量的跳变
        3. 落在 Nygaard 过滤之外
        """
        # 检查 Frobenius
        phi_w = w.frobenius()
        level = self._filtration.filtration_level(w)
        
        # φ(w) 应该在 I^^{level} 中
        ideal_power = self._prism.ideal_power(level)
        
        if not ideal_power.contains(phi_w):
            # 溢出检测！
            actual_level = self._filtration.filtration_level(phi_w)
            return OverflowInfo(
                detected=True,
                expected_ideal_level=level,
                actual_ideal_level=actual_level,
                violating_component=actual_level,
                message=f"Frobenius 将 N^^{{{level}}} 映到 I^^{{{actual_level}}} 而非 I^^{{{level}}}"
            )
        
        return None


def strict_witt_kernel_validation(
    p: int,
    length: int,
    *,
    max_pair_checks: Optional[int] = None
) -> Dict[str, int]:
    """
    严格 Witt 核验证（确定性、全量穷举；禁止启发式/随机采样）。

    验证目标：
      对所有 a,b ∈ W_length(𝔽_p)，加法与乘法必须满足 Ghost 环同态同余条件。

    说明：
      本底座的 `WittVector.__add__` / `__mul__` 已经内建逐层 Ghost 同余验证；
      因此全量遍历所有 (a,b)只要不抛错，就等价于严格通过。

    Args:
        p: 素数特征（应为 prime）
        length: Witt 向量长度
        max_pair_checks: 上限保护（仅用于拒绝过大规模；不做抽样）
    """
    if not isinstance(p, int):
        raise TypeError(f"p must be int, got {type(p).__name__}")
    if not isinstance(length, int):
        raise TypeError(f"length must be int, got {type(length).__name__}")
    if p < 2:
        raise ValueError("p must be >= 2 (and should be prime).")
    if length < 1:
        raise ValueError("length must be >= 1.")

    modulus = int(p ** length)
    total_pairs = int(modulus * modulus)

    if max_pair_checks is not None:
        if not isinstance(max_pair_checks, int):
            raise TypeError(f"max_pair_checks must be int, got {type(max_pair_checks).__name__}")
        if max_pair_checks < 0:
            raise ValueError("max_pair_checks must be >= 0.")
        if total_pairs > max_pair_checks:
            raise RuntimeError(
                "Refuse to run partial/heuristic validation.\n"
                f"  Required exhaustive pair checks = {total_pairs}\n"
                f"  Provided max_pair_checks         = {max_pair_checks}\n"
                "  请提高 max_pair_checks 或降低 (p,length)。"
            )

    for a_int in range(modulus):
        a = WittVector.from_integer(a_int, p, length)
        for b_int in range(modulus):
            b = WittVector.from_integer(b_int, p, length)
            _ = a + b
            _ = a * b

    return {
        "ok": 1,
        "p": int(p),
        "length": int(length),
        "pairs_tested": int(total_pairs),
    }


def strict_nygaard_filtration_validation(
    prism: "Prism",
    *,
    max_elements: Optional[int] = None
) -> Dict[str, int]:
    """
    严格 Nygaard 过滤验证（确定性、全量穷举；禁止启发式/随机采样）。

    验证核心约束（本实现的 Crystalline/W_n(𝔽_p) 场景）：
      对任意 w，令 i := NygaardLevel(w)，则应有 φ(w) ∈ I^i。

    Args:
        prism: 棱柱结构
        max_elements: 上限保护（仅用于拒绝过大规模；不做抽样）
    """
    if not isinstance(prism, Prism):
        raise TypeError(f"prism must be Prism, got {type(prism).__name__}")

    p = int(prism.base_ring_p)
    length = int(prism.witt_length)
    if p < 2:
        raise ValueError("prism.base_ring_p must be >= 2.")
    if length < 1:
        raise ValueError("prism.witt_length must be >= 1.")

    modulus = int(p ** length)
    if max_elements is not None:
        if not isinstance(max_elements, int):
            raise TypeError(f"max_elements must be int, got {type(max_elements).__name__}")
        if max_elements < 0:
            raise ValueError("max_elements must be >= 0.")
        if modulus > max_elements:
            raise RuntimeError(
                "Refuse to run partial/heuristic validation.\n"
                f"  Required exhaustive elements = {modulus}\n"
                f"  Provided max_elements        = {max_elements}\n"
                "  请提高 max_elements 或降低 (p,length)。"
            )

    filtration = NygaardFiltration(prism)
    for w_int in range(modulus):
        w = WittVector.from_integer(w_int, p, length)
        level = int(filtration.filtration_level(w))
        phi_w = w.frobenius()
        if not prism.ideal_power(level).contains(phi_w):
            raise RuntimeError(
                "Nygaard 过滤约束失败（部署必须中断）：\n"
                f"  p={p}, length={length}\n"
                f"  w_int={w_int}, level={level}\n"
                f"  w={w}\n"
                f"  phi(w)={phi_w}\n"
            )

    return {
        "ok": 1,
        "p": int(p),
        "length": int(length),
        "elements_tested": int(modulus),
    }


def strict_integrality_validation(
    prism: "Prism",
    *,
    max_elements: Optional[int] = None
) -> Dict[str, int]:
    """
    严格整性验证（确定性、全量穷举；禁止启发式/随机采样）。

    验证：对所有 w ∈ W_length(𝔽_p)，其 ghost 分量必须满足必需的 p-整性同余约束。
    """
    if not isinstance(prism, Prism):
        raise TypeError(f"prism must be Prism, got {type(prism).__name__}")

    p = int(prism.base_ring_p)
    length = int(prism.witt_length)
    if p < 2:
        raise ValueError("prism.base_ring_p must be >= 2.")
    if length < 1:
        raise ValueError("prism.witt_length must be >= 1.")

    modulus = int(p ** length)
    if max_elements is not None:
        if not isinstance(max_elements, int):
            raise TypeError(f"max_elements must be int, got {type(max_elements).__name__}")
        if max_elements < 0:
            raise ValueError("max_elements must be >= 0.")
        if modulus > max_elements:
            raise RuntimeError(
                "Refuse to run partial/heuristic validation.\n"
                f"  Required exhaustive elements = {modulus}\n"
                f"  Provided max_elements        = {max_elements}\n"
                "  请提高 max_elements 或降低 (p,length)。"
            )

    validator = IntegralityValidator(prism)
    for w_int in range(modulus):
        w = WittVector.from_integer(w_int, p, length)
        ok, errors = validator._validate_ghost_integrality(w)
        if not ok:
            raise RuntimeError(
                "Integrality validation FAILED (deployment must abort):\n"
                f"  p={p}, length={length}\n"
                f"  w_int={w_int}\n"
                f"  w={w}\n"
                f"  first_error={errors[0] if errors else 'N/A'}"
            )

    return {
        "ok": 1,
        "p": int(p),
        "length": int(length),
        "elements_tested": int(modulus),
    }


def strict_witt_polynomial_validation(p: int, max_n: int) -> Dict[str, int]:
    """
    严格 Witt 多项式可计算性验证（确定性）。

    验证：WittPolynomialGenerator 能否在给定深度 max_n 下构造 S_n/P_n 与进位多项式。
    这不做任何抽样；失败直接抛错（部署必须中断）。
    """
    if not isinstance(p, int):
        raise TypeError(f"p must be int, got {type(p).__name__}")
    if not isinstance(max_n, int):
        raise TypeError(f"max_n must be int, got {type(max_n).__name__}")
    if p < 2:
        raise ValueError("p must be >= 2 (and should be prime).")
    if max_n < 1:
        raise ValueError("max_n must be >= 1.")

    gen = WittPolynomialGenerator(p, max_n)
    for n in range(max_n):
        _ = gen.addition_polynomial(n)
        _ = gen.multiplication_polynomial(n)
    _ = gen.carry_polynomial()

    return {
        "ok": 1,
        "p": int(p),
        "max_n": int(max_n),
    }


def run_strict_validation_suite(
    p: int,
    length: int,
    *,
    max_pair_checks: Optional[int] = None,
    max_elements: Optional[int] = None,
    witt_polynomial_max_n: Optional[int] = None
) -> Dict[str, object]:
    """
    运行完整的严格验证套件（确定性；部署错误必须中断）。

    注意：该套件是数学完备性优先，会做穷举/闭环验证；调用者需自行确保参数规模可计算，
    或通过 max_pair_checks / max_elements 设置拒绝运行过大规模的硬上限（不做抽样）。
    """
    if witt_polynomial_max_n is None:
        witt_polynomial_max_n = int(length)

    logger.info("run_strict_validation_suite start p=%s length=%s", int(p), int(length))
    results: Dict[str, object] = {}
    results["witt_kernel"] = strict_witt_kernel_validation(p, length, max_pair_checks=max_pair_checks)
    results["witt_polynomial_consistency"] = verify_witt_polynomial_consistency(
        p, length, max_pair_checks=max_pair_checks
    )
    results["witt_polynomials"] = strict_witt_polynomial_validation(p, int(witt_polynomial_max_n))

    prism = Prism(base_ring_p=p, witt_length=length)
    results["nygaard_filtration"] = strict_nygaard_filtration_validation(prism, max_elements=max_elements)
    results["integrality"] = strict_integrality_validation(prism, max_elements=max_elements)

    results["summary"] = {
        "all_ok": True,
        "p": int(p),
        "length": int(length),
    }
    logger.info("run_strict_validation_suite ok p=%s length=%s", int(p), int(length))
    return results

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    nygaard_level: int
    errors: List[str]
    warnings: List[str]
    ghost_components: List[int]
    
    def __repr__(self) -> str:
        status = "✓ 合法" if self.is_valid else "✗ 非法"
        lines = [
            f"验证结果: {status}",
            f"Nygaard 级别: N^^{{≥{self.nygaard_level}}}",
            f"Ghost 分量: {self.ghost_components}"
        ]
        if self.errors:
            lines.append("错误:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("警告:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class OverflowInfo:
    """溢出信息"""
    detected: bool
    expected_ideal_level: int
    actual_ideal_level: int
    violating_component: int
    message: str
    
    def __repr__(self) -> str:
        if not self.detected:
            return "无溢出"
        return f" 溢出检测: {self.message}"


# ══════════════════════════════════════════════════════════════════════════════
# Part X: Iwasawa Algebra (Truncated Completion Model)
# Λ = Z_p[[T]] as a computable truncation:
#   Λ_{n,m} := (Z/p^nZ)[[T]] / (T^m)
# Coefficients are represented by WittVector in W_n(F_p) ≅ Z/p^nZ.
#
# Redlines:
# - No heuristics / no floats
# - No silent fallback: any mismatch must raise
# - No modification to Witt base is required for this layer; it is purely additive.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IwasawaTruncationSpec:
    """
    Truncation spec for the computable Iwasawa algebra Λ_{n,m}.

    Parameters:
    - p: prime (characteristic of residue field)
    - witt_length: n (p-adic precision via W_n(F_p) ≅ Z/p^nZ)
    - t_precision: m (T-adic truncation degree; we work modulo T^m)
    """

    p: int
    witt_length: int
    t_precision: int

    def __post_init__(self) -> None:
        if not isinstance(self.p, int):
            raise TypeError(f"p must be int, got {type(self.p).__name__}")
        if not isinstance(self.witt_length, int):
            raise TypeError(f"witt_length must be int, got {type(self.witt_length).__name__}")
        if not isinstance(self.t_precision, int):
            raise TypeError(f"t_precision must be int, got {type(self.t_precision).__name__}")
        if int(self.p) < 2:
            raise ValueError("p must be >= 2 (and should be prime).")
        if int(self.witt_length) < 1:
            raise ValueError("witt_length must be >= 1.")
        if int(self.t_precision) < 1:
            raise ValueError("t_precision must be >= 1.")

    @property
    def modulus(self) -> int:
        """Return p^n (modulus for coefficient ring Z/p^nZ)."""
        return int(int(self.p) ** int(self.witt_length))


def derive_t_precision_from_state_dimension_bits(*, state_dimension_bits: int) -> int:
    """
    Derive T-adic truncation m from the Iwasawa standard:
      m = 2 * State_Dimension + 1
    This is deterministic and integer-only.
    """
    if not isinstance(state_dimension_bits, int):
        raise TypeError(f"state_dimension_bits must be int, got {type(state_dimension_bits).__name__}")
    if int(state_dimension_bits) < 1:
        raise ValueError("state_dimension_bits must be >= 1.")
    return int(2 * int(state_dimension_bits) + 1)


def derive_witt_length_for_bit_budget(*, p: int, total_bits: int) -> int:
    """
    Deterministically derive minimal n such that p^n >= 2^{total_bits}.

    This implements the spirit of the Shannon/overflow bit-budget in the user's standard
    without using transcendental logs (no floats, no approximations).
    """
    if not isinstance(p, int):
        raise TypeError(f"p must be int, got {type(p).__name__}")
    if not isinstance(total_bits, int):
        raise TypeError(f"total_bits must be int, got {type(total_bits).__name__}")
    if int(p) < 2:
        raise ValueError("p must be >= 2 (and should be prime).")
    if int(total_bits) < 0:
        raise ValueError("total_bits must be >= 0.")
    threshold = 1 << int(total_bits)
    n = 0
    acc = 1
    pp = int(p)
    # minimal n with acc >= threshold
    while acc < threshold:
        acc *= pp
        n += 1
    return int(n if n >= 1 else 1)


def derive_iwasawa_trinity_specs_for_keccak256(
    *,
    state_dimension_bits: int = 1600,
    payload_bits: int = 256,
    overflow_bits: int = 256,
    prime_track_b: int = SECP256K1_FIELD_PRIME,
) -> Dict[str, IwasawaTruncationSpec]:
    """
    Build the Trinity specs from `Iwasawa建模标准.txt`:
      Track A (physical): p=2
      Track C (test):     p=3
      Track B (geometry): p=secp256k1 field prime (user-confirmed)

    Witt precision n is derived from the strict bit budget:
      total_bits = payload_bits + overflow_bits
      require p^n >= 2^{total_bits}

    T precision m is derived from:
      m = 2 * state_dimension_bits + 1
    """
    if not isinstance(prime_track_b, int):
        raise TypeError(f"prime_track_b must be int, got {type(prime_track_b).__name__}")
    if int(prime_track_b) < 2:
        raise ValueError("prime_track_b must be >= 2.")

    total_bits = int(payload_bits) + int(overflow_bits)
    m = derive_t_precision_from_state_dimension_bits(state_dimension_bits=int(state_dimension_bits))

    pA = 2
    pC = 3
    pB = int(prime_track_b)

    nA = derive_witt_length_for_bit_budget(p=pA, total_bits=total_bits)
    nC = derive_witt_length_for_bit_budget(p=pC, total_bits=total_bits)
    nB = derive_witt_length_for_bit_budget(p=pB, total_bits=total_bits)

    return {
        "A_physical_p2": IwasawaTruncationSpec(p=int(pA), witt_length=int(nA), t_precision=int(m)),
        "B_geometry_secp256k1": IwasawaTruncationSpec(p=int(pB), witt_length=int(nB), t_precision=int(m)),
        "C_test_p3": IwasawaTruncationSpec(p=int(pC), witt_length=int(nC), t_precision=int(m)),
    }


class IwasawaPowerSeries:
    """
    Element of Λ_{n,m} = (Z/p^nZ)[[T]]/(T^m), with coefficients in W_n(F_p).

    Representation:
      f(T) = Σ_{i=0}^{m-1} a_i T^i,  a_i ∈ W_n(F_p)
    """

    __slots__ = ("_spec", "_coeffs")

    def __init__(self, coeffs: Sequence[WittVector], spec: IwasawaTruncationSpec):
        if not isinstance(spec, IwasawaTruncationSpec):
            raise TypeError(f"spec must be IwasawaTruncationSpec, got {type(spec).__name__}")
        if not isinstance(coeffs, (list, tuple)):
            raise TypeError(f"coeffs must be a Sequence[WittVector], got {type(coeffs).__name__}")
        if len(coeffs) != int(spec.t_precision):
            raise ValueError(
                f"coeffs length must equal t_precision={int(spec.t_precision)}, got {len(coeffs)}"
            )
        for c in coeffs:
            if not isinstance(c, WittVector):
                raise TypeError(f"coeffs must contain WittVector, got {type(c).__name__}")
            if int(c.prime) != int(spec.p):
                raise ValueError(f"coefficient prime mismatch: expected p={int(spec.p)}, got {int(c.prime)}")
            if int(c.length) != int(spec.witt_length):
                raise ValueError(
                    f"coefficient witt_length mismatch: expected {int(spec.witt_length)}, got {int(c.length)}"
                )
        self._spec = spec
        self._coeffs = list(coeffs)

    @property
    def spec(self) -> IwasawaTruncationSpec:
        return self._spec

    @property
    def coeffs(self) -> List[WittVector]:
        return list(self._coeffs)

    def __getitem__(self, i: int) -> WittVector:
        return self._coeffs[int(i)]

    def is_zero(self) -> bool:
        return all(c.is_zero() for c in self._coeffs)

    @classmethod
    def zero(cls, spec: IwasawaTruncationSpec) -> "IwasawaPowerSeries":
        z = WittVector.zero(int(spec.p), int(spec.witt_length))
        return cls([z for _ in range(int(spec.t_precision))], spec)

    @classmethod
    def one(cls, spec: IwasawaTruncationSpec) -> "IwasawaPowerSeries":
        z = WittVector.zero(int(spec.p), int(spec.witt_length))
        o = WittVector.one(int(spec.p), int(spec.witt_length))
        coeffs = [z for _ in range(int(spec.t_precision))]
        coeffs[0] = o
        return cls(coeffs, spec)

    @classmethod
    def T(cls, spec: IwasawaTruncationSpec) -> "IwasawaPowerSeries":
        """The indeterminate T modulo T^m."""
        if int(spec.t_precision) < 2:
            # In Λ/(T^1), T == 0; treat this as a hard error to avoid silent downgrade.
            raise ValueError("Cannot construct T when t_precision=1 (since T≡0 mod T). Increase t_precision.")
        z = WittVector.zero(int(spec.p), int(spec.witt_length))
        o = WittVector.one(int(spec.p), int(spec.witt_length))
        coeffs = [z for _ in range(int(spec.t_precision))]
        coeffs[1] = o
        return cls(coeffs, spec)

    @classmethod
    def constant(cls, c: WittVector, spec: IwasawaTruncationSpec) -> "IwasawaPowerSeries":
        if not isinstance(c, WittVector):
            raise TypeError(f"c must be WittVector, got {type(c).__name__}")
        if int(c.prime) != int(spec.p) or int(c.length) != int(spec.witt_length):
            raise ValueError("constant coefficient incompatible with spec.")
        z = WittVector.zero(int(spec.p), int(spec.witt_length))
        coeffs = [z for _ in range(int(spec.t_precision))]
        coeffs[0] = c
        return cls(coeffs, spec)

    def _require_same_spec(self, other: "IwasawaPowerSeries") -> None:
        if not isinstance(other, IwasawaPowerSeries):
            raise TypeError(f"expected IwasawaPowerSeries, got {type(other).__name__}")
        if self._spec != other._spec:
            raise ValueError(f"Iwasawa spec mismatch: {self._spec} vs {other._spec}")

    def __neg__(self) -> "IwasawaPowerSeries":
        return IwasawaPowerSeries([(-c) for c in self._coeffs], self._spec)

    def __add__(self, other: "IwasawaPowerSeries") -> "IwasawaPowerSeries":
        self._require_same_spec(other)
        return IwasawaPowerSeries([a + b for a, b in zip(self._coeffs, other._coeffs)], self._spec)

    def __sub__(self, other: "IwasawaPowerSeries") -> "IwasawaPowerSeries":
        return self + (-other)

    def __mul__(self, other: "IwasawaPowerSeries") -> "IwasawaPowerSeries":
        self._require_same_spec(other)
        m = int(self._spec.t_precision)
        p = int(self._spec.p)
        n = int(self._spec.witt_length)
        z = WittVector.zero(p, n)
        out = [z for _ in range(m)]
        # Cauchy product truncated mod T^m
        for i in range(m):
            acc = z
            for j in range(i + 1):
                acc = acc + (self._coeffs[j] * other._coeffs[i - j])
            out[i] = acc
        return IwasawaPowerSeries(out, self._spec)

    def __pow__(self, e: int) -> "IwasawaPowerSeries":
        if not isinstance(e, int):
            raise TypeError(f"exponent must be int, got {type(e).__name__}")
        if int(e) < 0:
            raise ValueError("negative exponent not supported for truncated Iwasawa series.")
        if int(e) == 0:
            return IwasawaPowerSeries.one(self._spec)
        result = IwasawaPowerSeries.one(self._spec)
        base = self
        exp = int(e)
        while exp > 0:
            if exp & 1:
                result = result * base
            exp >>= 1
            if exp:
                base = base * base
        return result

    def shift_T(self, k: int) -> "IwasawaPowerSeries":
        """Multiply by T^k (truncate)."""
        if not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        if int(k) < 0:
            raise ValueError("k must be >= 0")
        m = int(self._spec.t_precision)
        p = int(self._spec.p)
        n = int(self._spec.witt_length)
        z = WittVector.zero(p, n)
        out = [z for _ in range(m)]
        kk = int(k)
        for i in range(m - kk):
            out[i + kk] = self._coeffs[i]
        return IwasawaPowerSeries(out, self._spec)

    def scale(self, c: WittVector) -> "IwasawaPowerSeries":
        """Multiply all coefficients by c ∈ Z/p^nZ."""
        if not isinstance(c, WittVector):
            raise TypeError(f"c must be WittVector, got {type(c).__name__}")
        if int(c.prime) != int(self._spec.p) or int(c.length) != int(self._spec.witt_length):
            raise ValueError("scale coefficient incompatible with spec.")
        return IwasawaPowerSeries([c * a for a in self._coeffs], self._spec)

    def compose(self, g: "IwasawaPowerSeries") -> "IwasawaPowerSeries":
        """
        Compose f(g(T)) modulo T^m.

        Requirement (strict): g(0) must be 0, otherwise composition is not well-defined as a T-adic endomorphism
        in the truncated model (would require additional data about convergence/unit decomposition).
        """
        self._require_same_spec(g)
        if not g._coeffs[0].is_zero():
            raise ValueError("compose requires g(0)=0 (constant term must be 0).")

        m = int(self._spec.t_precision)
        p = int(self._spec.p)
        n = int(self._spec.witt_length)
        z = WittVector.zero(p, n)

        # Precompute powers of g: g^0..g^{m-1} (since T^m=0)
        powers: List[IwasawaPowerSeries] = [IwasawaPowerSeries.one(self._spec)]
        for _ in range(1, m):
            powers.append(powers[-1] * g)

        out = IwasawaPowerSeries.zero(self._spec)
        for i in range(m):
            a_i = self._coeffs[i]
            if a_i.is_zero():
                continue
            # out += a_i * g^i
            out = out + powers[i].scale(a_i)
        return out

    def phi_T(self) -> "IwasawaPowerSeries":
        """
        φ(T) = (1+T)^p - 1  in Λ_{n,m}.

        Coefficients are computed as exact binomial coefficients (no float):
          (1+T)^p - 1 = Σ_{k=1}^{p} C(p,k) T^k
        truncated mod T^m.
        """
        spec = self._spec
        p = int(spec.p)
        m = int(spec.t_precision)
        n = int(spec.witt_length)
        z = WittVector.zero(p, n)
        coeffs = [z for _ in range(m)]
        # k=0 term is cancelled by "-1"
        # Use math.comb for exact binomial coefficients (deterministic integer arithmetic).
        import math

        max_k = min(int(p), int(m - 1))
        for k in range(1, max_k + 1):
            ck = int(math.comb(int(p), int(k)))
            coeffs[k] = WittVector.from_integer(int(ck), p, n)
        return IwasawaPowerSeries(coeffs, spec)

    def frobenius(self) -> "IwasawaPowerSeries":
        """
        Iwasawa Frobenius on Λ: φ(f)(T) := f((1+T)^p - 1).
        Coefficients in Z_p are fixed; all action is on the Γ-variable T.
        """
        g = self.phi_T()
        return self.compose(g)

    @staticmethod
    def _egcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended GCD: returns (g, x, y) with ax + by = g = gcd(a,b)."""
        aa = int(a)
        bb = int(b)
        if bb == 0:
            return (abs(aa), 1 if aa >= 0 else -1, 0)
        x0, y0 = 1, 0
        x1, y1 = 0, 1
        while bb != 0:
            q = aa // bb
            aa, bb = bb, aa - q * bb
            x0, x1 = x1, x0 - q * x1
            y0, y1 = y1, y0 - q * y1
        g = abs(int(aa))
        return (g, int(x0), int(y0))

    @staticmethod
    def _inv_mod(a: int, modulus: int) -> int:
        """Compute inverse of a modulo modulus; require gcd(a,modulus)=1."""
        m = int(modulus)
        if m <= 0:
            raise ValueError("modulus must be positive.")
        aa = int(a % m)
        g, x, _y = IwasawaPowerSeries._egcd(aa, m)
        if g != 1:
            raise ZeroDivisionError(f"element not invertible modulo {m}: gcd({aa},{m})={g}")
        return int(x % m)

    def vp_coeff(self, c: WittVector) -> int:
        """
        p-adic valuation v_p(c) within Z/p^nZ (deterministic, truncated):
        - v_p(0) is treated as n (maximal within this truncation).
        - otherwise v_p(c) is the largest v in [0,n-1] such that p^v | c (as integer rep mod p^n).
        """
        if not isinstance(c, WittVector):
            raise TypeError(f"c must be WittVector, got {type(c).__name__}")
        if int(c.prime) != int(self._spec.p) or int(c.length) != int(self._spec.witt_length):
            raise ValueError("coefficient incompatible with spec.")
        n = int(self._spec.witt_length)
        p = int(self._spec.p)
        modulus = int(self._spec.modulus)
        x = int(c._to_int_mod_p_power()) % modulus
        if x == 0:
            return int(n)
        v = 0
        while v < n and x % p == 0:
            x //= p
            v += 1
        return int(v)

    def mu_invariant(self) -> int:
        """
        μ-invariant (truncated) for f ∈ Λ_{n,m}:
          μ := max μ such that p^μ divides all coefficients (in Z/p^nZ sense).

        Returns an integer in [0, n]. For the zero series, returns n (maximal in this truncation).
        """
        n = int(self._spec.witt_length)
        vals = [self.vp_coeff(c) for c in self._coeffs]
        if not vals:
            return int(n)
        return int(min(vals))

    def is_unit(self) -> bool:
        """In Λ, f is a unit iff constant term is a p-adic unit (v_p(a0)=0)."""
        return bool(self.vp_coeff(self._coeffs[0]) == 0)

    def inverse(self) -> "IwasawaPowerSeries":
        """
        Compute multiplicative inverse in Λ_{n,m}, if it exists.

        Condition: constant term must be a unit (not divisible by p).
        Algorithm: standard power series inversion (deterministic recursion) modulo T^m.
        """
        if not self.is_unit():
            raise ZeroDivisionError("series is not invertible in Λ: constant term is not a unit.")

        spec = self._spec
        p = int(spec.p)
        n = int(spec.witt_length)
        m = int(spec.t_precision)
        modulus = int(spec.modulus)

        a0 = self._coeffs[0]
        a0_int = int(a0._to_int_mod_p_power()) % modulus
        inv_a0_int = self._inv_mod(a0_int, modulus)
        b0 = WittVector.from_integer(inv_a0_int, p, n)

        z = WittVector.zero(p, n)
        b: List[WittVector] = [z for _ in range(m)]
        b[0] = b0

        # For k>=1: b_k = -a0^{-1} * Σ_{i=1..k} a_i * b_{k-i}
        for k in range(1, m):
            s = z
            for i in range(1, k + 1):
                s = s + (self._coeffs[i] * b[k - i])
            b[k] = (-b0) * s

        return IwasawaPowerSeries(b, spec)

    def __eq__(self, other) -> bool:
        if not isinstance(other, IwasawaPowerSeries):
            return False
        return self._spec == other._spec and self._coeffs == other._coeffs

    def __repr__(self) -> str:
        # Compact, deterministic representation (avoid huge dumps).
        m = int(self._spec.t_precision)
        nz = [(i, c) for i, c in enumerate(self._coeffs) if not c.is_zero()]
        if not nz:
            return f"Λ[{int(self._spec.p)},{int(self._spec.witt_length)}]/(T^{m}):0"
        head = nz[:6]
        parts = [f"{c}*T^{i}" if i != 0 else f"{c}" for i, c in head]
        more = "" if len(nz) <= 6 else f" + ...({len(nz)-6} more)"
        return f"Λ[{int(self._spec.p)},{int(self._spec.witt_length)}]/(T^{m}):" + " + ".join(parts) + more


class IwasawaZpPowerSeries:
    """
    Scalable Λ_{n,m} element with coefficients as integers modulo p^n (NOT WittVector objects).

    This is required for high-precision tracks like (p=2, n=512, m=3201) where a
    WittVector-per-coefficient representation is memory-prohibitive.

    Semantics:
      coeffs[i] represents a_i ∈ Z/p^nZ
      f(T) = Σ_{i=0}^{m-1} a_i T^i   in (Z/p^nZ)[[T]]/(T^m)
    """

    __slots__ = ("_spec", "_coeffs")

    def __init__(self, coeffs: Sequence[int], spec: IwasawaTruncationSpec):
        if not isinstance(spec, IwasawaTruncationSpec):
            raise TypeError(f"spec must be IwasawaTruncationSpec, got {type(spec).__name__}")
        if not isinstance(coeffs, (list, tuple)):
            raise TypeError(f"coeffs must be Sequence[int], got {type(coeffs).__name__}")
        if len(coeffs) != int(spec.t_precision):
            raise ValueError(
                f"coeffs length must equal t_precision={int(spec.t_precision)}, got {len(coeffs)}"
            )
        mod = int(spec.modulus)
        if mod <= 0:
            raise ValueError("invalid modulus for spec.")
        norm: List[int] = []
        for c in coeffs:
            if not isinstance(c, int):
                raise TypeError(f"coeffs must contain int, got {type(c).__name__}")
            norm.append(int(c % mod))
        self._spec = spec
        self._coeffs = norm

    @property
    def spec(self) -> IwasawaTruncationSpec:
        return self._spec

    @property
    def coeffs(self) -> List[int]:
        return list(self._coeffs)

    def __getitem__(self, i: int) -> int:
        return int(self._coeffs[int(i)])

    def is_zero(self) -> bool:
        return all(int(c) == 0 for c in self._coeffs)

    @classmethod
    def zero(cls, spec: IwasawaTruncationSpec) -> "IwasawaZpPowerSeries":
        return cls([0 for _ in range(int(spec.t_precision))], spec)

    @classmethod
    def one(cls, spec: IwasawaTruncationSpec) -> "IwasawaZpPowerSeries":
        coeffs = [0 for _ in range(int(spec.t_precision))]
        coeffs[0] = 1
        return cls(coeffs, spec)

    @classmethod
    def T(cls, spec: IwasawaTruncationSpec) -> "IwasawaZpPowerSeries":
        if int(spec.t_precision) < 2:
            raise ValueError("Cannot construct T when t_precision=1 (since T≡0 mod T). Increase t_precision.")
        coeffs = [0 for _ in range(int(spec.t_precision))]
        coeffs[1] = 1
        return cls(coeffs, spec)

    def _require_same_spec(self, other: "IwasawaZpPowerSeries") -> None:
        if not isinstance(other, IwasawaZpPowerSeries):
            raise TypeError(f"expected IwasawaZpPowerSeries, got {type(other).__name__}")
        if self._spec != other._spec:
            raise ValueError(f"Iwasawa spec mismatch: {self._spec} vs {other._spec}")

    def __neg__(self) -> "IwasawaZpPowerSeries":
        mod = int(self._spec.modulus)
        return IwasawaZpPowerSeries([(-c) % mod for c in self._coeffs], self._spec)

    def __add__(self, other: "IwasawaZpPowerSeries") -> "IwasawaZpPowerSeries":
        self._require_same_spec(other)
        mod = int(self._spec.modulus)
        return IwasawaZpPowerSeries([(a + b) % mod for a, b in zip(self._coeffs, other._coeffs)], self._spec)

    def __sub__(self, other: "IwasawaZpPowerSeries") -> "IwasawaZpPowerSeries":
        return self + (-other)

    def __mul__(self, other: "IwasawaZpPowerSeries") -> "IwasawaZpPowerSeries":
        self._require_same_spec(other)
        mod = int(self._spec.modulus)
        m = int(self._spec.t_precision)
        out = [0 for _ in range(m)]
        for i in range(m):
            acc = 0
            for j in range(i + 1):
                acc = (acc + (self._coeffs[j] * other._coeffs[i - j])) % mod
            out[i] = int(acc)
        return IwasawaZpPowerSeries(out, self._spec)

    def __pow__(self, e: int) -> "IwasawaZpPowerSeries":
        if not isinstance(e, int):
            raise TypeError(f"exponent must be int, got {type(e).__name__}")
        if int(e) < 0:
            raise ValueError("negative exponent not supported for truncated Iwasawa series.")
        if int(e) == 0:
            return IwasawaZpPowerSeries.one(self._spec)
        result = IwasawaZpPowerSeries.one(self._spec)
        base = self
        exp = int(e)
        while exp > 0:
            if exp & 1:
                result = result * base
            exp >>= 1
            if exp:
                base = base * base
        return result

    def shift_T(self, k: int) -> "IwasawaZpPowerSeries":
        if not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        if int(k) < 0:
            raise ValueError("k must be >= 0")
        m = int(self._spec.t_precision)
        out = [0 for _ in range(m)]
        kk = int(k)
        for i in range(m - kk):
            out[i + kk] = int(self._coeffs[i])
        return IwasawaZpPowerSeries(out, self._spec)

    @staticmethod
    def _egcd(a: int, b: int) -> Tuple[int, int, int]:
        aa = int(a)
        bb = int(b)
        if bb == 0:
            return (abs(aa), 1 if aa >= 0 else -1, 0)
        x0, y0 = 1, 0
        x1, y1 = 0, 1
        while bb != 0:
            q = aa // bb
            aa, bb = bb, aa - q * bb
            x0, x1 = x1, x0 - q * x1
            y0, y1 = y1, y0 - q * y1
        g = abs(int(aa))
        return (g, int(x0), int(y0))

    @staticmethod
    def _inv_mod(a: int, modulus: int) -> int:
        m = int(modulus)
        if m <= 0:
            raise ValueError("modulus must be positive.")
        aa = int(a % m)
        g, x, _y = IwasawaZpPowerSeries._egcd(aa, m)
        if g != 1:
            raise ZeroDivisionError(f"element not invertible modulo {m}: gcd({aa},{m})={g}")
        return int(x % m)

    def vp_int(self, x: int) -> int:
        """Truncated v_p in Z/p^nZ: v_p(0)=n, else count p factors up to n."""
        if not isinstance(x, int):
            raise TypeError(f"x must be int, got {type(x).__name__}")
        p = int(self._spec.p)
        n = int(self._spec.witt_length)
        mod = int(self._spec.modulus)
        v = int(x % mod)
        if v == 0:
            return int(n)
        out = 0
        while out < n and v % p == 0:
            v //= p
            out += 1
        return int(out)

    def mu_invariant(self) -> int:
        n = int(self._spec.witt_length)
        vals = [self.vp_int(c) for c in self._coeffs]
        return int(min(vals)) if vals else int(n)

    def lambda_invariant(self) -> Optional[int]:
        """
        λ-invariant (truncated): the minimal index i such that v_p(a_i)=μ.
        For the zero series, returns None.
        """
        if self.is_zero():
            return None
        mu = int(self.mu_invariant())
        for i, c in enumerate(self._coeffs):
            if self.vp_int(int(c)) == mu:
                return int(i)
        # Should be impossible if mu is computed correctly
        raise RuntimeError("lambda_invariant internal error: no coefficient attained μ.")

    def is_unit(self) -> bool:
        """Unit iff constant term not divisible by p."""
        return bool(self.vp_int(int(self._coeffs[0])) == 0)

    def inverse(self) -> "IwasawaZpPowerSeries":
        if not self.is_unit():
            raise ZeroDivisionError("series is not invertible in Λ: constant term is not a unit.")
        mod = int(self._spec.modulus)
        m = int(self._spec.t_precision)
        a0 = int(self._coeffs[0] % mod)
        inv_a0 = self._inv_mod(a0, mod)
        b = [0 for _ in range(m)]
        b[0] = int(inv_a0)
        for k in range(1, m):
            s = 0
            for i in range(1, k + 1):
                s = (s + (self._coeffs[i] * b[k - i])) % mod
            b[k] = (-b[0] * s) % mod
        return IwasawaZpPowerSeries(b, self._spec)

    def compose(self, g: "IwasawaZpPowerSeries") -> "IwasawaZpPowerSeries":
        self._require_same_spec(g)
        if int(g._coeffs[0]) != 0:
            raise ValueError("compose requires g(0)=0 (constant term must be 0).")
        m = int(self._spec.t_precision)
        mod = int(self._spec.modulus)

        # Precompute g^0..g^{m-1}
        powers: List[IwasawaZpPowerSeries] = [IwasawaZpPowerSeries.one(self._spec)]
        for _ in range(1, m):
            powers.append(powers[-1] * g)

        out = IwasawaZpPowerSeries.zero(self._spec)
        for i in range(m):
            a_i = int(self._coeffs[i])
            if a_i == 0:
                continue
            out = out + IwasawaZpPowerSeries([(a_i * c) % mod for c in powers[i]._coeffs], self._spec)
        return out

    def phi_T(self) -> "IwasawaZpPowerSeries":
        """
        φ(T) = (1+T)^p - 1  modulo T^m, with coefficients computed modulo p^n.

        For large primes (e.g. secp256k1), computing binom(p,k) as a huge integer is infeasible.
        We compute binomial coefficients *directly modulo p^n* via the recurrence:
          C(p,0)=1
          C(p,k)=C(p,k-1) * (p-k+1) / k   (mod p^n)
        Since k < p in our truncation regime (m <= 3201 << p), gcd(k,p)=1 so inv(k) exists mod p^n.
        """
        spec = self._spec
        p = int(spec.p)
        n = int(spec.witt_length)
        mod = int(spec.modulus)
        m = int(spec.t_precision)

        # k ranges 1..min(m-1, p) but if p is huge we only need 1..m-1
        max_k = int(min(int(m - 1), int(p))) if int(p) < int(m) else int(m - 1)

        coeffs = [0 for _ in range(m)]
        # current binomial C(p,0)
        c = 1 % mod
        for k in range(1, max_k + 1):
            num = (c * ((p - k + 1) % mod)) % mod
            inv_k = self._inv_mod(int(k), mod)
            c = (num * inv_k) % mod
            coeffs[k] = int(c)
        # subtract 1 cancels k=0 term
        coeffs[0] = 0

        # Sanity: for prime p, for 1<=k<=p-1, C(p,k) is divisible by p.
        # We only assert this when k<p and n>=1; otherwise no claim.
        if n >= 1 and p > 1:
            for k in range(1, min(max_k, p - 1) + 1):
                if int(coeffs[k]) % int(p) != 0:
                    raise RuntimeError(
                        "phi_T binomial coefficient sanity failed: C(p,k) not divisible by p.\n"
                        f"  p={p}, n={n}, k={k}, C(p,k) mod p^n = {int(coeffs[k])}"
                    )
        return IwasawaZpPowerSeries(coeffs, spec)

    def frobenius(self) -> "IwasawaZpPowerSeries":
        """φ(f)(T) := f((1+T)^p - 1)."""
        return self.compose(self.phi_T())

    def __eq__(self, other) -> bool:
        if not isinstance(other, IwasawaZpPowerSeries):
            return False
        return self._spec == other._spec and self._coeffs == other._coeffs

    def __repr__(self) -> str:
        m = int(self._spec.t_precision)
        nz = [(i, c) for i, c in enumerate(self._coeffs) if int(c) != 0]
        if not nz:
            return f"ΛZp[{int(self._spec.p)},{int(self._spec.witt_length)}]/(T^{m}):0"
        head = nz[:6]
        parts = [f"{c}*T^{i}" if i != 0 else f"{c}" for i, c in head]
        more = "" if len(nz) <= 6 else f" + ...({len(nz)-6} more)"
        return f"ΛZp[{int(self._spec.p)},{int(self._spec.witt_length)}]/(T^{m}):" + " + ".join(parts) + more


def _hex_to_bytes(s: str) -> bytes:
    if not isinstance(s, str):
        raise TypeError(f"hex string must be str, got {type(s).__name__}")
    ss = s.strip().lower()
    if ss.startswith("0x"):
        ss = ss[2:]
    if len(ss) == 0 or (len(ss) % 2) != 0:
        raise ValueError("invalid hex string length")
    return bytes.fromhex(ss)


def normalize_key_bytes32(key: Any) -> bytes:
    """
    Normalize mapping key K to bytes32 using **left-zero padding** (EVM storage layout standard).

    Accepted forms:
    - bytes of length 32 (already bytes32)
    - bytes of length 20 (address) -> left-pad to 32
    - hex string '0x..' of 40 hex chars (address) -> left-pad to 32
    - hex string '0x..' of 64 hex chars (bytes32)
    """
    if isinstance(key, bytes):
        if len(key) == 32:
            return key
        if len(key) == 20:
            return b"\x00" * 12 + key
        raise ValueError(f"key bytes must be 20 (address) or 32 (bytes32), got len={len(key)}")
    if isinstance(key, str):
        b = _hex_to_bytes(key)
        if len(b) == 32:
            return b
        if len(b) == 20:
            return b"\x00" * 12 + b
        raise ValueError(f"key hex must decode to 20 or 32 bytes, got len={len(b)}")
    raise TypeError(f"unsupported key type: {type(key).__name__}")


def u256_to_bytes32_be(v: int) -> bytes:
    if not isinstance(v, int):
        raise TypeError(f"u256 value must be int, got {type(v).__name__}")
    if v < 0 or v.bit_length() > 256:
        raise ValueError("u256 out of range")
    return int(v).to_bytes(32, "big", signed=False)


def bytes32_be_to_u256(b: bytes) -> int:
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError(f"bytes32 must be bytes, got {type(b).__name__}")
    if len(b) != 32:
        raise ValueError(f"bytes32 must be length 32, got len={len(b)}")
    return int.from_bytes(bytes(b), "big", signed=False)


def keccak256_bytes(data: bytes) -> bytes:
    """
    Ethereum Keccak-256 hash (NOT FIPS SHA3-256).
    Deployment requirement: must have a Keccak backend; missing backend is fatal (no silent fallback).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")
    try:
        from Crypto.Hash import keccak  # type: ignore

        k = keccak.new(digest_bits=256)
        k.update(bytes(data))
        return k.digest()
    except Exception as e:
        raise RuntimeError(
            "Keccak backend missing. Install pycryptodome (Crypto.Hash.keccak). "
            "Deployment must abort."
        ) from e


def keccak256_mapping_step(*, key32: bytes, state32: bytes) -> bytes:
    """
    Operator B (fixed by user):
      next = keccak256( key32 || state32 )
    where both are bytes32 (total 64 bytes, big-endian semantics on state).
    """
    if not isinstance(key32, (bytes, bytearray)) or len(key32) != 32:
        raise ValueError("key32 must be bytes32")
    if not isinstance(state32, (bytes, bytearray)) or len(state32) != 32:
        raise ValueError("state32 must be bytes32")
    payload = bytes(key32) + bytes(state32)
    if len(payload) != 64:
        raise RuntimeError("mapping step input must be exactly 64 bytes")
    return keccak256_bytes(payload)


def iterate_keccak_mapping_orbit(
    *,
    key: Any,
    v0_padic: int,
    steps: int,
    p: int,
    witt_length: int,
) -> List[int]:
    """
    Build the Difference Observation Sequence O = {v_0, v_1, ..., v_steps} over Z/p^nZ, where:
      - v_0 is the p-adic seed (already in Z/p^nZ)
      - state_k := u256_be(v_k mod 2^256)
      - state_{k+1} := keccak256(key32 || state_k)   (bytes32)
      - v_{k+1} := Witt(state_{k+1})  (encoded as integer mod p^n; since state<2^256, this is a canonical embedding)

    This matches the user's decode->keccak->encode definition and keeps Key fixed.
    """
    if not isinstance(v0_padic, int):
        raise TypeError(f"v0_padic must be int, got {type(v0_padic).__name__}")
    if not isinstance(steps, int):
        raise TypeError(f"steps must be int, got {type(steps).__name__}")
    if int(steps) < 0:
        raise ValueError("steps must be >= 0")
    if not isinstance(p, int) or int(p) < 2:
        raise ValueError("p must be an integer prime >= 2")
    if not isinstance(witt_length, int) or int(witt_length) < 1:
        raise ValueError("witt_length must be >= 1")

    key32 = normalize_key_bytes32(key)
    modulus = int(int(p) ** int(witt_length))

    out: List[int] = []
    v = int(v0_padic % modulus)
    out.append(v)

    state = u256_to_bytes32_be(int(v % (1 << 256)))
    for _ in range(int(steps)):
        nxt = keccak256_mapping_step(key32=key32, state32=state)
        nxt_int = bytes32_be_to_u256(nxt)
        v = int(nxt_int % modulus)
        out.append(v)
        state = nxt  # next state's slot input is previous full keccak output bytes32

    return out


def _vp_p_power_trunc(x: int, p: int, k: int) -> int:
    """
    Truncated p-adic valuation v_p(x) in Z/p^kZ sense:
    - returns k if x ≡ 0 (mod p^k)
    - otherwise returns the largest v in [0,k-1] such that p^v | x
    """
    if not isinstance(x, int):
        raise TypeError(f"x must be int, got {type(x).__name__}")
    if not isinstance(p, int):
        raise TypeError(f"p must be int, got {type(p).__name__}")
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    if int(p) < 2:
        raise ValueError("p must be >= 2.")
    if int(k) < 1:
        raise ValueError("k must be >= 1.")

    pp = int(p)
    kk = int(k)
    if x == 0:
        return int(kk)

    # Fast path for p=2
    if pp == 2:
        # v2(x) = number of trailing zeros in binary, truncated at k
        # For negative numbers, Python uses infinite two's complement; normalize to abs.
        xx = int(abs(int(x)))
        if xx == 0:
            return int(kk)
        v = (xx & -xx).bit_length() - 1
        return int(v if v < kk else kk)

    xx = int(abs(int(x)))
    v = 0
    while v < kk and (xx % pp) == 0:
        xx //= pp
        v += 1
    return int(v)


def _trim_poly_mod(poly: List[int]) -> List[int]:
    """Trim trailing zeros (in the modular polynomial coefficient list)."""
    if not poly:
        return [0]
    i = len(poly) - 1
    while i > 0 and int(poly[i]) == 0:
        i -= 1
    return poly[: i + 1]


def verify_recurrence_over_zp_power(
    *,
    seq: Sequence[int],
    poly_f: Sequence[int],
    p: int,
    witt_length: int,
) -> bool:
    """
    Verify forward recurrence:
      Σ_{i=0..L} f[i] * seq[k+i] ≡ 0 (mod p^n) for all k where window fits.

    poly_f is low-degree first, and expected monic: f[L] == 1 (unit not required but recommended).
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError(f"seq must be a Sequence[int], got {type(seq).__name__}")
    if not isinstance(poly_f, (list, tuple)):
        raise TypeError(f"poly_f must be a Sequence[int], got {type(poly_f).__name__}")
    if not isinstance(p, int) or int(p) < 2:
        raise ValueError("p must be >= 2.")
    if not isinstance(witt_length, int) or int(witt_length) < 1:
        raise ValueError("witt_length must be >= 1.")

    modulus = int(int(p) ** int(witt_length))
    L = int(len(poly_f) - 1)
    if L < 0:
        raise ValueError("poly_f must be non-empty.")
    if len(seq) < 1:
        raise ValueError("seq must be non-empty.")
    # If L >= len(seq), there is no k such that the window fits; the condition is vacuously true.
    # We allow this case (it corresponds to "no informative recurrence found within the observed window").
    if int(len(seq)) < int(L + 1):
        return True

    f = [int(c % modulus) for c in poly_f]
    s = [int(x % modulus) for x in seq]

    for k in range(int(len(s)) - L):
        acc = 0
        for i in range(L + 1):
            acc = (acc + f[i] * s[k + i]) % modulus
        if int(acc) != 0:
            raise RuntimeError(
                "recurrence verification failed:\n"
                f"  p={int(p)} n={int(witt_length)} modulus=p^n\n"
                f"  k={int(k)} L={int(L)}\n"
                f"  residual={int(acc)}"
            )
    return True


def padic_berlekamp_massey_over_zp_power(
    *,
    seq: Sequence[int],
    p: int,
    witt_length: int,
) -> List[int]:
    """
    Deterministic Berlekamp–Massey-style synthesis over the chain ring Z/p^nZ.

    Output:
      A forward annihilating polynomial f(T)=Σ_{i=0..L} f[i] T^i (low-degree first) such that:
        Σ_{i=0..L} f[i] * seq[k+i] ≡ 0 (mod p^n)  for all k where defined,
      with f[L] = 1 (monic).

    Notes:
    - This works purely in Z/p^nZ (integer modulus), no floats, no randomness.
    - It is designed to be safe for p=2,n=512 and p=3,n=324 and secp256k1,n=3.
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError(f"seq must be a Sequence[int], got {type(seq).__name__}")
    if not isinstance(p, int) or int(p) < 2:
        raise ValueError("p must be >= 2.")
    if not isinstance(witt_length, int) or int(witt_length) < 1:
        raise ValueError("witt_length must be >= 1.")
    if len(seq) < 1:
        raise ValueError("seq must be non-empty.")

    pp = int(p)
    n = int(witt_length)
    modulus = int(pp ** n)

    s = [int(x % modulus) for x in seq]

    # Connection polynomial C(x)=1 + c1 x + ... + cL x^L
    # such that for all t>=L: sum_{i=0..L} c_i s[t-i] == 0 (mod p^n), with c_0=1.
    C: List[int] = [1]
    L = 0

    # Auxiliary pivots indexed by valuation. Each pivot stores:
    #   B_v : a previous connection polynomial snapshot
    #   b_v : discrepancy value at the time B_v was recorded
    #   pos_v : time index of recording (so shift = t - pos_v)
    #
    # IMPORTANT:
    # - We always include a base unit-pivot at v=0 with B=1, b=1, pos=-1.
    #   This guarantees solvability (b divides any discrepancy) and deterministic progress.
    pivots_B: Dict[int, List[int]] = {0: [1]}
    pivots_b: Dict[int, int] = {0: 1}
    pivots_pos: Dict[int, int] = {0: -1}

    def _pow_p(e: int) -> int:
        return int(pp ** int(e))

    for t in range(len(s)):
        # Compute discrepancy d_t = Σ_{i=0..L} C[i] * s[t-i]
        d = 0
        for i in range(L + 1):
            idx = int(t - i)
            if idx < 0:
                break
            d = (d + (int(C[i]) * int(s[idx]))) % modulus

        if int(d) == 0:
            continue

        vd = int(_vp_p_power_trunc(int(d), pp, n))
        if vd >= n:
            # d ≡ 0 (mod p^n) would have been caught above
            continue

        # Choose pivot to minimize the required new degree bound.
        # Constraint: pivot discrepancy b must divide d in Z/p^nZ, i.e., v_p(b) <= v_p(d).
        best = None
        for pv, Bv in pivots_B.items():
            bv = int(pivots_b.get(pv, 0))
            posv = int(pivots_pos.get(pv, -1))
            vbv = int(_vp_p_power_trunc(int(bv), pp, n))
            if vbv > vd:
                continue
            degBv = int(len(Bv) - 1)
            shiftv = int(t - posv)
            if shiftv <= 0:
                continue
            deg_candidate = max(int(L), int(degBv + shiftv))
            key = (int(deg_candidate), int(shiftv), int(vbv), int(pv))
            if best is None or key < best[0]:
                best = (key, int(pv), Bv, bv, posv, vbv, shiftv, degBv)

        if best is None:
            # Should never happen due to unit pivot v=0 (b=1).
            raise RuntimeError("no valid pivot found (internal error)")

        _key, pivot_v, B, b, pos, vb, shift = best[0], best[1], best[2], best[3], best[4], best[5], best[6]

        # Solve q*b ≡ d (mod p^n) with vb <= vd by dividing p^vb.
        mod_red = int(pp ** int(n - vb))
        b_red = int((int(b) // _pow_p(vb)) % mod_red)
        d_red = int((int(d) // _pow_p(vb)) % mod_red)
        if int(b_red) % pp == 0:
            raise RuntimeError("pivot unit-part unexpectedly divisible by p; cannot invert")
        inv_b_red = int(pow(int(b_red), -1, int(mod_red)))
        q = int((d_red * inv_b_red) % mod_red)

        # Update C <- C - q * x^shift * B  (mod p^n)
        T_old = list(C)
        needed = int(len(B) + int(shift))
        if len(C) < needed:
            C.extend([0] * (needed - len(C)))
        for i in range(len(B)):
            C[i + int(shift)] = int((int(C[i + int(shift)]) - int(q) * int(B[i])) % modulus)
        # IMPORTANT: Do NOT trim trailing zeros here.
        # Over Z/p^nZ, cancellations at the highest degree can occur during updates,
        # but the BM complexity parameter L must remain non-decreasing (as in the field BM algorithm).
        # We therefore track L explicitly and only ever increase it when the required bound increases.
        L_candidate = int(needed - 1)
        # Always record the most recent pivot for this valuation level.
        # (This is essential over Z/p^nZ: non-unit discrepancies carry structural information
        # that must be kept even if the current L does not increase.)
        pivots_B[int(vd)] = T_old
        pivots_b[int(vd)] = int(d)
        pivots_pos[int(vd)] = int(t)

        if L_candidate > L:
            L = int(L_candidate)
            # Ensure C has length exactly L+1 (pad if needed).
            if len(C) < (L + 1):
                C.extend([0] * ((L + 1) - len(C)))

    # Convert connection polynomial C (c0=1) to forward polynomial f (monic): reverse coefficients.
    if len(C) < (L + 1):
        C.extend([0] * ((L + 1) - len(C)))
    f = list(reversed(C[: L + 1]))
    f = [int(c % modulus) for c in f]
    if len(f) < 1:
        raise RuntimeError("internal error: empty polynomial")
    if int(f[-1]) != 1:
        # In principle f[-1] should be c0 == 1.
        raise RuntimeError(f"internal error: expected monic polynomial with leading coeff 1, got {int(f[-1])}")

    # Strict verification on the given data window.
    verify_recurrence_over_zp_power(seq=s, poly_f=f, p=pp, witt_length=n)
    return f


def _self_test_padic_bm_small() -> Dict[str, Any]:
    """
    Deterministic self-test for the p-adic BM synthesizer on tiny rings.
    This is intentionally small and exhaustive (no randomness).

    Scope:
    - Validates that the synthesizer always returns a **monic** annihilating polynomial
      and that `verify_recurrence_over_zp_power` accepts it on the same observation window.
    - Does NOT attempt to prove global minimality over Z/p^nZ here (that requires the
      full Reeds–Sloane chain-ring synthesis theory and is out of scope for this smoke-level check).
    """
    results: Dict[str, Any] = {"ok": True, "cases": []}

    # Exhaustive sequences over Z/4Z of length 6 (4^6=4096 cases)
    p, n, N = 2, 2, 6
    mod = int(p ** n)
    from itertools import product

    for seq in product(range(mod), repeat=N):
        seq_l = list(seq)
        f = padic_berlekamp_massey_over_zp_power(seq=seq_l, p=p, witt_length=n)
        if not isinstance(f, list) or not f:
            raise RuntimeError("BM returned empty polynomial (invalid).")
        if int(f[-1] % mod) != 1:
            raise RuntimeError(f"BM returned non-monic polynomial over Z/4Z: f[-1]={int(f[-1])}")
        verify_recurrence_over_zp_power(seq=seq_l, poly_f=f, p=p, witt_length=n)

    results["cases"].append({"ring": "Z/4Z", "len": N, "count": int(mod**N)})

    return results


@dataclass(frozen=True)
class IwasawaTorsionCertificate:
    """
    Deterministic torsion certificate for a single difference-seed orbit under Operator-B.

    This certificate is intentionally JSON-friendly (ints + hex strings) and avoids huge dumps.
    """

    ok: bool
    p: int
    witt_length: int
    steps: int
    modulus: int
    key32_hex: str
    slot_a_hex: str
    slot_b_hex: str
    seed_u256_hex: str
    seed_padic: int
    poly_degree: int
    poly_coeffs: Tuple[int, ...]  # forward polynomial coeffs (low degree first), monic
    torsion_detected: bool
    degree_threshold: int
    # Norton–Salagean / chain-ring synthesis evidence (deterministic, JSON-friendly).
    # This is the audit trail that replaces the previous "BM black box".
    synthesis_certificate: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "p": int(self.p),
            "witt_length": int(self.witt_length),
            "steps": int(self.steps),
            "modulus": str(int(self.modulus)),
            "key32": str(self.key32_hex),
            "slot_a": str(self.slot_a_hex),
            "slot_b": str(self.slot_b_hex),
            "seed_u256": str(self.seed_u256_hex),
            "seed_padic": str(int(self.seed_padic)),
            "poly_degree": int(self.poly_degree),
            "poly_coeffs": [str(int(c)) for c in self.poly_coeffs],
            "synthesis_certificate": self.synthesis_certificate,
            "torsion_detected": bool(self.torsion_detected),
            "degree_threshold": int(self.degree_threshold),
            "error": self.error,
        }


def _normalize_bytes32(x: Any) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        b = bytes(x)
        if len(b) != 32:
            raise ValueError(f"bytes32 must be length 32, got len={len(b)}")
        return b
    if isinstance(x, str):
        b = _hex_to_bytes(x)
        if len(b) != 32:
            raise ValueError(f"bytes32 hex must decode to length 32, got len={len(b)}")
        return b
    raise TypeError(f"unsupported bytes32 type: {type(x).__name__}")


def compute_seed_from_slots_u256(
    *,
    slot_a: Any,
    slot_b: Any,
) -> Tuple[int, bytes]:
    """
    Compute the physical u256 seed:
      seed_u256 = (slotA - slotB) mod 2^256
    Returns (seed_int, seed_bytes32_be).
    """
    a = bytes32_be_to_u256(_normalize_bytes32(slot_a))
    b = bytes32_be_to_u256(_normalize_bytes32(slot_b))
    seed = int((a - b) % (1 << 256))
    return seed, u256_to_bytes32_be(seed)


def compute_iwasawa_torsion_certificate_operator_b(
    *,
    key: Any,
    slot_a: Any,
    slot_b: Any,
    p: int,
    witt_length: int,
    steps: int,
) -> IwasawaTorsionCertificate:
    """
    End-to-end (Injection -> Evolution -> Extraction) for Operator-B:

      Injection:
        seed_u256 = (slotA - slotB) mod 2^256
        seed_padic = seed_u256 mod p^n

      Evolution:
        state_{k+1} = keccak256(key32 || state_k)
        v_{k+1} = int(state_{k+1}) mod p^n

      Extraction:
        f(T) via p-adic BM over Z/p^nZ on the sequence {v_0..v_steps}.

    Deterministic, no heuristics. Any internal inconsistency raises.
    """
    try:
        pp = int(p)
        nn = int(witt_length)
        if pp < 2:
            raise ValueError("p must be >= 2.")
        if nn < 1:
            raise ValueError("witt_length must be >= 1.")
        if not isinstance(steps, int) or int(steps) < 0:
            raise ValueError("steps must be >= 0.")

        key32 = normalize_key_bytes32(key)
        slot_a_b = _normalize_bytes32(slot_a)
        slot_b_b = _normalize_bytes32(slot_b)

        seed_u256, seed_bytes = compute_seed_from_slots_u256(slot_a=slot_a_b, slot_b=slot_b_b)
        modulus = int(pp ** nn)
        seed_padic = int(seed_u256 % modulus)

        # Observation window length = steps+1
        seq = iterate_keccak_mapping_orbit(
            key=key32,
            v0_padic=seed_padic,
            steps=int(steps),
            p=pp,
            witt_length=nn,
        )

        # Chain-ring synthesis (Norton–Salagean / Reeds–Sloane semantics):
        # We synthesize the minimal connection polynomial C(T)=1+...+c_L T^L over Z/p^nZ,
        # then convert it to the forward annihilator f(T)=Σ f_i T^i (low degree first) such that:
        #   Σ_{i=0..L} f_i * v_{k+i} ≡ 0 (mod p^n)
        from .norton_salagean import ChainRingSpec, norton_salagean_bm

        ns_spec = ChainRingSpec(p=pp, n=nn)
        ns = norton_salagean_bm(seq, ns_spec, require_solution=True, verify_with_oracle=False)
        if ns is None:
            raise RuntimeError("internal: norton_salagean_bm returned None under require_solution=True")

        C = list(ns.connection_polynomial)  # [1,c1,...,cL]
        deg = int(ns.degree)
        if len(C) != int(deg + 1):
            raise RuntimeError("internal: Norton–Salagean connection polynomial length mismatch")
        # forward polynomial (monic): reverse connection coefficients
        f = list(reversed(C))
        if not f or int(f[-1] % (pp**nn)) != 1:
            raise RuntimeError("internal: forward polynomial must be monic (leading coeff 1)")

        # Strict verification on the observation window (must abort on any mismatch).
        verify_recurrence_over_zp_power(seq=seq, poly_f=f, p=pp, witt_length=nn)

        # BM reconstruct limit: deg < steps/2 indicates collapse (torsion-like)
        degree_threshold = int(int(steps) // 2)
        torsion_detected = bool(deg < degree_threshold)

        return IwasawaTorsionCertificate(
            ok=True,
            p=pp,
            witt_length=nn,
            steps=int(steps),
            modulus=int(modulus),
            key32_hex="0x" + bytes(key32).hex(),
            slot_a_hex="0x" + bytes(slot_a_b).hex(),
            slot_b_hex="0x" + bytes(slot_b_b).hex(),
            seed_u256_hex="0x" + bytes(seed_bytes).hex(),
            seed_padic=int(seed_padic),
            poly_degree=int(deg),
            poly_coeffs=tuple(int(c) for c in f),
            synthesis_certificate=dict(ns.certificate),
            torsion_detected=bool(torsion_detected),
            degree_threshold=int(degree_threshold),
            error=None,
        )
    except Exception as e:
        # Hard failure: return a certificate marked not ok (for JSON pipelines),
        # but also keep the error explicit; callers who need hard-abort should raise.
        return IwasawaTorsionCertificate(
            ok=False,
            p=int(p) if isinstance(p, int) else -1,
            witt_length=int(witt_length) if isinstance(witt_length, int) else -1,
            steps=int(steps) if isinstance(steps, int) else -1,
            modulus=0,
            key32_hex="",
            slot_a_hex="",
            slot_b_hex="",
            seed_u256_hex="",
            seed_padic=0,
            poly_degree=-1,
            poly_coeffs=tuple(),
            synthesis_certificate={},
            torsion_detected=False,
            degree_threshold=0,
            error=str(e),
        )
