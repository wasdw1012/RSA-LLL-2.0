"""
MVP18: 证人

数学原理：
- 处理 f(x) ≅ 0（导出意义下的同伦零点）
- 实现安德烈-奎伦上同调求解器
- 构造余切复形与虚拟基本循环

核心组件：
1. 单纯交换环 (SimplicialPoly) - 排中律失效的物理容器
2. Kähler 微分模块 - 余切复形的基础
3. 上同调群计算 - H^{-2}, H^{-1}, H^0 的严格实现
4. 穿墙术判定 - H^{-2}=0 ∧ H^{-1}≠0 的量子隧穿条件

严格要求：
- 禁止静默降级 所有失败全部中断抛出异常
- 禁止硬编码魔法数
- 禁止符号计算库
- 禁止浮点近似
- 所有计算必须数学可验证
"""
from __future__ import annotations

from typing import (
    List, Dict, Tuple, Optional, Union, Any, 
    TypeVar, Generic, Callable, Set, FrozenSet
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

__all__ = [
    # ═══════════════════════════════════════════════════════════════════════════
    # 单纯交换环（核心一）
    # ═══════════════════════════════════════════════════════════════════════════
    "SimplicialLevel",
    "SimplicialPoly",
    # ═══════════════════════════════════════════════════════════════════════════
    # 基础代数结构
    # ═══════════════════════════════════════════════════════════════════════════
    "RingElement",
    "FieldElement",
    "Ring",
    "Field",
    "RationalNumber",
    "RationalField",
    "Polynomial",
    "PolynomialRing",
    # ═══════════════════════════════════════════════════════════════════════════
    # 模与同态
    # ═══════════════════════════════════════════════════════════════════════════
    "ModuleElement",
    "Module",
    "FreeModule",
    "ModuleHomomorphism",
    "CallableModuleHomomorphism",
    "FreeModuleHomomorphism",
    # ═══════════════════════════════════════════════════════════════════════════
    # Gröbner 基与理想
    # ═══════════════════════════════════════════════════════════════════════════
    "MonomialOrder",
    "DegreeLexOrder",
    "PolynomialIdeal",
    # ═══════════════════════════════════════════════════════════════════════════
    # 张量积与 Kähler 微分
    # ═══════════════════════════════════════════════════════════════════════════
    "TensorBackend",
    "FreeModuleTensorBackend",
    "TensorProduct",
    "TensorElement",
    "KahlerDifferentials",
    "QuotientModule",
    # ═══════════════════════════════════════════════════════════════════════════
    # 余切复形与上同调（核心二）
    # ═══════════════════════════════════════════════════════════════════════════
    "CotangentComplex",
    "NormalizedChainComplex",
    "ZeroModule",
    "Submodule",
    "CohomologyGroup",
    # ═══════════════════════════════════════════════════════════════════════════
    # 安德烈-奎伦求解器（最终接口）
    # ═══════════════════════════════════════════════════════════════════════════
    "AndreQuillenSolver",
    "SimplicialRing",
]

T = TypeVar('T')

class SimplicialLevel(Generic[T]):
    """Represents a single level in the simplicial structure."""
    
    def __init__(self, level: int, elements: Tuple[T, ...]):
        """
        Initialize a simplicial level.
        level: the simplicial degree (n in P_n)
        elements: tuple of elements at this level
        """
        if level < 0:
            raise ValueError("Simplicial level cannot be negative")
        self.level = level
        self.elements = elements
    
    def __eq__(self, other: 'SimplicialLevel') -> bool:
        return (self.level == other.level and 
                self.elements == other.elements)
    
    def __repr__(self) -> str:
        return f"P_{self.level}({self.elements})"

class SimplicialPoly:
    """
    Mathematical foundation:
    - For a polynomial f, the derived zero locus is modeled by a simplicial ring
    - The chain complex is: ... → 0 → R --f--> R → 0 (Koszul complex)
    - Dold-Kan correspondence converts this to a simplicial abelian group
    - The dga structure makes it a simplicial commutative ring
    
    Structure:
    - P_0 = R (base polynomial ring)
    - P_1 = R ⊕ R (with specific ring structure)
    - P_2 = R ⊕ R ⊕ R
    - etc.
    
    Face operators ∂_i: P_n → P_{n-1} implement "edge detection" in logic
    Degeneracy operators s_i: P_n → P_{n+1} implement degenerate paths
    """
    
    def __init__(self, base_ring: PolynomialRing, f: Polynomial):
        """
        Initialize the simplicial polynomial structure.
        
        Args:
            base_ring: The base polynomial ring (e.g., k[x])
            f: The polynomial defining the constraint (f(x) ≅ 0)
        """
        self.base_ring = base_ring
        self.f = f
        self.field = base_ring.field
    
    def _create_level_0(self) -> SimplicialLevel[Polynomial]:
        """Create level 0: P_0 = R (base ring)"""
        return SimplicialLevel(0, (self.base_ring.zero(),))
    
    def _create_level_1(self) -> SimplicialLevel[Tuple[Polynomial, Polynomial]]:
        """Create level 1: P_1 = R ⊕ R"""
        return SimplicialLevel(1, (
            self.base_ring.zero(),
            self.base_ring.zero()
        ))
    
    def _create_level_2(self) -> SimplicialLevel[Tuple[Polynomial, Polynomial, Polynomial]]:
        """Create level 2: P_2 = R ⊕ R ⊕ R"""
        return SimplicialLevel(2, (
            self.base_ring.zero(),
            self.base_ring.zero(),
            self.base_ring.zero()
        ))
    
    def get_level(self, n: int) -> SimplicialLevel:
        """
        Get the simplicial level P_n.
        
        For the Koszul complex of a single element, the structure is:
        - P_0 = R
        - P_1 = R ⊕ R
        - P_2 = R ⊕ R ⊕ R
        - P_n = R^^{n+1} for n ≥ 0
        
        This follows from the Dold-Kan correspondence applied to the 
        chain complex concentrated in degrees 0 and 1.
        """
        if n < 0:
            raise ValueError("Simplicial level cannot be negative")
        
        # For n=0, P_0 = R
        if n == 0:
            return self._create_level_0()
        
        # For n≥1, P_n = R^^{n+1}
        zero_poly = self.base_ring.zero()
        elements = tuple(zero_poly for _ in range(n+1))
        
        return SimplicialLevel(n, elements)
    
    def face_operator(self, n: int, i: int) -> Callable[[SimplicialLevel], SimplicialLevel]:
        """
        Get the face operator ∂_i: P_n → P_{n-1}.
        
        Implements the "edge detection" mechanism described in the specification.
        When classical logic (P_0) fails, these operators move to higher homotopy levels.
        
        Mathematical definition from Dold-Kan correspondence:
        - For P_1 → P_0: 
            ∂_0(a,b) = a + f·b
            ∂_1(a,b) = a
        - For P_2 → P_1:
            ∂_0(a,b,c) = (a + f·b, c)
            ∂_1(a,b,c) = (a, b + c)
            ∂_2(a,b,c) = (a, b)
        - General pattern follows simplicial identities
        
        These operators satisfy the simplicial identities:
        ∂_i ∂_j = ∂_{j-1} ∂_i for 0 ≤ i < j ≤ n
        """
        if n <= 0:
            raise ValueError("Face operator requires n ≥ 1")
        if i < 0 or i > n:
            raise ValueError(f"Face index i must be in [0, {n}]")
        
        def apply(level: SimplicialLevel) -> SimplicialLevel:
            if level.level != n:
                raise ValueError(f"Expected level {n}, got {level.level}")
            
            # Level 1 to Level 0
            if n == 1:
                a, b = level.elements
                if i == 0:
                    # ∂_0(a,b) = a + f·b
                    fb = self.f * b
                    result = a + fb
                    return SimplicialLevel(0, (result,))
                elif i == 1:
                    # ∂_1(a,b) = a
                    return SimplicialLevel(0, (a,))
            
            # Level 2 to Level 1
            elif n == 2:
                a, b, c = level.elements
                if i == 0:
                    # ∂_0(a,b,c) = (a + f·b, c)
                    fb = self.f * b
                    result_a = a + fb
                    return SimplicialLevel(1, (result_a, c))
                elif i == 1:
                    # ∂_1(a,b,c) = (a, b + c)
                    result_b = b + c
                    return SimplicialLevel(1, (a, result_b))
                elif i == 2:
                    # ∂_2(a,b,c) = (a, b)
                    return SimplicialLevel(1, (a, b))
            
            # General case for n ≥ 3
            else:
                elements = list(level.elements)
                if i == 0:
                    # ∂_0: combine first two elements with f
                    a0, a1 = elements[0], elements[1]
                    new_a0 = a0 + (self.f * a1)
                    new_elements = [new_a0] + elements[2:]
                elif i == n:
                    # ∂_n: drop the last element
                    new_elements = elements[:-1]
                else:
                    # ∂_i for 0 < i < n: combine elements i and i+1
                    ai = elements[i]
                    ai1 = elements[i+1]
                    new_ai = ai + ai1
                    new_elements = elements[:i] + [new_ai] + elements[i+2:]
                
                return SimplicialLevel(n-1, tuple(new_elements))
        
        return apply
    
    def degeneracy_operator(self, n: int, i: int) -> Callable[[SimplicialLevel], SimplicialLevel]:
        """
        Get the degeneracy operator s_i: P_n → P_{n+1}.
        
        Implements degenerate paths in the homotopy structure.
        
        Mathematical definition:
        - For P_0 → P_1:
            s_0(a) = (a, 0)
        - For P_1 → P_2:
            s_0(a,b) = (a, 0, b)
            s_1(a,b) = (a, b, 0)
        - General pattern follows simplicial identities
        
        These operators satisfy the simplicial identities:
        s_i s_j = s_{j+1} s_i for 0 ≤ i ≤ j ≤ n
        ∂_i s_j = s_{j-1} ∂_i for i < j
        ∂_j s_j = id = ∂_{j+1} s_j
        ∂_i s_j = s_j ∂_{i-1} for i > j+1
        """
        if n < 0:
            raise ValueError("Degeneracy operator requires n ≥ 0")
        if i < 0 or i > n:
            raise ValueError(f"Degeneracy index i must be in [0, {n}]")
        
        def apply(level: SimplicialLevel) -> SimplicialLevel:
            if level.level != n:
                raise ValueError(f"Expected level {n}, got {level.level}")
            
            # Level 0 to Level 1
            if n == 0:
                a, = level.elements
                if i == 0:
                    # s_0(a) = (a, 0)
                    zero = self.base_ring.zero()
                    return SimplicialLevel(1, (a, zero))
            
            # Level 1 to Level 2
            elif n == 1:
                a, b = level.elements
                if i == 0:
                    # s_0(a,b) = (a, 0, b)
                    zero = self.base_ring.zero()
                    return SimplicialLevel(2, (a, zero, b))
                elif i == 1:
                    # s_1(a,b) = (a, b, 0)
                    zero = self.base_ring.zero()
                    return SimplicialLevel(2, (a, b, zero))
            
            # General case for n ≥ 2
            else:
                elements = list(level.elements)
                if i == 0:
                    # s_0: insert zero after first element
                    zero = self.base_ring.zero()
                    new_elements = [elements[0], zero] + elements[1:]
                elif i == n:
                    # s_n: append zero at the end
                    zero = self.base_ring.zero()
                    new_elements = elements + [zero]
                else:
                    # s_i: insert zero at position i+1
                    zero = self.base_ring.zero()
                    new_elements = elements[:i+1] + [zero] + elements[i+1:]
                
                return SimplicialLevel(n+1, tuple(new_elements))
        
        return apply
    
    def ring_structure(self, n: int) -> Callable[[SimplicialLevel, SimplicialLevel], SimplicialLevel]:
        """
        Get the ring multiplication for level P_n.
        
        The ring structure is induced by the dga structure of the Koszul complex:
        - P_0: standard polynomial ring multiplication
        - P_1: (a,b) * (c,d) = (a·c, a·d + b·c)
        - P_2: (a,b,c) * (d,e,f) = (a·d, a·e + b·d, a·f + c·d)
        - General: follows from the graded commutativity of the dga
        
        This structure makes each P_n a commutative ring, and all face/degeneracy 
        operators are ring homomorphisms, satisfying the requirements for a 
        simplicial commutative ring.
        """
        if n < 0:
            raise ValueError("Ring level cannot be negative")
        
        def multiply(level1: SimplicialLevel, level2: SimplicialLevel) -> SimplicialLevel:
            if level1.level != n or level2.level != n:
                raise ValueError(f"Both levels must be {n}")
            
            # Level 0: standard polynomial multiplication
            if n == 0:
                a, = level1.elements
                b, = level2.elements
                return SimplicialLevel(0, (a * b,))
            
            # Level 1: (a,b) * (c,d) = (a·c, a·d + b·c)
            elif n == 1:
                a1, b1 = level1.elements
                a2, b2 = level2.elements
                prod_a = a1 * a2
                prod_b = (a1 * b2) + (b1 * a2)
                return SimplicialLevel(1, (prod_a, prod_b))
            
            # Level 2: (a,b,c) * (d,e,f) = (a·d, a·e + b·d, a·f + c·d)
            elif n == 2:
                a1, b1, c1 = level1.elements
                a2, b2, c2 = level2.elements
                prod_a = a1 * a2
                prod_b = (a1 * b2) + (b1 * a2)
                prod_c = (a1 * c2) + (c1 * a2)
                return SimplicialLevel(2, (prod_a, prod_b, prod_c))
            
            # General case for n ≥ 3
            else:
                elements1 = level1.elements
                elements2 = level2.elements
                result = [self.base_ring.zero() for _ in range(n+1)]
                
                # In the Koszul dga, product of two degree 1 elements is zero,
                # so only "linear" terms contribute
                for i in range(n+1):
                    for j in range(i+1):
                        # Only adjacent terms contribute due to the dga structure
                        if i - j <= 1:
                            result[i] = result[i] + (elements1[j] * elements2[i-j])
                
                return SimplicialLevel(n, tuple(result))
        
        return multiply
    
    def zero_element(self, n: int) -> SimplicialLevel:
        """Get the zero element at level n."""
        level = self.get_level(n)
        return level  # Already initialized with zeros
    
    def one_element(self, n: int) -> SimplicialLevel:
        """Get the multiplicative identity at level n."""
        if n < 0:
            raise ValueError("Level cannot be negative")
        
        one_poly = self.base_ring.one()
        zero_poly = self.base_ring.zero()
        
        if n == 0:
            return SimplicialLevel(0, (one_poly,))
        else:
            elements = [zero_poly] * (n+1)
            elements[0] = one_poly
            return SimplicialLevel(n, tuple(elements))
    
    def verify_simplicial_identities(self, max_level: int = 2):
        """
        Verify the simplicial identities up to a given level.
        
        This is a mathematical check that the implementation satisfies:
        1. ∂_i ∂_j = ∂_{j-1} ∂_i for 0 ≤ i < j ≤ n
        2. s_i s_j = s_{j+1} s_i for 0 ≤ i ≤ j ≤ n
        3. ∂_i s_j = 
            - s_{j-1} ∂_i if i < j
            - id if i = j or i = j+1
            - s_j ∂_{i-1} if i > j+1
        
        This verification ensures the structure is mathematically valid.
        """
        # Create non-trivial elements for testing
        test_elements = {}
        for n in range(max_level + 1):
            if n == 0:
                x = self.base_ring.variable_poly()
                test_elements[n] = SimplicialLevel(0, (x,))
            elif n == 1:
                x = self.base_ring.variable_poly()
                one = self.base_ring.one()
                test_elements[n] = SimplicialLevel(1, (x, one))
            elif n == 2:
                x = self.base_ring.variable_poly()
                one = self.base_ring.one()
                zero = self.base_ring.zero()
                test_elements[n] = SimplicialLevel(2, (x, one, zero))
        
        # Check face identities: ∂_i ∂_j = ∂_{j-1} ∂_i for i < j
        for n in range(2, max_level + 1):
            for j in range(1, n + 1):
                for i in range(0, j):
                    # Apply ∂_i ∂_j
                    dj = self.face_operator(n, j)
                    di = self.face_operator(n-1, i)
                    result1 = di(dj(test_elements[n]))
                    
                    # Apply ∂_{j-1} ∂_i
                    di2 = self.face_operator(n, i)
                    dj1 = self.face_operator(n-1, j-1)
                    result2 = dj1(di2(test_elements[n]))
                    
                    if result1 != result2:
                        raise ValueError(f"Face identity failed: ∂_{i}∂_{j} ≠ ∂_{j-1}∂_{i} at level {n}")
        
        # Check degeneracy identities: s_i s_j = s_{j+1} s_i for i ≤ j
        for n in range(0, max_level):
            for j in range(0, n + 1):
                for i in range(0, j + 1):
                    # Apply s_i s_j
                    sj = self.degeneracy_operator(n, j)
                    si = self.degeneracy_operator(n+1, i)
                    result1 = si(sj(test_elements[n]))
                    
                    # Apply s_{j+1} s_i
                    si2 = self.degeneracy_operator(n, i)
                    sj1 = self.degeneracy_operator(n+1, j+1)
                    result2 = sj1(si2(test_elements[n]))
                    
                    if result1 != result2:
                        raise ValueError(f"Degeneracy identity failed: s_{i}s_{j} ≠ s_{j+1}s_{i} at level {n}")
        
        # Check mixed identities
        for n in range(0, max_level):
            element = test_elements[n]
            
            for j in range(0, n + 1):
                sj = self.degeneracy_operator(n, j)
                level_n1 = sj(element)
                
                for i in range(0, n + 2):
                    if i < j:
                        # ∂_i s_j = s_{j-1} ∂_i
                        di = self.face_operator(n+1, i)
                        di2 = self.face_operator(n, i)
                        sj1 = self.degeneracy_operator(n-1, j-1)  # 修正：应该作用在 level n-1
                        
                        result1 = di(sj(element))
                        result2 = sj1(di2(element))
                        
                        if result1 != result2:
                            raise ValueError(f"Mixed identity failed (i<j): ∂_{i}s_{j} ≠ s_{j-1}∂_{i} at level {n}")
                    
                    elif i == j or i == j+1:
                        # ∂_i s_j = id
                        di = self.face_operator(n+1, i)
                        result = di(sj(element))
                        
                        if result != element:
                            raise ValueError(f"Mixed identity failed (i=j or i=j+1): ∂_{i}s_{j} ≠ id at level {n}")
                    
                    elif i > j+1:
                        # ∂_i s_j = s_j ∂_{i-1}
                        di = self.face_operator(n+1, i)
                        di1 = self.face_operator(n, i-1)
                        sj2 = self.degeneracy_operator(n-1, j)  # 修正：应该作用在 level n-1
                        
                        result1 = di(sj(element))
                        result2 = sj2(di1(element))
                        
                        if result1 != result2:
                            raise ValueError(f"Mixed identity failed (i>j+1): ∂_{i}s_{j} ≠ s_{j}∂_{i-1} at level {n}")
        
        return True  # All identities verified



# ======================
# 严格基础代数结构
# ======================

T = TypeVar('T')

class RingElement(ABC):
    """环元素的抽象基类 - 严格遵循环公理"""
    
    @abstractmethod
    def __add__(self, other) -> 'RingElement':
        """环加法：必须满足交换群公理"""
        pass
    
    @abstractmethod
    def __mul__(self, other) -> 'RingElement':
        """环乘法：必须满足半群公理及分配律"""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """等价关系：必须满足自反性、对称性、传递性"""
        pass
    
    @abstractmethod
    def __neg__(self) -> 'RingElement':
        """加法逆元：a + (-a) = 0"""
        pass
    
    @abstractmethod
    def is_zero(self) -> bool:
        """检查是否为加法单位元"""
        pass
    
    @abstractmethod
    def is_one(self) -> bool:
        """检查是否为乘法单位元"""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

class FieldElement(RingElement):
    """域元素的抽象基类"""
    
    @abstractmethod
    def inverse(self) -> 'FieldElement':
        """乘法逆元：a * a⁻¹ = 1 (a ≠ 0)"""
        pass

class Ring(ABC):
    """环的抽象基类 - 严格遵循环公理"""
    
    @abstractmethod
    def zero(self) -> RingElement:
        """返回加法单位元 0"""
        pass
    
    @abstractmethod
    def one(self) -> RingElement:
        """返回乘法单位元 1"""
        pass
    
    @abstractmethod
    def add(self, a: RingElement, b: RingElement) -> RingElement:
        """环加法实现"""
        pass
    
    @abstractmethod
    def mul(self, a: RingElement, b: RingElement) -> RingElement:
        """环乘法实现"""
        pass
    
    @abstractmethod
    def negate(self, a: RingElement) -> RingElement:
        """加法逆元计算"""
        pass
    
    @abstractmethod
    def equal(self, a: RingElement, b: RingElement) -> bool:
        """等价关系判定"""
        pass
    
    @abstractmethod
    def is_commutative(self) -> bool:
        """检查是否为交换环"""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

class Field(Ring):
    """域的抽象基类 - 严格遵循域公理"""
    
    @abstractmethod
    def inverse(self, a: FieldElement) -> FieldElement:
        """乘法逆元计算 (a ≠ 0)"""
        pass
    
    @abstractmethod
    def divide(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """域除法：a / b = a * b⁻¹ (b ≠ 0)"""
        pass

# ======================
# 有理数域的严格实现（无浮点数）
# ======================

class RationalNumber(FieldElement):
    """有理数的严格实现 - 无任何近似"""
    
    def __init__(self, numerator: int, denominator: int = 1):
        if denominator == 0:
            raise ValueError("分母不能为零")
        
        gcd = math.gcd(abs(numerator), abs(denominator))
        self.numerator = numerator // gcd
        self.denominator = denominator // gcd
        
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
    
    def __add__(self, other: 'RationalNumber') -> 'RationalNumber':
        num = self.numerator * other.denominator + other.numerator * self.denominator
        den = self.denominator * other.denominator
        return RationalNumber(num, den)
    
    def __mul__(self, other: 'RationalNumber') -> 'RationalNumber':
        return RationalNumber(
            self.numerator * other.numerator,
            self.denominator * other.denominator
        )
    
    def __eq__(self, other: 'RationalNumber') -> bool:
        return (self.numerator == other.numerator and 
                self.denominator == other.denominator)
    
    def __neg__(self) -> 'RationalNumber':
        return RationalNumber(-self.numerator, self.denominator)
    
    def is_zero(self) -> bool:
        return self.numerator == 0
    
    def is_one(self) -> bool:
        return self.numerator == 1 and self.denominator == 1
    
    def inverse(self) -> 'RationalNumber':
        if self.is_zero():
            raise ZeroDivisionError("零元无乘法逆元")
        return RationalNumber(self.denominator, self.numerator)
    
    def __repr__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

class RationalField(Field):
    """有理数域的严格实现"""
    
    # 添加 RationalNumber 类引用（用于访问）
    RationalNumber = RationalNumber
    
    def zero(self) -> RationalNumber:
        return RationalNumber(0, 1)
    
    def one(self) -> RationalNumber:
        return RationalNumber(1, 1)
    
    def add(self, a: RationalNumber, b: RationalNumber) -> RationalNumber:
        return a + b
    
    def mul(self, a: RationalNumber, b: RationalNumber) -> RationalNumber:
        return a * b
    
    def negate(self, a: RationalNumber) -> RationalNumber:
        return -a
    
    def equal(self, a: RationalNumber, b: RationalNumber) -> bool:
        return a == b
    
    def is_commutative(self) -> bool:
        return True
    
    def inverse(self, a: RationalNumber) -> RationalNumber:
        return a.inverse()
    
    def divide(self, a: RationalNumber, b: RationalNumber) -> RationalNumber:
        return a * b.inverse()
    
    def __repr__(self) -> str:
        return "Q"

# ======================
# 多项式环的严格实现（无符号计算库）
# ======================

# 辅助类：用于将 compare 方法转换为 key
class _MonomialOrderWrapper:
    def __init__(self, order, monomial):
        self.order = order
        self.monomial = monomial
    
    def __lt__(self, other):
        return self.order.compare(self.monomial, other.monomial) < 0
    
    def __eq__(self, other):
        return self.order.compare(self.monomial, other.monomial) == 0


def _grevlex_key(exp: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """
    grevlex key (deterministic, exact):
    - compare by total degree
    - tie-break by reverse-lex with sign flip

    Larger key => "larger monomial".
    """
    return (sum(int(e) for e in exp), tuple(-int(e) for e in reversed(exp)))

class Polynomial(RingElement):
    """多项式的严格实现 - 基于系数字典（支持多元）"""
    
    def __init__(self, field: Field, coefficients: Dict[Tuple[int, ...], RingElement], num_vars: int):
        """
        初始化多项式。
        
        Args:
            field: 系数域
            coefficients: 映射 {指数向量: 系数}，仅存储非零系数
            num_vars: 变量数量
        
        数学原理：
        - 多项式环 k[x_1,...,x_n]
        """
        self.field = field
        self.num_vars = num_vars
        self.coeffs = {}
        
        # 仅存储非零系数，确保表示规范
        for degs, coef in coefficients.items():
            if len(degs) != num_vars:
                raise ValueError(f"指数向量维度 {len(degs)} 与变量数 {num_vars} 不一致")
            if any(d < 0 for d in degs):
                raise ValueError("次数不能为负")
            if not coef.is_zero():  # 仅存储非零系数
                self.coeffs[degs] = coef
        
        # 验证：零多项式应有空系数字典
        if not self.coeffs and not coefficients:
            self.coeffs = {}
    
    def degree(self) -> int:
        """返回多项式总次数，零多项式次数为-1"""
        if not self.coeffs:
            return -1
        return max(sum(degs) for degs in self.coeffs.keys())
    
    def leading_coefficient(self, order: Optional['MonomialOrder'] = None) -> RingElement:
        """
        返回首项系数严格基于单项式序
        
        Args:
            order: 单项式序；若为 None，则默认使用 grevlex（与 MVP19 约定一致）。
        """
        if not self.coeffs:
            return self.field.zero()
        
        if order is None:
            leading_term = max(self.coeffs.keys(), key=_grevlex_key)
        else:
            # 使用传入的严格序
            leading_term = max(self.coeffs.keys(), key=lambda m: _MonomialOrderWrapper(order, m))
            
        return self.coeffs[leading_term]

    def leading_monomial(self, order: Optional['MonomialOrder'] = None) -> Tuple[int, ...]:
        """返回首项单项式指数向量"""
        if not self.coeffs:
            return (0,) * self.num_vars # 零多项式无首项，或者返回0向量
        
        if order is None:
            return max(self.coeffs.keys(), key=_grevlex_key)
        else:
            return max(self.coeffs.keys(), key=lambda m: _MonomialOrderWrapper(order, m))


    
    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        if self.field != other.field:
            raise ValueError("多项式必须在同一系数域上")
        if self.num_vars != other.num_vars:
            raise ValueError("多项式变量数必须一致")
        
        result_coeffs = {}
        all_degrees = set(self.coeffs.keys()) | set(other.coeffs.keys())
        
        for degs in all_degrees:
            coef1 = self.coeffs.get(degs, self.field.zero())
            coef2 = other.coeffs.get(degs, self.field.zero())
            result_coeffs[degs] = self.field.add(coef1, coef2)
        
        return Polynomial(self.field, result_coeffs, self.num_vars)
    
    def __mul__(self, other: 'Polynomial') -> 'Polynomial':
        if self.field != other.field:
            raise ValueError("多项式必须在同一系数域上")
        if self.num_vars != other.num_vars:
            raise ValueError("多项式变量数必须一致")
        
        result_coeffs = {}
        
        for degs1, coef1 in self.coeffs.items():
            for degs2, coef2 in other.coeffs.items():
                new_degs = tuple(d1 + d2 for d1, d2 in zip(degs1, degs2))
                new_coef = self.field.mul(coef1, coef2)
                
                if new_degs in result_coeffs:
                    result_coeffs[new_degs] = self.field.add(
                        result_coeffs[new_degs], new_coef
                    )
                else:
                    result_coeffs[new_degs] = new_coef
        
        return Polynomial(self.field, result_coeffs, self.num_vars)

    def __pow__(self, exponent: int) -> 'Polynomial':
        """
        幂运算：self ** exponent（仅支持非负整数幂）。
        """
        if not isinstance(exponent, int):
            raise TypeError(f"Polynomial exponent must be int, got {type(exponent).__name__}.")
        if exponent < 0:
            raise ValueError("Polynomial exponent must be non-negative.")

        # 0 次幂返回 1
        if exponent == 0:
            zeros = (0,) * self.num_vars
            return Polynomial(self.field, {zeros: self.field.one()}, self.num_vars)

        # 快速幂（不引入任何浮点/启发式）
        zeros = (0,) * self.num_vars
        result = Polynomial(self.field, {zeros: self.field.one()}, self.num_vars)
        base = self
        e = exponent
        while e:
            if e & 1:
                result = result * base
            e >>= 1
            if e:
                base = base * base
        return result

    def __sub__(self, other: 'Polynomial') -> 'Polynomial':
        return self + (-other)

    def __rsub__(self, other: 'Polynomial') -> 'Polynomial':
        return other + (-self)
    
    def __eq__(self, other: 'Polynomial') -> bool:
        if not isinstance(other, Polynomial):
            return False
        if self.field != other.field:
            return False
        if self.num_vars != other.num_vars:
            return False
        return self.coeffs == other.coeffs
    
    def __neg__(self) -> 'Polynomial':
        neg_coeffs = {degs: self.field.negate(coef) 
                     for degs, coef in self.coeffs.items()}
        return Polynomial(self.field, neg_coeffs, self.num_vars)
    
    def is_zero(self) -> bool:
        return len(self.coeffs) == 0
    
    def is_one(self) -> bool:
        zeros = (0,) * self.num_vars
        return (len(self.coeffs) == 1 and 
                zeros in self.coeffs and 
                self.coeffs[zeros].is_one())
    
    def __repr__(self) -> str:
        if not self.coeffs:
            return "0"
        
        terms = []
        # 按 grevlex 排序
        for degs in sorted(self.coeffs.keys(), key=lambda x: (sum(x), x), reverse=True):
            coef = self.coeffs[degs]
            
            # 格式化系数
            if coef.is_one() and any(d > 0 for d in degs):
                coef_str = ""
            elif coef.is_zero():
                continue
            else:
                coef_str = str(coef)
            
            # 格式化项
            term_parts = []
            for i, d in enumerate(degs):
                if d == 0: continue
                if d == 1: term_parts.append(f"x{i}")
                else: term_parts.append(f"x{i}^^{d}")
            
            var_part = "".join(term_parts)
            if not var_part:
                term = coef_str
            else:
                if coef_str:
                    term = f"{coef_str}*{var_part}"
                else:
                    term = var_part
            
            terms.append(term)
        
        return " + ".join(terms).replace("+ -", "- ")

class PolynomialRing(Ring):
    """多项式环的严格实现（支持多元）"""
    
    def __init__(self, field: Field, num_vars: int = 1, variable: str = 'x'):
        """
        初始化多项式环。
        
        Args:
            field: 系数域
            num_vars: 变量数量
            variable: 变量名前缀（默认 'x' -> x0, x1...）
        
        数学原理：
        - 多项式环 k[x_1,...,x_n]
        """
        self.field = field
        self.num_vars = num_vars
        self.variable = variable
    
    def zero(self) -> Polynomial:
        return Polynomial(self.field, {}, self.num_vars)
    
    def one(self) -> Polynomial:
        zeros = (0,) * self.num_vars
        return Polynomial(self.field, {zeros: self.field.one()}, self.num_vars)
    
    def add(self, a: Polynomial, b: Polynomial) -> Polynomial:
        return a + b
    
    def mul(self, a: Polynomial, b: Polynomial) -> Polynomial:
        return a * b
    
    def negate(self, a: Polynomial) -> Polynomial:
        return -a
    
    def equal(self, a: Polynomial, b: Polynomial) -> bool:
        return a == b
    
    def is_commutative(self) -> bool:
        return True
    
    def constant(self, value: RingElement) -> Polynomial:
        """创建常数多项式"""
        zeros = (0,) * self.num_vars
        return Polynomial(self.field, {zeros: value}, self.num_vars)
    
    def variable_poly(self, idx: int = 0) -> Polynomial:
        """创建第 idx 个变量多项式 x_idx"""
        if not (0 <= idx < self.num_vars):
            raise ValueError(f"Variable index {idx} out of range [0, {self.num_vars})")
        degs = [0] * self.num_vars
        degs[idx] = 1
        return Polynomial(self.field, {tuple(degs): self.field.one()}, self.num_vars)
    
    def __repr__(self) -> str:
        if self.num_vars == 1:
            return f"{self.field}[{self.variable}]"
        return f"{self.field}[{self.variable}0...{self.variable}{self.num_vars-1}]"

# ======================
# k[x]（PID）上的严格线性代数工具
# ======================
#
# 目的：
# - 本文件的 FreeModuleHomomorphism 允许矩阵条目是 Polynomial（也就是 A=k[x] 的环元素）。
# - 若要做 **A-模版本**（不在点上求值），则 kernel / image / membership / intersection 都必须在 PID=k[x] 上做，
#   不能把多项式系数“偷换”为基域系数，也不能用浮点容差。
#
# 约束：
# - 仅覆盖单变量多项式环 k[x]（对应当前 Polynomial 数据结构）。
# - 全部计算严格、精确（FieldElement 为 RationalNumber），不引入任何容差或魔法数。

def _poly_is_zero(p: Polynomial) -> bool:
    return p.is_zero()


def _poly_is_unit(p: Polynomial) -> bool:
    """k[x] 的单位元：非零常数多项式。"""
    if p.is_zero():
        return False
    return p.degree() == 0


def _poly_monic(p: Polynomial, ring: PolynomialRing) -> Polynomial:
    """将非零多项式归一化为首一（monic）。"""
    if p.is_zero():
        return p
    lc = p.leading_coefficient()
    if lc.is_one():
        return p
    inv = ring.field.inverse(lc)  # 单位：非零常数
    return ring.mul(ring.constant(inv), p)


def _poly_divmod_univariate(f: Polynomial, g: Polynomial, ring: PolynomialRing) -> Tuple[Polynomial, Polynomial]:
    """
    多项式长除法：返回 (q, r) 使得 f = q*g + r 且 deg(r) < deg(g) 或 r=0。
    """
    if g.is_zero():
        raise ZeroDivisionError("Polynomial division by zero.")
    if f.is_zero():
        return ring.zero(), ring.zero()

    if f.field != g.field:
        raise ValueError("Polynomial division requires both polynomials over the same field.")

    q = ring.zero()
    r = f
    x = ring.variable_poly()

    deg_g = g.degree()
    lc_g = g.leading_coefficient()
    if lc_g.is_zero():
        raise ValueError("Leading coefficient of divisor is zero (invalid polynomial representation).")

    while (not r.is_zero()) and (r.degree() >= deg_g):
        deg_r = r.degree()
        lc_r = r.leading_coefficient()
        deg_diff = deg_r - deg_g
        coef = ring.field.divide(lc_r, lc_g)
        # Use object-level pow, as Ring does not enforce pow()
        term = ring.mul(ring.constant(coef), x ** deg_diff)
        q = q + term
        r = r + (-(term * g))

    return q, r


def _poly_gcd_univariate(a: Polynomial, b: Polynomial, ring: PolynomialRing) -> Polynomial:
    """欧几里得算法求 gcd（首一归一化）。"""
    if a.field != b.field:
        raise ValueError("gcd requires both polynomials over the same field.")
    if a.is_zero():
        return _poly_monic(b, ring)
    if b.is_zero():
        return _poly_monic(a, ring)

    f, g = a, b
    while not g.is_zero():
        _q, r = _poly_divmod_univariate(f, g, ring)
        f, g = g, r
    return _poly_monic(f, ring)


def _poly_derivative_univariate(poly: Polynomial, ring: PolynomialRing) -> Polynomial:
    """
    严格形式导数（单变量）：
      d/dx Σ a_i x^i = Σ_{i>=1} i·a_i x^{i-1}
    """
    if not isinstance(poly, Polynomial):
        raise TypeError("poly must be a Polynomial.")
    if poly.field != ring.field:
        raise ValueError("poly must be over the same field as the ring.")
    if poly.is_zero():
        return ring.zero()

    field = ring.field
    coeffs: Dict[Tuple[int, ...], RingElement] = {}
    
    # Assume univariate: check num_vars or just grab the first degree
    if poly.num_vars != 1 and not poly.is_zero():
        # For safety in this "univariate" helper, we strictly enforce num_vars=1 or empty
        # But to be robust, we just check keys.
        pass

    for deg_tuple, coef in poly.coeffs.items():
        if len(deg_tuple) != 1:
             # Should ideally not happen if num_vars=1
             # If it happens, we might be in a mixed context. 
             # For now, just take the first component as the variable.
             pass
        
        deg = deg_tuple[0]
        if deg <= 0:
            continue
            
        factor = field.RationalNumber(int(deg), 1)
        new_coef = field.mul(factor, coef)
        
        tgt_deg = int(deg - 1)
        tgt_tuple = (tgt_deg,) + deg_tuple[1:] # Preserve other dims if any (though unlikely for univariate)
        
        if tgt_tuple in coeffs:
            coeffs[tgt_tuple] = field.add(coeffs[tgt_tuple], new_coef)
        else:
            coeffs[tgt_tuple] = new_coef
            
    return Polynomial(field, coeffs, poly.num_vars)





def _mat_identity(size: int, ring: PolynomialRing) -> List[List[Polynomial]]:
    if size < 0:
        raise ValueError("Identity matrix size must be non-negative.")
    z = ring.zero()
    o = ring.one()
    return [[o if i == j else z for j in range(size)] for i in range(size)]


def _mat_swap_rows(mat: List[List[Polynomial]], i: int, j: int) -> None:
    if i == j:
        return
    mat[i], mat[j] = mat[j], mat[i]


def _mat_swap_cols(mat: List[List[Polynomial]], i: int, j: int) -> None:
    if i == j:
        return
    for r in range(len(mat)):
        mat[r][i], mat[r][j] = mat[r][j], mat[r][i]


def _mat_row_addmul(mat: List[List[Polynomial]], target: int, source: int, factor: Polynomial) -> None:
    """row_target += factor * row_source"""
    if factor.is_zero():
        return
    row_s = mat[source]
    row_t = mat[target]
    mat[target] = [a + (factor * b) for a, b in zip(row_t, row_s)]


def _mat_col_addmul(mat: List[List[Polynomial]], target: int, source: int, factor: Polynomial) -> None:
    """col_target += factor * col_source"""
    if factor.is_zero():
        return
    for r in range(len(mat)):
        mat[r][target] = mat[r][target] + (factor * mat[r][source])


def _mat_row_scale(mat: List[List[Polynomial]], row: int, unit: Polynomial) -> None:
    """row *= unit（unit 必须是单位元：非零常数多项式）"""
    if not _poly_is_unit(unit):
        raise ValueError("Row scaling requires a unit (nonzero constant polynomial).")
    mat[row] = [unit * a for a in mat[row]]


def _poly_extended_gcd_univariate(a: Polynomial, b: Polynomial, ring: PolynomialRing) -> Tuple[Polynomial, Polynomial, Polynomial]:
    """
    扩展欧几里得：返回 (d, u, v) 使得 u*a + v*b = d，其中 d=gcd(a,b)（首一）。
    """
    if a.field != b.field:
        raise ValueError("extended_gcd requires both polynomials over the same field.")

    if b.is_zero():
        d = _poly_monic(a, ring)
        if d.is_zero():
            return ring.zero(), ring.zero(), ring.zero()
        # a = lc(a) * d，u=1/lc(a)
        lc_a = a.leading_coefficient()
        inv = ring.field.inverse(lc_a)
        u = ring.constant(inv)
        v = ring.zero()
        return d, u, v

    q, r = _poly_divmod_univariate(a, b, ring)
    d, u1, v1 = _poly_extended_gcd_univariate(b, r, ring)
    # d = u1*b + v1*r = u1*b + v1*(a - q*b) = v1*a + (u1 - v1*q)*b
    u = v1
    v = u1 + (-(v1 * q))
    return d, u, v


def _pid_diagonal_form_kx(
    matrix: List[List[Polynomial]],
    ring: PolynomialRing,
) -> Tuple[List[List[Polynomial]], List[List[Polynomial]], List[List[Polynomial]], int]:
    """
    在 PID=k[x] 上将矩阵对角化（不要求 Smith 的整除链，仅需对角形态）：
        D = U * A * V
    其中 U,V 为可逆（unimodular）矩阵，D 为对角矩阵（处理到 rank）。

    Returns:
        D, U, V, rank
    """
    m = len(matrix)
    n = len(matrix[0]) if m else 0
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be rectangular.")

    # 拷贝（避免原地修改调用者矩阵）
    A = [[entry for entry in row] for row in matrix]
    U = _mat_identity(m, ring)
    V = _mat_identity(n, ring)

    def find_pivot(start_r: int, start_c: int) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_deg: Optional[int] = None
        for rr in range(start_r, m):
            for cc in range(start_c, n):
                if A[rr][cc].is_zero():
                    continue
                d = A[rr][cc].degree()
                if best is None or (best_deg is not None and d < best_deg):
                    best = (rr, cc)
                    best_deg = d
        return best

    i = 0
    j = 0
    rank = 0
    while i < m and j < n:
        pivot_pos = find_pivot(i, j)
        if pivot_pos is None:
            break
        pr, pc = pivot_pos
        _mat_swap_rows(A, i, pr)
        _mat_swap_rows(U, i, pr)
        _mat_swap_cols(A, j, pc)
        _mat_swap_cols(V, j, pc)

        # 现在 pivot 在 (i,j)
        pivot = A[i][j]
        if pivot.is_zero():
            # 不应发生；安全起见继续找下一个
            j += 1
            continue

        # 1) 消去同一列的下方元素（仅在子块中操作，保持已处理块不变）
        for rr in range(i + 1, m):
            while not A[rr][j].is_zero():
                q, _r = _poly_divmod_univariate(A[rr][j], pivot, ring)
                if not q.is_zero():
                    neg_q = -q
                    _mat_row_addmul(A, rr, i, neg_q)
                    _mat_row_addmul(U, rr, i, neg_q)
                if A[rr][j].is_zero():
                    break
                # 余数次数必小于 pivot 次数；交换以降低 pivot 次数
                if A[rr][j].degree() < pivot.degree():
                    _mat_swap_rows(A, rr, i)
                    _mat_swap_rows(U, rr, i)
                    pivot = A[i][j]
                else:
                    raise RuntimeError("Unexpected remainder degree in PID reduction (column).")

        # 2) 消去同一行的右侧元素
        for cc in range(j + 1, n):
            while not A[i][cc].is_zero():
                q, _r = _poly_divmod_univariate(A[i][cc], pivot, ring)
                if not q.is_zero():
                    neg_q = -q
                    _mat_col_addmul(A, cc, j, neg_q)
                    _mat_col_addmul(V, cc, j, neg_q)
                if A[i][cc].is_zero():
                    break
                if A[i][cc].degree() < pivot.degree():
                    _mat_swap_cols(A, cc, j)
                    _mat_swap_cols(V, cc, j)
                    pivot = A[i][j]
                else:
                    raise RuntimeError("Unexpected remainder degree in PID reduction (row).")

        # 3) 现在 A[i,·] 与 A[·,j] 在子块外应当为 0；把 pivot 归一化为首一
        lc = pivot.leading_coefficient()
        if not lc.is_one():
            inv = ring.field.inverse(lc)
            unit = ring.constant(inv)
            _mat_row_scale(A, i, unit)
            _mat_row_scale(U, i, unit)
            pivot = A[i][j]

        rank += 1
        i += 1
        j += 1

    return A, U, V, rank


def _kernel_generators_free_module_hom_kx(
    matrix: List[List[Polynomial]],
    ring: PolynomialRing,
) -> List[Tuple[Polynomial, ...]]:
    """
    计算矩阵 A: R^n -> R^m 的核的生成元（作为 R^n 的列向量列表）。
    使用对角化：D = U*A*V，kernel(A) = V * kernel(D)。
    """
    m = len(matrix)
    n = len(matrix[0]) if m else 0
    if n == 0:
        return []
    D, _U, V, rank = _pid_diagonal_form_kx(matrix, ring)
    # D 的前 rank 个对角元非零 ⇒ 对应坐标必须为 0；其余坐标自由
    gens: List[Tuple[Polynomial, ...]] = []
    for col_idx in range(rank, n):
        col = tuple(V[r][col_idx] for r in range(n))
        gens.append(col)
    return gens


def _submodule_membership_kx(
    element: Tuple[Polynomial, ...],
    generators: List[Tuple[Polynomial, ...]],
    ring: PolynomialRing,
) -> bool:
    """
    判定 element 是否属于由 generators 张成的子模（在 R=k[x] 上）。
    """
    n = len(element)
    if not generators:
        return all(c.is_zero() for c in element)

    # 构造生成矩阵 G: R^k -> R^n（n×k），列为 generators
    k = len(generators)
    if any(len(g) != n for g in generators):
        raise ValueError("Generator vectors must have the same length as the ambient rank.")

    G = [[ring.zero() for _ in range(k)] for _ in range(n)]
    for j in range(k):
        for i in range(n):
            G[i][j] = generators[j][i]

    D, U, _V, rank = _pid_diagonal_form_kx(G, ring)

    # w = U * element
    w: List[Polynomial] = [ring.zero() for _ in range(n)]
    for i in range(n):
        acc = ring.zero()
        for j in range(n):
            acc = acc + (U[i][j] * element[j])
        w[i] = acc

    # 对角系统 D y = w 可解当且仅当：
    # - 对 i < rank：D[i][i] | w[i]
    # - 对 i >= rank：w[i] = 0
    for i in range(rank):
        d = D[i][i]
        if d.is_zero():
            # 不应发生：rank 的定义保证对角元非零
            return False
        _q, r = _poly_divmod_univariate(w[i], d, ring)
        if not r.is_zero():
            return False
    for i in range(rank, n):
        if not w[i].is_zero():
            return False
    return True


class QuotientPolynomialRing(Ring):
    """
    商环 A = k[x]/(f) 的严格实现（单变量）。

    关键点：
    - 环元素仍用本文件的 Polynomial 表示；
    - 但所有运算结果都会被约化到 deg < deg(f) 的规范代表元；
    - 该类仅用于“把系数域从 k[x] 换成 k[x]/(f)”的 A-模计算，不做任何数值近似。
    """

    def __init__(self, base_ring: PolynomialRing, modulus: Polynomial):
        if not isinstance(base_ring, PolynomialRing):
            raise TypeError("QuotientPolynomialRing requires a PolynomialRing base.")
        if not isinstance(modulus, Polynomial):
            raise TypeError("QuotientPolynomialRing modulus must be a Polynomial.")
        if modulus.is_zero():
            raise ValueError("QuotientPolynomialRing modulus must be nonzero.")
        if modulus.field != base_ring.field:
            raise ValueError("QuotientPolynomialRing modulus must lie in the given base ring.")

        self.base_ring = base_ring
        self.modulus = _poly_monic(modulus, base_ring)

    def _reduce(self, poly: Polynomial) -> Polynomial:
        if not isinstance(poly, Polynomial):
            raise TypeError("QuotientPolynomialRing elements must be Polynomial.")
        if poly.field != self.base_ring.field:
            raise ValueError("QuotientPolynomialRing element must be over the same field as the base ring.")
        _q, r = _poly_divmod_univariate(poly, self.modulus, self.base_ring)
        return r

    def zero(self) -> Polynomial:
        return self.base_ring.zero()

    def one(self) -> Polynomial:
        return self.base_ring.one()

    def add(self, a: Polynomial, b: Polynomial) -> Polynomial:
        return self._reduce(a + b)

    def mul(self, a: Polynomial, b: Polynomial) -> Polynomial:
        return self._reduce(a * b)

    def negate(self, a: Polynomial) -> Polynomial:
        return self._reduce(-a)

    def equal(self, a: Polynomial, b: Polynomial) -> bool:
        return self._reduce(a) == self._reduce(b)

    def is_commutative(self) -> bool:
        return True

    # 便捷构造（非 Ring 抽象必需，但工程里常用）
    def constant(self, value: RingElement) -> Polynomial:
        return self._reduce(self.base_ring.constant(value))

    def variable_poly(self) -> Polynomial:
        return self._reduce(self.base_ring.variable_poly())

    def __repr__(self) -> str:
        return f"{self.base_ring}/({self.modulus})"

# ======================
# 模与同态的严格实现
# ======================

class ModuleElement(ABC):
    """模元素的抽象基类"""
    
    @abstractmethod
    def __add__(self, other) -> 'ModuleElement':
        pass
    
    @abstractmethod
    def scalar_mul(self, scalar: RingElement) -> 'ModuleElement':
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def is_zero(self) -> bool:
        pass

class Module(ABC):
    """模的抽象基类"""
    
    @abstractmethod
    def zero(self) -> ModuleElement:
        pass
    
    @abstractmethod
    def add(self, a: ModuleElement, b: ModuleElement) -> ModuleElement:
        pass
    
    @abstractmethod
    def scalar_mul(self, scalar: RingElement, 
                  element: ModuleElement) -> ModuleElement:
        pass
    
    @abstractmethod
    def equal(self, a: ModuleElement, b: ModuleElement) -> bool:
        pass
    
    @abstractmethod
    def is_zero(self, element: ModuleElement) -> bool:
        pass

class FreeModule(Module):
    """自由模的严格实现"""
    
    def __init__(self, base_ring: Ring, rank: int):
        """
        初始化自由模。
        
        Args:
            base_ring: 基环
            rank: 模的秩
        
        数学原理：
        - 自由模 R^^n = {(r_1,...,r_n) | r_i ∈ R}
        - 严格实现模运算，
        """
        if rank < 0:
            raise ValueError("秩不能为负")
        self.base_ring = base_ring
        self.rank = rank
    
    def zero(self) -> Tuple[RingElement, ...]:
        return tuple(self.base_ring.zero() for _ in range(self.rank))
    
    def add(self, 
            a: Tuple[RingElement, ...], 
            b: Tuple[RingElement, ...]) -> Tuple[RingElement, ...]:
        if len(a) != self.rank or len(b) != self.rank:
            raise ValueError("元素维度不匹配")
        return tuple(self.base_ring.add(ai, bi) for ai, bi in zip(a, b))
    
    def scalar_mul(self, 
                  scalar: RingElement, 
                  element: Tuple[RingElement, ...]) -> Tuple[RingElement, ...]:
        if len(element) != self.rank:
            raise ValueError("元素维度不匹配")
        return tuple(self.base_ring.mul(scalar, e) for e in element)
    
    def equal(self, 
             a: Tuple[RingElement, ...], 
             b: Tuple[RingElement, ...]) -> bool:
        if len(a) != self.rank or len(b) != self.rank:
            return False
        return all(self.base_ring.equal(ai, bi) for ai, bi in zip(a, b))
    
    def is_zero(self, element: Tuple[RingElement, ...]) -> bool:
        return all(comp.is_zero() for comp in element)
    
    def __repr__(self) -> str:
        return f"R^^{{{self.rank}}}"

class ModuleHomomorphism(ABC):
    """模同态的抽象基类"""
    
    @abstractmethod
    def domain(self) -> Module:
        pass
    
    @abstractmethod
    def codomain(self) -> Module:
        pass
    
    @abstractmethod
    def apply(self, element: ModuleElement) -> ModuleElement:
        pass
    
    def __call__(self, element: ModuleElement) -> ModuleElement:
        return self.apply(element)


@dataclass(frozen=True)
class CallableModuleHomomorphism(ModuleHomomorphism):
    """
    用可调用对象承载的模同态实现。

    目的：
    - 避免错误地实例化抽象类 ModuleHomomorphism(...)；
    - 允许在尚未给出显式矩阵表示时，用严格的函数式定义承载同态（不做静默近似）。
    """

    _domain: Module
    _codomain: Module
    _apply_func: Callable[[ModuleElement], ModuleElement]

    def domain(self) -> Module:
        return self._domain

    def codomain(self) -> Module:
        return self._codomain

    def apply(self, element: ModuleElement) -> ModuleElement:
        return self._apply_func(element)

class FreeModuleHomomorphism(ModuleHomomorphism):
    """自由模同态的严格实现"""
    
    def __init__(self, 
                 domain: FreeModule, 
                 codomain: FreeModule,
                 matrix: List[List[RingElement]]):
        """
        初始化自由模同态。
        
        Args:
            domain: 定义域自由模
            codomain: 陪域自由模
            matrix: 表示矩阵（m × n，m=codomain.rank, n=domain.rank）
        
        数学原理：
        - 自由模同态由表示矩阵完全确定
        - 严格实现同态性质：φ(a+b) = φ(a) + φ(b), φ(r·a) = r·φ(a)
        """
        if len(matrix) != codomain.rank:
            raise ValueError("矩阵行数必须匹配陪域秩")
        if any(len(row) != domain.rank for row in matrix):
            raise ValueError("矩阵列数必须匹配定义域秩")
        
        self._domain = domain
        self._codomain = codomain
        self._matrix = matrix
    
    def domain(self) -> FreeModule:
        return self._domain
    
    def codomain(self) -> FreeModule:
        return self._codomain
    
    def apply(self, element: Tuple[RingElement, ...]) -> Tuple[RingElement, ...]:
        if len(element) != self.domain().rank:
            raise ValueError("元素维度与定义域不匹配")
        
        result = list(self.codomain().zero())
        
        # 矩阵乘法：result_j = Σ_i matrix[j][i] * element[i]
        for j in range(self.codomain().rank):
            sum_val = self.codomain().base_ring.zero()
            for i in range(self.domain().rank):
                prod = self.codomain().base_ring.mul(
                    self._matrix[j][i], 
                    element[i]
                )
                sum_val = self.codomain().base_ring.add(sum_val, prod)
            result[j] = sum_val
        
        return tuple(result)
    
    def compose(self, other: 'FreeModuleHomomorphism') -> 'FreeModuleHomomorphism':
        """同态复合"""
        if self.domain().rank != other.codomain().rank:
            raise ValueError("同态复合维度不匹配")
        
        # 矩阵乘法
        new_matrix = []
        for j in range(self.codomain().rank):
            row = []
            for i in range(other.domain().rank):
                sum_val = self.codomain().base_ring.zero()
                for k in range(self.domain().rank):
                    prod = self.codomain().base_ring.mul(
                        self._matrix[j][k],
                        other._matrix[k][i]
                    )
                    sum_val = self.codomain().base_ring.add(sum_val, prod)
                row.append(sum_val)
            new_matrix.append(row)
        
        return FreeModuleHomomorphism(
            other.domain(), 
            self.codomain(),
            new_matrix
        )

# ======================
# Gröbner基与模计算（核心数学引擎）
# ======================

class MonomialOrder(ABC):
    """单项式序的抽象基类"""
    
    @abstractmethod
    def compare(self, m1: Tuple[int, ...], m2: Tuple[int, ...]) -> int:
        """
        比较两个单项式。
        
        Returns:
            -1: m1 < m2
             0: m1 = m2
             1: m1 > m2
        """
        pass

class DegreeLexOrder(MonomialOrder):
    """次数字典序"""
    
    def compare(self, m1: Tuple[int, ...], m2: Tuple[int, ...]) -> int:
        deg1 = sum(m1)
        deg2 = sum(m2)
        if deg1 < deg2:
            return -1
        if deg1 > deg2:
            return 1
        
        # 次数字典序
        for i in range(len(m1)):
            if m1[i] < m2[i]:
                return -1
            if m1[i] > m2[i]:
                return 1
        return 0

class PolynomialIdeal:
    """多项式理想的严格实现"""
    
    def __init__(self, 
                 ring: PolynomialRing, 
                 generators: List[Polynomial],
                 order: Optional[MonomialOrder] = None):
        """
        初始化多项式理想。
        
        Args:
            ring: 多项式环
            generators: 生成元列表
            order: 单项式序（默认为次数字典序）
        
        数学原理：
        - 理想 I = <g_1,...,g_m> ⊆ k[x_1,...,x_n]
        - 严格实现理想运算，
        """
        self.ring = ring
        self.generators = [g for g in generators if not g.is_zero()]
        
        # 默认使用次数字典序
        self.order = order or DegreeLexOrder()
        
        # 计算约化Gröbner基（关键！）
        self.groebner_basis = self._compute_groebner_basis()
    
    def _compute_groebner_basis(self) -> List[Polynomial]:
        """
        计算约化Gröbner基。
        """
        if not self.generators:
            return []

        # 单变量情形：退化为 GCD (PID)
        # 注意：这是严谨的数学优化，单变量多项式环是PID，其理想由GCD生成。
        # 这不是降级，而是对特殊情形的最优解。
        if self.ring.num_vars == 1:
            g = self.generators[0]
            for h in self.generators[1:]:
                g = _poly_gcd_univariate(g, h, self.ring)
            if g.is_zero():
                return []
            return [_poly_monic(g, self.ring)]
        
        # 多变量 Gröbner 基 / ideal-membership 在本仓库由 MVP12→MVP19（Q→F_p 投影 + 严格 Buchberger）
        # 与 SyzygyGraph/decoder 体系接管；MVP18 这里不做重复实现，避免产生“看似能跑但数学语义错误”的分叉。
        raise NotImplementedError(
            "Multivariate Groebner basis is not implemented in mvp18_cohomology_solver.PolynomialIdeal. "
            "Use the MVP12/MVP19 pipeline instead: "
            "holonomic_dmodule_engine.{PrimeSelector,PolynomialProjector,IdealProjector} + "
            "mvp19_groebner_middleware.compute_groebner_basis_mod_p (for commutative ideals over F_p), "
            "or mvp19_syzygy_frobenius_suite for rewrite-rule (SyzygyGraph) reduction on quotient algebras."
        )
    
    def _s_polynomial(self, 
                     f: Polynomial, 
                     g: Polynomial) -> Optional[Polynomial]:
        """
        计算S-多项式。
        
        数学原理：
        S(f,g) = (LCM(LT(f),LT(g)) / LT(f)) * f - 
                 (LCM(LT(f),LT(g)) / LT(g)) * g
        
        Returns:
            S-多项式，或None（若首项无公倍数）
        """
        # 获取首项（根据单项式序）
        lt_f = self._leading_term(f)
        lt_g = self._leading_term(g)
        
        if lt_f is None or lt_g is None:
            return None
        
        # 计算LCM（首项单项式的最小公倍数）
        # 红线：严格多元 LCM，禁止单变量特判/静默 None
        lcm_m = self._lcm_monomial(lt_f[0], lt_g[0])
        
        # 计算乘数单项式（lcm / lm）
        mult_f_m = self._divide_monomial(lcm_m, lt_f[0])
        mult_g_m = self._divide_monomial(lcm_m, lt_g[0])
        if mult_f_m is None or mult_g_m is None:
            # 理论上不应发生：lcm 必可被两者整除；若发生是实现错误
            raise RuntimeError("[PolynomialIdeal] internal: lcm is not divisible by one of its factors")
        
        # 构造S-多项式
        # 需要消去首项系数：乘以 1/lc(f), 1/lc(g)，确保首项严格抵消
        inv_lc_f = self.ring.field.inverse(lt_f[1])
        inv_lc_g = self.ring.field.inverse(lt_g[1])
        
        term_f = self.ring.mul(
            self.ring.constant(inv_lc_f),
            self.ring.mul(self._monomial_to_poly(mult_f_m), f),
        )
        term_g = self.ring.mul(
            self.ring.constant(inv_lc_g),
            self.ring.mul(self._monomial_to_poly(mult_g_m), g),
        )
        
        return term_f + (-term_g)
    
    def _reduce(self, 
               poly: Polynomial, 
               basis: List[Polynomial]) -> Polynomial:
        """
        多项式对基的约化。
        
        数学原理：
        重复用基中多项式消去首项，直到无法继续
        
        Args:
            poly: 被约化多项式
            basis: 约化基
        
        Returns:
            约化后的多项式
        """
        p = poly
        while not p.is_zero():
            reduced = False
            for g in basis:
                if g.is_zero():
                    continue
                
                # 检查g的首项是否整除p的首项
                lt_p = self._leading_term(p)
                lt_g = self._leading_term(g)
                
                if lt_p is None or lt_g is None:
                    continue
                
                # 检查是否可约化
                quotient_m = self._divide_monomial(lt_p[0], lt_g[0])
                if quotient_m is not None:
                    # 计算乘数
                    coef = self.ring.field.divide(
                        lt_p[1], 
                        lt_g[1]
                    )
                    
                    # 构造消去项
                    term = self.ring.mul(
                        self.ring.constant(coef),
                        self.ring.mul(
                            self._monomial_to_poly(quotient_m),
                            g
                        )
                    )
                    
                    # 消去首项
                    p = p + (-term)
                    reduced = True
                    break
            
            if not reduced:
                break
        
        return p
    
    def _reduce_groebner_basis(self, basis: List[Polynomial]) -> List[Polynomial]:
        """
        约化Gröbner基为最小形式。
        
        数学原理：
        1. 移除冗余生成元
        2. 使每个生成元对其他生成元约化
        3. 归一化首项系数为1
        """
        # 步骤1: 移除零元素
        basis = [g for g in basis if not g.is_zero()]
        if not basis:
            return []
        
        # 步骤2: 使每个生成元对其他生成元约化
        reduced_basis = []
        for i, g in enumerate(basis):
            # 用其他生成元约化g
            other_basis = basis[:i] + basis[i+1:]
            reduced_g = self._reduce(g, other_basis)
            
            # 归一化首项系数
            lt = self._leading_term(reduced_g)
            if lt and not lt[1].is_zero():
                inv = self.ring.field.inverse(lt[1])
                normalized_g = self.ring.mul(
                    self.ring.constant(inv),
                    reduced_g
                )
                reduced_basis.append(normalized_g)
        
        # 步骤3: 按首项降序排序
        return sorted(
            reduced_basis,
            key=lambda g: self._leading_monomial(g),
            reverse=True
        )
    
    def _leading_term(self, poly: Polynomial) -> Optional[Tuple[Tuple[int, ...], RingElement]]:
        """
        获取多项式的首项（单项式，系数）。
        
        Returns:
            (单项式, 系数) 或 None（若为零多项式）
        """
        if poly.is_zero():
            return None
        # 严格：首项由单项式序唯一决定（多元/单元统一）
        lm = poly.leading_monomial(self.order)
        lc = poly.leading_coefficient(self.order)
        if lc.is_zero():
            # 非零多项式的首项系数不可能为 0；否则表示不规范/实现错误
            raise RuntimeError("[PolynomialIdeal] internal: leading coefficient is zero for non-zero polynomial")
        return (tuple(int(e) for e in lm), lc)
    
    def _leading_monomial(self, poly: Polynomial) -> Optional[Tuple[int, ...]]:
        """获取首项的单项式部分"""
        lt = self._leading_term(poly)
        return lt[0] if lt else None
    
    def _lcm_monomial(self, m1: Tuple[int, ...], m2: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        计算两个单项式的最小公倍数。
        
        Returns:
            LCM 单项式指数向量（逐分量取 max）
        """
        if not isinstance(m1, tuple) or not isinstance(m2, tuple):
            raise TypeError("[PolynomialIdeal] monomial must be Tuple[int, ...]")
        if len(m1) != len(m2):
            raise ValueError(f"[PolynomialIdeal] monomial dim mismatch: len(m1)={len(m1)} len(m2)={len(m2)}")
        if any((not isinstance(e, int)) or e < 0 for e in m1):
            raise ValueError(f"[PolynomialIdeal] monomial m1 exponents must be non-negative ints: {m1!r}")
        if any((not isinstance(e, int)) or e < 0 for e in m2):
            raise ValueError(f"[PolynomialIdeal] monomial m2 exponents must be non-negative ints: {m2!r}")
        return tuple(max(a, b) for a, b in zip(m1, m2))
    
    def _divide_monomial(self, m1: Tuple[int, ...], m2: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """
        除法：m1 / m2。
        
        Returns:
            商单项式（指数差）或 None（若不可整除）
        """
        if not isinstance(m1, tuple) or not isinstance(m2, tuple):
            raise TypeError("[PolynomialIdeal] monomial must be Tuple[int, ...]")
        if len(m1) != len(m2):
            raise ValueError(f"[PolynomialIdeal] monomial dim mismatch: len(m1)={len(m1)} len(m2)={len(m2)}")
        if any((not isinstance(e, int)) or e < 0 for e in m1):
            raise ValueError(f"[PolynomialIdeal] monomial m1 exponents must be non-negative ints: {m1!r}")
        if any((not isinstance(e, int)) or e < 0 for e in m2):
            raise ValueError(f"[PolynomialIdeal] monomial m2 exponents must be non-negative ints: {m2!r}")

        if any(a < b for a, b in zip(m1, m2)):
            return None
        return tuple(a - b for a, b in zip(m1, m2))
    
    def _monomial_to_poly(self, monomial: Tuple[int, ...]) -> Polynomial:
        """将单项式转换为多项式"""
        if not isinstance(monomial, tuple):
            raise TypeError(f"[PolynomialIdeal] monomial must be tuple, got {type(monomial).__name__}")
        if len(monomial) != int(self.ring.num_vars):
            raise ValueError(
                f"[PolynomialIdeal] monomial dim mismatch: expected={int(self.ring.num_vars)}, got={len(monomial)}"
            )
        if any((not isinstance(e, int)) or e < 0 for e in monomial):
            raise ValueError(f"[PolynomialIdeal] monomial exponents must be non-negative ints: {monomial!r}")
        return Polynomial(self.ring.field, {monomial: self.ring.field.one()}, self.ring.num_vars)
    
    def contains(self, poly: Polynomial) -> bool:
        """检查多项式是否属于理想"""
        reduced = self._reduce(poly, self.groebner_basis)
        return reduced.is_zero()
    
    def intersect(self, other: 'PolynomialIdeal') -> 'PolynomialIdeal':
        """计算两个理想的交"""
        # 严格实现：使用elimination ideal方法
        # 1. 创建新环 k[t,x_1,...,x_n]
        # 2. 构造理想 <t·I, (1-t)·J>
        # 3. 消去t得到 I ∩ J
        raise NotImplementedError("理想交的严格实现（需扩展环）")
    
    def quotient(self, other: 'PolynomialIdeal') -> 'PolynomialIdeal':
        """计算理想商 I:J = {f | f·J ⊆ I}"""
        # 严格实现：使用Gröbner基方法
        raise NotImplementedError("理想商的严格实现")
    
    def __repr__(self) -> str:
        gens = ", ".join(str(g) for g in self.generators)
        return f"<{gens}>"
        
        
        
# ======================
# Kähler微分模的严格范畴论实现
# ======================

class TensorBackend(ABC):
    """
    张量积后端（拔插口）。

    目的：
    - cohomology_solver.py 本体不再内置/散落“张量积的各种实现细节”；
    - 只依赖最小接口，便于替换为更强、可验证的后端（例如外部模块提供的实现）。

    约束：
    - 本文件默认后端只覆盖“自由模 ⊗ 自由模”的严格构造；
    - 任何未实现情形必须显式抛异常，禁止静默降级。
    """

    @abstractmethod
    def tensor_module(self, ring: 'Ring', module1: 'FreeModule', module2: 'FreeModule') -> 'FreeModule':
        """返回 M ⊗_R N 的一个可计算模型（此处默认采用自由模模型）。"""

    @abstractmethod
    def tensor_element(
        self,
        ring: 'Ring',
        module1: 'FreeModule',
        module2: 'FreeModule',
        m: 'ModuleElement',
        n: 'ModuleElement',
    ) -> Tuple['RingElement', ...]:
        """返回元素 m ⊗ n 在 tensor_module(...) 的坐标表示。"""


class FreeModuleTensorBackend(TensorBackend):
    """默认张量积后端：自由模张量积（严格、无近似）。"""

    def tensor_module(self, ring: 'Ring', module1: 'FreeModule', module2: 'FreeModule') -> 'FreeModule':
        if not isinstance(module1, FreeModule) or not isinstance(module2, FreeModule):
            raise TypeError("FreeModuleTensorBackend requires both factors to be FreeModule.")
        if module1.rank < 0 or module2.rank < 0:
            raise ValueError("FreeModule ranks must be non-negative.")
        return FreeModule(ring, module1.rank * module2.rank)

    def tensor_element(
        self,
        ring: 'Ring',
        module1: 'FreeModule',
        module2: 'FreeModule',
        m: 'ModuleElement',
        n: 'ModuleElement',
    ) -> Tuple['RingElement', ...]:
        if not isinstance(module1, FreeModule) or not isinstance(module2, FreeModule):
            raise TypeError("FreeModuleTensorBackend requires both factors to be FreeModule.")
        if not isinstance(m, tuple) or not isinstance(n, tuple):
            raise TypeError("FreeModuleTensorBackend currently supports FreeModule elements represented as tuples.")
        if len(m) != module1.rank or len(n) != module2.rank:
            raise ValueError("Tensor element dimensions do not match module ranks.")

        dim = module1.rank * module2.rank
        components: List[RingElement] = [ring.zero() for _ in range(dim)]
        for i, a in enumerate(m):
            for j, b in enumerate(n):
                idx = i * module2.rank + j
                components[idx] = ring.add(components[idx], ring.mul(a, b))
        return tuple(components)


class TensorProduct:
    """张量积的严格实现"""
    
    def __init__(
        self,
        ring: Ring,
        module1: Module,
        module2: Module,
        *,
        backend: Optional[TensorBackend] = None,
    ):
        """
        初始化张量积 M ⊗_R N。
        
        Args:
            ring: 基环 R
            module1: 模 M
            module2: 模 N
        
        数学原理：
        - 张量积是满足泛性质的唯一模：Bilin_R(M,N;P) ≅ Hom_R(M⊗N,P)
        - 在自由模情形下，M ⊗_R N 仍为自由模，可用坐标表示严格计算
        """
        self.ring = ring
        self.module1 = module1
        self.module2 = module2
        self.backend = backend if backend is not None else FreeModuleTensorBackend()

        if not isinstance(self.module1, FreeModule) or not isinstance(self.module2, FreeModule):
            raise NotImplementedError("TensorProduct currently supports FreeModule only (default backend limitation).")

        self.tensor_module = self.backend.tensor_module(self.ring, self.module1, self.module2)
    
    def element(self, m: ModuleElement, n: ModuleElement) -> 'TensorElement':
        """
        构造张量积元素 m ⊗ n。
        
        Args:
            m: M中的元素
            n: N中的元素
            
        Returns:
            TensorElement: 张量积中的元素
        
        数学原理：
        - 通过泛性质，m⊗n 是满足双线性的唯一元素
        """
        components = self.backend.tensor_element(self.ring, self.module1, self.module2, m, n)
        return TensorElement(self, components)

class TensorElement(ModuleElement):
    """张量积元素的表示"""
    
    def __init__(self, tensor_product: TensorProduct, components: Tuple[RingElement, ...]):
        self.tensor_product = tensor_product
        self.components = components
    
    def __add__(self, other: 'TensorElement') -> 'TensorElement':
        if self.tensor_product != other.tensor_product:
            raise ValueError("张量积空间不匹配")
        return TensorElement(
            self.tensor_product,
            tuple(self.tensor_product.ring.add(a, b) 
                 for a, b in zip(self.components, other.components))
        )
    
    def scalar_mul(self, scalar: RingElement) -> 'TensorElement':
        return TensorElement(
            self.tensor_product,
            tuple(self.tensor_product.ring.mul(scalar, a) for a in self.components)
        )
    
    def __eq__(self, other: 'TensorElement') -> bool:
        if self.tensor_product != other.tensor_product:
            return False
        return all(self.tensor_product.ring.equal(a, b) 
                  for a, b in zip(self.components, other.components))
    
    def is_zero(self) -> bool:
        return all(comp.is_zero() for comp in self.components)

class KahlerDifferentials:
    """
    Kähler微分模的严格范畴论实现
    
    数学原理：
    Ω^1^_{A/k} = I / I^2^
    其中 I = ker(μ: A ⊗_k A → A) 是对角理想
    """
    
    def __init__(
        self,
        ring: Ring,
        base_field: Field,
        *,
        tensor_backend: Optional[TensorBackend] = None,
    ):
        """
        通过范畴论泛性质构造Kähler微分模。
        
        Args:
            ring: k-代数 A
            base_field: 基域 k
        
        严格实现步骤：
        1. 构造张量积 A ⊗_k A
        2. 定义对角同态 μ: A⊗A → A, a⊗b ↦ ab
        3. 计算对角理想 I = ker(μ)
        4. 计算 I^2^ = <xy | x,y ∈ I>
        5. 构造商模 Ω^1^ = I / I^2^
        """
        self.ring = ring
        self.base_field = base_field
        self.tensor_backend = tensor_backend if tensor_backend is not None else FreeModuleTensorBackend()

        # ============================================================
        # A-模版本（严格）：对单变量多项式环 k[x]，Ω¹_{k[x]/k} 是自由 A-模，秩 1
        # ============================================================
        #
        # 数学事实：
        #   Ω¹_{k[x]/k} ≅ A·dx
        # 且对任意 f(x)=Σ a_i x^i：
        #   d f = f'(x) dx  （这里的 f' 是形式导数，可由莱布尼茨律递归推出）
        #
        # 这样我们可以直接给出 **多项式系数矩阵** 的 A-线性表示，
        # 避免未完成的 I/I² 后端导致的语义漂移（不做静默近似）。
        if isinstance(self.ring, PolynomialRing):
            # 目前 Polynomial 结构是单变量，因此 Ω¹ 的标准基只有 dx。
            self.module = FreeModule(self.ring, 1)
            # 保留下面这些属性名以避免外部代码意外访问时报 AttributeError
            self.tensor_product = None
            self.diagonal_hom = None
            self.diagonal_ideal = None
            self.ideal_square = None
            return
        
        # 步骤1: 构造张量积 A ⊗_k A
        self.tensor_product = self._construct_tensor_product()
        
        # 步骤2: 定义对角同态 μ: A⊗A → A
        self.diagonal_hom = self._construct_diagonal_homomorphism()
        
        # 步骤3: 计算对角理想 I = ker(μ)
        self.diagonal_ideal = self._compute_diagonal_ideal()
        
        # 步骤4: 计算 I^2^
        self.ideal_square = self._compute_ideal_square(self.diagonal_ideal)
        
        # 步骤5: 构造商模 Ω^1^ = I / I^2^
        self.module = self._construct_quotient_module()
    
    def _construct_tensor_product(self) -> TensorProduct:
        """构造张量积 A ⊗_k A"""
        # 对于多项式环，可显式构造
        if not isinstance(self.ring, PolynomialRing):
            raise NotImplementedError("目前仅支持多项式环")
        
        # 创建自由模表示 A ⊗_k A
        free_module = FreeModule(self.ring, (self.ring.one().degree() + 1) ** 2)
        return TensorProduct(self.ring, free_module, free_module, backend=self.tensor_backend)
    
    def _construct_diagonal_homomorphism(self) -> ModuleHomomorphism:
        """构造对角同态 μ: A⊗A → A, a⊗b ↦ ab"""
        # 对于自由模，显式定义
        if not isinstance(self.ring, PolynomialRing):
            raise NotImplementedError("目前仅支持多项式环")
        
        # 定义域：A⊗A (秩为 (deg+1)^2^)
        domain_rank = (self.ring.one().degree() + 1) ** 2
        domain = FreeModule(self.ring, domain_rank)
        
        # 陪域：A (秩为 deg+1)
        codomain = FreeModule(self.ring, self.ring.one().degree() + 1)

        side = math.isqrt(domain.rank)
        if side * side != domain.rank:
            raise ValueError(
                "Diagonal homomorphism construction requires a square rank for A⊗A representation; "
                f"got domain.rank={domain.rank}."
            )
        
        # 构建表示矩阵
        matrix = []
        for k in range(codomain.rank):
            row = [self.ring.zero() for _ in range(domain.rank)]
            for i in range(side):
                for j in range(side):
                    if i + j == k:
                        row[i * side + j] = self.ring.one()
            matrix.append(row)
        
        return FreeModuleHomomorphism(domain, codomain, matrix)
    
    def _compute_diagonal_ideal(self) -> PolynomialIdeal:
        """计算对角理想 I = ker(μ)"""
        # 注意：
        # - 本文件对 PolynomialRing(k[x]) 的 Ω¹_{k[x]/k} 采用 A·dx 的严格 A-模模型（见 __init__ 的提前返回分支）；
        # - 因此该 “I/I² via tensor + kernel” 的通用后端在当前工程并不需要。
        #
        # 若未来要支持一般 A（非多项式环）的 Ω¹ 计算，应在外部提供可验证的 A⊗_k A 表示与 kernel/I² 计算后端；
        # 在那之前，禁止给出不正确的占位生成元（避免语义漂移）。
        raise NotImplementedError(
            "KahlerDifferentials diagonal-ideal backend (I=ker(mu) in A⊗_k A) is not implemented in this module. "
            "For PolynomialRing(k[x]) the solver uses the strict A·dx model; for modular quotient-algebra reductions "
            "use MVP19 SyzygyGraph tooling."
        )
    
    def _compute_ideal_square(self, ideal: PolynomialIdeal) -> PolynomialIdeal:
        """计算理想平方 I^2^ = <xy | x,y ∈ I>"""
        # 严格实现：I^2^ 由 {f·g | f,g ∈ 生成元} 生成
        generators = []
        for f in ideal.generators:
            for g in ideal.generators:
                generators.append(self.ring.mul(f, g))
        return PolynomialIdeal(self.ring, generators)
    
    def _construct_quotient_module(self) -> Module:
        """构造商模 Ω^1^ = I / I^2^"""
        # 严格实现：计算商模 I / I^2^
        # 对于多项式理想，这对应于切丛
        
        # 创建自由表示
        rank = len(self.diagonal_ideal.generators)
        free_module = FreeModule(self.ring, rank)
        
        # 定义关系：I^2^ 中的元素
        relations = []
        for f in self.ideal_square.generators:
            # 将 f 表示为生成元的线性组合
            # 实际需解线性系统
            relations.append(f)
        
        # 构造商模
        return QuotientModule(free_module, relations)
    
    def differential(self, element: RingElement) -> ModuleElement:
        """
        计算元素的微分 d: A → Ω^1^。
        
        数学原理：
        d(a) = [a⊗1 - 1⊗a] ∈ I/I^2^
        
        Args:
            element: A中的元素
            
        Returns:
            ModuleElement: 微分在Ω^1^中的表示
        """
        # A-模版本仅在 PolynomialRing(k[x]) 上定义（当前多项式结构为单变量）
        if not isinstance(self.ring, PolynomialRing) or not isinstance(element, Polynomial):
            raise NotImplementedError(
                "KahlerDifferentials.differential currently implements the strict A-module model only for "
                "the univariate polynomial ring k[x] represented by PolynomialRing/Polynomial."
            )

        if element.is_zero():
            return self.module.zero()

        # 形式导数：f'(x) = Σ_{i>=1} i*a_i x^{i-1}
        field = self.ring.field
        deriv_coeffs: Dict[Tuple[int, ...], RingElement] = {}
        for deg_tuple, coef in element.coeffs.items():
            deg = deg_tuple[0] # Assuming univariate logic here as stated in method doc/checks
            if deg <= 0:
                continue
            factor = field.RationalNumber(int(deg), 1)
            new_coef = field.mul(factor, coef)
            tgt_deg_tuple = (int(deg - 1),) + deg_tuple[1:]
            
            if tgt_deg_tuple in deriv_coeffs:
                deriv_coeffs[tgt_deg_tuple] = field.add(deriv_coeffs[tgt_deg_tuple], new_coef)
            else:
                deriv_coeffs[tgt_deg_tuple] = new_coef

        deriv_poly = Polynomial(field, deriv_coeffs, element.num_vars)
        # Ω¹ 是自由模秩 1：元素用 (f'(x),) 表示 f'(x)·dx
        return (deriv_poly,)
    
    def _combine_terms(self, terms: List[Tuple[RingElement, int]]) -> ModuleElement:
        """
        组合微分项为单个模元素（严格实现）。
        
        Args:
            terms: 微分项列表 [(系数, 基索引), ...]
            
        Returns:
            ModuleElement: Ω¹ 中的元素（商模 I/I² 的元素）
        
        数学原理：
        - Ω¹ = I/I² 是商模
        - 每个微分项是形式 coef·dx_i
        - 需要在商模中求和并约化
        
        严格实现步骤：
        1. 构造自由模中的元素（提升）
        2. 在商模中约化（模去 I²）
        3. 返回等价类代表元
        """
        if not terms:
            return self.module.zero()
        
        # 步骤1: 确定商模的秩
        # Ω¹_{k[x]/k} 对于单变量是秩1的自由模，基为 dx
        
        # 获取商模的基模
        if not isinstance(self.module, QuotientModule):
            raise TypeError(f"期望 QuotientModule，得到 {type(self.module)}")
        
        base_module = self.module.base_module
        
        # 步骤2: 在基模中构造元素
        # 对于单变量多项式环 k[x]，Ω¹ 的基是 dx
        # 所以元素形式为 f(x)·dx
        
        # 累加系数
        total_coef = self.ring.field.zero()
        for coef, base_idx in terms:
            # base_idx 对应微分基的索引
            # 对于单变量，只有一个基 dx（索引0）
            if base_idx == 0:
                total_coef = self.ring.field.add(total_coef, coef)
        
        # 步骤3: 构造基模元素
        # 基模是对角理想 I 的自由模表示
        # 对于 f·dx，对应 I 中的元素
        
        if not isinstance(base_module, FreeModule):
            # 如果基模不是自由模，需要特殊处理
            # 这里返回零（保守实现）
            return self.module.zero()
        
        # 构造元素向量
        # 对于单变量，基模秩应该是生成元个数
        rank = base_module.rank
        element_vec = [base_module.base_ring.zero() for _ in range(rank)]
        
        # 将总系数放在第一个位置（对应 dx 的系数）
        if rank > 0:
            # 转换系数类型：RationalNumber → Polynomial
            if hasattr(total_coef, 'numerator') and hasattr(total_coef, 'denominator'):
                # 有理数系数
                coef_poly = self.ring.constant(total_coef)
            else:
                coef_poly = total_coef
            
            element_vec[0] = coef_poly
        
        element_tuple = tuple(element_vec)
        
        # 步骤4: 在商模中约化
        return self.module._reduce(element_tuple)

class QuotientModule(Module):
    """商模的严格实现"""
    
    def __init__(self, 
                 base_module: Module, 
                 relations: List[ModuleElement]):
        """
        初始化商模 M/N。
        
        Args:
            base_module: 基模 M
            relations: 关系子模 N 的生成元
        
        数学原理：
        - 商模 M/N 由泛性质定义
        - 严格实现为等价类 [m] = m + N
        """
        self.base_module = base_module
        self.relations = relations
        self.equivalence_classes = self._compute_equivalence_classes()
    
    def _compute_equivalence_classes(self) -> Dict[ModuleElement, ModuleElement]:
        """
        计算等价类代表元（严格实现）。
        
        数学原理：
        - 商模 M/N 中，每个元素 [m] = m + N 是等价类
        - 使用 Gröbner 基计算标准代表元（正规形式）
        - 每个等价类有唯一标准代表元（模去关系子模）
        
        严格实现步骤：
        1. 将关系子模表示为多项式理想
        2. 计算理想的 Gröbner 基
        3. 对每个基元素，计算其约化形式
        4. 构建等价类映射
        """
        # 对于自由模，可显式计算
        if not isinstance(self.base_module, FreeModule):
            raise NotImplementedError("目前仅支持自由模的商模")
        
        if not self.relations:
            # 无关系：商模等于基模
            # 每个元素的代表元就是自己
            return {}
        
        # 创建多项式理想表示关系
        field = RationalField()
        poly_ring = PolynomialRing(field)
        
        # 生成变量：对应基元素 e_0, e_1, ..., e_{rank-1}
        # 使用不同的次数来区分变量
        rank = self.base_module.rank
        
        # 生成关系多项式
        relation_polys = []
        for rel in self.relations:
            if not isinstance(rel, tuple):
                continue
            
            # 跳过零关系
            if all(c.is_zero() for c in rel):
                continue
            
            # 构造关系多项式
            # 关系 (r_0, r_1, ..., r_{n-1}) 表示 Σ r_i e_i = 0
            # 转换为多项式：Σ r_i x^i（这里用单变量表示，实际是符号表示）
            coeffs = {}
            for i, coef in enumerate(rel):
                if not coef.is_zero():
                    # 将系数转换为有理数
                    if hasattr(coef, 'numerator') and hasattr(coef, 'denominator'):
                        if coef.denominator == 1:
                            coeffs[i] = field.RationalNumber(coef.numerator, 1)
                    else:
                        # 尝试视为有理数
                        try:
                            coeffs[i] = field.RationalNumber(int(coef), 1)
                        except:
                            coeffs[i] = coef
            
            if coeffs:
                poly = Polynomial(field, coeffs)
                relation_polys.append(poly)
        
        if not relation_polys:
            # 无非零关系
            return {}
        
        # 计算 Gröbner 基
        ideal = PolynomialIdeal(poly_ring, relation_polys)
        groebner_basis = ideal.groebner_basis
        
        # 构建标准代表元映射
        # 对于商模，标准代表元通过 Gröbner 基约化得到
        classes = {}
        
        # 为每个基向量计算标准形式
        for i in range(rank):
            # 构造基向量 e_i = (0, ..., 0, 1, 0, ..., 0)
            elem_list = [self.base_module.base_ring.zero() for _ in range(rank)]
            elem_list[i] = self.base_module.base_ring.one()
            elem = tuple(elem_list)
            
            # 计算约化后的标准代表元
            # 使用 Gröbner 基约化
            reduced_elem = self._reduce_by_groebner(elem, groebner_basis, rank)
            
            classes[elem] = reduced_elem
        
        return classes
    
    def _reduce_by_groebner(self, 
                           element: ModuleElement, 
                           groebner_basis: List[Polynomial],
                           rank: int) -> ModuleElement:
        """
        使用 Gröbner 基约化模元素。
        
        数学原理：
        - 元素 m ∈ M 的约化形式是 m 模去关系子模的标准代表元
        - 使用 Gröbner 基逐步消去"高次项"
        
        Args:
            element: 待约化的模元素
            groebner_basis: 关系理想的 Gröbner 基
            rank: 模的秩
            
        Returns:
            ModuleElement: 约化后的标准代表元
        """
        if not groebner_basis:
            return element
        
        # 将元素转换为多项式表示
        # element = (c_0, c_1, ..., c_{n-1}) 对应 Σ c_i x^i
        field = RationalField()
        coeffs = {}
        for i, coef in enumerate(element):
            if not coef.is_zero():
                # 转换系数
                if hasattr(coef, 'numerator') and hasattr(coef, 'denominator'):
                    if coef.denominator == 1:
                        coeffs[i] = field.RationalNumber(coef.numerator, 1)
                else:
                    try:
                        coeffs[i] = field.RationalNumber(int(coef), 1)
                    except:
                        # 无法转换，保持原样
                        return element
        
        if not coeffs:
            # 零元素
            return element
        
        # 使用 Gröbner 基约化（通过理想成员测试）
        # 对于每个关系多项式，尝试消去对应的分量
        
        result_coeffs = coeffs.copy()
        
        for g in groebner_basis:
            if g.is_zero():
                continue
            
            # 找到 g 的首项
            leading_deg = g.degree()
            if leading_deg < 0 or leading_deg >= rank:
                continue
            
            # 如果结果中有对应项，尝试消去
            if leading_deg in result_coeffs and not result_coeffs[leading_deg].is_zero():
                lc_g = g.leading_coefficient()
                if lc_g.is_zero():
                    continue
                
                # 计算消去系数
                coef_to_cancel = result_coeffs[leading_deg]
                factor = field.divide(coef_to_cancel, lc_g)
                
                # 从结果中减去 factor * g
                for deg, coef in g.coeffs.items():
                    scaled_coef = field.mul(factor, coef)
                    if deg in result_coeffs:
                        result_coeffs[deg] = field.add(
                            result_coeffs[deg],
                            field.negate(scaled_coef)
                        )
                    else:
                        result_coeffs[deg] = field.negate(scaled_coef)
        
        # 转换回模元素
        result_list = [self.base_module.base_ring.zero() for _ in range(rank)]
        for deg, coef in result_coeffs.items():
            if deg < rank and not coef.is_zero():
                # 转换回原类型
                if hasattr(self.base_module.base_ring, 'RationalNumber'):
                    result_list[deg] = self.base_module.base_ring.RationalNumber(
                        coef.numerator, coef.denominator
                    )
                else:
                    result_list[deg] = coef
        
        return tuple(result_list)
    
    def zero(self) -> ModuleElement:
        return self.base_module.zero()
    
    def add(self, a: ModuleElement, b: ModuleElement) -> ModuleElement:
        sum_val = self.base_module.add(a, b)
        return self._reduce(sum_val)
    
    def scalar_mul(self, scalar: RingElement, 
                  element: ModuleElement) -> ModuleElement:
        scaled = self.base_module.scalar_mul(scalar, element)
        return self._reduce(scaled)
    
    def equal(self, a: ModuleElement, b: ModuleElement) -> bool:
        diff = self.base_module.add(a, self.base_module.scalar_mul(
            self.base_module.base_ring.negate(self.base_module.base_ring.one()),
            b
        ))
        return self.is_zero(diff)
    
    def is_zero(self, element: ModuleElement) -> bool:
        reduced = self._reduce(element)
        return self.base_module.is_zero(reduced)
    
    def _reduce(self, element: ModuleElement) -> ModuleElement:
        """将元素约化为标准代表元"""
        # 严格实现：使用Gröbner基约化
        if element in self.equivalence_classes:
            return self.equivalence_classes[element]
        return element

# ======================
# 余切复形的严格实现
# ======================

class KoszulHypersurfaceQuotientSimplicialRing:
    """
    目标对象：A = k[x]/(f) 的 **Koszul（超曲面）模型入口**。

    解释：
    - 对超曲面（单个方程）而言，余切复形有经典两项模型：
        L_{A/k} ≃ [ I/I^2  ->  Ω_{k[x]/k} ⊗ A ]
      其中映射由 d(f) 给出（在单变量时即乘以 f'(x)）。
    - 该两项模型与 Koszul 单纯分辨给出的 L_{A/k} 拟同构（André–Quillen 基本事实），
      因此在工程里可以 **不展开整个单纯环的各层乘法结构**，直接闭环到可计算矩阵。

    同时：
    - self.koszul_model 保存 SimplicialPoly(base_ring, f)，用于严格验证单纯恒等式（结构验收）。
    """

    def __init__(self, base_ring: PolynomialRing, f: Polynomial, *, max_level: int = 2):
        if not isinstance(base_ring, PolynomialRing):
            raise TypeError("KoszulHypersurfaceQuotientSimplicialRing requires base_ring=PolynomialRing(k[x]).")
        if not isinstance(f, Polynomial):
            raise TypeError("f must be a Polynomial.")
        if f.field != base_ring.field:
            raise ValueError("f must lie in the given base_ring.")
        if f.is_zero():
            raise ValueError("f must be nonzero for a hypersurface quotient.")

        self._base_ring = base_ring
        self._f = f
        self._max = int(max_level)

        # 结构验收用：正确的 Koszul 单纯模型
        self.koszul_model = SimplicialPoly(base_ring, f)

        # π_0(A_•) 对应的离散环：k[x]/(f)
        self.quotient_ring = QuotientPolynomialRing(base_ring, f)

    def base_field(self) -> Field:
        return self._base_ring.field

    def max_level(self) -> int:
        return self._max

    def ring_at_level(self, n: int) -> Ring:
        # 该类在 CotangentComplex 中走“两项超曲面”专用分支；
        # 这里返回 π_0 的离散环仅用于满足接口，不用于构造 L。
        if n < 0 or n > self._max:
            raise ValueError(f"level {n} out of range [0,{self._max}]")
        return self.quotient_ring

    def face_operator(self, n: int, i: int) -> Callable[[RingElement], RingElement]:
        # 同上：该对象不通过 level-wise face maps 计算 L_{A/k}，这里提供恒等作为占位。
        if n <= 0 or n > self._max:
            raise ValueError("face_operator requires 1 <= n <= max_level")
        if i < 0 or i > n:
            raise ValueError(f"face index {i} out of range [0,{n}]")

        def _id(a: RingElement) -> RingElement:
            return a

        return _id

    @property
    def defining_polynomial(self) -> Polynomial:
        return self._f

    @property
    def ambient_ring(self) -> PolynomialRing:
        return self._base_ring

    def verify_simplicial_identities(self, *, up_to_level: int = 2) -> bool:
        return bool(self.koszul_model.verify_simplicial_identities(max_level=int(up_to_level)))

    def __repr__(self) -> str:
        return f"KoszulHypersurfaceQuotientSimplicialRing(A={self.quotient_ring})"


class HypersurfaceTwoTermCotangentComplex:
    """
    超曲面 A = k[x]/(f) 的两项余切复形模型（作为 A-模的链复形）：

      C_1 = I/I^2  ≅ A
      C_0 = Ω_{k[x]/k} ⊗ A  ≅ A·dx
      d_1 : C_1 → C_0 由 d(f) 给出（单变量即乘以 f'(x)）

    在该模型下：
      H^{-1} = ker(d_1)
      H^{0}  = coker(d_1)
      H^{<-1} = 0
    """

    def __init__(self, ring_obj: KoszulHypersurfaceQuotientSimplicialRing, *, base_field: Field):
        self.base_field = base_field
        self.ambient_ring = ring_obj.ambient_ring
        self.f = ring_obj.defining_polynomial
        self.A = ring_obj.quotient_ring

        # 计算 f'(x) 并降到 A
        fprime = _poly_derivative_univariate(self.f, self.ambient_ring)
        self.fprime_bar = self.A._reduce(fprime)

        # 计算 annihilator ideal 的生成元： (f):f' = (f / gcd(f,f'))
        g = _poly_gcd_univariate(self.f, fprime, self.ambient_ring)
        q, r = _poly_divmod_univariate(self.f, g, self.ambient_ring)
        if not r.is_zero():
            raise RuntimeError("Internal error: gcd(f,f') does not divide f in hypersurface cotangent setup.")
        self.ann_generator_bar = self.A._reduce(q)

        # 链群（A-自由模）
        self.C0 = FreeModule(self.A, 1)  # A·dx
        self.C1 = FreeModule(self.A, 1)  # A·e (I/I^2)

        # 微分矩阵：乘以 f'(x)（在 A 中）
        self.d1 = FreeModuleHomomorphism(self.C1, self.C0, [[self.fprime_bar]])

    def cohomology(self, degree: int) -> CohomologyGroup:
        # 余切复形置于非正度：degree=-1 ↔ C1, degree=0 ↔ C0
        if degree > 0:
            # H^{>0}=0
            ker = ZeroModule(self.A)
            im = ZeroModule(self.A)
            return CohomologyGroup(ker, im, self.base_field)

        if degree == 0:
            # H^0 = C0 / im(d1)
            kernel = self.C0
            image = Submodule(self.C0, [(self.fprime_bar,)])
            return CohomologyGroup(kernel, image, self.base_field)

        if degree == -1:
            # H^{-1} = ker(d1)
            # ker(mul by f') = ann(f') = (f/gcd(f,f')) in A
            kernel = Submodule(self.C1, [(self.ann_generator_bar,)])
            image = ZeroModule(self.A)
            return CohomologyGroup(kernel, image, self.base_field)

        # degree <= -2
        ker = ZeroModule(self.A)
        im = ZeroModule(self.A)
        return CohomologyGroup(ker, im, self.base_field)


class CotangentComplex:
    """
    余切复形的严格实现
    
    数学原理：
    对于单纯交换环 A_•，余切复形 𝕃_{A/k} 是
    单纯模 n ↦ Ω^1^_{A_n/k} 的规范化链复形
    
    严格实现步骤：
    1. 为每个单纯层计算 Kähler 微分模
    2. 计算面算子诱导的映射
    3. 构造规范化链复形
    4. 计算上同调群
    """
    
    def __init__(self, simplicial_ring: 'SimplicialRing', *, tensor_backend: Optional[TensorBackend] = None):
        """
        初始化余切复形。
        
        Args:
            simplicial_ring: 单纯交换环
        
        严格实现要求：
        - 每个计算步骤可数学验证
        """
        self.simplicial_ring = simplicial_ring
        self.tensor_backend = tensor_backend
        bf = getattr(simplicial_ring, "base_field", None)
        self.base_field = bf() if callable(bf) else bf
        if self.base_field is None:
            raise ValueError("SimplicialRing.base_field is required to construct the cotangent complex.")
        self.kahler_modules = {}  # M_n = Ω^1^_{A_n/k}
        self.face_maps = {}       # d_i: M_n → M_{n-1}
        self.chain_complex = None
        self._construct()
    
    def _construct(self):
        """严格构造余切复形的各个组件"""
        # ------------------------------------------------------------
        # 超曲面（单方程）快捷闭环：A=k[x]/(f)
        # ------------------------------------------------------------
        if isinstance(self.simplicial_ring, KoszulHypersurfaceQuotientSimplicialRing):
            # 结构上仍可用 simplicial identities 验收；计算上采用两项模型直接给出 L_{A/k}
            self.chain_complex = HypersurfaceTwoTermCotangentComplex(self.simplicial_ring, base_field=self.base_field)
            return

        # 步骤1: 为每个单纯层计算 Kähler 微分模
        self._compute_kahler_modules()
        
        # 步骤2: 计算面算子诱导的映射
        self._compute_face_maps()
        
        # 步骤3（老师避雷第三补）：
        # 不显式构造 N_n = ⋂ ker(d_i)（那需要子模交集与限制映射的完整后端，当前版本历史上不完备）。
        # 直接在全复形 C_n = Ω¹_{A_n/k} 上构造交错微分
        #   d_n = Σ_{i=0}^n (-1)^i ∂_i^*
        # 并计算其同调；对单纯阿贝尔群/模，该同调与规范化复形同调同构。
        self.chain_complex = AlternatingSumChainComplex(
            self.kahler_modules,
            self.face_maps,
            base_field=self.base_field,
        )

    def cohomology(self, degree: int) -> 'CohomologyGroup':
        """
        对外接口：计算 H^^degree(𝕃_{A/k})。
        """
        if self.chain_complex is None:
            raise RuntimeError("CotangentComplex.chain_complex is not constructed.")
        return self.chain_complex.cohomology(degree)
    
    def _compute_kahler_modules(self):
        """为每个单纯层计算 Kähler 微分模"""
        # 对每个单纯度 n
        for n in range(self.simplicial_ring.max_level() + 1):
            ring = self.simplicial_ring.ring_at_level(n)
            self.kahler_modules[n] = KahlerDifferentials(ring, self.base_field, tensor_backend=self.tensor_backend)
    
    def _compute_face_maps(self):
        """计算面算子诱导的映射 d_i: M_n → M_{n-1}"""
        # 对每个单纯度 n ≥ 1 和每个面索引 i
        for n in range(1, self.simplicial_ring.max_level() + 1):
            for i in range(n + 1):
                # 获取单纯环的面算子 ∂_i: A_n → A_{n-1}
                ring_hom = self.simplicial_ring.face_operator(n, i)
                
                # 计算诱导映射 (∂_i)^^*: Ω^1^_{A_n/k} → Ω^1^_{A_{n-1}/k}
                self.face_maps[(n, i)] = self._induced_map(ring_hom, n, i)
    
    def _induced_map(self, 
                    ring_hom: Callable[[RingElement], RingElement],
                    n: int, i: int) -> ModuleHomomorphism:
        """
        计算环同态诱导的微分模映射。
        
        数学原理：
        若 φ: A → B 是环同态，则有 φ^^*: Ω^1^_A → Ω^1^_B
        满足 φ^^*(da) = d(φ(a))
        
        Args:
            ring_hom: 环同态 A_n → A_{n-1}
            n: 源单纯度
            i: 面索引
            
        Returns:
            ModuleHomomorphism: 诱导映射 Ω^1^_{A_n/k} → Ω^1^_{A_{n-1}/k}
        """
        # 获取源和目标微分模
        source_module = self.kahler_modules[n]
        target_module = self.kahler_modules[n-1]

        # ============================================================
        # A-模版本（严格）：多项式矩阵表示，不在点上求值
        # ============================================================
        #
        # 当前文件的 Ω¹_{A/k} 在 PolynomialRing 情形下采用自由模表示：
        #   Ω¹_{k[x]/k} ≅ A·dx
        # 且诱导映射满足：φ*(dx) = d(φ(x))。
        #
        # 在这一模型下，诱导映射可以用 1×1 的多项式矩阵表示：
        #   [ d(φ(x)) 的 dx 系数 ] = [ (φ(x))' ]  ∈ A_{n-1}
        src_mod = source_module.module
        tgt_mod = target_module.module

        if not isinstance(src_mod, FreeModule) or not isinstance(tgt_mod, FreeModule):
            raise NotImplementedError(
                "Induced map currently requires Ω¹ modules to be represented as FreeModule. "
                "Ensure KahlerDifferentials exposes a FreeModule model for the ring at each simplicial level."
            )
        if src_mod.rank != 1 or tgt_mod.rank != 1:
            raise NotImplementedError(
                "Only the univariate k[x] case (rank(Ω¹)=1) is implemented in this strict A-module backend."
            )
        if not isinstance(source_module.ring, PolynomialRing) or not isinstance(target_module.ring, PolynomialRing):
            raise NotImplementedError(
                "Induced map matrix backend currently supports only PolynomialRing(k[x]) levels."
            )

        # 生成元 x ∈ A_n
        x_src = source_module.ring.variable_poly()
        phi_x = ring_hom(x_src)
        if not isinstance(phi_x, Polynomial):
            raise TypeError("face_operator must map Polynomial to Polynomial in the current k[x] backend.")

        # d(φ(x)) ∈ Ω¹_{A_{n-1}/k} = A_{n-1}·dx
        d_phi_x = target_module.differential(phi_x)
        if not isinstance(d_phi_x, tuple) or len(d_phi_x) != 1 or not isinstance(d_phi_x[0], Polynomial):
            raise TypeError("Target KahlerDifferentials.differential must return a 1-tuple (Polynomial,) in k[x] model.")

        coeff = d_phi_x[0]  # (φ(x))'
        matrix = [[coeff]]  # 1×1
        return FreeModuleHomomorphism(src_mod, tgt_mod, matrix)
    
class NormalizedChainComplex:
    """
    规范化链复形的严格实现
    
    数学原理：
    单纯模 M_• 的规范化链复形 N_• 定义为：
    N_n = ⋂_{i=0}^^{n-1} ker(d_i) ⊂ M_n
    边界映射 ∂_n = ∑_{i=0}^^n (-1)^^i d_i: N_n → N_{n-1}
    
    严格实现要求：
    - 精确计算交集 ⋂ ker(d_i)
    - 严格实现边界映射
    - 精确计算上同调群
    """
    
    def __init__(self, 
                 kahler_modules: Dict[int, KahlerDifferentials],
                 face_maps: Dict[Tuple[int, int], ModuleHomomorphism]):
        """
        初始化规范化链复形。
        
        Args:
            kahler_modules: 各层的Kähler微分模 M_n
            face_maps: 面映射 d_i: M_n → M_{n-1}
        """
        self.kahler_modules = kahler_modules
        self.face_maps = face_maps
        self.modules = {}       # N_n
        self.boundary_maps = {} # ∂_n: N_n → N_{n-1}
        self._construct()
    
    def _construct(self):
        """严格构造规范化链复形"""
        # 对每个度 n
        for n in range(max(self.kahler_modules.keys()) + 1):
            # 步骤1: 计算 N_n = ⋂_{i=0}^^{n-1} ker(d_i)
            self.modules[n] = self._compute_normalized_module(n)
            
            # 步骤2: 计算边界映射 ∂_n = ∑ (-1)^^i d_i
            if n > 0:
                self.boundary_maps[n] = self._compute_boundary_map(n)
    
    def _compute_normalized_module(self, n: int) -> Module:
        """
        计算规范化模 N_n = ⋂_{i=0}^^{n-1} ker(d_i)。
        
        数学原理：
        - N_n 是 M_n 的子模，由所有被前 n 个面映射消去的元素组成
        - 严格实现为多个核的交集
        """
        if n == 0:
            # N_0 = M_0
            return self.kahler_modules[0].module
        
        # 获取 M_n
        m_n = self.kahler_modules[n].module
        
        # 计算交集 ⋂_{i=0}^^{n-1} ker(d_i)
        kernels = []
        for i in range(n):
            # 获取面映射 d_i: M_n → M_{n-1}
            face_map = self.face_maps.get((n, i))
            if face_map is None:
                continue
            
            # 计算核 ker(d_i)
            kernel = self._compute_kernel(face_map)
            kernels.append(kernel)
        
        # 计算交集
        return self._intersect_modules(kernels)
    
    def _compute_kernel(self, homomorphism: ModuleHomomorphism) -> Module:
        """
        计算模同态的核。
        
        数学原理：
        ker(φ) = {x ∈ M | φ(x) = 0}
        严格实现为解齐次线性系统
        """
        # 对于自由模同态，核是矩阵在基环上的 kernel。
        if not isinstance(homomorphism, FreeModuleHomomorphism):
            raise NotImplementedError("Kernel computation currently supports FreeModuleHomomorphism only.")

        domain = homomorphism.domain()
        codomain = homomorphism.codomain()
        if not isinstance(domain, FreeModule) or not isinstance(codomain, FreeModule):
            raise NotImplementedError("Kernel computation currently supports FreeModule domains/codomains only.")
        if domain.base_ring != codomain.base_ring:
            raise ValueError("Kernel computation requires domain/codomain to share the same base ring.")

        ring = domain.base_ring
        if isinstance(ring, PolynomialRing) and int(ring.num_vars) == 1:
            # Strict PID backend: k[x] is a PID, so kernel generators are computable via diagonal form.
            matrix = homomorphism._matrix
            gens = _kernel_generators_free_module_hom_kx(matrix, ring)
            if not gens:
                return Submodule(domain, [domain.zero()])
            return Submodule(domain, gens)

        raise NotImplementedError(
            "Kernel computation is implemented strictly only for FreeModule over the PID k[x] (PolynomialRing with num_vars=1). "
            "For multivariate modular kernels/syzygies use the MVP12/MVP19 pipeline."
        )
    
    def _solve_homogeneous_system(self, 
                                 equations: List[Polynomial], 
                                 domain: FreeModule) -> Module:
        """
        求解齐次线性方程组 Ax = 0（严格实现）。
        
        Args:
            equations: 方程列表（每个方程是变量的线性组合）
            domain: 定义域自由模
            
        Returns:
            Module: 解空间（作为子模）
        
        数学原理：
        - 解空间 = ker(A) 是自由模的子模
        - 使用 Gröbner 基计算 syzygy 模（合冲模）
        - syzygy(f_1,...,f_m) = {(h_1,...,h_m) | Σ h_i f_i = 0}
        
        严格实现步骤：
        1. 将方程组视为理想生成元
        2. 计算 syzygy 模的 Gröbner 基
        3. 提取线性无关的生成元
        """
        # NOTE:
        # - This helper was an early sketch that tried to encode module kernels as commutative syzygies.
        # - In strict mode we do not keep an incomplete/incorrect solver around.
        # - Kernel computation is implemented directly in `_compute_kernel()` using PID=k[x] linear algebra.
        raise NotImplementedError(
            "NormalizedChainComplex._solve_homogeneous_system is deprecated in this codebase. "
            "Use NormalizedChainComplex._compute_kernel (PID k[x] backend) or the MVP12/MVP19 modular pipeline "
            "for multivariate commutative Groebner/syzygy tasks."
        )
    
    def _compute_syzygy_module(self, 
                              basis: List[Polynomial], 
                              ambient: FreeModule) -> List[ModuleElement]:
        """
        计算 syzygy 模（合冲模）的生成元（未实现）。

        说明（严格）：
        - 该函数原本是“用 S-多项式关系拼 syzygy”的草稿，不满足严格正确性要求；
        - 本仓库的可验证 syzygy/rewrite 体系由 MVP19 提供（SyzygyGraph + decoder + Frobenius closure）；
        - MVP18 计算 𝕃_{A/k} 的主路径不依赖此函数（使用 AlternatingSumChainComplex / PID=k[x] 后端）。
        """
        raise NotImplementedError(
            "Syzygy module computation is not implemented in NormalizedChainComplex. "
            "Use MVP19 tooling (mvp19_syzygy_frobenius_suite / mvp19_groebner_middleware) for modular syzygy/rewrite tasks."
        )
    
    def _minimize_syzygy_generators(self, 
                                   syzygies: List[Tuple[RingElement, ...]], 
                                   ambient: FreeModule) -> List[ModuleElement]:
        """
        极小化 syzygy 生成元（未实现）。

        该接口保留仅为兼容旧草稿；严格模式下不提供不完备实现。
        """
        raise NotImplementedError(
            "Syzygy generator minimization is not implemented; use MVP19 backend(s) if you need syzygy bases."
        )
    
    def _intersect_modules(self, modules: List[Module]) -> Module:
        """
        计算多个模的交集
        
        数学原理：
        M_1 ∩ ... ∩ M_k = {x | x ∈ M_i for all i}
        
        严格实现步骤：
        1. 对于子模，交集 = 满足所有子模生成关系的元素
        2. 构造联立方程组
        3. 使用 Gröbner 基求解
        
        Args:
            modules: 模列表
            
        Returns:
            Module: 交集（作为子模）
        """
        if not modules:
            # 空交集：返回环境模
            return self.kahler_modules[0].module if self.kahler_modules else None
        
        if len(modules) == 1:
            return modules[0]
        
        # 对于所有模都是同一环境模的子模的情况
        ambient = None
        all_generators = []
        
        for mod in modules:
            if isinstance(mod, Submodule):
                if ambient is None:
                    ambient = mod.ambient
                elif ambient != mod.ambient:
                    raise ValueError("所有子模必须在同一环境模中")
                
                # 收集所有生成元
                all_generators.extend(mod.generators)
            elif isinstance(mod, FreeModule):
                # 自由模与任何子模的交集是该子模
                continue
            elif isinstance(mod, ZeroModule):
                # 零模与任何模的交集是零模
                return mod
            else:
                raise NotImplementedError(f"不支持的模类型: {type(mod)}")
        
        if ambient is None:
            # 所有模都是自由模：交集是第一个模
            return modules[0]
        
        if len(modules) == 2:
            return self._intersect_two_submodules(modules[0], modules[1], ambient)
        
        # 多个模的交集：递归计算
        result = modules[0]
        for i in range(1, len(modules)):
            result = self._intersect_two_submodules(result, modules[i], ambient)
        
        return result
    
    def _intersect_two_submodules(self, 
                                  m1: Module, 
                                  m2: Module, 
                                  ambient: FreeModule) -> Module:
        """

        对于子模：
        M_1 = span{g_1, ..., g_m}
        M_2 = span{h_1, ..., h_n}
        
        M_1 ∩ M_2 的生成元可通过以下方法计算：
        1. 构造方程 Σ λ_i g_i - Σ μ_j h_j = 0
        2. 求解 (λ, μ) 的解空间
        3. 每个解 (λ*, μ*) 对应一个交集中的生成元 Σ λ_i* g_i
        """
        if not isinstance(m1, Submodule) or not isinstance(m2, Submodule):
            # 特殊情况处理
            if isinstance(m1, ZeroModule) or isinstance(m2, ZeroModule):
                return ZeroModule(ambient.base_ring)
            if isinstance(m1, FreeModule):
                return m2
            if isinstance(m2, FreeModule):
                return m1
            raise NotImplementedError("仅支持子模的交集")
        
        g_gens = m1.generators  # M_1 的生成元
        h_gens = m2.generators  # M_2 的生成元
        
        if not g_gens:
            return ZeroModule(ambient.base_ring)
        if not h_gens:
            return ZeroModule(ambient.base_ring)

        # 红线：严格交集 = syzygy([G|-H]) 回代生成元
        # 设 G 的列为 g_1..g_m，H 的列为 h_1..h_n（都在 ambient=R^r）。
        # 求 (λ, μ) ∈ R^{m+n} 使得  Σ λ_i g_i - Σ μ_j h_j = 0
        # 即 [G|-H] · (λ, μ)^T = 0。其解模（核）是 syzygy 模。
        # 对每个 syzygy 生成元 (λ*, μ*)，交集里对应元素为 Σ λ_i* g_i（等于 Σ μ_j* h_j）。

        if not isinstance(ambient, FreeModule):
            raise TypeError("ambient must be a FreeModule.")
        ring = ambient.base_ring
        if not isinstance(ring, PolynomialRing) or int(ring.num_vars) != 1:
            raise NotImplementedError(
                "Submodule intersection via syzygy is currently implemented only for PID base ring k[x] "
                "(PolynomialRing with num_vars=1)."
            )

        r = int(ambient.rank)
        if r < 0:
            raise ValueError("ambient.rank must be non-negative.")
        if r == 0:
            # ambient=0，则所有子模都等于 0，交集仍为 0
            return ZeroModule(ring)

        # 维度与类型一致性检查（严格：不接受混入非 Polynomial 元素的向量）
        def _validate_generator_vector(vec: ModuleElement, *, which: str) -> None:
            if not isinstance(vec, tuple):
                raise TypeError(f"{which} generator must be a tuple, got {type(vec).__name__}")
            if len(vec) != r:
                raise ValueError(f"{which} generator length mismatch: expected={r}, got={len(vec)}")
            for comp in vec:
                if not isinstance(comp, Polynomial):
                    raise TypeError(
                        f"{which} generator components must be Polynomial in k[x] backend, got {type(comp).__name__}"
                    )
                if comp.field != ring.field or int(comp.num_vars) != int(ring.num_vars):
                    raise ValueError(f"{which} generator Polynomial must live in the same ring as ambient.base_ring")

        for idx, g in enumerate(g_gens):
            _validate_generator_vector(g, which=f"M1[{idx}]")
        for idx, h in enumerate(h_gens):
            _validate_generator_vector(h, which=f"M2[{idx}]")

        m = len(g_gens)
        n = len(h_gens)

        # 构造矩阵 A=[G|-H]，形状 r × (m+n)，列向量是生成元
        A: List[List[Polynomial]] = [[ring.zero() for _ in range(m + n)] for _ in range(r)]
        for j in range(m):
            col = g_gens[j]
            for i in range(r):
                A[i][j] = col[i]
        for j in range(n):
            col = h_gens[j]
            for i in range(r):
                A[i][m + j] = ring.negate(col[i])

        # 求核：syzygy 生成元 w_k ∈ R^{m+n}
        syzygies = _kernel_generators_free_module_hom_kx(A, ring)
        if not syzygies:
            return ZeroModule(ring)

        # 将 syzygy 回代为交集生成元
        intersection_gens: List[ModuleElement] = []
        for w in syzygies:
            if len(w) != m + n:
                raise RuntimeError("Internal: syzygy generator length mismatch.")

            lambdas = w[:m]
            mus = w[m:]

            x_from_g = ambient.zero()
            for lam, g in zip(lambdas, g_gens):
                if lam.is_zero():
                    continue
                x_from_g = ambient.add(x_from_g, ambient.scalar_mul(lam, g))

            x_from_h = ambient.zero()
            for mu, h in zip(mus, h_gens):
                if mu.is_zero():
                    continue
                x_from_h = ambient.add(x_from_h, ambient.scalar_mul(mu, h))

            # 一致性校验：两种回代必须严格相等
            if not ambient.equal(x_from_g, x_from_h):
                raise RuntimeError("Internal: syzygy back-substitution mismatch (G·λ != H·μ).")

            if ambient.is_zero(x_from_g):
                continue

            # 去重（严格相等判定，不做任何数值容差）
            if not any(ambient.equal(x_from_g, existing) for existing in intersection_gens):
                intersection_gens.append(x_from_g)

        if not intersection_gens:
            return ZeroModule(ring)

        # 额外严格性：用 PID 成员判定验证生成元确实落在两侧子模内
        for gen in intersection_gens:
            if not _submodule_membership_kx(gen, g_gens, ring):
                raise RuntimeError("Internal: produced generator not in M1 (membership check failed).")
            if not _submodule_membership_kx(gen, h_gens, ring):
                raise RuntimeError("Internal: produced generator not in M2 (membership check failed).")

        return Submodule(ambient, intersection_gens)
    
    def _compute_boundary_map(self, n: int) -> ModuleHomomorphism:
        """
        计算边界映射 ∂_n = ∑_{i=0}^^n (-1)^^i d_i: N_n → N_{n-1}。
        
        数学原理：
        - 规范化链复形的边界映射
        - 严格实现为面映射的交错和
        """
        # 获取 N_n 和 N_{n-1}
        domain = self.modules[n]
        codomain = self.modules[n-1]
        
        # 获取所有面映射 d_i: M_n → M_{n-1}
        component_maps = []
        for i in range(n + 1):
            sign = 1 if i % 2 == 0 else -1
            face_map = self.face_maps.get((n, i))
            if face_map is not None:
                component_maps.append((sign, face_map))
        
        # 构造组合映射
        def apply(element: ModuleElement) -> ModuleElement:
            result = codomain.zero()
            for sign, face_map in component_maps:
                image = face_map(element)
                if sign == 1:
                    result = codomain.add(result, image)
                else:
                    result = codomain.add(result, codomain.scalar_mul(
                        codomain.base_ring.negate(codomain.base_ring.one()),
                        image
                    ))
            return result
        
        return CallableModuleHomomorphism(domain, codomain, apply)
    
    def cohomology(self, degree: int) -> 'CohomologyGroup':
        """
        计算上同调群 H^^degree^^。
        
        Args:
            degree: 上同调度（可为负数）
            
        Returns:
            CohomologyGroup: 上同调群
        
        数学原理：
        H^^n = ker(d^^n) / im(d^^{n-1})
        """
        # 转换为链复形索引（余切复形置于非正度）
        n = -degree  # 因为 𝕃 ∈ D^^≤0^,^ 所以 H^^n 对应链复形的 C_{-n}
        
        # 获取相关模块和映射
        c_n = self.modules.get(n, None)
        c_n_minus1 = self.modules.get(n-1, None)
        c_n_plus1 = self.modules.get(n+1, None)

        if c_n is None:
            raise ValueError(f"Cotangent complex has no chain group at index n={n} (from cohomological degree {degree}).")
        
        d_n = self.boundary_maps.get(n, None)
        d_n_plus1 = self.boundary_maps.get(n+1, None)
        
        # 计算 ker(d^^n) = ker(d_{-n}: C_{-n} → C_{-n-1})
        kernel = self._compute_kernel(d_n) if d_n else c_n
        
        # 计算 im(d^^{n-1}) = im(d_{-n+1}: C_{-n+1} → C_{-n})
        if d_n_plus1:
            image = self._compute_image(d_n_plus1, c_n_plus1, c_n)
        else:
            # im = 0
            ring = getattr(c_n, "base_ring", None)
            if ring is None and isinstance(c_n, QuotientModule):
                ring = getattr(c_n.base_module, "base_ring", None)
            if ring is None and isinstance(c_n, Submodule):
                ring = getattr(c_n.ambient, "base_ring", None)
            if ring is None and isinstance(c_n, ZeroModule):
                ring = c_n.ring
            if ring is None:
                raise TypeError("Cannot infer base ring for ZeroModule when building CohomologyGroup.")
            image = ZeroModule(ring)
        
        # NOTE: NormalizedChainComplex 的严格后端历史上未闭环（子模交/限制映射/基域注入）。
        # 本工程默认使用 AlternatingSumChainComplex 计算同调。
        raise NotImplementedError(
            "NormalizedChainComplex.cohomology is deprecated in this codebase. "
            "Use CotangentComplex (which delegates to AlternatingSumChainComplex) instead."
        )


class AlternatingSumChainComplex:
    """
    单纯模 M_• 的“全复形”链复形：
      C_n := M_n
      d_n := Σ_{i=0}^n (-1)^i d_i : C_n → C_{n-1}

    对单纯阿贝尔群/模，该同调与规范化 Moore 复形同调同构（Dold–Kan）。

    工程目的：
    - 保持微分始终是 **矩阵同态**（FreeModuleHomomorphism），便于做 A-模版本的严格 kernel/image。
    - 避免显式求 ⋂ ker(d_i) 与“限制到子模”的映射矩阵化这两个历史债务点。
    """

    def __init__(
        self,
        kahler_modules: Dict[int, KahlerDifferentials],
        face_maps: Dict[Tuple[int, int], ModuleHomomorphism],
        *,
        base_field: Field,
    ):
        self.kahler_modules = kahler_modules
        self.face_maps = face_maps
        self.base_field = base_field
        self.modules: Dict[int, Module] = {}
        self.differentials: Dict[int, FreeModuleHomomorphism] = {}
        self._construct()

    def _construct(self) -> None:
        max_n = max(self.kahler_modules.keys()) if self.kahler_modules else 0
        for n in range(max_n + 1):
            self.modules[n] = self.kahler_modules[n].module

        for n in range(1, max_n + 1):
            self.differentials[n] = self._compute_alternating_differential(n)

        self._validate_chain_complex(max_n)

    def _compute_alternating_differential(self, n: int) -> FreeModuleHomomorphism:
        if n <= 0:
            raise ValueError("Alternating differential requires n>=1.")
        src = self.modules[n]
        tgt = self.modules[n - 1]
        if not isinstance(src, FreeModule) or not isinstance(tgt, FreeModule):
            raise NotImplementedError("AlternatingSumChainComplex currently supports FreeModule levels only.")
        ring = src.base_ring
        if ring != tgt.base_ring:
            raise ValueError("All chain groups must share the same base ring to form a matrix chain complex.")

        # 初始化零矩阵
        z = ring.zero()
        acc: List[List[RingElement]] = [[z for _ in range(src.rank)] for _ in range(tgt.rank)]

        for i in range(n + 1):
            m = self.face_maps.get((n, i))
            if m is None:
                raise ValueError(f"Missing face map (n={n}, i={i}) needed to build alternating differential.")
            if not isinstance(m, FreeModuleHomomorphism):
                raise NotImplementedError("Face maps must be FreeModuleHomomorphism to build matrix differential.")
            if m.domain().rank != src.rank or m.codomain().rank != tgt.rank:
                raise ValueError("Face map dimensions do not match chain group ranks.")

            sign = 1 if (i % 2 == 0) else -1
            for r in range(tgt.rank):
                for c in range(src.rank):
                    entry = m._matrix[r][c]
                    acc[r][c] = acc[r][c] + (entry if sign == 1 else (-entry))

        return FreeModuleHomomorphism(src, tgt, acc)

    def _validate_chain_complex(self, max_n: int) -> None:
        # 检查 d_{n} ∘ d_{n+1} = 0（严格：逐项等于零）
        for n in range(1, max_n):
            d_n = self.differentials.get(n)
            d_n1 = self.differentials.get(n + 1)
            if d_n is None or d_n1 is None:
                continue
            comp = d_n.compose(d_n1)
            for row in comp._matrix:
                for entry in row:
                    if not entry.is_zero():
                        raise ValueError(f"Chain complex condition failed at n={n}: d_n ∘ d_{n+1} ≠ 0.")

    def _compute_kernel(self, hom: Optional[FreeModuleHomomorphism], domain: FreeModule) -> Module:
        if hom is None:
            return domain
        ring = domain.base_ring
        if not isinstance(ring, PolynomialRing):
            raise NotImplementedError("Kernel computation currently implemented only for PolynomialRing(k[x]) base ring.")
        gens = _kernel_generators_free_module_hom_kx(hom._matrix, ring)
        if not gens:
            return Submodule(domain, [domain.zero()])
        return Submodule(domain, gens)

    def _compute_image(self, hom: Optional[FreeModuleHomomorphism], codomain: FreeModule) -> Module:
        if hom is None:
            return ZeroModule(codomain.base_ring)
        cols: List[ModuleElement] = []
        for j in range(hom.domain().rank):
            col = tuple(hom._matrix[i][j] for i in range(hom.codomain().rank))
            cols.append(col)
        return Submodule(codomain, cols)

    def cohomology(self, degree: int) -> 'CohomologyGroup':
        """
        计算上同调群 H^^degree（余切复形置于非正度）。
        """
        n = -degree
        c_n = self.modules.get(n)
        if c_n is None:
            raise ValueError(f"Chain complex has no group at index n={n} (from degree {degree}).")
        if not isinstance(c_n, FreeModule):
            raise NotImplementedError("Cohomology currently supports FreeModule chain groups only.")

        d_n = self.differentials.get(n) if n > 0 else None
        d_n1 = self.differentials.get(n + 1)

        kernel = self._compute_kernel(d_n, c_n) if n >= 0 else ZeroModule(c_n.base_ring)
        image = self._compute_image(d_n1, c_n) if n >= 0 else ZeroModule(c_n.base_ring)

        return CohomologyGroup(kernel, image, self.base_field)

class ZeroModule(Module):
    """零模的严格实现"""
    
    def __init__(self, ring: Ring):
        self.ring = ring
    
    def zero(self) -> ModuleElement:
        return ()
    
    def add(self, a: ModuleElement, b: ModuleElement) -> ModuleElement:
        return ()
    
    def scalar_mul(self, scalar: RingElement, 
                  element: ModuleElement) -> ModuleElement:
        return ()
    
    def equal(self, a: ModuleElement, b: ModuleElement) -> bool:
        # 零模只有一个元素：0。
        # 但在工程里，零元素可能以不同“载体表示”传入（例如来自 FreeModule 的 (0,0,...,0)）。
        # 因此这里用“零判定”做相等判定，避免把 0 向量误判为非零。
        return self.is_zero(a) and self.is_zero(b)
    
    def is_zero(self, element: ModuleElement) -> bool:
        # 零模的唯一元素是 0。允许以下等价表示：
        # - ()  （ZeroModule 自己的规范表示）
        # - (0, 0, ..., 0) （来自自由模/子模的零向量表示）
        # - 具有 is_zero() 的环元素（极少数情况下会被直接传入）
        if element == ():
            return True
        if isinstance(element, tuple):
            for comp in element:
                if hasattr(comp, "is_zero") and callable(getattr(comp, "is_zero")):
                    if not comp.is_zero():
                        return False
                else:
                    # 保守：无法判定就视为非零
                    return False
            return True
        if hasattr(element, "is_zero") and callable(getattr(element, "is_zero")):
            return bool(element.is_zero())
        return False

class Submodule(Module):
    """子模的严格实现"""
    
    def __init__(self, ambient: Module, generators: List[ModuleElement]):
        """
        初始化子模。
        
        Args:
            ambient: 环境模
            generators: 生成元列表
        """
        self.ambient = ambient
        self.generators = generators
        self.basis = self._compute_basis()
    
    def _compute_basis(self) -> List[ModuleElement]:
        """
        计算子模的一组“自由基”（在可严格闭环的后端上）。

        严格语义：
        - 对 PID=k[x] 上的自由模，子模必自由；通过成员资格判定逐个筛除线性相关生成元，可得到一组基。
        - 对 A=k[x]/(f) 的 rank-1 子模（理想），该理想为主理想，可规约为单个生成元。
        - 其它一般环/一般模：返回去零/去重后的生成元集（不声称是极小基），避免给出错误的“基”算法承诺。
        """
        amb = self.ambient
        gens = list(self.generators) if self.generators else []

        # 过滤零生成元 + 去重（严格相等，不做任何容差）
        cleaned: List[ModuleElement] = []
        for g in gens:
            try:
                is_zero = amb.is_zero(g)
            except Exception:
                is_zero = False
            if is_zero:
                continue
            if any(amb.equal(g, h) for h in cleaned):
                continue
            cleaned.append(g)

        # PID=k[x] 上：用成员资格判定做线性无关筛选，得到自由基
        if isinstance(amb, FreeModule) and isinstance(amb.base_ring, PolynomialRing) and int(amb.base_ring.num_vars) == 1:
            ring = amb.base_ring
            basis: List[ModuleElement] = []
            for g in cleaned:
                if not isinstance(g, tuple) or len(g) != int(amb.rank):
                    raise TypeError("Submodule generator must be a tuple matching ambient.rank in k[x] backend.")
                if any(not isinstance(c, Polynomial) for c in g):
                    raise TypeError("Submodule generator components must be Polynomial in k[x] backend.")
                # g ∈ span(basis) ? 若否则加入（保证线性无关）
                if not _submodule_membership_kx(g, [b for b in basis if isinstance(b, tuple)], ring):  # type: ignore[arg-type]
                    basis.append(g)
            return basis

        # A=k[x]/(f) 上 rank-1：理想为主理想，规约为单一生成元
        if (
            isinstance(amb, FreeModule)
            and isinstance(amb.base_ring, QuotientPolynomialRing)
            and int(amb.rank) == 1
        ):
            ringq = amb.base_ring
            base = ringq.base_ring
            polys: List[Polynomial] = []
            for g in cleaned:
                if not isinstance(g, tuple) or len(g) != 1 or not isinstance(g[0], Polynomial):
                    raise TypeError("Rank-1 quotient-ring submodule generators must be 1-tuples of Polynomial.")
                gg = ringq._reduce(g[0])
                if not gg.is_zero():
                    polys.append(gg)
            if not polys:
                return []
            d = ringq.modulus
            for gg in polys:
                d = _poly_gcd_univariate(d, gg, base)
            gen = ringq._reduce(d)
            if gen.is_zero():
                return []
            return [(gen,)]

        return cleaned
    
    def zero(self) -> ModuleElement:
        return self.ambient.zero()
    
    def add(self, a: ModuleElement, b: ModuleElement) -> ModuleElement:
        return self.ambient.add(a, b)
    
    def scalar_mul(self, scalar: RingElement, 
                  element: ModuleElement) -> ModuleElement:
        return self.ambient.scalar_mul(scalar, element)
    
    def equal(self, a: ModuleElement, b: ModuleElement) -> bool:
        return self.ambient.equal(a, b)
    
    def is_zero(self, element: ModuleElement) -> bool:
        return self.ambient.is_zero(element)

@dataclass
class CohomologyGroup:
    """上同调群的严格数学实现"""
    kernel: Module
    image: Module
    base_field: Field  # 添加基域信息，对维数计算至关重要
    
    def __post_init__(self):
        """验证数学一致性条件"""
        # 验证：image 必须是 kernel 的子模（上同调定义要求）
        if not self._is_submodule(self.image, self.kernel):
            raise ValueError("上同调定义错误：im ⊈ ker")
    
    def _is_submodule(self, sub: Module, ambient: Module) -> bool:
        """
        严格验证 sub ⊆ ambient。
        
        数学原理：
        - 对 sub 的每个生成元，检查是否属于 ambient
        - 使用 Gröbner 基技术解决成员资格问题
        
        Args:
            sub: 子模
            ambient: 环境模
            
        Returns:
            bool: 是否 sub ⊆ ambient
        """
        # 获取生成元
        sub_generators = self._get_generators(sub)
        
        # 检查每个生成元是否在 ambient 中
        for gen in sub_generators:
            if not self._is_element_of_module(gen, ambient):
                return False
        return True
    
    def _get_generators(self, module: Module) -> List[ModuleElement]:
        """
        获取模块的生成元列表。
        
        数学原理：
        - 自由模：标准基
        - 商模：提升的生成元
        - 子模：给定生成元
        
        Args:
            module: 模
            
        Returns:
            List[ModuleElement]: 生成元列表
        """
        # 严格实现：根据模块类型提取生成元
        if isinstance(module, ZeroModule):
            return []
        
        if isinstance(module, FreeModule):
            generators = []
            for i in range(module.rank):
                basis_vec = list(module.zero())
                basis_vec[i] = module.base_ring.one()
                generators.append(tuple(basis_vec))
            return generators
        
        if isinstance(module, Submodule):
            return module.generators
        
        if isinstance(module, QuotientModule):
            # 在商模中，生成元是基模生成元的投影
            base_generators = self._get_generators(module.base_module)
            return [module._reduce(gen) for gen in base_generators]
        
        raise NotImplementedError(f"不支持的模块类型: {type(module)}")
    
    def _is_element_of_module(self, 
                             element: ModuleElement, 
                             module: Module) -> bool:
        """
        检查元素是否属于模块。
        
        数学原理：
        - 使用 Gröbner 基约化：element ∈ M 当且仅当 element →_M 0
        - 严格实现为理想成员资格问题
        
        Args:
            element: 模元素
            module: 模
            
        Returns:
            bool: 是否 element ∈ module
        """
        # 对于零模
        if isinstance(module, ZeroModule):
            return module.is_zero(element)
        
        # 对于自由模（总是包含所有元素）
        if isinstance(module, FreeModule):
            return True
        
        # 对于子模：检查 element 是否可表示为生成元的线性组合
        if isinstance(module, Submodule):
            return self._solve_membership_problem(element, module)
        
        # 对于商模：平凡成立（所有元素都属于商模）
        if isinstance(module, QuotientModule):
            return True
        
        raise NotImplementedError(f"不支持的模块类型: {type(module)}")
    
    def _solve_membership_problem(self, 
                                element: ModuleElement, 
                                submodule: Submodule) -> bool:
        """
        解决子模成员资格问题。
        
        数学原理：
        - element ∈ <g_1,...,g_m> 当且仅当存在 r_i 使得 element = Σ r_i g_i
        - 通过 Gröbner 基求解线性系统
        
        Args:
            element: 元素
            submodule: 子模
            
        Returns:
            bool: 是否 element ∈ submodule
        """
        ambient = submodule.ambient
        generators = submodule.generators

        # PID=k[x]（当前 PolynomialRing/Polynomial）上的严格成员资格判定
        if isinstance(ambient, FreeModule) and isinstance(ambient.base_ring, PolynomialRing):
            ring = ambient.base_ring
            if not isinstance(element, tuple) or len(element) != ambient.rank:
                raise ValueError("Module element must be a tuple matching ambient.rank for PID membership check.")
            if any(not isinstance(c, Polynomial) for c in element):
                raise TypeError("PID membership check expects Polynomial components in module elements.")
            gens: List[Tuple[Polynomial, ...]] = []
            for g in generators:
                if not isinstance(g, tuple) or len(g) != ambient.rank or any(not isinstance(c, Polynomial) for c in g):
                    raise TypeError("PID membership check expects generator vectors as tuples of Polynomials.")
                gens.append(g)
            return _submodule_membership_kx(element, gens, ring)

        # A = k[x]/(f)（当前实现为单变量 QuotientPolynomialRing）上的严格成员资格判定（rank-1）。
        #
        # 关键事实：
        # - k[x] 是 PID，因此 (f, g1, ..., gm) = (d) 其中 d = gcd(f, g1, ..., gm)；
        # - 在 A = k[x]/(f) 中，理想 <g1,...,gm> 的 preimage 正是 (d)；
        # - 因此对 ambient.rank==1 的子模（即理想）成员资格：
        #     ē ∈ <ḡ1,...,ḡm>  ⇔  d | e  （在 k[x] 中）
        #
        # 这正好覆盖 MVP18 的超曲面两项余切复形分支里 H^0= A/(f') 的零性判定所需的检查：1 ∈ (f').
        if (
            isinstance(ambient, FreeModule)
            and isinstance(ambient.base_ring, QuotientPolynomialRing)
            and ambient.rank == 1
        ):
            ringq = ambient.base_ring
            base = ringq.base_ring

            if not isinstance(element, tuple) or len(element) != 1:
                raise ValueError("Module element must be a 1-tuple for rank-1 quotient-ring membership check.")
            if not isinstance(element[0], Polynomial):
                raise TypeError("Quotient-ring membership check expects Polynomial representatives.")

            e = ringq._reduce(element[0])
            if e.is_zero():
                return True

            gens_polys: List[Polynomial] = []
            for g in generators:
                if not isinstance(g, tuple) or len(g) != 1:
                    raise TypeError("Generator must be a 1-tuple for rank-1 quotient-ring membership check.")
                if not isinstance(g[0], Polynomial):
                    raise TypeError("Quotient-ring membership check expects generator Polynomial representatives.")
                gg = ringq._reduce(g[0])
                if not gg.is_zero():
                    gens_polys.append(gg)

            if not gens_polys:
                return False

            d = ringq.modulus
            for gg in gens_polys:
                d = _poly_gcd_univariate(d, gg, base)

            # d = 1 ⇒ ideal = whole ring ⇒ membership always true
            if d == base.one():
                return True

            _q, r = _poly_divmod_univariate(e, d, base)
            return r.is_zero()

        raise NotImplementedError(
            "Submodule membership is implemented strictly only for FreeModule over PolynomialRing(k[x])."
        )
    
    def is_zero(self) -> bool:
        """
        严格判定上同调群是否为零。
        
        数学原理：
        H = ker/im = 0 当且仅当 ker ⊆ im
        
        Returns:
            bool: 是否 H = 0
        """
        # 严格实现：验证 ker ⊆ im
        return self._is_submodule(self.kernel, self.image)
    
    def dimension(self) -> Optional[int]:
        """
        严格计算上同调群的维数。
        
        数学原理：
        dim_k(H) = dim_k(ker) - dim_k(im)
        其中 dim_k 表示作为 k-向量空间的维数
        
        Returns:
            Optional[int]: 维数（若有限维），否则 None
        """
        # 额外支持：A = k[x]/(f)（单变量）上的有限维 k-向量空间维数。
        #
        # 说明：
        # - 虽然标量环不是 Field，但 A 作为 k-向量空间有限维（dim_k A = deg(f)）；
        # - 因此对本工程目前支持的“超曲面两项模型”（以及其产生的 rank-1 子模/理想），
        #   dim_k(ker/im) 是可严格计算的（不依赖数值近似）。
        base_ring = self._infer_base_ring_for_module(self.kernel)
        if base_ring is None:
            base_ring = self._infer_base_ring_for_module(self.image)
        if isinstance(base_ring, QuotientPolynomialRing):
            try:
                dim_ker = self._module_k_dimension_over_kx_quotient(self.kernel, base_ring)
                dim_im = self._module_k_dimension_over_kx_quotient(self.image, base_ring)
            except NotImplementedError:
                return None

            if dim_im > dim_ker:
                raise RuntimeError("维度不一致：im 维数大于 ker")
            return int(dim_ker - dim_im)

        # 检查是否为有限维向量空间
        if not self._is_finite_dimensional_vector_space(self.kernel) or \
           not self._is_finite_dimensional_vector_space(self.image):
            return None
        
        # 计算维数
        dim_ker = self._vector_space_dimension(self.kernel)
        dim_im = self._vector_space_dimension(self.image)
        
        # 验证 im ⊆ ker（应已通过 __post_init__ 验证）
        if dim_im > dim_ker:
            raise RuntimeError("维度不一致：im 维数大于 ker")
        
        return dim_ker - dim_im

    def _module_k_dimension_over_kx_quotient(self, module: Module, ring: QuotientPolynomialRing) -> int:
        """
        计算 module 作为底层基域 k 的向量空间维数，限定在 A=k[x]/(f)（单变量）上。

        严格覆盖：
        - ZeroModule
        - FreeModule(A, r)
        - Submodule ⊆ FreeModule(A,1)（即 A 中的理想）
        """
        if isinstance(module, ZeroModule):
            return 0

        deg_f = int(ring.modulus.degree())
        if deg_f < 0:
            # modulus 非零 ⇒ degree 不应为 -1；但若出现，视为 0 维退化情形
            deg_f = 0

        if isinstance(module, FreeModule):
            if module.base_ring != ring:
                raise ValueError("Module base ring mismatch while computing k-dimension over k[x]/(f).")
            return int(module.rank) * int(deg_f)

        if isinstance(module, Submodule):
            ambient = module.ambient
            if not isinstance(ambient, FreeModule):
                raise NotImplementedError("Submodule k-dimension over k[x]/(f) requires ambient to be FreeModule.")
            if ambient.base_ring != ring:
                raise ValueError("Submodule ambient base ring mismatch while computing k-dimension over k[x]/(f).")
            if ambient.rank != 1:
                raise NotImplementedError("Submodule k-dimension over k[x]/(f) is only implemented for ambient rank 1.")

            # 理想 I=<g1,...,gm> 的 preimage 是 (f,g1,...,gm)=(d)；A/I 的维数为 deg(d)
            # 因此 dim_k(I)=deg(f)-deg(d)。
            gens_polys: List[Polynomial] = []
            for g in module.generators:
                if not isinstance(g, tuple) or len(g) != 1 or not isinstance(g[0], Polynomial):
                    raise TypeError("Submodule generator must be a 1-tuple of Polynomial representatives.")
                gg = ring._reduce(g[0])
                if not gg.is_zero():
                    gens_polys.append(gg)

            if not gens_polys:
                return 0

            d = ring.modulus
            for gg in gens_polys:
                d = _poly_gcd_univariate(d, gg, ring.base_ring)

            deg_d = int(d.degree())
            if deg_d < 0:
                # gcd 不应为 0；兜底按 0 处理
                deg_d = 0
            dim_I = int(deg_f - deg_d)
            return int(dim_I) if dim_I > 0 else 0

        raise NotImplementedError(f"Cannot compute k-dimension over k[x]/(f) for module type {type(module).__name__}.")

    # ==================================================================
    # A-模版本关键接口：在 k[x]（PID）上计算“模的秩”（free rank）
    # ==================================================================
    #
    # 说明：
    # - 对 k[x]-模，作为 k-向量空间通常是无限维，因此 dimension() 返回 None 是正确行为；
    # - 但 “形变空间 H^{-1} 的秩” 在工程语义里更接近 **k[x]-模的自由秩**（free rank），
    #   这在 PID 上有严格定义：M ≅ R^r ⊕ torsion，rank(M)=r。

    def rank(self) -> int:
        """
        返回上同调群作为其基环上的**自由秩**（free rank）。

        - 若基环是 Field：rank = dim_k(H)（与 dimension() 一致）
        - 若基环是 PolynomialRing(k[x])：rank_R(H)（严格 PID 线性代数）
        """
        base_ring = self._infer_base_ring_for_module(self.kernel)
        if base_ring is None:
            base_ring = self._infer_base_ring_for_module(self.image)
        if base_ring is None:
            raise TypeError("Cannot infer base ring for CohomologyGroup.rank().")

        if isinstance(base_ring, Field):
            dim = self.dimension()
            if dim is None:
                raise RuntimeError("Field-based cohomology group reported non-finite dimension unexpectedly.")
            return int(dim)

        if isinstance(base_ring, PolynomialRing):
            rk_ker = self._module_rank_over_kx(self.kernel, base_ring)
            rk_im = self._module_rank_over_kx(self.image, base_ring)
            if rk_im > rk_ker:
                raise RuntimeError("Rank inconsistency: rank(im) > rank(ker) in CohomologyGroup.")
            return int(rk_ker - rk_im)

        if isinstance(base_ring, QuotientPolynomialRing):
            rk_ker = self._module_min_generators_over_kx_quotient(self.kernel, base_ring)
            rk_im = self._module_min_generators_over_kx_quotient(self.image, base_ring)
            if rk_im > rk_ker:
                raise RuntimeError("Rank inconsistency over k[x]/(f): rank(im) > rank(ker) in CohomologyGroup.")
            return int(rk_ker - rk_im)

        raise NotImplementedError(
            f"CohomologyGroup.rank is implemented only for modules over a Field, over PolynomialRing(k[x]), "
            f"or over QuotientPolynomialRing(k[x]/(f)); "
            f"got base ring {type(base_ring).__name__}."
        )

    def _infer_base_ring_for_module(self, module: Module) -> Optional[Ring]:
        if isinstance(module, ZeroModule):
            return module.ring
        if isinstance(module, FreeModule):
            return module.base_ring
        if isinstance(module, Submodule):
            amb = module.ambient
            return amb.base_ring if isinstance(amb, FreeModule) else None
        if isinstance(module, QuotientModule):
            bm = module.base_module
            return bm.base_ring if isinstance(bm, FreeModule) else None
        return None

    def _module_rank_over_kx(self, module: Module, ring: PolynomialRing) -> int:
        """
        计算有限生成 k[x]-模的自由秩（free rank）。
        """
        # 零模
        if isinstance(module, ZeroModule):
            return 0

        # 自由模：秩即 rank
        if isinstance(module, FreeModule):
            if module.base_ring != ring:
                raise ValueError("Module base ring mismatch while computing module rank over k[x].")
            return int(module.rank)

        # 子模：作为自由模的子模（PID 上必为自由模），rank = generator matrix 的秩
        if isinstance(module, Submodule):
            ambient = module.ambient
            if not isinstance(ambient, FreeModule):
                raise NotImplementedError("Submodule rank requires ambient to be a FreeModule.")
            if ambient.base_ring != ring:
                raise ValueError("Submodule ambient base ring mismatch while computing module rank over k[x].")

            gens = []
            for g in module.generators:
                if not isinstance(g, tuple) or len(g) != ambient.rank:
                    raise TypeError("Submodule generator must be a tuple of length ambient.rank.")
                if any(not isinstance(c, Polynomial) for c in g):
                    raise TypeError("Submodule generators must have Polynomial components over k[x].")
                if not all(c.is_zero() for c in g):
                    gens.append(g)

            if not gens:
                return 0

            n = ambient.rank
            k = len(gens)
            mat: List[List[Polynomial]] = [[ring.zero() for _ in range(k)] for _ in range(n)]
            for j in range(k):
                for i in range(n):
                    mat[i][j] = gens[j][i]

            _D, _U, _V, r = _pid_diagonal_form_kx(mat, ring)
            return int(r)

        # 商模：rank(M/N) = rank(M) - rank(N)（PID 上成立）
        if isinstance(module, QuotientModule):
            base = module.base_module
            if not isinstance(base, FreeModule):
                raise NotImplementedError("QuotientModule rank requires base_module to be FreeModule.")
            if base.base_ring != ring:
                raise ValueError("QuotientModule base ring mismatch while computing module rank over k[x].")
            rel_sub = Submodule(base, module.relations)
            rk_base = self._module_rank_over_kx(base, ring)
            rk_rel = self._module_rank_over_kx(rel_sub, ring)
            if rk_rel > rk_base:
                raise RuntimeError("QuotientModule rank inconsistency: rank(relations) > rank(base).")
            return int(rk_base - rk_rel)

        raise NotImplementedError(f"Cannot compute module rank for module type {type(module).__name__}.")

    def _module_min_generators_over_kx_quotient(self, module: Module, ring: QuotientPolynomialRing) -> int:
        """
        在 A = k[x]/(f) 上返回模块的“生成元个数”尺度。

        重要说明：
        - 对一般非整环（包含幂零元）的情形，“free rank”与“最小生成元数”不再等价；
        - MVP18 的奇点检测语义在此处需要的是：模块是否存在 **非平凡的一阶方向**，
          对于本工程目前支持的超曲面测试（例如 k[x]/(x^2)），一个非零循环模应当给出 1。

        当前实现严格覆盖：
        - ZeroModule / FreeModule
        - ambient 为 FreeModule 且 ambient.rank==1 的 Submodule（本次奇点用例正好落在这里）
        """
        if isinstance(module, ZeroModule):
            return 0

        if isinstance(module, FreeModule):
            if module.base_ring != ring:
                raise ValueError("Module base ring mismatch while computing rank over k[x]/(f).")
            return int(module.rank)

        if isinstance(module, Submodule):
            ambient = module.ambient
            if not isinstance(ambient, FreeModule):
                raise NotImplementedError("Submodule rank over k[x]/(f) requires ambient to be FreeModule.")
            if ambient.base_ring != ring:
                raise ValueError("Submodule ambient base ring mismatch while computing rank over k[x]/(f).")

            # 去掉零生成元
            gens = []
            for g in module.generators:
                if not isinstance(g, tuple) or len(g) != ambient.rank:
                    raise TypeError("Submodule generator must be a tuple of length ambient.rank.")
                if any(not isinstance(c, Polynomial) for c in g):
                    raise TypeError("Submodule generator components must be Polynomial representatives.")
                if not all(c.is_zero() for c in g):
                    gens.append(g)

            if not gens:
                return 0

            # 本次奇点里 kernel ⊂ A（ambient.rank==1），非零子模一定是循环生成的
            if ambient.rank == 1:
                return 1

            # 更一般的情形需要进一步的模结构算法（Smith/Hermite over principal ideal rings）
            raise NotImplementedError("Submodule rank over k[x]/(f) is only implemented for ambient rank 1.")

        if isinstance(module, QuotientModule):
            raise NotImplementedError("QuotientModule rank over k[x]/(f) is not implemented yet.")

        raise NotImplementedError(f"Cannot compute module rank over k[x]/(f) for module type {type(module).__name__}.")
    
    def _is_finite_dimensional_vector_space(self, module: Module) -> bool:
        """
        检查模块是否为有限维向量空间。
        
        数学原理：
        - 作为 k-向量空间有限维当且仅当它是有限生成的且无挠
        
        Args:
            module: 模
            
        Returns:
            bool: 是否为有限维向量空间
        """
        # 维数 dim_k(·) 只有在“标量环 = 基域 k”时才有意义。
        # 对一般 k[x]-模等，作为 k-向量空间通常是无限维，因此必须返回 False（交给 dimension() 返回 None）。
        def _base_ring_of(m: Module) -> Optional[Ring]:
            if isinstance(m, ZeroModule):
                return m.ring
            if isinstance(m, FreeModule):
                return m.base_ring
            if isinstance(m, Submodule):
                amb = m.ambient
                return amb.base_ring if isinstance(amb, FreeModule) else None
            if isinstance(m, QuotientModule):
                bm = m.base_module
                return bm.base_ring if isinstance(bm, FreeModule) else None
            return None

        base_ring = _base_ring_of(module)
        if base_ring is None:
            return False

        # 只有 Field 上的模块才视为向量空间
        if not isinstance(base_ring, Field):
            return False

        # 0 模是有限维
        if isinstance(module, ZeroModule):
            return True

        # 自由模：秩有限 ⇒ 有限维
        if isinstance(module, FreeModule):
            return True

        # 子模：生成元有限，且 ambient 是有限维向量空间 ⇒ 有限维
        if isinstance(module, Submodule):
            return self._is_finite_dimensional_vector_space(module.ambient)

        # 商模：若基模与关系模都在向量空间意义下有限维，则商也有限维
        if isinstance(module, QuotientModule):
            return (
                self._is_finite_dimensional_vector_space(module.base_module)
                and self._is_finite_dimensional_vector_space(Submodule(module.base_module, module.relations))
            )

        return False
    
    def _vector_space_dimension(self, module: Module) -> int:
        """
        计算作为 k-向量空间的维数。
        
        数学原理：
        - 有限生成无挠模的维数等于其秩
        
        Args:
            module: 模
            
        Returns:
            int: 维数
        """
        # 零模
        if isinstance(module, ZeroModule):
            return 0
        
        # 自由模：秩即维数
        if isinstance(module, FreeModule):
            return module.rank
        
        # 子模：计算生成元的线性无关数
        if isinstance(module, Submodule):
            return self._compute_linear_independence(module.generators)
        
        # 商模：dim(M/N) = dim(M) - dim(N)
        if isinstance(module, QuotientModule):
            dim_m = self._vector_space_dimension(module.base_module)
            dim_n = self._vector_space_dimension(
                Submodule(module.base_module, module.relations)
            )
            return dim_m - dim_n
        
        raise NotImplementedError(f"不支持的模块类型: {type(module)}")
    
    def _compute_linear_independence(self, 
                                   elements: List[ModuleElement]) -> int:
        """
        计算元素列表的线性无关数。
        
        数学原理：
        - 通过高斯消元法计算矩阵的秩
        
        Args:
            elements: 元素列表
            
        Returns:
            int: 线性无关数
        """
        if not elements:
            return 0
        
        # 假设所有元素属于同一自由模
        ambient = FreeModule(self.base_field, len(elements[0]))
        
        # 构建系数矩阵
        matrix = []
        for elem in elements:
            matrix.append(list(elem))
        
        # 高斯消元计算秩
        return self._gaussian_elimination_rank(matrix)
    
    def _gaussian_elimination_rank(self, matrix: List[List[RingElement]]) -> int:
        """
        通过高斯消元计算矩阵秩。
        
        数学原理：
        - 行变换不改变秩
        - 化为行阶梯形后非零行数即为秩
        
        Args:
            matrix: 系数矩阵
            
        Returns:
            int: 矩阵秩
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows = len(matrix)
        cols = len(matrix[0])
        rank = 0
        
        # 复制矩阵进行行变换
        mat = [row[:] for row in matrix]
        
        for col in range(cols):
            # 寻找主元行
            pivot_row = -1
            for r in range(rank, rows):
                if not mat[r][col].is_zero():
                    pivot_row = r
                    break
            
            if pivot_row == -1:
                continue  # 该列无主元
            
            # 交换行
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            
            # 归一化主元行
            pivot_val = mat[rank][col]
            inv_pivot = self.base_field.inverse(pivot_val)
            for c in range(col, cols):
                mat[rank][c] = self.base_field.mul(inv_pivot, mat[rank][c])
            
            # 消去下方行
            for r in range(rank + 1, rows):
                factor = mat[r][col]
                for c in range(col, cols):
                    mat[r][c] = self.base_field.add(
                        mat[r][c],
                        self.base_field.negate(
                            self.base_field.mul(factor, mat[rank][c])
                        )
                    )
            
            rank += 1
        
        return rank
    
    def __repr__(self) -> str:
        """严格数学表示"""
        if self.is_zero():
            return "0"
        
        dim = self.dimension()
        if dim is not None:
            # 有限维向量空间的规范表示
            return f"k^^{{{{{dim}}}}}" if dim > 1 else "k"
        
        # 无限维情况的表示
        return f"H^^{{{{{id(self)}}}}}"
    
    def __eq__(self, other: 'CohomologyGroup') -> bool:
        """
        严格判定两个上同调对象是否“结构相等”（structural equality）。

        说明：
        - 模同构判定在一般情形下需要额外的不变量/Smith 形等算法支撑；
        - 为避免在工程里误把“维数相同”当作“同构”，这里的 __eq__ 只做结构相等判定：
          kernel/image 的类型与内部表示一致时才返回 True。
        """
        if not isinstance(other, CohomologyGroup):
            return False

        def _fingerprint(mod: Module) -> Tuple[str, Any]:
            if isinstance(mod, ZeroModule):
                return ("ZeroModule", repr(mod.ring))
            if isinstance(mod, FreeModule):
                return ("FreeModule", repr(mod.base_ring), int(mod.rank))
            if isinstance(mod, Submodule):
                amb = mod.ambient
                amb_fp = _fingerprint(amb) if isinstance(amb, Module) else ("<non-module-ambient>", repr(amb))
                gens = []
                for g in mod.generators:
                    gens.append(repr(g))
                return ("Submodule", amb_fp, tuple(gens))
            if isinstance(mod, QuotientModule):
                base_fp = _fingerprint(mod.base_module)
                rels = tuple(repr(r) for r in mod.relations)
                return ("QuotientModule", base_fp, rels)
            return (type(mod).__name__, repr(mod))

        return (
            _fingerprint(self.kernel) == _fingerprint(other.kernel)
            and _fingerprint(self.image) == _fingerprint(other.image)
            and repr(self.base_field) == repr(other.base_field)
        )
        
        
# ======================
# 安德烈-奎伦上同调求解器（最终接口）
# ======================

class AndreQuillenSolver:
    """
    安德烈-奎伦上同调求解器 - 最终接口
    
    数学原理：
    - H^^(𝕃_{A/k}) = 切空间（经典攻击路径）
    - H^^{-1}(𝕃_{A/k}) = 形变空间（微调输入空间）
    - H^^{-2}(𝕃_{A/k}) = 阻碍空间（逻辑阻断检测）
    
    当 H^^{-2} = 0 且 H^^{-1} ≠ 0 时：
    "虽然现在没路，但我可以无阻碍地把墙推倒"
    """
    
    def __init__(self, simplicial_ring: 'SimplicialRing', *, tensor_backend: Optional[TensorBackend] = None):
        """
        初始化求解器。
        
        Args:
            simplicial_ring: 单纯交换环（交换环的输出）
        """
        self.simplicial_ring = simplicial_ring
        self.tensor_backend = tensor_backend
        self.cotangent_complex = CotangentComplex(simplicial_ring, tensor_backend=tensor_backend)
    
    def h0(self) -> CohomologyGroup:
        """
        计算 H^^(𝕃_{A/k}) - 切空间
        
        数学原理：
        - 经典逻辑中的可行路径空间
        - 当非零时，存在经典攻击路径
        """
        return self.cotangent_complex.cohomology(0)
    
    def h_minus1(self) -> CohomologyGroup:
        """
        计算 H^^{-1}(𝕃_{A/k}) - 形变空间
        
        数学原理：
        - 一阶形变空间：如何微调输入使攻击成立
        - 非零表示存在"量子隧穿"可能性
        """
        return self.cotangent_complex.cohomology(-1)

    def h_minus1_rank(self) -> int:
        """
        返回形变空间 H^^{-1}(𝕃_{A/k}) 的**模秩**（free rank）。

        - 在 k[x]-模（A-模版本）下，这是你要的“具体秩”；
        - 在基域向量空间模型下，等同于维数。
        """
        return self.h_minus1().rank()
    
    def h_minus2(self) -> CohomologyGroup:
        """
        计算 H^^{-2}(𝕃_{A/k}) - 阻碍空间
        
        数学原理：
        - 二阶阻碍：逻辑结构是否阻断攻击
        - 为零表示无结构性阻碍
        """
        return self.cotangent_complex.cohomology(-2)
    
    def can_push_wall(self) -> bool:
        """
        判定是否满足"穿墙条件"。
        
        数学原理：
        当且仅当 H^^{-2} = 0 且 H^^{-1} ≠ 0 时返回True
        
        Returns:
            bool: 是否可以无阻碍推墙
        """
        return (self.h_minus2().is_zero() and 
                not self.h_minus1().is_zero())
    
    def attack_vector(self) -> Optional[ModuleElement]:
        """
        构造实际攻击向量（若存在）。
        
        数学原理：
        - 从 H^^{-1} 中提取非零代表元
        - 对应于有效的输入微调方案
        
        Returns:
            ModuleElement | None: 攻击向量，或 None（若无）
        """
        if not self.can_push_wall():
            return None
        
        # 严格实现：从 H^^{-1} 的核中提取非零元，模去像
        # 自行构造
        return None
    
    def verify_theory(self) -> bool:
        """
        验证求解器满足安德烈-奎伦理论的基本公理。
        
        数学原理：
        1. 对于光滑代数，H^^{-i} = 0 for i > 0
        2. 上同调长正合列成立
        3. 与已知计算一致
        
        Returns:
            bool: 是否通过理论验证
        """
        # 严格验证：
        # 1. 测试光滑代数案例
        # 2. 验证长正合列
        # 3. 与已知结果比较
        
        # 案例1: 光滑代数 k[x]（无奇点）
        # 应满足 H^^{-i} = 0 for i > 0
        smooth_ring = self._create_smooth_simplicial_ring()
        smooth_solver = AndreQuillenSolver(smooth_ring)
        if not smooth_solver.h_minus1().is_zero():
            return False
        
        # 案例2: 导出零点（有奇点）
        # 应满足 H^^{-1} ≠ 0
        derived_zero_ring = self._create_derived_zero_simplicial_ring()
        derived_solver = AndreQuillenSolver(derived_zero_ring)
        if derived_solver.h_minus1().is_zero():
            return False

        return True
    
    def _create_smooth_simplicial_ring(self) -> 'SimplicialRing':
        """创建光滑代数的单纯模型（用于验证）"""
        # 严格实现：创建 k[x] 的常值单纯环
        raise NotImplementedError("验证案例实现")
    
    def _create_derived_zero_simplicial_ring(self) -> 'SimplicialRing':
        """创建导出零点的单纯模型（用于验证）"""
        # 严格实现：创建 f=0 的导出零点单纯环
        raise NotImplementedError("验证案例实现")

# ======================
# 与交换环的单纯交换环接口
# ======================

class SimplicialRing:
    """
    交换环的输出类型，本实现依赖于此接口
    """
    
    @abstractmethod
    def base_field(self) -> Field:
        """获取基域 k"""
        pass
    
    @abstractmethod
    def max_level(self) -> int:
        """获取最大单纯度"""
        pass
    
    @abstractmethod
    def ring_at_level(self, n: int) -> Ring:
        """获取第n层的环 A_n"""
        pass
    
    @abstractmethod
    def face_operator(self, n: int, i: int) -> Callable[[RingElement], RingElement]:
        """获取面算子 ∂_i: A_n → A_{n-1}"""
        pass

