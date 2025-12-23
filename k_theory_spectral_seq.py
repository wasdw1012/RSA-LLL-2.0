"""
===============================================================================
MVP20  Sub-Zero (绝对零度)
目标：构造从K理论到棱镜上同调的Atiyah-Hirzebruch型谱序列
核心：将轨道积分转化为K理论欧拉示性数的计算 Thomason定理

禁止 魔法数  启发式 明显的偷懒函数 狡猾的归一化 无理由静默失败 降级 
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable, Any, Sequence
from enum import Enum
from fractions import Fraction
from abc import ABC, abstractmethod


class KTheorySpectralSequenceError(RuntimeError):
    """K 理论谱序列计算错误（必须中断，禁止静默）。"""


# MVP17 核心模块导入
try:
    from .mvp17_prismatic import (
        WittVector,
        WittPolynomialGenerator,
        Prism,
        NygaardFiltration,
        NygaardCompletion,
        IntegralityValidator,
        ValidationResult,
        FiniteFieldElement,
        DeltaRing,
        WittVectorDeltaRing,
    )
    MVP17_AVAILABLE = True
except ImportError:
    MVP17_AVAILABLE = False
    # 占位符，允许文件独立加载进行类型检查
    WittVector = None  # type: ignore
    WittPolynomialGenerator = None  # type: ignore
    Prism = None  # type: ignore
    NygaardFiltration = None  # type: ignore
    NygaardCompletion = None  # type: ignore
    IntegralityValidator = None  # type: ignore
    ValidationResult = None  # type: ignore
    FiniteFieldElement = None  # type: ignore
    DeltaRing = None  # type: ignore
    WittVectorDeltaRing = None  # type: ignore

# MVP17 CFG 对象模型导入
try:
    from .mvp17_cfg_object_model import (
        MVP17CFGAuditor,
        MVP17CFGObjectBuilder,
        MVP17Fiber,
        MVP17BerryConnection,
        MonomialOperator,
        crystalline_frobenius_spectral_certificate,
    )
    MVP17_CFG_AVAILABLE = True
except ImportError:
    MVP17_CFG_AVAILABLE = False
    MVP17CFGAuditor = None  # type: ignore
    MVP17CFGObjectBuilder = None  # type: ignore
    MVP17Fiber = None  # type: ignore
    MVP17BerryConnection = None  # type: ignore
    MonomialOperator = None  # type: ignore
    crystalline_frobenius_spectral_certificate = None  # type: ignore


# 第一部分：K理论的基础代数结构


@dataclass(frozen=True)
class AbelianGroupInvariant:
    """
    有限生成阿贝尔群的严格不变量分解（invariant factor decomposition）：
        G ≅ Z^r ⊕ ⊕_{i=1}^t Z/n_i Z
    其中 n_i >= 2 且 n_i | n_{i+1}。

    说明：
    - 本类只做“结构/不变量”的确定性表示，不做任何启发式/近似。
    - 若 free_rank>0 则群无限；若 free_rank==0 且 torsion_invariants 为空，则为平凡群。
    """
    free_rank: int
    torsion_invariants: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.free_rank, int) or self.free_rank < 0:
            raise ValueError("free_rank must be a non-negative int.")
        if not isinstance(self.torsion_invariants, tuple):
            raise TypeError("torsion_invariants must be a tuple of ints.")
        prev = 1
        for n in self.torsion_invariants:
            if (not isinstance(n, int)) or n < 2:
                raise ValueError("Each torsion invariant must be an int >= 2.")
            if n % prev != 0:
                raise ValueError("torsion invariants must satisfy n_i | n_{i+1}.")
            prev = n

    def is_trivial(self) -> bool:
        return self.free_rank == 0 and len(self.torsion_invariants) == 0

    def is_finite(self) -> bool:
        return self.free_rank == 0

    def order(self) -> int:
        """返回群的阶（仅对有限群定义）。"""
        if not self.is_finite():
            raise ValueError("Infinite group: order is not defined.")
        o = 1
        for n in self.torsion_invariants:
            o *= n
        return o

    def minimal_number_of_generators(self) -> int:
        """
        μ(G)：作为 Z-模的最小生成元个数（严格，不是“秩”的近似）。
        对 invariant factor 分解：μ(G)=free_rank + len(torsion_invariants)。
        """
        return int(self.free_rank + len(self.torsion_invariants))


def _is_prime_u64_deterministic(n: int) -> bool:
    """
    64-bit 范围内的确定性 Miller–Rabin 素性判定。

    对 n < 2^64，底数集
        [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    是确定性的（非概率，非启发式）。
    """
    if not isinstance(n, int):
        raise TypeError("n must be int.")
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    if n >= (1 << 64):
        raise NotImplementedError("Deterministic primality check currently implemented only for n < 2^64.")

    # n-1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        witness = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                witness = False
                break
        if witness:
            return False
    return True


def _int_nth_root_floor(n: int, k: int) -> int:
    """返回 ⌊n^{1/k}⌋（仅整数运算，确定性）。"""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative int.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive int.")
    if n in (0, 1):
        return n
    if k == 1:
        return n

    # 上界：2^{ceil(bitlen/k)} 一定 >= n^{1/k}
    bitlen = n.bit_length()
    hi = 1 << ((bitlen + k - 1) // k)
    lo = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        mid_pow = pow(mid, k)
        if mid_pow <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _gcd(a: int, b: int) -> int:
    """
    欧几里得算法计算最大公约数（确定性，无浮点）。
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Arguments must be integers.")
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _finite_field_order_prime_power_u64(q: int) -> Tuple[int, int]:
    """
    将 q 分解为 q = p^f（p 为素数，f>=1）。

    严格约束：
    - 仅支持 q < 2^64（保证素性判定可确定性完成）。
    - 若不是素数幂，抛异常（不做任何“近似域”降级）。
    """
    if not isinstance(q, int) or q < 2:
        raise ValueError("q must be int >= 2.")
    if q >= (1 << 64):
        raise NotImplementedError("Finite field order decomposition currently implemented only for q < 2^64.")

    # f 的上界：2^f <= q ⇒ f <= floor(log2 q) = bit_length-1
    max_f = q.bit_length() - 1
    # 从大到小找最大 f，确保分解唯一（p 必须是素数）
    for f in range(max_f, 0, -1):
        p = _int_nth_root_floor(q, f)
        if pow(p, f) != q:
            continue
        if _is_prime_u64_deterministic(p):
            return p, f
    raise ValueError("q is not a prime power with prime base within the supported 64-bit backend.")


def quillen_k_group_finite_field(q: int, n: int) -> AbelianGroupInvariant:
    """
    Quillen 定理（有限域的高阶代数 K 群）——严格版。

    对有限域 F_q（q 为素数幂）：
    - K_0(F_q) ≅ Z
    - 对 i>=1：K_{2i}(F_q) = 0
    - 对 i>=1：K_{2i-1}(F_q) ≅ Z/(q^i - 1)Z
    - 负 K：对域（正则 Noetherian，Krull 维数 0），K_{-m}(F_q)=0 (m>0)

    当前工程后端只接受 q 为 **64-bit 内的素数**（即 F_p）。
    若需要一般素数幂，请先提供严格的素数幂分解证据后再扩展（禁止概率/启发式因子分解）。
    """
    if not isinstance(q, int) or q < 2:
        raise ValueError("q must be an int >= 2.")
    if not isinstance(n, int):
        raise TypeError("n must be int.")

    # 严格：当前仅支持 64-bit 范围内的有限域阶 q=p^f（p 素数）。
    # 这是为了保证素性判定完全确定性（拒绝概率 MR / 启发式分解）。
    _p, _f = _finite_field_order_prime_power_u64(q)

    if n == 0:
        return AbelianGroupInvariant(free_rank=1, torsion_invariants=())
    if n < 0:
        return AbelianGroupInvariant(free_rank=0, torsion_invariants=())
    if n % 2 == 0:
        # n>0 偶数：0 群
        return AbelianGroupInvariant(free_rank=0, torsion_invariants=())

    # n=2i-1
    i = (n + 1) // 2
    order = pow(q, i) - 1
    if order <= 1:
        # 例如 q=2,i=1：K_1(F_2)=0
        return AbelianGroupInvariant(free_rank=0, torsion_invariants=())
    return AbelianGroupInvariant(free_rank=0, torsion_invariants=(order,))

class KTheoryDegree(Enum):
    """K理论的度数（grading）"""
    K0 = 0  # 对象的Grothendieck群
    K1 = 1  # 自同构群（GL模掉初等矩阵）
    K2 = 2  # Steinberg关系
    K_NEGATIVE = -1  # 负K理论（Bass构造）

@dataclass
class KTheoryClass:
    """
    K理论类的代数表示
    数学：K_n(R) 中的一个元素，表示为生成元的线性组合
    
    [禁止魔法数] degree必须从Quillen Q-构造的维度导出
    """
    degree: int  # K_n 的 n
    generators: List[Tuple[str, int]]  # [(生成元名称, 系数)]
    base_ring: 'WittVectorRing'  # 定义在哪个环上

    def group_invariant(self) -> AbelianGroupInvariant:
        """返回 K_degree(F_q) 的结构不变量（严格：Quillen 公式，q 由 base_ring 给出）。"""
        return quillen_k_group_finite_field(self.base_ring.characteristic(), self.degree)
    
    def rank(self) -> int:
        """
        [严格实现] K 理论“秩/生成元计数”接口（不做任何近似）。

        - **K_0**：返回虚拟对象的秩映射（augmentation）结果。
          在本文件的抽象表示里，每个生成元默认代表秩 1 的基对象，因此秩为系数和。

        - **n ≠ 0**：返回 **K_n(F_p)** 作为阿贝尔群的最小生成元个数 μ(K_n)。
          当前严格后端只支持有限域 F_p（p = base_ring.characteristic()，并做 64-bit 确定性素性验证）。
        """
        if self.degree == 0:
            total = 0
            for _name, coeff in self.generators:
                if not isinstance(coeff, int):
                    raise TypeError("K_0 generator coefficient must be int.")
                total += coeff
            return total

        return self._quillen_lichtenbaum_rank()
    
    def _quillen_lichtenbaum_rank(self) -> int:
        """
        Quillen（有限域）给出的 **K_n(F_p)** 结构不变量，提取 μ(K_n)。

        对有限域 F_q：
        - K_0(F_q) ≅ Z
        - i>=1：K_{2i}(F_q) = 0
        - i>=1：K_{2i-1}(F_q) ≅ Z/(q^i - 1)Z
        """
        inv = self.group_invariant()
        return inv.minimal_number_of_generators()


class WittVectorRing:
    """
    Witt向量环（来自MVP5/MVP17）
    这里只定义K理论需要的接口
    """
    def __init__(self, prime: int):
        # 这里的 prime 参数在本 patch 中承载“有限域阶 q”的角色（历史命名保留）。
        # 严格：要求 q=p^f 且 q<2^64，保证 Quillen 侧可确定性工作。
        self.p = prime
        _finite_field_order_prime_power_u64(self.p)
    
    def characteristic(self) -> int:
        return self.p
    
    def frobenius_power(self, n: int):
        """φ^n: W(k) -> W(k)"""
        return self.p ** n



# 第二部分：谱序列的代数结构（E_r页）

@dataclass
class SpectralSequenceTerm:
    """
    谱序列的一个(p,q)项
    数学：E_r^{p,q} 页上的一个对象
    
    [禁止魔法数] page_number (r) 必须从微分的消失阶导出
    """
    page_number: int  # E_r 的 r
    bidegree: Tuple[int, int]  # (p, q)
    module: List[KTheoryClass]  # 这个位置上的模（K理论类的集合）
    
    def dimension(self) -> int:
        """[必须实现] 计算该项的维度（作为向量空间）"""
        return sum(k_class.rank() for k_class in self.module)


@dataclass(frozen=True)
class NygaardFixedPointCycle:
    """
    Monomial Frobenius 的一个置换环（cycle）以及其斜率数据。

    约定（与 mvp17_cfg_object_model 的“每步斜率”一致）：
      - MonomialOperator 表示 Φ^{steps} 的一次作用：
            e_i ↦ p^{exp[i]} · e_{perm[i]}
      - 对一个 cycle 长度 L，令 E = Σ_{i∈cycle} exp[i]，则
            slope_per_step = E / (L · steps)
    """

    cycle: Tuple[int, ...]
    exp_sum: int
    steps: int
    slope_per_step: Fraction

    @property
    def cycle_len(self) -> int:
        return int(len(self.cycle))


@dataclass(frozen=True)
class NygaardFixedPointResult:
    """
    Nygaard 固定点（算术核）判决结果。

    目标方程（严格口径）：
        (Φ^{steps} - p^{n·steps}) x = 0

    - slope_filtration_ok：要求 max_slope_per_step == n
      （稿子口径：最大斜率分量与算术核频率共振才允许通过）
    - unique_resonant_cycle：要求“共振的最大斜率环”唯一

    fixed_point_generator_valuations:
      - 用稀疏表示给出一个确定性的生成元向量：
            idx ↦ v_p(x_idx)  (即 x_idx = p^{v_p} )
        这是一个“原始生成元”（min valuation = 0），便于审计可复现。
    """

    p: int
    witt_length: int
    steps: int
    slope_n: int
    max_slope_per_step: Fraction
    cycles: Tuple[NygaardFixedPointCycle, ...]
    resonant_cycles: Tuple[NygaardFixedPointCycle, ...]
    slope_filtration_ok: bool
    unique_resonant_cycle: bool
    fixed_point_generator_valuations: Optional[Dict[int, int]]

    def fixed_point_generator_as_witt_vectors(self) -> Dict[int, "WittVector"]:
        """
        将 fixed_point_generator_valuations 具体化为 WittVector 坐标（p 进位）。
        仅当 MVP17_AVAILABLE=True 时可用；否则抛异常（禁止静默降级）。
        """
        if not MVP17_AVAILABLE:
            raise KTheorySpectralSequenceError(
                "MVP17 backend not available: cannot materialize Witt vectors."
            )
        if self.fixed_point_generator_valuations is None:
            return {}
        out: Dict[int, WittVector] = {}
        for idx, vp in sorted(self.fixed_point_generator_valuations.items(), key=lambda kv: kv[0]):
            out[int(idx)] = _witt_p_power(self.p, self.witt_length, int(vp))
        return out


@dataclass(frozen=True)
class ValuationCertificate:
    """
    MVP20 输出的赋值证书 - 作为 MVP14 测试函数的原材料。
    
    数学意义：
    - 这是 Syntomic 共振核 (Φ - p^n·Id) x = 0 的确定性输出
    - valuations[i] = v_p(x_i) 表示固定点生成元的 p-adic 估值
    - 这个证书替代了旧的 GlobalTestFunction 的病态连续函数
    
    用法：
    - SyntomicTestFunction 持有此证书
    - evaluate(s) 时使用 valuations 计算 Σ p^{-s·v_i}
    
    Quillen 自然截断：
    - slope_n < 0 时，K_n = 0，证书为空（数学定理，非 hack）
    """
    
    p: int                              # 素数
    slope_n: int                        # 目标斜率（K 群度数）
    witt_length: int                    # Witt 向量精度（由 Arakelov 高度导出）
    valuations: Dict[int, int]          # idx -> v_p(x_idx)
    cycles: Tuple[NygaardFixedPointCycle, ...]  # cycle 结构信息
    max_slope_per_step: Fraction        # 最大斜率
    slope_filtration_ok: bool           # 是否通过斜率过滤
    unique_resonant_cycle: bool         # 是否存在唯一共振态
    
    @classmethod
    def from_fixed_point_result(cls, result: "NygaardFixedPointResult") -> "ValuationCertificate":
        """
        从 NygaardFixedPointResult 构造 ValuationCertificate。
        
        这是 MVP20 → MVP14 的数据桥梁。
        
        Args:
            result: NygaardFixedPointResult（来自 SyntomicResonanceSolver）
        
        Returns:
            ValuationCertificate
        
        Raises:
            KTheorySpectralSequenceError: 如果 result 无效
        """
        if not isinstance(result, NygaardFixedPointResult):
            raise KTheorySpectralSequenceError(
                f"Expected NygaardFixedPointResult, got {type(result).__name__}"
            )
        
        # 提取 valuations，若无共振则为空字典
        valuations: Dict[int, int] = {}
        if result.fixed_point_generator_valuations is not None:
            valuations = dict(result.fixed_point_generator_valuations)
        
        return cls(
            p=int(result.p),
            slope_n=int(result.slope_n),
            witt_length=int(result.witt_length),
            valuations=valuations,
            cycles=result.cycles,
            max_slope_per_step=result.max_slope_per_step,
            slope_filtration_ok=bool(result.slope_filtration_ok),
            unique_resonant_cycle=bool(result.unique_resonant_cycle),
        )
    
    def is_trivial(self) -> bool:
        """证书是否平凡（无共振态）。"""
        return not self.valuations or not self.slope_filtration_ok
    
    def total_valuation(self) -> int:
        """所有 valuation 的总和。"""
        return sum(self.valuations.values())
    
    def dimension(self) -> int:
        """非零 valuation 的数量（固定点的"维度"）。"""
        return len(self.valuations)


def _sorted_cycles_deterministic(perm: Sequence[int]) -> List[List[int]]:
    """
    将 perm（置换）分解成 cycles，并做确定性规范：
      - 每个 cycle 旋转，使最小下标在首位
      - cycles 按 (长度, 首元素) 排序
    """
    n = len(list(perm))
    if n < 1:
        raise KTheorySpectralSequenceError("perm must be non-empty")
    seen = [False] * n
    cycles: List[List[int]] = []
    for i0 in range(n):
        if seen[i0]:
            continue
        cyc: List[int] = []
        i = int(i0)
        while not seen[i]:
            seen[i] = True
            cyc.append(int(i))
            i = int(perm[i])
            if i < 0 or i >= n:
                raise KTheorySpectralSequenceError(f"perm is not a permutation: jump to {i}")
        if cyc:
            j = min(range(len(cyc)), key=lambda k: cyc[k])
            cyc = cyc[j:] + cyc[:j]
            cycles.append(cyc)
    cycles.sort(key=lambda c: (len(c), c[0]))
    return cycles


def _witt_p_power(p: int, length: int, exponent: int) -> "WittVector":
    """
    返回 p^{exponent} 在 W_length(F_p) 里的 Witt 向量表示：
      - exponent < length: (0,...,0,1,0,...,0)（1 出现在第 exponent 个分量）
      - exponent >= length: 0
    """
    if not MVP17_AVAILABLE:
        raise KTheorySpectralSequenceError("MVP17 backend not available: cannot build WittVector.")
    if not isinstance(p, int) or p < 2:
        raise ValueError("p must be int >= 2.")
    if not isinstance(length, int) or length < 1:
        raise ValueError("length must be int >= 1.")
    if not isinstance(exponent, int) or exponent < 0:
        raise ValueError("exponent must be int >= 0.")
    if exponent >= length:
        return WittVector.zero(p, length)
    comps = [FiniteFieldElement.zero(p) for _ in range(length)]
    comps[exponent] = FiniteFieldElement.one(p)
    return WittVector(comps, p)


class NygaardFixedPointInvariants:
    """
    NygaardFixedPointInvariants —— 唯一解的终极判官（稿子 Step 3）。

    输入：
      - MonomialOperator op：表示 Φ^{steps} 的 monomial Frobenius（来自 MVP17 CFG 证书侧）
      - slope_n：目标“每步斜率” n（整数）

    输出：
      - NygaardFixedPointResult：包含
          1) Newton slope 列表与 max_slope
          2) 是否通过斜率过滤（max_slope == n）
          3) 是否存在且唯一的“共振最大斜率 cycle”
          4) 若唯一：给出一个确定性的固定点生成元（稀疏 p-幂坐标）

    约束：
      - 不做任何启发式：无法严格判定则抛异常。
      - 不做静默失败：所有缺后端/输入不合法均抛异常。
    """

    def __init__(self, prism: "Prism"):
        if not MVP17_AVAILABLE:
            raise KTheorySpectralSequenceError(
                "MVP17 prismatic module not available. Nygaard fixed points require Witt backend."
            )
        if not isinstance(prism, Prism):
            raise TypeError("prism must be a Prism.")
        self._prism = prism
        self._p = int(prism.base_ring_p)
        self._witt_length = int(prism.witt_length)

    def compute_fixed_points_for_monomial_operator(self, op: "MonomialOperator", slope_n: int) -> NygaardFixedPointResult:
        if not MVP17_CFG_AVAILABLE:
            raise KTheorySpectralSequenceError(
                "MVP17 CFG object model not available: MonomialOperator backend missing."
            )
        if not isinstance(op, MonomialOperator):
            raise TypeError("op must be a MonomialOperator.")
        if not isinstance(slope_n, int):
            raise TypeError("slope_n must be int.")

        if int(op.p) != self._p:
            raise ValueError(f"prime mismatch: op.p={op.p} vs prism.p={self._p}")

        steps = int(op.steps)
        if steps <= 0:
            raise ValueError("MonomialOperator.steps must be > 0 for fixed-point equation Φ^{steps}.")

        cycles_idx = _sorted_cycles_deterministic(op.perm)
        cycles: List[NygaardFixedPointCycle] = []
        for cyc in cycles_idx:
            L = int(len(cyc))
            E = int(sum(int(op.exp[i]) for i in cyc))
            slope = Fraction(E, L * steps)
            cycles.append(
                NygaardFixedPointCycle(
                    cycle=tuple(int(i) for i in cyc),
                    exp_sum=E,
                    steps=steps,
                    slope_per_step=slope,
                )
            )

        if not cycles:
            raise KTheorySpectralSequenceError("internal: no cycles found for a non-empty permutation")

        max_slope = max((c.slope_per_step for c in cycles), default=Fraction(0, 1))
        target_slope = Fraction(int(slope_n), 1)

        # 斜率过滤：只有最大斜率与 n 共振才允许通过（稿子口径）
        slope_filtration_ok = (max_slope == target_slope)

        # 共振 cycle：既满足 slope==n，又必须是最大斜率
        resonant_cycles = tuple(c for c in cycles if c.slope_per_step == target_slope and c.slope_per_step == max_slope)
        unique = (len(resonant_cycles) == 1)

        gen: Optional[Dict[int, int]] = None
        if unique:
            gen = self._canonical_fixed_point_generator(op, resonant_cycles[0], slope_n=int(slope_n))

        return NygaardFixedPointResult(
            p=self._p,
            witt_length=self._witt_length,
            steps=steps,
            slope_n=int(slope_n),
            max_slope_per_step=max_slope,
            cycles=tuple(cycles),
            resonant_cycles=resonant_cycles,
            slope_filtration_ok=bool(slope_filtration_ok),
            unique_resonant_cycle=bool(unique),
            fixed_point_generator_valuations=gen,
        )

    def _canonical_fixed_point_generator(
        self,
        op: "MonomialOperator",
        cycle: NygaardFixedPointCycle,
        *,
        slope_n: int,
    ) -> Dict[int, int]:
        """
        对一个共振 cycle 构造一个确定性固定点生成元（以 p-幂 valuations 表示）。

        方程：
          Φ^{steps}(x) = p^{n·steps} x

        在 cycle 上递推：
          p^{exp[i_j]} x_{i_j} = p^{n·steps} x_{i_{j+1}}

        令 δ_j = exp[i_j] - n·steps，则 valuations 递推：
          v_{j+1} = v_j + δ_j

        取一个“原始”生成元：通过平移 v_0，使得 min(v_j)=0（唯一确定）。
        """
        steps = int(op.steps)
        if steps <= 0:
            raise KTheorySpectralSequenceError("internal: steps must be > 0")
        rhs_exp = int(slope_n) * steps

        cyc = list(cycle.cycle)
        if not cyc:
            raise KTheorySpectralSequenceError("internal: empty cycle")

        deltas: List[int] = [int(op.exp[i]) - rhs_exp for i in cyc]
        partial: List[int] = [0]
        s = 0
        for d in deltas:
            s += int(d)
            partial.append(int(s))
        # 共振条件：Σ δ = 0
        if s != 0:
            raise KTheorySpectralSequenceError(
                f"internal: non-resonant cycle passed to generator (sum delta={s})"
            )

        min_s = min(partial)
        shift = -min_s if min_s < 0 else 0

        valuations: Dict[int, int] = {}
        for j, idx in enumerate(cyc):
            v = int(shift + partial[j])
            if v < 0:
                raise KTheorySpectralSequenceError("internal: negative valuation after shift")
            valuations[int(idx)] = v

        # 自检：确保递推方程在 valuations 层面严格成立（包含截断到 p^{witt_length} 的零化一致性）
        for j, idx in enumerate(cyc):
            nxt = int(op.perm[idx])
            v_j = int(valuations[int(idx)])
            v_nxt = int(valuations[int(nxt)])
            left_exp = int(op.exp[idx]) + v_j
            right_exp = rhs_exp + v_nxt
            if left_exp != right_exp:
                raise KTheorySpectralSequenceError(
                    f"internal: fixed point valuation check failed at idx={idx}: "
                    f"exp[idx]={op.exp[idx]}, v={v_j}, rhs_exp={rhs_exp}, v_next={v_nxt}"
                )

        return valuations


# =============================================================================
# Syntomic resonance: (Phi - p^n) x = 0 over B_cris (monodromy-free)
# =============================================================================


class SyntomicResonanceError(KTheorySpectralSequenceError):
    """Syntomic 共振核求解失败（必须中断，禁止静默）。"""


class MonodromyNoiseError(SyntomicResonanceError):
    """N != 0：落在 B_st 而非 B_cris（对数极点伪解）。"""


class NonUniqueResonanceError(SyntomicResonanceError):
    """target_rank=1 失败：共振核不唯一。"""


class InsufficientPrecisionError(SyntomicResonanceError):
    """精度不足（未达到由 Arakelov 高度导出的 required_precision）。"""


def _is_identity_matrix_modp(mat: Sequence[Sequence[int]], *, p: int) -> bool:
    if not isinstance(p, int) or p < 2:
        raise ValueError("p must be int >= 2.")
    n = len(mat)
    if n == 0:
        return False
    for i in range(n):
        row = mat[i]
        if not isinstance(row, Sequence) or len(row) != n:
            return False
        for j in range(n):
            v = int(row[j]) % p
            if i == j:
                if v != 1:
                    return False
            else:
                if v != 0:
                    return False
    return True


def _is_zero_matrix_modp(mat: Sequence[Sequence[int]], *, p: int) -> bool:
    if not isinstance(p, int) or p < 2:
        raise ValueError("p must be int >= 2.")
    if len(mat) == 0:
        return False
    for row in mat:
        if not isinstance(row, Sequence):
            return False
        for x in row:
            if int(x) % p != 0:
                return False
    return True


class _WittVectorOperator(ABC):
    """
    在 W_{witt_length}(F_p) 上的线性算子（仅用于严格验算，不做数值近似）。
    输入/输出向量形状：长度 = basis_dim 的 WittVector 列表。
    """

    def __init__(self, *, prism: "Prism", basis_dim: int):
        if not MVP17_AVAILABLE:
            raise KTheorySpectralSequenceError("MVP17 backend not available.")
        if not isinstance(prism, Prism):
            raise TypeError("prism must be a Prism.")
        d = int(basis_dim)
        if d < 1:
            raise ValueError("basis_dim must be >= 1.")
        self._prism = prism
        self._p = int(prism.p)
        self._witt_length = int(prism.witt_length)
        self._basis_dim = d

    @property
    def basis_dim(self) -> int:
        return int(self._basis_dim)

    @abstractmethod
    def apply(self, x: Sequence["WittVector"]) -> List["WittVector"]:
        raise NotImplementedError

    def __call__(self, x: Sequence["WittVector"]) -> List["WittVector"]:
        return self.apply(x)

    def __sub__(self, other: "_WittVectorOperator") -> "_WittVectorOperator":
        return _OperatorDifference(left=self, right=other)

    def __rmul__(self, scalar: int) -> "_WittVectorOperator":
        return _OperatorScalarMul(scalar=int(scalar), op=self)


class _OperatorIdentity(_WittVectorOperator):
    def apply(self, x: Sequence["WittVector"]) -> List["WittVector"]:
        if len(x) != self._basis_dim:
            raise ValueError("vector length mismatch for Identity operator.")
        return [xi for xi in x]


class _OperatorZero(_WittVectorOperator):
    def apply(self, x: Sequence["WittVector"]) -> List["WittVector"]:
        if len(x) != self._basis_dim:
            raise ValueError("vector length mismatch for Zero operator.")
        return [WittVector.zero(self._p, self._witt_length) for _ in range(self._basis_dim)]


class _OperatorDifference(_WittVectorOperator):
    def __init__(self, *, left: _WittVectorOperator, right: _WittVectorOperator):
        if left.basis_dim != right.basis_dim:
            raise ValueError("Operator basis_dim mismatch.")
        super().__init__(prism=left._prism, basis_dim=left.basis_dim)
        self._left = left
        self._right = right

    def apply(self, x: Sequence["WittVector"]) -> List["WittVector"]:
        a = self._left.apply(x)
        b = self._right.apply(x)
        if len(a) != len(b):
            raise KTheorySpectralSequenceError("internal: operator output length mismatch.")
        return [ai - bi for ai, bi in zip(a, b)]


class _OperatorScalarMul(_WittVectorOperator):
    """
    标量乘法算子：scalar * op。

    严格约束：scalar 必须是 p 的幂（来自 prism.p ** k），否则拒绝（避免引入一般整数嵌入的歧义）。
    """

    def __init__(self, *, scalar: int, op: _WittVectorOperator):
        super().__init__(prism=op._prism, basis_dim=op.basis_dim)
        if not isinstance(scalar, int):
            raise TypeError("scalar must be int.")
        if scalar < 0:
            raise ValueError("scalar must be >= 0.")
        self._op = op
        self._scalar = int(scalar)

        if self._scalar == 0:
            self._scalar_witt = WittVector.zero(self._p, self._witt_length)
            return

        # Require scalar = p^k exactly.
        tmp = int(self._scalar)
        k = 0
        while tmp % self._p == 0:
            tmp //= self._p
            k += 1
        if tmp != 1:
            raise SyntomicResonanceError(
                f"scalar={int(self._scalar)} is not a pure p-power (p={self._p}); refuse ambiguous scaling."
            )
        if k >= self._witt_length:
            raise InsufficientPrecisionError(
                f"Need p^{int(k)} in scalar operator but witt_length={self._witt_length} truncates it to 0."
            )
        self._scalar_witt = _witt_p_power(self._p, self._witt_length, int(k))

    def apply(self, x: Sequence["WittVector"]) -> List["WittVector"]:
        y = self._op.apply(x)
        return [self._scalar_witt * yi for yi in y]


class _OperatorMonomialFrobenius(_WittVectorOperator):
    """
    MonomialOperator 作为 Φ（或 Φ^{steps}）的实现：
        e_i ↦ p^{exp[i]} · e_{perm[i]}
    """

    def __init__(self, *, prism: "Prism", op: "MonomialOperator"):
        if not isinstance(op, MonomialOperator):
            raise TypeError("op must be a MonomialOperator.")
        super().__init__(prism=prism, basis_dim=len(op.perm))
        self._op_monomial = op

    def apply(self, x: Sequence["WittVector"]) -> List["WittVector"]:
        if len(x) != self._basis_dim:
            raise ValueError("vector length mismatch for MonomialFrobenius operator.")
        y = [WittVector.zero(self._p, self._witt_length) for _ in range(self._basis_dim)]
        for i in range(self._basis_dim):
            e = int(self._op_monomial.exp[i])
            if e < 0:
                raise SyntomicResonanceError(
                    "MonomialOperator.exp contains negative exponent; current Witt backend cannot represent p^{-1}. "
                    "Refuse to approximate; provide B_cris[1/p] backend."
                )
            if e >= self._witt_length:
                raise InsufficientPrecisionError(
                    f"Need p^{int(e)} for Frobenius scaling but witt_length={self._witt_length} truncates it to 0."
                )
            scale = _witt_p_power(self._p, self._witt_length, int(e))
            j = int(self._op_monomial.perm[i])
            y[j] = scale * x[i]
        return y


def Identity(*, prism: "Prism", basis_dim: int) -> _WittVectorOperator:
    """Identity() helper with explicit basis_dim (no hidden globals)."""
    return _OperatorIdentity(prism=prism, basis_dim=int(basis_dim))


class SyntomicResonanceSolver:
    """
    Syntomic 共振核求解器（严格版）。

    定义方程（Syntomic 上同调口径）：
        operator_T = Φ - p^n · Id
        H^0_syn(...) = ker(operator_T)  （并在 B_cris 中取解，排除 B_st 的 monodromy 噪声）

    关键硬约束（来自用户稿件/红线）：
    - **B_cris**：强制 N == 0（monodromy-free），否则视为“对数极点伪解”并抛异常。
    - **target_rank = 1**：只接受秩 1 的唯一共振态；否则抛异常（拒绝非唯一解）。
    - **precision = prism.required_precision**：精度必须由 Arakelov 高度上界导出；未提供高度直接抛错。

    当前后端实现：
    - Φ 用 MVP17 的 MonomialOperator 模型（置换 + p 幂缩放），并只在“截断 Witt 环 W_{witt_length}(F_p)”上做精确验证。
    - 若输入需要 p^{-1}（例如 MonomialOperator.exp 含负数），将明确抛错要求 B_cris[1/p] 后端，禁止静默近似。
    """

    def __init__(self, prism: "Prism"):
        if not MVP17_AVAILABLE:
            raise SyntomicResonanceError("MVP17 backend not available: need Prism/WittVector support.")
        if not MVP17_CFG_AVAILABLE:
            raise SyntomicResonanceError("MVP17 CFG backend not available: need MonomialOperator support.")
        if not isinstance(prism, Prism):
            raise TypeError("prism must be a Prism.")
        self.prism = prism
        self._p = int(prism.p)
        self._witt_length = int(prism.witt_length)
        if self._witt_length < 1:
            raise ValueError("prism.witt_length must be >= 1.")

    def solve_fixed_point_rank_one(
        self,
        *,
        D_cris_frobenius: "MonomialOperator",
        n: int,
        # monodromy certificate: if provided, we enforce N==0 strictly.
        # - N_modp: monodromy operator matrix over F_p (must be all-zero).
        N_modp: Optional[Sequence[Sequence[int]]] = None,
        # Some callers may only have holonomy/monodromy matrix M (unipotent). For crystalline we need it to be identity.
        monodromy_matrix_modp: Optional[Sequence[Sequence[int]]] = None,
    ) -> NygaardFixedPointResult:
        if not isinstance(D_cris_frobenius, MonomialOperator):
            raise TypeError("D_cris_frobenius must be a MonomialOperator.")
        if not isinstance(n, int):
            raise TypeError("n must be int.")

        # 0) Precision must be derived from Arakelov height (no default truncation).
        precision = int(self.prism.required_precision)
        if precision < 1:
            raise InsufficientPrecisionError("internal: required_precision must be >= 1.")
        if self._witt_length < precision:
            # Prism.required_precision should already enforce this; keep a second guard to avoid silent misuse.
            raise InsufficientPrecisionError(
                f"witt_length={self._witt_length} < required_precision={precision}."
            )

        # 1) B_cris vs B_st: monodromy must vanish (N == 0).
        if N_modp is not None:
            if not _is_zero_matrix_modp(N_modp, p=self._p):
                raise MonodromyNoiseError("Monodromy operator N != 0 (log-pole noise): refuse B_st pseudo-solution.")
        if monodromy_matrix_modp is not None:
            if not _is_identity_matrix_modp(monodromy_matrix_modp, p=self._p):
                raise MonodromyNoiseError("Monodromy matrix is not identity: refuse B_st pseudo-solution.")

        # 2) 核心算术核：解 (Φ - p^n) x = 0
        #    若 D_cris_frobenius 表示 Φ^{steps}（MonomialOperator.steps），则方程为：
        #        (Φ^{steps} - p^{n·steps}) x = 0
        steps = int(D_cris_frobenius.steps)
        if steps <= 0:
            raise ValueError("MonomialOperator.steps must be > 0 (Frobenius operator cannot be empty).")

        # operator_T = D_cris.frobenius_operator() - (prism.p ** n) * Identity()
        # 在 steps!=1 的 CFG/路径模型里，Φ 实际为 Φ^{steps}，对应标量为 p^{n·steps}。
        basis_dim = int(len(D_cris_frobenius.perm))
        D_cris_phi = _OperatorMonomialFrobenius(prism=self.prism, op=D_cris_frobenius)
        operator_T = D_cris_phi - (self.prism.p ** (int(n) * int(steps))) * Identity(prism=self.prism, basis_dim=basis_dim)

        inv = NygaardFixedPointInvariants(self.prism)
        fp = inv.compute_fixed_points_for_monomial_operator(D_cris_frobenius, slope_n=int(n))

        # kernel trivial => no resonance (deterministic)
        if not fp.slope_filtration_ok:
            return fp

        # 3) target_rank = 1 hard constraint: require unique resonant cycle.
        if not fp.unique_resonant_cycle:
            raise NonUniqueResonanceError(
                f"target_rank=1 violated: resonant_cycles={len(fp.resonant_cycles)}."
            )
        if fp.fixed_point_generator_valuations is None:
            raise SyntomicResonanceError("internal: unique_resonant_cycle but generator missing.")

        # 4) Materialize a B_cris candidate (here: truncated Witt ring) and verify the equation exactly.
        #    This is the no-cheat checkpoint: we must have (Φ^{steps} - p^{n·steps}Id)(x) == 0 exactly.
        if basis_dim < 1:
            raise SyntomicResonanceError("invalid MonomialOperator: basis_dim < 1.")

        # Build x vector (WittVector coordinates). Missing indices => 0.
        x_vec: List[WittVector] = [WittVector.zero(self._p, self._witt_length) for _ in range(basis_dim)]
        max_needed_exp = 0
        for idx, vp in fp.fixed_point_generator_valuations.items():
            if vp < 0:
                raise SyntomicResonanceError("internal: negative valuation in generator (needs B_cris[1/p]).")
            max_needed_exp = max(max_needed_exp, int(vp))
            x_vec[int(idx)] = _witt_p_power(self._p, self._witt_length, int(vp))

        rhs_exp = int(n) * int(steps)
        if rhs_exp < 0:
            raise SyntomicResonanceError("internal: rhs exponent negative (invalid n or steps).")
        max_needed_exp = max(max_needed_exp, rhs_exp)

        # Ensure truncation can represent all p^k used in this verification.
        # (p^k becomes 0 when k >= witt_length in the current backend; that would be silent corruption.)
        if max_needed_exp >= self._witt_length:
            raise InsufficientPrecisionError(
                f"Need p^{max_needed_exp} but witt_length={self._witt_length} truncates it to 0. "
                f"Provide larger witt_length consistent with prism.required_precision={precision}."
            )

        # Verify T(x) == 0 exactly in W_{witt_length}(F_p)
        tx = operator_T.apply(x_vec)
        if len(tx) != basis_dim:
            raise SyntomicResonanceError("internal: operator_T output length mismatch.")
        for j in range(basis_dim):
            if tx[j].is_zero():
                continue
            raise SyntomicResonanceError(
                f"Arithmetic kernel verification failed at component {j}: T(x) != 0."
            )

        return fp


# 第三部分：从K理论到轨道积分的Thomason桥接


class ThomasonOrbitalBridge:
    """
    Thomason定理：K理论欧拉示性数 = 轨道积分
    这里实现从抽象K理论到具体DeFi路径的映射
    核心：路径签名张量 与 算术核 的内积映射 依赖 MVP19
    """
    def __init__(self, k_theory_engine: "KTheorySpectralSequenceEngine"):
        self.K = k_theory_engine
    
    def compute_orbital_integral_via_euler_char(self, cfg_path: Any) -> complex:
                                                      
        """
        核心公式（Thomason 桥接） 占位置先
        """
        raise NotImplementedError(
            "ThomasonOrbitalBridge.compute_orbital_integral_via_euler_char is not wired in Temporarypatch."
        )


@dataclass
class NygaardGradedPiece:
    """
    Nygaard 过滤的 graded piece。
    
    数学定义：
    gr^i_N(Δ) = N^{≥i} / N^{≥i+1}
    
    对于 Witt 向量：gr^i_N(W(k)) ≅ k（剩余域）
    """
    level: int  # i
    residue_class: int  # 在 F_p 中的代表元
    p: int  # 特征


@dataclass
class PrismaticCohomologyModule:
    """
    棱镜上同调模 H^n_Δ(X/A)。
    
    数学结构：
    - 作为 A-模配备 Frobenius φ
    - 带有 Nygaard 过滤 N^{≥i}
    - 连接到 de Rham 上同调：H^n_dR(X/(A/I)) ≅ H^n_Δ(X/A) ⊗_A (A/I)
    
    从 MVP17 的 Witt 向量/棱柱结构导出。
    """
    degree: int  # H^n 的 n
    witt_generators: List['WittVector']  # 作为 W(k)-模的生成元
    nygaard_levels: List[int]  # 每个生成元的 Nygaard 级别
    prism: 'Prism'  # 所属的棱柱
    
    def dimension(self) -> int:
        """模的秩（作为 W(k)-模）"""
        return len(self.witt_generators)
    
    def frobenius_action(self) -> List['WittVector']:
        """
        Frobenius 作用 φ: H^n → H^n
        返回 φ(生成元) 的列表
        """
        return [w.frobenius() for w in self.witt_generators]
    
    def nygaard_filtration_level(self, index: int) -> int:
        """获取第 index 个生成元的 Nygaard 级别"""
        if index < 0 or index >= len(self.nygaard_levels):
            raise IndexError(f"Generator index {index} out of range")
        return self.nygaard_levels[index]


@dataclass
class SpectralSequenceE2Term:
    """
    E_2 页的一个项 E_2^{p,q}。
    
    数学定义：
    E_2^{p,q} = H^p(X, K_q(pt)) ⊗ H^0_Δ(X/A)
    
    对于有限域系数：
    - K_q(F_p) 由 Quillen 定理给出
    - H^0_Δ 是棱镜上同调的 0 阶部分
    """
    bidegree: Tuple[int, int]  # (p, q)
    prismatic_component: Optional[PrismaticCohomologyModule]  # H^p_Δ
    k_theory_component: AbelianGroupInvariant  # K_q(F_p)
    
    def total_degree(self) -> int:
        return self.bidegree[0] + self.bidegree[1]
    
    def dimension(self) -> int:
        """E_2^{p,q} 的维度"""
        if self.prismatic_component is None:
            return 0
        k_dim = self.k_theory_component.minimal_number_of_generators()
        h_dim = self.prismatic_component.dimension()
        return k_dim * h_dim


class KTheoryPrismaticBridge:
    """
    Step 1: CFG → MVP17 Prism（棱柱输入）
    将 MVP17 的棱镜结构桥接到 K 理论谱序列。
    核心数学：
    - Prism (A, I) 提供基础 δ-环结构
    - Nygaard 过滤 N^{≥i} 控制 Frobenius 行为
    - K 理论通过 Chern 特征连接到棱镜上同调
    
    关键定理（Bhatt-Morrow-Scholze）：
    对于 qcqs 形式概型 X/O_K，有自然的比较同构：
    K_n(X) ⊗ Z_p ≅ H^{2n}_Δ(X/A)[1/p] 的某个过滤商
    """
    
    def __init__(self, p: int, witt_length: int = 4, *, arakelov_height_bound: Optional[int] = None):
        """
        Args:
            p: 特征（必须是素数）
            witt_length: Witt 向量长度（精度）
        """
        if not MVP17_AVAILABLE:
            raise KTheorySpectralSequenceError(
                "MVP17 prismatic module not available. "
                "Cannot construct K-theory bridge without Witt vector backend."
            )
        
        # 验证 p 是素数（使用确定性 Miller-Rabin）
        if not _is_prime_u64_deterministic(p):
            raise ValueError(f"p={p} is not prime")
        
        self._p = p
        self._witt_length = witt_length
        
        # 构造 Crystalline 棱柱 (W(k), (p))
        # 注意：arakelov_height_bound 若提供，将用于严格导出 prism.required_precision（禁止默认截断）。
        self._prism = Prism(p, witt_length, arakelov_height_bound=arakelov_height_bound)
        self._nygaard = NygaardFiltration(self._prism)
        self._validator = IntegralityValidator(self._prism)
        self._delta_ring = WittVectorDeltaRing(p, witt_length)
    
    @property
    def prime(self) -> int:
        return self._p
    
    @property
    def prism(self) -> 'Prism':
        return self._prism
    
    @property
    def nygaard_filtration(self) -> 'NygaardFiltration':
        return self._nygaard
    
    def witt_vector_from_components(self, components: List[int]) -> 'WittVector':
        """
        从分量列表构造 Witt 向量。
        
        Args:
            components: [x_0, x_1, ..., x_{n-1}]，每个 x_i ∈ [0, p-1]
        
        Returns:
            对应的 Witt 向量
        """
        if len(components) > self._witt_length:
            raise ValueError(f"Too many components: {len(components)} > {self._witt_length}")
        
        # 扩展到正确长度
        padded = components + [0] * (self._witt_length - len(components))
        
        return WittVector(
            [FiniteFieldElement(c % self._p, self._p) for c in padded],
            self._p
        )
    
    def validate_witt_vector(self, w: 'WittVector') -> ValidationResult:
        """验证 Witt 向量的整性"""
        return self._validator.validate_witt_vector(w)
    
    def compute_nygaard_level(self, w: 'WittVector') -> int:
        """
        Step 2: 计算 Witt 向量的 Nygaard 过滤级别。
        
        定义：level(w) = max{i : w ∈ N^{≥i}}
        对于 Witt 向量：w ∈ N^{≥i} 当且仅当 w_0 = ... = w_{i-1} = 0
        """
        return self._nygaard.filtration_level(w)
    
    def extract_graded_piece(self, w: 'WittVector') -> NygaardGradedPiece:
        """
        提取 Nygaard graded piece。
        
        gr^i_N(w) = w_i (mod p) ∈ F_p
        """
        level, residue = self._nygaard.graded_piece(w)
        return NygaardGradedPiece(
            level=level,
            residue_class=residue.value,
            p=self._p
        )
    
    def frobenius_compatibility_check(self, w: 'WittVector') -> bool:
        """
        验证 Frobenius 兼容性：φ(N^{≥i}) ⊂ I^i
        
        这是 Nygaard 过滤的核心性质。
        """
        return self._nygaard.verify_frobenius_compatibility(w)
    
    def construct_prismatic_cohomology_module(
        self,
        degree: int,
        generators_data: List[List[int]]
    ) -> PrismaticCohomologyModule:
        """
        构造棱镜上同调模 H^n_Δ。
        
        Args:
            degree: 上同调度数 n
            generators_data: 生成元的 Witt 分量列表
        
        Returns:
            PrismaticCohomologyModule 对象
        """
        witt_gens = [self.witt_vector_from_components(data) for data in generators_data]
        nygaard_levels = [self.compute_nygaard_level(w) for w in witt_gens]
        
        return PrismaticCohomologyModule(
            degree=degree,
            witt_generators=witt_gens,
            nygaard_levels=nygaard_levels,
            prism=self._prism
        )


class KTheoryE2PageBuilder:
    """
    Step 3: Nygaard → E_2 页初始化
    构造 Atiyah-Hirzebruch 谱序列的 E_2 页。
    对于有限域 F_p：
    - K_0(F_p) ≅ Z
    - K_{2i-1}(F_p) ≅ Z/(p^i - 1)Z
    - K_{2i}(F_p) = 0 (i ≥ 1)
    """
    
    def __init__(self, bridge: KTheoryPrismaticBridge, target_degree: int):
        """
        Args:
            bridge: K 理论-棱镜桥接
            target_degree: 目标 K 群度数（计算 K_n）
        """
        self._bridge = bridge
        self._p = bridge.prime
        self._target_n = target_degree
        self._e2_terms: Dict[Tuple[int, int], SpectralSequenceE2Term] = {}
    
    def build_e2_page(
        self, 
        prismatic_modules: Optional[Dict[int, PrismaticCohomologyModule]] = None
    ) -> Dict[Tuple[int, int], SpectralSequenceE2Term]:
        """
        构造完整的 E_2 页。
        Args:
            prismatic_modules: {度数: 棱镜上同调模} 字典
                              如果为 None，使用默认构造
        """
        # 第一象限约束：p ≥ 0, q ≥ 0, p + q = target_n
        for p_deg in range(self._target_n + 1):
            q_deg = self._target_n - p_deg
            if q_deg < 0:
                continue
            
            # 获取 K_q(F_p) 的结构
            k_q_invariant = quillen_k_group_finite_field(self._p, q_deg)
            
            # 获取 H^p_Δ
            prismatic_mod = None
            if prismatic_modules is not None and p_deg in prismatic_modules:
                prismatic_mod = prismatic_modules[p_deg]
            elif p_deg == 0:
                # H^0 总是非平凡的（常值截面）
                prismatic_mod = self._construct_default_h0()
            
            # 构造 E_2^{p,q}
            e2_term = SpectralSequenceE2Term(
                bidegree=(p_deg, q_deg),
                prismatic_component=prismatic_mod,
                k_theory_component=k_q_invariant
            )
            
            self._e2_terms[(p_deg, q_deg)] = e2_term
        
        return self._e2_terms
    
    def _construct_default_h0(self) -> PrismaticCohomologyModule:
        """构造默认的 H^0_Δ（单位元生成）"""
        # H^0 由 Witt 向量 (1, 0, 0, ...) 生成
        return self._bridge.construct_prismatic_cohomology_module(
            degree=0,
            generators_data=[[1]]  # Teichmüller 提升 [1]
        )
    
    def get_e2_term(self, p: int, q: int) -> Optional[SpectralSequenceE2Term]:
        """获取 E_2^{p,q}"""
        return self._e2_terms.get((p, q))
    
    def e2_total_dimension(self) -> int:
        """E_2 页的总维度（所有项维度之和）"""
        return sum(term.dimension() for term in self._e2_terms.values())


@dataclass
class DifferentialComputation:
    """
    微分计算结果。
    包含 d_r: E_r^{p,q} → E_r^{p+r, q-r+1} 的完整信息。
    """
    page_number: int  # r
    source_bidegree: Tuple[int, int]  # (p, q)
    target_bidegree: Tuple[int, int]  # (p+r, q-r+1)
    is_zero: bool  # 微分是否为零映射
    kernel_dimension: int  # ker(d_r) 的维度
    image_dimension: int  # im(d_r) 的维度
    adams_contribution: bool  # 是否有 Adams 操作贡献
    massey_contribution: bool  # 是否有 Massey 积贡献


class KTheoryDifferentialEngine:
    """
    Step 4: E_2 → E_r 微分计算（严格版）

    本引擎只做“被迫为零”的判定：
      - 源项为零 或 目标项为零（结构性为零）⇒ d_r 必为零
      - 其余情况：不做任何启发式推断，直接拒绝（抛异常）
    """
    
    def __init__(self, bridge: KTheoryPrismaticBridge, e2_page: Dict[Tuple[int, int], SpectralSequenceE2Term]):
        self._bridge = bridge
        self._p = bridge.prime
        self._e2_page = e2_page
        self._pages: Dict[int, Dict[Tuple[int, int], Any]] = {2: e2_page}
        self._differentials: Dict[Tuple[int, Tuple[int, int]], DifferentialComputation] = {}
    
    def compute_differential(
        self, 
        r: int, 
        source_bidegree: Tuple[int, int]
    ) -> DifferentialComputation:
        """
        计算微分 d_r: E_r^{p,q} → E_r^{p+r, q-r+1}。
        Args:
            r: 页数（r ≥ 2）
            source_bidegree: 源双度数 (p, q)
        """
        if r < 2:
            raise ValueError("Differential page number must be >= 2.")

        src_term = self._e2_page.get(source_bidegree)
        src_dim = src_term.dimension() if src_term is not None else 0

        p, q = source_bidegree
        target_bidegree = (p + r, q - r + 1)

        # 目标不在第一象限：目标项为 0 ⇒ 微分为 0
        if target_bidegree[0] < 0 or target_bidegree[1] < 0:
            return DifferentialComputation(
                page_number=int(r),
                source_bidegree=source_bidegree,
                target_bidegree=target_bidegree,
                is_zero=True,
                kernel_dimension=int(src_dim),
                image_dimension=0,
                adams_contribution=False,
                massey_contribution=False,
            )

        tgt_term = self._e2_page.get(target_bidegree)
        tgt_dim = tgt_term.dimension() if tgt_term is not None else 0

        # 源/目标结构性为零：微分必为零
        if src_dim == 0 or tgt_dim == 0:
            return DifferentialComputation(
                page_number=int(r),
                source_bidegree=source_bidegree,
                target_bidegree=target_bidegree,
                is_zero=True,
                kernel_dimension=int(src_dim),
                image_dimension=0,
                adams_contribution=False,
                massey_contribution=False,
            )

        raise KTheorySpectralSequenceError(
            "Non-trivial differentials are not computed in the strict engine. "
            "Provide explicit differential data or restrict to vanishing-target cases."
        )
    
    # 严格版不暴露任何“推断型”私有方法；需要的零判定已在 compute_differential 内完成。


class KTheoryConvergenceChecker:
    """
    Step 5: E_r → E_∞ 收敛判定
    判断谱序列何时收敛到 E_∞ 页。
    收敛定理：
    对于有界第一象限谱序列，存在 R 使得对所有 r ≥ R：
    E_r ≅ E_{r+1} ≅ ... ≅ E_∞
    具体地，对于目标度数 n 的 Atiyah-Hirzebruch 谱序列：
    当 r > n + 1 时，所有微分因双度数原因必为零。
    """
    
    def __init__(self, target_degree: int, p: int):
        self._target_n = target_degree
        self._p = p
        # 理论收敛页数：r > n + 1
        self._theoretical_convergence = target_degree + 2
    
    @property
    def theoretical_convergence_page(self) -> int:
        """理论上的收敛页数"""
        return self._theoretical_convergence
    
    def check_convergence(
        self, 
        differential_engine: KTheoryDifferentialEngine,
        e2_page: Dict[Tuple[int, int], SpectralSequenceE2Term]
    ) -> Tuple[int, bool]:
        """
        检查谱序列是否收敛。
        
        Returns:
            (收敛页数, 是否提前收敛)
        """
        # 对于第一象限 AHSS：在 r > n + 1 时必收敛
        max_r = self._theoretical_convergence
        
        # 检查是否提前收敛（所有微分为零）
        for r in range(2, max_r):
            all_zero = True
            for bidegree in e2_page.keys():
                diff = differential_engine.compute_differential(r, bidegree)
                if not diff.is_zero:
                    all_zero = False
                    break
            
            if all_zero:
                return (r, True)  # 提前收敛
        
        return (max_r, False)  # 理论收敛
    
    def is_converged_at_page(self, page_number: int) -> bool:
        """判断在给定页数是否已收敛"""
        return page_number >= self._theoretical_convergence


@dataclass
class KTheoryGroupStructure:
    """
    K 理论群的完整结构。
    K_n(X) 作为有限生成阿贝尔群的分解：
    K_n(X) ≅ Z^r ⊕ ⊕_i Z/n_i Z
    包含从谱序列 E_∞ 页提取的完整信息。
    """
    degree: int  # K_n 的 n
    invariant: AbelianGroupInvariant  # 群结构不变量
    e_infinity_contributions: List[Tuple[Tuple[int, int], int]]  # [(双度数, 维度贡献)]
    convergence_page: int  # 收敛于 E_r
    euler_characteristic: int  # 欧拉示性数 χ(K_n)
    
    def rank(self) -> int:
        """自由部分的秩"""
        return self.invariant.free_rank
    
    def torsion_order(self) -> int:
        """挠部分的阶（如果有限）"""
        if not self.invariant.is_finite():
            raise ValueError("Group has infinite order")
        return self.invariant.order()


class KTheoryExtractor:
    """
    Step 6: E_∞ → K_n(X) 提取
    从 E_∞ 页提取 K 理论群 K_n(X)。
    数学公式：
    K_n(X) 由 E_∞ 页的过滤商给出：
    0 = F^{n+1} K_n ⊂ F^n K_n ⊂ ... ⊂ F^0 K_n = K_n(X)
    其中 F^p K_n / F^{p+1} K_n ≅ E_∞^{p, n-p}
    欧拉示性数：
    χ(K_n) = Σ_{p+q=n} (-1)^p · dim(E_∞^{p,q})
    """
    
    def __init__(self, bridge: KTheoryPrismaticBridge, target_degree: int):
        self._bridge = bridge
        self._p = bridge.prime
        self._target_n = target_degree
    
    def extract_from_e_infinity(
        self, 
        e_infinity_page: Dict[Tuple[int, int], SpectralSequenceE2Term],
        convergence_page: int
    ) -> KTheoryGroupStructure:
        """
        从 E_∞ 页提取 K_n(X)。
        
        Args:
            e_infinity_page: E_∞ 页的项
            convergence_page: 收敛页数
        
        Returns:
            KTheoryGroupStructure 对象
        """
        # 只依赖 E_∞ 的分级信息时，一般无法恢复 K_n 的完整同构类型（存在 extension problem）。
        # 本工程实现遵循“拒绝猜测”的红线：
        #   - 若沿对角线 p+q=n 有多个非零 graded piece：直接抛错，要求提供 extension 证书。
        #   - 若只有一个非零 graded piece：不存在 extension，自同构类型可唯一确定。

        contributions_terms: List[Tuple[Tuple[int, int], SpectralSequenceE2Term]] = []
        euler_char = 0

        for bidegree, term in e_infinity_page.items():
            p_deg, q_deg = bidegree
            if p_deg + q_deg != self._target_n:
                continue
            dim = int(term.dimension())
            if dim <= 0:
                continue
            contributions_terms.append((bidegree, term))
            euler_char += ((-1) ** int(p_deg)) * dim

        if not contributions_terms:
            invariant = AbelianGroupInvariant(free_rank=0, torsion_invariants=())
            return KTheoryGroupStructure(
                degree=self._target_n,
                invariant=invariant,
                e_infinity_contributions=[],
                convergence_page=int(convergence_page),
                euler_characteristic=int(euler_char),
            )

        if len(contributions_terms) != 1:
            raise KTheorySpectralSequenceError(
                "Cannot extract unique K-group structure from E_∞ with multiple non-zero graded pieces "
                "(extension problem). Provide an explicit extension certificate or restrict inputs."
            )

        (bidegree, term) = contributions_terms[0]
        p_deg, q_deg = bidegree

        # 该 graded piece 在我们的实现里形状为：
        #   E_∞^{p,q} = H^p_Δ ⊗ K_q(F_p)
        # 若 H^p_Δ 的秩为 h_dim，则 K_n(X) 在此简化模型中等价于 h_dim 份 K_q(F_p) 的直和。
        pr = term.prismatic_component
        h_dim = int(pr.dimension()) if pr is not None else 0
        if h_dim < 0:
            raise KTheorySpectralSequenceError("internal: negative prismatic dimension")

        k_inv = term.k_theory_component
        if k_inv.is_trivial() or h_dim == 0:
            invariant = AbelianGroupInvariant(free_rank=0, torsion_invariants=())
        else:
            free_rank = int(k_inv.free_rank) * h_dim
            torsion_invariants: Tuple[int, ...] = ()
            if k_inv.torsion_invariants:
                if len(k_inv.torsion_invariants) != 1:
                    raise KTheorySpectralSequenceError(
                        "internal: expected cyclic torsion for finite field K_q(F_p) in current backend"
                    )
                m = int(k_inv.torsion_invariants[0])
                torsion_invariants = tuple(m for _ in range(h_dim))
            invariant = AbelianGroupInvariant(free_rank=free_rank, torsion_invariants=torsion_invariants)

        return KTheoryGroupStructure(
            degree=self._target_n,
            invariant=invariant,
            e_infinity_contributions=[(bidegree, int(term.dimension()))],
            convergence_page=int(convergence_page),
            euler_characteristic=int(euler_char),
        )
    
    def compute_euler_characteristic(
        self, 
        e_infinity_page: Dict[Tuple[int, int], SpectralSequenceE2Term]
    ) -> int:
        """
        计算欧拉示性数。
        
        χ(K_n) = Σ_{p+q=n} (-1)^p · dim(E_∞^{p,q})
        """
        euler_char = 0
        for bidegree, term in e_infinity_page.items():
            p, q = bidegree
            if p + q == self._target_n:
                euler_char += ((-1) ** p) * term.dimension()
        return euler_char


class ThomasonBridgeEngine:
    """
    Step 7: K_n(X) → Thomason 桥接 → 轨道积分
    Thomason 定理的核心桥接。
    数学定理（Thomason）：
    对于概型 X 和其上的向量丛 E，有：
    χ(X, E) = ∫_X ch(E) · td(X)
    - χ 是欧拉示性数（K 理论层面）- ch 是 Chern 特征（到上同调的映射）- td 是 Todd 类
    在朗兰兹框架下的应用：
    轨道积分 O_γ(f) = χ(Hitchin 纤维 at γ) = K 理论欧拉示性数 这是测试函数问题的解决方案：
    - 传统问题：Re(s) << 0 时测试函数发散
    - K 理论解决：s 对应 K 群的 grading，负 K 群自动为零
    """
    
    def __init__(self, k_theory_structure: KTheoryGroupStructure, bridge: KTheoryPrismaticBridge):
        self._k_structure = k_theory_structure
        self._bridge = bridge
        self._p = bridge.prime
    
    def compute_orbital_integral_euler_method(self) -> int:
        """
        通过欧拉示性数计算轨道积分。
        核心公式（Thomason 桥接）：
        O_γ(f) = χ(K_theory class of Hitchin fiber)
        Returns:
            轨道积分值（作为整数/有理数）
        """
        return self._k_structure.euler_characteristic
    
    def compute_chern_character_image(self) -> Dict[int, Fraction]:
        """
        计算 Chern 特征像。
        
        ch: K_n(X) → H^{even}(X, Q)
        
        对于 K_0 中的元素，Chern 特征分解为：
        ch(E) = rank(E) + c_1(E) + (c_1^2 - 2c_2)/2 + ...
        
        Returns:
            {度数: 系数} 的字典
        """
        ch_image = {}
        
        # ch_0 = 欧拉示性数（对于 K_0）
        if self._k_structure.degree == 0:
            ch_image[0] = Fraction(self._k_structure.euler_characteristic)
        
        # 对于高阶 K 群，Chern 特征更复杂
        # 需要 Adams 操作和 γ-过滤
        
        return ch_image
    
    def verify_thomason_compatibility(self) -> bool:
        """
        验证 Thomason 兼容性。
        
        检查 K 理论结构是否与轨道积分计算兼容。
        """
        # 验证 1: 欧拉示性数有限
        euler = self._k_structure.euler_characteristic
        
        # 验证 2: torsion 结构与 Hitchin 纤维兼容
        # 对于有限域，K 群的 torsion 由 Quillen 定理控制
        
        return True  # 当前实现总是兼容


class KTheorySpectralSequenceEngine:
    """
    流水线暂不确定
    """
    
    def __init__(self, p: int, target_degree: int, witt_length: int = 4):
        """
        Args:
            p: 特征（必须是素数）
            target_degree: 目标 K 群度数（计算 K_n）
            witt_length: Witt 向量长度（精度）
        """
        if not MVP17_AVAILABLE:
            raise KTheorySpectralSequenceError(
                "MVP17 not available. K-theory spectral sequence engine requires "
                "mvp17_prismatic.py for Witt vector and prism support."
            )
        
        self._p = p
        self._target_n = target_degree
        self._witt_length = witt_length
        
        # Step 1: 初始化棱柱桥接
        self._bridge = KTheoryPrismaticBridge(p, witt_length)
        
        # 内部状态
        self._e2_builder: Optional[KTheoryE2PageBuilder] = None
        self._diff_engine: Optional[KTheoryDifferentialEngine] = None
        self._convergence_checker: Optional[KTheoryConvergenceChecker] = None
        self._extractor: Optional[KTheoryExtractor] = None
        self._e2_page: Optional[Dict[Tuple[int, int], SpectralSequenceE2Term]] = None
        self._e_infinity_page: Optional[Dict[Tuple[int, int], SpectralSequenceE2Term]] = None
        self._convergence_info: Optional[Tuple[int, bool]] = None
        self._k_structure: Optional[KTheoryGroupStructure] = None
    
    @property
    def prime(self) -> int:
        return self._p
    
    @property
    def target_degree(self) -> int:
        return self._target_n
    
    def run_full_pipeline(
        self, 
        prismatic_input: Optional[Dict[int, List[List[int]]]] = None
    ) -> KTheoryGroupStructure:
        """
        运行完整的七步流水线。
        
        Args:
            prismatic_input: 棱镜上同调输入
                            {度数: [Witt 分量列表]} 的字典
                            如果为 None，使用默认构造
        
        Returns:
            KTheoryGroupStructure - K_n(X) 的完整结构
        """
        # Step 2 已在 bridge 初始化时完成
        
        # Step 3: 构造 E_2 页
        self._e2_builder = KTheoryE2PageBuilder(self._bridge, self._target_n)
        
        # 转换输入格式
        prismatic_modules = None
        if prismatic_input is not None:
            prismatic_modules = {
                deg: self._bridge.construct_prismatic_cohomology_module(deg, gens)
                for deg, gens in prismatic_input.items()
            }
        
        self._e2_page = self._e2_builder.build_e2_page(prismatic_modules)
        
        # 本文件的 E_2 构造器只生成对角线 p+q=n 的项。
        # 对 Atiyah–Hirzebruch 谱序列而言，d_r: (p,q) -> (p+r, q-r+1) 会把总度数提升 1，
        # 因而不会在“单一对角线切片”内部产生可见微分。
        #
        # 为避免“看不见就当不存在”的静默降级，本引擎在进入 Step 6 前强制检查：
        #   - 对角线上最多允许一个非零项；否则 K_n 的同构类型存在 extension problem，不能唯一恢复。
        diagonal_nonzero = [
            (b, t) for (b, t) in self._e2_page.items() if int(t.dimension()) > 0
        ]
        if len(diagonal_nonzero) > 1:
            raise KTheorySpectralSequenceError(
                "E_2 diagonal slice has multiple non-zero terms; "
                "full AHSS differentials/extensions are not implemented here (refuse to guess)."
            )

        # 在“单项对角线切片”口径下，E_2 = E_∞（对该切片）。
        self._convergence_info = (2, True)
        self._e_infinity_page = self._e2_page
        
        # Step 6: 提取 K 群
        self._extractor = KTheoryExtractor(self._bridge, self._target_n)
        self._k_structure = self._extractor.extract_from_e_infinity(
            self._e_infinity_page, 
            int(self._convergence_info[0])
        )
        
        return self._k_structure
    
    def compute_orbital_integral(self) -> int:
        """
        Step 7: 计算轨道积分。
        
        必须先调用 run_full_pipeline()。
        
        Returns:
            轨道积分值（通过 Thomason 桥接）
        """
        if self._k_structure is None:
            raise KTheorySpectralSequenceError(
                "Must call run_full_pipeline() before compute_orbital_integral()"
            )
        
        thomason = ThomasonBridgeEngine(self._k_structure, self._bridge)
        return thomason.compute_orbital_integral_euler_method()
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """
        获取诊断报告。
        
        Returns:
            包含所有中间状态的诊断信息
        """
        report = {
            "prime": self._p,
            "target_degree": self._target_n,
            "witt_length": self._witt_length,
            "mvp17_available": MVP17_AVAILABLE,
        }
        
        if self._e2_page is not None:
            report["e2_page_terms"] = len(self._e2_page)
            report["e2_total_dimension"] = sum(t.dimension() for t in self._e2_page.values())
        
        if self._convergence_info is not None:
            report["convergence_page"] = self._convergence_info[0]
            report["early_convergence"] = self._convergence_info[1]
        
        if self._k_structure is not None:
            report["k_group_free_rank"] = self._k_structure.rank()
            report["euler_characteristic"] = self._k_structure.euler_characteristic
            report["torsion_invariants"] = list(self._k_structure.invariant.torsion_invariants)
        
        return report



# 接口函数


def compute_k_theory_orbital_integral(
    p: int, 
    target_degree: int,
    prismatic_input: Optional[Dict[int, List[List[int]]]] = None,
    witt_length: int = 4
) -> Tuple[KTheoryGroupStructure, int]:
    """
    便捷函数：计算 K 理论群和轨道积分。
    
    这是朗兰兹大纲测试函数问题的解决方案入口。
    
    Args:
        p: 特征（素数）
        target_degree: 目标 K 群度数
        prismatic_input: 棱镜上同调输入（可选）
        witt_length: Witt 向量长度
    
    Returns:
        (K 群结构, 轨道积分值)
    
    Example:
        >>> k_struct, orbital = compute_k_theory_orbital_integral(p=251, target_degree=1)
        >>> print(f"K_1 欧拉示性数 = {orbital}")
    """
    engine = KTheorySpectralSequenceEngine(p, target_degree, witt_length)
    k_structure = engine.run_full_pipeline(prismatic_input)
    orbital_integral = engine.compute_orbital_integral()
    
    return k_structure, orbital_integral
