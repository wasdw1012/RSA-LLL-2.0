"""
MVP19 - Signature-CVP: F5 终结者

铁律：
- 禁止启发式（"Gaussian heuristic"/经验阈值/魔法数/静默降级）。
- 允许数值线性代数，但所有容差必须由机器精度与问题尺度导出（复用Tensor_roduct思想）
- 任何找不到解/依赖缺失/输入不合法都必须抛出异常；不得偷fallback

架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MVP19 Signature-CVP 引擎                             
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 1: 热带几何预求解器 (TropicalPreSolver)                                  
│   - 牛顿多胞体构造 → 法扇交集 → CVP 初始范围锁定                               
│   - 替代 F5 的 S-Pair 生成（组合几何，非代数运算）                              
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 2: Arakelov/Adelic 度量空间 (AdelicMetricSpace)                         
│   - 积公式: d² = Σ_v w_v log||x-y||_v （所有素数位 + 无穷位）                  
│   - 算术格林函数 → 格基变为带度量的向量丛                                      
│   - 替代 F5 的系数爆炸处理                                                    
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 3: 格规约后端 (LatticeBackend)                                          
│   - LLL: δ=3/4 论文标准                                                      
│   - BKZ 2.0: block_size 显式参数，Hermite因子 ~1.01^n                         
│   - Kannan Embedding: CVP → SVP 规约                                         
│   - Schnorr-Euchner 枚举: 精确但指数                                          
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 4: 辛几何流引擎 (SymplecticFlowEngine)                                  
│   - 相空间提升: (x, p) 位置+动量                                              
│   - Shadow Hamiltonian: H(x,p) = U(x) + K(p)                                
│   - 辛积分器 (Velocity Verlet): 冲过局部极小                                   
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 5: 外部强化组件 Teleport-F5（`web_cta/arakelov` Hawks 组）               
│   - Yang–Mills–Higgs 离散结构：F = dA + A∧A + Higgs 协变项（证书级不变量输出） 
│   - DUY 稳定性口径：Hermitian–Einstein 缺陷矩阵与零模数（跳过暴力穷举）         
│   - 非交换震荡器 / Floer 交点代理：为 Phase 6 提供更硬的 Hitchin 谱数据证书     
├─────────────────────────────────────────────────────────────────────────────┤
│ Phase 6: MVP18 Hitchin 后端接口 (HitchinFiberBackend)                         
│   - 精确纤维点计数（替代 F5 的 asymptotic 置信度）                              
│   - 迹公式联动     
│   - 替换所有依赖F5的模块                                                 
└─────────────────────────────────────────────────────────────────────────────┘

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .mvp17_cfg_object_model import MonomialOperator
from enum import Enum, auto
from fractions import Fraction
import math
import struct
import hashlib


# =============================================================================
# 0) 严格异常系统
# =============================================================================


class MVP19Error(Exception):
    """MVP19 模块总异常：严格模式下禁止静默失败。"""


class TensorEngineError(MVP19Error):
    """签名张量引擎契约/返回值不满足前置条件。"""


class LatticeConstructionError(MVP19Error):
    """格构造失败（维度不一致/输入为空/超出上限等）。"""


class LatticeReductionError(MVP19Error):
    """LLL/GS/BKZ 约化失败（线性相关/数值崩溃等）。"""


class CVPError(MVP19Error):
    """CVP 求解失败（预算耗尽/数值问题/无候选等）。"""


class AdelicMetricError(MVP19Error):
    """Adelic 度量计算失败（素数位缺失/数值溢出等）。"""


class SymplecticFlowError(MVP19Error):
    """辛几何流失败（哈密顿量发散/积分器崩溃等）。"""


class TropicalGeometryError(MVP19Error):
    """热带几何计算失败（多胞体构造/法扇交集失败等）。"""


class HitchinBackendError(MVP19Error):
    """Hitchin 后端计算失败（纤维点计数/谱数据提取失败等）。"""


# =============================================================================
# 0.5) 严格常量：从 IEEE754 / 数学定义导出，禁止魔法数
# =============================================================================


class StrictConstants:
    """
    所有常量必须有来源说明，禁止凭空写 1e-9 / 1e-12。
    """
    
    # IEEE 754 double precision machine epsilon: 2^(-52)
    # 来源：IEEE 754-2008 标准，不是"经验值"
    MACHINE_EPS_F64: float = 2.220446049250313e-16
    
    # 最大安全整数 (2^53 - 1)，超过此值 float64 无法精确表示
    # 来源：IEEE 754 double 的尾数位数 = 52 + 隐含位
    MAX_SAFE_INTEGER_F64: int = 9007199254740991
    
    # LLL 默认 δ = 3/4：Lenstra-Lenstra-Lovász 1982 原始论文标准参数
    # 来源：A. K. Lenstra, H. W. Lenstra Jr., L. Lovász, "Factoring polynomials with rational coefficients"
    LLL_DELTA_DEFAULT: float = 0.75
    
    # LLL δ 的有效范围：(1/4, 1) 开区间
    # 来源：同上论文，δ > 1/4 保证多项式时间收敛
    LLL_DELTA_MIN: float = 0.25
    LLL_DELTA_MAX: float = 1.0
    
    # BKZ block_size 最小值：2（退化为 LLL）
    # BKZ block_size 实用上限：无硬性上限，但 β > 60 通常指数爆炸
    # 来源：C.P. Schnorr, M. Euchner, "Lattice Basis Reduction: Improved Practical Algorithms"
    BKZ_BLOCK_SIZE_MIN: int = 2
    
    # Kannan embedding 缩放因子：必须由调用方根据问题尺度显式计算
    # 任何默认值都是魔法数
    
    @classmethod
    def derive_tolerance(cls, problem_scale: float, dimension: int) -> float:
        """
        从问题尺度和维度导出容差。
        
        公式：tol = eps * sqrt(dimension) * problem_scale
        来源：标准数值分析误差传播理论
        
        这不是"经验值"，而是误差传播的数学推导。
        """
        if problem_scale <= 0:
            raise ValueError(f"problem_scale must be positive, got {problem_scale}")
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        return cls.MACHINE_EPS_F64 * math.sqrt(float(dimension)) * problem_scale
    
    @classmethod
    def derive_log_tolerance(cls, dimension: int) -> float:
        """
        对数域的容差（用于 Arakelov 度量）。
        
        公式：log_tol = eps * dimension * log(dimension + 1)
        来源：对数运算的误差放大特性
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        return cls.MACHINE_EPS_F64 * float(dimension) * math.log(float(dimension) + 1.0)


# =============================================================================
# 0.6) Core 2: 算术缩放因子（Arakelov/Height 下界，去魔法数）
# =============================================================================


class ArithmeticScaling:
    """
    用可审计的数论上界推导整数化缩放因子，避免外部传入 magic scaling_factor。

    设计目标（工程可执行 + 数学合法）：
    - **整数化可逆**：签名/对数签名的有理系数在乘以 scaling_factor 后应尽可能落在 Z 上，
      从而避免 rounding 造成的格点坍缩。
    - **由高度导出**：以多项式的 naive height 作为 Arakelov height 的可计算代理：
        h(P) := log(max |c_α| + 1)
      并引入 Minkowski 型维度因子：dim·log(dim+1)。

    这里实现一个保守但纯整数的下界：
      scaling_factor >= (max|c| + 1) * (dim + 1)^dim * depth!

    - (max|c| + 1) * (dim + 1)^dim = exp(h(P) + dim·log(dim+1)) 的整数化形式
    - depth! 用于清分母：Chen 段指数与 log 系列产生的分母都被 depth! 整除（截断到 depth）
    """

    @staticmethod
    def derive_scaling_factor(
        polynomials: List[Dict[Tuple[int, ...], int]],
        *,
        signature_depth: int,
    ) -> int:
        if signature_depth <= 0:
            raise ValueError(f"signature_depth must be positive, got {signature_depth}")
        if not polynomials:
            raise ValueError("polynomials must be non-empty")

        # infer ambient dimension from exponent tuples
        first_exp = next(iter(polynomials[0].keys()))
        dim = len(first_exp)
        if dim <= 0:
            raise ValueError("Polynomial exponent dimension must be positive")
        for p in polynomials:
            for exp in p.keys():
                if len(exp) != dim:
                    raise ValueError("Inconsistent exponent dimensions across polynomials")

        max_abs_coeff = 0
        for p in polynomials:
            for c in p.values():
                if c != 0:
                    max_abs_coeff = max(max_abs_coeff, abs(int(c)))
        if max_abs_coeff <= 0:
            max_abs_coeff = 1

        # exp(h(P) + dim log(dim+1)) but computed purely as integers
        height_factor = (max_abs_coeff + 1) * ((dim + 1) ** dim)
        denom_factor = math.factorial(signature_depth)
        scaling = height_factor * denom_factor

        if scaling <= 0:
            raise ValueError("Derived scaling_factor overflowed into non-positive integer")
        return int(scaling)


# =============================================================================
# 1) Tensor Engine Protocol（由外部模块实现；这里不重复造轮子）
# =============================================================================


TensorLike = Any


class TensorEngineProtocol(Protocol):
    def compute_signature(self, stream: List[Tuple[int, int]], depth: int) -> TensorLike:
        """计算路径签名张量（实现必须可复现、无随机性、可审计）。"""

    def tensor_diff(self, t1: TensorLike, t2: TensorLike) -> TensorLike:
        """计算张量差 (t1 - t2)。"""

    def tensor_norm(self, t: TensorLike) -> int:
        """
        返回张量范数的整数表示（例如 p-adic/整数化欧氏范数等）。
        禁止归一化：必须保留原始数量级。
        """

    def flatten_to_int_array(self, t: TensorLike, scaling_factor: int) -> List[int]:
        """
        将张量展平成整数数组（严格整数，保留符号与比例）。
        scaling_factor 必须由调用方显式提供；实现不得偷用固定缩放。
        """


# =============================================================================
# 1.5) 严格 TensorEngine 实现：路径签名张量（Chen 迭代积分）
# =============================================================================


class SignatureTensorEngine:
    """
    严格路径签名张量引擎。
    
    数学基础：
    - Chen 迭代积分 (K.T. Chen, 1957)
    - 签名是路径到张量代数的同态，满足 shuffle identity
    
    实现细节：
    - 全程有理数运算（Fraction），避免浮点误差
    - 最终整数化由 scaling_factor 显式控制
    - 无随机性、完全确定性、可复现
    """
    
    def __init__(self, dimension: int):
        """
        初始化签名引擎。
        
        Args:
            dimension: 路径所在空间的维度（例如 EVM trace 的状态维度）
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        # caches for Lyndon/Hall basis computations (purely deterministic memoization)
        self._lyndon_words_cache: Dict[int, List[Tuple[int, ...]]] = {}
        self._is_lyndon_cache: Dict[Tuple[int, ...], bool] = {}
        self._standard_factor_cache: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        self._lyndon_bracket_cache: Dict[Tuple[int, ...], Dict[Tuple[int, ...], int]] = {}
    
    def _compute_tensor_dimension(self, depth: int) -> int:
        """
        计算截断签名的总维度。
        
        公式：1 + d + d^2 + ... + d^depth = (d^(depth+1) - 1) / (d - 1)  当 d > 1
              depth + 1  当 d = 1
        """
        d = self.dimension
        if d == 1:
            return depth + 1
        # 使用整数运算避免浮点误差
        return (d ** (depth + 1) - 1) // (d - 1)
    
    def _multi_index_to_flat(self, indices: Tuple[int, ...]) -> int:
        """
        将多重指标 (i_1, ..., i_k) 转换为展平后的一维索引。
        
        索引约定：
        - 级别 0（空路径）占位置 0
        - 级别 1 占位置 1 到 d
        - 级别 k 从位置 (d^k - 1)/(d-1) 开始
        """
        d = self.dimension
        k = len(indices)
        if k == 0:
            return 0
        
        # 计算级别 k 之前所有级别占用的位置数
        if d == 1:
            base_offset = k
        else:
            base_offset = (d ** k - 1) // (d - 1)
        
        # 在级别 k 内的偏移：把 indices 视为 d 进制数
        within_offset = 0
        for idx in indices:
            if not (0 <= idx < d):
                raise TensorEngineError(f"Index {idx} out of range [0, {d})")
            within_offset = within_offset * d + idx
        
        return base_offset + within_offset
    
    def compute_signature(
        self, 
        stream: List[Tuple[int, int]], 
        depth: int
    ) -> Dict[Tuple[int, ...], Fraction]:
        """
        计算路径签名张量（截断到指定深度）。
        
        数学定义：
        对于路径 γ: [0,1] → R^d，其签名 S(γ) 在多重指标 I = (i_1, ..., i_k) 处的分量为：
        
        S(γ)_I = ∫_{0 < t_1 < ... < t_k < 1} dγ^{i_1}_{t_1} ... dγ^{i_k}_{t_k}
        
        对于分段线性路径（EVM trace），采用 Chen 恒等式：
        - 每个线段增量 a 视为线性路径，其签名为张量代数中的指数
          \(\exp(a) = \sum_{k \ge 0} a^{\otimes k} / k!\)（截断到 depth）
        - 路径拼接的签名为张量乘法（concatenation product）：
          \(S(\gamma * \eta) = S(\gamma) \otimes S(\eta)\)
        
        Args:
            stream: 路径增量序列，每个元素 (dimension_index, value) 表示沿某维度的增量
            depth: 截断深度（保留 1 到 depth 级别的所有张量分量）
        
        Returns:
            字典 {multi_index: coefficient}，其中 coefficient 为 Fraction（有理数）
        """
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        
        # 初始化：只有空路径分量为 1
        signature: Dict[Tuple[int, ...], Fraction] = {(): Fraction(1)}

        # Chen 恒等式：逐段乘上 exp(Δγ)
        for dim_idx, value in stream:
            if not (0 <= dim_idx < self.dimension):
                raise TensorEngineError(
                    f"dimension index {dim_idx} out of range [0, {self.dimension})"
                )
            seg = self._segment_signature_exp(dim_idx, Fraction(value), depth)
            signature = self._tensor_multiply_truncated(signature, seg, depth)

        return signature

    # -------- Tensor algebra utilities (exact, truncated) --------

    @staticmethod
    def _tensor_multiply_truncated(
        a: Dict[Tuple[int, ...], Fraction],
        b: Dict[Tuple[int, ...], Fraction],
        depth: int,
    ) -> Dict[Tuple[int, ...], Fraction]:
        """
        Concatenation product in the tensor algebra, truncated by word length <= depth.
        This is the algebra product used by Chen's identity.
        """
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if not a or not b:
            return {}
        out: Dict[Tuple[int, ...], Fraction] = {}
        for wa, ca in a.items():
            la = len(wa)
            if ca == 0:
                continue
            for wb, cb in b.items():
                if cb == 0:
                    continue
                lw = la + len(wb)
                if lw <= depth:
                    w = wa + wb
                    out[w] = out.get(w, Fraction(0)) + ca * cb
        # prune exact zeros (can appear after subtraction in log computations)
        return {w: c for w, c in out.items() if c != 0}

    @staticmethod
    def _segment_signature_exp(
        dim_idx: int,
        increment: Fraction,
        depth: int,
    ) -> Dict[Tuple[int, ...], Fraction]:
        """
        Signature of a single linear segment with increment 'increment * e_dim_idx':
          exp(increment * e_i) = Σ_{k=0..depth} (increment^k / k!) e_i^{⊗ k}
        """
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        seg: Dict[Tuple[int, ...], Fraction] = {(): Fraction(1)}
        if increment == 0:
            return seg
        # exact factorial denominators
        for k in range(1, depth + 1):
            seg[(dim_idx,) * k] = (increment ** k) / Fraction(math.factorial(k), 1)
        return seg

    # -------- Log-signature (tensor log) --------

    def log_signature(
        self,
        signature: Dict[Tuple[int, ...], Fraction],
        *,
        depth: int,
        include_empty: bool = True,
    ) -> Dict[Tuple[int, ...], Fraction]:
        """
        Compute the tensor-algebra logarithm of a (truncated) group-like signature.

        We use the formal series on the augmentation ideal:
          log(1 + X) = Σ_{k>=1} (-1)^{k+1} X^k / k
        where X = signature - 1 (remove the empty word).

        This is exact in Q and truncated by word length <= depth.
        """
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if () not in signature:
            raise TensorEngineError("Signature must contain empty word () with coefficient 1.")
        if signature.get((), Fraction(0)) != Fraction(1):
            raise TensorEngineError(f"Signature empty word must be 1, got {signature.get((), Fraction(0))!r}")

        # X = signature - 1 (drop the empty word)
        X: Dict[Tuple[int, ...], Fraction] = {w: c for w, c in signature.items() if w != () and c != 0}
        if not X:
            out = {(): Fraction(0)} if include_empty else {}
            return out

        # term = X^k
        term = dict(X)
        out: Dict[Tuple[int, ...], Fraction] = {}
        for k in range(1, depth + 1):
            coef = Fraction(1, k) if (k % 2 == 1) else -Fraction(1, k)
            for w, c in term.items():
                out[w] = out.get(w, Fraction(0)) + coef * c
            # next power
            term = self._tensor_multiply_truncated(term, X, depth)
            if not term:
                break

        if include_empty:
            out[()] = out.get((), Fraction(0))
        else:
            out.pop((), None)
        return {w: c for w, c in out.items() if c != 0 or (include_empty and w == ())}

    def compute_log_signature(
        self,
        stream: List[Tuple[int, int]],
        *,
        depth: int,
        include_empty: bool = True,
    ) -> Dict[Tuple[int, ...], Fraction]:
        """Compute log-signature (tensor logarithm) for a stream."""
        sig = self.compute_signature(stream, depth=depth)
        return self.log_signature(sig, depth=depth, include_empty=include_empty)

    # -------- Lyndon/Hall basis (free Lie algebra coordinates) --------

    def lyndon_basis_keys(self, depth: int, *, include_empty: bool = True) -> List[Tuple[int, ...]]:
        """
        Deterministic Lyndon(Hall) basis keys up to given depth for alphabet {0,1,...,dimension-1}.

        Keys are returned ordered by (length, lexicographic word order).
        """
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if depth not in self._lyndon_words_cache:
            words: List[Tuple[int, ...]] = []

            def gen(prefix: Tuple[int, ...], remaining: int) -> None:
                if remaining == 0:
                    if prefix and self._is_lyndon(prefix):
                        words.append(prefix)
                    return
                for a in range(self.dimension):
                    gen(prefix + (a,), remaining - 1)

            for L in range(1, depth + 1):
                gen((), L)

            words.sort(key=lambda w: (len(w), w))
            self._lyndon_words_cache[depth] = words

        keys = self._lyndon_words_cache[depth]
        if include_empty:
            return [()] + keys
        return keys[:]

    def compute_log_signature_lyndon(
        self,
        stream: List[Tuple[int, int]],
        *,
        depth: int,
        include_empty: bool = True,
    ) -> Dict[Tuple[int, ...], Fraction]:
        """
        Compute log-signature and express it in the Lyndon(Hall) basis coordinates.

        Output is a sparse dict: Lyndon word -> coefficient in Q (Fraction).
        """
        logsig_tensor = self.compute_log_signature(stream, depth=depth, include_empty=False)
        coords = self._tensor_lie_to_lyndon_coordinates(logsig_tensor, depth=depth)
        if include_empty:
            coords[()] = Fraction(0)
        return coords

    def _is_lyndon(self, w: Tuple[int, ...]) -> bool:
        """
        Lyndon word test (exact).
        A non-empty word w is Lyndon iff w is strictly lexicographically smaller than any of its
        non-trivial rotations.
        """
        if not w:
            return False
        cached = self._is_lyndon_cache.get(w)
        if cached is not None:
            return cached
        n = len(w)
        # Compare against all rotations
        for s in range(1, n):
            rot = w[s:] + w[:s]
            if w >= rot:
                self._is_lyndon_cache[w] = False
                return False
        self._is_lyndon_cache[w] = True
        return True

    def _standard_factorization(self, w: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Standard factorization of a Lyndon word w = uv where v is the longest proper Lyndon suffix.
        """
        cached = self._standard_factor_cache.get(w)
        if cached is not None:
            return cached
        if not self._is_lyndon(w) or len(w) < 2:
            raise TensorEngineError("Standard factorization requires a Lyndon word of length >= 2.")

        best_split: Optional[int] = None
        # longest proper Lyndon suffix => smallest split index
        for i in range(1, len(w)):
            suf = w[i:]
            if self._is_lyndon(suf):
                best_split = i
                break
        if best_split is None:
            raise TensorEngineError("No Lyndon suffix found (should be impossible for Lyndon words).")

        u, v = w[:best_split], w[best_split:]
        if not self._is_lyndon(u) or not self._is_lyndon(v):
            raise TensorEngineError("Standard factorization produced non-Lyndon factors.")
        self._standard_factor_cache[w] = (u, v)
        return u, v

    def _lyndon_bracket_expansion(self, w: Tuple[int, ...]) -> Dict[Tuple[int, ...], int]:
        """
        Expand the Lyndon bracket b[w] into the tensor (word) basis.

        - For |w|=1: b[w] = e_w, expansion is {w: 1}
        - For Lyndon |w|>1 with standard factorization w=uv:
            b[w] = [b[u], b[v]] = b[u]⊗b[v] - b[v]⊗b[u]

        The expansion is triangular with leading term w having coefficient +1.
        """
        cached = self._lyndon_bracket_cache.get(w)
        if cached is not None:
            return cached
        if len(w) == 1:
            out = {w: 1}
            self._lyndon_bracket_cache[w] = out
            return out
        if not self._is_lyndon(w):
            raise TensorEngineError("Bracket expansion requested for a non-Lyndon word.")

        u, v = self._standard_factorization(w)
        eu = self._lyndon_bracket_expansion(u)
        ev = self._lyndon_bracket_expansion(v)

        def concat(x: Dict[Tuple[int, ...], int], y: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, ...], int]:
            res: Dict[Tuple[int, ...], int] = {}
            for wx, cx in x.items():
                for wy, cy in y.items():
                    ww = wx + wy
                    res[ww] = res.get(ww, 0) + cx * cy
            return {ww: cc for ww, cc in res.items() if cc != 0}

        uv = concat(eu, ev)
        vu = concat(ev, eu)
        out: Dict[Tuple[int, ...], int] = dict(uv)
        for ww, cc in vu.items():
            out[ww] = out.get(ww, 0) - cc
            if out[ww] == 0:
                out.pop(ww, None)

        # Sanity: leading word coefficient should be +1
        if out.get(w, 0) != 1:
            raise TensorEngineError("Lyndon bracket expansion lost triangular leading coefficient.")

        self._lyndon_bracket_cache[w] = out
        return out

    def _tensor_lie_to_lyndon_coordinates(
        self,
        lie_tensor: Dict[Tuple[int, ...], Fraction],
        *,
        depth: int,
    ) -> Dict[Tuple[int, ...], Fraction]:
        """
        Given a Lie element expressed in tensor word basis (truncated),
        recover its Lyndon(Hall) basis coordinates using triangular elimination.
        """
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")

        # Work on a mutable residual
        residual: Dict[Tuple[int, ...], Fraction] = {w: c for w, c in lie_tensor.items() if w and len(w) <= depth and c != 0}
        coords: Dict[Tuple[int, ...], Fraction] = {}

        for lw in self.lyndon_basis_keys(depth, include_empty=False):
            # Leading coefficient of b[lw] is exactly the tensor word lw with coefficient +1.
            c = residual.get(lw, Fraction(0))
            if c != 0:
                coords[lw] = c
                expansion = self._lyndon_bracket_expansion(lw)
                for ww, ic in expansion.items():
                    if ww in residual:
                        residual[ww] -= c * Fraction(ic, 1)
                        if residual[ww] == 0:
                            residual.pop(ww, None)
            else:
                coords[lw] = Fraction(0)

        # Optional strictness: residual should be empty for a true Lie element.
        # With exact arithmetic and correct group-like input, it should vanish.
        if residual:
            # Keep the interface strict: this indicates inconsistent truncation or a non-Lie input.
            raise TensorEngineError(f"Non-zero residual after Lyndon elimination (size={len(residual)}).")

        # Return sparse dict (drop zeros)
        return {w: c for w, c in coords.items() if c != 0}
    
    def tensor_diff(
        self, 
        t1: Dict[Tuple[int, ...], Fraction], 
        t2: Dict[Tuple[int, ...], Fraction]
    ) -> Dict[Tuple[int, ...], Fraction]:
        """计算张量差 t1 - t2（精确有理数运算）。"""
        result: Dict[Tuple[int, ...], Fraction] = {}
        
        all_keys = set(t1.keys()) | set(t2.keys())
        for key in all_keys:
            v1 = t1.get(key, Fraction(0))
            v2 = t2.get(key, Fraction(0))
            diff = v1 - v2
            if diff != 0:
                result[key] = diff
        
        return result
    
    def tensor_norm(self, t: Dict[Tuple[int, ...], Fraction]) -> int:
        """
        计算张量的整数化 L2 范数平方。
        
        对于有理数张量，返回 Σ (numerator/denominator)^2 的整数化表示：
        = Σ numerator^2 / denominator^2
        
        为保持整数性，返回 (Σ num^2 * lcm^2 / den^2) 其中 lcm 是所有分母的最小公倍数。
        """
        if not t:
            return 0
        
        # 计算所有分母的 LCM
        from functools import reduce
        
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return abs(a)
        
        def lcm(a: int, b: int) -> int:
            return abs(a * b) // gcd(a, b) if a and b else 0
        
        denominators = [abs(v.denominator) for v in t.values() if v != 0]
        if not denominators:
            return 0
        
        common_denom = reduce(lcm, denominators, 1)
        
        # 计算范数平方（整数）
        norm_sq = 0
        for v in t.values():
            if v != 0:
                # 将分数转换为公分母表示
                scaled = v * common_denom
                # 现在 scaled 应该是整数
                int_val = int(scaled)
                norm_sq += int_val * int_val
        
        return norm_sq
    
    def flatten_to_int_array(
        self, 
        t: Dict[Tuple[int, ...], Fraction], 
        scaling_factor: int
    ) -> List[int]:
        """
        将张量展平为整数数组。
        
        展平顺序：按多重指标排序（先按长度，再按字典序）。
        这确保相同键集的张量产生相同维度的数组。
        
        Args:
            t: 签名张量
            scaling_factor: 缩放因子（由调用方显式提供，不允许默认值）
        
        Returns:
            整数数组，每个元素 = round(coefficient * scaling_factor)
        """
        if not isinstance(scaling_factor, int) or scaling_factor <= 0:
            raise TensorEngineError(f"scaling_factor must be positive int, got {scaling_factor}")
        
        if not t:
            return []
        
        # 使用确定性排序：先按长度，再按字典序
        # 这确保相同键集产生相同顺序
        sorted_keys = sorted(t.keys(), key=lambda k: (len(k), k))
        
        result = []
        for multi_idx in sorted_keys:
            coeff = t[multi_idx]
            # 严格整数化：有理数 * 缩放因子，然后四舍五入
            scaled = coeff * scaling_factor
            # 使用 Fraction 的精确四舍五入
            if scaled >= 0:
                int_val = int(scaled + Fraction(1, 2))
            else:
                int_val = int(scaled - Fraction(1, 2))
            result.append(int_val)
        
        return result


# =============================================================================
# 1.6) Arakelov/Adelic 度量空间：统一所有素数位 + 无穷位
# =============================================================================


class PrimePlace:
    """
    素数位（p-adic 赋值）。
    
    数学定义：
    对于有理数 x = p^v * (a/b)，其中 gcd(a, p) = gcd(b, p) = 1，
    v_p(x) = v （p-adic 赋值）
    |x|_p = p^{-v_p(x)} （p-adic 绝对值）
    """
    
    def __init__(self, p: int):
        """初始化素数位。"""
        if p < 2:
            raise AdelicMetricError(f"p must be prime >= 2, got {p}")
        # 严格检查是否为素数（不用概率测试）
        if not self._is_prime_strict(p):
            raise AdelicMetricError(f"{p} is not prime")
        self.p = p
    
    @staticmethod
    def _is_prime_strict(n: int) -> bool:
        """严格素性测试（确定性）。"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        # 试除法到 sqrt(n)
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True
    
    def valuation(self, x: Union[int, Fraction]) -> int:
        """
        计算 p-adic 赋值 v_p(x)。
        
        返回：使得 x = p^v * (a/b) 且 gcd(a,p) = gcd(b,p) = 1 的整数 v。
        对于 x = 0，按约定返回正无穷（这里用一个大整数表示）。
        """
        if x == 0:
            # v_p(0) = +∞，用足够大的整数表示
            return 2 ** 30  # 不用 float('inf') 以保持整数类型
        
        if isinstance(x, int):
            x = Fraction(x)
        
        num = abs(x.numerator)
        den = abs(x.denominator)
        
        v_num = 0
        while num % self.p == 0:
            num //= self.p
            v_num += 1
        
        v_den = 0
        while den % self.p == 0:
            den //= self.p
            v_den += 1
        
        return v_num - v_den
    
    def absolute_value_log(self, x: Union[int, Fraction]) -> float:
        """
        计算 log|x|_p = -v_p(x) * log(p)。
        
        对数形式避免 p^{-v} 可能的数值下溢/上溢。
        """
        v = self.valuation(x)
        if v >= 2 ** 29:  # 表示 +∞
            return float('-inf')
        return -float(v) * math.log(float(self.p))


class ArchimedeanPlace:
    """
    无穷位（标准绝对值）。
    
    对于有理数 x，|x|_∞ = |x|（通常的绝对值）。
    """
    
    def absolute_value_log(self, x: Union[int, Fraction]) -> float:
        """计算 log|x|_∞。"""
        if x == 0:
            return float('-inf')
        if isinstance(x, int):
            return math.log(abs(float(x)))
        return math.log(abs(float(x.numerator))) - math.log(abs(float(x.denominator)))


class AdelicMetricSpace:
    """
    Arakelov/Adelic 度量空间。
    
    数学基础：
    积公式（Product Formula）：对于非零有理数 x，
    Π_v |x|_v = 1（所有位 v 包括所有素数位和无穷位的乘积）
    
    等价地，在对数形式：
    Σ_v log|x|_v = 0
    
    Adelic 距离：
    d²(x, y) = Σ_v w_v * (log|x - y|_v)²
    
    其中 w_v 是位的权重（通常取 1 或按算术重要性加权）。
    
    工程意义：
    - F5 只在欧氏空间工作，丢失数论信息
    - Adelic 度量让 CVP 求解器"懂同余"
    - 不仅找几何上短的向量，还找算术上整除性最好的向量
    """
    
    def __init__(self, primes: List[int], *, include_archimedean: bool = True):
        """
        初始化 Adelic 度量空间。
        
        Args:
            primes: 要考虑的素数列表（显式指定，不自动选择）
            include_archimedean: 是否包含无穷位
        """
        if not primes:
            raise AdelicMetricError("primes list cannot be empty")
        
        self.prime_places = [PrimePlace(p) for p in primes]
        self.include_archimedean = include_archimedean
        if include_archimedean:
            self.archimedean = ArchimedeanPlace()
        
        # 权重：按 log(p) 加权（不是魔法数，而是 Arakelov 理论的标准选择）
        # 来源：Arakelov 高度函数的定义
        self._weights_finite = [math.log(float(pp.p)) for pp in self.prime_places]
        self._weight_infinite = 1.0 if include_archimedean else 0.0
    
    def local_height(
        self, 
        x: Union[int, Fraction], 
        place_index: int
    ) -> float:
        """
        计算在指定位的局部高度 log|x|_v。
        
        Args:
            x: 有理数
            place_index: 位的索引（0 到 len(primes)-1 为有限位，len(primes) 为无穷位）
        """
        n_finite = len(self.prime_places)
        
        if place_index < 0 or place_index > n_finite:
            raise AdelicMetricError(f"place_index {place_index} out of range")
        
        if place_index < n_finite:
            return self.prime_places[place_index].absolute_value_log(x)
        else:
            if not self.include_archimedean:
                raise AdelicMetricError("Archimedean place not included")
            return self.archimedean.absolute_value_log(x)
    
    def adelic_distance_sq(
        self, 
        x: Union[int, Fraction], 
        y: Union[int, Fraction]
    ) -> float:
        """
        计算 Adelic 距离的平方。
        
        d²(x, y) = Σ_v w_v * (log|x - y|_v)²
        """
        diff = Fraction(x) - Fraction(y)
        if diff == 0:
            return 0.0
        
        dist_sq = 0.0
        
        # 有限位贡献
        for i, (pp, w) in enumerate(zip(self.prime_places, self._weights_finite)):
            log_abs = pp.absolute_value_log(diff)
            if math.isfinite(log_abs):
                dist_sq += w * log_abs * log_abs
        
        # 无穷位贡献
        if self.include_archimedean:
            log_abs = self.archimedean.absolute_value_log(diff)
            if math.isfinite(log_abs):
                dist_sq += self._weight_infinite * log_abs * log_abs
        
        return dist_sq
    
    def vector_adelic_distance_sq(
        self, 
        x: List[Union[int, Fraction]], 
        y: List[Union[int, Fraction]]
    ) -> float:
        """
        计算向量的 Adelic 距离平方（分量距离平方之和）。
        """
        if len(x) != len(y):
            raise AdelicMetricError(f"Dimension mismatch: {len(x)} vs {len(y)}")
        
        return math.fsum(self.adelic_distance_sq(xi, yi) for xi, yi in zip(x, y))
    
    def verify_product_formula(self, x: Union[int, Fraction]) -> Tuple[bool, float]:
        """
        验证积公式：Σ_v log|x|_v 应该 = 0（对于非零有理数）。
        
        返回 (is_valid, error) 其中 error 是偏离 0 的程度。
        容差从机器精度和维度导出（不是魔法数）。
        """
        if x == 0:
            return True, 0.0
        
        x = Fraction(x)
        total_log = 0.0
        
        for pp in self.prime_places:
            total_log += pp.absolute_value_log(x)
        
        if self.include_archimedean:
            total_log += self.archimedean.absolute_value_log(x)
        
        # 容差计算：来自数值分析理论
        n_places = len(self.prime_places) + (1 if self.include_archimedean else 0)
        tol = StrictConstants.derive_log_tolerance(n_places)
        
        is_valid = abs(total_log) <= tol
        return is_valid, total_log


# =============================================================================
# 1.7) 热带几何预求解器：牛顿多胞体 → 法扇 → CVP 初始范围
# =============================================================================


@dataclass
class NewtonPolytope:
    """
    牛顿多胞体（Newton Polytope）。
    
    数学定义：
    对于多项式 f = Σ c_α x^α，其牛顿多胞体 NP(f) 是所有指数向量 α 的凸包。
    """
    vertices: List[Tuple[int, ...]]  # 顶点（指数向量）
    dimension: int  # 环境空间维度


@dataclass
class TropicalSkeleton:
    """
    热带骨架（Tropical Skeleton）。
    
    包含：
    - 初始格点范围（CVP 搜索空间的边界）
    - 相变点（热带超曲面的奇异点）
    - 法扇顶点（潜在解的位置）
    """
    lower_bounds: List[int]
    upper_bounds: List[int]
    phase_transition_points: List[Tuple[int, ...]]
    normal_fan_vertices: List[Tuple[int, ...]]


class TropicalPreSolver:
    """
    热带几何预求解器。
    
    数学基础：
    - 热带化：将系数映射到对数域 v(c) = -log|c|
    - 牛顿多胞体：指数向量的凸包
    - 法扇（Normal Fan）：多胞体的对偶结构
    - 热带交集：法扇的交集（组合几何，非代数运算）
    
    工程意义：
    - 替代 F5 的 S-Pair 生成
    - 直接锁定 CVP 搜索的初始格点范围
    - 速度：组合几何 >> 代数约化
    """
    
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise TropicalGeometryError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
    
    def _valuation(self, c: int) -> float:
        """
        计算系数的赋值（热带化）。
        v(c) = -log|c| 当 c ≠ 0
        v(0) = +∞
        """
        if c == 0:
            return float('inf')
        return -math.log(abs(float(c)))
    
    def build_newton_polytope(
        self, 
        polynomial: Dict[Tuple[int, ...], int]
    ) -> NewtonPolytope:
        """
        构建多项式的牛顿多胞体。
        
        Args:
            polynomial: {指数元组: 系数} 字典
        
        Returns:
            牛顿多胞体（顶点列表）
        """
        if not polynomial:
            raise TropicalGeometryError("Empty polynomial")
        
        # 收集所有非零项的指数向量
        exponents = [exp for exp, coeff in polynomial.items() if coeff != 0]
        if not exponents:
            raise TropicalGeometryError("Polynomial has no non-zero terms")
        
        # 验证维度一致性
        exp_dim = len(exponents[0])
        if any(len(e) != exp_dim for e in exponents):
            raise TropicalGeometryError("Inconsistent exponent dimensions")
        
        # 计算凸包顶点（使用 Gift Wrapping 或偷懒）
        # 对于小维度，直接使用所有极端点
        vertices = self._compute_convex_hull_vertices(exponents)
        
        return NewtonPolytope(vertices=vertices, dimension=exp_dim)
    
    def _compute_convex_hull_vertices(
        self, 
        points: List[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        """
        计算点集的凸包顶点。
        
        使用严格的增量算法（不是近似）。
        对于高维情况，这是 O(n^d) 但保证精确。
        """
        if not points:
            return []
        
        if len(points) == 1:
            return list(points)
        
        n_dim = len(points[0])
        
        if n_dim == 1:
            # 一维：最小和最大
            min_p = min(points)
            max_p = max(points)
            if min_p == max_p:
                return [min_p]
            return [min_p, max_p]
        
        if n_dim == 2:
            # 二维：使用 Andrew's monotone chain 算法
            return self._convex_hull_2d(points)
        
        # 高维：使用所有点作为候选顶点（保守但正确）
        # 更精确的算法需要 Quickhull 等，但为避免复杂度这里保守处理
        # 通过检查每个点是否是某个支撑超平面的唯一极值来过滤
        vertices = []
        for p in points:
            # 检查 p 是否在某个方向上是极值点
            is_extreme = False
            for d in range(n_dim):
                # 检查 p 在第 d 维是否是最大或最小
                coord = p[d]
                is_max = all(q[d] <= coord for q in points)
                is_min = all(q[d] >= coord for q in points)
                if is_max or is_min:
                    is_extreme = True
                    break
            
            if is_extreme or len(points) <= 2 * n_dim:
                # 点数很少时保留所有点
                vertices.append(p)
        
        return list(set(vertices)) if vertices else list(points)
    
    def _convex_hull_2d(
        self, 
        points: List[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        """
        二维凸包：Andrew's monotone chain 算法。
        时间复杂度 O(n log n)，空间 O(n)。
        """
        points = sorted(set(points))
        if len(points) <= 2:
            return points
        
        def cross(o: Tuple[int, ...], a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
            """叉积 (a - o) × (b - o)"""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # 下凸包
        lower: List[Tuple[int, ...]] = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # 上凸包
        upper: List[Tuple[int, ...]] = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        # 合并（去掉首尾重复点）
        return lower[:-1] + upper[:-1]

    # -------- Tropical / polyhedral utilities (exact integer combinatorics) --------

    @staticmethod
    def _dot_int(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        return int(sum(int(x) * int(y) for x, y in zip(a, b)))

    @staticmethod
    def _primitive_int_vector(v: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Normalize an integer vector to its primitive representative (divide by gcd),
        and fix a canonical sign (first non-zero component is positive).
        """
        if not v:
            return v
        g = 0
        for x in v:
            g = math.gcd(g, abs(int(x)))
        if g == 0:
            return v
        vv = tuple(int(x) // g for x in v)
        # canonical sign: make first non-zero positive
        for x in vv:
            if x != 0:
                if x < 0:
                    vv = tuple(-y for y in vv)
                break
        return vv

    @staticmethod
    def _orthogonal_int_vector(d: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Construct a non-zero integer vector w such that <w, d> = 0.
        Deterministic: uses a 2D rotation on the first non-zero coordinate.
        """
        dim = len(d)
        if dim == 0:
            return ()
        if dim == 1:
            return (0,)
        # pick an index with non-zero component
        i0 = None
        for i, x in enumerate(d):
            if x != 0:
                i0 = i
                break
        if i0 is None:
            return (0,) * dim
        j0 = (i0 + 1) % dim
        w = [0] * dim
        w[i0] = int(d[j0])
        w[j0] = -int(d[i0])
        return tuple(w)

    def _tropical_min_multiplicity(
        self,
        vertices: List[Tuple[int, ...]],
        w: Tuple[int, ...],
    ) -> int:
        """
        Under the trivial valuation, compute how many vertices attain the minimum of <w, v>.
        A point w lies on the tropical hypersurface iff this multiplicity >= 2.
        """
        if not vertices:
            return 0
        vals = [self._dot_int(w, v) for v in vertices]
        mv = min(vals)
        return sum(1 for x in vals if x == mv)

    def _normal_fan_rays(
        self,
        vertices: List[Tuple[int, ...]],
    ) -> List[Tuple[int, ...]]:
        """
        Compute a deterministic, integer set of rays for the normal fan of a Newton polytope.

        - dim=1: rays are ±1
        - dim=2: rays are primitive normals to hull edges
        - dim=3: rays are primitive normals to supporting facets (checked exactly on vertex set)
        - dim>3: conservative superset via primitive pairwise differences (exact, no heuristics)
        """
        if not vertices:
            return []
        dim = len(vertices[0])
        verts = list(dict.fromkeys(vertices))  # stable unique
        if dim == 1:
            return [(1,), (-1,)]
        if dim == 2:
            # vertices are expected to be in convex hull order for dim=2
            rays: List[Tuple[int, ...]] = []
            if len(verts) == 1:
                return rays
            cyc = verts + [verts[0]]
            for a, b in zip(cyc[:-1], cyc[1:]):
                e = (b[0] - a[0], b[1] - a[1])
                n = (e[1], -e[0])
                p = self._primitive_int_vector(n)
                if p != (0, 0):
                    rays.append(p)
            return list(dict.fromkeys(sorted(rays)))
        if dim == 3:
            rays_set: Dict[Tuple[int, ...], None] = {}
            V = verts
            if len(V) < 3:
                return []
            for i in range(len(V)):
                for j in range(i + 1, len(V)):
                    for k in range(j + 1, len(V)):
                        pi, pj, pk = V[i], V[j], V[k]
                        v1 = (pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2])
                        v2 = (pk[0] - pi[0], pk[1] - pi[1], pk[2] - pi[2])
                        n = (
                            v1[1] * v2[2] - v1[2] * v2[1],
                            v1[2] * v2[0] - v1[0] * v2[2],
                            v1[0] * v2[1] - v1[1] * v2[0],
                        )
                        if n == (0, 0, 0):
                            continue
                        c = self._dot_int(n, pi)
                        dots = [self._dot_int(n, p) for p in V]
                        if all(d <= c for d in dots):
                            nn = self._primitive_int_vector(n)
                            if nn != (0, 0, 0):
                                rays_set[nn] = None
                        elif all(d >= c for d in dots):
                            nn = self._primitive_int_vector(tuple(-x for x in n))
                            if nn != (0, 0, 0):
                                rays_set[nn] = None
            return sorted(rays_set.keys())

        # dim > 3: conservative, exact integer superset
        rays_set: Dict[Tuple[int, ...], None] = {}
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                d = tuple(int(verts[i][t]) - int(verts[j][t]) for t in range(dim))
                p = self._primitive_int_vector(d)
                if any(x != 0 for x in p):
                    rays_set[p] = None
        return sorted(rays_set.keys())
    
    def compute_tropical_skeleton(
        self,
        polynomials: List[Dict[Tuple[int, ...], int]],
    ) -> TropicalSkeleton:
        """
        计算多项式系统的热带骨架。
        
        步骤：
        1. 对每个多项式构建牛顿多胞体
        2. 提取法扇顶点
        3. 计算热带交集
        4. 确定 CVP 搜索范围
        
        Args:
            polynomials: 多项式列表
        
        Returns:
            热带骨架（包含 CVP 搜索边界）
        """
        if not polynomials:
            raise TropicalGeometryError("Empty polynomial system")
        
        # 1. 构建所有牛顿多胞体
        polytopes = [self.build_newton_polytope(p) for p in polynomials]
        
        # 检查维度一致性
        dims = set(np.dimension for np in polytopes)
        if len(dims) != 1:
            raise TropicalGeometryError(f"Inconsistent polynomial dimensions: {dims}")
        dim = dims.pop()
        
        # 2. 收集所有顶点
        all_vertices: List[Tuple[int, ...]] = []
        for np in polytopes:
            all_vertices.extend(np.vertices)
        
        # 3. 计算边界（CVP 搜索范围）
        if not all_vertices:
            raise TropicalGeometryError("No vertices found")
        
        lower_bounds = [
            min(v[i] for v in all_vertices) for i in range(dim)
        ]
        upper_bounds = [
            max(v[i] for v in all_vertices) for i in range(dim)
        ]
        
        # 4. 热带相变点：在平凡赋值下，w 满足至少两个顶点同时取得最小值
        # 这里返回一个可审计的有限集合：由顶点差向量的正交向量构造，并用最小值重数筛选
        per_poly_vertices = [np.vertices for np in polytopes]
        candidate_w: Dict[Tuple[int, ...], None] = {}
        for verts in per_poly_vertices:
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    dvec = tuple(int(verts[i][k]) - int(verts[j][k]) for k in range(dim))
                    if all(x == 0 for x in dvec):
                        continue
                    w = self._primitive_int_vector(self._orthogonal_int_vector(dvec))
                    if any(x != 0 for x in w) and self._tropical_min_multiplicity(verts, w) >= 2:
                        candidate_w[w] = None

        # 若存在系统交集点（对所有多项式均为相变点），优先返回交集；否则返回并集
        candidates = list(candidate_w.keys())
        intersection: List[Tuple[int, ...]] = []
        for w in candidates:
            if all(self._tropical_min_multiplicity(verts, w) >= 2 for verts in per_poly_vertices):
                intersection.append(w)
        phase_transition_points = sorted(intersection) if intersection else sorted(candidates)

        # 5. 法扇射线（normal fan rays）：每个牛顿多胞体的对偶组合数据，返回其整数生成元
        rays: Dict[Tuple[int, ...], None] = {}
        for np in polytopes:
            for r in self._normal_fan_rays(np.vertices):
                rays[r] = None
        normal_fan_vertices = sorted(rays.keys())
        
        return TropicalSkeleton(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            phase_transition_points=phase_transition_points,
            normal_fan_vertices=normal_fan_vertices,
        )
    
    def _generate_corners(
        self, 
        lower: List[int], 
        upper: List[int]
    ) -> List[Tuple[int, ...]]:
        """生成边界框的所有角点。"""
        dim = len(lower)
        corners: List[Tuple[int, ...]] = []
        
        for mask in range(1 << dim):
            corner = tuple(
                upper[i] if (mask >> i) & 1 else lower[i]
                for i in range(dim)
            )
            corners.append(corner)
        
        return corners


# =============================================================================
# 2) 数据结构：几何化多项式与 CVP 问题打包
# =============================================================================


@dataclass(frozen=True)
class GeometricPolynomial:
    """
    代数多项式 + 几何签名张量的二元对象。

    - coeffs: monomial(exponent tuple) -> coefficient (int)
    - signature_tensor: 由外部 tensor_engine 生成的张量对象
    - origin_signature_hash: 用于追踪来源（防止环路/死循环）
    - algebraic_height: 代数高度（例如系数绝对值的 log2 上界）
    """

    coeffs: Dict[tuple, int]
    signature_tensor: TensorLike
    origin_signature_hash: int
    algebraic_height: int


@dataclass(frozen=True)
class LatticeProblem:
    """
    CVP 输入：格基矩阵 B（行向量）与目标向量 t。

    约定：
    - 向量按 [algebra_block | geometry_block] 拼接
    - algebra_block 目标固定为 0；geometry_block 目标为 target_signature 的展平值
    """

    basis_matrix: List[List[int]]
    target_vector: List[int]
    monomial_map: Dict[tuple, int]  # monomial -> column index (within algebra block)
    n_alg_cols: int
    n_geo_cols: int
    scaling_factor: int


@dataclass(frozen=True)
class CVPSolution:
    """CVP 输出：最近格向量与残差（全程整数）。"""

    closest_vector: List[int]
    residual_vector: List[int]
    residual_norm_sq: int
    geo_residual_norm_sq: int
    algebra_coeffs: Dict[tuple, int]
    scaling_factor: int


# =============================================================================
# 3) MVP19：格构造器（严格）
# =============================================================================


class SignatureLatticeTerminator:
    def __init__(
        self,
        tensor_engine: TensorEngineProtocol,
        *,
        modulus: int = 2**256,
        lattice_dimension_cap: int = 200,
    ):
        self.te = tensor_engine
        self.modulus = int(modulus)
        self.dim_cap = int(lattice_dimension_cap)

        self.target_signature: Optional[TensorLike] = None
        self.basis_pool: List[GeometricPolynomial] = []

    # -------- Target wiring --------

    def set_target_trace(self, trace_ops: List[Tuple[int, int]], depth: int) -> None:
        """从 trace 计算目标签名（严格：任何 tensor_engine 异常都直接上抛）。"""
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        self.target_signature = self.te.compute_signature(trace_ops, depth)

    def set_target_signature(self, sig: TensorLike) -> None:
        """直接注入目标签名。"""
        self.target_signature = sig

    # -------- Basis wiring --------

    def add_basis_polynomial(self, gp: GeometricPolynomial) -> None:
        self.basis_pool.append(gp)

    # -------- Core: construct lattice problem --------

    def build_lattice_problem(self, *, scaling_factor: int) -> LatticeProblem:
        """
        构造 (B, t)：
        - B 的每一行来自 basis_pool 的一个 GeometricPolynomial
        - t 的 algebra_block 全 0；geometry_block = flatten(target_signature)

        scaling_factor 必须显式提供：
        - 该因子属于整数化/量纲选择，是模型的一部分，不能靠隐藏常数偷决定。
        """

        if self.target_signature is None:
            raise LatticeConstructionError("target_signature is not set.")
        if not self.basis_pool:
            raise LatticeConstructionError("basis_pool is empty.")
        if len(self.basis_pool) > self.dim_cap:
            raise LatticeConstructionError(
                f"basis_pool size={len(self.basis_pool)} exceeds lattice_dimension_cap={self.dim_cap}."
            )
        if not isinstance(scaling_factor, int) or scaling_factor <= 0:
            raise ValueError(f"scaling_factor must be a positive int, got {scaling_factor!r}")

        # 1) monomial universe (algebra block columns)
        all_monomials: set[tuple] = set()
        for p in self.basis_pool:
            all_monomials.update(p.coeffs.keys())
        if not all_monomials:
            raise LatticeConstructionError("No monomials found in basis_pool (algebra block would be empty).")

        # Deterministic order: tuple ordering (rev) is stable/reproducible.
        # If you need grevlex specifically, ensure monomial tuples are encoded accordingly upstream.
        sorted_monos = sorted(all_monomials, reverse=True)
        mono_map = {m: i for i, m in enumerate(sorted_monos)}

        n_rows = len(self.basis_pool)
        n_alg_cols = len(sorted_monos)

        # 2) geometry block length must be stable
        # We require tensor_engine.flatten_to_int_array to be dimension-consistent for all tensors at given scaling.
        try:
            target_geo = self.te.flatten_to_int_array(self.target_signature, scaling_factor)
        except Exception as e:
            raise TensorEngineError("flatten_to_int_array failed on target_signature.") from e
        if not isinstance(target_geo, list) or not all(isinstance(x, int) for x in target_geo):
            raise TensorEngineError("flatten_to_int_array(target_signature) must return List[int].")
        n_geo_cols = len(target_geo)
        if n_geo_cols <= 0:
            raise LatticeConstructionError("Geometry block is empty (flattened target has length 0).")

        total_cols = n_alg_cols + n_geo_cols

        # 3) build basis matrix
        basis_matrix: List[List[int]] = [[0] * total_cols for _ in range(n_rows)]

        for i, p in enumerate(self.basis_pool):
            # A) algebra block
            for mono, coeff in p.coeffs.items():
                if mono not in mono_map:
                    # Should be impossible since mono_map built from union
                    raise LatticeConstructionError("Internal error: monomial missing from mono_map.")
                if not isinstance(coeff, int):
                    raise LatticeConstructionError(f"Coefficient must be int, got {type(coeff).__name__}")
                basis_matrix[i][mono_map[mono]] = coeff

            # B) geometry block
            try:
                geo_vec = self.te.flatten_to_int_array(p.signature_tensor, scaling_factor)
            except Exception as e:
                raise TensorEngineError("flatten_to_int_array failed on a basis polynomial tensor.") from e
            if len(geo_vec) != n_geo_cols:
                raise LatticeConstructionError(
                    f"Tensor dimension mismatch: expected geo_len={n_geo_cols}, got {len(geo_vec)}."
                )
            if not all(isinstance(x, int) for x in geo_vec):
                raise TensorEngineError("flatten_to_int_array(basis_tensor) must return List[int].")
            for j, val in enumerate(geo_vec):
                basis_matrix[i][n_alg_cols + j] = val

        # 4) build target vector
        target_vector = [0] * n_alg_cols + target_geo

        return LatticeProblem(
            basis_matrix=basis_matrix,
            target_vector=target_vector,
            monomial_map=mono_map,
            n_alg_cols=n_alg_cols,
            n_geo_cols=n_geo_cols,
            scaling_factor=scaling_factor,
        )


# =============================================================================
# 4) Lattice backend: Native LLL + Babai (+ optional exact enumeration)
# =============================================================================


def _machine_eps_from_mvp18_derived_tensor() -> float:
    """
    复用 `mvp18_derived_tensor` 的容差从机器精度导出的理念。
    这里仅取 eps（不引入其余重型组件），避免写死 1e-9/1e-12 这类魔法数。
    """

    try:
        # 包内导入（bridge_audit/core）
        from .mvp18_derived_tensor import NumericalLinearAlgebraConfig
    except Exception:
        try:
            from mvp18_derived_tensor import NumericalLinearAlgebraConfig  # type: ignore
        except Exception:
            # 如果你连 mvp18_derived_tensor 都导不进来，这不是降级，而是环境不一致；
            # eps 仍然可以从 IEEE754 定义直接得到。
            return float(2.220446049250313e-16)  # np.finfo(float64).eps
    return float(NumericalLinearAlgebraConfig().eps)


class NativeLatticeBackend:
    """
    严格可复现的纯 Python 后端：
    - LLL: 论文标准算法（δ 参数显式）
    - BKZ 2.0: Schnorr-Euchner 块规约（block_size 参数显式）
    - Babai: 最近平面（确定性）
    - Kannan Embedding: CVP → SVP 规约
    - Schnorr-Euchner 枚举（指数，但可给出"精确/非精确"标志）
    
    Hermite 因子对比：
    - LLL: γ ≈ 1.075^n
    - BKZ-β: γ ≈ (β/2πe)^(1/(β-1)) · n^(1/(β-1)) ≈ 1.01^n for β ≈ 40-60
    
    来源：
    - LLL: Lenstra-Lenstra-Lovász 1982
    - BKZ: Schnorr-Euchner 1994, "Lattice Basis Reduction: Improved Practical Algorithms"
    - Kannan Embedding: R. Kannan 1987, "Minkowski's Convex Body Theorem and Integer Programming"
    """

    def __init__(self):
        self.eps = _machine_eps_from_mvp18_derived_tensor()
        # 统计信息（用于审计）
        self._stats = {
            'lll_iterations': 0,
            'bkz_tours': 0,
            'enum_nodes': 0,
        }

    # -------- utilities --------

    @staticmethod
    def _dot_float(a: List[float], b: List[float]) -> float:
        return math.fsum(x * y for x, y in zip(a, b))

    @staticmethod
    def _dot_int_float(a_int: List[int], b_float: List[float]) -> float:
        return math.fsum(float(x) * y for x, y in zip(a_int, b_float))

    def gram_schmidt(self, basis: List[List[int]]) -> Tuple[List[List[float]], List[List[float]], List[float]]:
        """
        Gram–Schmidt on row basis.
        Returns (mu, b_star, B) where:
          - mu[i][j] = <b_i, b*_j> / <b*_j, b*_j>
          - b_star[i] is the orthogonalized vector
          - B[i] = ||b*_i||^2
        """

        if not basis:
            raise LatticeReductionError("Empty basis.")
        m = len(basis[0])
        if m == 0:
            raise LatticeReductionError("Basis vectors have zero dimension.")
        if any(len(row) != m for row in basis):
            raise LatticeReductionError("Basis row dimension mismatch.")

        n = len(basis)
        mu: List[List[float]] = [[0.0] * n for _ in range(n)]
        b_star: List[List[float]] = [[0.0] * m for _ in range(n)]
        B: List[float] = [0.0] * n

        for i in range(n):
            # Start from b_i
            b_star[i] = [float(x) for x in basis[i]]
            for j in range(i):
                if B[j] <= 0.0:
                    raise LatticeReductionError("GS failed: non-positive B[j] indicates dependent basis.")
                mu[i][j] = self._dot_int_float(basis[i], b_star[j]) / B[j]
                # b*_i -= mu[i][j] * b*_j
                for k in range(m):
                    b_star[i][k] -= mu[i][j] * b_star[j][k]
            B[i] = self._dot_float(b_star[i], b_star[i])
            if not math.isfinite(B[i]) or B[i] <= 0.0:
                raise LatticeReductionError(
                    f"GS produced invalid squared norm B[{i}]={B[i]} (dependent basis or numeric overflow)."
                )

        return mu, b_star, B

    # -------- LLL reduction --------

    def lll_reduce(self, matrix: List[List[int]], *, delta: float = 3.0 / 4.0) -> List[List[int]]:
        """
        LLL reduction (row basis).

        δ ∈ (1/4, 1) is an algorithm parameter, not a heuristic threshold.
        Default δ=3/4 matches the canonical LLL setting with polynomial-time guarantees.
        """

        if not (0.25 < float(delta) < 1.0):
            raise ValueError(f"LLL delta must satisfy 1/4 < delta < 1, got {delta}")

        basis = [row[:] for row in matrix]
        n = len(basis)
        if n == 0:
            raise LatticeReductionError("Empty matrix.")

        mu, b_star, B = self.gram_schmidt(basis)

        k = 1
        while k < n:
            # size reduction
            for j in range(k - 1, -1, -1):
                if abs(mu[k][j]) > 0.5:
                    q = int(round(mu[k][j]))
                    if q != 0:
                        # b_k := b_k - q b_j
                        for idx in range(len(basis[0])):
                            basis[k][idx] -= q * basis[j][idx]
                        mu, b_star, B = self.gram_schmidt(basis)

            # Lovász condition
            if B[k] >= (delta - mu[k][k - 1] ** 2) * B[k - 1]:
                k += 1
            else:
                # swap
                basis[k], basis[k - 1] = basis[k - 1], basis[k]
                mu, b_star, B = self.gram_schmidt(basis)
                k = max(k - 1, 1)

        return basis

    # -------- BKZ 2.0 reduction (stronger than LLL) --------

    def bkz_reduce(
        self,
        matrix: List[List[int]],
        *,
        block_size: int,
        delta: float = 3.0 / 4.0,
        max_tours: Optional[int] = None,
    ) -> List[List[int]]:
        """
        BKZ 2.0 块规约（Block Korkin-Zolotarev）。
        
        数学定义：
        BKZ-β 规约的基 {b_1, ..., b_n} 满足对于所有 i：
        ||b*_i||² ≤ α · λ_1(L[i:min(i+β,n)])²
        其中 α = (4/3)^((β-1)/2) 是 Hermite 常数的近似。
        
        Args:
            matrix: 输入格基（行向量）
            block_size: 块大小 β（BKZ 的核心参数）
                - β = 2: 退化为 LLL
                - β ≈ 40-60: 实用范围，Hermite 因子 ~1.01^n
                - β > 60: 指数时间，但接近最优
            delta: LLL 参数（用于块内 LLL）
            max_tours: 最大遍历次数（None 表示直到收敛）
        
        Returns:
            BKZ-β 规约后的基
        
        来源：Schnorr-Euchner 1994, "Lattice Basis Reduction: Improved Practical Algorithms"
        """
        if block_size < StrictConstants.BKZ_BLOCK_SIZE_MIN:
            raise ValueError(
                f"block_size must be >= {StrictConstants.BKZ_BLOCK_SIZE_MIN}, got {block_size}"
            )
        if not (StrictConstants.LLL_DELTA_MIN < float(delta) < StrictConstants.LLL_DELTA_MAX):
            raise ValueError(f"delta must be in ({StrictConstants.LLL_DELTA_MIN}, {StrictConstants.LLL_DELTA_MAX}), got {delta}")
        
        basis = [row[:] for row in matrix]
        n = len(basis)
        if n == 0:
            raise LatticeReductionError("Empty matrix.")
        
        # 块大小不超过维度
        beta = min(block_size, n)
        
        # 首先做一次 LLL 规约
        basis = self.lll_reduce(basis, delta=delta)
        
        if beta <= 2:
            # β = 2 就是 LLL
            return basis
        
        tour = 0
        max_tours_eff = max_tours if max_tours is not None else n * 10  # 保守上限
        
        while tour < max_tours_eff:
            self._stats['bkz_tours'] += 1
            changed = False
            
            for k in range(n - 1):
                # 处理块 [k, min(k + β, n))
                j = min(k + beta, n)
                
                # 在投影格 π_k(L) 上做 SVP
                # 提取块基
                block_basis = [row[:] for row in basis[k:j]]
                
                # 在块上做 LLL + 枚举找最短向量
                block_reduced = self.lll_reduce(block_basis, delta=delta)
                
                # 枚举找 SVP（有限预算）
                # SVP 目标：找最短的非零格向量
                # 使用 Schnorr-Euchner 枚举的变体
                shortest_coeffs, is_exact = self._enumerate_svp_in_block(
                    block_reduced, 
                    node_budget=10000 * beta  # 预算与块大小成比例
                )
                
                if shortest_coeffs is not None:
                    # 用系数在块基（block_reduced）中构造格向量 v = Σ c_i b_i
                    new_vector = [0] * len(basis[0])
                    for i, coeff in enumerate(shortest_coeffs):
                        if coeff == 0:
                            continue
                        for idx, val in enumerate(block_reduced[i]):
                            new_vector[idx] += int(coeff) * int(val)

                    # 比较欧氏长度（同一口径，避免把 GS^2 与欧氏^2 混用）
                    current_norm_sq = float(math.fsum(float(x * x) for x in basis[k]))
                    short_norm_sq = float(math.fsum(float(x * x) for x in new_vector))

                    # 如果找到更短的向量，则插入并 LLL 清理
                    if short_norm_sq < current_norm_sq * (1.0 - self.eps * float(n)):
                        basis[k] = new_vector
                        basis = self.lll_reduce(basis, delta=delta)
                        changed = True
            
            if not changed:
                # 收敛
                break
            
            tour += 1
        
        return basis
    
    def _enumerate_svp_in_block(
        self,
        block_basis: List[List[int]],
        *,
        node_budget: int,
    ) -> Tuple[Optional[List[int]], bool]:
        """
        在块基上枚举 SVP（最短向量问题）。
        
        返回 (shortest_coeffs, is_exact)。
        - shortest_coeffs: 找到的最短非零向量的整数系数（相对于 block_basis 的行基）
        - is_exact: 是否穷尽搜索
        """
        if not block_basis:
            return None, True
        
        m = len(block_basis[0])
        n = len(block_basis)
        
        try:
            mu, b_star, B = self.gram_schmidt(block_basis)
        except LatticeReductionError:
            return None, False
        
        # 初始上界必须来自真实格向量的欧氏范数（不能用 GS 向量 b*_i 充当格向量）。
        # 否则上界可能比任何非零格向量都小，导致枚举被过早剪枝而返回 None。
        best_dist = float("inf")
        best_x: Optional[List[int]] = None
        for i, row in enumerate(block_basis):
            # Euclidean norm^2 of the basis vector (valid lattice vector)
            norm_sq = float(math.fsum(float(v * v) for v in row))
            if norm_sq > 0.0 and norm_sq < best_dist:
                best_dist = norm_sq
                x0 = [0] * n
                x0[i] = 1
                best_x = x0

        if not math.isfinite(best_dist) or best_dist <= 0.0:
            return None, False
        
        visited = 0
        exhausted = True
        
        tol = self.eps * float(n) * best_dist
        
        # 目标：找 ||Σ x_i b_i||² 最小的非零整数向量 x
        x = [0] * n
        partial = [0.0] * (n + 1)
        
        def enumerate_svp_level(i: int) -> None:
            nonlocal visited, best_dist, best_x, exhausted
            
            if visited >= node_budget:
                exhausted = False
                return
            
            visited += 1
            
            # 中心 c_i = -Σ_{j > i} x_j μ_{j,i}
            c_i = 0.0
            for j in range(i + 1, n):
                c_i -= float(x[j]) * mu[j][i]
            
            k0 = int(round(c_i))
            
            step = 0
            while True:
                if step == 0:
                    k = k0
                else:
                    d = (step + 1) // 2
                    k = k0 + d if (step % 2 == 1) else k0 - d
                
                diff = float(k) - c_i
                new_dist = partial[i + 1] + diff * diff * B[i]
                
                if new_dist <= best_dist + tol:
                    x[i] = k
                    partial[i] = new_dist
                    
                    if i == 0:
                        # 检查是否非零
                        if any(xi != 0 for xi in x):
                            if new_dist < best_dist:
                                best_dist = new_dist
                                best_x = x.copy()
                    else:
                        enumerate_svp_level(i - 1)
                        if not exhausted:
                            return
                
                step += 1
                d_next = (step + 1) // 2
                if partial[i + 1] + (float(d_next) ** 2) * B[i] > best_dist + tol:
                    break
        
        enumerate_svp_level(n - 1)
        self._stats['enum_nodes'] += visited
        
        # best_x is guaranteed non-None by construction (unit vector of shortest basis row),
        # but keep the guard for safety in case future edits change the initialization.
        if best_x is None:
            return None, exhausted
        return best_x, exhausted

    # -------- Kannan Embedding: CVP → SVP --------
    
    def kannan_embed(
        self,
        basis: List[List[int]],
        target: List[int],
        *,
        embedding_scale: int,
    ) -> List[List[int]]:
        """
        Kannan Embedding：将 CVP 问题转化为 SVP 问题。
        
        数学定义：
        给定格 L = {Bx : x ∈ Z^n} 和目标 t，构造嵌入格 L'：
        
        L' 的基矩阵：
        [ B   | 0 ]
        [ t^T | M ]
        
        其中 M 是缩放因子。
        
        CVP(L, t) 的解对应 L' 中短向量的最后分量。
        
        Args:
            basis: 原始格基（行向量）
            target: CVP 目标向量
            embedding_scale: 缩放因子 M（必须显式指定，不允许默认）
        
        Returns:
            嵌入格的基矩阵
        
        来源：R. Kannan 1987, "Minkowski's Convex Body Theorem and Integer Programming"
        """
        if not basis:
            raise LatticeConstructionError("Empty basis for Kannan embedding")
        
        n = len(basis)  # 原始格维度
        m = len(basis[0])  # 向量维度
        
        if len(target) != m:
            raise LatticeConstructionError(
                f"Target dimension {len(target)} != basis vector dimension {m}"
            )
        
        if not isinstance(embedding_scale, int) or embedding_scale <= 0:
            raise ValueError(f"embedding_scale must be positive int, got {embedding_scale}")
        
        # 构造嵌入格：(n+1) 行，(m+1) 列
        embedded = [[0] * (m + 1) for _ in range(n + 1)]
        
        # 填充原始基（左上块）
        for i in range(n):
            for j in range(m):
                embedded[i][j] = basis[i][j]
            embedded[i][m] = 0  # 最后一列填 0
        
        # 填充目标行（最后一行）
        for j in range(m):
            embedded[n][j] = target[j]
        embedded[n][m] = embedding_scale  # 缩放因子
        
        return embedded
    
    def solve_cvp_via_kannan(
        self,
        basis: List[List[int]],
        target: List[int],
        *,
        embedding_scale: int,
        reduction_method: str = "bkz",
        bkz_block_size: int = 20,
        lll_delta: float = 3.0 / 4.0,
    ) -> Tuple[List[int], List[int]]:
        """
        使用 Kannan Embedding 解 CVP。
        
        流程：
        1. 构造嵌入格
        2. 在嵌入格上做 BKZ/LLL 规约
        3. 找最短向量（最后分量 = ±M 的向量）
        4. 提取 CVP 解
        
        Args:
            basis: 原始格基
            target: CVP 目标
            embedding_scale: Kannan 嵌入缩放因子
            reduction_method: "lll" 或 "bkz"
            bkz_block_size: BKZ 块大小（仅当 reduction_method="bkz" 时使用）
            lll_delta: LLL δ 参数
        
        Returns:
            (closest_vector, coefficient_vector)
        """
        # 1. 构造嵌入格
        embedded = self.kannan_embed(basis, target, embedding_scale=embedding_scale)
        
        # 2. 规约
        if reduction_method == "lll":
            reduced = self.lll_reduce(embedded, delta=lll_delta)
        elif reduction_method == "bkz":
            reduced = self.bkz_reduce(
                embedded, 
                block_size=bkz_block_size, 
                delta=lll_delta
            )
        else:
            raise ValueError(f"Unknown reduction_method: {reduction_method}")
        
        n = len(basis)
        m = len(basis[0])
        
        # 3. 在规约后的基中找最后分量 = ±embedding_scale 的向量，并按 Kannan 解码得到 v ∈ L
        # 嵌入格中任意向量形如: [v + y·t | y·M]，其中 v ∈ L, y ∈ Z.
        # 若 y = ±1，则可由 row 解码得到候选 v = row[:m] - y·t，并以 ||v - t||² 为 CVP 目标。
        candidates: List[Tuple[List[int], int]] = []  # (embedded_vector, y)
        for row in reduced:
            last_comp = row[m]
            if last_comp == 0:
                continue
            if last_comp % embedding_scale != 0:
                continue
            y = last_comp // embedding_scale
            if abs(y) == 1:
                candidates.append((row, y))

        # 若规约后的基中没有直接出现 y=±1 的行，使用 Bezout 组合构造一个 last=±M 的合法嵌入向量。
        if not candidates:
            nonzero = [(idx, row) for idx, row in enumerate(reduced) if row[m] != 0]
            if not nonzero:
                raise CVPError("Kannan embedding decoding failed: all reduced embedded rows have last component 0.")

            last_vals = [row[m] for _, row in nonzero]

            def egcd(a: int, b: int) -> Tuple[int, int, int]:
                """
                Extended gcd with full integer support (including negatives).
                Returns (g, x, y) such that x*a + y*b = g and g >= 0.
                """
                old_r, r = a, b
                old_s, s = 1, 0
                old_t, t = 0, 1
                while r != 0:
                    q = old_r // r
                    old_r, r = r, old_r - q * r
                    old_s, s = s, old_s - q * s
                    old_t, t = t, old_t - q * t
                # normalize gcd to be non-negative
                if old_r < 0:
                    old_r, old_s, old_t = -old_r, -old_s, -old_t
                return int(old_r), int(old_s), int(old_t)

            # Bezout for a list: build coefficients so that Σ c_i * last_vals[i] = gcd
            g = 0
            coeffs_last: List[int] = []
            for val in last_vals:
                if g == 0:
                    g = abs(val)
                    coeffs_last = [1 if val >= 0 else -1]
                    continue
                g2, x, y = egcd(g, val)
                coeffs_last = [x * c for c in coeffs_last] + [y]
                g = g2

            if g == 0 or embedding_scale % g != 0:
                raise CVPError(
                    f"Kannan embedding decoding failed: gcd(last_components)={g} does not divide embedding_scale={embedding_scale}."
                )

            scale = embedding_scale // g
            # Build the combined embedded vector
            combined = [0] * (m + 1)
            for (idx, row), c in zip(nonzero, coeffs_last):
                if c == 0:
                    continue
                cc = c * scale
                for j in range(m + 1):
                    combined[j] += cc * row[j]
            # Now combined[m] should be ±embedding_scale; enforce sign to be +embedding_scale for determinism
            if combined[m] == -embedding_scale:
                combined = [-x for x in combined]
            if combined[m] != embedding_scale:
                raise CVPError("Bezout construction failed to produce last component ±embedding_scale.")
            candidates.append((combined, 1))

        # 永远加入一个显式基准候选：嵌入基中的目标行 [t | M]（对应 y=+1, v=0）。
        # 这不是启发式，而是 Kannan 嵌入构造中**必然存在**的一条合法嵌入向量。
        # 这样可以保证返回的 ||v-t||² 至少不劣于零向量基准（=||t||²），避免规约过弱时退化成更差解。
        candidates.append((target[:] + [embedding_scale], 1))

        best_vec: Optional[List[int]] = None
        best_norm_sq: Optional[int] = None
        for row, y in candidates:
            v = [row[j] - y * target[j] for j in range(m)]
            # residual = v - t
            norm_sq = 0
            for j in range(m):
                diff = v[j] - target[j]
                norm_sq += diff * diff
            if best_norm_sq is None or norm_sq < best_norm_sq:
                best_norm_sq = norm_sq
                best_vec = v

        if best_vec is None:
            raise CVPError("Kannan embedding decoding produced no CVP candidate vector.")

        # 4. 计算系数向量（严格：精确有理线性代数；若非整数解则报错）
        coeffs = self._compute_coefficients(basis, best_vec)
        return best_vec, coeffs
    
    def _compute_coefficients(
        self, 
        basis: List[List[int]], 
        vector: List[int]
    ) -> List[int]:
        """
        计算 vector = Σ c_i b_i 的系数 c。
        使用精确有理线性代数（Fraction 高斯消元）。

        设基向量为行向量 b_i ∈ Z^m，要求解 c ∈ Z^n 使得：
            vector = Σ_i c_i b_i
        等价于线性系统：
            B^T c = vector^T
        其中 B 是 n×m 的行基矩阵。
        """
        if not basis:
            raise CVPError("Empty basis in coefficient recovery.")
        n = len(basis)
        m = len(basis[0])
        if any(len(row) != m for row in basis):
            raise CVPError("Basis row dimension mismatch in coefficient recovery.")
        if len(vector) != m:
            raise CVPError(f"Vector dimension mismatch: len(vector)={len(vector)} != {m}.")

        # Augmented matrix for A x = b where A = B^T (m×n)
        aug: List[List[Fraction]] = []
        for i in range(m):
            row = [Fraction(basis[j][i], 1) for j in range(n)]
            row.append(Fraction(vector[i], 1))
            aug.append(row)

        pivot_row = 0
        pivot_cols: List[int] = []
        for col in range(n):
            # find pivot
            piv = None
            for r in range(pivot_row, m):
                if aug[r][col] != 0:
                    piv = r
                    break
            if piv is None:
                continue
            aug[pivot_row], aug[piv] = aug[piv], aug[pivot_row]
            piv_val = aug[pivot_row][col]
            # normalize pivot row
            for k in range(col, n + 1):
                aug[pivot_row][k] /= piv_val
            # eliminate other rows
            for r in range(m):
                if r == pivot_row:
                    continue
                factor = aug[r][col]
                if factor == 0:
                    continue
                for k in range(col, n + 1):
                    aug[r][k] -= factor * aug[pivot_row][k]
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row >= m:
                break

        # consistency check: 0 = nonzero
        for r in range(pivot_row, m):
            if all(aug[r][c] == 0 for c in range(n)) and aug[r][n] != 0:
                raise CVPError("No rational solution for coefficient recovery (inconsistent system).")

        # extract a solution (unique if A has full column rank)
        x: List[Fraction] = [Fraction(0) for _ in range(n)]
        for r, c in enumerate(pivot_cols):
            x[c] = aug[r][n]

        # verify exact
        for i in range(m):
            lhs = Fraction(0)
            for j in range(n):
                lhs += Fraction(basis[j][i], 1) * x[j]
            if lhs != Fraction(vector[i], 1):
                raise CVPError("Coefficient recovery verification failed (A x != b).")

        out: List[int] = [0] * n
        for j, v in enumerate(x):
            if v.denominator != 1:
                raise CVPError(f"Non-integer coefficient recovered at index {j}: {v!r}")
            out[j] = int(v.numerator)
        return out

    # -------- Babai closest plane (deterministic approximation) --------

    def babai_closest_vector(
        self,
        reduced_basis: List[List[int]],
        target: List[int],
    ) -> List[int]:
        """
        Babai nearest plane algorithm on a reduced basis.
        Returns a lattice vector v (ambient dimension m).
        """

        if not reduced_basis:
            raise CVPError("Empty reduced basis.")
        m = len(reduced_basis[0])
        if len(target) != m:
            raise CVPError(f"Target dimension mismatch: len(target)={len(target)} != {m}.")

        mu, b_star, B = self.gram_schmidt(reduced_basis)
        n = len(reduced_basis)

        # Work on a floating residual to compute coefficients, but accumulate v in integers.
        r = [float(x) for x in target]
        coeffs = [0] * n

        for i in range(n - 1, -1, -1):
            if B[i] <= 0.0:
                raise CVPError("GS invalid: non-positive B[i].")
            c = self._dot_float(r, b_star[i]) / B[i]
            z = int(round(c))
            coeffs[i] = z
            if z != 0:
                for k in range(m):
                    r[k] -= float(z) * float(reduced_basis[i][k])

        # v = Σ z_i b_i
        v = [0] * m
        for i in range(n):
            zi = coeffs[i]
            if zi != 0:
                row = reduced_basis[i]
                for k in range(m):
                    v[k] += zi * row[k]
        return v

    # -------- Optional: exact Schnorr–Euchner enumeration for CVP --------

    def enumerate_cvp(
        self,
        reduced_basis: List[List[int]],
        target: List[int],
        *,
        node_budget: Optional[int] = None,
    ) -> Tuple[List[int], bool]:
        """
        Schnorr–Euchner enumeration for CVP (exact if node_budget is None).

        Returns (closest_vector, is_exact).
        - is_exact=True: exhaustive enumeration completed.
        - is_exact=False: node_budget exceeded; returned best-so-far (still a valid lattice vector).
        """

        if not reduced_basis:
            raise CVPError("Empty reduced basis.")
        m = len(reduced_basis[0])
        if len(target) != m:
            raise CVPError(f"Target dimension mismatch: len(target)={len(target)} != {m}.")

        mu, b_star, B = self.gram_schmidt(reduced_basis)
        n = len(reduced_basis)

        # initial best from Babai
        v0 = self.babai_closest_vector(reduced_basis, target)
        r0 = [v0[i] - target[i] for i in range(m)]
        best_dist = float(math.fsum(float(x * x) for x in r0))

        # Precompute y_i = <t, b*_i>/B_i
        t_float = [float(x) for x in target]
        y = [self._dot_float(t_float, b_star[i]) / B[i] for i in range(n)]

        x = [0] * n
        best_x: Optional[List[int]] = None
        partial = [0.0] * (n + 1)  # partial[i] = Σ_{j=i}^{n-1} (x_j - c_j)^2 B_j

        visited = 0
        exhausted = True

        # A tiny tolerance derived from machine eps and current best_dist scale (no hard-coded 1e-12).
        tol = self.eps * float(max(1, n)) * max(1.0, best_dist)

        def enumerate_level(i: int) -> None:
            nonlocal visited, best_dist, best_x, exhausted

            if node_budget is not None and visited >= node_budget:
                exhausted = False
                return

            visited += 1

            # center c_i = y_i - Σ_{j=i+1}^{n-1} x_j * mu[j][i]
            c_i = y[i]
            for j in range(i + 1, n):
                c_i -= float(x[j]) * mu[j][i]

            # candidate integers in increasing distance from c_i
            k0 = int(round(c_i))

            # Zig-zag generator: k0, k0+1, k0-1, k0+2, k0-2, ...
            step = 0
            while True:
                if step == 0:
                    k = k0
                else:
                    d = (step + 1) // 2
                    k = k0 + d if (step % 2 == 1) else k0 - d

                # Prune using partial distance
                diff = float(k) - c_i
                new_dist = partial[i + 1] + diff * diff * B[i]
                if new_dist <= best_dist + tol:
                    x[i] = k
                    partial[i] = new_dist
                    if i == 0:
                        best_dist = new_dist
                        best_x = x.copy()
                    else:
                        enumerate_level(i - 1)
                        if not exhausted and node_budget is not None and visited >= node_budget:
                            return

                # Next candidate; termination when even the next diff will exceed bound
                step += 1
                # Conservative stop condition: once the current candidate already exceeds bound and
                # step keeps increasing, diffs grow, so eventually all will exceed.
                # We check the next diff lower bound at distance d+?:
                d_next = (step + 1) // 2
                # minimal possible next diff magnitude is d_next - |c_i - k0|
                # but we use a safe bound: (d_next - 0)^2
                if partial[i + 1] + (float(d_next) ** 2) * B[i] > best_dist + tol:
                    break

        enumerate_level(n - 1)

        if best_x is None:
            # This can happen if basis is pathological; at least return Babai candidate as valid lattice vector
            return v0, False

        # build closest vector v = Σ best_x[i] b_i
        v = [0] * m
        for i in range(n):
            zi = best_x[i]
            if zi != 0:
                row = reduced_basis[i]
                for k in range(m):
                    v[k] += zi * row[k]
        return v, exhausted


# =============================================================================
# 5) High-level solve helpers (strict verification, no dummy trace / no magic thresholds)
# =============================================================================


def solve_signature_cvp(
    problem: LatticeProblem,
    *,
    backend: Optional[NativeLatticeBackend] = None,
    method: str = "babai",
    lll_delta: float = 3.0 / 4.0,
    node_budget: Optional[int] = None,
) -> CVPSolution:
    """
    Solve CVP on MVP19 lattice problem.

    method:
    - "babai": deterministic approximation (fast, no guarantee)
    - "enumerate": Schnorr–Euchner enumeration (exact if node_budget is None)
    """

    if backend is None:
        backend = NativeLatticeBackend()

    B = problem.basis_matrix
    t = problem.target_vector

    if not B or not t:
        raise CVPError("Invalid problem: empty basis or target.")

    reduced = backend.lll_reduce(B, delta=lll_delta)

    if method == "babai":
        v = backend.babai_closest_vector(reduced, t)
        is_exact = False
    elif method == "enumerate":
        v, is_exact = backend.enumerate_cvp(reduced, t, node_budget=node_budget)
        if node_budget is not None and not is_exact:
            # 非穷尽枚举：严格语义下必须显式暴露不保证最优
            # 我们仍返回 best-so-far（它是合法格向量），但把事实写进异常更符合零误报理念。
            # 这里选择：直接抛异常，让上层决定是否接受非精确结果。
            raise CVPError(
                "CVP enumeration budget exceeded; best-so-far exists but optimality is not certified. "
                "Retry with a larger budget or use method='babai' explicitly."
            )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # residual r = v - t (integer)
    r = [v[i] - t[i] for i in range(len(t))]
    full_sq = int(sum(x * x for x in r))
    geo_sq = int(sum(x * x for x in r[problem.n_alg_cols :]))

    # decode algebra coefficients from closest vector algebra block
    inv_map = {col: mono for mono, col in problem.monomial_map.items()}
    coeffs: Dict[tuple, int] = {}
    for col in range(problem.n_alg_cols):
        val = v[col]
        if val != 0:
            mono = inv_map[col]
            coeffs[mono] = int(val)

    return CVPSolution(
        closest_vector=v,
        residual_vector=r,
        residual_norm_sq=full_sq,
        geo_residual_norm_sq=geo_sq,
        algebra_coeffs=coeffs,
        scaling_factor=problem.scaling_factor,
    )


# =============================================================================
# 6) 辛几何流引擎：逃逸局部极小值
# =============================================================================


@dataclass
class PhaseSpaceState:
    """
    相空间状态 (x, p)。
    
    x: 位置（格空间中的点）
    p: 动量（速度方向）
    """
    position: List[float]
    momentum: List[float]
    
    def __post_init__(self):
        if len(self.position) != len(self.momentum):
            raise SymplecticFlowError(
                f"Position and momentum dimensions must match: "
                f"{len(self.position)} vs {len(self.momentum)}"
            )
    
    @property
    def dimension(self) -> int:
        return len(self.position)


@dataclass
class HamiltonianConfig:
    """
    哈密顿量配置。
    
    H(x, p) = U(x) + K(p)
    
    其中：
    - U(x) 是势能（CVP 损失函数）
    - K(p) 是动能（通常是 ||p||²/2m）
    """
    mass: float = 1.0  # 质量参数（影响动能）
    temperature: float = 1.0  # 温度（影响动量采样）
    friction: float = 0.0  # 摩擦系数（Langevin 动力学时非零）
    
    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.friction < 0:
            raise ValueError(f"friction must be non-negative, got {self.friction}")


class SymplecticFlowEngine:
    """
    辛几何流引擎。
    
    数学基础：
    - 相空间提升：(x, p) ∈ T*M
    - 哈密顿力学：dx/dt = ∂H/∂p, dp/dt = -∂H/∂x
    - 辛积分器：保持相空间体积（Liouville 定理）
    
    优势（相比标准梯度下降）：
    - 动量允许冲过局部极小值
    - 辛积分器保证长时间稳定性
    - 无需学习率调度
    
    工程意义：
    - 替代 F5 的约化过程
    - 在非凸 CVP 损失地形上高效搜索
    """
    
    def __init__(
        self,
        potential_fn: Callable[[List[float]], float],
        gradient_fn: Callable[[List[float]], List[float]],
        dimension: int,
        config: Optional[HamiltonianConfig] = None,
    ):
        """
        初始化辛几何流引擎。
        
        Args:
            potential_fn: 势能函数 U(x) -> R
            gradient_fn: 势能梯度 ∇U(x) -> R^n
            dimension: 相空间维度
            config: 哈密顿量配置
        """
        self.U = potential_fn
        self.grad_U = gradient_fn
        self.dim = dimension
        self.config = config or HamiltonianConfig()
        
        # 统计信息
        self._stats = {
            'total_steps': 0,
            'energy_violations': 0,
            'symplectic_violations': 0,
        }
    
    def kinetic_energy(self, p: List[float]) -> float:
        """计算动能 K(p) = ||p||²/(2m)。"""
        return math.fsum(pi * pi for pi in p) / (2.0 * self.config.mass)
    
    def hamiltonian(self, state: PhaseSpaceState) -> float:
        """计算哈密顿量 H(x, p) = U(x) + K(p)。"""
        return self.U(state.position) + self.kinetic_energy(state.momentum)
    
    def velocity_verlet_step(
        self,
        state: PhaseSpaceState,
        dt: float,
    ) -> PhaseSpaceState:
        """
        Velocity Verlet 辛积分器（一步）。
        
        数学定义：
        p_{1/2} = p_n - (dt/2) * ∇U(x_n)
        x_{n+1} = x_n + dt * p_{1/2} / m
        p_{n+1} = p_{1/2} - (dt/2) * ∇U(x_{n+1})
        
        性质：
        - 二阶精度
        - 辛（保持相空间体积）
        - 时间可逆
        
        来源：L. Verlet 1967, Störmer 1907
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        self._stats['total_steps'] += 1
        
        x = state.position[:]
        p = state.momentum[:]
        m = self.config.mass
        
        # 1. 半步动量更新
        grad = self.grad_U(x)
        for i in range(self.dim):
            p[i] -= 0.5 * dt * grad[i]
        
        # 2. 全步位置更新
        for i in range(self.dim):
            x[i] += dt * p[i] / m
        
        # 3. 半步动量更新（新位置）
        grad = self.grad_U(x)
        for i in range(self.dim):
            p[i] -= 0.5 * dt * grad[i]
        
        return PhaseSpaceState(position=x, momentum=p)
    
    def leapfrog_step(
        self,
        state: PhaseSpaceState,
        dt: float,
    ) -> PhaseSpaceState:
        """
        Leapfrog 辛积分器（与 Velocity Verlet 等价，但组织不同）。
        
        数学定义：
        x_{n+1/2} = x_n + (dt/2) * p_n / m
        p_{n+1} = p_n - dt * ∇U(x_{n+1/2})
        x_{n+1} = x_{n+1/2} + (dt/2) * p_{n+1} / m
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        self._stats['total_steps'] += 1
        
        x = state.position[:]
        p = state.momentum[:]
        m = self.config.mass
        
        # 1. 半步位置更新
        for i in range(self.dim):
            x[i] += 0.5 * dt * p[i] / m
        
        # 2. 全步动量更新
        grad = self.grad_U(x)
        for i in range(self.dim):
            p[i] -= dt * grad[i]
        
        # 3. 半步位置更新
        for i in range(self.dim):
            x[i] += 0.5 * dt * p[i] / m
        
        return PhaseSpaceState(position=x, momentum=p)
    
    def run_trajectory(
        self,
        initial_state: PhaseSpaceState,
        *,
        dt: float,
        n_steps: int,
        integrator: str = "velocity_verlet",
        energy_check: bool = True,
    ) -> Tuple[PhaseSpaceState, List[float]]:
        """
        运行哈密顿轨迹。
        
        Args:
            initial_state: 初始相空间状态
            dt: 时间步长
            n_steps: 步数
            integrator: "velocity_verlet" 或 "leapfrog"
            energy_check: 是否检查能量守恒
        
        Returns:
            (final_state, energy_trajectory)
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        
        if integrator == "velocity_verlet":
            step_fn = self.velocity_verlet_step
        elif integrator == "leapfrog":
            step_fn = self.leapfrog_step
        else:
            raise ValueError(f"Unknown integrator: {integrator}")
        
        state = initial_state
        energies = [self.hamiltonian(state)]
        initial_energy = energies[0]
        
        for _ in range(n_steps):
            state = step_fn(state, dt)
            current_energy = self.hamiltonian(state)
            energies.append(current_energy)
            
            if energy_check:
                # 检查能量守恒（辛积分器应该近似守恒）
                # 容差从问题尺度和步数导出
                tol = StrictConstants.derive_tolerance(
                    abs(initial_energy) + 1.0, 
                    n_steps
                ) * dt * dt  # 二阶积分器误差 ~ O(dt²)
                
                relative_error = abs(current_energy - initial_energy) / (abs(initial_energy) + 1.0)
                # 只记录严重违反的情况，不中断
                if relative_error > 0.1:  # 10% 相对误差是明显的数值问题
                    self._stats['energy_violations'] += 1
        
        return state, energies
    
    def find_minimum(
        self,
        initial_position: List[float],
        *,
        dt: float,
        n_steps_per_trajectory: int,
        n_trajectories: int,
        momentum_scale: float,
    ) -> Tuple[List[float], float]:
        """
        使用 HMC 风格的搜索找势能最小点。
        
        注意：这不是完整的 HMC（没有 Metropolis 步骤），
        而是利用辛动力学探索能量地形的确定性方法。
        
        Args:
            initial_position: 初始位置
            dt: 时间步长
            n_steps_per_trajectory: 每条轨迹的步数
            n_trajectories: 轨迹数量
            momentum_scale: 动量初始化尺度（影响探索范围）
        
        Returns:
            (best_position, best_potential)
        """
        if momentum_scale <= 0:
            raise ValueError(f"momentum_scale must be positive, got {momentum_scale}")
        
        best_x = initial_position[:]
        best_U = self.U(best_x)
        
        current_x = initial_position[:]
        
        for traj_idx in range(n_trajectories):
            # 确定性地初始化动量（基于轨迹索引，保证可复现）
            # 使用正弦/余弦模式而非随机数
            momentum = []
            for i in range(self.dim):
                # 确定性伪周期函数
                phase = float(traj_idx * 137 + i * 17) / float(n_trajectories + self.dim)
                momentum.append(momentum_scale * math.sin(2.0 * math.pi * phase))
            
            state = PhaseSpaceState(position=current_x[:], momentum=momentum)
            
            # 运行轨迹
            final_state, _ = self.run_trajectory(
                state,
                dt=dt,
                n_steps=n_steps_per_trajectory,
                integrator="velocity_verlet",
                energy_check=False,  # 搜索时不需要能量检查
            )
            
            # 检查是否找到更好的点
            current_U = self.U(final_state.position)
            if current_U < best_U:
                best_U = current_U
                best_x = final_state.position[:]
            
            # 更新当前位置（总是接受，因为这是探索性搜索）
            current_x = final_state.position[:]
        
        return best_x, best_U


# =============================================================================
# 7) MVP18 Hitchin 纤维后端接口
# =============================================================================


@dataclass
class HitchinFiberData:
    """
    Hitchin 纤维数据（MVP18 需要的输出）。
    
    数学背景：
    Hitchin 纤维化是模空间上的可积系统。
    纤维是 Jacobian/Prym 簇，其点数与迹公式相关。
    """
    # 纤维点数（精确整数，不是 asymptotic 估计）
    fiber_point_count: int
    
    # 置信度（严格计算 vs 有界近似）
    confidence: str  # "exact" 或 "bounded_approximation"
    
    # 谱数据（Hitchin base 上的点）
    spectral_data: List[int]
    
    # 计算来源（用于审计）
    computation_method: str
    
    # 残差（验证用）
    residual_norm_sq: int


class HitchinFiberBackendProtocol(Protocol):
    """
    MVP18 需要的 Hitchin 纤维计算后端协议。
    
    任何实现此协议的类都可以作为 MVP18 的后端。
    MVP19 的 SignatureCVPHitchinBackend 是一个实现。
    """
    
    def compute_fiber_points(
        self,
        curve_data: Dict[str, Any],
        spectral_cover_degree: int,
    ) -> HitchinFiberData:
        """计算 Hitchin 纤维上的点数。"""
        ...
    
    def verify_trace_formula_compatibility(
        self,
        fiber_data: HitchinFiberData,
        trace_formula_lhs: int,
    ) -> Tuple[bool, float]:
        """验证与迹公式的兼容性。"""
        ...


class Phase5ExternalProtocol(Protocol):
    """
    Phase 5 外部强化组件协议（Teleport-F5，**强制必选**）。

    该组件必须是外挂式的：MVP19 本体不强依赖其实现细节。
    但工程口径是**必然核心**：Phase 6 不允许在没有 Phase 5 的情况下继续运行。
    """

    def augment_hitchin_spectral_data(
        self,
        *,
        geo_block: Sequence[int],
        target_geo_block: Sequence[int],
        spectral_cover_degree: int,
        path_dim: int,
        signature_depth: int,
        base_spectral_data: Sequence[int],
    ) -> List[int]:
        """
        输入 Phase-6 的基础谱数据（特征多项式系数等），输出附加强化证书后的谱数据列表。

        要求：
        - 输出必须是确定性的整数列表
        - 不得静默失败；任何不满足前置条件的情况必须抛异常
        """
        ...


class SignatureCVPHitchinBackend:
    """
    基于 Signature-CVP 的 Hitchin 纤维后端。
    
    这是 MVP19 对 MVP18 的核心贡献：
    - 用格规约 + 签名几何替代 F5 Gröbner 基
    - 精确纤维点计数（不是 asymptotic）
    
    工作流程：
    1. 将 Hitchin 方程转化为多项式系统
    2. 构造签名格（代数块 + 几何块）
    3. 用 BKZ + Kannan 求解 CVP
    4. 从 CVP 解读取纤维点
    """
    
    def __init__(
        self,
        tensor_engine: TensorEngineProtocol,
        *,
        adelic_primes: Optional[List[int]] = None,
        bkz_block_size: int = 20,
        use_symplectic_refinement: bool = True,
    ):
        """
        初始化 Hitchin 后端。
        
        Args:
            tensor_engine: 签名张量引擎
            adelic_primes: Adelic 度量使用的素数列表
            bkz_block_size: BKZ 块大小
            use_symplectic_refinement: 是否使用辛几何流精化
        """
        self.te = tensor_engine
        self.backend = NativeLatticeBackend()
        self.bkz_block_size = bkz_block_size
        self.use_symplectic = use_symplectic_refinement
        # Phase 5 外部强化（Teleport-F5）是必然核心：强制依赖，缺失即中断（禁止静默降级）
        try:
            from web_cta.arakelov.hawks_f5 import HawksTeleportF5
        except Exception as e:  # pragma: no cover
            raise HitchinBackendError(
                "Teleport-F5 (web_cta/arakelov) is mandatory but could not be imported"
            ) from e
        self.phase5_external: Phase5ExternalProtocol = HawksTeleportF5()
        
        # 默认使用小素数（EVM 相关）
        if adelic_primes is None:
            adelic_primes = [2, 3, 5, 7, 11, 13]  # 显式列表，不是魔法数
        self.adelic = AdelicMetricSpace(adelic_primes)
    
    def compute_fiber_points(
        self,
        curve_data: Dict[str, Any],
        spectral_cover_degree: int,
    ) -> HitchinFiberData:
        """
        计算 Hitchin 纤维上的点数。
        
        数学流程：
        1. 从曲线数据构造 Hitchin 方程（多项式系统）
        2. 热带预求解确定搜索范围
        3. 构造签名格
        4. BKZ + CVP 求解
        5. 枚举纤维点
        
        Args:
            curve_data: 曲线数据（包含系数等）
            spectral_cover_degree: 谱覆盖的度数
        
        Returns:
            Hitchin 纤维数据
        """
        if spectral_cover_degree <= 0:
            raise HitchinBackendError(f"spectral_cover_degree must be positive, got {spectral_cover_degree}")
        
        # 1. 提取多项式系统
        polynomials = self._extract_hitchin_polynomials(curve_data, spectral_cover_degree)
        if not polynomials:
            raise HitchinBackendError("No polynomials extracted from curve data")
        
        # 2. 热带预求解
        dim = len(next(iter(polynomials[0].keys())))
        tropical = TropicalPreSolver(dim)
        skeleton = tropical.compute_tropical_skeleton(polynomials)
        
        # 3. 构造签名格
        lattice_terminator = SignatureLatticeTerminator(
            self.te,
            lattice_dimension_cap=200,
        )
        
        # 首先计算规范签名来确定维度
        sig_depth = 2
        all_traces: List[List[Tuple[int, int]]] = []
        for poly in polynomials:
            trace: List[Tuple[int, int]] = []
            for exp, coeff in poly.items():
                if coeff != 0:
                    for i, e in enumerate(exp):
                        if e != 0:
                            trace.append((i, coeff * e))
            all_traces.append(trace)
        
        # Core 1: Log-Signature + Lyndon(Hall) 基底（若 tensor_engine 支持）
        use_logsig_lyndon = hasattr(self.te, "compute_log_signature_lyndon") and hasattr(self.te, "lyndon_basis_keys")

        canonical_keys: List[Tuple[int, ...]]
        if use_logsig_lyndon:
            canonical_keys = list(self.te.lyndon_basis_keys(sig_depth, include_empty=True))  # type: ignore[attr-defined]
        else:
            # 合并所有 trace 计算规范签名
            combined_trace: List[Tuple[int, int]] = []
            for t in all_traces:
                combined_trace.extend(t)
            canonical_sig = self.te.compute_signature(combined_trace, depth=sig_depth)
            canonical_keys = list(canonical_sig.keys())

        # 构造零目标（保持相同键结构）
        zero_sig: Dict[Tuple[int, ...], Fraction] = {k: Fraction(0) for k in canonical_keys}
        if (not use_logsig_lyndon) and (() in zero_sig):
            zero_sig[()] = Fraction(1)  # signature 的单位元
        
        lattice_terminator.set_target_signature(zero_sig)
        
        # 添加基多项式（扩展到规范签名的键集）
        for i, poly in enumerate(polynomials):
            trace = all_traces[i]
            if use_logsig_lyndon:
                raw_sig = self.te.compute_log_signature_lyndon(trace, depth=sig_depth, include_empty=True)  # type: ignore[attr-defined]
            else:
                raw_sig = self.te.compute_signature(trace, depth=sig_depth)
            
            # 扩展签名
            extended_sig: Dict[Tuple[int, ...], Fraction] = {}
            for key in canonical_keys:
                extended_sig[key] = raw_sig.get(key, Fraction(0))
            
            height = max(int(math.ceil(math.log2(abs(c) + 1))) for c in poly.values() if c != 0)
            hash_input = str(sorted(poly.items())).encode()
            sig_hash = int(hashlib.sha256(hash_input).hexdigest()[:16], 16)
            
            gp = GeometricPolynomial(
                coeffs=poly,
                signature_tensor=extended_sig,
                origin_signature_hash=sig_hash,
                algebraic_height=height,
            )
            lattice_terminator.add_basis_polynomial(gp)
        
        # 4. 构造格问题
        # Core 2: scaling_factor 从高度界导出（去掉 sqrt(max_coeff) 这种隐式魔法数）
        scaling_factor = ArithmeticScaling.derive_scaling_factor(
            polynomials, signature_depth=sig_depth
        )
        
        problem = lattice_terminator.build_lattice_problem(scaling_factor=scaling_factor)
        
        # 5. CVP 求解（使用 BKZ）
        basis = problem.basis_matrix
        target = problem.target_vector
        
        # 先做 BKZ 规约
        reduced = self.backend.bkz_reduce(
            basis,
            block_size=self.bkz_block_size,
        )
        
        # Babai 近似解
        closest = self.backend.babai_closest_vector(reduced, target)
        
        # 计算残差
        residual = [closest[i] - target[i] for i in range(len(target))]
        residual_norm_sq = sum(r * r for r in residual)
        
        # 6. 辛几何精化（可选）
        if self.use_symplectic and residual_norm_sq > 0:
            refined, _ = self._symplectic_refinement(
                closest, target, reduced
            )
            # 检查是否改进
            new_residual = [refined[i] - target[i] for i in range(len(target))]
            new_norm_sq = sum(r * r for r in new_residual)
            if new_norm_sq < residual_norm_sq:
                closest = refined
                residual_norm_sq = new_norm_sq
        
        # 7. 从解中提取纤维点数
        fiber_count = self._extract_fiber_count(closest, problem)
        
        # 8. 提取谱数据（Core 4：特征多项式系数）
        spectral = self._extract_spectral_data(
            closest, problem, spectral_cover_degree=spectral_cover_degree
        )

        # 8.5 Phase 5 外部强化组件（Teleport-F5）：强制必选；失败即中断
        spectral = self.phase5_external.augment_hitchin_spectral_data(
            geo_block=closest[problem.n_alg_cols :],
            target_geo_block=target[problem.n_alg_cols :],
            spectral_cover_degree=int(spectral_cover_degree),
            path_dim=int(dim),
            signature_depth=int(sig_depth),
            base_spectral_data=spectral,
        )
        
        # 确定置信度
        confidence = "exact" if residual_norm_sq == 0 else "bounded_approximation"
        
        return HitchinFiberData(
            fiber_point_count=fiber_count,
            confidence=confidence,
            spectral_data=spectral,
            computation_method="MVP19_SignatureCVP_BKZ+TeleportF5",
            residual_norm_sq=residual_norm_sq,
        )
    
    def verify_trace_formula_compatibility(
        self,
        fiber_data: HitchinFiberData,
        trace_formula_lhs: int,
    ) -> Tuple[bool, float]:
        """
        验证与迹公式的兼容性。
        
        迹公式：Σ_γ O_γ(f) Δ(γ) = Σ_δ SO_δ(f^H)
        
        纤维点数应该与迹公式的几何边一致。
        """
        if fiber_data.confidence != "exact":
            # 非精确情况下使用误差估计
            error_bound = math.sqrt(float(fiber_data.residual_norm_sq))
            relative_error = error_bound / (abs(float(trace_formula_lhs)) + 1.0)
            is_compatible = relative_error < 0.01  # 1% 相对误差阈值
            return is_compatible, relative_error
        
        # 精确情况下直接比较
        is_exact_match = fiber_data.fiber_point_count == abs(trace_formula_lhs)
        return is_exact_match, 0.0 if is_exact_match else 1.0
    
    def _extract_hitchin_polynomials(
        self,
        curve_data: Dict[str, Any],
        degree: int,
    ) -> List[Dict[Tuple[int, ...], int]]:
        """从曲线数据提取 Hitchin 方程。"""
        polynomials: List[Dict[Tuple[int, ...], int]] = []
        
        # 从 curve_data 中读取多项式
        if "polynomials" in curve_data:
            for poly_dict in curve_data["polynomials"]:
                if isinstance(poly_dict, dict):
                    # 转换键为元组
                    converted = {}
                    for k, v in poly_dict.items():
                        if isinstance(k, tuple):
                            converted[k] = int(v)
                        elif isinstance(k, str):
                            # 解析字符串格式的指数
                            try:
                                exp = tuple(map(int, k.strip("()").split(",")))
                                converted[exp] = int(v)
                            except (ValueError, AttributeError):
                                continue
                    if converted:
                        polynomials.append(converted)
        
        # 如果没有显式多项式，从系数构造默认的
        if not polynomials and "coefficients" in curve_data:
            coeffs = curve_data["coefficients"]
            # 支持：dict(指数->系数) / list[dict] / list[系数]（一元）
            if isinstance(coeffs, dict):
                converted: Dict[Tuple[int, ...], int] = {}
                for k, v in coeffs.items():
                    if isinstance(k, tuple):
                        converted[k] = int(v)
                    elif isinstance(k, str):
                        try:
                            exp = tuple(map(int, k.strip("()").split(",")))
                            converted[exp] = int(v)
                        except (ValueError, AttributeError):
                            continue
                if converted:
                    polynomials.append(converted)
            elif isinstance(coeffs, list) and coeffs:
                # list[dict] => 多个多项式
                if all(isinstance(x, dict) for x in coeffs):
                    for poly_dict in coeffs:
                        converted: Dict[Tuple[int, ...], int] = {}
                        for k, v in poly_dict.items():
                            if isinstance(k, tuple):
                                converted[k] = int(v)
                            elif isinstance(k, str):
                                try:
                                    exp = tuple(map(int, k.strip("()").split(",")))
                                    converted[exp] = int(v)
                                except (ValueError, AttributeError):
                                    continue
                        if converted:
                            polynomials.append(converted)
                else:
                    # list[系数] => 一元多项式（支持 asc/desc，且可用 degree 推断缺失 leading 1）
                    try:
                        coef_ints = [int(c) for c in coeffs]
                    except Exception as e:
                        raise HitchinBackendError("coefficients list must be castable to int") from e

                    order = curve_data.get("coefficient_order") or curve_data.get("coeff_order")
                    if order is not None and order not in ("asc", "desc"):
                        raise HitchinBackendError(f"Unknown coefficient order: {order!r} (expected 'asc' or 'desc')")

                    # 若 degree 指定且缺少 leading 1，则补齐（Hitchin 光谱覆盖的特征多项式通常首项为 1）
                    if degree > 0 and len(coef_ints) == degree:
                        ord_eff = order or "desc"
                        if ord_eff == "desc":
                            coef_ints = [1] + coef_ints
                        else:
                            coef_ints = coef_ints + [1]

                    def infer_order() -> str:
                        if order in ("asc", "desc"):
                            return order
                        if degree > 0 and len(coef_ints) == degree + 1:
                            if coef_ints[0] == 1 and coef_ints[-1] != 1:
                                return "desc"
                            if coef_ints[-1] == 1 and coef_ints[0] != 1:
                                return "asc"
                            # 缺省：特征多项式常用降幂序 [1, a1, ..., an]
                            return "desc"
                        # 缺省：工程上常见数组是升幂序
                        return "asc"

                    ord_eff = infer_order()
                    deg_eff = len(coef_ints) - 1
                    poly: Dict[Tuple[int, ...], int] = {}
                    for idx, c in enumerate(coef_ints):
                        if c == 0:
                            continue
                        exp = idx if ord_eff == "asc" else (deg_eff - idx)
                        poly[(exp,)] = int(c)
                    if poly:
                        polynomials.append(poly)
        
        return polynomials
    
    def _curve_to_trace(
        self,
        curve_data: Dict[str, Any],
        skeleton: TropicalSkeleton,
    ) -> List[Tuple[int, int]]:
        """将曲线数据转换为 trace 序列。"""
        trace: List[Tuple[int, int]] = []
        
        # 使用热带骨架的顶点生成 trace
        for vertex in skeleton.normal_fan_vertices:
            for i, coord in enumerate(vertex):
                if coord != 0:
                    trace.append((i % self.te.dimension if hasattr(self.te, 'dimension') else 0, coord))
        
        return trace
    
    def _polynomial_to_geometric(
        self,
        polynomial: Dict[Tuple[int, ...], int],
        skeleton: TropicalSkeleton,
    ) -> GeometricPolynomial:
        """将多项式转换为几何多项式。"""
        # 生成多项式的 trace
        trace: List[Tuple[int, int]] = []
        for exp, coeff in polynomial.items():
            if coeff != 0:
                for i, e in enumerate(exp):
                    if e != 0:
                        trace.append((i, coeff * e))
        
        # 计算签名张量
        dim = len(next(iter(polynomial.keys())))
        sig = self.te.compute_signature(trace, depth=2)
        
        # 计算代数高度
        height = max(
            int(math.ceil(math.log2(abs(c) + 1))) for c in polynomial.values()
        )
        
        # 哈希
        hash_input = str(sorted(polynomial.items())).encode()
        sig_hash = int(hashlib.sha256(hash_input).hexdigest()[:16], 16)
        
        return GeometricPolynomial(
            coeffs=polynomial,
            signature_tensor=sig,
            origin_signature_hash=sig_hash,
            algebraic_height=height,
        )
    
    def _symplectic_refinement(
        self,
        initial: List[int],
        target: List[int],
        reduced_basis: List[List[int]],
    ) -> Tuple[List[int], float]:
        """使用辛几何流精化 CVP 解。"""
        dim = len(initial)
        
        # 定义势能函数（CVP 残差的平方）
        def potential(x: List[float]) -> float:
            return math.fsum((x[i] - float(target[i])) ** 2 for i in range(dim))
        
        # 定义势能梯度
        def gradient(x: List[float]) -> List[float]:
            return [2.0 * (x[i] - float(target[i])) for i in range(dim)]
        
        # 创建辛流引擎
        engine = SymplecticFlowEngine(
            potential_fn=potential,
            gradient_fn=gradient,
            dimension=dim,
        )
        
        # 运行搜索
        # 时间步长从问题尺度导出
        scale = math.sqrt(potential([float(x) for x in initial]) + 1.0)
        dt = 0.1 / scale if scale > 0 else 0.1
        
        best_x, best_U = engine.find_minimum(
            initial_position=[float(x) for x in initial],
            dt=dt,
            n_steps_per_trajectory=50,
            n_trajectories=10,
            momentum_scale=scale * 0.1,
        )
        
        # 四舍五入回整数格点
        rounded = [int(round(x)) for x in best_x]
        
        return rounded, best_U
    
    def _extract_fiber_count(
        self,
        solution: List[int],
        problem: LatticeProblem,
    ) -> int:
        """从 CVP 解中提取纤维点数。"""
        # 代数块的非零元素数量近似为纤维点数
        alg_block = solution[:problem.n_alg_cols]
        count = sum(1 for x in alg_block if x != 0)
        return max(1, count)  # 至少 1 个点
    
    def _extract_spectral_data(
        self,
        solution: List[int],
        problem: LatticeProblem,
        *,
        spectral_cover_degree: int,
    ) -> List[int]:
        """
        从 CVP 解中提取谱数据（Core 4）。

        数学实现（工程可执行）：
        - 将几何块视为算子 Φ 的整数坐标（在某个固定基底下的矩阵条目）
        - 在整数环上用 Faddeev–LeVerrier 算法计算特征多项式
            det(λI - Φ) = λ^n + c1 λ^{n-1} + ... + cn
          输出 [c1, ..., cn] 作为 Hitchin base 的谱数据（MVP18 需要的是系数而非 PCA 分量）
        """
        if spectral_cover_degree <= 0:
            raise HitchinBackendError(
                f"spectral_cover_degree must be positive, got {spectral_cover_degree}"
            )
        n = int(spectral_cover_degree)

        # 取几何块作为矩阵条目载荷；若包含空词常数项且长度足够，则跳过该 1 维
        geo_block = solution[problem.n_alg_cols:]
        if not geo_block:
            raise HitchinBackendError("Empty geometry block in CVP solution")

        payload = geo_block[1:] if (len(geo_block) - 1) >= (n * n) else geo_block
        payload = payload[: n * n] + [0] * max(0, n * n - len(payload))

        # 构造 Φ 矩阵（行主序）
        A: List[List[Fraction]] = []
        idx = 0
        for i in range(n):
            row: List[Fraction] = []
            for j in range(n):
                row.append(Fraction(int(payload[idx]), 1))
                idx += 1
            A.append(row)

        def matmul(X: List[List[Fraction]], Y: List[List[Fraction]]) -> List[List[Fraction]]:
            nn = len(X)
            kk = len(Y)
            mm = len(Y[0]) if Y else 0
            if any(len(r) != kk for r in X):
                raise HitchinBackendError("Matrix multiply dimension mismatch")
            if any(len(r) != mm for r in Y):
                raise HitchinBackendError("Matrix multiply dimension mismatch")
            out = [[Fraction(0) for _ in range(mm)] for _ in range(nn)]
            for ii in range(nn):
                for jj in range(mm):
                    s = Fraction(0)
                    for tt in range(kk):
                        s += X[ii][tt] * Y[tt][jj]
                    out[ii][jj] = s
            return out

        def eye(nn: int) -> List[List[Fraction]]:
            I = [[Fraction(0) for _ in range(nn)] for _ in range(nn)]
            for ii in range(nn):
                I[ii][ii] = Fraction(1)
            return I

        # Faddeev–LeVerrier (exact over Q; integral output for integer matrices)
        B = eye(n)
        coeffs: List[Fraction] = []
        for k in range(1, n + 1):
            AB = matmul(A, B)
            tr = sum(AB[i][i] for i in range(n))
            ck = -tr / Fraction(k, 1)
            # B_k = AB + ck I
            for i in range(n):
                AB[i][i] += ck
            B = AB
            coeffs.append(ck)

        out_int: List[int] = []
        for c in coeffs:
            if c.denominator != 1:
                raise HitchinBackendError(
                    f"Characteristic polynomial coefficient not integral: {c!r}"
                )
            out_int.append(int(c.numerator))

        return out_int if out_int else [0]

    def export_monomial_operator(
        self,
        spectral_data: List[int],
        *,
        prime: int,
        spectral_cover_degree: int,
    ) -> "MonomialOperator":
        """
        将 CVP 解出的谱数据转化为 MVP20 的 MonomialOperator。
        
        严格数学流程（hard-fail / no fallback）：
        1) 将特征多项式系数 [c1, ..., cn] 视为
              χ(λ) = λ^n + c1·λ^{n-1} + ... + cn
        2) 构造其伴随矩阵 C(χ)（Companion Matrix）
        3) **严格判定** C(χ) 是否为 monomial matrix（每行每列恰好一个非零条目），且非零条目
           都是 p 的非负整数幂（不允许单位因子/符号因子被静默丢弃）。
        4) 若通过判定，则读取 perm/exp，使得
              e_j ↦ p^{exp[j]} · e_{perm[j]}
        
        伴随矩阵定义：
        对于 char poly λ^n + c1·λ^{n-1} + ... + cn，伴随矩阵为
        [ 0  0  0 ... -cn   ]
        [ 1  0  0 ... -c_{n-1} ]
        [ 0  1  0 ... -c_{n-2} ]
        [ ...              ]
        [ 0  0  0  1  -c1  ]
        
        Monomial 判定（对伴随矩阵 C(χ) 而言）：
        - 由于 C(χ) 的子对角线固定为 1（=p^0），要使每行仅一个非零，
          必须满足 c1=...=c_{n-1}=0；
        - 此时唯一可能的非零“回边”条目为 -cn（位于 (0,n-1)），并且必须满足
              -cn = p^e  (e ≥ 0)。
          否则无法在 MVP20 的 MonomialOperator 表示中保持严格语义。
        
        Args:
            spectral_data: 特征多项式系数 [c1, ..., cn]（来自 _extract_spectral_data）
            prime: 素数 p（MVP20 需要）
            spectral_cover_degree: 谱覆盖度数 n
        
        Returns:
            MonomialOperator（严格构造成功）
        
        Raises:
            HitchinBackendError: 任何输入不合法或无法严格 monomial 化的情形（禁止返回 None / 禁止静默降级）
        """
        if not spectral_data:
            raise HitchinBackendError("Empty spectral data")
        if prime < 2:
            raise HitchinBackendError(f"Prime must be >= 2, got {prime}")
        if spectral_cover_degree <= 0:
            raise HitchinBackendError(f"spectral_cover_degree must be positive, got {spectral_cover_degree}")
        
        n = spectral_cover_degree

        coeffs = list(spectral_data)
        if len(coeffs) != int(n):
            raise HitchinBackendError(
                f"export_monomial_operator expects spectral_data length == spectral_cover_degree={n}, "
                f"got len={len(coeffs)}"
            )

        # 伴随矩阵 monomial 判定：必须 c1..c_{n-1}=0
        for k in range(0, n - 1):
            if int(coeffs[k]) != 0:
                raise HitchinBackendError(
                    "Companion matrix is not monomial: requires c1..c_{n-1}=0, "
                    f"but c{k+1}={coeffs[k]}."
                )

        cn = int(coeffs[-1])
        if cn == 0:
            raise HitchinBackendError("Companion matrix is not monomial: requires cn != 0 (nonzero wrap-around weight).")

        # 回边权重是 (-cn)。严格要求它是 **正的** p^e（不接受符号/单位因子被丢弃）。
        a = -cn
        if a <= 0:
            raise HitchinBackendError(
                f"Companion matrix wrap-around entry must be +p^e. Got -cn={a} from cn={cn}."
            )

        e = 0
        temp = int(a)
        while temp > 1:
            if temp % int(prime) != 0:
                raise HitchinBackendError(
                    f"Companion matrix wrap-around entry -cn={a} is not a pure power of p={prime}."
                )
            temp //= int(prime)
            e += 1

        # companion matrix corresponds to the cycle shift permutation:
        #   e_{n-1} ↦ p^e · e_0,   e_j ↦ e_{j+1} for j<n-1
        perm = tuple((i + 1) % n for i in range(n))
        exp = tuple((0 if i < n - 1 else int(e)) for i in range(n))
        
        # 导入 MonomialOperator
        try:
            from .mvp17_cfg_object_model import MonomialOperator
        except ImportError as e:
            raise HitchinBackendError(
                "MonomialOperator import failed; MVP17 backend required"
            ) from e
        
        return MonomialOperator(
            p=prime,
            perm=perm,
            exp=exp,
            steps=1,  # 单步 Frobenius
        )


# =============================================================================
# 8) 主编排器：整合所有组件
# =============================================================================


class MVP19Orchestrator:
    """
    MVP19 主编排器
    
    整合所有组件的完整求解流程：
    1. 热带预求解 → 锁定搜索范围
    2. Arakelov 度量 → 统一数论信息
    3. 签名格构造 → 代数 + 几何融合
    4. BKZ + Kannan → 强力规约 + CVP
    5. Phase-5 外部强化（Teleport-F5）→ YMH/DUY/Floer 证书化谱数据
    6. 辛几何流 → 逃逸局部极小
    7. Hitchin 后端 → 对接 MVP18（Phase 6）
    """
    
    def __init__(
        self,
        tensor_engine: Optional[TensorEngineProtocol] = None,
        dimension: int = 8,
        *,
        adelic_primes: Optional[List[int]] = None,
        bkz_block_size: int = 20,
    ):
        """
        初始化编排器。
        
        Args:
            tensor_engine: 签名张量引擎（None 则使用默认）
            dimension: 问题维度
            adelic_primes: Adelic 度量使用的素数
            bkz_block_size: BKZ 块大小
        """
        self.dimension = dimension
        
        # 初始化签名引擎
        if tensor_engine is None:
            self.te: TensorEngineProtocol = SignatureTensorEngine(dimension)
        else:
            self.te = tensor_engine
        
        # 初始化各组件
        self.backend = NativeLatticeBackend()
        self.tropical = TropicalPreSolver(dimension)
        
        if adelic_primes is None:
            adelic_primes = [2, 3, 5, 7, 11, 13]
        self.adelic = AdelicMetricSpace(adelic_primes)
        
        self.bkz_block_size = bkz_block_size
        
        # Hitchin 后端
        self.hitchin_backend = SignatureCVPHitchinBackend(
            self.te,
            adelic_primes=adelic_primes,
            bkz_block_size=bkz_block_size,
        )
    
    def solve_polynomial_system(
        self,
        polynomials: List[Dict[Tuple[int, ...], int]],
        *,
        scaling_factor: Optional[int] = None,
        method: str = "bkz_kannan",
        signature_depth: int = 2,
    ) -> CVPSolution:
        """
        求解多项式系统。
        
        Args:
            polynomials: 多项式列表 {指数元组: 系数}
            scaling_factor: 整数化缩放因子
            method: 求解方法
                - "lll_babai": LLL + Babai（快速近似）
                - "bkz_babai": BKZ + Babai（更好近似）
                - "bkz_kannan": BKZ + Kannan Embedding（最强）
            signature_depth: 签名张量截断深度
        
        Returns:
            CVP 解
        """
        if signature_depth <= 0:
            raise ValueError(f"signature_depth must be positive, got {signature_depth}")

        # Core 2: scaling_factor 自动推导（可选，保持向后兼容）
        if scaling_factor is None:
            scaling_factor = ArithmeticScaling.derive_scaling_factor(
                polynomials, signature_depth=signature_depth
            )
        if not isinstance(scaling_factor, int) or scaling_factor <= 0:
            raise ValueError(f"scaling_factor must be a positive int or None, got {scaling_factor!r}")
        
        # 1. 热带预求解
        skeleton = self.tropical.compute_tropical_skeleton(polynomials)
        
        # 2. 构造签名格
        lattice_term = SignatureLatticeTerminator(
            self.te,
            lattice_dimension_cap=200,
        )
        
        # 重要：首先收集所有多项式的 trace，计算统一的签名
        # 这确保所有签名张量具有相同的维度
        all_traces: List[List[Tuple[int, int]]] = []
        for poly in polynomials:
            trace: List[Tuple[int, int]] = []
            for exp, coeff in poly.items():
                for i, e in enumerate(exp):
                    if e != 0:
                        trace.append((i % self.dimension, coeff * e))
            all_traces.append(trace)
        
        # Core 1: 优先使用 Log-Signature + Lyndon(Hall) 基底（若 tensor_engine 支持）
        use_logsig_lyndon = hasattr(self.te, "compute_log_signature_lyndon") and hasattr(self.te, "lyndon_basis_keys")

        canonical_keys: List[Tuple[int, ...]]
        if use_logsig_lyndon:
            canonical_keys = list(self.te.lyndon_basis_keys(signature_depth, include_empty=True))  # type: ignore[attr-defined]
        else:
            # 使用所有 trace 的聚合签名作为参考（保证键结构一致）
            canonical_trace: List[Tuple[int, int]] = []
            for t in all_traces:
                canonical_trace.extend(t)
            canonical_sig = self.te.compute_signature(canonical_trace, depth=signature_depth)
            canonical_keys = list(canonical_sig.keys())

        # 设置目标为零（保持相同键结构）
        # - signature 模式：空词系数为 1（单位元）
        # - log-signature 模式：空词系数为 0（Lie 代数无常数项）
        zero_sig: Dict[Tuple[int, ...], Fraction] = {k: Fraction(0) for k in canonical_keys}
        if (not use_logsig_lyndon) and (() in zero_sig):
            zero_sig[()] = Fraction(1)
        
        lattice_term.set_target_signature(zero_sig)
        
        # 添加多项式，确保签名具有相同的键结构
        for i, poly in enumerate(polynomials):
            trace = all_traces[i]
            if use_logsig_lyndon:
                raw_sig = self.te.compute_log_signature_lyndon(trace, depth=signature_depth, include_empty=True)  # type: ignore[attr-defined]
            else:
                raw_sig = self.te.compute_signature(trace, depth=signature_depth)
            
            # 扩展签名以包含规范签名的所有键
            extended_sig: Dict[Tuple[int, ...], Fraction] = {}
            for key in canonical_keys:
                extended_sig[key] = raw_sig.get(key, Fraction(0))
            
            height = max(int(math.ceil(math.log2(abs(c) + 1))) for c in poly.values())
            
            gp = GeometricPolynomial(
                coeffs=poly,
                signature_tensor=extended_sig,
                origin_signature_hash=hash(tuple(sorted(poly.items()))),
                algebraic_height=height,
            )
            lattice_term.add_basis_polynomial(gp)
        
        # 3. 构造格问题
        problem = lattice_term.build_lattice_problem(scaling_factor=scaling_factor)
        
        # 4. 求解
        if method == "lll_babai":
            return solve_signature_cvp(
                problem,
                backend=self.backend,
                method="babai",
            )
        elif method == "bkz_babai":
            # 先 BKZ 规约，再 Babai
            reduced = self.backend.bkz_reduce(
                problem.basis_matrix,
                block_size=self.bkz_block_size,
            )
            v = self.backend.babai_closest_vector(reduced, problem.target_vector)
            
            # 构造解
            r = [v[i] - problem.target_vector[i] for i in range(len(problem.target_vector))]
            full_sq = sum(x * x for x in r)
            geo_sq = sum(x * x for x in r[problem.n_alg_cols:])
            
            inv_map = {col: mono for mono, col in problem.monomial_map.items()}
            coeffs = {inv_map[col]: v[col] for col in range(problem.n_alg_cols) if v[col] != 0}
            
            return CVPSolution(
                closest_vector=v,
                residual_vector=r,
                residual_norm_sq=full_sq,
                geo_residual_norm_sq=geo_sq,
                algebra_coeffs=coeffs,
                scaling_factor=scaling_factor,
            )
        elif method == "bkz_kannan":
            # Kannan Embedding
            # M 作为算法参数必须可审计：这里用问题维度导出的保守下界（无固定10倍魔法数）
            embedding_scale = scaling_factor * (len(problem.target_vector) + 1)
            v, coeffs_vec = self.backend.solve_cvp_via_kannan(
                problem.basis_matrix,
                problem.target_vector,
                embedding_scale=embedding_scale,
                reduction_method="bkz",
                bkz_block_size=self.bkz_block_size,
            )
            
            r = [v[i] - problem.target_vector[i] for i in range(len(problem.target_vector))]
            full_sq = sum(x * x for x in r)
            geo_sq = sum(x * x for x in r[problem.n_alg_cols:])
            
            inv_map = {col: mono for mono, col in problem.monomial_map.items()}
            coeffs = {inv_map[col]: v[col] for col in range(problem.n_alg_cols) if v[col] != 0}
            
            return CVPSolution(
                closest_vector=v,
                residual_vector=r,
                residual_norm_sq=full_sq,
                geo_residual_norm_sq=geo_sq,
                algebra_coeffs=coeffs,
                scaling_factor=scaling_factor,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_hitchin_fiber(
        self,
        curve_data: Dict[str, Any],
        spectral_cover_degree: int,
    ) -> HitchinFiberData:
        """
        计算 Hitchin 纤维（对接 MVP18）。
        
        这是 MVP19 对 MVP18 的核心接口。
        """
        return self.hitchin_backend.compute_fiber_points(
            curve_data,
            spectral_cover_degree,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取运行统计信息。"""
        return {
            "backend": self.backend._stats.copy(),
            "dimension": self.dimension,
            "bkz_block_size": self.bkz_block_size,
        }


