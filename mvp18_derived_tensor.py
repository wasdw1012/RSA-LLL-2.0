"""
MVP18 证人 强化组件 - 导出张量积完整的数学合法实现
"""

__all__ = [
    # ═══════════════════════════════════════════════════════════════════════════
    # 异常系统
    # ═══════════════════════════════════════════════════════════════════════════
    "TensorProductError",
    "LinearAlgebraError",
    "ChainComplexValidationError",
    "SimplicialIdentityError",
    # ═══════════════════════════════════════════════════════════════════════════
    # 数值线性代数配置
    # ═══════════════════════════════════════════════════════════════════════════
    "NumericalLinearAlgebraConfig",
    # ═══════════════════════════════════════════════════════════════════════════
    # 代数结构（数值版）
    # ═══════════════════════════════════════════════════════════════════════════
    "Ring",
    "PolynomialRing",
    "Module",
    "FreeModule",
    # ═══════════════════════════════════════════════════════════════════════════
    # 链复形与投射分解
    # ═══════════════════════════════════════════════════════════════════════════
    "ChainComplex",
    "ProjectiveResolution",
    # ═══════════════════════════════════════════════════════════════════════════
    # 导出张量积
    # ═══════════════════════════════════════════════════════════════════════════
    "DerivedTensorProduct",
    # ═══════════════════════════════════════════════════════════════════════════
    # 单纯结构
    # ═══════════════════════════════════════════════════════════════════════════
    "SimplicialStructure",
    # ═══════════════════════════════════════════════════════════════════════════
    # 中间件
    # ═══════════════════════════════════════════════════════════════════════════
    "DerivedTensorProductMiddleware",
    # ═══════════════════════════════════════════════════════════════════════════
    # 余切复形上同调（数值版）
    # ═══════════════════════════════════════════════════════════════════════════
    "CotangentComplexCohomology",
    # ═══════════════════════════════════════════════════════════════════════════
    # Voronoi 积分权重
    # ═══════════════════════════════════════════════════════════════════════════
    "VoronoiIntegrationWeights",
    # ═══════════════════════════════════════════════════════════════════════════
    # 完整虚拟循环积分器
    # ═══════════════════════════════════════════════════════════════════════════
    "CompleteVirtualCycleIntegrator",
    # ═══════════════════════════════════════════════════════════════════════════
    # Kähler 诱导映射 (Merged from kahler_induced_map.py)
    # ═══════════════════════════════════════════════════════════════════════════
    "Poly",
    "RingHomomorphism",
    "SimplicialPolyRing",
    "SimplicialKahlerMatrices",
    "compute_cotangent_cohomology",
    "analyze_derived_intersection",
    "create_standard_simplicial_polyring",
    "create_evm_simplicial_polyring",
    # ═══════════════════════════════════════════════════════════════════════════
    # MVP19 Syzygy Adapter
    # ═══════════════════════════════════════════════════════════════════════════
    "SyzygyKahlerAdapter",
]

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Set, TypeVar, Generic, Mapping, Union, Sequence, Iterable
from dataclasses import dataclass
import scipy.linalg
import scipy.sparse
from scipy.sparse.linalg import svds
from itertools import product
import math
import warnings
from enum import Enum
from fractions import Fraction

# ======================================
# 0. 数值线性代数的非魔法数配置与异常
# ======================================


class TensorProductError(Exception):
    """本文件内张量/导出张量积相关错误的基类。"""


class LinearAlgebraError(TensorProductError):
    """数值线性代数失败（SVD/QR/求解等）。"""


class ChainComplexValidationError(TensorProductError):
    """链复形条件/同调定义前提不满足。"""


class SimplicialIdentityError(TensorProductError):
    """单纯恒等式不满足。"""


@dataclass(frozen=True)
class NumericalLinearAlgebraConfig:
    """
    数值容差全部由机器精度和问题尺度导出，避免硬编码1e-10/0.999类魔法数。

    约定：
    - 仅针对浮点矩阵（np.floating）。若输入为非浮点（例如多项式系数对象），
      本实现会明确抛异常要求调用方提供可比的范数/零判定。
    """

    dtype: np.dtype = np.dtype(np.float64)

    @property
    def eps(self) -> float:
        return float(np.finfo(self.dtype).eps)

    def svd_rank_tol(self, s_max: float, shape: Tuple[int, int]) -> float:
        """
        SVD秩判定阈值：tol = eps * max(shape) * s_max
        这是数值线性代数的标准尺度（与 LAPACK 的经验规则一致），但不硬编码常数。
        """
        if s_max == 0:
            return 0.0
        scale = float(max(shape))
        return self.eps * scale * float(s_max)

    def qr_rank_tol(self, diag_r_max: float, shape: Tuple[int, int]) -> float:
        """
        QR秩判定阈值：tol = eps * max(shape) * max(|diag(R)|)
        """
        if diag_r_max == 0:
            return 0.0
        scale = float(max(shape))
        return self.eps * scale * float(diag_r_max)

    def composition_zero_tol(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        判断 A@B 是否应当视为数值零的阈值，按误差传播给出尺度：
        ||A@B|| 约为 O(eps) * ||A|| * ||B||。
        """
        if a.size == 0 or b.size == 0:
            return 0.0
        scale = float(max(a.shape[0], a.shape[1], b.shape[0], b.shape[1]))
        a_norm = float(np.linalg.norm(a, ord=2))
        b_norm = float(np.linalg.norm(b, ord=2))
        return self.eps * scale * a_norm * b_norm


def _require_floating_matrix(matrix: np.ndarray, *, context: str) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{context}: expected np.ndarray, got {type(matrix).__name__}")
    if matrix.size == 0:
        return
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError(
            f"{context}: expected floating dtype for numerical linear algebra, got {matrix.dtype}. "
            f"If you are working over a symbolic ring, provide an exact-linear-algebra backend."
        )


def _svd_image_basis(matrix: np.ndarray, cfg: NumericalLinearAlgebraConfig) -> Tuple[np.ndarray, int]:
    """返回列空间的一个正交基（U 的前 r 列）。"""
    _require_floating_matrix(matrix, context="image_basis")
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=cfg.dtype), 0
    try:
        u, s, _vh = scipy.linalg.svd(matrix, full_matrices=False)
    except Exception as e:
        raise LinearAlgebraError("SVD failed while computing image basis.") from e
    tol = cfg.svd_rank_tol(float(np.max(s) if s.size else 0.0), matrix.shape)
    r = int(np.sum(s > tol))
    if r == 0:
        return np.zeros((matrix.shape[0], 0), dtype=cfg.dtype), 0
    return u[:, :r], r


def _svd_kernel_basis(matrix: np.ndarray, cfg: NumericalLinearAlgebraConfig) -> Tuple[np.ndarray, int]:
    """返回零空间的一个基（V 的后 n-r 列）。"""
    _require_floating_matrix(matrix, context="kernel_basis")
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=cfg.dtype), int(matrix.shape[1])
    try:
        _u, s, vh = scipy.linalg.svd(matrix, full_matrices=True)
    except Exception as e:
        raise LinearAlgebraError("SVD failed while computing kernel basis.") from e
    tol = cfg.svd_rank_tol(float(np.max(s) if s.size else 0.0), matrix.shape)
    r = int(np.sum(s > tol))
    null_dim = int(vh.shape[0] - r)
    if null_dim == 0:
        return np.zeros((matrix.shape[1], 0), dtype=cfg.dtype), 0
    return vh[r:, :].T, null_dim


def _qr_orthonormal_columns(matrix: np.ndarray, cfg: NumericalLinearAlgebraConfig) -> np.ndarray:
    """对列向量做正交化，返回列空间的正交基 Q。"""
    _require_floating_matrix(matrix, context="orthonormalize")
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=cfg.dtype)
    try:
        q, r = scipy.linalg.qr(matrix, mode="economic")
    except Exception as e:
        raise LinearAlgebraError("QR failed while orthonormalizing columns.") from e
    diag_r = np.abs(np.diag(r)) if r.size else np.array([], dtype=cfg.dtype)
    diag_max = float(np.max(diag_r) if diag_r.size else 0.0)
    tol = cfg.qr_rank_tol(diag_max, matrix.shape)
    keep = diag_r > tol
    return q[:, keep]


# ==============================
# 1. 代数基础结构定义
# ==============================

T = TypeVar('T')

class Ring:
    """环的抽象基类"""
    def add(self, a: T, b: T) -> T:
        raise NotImplementedError
    
    def mul(self, a: T, b: T) -> T:
        raise NotImplementedError
    
    def zero(self) -> T:
        raise NotImplementedError
    
    def one(self) -> T:
        raise NotImplementedError
    
    def neg(self, a: T) -> T:
        raise NotImplementedError

class PolynomialRing(Ring):
    """多项式环实现"""
    
    def __init__(self, base_ring, variables: List[str]):
        self.base_ring = base_ring
        self.variables = variables
    
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """多项式加法"""
        max_deg = max(a.shape[0], b.shape[0])
        a_pad = np.pad(a, (0, max_deg - a.shape[0]), 'constant')
        b_pad = np.pad(b, (0, max_deg - b.shape[0]), 'constant')
        return a_pad + b_pad
    
    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """多项式乘法（卷积）"""
        deg_a = a.shape[0] - 1
        deg_b = b.shape[0] - 1
        result = np.zeros(deg_a + deg_b + 1)
        
        for i in range(deg_a + 1):
            for j in range(deg_b + 1):
                result[i + j] += a[i] * b[j]
        
        return result
    
    def zero(self) -> np.ndarray:
        """零多项式"""
        return np.array([0.0])
    
    def one(self) -> np.ndarray:
        """常数多项式1"""
        return np.array([1.0])
    
    def neg(self, a: np.ndarray) -> np.ndarray:
        """多项式取负"""
        return -a

class Module:
    """模的抽象基类"""
    def __init__(self, ring: Ring):
        self.ring = ring
    
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def scalar_mul(self, r: T, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def zero(self) -> np.ndarray:
        raise NotImplementedError

class FreeModule(Module):
    """自由模实现"""
    
    def __init__(self, ring: Ring, rank: int):
        super().__init__(ring)
        self.rank = rank
    
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """向量加法"""
        return a + b
    
    def scalar_mul(self, r: T, v: np.ndarray) -> np.ndarray:
        """标量乘法"""
        return r * v
    
    def zero(self) -> np.ndarray:
        """零向量"""
        return np.zeros(self.rank)
    
    def basis(self) -> List[np.ndarray]:
        """标准基"""
        basis = []
        for i in range(self.rank):
            e = np.zeros(self.rank)
            e[i] = 1.0
            basis.append(e)
        return basis

# ==============================
# 2. 投射分解实现
# ==============================

class ChainComplex:
    """链复形实现"""
    
    def __init__(
        self,
        modules: List[Module],
        differentials: List[np.ndarray],
        *,
        la_cfg: Optional[NumericalLinearAlgebraConfig] = None,
        validate: bool = True,
    ):
        """
        modules: [C_0, C_1, C_2, ...] 从0开始
        differentials: [d_1: C_1 → C_0, d_2: C_2 → C_1, ...]
        """
        self.modules = modules
        self.differentials = differentials
        self.la_cfg = la_cfg or NumericalLinearAlgebraConfig(
            dtype=(differentials[0].dtype if differentials else np.dtype(np.float64))
        )
        
        # 验证链复形条件: d_{n} ∘ d_{n+1} = 0
        if validate:
            self._validate_chain_complex()
    
    def _validate_chain_complex(self):
        """验证链复形条件 d_{n} ∘ d_{n+1} = 0"""
        for n in range(len(self.differentials) - 1):
            d_n = self.differentials[n]
            d_n1 = self.differentials[n+1]
            
            # 计算复合映射
            composition = d_n @ d_n1
            
            # 检查是否为零映射（容差由机器精度与算子尺度推导）
            tol = self.la_cfg.composition_zero_tol(d_n, d_n1)
            comp_norm = float(np.linalg.norm(composition, ord=2)) if composition.size else 0.0
            if comp_norm > tol:
                raise ChainComplexValidationError(
                    "链复形条件失败: d_n ∘ d_{n+1} ≠ 0 "
                    f"(n={n}, ||d_n d_{n+1}||_2={comp_norm}, tol={tol})."
                )
    
    def homology(self, n: int) -> Tuple[np.ndarray, int]:
        """
        计算第n个同调群 H_n = Ker(d_n) / Im(d_{n+1})
        
        返回: (同调群基, 维数)
        """
        if n < 0 or n >= len(self.modules):
            return np.zeros((self.modules[0].rank, 0)), 0
        
        # 获取微分算子
        d_n = self.differentials[n-1] if n > 0 else None  # d_n: C_n → C_{n-1}
        d_n1 = self.differentials[n] if n < len(self.differentials) else None  # d_{n+1}: C_{n+1} → C_n
        
        # 计算核空间 Ker(d_n)
        kernel_basis, kernel_dim = self._compute_kernel_basis(d_n, self.modules[n].rank)
        
        # 计算像空间 Im(d_{n+1})
        if d_n1 is not None:
            image_basis, image_dim = self._compute_image_basis(d_n1)
        else:
            image_basis, image_dim = np.zeros((self.modules[n].rank, 0)), 0
        
        # 计算商空间 Ker(d_n) / Im(d_{n+1})
        homology_basis, homology_dim = self._compute_quotient_space(kernel_basis, image_basis)
        
        return homology_basis, homology_dim
    
    def _compute_kernel_basis(self, matrix: Optional[np.ndarray], dim: int) -> Tuple[np.ndarray, int]:
        """计算矩阵的核空间基"""
        if matrix is None or matrix.size == 0:
            return np.eye(dim, dtype=self.la_cfg.dtype), dim

        kernel_basis, kernel_dim = _svd_kernel_basis(matrix, self.la_cfg)
        return kernel_basis, kernel_dim
    
    def _compute_image_basis(self, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """计算矩阵的像空间基"""
        if matrix.size == 0:
            return np.zeros((matrix.shape[0], 0), dtype=self.la_cfg.dtype), 0

        image_basis, image_dim = _svd_image_basis(matrix, self.la_cfg)
        return image_basis, image_dim
    
    def _compute_quotient_space(self, kernel_basis: np.ndarray, 
                               image_basis: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        计算商空间 Ker(d_n) / Im(d_{n+1})
        
        找到 Ker 中与 Im 正交的补空间
        """
        if kernel_basis.shape[1] == 0:
            return np.zeros((kernel_basis.shape[0], 0)), 0
        
        if image_basis.shape[1] == 0:
            return kernel_basis, kernel_basis.shape[1]
        
        # 数学上要求 Im(d_{n+1}) ⊆ Ker(d_n) 才能定义商空间。
        # 这里用正交投影做严格检查：若 image 中存在明显不在 kernel 中的分量，直接抛异常。
        qk = _qr_orthonormal_columns(kernel_basis, self.la_cfg)
        qi = _qr_orthonormal_columns(image_basis, self.la_cfg)

        if qi.shape[1] == 0:
            return qk, qk.shape[1]

        # 检查 image 是否包含在 kernel 张成空间中
        proj_i_onto_k = qk @ (qk.T @ image_basis) if qk.shape[1] else np.zeros_like(image_basis)
        residual = image_basis - proj_i_onto_k
        # residual 的应为零阈值：eps * 规模 * ||image_basis||
        residual_norm = float(np.linalg.norm(residual, ord=2)) if residual.size else 0.0
        image_norm = float(np.linalg.norm(image_basis, ord=2)) if image_basis.size else 0.0
        tol = self.la_cfg.eps * float(max(image_basis.shape)) * image_norm
        if residual_norm > tol:
            raise ChainComplexValidationError(
                "无法计算同调：检测到 Im(d_{n+1}) 非 Ker(d_n) 的子空间 "
                f"(||residual||_2={residual_norm}, tol={tol})."
            )

        # Ker 中与 Im 正交的补空间：Q = (I - P_im) Ker
        #
        # 重要数值细节：
        # - 当 Im == Ker（同调为 0）时，qk_perp 理论上应当为 0；
        # - 但 qk_perp 由两项“同尺度矩阵差”构成，舍入误差会留下 O(eps) 量级的残差；
        # - 若直接用“相对 qk_perp 自身尺度”的秩阈值，会把这类残差误判为非零方向。
        #
        # 因此：先用以 qk/qi 的尺度为基准的绝对阈值过滤近零列，再做 QR 正交化。
        qk_perp = qk - qi @ (qi.T @ qk)
        if qk_perp.size == 0 or qk_perp.shape[1] == 0:
            return np.zeros((qk.shape[0], 0), dtype=self.la_cfg.dtype), 0

        col_norms = np.sqrt(np.sum(qk_perp * qk_perp, axis=0))
        scale = max(
            float(np.linalg.norm(qk, ord=2)) if qk.size else 0.0,
            float(np.linalg.norm(qi, ord=2)) if qi.size else 0.0,
        )
        tol = self.la_cfg.eps * float(max(qk_perp.shape)) * scale
        keep = col_norms > tol
        qk_perp = qk_perp[:, keep]
        if qk_perp.shape[1] == 0:
            return np.zeros((qk.shape[0], 0), dtype=self.la_cfg.dtype), 0

        qq = _qr_orthonormal_columns(qk_perp, self.la_cfg)
        return qq, qq.shape[1]

class ProjectiveResolution:
    """投射分解实现"""
    
    def __init__(self, module: Module, max_length: int = 10):
        """
        构建模的投射分解
        
        参数:
        module: 要分解的模
        max_length: 最大分解长度
        """
        self.module = module
        self.max_length = max_length
        self.resolution = self._construct_resolution()
    
    def _construct_resolution(self) -> ChainComplex:
        """
        构造投射分解 P_• → M。

        数学事实：
        - 自由模是投射模，因此其投射分解可以取长度 0的平凡分解：P_0 = M，P_n=0 (n>0)。

        由于本文件的 `Module` 抽象没有给出有限生成表示/呈示矩阵/环上 syzygy 算法等数据，
        对一般模无法在不做启发式假设的前提下自动构造投射分解，因此这里**明确抛异常**。
        """
        if isinstance(self.module, FreeModule):
            return ChainComplex([self.module], [], validate=True)

        # 允许模块提供自己的严格构造（避免本处做任何猜测/近似）
        if hasattr(self.module, "projective_resolution") and callable(getattr(self.module, "projective_resolution")):
            resolution = self.module.projective_resolution(max_length=self.max_length)
            if not isinstance(resolution, ChainComplex):
                raise TypeError(
                    "module.projective_resolution(...) must return a ChainComplex; "
                    f"got {type(resolution).__name__}."
                )
            return resolution

        raise NotImplementedError(
            "Cannot construct a projective resolution for this Module without additional algebraic data "
            "(e.g. a finite presentation and a syzygy algorithm over the base ring). "
            "Provide module.projective_resolution(max_length=...) or supply resolutions explicitly."
        )
    
    def get_chain_complex(self) -> ChainComplex:
        """获取完整的链复形"""
        return self.resolution

# ==============================
# 3. 导出张量积核心实现
# ==============================

class DerivedTensorProduct:
    """导出张量积的完整实现"""
    
    def __init__(
        self,
        ring: Ring,
        max_homological_degree: int = 5,
        *,
        la_cfg: Optional[NumericalLinearAlgebraConfig] = None,
    ):
        """
        ring: 基础环
        max_homological_degree: 最大同调度数
        """
        self.ring = ring
        self.max_degree = max_homological_degree
        self.la_cfg = la_cfg or NumericalLinearAlgebraConfig()
    
    def compute(self, module1: Module, module2: Module) -> ChainComplex:
        """
        计算两个模的导出张量积 M ⊗^L_R N
        
        严格遵循数学定义:
        1. 取 M 的投射分解 P_•
        2. 取 N 的投射分解 Q_•
        3. 构造双复形 P_• ⊗ Q_•
        4. 取总复形 Tot(P_• ⊗ Q_•)
        5. 返回该总复形
        """
        # 步骤1: 构建（或获取）投射分解
        resolution1 = ProjectiveResolution(module1, self.max_degree).get_chain_complex()
        resolution2 = ProjectiveResolution(module2, self.max_degree).get_chain_complex()

        # 步骤2: 取张量积双复形的总复形（严格 Kronecker 构造）
        return self._total_complex_of_tensor_product(resolution1, resolution2)

    def _truncate_resolution(self, resolution: ChainComplex) -> ChainComplex:
        """
        仅保留 0..max_degree 的链复形项（differentials 同步截断）。
        """
        if self.max_degree is None:
            return resolution
        keep = min(len(resolution.modules), self.max_degree + 1)
        return ChainComplex(
            resolution.modules[:keep],
            resolution.differentials[: max(0, keep - 1)],
            la_cfg=resolution.la_cfg,
            validate=True,
        )

    def _require_free_chain_complex(self, resolution: ChainComplex, *, name: str) -> None:
        for idx, m in enumerate(resolution.modules):
            if not hasattr(m, "rank"):
                raise TypeError(f"{name}.modules[{idx}] has no `rank` attribute; expected free modules.")
            if not isinstance(m.rank, int):
                raise TypeError(f"{name}.modules[{idx}].rank must be int; got {type(m.rank).__name__}.")
            if m.rank < 0:
                raise ValueError(f"{name}.modules[{idx}].rank must be non-negative.")
        for i, d in enumerate(resolution.differentials):
            _require_floating_matrix(d, context=f"{name}.differentials[{i}]")
            src_rank = resolution.modules[i + 1].rank
            dst_rank = resolution.modules[i].rank
            if d.shape != (dst_rank, src_rank):
                raise ValueError(
                    f"{name}.differentials[{i}] has shape {d.shape}, expected {(dst_rank, src_rank)}."
                )

    def _total_complex_of_tensor_product(self, res1: ChainComplex, res2: ChainComplex) -> ChainComplex:
        """
        给定两个链复形（通常是投射分解）P_• 与 Q_•，构造总复形 Tot(P ⊗ Q)：
        Tot_n = ⊕_{p+q=n} (P_p ⊗ Q_q)
        d_tot = d^h + d^v, 其中 d^h = d_P ⊗ id, d^v = (-1)^p id ⊗ d_Q。
        """
        res1 = self._truncate_resolution(res1)
        res2 = self._truncate_resolution(res2)

        self._require_free_chain_complex(res1, name="resolution1")
        self._require_free_chain_complex(res2, name="resolution2")

        p_len = len(res1.modules)
        q_len = len(res2.modules)
        if p_len == 0 or q_len == 0:
            raise ValueError("Resolutions must contain at least degree-0 term.")

        max_n = (p_len - 1) + (q_len - 1)
        dtype = np.result_type(
            *(d.dtype for d in (res1.differentials + res2.differentials) if isinstance(d, np.ndarray)),
            self.la_cfg.dtype,
        )
        la_cfg = NumericalLinearAlgebraConfig(dtype=np.dtype(dtype))

        components_by_n: List[List[Tuple[int, int]]] = []
        offsets_by_n: List[Dict[Tuple[int, int], Tuple[int, int]]] = []
        total_modules: List[FreeModule] = []

        for n in range(max_n + 1):
            pairs: List[Tuple[int, int]] = []
            p_min = max(0, n - (q_len - 1))
            p_max = min(n, p_len - 1)
            for p in range(p_min, p_max + 1):
                q = n - p
                if 0 <= q <= (q_len - 1):
                    pairs.append((p, q))

            offsets: Dict[Tuple[int, int], Tuple[int, int]] = {}
            cursor = 0
            for p, q in pairs:
                dim = int(res1.modules[p].rank) * int(res2.modules[q].rank)
                offsets[(p, q)] = (cursor, cursor + dim)
                cursor += dim

            components_by_n.append(pairs)
            offsets_by_n.append(offsets)
            total_modules.append(FreeModule(self.ring, cursor))

        total_differentials: List[np.ndarray] = []
        for n in range(1, max_n + 1):
            dim_src = total_modules[n].rank
            dim_dst = total_modules[n - 1].rank
            d_tot = np.zeros((dim_dst, dim_src), dtype=dtype)

            for p, q in components_by_n[n]:
                col_start, col_end = offsets_by_n[n][(p, q)]

                # 水平： (p,q) -> (p-1,q)
                if p > 0:
                    row_start, row_end = offsets_by_n[n - 1][(p - 1, q)]
                    d_p = res1.differentials[p - 1].astype(dtype, copy=False)
                    id_q = np.eye(res2.modules[q].rank, dtype=dtype)
                    block = np.kron(d_p, id_q)
                    d_tot[row_start:row_end, col_start:col_end] += block

                # 垂直： (p,q) -> (p,q-1)，带符号 (-1)^p
                if q > 0:
                    row_start, row_end = offsets_by_n[n - 1][(p, q - 1)]
                    d_q = res2.differentials[q - 1].astype(dtype, copy=False)
                    id_p = np.eye(res1.modules[p].rank, dtype=dtype)
                    block = np.kron(id_p, d_q)
                    if (p % 2) == 1:
                        block = -block
                    d_tot[row_start:row_end, col_start:col_end] += block

            total_differentials.append(d_tot)

        return ChainComplex(total_modules, total_differentials, la_cfg=la_cfg, validate=True)
    
    def compute_homology(self, total_complex: ChainComplex, n: int) -> Tuple[np.ndarray, int]:
        """
        计算导出张量积的第n个同调群
        
        H_n(M ⊗^L_R N) = H_n(Tot(P_• ⊗ Q_•))
        """
        return total_complex.homology(n)

# ==============================
# 4. 单纯结构与面算子实现
# ==============================

class SimplicialStructure:
    """单纯结构实现，用于维护导出张量积的单纯性质"""
    
    def __init__(self, max_dimension: int):
        self.max_dim = max_dimension
        self.face_operators = self._construct_face_operators()
        self.degeneracy_operators = self._construct_degeneracy_operators()
    
    def _construct_face_operators(self) -> List[List[np.ndarray]]:
        """
        构造面算子 ∂_i: P_n → P_{n-1}
        
        满足单纯恒等式:
        ∂_i ∂_j = ∂_{j-1} ∂_i  for i < j
        """
        face_ops = []
        
        for n in range(self.max_dim + 1):
            ops_at_level = []
            
            for i in range(n + 1):
                # 构造 ∂_i: P_n → P_{n-1}
                if n == 0:
                    # P_0 没有面算子
                    op = np.zeros((0, 1))
                else:
                    # 创建随机矩阵但确保单纯恒等式
                    op = self._create_face_operator(n, i)
                
                ops_at_level.append(op)
            
            face_ops.append(ops_at_level)
        
        # 验证单纯恒等式
        self._validate_simplicial_identities(face_ops)
        
        return face_ops
    
    def _construct_degeneracy_operators(self) -> List[List[np.ndarray]]:
        """
        构造退化算子 s_i: P_n → P_{n+1}
        
        满足单纯恒等式:
        s_i s_j = s_{j+1} s_i  for i ≤ j
        """
        degen_ops = []
        
        for n in range(self.max_dim):
            ops_at_level = []
            
            for i in range(n + 1):
                # 构造 s_i: P_n → P_{n+1}
                op = self._create_degeneracy_operator(n, i)
                ops_at_level.append(op)
            
            degen_ops.append(ops_at_level)
        
        # 验证单纯恒等式
        self._validate_simplicial_identities_with_degeneracies(
            self.face_operators, degen_ops)
        
        return degen_ops
    
    def _create_face_operator(self, n: int, i: int) -> np.ndarray:
        """
        创建满足单纯恒等式的面算子 ∂_i: P_n → P_{n-1}
        
        严格实现: 使用标准单纯面算子的矩阵表示
        """
        # 对于单纯集合，面算子可以表示为删除第i个坐标的映射
        # 在自由模的上下文中，我们需要确保单纯恒等式
        
        # 创建 (n) × (n+1) 矩阵
        matrix = np.zeros((n, n+1))
        
        # 对于 ∂_i，它是删除第i个坐标的映射
        for j in range(n+1):
            if j < i:
                matrix[j, j] = 1.0
            elif j > i:
                matrix[j-1, j] = 1.0
        
        return matrix
    
    def _create_degeneracy_operator(self, n: int, i: int) -> np.ndarray:
        """
        创建满足单纯恒等式的退化算子 s_i: P_n → P_{n+1}
        
        严格实现: 使用标准单纯退化算子的矩阵表示
        """
        # 对于单纯集合，退化算子可以表示为重复第i个坐标的映射
        # 在自由模的上下文中，我们需要确保单纯恒等式
        
        # 创建 (n+2) × (n+1) 矩阵
        matrix = np.zeros((n+2, n+1))
        
        # 对于 s_i，它是重复第i个坐标的映射
        for j in range(n+1):
            if j <= i:
                matrix[j, j] = 1.0
            if j >= i:
                matrix[j+1, j] = 1.0
        
        return matrix
    
    def _validate_simplicial_identities(self, face_ops: List[List[np.ndarray]]):
        """验证面算子的单纯恒等式 ∂_i ∂_j = ∂_{j-1} ∂_i for i < j"""
        for n in range(2, self.max_dim + 1):
            for i in range(n):
                for j in range(i+1, n+1):
                    # 计算 ∂_i ∂_j
                    left = face_ops[n-1][i] @ face_ops[n][j]
                    
                    # 计算 ∂_{j-1} ∂_i
                    right = face_ops[n-1][j-1] @ face_ops[n][i]
                    
                    # 检查是否相等
                    if left.shape != right.shape or not np.array_equal(left, right):
                        raise SimplicialIdentityError(
                            f"面算子恒等式失败: ∂_{i}∂_{j} ≠ ∂_{j-1}∂_{i} (n={n})"
                        )
    
    def _validate_simplicial_identities_with_degeneracies(
            self, face_ops: List[List[np.ndarray]], degen_ops: List[List[np.ndarray]]):
        """验证面算子和退化算子的完整单纯恒等式"""
        # 1) 退化恒等式：s_i s_j = s_{j+1} s_i (i ≤ j)
        max_level_for_ss = min(len(degen_ops) - 2, self.max_dim - 2) if self.max_dim >= 2 else -1
        for n in range(max_level_for_ss + 1):
            for j in range(n + 1):
                for i in range(j + 1):
                    left = degen_ops[n + 1][i] @ degen_ops[n][j]
                    right = degen_ops[n + 1][j + 1] @ degen_ops[n][i]
                    if left.shape != right.shape or not np.array_equal(left, right):
                        raise SimplicialIdentityError(
                            f"退化恒等式失败: s_{i}s_{j} ≠ s_{j+1}s_{i} (n={n})"
                        )

        # 2) 面-退化恒等式：d_i s_j 分三种情况
        # 这里用标准公式（不使用 allclose / 容差）：本文件的面/退化算子是 0-1 矩阵，恒等式应当严格成立。
        max_level_for_ds = min(len(degen_ops) - 1, self.max_dim - 1)
        for n in range(1, max_level_for_ds + 1):
            for j in range(n + 1):
                for i in range(n + 2):
                    left = face_ops[n + 1][i] @ degen_ops[n][j]

                    if i < j:
                        right = degen_ops[n - 1][j - 1] @ face_ops[n][i]
                    elif i == j or i == (j + 1):
                        dim = left.shape[0]
                        right = np.eye(dim, dtype=left.dtype)
                    else:
                        right = degen_ops[n - 1][j] @ face_ops[n][i - 1]

                    if left.shape != right.shape or not np.array_equal(left, right):
                        raise SimplicialIdentityError(
                            f"面-退化恒等式失败: d_{i}s_{j} (n={n})"
                        )

# ==============================
# 5. 完整导出张量积中间件
# ==============================

class DerivedTensorProductMiddleware:
    """导出张量积中间件，用于连接不同组件"""
    
    def __init__(self, ring: Ring, max_homological_degree: int = 5):
        self.derived_tensor = DerivedTensorProduct(ring, max_homological_degree)
        self.simplicial_structure = SimplicialStructure(max_homological_degree)
    
    def compute_derived_tensor(self, module1: Module, module2: Module) -> Dict:
        """
        计算导出张量积并返回完整信息
        """
        # 计算导出张量积
        total_complex = self.derived_tensor.compute(module1, module2)
        
        # 计算同调群
        homology_groups = {}
        for n in range(self.derived_tensor.max_degree):
            basis, dim = total_complex.homology(n)
            homology_groups[n] = {
                "dimension": dim,
                "basis": basis if dim > 0 else None
            }
        
        return {
            "total_complex": total_complex,
            "homology_groups": homology_groups,
            "simplicial_structure": {
                "face_operators": self.simplicial_structure.face_operators,
                "degeneracy_operators": self.simplicial_structure.degeneracy_operators
            },
            "metadata": {
                "max_homological_degree": self.derived_tensor.max_degree,
                "ring_type": type(module1.ring).__name__,
                "module1_rank": getattr(module1, "rank", None),
                "module2_rank": getattr(module2, "rank", None),
            }
        }
    
    def verify_derived_property(self, result: Dict) -> bool:
        """
        验证导出张量积的导出性质
        
        关键性质: 如果 M 或 N 是平坦模，则导出张量积应与普通张量积同构
        """
        # 检查 H_0 是否等于普通张量积
        h0 = result["homology_groups"].get(0, {})
        dim_h0 = h0.get("dimension", 0)
        
        # 普通张量积的维数
        meta = result.get("metadata", {})
        r1 = meta.get("module1_rank")
        r2 = meta.get("module2_rank")
        if r1 is None or r2 is None:
            raise ValueError("verify_derived_property requires module ranks in result['metadata'].")
        expected_dim = int(r1) * int(r2)
        
        # 检查是否匹配
        return int(dim_h0) == expected_dim
    
    def integrate_with_virtual_cycle(self, 
                                   result: Dict, 
                                   virtual_cycle_points: np.ndarray) -> np.ndarray:
        """
        将导出张量积与虚拟循环集成
        
        这是"把鬼抓出来，变成人"的关键步骤
        """
        # 链复形采用同调分级（n>=0）。若要使用负次数同调=形变空间的约定，
        raise NotImplementedError(
            "integrate_with_virtual_cycle is not defined for the current (nonnegative) homological grading. "
            "Provide a deformation complex with the desired grading and an explicit integration model."
        )
        


# ==============================================
# 核心一：余调群计算的完整数学实现
# ==============================================

class CotangentComplexCohomology:
    """
    安德烈-奎伦余调群的数值线性代数实现（以矩阵表示为输入）。

    重要约束：
    - 只在给出明确的面算子/退化算子的矩阵表示时工作
    - 不对负次数项/隐含分级延拓做任何猜测：若需要负次数，请显式提供对应复形。
    """
    
    def __init__(
        self,
        face_operators: List[List[np.ndarray]],
        degeneracy_operators: List[List[np.ndarray]],
        *,
        la_cfg: Optional[NumericalLinearAlgebraConfig] = None,
    ):
        """
        face_operators: 面算子 ∂_i: P_n → P_{n-1}
        degeneracy_operators: 退化算子 s_i: P_n → P_{n+1}
        """
        self.face_ops = face_operators
        self.degen_ops = degeneracy_operators
        self.la_cfg = la_cfg or NumericalLinearAlgebraConfig(dtype=self._infer_dtype())
        self.differentials = self._build_differentials()

    def _infer_dtype(self) -> np.dtype:
        for levels in (self.face_ops, self.degen_ops):
            for ops_at_level in levels:
                for op in ops_at_level:
                    if isinstance(op, np.ndarray) and op.size:
                        return np.dtype(op.dtype)
        return self.la_cfg.dtype if hasattr(self, "la_cfg") else np.dtype(np.float64)

    def _dim_of_level(self, n: int) -> int:
        """
        从面/退化算子的矩阵形状推断 L_n 的维数。
        """
        if n < 0:
            raise NotImplementedError("Negative degrees are not inferred; provide them explicitly if needed.")

        if n < len(self.face_ops) and self.face_ops[n]:
            return int(self.face_ops[n][0].shape[1])

        if n < len(self.degen_ops) and self.degen_ops[n]:
            return int(self.degen_ops[n][0].shape[1])

        if (n - 1) >= 0 and (n - 1) < len(self.degen_ops) and self.degen_ops[n - 1]:
            return int(self.degen_ops[n - 1][0].shape[0])

        raise ValueError(f"Cannot infer dimension of level {n} from provided operators.")
        
    def _build_differentials(self) -> List[np.ndarray]:
        """
        构建余切复形的微分算子 d_n: L_n → L_{n-1}
        
        从单纯环的面算子构造：
        d_n = Σ_{i=0}^n (-1)^i ∂_i
        其中 ∂_i 满足单纯恒等式
        """
        differentials = []

        # 约定：L_{-1} = 0，因此 d_0: L_0 -> 0 是零映射。
        dim0 = self._dim_of_level(0)
        differentials.append(np.zeros((0, dim0), dtype=self.la_cfg.dtype))

        # 对于每个维度 n>=1，构造微分 d_n = Σ_{i=0}^n (-1)^i ∂_i
        for n in range(1, len(self.face_ops)):
            faces_at_level_n = self._get_faces_at_level(n)
            diff = self._compute_simplicial_differential(faces_at_level_n)
            differentials.append(diff)
        
        return differentials
    
    def _get_faces_at_level(self, n: int) -> List[np.ndarray]:
        """获取第n层的所有面算子"""
        faces = []
        for i in range(n + 1):
            # ∂_i: P_n → P_{n-1}
            face_op = self.face_ops[n][i] if n < len(self.face_ops) else None
            if face_op is not None:
                faces.append(face_op)
        return faces
    
    def _compute_simplicial_differential(self, faces: List[np.ndarray]) -> np.ndarray:
        """计算单纯微分 d = Σ (-1)^i ∂_i"""
        if not faces:
            return np.zeros((1, 1))
        
        # 检查所有面算子是否有相同维度
        output_dim = faces[0].shape[0]
        input_dim = faces[0].shape[1]
        
        diff = np.zeros((output_dim, input_dim))
        for i, face in enumerate(faces):
            sign = (-1) ** i
            diff += sign * face
        
        return diff
    
    def compute_cohomology(self, degree: int) -> Dict:
        """
        计算指定度数的余调群 H^n(L)
        
        数学定义：H^n(L) = Ker(d_n) / Im(d_{n+1})
        
        实现步骤：
        1. 计算微分算子 d_n 和 d_{n+1}
        2. 计算 Ker(d_n) - d_n 的零空间
        3. 计算 Im(d_{n+1}) - d_{n+1} 的列空间
        4. 计算商空间 Ker(d_n) / Im(d_{n+1})
        """
        # ------------------------------------------------------------
        # 分级约定（不做启发式）：
        # - 由单纯对象经 Dold–Kan 得到的复形是同调分级 C_n (n>=0)；
        # - 余切复形 L_{A/k} 按导出几何惯例置于上同调非正度：
        #       H^{-n}(L_{A/k})  ≅  H_n(C_•)
        # 因此：degree <= 0 时，转为同调度 n = -degree；degree > 0 时应为 0。
        # ------------------------------------------------------------
        if degree > 0:
            return {
                "dim": 0,
                "representatives": [],
                "basis": np.zeros((0, 0), dtype=self.la_cfg.dtype),
                "matrix": None,
                "kernel_dim": 0,
                "image_dim": 0,
                "kernel_basis": np.zeros((0, 0), dtype=self.la_cfg.dtype),
                "image_basis": np.zeros((0, 0), dtype=self.la_cfg.dtype),
            }

        homological_degree = int(-degree)  # degree<=0

        if homological_degree >= len(self.differentials):
            return {
                "dim": 0,
                "representatives": [],
                "basis": np.zeros((0, 0), dtype=self.la_cfg.dtype),
                "matrix": None,
                "kernel_dim": 0,
                "image_dim": 0,
                "kernel_basis": np.zeros((0, 0), dtype=self.la_cfg.dtype),
                "image_basis": np.zeros((0, 0), dtype=self.la_cfg.dtype),
            }

        d_current = self.differentials[homological_degree]  # d_n: L_n -> L_{n-1}; for n=0 it maps to 0
        d_next = self.differentials[homological_degree + 1] if (homological_degree + 1) < len(self.differentials) else None

        _require_floating_matrix(d_current, context=f"cotangent.differentials[{degree}]")
        if d_next is not None:
            _require_floating_matrix(d_next, context=f"cotangent.differentials[{degree + 1}]")

        dim = int(d_current.shape[1])
        if dim == 0:
            return {"dim": 0, "representatives": [], "basis": np.zeros((0, 0), dtype=self.la_cfg.dtype), "matrix": d_current}
        
        # 步骤1: 计算核空间 Ker(d_n)
        if d_current.size == 0 or d_current.shape[0] == 0:
            kernel_basis = np.eye(dim, dtype=self.la_cfg.dtype)
            kernel_dim = dim
        else:
            kernel_basis, kernel_dim = self._compute_kernel_basis(d_current)

        # 步骤2: 计算像空间 Im(d_{n+1})
        if d_next is not None and d_next.size and d_next.shape[0] and d_next.shape[1]:
            image_basis, image_dim = self._compute_image_basis(d_next)
        else:
            image_basis = np.zeros((dim, 0), dtype=self.la_cfg.dtype)
            image_dim = 0

        # 链复形条件检查：d_n ∘ d_{n+1} = 0
        if d_next is not None and d_current.size and d_next.size:
            comp = d_current @ d_next
            tol = self.la_cfg.composition_zero_tol(d_current, d_next)
            comp_norm = float(np.linalg.norm(comp, ord=2)) if comp.size else 0.0
            if comp_norm > tol:
                raise ChainComplexValidationError(
                    f"Cotangent complex is not a chain complex at degree {degree}: "
                    f"||d_n d_{n+1}||_2={comp_norm}, tol={tol}."
                )
        
        # 步骤3: 计算商空间 Ker(d_n) / Im(d_{n+1})
        cohomology_basis, cohomology_dim = self._compute_quotient_space(kernel_basis, image_basis)
        
        # 步骤4: 验证结果（严格：若失败直接抛异常）
        self._validate_cohomology(cohomology_basis, d_current, d_next, image_basis=image_basis)
        
        # 步骤5: 生成代表元
        representatives = []
        for i in range(cohomology_basis.shape[1]):
            representatives.append(cohomology_basis[:, i])
        
        return {
            "dim": cohomology_dim,
            "representatives": representatives,
            "basis": cohomology_basis,
            "matrix": d_current,
            "kernel_dim": kernel_dim,
            "image_dim": image_dim,
            "kernel_basis": kernel_basis,
            "image_basis": image_basis
        }
    
    def _get_differential(self, n: int) -> Optional[np.ndarray]:
        """获取第n个微分算子"""
        if 0 <= n < len(self.differentials):
            return self.differentials[n]
        return None
    
    def _compute_kernel_basis(self, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        计算矩阵的核空间基
        
        使用SVD分解：A = U Σ V^T
        Ker(A) = V中对应奇异值为0的列
        """
        _require_floating_matrix(matrix, context="cotangent_kernel_basis")
        if matrix.size == 0:
            dim = int(matrix.shape[1])
            return np.eye(dim, dtype=self.la_cfg.dtype), dim
        return _svd_kernel_basis(matrix, self.la_cfg)
    
    def _compute_image_basis(self, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        计算矩阵的像空间基
        
        使用SVD分解：A = U Σ V^T
        Im(A) = U中对应奇异值>0的列
        """
        _require_floating_matrix(matrix, context="cotangent_image_basis")
        if matrix.size == 0:
            return np.zeros((matrix.shape[0], 0), dtype=self.la_cfg.dtype), 0
        return _svd_image_basis(matrix, self.la_cfg)
    
    def _compute_quotient_space(self, kernel_basis: np.ndarray, 
                               image_basis: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        计算商空间 Ker(d_n) / Im(d_{n+1})
        
        数学实现：找到 Ker 中与 Im 正交的补空间
        """
        if kernel_basis.shape[1] == 0:
            return np.zeros((kernel_basis.shape[0], 0), dtype=self.la_cfg.dtype), 0

        qk = _qr_orthonormal_columns(kernel_basis, self.la_cfg)
        if image_basis.shape[1] == 0:
            return qk, qk.shape[1]

        qi = _qr_orthonormal_columns(image_basis, self.la_cfg)
        if qi.shape[1] == 0:
            return qk, qk.shape[1]

        # 数学上要求 Im ⊆ Ker；用正交投影检查。
        proj_i_onto_k = qk @ (qk.T @ image_basis) if qk.shape[1] else np.zeros_like(image_basis)
        residual = image_basis - proj_i_onto_k
        residual_norm = float(np.linalg.norm(residual, ord=2)) if residual.size else 0.0
        image_norm = float(np.linalg.norm(image_basis, ord=2)) if image_basis.size else 0.0
        tol = self.la_cfg.eps * float(max(image_basis.shape)) * image_norm
        if residual_norm > tol:
            raise ChainComplexValidationError(
                "Cannot form quotient Ker/Im: detected Im not contained in Ker "
                f"(||residual||_2={residual_norm}, tol={tol})."
            )

        qk_perp = qk - qi @ (qi.T @ qk)
        if qk_perp.size == 0 or qk_perp.shape[1] == 0:
            return np.zeros((qk.shape[0], 0), dtype=self.la_cfg.dtype), 0

        # 同上：避免把“理论应为 0 的消去残差”误判为非零方向。
        col_norms = np.sqrt(np.sum(qk_perp * qk_perp, axis=0))
        scale = max(
            float(np.linalg.norm(qk, ord=2)) if qk.size else 0.0,
            float(np.linalg.norm(qi, ord=2)) if qi.size else 0.0,
        )
        tol = self.la_cfg.eps * float(max(qk_perp.shape)) * scale
        keep = col_norms > tol
        qk_perp = qk_perp[:, keep]
        if qk_perp.shape[1] == 0:
            return np.zeros((qk.shape[0], 0), dtype=self.la_cfg.dtype), 0

        qq = _qr_orthonormal_columns(qk_perp, self.la_cfg)
        return qq, qq.shape[1]
    
    def _validate_cohomology(
        self,
        cohomology_basis: np.ndarray,
        d_current: np.ndarray,
        d_next: Optional[np.ndarray],
        *,
        image_basis: Optional[np.ndarray] = None,
    ) -> None:
        """验证余调群计算的正确性"""
        if cohomology_basis.shape[1] == 0:
            return
        
        _require_floating_matrix(d_current, context="cotangent_validate_d_current")

        # 1) 余调群代表元必须在 Ker(d_current) 中
        if d_current.size and d_current.shape[0]:
            d_norm = float(np.linalg.norm(d_current, ord=2))
            for i in range(cohomology_basis.shape[1]):
                v = cohomology_basis[:, i]
                dv = d_current @ v
                dv_norm = float(np.linalg.norm(dv, ord=2)) if dv.size else 0.0
                v_norm = float(np.linalg.norm(v, ord=2)) if v.size else 0.0
                tol = self.la_cfg.eps * float(max(d_current.shape)) * d_norm * v_norm
                if dv_norm > tol:
                    raise ChainComplexValidationError(
                        f"Cohomology basis vector {i} is not in Ker(d): ||dv||_2={dv_norm}, tol={tol}."
                    )

        # 2) 代表元应与 Im(d_next) 正交（我们构造时即如此），否则说明商空间构造失败。
        if image_basis is not None and image_basis.size and image_basis.shape[1]:
            qi = _qr_orthonormal_columns(image_basis, self.la_cfg)
            for i in range(cohomology_basis.shape[1]):
                v = cohomology_basis[:, i]
                overlap = qi.T @ v
                overlap_norm = float(np.linalg.norm(overlap, ord=2)) if overlap.size else 0.0
                v_norm = float(np.linalg.norm(v, ord=2)) if v.size else 0.0
                tol = self.la_cfg.eps * float(max(qi.shape[0], qi.shape[1])) * v_norm
                if overlap_norm > tol:
                    raise ChainComplexValidationError(
                        f"Cohomology basis vector {i} has nontrivial component in Im(d_next): "
                        f"||proj||_2={overlap_norm}, tol={tol}."
                    )


# ==============================================
# 核心二：Voronoi细胞体积权重的完整计算
# ==============================================

class VoronoiIntegrationWeights:
    """
    计算 Voronoi 单元体积权重（严格、无蒙特卡洛/无启发式阈值）：

    数学定义：
    - Voronoi 单元 V_i = {x | ||x-p_i|| <= ||x-p_j||, ∀j}
    - 为了得到有限体积，默认在有界域 Ω = Conv(points) 内裁剪：C_i = V_i ∩ Ω
    - 权重 w_i = Vol(C_i) / Σ_k Vol(C_k)

    实现采用半空间交：
    - V_i 由所有垂直平分超平面的半空间交给出；
    - Ω 由凸包的半空间表示给出；
    - C_i 的顶点由 `scipy.spatial.HalfspaceIntersection` 计算，体积由 `ConvexHull(...).volume` 给出。
    """

    def __init__(
        self,
        points: np.ndarray,
        dimension: int,
        *,
        domain_halfspaces: Optional[np.ndarray] = None,
    ):
        points = np.asarray(points)
        if points.ndim != 2:
            raise ValueError("points must be a 2D array of shape (n_points, dimension).")
        if points.shape[1] != dimension:
            raise ValueError(f"points.shape[1]={points.shape[1]} does not match dimension={dimension}.")
        _require_floating_matrix(points, context="VoronoiIntegrationWeights.points")

        self.points = points
        self.dim = int(dimension)
        self.n_points = int(points.shape[0])

        if self.n_points < (self.dim + 1):
            raise ValueError(
                f"Need at least dim+1 points for a full-dimensional convex hull (dim={self.dim})."
            )

        unique = np.unique(points, axis=0)
        if unique.shape[0] != self.n_points:
            raise ValueError("Duplicate points detected; Voronoi cells are not well-defined.")

        self.domain_halfspaces = (
            np.asarray(domain_halfspaces)
            if domain_halfspaces is not None
            else self._convex_hull_halfspaces(points)
        )
        if self.domain_halfspaces.ndim != 2 or self.domain_halfspaces.shape[1] != (self.dim + 1):
            raise ValueError(
                "domain_halfspaces must have shape (n_halfspaces, dim+1) encoding a·x + b <= 0."
            )

    def _convex_hull_halfspaces(self, points: np.ndarray) -> np.ndarray:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        if hull.volume <= 0:
            raise ValueError("Convex hull volume is zero; points are not full-dimensional.")
        return np.asarray(hull.equations, dtype=points.dtype)

    def _find_strictly_feasible_point(self, halfspaces: np.ndarray) -> np.ndarray:
        """
        HalfspaceIntersection 需要严格可行点 x0（满足 a·x0 + b < 0）。
        我们通过线性规划最大化最小松弛量来找这样的点。
        """
        from scipy.optimize import linprog

        a = halfspaces[:, : self.dim]
        b = halfspaces[:, self.dim]

        # 变量为 (x, t)，目标最大化 t（等价于最小化 -t）
        a_ub = np.hstack([a, np.ones((a.shape[0], 1), dtype=a.dtype)])
        b_ub = -b
        c = np.zeros(self.dim + 1, dtype=a.dtype)
        c[-1] = -1

        bounds = [(None, None)] * self.dim + [(0, None)]
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds)
        if not res.success:
            raise LinearAlgebraError(f"Failed to find an interior point: {res.message}")
        if res.x[-1] <= 0:
            raise LinearAlgebraError("No strictly feasible point exists for the given halfspaces.")
        return res.x[: self.dim]

    def _cell_volume(self, i: int, *, domain_halfspaces: np.ndarray, norm_sq: np.ndarray) -> float:
        from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

        p_i = self.points[i]
        diffs = self.points - p_i
        # 不等式： (p_j - p_i)·x <= (||p_j||^2 - ||p_i||^2)/2
        a = diffs
        b = (norm_sq - norm_sq[i]) / 2

        # 去掉 j=i 的平凡约束（a=0,b=0）
        mask = np.ones(self.n_points, dtype=bool)
        mask[i] = False
        a = a[mask]
        b = b[mask]

        vor_halfspaces = np.hstack([a, (-b).reshape(-1, 1)])
        halfspaces = np.vstack([domain_halfspaces, vor_halfspaces])

        x0 = self._find_strictly_feasible_point(halfspaces)
        try:
            hs = HalfspaceIntersection(halfspaces, x0)
        except QhullError as e:
            raise LinearAlgebraError(f"Halfspace intersection failed for cell {i}.") from e

        vertices = hs.intersections
        if vertices.size == 0 or vertices.shape[0] < (self.dim + 1):
            raise LinearAlgebraError(f"Degenerate Voronoi cell {i}: insufficient vertices for volume.")

        try:
            hull = ConvexHull(vertices)
        except QhullError as e:
            raise LinearAlgebraError(f"ConvexHull failed while computing volume for cell {i}.") from e
        return float(hull.volume)

    def compute_weights(self) -> np.ndarray:
        """
        返回 w_i = Vol(V_i ∩ Conv(points)) / Σ Vol(V_k ∩ Conv(points))
        """
        norm_sq = np.einsum("ij,ij->i", self.points, self.points)
        volumes = np.zeros(self.n_points, dtype=self.points.dtype)
        for i in range(self.n_points):
            volumes[i] = self._cell_volume(i, domain_halfspaces=self.domain_halfspaces, norm_sq=norm_sq)

        total = float(np.sum(volumes))
        if total <= 0:
            raise ValueError("Total clipped Voronoi volume is non-positive; cannot normalize weights.")
        return volumes / total

# ==============================================
# 完整实现的VirtualCycleIntegrator
# ==============================================

class CompleteVirtualCycleIntegrator:
    """使用完整数学实现的虚拟循环积分器"""
    
    def __init__(self, cotangent_complex: List[np.ndarray], 
                 face_operators: List[np.ndarray],
                 degeneracy_operators: List[np.ndarray]):
        
        # 核心1：余调群计算
        self.cohomology_calculator = CotangentComplexCohomology(face_operators, degeneracy_operators)
        
        # 存储原始数据
        self.cotangent_complex = cotangent_complex
        self.face_ops = face_operators
        self.degen_ops = degeneracy_operators
        
    def compute_virtual_cycle_cohomology(self, degree: int) -> Dict:
        """计算虚拟循环的完整余调群"""
        return self.cohomology_calculator.compute_cohomology(degree)
    
    def integrate_on_virtual_cycle(self, points: np.ndarray, 
                                   differential_form: np.ndarray,
                                   symplectic_form: np.ndarray) -> Dict:
        """
        在虚拟循环上积分微分形式
        
        使用完整的Voronoi权重计算
        """
        if len(points) == 0:
            return {
                "integral_value": np.zeros_like(differential_form),
                "weights": np.array([]),
                "volumes": np.array([]),
                "method": "empty"
            }
        
        # 计算Voronoi权重
        voronoi_calculator = VoronoiIntegrationWeights(points, points.shape[1])
        weights = voronoi_calculator.compute_weights()
        
        # 计算辛体积形式
        symplectic_volumes = self._compute_symplectic_volumes(points, symplectic_form)
        
        # 结合权重计算积分
        integral_value = np.zeros_like(differential_form)
        total_weighted_volume = 0.0
        
        for i, point in enumerate(points):
            weight = weights[i]
            symplectic_vol = symplectic_volumes[i] if i < len(symplectic_volumes) else 1.0
            
            # 微分形式在点处的值
            form_value = differential_form @ point
            
            # 加权积分
            weighted_value = weight * symplectic_vol * form_value * point
            
            integral_value += weighted_value
            total_weighted_volume += weight * symplectic_vol
        
        # 归一化
        if total_weighted_volume > 0:
            integral_value /= total_weighted_volume
        
        return {
            "integral_value": integral_value,
            "weights": weights,
            "symplectic_volumes": symplectic_volumes,
            "total_weighted_volume": total_weighted_volume,
            "method": "voronoi_symplectic"
        }
    
    def _compute_symplectic_volumes(self, points: np.ndarray, 
                                   symplectic_form: np.ndarray) -> np.ndarray:
        """
        计算每个点的辛体积元素
        
        对于辛流形上的积分，体积元素由辛形式给出：
        ω^n / n!，其中n是辛空间的维度的一半
        """
        if symplectic_form.ndim != 2 or symplectic_form.shape[0] != symplectic_form.shape[1]:
            raise ValueError("symplectic_form must be a square matrix.")

        # 对反对称矩阵 ω，有 Pf(ω)^2 = det(ω)。用于体积元素时只需要 √|det(ω)|
        det = np.linalg.det(symplectic_form)
        volume_element = float(np.sqrt(abs(det)))

        volumes = np.ones(len(points), dtype=float) * volume_element
        
        # 归一化
        if np.sum(volumes) > 0:
            volumes = volumes / np.sum(volumes)
        
        return volumes
    
   
    def complete_virtual_integration(self, 
                                   points: np.ndarray,
                                   differential_form: np.ndarray,
                                   symplectic_form: np.ndarray,
                                   constraints: List[Callable]) -> Dict:
        """
        完整的虚拟循环积分流程
        
        包括：
        1. 余调群计算
        2. Voronoi权重计算
        3. 辛体积计算
        4. 约束投影
        5. 积分
        """
        
        # 步骤1: 计算H^{-2}和H^{-1}
        cohomology_minus2 = self.compute_virtual_cycle_cohomology(-2)
        cohomology_minus1 = self.compute_virtual_cycle_cohomology(-1)
        
        print(f"H^{-2}(阻碍空间) 维数: {cohomology_minus2['dim']}")
        print(f"H^{-1}(形变空间) 维数: {cohomology_minus1['dim']}")
        print(f"H^0(切空间) 维数: {self.compute_virtual_cycle_cohomology(0)['dim']}")
        
        # 检查虚拟循环存在条件
        if cohomology_minus2['dim'] != 0:
            raise ValueError(f"阻碍空间非零: dim(H^{-2}) = {cohomology_minus2['dim']}，无法构造虚拟循环")
        
        if cohomology_minus1['dim'] == 0:
            raise ValueError(f"形变空间为零: dim(H^{-1}) = 0，无法构造虚拟循环")
        
        # 步骤2: 计算Voronoi权重
        print(f"计算 {len(points)} 个点的Voronoi权重...")
        integration_result = self.integrate_on_virtual_cycle(points, differential_form, symplectic_form)
        
        # 步骤3: 约束投影
        projected_points = []
        for point in points:
            projected = self._project_to_constraints(point, constraints)
            projected_points.append(projected)
        
        projected_points = np.array(projected_points)
        
        # 步骤4: 在投影点上再次积分
        final_integral = self.integrate_on_virtual_cycle(projected_points, differential_form, symplectic_form)
        
        return {
            "cohomology": {
                "H^{-2}": cohomology_minus2,
                "H^{-1}": cohomology_minus1,
                "H^0": self.compute_virtual_cycle_cohomology(0)
            },
            "voronoi_weights": integration_result["weights"],
            "symplectic_volumes": integration_result["symplectic_volumes"],
            "initial_integral": integration_result["integral_value"],
            "final_integral": final_integral["integral_value"],
            "projected_points": projected_points,
            "constraint_norms": [np.linalg.norm(c(p)) for p, c in zip(projected_points, [constraints[0]]*len(projected_points))]
        }
    
    def _project_to_constraints(self, point: np.ndarray, constraints: List[Callable]) -> np.ndarray:
        """将点投影到约束流形上"""
        # 使用约束优化进行投影
        from scipy.optimize import minimize
        
        def constraint_loss(x):
            total_loss = 0.0
            for constraint in constraints:
                c_val = constraint(x)
                if isinstance(c_val, np.ndarray):
                    total_loss += np.sum(c_val**2)
                else:
                    total_loss += c_val**2
            return total_loss
        
        # 最小化约束违反
        result = minimize(constraint_loss, point, method="BFGS")
        if not result.success:
            raise LinearAlgebraError(f"Constraint projection failed: {result.message}")
        return result.x


# ═══════════════════════════════════════════════════════════════════════════
# Kähler 诱导映射 (Merged from kahler_induced_map.py)
# ═══════════════════════════════════════════════════════════════════════════

# 本模块允许 point 使用 int/Fraction/float；导数与代数结构仍精确（Fraction），
# 仅在在点上取值时可能落到 float（如果 point 给的是 float）。
Scalar = Union[int, Fraction, float]


def _as_fraction(x: Union[int, Fraction]) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    raise TypeError(f"Expected int|Fraction, got {type(x).__name__}.")


def _zero_point(num_vars: int) -> Tuple[Fraction, ...]:
    return tuple(Fraction(0, 1) for _ in range(int(num_vars)))


def _require_num_vars(num_vars: int) -> int:
    if not isinstance(num_vars, int) or num_vars < 0:
        raise ValueError(f"num_vars must be a non-negative int, got {num_vars!r}.")
    return num_vars


def _require_exponents(exponents: Tuple[int, ...], num_vars: int) -> None:
    if len(exponents) != num_vars:
        raise ValueError(f"Exponent length {len(exponents)} != num_vars {num_vars}.")
    if any((not isinstance(e, int) or e < 0) for e in exponents):
        raise ValueError(f"Exponents must be non-negative ints, got {exponents!r}.")


class Poly:
    """
    稀疏多项式：Σ c_m x^m，m 用指数向量（tuple）表示，系数用 Fraction 精确表示。
    """

    __slots__ = ("_num_vars", "_terms")

    def __init__(self, num_vars: int, terms: Optional[Mapping[Tuple[int, ...], Fraction]] = None):
        self._num_vars = _require_num_vars(num_vars)
        clean: Dict[Tuple[int, ...], Fraction] = {}
        if terms:
            for exps, coeff in terms.items():
                _require_exponents(tuple(exps), self._num_vars)
                if not isinstance(coeff, Fraction):
                    raise TypeError("Poly coefficients must be Fraction for exactness.")
                if coeff != 0:
                    clean[tuple(exps)] = coeff
        self._terms = clean

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def terms(self) -> Dict[Tuple[int, ...], Fraction]:
        return dict(self._terms)

    def is_zero(self) -> bool:
        return not self._terms

    @staticmethod
    def zero(num_vars: int) -> "Poly":
        return Poly(_require_num_vars(num_vars), {})

    @staticmethod
    def one(num_vars: int) -> "Poly":
        n = _require_num_vars(num_vars)
        return Poly(n, {(0,) * n: Fraction(1, 1)})

    @staticmethod
    def const(num_vars: int, value: Union[int, Fraction]) -> "Poly":
        n = _require_num_vars(num_vars)
        c = _as_fraction(value)
        if c == 0:
            return Poly.zero(n)
        return Poly(n, {(0,) * n: c})

    @staticmethod
    def var(var_idx: int, num_vars: int) -> "Poly":
        n = _require_num_vars(num_vars)
        if not isinstance(var_idx, int) or not (0 <= var_idx < n):
            raise ValueError(f"var_idx out of range: {var_idx} for num_vars={n}.")
        exps = tuple(1 if i == var_idx else 0 for i in range(n))
        return Poly(n, {exps: Fraction(1, 1)})

    def __add__(self, other: "Poly") -> "Poly":
        if self.num_vars != other.num_vars:
            raise ValueError("Poly num_vars mismatch in addition.")
        out: Dict[Tuple[int, ...], Fraction] = dict(self._terms)
        for exps, c in other._terms.items():
            out[exps] = out.get(exps, Fraction(0, 1)) + c
            if out[exps] == 0:
                del out[exps]
        return Poly(self.num_vars, out)

    def __sub__(self, other: "Poly") -> "Poly":
        return self + (-other)

    def __neg__(self) -> "Poly":
        return Poly(self.num_vars, {e: -c for e, c in self._terms.items()})

    def __mul__(self, other: "Poly") -> "Poly":
        if self.num_vars != other.num_vars:
            raise ValueError("Poly num_vars mismatch in multiplication.")
        if self.is_zero() or other.is_zero():
            return Poly.zero(self.num_vars)
        out: Dict[Tuple[int, ...], Fraction] = {}
        for e1, c1 in self._terms.items():
            for e2, c2 in other._terms.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                out[e] = out.get(e, Fraction(0, 1)) + (c1 * c2)
                if out[e] == 0:
                    del out[e]
        return Poly(self.num_vars, out)

    def __rmul__(self, scalar: Union[int, Fraction]) -> "Poly":
        c = _as_fraction(scalar)
        if c == 0 or self.is_zero():
            return Poly.zero(self.num_vars)
        return Poly(self.num_vars, {e: c * a for e, a in self._terms.items()})

    def pow(self, exponent: int) -> "Poly":
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Poly.pow exponent must be a non-negative int.")
        if exponent == 0:
            return Poly.one(self.num_vars)
        result = Poly.one(self.num_vars)
        base = self
        e = exponent
        while e:
            if e & 1:
                result = result * base
            e >>= 1
            if e:
                base = base * base
        return result

    def partial(self, var_idx: int) -> "Poly":
        if not isinstance(var_idx, int) or not (0 <= var_idx < self.num_vars):
            raise ValueError(f"var_idx out of range: {var_idx} for num_vars={self.num_vars}.")
        out: Dict[Tuple[int, ...], Fraction] = {}
        for exps, c in self._terms.items():
            e = exps[var_idx]
            if e == 0:
                continue
            new_exps = list(exps)
            new_exps[var_idx] = e - 1
            new_exps_t = tuple(new_exps)
            out[new_exps_t] = out.get(new_exps_t, Fraction(0, 1)) + c * e
            if out[new_exps_t] == 0:
                del out[new_exps_t]
        return Poly(self.num_vars, out)

    def eval(self, point: Sequence[Scalar]) -> Scalar:
        if len(point) != self.num_vars:
            raise ValueError(f"Point length {len(point)} != num_vars {self.num_vars}.")
        if self.is_zero():
            return Fraction(0, 1)

        # 尽可能保持精确：若 point 全是 int/Fraction，则返回 Fraction；否则返回 float。
        all_exact = all(isinstance(v, (int, Fraction)) for v in point)
        if all_exact:
            pt = tuple(_as_fraction(int(v) if isinstance(v, int) else v) for v in point)  # type: ignore[arg-type]
            acc = Fraction(0, 1)
            for exps, c in self._terms.items():
                term = c
                for v, e in zip(pt, exps):
                    if e:
                        term *= v ** e
                acc += term
            return acc

        # 允许 float：用于在点上取纤维的数值化线性代数
        ptf = [float(v) for v in point]
        accf = 0.0
        for exps, c in self._terms.items():
            termf = float(c)
            for v, e in zip(ptf, exps):
                if e:
                    termf *= v ** e
            accf += termf
        return accf

    def substitute(self, images: Sequence["Poly"]) -> "Poly":
        """
        将当前多项式中的变量 y_j 替换为 images[j]（多项式），得到新的多项式。
        """
        if len(images) != self.num_vars:
            raise ValueError("Substitution images length mismatch.")
        if any(img.num_vars != images[0].num_vars for img in images):
            raise ValueError("All substitution polynomials must share the same num_vars.")
        tgt_vars = images[0].num_vars if images else 0
        out = Poly.zero(tgt_vars)
        for exps, c in self._terms.items():
            term = Poly.one(tgt_vars)
            for j, e in enumerate(exps):
                if e:
                    term = term * images[j].pow(e)
            out = out + (c * term)
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Poly) and self.num_vars == other.num_vars and self._terms == other._terms

    def __repr__(self) -> str:
        if self.is_zero():
            return "0"
        # 按总次数降序，指数词典序升序输出
        items = sorted(self._terms.items(), key=lambda kv: (-sum(kv[0]), kv[0]))
        parts: List[str] = []
        for exps, c in items:
            if c == 0:
                continue
            coeff_str = "" if c == 1 and any(e > 0 for e in exps) else ("-" if c == -1 and any(e > 0 for e in exps) else str(c))
            mon_parts: List[str] = []
            for i, e in enumerate(exps):
                if e == 0:
                    continue
                if e == 1:
                    mon_parts.append(f"x{i}")
                else:
                    mon_parts.append(f"x{i}^^{e}")
            mon_str = "".join(mon_parts) if mon_parts else "1"
            if mon_parts:
                if coeff_str in ("", "-"):
                    parts.append(f"{coeff_str}{mon_str}" if coeff_str else mon_str)
                else:
                    parts.append(f"{coeff_str}*{mon_str}")
            else:
                parts.append(coeff_str if coeff_str else "1")
        return " + ".join(parts).replace("+ -", "- ")


@dataclass(frozen=True)
class RingHomomorphism:
    """
    环同态 φ: k[x_0..x_{m-1}] → k[y_0..y_{n-1}]，由 x_i 的像给出。
    """

    source_num_vars: int
    target_num_vars: int
    images: Tuple[Poly, ...]  # len = source_num_vars, each in target vars

    def __post_init__(self) -> None:
        if len(self.images) != self.source_num_vars:
            raise ValueError("RingHomomorphism images length mismatch.")
        for p in self.images:
            if p.num_vars != self.target_num_vars:
                raise ValueError("RingHomomorphism image num_vars mismatch with target.")

    def apply(self, f: Poly) -> Poly:
        if f.num_vars != self.source_num_vars:
            raise ValueError("Poly num_vars mismatch with homomorphism source.")
        return f.substitute(self.images)

    def compose(self, after: "RingHomomorphism") -> "RingHomomorphism":
        """
        复合：after ∘ self。
        self: A→B, after: B→C，则返回 A→C。
        """
        if after.source_num_vars != self.target_num_vars:
            raise ValueError("Homomorphism composition var-count mismatch.")
        new_images = tuple(after.apply(p) for p in self.images)
        return RingHomomorphism(
            source_num_vars=self.source_num_vars,
            target_num_vars=after.target_num_vars,
            images=new_images,
        )

    def jacobian_at(self, point: Optional[Sequence[Scalar]] = None) -> np.ndarray:
        """
        返回 J[j,i] = ∂(φ(x_i))/∂y_j 在 point 处的取值（float 矩阵）。
        shape = (target_num_vars, source_num_vars)
        """
        pt = list(point) if point is not None else list(_zero_point(self.target_num_vars))
        if len(pt) != self.target_num_vars:
            raise ValueError("Jacobian evaluation point length mismatch with target vars.")
        J = np.zeros((self.target_num_vars, self.source_num_vars), dtype=np.float64)
        for i, img in enumerate(self.images):
            for j in range(self.target_num_vars):
                v = img.partial(j).eval(pt)
                J[j, i] = float(v)
        return J


class SimplicialPolyRing:
    """
    单纯多项式代数：每层 A_n = k[x_0..x_{d_n-1}]，面算子由 RingHomomorphism 给出。
    """

    def __init__(self, level_dims: Sequence[int]):
        self._level_dims = [int(d) for d in level_dims]
        if not self._level_dims:
            raise ValueError("SimplicialPolyRing requires at least one level.")
        if any(d < 0 for d in self._level_dims):
            raise ValueError("Level dimensions must be non-negative.")
        self._face_maps: Dict[Tuple[int, int], RingHomomorphism] = {}

    def num_levels(self) -> int:
        return len(self._level_dims)

    def level_dim(self, n: int) -> int:
        if not (0 <= n < self.num_levels()):
            raise ValueError(f"Level {n} out of range.")
        return self._level_dims[n]

    def set_face_map(self, n: int, i: int, images: Sequence[Poly]) -> None:
        if not (1 <= n < self.num_levels()):
            raise ValueError(f"Face map source level n={n} out of range.")
        if not (0 <= i <= n):
            raise ValueError(f"Face index i={i} out of range for level n={n}.")
        src = self.level_dim(n)
        tgt = self.level_dim(n - 1)
        hom = RingHomomorphism(src, tgt, tuple(images))
        self._face_maps[(n, i)] = hom

    def face_map(self, n: int, i: int) -> RingHomomorphism:
        hom = self._face_maps.get((n, i))
        if hom is None:
            raise KeyError(f"Face map (n={n}, i={i}) not set.")
        return hom

    def validate_simplicial_identities(self) -> None:
        """
        严格验证 ∂_i ∂_j = ∂_{j-1} ∂_i (i<j) 在生成元上的一致性。
        """
        for n in range(2, self.num_levels()):
            for i in range(n):
                for j in range(i + 1, n + 1):
                    d_n_j = self.face_map(n, j)
                    d_n_i = self.face_map(n, i)
                    d_n1_i = self.face_map(n - 1, i)
                    d_n1_j1 = self.face_map(n - 1, j - 1)

                    left = d_n_j.compose(d_n1_i)    # (n-1)->(n-2) ∘ (n)->(n-1)
                    right = d_n_i.compose(d_n1_j1)  # (n-1)->(n-2) ∘ (n)->(n-1)

                    # 检查每个生成元 x_k
                    for k in range(self.level_dim(n)):
                        xk = Poly.var(k, self.level_dim(n))
                        if left.apply(xk) != right.apply(xk):
                            raise ValueError(
                                f"Simplicial identity failed at n={n}, i={i}, j={j}, k={k}: "
                                f"∂_{i}∂_{j} != ∂_{j-1}∂_{i}"
                            )


class SimplicialKahlerMatrices:
    """
    单纯 Kähler 微分模 Ω¹_{A_•/k} 的面算子矩阵：∂_i*: Ω¹_{A_n} → Ω¹_{A_{n-1}}。
    """

    def __init__(self, ring: SimplicialPolyRing, *, points: Optional[Mapping[int, Sequence[Scalar]]] = None):
        self.ring = ring
        self.points: Dict[int, Sequence[Scalar]] = dict(points) if points is not None else {}

    def induced_face_matrix(self, n: int, i: int) -> np.ndarray:
        hom = self.ring.face_map(n, i)
        pt = self.points.get(n - 1)
        return hom.jacobian_at(pt)

    def face_operators(self) -> List[List[np.ndarray]]:
        """
        返回 face_ops[n][i] = ∂_i* 的矩阵（float），并给出 n=0 的占位矩阵用于维度推断。
        """
        face_ops: List[List[np.ndarray]] = []
        dim0 = self.ring.level_dim(0)
        face_ops.append([np.zeros((0, dim0), dtype=np.float64)])
        for n in range(1, self.ring.num_levels()):
            ops_at_level: List[np.ndarray] = []
            for i in range(n + 1):
                ops_at_level.append(self.induced_face_matrix(n, i))
            face_ops.append(ops_at_level)
        return face_ops


def compute_cotangent_cohomology(
    simplicial_ring: SimplicialPolyRing,
    *,
    points: Optional[Mapping[int, Sequence[Scalar]]] = None,
    max_degree: Optional[int] = None,
) -> Dict[int, Dict]:
    """
    便捷入口：构造 ∂_i* 的矩阵后，用 `CotangentComplexCohomology` 计算 H^{degree}。

    返回：degree -> compute_cohomology(degree) 的原始字典。
    """
    # CotangentComplexCohomology is defined in this file

    mats = SimplicialKahlerMatrices(simplicial_ring, points=points).face_operators()
    # 入口只需要面算子；退化算子在当前同调计算中不参与（维度推断也由 face_ops[0] 给定）。
    cc = CotangentComplexCohomology(mats, [])

    if max_degree is None:
        max_degree = -(simplicial_ring.num_levels() - 1)  # 最低到 H^{-(N-1)}

    result: Dict[int, Dict] = {}
    # 计算 degree ∈ [max_degree, 0]（max_degree 为负或 0）
    for deg in range(int(max_degree), 1):
        result[deg] = cc.compute_cohomology(deg)
    return result


def analyze_derived_intersection(
    simplicial_ring: SimplicialPolyRing,
    *,
    points: Optional[Mapping[int, Sequence[Scalar]]] = None,
) -> Dict[str, object]:
    """
    端到端分析封装（不做任何步长/阈值启发式）：
    - 计算 H^0, H^{-1}, H^{-2} 的维数
    - 给出穿墙判定：H^{-2}=0 且 H^{-1}≠0
    - 返回 H^{-1} 的代表元（作为无穷小形变方向）
    """
    cohom = compute_cotangent_cohomology(simplicial_ring, points=points, max_degree=-2)
    h0 = int(cohom.get(0, {}).get("dim", 0))
    hm1_block = cohom.get(-1, {})
    hm2 = int(cohom.get(-2, {}).get("dim", 0))
    hm1 = int(hm1_block.get("dim", 0))

    directions = list(hm1_block.get("representatives", [])) if isinstance(hm1_block, dict) else []
    can_push = (hm2 == 0 and hm1 > 0)

    attack_vec = None
    if can_push and directions:
        v = np.asarray(directions[0], dtype=float)
        nrm = float(np.linalg.norm(v, ord=2)) if v.size else 0.0
        if nrm == 0.0:
            raise ValueError("Nontrivial H^{-1} reported but first representative has zero norm.")
        attack_vec = v / nrm  # 仅提供无穷小方向（不做 Maurer–Cartan 积分启发式）

    return {
        "h0": h0,
        "h_minus_1": hm1,
        "h_minus_2": hm2,
        "can_push_wall": can_push,
        "deformation_directions": directions,
        "attack_vector_infinitesimal": attack_vec,
        "raw_cohomology": cohom,
    }


def create_standard_simplicial_polyring(level_dims: Sequence[int]) -> SimplicialPolyRing:
    """
    构造“标准”单纯面算子：
      ∂_i(x_j) = x_j      (j < i)
      ∂_i(x_i) = 0
      ∂_i(x_j) = x_{j-1}  (j > i)

    - 这给出的是自由代数上的一个典型单纯结构；
    - 若 level_dims 不是逐层“+1”的形状，仍可构造映射，但不保证满足单纯恒等式；
      调用者应当显式调用 `validate_simplicial_identities()` 进行验证。
    """
    ring = SimplicialPolyRing(level_dims)
    for n in range(1, ring.num_levels()):
        src_dim = ring.level_dim(n)
        tgt_dim = ring.level_dim(n - 1)
        for i in range(n + 1):
            images: List[Poly] = []
            for j in range(src_dim):
                if j < i:
                    images.append(Poly.var(j, tgt_dim) if j < tgt_dim else Poly.zero(tgt_dim))
                elif j == i:
                    images.append(Poly.zero(tgt_dim))
                else:
                    images.append(Poly.var(j - 1, tgt_dim) if (j - 1) < tgt_dim else Poly.zero(tgt_dim))
            ring.set_face_map(n, i, images)
    return ring


def create_evm_simplicial_polyring(*, base_vars: int, constraint_count: int) -> SimplicialPolyRing:
    """
    EVM模板维度骨架：
    - A_0：基础状态变量（base_vars）
    - A_1：一阶约束变量（base_vars + constraint_count）
    - A_2：二阶关系变量（base_vars + 2*constraint_count）
    """
    dims = [int(base_vars), int(base_vars) + int(constraint_count), int(base_vars) + 2 * int(constraint_count)]
    return create_standard_simplicial_polyring(dims)


# ===========================================================================
# MVP19 Syzygy Graph Adapter
# ===========================================================================

class SyzygyKahlerAdapter:
    """
    Adapter to bridge MVP19 SyzygyGraph (multivariate rewrite rules)
    to MVP18 SimplicialPolyRing (for Kahler/Cotangent calculations).
    """

    def __init__(self, graph):
        """
        Args:
            graph: bridge_audit.core.mvp19_syzygy_frobenius_suite.SyzygyGraph
        """
        self.graph = graph
        self.modulus = int(graph.modulus)
        
        # Validate graph properties needed for conversion
        if not graph.basis_monomials:
            raise ValueError("SyzygyGraph has no basis monomials.")
        
        self.num_vars = len(graph.basis_monomials[0])
        self.rules = list(graph.rewrite_rules.items())
        # Sort rules for deterministic order (by pivot grevlex)
        # Helper to sort monomials (same as graph's internal logic ideally)
        def _sort_key(item):
            piv = item[0]
            return (sum(piv), tuple(-x for x in reversed(piv)))
        self.rules.sort(key=_sort_key)

    def to_simplicial_ring(self) -> SimplicialPolyRing:
        """
        Construct a SimplicialPolyRing representing the first step of resolution:
        A_0 = k[x_1...x_n]
        A_1 = k[x_1...x_n, e_1...e_m] (where m = num_rules)
        
        Face maps:
        d_0(e_k) = pivot_k - rhs_k
        d_1(e_k) = 0
        """
        num_relations = len(self.rules)
        
        # Level dimensions:
        # Level 0: A_0 is free algebra on num_vars.
        # Level 1: A_1 has variables for A_0 + variables for relations.
        # But SimplicialPolyRing defines "level_dim" as number of polynomial variables.
        # A_0 = k[x_1...x_n] -> dim = n
        # A_1 = A_0[e_1...e_m] -> dim = n + m
        # (Assuming standard simplicial resolution structure)
        
        # Wait, SimplicialPolyRing expects level_dims to be simply integers.
        # And face maps define homomorphism A_n -> A_{n-1}.
        # A_1 -> A_0 maps n+m variables to n variables.
        # x_i -> x_i (for i < n)
        # e_k -> relation_k (for k)
        
        level_dims = [self.num_vars, self.num_vars + num_relations]
        ring = SimplicialPolyRing(level_dims)
        
        # Define face maps for n=1
        # There are n+1 = 2 face maps: d_0, d_1.
        
        # d_0: The "target" map that imposes relations.
        # x_i -> x_i
        # e_k -> relation_k (pivot - rhs)
        
        d0_images = []
        # 1. Map original vars to themselves
        for i in range(self.num_vars):
            d0_images.append(Poly.var(i, self.num_vars))
            
        # 2. Map relation vars to relations
        for i, (pivot, rhs_dict) in enumerate(self.rules):
            # Construct relation poly: pivot - rhs
            p_pivot = self._monomial_to_poly(pivot)
            p_rhs = self._rhs_to_poly(rhs_dict)
            relation = p_pivot - p_rhs
            d0_images.append(relation)
            
        ring.set_face_map(1, 0, d0_images)
        
        # d_1: The "base" map (normalization or degeneracy target)
        # Usually d_1(x_i) = x_i, d_1(e_k) = 0
        
        d1_images = []
        for i in range(self.num_vars):
            d1_images.append(Poly.var(i, self.num_vars))
        for i in range(num_relations):
            d1_images.append(Poly.zero(self.num_vars))
            
        ring.set_face_map(1, 1, d1_images)
        
        return ring

    def _monomial_to_poly(self, mono: Tuple[int, ...]) -> Poly:
        """Convert a monomial tuple to a Poly."""
        terms = {mono: Fraction(1, 1)}
        return Poly(self.num_vars, terms)

    def _rhs_to_poly(self, rhs: Dict[Tuple[int, ...], int]) -> Poly:
        """Convert RHS dict {monomial: coeff} to Poly."""
        terms = {}
        for m, c in rhs.items():
            # Coeffs in graph are int (mod p).
            # We convert them to Fraction.
            # Note: We represent them as positive integers or handle modulus locally?
            # Poly uses Fraction (field 0). It doesn't enforce modulus.
            # But the relations are valid over Z/pZ.
            # If we compute Jacobian over Q, it might be different.
            # However, for MVP18/19 "Numerical", we usually lift or work with "float" representatives.
            # Using Fraction(c, 1) preserves the integer representative.
            terms[m] = Fraction(c, 1)
        return Poly(self.num_vars, terms)

    @staticmethod
    def build_differential_matrix(graph) -> List[np.ndarray]:
        """
        Static helper to build the d_1 matrix directly from a graph.
        Useful if you don't need the full ring structure.
        """
        adapter = SyzygyKahlerAdapter(graph)
        ring = adapter.to_simplicial_ring()
        mats = SimplicialKahlerMatrices(ring).face_operators()
        # mats[1] contains [d_0, d_1].
        # Differential is d_0 - d_1.
        d0 = mats[1][0]
        d1 = mats[1][1]
        return d0 - d1
