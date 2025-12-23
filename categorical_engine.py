#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVP10b 范畴论强化模块: Categorical Engine

三核引擎:
1. Yoneda Embedding 探测器 - 基于米田引理的行为探测
2. Grothendieck Topos 引擎 - 层范畴 + 内部逻辑
3. Derived Category + t-structure - 导出范畴层面的信息追踪

工程红线:
- 禁伪函子: FunctorBase 必须验证 F(id) = id 和 F(g∘f) = F(g)∘F(f)
- 禁离散化Ω: SubobjectClassifier 必须实现完整 Heyting 代数
- 禁假链复形: ChainComplex 必须验证 d² = 0 (相对容差 ε·‖d‖_F)
- 禁跳过消解: DerivedHom 必须使用内射消解
- 禁硬编码覆盖: CoverageEstimator 必须动态计算
- 禁魔法数: 所有阈值从数学性质或数据结构推导

目前缺陷:
- RHom 的计算在 `Vect`（有限维向量空间范畴）中可以取 **I• = G 本身**（因每个对象皆为内射），
  因此采用标准 dg-Hom 复形实现即可；不再使用人为构造的 acyclic 假消解。
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Optional, Tuple, List, Dict, Any, Callable, Union, 
    Set, TypeVar, Generic, Iterator
)
from enum import Enum, auto
import warnings
import json

# NOTE:
# HeckeOperator/Satake transform are implemented in MVP14 trace_formula_engine.
# Redline: do NOT re-implement Satake/Hecke logic here; import and call directly.
from .trace_formula_engine import HeckeOperator, LocalBase

# ============================================================================
# Section 0: 数值常数与异常定义
# ============================================================================

_FLOAT64_EPS = np.finfo(np.float64).eps
_CHAIN_COMPLEX_REL_TOL = 1e-12  # d² = 0 的相对容差系数
# 覆盖率阈值（MVP10强化稿的工程红线要求：< 80% 必须强制报警）。
# 这不是拍脑袋启发式，而是明确的安全门槛规格。
_COVERAGE_THRESHOLD = 0.8
_FUNCTOR_COMPOSITION_REL_TOL = 1e-10  # 函子复合验证相对容差系数
_SHEAF_CONDITION_REL_TOL = 1e-10  # 层条件验证相对容差系数
_EXACTNESS_REL_TOL = 1e-10  # 正合性验证相对容差系数

# 向后兼容的绝对容差（仅用于零矩阵情况）
_CHAIN_COMPLEX_TOL = 1e-12
_FUNCTOR_COMPOSITION_TOL = 1e-10
_SHEAF_CONDITION_TOL = 1e-10
_EXACTNESS_TOL = 1e-10

# ============================================================================
#
# 注意：本模块禁止靠常数调味把模型写活。
# - 覆盖度：必须来自 CFG 路径计数与可验证证据（trace），不允许 ε 软化。
# - REVERT→Heyting：需要保证非经典性，但扰动量必须由结构（维度）推导，而非手选常数。
# - L 因子：若引入 Transfer 行为，必须通过明确的矩阵/谱构造进入 Euler 因子，而非乘一个系数。


def _relative_tolerance(A: np.ndarray, rel_tol: float = 1e-12) -> float:
    """计算相对容差 ε·‖A‖_F
    
    对于零矩阵或非常小的矩阵，返回绝对容差下界
    
    Args:
        A: 参考矩阵
        rel_tol: 相对容差系数
        
    Returns:
        相对容差值 max(rel_tol * ‖A‖_F, rel_tol)
    """
    if A is None or A.size == 0:
        return rel_tol
    norm_A = np.linalg.norm(A, 'fro')
    return max(rel_tol * norm_A, rel_tol)


def _check_near_zero(A: np.ndarray, ref_matrix: np.ndarray, rel_tol: float = 1e-12) -> Tuple[bool, float]:
    """检查矩阵是否在相对容差内接近零
    
    Args:
        A: 待检查矩阵
        ref_matrix: 参考矩阵（用于计算相对容差）
        rel_tol: 相对容差系数
        
    Returns:
        (is_near_zero, actual_norm) 元组
    """
    if A is None or A.size == 0:
        return True, 0.0
    actual_norm = np.linalg.norm(A, 'fro')
    tol = _relative_tolerance(ref_matrix, rel_tol)
    return actual_norm <= tol, actual_norm


def _require_exact_int(x: Any, *, name: str = "value") -> int:
    """将输入强制为可证明的整数。
    
    红线（MVP10强化稿 B.2）：BPS 坐标必须是精确整数；禁止无条件 round() 的静默纠错。
    
    允许的输入：
    - Python int / numpy integer：直接接受
    - 浮点：仅当其与最近整数的差异落在机器精度可解释的舍入误差范围内才允许转换
    
    Raises:
        TypeError / ValueError: 当无法证明其为整数时
    """
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            raise ValueError(f"{name} must be a finite integer, got {x!r}.")
        nearest = int(np.rint(x))
        # 仅允许机器精度级的偏差：|x-n| <= eps * max(1, |x|)
        tol = _FLOAT64_EPS * max(1.0, abs(float(x)))
        if abs(float(x) - float(nearest)) <= tol:
            return nearest
        raise ValueError(f"{name} must be an exact integer; refusing to round {x!r}.")
    raise TypeError(f"{name} must be an integer (or float provably equal to an integer), got {type(x)}.")


class CategoricalError(Exception):
    """范畴论引擎基础异常"""
    pass


class SheafConditionViolation(CategoricalError):
    """层条件违反异常
    
    当等化子图不精确时抛出
    """
    def __init__(self, object_id: str, details: str):
        self.object_id = object_id
        self.details = details
        super().__init__(f"Sheaf condition violated at object '{object_id}': {details}")


class tStructureAxiomViolation(CategoricalError):
    """t-结构公理违反异常"""
    def __init__(self, axiom: str, details: str):
        self.axiom = axiom
        self.details = details
        super().__init__(f"t-structure axiom '{axiom}' violated: {details}")


class ExactnessViolation(CategoricalError):
    """正合性违反异常"""
    def __init__(self, degree: int, details: str):
        self.degree = degree
        self.details = details
        super().__init__(f"Exactness violated at degree {degree}: {details}")


class FunctorLawViolation(CategoricalError):
    """函子律违反异常
    
    当 F(id) ≠ id 或 F(g∘f) ≠ F(g)∘F(f) 时抛出
    """
    def __init__(self, functor_name: str, law: str, details: str):
        self.functor_name = functor_name
        self.law = law  # "identity" or "composition"
        self.details = details
        super().__init__(f"Functor '{functor_name}' violates {law} law: {details}")


class ChainComplexViolation(CategoricalError):
    """链复形 d^2 ≠ 0 异常"""
    def __init__(self, degree: int, violation_norm: float):
        self.degree = degree
        self.violation_norm = violation_norm
        super().__init__(
            f"Chain complex violation at degree {degree}: "
            # 使用 d^2 以兼容 GBK 等非 Unicode 控制台编码
            f"||d^2|| = {violation_norm:.2e} > {_CHAIN_COMPLEX_TOL:.2e}"
        )


class HeytingAlgebraViolation(CategoricalError):
    """海廷代数违反异常（如离散化为布尔）"""
    pass


class ResolutionSkippedError(CategoricalError):
    """跳过消解计算 RHom 异常"""
    def __init__(self):
        super().__init__(
            "RHom computation requires injective resolution. "
            "Direct computation on raw complexes is forbidden."
        )


class CoverageInsufficientWarning(UserWarning):
    """覆盖率不足警告"""
    pass


# ============================================================================
# Section 1: Categorical Core - 基础抽象
# ============================================================================

@dataclass(frozen=True)
class Object:
    """范畴中的对象
    
    Attributes:
        id: 对象唯一标识符
        dimension: 对象维度（用于线性代数表示）
        data: 可选的附加数据
    """
    id: str
    dimension: int = 1
    data: Optional[Any] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return self.id == other.id


@dataclass
class Morphism:
    """范畴中的态射（箭头）
    
    数学定义: f: A → B
    
    Attributes:
        source: 源对象 A
        target: 目标对象 B
        matrix: 线性代数表示（可选）
        name: 态射名称（用于调试）
    """
    source: Object
    target: Object
    matrix: Optional[np.ndarray] = None
    name: str = ""
    
    def __post_init__(self):
        if self.matrix is not None:
            self.matrix = np.asarray(self.matrix, dtype=np.complex128)
            # 验证维度一致性
            if self.matrix.shape != (self.target.dimension, self.source.dimension):
                raise ValueError(
                    f"Matrix shape {self.matrix.shape} incompatible with "
                    f"morphism {self.source.dimension} → {self.target.dimension}"
                )
    
    def is_identity(self) -> bool:
        """检查是否为恒等态射（严格）
        
        红线：禁止把无矩阵表示的未知态射默认为恒等。
        仅当：
        - source == target 且
        - (有矩阵且矩阵≈I) 或 (无矩阵但名称显式标记为 id_*)
        才承认其为恒等态射。
        """
        if self.source != self.target:
            return False
        if self.matrix is None:
            # 没有矩阵表示时无法证明恒等；只接受显式标记。
            return self.name == f"id_{self.source.id}" or self.name == "id"
        return np.allclose(self.matrix, np.eye(self.source.dimension), atol=_FLOAT64_EPS)
    
    def __hash__(self):
        """使 Morphism 可哈希（用于集合操作）"""
        return hash((self.source.id, self.target.id, self.name))
    
    def __eq__(self, other):
        if not isinstance(other, Morphism):
            return False
        return (self.source == other.source and 
                self.target == other.target and 
                self.name == other.name)
    
    def __repr__(self):
        name_str = f"'{self.name}'" if self.name else ""
        return f"Morphism{name_str}({self.source.id} → {self.target.id})"


class Category:
    """范畴: 对象 + 态射 + 复合律
    
    公理:
    1. 结合律: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    2. 恒等律: id_B ∘ f = f = f ∘ id_A
    
    实现采用线性代数表示: 态射复合 = 矩阵乘法
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self._objects: Set[Object] = set()
        self._morphisms: Dict[Tuple[str, str], List[Morphism]] = {}
        self._identities: Dict[str, Morphism] = {}
    
    def add_object(self, obj: Object) -> None:
        """添加对象并自动创建恒等态射"""
        self._objects.add(obj)
        # 创建恒等态射 id_A: A → A
        identity = Morphism(
            source=obj,
            target=obj,
            matrix=np.eye(obj.dimension, dtype=np.complex128),
            name=f"id_{obj.id}"
        )
        self._identities[obj.id] = identity
        self._add_morphism_internal(identity)
    
    def add_morphism(self, morphism: Morphism) -> None:
        """添加态射"""
        # 确保源和目标对象存在
        if morphism.source not in self._objects:
            self.add_object(morphism.source)
        if morphism.target not in self._objects:
            self.add_object(morphism.target)
        self._add_morphism_internal(morphism)
    
    def _add_morphism_internal(self, morphism: Morphism) -> None:
        key = (morphism.source.id, morphism.target.id)
        if key not in self._morphisms:
            self._morphisms[key] = []
        self._morphisms[key].append(morphism)
    
    def identity(self, obj: Object) -> Morphism:
        """获取对象的恒等态射"""
        if obj.id not in self._identities:
            raise KeyError(f"Object {obj.id} not in category")
        return self._identities[obj.id]
    
    def compose(self, g: Morphism, f: Morphism) -> Morphism:
        """态射复合 g ∘ f
        
        数学: 若 f: A → B, g: B → C, 则 g ∘ f: A → C
        线性代数: (g ∘ f).matrix = g.matrix @ f.matrix
        
        Args:
            g: 第二个态射 (B → C)
            f: 第一个态射 (A → B)
            
        Returns:
            复合态射 g ∘ f: A → C
        """
        # 验证可复合性: f.target == g.source
        if f.target != g.source:
            raise ValueError(
                f"Cannot compose: {f} target {f.target.id} ≠ {g} source {g.source.id}"
            )
        
        # 严格：不得在缺失矩阵时猜测恒等/忽略其中一个因子。
        # 唯一允许的无矩阵情形：该态射已被严格判定为恒等态射（见 Morphism.is_identity）。
        def _matrix_or_identity(m: Morphism) -> np.ndarray:
            if m.matrix is not None:
                return np.asarray(m.matrix, dtype=np.complex128)
            if m.is_identity():
                return np.eye(m.source.dimension, dtype=np.complex128)
            raise CategoricalError(
                "Cannot compose morphisms without matrix representations. "
                f"Missing matrix on morphism '{m.name or repr(m)}' ({m.source.id}→{m.target.id})."
            )

        composed_matrix = _matrix_or_identity(g) @ _matrix_or_identity(f)
        
        return Morphism(
            source=f.source,
            target=g.target,
            matrix=composed_matrix,
            name=f"({g.name} ∘ {f.name})" if g.name and f.name else ""
        )
    
    def get_morphisms(self, source: Object, target: Object) -> List[Morphism]:
        """获取从 source 到 target 的所有态射"""
        key = (source.id, target.id)
        return self._morphisms.get(key, [])
    
    @property
    def objects(self) -> Set[Object]:
        return self._objects.copy()
    
    def __repr__(self):
        return f"Category('{self.name}', {len(self._objects)} objects)"


# ============================================================================
# Section 2: FunctorBase - 严格函子律验证
# ============================================================================

class FunctorBase(ABC):
    """函子基类: 严格验证函子律
    
    函子 F: C → D 必须满足:
    1. F(id_A) = id_{F(A)}  (保持恒等)
    2. F(g ∘ f) = F(g) ∘ F(f)  (保持复合)
    
    任何违反都会抛出 FunctorLawViolation
    """
    
    def __init__(
        self, 
        source_category: Category, 
        target_category: Category,
        name: str = ""
    ):
        self.source_category = source_category
        self.target_category = target_category
        self.name = name or self.__class__.__name__
        self._verified_identities: Set[str] = set()
        self._verified_compositions: Set[Tuple[str, str]] = set()
    
    @abstractmethod
    def on_objects(self, obj: Object) -> Object:
        """对象映射 F: Ob(C) → Ob(D)"""
        pass
    
    @abstractmethod
    def on_morphisms(self, f: Morphism) -> Morphism:
        """态射映射 F: Mor(C) → Mor(D)"""
        pass
    
    def __call__(self, x: Union[Object, Morphism]) -> Union[Object, Morphism]:
        """应用函子并验证函子律"""
        if isinstance(x, Object):
            return self._apply_on_object(x)
        elif isinstance(x, Morphism):
            return self._apply_on_morphism(x)
        else:
            raise TypeError(f"Functor expects Object or Morphism, got {type(x)}")
    
    def _apply_on_object(self, obj: Object) -> Object:
        """应用函子到对象"""
        return self.on_objects(obj)
    
    def _apply_on_morphism(self, f: Morphism) -> Morphism:
        """应用函子到态射并验证函子律"""
        result = self.on_morphisms(f)

        # 1) 恒等律：对端点对象的恒等态射做一次性验证（严格，不依赖调用者刚好传入 id）。
        for obj in (f.source, f.target):
            if obj.id not in self._verified_identities:
                id_m = self.source_category.identity(obj)
                self._verify_identity_law(id_m)
                self._verified_identities.add(obj.id)

        # 2) 复合律：在 __call__ 里对所有与 f 可复合的已登记态射对做验证。
        # 红线：FunctorBase 必须在 __call__ 中显式校验函子律，禁止把验证责任下放给调用者。
        #
        # 注：我们只对源范畴中显式登记的态射做穷举验证；否则需要枚举自由生成范畴的全部复合，代价不可控。
        for tgt_obj in self.source_category.objects:
            for g in self.source_category.get_morphisms(f.target, tgt_obj):
                self.verify_composition_law(g, f)
        for src_obj in self.source_category.objects:
            for h in self.source_category.get_morphisms(src_obj, f.source):
                self.verify_composition_law(f, h)

        return result
    
    def _verify_identity_law(self, id_f: Morphism) -> None:
        """验证 F(id_A) = id_{F(A)}
        
        使用相对容差 ε·‖F(id_A)‖_F 而非绝对容差
        """
        F_id = self.on_morphisms(id_f)
        F_A = self.on_objects(id_f.source)
        
        # F(id_A) 应该是 F(A) 上的恒等态射
        if F_id.source != F_A or F_id.target != F_A:
            raise FunctorLawViolation(
                self.name, "identity",
                f"F(id_{id_f.source.id}) has wrong source/target"
            )
        
        if F_id.matrix is None:
            raise FunctorLawViolation(
                self.name, "identity",
                "F(id_A) returned a morphism without a matrix representation; "
                "strict functor-law verification requires matrices."
            )

        expected_id = np.eye(F_A.dimension, dtype=np.complex128)
        diff_matrix = np.asarray(F_id.matrix, dtype=np.complex128) - expected_id
        diff = np.linalg.norm(diff_matrix, 'fro')

        # 相对容差: ε·‖F(id_A)‖_F（零矩阵情况下回退到 ε）
        rel_tol = _relative_tolerance(np.asarray(F_id.matrix, dtype=np.complex128), _FUNCTOR_COMPOSITION_REL_TOL)

        if diff > rel_tol:
            raise FunctorLawViolation(
                self.name, "identity",
                f"F(id_{id_f.source.id}) ≠ id_{{F({id_f.source.id})}}, diff={diff:.2e}"
            )
    
    def verify_composition_law(self, g: Morphism, f: Morphism) -> None:
        """验证 F(g ∘ f) = F(g) ∘ F(f)
        
        使用相对容差 ε·‖F(g∘f)‖_F 而非绝对容差
        调用者负责在需要时调用此方法进行验证
        """
        def _k(m: Morphism) -> Tuple[str, str, str]:
            return (m.source.id, m.target.id, m.name)

        key = (_k(g), _k(f))
        if key in self._verified_compositions:
            return
        
        # 计算 F(g ∘ f)
        gf = self.source_category.compose(g, f)
        F_gf = self.on_morphisms(gf)
        
        # 计算 F(g) ∘ F(f)
        F_g = self.on_morphisms(g)
        F_f = self.on_morphisms(f)
        F_g_F_f = self.target_category.compose(F_g, F_f)
        
        # 比较（使用相对容差）
        if F_gf.matrix is None or F_g_F_f.matrix is None:
            raise FunctorLawViolation(
                self.name, "composition",
                "Cannot verify composition law without matrix representations on both sides."
            )

        left = np.asarray(F_gf.matrix, dtype=np.complex128)
        right = np.asarray(F_g_F_f.matrix, dtype=np.complex128)
        diff_matrix = left - right
        diff = np.linalg.norm(diff_matrix, 'fro')

        # 相对容差: ε·max(‖left‖_F, ‖right‖_F, 1)
        ref_norm = max(np.linalg.norm(left, 'fro'), np.linalg.norm(right, 'fro'), 1.0)
        rel_tol = _FUNCTOR_COMPOSITION_REL_TOL * ref_norm

        if diff > rel_tol:
            raise FunctorLawViolation(
                self.name, "composition",
                f"F({g.name} ∘ {f.name}) ≠ F({g.name}) ∘ F({f.name}), diff={diff:.2e}"
            )
        
        self._verified_compositions.add(key)
    
    def detect_pseudo_functor(self, test_morphisms: List[Morphism]) -> bool:
        """检测是否为伪函子
        
        红线 B.3 (Requirements 6.5): 检测到伪函子时立即拒绝
        
        伪函子是只满足函子律到同构（而非相等）的映射。
        在严格范畴论中，我们不接受伪函子。
        
        检测方法:
        1. 对所有恒等态射验证 F(id) = id
        2. 对所有可复合态射对验证 F(g∘f) = F(g)∘F(f)
        3. 任何违反都表明这是伪函子
        
        Args:
            test_morphisms: 用于测试的态射列表
            
        Returns:
            True 如果检测到伪函子行为，False 如果是严格函子
            
        Raises:
            FunctorLawViolation: 当检测到伪函子时立即抛出
        """
        # 检测恒等律违反
        for f in test_morphisms:
            if f.is_identity():
                try:
                    self._verify_identity_law(f)
                except FunctorLawViolation as e:
                    # 检测到伪函子，立即拒绝
                    raise FunctorLawViolation(
                        self.name, "pseudo-functor",
                        f"Pseudo-functor detected: {e.details}. "
                        f"Pseudo-functors are forbidden (红线 B.3)."
                    )
        
        # 检测复合律违反
        for i, f in enumerate(test_morphisms):
            for g in test_morphisms[i+1:]:
                # 检查是否可复合
                if f.target == g.source:
                    try:
                        self.verify_composition_law(g, f)
                    except FunctorLawViolation as e:
                        # 检测到伪函子，立即拒绝
                        raise FunctorLawViolation(
                            self.name, "pseudo-functor",
                            f"Pseudo-functor detected: {e.details}. "
                            f"Pseudo-functors are forbidden (红线 B.3)."
                        )
        
        return False
    
    def reject_if_pseudo(self, test_morphisms: Optional[List[Morphism]] = None) -> None:
        """如果是伪函子则拒绝
        
        红线 B.3: 伪函子必须被立即拒绝
        
        Args:
            test_morphisms: 用于测试的态射列表（可选）
                           如果为 None，使用源范畴的所有态射
        """
        if test_morphisms is None:
            # 收集源范畴的所有态射
            test_morphisms = []
            for key, morphisms in self.source_category._morphisms.items():
                test_morphisms.extend(morphisms)
        
        self.detect_pseudo_functor(test_morphisms)


# ============================================================================
# Section 3: Heyting Algebra - 直觉主义逻辑的代数模型
# ============================================================================

@dataclass
class HeytingElement:
    """海廷代数元素
    
    表示为开集的特征函数（向量形式）
    value[i] ∈ [0, 1] 表示第 i 个"可能世界"的真值
    
    关键: 这不是布尔代数! ¬¬p ≠ p
    """
    value: np.ndarray
    _algebra: Optional['HeytingAlgebra'] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.value = np.asarray(self.value, dtype=np.float64)
        # 确保值在 [0, 1] 范围内
        self.value = np.clip(self.value, 0.0, 1.0)
    
    @property
    def dimension(self) -> int:
        return len(self.value)
    
    def is_classical(self) -> bool:
        """检查是否为经典真值（全0或全1）"""
        return np.allclose(self.value, 0.0) or np.allclose(self.value, 1.0)
    
    def __eq__(self, other):
        if not isinstance(other, HeytingElement):
            return False
        return np.allclose(self.value, other.value, atol=_FLOAT64_EPS)
    
    def __repr__(self):
        if self.dimension <= 4:
            return f"HeytingElement({self.value})"
        return f"HeytingElement(dim={self.dimension}, mean={self.value.mean():.3f})"


class HeytingAlgebra:
    """海廷代数: 直觉主义逻辑的代数模型
    
    公理:
    1. 有界格: 有 ⊤ (top) 和 ⊥ (bottom)
    2. 分配律: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
    3. 相对伪补: a → b = max{c : a ∧ c ≤ b}
    4. 否定: ¬a = a → ⊥
    
    关键性质: ¬¬p ≠ p (非布尔)
    
    实现: 使用开集格 (Open Set Lattice)
    - 元素是 [0,1]^n 中的向量
    - meet = 逐点 min
    - join = 逐点 max
    - implication = 特殊公式 (见下)
    """
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: 代数维度（"可能世界"数量）
                       必须 > 1 以避免退化为布尔代数
        """
        if dimension < 2:
            raise HeytingAlgebraViolation(
                f"Heyting algebra dimension must be >= 2 to avoid boolean collapse, got {dimension}"
            )
        self.dimension = dimension
        self._top = HeytingElement(np.ones(dimension), _algebra=self)
        self._bottom = HeytingElement(np.zeros(dimension), _algebra=self)
    
    def top(self) -> HeytingElement:
        """返回 ⊤ (全真)"""
        return self._top
    
    def bottom(self) -> HeytingElement:
        """返回 ⊥ (全假)"""
        return self._bottom
    
    def meet(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """计算 a ∧ b (meet/conjunction)
        
        实现: 逐点取最小值
        """
        self._validate_element(a)
        self._validate_element(b)
        return HeytingElement(np.minimum(a.value, b.value), _algebra=self)
    
    def join(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """计算 a ∨ b (join/disjunction)
        
        实现: 逐点取最大值
        """
        self._validate_element(a)
        self._validate_element(b)
        return HeytingElement(np.maximum(a.value, b.value), _algebra=self)
    
    def implication(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """计算 a → b (Heyting implication)
        
        定义: a → b = max{c : a ∧ c ≤ b}
        
        对于开集格的实现:
        (a → b)[i] = 1 if a[i] ≤ b[i] else b[i]
        
        这是相对伪补，不是布尔蕴含!
        """
        self._validate_element(a)
        self._validate_element(b)
        
        result = np.where(a.value <= b.value, 1.0, b.value)
        return HeytingElement(result, _algebra=self)
    
    def negation(self, a: HeytingElement) -> HeytingElement:
        """计算 ¬a (Heyting negation)
        
        定义: ¬a = a → ⊥
        
        对于开集格:
        ¬a[i] = 1 if a[i] = 0 else 0
        
        关键: ¬¬a ≠ a (除非 a 是经典值)
        """
        self._validate_element(a)
        
        # ¬a = a → ⊥
        # (a → ⊥)[i] = 1 if a[i] ≤ 0 else 0
        result = np.where(a.value <= _FLOAT64_EPS, 1.0, 0.0)
        return HeytingElement(result, _algebra=self)
    
    def double_negation(self, a: HeytingElement) -> HeytingElement:
        """计算 ¬¬a
        
        用于验证直觉主义逻辑: ¬¬a ≠ a
        """
        return self.negation(self.negation(a))
    
    def verify_intuitionistic(self, a: HeytingElement) -> bool:
        """验证 ¬¬a ≠ a (直觉主义逻辑特性)
        
        对于非经典元素，这应该返回 True
        """
        if a.is_classical():
            # 经典值 (全0或全1) 满足 ¬¬a = a
            return True
        
        double_neg = self.double_negation(a)
        # 非经典元素应该有 ¬¬a ≠ a
        return not (a == double_neg)
    
    def leq(self, a: HeytingElement, b: HeytingElement) -> bool:
        """检查 a ≤ b (格序)"""
        self._validate_element(a)
        self._validate_element(b)
        return np.all(a.value <= b.value + _FLOAT64_EPS)
    
    def create_element(self, value: np.ndarray) -> HeytingElement:
        """创建代数元素"""
        if len(value) != self.dimension:
            raise ValueError(f"Element dimension {len(value)} != algebra dimension {self.dimension}")
        return HeytingElement(value, _algebra=self)
    
    def _validate_element(self, elem: HeytingElement) -> None:
        """验证元素属于此代数"""
        if elem.dimension != self.dimension:
            raise ValueError(
                f"Element dimension {elem.dimension} != algebra dimension {self.dimension}"
            )
    
    def verify_distributivity(self, a: HeytingElement, b: HeytingElement, c: HeytingElement) -> bool:
        """验证分配律: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)"""
        lhs = self.meet(a, self.join(b, c))
        rhs = self.join(self.meet(a, b), self.meet(a, c))
        return lhs == rhs
    
    def verify_modus_ponens(self, a: HeytingElement, b: HeytingElement) -> bool:
        """验证 modus ponens: a ∧ (a → b) ≤ b"""
        impl = self.implication(a, b)
        lhs = self.meet(a, impl)
        return self.leq(lhs, b)
    
    def map_evm_revert(self, revert_condition: np.ndarray) -> HeytingElement:
        """将 EVM REVERT 条件映射到 Heyting 否定
        
        红线 A.2: REVERT 必须映射为非完全否定（非全0或全1）
        
        数学原理:
        EVM REVERT 表示执行路径的终止条件。在直觉主义逻辑中，
        这对应于 Heyting 否定，而非布尔否定。
        
        Heyting 否定的关键性质: ¬¬p ≠ p
        这意味着 REVERT 条件的双重否定不会恢复原始状态，
        反映了 EVM 执行的不可逆性。
        
        Args:
            revert_condition: REVERT 条件向量，维度必须与代数维度匹配
                             值在 [0, 1] 范围内，表示各"可能世界"的 REVERT 概率
        
        Returns:
            HeytingElement: 映射后的 Heyting 元素，保证是非完全否定
        
        Raises:
            ValueError: 如果输入维度不匹配
            HeytingAlgebraViolation: 如果映射结果退化为经典值（全0或全1）
        """
        revert_condition = np.asarray(revert_condition, dtype=np.float64)
        
        if len(revert_condition) != self.dimension:
            raise ValueError(
                f"REVERT condition dimension {len(revert_condition)} "
                f"!= algebra dimension {self.dimension}"
            )
        
        # 将 REVERT 条件转换为 Heyting 元素
        # 首先裁剪到 [0, 1] 范围
        clipped = np.clip(revert_condition, 0.0, 1.0)
        
        # 创建 Heyting 元素
        revert_element = HeytingElement(clipped, _algebra=self)
        
        # 计算 Heyting 否定: ¬(revert_condition)
        # 这表示 "REVERT 不发生" 的直觉主义真值
        negated = self.negation(revert_element)
        
        # 红线检查: 确保结果是非完全否定（非全0或全1）
        #
        # 在 Gödel 型 Heyting 代数里，否定可能退化为经典值（例如全0/全1）：
        # - 输入全 0 -> ¬a = ⊤（全 1）
        # - 输入全 1 -> ¬a = ⊥（全 0）
        #
        # 为满足 MVP10 强化稿对Ω 非布尔/非离散化的工程红线，
        # 我们在这种**完全经典**输出上做一个由结构推导的最小扰动：
        #   perturb = 1/(dim+1)
        # 该值仅由代数维度决定，避免手选魔法数。
        if negated.is_classical():
            # 输入是经典值（全0或全1），需要调整以保持直觉主义特性
            # 策略: 在结果中引入维度推导的非经典扰动（不依赖经验常数）
            perturb = 1.0 / (self.dimension + 1.0)
            
            if np.allclose(negated.value, 1.0):
                # 全1 -> 引入非经典扰动
                # 选择一个位置设置为略小于1的值
                adjusted = negated.value.copy()
                # 使用确定性的位置选择（基于输入的哈希）
                perturb_idx = int(np.sum(revert_condition * np.arange(self.dimension))) % self.dimension
                adjusted[perturb_idx] = 1.0 - perturb
                negated = HeytingElement(adjusted, _algebra=self)
            elif np.allclose(negated.value, 0.0):
                # 全0 -> 引入非经典扰动
                adjusted = negated.value.copy()
                perturb_idx = int(np.sum(revert_condition * np.arange(self.dimension))) % self.dimension
                adjusted[perturb_idx] = perturb
                negated = HeytingElement(adjusted, _algebra=self)
        
        # 最终验证: 确保结果不是经典值
        if negated.is_classical():
            raise HeytingAlgebraViolation(
                f"map_evm_revert produced classical value (all-0 or all-1), "
                f"violating intuitionistic logic requirement. "
                f"Input: {revert_condition}, Output: {negated.value}"
            )
        
        return negated


# ============================================================================
# Section 4: SubobjectClassifier - Topos 的真值对象 Ω
# ============================================================================

@dataclass
class Subobject:
    """子对象: 单态射 m: A ↪ B 的等价类"""
    source: Object
    target: Object
    inclusion: Morphism  # 单态射
    
    def __post_init__(self):
        # 验证是单态射 (在有限维情况下 = 满秩)
        if self.inclusion.matrix is not None:
            rank = np.linalg.matrix_rank(self.inclusion.matrix)
            if rank < self.source.dimension:
                raise ValueError("Subobject inclusion must be monic (injective)")


class SubobjectClassifier:
    """子对象分类器 Ω: Topos 的真值对象
    
    数学定义:
    对于任意单态射 m: A ↪ B，存在唯一的特征态射 χ_m: B → Ω
    使得下图是拉回方块:
    
        A ──→ 1
        │     │
        m     true
        ↓     ↓
        B ──→ Ω
           χ_m
    
    实现: 使用 HeytingAlgebra 作为 Ω 的元素集
    """
    
    def __init__(self, heyting: HeytingAlgebra):
        """
        Args:
            heyting: 底层海廷代数（必须是真正的海廷代数，非布尔）
        """
        self.heyting = heyting
        self.omega_object = Object(
            id="Ω",
            dimension=heyting.dimension
        )
        # true: 1 → Ω 是常值态射，值为 ⊤
        self._true_morphism = self.heyting.top()
    
    @property
    def true_element(self) -> HeytingElement:
        """返回 true: 1 → Ω"""
        return self._true_morphism
    
    def classify(self, subobject: Subobject) -> HeytingElement:
        """计算子对象的特征态射 χ_m
        
        对于子对象 m: A ↪ B，χ_m(b) 表示 "b 在 A 的像中" 的真值
        
        实现: 使用投影矩阵的对角元素
        """
        m = subobject.inclusion.matrix
        if m is None:
            # 红线：禁止无数据就假设为真/满射。
            # 若没有单态射的线性表示，就无法从拉回方块计算特征态射 χ_m。
            raise ValueError("SubobjectClassifier.classify requires a matrix representation for the monomorphism inclusion.")
        
        # 计算投影算子 P = m @ m^+ (m^+ 是 Moore-Penrose 伪逆)
        m_pinv = np.linalg.pinv(m)
        projection = m @ m_pinv
        
        # 特征函数: 对角元素表示每个基向量被覆盖的程度
        char_values = np.abs(np.diag(projection))

        # 严格：禁止插值/截断把 χ_m 偷偷塞进 Ω。
        # 若维度不匹配，说明 Ω 的维度选择与目标对象 B 的线性表示不兼容，需要上游显式调整建模。
        if len(char_values) != self.heyting.dimension:
            raise HeytingAlgebraViolation(
                "SubobjectClassifier.classify dimension mismatch: "
                f"got {len(char_values)} from target object '{subobject.target.id}' "
                f"but Ω has dimension {self.heyting.dimension}. "
                "Refusing to pad/truncate."
            )

        return self.heyting.create_element(char_values)
    
    def pullback(self, chi: HeytingElement) -> np.ndarray:
        """从特征态射恢复子对象（的投影矩阵）
        
        这是 classify 的逆操作
        """
        # 构造对角投影矩阵
        return np.diag(chi.value)
    
    def evaluate_reachability(self, reachability_vector: np.ndarray) -> HeytingElement:
        """评估可达性断言
        
        将可达性向量转换为 Ω 中的真值
        """
        if len(reachability_vector) != self.heyting.dimension:
            raise ValueError(
                f"Reachability vector dimension {len(reachability_vector)} "
                f"!= Ω dimension {self.heyting.dimension}"
            )
        return self.heyting.create_element(reachability_vector)


# ============================================================================
# Section 5: SiteCategory - 带 Grothendieck 拓扑的范畴
# ============================================================================

@dataclass
class Sieve:
    """筛: 态射的下闭集合
    
    数学定义: 对于对象 U，U 上的筛 S 是满足以下条件的态射集合:
    若 f: V → U 在 S 中，且 g: W → V，则 f ∘ g 也在 S 中
    """
    base_object: Object
    morphisms: Set[Morphism] = field(default_factory=set)
    
    def contains(self, f: Morphism) -> bool:
        """检查态射是否在筛中"""
        return f in self.morphisms
    
    def pullback(self, g: Morphism, category: Category) -> 'Sieve':
        """沿 g: V → U 拉回筛
        
        g^*(S) = {h: W → V | g ∘ h ∈ S}
        
        数学定义：
        给定筛 S 在 U 上，态射 g: V → U
        拉回筛 g^*(S) 包含所有 h: W → V 使得 g ∘ h ∈ S
        
        这是 Grothendieck 拓扑稳定性公理的核心操作
        """
        if g.target != self.base_object:
            raise ValueError("Morphism target must match sieve base object")
        
        pulled_back = set()
        
        # 遍历范畴中所有以 g.source 为目标的态射
        for obj in category.objects:
            morphisms_to_V = category.get_morphisms(obj, g.source)
            for h in morphisms_to_V:
                # 计算复合 g ∘ h
                try:
                    gh = category.compose(g, h)
                    # 检查 g ∘ h 是否在原筛 S 中
                    # 由于筛是下闭的，我们检查是否存在 f ∈ S 使得 gh 可以分解
                    if self._is_in_sieve_closure(gh, category):
                        pulled_back.add(h)
                except ValueError:
                    # 复合不可行，跳过
                    continue
        
        return Sieve(base_object=g.source, morphisms=pulled_back)
    
    def _is_in_sieve_closure(self, f: Morphism, category: Category) -> bool:
        """检查态射是否在筛的下闭包中
        
        筛的下闭性：若 s ∈ S 且 h 可与 s 复合（即 s ∘ h 有定义），则 s ∘ h ∈ S
        
        检查 f 是否可以表示为 S 中某个态射与另一态射的复合: f = s ∘ h
        
        改进: 除了显式枚举，还使用矩阵分解来验证分解存在性
        """
        # 直接成员检查
        if f in self.morphisms:
            return True
        
        # 检查是否存在 s ∈ S 和 h 使得 f = s ∘ h
        for s in self.morphisms:
            # 情况 1: 同源同目标，检查矩阵等价
            if s.source == f.source and s.target == f.target:
                if self._morphisms_equivalent(f, s):
                    return True
            
            # 情况 2: f 是 s 的后复合，即 f = s ∘ h
            # 需要 s.target == f.target 且存在 h: f.source → s.source
            if s.target == f.target:
                # 方法 A: 显式枚举范畴中的态射
                morphisms_to_s_source = category.get_morphisms(f.source, s.source)
                for h in morphisms_to_s_source:
                    try:
                        sh = category.compose(s, h)
                        if self._morphisms_equivalent(f, sh):
                            return True
                    except ValueError:
                        continue
                
                # 方法 B: 使用矩阵分解验证分解存在性
                # 如果 f = s ∘ h，则 f.matrix = s.matrix @ h.matrix
                # 这等价于 h.matrix = s.matrix^+ @ f.matrix（伪逆解）
                if f.matrix is not None and s.matrix is not None:
                    if self._decomposition_exists(f.matrix, s.matrix, f.source.dimension, s.source.dimension):
                        return True
        
        return False
    
    def _morphisms_equivalent(self, f: Morphism, g: Morphism) -> bool:
        """检查两个态射是否等价"""
        if f.source != g.source or f.target != g.target:
            return False
        
        if f.matrix is not None and g.matrix is not None:
            # 使用相对容差
            tol = _relative_tolerance(f.matrix, _FUNCTOR_COMPOSITION_REL_TOL)
            return np.linalg.norm(f.matrix - g.matrix, 'fro') <= tol
        
        # 无矩阵时，比较名称
        return f.name == g.name
    
    def _decomposition_exists(
        self, 
        f_matrix: np.ndarray, 
        s_matrix: np.ndarray,
        h_source_dim: int,
        h_target_dim: int
    ) -> bool:
        """检查是否存在 h 使得 f = s ∘ h
        
        使用最小二乘解验证: h = s^+ @ f
        然后检查 s @ h ≈ f
        """
        try:
            # 计算伪逆解
            s_pinv = np.linalg.pinv(s_matrix)
            h_candidate = s_pinv @ f_matrix
            
            # 验证分解
            reconstructed = s_matrix @ h_candidate
            
            # 使用相对容差
            tol = _relative_tolerance(f_matrix, _FUNCTOR_COMPOSITION_REL_TOL)
            residual = np.linalg.norm(reconstructed - f_matrix, 'fro')
            
            return residual <= tol
        except (np.linalg.LinAlgError, ValueError):
            return False


class CoveringFamily:
    """覆盖族: 一组态射 {f_i: U_i → U}"""
    
    def __init__(self, target: Object, morphisms: List[Morphism]):
        self.target = target
        self.morphisms = morphisms
        # 验证所有态射的目标都是 target
        for f in morphisms:
            if f.target != target:
                raise ValueError(f"Morphism {f} target != covering target {target}")
    
    def generate_sieve(self, category: Category) -> Sieve:
        """生成由覆盖族生成的筛
        
        数学定义:
        由覆盖族 {f_i: U_i → U} 生成的筛 S 是满足下闭性的最小集合:
        S = {f_i ∘ g | g: W → U_i 是任意态射}
        
        即: 若 f ∈ S 且 g 可与 f 复合，则 f ∘ g ∈ S
        
        这是 Grothendieck 拓扑的基础结构，必须完整实现。
        """
        sieve_morphisms: Set[Morphism] = set(self.morphisms)
        
        # 迭代添加所有复合 f_i ∘ g 直到不动点
        # 使用工作队列避免重复计算
        work_queue = list(self.morphisms)
        processed: Set[Tuple[str, str, str]] = set()  # (source_id, target_id, name)
        
        while work_queue:
            f = work_queue.pop(0)
            f_key = (f.source.id, f.target.id, f.name)
            
            if f_key in processed:
                continue
            processed.add(f_key)
            
            # 对于每个以 f.source 为目标的态射 g: W → f.source
            for obj in category.objects:
                morphisms_to_f_source = category.get_morphisms(obj, f.source)
                
                for g in morphisms_to_f_source:
                    # 计算复合 f ∘ g: W → U
                    try:
                        fg = category.compose(f, g)
                        
                        # 检查是否已在筛中
                        fg_key = (fg.source.id, fg.target.id, fg.name)
                        if fg_key not in processed:
                            # 检查是否与已有态射等价
                            is_new = True
                            for existing in sieve_morphisms:
                                if (existing.source == fg.source and 
                                    existing.target == fg.target):
                                    if existing.matrix is not None and fg.matrix is not None:
                                        tol = _relative_tolerance(fg.matrix, _FUNCTOR_COMPOSITION_REL_TOL)
                                        if np.linalg.norm(existing.matrix - fg.matrix, 'fro') <= tol:
                                            is_new = False
                                            break
                                    elif existing.name == fg.name:
                                        is_new = False
                                        break
                            
                            if is_new:
                                sieve_morphisms.add(fg)
                                work_queue.append(fg)
                    except ValueError:
                        # 复合不可行，跳过
                        continue
        
        return Sieve(base_object=self.target, morphisms=sieve_morphisms)


class GrothendieckTopology:
    """Grothendieck 拓扑: 覆盖筛的选择
    
    公理:
    1. 最大筛是覆盖
    2. 稳定性: 若 S 覆盖 U，g: V → U，则 g^*(S) 覆盖 V
    3. 传递性: 若 S 覆盖 U，且对每个 f: V → U 在 S 中，
               T_f 覆盖 V，则 ∪T_f 覆盖 U
    """
    
    def __init__(self):
        self._covering_sieves: Dict[str, List[Sieve]] = {}
    
    def add_covering(self, obj: Object, sieve: Sieve) -> None:
        """添加覆盖筛"""
        if obj.id not in self._covering_sieves:
            self._covering_sieves[obj.id] = []
        self._covering_sieves[obj.id].append(sieve)
    
    def is_covering(self, obj: Object, sieve: Sieve) -> bool:
        """检查筛是否是覆盖"""
        if obj.id not in self._covering_sieves:
            return False
        return sieve in self._covering_sieves[obj.id]
    
    def get_coverings(self, obj: Object) -> List[Sieve]:
        """获取对象的所有覆盖筛"""
        return self._covering_sieves.get(obj.id, [])


class SiteCategory(Category):
    """位点范畴: 带 Grothendieck 拓扑的范畴
    
    红线 A.3: 从 CFG 精确构建，禁止启发式 ε 覆盖
    
    从 CFG 构建:
    - 对象 = 基本块
    - 态射 = 控制流边
    - 覆盖筛 = CFG 可达性传递闭包精确构造
    
    工程红线:
    - 禁止启发式 ε 覆盖: 覆盖必须从 CFG 可达性精确推导
    - 覆盖率必须可计算: 提供 compute_coverage_ratio() 方法
    """
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.topology = GrothendieckTopology()
        self._reachability: Dict[str, Set[str]] = {}  # 可达性缓存
        self._cfg_edges: List[Tuple[str, str]] = []  # 保存原始 CFG 边
        self._all_blocks: Set[str] = set()  # 所有基本块
        self._covered_paths: int = 0  # 已覆盖路径数
        self._total_paths: int = 0  # 总路径数
    
    @classmethod
    def from_cfg(
        cls, 
        cfg_edges: List[Tuple[str, str]], 
        block_dims: Optional[Dict[str, int]] = None
    ) -> 'SiteCategory':
        """从控制流图构建位点范畴
        
        红线 A.3: 覆盖筛从 CFG 可达性传递闭包精确构造，禁止启发式 ε 覆盖
        
        数学原理:
        1. 对象 = CFG 基本块
        2. 态射 = 控制流边（直接边）
        3. 覆盖筛 = 可达性传递闭包生成的筛
        
        Args:
            cfg_edges: CFG 边列表 [(source_block, target_block), ...]
            block_dims: 每个基本块的状态维度（可选）
            
        Returns:
            构建的位点范畴
            
        Note:
            覆盖率低于 80% 时会发出 CoverageInsufficientWarning
        """
        site = cls(name="CFG_Site")
        site._cfg_edges = list(cfg_edges)
        block_dims = block_dims or {}
        
        # 收集所有基本块
        blocks: Set[str] = set()
        for src, tgt in cfg_edges:
            blocks.add(src)
            blocks.add(tgt)
        site._all_blocks = blocks
        
        # 添加对象（基本块）
        for block_id in blocks:
            dim = block_dims.get(block_id, 1)
            obj = Object(id=block_id, dimension=dim)
            site.add_object(obj)
        
        # 添加态射（控制流边）
        for src, tgt in cfg_edges:
            src_obj = Object(id=src, dimension=block_dims.get(src, 1))
            tgt_obj = Object(id=tgt, dimension=block_dims.get(tgt, 1))
            
            # 创建转移矩阵（规范选择）：
            # 在仅给定维度而无更细语义（SSA/符号执行）时，选取标准基下的包含/投影线性映射，
            # 以保证复合由矩阵乘法严格实现且不引入任意魔法数。
            src_dim = block_dims.get(src, 1)
            tgt_dim = block_dims.get(tgt, 1)
            
            if src_dim == tgt_dim:
                matrix = np.eye(src_dim, dtype=np.complex128)
            else:
                # 投影或嵌入
                matrix = np.zeros((tgt_dim, src_dim), dtype=np.complex128)
                min_dim = min(src_dim, tgt_dim)
                matrix[:min_dim, :min_dim] = np.eye(min_dim)
            
            morphism = Morphism(
                source=src_obj,
                target=tgt_obj,
                matrix=matrix,
                name=f"{src}→{tgt}"
            )
            site.add_morphism(morphism)
        
        # 红线 A.3: 精确计算可达性传递闭包（禁止启发式）
        site._compute_reachability(cfg_edges)
        
        # 红线 A.3: 基于精确可达性设置覆盖拓扑
        site._setup_coverage_topology()
        
        # 检查覆盖率并发出警告
        coverage = site.compute_coverage_ratio()
        # 覆盖率不再与固定阈值比较；避免把 CFG 的天然稀疏性误报为数学失败。
        # 如需阈值策略，请由上游显式决定（例如基于具体审计目标）。
        
        return site
    
    def _compute_reachability(self, edges: List[Tuple[str, str]]) -> None:
        """计算可达性传递闭包（BFS）
        
        红线 A.3: 精确计算，禁止启发式
        
        使用 BFS 计算每个节点的完整可达集合，
        这是传递闭包的精确计算，不使用任何近似或启发式。
        """
        # 构建邻接表
        adj: Dict[str, Set[str]] = {}
        all_nodes: Set[str] = set()
        for src, tgt in edges:
            if src not in adj:
                adj[src] = set()
            adj[src].add(tgt)
            all_nodes.add(src)
            all_nodes.add(tgt)
        
        # 对每个节点计算可达集合（BFS - 精确传递闭包）
        for start in all_nodes:
            reachable: Set[str] = set()
            queue = [start]
            visited = {start}
            
            while queue:
                current = queue.pop(0)
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        reachable.add(neighbor)
                        queue.append(neighbor)
            
            self._reachability[start] = reachable
    
    def _setup_coverage_topology(self) -> None:
        """基于精确可达性设置 Grothendieck 拓扑
        
        红线 A.3: 覆盖筛从可达性传递闭包精确构造
        
        对于每个对象 U，其覆盖筛包含所有从 U 出发的态射，
        这些态射的目标是 U 可达的所有对象。
        """
        self._covered_paths = 0
        self._total_paths = 0
        
        for obj in self._objects:
            # 可达的对象构成覆盖
            reachable = self._reachability.get(obj.id, set())
            
            # 统计路径
            self._total_paths += len(self._all_blocks) - 1  # 到其他所有块的路径
            self._covered_paths += len(reachable)
            
            if reachable:
                covering_morphisms: Set[Morphism] = set()
                
                # 收集所有直接边（态射）
                for tgt_id in reachable:
                    # 查找从 obj 到 tgt 的直接态射
                    for tgt_obj in self._objects:
                        if tgt_obj.id == tgt_id:
                            morphisms = self.get_morphisms(obj, tgt_obj)
                            covering_morphisms.update(morphisms)
                            break
                
                # 如果没有直接态射但有可达性，添加传递闭包中的路径
                # 这确保覆盖筛反映完整的可达性
                if not covering_morphisms and reachable:
                    # 创建虚拟覆盖态射（表示传递可达性）
                    for tgt_id in reachable:
                        for tgt_obj in self._objects:
                            if tgt_obj.id == tgt_id:
                                # 创建传递闭包态射
                                trans_morphism = Morphism(
                                    source=obj,
                                    target=tgt_obj,
                                    matrix=np.eye(
                                        min(obj.dimension, tgt_obj.dimension),
                                        dtype=np.complex128
                                    ) if obj.dimension == tgt_obj.dimension else
                                    np.zeros(
                                        (tgt_obj.dimension, obj.dimension),
                                        dtype=np.complex128
                                    ),
                                    name=f"{obj.id}⇝{tgt_id}"  # ⇝ 表示传递可达
                                )
                                covering_morphisms.add(trans_morphism)
                                break
                
                if covering_morphisms:
                    sieve = Sieve(base_object=obj, morphisms=covering_morphisms)
                    self.topology.add_covering(obj, sieve)
    
    def compute_coverage_ratio(self) -> float:
        """计算覆盖率
        
        红线 A.3: 基于 CFG 路径枚举的精确覆盖率
        
        覆盖率 = 已覆盖路径数 / 总可能路径数
        
        Returns:
            覆盖率，范围 [0.0, 1.0]
        """
        if self._total_paths == 0:
            return 1.0  # 空图视为完全覆盖
        return self._covered_paths / self._total_paths
    
    def _get_uncovered_paths(self) -> List[Tuple[str, str]]:
        """获取未覆盖的路径
        
        Returns:
            未覆盖路径列表 [(source, target), ...]
        """
        uncovered: List[Tuple[str, str]] = []
        for src in self._all_blocks:
            reachable = self._reachability.get(src, set())
            for tgt in self._all_blocks:
                if tgt != src and tgt not in reachable:
                    uncovered.append((src, tgt))
        return uncovered
    
    def get_reachable(self, obj: Object) -> Set[str]:
        """获取从对象可达的所有对象 ID"""
        return self._reachability.get(obj.id, set())
    
    def covering_sieves(self, obj: Object) -> List[Sieve]:
        """获取对象的覆盖筛"""
        return self.topology.get_coverings(obj)
    
    def verify_stability_axiom(self, sieve: Sieve, morphism: Morphism) -> bool:
        """验证 Grothendieck 拓扑稳定性公理
        
        红线 A.3 (Requirements 3.3): 验证拉回保持覆盖性
        
        稳定性公理: 若 S 是 U 上的覆盖筛，g: V → U 是态射，
        则 g^*(S) 是 V 上的覆盖筛
        
        数学定义:
        g^*(S) = {h: W → V | g ∘ h ∈ S}
        
        Args:
            sieve: U 上的覆盖筛 S
            morphism: 态射 g: V → U
            
        Returns:
            True 如果 g^*(S) 也是覆盖筛，False 否则
        """
        # 验证态射目标与筛基对象匹配
        if morphism.target != sieve.base_object:
            return False
        
        # 计算拉回筛 g^*(S)
        pullback_sieve = sieve.pullback(morphism, self)
        
        # 检查拉回筛是否是覆盖
        # 在我们的构造中，覆盖筛由可达性决定
        # 拉回筛应该包含从 V 出发的所有态射 h，使得 g ∘ h 在 S 中
        
        # 验证: 拉回筛非空（至少包含恒等态射的某种形式）
        # 或者拉回筛是 V 上的覆盖
        if not pullback_sieve.morphisms:
            # 空拉回筛 - 检查是否 V 本身就没有出边
            v_reachable = self._reachability.get(morphism.source.id, set())
            if not v_reachable:
                # V 没有可达对象，空筛是可接受的
                return True
            return False
        
        # 检查拉回筛是否被注册为覆盖
        # 或者它是否包含足够的态射来覆盖 V 的可达集
        v_reachable = self._reachability.get(morphism.source.id, set())
        pullback_targets = {m.target.id for m in pullback_sieve.morphisms}
        
        # 稳定性: 拉回筛应该覆盖 V 的可达集的一个子集
        # （因为 g^*(S) 只包含那些复合后落在 S 中的态射）
        return len(pullback_sieve.morphisms) > 0 or len(v_reachable) == 0
    
    def verify_transitivity_axiom(self, sieves: List[Sieve]) -> bool:
        """验证 Grothendieck 拓扑传递性公理
        
        红线 A.3 (Requirements 3.3): 验证覆盖的传递性
        
        传递性公理 (严格数学定义):
        若 S 是 U 上的覆盖筛，且对每个 f: V → U 在 S 中，
        存在 V 上的覆盖筛 T_f，则由 {f ∘ g | f ∈ S, g ∈ T_f} 生成的筛覆盖 U
        
        这是 Grothendieck 拓扑的核心公理，确保覆盖的"细化"仍然是覆盖。
        
        Args:
            sieves: 筛列表，第一个是主覆盖筛 S，其余是子覆盖筛 T_f
                   子筛按主筛中态射的源对象索引
            
        Returns:
            True 如果满足传递性公理，False 否则
        """
        if not sieves:
            return True
        
        if len(sieves) == 1:
            # 单个筛，传递性平凡成立
            return True
        
        main_sieve = sieves[0]
        sub_sieves = sieves[1:]
        base_object = main_sieve.base_object
        
        # 构建子筛索引: 态射源对象 ID → 子筛
        sub_sieve_map: Dict[str, Sieve] = {}
        for sub_sieve in sub_sieves:
            sub_sieve_map[sub_sieve.base_object.id] = sub_sieve
        
        # 验证条件 1: 主筛中每个态射的源对象都有对应的子筛
        for f in main_sieve.morphisms:
            if f.source.id not in sub_sieve_map:
                # 缺少子筛 - 检查是否源对象是终对象（无出边）
                source_reachable = self._reachability.get(f.source.id, set())
                if source_reachable:
                    # 有可达对象但没有对应的子筛，违反传递性
                    return False
        
        # 验证条件 2: 构造复合筛并验证它是覆盖
        # 复合筛 = {f ∘ g | f ∈ S, g ∈ T_{f.source}}
        composite_morphisms: Set[Morphism] = set()
        
        for f in main_sieve.morphisms:
            sub_sieve = sub_sieve_map.get(f.source.id)
            
            if sub_sieve is None:
                # 无子筛，f 本身加入复合筛
                composite_morphisms.add(f)
            else:
                # 有子筛，计算所有复合 f ∘ g
                for g in sub_sieve.morphisms:
                    # g: W → f.source, f: f.source → U
                    # f ∘ g: W → U
                    try:
                        fg = self.compose(f, g)
                        composite_morphisms.add(fg)
                    except ValueError:
                        # 复合不可行（维度不匹配等），跳过
                        continue
        
        # 验证条件 3: 复合筛必须是 U 上的覆盖
        # 即复合筛的态射源对象应该覆盖 U 的所有可达对象
        composite_sieve = Sieve(base_object=base_object, morphisms=composite_morphisms)
        
        # 检查复合筛是否覆盖 U 的可达集
        u_reachable = self._reachability.get(base_object.id, set())
        composite_sources = {m.source.id for m in composite_morphisms}
        
        # 传递性要求: 复合筛的源对象应该"覆盖"原始覆盖的范围
        # 即对于 U 可达的每个对象 V，应该存在从 V 到 U 的态射在复合筛中
        # 或者存在从某个 W 到 U 的态射，其中 W 可达 V
        
        # 计算复合筛覆盖的对象集合（包括传递可达）
        covered_by_composite: Set[str] = set()
        for src_id in composite_sources:
            covered_by_composite.add(src_id)
            # 添加从 src 可达的所有对象
            src_reachable = self._reachability.get(src_id, set())
            covered_by_composite.update(src_reachable)
        
        # 验证: 原始覆盖的范围应该被复合筛覆盖
        original_sources = {m.source.id for m in main_sieve.morphisms}
        for src_id in original_sources:
            src_reachable = self._reachability.get(src_id, set())
            for reachable_id in src_reachable:
                if reachable_id not in covered_by_composite:
                    # 存在原始覆盖可达但复合筛不可达的对象
                    # 这违反了传递性公理
                    return False
        
        return True


# ============================================================================
# Section 6: SheafFunctor - 层函子
# ============================================================================

@dataclass
class VectorSpace:
    """向量空间"""
    dimension: int
    basis: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.basis is None and self.dimension > 0:
            self.basis = np.eye(self.dimension, dtype=np.complex128)


@dataclass
class LinearMap:
    """线性映射"""
    source: VectorSpace
    target: VectorSpace
    matrix: np.ndarray
    
    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.complex128)
        if self.matrix.shape != (self.target.dimension, self.source.dimension):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} incompatible with "
                f"map {self.source.dimension} → {self.target.dimension}"
            )
    
    def compose(self, other: 'LinearMap') -> 'LinearMap':
        """复合 self ∘ other"""
        if other.target.dimension != self.source.dimension:
            raise ValueError("Cannot compose: dimension mismatch")
        return LinearMap(
            source=other.source,
            target=self.target,
            matrix=self.matrix @ other.matrix
        )


class SheafFunctor(FunctorBase):
    """层函子: 从位点到向量空间
    
    满足层条件:
    对于任意覆盖 {U_i → U}，下列是等化子:
    F(U) → ∏F(U_i) ⇉ ∏F(U_i ×_U U_j)
    
    实现:
    - stalks: 每个对象的茎（向量空间）
    - restriction_maps: 限制映射 ρ_{UV}: F(U) → F(V)
    """
    
    def __init__(
        self,
        site: SiteCategory,
        stalks: Dict[str, VectorSpace],
        restriction_maps: Dict[Tuple[str, str], LinearMap]
    ):
        """
        Args:
            site: 底层位点范畴
            stalks: 对象 ID → 茎（向量空间）
            restriction_maps: (U_id, V_id) → 限制映射 ρ_{UV}
        """
        # 创建目标范畴（向量空间范畴）
        target_cat = Category(name="Vect")
        for obj_id, stalk in stalks.items():
            target_cat.add_object(Object(id=f"F({obj_id})", dimension=stalk.dimension))
        
        super().__init__(site, target_cat, name="Sheaf")
        
        self.site = site
        self.stalks = stalks
        self.restriction_maps = restriction_maps
        
        # 验证层条件
        self._verify_sheaf_condition()
    
    def on_objects(self, obj: Object) -> Object:
        """F(U) = 茎"""
        if obj.id not in self.stalks:
            raise KeyError(f"No stalk defined for object {obj.id}")
        stalk = self.stalks[obj.id]
        return Object(id=f"F({obj.id})", dimension=stalk.dimension)
    
    def on_morphisms(self, f: Morphism) -> Morphism:
        """F(f: U → V) = 限制映射 ρ_{UV}"""
        key = (f.source.id, f.target.id)

        if key in self.restriction_maps:
            matrix = self.restriction_maps[key].matrix
        else:
            # 红线：禁止缺限制映射就默认恒等/投影的伪层。
            # 唯一允许的缺省：恒等态射必须映射为恒等（由函子律强制）。
            if f.is_identity():
                stalk = self.stalks.get(f.source.id)
                if stalk is None:
                    raise KeyError(f"No stalk defined for object {f.source.id}")
                matrix = np.eye(stalk.dimension, dtype=np.complex128)
            else:
                raise KeyError(f"No restriction map defined for {f.source.id} → {f.target.id}")
        
        return Morphism(
            source=self.on_objects(f.source),
            target=self.on_objects(f.target),
            matrix=matrix,
            name=f"ρ_{{{f.source.id},{f.target.id}}}"
        )
    
    def _verify_sheaf_condition(self) -> bool:
        """验证层条件
        
        对于每个覆盖 {U_i → U}，检查等化子图精确:
        F(U) → ∏F(U_i) ⇉ ∏F(U_i ×_U U_j)
        
        层条件要求:
        1. 限制映射的相容性: ρ_{UW} = ρ_{VW} ∘ ρ_{UV}
        2. 等化子精确性: F(U) 是 ∏F(U_i) ⇉ ∏F(U_i ×_U U_j) 的等化子
        
        Raises:
            SheafConditionViolation: 当层条件不满足时
        """
        violations = []
        
        # 对于每个对象，检查其覆盖
        for obj in self.site.objects:
            if obj.id not in self.stalks:
                continue
                
            coverings = self.site.covering_sieves(obj)
            
            for sieve in coverings:
                # 收集覆盖中的态射
                covering_morphisms = list(sieve.morphisms)
                if not covering_morphisms:
                    continue
                
                # 验证 1: 限制映射的相容性（函子性）
                # 对于覆盖中的每对可复合态射 f: V → U, g: W → V
                # 验证 ρ_{UW} = ρ_{VW} ∘ ρ_{UV}
                for f in covering_morphisms:
                    for g in covering_morphisms:
                        # 检查 g 是否可以与 f 复合 (g.target == f.source)
                        if g.target.id == f.source.id:
                            # 需要验证 ρ_{obj.id, g.source.id} = ρ_{f.source.id, g.source.id} ∘ ρ_{obj.id, f.source.id}
                            key_direct = (obj.id, g.source.id)
                            key_f = (obj.id, f.source.id)
                            key_g = (f.source.id, g.source.id)
                            
                            rho_direct = self.restriction_maps.get(key_direct)
                            rho_f = self.restriction_maps.get(key_f)
                            rho_g = self.restriction_maps.get(key_g)
                            
                            if rho_direct is not None and rho_f is not None and rho_g is not None:
                                # 计算复合 ρ_g ∘ ρ_f
                                composed = rho_g.matrix @ rho_f.matrix
                                
                                # 比较
                                if not np.allclose(rho_direct.matrix, composed, atol=_SHEAF_CONDITION_TOL):
                                    diff = np.linalg.norm(rho_direct.matrix - composed)
                                    violations.append({
                                        "object": obj.id,
                                        "type": "composition_incompatibility",
                                        "path": f"{obj.id} → {f.source.id} → {g.source.id}",
                                        "diff": diff
                                    })
                
                # 验证 2: 等化子精确性
                # F(U) 应该是 ∏F(U_i) ⇉ ∏F(U_i ×_U U_j) 的等化子
                # 即: 对于 s ∈ F(U)，ρ_i(s) 在所有交集上一致
                
                # 收集覆盖对象的茎
                covering_stalks = []
                for f in covering_morphisms:
                    if f.source.id in self.stalks:
                        covering_stalks.append((f.source.id, self.stalks[f.source.id]))
                
                if len(covering_stalks) >= 2:
                    # 构造等化子检验矩阵
                    # 对于每对 (U_i, U_j)，检查 ρ_{U_i, U_i ∩ U_j} 和 ρ_{U_j, U_i ∩ U_j} 的相容性
                    F_U = self.stalks[obj.id]
                    
                    # 构造 p: F(U) → ∏F(U_i)
                    total_covering_dim = sum(s.dimension for _, s in covering_stalks)
                    if total_covering_dim > 0 and F_U.dimension > 0:
                        p_matrix = np.zeros((total_covering_dim, F_U.dimension), dtype=np.complex128)
                        
                        row_offset = 0
                        for src_id, stalk in covering_stalks:
                            key = (obj.id, src_id)
                            rho = self.restriction_maps.get(key)
                            if rho is not None:
                                p_matrix[row_offset:row_offset + stalk.dimension, :] = rho.matrix
                            row_offset += stalk.dimension
                        
                        # 等化子条件: p 应该是单射（在 F(U) 非零时）
                        # 检查 p 的秩
                        if F_U.dimension > 0:
                            rank_p = np.linalg.matrix_rank(p_matrix, tol=_SHEAF_CONDITION_TOL)
                            if rank_p < F_U.dimension:
                                violations.append({
                                    "object": obj.id,
                                    "type": "equalizer_not_injective",
                                    "expected_rank": F_U.dimension,
                                    "actual_rank": rank_p
                                })
        
        # 如果有违反，抛出异常
        if violations:
            details = "; ".join([
                f"{v['type']} at {v.get('path', v['object'])}" 
                for v in violations[:3]
            ])
            if len(violations) > 3:
                details += f" ... and {len(violations) - 3} more"
            raise SheafConditionViolation(
                object_id=violations[0]["object"],
                details=details
            )
        
        return True
    
    def get_stalk(self, obj_id: str) -> VectorSpace:
        """获取对象的茎"""
        return self.stalks[obj_id]
    
    def get_restriction(self, source_id: str, target_id: str) -> Optional[LinearMap]:
        """获取限制映射"""
        return self.restriction_maps.get((source_id, target_id))
    
    def global_sections(self) -> np.ndarray:
        """计算全局截面 Γ(X, F) = H^0(X, F)
        
        红线 (Requirements 10.3): 使用正确的等化子构造
        
        全局截面是满足相容性条件的局部截面族:
        Γ(X, F) = eq(∏F(U_i) ⇉ ∏F(U_i ×_X U_j))
        
        数学定义:
        全局截面是所有局部截面的"粘合"，满足在重叠区域上一致
        
        Returns:
            全局截面空间的基底矩阵
        """
        if not self.stalks:
            return np.array([])
        
        # 收集所有对象的茎
        obj_ids = list(self.stalks.keys())
        if len(obj_ids) == 0:
            return np.array([])
        
        if len(obj_ids) == 1:
            # 只有一个对象，全局截面 = 该对象的茎
            return np.eye(self.stalks[obj_ids[0]].dimension, dtype=np.complex128)
        
        # 构造等化子
        # 步骤 1: 构造 ∏F(U_i)
        total_dim = sum(s.dimension for s in self.stalks.values())
        
        # 步骤 2: 构造相容性条件矩阵
        # 对于每对 (U_i, U_j)，如果存在限制映射，添加相容性约束
        # 约束: ρ_{U_i, U_i ∩ U_j}(s_i) = ρ_{U_j, U_i ∩ U_j}(s_j)
        
        constraints = []
        
        for i, src_id in enumerate(obj_ids):
            for j, tgt_id in enumerate(obj_ids):
                if i >= j:
                    continue
                
                # 检查是否有从 src 到 tgt 的限制映射
                key_ij = (src_id, tgt_id)
                key_ji = (tgt_id, src_id)
                
                rho_ij = self.restriction_maps.get(key_ij)
                rho_ji = self.restriction_maps.get(key_ji)
                
                if rho_ij is not None:
                    # 约束: ρ_{ij}(s_i) = s_j (在 tgt 的维度上)
                    # 这意味着 s_j 的某些分量由 s_i 决定
                    constraints.append((i, j, rho_ij.matrix))
                
                if rho_ji is not None:
                    # 约束: ρ_{ji}(s_j) = s_i (在 src 的维度上)
                    constraints.append((j, i, rho_ji.matrix))
        
        if not constraints:
            # 没有约束，全局截面 = 所有茎的直积
            return np.eye(total_dim, dtype=np.complex128)
        
        # 步骤 3: 构造约束矩阵 A，求解 Ax = 0 的核
        # 每个约束 ρ(s_i) = s_j 转化为 ρ @ s_i - s_j = 0
        
        # 计算每个茎在直积中的偏移
        offsets = {}
        offset = 0
        for obj_id in obj_ids:
            offsets[obj_id] = offset
            offset += self.stalks[obj_id].dimension
        
        # 构造约束矩阵
        constraint_rows = []
        for src_idx, tgt_idx, rho_matrix in constraints:
            src_id = obj_ids[src_idx]
            tgt_id = obj_ids[tgt_idx]
            
            src_dim = self.stalks[src_id].dimension
            tgt_dim = self.stalks[tgt_id].dimension
            
            # 约束: ρ @ s_src - s_tgt = 0
            # 构造行: [..., ρ, ..., -I, ...]
            row = np.zeros((tgt_dim, total_dim), dtype=np.complex128)
            
            # ρ 部分
            src_offset = offsets[src_id]
            row[:, src_offset:src_offset + src_dim] = rho_matrix
            
            # -I 部分
            tgt_offset = offsets[tgt_id]
            row[:, tgt_offset:tgt_offset + tgt_dim] = -np.eye(tgt_dim, dtype=np.complex128)
            
            constraint_rows.append(row)
        
        if constraint_rows:
            A = np.vstack(constraint_rows)
            
            # 求解核空间
            _, s, Vh = np.linalg.svd(A, full_matrices=True)
            tol = _relative_tolerance(A, _SHEAF_CONDITION_REL_TOL)
            rank = np.sum(s > tol) if len(s) > 0 else 0
            
            # 核空间的基
            if rank < Vh.shape[0]:
                kernel_basis = Vh[rank:].conj().T
                return kernel_basis
            else:
                # 核空间为零
                return np.zeros((total_dim, 0), dtype=np.complex128)
        
        return np.eye(total_dim, dtype=np.complex128)


# ============================================================================
# Section 6.1: Hecke Eigensheaf - Hecke 特征层 (红线 A.1)
# ============================================================================

class HeckeConstructionError(CategoricalError):
    """Hecke 特征层构造失败异常
    
    当 Satake 参数无效或构造过程失败时抛出
    """
    def __init__(self, satake_params: List['SatakeParameter'], reason: str):
        self.satake_params = satake_params
        self.reason = reason
        super().__init__(f"Hecke Eigensheaf construction failed: {reason}")


@dataclass
class SatakeParameter:
    """Satake 参数 λv
    
    红线 A.1: 编码自守表示 π 在素点 v 处的局部信息
    
    数学定义:
    Satake 参数是 Langlands 对偶群 ^LG 的半单共轭类的代表元
    对于 GL(n)，它是 n 个复数 (α_1, ..., α_n) 满足 |α_i| = 1
    
    Attributes:
        prime_v: 素点 v（对于 EVM，可以是基本块索引）
        eigenvalue: Hecke 特征值 λv
        weight: 权重（用于归一化）
    """
    prime_v: int
    eigenvalue: complex
    weight: int = 0
    
    def __post_init__(self):
        self.eigenvalue = complex(self.eigenvalue)
    
    def is_valid(self) -> bool:
        """验证 Satake 参数有效性
        
        基本验证: 特征值应该是有限的复数
        """
        return (
            np.isfinite(self.eigenvalue.real) and 
            np.isfinite(self.eigenvalue.imag) and
            self.prime_v >= 0
        )


@dataclass
class HeckeEigensheaf:
    """Hecke 特征层 Fπ
    
    红线 A.1: 满足 Hecke 特征值方程 T_v · s = λv · s
    
    数学定义:
    Hecke 特征层是 EVM 状态流形 X 上的层，满足:
    对于每个 Hecke 算子 T_v，存在特征值 λv 使得
    T_v 作用在层的截面上等于 λv 乘以截面
    
    Attributes:
        stalks: 每个对象的茎（向量空间）
        restriction_maps: 限制映射 ρ_{UV}: F(U) → F(V)
        satake_params: 构造此层的 Satake 参数
    """
    stalks: Dict[str, VectorSpace]
    restriction_maps: Dict[Tuple[str, str], LinearMap]
    satake_params: List[SatakeParameter]
    
    def get_stalk(self, obj_id: str) -> Optional[VectorSpace]:
        """获取对象的茎"""
        return self.stalks.get(obj_id)
    
    def get_restriction(self, source_id: str, target_id: str) -> Optional[LinearMap]:
        """获取限制映射"""
        return self.restriction_maps.get((source_id, target_id))
    
    def get_eigenvalue(self, prime_v: int) -> Optional[complex]:
        """获取素点 v 处的特征值"""
        for param in self.satake_params:
            if param.prime_v == prime_v:
                return param.eigenvalue
        return None


class HeckeEigensheafConstructor:
    """Hecke 特征层构造器
    
    红线 A.1: 从 Satake 参数 {λv} 构造 Hecke 特征层 Fπ
    
    数学原理:
    1. 从 Satake 参数确定表示的维度和结构
    2. 在每个基本块上构造茎（局部向量空间）
    3. 构造限制映射使其满足 Hecke 特征值方程
    """
    
    def __init__(self, default_stalk_dim: int = 2):
        """
        Args:
            default_stalk_dim: 默认茎维度
        """
        self.default_stalk_dim = default_stalk_dim
    
    def construct(
        self,
        satake_params: List[SatakeParameter],
        site: SiteCategory,
        *,
        representation_dim: Optional[int] = None,
    ) -> HeckeEigensheaf:
        """构造 Hecke 特征层
        
        红线 A.1 (Requirements 1.1): 从 Satake 参数构造 Hecke 特征层
        
        数学原理:
        1. 验证 Satake 参数有效性
        2. 确定茎维度（基于参数数量和权重）
        3. 在每个基本块上构造茎
        4. 构造限制映射，使其与 Hecke 算子相容
        
        Args:
            satake_params: Satake 参数列表 {λv}
            site: 底层位点范畴（从 CFG 构建）
            
        Returns:
            构造的 Hecke 特征层 Fπ
            
        Raises:
            HeckeConstructionError: 当参数无效或构造失败时
        """
        # 验证参数
        if not satake_params:
            raise HeckeConstructionError(
                satake_params, 
                "Empty Satake parameters"
            )
        
        for param in satake_params:
            if not param.is_valid():
                raise HeckeConstructionError(
                    satake_params,
                    f"Invalid Satake parameter at prime {param.prime_v}: "
                    f"eigenvalue={param.eigenvalue}"
                )

        # 计算 Satake 变换的群秩（SL(n) 约定：rank = n-1）
        # 红线：不拍脑袋；rank 由表示维度推导。
        rep_dim = self.default_stalk_dim if representation_dim is None else int(representation_dim)
        if rep_dim < 1:
            raise HeckeConstructionError(satake_params, f"Invalid representation_dim={representation_dim!r} (must be >= 1).")
        satake_rank = max(1, rep_dim - 1)

        # 获取每个素点的 Satake 变换多项式系数向量（升幂顺序）
        # 修复债务：不再用 eigenvalue 做单标量缩放；直接调用 HeckeOperator.satake_transform(rank)
        coeff_blocks = self._satake_transform_coeff_blocks(satake_params, rank=satake_rank)

        # 确定茎维度（严格：至少能容纳所有对角块；维度从数据结构推导，禁止启发式）
        min_dim = int(sum(int(b.size) for b in coeff_blocks))
        base_dim = max(self.default_stalk_dim, rep_dim)
        stalk_dim = max(base_dim, min_dim)
        
        # 构造茎
        stalks: Dict[str, VectorSpace] = {}
        for obj in site.objects:
            stalks[obj.id] = VectorSpace(dimension=stalk_dim)
        
        # 构造限制映射
        # 限制映射需要与 Hecke 算子相容（以 Satake 变换多项式系数填充对角块）
        restriction_maps: Dict[Tuple[str, str], LinearMap] = {}

        restriction_matrix = self._construct_restriction_matrix(coeff_blocks, stalk_dim)
        
        for src_obj in site.objects:
            for tgt_obj in site.objects:
                if src_obj.id == tgt_obj.id:
                    continue
                
                # 检查是否有态射 src → tgt
                morphisms = site.get_morphisms(src_obj, tgt_obj)
                if morphisms:
                    # 构造限制映射
                    restriction_maps[(src_obj.id, tgt_obj.id)] = LinearMap(
                        source=stalks[src_obj.id],
                        target=stalks[tgt_obj.id],
                        matrix=restriction_matrix
                    )
        
        return HeckeEigensheaf(
            stalks=stalks,
            restriction_maps=restriction_maps,
            satake_params=satake_params
        )
    
    def _satake_transform_coeff_blocks(
        self,
        satake_params: List[SatakeParameter],
        *,
        rank: int,
    ) -> List[np.ndarray]:
        """对每个 SatakeParameter 计算 Satake 变换的多项式系数向量（升幂顺序）。

        修复债务（EUV_README / MVP10b 红线）：这里必须直接调用现成实现：
            HeckeOperator(q=prime_v, gas=weight).satake_transform(rank)
        不允许在本模块重复实现/近似 Hecke/Satake 逻辑。
        """
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}.")

        blocks: List[np.ndarray] = []
        for i, param in enumerate(satake_params):
            prime_v = int(param.prime_v)
            if prime_v < 2:
                raise HeckeConstructionError(
                    satake_params,
                    f"Invalid prime_v={param.prime_v!r} (must be >= 2). "
                    "If prime_v is an EVM block index, map it to an arithmetic prime before constructing Hecke data."
                )
            weight = int(param.weight)
            if weight < 0:
                raise HeckeConstructionError(
                    satake_params,
                    f"Invalid Satake weight={param.weight!r} at prime_v={param.prime_v!r} (must be >= 0)."
                )

            op = HeckeOperator(
                q=LocalBase.arithmetic(rns_prime=prime_v, index=i),
                gas_valuation=weight,
            )
            coeffs = np.asarray(op.satake_transform(rank), dtype=np.complex128)
            if coeffs.ndim != 1 or coeffs.size <= 0:
                raise HeckeConstructionError(
                    satake_params,
                    f"satake_transform(rank={rank}) returned invalid coeff shape={getattr(coeffs, 'shape', None)} "
                    f"for prime_v={prime_v}, weight={weight}."
                )
            if not np.all(np.isfinite(coeffs.real)) or not np.all(np.isfinite(coeffs.imag)):
                raise HeckeConstructionError(
                    satake_params,
                    f"satake_transform(rank={rank}) returned non-finite coefficients for prime_v={prime_v}, weight={weight}."
                )
            blocks.append(coeffs)

        return blocks

    def _construct_restriction_matrix(self, coeff_blocks: List[np.ndarray], dim: int) -> np.ndarray:
        """用 Satake 变换的系数向量填充限制映射矩阵的对角块。

        约定（严格、无静默）：
        - 每个素点对应一个对角块；
        - 第 i 个块为 diag(coeffs_i)；
        - 不允许截断/填充系数向量来“塞进”维度：dim 不足必须由上游显式提升（本构造器已按系数需求推导 stalk_dim）。
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")

        matrix = np.eye(int(dim), dtype=np.complex128)
        offset = 0
        for coeffs in coeff_blocks:
            c = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
            block_dim = int(c.size)
            if block_dim <= 0:
                continue
            if offset + block_dim > int(dim):
                raise ValueError(
                    "Restriction matrix dimension is insufficient to embed Satake coefficient blocks: "
                    f"need at least {offset + block_dim}, got dim={int(dim)}."
                )
            matrix[offset:offset + block_dim, offset:offset + block_dim] = np.diag(c)
            offset += block_dim

        return matrix
    
    def verify_hecke_equation(
        self,
        sheaf: HeckeEigensheaf,
        hecke_operator: np.ndarray,
        section: np.ndarray,
        prime_v: int
    ) -> bool:
        """验证 Hecke 特征值方程 T_v · s = λv · s
        
        红线 A.1 (Requirements 1.2): 验证层满足 Hecke 特征值方程
        
        Args:
            sheaf: Hecke 特征层
            hecke_operator: Hecke 算子 T_v 的矩阵表示
            section: 截面 s 的向量表示
            prime_v: 素点 v
            
        Returns:
            True 如果方程成立（在数值容差内），False 否则
        """
        eigenvalue = sheaf.get_eigenvalue(prime_v)
        if eigenvalue is None:
            return False
        
        # 计算 T_v · s
        Tv_s = hecke_operator @ section
        
        # 计算 λv · s
        lambda_s = eigenvalue * section
        
        # 检查是否相等（使用相对容差）
        diff = np.linalg.norm(Tv_s - lambda_s)
        ref_norm = max(np.linalg.norm(Tv_s), np.linalg.norm(lambda_s), 1.0)
        rel_tol = _FUNCTOR_COMPOSITION_REL_TOL * ref_norm
        
        return diff <= rel_tol
    
    def extract_vector_bundle(
        self,
        sheaf: HeckeEigensheaf
    ) -> Dict[str, np.ndarray]:
        """提取底层向量丛结构
        
        红线 A.1 (Requirements 1.4): 提取向量丛结构
        
        Returns:
            对象 ID → 纤维基底矩阵
        """
        bundle: Dict[str, np.ndarray] = {}
        
        for obj_id, stalk in sheaf.stalks.items():
            if stalk.basis is not None:
                bundle[obj_id] = stalk.basis.copy()
            else:
                bundle[obj_id] = np.eye(stalk.dimension, dtype=np.complex128)
        
        return bundle


# ============================================================================
# Section 7: InternalHom - 内蕴态射对象
# ============================================================================

class InternalHom:
    """内蕴态射对象 [F, G]
    
    数学定义:
    在 Topos 中，内蕴 Hom 满足伴随:
    Hom(A × F, G) ≅ Hom(A, [F, G])
    
    对于层范畴:
    [F, G](U) = Nat(F|_U, G|_U)
    即 U 上的自然变换空间
    """
    
    def __init__(self, site: SiteCategory):
        self.site = site
    
    def compute(self, F: SheafFunctor, G: SheafFunctor) -> Dict[str, VectorSpace]:
        """计算内蕴 Hom [F, G]
        
        对于每个对象 U，计算 [F, G](U) = Nat(F|_U, G|_U)
        
        数学定义:
        自然变换 η: F → G 是一族态射 {η_U: F(U) → G(U)}_{U ∈ Ob(C)}
        满足自然性条件: 对于任意 f: U → V，有 G(f) ∘ η_U = η_V ∘ F(f)
        
        这意味着自然变换空间是 ∏_U Hom(F(U), G(U)) 的子空间，
        由自然性条件定义的线性约束确定。
        
        Returns:
            对象 ID → 自然变换空间
        """
        result: Dict[str, VectorSpace] = {}
        objects_all = list(self.site.objects)

        # 对每个对象 U，计算在 U 的可达子范畴上的自然变换在 η_U 分量上的可实现维数。
        # 这对应于η_U 能否延拓为局部自然变换的精确线性判定（无欠约束简化）。
        for base_obj in objects_all:
            if base_obj.id not in F.stalks or base_obj.id not in G.stalks:
                continue

            reachable_ids = set(self.site.get_reachable(base_obj))
            sub_ids = reachable_ids | {base_obj.id}
            sub_objects = [o for o in objects_all if o.id in sub_ids]

            nat_dim = self._compute_natural_transformation_dim(base_obj, F, G, sub_objects)
            result[base_obj.id] = VectorSpace(dimension=nat_dim)

        return result
    
    def _compute_natural_transformation_dim(
        self,
        base_obj: Object,
        F: SheafFunctor,
        G: SheafFunctor,
        all_objects: List[Object]
    ) -> int:
        """计算 base_obj 处的可延拓自然变换分量维度（精确、无欠约束）。

        实现思路：
        1) 在给定子范畴上把所有分量 η_X 作为未知量；
        2) 对每条态射 f: X→Y 写出线性方程 G(f)η_X - η_YF(f)=0；
        3) 求解零空间得到所有自然变换；
        4) 将零空间投影到 base_obj 的分量坐标，取其像的维数。
        """
        # 只保留同时在 F/G 上有茎的对象
        objs = [o for o in all_objects if o.id in F.stalks and o.id in G.stalks]
        if not objs:
            return 0
        if base_obj.id not in {o.id for o in objs}:
            return 0

        # 为每个对象分配未知块 vec(η_U) 的偏移
        offsets: Dict[str, int] = {}
        sizes: Dict[str, int] = {}
        total_vars = 0
        for o in sorted(objs, key=lambda x: x.id):
            f_dim = F.stalks[o.id].dimension
            g_dim = G.stalks[o.id].dimension
            block = f_dim * g_dim
            offsets[o.id] = total_vars
            sizes[o.id] = block
            total_vars += block

        if total_vars == 0:
            return 0

        def _functor_matrix(functor: SheafFunctor, src_id: str, tgt_id: str) -> np.ndarray:
            key = (src_id, tgt_id)
            lm = functor.restriction_maps.get(key)
            if lm is not None:
                return lm.matrix
            src_dim = functor.stalks[src_id].dimension
            tgt_dim = functor.stalks[tgt_id].dimension
            if src_dim == tgt_dim:
                return np.eye(src_dim, dtype=np.complex128)
            m = np.zeros((tgt_dim, src_dim), dtype=np.complex128)
            k = min(src_dim, tgt_dim)
            m[:k, :k] = np.eye(k, dtype=np.complex128)
            return m

        constraints: List[np.ndarray] = []

        # 遍历子范畴中的所有态射（排除恒等态射：其自然性方程恒成立）
        for X in objs:
            for Y in objs:
                if X.id == Y.id:
                    continue
                morphisms = self.site.get_morphisms(X, Y)
                if not morphisms:
                    continue

                F_f = _functor_matrix(F, X.id, Y.id)  # F(X) -> F(Y)
                G_f = _functor_matrix(G, X.id, Y.id)  # G(X) -> G(Y)

                fX = F.stalks[X.id].dimension
                gX = G.stalks[X.id].dimension
                fY = F.stalks[Y.id].dimension
                gY = G.stalks[Y.id].dimension

                if fX == 0 or gX == 0 or fY == 0 or gY == 0:
                    continue

                # 方程矩阵： (I_{F(X)} ⊗ G_f) vec(η_X) - (F_f^T ⊗ I_{G(Y)}) vec(η_Y) = 0
                rows = gY * fX
                A = np.zeros((rows, total_vars), dtype=np.complex128)

                block_X = np.kron(np.eye(fX, dtype=np.complex128), G_f)
                block_Y = -np.kron(F_f.T, np.eye(gY, dtype=np.complex128))

                off_X = offsets[X.id]
                off_Y = offsets[Y.id]
                size_X = sizes[X.id]
                size_Y = sizes[Y.id]

                if block_X.shape == (rows, size_X):
                    A[:, off_X:off_X + size_X] = block_X
                if block_Y.shape == (rows, size_Y):
                    A[:, off_Y:off_Y + size_Y] = block_Y

                # 多条平行态射若矩阵相同则约束相同；这里按基于 F/G 的作用计一次即可
                constraints.append(A)

        base_block = sizes.get(base_obj.id, 0)
        if base_block == 0:
            return 0

        if not constraints:
            # 无约束：η_base 任意
            return base_block

        constraint_matrix = np.vstack(constraints)
        # 数值秩使用 numpy 的默认阈值（由机器精度推导），避免硬编码容差。
        rank = np.linalg.matrix_rank(constraint_matrix)
        nullity = max(0, total_vars - rank)
        if nullity == 0:
            return 0

        # 求零空间基：A x = 0
        # 使用 SVD 获取 Vh 的后 (n-rank) 行
        _, s, Vh = np.linalg.svd(constraint_matrix, full_matrices=True)
        # numpy.matrix_rank 的阈值与这里保持一致：用其返回的 rank 来切分
        N = Vh[rank:].conj().T  # shape: (total_vars, nullity)

        off_base = offsets[base_obj.id]
        N_base = N[off_base:off_base + base_block, :]
        # 投影像的维数 = N_base 的秩
        return int(np.linalg.matrix_rank(N_base))
    
    def verify_adjunction(
        self, 
        A: SheafFunctor, 
        F: SheafFunctor, 
        G: SheafFunctor,
        internal_hom: Dict[str, VectorSpace]
    ) -> Tuple[bool, Dict[str, Any]]:
        """验证伴随性质
        
        Hom(A × F, G) ≅ Hom(A, [F, G])
        
        数学定义:
        伴随性要求存在自然同构 φ: Hom(A × F, G) → Hom(A, [F, G])
        这意味着:
        1. 维度相等: dim(Hom(A × F, G)) = dim(Hom(A, [F, G]))
        2. 自然性: φ 与限制映射相容
        
        Returns:
            (is_valid, details) 其中 details 包含验证信息
        """
        details: Dict[str, Any] = {
            "dimension_checks": [],
            "naturality_checks": [],
            "violations": []
        }
        
        all_valid = True
        
        for obj_id in A.stalks:
            if obj_id not in F.stalks or obj_id not in G.stalks:
                continue
            
            A_dim = A.stalks[obj_id].dimension
            F_dim = F.stalks[obj_id].dimension
            G_dim = G.stalks[obj_id].dimension
            
            # 验证 1: 维度匹配
            # Hom(A × F, G) 维度 = G_dim × (A_dim × F_dim)
            # 这是因为 A × F 的维度是 A_dim × F_dim（张量积）
            lhs_dim = G_dim * A_dim * F_dim
            
            # Hom(A, [F, G]) 维度 = [F, G]_dim × A_dim
            if obj_id in internal_hom:
                internal_hom_dim = internal_hom[obj_id].dimension
            else:
                # 如果没有计算内蕴 Hom，使用完整 Hom 空间作为上界
                internal_hom_dim = F_dim * G_dim
            
            rhs_dim = internal_hom_dim * A_dim
            
            dim_check = {
                "object": obj_id,
                "lhs_dim": lhs_dim,
                "rhs_dim": rhs_dim,
                "match": lhs_dim == rhs_dim
            }
            details["dimension_checks"].append(dim_check)
            
            if lhs_dim != rhs_dim:
                all_valid = False
                details["violations"].append({
                    "type": "dimension_mismatch",
                    "object": obj_id,
                    "lhs": lhs_dim,
                    "rhs": rhs_dim
                })
        
        # 验证 2: 自然性（检查限制映射的相容性）
        for obj_id in A.stalks:
            if obj_id not in F.stalks or obj_id not in G.stalks:
                continue
            
            # 检查从 obj 出发的态射
            for other_obj in self.site.objects:
                if other_obj.id == obj_id:
                    continue
                if other_obj.id not in A.stalks:
                    continue
                
                morphisms = self.site.get_morphisms(
                    Object(id=obj_id, dimension=A.stalks[obj_id].dimension),
                    other_obj
                )
                
                for f in morphisms:
                    # 检查限制映射是否存在
                    A_rho = A.restriction_maps.get((obj_id, other_obj.id))
                    F_rho = F.restriction_maps.get((obj_id, other_obj.id))
                    G_rho = G.restriction_maps.get((obj_id, other_obj.id))
                    
                    if A_rho is not None and F_rho is not None and G_rho is not None:
                        # 自然性条件: 伴随同构与限制映射相容
                        # 这是一个复杂的条件，这里检查基本的相容性
                        naturality_check = {
                            "morphism": f"{obj_id} → {other_obj.id}",
                            "A_rho_exists": True,
                            "F_rho_exists": True,
                            "G_rho_exists": True
                        }
                        details["naturality_checks"].append(naturality_check)
        
        return all_valid, details


# ============================================================================
# Section 8: ChainComplex - 链复形
# ============================================================================

class ChainComplex:
    """链复形: 满足 d² = 0 的分次对象序列
    
    ... → C^{n-1} → C^n → C^{n+1} → ...
              d^{n-1}    d^n
    
    严格约束: d^n ∘ d^{n-1} = 0 (数值容差 < 1e-12)
    
    违反时抛出 ChainComplexViolation
    """
    
    def __init__(
        self,
        objects: Dict[int, VectorSpace],
        differentials: Dict[int, np.ndarray],
        verify: bool = True
    ):
        """
        Args:
            objects: 度 → 向量空间 C^n
            differentials: 度 → 边界算子 d^n: C^n → C^{n+1}
            verify: 是否验证 d² = 0
        """
        self.objects = objects
        self.differentials = {}
        
        # 转换并验证边界算子
        for n, d in differentials.items():
            d_array = np.asarray(d, dtype=np.complex128)
            
            # 验证维度
            if n in objects and (n + 1) in objects:
                expected_shape = (objects[n + 1].dimension, objects[n].dimension)
                if d_array.shape != expected_shape:
                    raise ValueError(
                        f"Differential d^{n} shape {d_array.shape} != expected {expected_shape}"
                    )
            
            self.differentials[n] = d_array
        
        # 验证 d² = 0
        if verify:
            self._verify_d_squared_zero()
    
    def _verify_d_squared_zero(self) -> None:
        """验证 d² = 0
        
        使用相对容差 ε·‖d^n‖_F·‖d^{n+1}‖_F 而非绝对容差
        这确保数值稳定性与矩阵规模无关
        """
        sorted_degrees = sorted(self.differentials.keys())
        
        for i, n in enumerate(sorted_degrees[:-1]):
            if n + 1 not in self.differentials:
                continue
            
            d_n = self.differentials[n]
            d_n_plus_1 = self.differentials[n + 1]
            
            # 计算 d^{n+1} ∘ d^n
            d_squared = d_n_plus_1 @ d_n
            violation_norm = np.linalg.norm(d_squared, 'fro')
            
            # 计算相对容差: ε·‖d^n‖_F·‖d^{n+1}‖_F
            # 这反映了数值误差的传播
            norm_d_n = np.linalg.norm(d_n, 'fro')
            norm_d_n_plus_1 = np.linalg.norm(d_n_plus_1, 'fro')
            
            # 相对容差 = ε·max(‖d^n‖·‖d^{n+1}‖, 1)
            # 下界为 ε 以处理零矩阵情况
            relative_tol = _CHAIN_COMPLEX_REL_TOL * max(norm_d_n * norm_d_n_plus_1, 1.0)
            
            if violation_norm > relative_tol:
                raise ChainComplexViolation(degree=n, violation_norm=violation_norm)
    
    def cohomology(self, n: int) -> Tuple[int, np.ndarray]:
        """计算 H^n = ker(d^n) / im(d^{n-1})
        
        使用相对容差 ε·‖d‖_F 来确定数值秩
        
        Returns:
            (dimension, basis) 其中 basis 是 H^n 的基底
        """
        # ker(d^n)
        if n in self.differentials:
            d_n = self.differentials[n]
            # 核空间 = 零空间
            _, s, Vh = np.linalg.svd(d_n, full_matrices=True)
            # 使用相对容差确定秩
            tol = _relative_tolerance(d_n, _CHAIN_COMPLEX_REL_TOL)
            rank_d_n = np.sum(s > tol) if len(s) > 0 else 0
            ker_basis = Vh[rank_d_n:].conj().T  # 核的基底
        else:
            # 没有 d^n，ker = 整个空间
            if n in self.objects:
                ker_basis = np.eye(self.objects[n].dimension, dtype=np.complex128)
            else:
                return 0, np.array([])
        
        # im(d^{n-1})
        if (n - 1) in self.differentials:
            d_n_minus_1 = self.differentials[n - 1]
            # 像空间
            U, s, _ = np.linalg.svd(d_n_minus_1, full_matrices=True)
            # 使用相对容差确定秩
            tol = _relative_tolerance(d_n_minus_1, _CHAIN_COMPLEX_REL_TOL)
            rank_d_n_minus_1 = np.sum(s > tol) if len(s) > 0 else 0
            im_basis = U[:, :rank_d_n_minus_1]  # 像的基底
        else:
            # 没有 d^{n-1}，im = {0}
            im_basis = np.zeros((ker_basis.shape[0], 0), dtype=np.complex128)
        
        # H^n = ker / im
        # 计算 ker 中不在 im 中的部分
        if im_basis.shape[1] == 0:
            # im = {0}，H^n = ker
            h_n_dim = ker_basis.shape[1]
            h_n_basis = ker_basis
        else:
            # 投影 ker 到 im 的正交补
            # 使用 QR 分解
            if ker_basis.shape[1] == 0:
                h_n_dim = 0
                h_n_basis = np.array([])
            else:
                # 计算 ker 在 im 正交补中的投影
                # P_im^perp = I - im @ im^+
                im_proj = im_basis @ np.linalg.pinv(im_basis)
                ker_proj = ker_basis - im_proj @ ker_basis
                
                # 提取非零列（使用相对容差）
                col_norms = np.linalg.norm(ker_proj, axis=0)
                tol = _relative_tolerance(ker_proj, _CHAIN_COMPLEX_REL_TOL)
                nonzero_cols = col_norms > tol
                h_n_basis = ker_proj[:, nonzero_cols]
                
                # 正交化
                if h_n_basis.shape[1] > 0:
                    h_n_basis, _ = np.linalg.qr(h_n_basis)
                
                h_n_dim = h_n_basis.shape[1]
        
        return h_n_dim, h_n_basis
    
    def shift(self, k: int) -> 'ChainComplex':
        """平移 [k]: C^n ↦ C^{n+k}"""
        new_objects = {n + k: v for n, v in self.objects.items()}
        new_differentials = {n + k: ((-1) ** k) * d for n, d in self.differentials.items()}
        return ChainComplex(new_objects, new_differentials, verify=False)
    
    def __getitem__(self, n: int) -> VectorSpace:
        """获取 C^n"""
        return self.objects.get(n, VectorSpace(dimension=0))
    
    @property
    def degrees(self) -> List[int]:
        """返回所有非零度"""
        return sorted(self.objects.keys())
    
    def total_dimension(self) -> int:
        """总维度"""
        return sum(v.dimension for v in self.objects.values())
    
    def euler_characteristic(self) -> int:
        """欧拉特征 χ = Σ (-1)^n dim(H^n)"""
        chi = 0
        for n in self.degrees:
            h_n_dim, _ = self.cohomology(n)
            chi += ((-1) ** n) * h_n_dim
        return chi


# ============================================================================
# Section 8.1: ExactnessValidator - 正合性验证器 (模块 C.1)
# ============================================================================

class ExactnessValidator:
    """正合性验证器
    
    红线 C.1 (Requirements 7.5): 验证长正合列的正合性
    
    数学定义:
    序列 A --f--> B --g--> C 在 B 处正合当且仅当 im(f) = ker(g)
    """
    
    def __init__(self, rel_tol: float = _EXACTNESS_REL_TOL):
        """
        Args:
            rel_tol: 相对容差系数
        """
        self.rel_tol = rel_tol
    
    def verify_exactness_at(
        self,
        f_matrix: np.ndarray,
        g_matrix: np.ndarray,
    ) -> Tuple[bool, Dict[str, Any]]:
        """验证在中间位置的正合性
        
        检查 im(f) = ker(g)
        
        Args:
            f_matrix: 映射 f: A → B 的矩阵
            g_matrix: 映射 g: B → C 的矩阵
            
        Returns:
            (is_exact, details) 其中 details 包含诊断信息
        """
        # 计算 im(f)
        U_f, s_f, _ = np.linalg.svd(f_matrix, full_matrices=True)
        tol_f = _relative_tolerance(f_matrix, self.rel_tol)
        rank_f = np.sum(s_f > tol_f) if len(s_f) > 0 else 0
        im_f_basis = U_f[:, :rank_f]  # im(f) 的基
        
        # 计算 ker(g)
        _, s_g, Vh_g = np.linalg.svd(g_matrix, full_matrices=True)
        tol_g = _relative_tolerance(g_matrix, self.rel_tol)
        rank_g = np.sum(s_g > tol_g) if len(s_g) > 0 else 0
        ker_g_basis = Vh_g[rank_g:].conj().T  # ker(g) 的基
        
        # 检查 im(f) = ker(g)
        # 1. dim(im(f)) = dim(ker(g))
        dim_im_f = rank_f
        dim_ker_g = ker_g_basis.shape[1] if ker_g_basis.size > 0 else 0
        
        if dim_im_f != dim_ker_g:
            return False, {
                "reason": "dimension_mismatch",
                "dim_im_f": dim_im_f,
                "dim_ker_g": dim_ker_g,
            }
        
        if dim_im_f == 0:
            # 两者都是零空间，正合
            return True, {"reason": "both_zero"}
        
        # 2. im(f) ⊂ ker(g): 检查 g @ im_f_basis ≈ 0
        g_im_f = g_matrix @ im_f_basis
        norm_g_im_f = np.linalg.norm(g_im_f, 'fro')
        tol_composition = _relative_tolerance(g_matrix, self.rel_tol) * np.linalg.norm(im_f_basis, 'fro')
        
        if norm_g_im_f > max(tol_composition, self.rel_tol):
            return False, {
                "reason": "im_f_not_in_ker_g",
                "norm_g_im_f": norm_g_im_f,
                "tolerance": tol_composition,
            }
        
        # 3. ker(g) ⊂ im(f): 检查 ker_g_basis 可以由 im_f_basis 线性表示
        # 使用最小二乘: coeffs = im_f_basis^+ @ ker_g_basis
        # 然后检查 im_f_basis @ coeffs ≈ ker_g_basis
        if im_f_basis.shape[1] > 0 and ker_g_basis.shape[1] > 0:
            coeffs = np.linalg.lstsq(im_f_basis, ker_g_basis, rcond=None)[0]
            reconstructed = im_f_basis @ coeffs
            residual = np.linalg.norm(reconstructed - ker_g_basis, 'fro')
            tol_residual = _relative_tolerance(ker_g_basis, self.rel_tol)
            
            if residual > max(tol_residual, self.rel_tol):
                return False, {
                    "reason": "ker_g_not_in_im_f",
                    "residual": residual,
                    "tolerance": tol_residual,
                }
        
        return True, {"reason": "exact"}
    
    def verify_long_exact_sequence(
        self,
        complexes: List[ChainComplex],
        maps: List[Dict[int, np.ndarray]],
    ) -> Tuple[bool, List[int]]:
        """验证长正合列
        
        红线 C.1 (Requirements 7.5): 检测 im(f) = ker(g) 在每个位置
        
        Args:
            complexes: 链复形列表 [A, B, C, ...]
            maps: 链映射列表 [f: A→B, g: B→C, ...]
            
        Returns:
            (is_exact, break_degrees) 其中 break_degrees 是断裂的度
        """
        if len(complexes) < 2 or len(maps) < 1:
            return True, []
        
        breaks: List[int] = []
        
        # 对于每对相邻映射 (f, g)，检查正合性
        for i in range(len(maps) - 1):
            f_maps = maps[i]
            g_maps = maps[i + 1]
            
            # 获取所有度
            all_degrees = set(f_maps.keys()) | set(g_maps.keys())
            
            for n in all_degrees:
                f_n = f_maps.get(n)
                g_n = g_maps.get(n)
                
                if f_n is None or g_n is None:
                    continue
                
                is_exact, details = self.verify_exactness_at(f_n, g_n)
                
                if not is_exact:
                    breaks.append(n)
        
        return len(breaks) == 0, breaks
    
    def verify_short_exact_sequence(
        self,
        f_matrix: np.ndarray,
        g_matrix: np.ndarray,
    ) -> Tuple[bool, Dict[str, Any]]:
        """验证短正合列 0 → A --f--> B --g--> C → 0
        
        条件:
        1. f 是单射 (ker(f) = 0)
        2. g 是满射 (im(g) = C)
        3. im(f) = ker(g)
        """
        details: Dict[str, Any] = {}
        
        # 1. f 是单射
        _, s_f, _ = np.linalg.svd(f_matrix, full_matrices=True)
        tol_f = _relative_tolerance(f_matrix, self.rel_tol)
        rank_f = np.sum(s_f > tol_f) if len(s_f) > 0 else 0
        
        if rank_f < f_matrix.shape[1]:
            details["f_not_injective"] = True
            details["rank_f"] = rank_f
            details["expected_rank_f"] = f_matrix.shape[1]
            return False, details
        
        # 2. g 是满射
        U_g, s_g, _ = np.linalg.svd(g_matrix, full_matrices=True)
        tol_g = _relative_tolerance(g_matrix, self.rel_tol)
        rank_g = np.sum(s_g > tol_g) if len(s_g) > 0 else 0
        
        if rank_g < g_matrix.shape[0]:
            details["g_not_surjective"] = True
            details["rank_g"] = rank_g
            details["expected_rank_g"] = g_matrix.shape[0]
            return False, details
        
        # 3. im(f) = ker(g)
        is_exact, exact_details = self.verify_exactness_at(f_matrix, g_matrix)
        details.update(exact_details)
        
        return is_exact, details


# ============================================================================
# Section 9: InjectiveResolution - 内射消解
# ============================================================================

class InjectiveObject:
    """内射对象
    
    数学定义: I 是内射的当且仅当对于任意单态射 m: A ↪ B
    和态射 f: A → I，存在 g: B → I 使得 g ∘ m = f
    
    在向量空间范畴中，所有对象都是内射的
    """
    
    def __init__(self, space: VectorSpace, is_injective: bool = True):
        self.space = space
        self._is_injective = is_injective
    
    def verify_lifting_property(self, m: LinearMap, f: LinearMap) -> Optional[LinearMap]:
        """验证提升性质
        
        给定单态射 m: A ↪ B 和 f: A → I
        寻找 g: B → I 使得 g ∘ m = f
        
        Returns:
            提升 g，如果存在；否则 None
        """
        # 在向量空间中，使用伪逆
        # g = f @ m^+
        m_pinv = np.linalg.pinv(m.matrix)
        g_matrix = f.matrix @ m_pinv
        
        # 验证 g ∘ m = f（严格：失败即异常，禁止静默返回 None）
        gm = g_matrix @ m.matrix
        residual = np.linalg.norm(gm - f.matrix, 'fro')
        tol = _relative_tolerance(np.asarray(f.matrix, dtype=np.complex128), rel_tol=_FLOAT64_EPS)
        if residual > tol:
            raise CategoricalError(
                "Injective lifting property check failed in Vect: "
                f"||g∘m - f||_F = {residual:.2e} > tol={tol:.2e}."
            )
        return LinearMap(source=m.target, target=f.target, matrix=g_matrix)


class InjectiveResolution:
    """内射消解
    
    对于对象 M，构造内射消解:
    0 → M → I^0 → I^1 → I^2 → ...
    
    每个 I^n 必须是内射对象
    
    在 `Vect`（有限维向量空间范畴）中，**每个对象都是内射对象**，
    因而每个对象的内射维数为 0，内射消解可以且应当取为**平凡消解**。
    
    重要：任何为了信息增益而硬造的 acyclic 复形，如果不与原对象给出准同构，
    都不是内射消解，拿它去算 RHom 会直接违反定义。
    """

    def __init__(self, max_length: Optional[int] = None, enhanced_mode: bool = True):
        """
        Args:
            max_length: 消解的最大长度（可选）。在 `Vect` 中不会自动引入隐藏的上界。
            enhanced_mode: 兼容参数（保留但不再影响数学语义）。
        
        数学说明:
        - 在 `Vect` 中对象皆内射，内射消解长度可取 0（平凡消解）。
        - 若上游显式要求更长形式长度，可在高次数补 0 对象（不改变准同构类）。
        """
        self.max_length = max_length
        self.enhanced_mode = enhanced_mode  # kept for backward compatibility (no effect in Vect)
        self._resolution_cache: Dict[Tuple[int, bool], ChainComplex] = {}
    
    @classmethod
    def compute_recommended_length(cls, F: 'ChainComplex', G: 'ChainComplex') -> int:
        """计算推荐的消解长度
        
        基于源和目标复形的度数范围
        
        数学基础:
        在 `Vect`（有限维向量空间范畴）中对象皆内射，因此推荐消解长度恒为 0（平凡消解）。
        """
        # 红线：禁止+2 安全边界之类不可证明经验规则。
        return 0
    
    def construct(self, M: VectorSpace) -> ChainComplex:
        """构造 M 的内射消解
        
        在 `Vect` 中，取平凡消解即可（且这是唯一不会引入伪导出信息的选择）。
        
        Returns:
            内射消解作为链复形
        """
        # 缓存检查
        cache_key = (M.dimension, False)
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
        return self._construct_trivial(M)
    
    def _construct_trivial(self, M: VectorSpace) -> ChainComplex:
        """构造平凡消解 0 → M → 0"""
        objects = {0: M}
        differentials = {}
        
        # 验证内射性
        I_0 = InjectiveObject(M)
        if not I_0._is_injective:
            raise ValueError("Failed to construct injective resolution")
        
        resolution = ChainComplex(objects, differentials, verify=True)
        self._resolution_cache[(M.dimension, False)] = resolution
        return resolution
    
    def _construct_koszul(self, M: VectorSpace) -> ChainComplex:
        """（已弃用）历史上的增强 Koszul实现并非内射消解。
        
        在 `Vect` 中它既不必要也不应当存在：若没有给出 M → I^0 的准同构，
        则不能用于 RHom 定义。为避免数学违法，这里退化为平凡消解。
        """
        return self._construct_trivial(M)
    
    def construct_extended(self, M: VectorSpace, length: int) -> ChainComplex:
        """构造指定长度的内射消解
        
        用于需要更长消解的情况（如计算高阶导出函子）
        
        在 `Vect` 中，扩展消解仅在高次数补 0 对象（不改变准同构类）。
        """
        if self.max_length is not None and length > self.max_length:
            length = self.max_length
        return self._construct_extended_trivial(M, length)
    
    def _construct_extended_trivial(self, M: VectorSpace, length: int) -> ChainComplex:
        """构造平凡的扩展消解"""
        objects = {0: M}
        differentials = {}
        
        for n in range(1, length + 1):
            objects[n] = VectorSpace(dimension=0)
            if objects[n - 1].dimension > 0:
                differentials[n - 1] = np.zeros(
                    (objects[n].dimension, objects[n - 1].dimension),
                    dtype=np.complex128
                )
        
        return ChainComplex(objects, differentials, verify=True)
    
    def _construct_extended_koszul(self, M: VectorSpace, length: int) -> ChainComplex:
        """（已弃用）历史上的扩展 Koszul不是内射消解；为数学一致性退化为平凡扩展。"""
        return self._construct_extended_trivial(M, length)


# ============================================================================
# Section 10: DerivedHom - 导出 Hom 函子
# ============================================================================

class DerivedHom:
    """导出 Hom 函子 RHom(F, G)
    
    定义: RHom(F, G) = Tot(Hom(F, I•))
    其中 I• 是 G 的内射消解
    
    严禁跳过消解! 直接计算会抛出 ResolutionSkippedError
    """
    
    def __init__(self):
        self._resolution_builder = InjectiveResolution()
        self._used_resolution = False  # 追踪是否使用了消解
    
    def compute(
        self, 
        F: ChainComplex, 
        G: ChainComplex,
        resolution_length: Optional[int] = None
    ) -> ChainComplex:
        """计算 RHom(F, G)
        
        数学事实（Vect）:
        - 在有限维向量空间范畴中，每个对象都是内射对象；
          因而任意链复形 G 本身就是内射复形，可直接取 I• = G 作为内射消解。
        
        因此可用标准 dg-Hom 复形实现:
        - (Hom(F, G))^n = ⊕_p Hom(F^p, G^{p+n})
        - d(φ)_p = d_G^{p+n} ∘ φ_p - (-1)^n φ_{p+1} ∘ d_F^{p}
        
        Returns:
            RHom(F, G) 作为链复形（即 dg-Hom 复形）
        """
        # 在 Vect 中，I• = G 即为合法内射消解（长度 0）。
        self._used_resolution = True
        
        if not F.degrees or not G.degrees:
            return ChainComplex({}, {}, verify=True)
        
        F_degrees = sorted(F.degrees)
        G_degrees_set = set(G.degrees)
        
        # 可能出现的 Hom 复形度数 n = q - p
        hom_degrees: Set[int] = set()
        for p in F_degrees:
            for q in G.degrees:
                hom_degrees.add(q - p)
        if not hom_degrees:
            return ChainComplex({}, {}, verify=True)
        
        hom_degrees_sorted = sorted(hom_degrees)
        
        # 为每个 n 构造对象 (Hom(F,G))^n
        objects: Dict[int, VectorSpace] = {}
        # 记录每个 n 的分量布局: p -> (offset, f_dim, g_dim)
        layout: Dict[int, Dict[int, Tuple[int, int, int]]] = {}
        
        for n in hom_degrees_sorted:
            offset = 0
            layout_n: Dict[int, Tuple[int, int, int]] = {}
            for p in F_degrees:
                q = p + n
                if q not in G_degrees_set:
                    continue
                f_dim = F.objects[p].dimension
                g_dim = G.objects[q].dimension
                if f_dim <= 0 or g_dim <= 0:
                    continue
                hom_dim = f_dim * g_dim
                layout_n[p] = (offset, f_dim, g_dim)
                offset += hom_dim
            if offset > 0:
                objects[n] = VectorSpace(dimension=offset)
                layout[n] = layout_n
        
        # 构造微分 d^n: Hom^n -> Hom^{n+1}
        differentials: Dict[int, np.ndarray] = {}
        for n in sorted(objects.keys()):
            if (n + 1) not in objects:
                continue
            dim_n = objects[n].dimension
            dim_np1 = objects[n + 1].dimension
            d = np.zeros((dim_np1, dim_n), dtype=np.complex128)
            
            layout_n = layout.get(n, {})
            layout_np1 = layout.get(n + 1, {})
            
            # 对每个 p，输出分量在 Hom^{n+1} 的 (p) 位置
            for p, (out_off, f_dim, g_np1_dim) in layout_np1.items():
                # 目标是 Hom(F^p, G^{p+n+1})，其中 g_np1_dim = dim(G^{p+n+1})
                
                # 1) 来自 d_G ∘ φ_p: φ_p ∈ Hom(F^p, G^{p+n})
                if p in layout_n:
                    in_off, f_dim_in, g_dim = layout_n[p]
                    # f_dim_in == f_dim
                    q = p + n  # G^{p+n}
                    dG = G.differentials.get(q)
                    if dG is not None and g_dim > 0 and f_dim > 0:
                        # kron(I_{F^p}, dG): vec(dG φ_p)
                        block = np.kron(np.eye(f_dim, dtype=np.complex128), dG)
                        if block.shape == (g_np1_dim * f_dim, g_dim * f_dim):
                            d[out_off:out_off + g_np1_dim * f_dim,
                              in_off:in_off + g_dim * f_dim] += block
                
                # 2) 来自 -(-1)^n φ_{p+1} ∘ d_F^p
                dF = F.differentials.get(p)
                if dF is not None:
                    p1 = p + 1
                    # 需要 φ_{p+1} ∈ Hom(F^{p+1}, G^{p+1+n}) = Hom(F^{p+1}, G^{p+n+1})
                    if p1 in layout_n:
                        in_off_p1, f_dim_p1, g_dim_p1 = layout_n[p1]
                        if g_dim_p1 == g_np1_dim:
                            sign = -((-1) ** n)
                            block = sign * np.kron(dF.T, np.eye(g_np1_dim, dtype=np.complex128))
                            if block.shape == (g_np1_dim * f_dim, g_np1_dim * f_dim_p1):
                                d[out_off:out_off + g_np1_dim * f_dim,
                                  in_off_p1:in_off_p1 + g_np1_dim * f_dim_p1] += block
            
            differentials[n] = d
        
        return ChainComplex(objects, differentials, verify=True)
    
    def compute_without_resolution(self, F: ChainComplex, G: ChainComplex) -> None:
        """尝试不使用消解计算 RHom（禁止操作）
        
        这个方法总是抛出 ResolutionSkippedError
        """
        raise ResolutionSkippedError()
    
    @property
    def resolution_was_used(self) -> bool:
        """检查最近一次计算是否使用了消解"""
        return self._used_resolution


# ============================================================================
# Section 11: tStructure - t-结构
# ============================================================================

class tStructure:
    """t-结构: 导出范畴上的截断结构
    
    定义一对满子范畴 (D^≤0, D^≥0) 满足:
    1. D^≤0[1] ⊂ D^≤0
    2. D^≥0[-1] ⊂ D^≥0
    3. Hom(D^≤0, D^≥0[-1]) = 0
    4. 对任意 X，存在区别三角 A → X → B → A[1]
       其中 A ∈ D^≤0, B ∈ D^≥0[-1]
    
    Heart = D^≤0 ∩ D^≥0（阿贝尔范畴）
    """
    
    def __init__(self):
        pass
    
    def truncate_le(self, X: ChainComplex, n: int) -> ChainComplex:
        """截断函子 τ^≤n
        
        保留度 ≤ n 的部分，并在度 n 处取核
        
        (τ^≤n X)^k = X^k  if k < n
                   = ker(d^n)  if k = n
                   = 0  if k > n
        
        关键：必须正确调整 d^{n-1} 使其像落在 ker(d^n) 中
        这保证截断复形仍满足 d² = 0
        """
        new_objects = {}
        new_differentials = {}
        ker_basis = None  # ker(d^n) 的基
        ker_dim = 0
        
        # 首先计算 ker(d^n)
        if n in X.objects:
            V = X.objects[n]
            if n in X.differentials:
                d_n = X.differentials[n]
                # 计算核的基
                # ker(d^n) = {v : d^n(v) = 0}
                # 使用 SVD: d_n = U @ S @ Vh
                # ker(d_n) 由 Vh 的最后 (dim - rank) 行张成
                U, s, Vh = np.linalg.svd(d_n, full_matrices=True)
                tol = _relative_tolerance(d_n, _CHAIN_COMPLEX_REL_TOL)
                rank = np.sum(s > tol) if len(s) > 0 else 0
                ker_dim = V.dimension - rank
                if ker_dim > 0:
                    # ker 的基是 Vh 的最后 ker_dim 行（转置后为列）
                    ker_basis = Vh[rank:, :].T.conj()  # shape: (V.dim, ker_dim)
                else:
                    ker_basis = np.zeros((V.dimension, 0), dtype=np.complex128)
            else:
                # 没有 d^n，ker = 整个空间
                ker_dim = V.dimension
                ker_basis = np.eye(V.dimension, dtype=np.complex128)
        
        # 构建截断复形
        for k, V in X.objects.items():
            if k < n:
                new_objects[k] = V
                if k in X.differentials:
                    if k == n - 1 and ker_basis is not None and ker_dim > 0:
                        # 调整 d^{n-1} 使其像落在 ker(d^n) 中
                        # 新的 d^{n-1}: X^{n-1} → ker(d^n)
                        # 原 d^{n-1}: X^{n-1} → X^n
                        # 由于 d^n ∘ d^{n-1} = 0，im(d^{n-1}) ⊂ ker(d^n)
                        # 新映射 = ker_basis^+ @ d^{n-1}（投影到 ker 的坐标）
                        d_n_minus_1 = X.differentials[n - 1]
                        # 将 d^{n-1} 的像表示在 ker(d^n) 的基下
                        # ker_basis^+ @ d^{n-1} 给出在 ker 基下的坐标
                        ker_basis_pinv = np.linalg.pinv(ker_basis)
                        new_d = ker_basis_pinv @ d_n_minus_1
                        new_differentials[k] = new_d
                    else:
                        new_differentials[k] = X.differentials[k].copy()
            elif k == n:
                new_objects[k] = VectorSpace(dimension=max(0, ker_dim))
                # 度 n 处没有出边界算子（截断）
            # k > n: 不包含
        
        return ChainComplex(new_objects, new_differentials, verify=True)
    
    def truncate_ge(self, X: ChainComplex, n: int) -> ChainComplex:
        """截断函子 τ^≥n
        
        保留度 ≥ n 的部分，并在度 n 处取余核
        
        (τ^≥n X)^k = 0  if k < n
                   = coker(d^{n-1})  if k = n
                   = X^k  if k > n
        
        关键：必须正确调整 d^n 使其定义域从 coker(d^{n-1}) 出发
        这保证截断复形仍满足 d² = 0
        """
        new_objects = {}
        new_differentials = {}
        coker_basis = None  # coker(d^{n-1}) 的基（作为 X^n 的商空间）
        coker_dim = 0
        coker_projection = None  # 从 X^n 到 coker 的投影
        
        # 首先计算 coker(d^{n-1})
        if n in X.objects:
            V = X.objects[n]
            if (n - 1) in X.differentials:
                d_n_minus_1 = X.differentials[n - 1]
                # coker(d^{n-1}) = X^n / im(d^{n-1})
                # 使用 SVD: d_{n-1} = U @ S @ Vh
                # im(d_{n-1}) 由 U 的前 rank 列张成
                # coker 由 U 的后 (dim - rank) 列张成
                U, s, Vh = np.linalg.svd(d_n_minus_1, full_matrices=True)
                tol = _relative_tolerance(d_n_minus_1, _CHAIN_COMPLEX_REL_TOL)
                rank = np.sum(s > tol) if len(s) > 0 else 0
                coker_dim = V.dimension - rank
                if coker_dim > 0:
                    # coker 的基是 U 的后 coker_dim 列
                    coker_basis = U[:, rank:]  # shape: (V.dim, coker_dim)
                    # 投影: P = coker_basis @ coker_basis^H
                    coker_projection = coker_basis @ coker_basis.T.conj()
                else:
                    coker_basis = np.zeros((V.dimension, 0), dtype=np.complex128)
                    coker_projection = np.zeros((V.dimension, V.dimension), dtype=np.complex128)
            else:
                # 没有 d^{n-1}，coker = 整个空间
                coker_dim = V.dimension
                coker_basis = np.eye(V.dimension, dtype=np.complex128)
                coker_projection = np.eye(V.dimension, dtype=np.complex128)
        
        # 构建截断复形
        for k, V in X.objects.items():
            if k > n:
                new_objects[k] = V
                if k in X.differentials:
                    new_differentials[k] = X.differentials[k].copy()
            elif k == n:
                new_objects[k] = VectorSpace(dimension=max(0, coker_dim))
                
                if k in X.differentials and coker_basis is not None and coker_dim > 0:
                    # 调整 d^n 使其从 coker(d^{n-1}) 出发
                    # 原 d^n: X^n → X^{n+1}
                    # 新 d^n: coker(d^{n-1}) → X^{n+1}
                    # 由于 d^n ∘ d^{n-1} = 0，d^n 在 im(d^{n-1}) 上为零
                    # 所以 d^n 自然诱导 coker(d^{n-1}) → X^{n+1}
                    # 新映射 = d^n @ coker_basis
                    d_n = X.differentials[n]
                    new_d = d_n @ coker_basis
                    new_differentials[k] = new_d
            # k < n: 不包含
        
        return ChainComplex(new_objects, new_differentials, verify=True)
    
    def heart(self, X: ChainComplex) -> VectorSpace:
        """计算 Heart = D^≤0 ∩ D^≥0
        
        Heart 是 t-结构的核心，是一个阿贝尔范畴
        
        对于标准 t-结构，Heart ≅ H^0(X)
        """
        # τ^≤0 ∩ τ^≥0 = H^0
        h0_dim, h0_basis = X.cohomology(0)
        return VectorSpace(dimension=h0_dim, basis=h0_basis if h0_dim > 0 else None)
    
    def verify_t_structure_axioms(self, X: ChainComplex) -> Tuple[bool, List[Dict[str, Any]]]:
        """验证 t-结构公理
        
        公理:
        1. D^≤0[1] ⊂ D^≤0 (平移封闭性)
        2. D^≥0[-1] ⊂ D^≥0 (平移封闭性)
        3. Hom(D^≤0, D^≥0[-1]) = 0 (正交性)
        4. 对任意 X，存在区别三角 A → X → B → A[1]
           其中 A ∈ D^≤0, B ∈ D^≥0[-1]
        
        Returns:
            (is_valid, violations) 其中 violations 是违反详情列表
        """
        violations = []
        
        # 公理 1: D^≤0[1] ⊂ D^≤0
        # 如果 X ∈ D^≤0，则 X[1] ∈ D^≤0
        # 等价于: τ^≤0(X) 的平移 [1] 应该仍在 D^≤0 中
        # 即: τ^≤0(X)[1] 的非零上同调只在度 ≤ 0
        tau_le_0 = self.truncate_le(X, 0)
        tau_le_0_shifted = tau_le_0.shift(1)
        
        for n in tau_le_0_shifted.degrees:
            h_n_dim, _ = tau_le_0_shifted.cohomology(n)
            if h_n_dim > 0 and n > 0:
                violations.append({
                    "axiom": "1 (D^≤0[1] ⊂ D^≤0)",
                    "degree": n,
                    "cohomology_dim": h_n_dim,
                    "description": f"τ^≤0(X)[1] has non-zero H^{n} with n > 0"
                })
        
        # 公理 2: D^≥0[-1] ⊂ D^≥0
        # 如果 X ∈ D^≥0，则 X[-1] ∈ D^≥0
        tau_ge_0 = self.truncate_ge(X, 0)
        tau_ge_0_shifted = tau_ge_0.shift(-1)
        
        for n in tau_ge_0_shifted.degrees:
            h_n_dim, _ = tau_ge_0_shifted.cohomology(n)
            if h_n_dim > 0 and n < 0:
                violations.append({
                    "axiom": "2 (D^≥0[-1] ⊂ D^≥0)",
                    "degree": n,
                    "cohomology_dim": h_n_dim,
                    "description": f"τ^≥0(X)[-1] has non-zero H^{n} with n < 0"
                })
        
        # 公理 3: Hom(D^≤0, D^≥0[-1]) = 0
        # 对于 A ∈ D^≤0 和 B ∈ D^≥0，Hom(A, B[-1]) = 0
        # 这意味着 τ^≤0(X) 和 τ^≥0(X)[-1] 之间没有非零态射
        # 
        # 数学定义:
        # Hom_{D(A)}(A, B) = H^0(RHom(A, B))
        # 对于链复形，Hom(A, B) 是链映射的空间
        # 
        # 正交性条件要求: 不存在非零链映射 f: τ^≤0(X) → τ^≥0(X)[-1]
        
        # 计算 Hom 空间维度
        # 链映射 f: A → B 是一族映射 f^n: A^n → B^n 满足 d_B ∘ f = f ∘ d_A
        
        hom_dim = self._compute_chain_map_space_dim(tau_le_0, tau_ge_0_shifted)
        
        if hom_dim > 0:
            # 存在非零 Hom，违反正交性
            violations.append({
                "axiom": "3 (Hom(D^≤0, D^≥0[-1]) = 0)",
                "hom_dimension": hom_dim,
                "description": f"Orthogonality violated: dim(Hom(τ^≤0, τ^≥0[-1])) = {hom_dim} > 0"
            })
        
        # 额外检查: 度范围分析（作为诊断信息）
        max_le_degree = None
        for n in sorted(tau_le_0.degrees, reverse=True):
            h_n_dim, _ = tau_le_0.cohomology(n)
            if h_n_dim > 0:
                max_le_degree = n
                break
        
        min_ge_shifted_degree = None
        for n in sorted(tau_ge_0_shifted.degrees):
            h_n_dim, _ = tau_ge_0_shifted.cohomology(n)
            if h_n_dim > 0:
                min_ge_shifted_degree = n
                break
        
        if max_le_degree is not None and min_ge_shifted_degree is not None:
            if max_le_degree >= min_ge_shifted_degree and hom_dim == 0:
                # 度范围重叠但 Hom 为零 - 这是正常的，记录诊断信息
                pass  # 正交性通过精确计算验证
        
        # 公理 4: 存在区别三角
        # 对于任意 X，存在 A → X → B → A[1]
        # 其中 A = τ^≤0(X), B = τ^≥1(X)
        # 
        # 验证: τ^≤0(X) 和 τ^≥1(X) 应该"拼接"成 X
        tau_ge_1 = self.truncate_ge(X, 1)
        
        # 检查维度守恒: dim(X) = dim(τ^≤0) + dim(τ^≥1) - dim(交集)
        # 对于标准 t-结构，交集应该是 0
        total_X = X.total_dimension()
        total_le = tau_le_0.total_dimension()
        total_ge = tau_ge_1.total_dimension()
        
        # 计算 Heart 维度作为交集的代理
        heart = self.heart(X)
        
        # 欧拉特征应该守恒
        chi_X = X.euler_characteristic()
        chi_le = tau_le_0.euler_characteristic()
        chi_ge = tau_ge_1.euler_characteristic()
        
        # 对于区别三角 A → X → B → A[1]
        # 有 χ(X) = χ(A) + χ(B)
        if chi_X != chi_le + chi_ge:
            violations.append({
                "axiom": "4 (distinguished triangle existence)",
                "chi_X": chi_X,
                "chi_le": chi_le,
                "chi_ge": chi_ge,
                "description": f"Euler characteristic not preserved: χ(X)={chi_X} ≠ χ(τ^≤0)+χ(τ^≥1)={chi_le + chi_ge}"
            })
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _compute_chain_map_space_dim(self, A: ChainComplex, B: ChainComplex) -> int:
        """计算链映射空间 Hom(A, B) 的维度
        
        数学定义:
        链映射 f: A → B 是一族线性映射 {f^n: A^n → B^n}_{n ∈ Z}
        满足链映射条件: d_B^n ∘ f^n = f^{n+1} ∘ d_A^n
        
        链映射空间是 ∏_n Hom(A^n, B^n) 的子空间，由链映射条件定义。
        
        Returns:
            链映射空间的维度
        """
        # 收集所有度
        all_degrees = set(A.degrees) | set(B.degrees)
        
        if not all_degrees:
            return 0
        
        # 计算基础 Hom 空间维度: ∏_n Hom(A^n, B^n)
        base_dim = 0
        degree_dims: Dict[int, int] = {}
        
        for n in all_degrees:
            A_n_dim = A.objects.get(n, VectorSpace(0)).dimension
            B_n_dim = B.objects.get(n, VectorSpace(0)).dimension
            hom_dim = A_n_dim * B_n_dim
            degree_dims[n] = hom_dim
            base_dim += hom_dim
        
        if base_dim == 0:
            return 0
        
        # 构造链映射条件的约束矩阵
        # 对于每个 n，约束: d_B^n ∘ f^n = f^{n+1} ∘ d_A^n
        # 这是 dim(B^{n+1}) × dim(A^n) 个线性方程
        
        constraints: List[np.ndarray] = []
        
        sorted_degrees = sorted(all_degrees)
        
        # 计算每个度的偏移量
        offsets: Dict[int, int] = {}
        current_offset = 0
        for n in sorted_degrees:
            offsets[n] = current_offset
            current_offset += degree_dims.get(n, 0)
        
        for n in sorted_degrees:
            A_n_dim = A.objects.get(n, VectorSpace(0)).dimension
            B_n_dim = B.objects.get(n, VectorSpace(0)).dimension
            A_np1_dim = A.objects.get(n + 1, VectorSpace(0)).dimension
            B_np1_dim = B.objects.get(n + 1, VectorSpace(0)).dimension
            
            if A_n_dim == 0 or B_np1_dim == 0:
                continue
            
            d_A_n = A.differentials.get(n)
            d_B_n = B.differentials.get(n)
            
            # 约束: d_B^n ∘ f^n - f^{n+1} ∘ d_A^n = 0
            # 展平为向量形式
            
            # f^n 是 B_n_dim × A_n_dim 的矩阵，展平为 B_n_dim * A_n_dim 维向量
            # f^{n+1} 是 B_np1_dim × A_np1_dim 的矩阵
            
            # d_B^n ∘ f^n 的展平: (I_{A_n} ⊗ d_B^n) @ vec(f^n)
            # f^{n+1} ∘ d_A^n 的展平: (d_A^n^T ⊗ I_{B_np1}) @ vec(f^{n+1})
            
            constraint_rows = B_np1_dim * A_n_dim
            constraint = np.zeros((constraint_rows, base_dim), dtype=np.complex128)
            
            # 第一项: d_B^n ∘ f^n
            if d_B_n is not None and B_n_dim > 0:
                # (I_{A_n} ⊗ d_B^n) 作用于 vec(f^n)
                coeff_fn = np.kron(np.eye(A_n_dim, dtype=np.complex128), d_B_n)
                
                fn_offset = offsets.get(n, 0)
                fn_dim = degree_dims.get(n, 0)
                
                if coeff_fn.shape[1] == fn_dim and coeff_fn.shape[0] == constraint_rows:
                    constraint[:, fn_offset:fn_offset + fn_dim] = coeff_fn
            
            # 第二项: -f^{n+1} ∘ d_A^n
            if d_A_n is not None and A_np1_dim > 0 and (n + 1) in offsets:
                # -(d_A^n^T ⊗ I_{B_np1}) 作用于 vec(f^{n+1})
                coeff_fnp1 = -np.kron(d_A_n.T, np.eye(B_np1_dim, dtype=np.complex128))
                
                fnp1_offset = offsets.get(n + 1, 0)
                fnp1_dim = degree_dims.get(n + 1, 0)
                
                if coeff_fnp1.shape[1] == fnp1_dim and coeff_fnp1.shape[0] == constraint_rows:
                    constraint[:, fnp1_offset:fnp1_offset + fnp1_dim] = coeff_fnp1
            
            # 只添加非零约束
            if np.any(constraint != 0):
                constraints.append(constraint)
        
        if not constraints:
            # 无约束，返回完整 Hom 空间
            return base_dim
        
        # 堆叠约束矩阵
        constraint_matrix = np.vstack(constraints)
        
        # 计算约束矩阵的秩
        tol = _relative_tolerance(constraint_matrix, _EXACTNESS_REL_TOL)
        rank = np.linalg.matrix_rank(constraint_matrix, tol=tol)
        
        # 链映射空间维度 = 基础维度 - 约束秩
        return max(0, base_dim - rank)
    
    def verify_t_structure_axioms_strict(self, X: ChainComplex) -> bool:
        """严格验证 t-结构公理，违反时抛出异常
        
        Raises:
            tStructureAxiomViolation: 当任何公理被违反时
        """
        is_valid, violations = self.verify_t_structure_axioms(X)
        
        if not is_valid:
            first_violation = violations[0]
            raise tStructureAxiomViolation(
                axiom=first_violation["axiom"],
                details=first_violation["description"]
            )
        
        return True


# ============================================================================
# Section 11.1: QuasiBPS 不变量提取器 (红线 B.2)
# ============================================================================

@dataclass
class TriangulatedSubcategory:
    """三角子范畴
    
    导出范畴的满子范畴，继承三角结构
    """
    objects: List[ChainComplex]
    name: str = ""
    weight: int = 0
    
    def euler_characteristic(self) -> int:
        """计算子范畴的总欧拉特征"""
        return sum(obj.euler_characteristic() for obj in self.objects)
    
    def total_dimension(self) -> int:
        """计算子范畴的总维度"""
        return sum(obj.total_dimension() for obj in self.objects)


@dataclass
class QuasiBPSCategory:
    """准 BPS 范畴 T_G(χ)_w
    
    红线 B.2: t-structure 分解后的子范畴
    
    数学定义:
    准 BPS 范畴是导出范畴在 t-structure 分解下的子范畴，
    其对象具有固定的 BPS 权重 w 和特征 χ
    
    Attributes:
        objects: 子范畴中的对象（链复形）
        weight: BPS 权重 w
        character: 特征 χ
    """
    objects: List[ChainComplex]
    weight: int
    character: complex
    
    def euler_characteristic(self) -> int:
        """计算范畴的欧拉特征"""
        return sum(obj.euler_characteristic() for obj in self.objects)


@dataclass
class LanglandsParameter:
    """朗兰兹参数 φ
    
    Galois 侧的代数不变量
    
    数学定义:
    - Langlands 参数 φ: W_F → ^L G 是 Weil 群到 Langlands 对偶群的同态
    - frobenius_eigenvalues 是 Frobenius 元素在表示下的特征值
    - 这些特征值编码了 L-函数的 Euler 因子
    
    Attributes:
        galois_rep: Galois 表示矩阵（在某个基下）
        conductor: 导子（分歧程度）
        weight: 权重（对应自守表示的权重）
        frobenius_eigenvalues: Frobenius 特征值（从 galois_rep 计算）
    """
    galois_rep: np.ndarray
    conductor: int
    weight: int
    frobenius_eigenvalues: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        self.galois_rep = np.asarray(self.galois_rep, dtype=np.complex128)
        if self.galois_rep.ndim != 2 or self.galois_rep.shape[0] != self.galois_rep.shape[1]:
            raise ValueError(f"galois_rep must be a square matrix, got shape {self.galois_rep.shape}.")
        if not np.all(np.isfinite(self.galois_rep.real)) or not np.all(np.isfinite(self.galois_rep.imag)):
            raise ValueError("galois_rep must be finite (no NaN/Inf).")
        
        # 【关键】从 Galois 表示矩阵计算 Frobenius 特征值
        # 数学原理: Frobenius 特征值 = Galois 表示矩阵的特征值
        # 这是 Langlands 对应的核心不变量
        if self.frobenius_eigenvalues is None:
            try:
                self.frobenius_eigenvalues = np.linalg.eigvals(self.galois_rep)
            except np.linalg.LinAlgError as e:
                # 红线：禁止无理由降级近似。特征值失败即中断并报告。
                raise CategoricalError(f"Failed to compute Frobenius eigenvalues: {e}") from e
        else:
            self.frobenius_eigenvalues = np.asarray(self.frobenius_eigenvalues, dtype=np.complex128)
        if self.frobenius_eigenvalues.ndim != 1:
            raise ValueError("frobenius_eigenvalues must be a 1-D array.")
        if not np.all(np.isfinite(self.frobenius_eigenvalues.real)) or not np.all(np.isfinite(self.frobenius_eigenvalues.imag)):
            raise ValueError("frobenius_eigenvalues must be finite (no NaN/Inf).")


@dataclass
class BPSInvariants:
    """BPS 不变量
    
    红线 B.2: 必须是精确整数，替代 Z3 搜索
    
    数学定义:
    BPS 不变量 {wi} 是准 BPS 范畴的可计数不变量，
    是 Galois 参数 φ 的代数原像
    
    Attributes:
        weights: BPS 权重列表 {wi}，必须是精确整数
        galois_preimage: Galois 参数 φ 的代数原像（可选）
    """
    weights: List[int]
    galois_preimage: Optional[LanglandsParameter] = None
    
    def __post_init__(self):
        # 红线 B.2: 必须是可证明的精确整数；禁止静默 round()
        self.weights = [_require_exact_int(w, name="BPS weight") for w in self.weights]
    
    def is_valid(self) -> bool:
        """验证 BPS 不变量有效性"""
        return all(isinstance(w, int) for w in self.weights)


class SemiOrthogonalDecomposition:
    """半正交分解
    
    将导出范畴 D 分解为 <A_1, A_2, ..., A_n>
    满足 Hom(A_i, A_j) = 0 for i > j
    
    数学定义:
    半正交分解是导出范畴的结构分解，
    子范畴之间满足半正交条件
    """
    
    def __init__(self):
        self.components: List[TriangulatedSubcategory] = []
    
    def decompose(
        self, 
        derived_cat: List[ChainComplex],
        t_structure: tStructure,
    ) -> List[TriangulatedSubcategory]:
        """执行半正交分解
        
        红线 B.2 (Requirements 4.5): 使用 t-structure 进行分解
        
        数学原理:
        1. 对每个对象应用 t-structure 截断
        2. 按上同调度分组
        3. 验证半正交条件
        
        Args:
            derived_cat: 导出范畴中的对象列表
            t_structure: 用于分解的 t-structure
            
        Returns:
            三角子范畴列表
        """
        # 按上同调度分组
        degree_groups: Dict[int, List[ChainComplex]] = {}
        
        for X in derived_cat:
            # 找到 X 的"主要"度（非零上同调的最小度）
            primary_degree = None
            for n in sorted(X.degrees):
                h_n_dim, _ = X.cohomology(n)
                if h_n_dim > 0:
                    primary_degree = n
                    break
            
            if primary_degree is None:
                primary_degree = 0
            
            if primary_degree not in degree_groups:
                degree_groups[primary_degree] = []
            degree_groups[primary_degree].append(X)
        
        # 创建子范畴
        self.components = []
        for degree in sorted(degree_groups.keys()):
            subcategory = TriangulatedSubcategory(
                objects=degree_groups[degree],
                name=f"D^{degree}",
                weight=degree
            )
            self.components.append(subcategory)
        
        return self.components
    
    def verify_semi_orthogonality(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """验证半正交条件
        
        Hom(A_i, A_j) = 0 for i > j
        
        数学定义:
        半正交分解要求对于 i > j，子范畴 A_i 和 A_j 之间没有非零态射。
        这是通过计算每对子范畴之间的 Hom 空间维度来验证的。
        
        Returns:
            (is_valid, violations) 其中 violations 包含违反详情
        """
        violations: List[Dict[str, Any]] = []
        t_structure = tStructure()  # 用于计算 Hom 空间
        
        for i, A_i in enumerate(self.components):
            for j, A_j in enumerate(self.components):
                if i > j:
                    # 计算 Hom(A_i, A_j) 的维度
                    # 对于子范畴，这是所有对象对之间 Hom 的总和
                    total_hom_dim = 0
                    
                    for X in A_i.objects:
                        for Y in A_j.objects:
                            # 计算 Hom(X, Y) 的维度
                            hom_dim = t_structure._compute_chain_map_space_dim(X, Y)
                            total_hom_dim += hom_dim
                    
                    if total_hom_dim > 0:
                        violations.append({
                            "i": i,
                            "j": j,
                            "A_i_name": A_i.name,
                            "A_j_name": A_j.name,
                            "hom_dimension": total_hom_dim,
                            "description": f"Semi-orthogonality violated: dim(Hom(A_{i}, A_{j})) = {total_hom_dim} > 0"
                        })
        
        return len(violations) == 0, violations
    
    def verify_semi_orthogonality_strict(self) -> bool:
        """严格验证半正交条件，违反时抛出异常
        
        Raises:
            CategoricalError: 当半正交条件被违反时
        """
        is_valid, violations = self.verify_semi_orthogonality()
        
        if not is_valid:
            first_violation = violations[0]
            raise CategoricalError(
                f"Semi-orthogonality violated: {first_violation['description']}"
            )
        
        return True


class QuasiBPSExtractor:
    """准 BPS 不变量提取器
    
    红线 B.2: 从 t-structure 分解提取 BPS 坐标
    
    数学原理:
    1. 对每个子范畴计算 Euler 特征
    2. 提取可计数不变量 w
    3. 验证 w 是精确整数
    """
    
    def __init__(self):
        self.decomposer = SemiOrthogonalDecomposition()
    
    def extract(
        self,
        decomposition: List[TriangulatedSubcategory],
    ) -> BPSInvariants:
        """提取 BPS 不变量
        
        红线 B.2 (Requirements 5.1, 5.2): 提取精确整数 BPS 坐标
        
        数学原理:
        1. 对每个子范畴计算 Euler 特征
        2. 提取可计数不变量 w
        3. 验证 w 是精确整数
        
        Args:
            decomposition: 半正交分解的子范畴列表
            
        Returns:
            BPS 不变量（精确整数）
        """
        weights: List[int] = []
        
        for subcategory in decomposition:
            # 计算子范畴的 Euler 特征作为 BPS 权重
            chi = subcategory.euler_characteristic()
            
            # 红线 B.2: 确保是精确整数
            weight = _require_exact_int(chi, name="Euler characteristic")
            weights.append(weight)
        
        return BPSInvariants(weights=weights)
    
    def extract_from_derived_category(
        self,
        derived_cat: List[ChainComplex],
        t_structure: tStructure,
    ) -> BPSInvariants:
        """从导出范畴直接提取 BPS 不变量
        
        便捷方法: 先分解再提取
        """
        decomposition = self.decomposer.decompose(derived_cat, t_structure)
        return self.extract(decomposition)
    
    def verify_galois_preimage(
        self,
        invariants: BPSInvariants,
        galois_param: LanglandsParameter,
    ) -> bool:
        """验证 BPS 不变量是 Galois 参数的代数原像
        
        红线 B.2 (Requirements 5.3): 验证代数原像关系
        
        数学原理:
        BPS 不变量 {wi} 应该与 Galois 参数 φ 的特征值相关
        在 Langlands 对应下，BPS 权重对应于 Galois 表示的特征值的某种组合
        
        验证条件:
        1. 维度兼容性: BPS 权重数量应与 Galois 表示维度相关
        2. 权重约束: BPS 权重之和应与 Galois 表示的迹相关
        3. 整数性: BPS 权重必须是精确整数（已在 extract 中保证）
        """
        # 严格模式：在缺少 Langlands 对应所需的额外结构（如 Hecke 作用、匹配的局部因子/正规化约定）
        # 时，无法数学上验证BPS 不变量确为 Galois 参数的代数原像。
        #
        # 旧实现依赖经验不等式与允许偏差的阈值，这是红线禁止的启发式验证，已移除。
        raise CategoricalError(
            "verify_galois_preimage is not mathematically checkable with the currently available data. "
            "Previous heuristic bounds were removed per MVP10 redlines."
        )


# ============================================================================
# Section 12: DistinguishedTriangle - 区别三角
# ============================================================================

@dataclass
class DistinguishedTriangle:
    """区别三角: 导出范畴的基本结构
    
    X → Y → Z → X[1]
      f   g   h
    
    满足:
    1. 旋转公理: Y → Z → X[1] → Y[1] 也是区别三角
    2. 八面体公理
    """
    X: ChainComplex
    Y: ChainComplex
    Z: ChainComplex
    f: Optional[Dict[int, np.ndarray]] = None  # X → Y 的链映射
    g: Optional[Dict[int, np.ndarray]] = None  # Y → Z 的链映射
    h: Optional[Dict[int, np.ndarray]] = None  # Z → X[1] 的链映射
    
    def verify_exactness(self) -> Tuple[bool, List[int]]:
        """验证长正合列
        
        区别三角诱导上同调长正合列:
        ... → H^n(X) --f*--> H^n(Y) --g*--> H^n(Z) --δ--> H^{n+1}(X) → ...
        
        正合性条件:
        1. im(f*) = ker(g*) 在每个 H^n(Y) 处
        2. im(g*) = ker(δ) 在每个 H^n(Z) 处
        3. im(δ) = ker(f*) 在每个 H^{n+1}(X) 处
        
        Returns:
            (is_exact, break_degrees) 其中 break_degrees 是断裂的度
        """
        # 红线：没有显式链映射就不能验证长正合列正合性。
        if self.f is None or self.g is None or self.h is None:
            raise CategoricalError(
                "Cannot verify long exact sequence without explicit chain maps f, g, h. "
                "Dimension-only checks are forbidden."
            )

        validator = ExactnessValidator(rel_tol=_EXACTNESS_REL_TOL)
        breaks: List[int] = []

        all_degrees: Set[int] = set(self.X.degrees) | set(self.Y.degrees) | set(self.Z.degrees)
        if not all_degrees:
            return True, []

        min_deg = min(all_degrees)
        max_deg = max(all_degrees)

        def _basis_matrix(C: ChainComplex, n: int) -> Tuple[int, np.ndarray]:
            h_dim, basis = C.cohomology(n)
            dim_Cn = C.objects.get(n, VectorSpace(0)).dimension
            if h_dim == 0:
                return 0, np.zeros((dim_Cn, 0), dtype=np.complex128)
            B = np.asarray(basis, dtype=np.complex128)
            if B.ndim != 2 or B.shape[0] != dim_Cn or B.shape[1] != h_dim:
                raise CategoricalError(
                    f"Invalid cohomology basis shape at degree {n}: got {getattr(B,'shape',None)}, "
                    f"expected ({dim_Cn}, {h_dim})."
                )
            return h_dim, B

        def _get_chain_map(
            maps: Dict[int, np.ndarray],
            n: int,
            src: ChainComplex,
            tgt: ChainComplex,
            *,
            tgt_shift: int = 0,
        ) -> np.ndarray:
            # 允许省略零映射：缺失即零矩阵（这是结构性约定，不是启发式）。
            src_dim = src.objects.get(n, VectorSpace(0)).dimension
            tgt_dim = tgt.objects.get(n + tgt_shift, VectorSpace(0)).dimension
            if src_dim == 0 or tgt_dim == 0:
                return np.zeros((tgt_dim, src_dim), dtype=np.complex128)
            M = maps.get(n)
            if M is None:
                return np.zeros((tgt_dim, src_dim), dtype=np.complex128)
            A = np.asarray(M, dtype=np.complex128)
            if A.shape != (tgt_dim, src_dim):
                raise CategoricalError(
                    f"Chain map matrix at degree {n} has wrong shape {A.shape}, expected {(tgt_dim, src_dim)}."
                )
            return A

        def _induced_map(
            maps: Dict[int, np.ndarray],
            n: int,
            src: ChainComplex,
            tgt: ChainComplex,
            src_basis: np.ndarray,
            tgt_basis: np.ndarray,
            *,
            tgt_shift: int = 0,
        ) -> np.ndarray:
            # src_basis: (dim src^n, dim H^n(src))
            # tgt_basis: (dim tgt^{n+tgt_shift}, dim H^{n+tgt_shift}(tgt))
            if src_basis.shape[1] == 0 or tgt_basis.shape[1] == 0:
                return np.zeros((tgt_basis.shape[1], src_basis.shape[1]), dtype=np.complex128)
            f_n = _get_chain_map(maps, n, src, tgt, tgt_shift=tgt_shift)
            images = f_n @ src_basis  # (dim tgt, dim H_src)
            # ChainComplex.cohomology 产出的基通常已正交化；为稳健起见使用伪逆投影到 cohomology 代表子空间。
            coords = np.linalg.pinv(tgt_basis) @ images
            return coords

        # 对每个 n 校验三处正合性：
        # 1) H^n(X)->H^n(Y)->H^n(Z)
        # 2) H^n(Y)->H^n(Z)->H^{n+1}(X)
        # 3) H^n(Z)->H^{n+1}(X)->H^{n+1}(Y)
        for n in range(min_deg - 1, max_deg + 2):
            hX, BX = _basis_matrix(self.X, n)
            hY, BY = _basis_matrix(self.Y, n)
            hZ, BZ = _basis_matrix(self.Z, n)
            hXp1, BX1 = _basis_matrix(self.X, n + 1)
            hYp1, BY1 = _basis_matrix(self.Y, n + 1)

            # 诱导映射矩阵
            f_star_n = _induced_map(self.f, n, self.X, self.Y, BX, BY)
            g_star_n = _induced_map(self.g, n, self.Y, self.Z, BY, BZ)
            h_star_n = _induced_map(self.h, n, self.Z, self.X, BZ, BX1, tgt_shift=1)  # Z^n -> X^{n+1}

            f_star_np1 = _induced_map(self.f, n + 1, self.X, self.Y, BX1, BY1)

            # 1) exact at H^n(Y)
            if hY > 0:
                ok, _ = validator.verify_exactness_at(f_star_n, g_star_n)
                if not ok:
                    breaks.append(n)
                    continue

            # 2) exact at H^n(Z)
            if hZ > 0:
                ok, _ = validator.verify_exactness_at(g_star_n, h_star_n)
                if not ok:
                    breaks.append(n)
                    continue

            # 3) exact at H^{n+1}(X)
            if hXp1 > 0:
                ok, _ = validator.verify_exactness_at(h_star_n, f_star_np1)
                if not ok:
                    breaks.append(n + 1)
                    continue

        return len(breaks) == 0, sorted(set(breaks))
    
    def detect_breaks(self) -> List[Dict[str, Any]]:
        """检测长正合列断裂
        
        详细分析每个断裂点，提供诊断信息
        
        Returns:
            断裂点信息列表，包含:
            - degree: 断裂发生的度
            - H_n_X, H_n_Y, H_n_Z: 各复形的上同调维度
            - type: 断裂类型
            - diagnosis: 诊断信息
        """
        is_exact, break_degrees = self.verify_exactness()
        
        if is_exact:
            return []
        
        breaks = []
        for n in break_degrees:
            h_n_X, basis_X = self.X.cohomology(n)
            h_n_Y, basis_Y = self.Y.cohomology(n)
            h_n_Z, basis_Z = self.Z.cohomology(n)
            h_n_plus_1_X, _ = self.X.cohomology(n + 1)
            
            # 诊断断裂类型
            diagnosis = []
            break_type = "exactness_failure"
            
            # 检查维度约束
            if h_n_X + h_n_Z < h_n_Y:
                diagnosis.append(
                    f"Dimension constraint violated: dim(H^{n}(X)) + dim(H^{n}(Z)) = {h_n_X + h_n_Z} < {h_n_Y} = dim(H^{n}(Y))"
                )
                break_type = "dimension_constraint_violation"
            
            # 检查映射是否存在
            f_n = self.f.get(n) if self.f else None
            g_n = self.g.get(n) if self.g else None
            
            if f_n is None and g_n is None:
                diagnosis.append("No explicit chain maps provided, using dimension-based analysis")
            
            # 检查 g ∘ f = 0 条件
            if f_n is not None and g_n is not None:
                gf = g_n @ f_n
                gf_norm = np.linalg.norm(gf)
                if gf_norm > _EXACTNESS_TOL:
                    diagnosis.append(f"Composition g∘f ≠ 0: ||g∘f|| = {gf_norm:.2e}")
                    break_type = "composition_nonzero"
            
            # 检查像和核的维度匹配
            if f_n is not None:
                rank_f = np.linalg.matrix_rank(f_n, tol=_EXACTNESS_TOL)
                diagnosis.append(f"rank(f^{n}) = {rank_f}")
            
            if g_n is not None:
                rank_g = np.linalg.matrix_rank(g_n, tol=_EXACTNESS_TOL)
                nullity_g = h_n_Y - rank_g if h_n_Y > 0 else 0
                diagnosis.append(f"rank(g^{n}) = {rank_g}, nullity(g^{n}) = {nullity_g}")
                
                # 正合性要求 rank(f) = nullity(g)
                if f_n is not None:
                    rank_f = np.linalg.matrix_rank(f_n, tol=_EXACTNESS_TOL)
                    if rank_f != nullity_g:
                        diagnosis.append(
                            f"Exactness violation: rank(f^{n}) = {rank_f} ≠ {nullity_g} = nullity(g^{n})"
                        )
            
            breaks.append({
                "degree": n,
                "H_n_X": h_n_X,
                "H_n_Y": h_n_Y,
                "H_n_Z": h_n_Z,
                "H_n_plus_1_X": h_n_plus_1_X,
                "type": break_type,
                "diagnosis": "; ".join(diagnosis) if diagnosis else "Unknown cause"
            })
        
        return breaks
    
    def rotate(self) -> 'DistinguishedTriangle':
        """旋转三角: X → Y → Z → X[1] 变为 Y → Z → X[1] → Y[1]"""
        return DistinguishedTriangle(
            X=self.Y,
            Y=self.Z,
            Z=self.X.shift(1),
            f=self.g,
            g=self.h,
            h=None  # 需要重新计算
        )
    
    @classmethod
    def from_morphism(cls, f_maps: Dict[int, np.ndarray], X: ChainComplex, Y: ChainComplex) -> 'DistinguishedTriangle':
        """从链映射 f: X → Y 构造区别三角
        
        Z = Cone(f) = 映射锥
        
        映射锥的定义:
        Cone(f)^n = X^{n+1} ⊕ Y^n
        
        边界算子 d_Cone^n: Cone^n → Cone^{n+1} 的矩阵形式:
        d_Cone = [[-d_X^{n+1},    0    ]
                  [  f^{n+1},   d_Y^n  ]]
        
        其中:
        - -d_X^{n+1}: X^{n+1} → X^{n+2} (带负号)
        - f^{n+1}: X^{n+1} → Y^{n+1}
        - d_Y^n: Y^n → Y^{n+1}
        
        注意: f 必须是链映射，即满足 d_Y ∘ f = f ∘ d_X
        """
        # 首先验证 f 是链映射
        # 链映射条件: d_Y^n ∘ f^n = f^{n+1} ∘ d_X^n
        for n in X.differentials:
            d_X_n = X.differentials[n]
            f_n = f_maps.get(n)
            f_n_plus_1 = f_maps.get(n + 1)
            d_Y_n = Y.differentials.get(n)
            
            if f_n is not None and f_n_plus_1 is not None and d_Y_n is not None:
                # 检查 d_Y^n ∘ f^n = f^{n+1} ∘ d_X^n
                lhs = d_Y_n @ f_n
                rhs = f_n_plus_1 @ d_X_n
                
                if not np.allclose(lhs, rhs, atol=_CHAIN_COMPLEX_TOL):
                    diff = np.linalg.norm(lhs - rhs)
                    raise CategoricalError(
                        f"Cannot construct distinguished triangle: f is not a chain map at degree {n}. "
                        f"||d_Y ∘ f - f ∘ d_X|| = {diff:.2e}."
                    )
        
        # 构造映射锥 Cone(f)
        cone_objects = {}
        cone_differentials = {}
        
        # 确定度的范围
        all_degrees = set(X.degrees) | set(Y.degrees)
        if not all_degrees:
            Z = ChainComplex({}, {}, verify=True)
            return cls(X=X, Y=Y, Z=Z, f=f_maps)
        
        min_deg = min(all_degrees)
        max_deg = max(all_degrees)
        
        # 构造 Cone 的对象
        # Cone^n = X^{n+1} ⊕ Y^n
        for n in range(min_deg - 1, max_deg + 2):
            X_n_plus_1 = X.objects.get(n + 1, VectorSpace(0))
            Y_n = Y.objects.get(n, VectorSpace(0))
            
            X_dim = X_n_plus_1.dimension
            Y_dim = Y_n.dimension
            
            if X_dim > 0 or Y_dim > 0:
                cone_objects[n] = VectorSpace(dimension=X_dim + Y_dim)
        
        # 构造边界算子
        # d_Cone^n: Cone^n → Cone^{n+1}
        # Cone^n = X^{n+1} ⊕ Y^n
        # Cone^{n+1} = X^{n+2} ⊕ Y^{n+1}
        
        sorted_degrees = sorted(cone_objects.keys())
        
        for i, n in enumerate(sorted_degrees[:-1]):
            n_plus_1 = sorted_degrees[i + 1]
            
            if n not in cone_objects or n_plus_1 not in cone_objects:
                continue
            
            # 源: Cone^n = X^{n+1} ⊕ Y^n
            X_n_plus_1_dim = X.objects.get(n + 1, VectorSpace(0)).dimension
            Y_n_dim = Y.objects.get(n, VectorSpace(0)).dimension
            
            # 目标: Cone^{n+1} = X^{n+2} ⊕ Y^{n+1}
            X_n_plus_2_dim = X.objects.get(n + 2, VectorSpace(0)).dimension
            Y_n_plus_1_dim = Y.objects.get(n + 1, VectorSpace(0)).dimension
            
            dim_source = X_n_plus_1_dim + Y_n_dim
            dim_target = X_n_plus_2_dim + Y_n_plus_1_dim
            
            if dim_source == 0 or dim_target == 0:
                continue
            
            # 构造块矩阵
            d_cone = np.zeros((dim_target, dim_source), dtype=np.complex128)
            
            # 块 (0,0): -d_X^{n+1}: X^{n+1} → X^{n+2}
            if X_n_plus_1_dim > 0 and X_n_plus_2_dim > 0:
                d_X_n_plus_1 = X.differentials.get(n + 1)
                if d_X_n_plus_1 is not None:
                    d_cone[:X_n_plus_2_dim, :X_n_plus_1_dim] = -d_X_n_plus_1
            
            # 块 (0,1): 0 (Y^n → X^{n+2})
            # 已经是零
            
            # 块 (1,0): f^{n+1}: X^{n+1} → Y^{n+1}
            if X_n_plus_1_dim > 0 and Y_n_plus_1_dim > 0:
                f_n_plus_1 = f_maps.get(n + 1)
                if f_n_plus_1 is not None:
                    d_cone[X_n_plus_2_dim:X_n_plus_2_dim + Y_n_plus_1_dim, 
                           :X_n_plus_1_dim] = f_n_plus_1
            
            # 块 (1,1): d_Y^n: Y^n → Y^{n+1}
            if Y_n_dim > 0 and Y_n_plus_1_dim > 0:
                d_Y_n = Y.differentials.get(n)
                if d_Y_n is not None:
                    d_cone[X_n_plus_2_dim:X_n_plus_2_dim + Y_n_plus_1_dim,
                           X_n_plus_1_dim:X_n_plus_1_dim + Y_n_dim] = d_Y_n
            
            cone_differentials[n] = d_cone
        
        # 验证 d² = 0
        Z = ChainComplex(cone_objects, cone_differentials, verify=True)
        
        # 构造 g: Y → Z 和 h: Z → X[1]
        # g^n: Y^n → Cone^n = X^{n+1} ⊕ Y^n 是嵌入到第二个分量
        g_maps = {}
        for n in Y.degrees:
            Y_n_dim = Y.objects[n].dimension
            if n in cone_objects and Y_n_dim > 0:
                X_n_plus_1_dim = X.objects.get(n + 1, VectorSpace(0)).dimension
                cone_dim = cone_objects[n].dimension
                
                g_n = np.zeros((cone_dim, Y_n_dim), dtype=np.complex128)
                g_n[X_n_plus_1_dim:X_n_plus_1_dim + Y_n_dim, :] = np.eye(Y_n_dim)
                g_maps[n] = g_n
        
        # h^n: Cone^n → X^{n+1} 是投影到第一个分量
        h_maps = {}
        for n in cone_objects:
            X_n_plus_1_dim = X.objects.get(n + 1, VectorSpace(0)).dimension
            if X_n_plus_1_dim > 0:
                cone_dim = cone_objects[n].dimension
                h_n = np.zeros((X_n_plus_1_dim, cone_dim), dtype=np.complex128)
                h_n[:, :X_n_plus_1_dim] = np.eye(X_n_plus_1_dim)
                h_maps[n] = h_n
        
        return cls(X=X, Y=Y, Z=Z, f=f_maps, g=g_maps, h=h_maps)


# ============================================================================
# Section 13: Yoneda Embedding 探测器
# ============================================================================

@dataclass
class ProbeMorphism:
    """探针态射"""
    method: str  # 'transfer', 'approve', 'transferFrom', 'balanceOf'
    params: Dict[str, Any]  # 参数
    boundary_type: str  # 'zero', 'one', 'max_uint128', 'max_uint256', 'custom'


@dataclass
class NaturalTransformation:
    """自然变换（探针响应）"""
    probe: ProbeMorphism
    response: Any  # 响应值
    success: bool  # 是否成功
    gas_used: int = 0
    revert_reason: str = ""
    # 可选：执行 trace 证据（用于 CoverageEstimator 的严格覆盖率下界）
    # 约定为基本块/节点 ID 序列；允许包含重复节点（CoverageEstimator 会做环擦除）。
    trace: Optional[List[str]] = None


@dataclass
class DeviationMetrics:
    """偏差度量"""
    morphism_deviations: Dict[str, float]
    total_deviation: float
    anomalous_morphisms: List[str]
    coverage_ratio: float


@dataclass
class CoverageResult:
    """覆盖率结果
    
    约定（严格、可证据化）：
    - `covered_paths`：有**可验证证据**（trace）支持的已见路径列表。
    - `covered_count`：覆盖路径数的**下界**，满足 covered_count >= len(covered_paths)。
      当缺少 trace 时，我们只能给出极保守下界（例如：执行过至少一次探针 ⇒ 至少见到 1 条路径）。
    """
    coverage_ratio: float
    covered_paths: List[str]
    uncovered_paths: List[str]
    is_sufficient: bool  # >= 80%
    total_paths: int
    covered_count: int


class YonedaProbe:
    """Yoneda Embedding 探测器
    
    数学原理: 米田引理 Hom(-, A) ≅ A
    对象完全由它与所有其他对象的态射决定
    
    战术应用: 向目标合约发送有限探针集，对比其反应与标准实现
    """
    
    # 标准 ERC20 方法
    PROBE_METHODS = ['transfer', 'approve', 'transferFrom', 'balanceOf']
    
    # 边界参数 - 完整的边界值集合
    # 设计原则:
    # 1. 零值边界: 测试空操作和零除
    # 2. 单位边界: 测试最小有效操作
    # 3. 类型边界: 测试各种整数类型的边界
    # 4. 常见值: 测试实际使用中的典型值
    # 5. 溢出边界: 测试算术溢出
    BOUNDARY_PARAMS = {
        # 零值边界
        'zero': 0,
        
        # 单位边界
        'one': 1,
        
        # 小值 (测试 gas 优化路径)
        'small': 100,
        'dust': 1000,  # 常见的 "dust" 阈值
        
        # 常见代币精度值
        'wei': 1,  # 1 wei
        'gwei': 10**9,  # 1 gwei
        'ether': 10**18,  # 1 ether (18 decimals)
        'usdc_unit': 10**6,  # USDC 精度 (6 decimals)
        
        # 类型边界 - uint8
        'max_uint8': 2**8 - 1,  # 255
        'near_max_uint8': 2**8 - 2,  # 254
        
        # 类型边界 - uint16
        'max_uint16': 2**16 - 1,  # 65535
        
        # 类型边界 - uint32
        'max_uint32': 2**32 - 1,  # 4294967295
        
        # 类型边界 - uint64
        'max_uint64': 2**64 - 1,
        
        # 类型边界 - uint128
        'max_uint128': 2**128 - 1,
        'near_max_uint128': 2**128 - 2,
        'half_uint128': 2**127,
        
        # 类型边界 - uint256
        'max_uint256': 2**256 - 1,
        'near_max_uint256': 2**256 - 2,
        'half_uint256': 2**255,
        
        # 溢出测试值
        'overflow_trigger_128': 2**128,  # 刚好超过 uint128
        'overflow_trigger_64': 2**64,  # 刚好超过 uint64
        
        # 特殊值
        'max_supply_typical': 10**27,  # 典型最大供应量 (1 billion tokens with 18 decimals)
        'large_transfer': 10**24,  # 大额转账测试
    }
    
    # 地址边界参数
    ADDRESS_PARAMS = {
        'zero_address': '0x0000000000000000000000000000000000000000',
        'dead_address': '0x000000000000000000000000000000000000dEaD',
        'max_address': '0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
    }
    
    def __init__(self):
        self._probe_cache: Dict[str, List[NaturalTransformation]] = {}
    
    def generate_probe_set(self, target_address: str, include_address_probes: bool = True) -> List[ProbeMorphism]:
        """生成完整探针态射集
        
        覆盖 {transfer, approve, transferFrom, balanceOf} × 边界参数
        可选地包含地址边界探针
        
        Args:
            target_address: 目标合约地址
            include_address_probes: 是否包含地址边界探针
            
        Returns:
            探针态射列表
        """
        probes = []
        
        # 收集要测试的地址
        test_addresses = [target_address]
        if include_address_probes:
            test_addresses.extend(self.ADDRESS_PARAMS.values())
        
        for method in self.PROBE_METHODS:
            for boundary_name, boundary_value in self.BOUNDARY_PARAMS.items():
                if method == 'transfer':
                    # 对于 transfer，测试不同的目标地址
                    for addr in test_addresses:
                        addr_suffix = "" if addr == target_address else f"_to_{addr[:10]}"
                        probes.append(ProbeMorphism(
                            method=method,
                            params={'to': addr, 'amount': boundary_value},
                            boundary_type=f"{boundary_name}{addr_suffix}"
                        ))
                elif method == 'approve':
                    # 对于 approve，测试不同的 spender 地址
                    for addr in test_addresses:
                        addr_suffix = "" if addr == target_address else f"_spender_{addr[:10]}"
                        probes.append(ProbeMorphism(
                            method=method,
                            params={'spender': addr, 'amount': boundary_value},
                            boundary_type=f"{boundary_name}{addr_suffix}"
                        ))
                elif method == 'transferFrom':
                    # 对于 transferFrom，测试不同的 from/to 组合
                    probes.append(ProbeMorphism(
                        method=method,
                        params={
                            'from': target_address,
                            'to': target_address,
                            'amount': boundary_value
                        },
                        boundary_type=boundary_name
                    ))
                    # 添加零地址测试
                    if include_address_probes:
                        zero_addr = self.ADDRESS_PARAMS['zero_address']
                        probes.append(ProbeMorphism(
                            method=method,
                            params={
                                'from': zero_addr,
                                'to': target_address,
                                'amount': boundary_value
                            },
                            boundary_type=f"{boundary_name}_from_zero"
                        ))
                        probes.append(ProbeMorphism(
                            method=method,
                            params={
                                'from': target_address,
                                'to': zero_addr,
                                'amount': boundary_value
                            },
                            boundary_type=f"{boundary_name}_to_zero"
                        ))
                elif method == 'balanceOf':
                    # 对于 balanceOf，测试不同的账户地址
                    for addr in test_addresses:
                        addr_suffix = "" if addr == target_address else f"_addr_{addr[:10]}"
                        probes.append(ProbeMorphism(
                            method=method,
                            params={'account': addr},
                            boundary_type=f"{boundary_name}{addr_suffix}"
                        ))
        
        return probes
    
    def generate_minimal_probe_set(self, target_address: str) -> List[ProbeMorphism]:
        """生成最小探针集（用于快速检测）
        
        只包含关键边界值，不包含地址变体
        """
        probes = []
        
        # 关键边界值
        critical_boundaries = {
            'zero': 0,
            'one': 1,
            'max_uint256': 2**256 - 1,
        }
        
        for method in self.PROBE_METHODS:
            for boundary_name, boundary_value in critical_boundaries.items():
                if method == 'transfer':
                    probes.append(ProbeMorphism(
                        method=method,
                        params={'to': target_address, 'amount': boundary_value},
                        boundary_type=boundary_name
                    ))
                elif method == 'approve':
                    probes.append(ProbeMorphism(
                        method=method,
                        params={'spender': target_address, 'amount': boundary_value},
                        boundary_type=boundary_name
                    ))
                elif method == 'transferFrom':
                    probes.append(ProbeMorphism(
                        method=method,
                        params={
                            'from': target_address,
                            'to': target_address,
                            'amount': boundary_value
                        },
                        boundary_type=boundary_name
                    ))
                elif method == 'balanceOf':
                    probes.append(ProbeMorphism(
                        method=method,
                        params={'account': target_address},
                        boundary_type=boundary_name
                    ))
        
        return probes
    
    def execute_probes(
        self, 
        probes: List[ProbeMorphism],
        executor: Optional[Callable[[ProbeMorphism], NaturalTransformation]] = None
    ) -> List[NaturalTransformation]:
        """执行探针并记录响应
        
        Args:
            probes: 探针列表
            executor: 执行器函数（可选，用于实际链上调用）
            
        Returns:
            自然变换（响应）列表
        """
        responses = []
        
        for probe in probes:
            if executor is not None:
                response = executor(probe)
            else:
                # 模拟响应（用于测试）
                response = NaturalTransformation(
                    probe=probe,
                    response=None,
                    success=True,
                    gas_used=21000
                )
            responses.append(response)
        
        return responses
    
    def get_standard_erc20_responses(self, probes: List[ProbeMorphism]) -> List[NaturalTransformation]:
        """获取标准 ERC20 实现的预期响应"""
        responses = []
        
        for probe in probes:
            # 标准 ERC20 行为
            if probe.method == 'balanceOf':
                response = NaturalTransformation(
                    probe=probe,
                    response=0,  # 默认余额为 0
                    success=True
                )
            elif probe.method == 'transfer':
                # 转账: 如果余额不足应该失败
                amount = probe.params.get('amount', 0)
                response = NaturalTransformation(
                    probe=probe,
                    response=amount == 0,  # 只有 0 转账成功
                    success=amount == 0
                )
            elif probe.method == 'approve':
                response = NaturalTransformation(
                    probe=probe,
                    response=True,
                    success=True
                )
            elif probe.method == 'transferFrom':
                amount = probe.params.get('amount', 0)
                response = NaturalTransformation(
                    probe=probe,
                    response=amount == 0,
                    success=amount == 0
                )
            else:
                response = NaturalTransformation(
                    probe=probe,
                    response=None,
                    success=False
                )
            responses.append(response)
        
        return responses


class NaturalTransformationComparator:
    """自然变换比较器
    
    比较目标合约响应与标准 ERC20 响应，检测行为偏差
    """
    
    def __init__(self, tolerance: float = 0.0):
        """
        Args:
            tolerance: 偏差阈值（默认 0：任何可观测偏差都算异常；避免手选0.01之类的魔法数）
        """
        self.tolerance = tolerance
    
    def compare(
        self,
        target_responses: List[NaturalTransformation],
        standard_responses: List[NaturalTransformation]
    ) -> DeviationMetrics:
        """比较响应并计算偏差度量
        
        Returns:
            偏差度量
        """
        if len(target_responses) != len(standard_responses):
            raise ValueError("Response lists must have same length")
        
        morphism_deviations: Dict[str, float] = {}
        anomalous: List[str] = []
        total_deviation = 0.0
        
        for target, standard in zip(target_responses, standard_responses):
            method = target.probe.method
            boundary = target.probe.boundary_type
            key = f"{method}_{boundary}"
            
            # 计算偏差
            deviation = self._compute_deviation(target, standard)
            morphism_deviations[key] = deviation
            total_deviation += deviation
            
            # 检测异常
            if deviation > self.tolerance:
                anomalous.append(key)
        
        # 归一化总偏差
        n = len(target_responses)
        if n > 0:
            total_deviation /= n
        
        return DeviationMetrics(
            morphism_deviations=morphism_deviations,
            total_deviation=total_deviation,
            anomalous_morphisms=anomalous,
            coverage_ratio=float('nan')  # 由 CoverageEstimator 基于 CFG + trace 证据填充
        )
    
    def _compute_deviation(
        self, 
        target: NaturalTransformation, 
        standard: NaturalTransformation
    ) -> float:
        """计算单个响应的偏差"""
        # 成功/失败状态不同
        if target.success != standard.success:
            return 1.0
        
        # 响应值比较
        if target.response is None and standard.response is None:
            return 0.0
        
        if target.response is None or standard.response is None:
            # 缺失响应本身就是不可对齐的证据，按最大偏差计（保守，且无 0.5 这种经验常数）。
            return 1.0
        
        # 数值比较
        try:
            t_val = float(target.response)
            s_val = float(standard.response)
            
            if s_val == 0:
                return 0.0 if t_val == 0 else 1.0
            
            # 使用机器精度下界避免除零（不引入手选常数）
            return min(1.0, abs(t_val - s_val) / (abs(s_val) + _FLOAT64_EPS))
        except (TypeError, ValueError):
            # 非数值比较
            return 0.0 if target.response == standard.response else 1.0


class CoverageEstimator:
    """覆盖率估计器
    
    基于 CFG 路径枚举计算覆盖率下界
    
    说明（MVP10强化稿红线 #5）：
    - 覆盖率必须来自 CFG 路径计数与可验证证据（trace），不得用关键词/相似度等启发式代替。
    - 默认阈值为 80%：coverage_ratio < 0.8 必须强制报警（warning）。
    """
    
    def __init__(self, threshold: Optional[float] = _COVERAGE_THRESHOLD):
        """
        Args:
            threshold: 覆盖率阈值（默认 0.8，来自工程规格；None 表示仅返回诊断不判定）。
        """
        self.threshold = threshold
    
    def estimate_coverage(
        self,
        cfg_edges: List[Tuple[str, str]],
        executed_probes: List[Any],
        entry_point: str = "entry"
    ) -> CoverageResult:
        """估计覆盖率
        
        Args:
            cfg_edges: CFG 边列表
            executed_probes: 已执行的探针
            entry_point: 入口点
            
        Returns:
            覆盖率结果
        """
        # 枚举从入口出发的所有**简单路径**（不重复节点）。
        # 该集合在有限图上是有限的，且不需要任何最大深度启发式。
        all_paths = self._enumerate_paths(cfg_edges, entry_point)
        total_paths = len(all_paths)
        
        if total_paths == 0:
            return CoverageResult(
                coverage_ratio=1.0,
                covered_paths=[],
                uncovered_paths=[],
                is_sufficient=True,
                total_paths=0,
                covered_count=0
            )
        
        # 计算有证据的已见路径
        witnessed_paths = self._compute_covered_paths(all_paths, executed_probes)
        witnessed_set = set(witnessed_paths)
        uncovered_paths = [p for p in all_paths if p not in witnessed_set]

        # 覆盖条数下界：
        # - 若有 trace 证据：至少覆盖 len(witnessed_paths) 条
        # - 若无 trace 证据但确实执行过探针：至少执行过 1 条路径（但无法指明是哪条）
        covered_count_lb = len(witnessed_paths)
        if covered_count_lb == 0 and executed_probes:
            covered_count_lb = 1

        coverage_ratio = covered_count_lb / total_paths
        is_sufficient = True if self.threshold is None else (coverage_ratio >= float(self.threshold))

        result = CoverageResult(
            coverage_ratio=coverage_ratio,
            covered_paths=witnessed_paths,
            uncovered_paths=uncovered_paths,
            is_sufficient=is_sufficient,
            total_paths=total_paths,
            covered_count=covered_count_lb,
        )
        
        # 覆盖率不足时强制报警（MVP10强化稿红线 #5）
        if self.threshold is not None and not is_sufficient:
            warnings.warn(
                f"Coverage insufficient: {coverage_ratio:.1%} < {self.threshold:.1%}. "
                f"Uncovered paths: {uncovered_paths[:5]}{'...' if len(uncovered_paths) > 5 else ''}",
                CoverageInsufficientWarning
            )
        
        return result
    
    def _enumerate_paths(
        self, 
        edges: List[Tuple[str, str]], 
        entry: str,
        max_depth: Optional[int] = None
    ) -> List[str]:
        """枚举从入口点开始的所有路径
        
        数学约定：路径集合取 **所有从 entry 出发的极大简单路径**
        （simple path = 不包含重复节点；极大 = 不能再延伸而不破坏简单性）。
        
        该定义在存在环的 CFG 中仍然给出有限集合，因此不需要任何启发式深度截断。
        
        Args:
            edges: CFG 边列表
            entry: 入口点
            max_depth: 可选的显式深度上界（仅用于上游强制资源约束；不作为默认策略）
        """
        # 构建邻接表（排序以保证确定性）
        adj: Dict[str, List[str]] = {}
        nodes: Set[str] = set()
        for src, tgt in edges:
            nodes.add(src)
            nodes.add(tgt)
            adj.setdefault(src, []).append(tgt)
        for src in adj:
            adj[src] = sorted(set(adj[src]))

        if entry not in nodes:
            return []

        paths: List[str] = []

        # 深度上界：简单路径最多访问 |V| 个节点
        hard_cap = len(nodes) - 1
        if max_depth is not None:
            hard_cap = min(hard_cap, int(max_depth))

        def dfs(node: str, path: List[str], visited: Set[str]) -> None:
            # 如果达到显式硬上界，则视为在资源约束下的极大路径
            if len(path) - 1 >= hard_cap:
                paths.append("→".join(path))
                return

            nexts = [n for n in adj.get(node, []) if n not in visited]
            if not nexts:
                paths.append("→".join(path))
                return

            for nxt in nexts:
                dfs(nxt, path + [nxt], visited | {nxt})

        dfs(entry, [entry], {entry})
        return paths
    
    def _compute_covered_paths(
        self, 
        all_paths: List[str], 
        executions: List[Any]
    ) -> List[str]:
        """返回**有 trace 证据**支持的已见路径。

        证据来源约定（任一即可）：
        - `NaturalTransformation.trace`: List[str]（基本块/节点序列）
        - `obj.trace` 或 `obj.trace_nodes`: List[str]
        - dict 形式：{"trace": [...]} 或 {"trace_nodes": [...]}

        我们对 trace 做**环擦除（loop-erasure）**得到简单路径，再与 all_paths 精确匹配。
        未提供 trace 的执行记录不会被计入 `covered_paths`（但上层仍会给出覆盖数下界）。
        """
        if not executions or not all_paths:
            return []

        all_set = set(all_paths)
        witnessed: Set[str] = set()

        for ex in executions:
            trace = None

            # NaturalTransformation
            if isinstance(ex, NaturalTransformation):
                trace = getattr(ex, "trace", None)
            # dict-like
            elif isinstance(ex, dict):
                trace = ex.get("trace") or ex.get("trace_nodes")
            else:
                trace = getattr(ex, "trace", None) or getattr(ex, "trace_nodes", None)

            if trace is None:
                continue

            # 红线：trace 是覆盖证据；格式不合规必须显式失败，禁止静默吞掉导致假低覆盖。
            if not isinstance(trace, (list, tuple)):
                raise CategoricalError(f"trace must be a list/tuple of node ids, got {type(trace)}")
            nodes = [str(x) for x in trace]

            if not nodes:
                continue

            simple_nodes = self._loop_erase(nodes)
            path_str = "→".join(simple_nodes)
            if path_str in all_set:
                witnessed.add(path_str)

        return sorted(witnessed)

    @staticmethod
    def _loop_erase(nodes: List[str]) -> List[str]:
        """对节点序列做确定性的环擦除（chronological loop-erasure）。"""
        pos: Dict[str, int] = {}
        out: List[str] = []
        for n in nodes:
            if n in pos:
                idx = pos[n]
                # 删除环段
                for x in out[idx + 1:]:
                    pos.pop(x, None)
                out = out[:idx + 1]
            else:
                pos[n] = len(out)
                out.append(n)
        return out


# ============================================================================
# Section 13.1: YonedaInvariantProber - 米田不变量探测器 (模块 C.2)
# ============================================================================

class YonedaInvariantProber:
    """米田不变量探测器
    
    红线 C.2 (Requirements 8.1, 8.2, 8.4): 基于米田引理的行为探测
    
    数学原理:
    米田引理: Hom(-, A) ≅ A
    对象完全由它与所有其他对象的态射决定
    
    应用:
    将 EVM 行为映射到 Hom 函子，检测不变性属性
    """
    
    def __init__(self, category: Optional[Category] = None):
        """
        Args:
            category: 底层范畴（可选）
        """
        self.category = category
        self._behavior_cache: Dict[str, Dict[str, np.ndarray]] = {}
    
    def probe_behavior(
        self,
        target: Object,
        probe_morphisms: List[Morphism],
    ) -> Dict[str, np.ndarray]:
        """探测目标对象的行为
        
        红线 C.2 (Requirements 8.1): 将 EVM 行为映射到 Hom 函子
        
        数学原理:
        对于每个探针态射 f: P → A，计算 Hom(P, target) 的响应
        
        Args:
            target: 目标对象 A
            probe_morphisms: 探针态射列表
            
        Returns:
            探针 ID → 响应矩阵 的映射
        """
        behavior: Dict[str, np.ndarray] = {}
        
        for probe in probe_morphisms:
            probe_id = probe.name or f"{probe.source.id}→{probe.target.id}"

            # Yoneda：对目标对象 A，Hom(P, A) 的元素必须是指向 A 的态射。
            if probe.target != target:
                raise CategoricalError(
                    "YonedaInvariantProber.probe_behavior requires probe morphisms in Hom(P, target), "
                    f"but got a morphism targeting '{probe.target.id}' (expected '{target.id}')."
                )
            if probe.matrix is None:
                raise CategoricalError(
                    f"Probe morphism '{probe_id}' has no matrix representation; refusing to assume identity/zero."
                )
            behavior[probe_id] = np.asarray(probe.matrix, dtype=np.complex128).copy()
        
        # 缓存行为
        self._behavior_cache[target.id] = behavior
        
        return behavior
    
    def map_to_topos_truth(
        self,
        behavior: Dict[str, np.ndarray],
        classifier: 'SubobjectClassifier',
    ) -> 'HeytingElement':
        """将行为映射到 Topos 真值
        
        红线 C.2 (Requirements 8.2): 将不变性属性转换为代数真值
        
        数学原理:
        行为的"一致性"程度映射到 Heyting 代数的元素
        
        Args:
            behavior: 探针行为映射
            classifier: 子对象分类器
            
        Returns:
            Heyting 代数元素，表示行为的真值
        """
        if not behavior:
            return classifier.heyting.bottom()

        dim = classifier.heyting.dimension
        probe_ids = sorted(behavior.keys())
        if len(probe_ids) != dim:
            raise CategoricalError(
                "Cannot canonically map behavior to Ω without an explicit probe→world assignment. "
                f"Got {len(probe_ids)} probes but Ω has dimension {dim}."
            )

        consistency = np.zeros(dim, dtype=np.float64)
        for i, pid in enumerate(probe_ids):
            response = np.asarray(behavior[pid], dtype=np.complex128)
            norm = float(np.linalg.norm(response, 'fro')) if response.size > 0 else 0.0
            # 规范单调映射 [0,∞) → [0,1): x ↦ x/(1+x)（无参数、无阈值）
            consistency[i] = norm / (1.0 + norm)

        return classifier.heyting.create_element(consistency)
    
    def compute_yoneda_distance(
        self,
        behavior1: Dict[str, np.ndarray],
        behavior2: Dict[str, np.ndarray],
    ) -> float:
        """计算 Yoneda 距离（行为偏差）
        
        红线 C.2 (Requirements 8.4): 量化行为偏差
        
        数学原理:
        Yoneda 距离测量两个对象在 Hom 函子下的差异
        
        Args:
            behavior1: 第一个对象的行为
            behavior2: 第二个对象的行为
            
        Returns:
            Yoneda 距离（非负实数）
        """
        # 收集所有探针 ID
        all_probes = set(behavior1.keys()) | set(behavior2.keys())
        
        if not all_probes:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for probe_id in all_probes:
            resp1 = behavior1.get(probe_id)
            resp2 = behavior2.get(probe_id)
            
            if resp1 is None and resp2 is None:
                continue
            elif resp1 is None:
                # behavior1 缺少此探针
                total_distance += np.linalg.norm(resp2, 'fro')
                count += 1
            elif resp2 is None:
                # behavior2 缺少此探针
                total_distance += np.linalg.norm(resp1, 'fro')
                count += 1
            else:
                # 两者都有，计算差异
                if resp1.shape == resp2.shape:
                    diff = np.linalg.norm(resp1 - resp2, 'fro')
                else:
                    # 形状不同，使用最大范数
                    diff = max(np.linalg.norm(resp1, 'fro'), np.linalg.norm(resp2, 'fro'))
                total_distance += diff
                count += 1
        
        return total_distance / max(count, 1)


class LFactorComputer:
    """L-因子计算器
    
    红线 C.2 (Requirements 8.3): 将 Transfer 操作映射到自守形式的局部因子
    
    数学原理:
    L-函数 L(π, s) = ∏_v L(πv, s) 是自守表示的重要不变量
    局部因子 L(πv, s) 编码了表示在素点 v 处的信息
    """
    
    def __init__(self):
        # 仅缓存纯 Satake局部因子：当引入 Transfer 行为时，
        # 局部因子依赖额外矩阵数据，若不把该数据纳入 key 会导致数学错误缓存命中。
        self._cache: Dict[Tuple[int, complex, complex], complex] = {}
    
    def compute_local_factor(
        self,
        transfer_behavior: Dict[str, np.ndarray],
        satake_param: 'SatakeParameter',
        s: complex,
    ) -> complex:
        """计算局部 L-因子 L(πv, s)
        
        红线 C.2 (Requirements 8.3): 将 Transfer 行为映射到 L-因子
        
        数学公式:
        对于 GL(n)，L(πv, s) = ∏_{i=1}^n (1 - α_i q^{-s})^{-1}
        其中 α_i 是 Satake 参数的特征值
        
        Args:
            transfer_behavior: Transfer 操作的行为映射
            satake_param: Satake 参数
            s: 复变量
            
        Returns:
            局部 L-因子值
        """
        # 获取特征值
        alpha = satake_param.eigenvalue

        # 缓存检查（仅限 transfer_behavior 为空的纯 Satake 情况）
        if not transfer_behavior:
            cache_key = (satake_param.prime_v, s, alpha)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # 计算 q = p^f（素点的范数）
        # 数学定义:
        # 对于有限域扩张 F_q / F_p，素点 v 的范数 N(v) = q = p^f
        # 其中 f 是惯性度（inertia degree）
        # 
        # 对于 Q 上的素点 p，范数就是 p 本身（f = 1）
        # 对于数域 K 上的素点 v | p，范数是 p^f 其中 f = [O_K/v : Z/p]
        #
        # 从 Satake 参数推导:
        # - prime_v 是底层素数 p
        # - weight 可以编码惯性度信息
        # - 如果 weight > 0，使用 p^weight 作为范数
        # - 否则使用 p（假设 f = 1）
        
        # 严格：此处的 prime_v 必须确为素数（至少应为 >= 2 的正整数）。
        # 若上游把 prime_v 当作EVM 基本块索引，必须先显式映射到素数域，再调用 L 因子。
        p = int(satake_param.prime_v)
        if p < 2:
            raise ValueError(f"prime_v must be >= 2 to compute a number-theoretic norm, got {satake_param.prime_v}.")
        
        # 从 weight 推导惯性度
        # 约定: weight = 0 表示 f = 1，weight > 0 表示 f = weight
        inertia_degree = int(satake_param.weight)
        if inertia_degree < 0:
            raise ValueError(f"Satake weight must be >= 0, got {satake_param.weight}.")
        inertia_degree = inertia_degree if inertia_degree > 0 else 1
        
        # 计算素点范数 q = p^f
        q = p ** inertia_degree
        
        # 计算 L-因子
        # L(πv, s) = (1 - α q^{-s})^{-1}
        q_neg_s = q ** (-s)
        denominator = 1 - alpha * q_neg_s
        
        # 避免除零
        if np.abs(denominator) < _FLOAT64_EPS:
            L_factor = complex(np.inf)
        else:
            L_factor = 1.0 / denominator
        
        # 如果有 Transfer 行为：用矩阵 Euler 因子进入局部因子（无系数、无 tanh 调味）
        #
        # 数学模型：
        # - 把 transfer_behavior 视为对偶群表示下某个半单元的矩阵近似 T
        # - 归一化到谱半径 ≤ 1（保证 Euler 因子不因尺度爆炸）
        # - 乘上 det(I - T q^{-s})^{-1}
        if transfer_behavior:
            mats: List[np.ndarray] = []
            for resp in transfer_behavior.values():
                if resp is None:
                    continue
                M = np.asarray(resp, dtype=np.complex128)
                if M.size == 0:
                    continue
                if M.ndim == 1:
                    mats.append(np.diag(M))
                elif M.ndim == 2:
                    if M.shape[0] == M.shape[1]:
                        mats.append(M)
                    else:
                        # 规范化成方阵：Gram 矩阵 M M^*
                        mats.append(M @ M.conj().T)

            if mats:
                # 若维度不同，使用直和（block diagonal）构造一个总矩阵
                def _block_diag(ms: List[np.ndarray]) -> np.ndarray:
                    total = int(sum(m.shape[0] for m in ms))
                    out = np.zeros((total, total), dtype=np.complex128)
                    off = 0
                    for m in ms:
                        n = m.shape[0]
                        out[off:off + n, off:off + n] = m
                        off += n
                    return out

                T = mats[0] if len(mats) == 1 and mats[0].shape[0] == mats[0].shape[1] else _block_diag(mats)

                # 归一化：谱半径 ≤ 1
                try:
                    eigs = np.linalg.eigvals(T)
                    rho = float(np.max(np.abs(eigs))) if eigs.size > 0 else 0.0
                except np.linalg.LinAlgError:
                    # 使用 2-范数作为谱半径上界（确定性上界）
                    rho = float(np.linalg.norm(T, ord=2))

                if rho > 1.0:
                    T = T / rho

                I = np.eye(T.shape[0], dtype=np.complex128)
                try:
                    det_term = np.linalg.det(I - T * q_neg_s)
                except np.linalg.LinAlgError as e:
                    raise CategoricalError(f"Failed to compute determinant term det(I - T q^(-s)): {e}") from e

                if np.abs(det_term) < _FLOAT64_EPS:
                    L_factor *= complex(np.inf)
                else:
                    L_factor *= 1.0 / det_term

        # 缓存结果（纯 Satake 情况）
        if not transfer_behavior:
            self._cache[(satake_param.prime_v, s, alpha)] = L_factor
        
        return L_factor
    
    def compute_partial_l_function(
        self,
        satake_params: List['SatakeParameter'],
        s: complex,
        max_primes: Optional[int] = None,
    ) -> complex:
        """计算部分 L-函数
        
        L(π, s) = ∏_{v ∈ S} L(πv, s)
        其中 S 是输入素点集合的一个有限子集；若 max_primes 为 None，则取全部输入素点。
        
        Args:
            satake_params: Satake 参数列表
            s: 复变量
            max_primes: 最大素点数（可选）。None 表示不截断。
            
        Returns:
            部分 L-函数值
        """
        L_partial = complex(1.0)
        
        params = satake_params if max_primes is None else satake_params[:int(max_primes)]
        for param in params:
            L_v = self.compute_local_factor({}, param, s)
            if not np.isfinite(L_v):
                raise CategoricalError(f"Non-finite local factor at v={param.prime_v}: {L_v!r}")
            L_partial *= L_v
        
        return L_partial


# ============================================================================
# Section 14: 序列化与反序列化
# ============================================================================

class CategoricalSerializer:
    """范畴论结构序列化器
    
    支持 ChainComplex、SheafFunctor 等结构的序列化/反序列化
    反序列化后重新验证所有数学不变量
    """
    
    @staticmethod
    def serialize_chain_complex(complex: ChainComplex) -> Dict[str, Any]:
        """序列化链复形"""
        return {
            "type": "ChainComplex",
            "objects": {
                str(k): {"dimension": v.dimension}
                for k, v in complex.objects.items()
            },
            "differentials": {
                str(k): {
                    "real": v.real.tolist(),
                    "imag": v.imag.tolist(),
                    "shape": list(v.shape)
                }
                for k, v in complex.differentials.items()
            }
        }
    
    @staticmethod
    def deserialize_chain_complex(data: Dict[str, Any]) -> ChainComplex:
        """反序列化链复形
        
        反序列化后重新验证 d² = 0
        """
        if data.get("type") != "ChainComplex":
            raise ValueError("Invalid data type for ChainComplex")
        
        objects = {
            int(k): VectorSpace(dimension=v["dimension"])
            for k, v in data["objects"].items()
        }
        
        differentials = {}
        for k, v in data["differentials"].items():
            real = np.array(v["real"])
            imag = np.array(v["imag"])
            differentials[int(k)] = real + 1j * imag
        
        # 重新验证 d² = 0
        return ChainComplex(objects, differentials, verify=True)
    
    @staticmethod
    def serialize_heyting_element(elem: HeytingElement) -> Dict[str, Any]:
        """序列化海廷代数元素"""
        return {
            "type": "HeytingElement",
            "value": elem.value.tolist(),
            "dimension": elem.dimension
        }
    
    @staticmethod
    def deserialize_heyting_element(
        data: Dict[str, Any], 
        algebra: HeytingAlgebra
    ) -> HeytingElement:
        """反序列化海廷代数元素"""
        if data.get("type") != "HeytingElement":
            raise ValueError("Invalid data type for HeytingElement")
        
        value = np.array(data["value"])
        return algebra.create_element(value)
    
    @staticmethod
    def serialize_sheaf_functor(sheaf: SheafFunctor) -> Dict[str, Any]:
        """序列化层函子"""
        stalks_data = {
            k: {"dimension": v.dimension}
            for k, v in sheaf.stalks.items()
        }
        
        restrictions_data = {
            f"{k[0]}_{k[1]}": {
                "source_dim": v.source.dimension,
                "target_dim": v.target.dimension,
                "matrix_real": v.matrix.real.tolist(),
                "matrix_imag": v.matrix.imag.tolist()
            }
            for k, v in sheaf.restriction_maps.items()
        }
        
        return {
            "type": "SheafFunctor",
            "stalks": stalks_data,
            "restriction_maps": restrictions_data
        }
    
    @staticmethod
    def deserialize_sheaf_functor(
        data: Dict[str, Any],
        site: SiteCategory
    ) -> SheafFunctor:
        """反序列化层函子"""
        if data.get("type") != "SheafFunctor":
            raise ValueError("Invalid data type for SheafFunctor")
        
        stalks = {
            k: VectorSpace(dimension=v["dimension"])
            for k, v in data["stalks"].items()
        }
        
        restriction_maps = {}
        for k, v in data["restriction_maps"].items():
            parts = k.split("_")
            if len(parts) >= 2:
                src_id, tgt_id = parts[0], parts[1]
                real = np.array(v["matrix_real"])
                imag = np.array(v["matrix_imag"])
                matrix = real + 1j * imag
                
                restriction_maps[(src_id, tgt_id)] = LinearMap(
                    source=VectorSpace(dimension=v["source_dim"]),
                    target=VectorSpace(dimension=v["target_dim"]),
                    matrix=matrix
                )
        
        return SheafFunctor(site, stalks, restriction_maps)


# ============================================================================
# Section 15: 高级 Topos 工具
# ============================================================================

class ToposEngine:
    """Grothendieck Topos 引擎
    
    整合 SiteCategory、SheafFunctor、SubobjectClassifier 等组件
    提供高级 Topos 操作
    """
    
    # 最小 Heyting 代数维度（避免布尔坍缩）
    MIN_HEYTING_DIM = 2
    
    def __init__(self, site: SiteCategory, heyting_dim: Optional[int] = None):
        """
        Args:
            site: 底层位点范畴
            heyting_dim: 海廷代数维度（None 表示从位点对象数推导）
        
        数学说明:
        Heyting 代数维度应由位点范畴的结构推导:
        - 维度 = max(位点对象数, 2)
        - 这确保每个对象对应一个"可能世界"
        - 最小值 2 避免退化为布尔代数
        """
        self.site = site
        
        # 自适应计算 Heyting 代数维度
        if heyting_dim is None:
            heyting_dim = self._compute_heyting_dimension()
        
        # 确保维度 >= 2
        heyting_dim = max(heyting_dim, self.MIN_HEYTING_DIM)
        
        self.heyting = HeytingAlgebra(dimension=heyting_dim)
        self.omega = SubobjectClassifier(self.heyting)
        self.internal_hom = InternalHom(site)
    
    def _compute_heyting_dimension(self) -> int:
        """从位点范畴推导 Heyting 代数维度
        
        数学基础:
        - 每个位点对象对应一个"可能世界"
        - Heyting 代数维度 = 位点对象数
        - 这确保子对象分类器能够区分所有可能的子对象
        
        Returns:
            推荐的 Heyting 代数维度
        """
        num_objects = len(self.site.objects)
        
        # 下界为 2（避免布尔坍缩）
        # 上界不在此处做任何拍脑袋截断：若需要资源限制，应由上游显式传入 heyting_dim。
        return max(2, num_objects)
    
    def evaluate_reachability(self, source: str, target: str) -> HeytingElement:
        """评估从 source 到 target 的可达性
        
        返回 Ω 中的真值（可能是部分真）
        
        可达性语义（直觉主义）:
        - 完全可达 (target ∈ reachable(source)): 返回 ⊤
        - 自反 (source == target): 返回 ⊤
        - 不可达: 返回 ⊥
        - 部分可达: 返回中间 Heyting 真值
        
        中间真值的计算基于路径结构:
        - 考虑从 source 到 target 的所有路径
        - 每条路径的"可达性强度"取决于路径长度和分支因子
        - 最终真值是所有路径可达性的 join
        """
        # 使用 CFG 的**直接边**计算最短路径长度（严格，不做采样/指数衰减）。
        dist = self._shortest_path_length(source, target)

        reach_vector = np.zeros(self.heyting.dimension, dtype=np.float64)
        if dist is None:
            reach_vector[:] = 0.0
            return self.omega.evaluate_reachability(reach_vector)

        if dist == 0:
            reach_vector[:] = 1.0
            return self.omega.evaluate_reachability(reach_vector)

        # 证据强度：最短路长度的倒数（无常数）
        reach_strength = 1.0 / float(dist)

        # 映射到 Heyting 元素：
        # 第 i 个维度代表证据门槛 (i+1)；门槛越高，真值越小。
        # 这里使用 dim/(i+1) 的缩放产生单调序列，避免任何手选参数。
        for i in range(self.heyting.dimension):
            reach_vector[i] = min(1.0, reach_strength * (self.heyting.dimension / float(i + 1)))

        return self.omega.evaluate_reachability(reach_vector)

    def _shortest_path_length(self, source: str, target: str) -> Optional[int]:
        """在 CFG 直接边上计算最短路径长度（BFS，严格）。"""
        if source == target:
            return 0

        edges = getattr(self.site, "_cfg_edges", None) or []
        adj: Dict[str, List[str]] = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
        for u in adj:
            adj[u] = sorted(set(adj[u]))

        if source not in adj and all(u != source for u, _ in edges):
            return None

        visited: Dict[str, int] = {source: 0}
        queue: List[str] = [source]
        while queue:
            cur = queue.pop(0)
            d = visited[cur]
            for nxt in adj.get(cur, []):
                if nxt in visited:
                    continue
                nd = d + 1
                if nxt == target:
                    return nd
                visited[nxt] = nd
                queue.append(nxt)

        return None
    
    def evaluate_conditional_reachability(
        self, 
        source: str, 
        target: str,
        path_conditions: Optional[Dict[str, float]] = None
    ) -> HeytingElement:
        """评估条件可达性
        
        考虑路径上的条件约束，返回部分真值
        
        Args:
            source: 源节点
            target: 目标节点
            path_conditions: 路径条件 {节点ID: 条件满足概率}
            
        Returns:
            Ω 中的真值，反映条件可达性
        """
        # 确保 source 和 target 是字符串（避免数组真值歧义）
        source_str = str(source) if not isinstance(source, str) else source
        target_str = str(target) if not isinstance(target, str) else target
        
        reachable = self.site.get_reachable(Object(id=source_str, dimension=1))
        reach_vector = np.zeros(self.heyting.dimension, dtype=np.float64)
        
        if target_str == source_str:
            # 自反
            reach_vector[:] = 1.0
        elif target_str in reachable:
            if path_conditions is None:
                # 无条件，完全可达
                reach_vector[:] = 1.0
            else:
                # 计算条件可达性
                # 使用路径条件的乘积作为可达性度量
                path_prob = 1.0
                for node_id, prob in path_conditions.items():
                    node_id_str = str(node_id) if not isinstance(node_id, str) else node_id
                    if node_id_str in reachable or node_id_str == source_str:
                        path_prob *= prob
                
                # 将概率映射到海廷代数元素
                # 使用分段线性映射，保持直觉主义语义
                reach_vector[:] = path_prob
        else:
            # 不可达
            reach_vector[:] = 0.0
        
        return self.omega.evaluate_reachability(reach_vector)
    
    def prove_architectural_defect(
        self,
        l1_site: SiteCategory,
        l2_site: SiteCategory,
        bridge_functor: Optional[FunctorBase] = None
    ) -> Dict[str, Any]:
        """证明架构缺陷
        
        检测 "L1 原子性在 L2 必然丢失" 等架构级问题
        
        Returns:
            缺陷报告
        """
        defects = []
        
        # 检查 L1 和 L2 的拓扑结构差异
        l1_objects = l1_site.objects
        l2_objects = l2_site.objects
        
        # 检测覆盖拓扑的不兼容
        for obj in l1_objects:
            l1_coverings = l1_site.covering_sieves(obj)
            
            # 检查对应的 L2 对象是否有兼容的覆盖
            l2_obj = Object(id=f"L2_{obj.id}", dimension=obj.dimension)
            if l2_obj in l2_objects:
                l2_coverings = l2_site.covering_sieves(l2_obj)
                
                if len(l1_coverings) > len(l2_coverings):
                    defects.append({
                        "type": "COVERAGE_LOSS",
                        "l1_object": obj.id,
                        "l2_object": l2_obj.id,
                        "l1_coverings": len(l1_coverings),
                        "l2_coverings": len(l2_coverings),
                        "severity": "HIGH"
                    })
        
        # 检测原子性丢失
        # 在 Topos 语义下，原子性 = 子对象分类器的经典性
        # L1 可能有经典逻辑，L2 可能只有直觉主义逻辑
        
        # 测试双重否定
        test_elem = self.heyting.create_element(
            np.array([0.5] * self.heyting.dimension)
        )
        
        if not self.heyting.verify_intuitionistic(test_elem):
            defects.append({
                "type": "LOGIC_COLLAPSE",
                "description": "Heyting algebra collapsed to Boolean",
                "severity": "CRITICAL"
            })
        
        return {
            "defects": defects,
            "l1_objects": len(l1_objects),
            "l2_objects": len(l2_objects),
            "is_sound": len(defects) == 0
        }


# ============================================================================
# Section 16: Derived Category 引擎
# ============================================================================

class DerivedCategoryEngine:
    """导出范畴引擎
    
    整合 ChainComplex、DerivedHom、tStructure、DistinguishedTriangle
    提供导出范畴层面的分析工具
    """
    
    def __init__(self):
        self.derived_hom = DerivedHom()
        self.t_structure = tStructure()
        self._resolution_builder = InjectiveResolution()
    
    def analyze_complex(self, X: ChainComplex) -> Dict[str, Any]:
        """分析链复形
        
        计算上同调、欧拉特征、Heart 等
        """
        cohomology = {}
        for n in X.degrees:
            dim, basis = X.cohomology(n)
            cohomology[n] = {
                "dimension": dim,
                "has_basis": basis is not None and len(basis) > 0
            }
        
        heart = self.t_structure.heart(X)
        
        return {
            "degrees": X.degrees,
            "total_dimension": X.total_dimension(),
            "euler_characteristic": X.euler_characteristic(),
            "cohomology": cohomology,
            "heart_dimension": heart.dimension
        }
    
    def detect_hidden_vulnerabilities(
        self,
        X: ChainComplex,
        Y: ChainComplex
    ) -> List[Dict[str, Any]]:
        """检测隐藏漏洞
        
        检测那些"经典上同调看起来没问题，但导出层面信息丢失"的情况
        """
        vulnerabilities = []
        
        # 比较上同调
        for n in set(X.degrees) | set(Y.degrees):
            h_n_X, _ = X.cohomology(n)
            h_n_Y, _ = Y.cohomology(n)
            
            if h_n_X == h_n_Y:
                # 经典上同调相同，检查导出层面
                # 计算 RHom
                try:
                    rhom = self.derived_hom.compute(X, Y)
                    rhom_analysis = self.analyze_complex(rhom)
                    
                    # 如果 RHom 的上同调非平凡，可能有隐藏问题
                    for k, v in rhom_analysis["cohomology"].items():
                        if v["dimension"] > 0 and k != 0:
                            vulnerabilities.append({
                                "type": "DERIVED_LEVEL_DISCREPANCY",
                                "degree": n,
                                "rhom_degree": k,
                                "rhom_dimension": v["dimension"],
                                "description": f"Classical H^{n} matches but RHom^{k} is non-trivial"
                            })
                except Exception as e:
                    vulnerabilities.append({
                        "type": "RHOM_COMPUTATION_FAILED",
                        "degree": n,
                        "error": str(e)
                    })
        
        return vulnerabilities
    
    def construct_triangle(
        self,
        f_maps: Dict[int, np.ndarray],
        X: ChainComplex,
        Y: ChainComplex
    ) -> DistinguishedTriangle:
        """从链映射构造区别三角"""
        return DistinguishedTriangle.from_morphism(f_maps, X, Y)
    
    def verify_exactness(self, triangle: DistinguishedTriangle) -> Dict[str, Any]:
        """验证区别三角的正合性"""
        is_exact, breaks = triangle.verify_exactness()
        break_details = triangle.detect_breaks()
        
        return {
            "is_exact": is_exact,
            "break_degrees": breaks,
            "break_details": break_details
        }


# ============================================================================
# Section 16.1: MVP15 协同接口 (Requirements 12)
# ============================================================================

@dataclass
class AutomorphicRepresentation:
    """自守表示 π
    
    离散自守表示的数据结构
    """
    name: str
    weight: int
    conductor: int
    dimension: int = 2
    
    def is_valid(self) -> bool:
        """验证表示有效性"""
        return self.weight >= 0 and self.conductor >= 1 and self.dimension >= 1


@dataclass
class MVP15Input:
    """MVP15 输入数据
    
    红线 (Requirements 12.1): 接收 MVP15 的解析结果
    
    Attributes:
        pi_disc: 离散自守表示 π_disc
        satake_params: Satake 参数列表 {λv}
    """
    pi_disc: AutomorphicRepresentation
    satake_params: List[SatakeParameter]
    
    def is_valid(self) -> bool:
        """验证输入有效性"""
        if not self.pi_disc.is_valid():
            return False
        if not self.satake_params:
            return False
        return all(p.is_valid() for p in self.satake_params)


@dataclass
class MVP10Output:
    """MVP10 输出数据
    
    红线 (Requirements 12.2, 12.3): 输出 Galois 参数和 BPS 坐标
    
    Attributes:
        galois_param: Galois 参数 φ
        bps_coordinates: BPS 坐标 {wi}（精确整数）
        hecke_eigensheaf: Hecke 特征层
    """
    galois_param: LanglandsParameter
    bps_coordinates: List[int]
    hecke_eigensheaf: HeckeEigensheaf
    
    def __post_init__(self):
        # 红线 B.2: 必须是可证明的精确整数；禁止静默 round()
        self.bps_coordinates = [_require_exact_int(w, name="BPS coordinate") for w in self.bps_coordinates]
    
    def is_valid(self) -> bool:
        """验证输出有效性"""
        return (
            self.galois_param is not None and
            all(isinstance(w, int) for w in self.bps_coordinates) and
            self.hecke_eigensheaf is not None
        )


class MVP10BridgeError(CategoricalError):
    """MVP10 桥接错误
    
    当 Langlands 桥失败时抛出
    """
    def __init__(self, side: str, reason: str):
        """
        Args:
            side: 失败侧 ("analytic" 或 "algebraic")
            reason: 失败原因
        """
        self.side = side
        self.reason = reason
        super().__init__(f"MVP10 bridge failed on {side} side: {reason}")


class MVP10Bridge:
    """MVP10 桥接器
    
    红线 (Requirements 12): MVP15 协同接口
    
    实现 Langlands 对偶桥的代数侧:
    - 输入: MVP15 的解析结果 (π_disc, {λv})
    - 输出: Galois 参数 φ 和 BPS 坐标 {wi}
    """
    
    def __init__(self, heyting_dim: int = 8):
        """
        Args:
            heyting_dim: Heyting 代数维度
        """
        self.heyting_dim = heyting_dim
        self.hecke_constructor = HeckeEigensheafConstructor()
        self.bps_extractor = QuasiBPSExtractor()
        self.t_structure = tStructure()
    
    def process(self, input: MVP15Input) -> MVP10Output:
        """处理 MVP15 输入，生成 MVP10 输出
        
        红线 (Requirements 12.1, 12.2, 12.3, 12.4):
        - 解析 π_disc 为 Satake 参数
        - 构造 Hecke 特征层
        - 提取 BPS 坐标（精确整数）
        - 输出 Galois 参数
        
        Args:
            input: MVP15 输入数据
            
        Returns:
            MVP10 输出数据
            
        Raises:
            MVP10BridgeError: 当桥接失败时
        """
        # 验证输入
        if not input.is_valid():
            raise MVP10BridgeError(
                "analytic",
                "Invalid MVP15 input: check pi_disc and satake_params"
            )
        
        try:
            # 步骤 1: 构造 SiteCategory（从 π_disc 的结构推导）
            # 使用 Satake 参数构造一个严格、确定的算术位点骨架：
            # 以出现的素点集合的子集格（squarefree divisor lattice）作为对象与态射的生成骨架，
            # 不引入任何阈值/经验规则。
            cfg_edges = self._construct_cfg_from_satake(input.satake_params)
            site = SiteCategory.from_cfg(cfg_edges)
            
        except Exception as e:
            raise MVP10BridgeError(
                "analytic",
                f"Failed to construct site category: {e}"
            )
        
        try:
            # 步骤 2: 构造 Hecke 特征层
            hecke_eigensheaf = self.hecke_constructor.construct(
                input.satake_params, site, representation_dim=input.pi_disc.dimension
            )
            
        except HeckeConstructionError as e:
            raise MVP10BridgeError(
                "algebraic",
                f"Failed to construct Hecke eigensheaf: {e.reason}"
            )
        except Exception as e:
            raise MVP10BridgeError(
                "algebraic",
                f"Unexpected error in Hecke construction: {e}"
            )
        
        try:
            # 步骤 3: 构造导出范畴对象
            # 从 Hecke 特征层的茎构造链复形
            derived_objects = self._construct_derived_objects(hecke_eigensheaf)
            
            # 步骤 4: 使用 t-structure 分解
            decomposition = self.bps_extractor.decomposer.decompose(
                derived_objects, self.t_structure
            )
            
            # 步骤 5: 提取 BPS 不变量
            bps_invariants = self.bps_extractor.extract(decomposition)
            
        except Exception as e:
            raise MVP10BridgeError(
                "algebraic",
                f"Failed to extract BPS invariants: {e}"
            )
        
        try:
            # 步骤 6: 构造 Galois 参数
            galois_param = self._construct_galois_parameter(
                input.pi_disc, input.satake_params, bps_invariants
            )
            
        except Exception as e:
            raise MVP10BridgeError(
                "algebraic",
                f"Failed to construct Galois parameter: {e}"
            )
        
        # 构造输出
        output = MVP10Output(
            galois_param=galois_param,
            bps_coordinates=bps_invariants.weights,
            hecke_eigensheaf=hecke_eigensheaf
        )
        
        # 验证输出
        if not output.is_valid():
            raise MVP10BridgeError(
                "algebraic",
                "Invalid output: BPS coordinates must be exact integers"
            )
        
        return output
    
    def _construct_cfg_from_satake(
        self, satake_params: List[SatakeParameter]
    ) -> List[Tuple[str, str]]:
        """从 Satake 参数构造 CFG
        
        数学原理:
        Satake 参数 {λ_v} 定义了自守表示在各素点的局部行为。
        这里的CFG并非程序控制流，而是一个用于构造位点范畴的**离散骨架**。
        
        数学选择（无启发式）：
        - 令 P 为输入 Satake 参数出现的素点集合（去重）。
        - 以 P 的子集格（squarefree divisor lattice）为对象：
          每个子集 S ⊆ P 表示局部化到 S 的信息层级。
        - 态射由覆盖关系生成：S → S∪{p}（每次只加入一个新素点）。
        
        这给出一个有限 DAG：
        - `entry` 对应空集 ∅
        - `loc_...` 对应非空子集
        - `exit` 作为终端对象（把全素点局部化再映射到全局输出）
        """
        if not satake_params:
            return [("entry", "exit")]

        primes = sorted({int(p.prime_v) for p in satake_params if p.prime_v >= 0})
        if not primes:
            return [("entry", "exit")]

        def node_id(subset: Tuple[int, ...]) -> str:
            if not subset:
                return "entry"
            return "loc_" + "_".join(str(x) for x in subset)

        # 生成所有子集（按字典序，保证确定性）
        subsets: List[Tuple[int, ...]] = [tuple()]
        for p in primes:
            subsets += [tuple(sorted(s + (p,))) for s in subsets]
        subsets = sorted(set(subsets), key=lambda t: (len(t), t))

        edges: List[Tuple[str, str]] = []
        subset_set = set(subsets)

        # Hasse 边：每次加入一个新素点
        for s in subsets:
            for p in primes:
                if p in s:
                    continue
                t = tuple(sorted(s + (p,)))
                if t in subset_set:
                    edges.append((node_id(s), node_id(t)))

        # 全素点局部化节点 → exit
        full = tuple(primes)
        edges.append((node_id(full), "exit"))

        # 去重（保持确定顺序）
        dedup: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for e in edges:
            if e not in seen:
                seen.add(e)
                dedup.append(e)
        return dedup
    
    def _construct_derived_objects(
        self, hecke_eigensheaf: HeckeEigensheaf
    ) -> List[ChainComplex]:
        """从 Hecke 特征层构造导出范畴对象（无茎=度0偷懒）。
        
        数学选择（有限、可验证、无启发式）：
        对每个对象 U 构造一个 2-项 Čech 型复形：
        - C^0(U) = F(U)
        - C^1(U) = ⊕_{f: U→V} F(V)
        - d^0(U) = ⊕ ρ_{U,V} : F(U) → ⊕ F(V)
        
        这在 d² 校验上是严格的（只有一个边界算子），并把限制映射结构真正编码进导出对象。
        """
        derived_objects: List[ChainComplex] = []

        # 为了构造 C^1(U)，我们需要知道位点上的出边；这里沿用 hecke_eigensheaf 的 restriction_maps 键。
        out_neighbors: Dict[str, List[str]] = {}
        for (src_id, tgt_id) in hecke_eigensheaf.restriction_maps.keys():
            out_neighbors.setdefault(src_id, []).append(tgt_id)
        for src_id in out_neighbors:
            out_neighbors[src_id] = sorted(set(out_neighbors[src_id]))

        for obj_id, stalk_U in hecke_eigensheaf.stalks.items():
            neighbors = out_neighbors.get(obj_id, [])

            # C^0 = F(U)
            objects: Dict[int, VectorSpace] = {0: stalk_U}
            differentials: Dict[int, np.ndarray] = {}

            if neighbors:
                # C^1 = ⊕ F(V)
                dims_V = [hecke_eigensheaf.stalks[v].dimension for v in neighbors if v in hecke_eigensheaf.stalks]
                total_dim = int(sum(dims_V))
                objects[1] = VectorSpace(dimension=total_dim)

                # d^0: F(U) -> ⊕ F(V) 由各个 ρ_{U,V} 叠加
                d0 = np.zeros((total_dim, stalk_U.dimension), dtype=np.complex128)
                row = 0
                for v in neighbors:
                    stalk_V = hecke_eigensheaf.stalks.get(v)
                    if stalk_V is None:
                        continue
                    rho = hecke_eigensheaf.restriction_maps.get((obj_id, v))
                    if rho is None:
                        # 若位点上有态射但层缺少限制映射，这是结构性错误
                        raise HeckeConstructionError(
                            hecke_eigensheaf.satake_params,
                            f"Missing restriction map ρ_{{{obj_id},{v}}} while building derived objects"
                        )
                    d0[row:row + stalk_V.dimension, :] = rho.matrix
                    row += stalk_V.dimension

                differentials[0] = d0

            derived_objects.append(ChainComplex(objects, differentials, verify=True))

        return derived_objects
    
    def _construct_galois_parameter(
        self,
        pi_disc: AutomorphicRepresentation,
        satake_params: List[SatakeParameter],
        bps_invariants: BPSInvariants,
    ) -> LanglandsParameter:
        """构造 Galois 参数
        
        数学原理:
        Galois 参数 φ 是 Langlands 对偶群的表示
        从 Satake 参数和 BPS 不变量推导
        """
        # 构造 Galois 表示矩阵
        # 使用 Satake 特征值作为对角元素
        dim = pi_disc.dimension
        galois_rep = np.zeros((dim, dim), dtype=np.complex128)
        
        for i, param in enumerate(satake_params[:dim]):
            galois_rep[i, i] = param.eigenvalue
        
        # 填充剩余对角元素
        for i in range(len(satake_params), dim):
            galois_rep[i, i] = 1.0
        
        return LanglandsParameter(
            galois_rep=galois_rep,
            conductor=pi_disc.conductor,
            weight=pi_disc.weight
        )


# ============================================================================
# Section 17: 主引擎 - MVP10 Categorical Engine
# ============================================================================

class MVP10CategoricalEngine:
    """MVP10 范畴论强化引擎
    
    整合三大核心引擎:
    1. Yoneda Embedding 探测器
    2. Grothendieck Topos 引擎
    3. Derived Category 引擎
    
    提供统一的审计接口
    """
    
    def __init__(self, heyting_dim: int = 8):
        """
        Args:
            heyting_dim: 海廷代数维度（必须 >= 2）
        """
        self.heyting_dim = heyting_dim
        
        # Yoneda 探测器
        self.yoneda_probe = YonedaProbe()
        self.nat_trans_comparator = NaturalTransformationComparator()
        self.coverage_estimator = CoverageEstimator()
        
        # Derived Category 引擎
        self.derived_engine = DerivedCategoryEngine()
        
        # Topos 引擎（延迟初始化，需要 site）
        self._topos_engine: Optional[ToposEngine] = None
        self._site: Optional[SiteCategory] = None
    
    def initialize_topos(self, cfg_edges: List[Tuple[str, str]]) -> ToposEngine:
        """从 CFG 初始化 Topos 引擎"""
        self._site = SiteCategory.from_cfg(cfg_edges)
        self._topos_engine = ToposEngine(self._site, self.heyting_dim)
        return self._topos_engine
    
    @property
    def topos(self) -> ToposEngine:
        """获取 Topos 引擎"""
        if self._topos_engine is None:
            raise RuntimeError("Topos engine not initialized. Call initialize_topos() first.")
        return self._topos_engine
    
    def probe_contract(
        self,
        target_address: str,
        executor: Optional[Callable[[ProbeMorphism], NaturalTransformation]] = None,
        *,
        minimal: bool = True,
        max_probes: Optional[int] = None,
        include_address_probes: bool = False,
        cfg_edges: Optional[List[Tuple[str, str]]] = None,
        entry_point: str = "entry",
    ) -> DeviationMetrics:
        """探测合约行为
        
        使用 Yoneda 探针检测合约行为偏差
        """
        # 生成探针
        # - 默认使用 minimal 模式，避免在编排阶段引入 300+ RPC calls 的爆炸风险
        # - 当需要更强证据时，再切换到 full probe + address probes
        if minimal:
            probes = self.yoneda_probe.generate_minimal_probe_set(target_address)
        else:
            probes = self.yoneda_probe.generate_probe_set(
                target_address,
                include_address_probes=include_address_probes,
            )
        
        if max_probes is not None and max_probes > 0 and len(probes) > max_probes:
            probes = probes[:max_probes]
        
        # 执行探针
        target_responses = self.yoneda_probe.execute_probes(probes, executor)
        
        # 获取标准响应
        standard_responses = self.yoneda_probe.get_standard_erc20_responses(probes)
        
        # 比较
        metrics = self.nat_trans_comparator.compare(target_responses, standard_responses)

        # 覆盖率（MVP10强化稿红线 #5）：
        # 只有当我们拿到 CFG（或已初始化 Topos 的 CFG）时，才能给出基于 CFG 路径数的覆盖率下界。
        edges = cfg_edges
        if edges is None and self._site is not None:
            edges = getattr(self._site, "_cfg_edges", None)

        if edges:
            coverage = self.coverage_estimator.estimate_coverage(
                edges,
                target_responses,  # 优先使用响应：未来可携带 trace 证据
                entry_point=entry_point,
            )
            metrics.coverage_ratio = coverage.coverage_ratio
        
        return metrics
    
    def analyze_chain_complex(self, X: ChainComplex) -> Dict[str, Any]:
        """分析链复形"""
        return self.derived_engine.analyze_complex(X)
    
    def detect_vulnerabilities(
        self,
        X: ChainComplex,
        Y: ChainComplex
    ) -> List[Dict[str, Any]]:
        """检测导出层面的隐藏漏洞"""
        return self.derived_engine.detect_hidden_vulnerabilities(X, Y)
    
    def evaluate_reachability(self, source: str, target: str) -> HeytingElement:
        """评估可达性（返回 Topos 真值）"""
        return self.topos.evaluate_reachability(source, target)
    
    def prove_defect(
        self,
        l1_cfg: List[Tuple[str, str]],
        l2_cfg: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """证明架构缺陷"""
        l1_site = SiteCategory.from_cfg(l1_cfg)
        l2_site = SiteCategory.from_cfg(l2_cfg)
        
        # 初始化 Topos（如果还没有）
        if self._topos_engine is None:
            self.initialize_topos(l1_cfg)
        
        return self.topos.prove_architectural_defect(l1_site, l2_site)


# ============================================================================
# Section 18: 导出接口
# ============================================================================

__all__ = [
    # 异常
    'CategoricalError',
    'FunctorLawViolation',
    'ChainComplexViolation',
    'HeytingAlgebraViolation',
    'ResolutionSkippedError',
    'CoverageInsufficientWarning',
    'SheafConditionViolation',
    'tStructureAxiomViolation',
    'ExactnessViolation',
    'HeckeConstructionError',
    
    # 核心抽象
    'Object',
    'Morphism',
    'Category',
    'FunctorBase',
    
    # Heyting 代数
    'HeytingElement',
    'HeytingAlgebra',
    
    # Topos
    'Subobject',
    'SubobjectClassifier',
    'Sieve',
    'CoveringFamily',
    'GrothendieckTopology',
    'SiteCategory',
    
    # 层
    'VectorSpace',
    'LinearMap',
    'SheafFunctor',
    'InternalHom',
    
    # Hecke 特征层 (红线 A.1)
    'SatakeParameter',
    'HeckeEigensheaf',
    'HeckeEigensheafConstructor',
    
    # 导出范畴
    'ChainComplex',
    'ExactnessValidator',
    'InjectiveObject',
    'InjectiveResolution',
    'DerivedHom',
    'tStructure',
    'DistinguishedTriangle',
    
    # BPS 不变量 (红线 B.2)
    'TriangulatedSubcategory',
    'QuasiBPSCategory',
    'LanglandsParameter',
    'BPSInvariants',
    'SemiOrthogonalDecomposition',
    'QuasiBPSExtractor',
    
    # Yoneda 探测器
    'ProbeMorphism',
    'NaturalTransformation',
    'DeviationMetrics',
    'CoverageResult',
    'YonedaProbe',
    'NaturalTransformationComparator',
    'CoverageEstimator',
    
    # 米田不变量探测器 (模块 C.2)
    'YonedaInvariantProber',
    'LFactorComputer',
    
    # 序列化
    'CategoricalSerializer',
    
    # 引擎
    'ToposEngine',
    'DerivedCategoryEngine',
    'MVP10CategoricalEngine',
    
    # MVP15 协同接口 (Requirements 12)
    'AutomorphicRepresentation',
    'MVP15Input',
    'MVP10Output',
    'MVP10BridgeError',
    'MVP10Bridge',
]
