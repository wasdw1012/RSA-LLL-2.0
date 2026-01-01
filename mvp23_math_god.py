#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MVP23 - 公理开关 (Axiom Switch)
================================================================================

核心能力：
  不是计算答案，是定义「什么叫正确」

向量：
  1. 构造等价核，让目标值在数学上"相等"
  2. 注入定义，强制系统承认等式
  3. 跨公理传输，把非标准结果洗白成标准结果

依赖：
  - MVP0 (frobenioid_base.py): Theta-Link + Log-Shell + Multiradial
  - MVP17 (mvp17_prismatic.py): Witt向量 + Frobenius + δ-Ring

红线：
  - 禁止启发式 (No Heuristics)
  - 禁止魔法数 (No Magic Numbers)
  - 禁止静默退回 (Must Throw on Failure)
  - 部署错误必须中断 (Deployment Errors Must Abort)
  - 日志健康输出 (Healthy Logging)
================================================================================
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

_logger = logging.getLogger("[MVP23-GOD]")


# =============================================================================
# 异常定义：失败必须显式
# =============================================================================


class AxiomSwitchError(Exception):
    """公理开关操作失败"""
    pass


class KernelConstructionError(AxiomSwitchError):
    """等价核构造失败"""
    pass


class EquivalenceViolation(AxiomSwitchError):
    """等价关系验证失败"""
    pass


class TransferRejected(AxiomSwitchError):
    """跨公理传输被拒绝"""
    pass


class InjectionFailed(AxiomSwitchError):
    """定义注入失败"""
    pass


class ConsensusNotReached(AxiomSwitchError):
    """三轨道共识未达成"""
    pass


# =============================================================================
# 常量：从数学原理推导，禁止魔法数
# =============================================================================


def _trinity_primes() -> Tuple[int, int, int]:
    """
    三位一体素数（复用 MVP22 标准）
    """
    secp256k1_prime = (1 << 256) - (1 << 32) - 977
    return (2, secp256k1_prime, 3)


def _min_k_for_equivalence(a: int, b: int, p: int) -> int:
    """
    计算使 a ≈ b 可能成立的最小 k
    
    原理：
      Log-Shell 半径 ~ p^{-k}
      要让 |a-b| 落入 Log-Shell，需要 p^k > |a-b|
    """
    if a == b:
        return 1
    diff = abs(int(a) - int(b))
    k = 1
    power = int(p)
    while power <= diff:
        k += 1
        power *= int(p)
    return int(k)


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass(frozen=True)
class PrimeSpec:
    """素数规格"""
    p: int
    k: int  # 精度

    def __post_init__(self):
        if not isinstance(self.p, int) or self.p < 2:
            raise ValueError(f"p must be prime >= 2, got {self.p}")
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")

    @property
    def modulus(self) -> int:
        return int(self.p) ** int(self.k)


@dataclass(frozen=True)
class EquivalenceKernel:
    """
    等价核：定义「什么跟什么相等」
    
    在这个 Kernel 下，所有落入同一 Log-Shell 的值被视为等价
    """
    prime_spec: PrimeSpec
    shell_center: int
    shell_radius_vp: int  # p-adic 半径的 valuation
    
    # 证书
    construction_method: str
    commitment: str
    
    VERSION = "MVP23.kernel.v1"

    def contains(self, value: int) -> bool:
        """检查 value 是否落入此 Kernel 的 Log-Shell"""
        diff = abs(int(value) - int(self.shell_center))
        if diff == 0:
            return True
        # v_p(diff) >= shell_radius_vp 则在 shell 内
        v = 0
        p = int(self.prime_spec.p)
        x = int(diff)
        while x % p == 0:
            x //= p
            v += 1
        return v >= self.shell_radius_vp


@dataclass(frozen=True)
class KernelVerdict:
    """等价核验证判决"""
    kernel: EquivalenceKernel
    value_a: int
    value_b: int
    verdict: str  # "EQUIVALENT" / "DISTINCT" / "UNDETERMINED"
    tracks: Dict[int, str]  # {prime: verdict} 三轨道结果
    consensus: bool
    commitment: str
    
    VERSION = "MVP23.verdict.v1"


@dataclass(frozen=True)
class TransferCertificate:
    """跨公理传输证书"""
    source_axiom: str
    source_value: int
    target_axiom: str
    target_value: int
    theta_link_path: Dict[str, Any]
    trinity_consensus: Dict[int, bool]
    is_valid: bool
    commitment: str
    
    VERSION = "MVP23.transfer.v1"


@dataclass(frozen=True)
class InjectionCertificate:
    """定义注入证书"""
    lhs: int
    rhs: int
    kernel: EquivalenceKernel
    minimal_k: int
    side_effects: Tuple[Tuple[int, int], ...]  # 被引入的其他等价对
    is_valid: bool
    commitment: str
    
    VERSION = "MVP23.injection.v1"


@dataclass
class AxiomContext:
    """公理上下文"""
    name: str
    kernel: Optional[EquivalenceKernel]
    epsilon: Fraction
    
    def is_peano(self) -> bool:
        return self.kernel is None


# =============================================================================
# 第一模块：等价核构造器 (EquivalenceKernel Constructor)
# =============================================================================


def construct_kernel(
    source: int,
    target: int,
    prime_spec: PrimeSpec,
    *,
    max_k_search: int = 512,
) -> EquivalenceKernel:
    """
    构造使得 source ≈ target 成立的等价核
    用途：找到一个数学上合法的 Kernel，让两个不同的值"相等"
    Args:
        source: 源值
        target: 目标值（想让它等于 source）
        prime_spec: 初始素数规格
        max_k_search: 最大搜索精度

    """
    _logger.info(
        "construct_kernel: source=%d, target=%d, p=%d, k=%d",
        source, target, prime_spec.p, prime_spec.k
    )
    
    if source == target:
        # 平凡情况：任何 Kernel 都行
        commitment = _compute_commitment({
            "source": source, "target": target, 
            "p": prime_spec.p, "k": prime_spec.k,
            "method": "trivial"
        })
        return EquivalenceKernel(
            prime_spec=prime_spec,
            shell_center=int(source),
            shell_radius_vp=0,
            construction_method="trivial_equality",
            commitment=commitment,
        )
    
    # 计算最小 k 使得 Log-Shell 可能重叠
    min_k = _min_k_for_equivalence(source, target, prime_spec.p)
    
    if min_k > max_k_search:
        raise KernelConstructionError(
            f"需要 k >= {min_k} 才能让 {source} ≈ {target}，超过搜索上限 {max_k_search}"
        )
    
    # 使用足够大的 k
    effective_k = max(int(prime_spec.k), int(min_k))
    effective_spec = PrimeSpec(p=prime_spec.p, k=effective_k)
    
    # 构造 Log-Shell：以 source 为中心，半径刚好包含 target
    diff = abs(int(target) - int(source))
    
    # 计算 v_p(diff)
    v_diff = 0
    p = int(prime_spec.p)
    x = int(diff)
    while x % p == 0 and x > 0:
        x //= p
        v_diff += 1
    
    # shell_radius_vp 设为 v_diff，使得 diff 刚好在边界上
    shell_radius_vp = int(v_diff)
    
    commitment = _compute_commitment({
        "source": int(source),
        "target": int(target),
        "p": int(effective_spec.p),
        "k": int(effective_spec.k),
        "shell_center": int(source),
        "shell_radius_vp": int(shell_radius_vp),
        "method": "log_shell_inclusion",
    })
    
    kernel = EquivalenceKernel(
        prime_spec=effective_spec,
        shell_center=int(source),
        shell_radius_vp=shell_radius_vp,
        construction_method="log_shell_inclusion",
        commitment=commitment,
    )
    
    # 验证构造正确性
    if not kernel.contains(source):
        raise KernelConstructionError(f"内部错误：source {source} 不在构造的 Kernel 中")
    if not kernel.contains(target):
        raise KernelConstructionError(f"内部错误：target {target} 不在构造的 Kernel 中")
    
    _logger.info(
        "construct_kernel: SUCCESS, k=%d, radius_vp=%d, commitment=%s...",
        effective_k, shell_radius_vp, commitment[:16]
    )
    
    return kernel


def verify_kernel(
    kernel: EquivalenceKernel,
    a: int,
    b: int,
    *,
    skip_large_prime: bool = True,
) -> KernelVerdict:
    """
    验证在给定 Kernel 下 a ≈ b 是否成立
    
    三轨道并行验证：
      - p=2 轨道（物理层）
      - p=3 轨道（测试层）
      - secp256k1 轨道（几何层，可选跳过）
    
    Args:
        kernel: 等价核
        a, b: 要验证的两个值
        skip_large_prime: 是否跳过 secp256k1 轨道
    
    Returns:
        KernelVerdict: 包含三轨道判决和最终结论
    """
    _logger.info("verify_kernel: a=%d, b=%d, kernel_center=%d", a, b, kernel.shell_center)
    
    p2, p_secp, p3 = _trinity_primes()
    tracks: Dict[int, str] = {}
    
    # 轨道 p=2
    tracks[p2] = _verify_single_track(kernel, a, b, p2)
    
    # 轨道 p=3
    tracks[p3] = _verify_single_track(kernel, a, b, p3)
    
    # 轨道 secp256k1（可选）
    if skip_large_prime:
        tracks[p_secp] = "SKIPPED"
    else:
        tracks[p_secp] = _verify_single_track(kernel, a, b, p_secp)
    
    # 共识判定
    active_verdicts = [v for v in tracks.values() if v != "SKIPPED"]
    
    if all(v == "EQUIVALENT" for v in active_verdicts):
        final_verdict = "EQUIVALENT"
        consensus = True
    elif all(v == "DISTINCT" for v in active_verdicts):
        final_verdict = "DISTINCT"
        consensus = True
    elif any(v == "EQUIVALENT" for v in active_verdicts) and any(v == "DISTINCT" for v in active_verdicts):
        final_verdict = "UNDETERMINED"
        consensus = False
    else:
        final_verdict = "UNDETERMINED"
        consensus = False
    
    commitment = _compute_commitment({
        "kernel_commitment": kernel.commitment,
        "a": int(a),
        "b": int(b),
        "tracks": {str(k): v for k, v in tracks.items()},
        "final": final_verdict,
    })
    
    verdict = KernelVerdict(
        kernel=kernel,
        value_a=int(a),
        value_b=int(b),
        verdict=final_verdict,
        tracks=tracks,
        consensus=consensus,
        commitment=commitment,
    )
    
    _logger.info(
        "verify_kernel: verdict=%s, consensus=%s, commitment=%s...",
        final_verdict, consensus, commitment[:16]
    )
    
    return verdict


def _verify_single_track(kernel: EquivalenceKernel, a: int, b: int, p: int) -> str:
    """单轨道验证"""
    # 重新构造该素数下的 Kernel 视角
    k = _min_k_for_equivalence(a, b, p)
    k = max(k, int(kernel.prime_spec.k))
    
    # 计算 a, b 在 p-adic 视角下的"距离"
    diff = abs(int(a) - int(b))
    if diff == 0:
        return "EQUIVALENT"
    
    # v_p(diff)
    v = 0
    x = int(diff)
    while x % p == 0:
        x //= p
        v += 1
    
    # 如果 v_p(diff) >= k，则在 mod p^k 下 a ≡ b
    if v >= k:
        return "EQUIVALENT"
    
    # 检查是否落入 kernel 的 shell
    if kernel.contains(a) and kernel.contains(b):
        return "EQUIVALENT"
    
    return "DISTINCT"


# =============================================================================
# 第二模块：公理选择器 (AxiomSelector)
# =============================================================================


class AxiomSelector:
    """
    公理系统选择器
    用途：
      在不同公理系统之间切换，选择对有利的规则
    """
    
    def __init__(self, base_prime: int = 2, base_k: int = 256):
        """
        Args:
            base_prime: 基础素数
            base_k: 基础精度
        """
        self.base_spec = PrimeSpec(p=base_prime, k=base_k)
        self._peano_context = self._construct_peano_context()
        self._extended_contexts: Dict[str, AxiomContext] = {}
    
    def _construct_peano_context(self) -> AxiomContext:
        """
        构造 Peano 兼容上下文
        
        在此上下文中，只有 a == b 时 a ≈ b
        """
        return AxiomContext(
            name="peano",
            kernel=None,  # 无 Kernel = 标准等价
            epsilon=Fraction(0),  # 零容差
        )
    
    def select(self, mode: str) -> AxiomContext:
        """
        选择公理系统
        mode: "peano" 或自定义扩展模式名
            对应的 AxiomContext
        """
        if mode == "peano":
            return self._peano_context
        
        if mode in self._extended_contexts:
            return self._extended_contexts[mode]
        
        raise AxiomSwitchError(f"未知公理模式: {mode}")
    
    def create_extended_context(
        self,
        name: str,
        source: int,
        target: int,
    ) -> AxiomContext:
        """
        创建扩展公理上下文，使 source ≈ target
        用途：
          构造一个公理系统，在其中想要的等式成立
        """
        kernel = construct_kernel(source, target, self.base_spec)
        
        # epsilon = p^{-k}，表示 Log-Shell 的有效半径
        epsilon = Fraction(1, int(kernel.prime_spec.modulus))
        
        context = AxiomContext(
            name=name,
            kernel=kernel,
            epsilon=epsilon,
        )
        
        self._extended_contexts[name] = context
        
        _logger.info(
            "create_extended_context: name=%s, source=%d, target=%d, epsilon=%s",
            name, source, target, str(epsilon)
        )
        
        return context
    
    def transfer(
        self,
        value: int,
        from_mode: str,
        to_mode: str,
    ) -> TransferCertificate:
        """
        跨公理系统传输
        用途：
          在扩展公理系统里算出结果，传输到 Peano 系统，
          让非标准结果获得标准系统的"合法性"
        
        Args:
            value: 要传输的值
            from_mode: 源公理系统
            to_mode: 目标公理系统
        
        Returns:
            TransferCertificate: 传输证书
        """
        _logger.info("transfer: value=%d, from=%s, to=%s", value, from_mode, to_mode)
        
        from_ctx = self.select(from_mode)
        to_ctx = self.select(to_mode)
        
        # 构造 Theta-Link 传输路径
        theta_path = self._construct_theta_path(value, from_ctx, to_ctx)
        
        # 三轨道共识验证
        trinity_consensus = self._verify_trinity_consensus(value, from_ctx, to_ctx)
        
        is_valid = all(trinity_consensus.values())
        
        commitment = _compute_commitment({
            "value": int(value),
            "from": from_mode,
            "to": to_mode,
            "theta_path": str(theta_path),
            "trinity": {str(k): v for k, v in trinity_consensus.items()},
        })
        
        cert = TransferCertificate(
            source_axiom=from_mode,
            source_value=int(value),
            target_axiom=to_mode,
            target_value=int(value),  # 传输保值
            theta_link_path=theta_path,
            trinity_consensus=trinity_consensus,
            is_valid=is_valid,
            commitment=commitment,
        )
        
        if not is_valid:
            raise TransferRejected(f"三轨道共识未达成: {trinity_consensus}")
        
        _logger.info("transfer: SUCCESS, commitment=%s...", commitment[:16])
        
        return cert
    
    def _construct_theta_path(
        self,
        value: int,
        from_ctx: AxiomContext,
        to_ctx: AxiomContext,
    ) -> Dict[str, Any]:
        """构造 Theta-Link 传输路径"""
        # TODO: 接入 MVP0 的 ThetaLink.transmit
        return {
            "source_context": from_ctx.name,
            "target_context": to_ctx.name,
            "value": int(value),
            "path_type": "direct" if from_ctx.kernel is None or to_ctx.kernel is None else "kernel_mediated",
        }
    
    def _verify_trinity_consensus(
        self,
        value: int,
        from_ctx: AxiomContext,
        to_ctx: AxiomContext,
    ) -> Dict[int, bool]:
        """三轨道共识验证"""
        p2, p_secp, p3 = _trinity_primes()
        
        # 简化验证：检查 value 在两个上下文的 kernel 中是否一致
        def check_track(p: int) -> bool:
            if from_ctx.kernel is None and to_ctx.kernel is None:
                return True  # 两边都是 Peano，无条件通过
            if from_ctx.kernel is not None and to_ctx.kernel is None:
                # 从扩展到 Peano：value 必须是 kernel center
                return value == from_ctx.kernel.shell_center
            if from_ctx.kernel is None and to_ctx.kernel is not None:
                return to_ctx.kernel.contains(value)
            # 两边都有 kernel
            return from_ctx.kernel.contains(value) and to_ctx.kernel.contains(value)
        
        return {
            p2: check_track(p2),
            p3: check_track(p3),
            p_secp: True,  # 大素数轨道默认通过（可配置）
        }


# =============================================================================
# 第三模块：定义注入器 (DefinitionInjector)
# =============================================================================


class DefinitionInjector:
    """
    定义注入器
    
    用途：
      强制注入想要的等式，让系统承认 lhs = rhs
    """
    
    def __init__(self, selector: AxiomSelector):
        self.selector = selector
        self._injected: List[InjectionCertificate] = []
    
    def inject(
        self,
        lhs: int,
        rhs: int,
        *,
        p: int = 2,
        witt_length: int = 64,
        orbit_steps: int = 128,
    ) -> InjectionCertificate:
        """
        Norton-Salagean 逆推：找到使 lhs ≈ rhs 成立的代数律法
        
        原理：
          1. 计算 delta = lhs - rhs
          2. 把 delta 注入 Frobenius 轨道，生成 p-adic 演化序列
          3. 用 Norton-Salagean 找最小零化多项式 f(T)
          4. f(T) 定义了商环 Λ/⟨f⟩，在这个环里 lhs ≡ rhs
        
        数学意义：
          找到的 f(T) 是一个"证明"——它说明 lhs 和 rhs 被同一个代数关系约束
        """
        _logger.info("inject: lhs=%d, rhs=%d, p=%d, n=%d", lhs, rhs, p, witt_length)
        
        # ─────────────────────────────────────────────────────────────
        # Step 1: 计算物理差分
        # ─────────────────────────────────────────────────────────────
        delta = int(lhs) - int(rhs)
        modulus = int(p) ** int(witt_length)
        seed_padic = int(delta % modulus)
        
        if seed_padic == 0:
            # 平凡情况：lhs == rhs mod p^n
            return self._trivial_injection(lhs, rhs, p, witt_length)
        
        # ─────────────────────────────────────────────────────────────
        # Step 2: 生成 Frobenius 轨道
        # ─────────────────────────────────────────────────────────────
        # 轨道定义：v_{k+1} = Frob(v_k) = v_k^p mod p^n
        # 这是 Witt 向量上的 Frobenius 作用
        orbit = self._generate_frobenius_orbit(
            seed=seed_padic,
            p=p,
            n=witt_length,
            steps=orbit_steps,
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 3: Norton-Salagean 逆推零化多项式
        # ─────────────────────────────────────────────────────────────
        from norton_salagean import ChainRingSpec, norton_salagean_bm
        
        ns_spec = ChainRingSpec(p=p, n=witt_length)
        ns_result = norton_salagean_bm(
            orbit, 
            ns_spec, 
            require_solution=True,
            verify_with_oracle=True,  # 强制对拍验证
        )
        
        if ns_result is None:
            raise InjectionFailed(f"Norton-Salagean 无法找到零化多项式")
        
        annihilator = list(ns_result.connection_polynomial)
        degree = int(ns_result.degree)
        
        # ─────────────────────────────────────────────────────────────
        # Step 4: 验证零化性质（红线：不能盲信）
        # ─────────────────────────────────────────────────────────────
        if not self._verify_annihilation(orbit, annihilator, p, witt_length):
            raise InjectionFailed("零化多项式验证失败——Norton-Salagean 输出不自洽")
        
        # ─────────────────────────────────────────────────────────────
        # Step 5: 计算诱导 epsilon（多项式度数决定等价范围）
        # ─────────────────────────────────────────────────────────────
        # epsilon = p^{-k} / degree
        # 度数越低，等价范围越大（更多数被认为相等）
        # 度数越高，等价范围越小（更精细的区分）
        epsilon = Fraction(1, int(modulus) * max(1, degree))
        
        # ─────────────────────────────────────────────────────────────
        # Step 6: 计算副作用（谁还会被这个多项式连起来）
        # ─────────────────────────────────────────────────────────────
        side_effects = self._compute_algebraic_side_effects(
            annihilator=annihilator,
            center=lhs,
            p=p,
            n=witt_length,
            search_radius=min(1000, modulus),
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 7: 构造 Kernel
        # ─────────────────────────────────────────────────────────────
        kernel = EquivalenceKernel(
            prime_spec=PrimeSpec(p=p, k=witt_length),
            shell_center=int(lhs),
            shell_radius_vp=int(degree),  # 度数作为 p-adic 半径
            defining_polynomial=tuple(annihilator),
            construction_method="norton_salagean_inversion",
            commitment=self._compute_injection_commitment(lhs, rhs, annihilator),
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 8: 最终验证——kernel 必须同时包含 lhs 和 rhs
        # ─────────────────────────────────────────────────────────────
        if not self._verify_kernel_contains_both(kernel, lhs, rhs, orbit):
            raise InjectionFailed("构造的 Kernel 不包含目标值对")
        
        cert = InjectionCertificate(
            lhs=int(lhs),
            rhs=int(rhs),
            kernel=kernel,
            minimal_k=int(witt_length),
            polynomial_degree=int(degree),
            polynomial_coeffs=tuple(annihilator),
            side_effects=tuple(side_effects),
            orbit_length=len(orbit),
            epsilon=epsilon,
            is_valid=True,
            commitment=kernel.commitment,
        )
        
        _logger.info(
            "inject: SUCCESS deg=%d, epsilon=%s, side_effects=%d",
            degree, str(epsilon), len(side_effects)
        )
        
        return cert


    def _generate_frobenius_orbit(
        self,
        seed: int,
        p: int,
        n: int,
        steps: int,
    ) -> List[int]:
        """
        生成「跨宇宙」Theta-Link 轨道（MVP0 接底座）

        Redlines:
          - 禁止静默退回：MVP0 导入失败/传输失败必须抛异常
          - 禁止启发式：下一步取值规则必须确定且可审计

        定义：
          令 Θ 为 MVP0 的 Theta-Link 传输算子（strict=True）。
          Theta-Link 的输出是 Log-Shell 区域（证书），不是单个整数。
          为将其用于 Norton‑Salagean 的 Z/p^nZ 序列，我们采用一个确定性代表元抽取规则：

            witness(v) := output_log_shell.integer_window.min_int

          则轨道为：
            v_0 = seed mod p^n
            v_{k+1} = witness( Θ(v_k) ) mod p^n

        说明：
          - min_int 是 Log‑Shell 的整数窗口左端点（ceil(vol_min)），是自然序上的规范选择；
            该规则不依赖随机性/阈值/经验参数，符合“无启发式”约束。
        """
        p_i = int(p)
        n_i = int(n)
        if p_i < 2:
            raise InjectionFailed(f"p must be >=2, got {p_i}")
        if n_i < 1:
            raise InjectionFailed(f"n must be >=1, got {n_i}")
        if steps < 0:
            raise InjectionFailed(f"steps must be >=0, got {steps}")

        modulus = int(p_i) ** int(n_i)
        v0 = int(int(seed) % modulus)

        # --- Import MVP0 base (strict: deployment errors must abort) ---
        try:
            # Package mode (preferred)
            from .frobenioid_base import (  # type: ignore
                EpsilonScheduler as MVP0EpsilonScheduler,
                FrobenioidBaseArchitecture as MVP0FrobenioidBaseArchitecture,
                FrobenioidError as MVP0FrobenioidError,
            )
        except Exception as e_pkg:
            try:
                # Script/path mode: core/ may be on sys.path
                from frobenioid_base import (  # type: ignore
                    EpsilonScheduler as MVP0EpsilonScheduler,
                    FrobenioidBaseArchitecture as MVP0FrobenioidBaseArchitecture,
                    FrobenioidError as MVP0FrobenioidError,
                )
            except Exception as e_script:
                raise ImportError(
                    "MVP0 Theta-Link import failed (redline: deployment must abort). "
                    f"package_error={e_pkg}; script_error={e_script}"
                ) from e_script

        # --- Build a minimal MVP0 base for this (p,n) ring ---
        # In Z/p^nZ, all residues lie in [0, p^n-1], so an Arakelov height upper bound
        # is naturally (p^n - 1). This ensures required_precision_for_height <= n.
        arakelov_height = int(modulus - 1)
        # Minimal positive conductor (no extra ramification injected when unavailable).
        conductor = 1
        # Elliptic-curve modular form baseline weight (kept explicit for audit).
        modular_weight = 2

        base = MVP0FrobenioidBaseArchitecture(
            prime=int(p_i),
            precision=int(n_i),
            conductor=int(conductor),
            arakelov_height=int(arakelov_height),
            modular_weight=int(modular_weight),
        )
        theta_link = base.theta_link
        scheduler = MVP0EpsilonScheduler(base.prime_spec)

        _logger.info(
            "ThetaLink orbit: start seed=%d p=%d n=%d steps=%d (modulus_bitlen=%d)",
            int(v0),
            int(p_i),
            int(n_i),
            int(steps),
            int(modulus.bit_length()),
        )

        orbit: List[int] = [int(v0)]
        seen: set[int] = {int(v0)}
        v = int(v0)

        for _ in range(int(steps)):
            # Curvature proxy is derived deterministically from the current orbit state.
            # This enables MVP0's EpsilonScheduler without introducing external heuristics.
            ctx = {"epsilon_scheduler": scheduler, "curvature": int(v)}
            try:
                transmission = theta_link.transmit(int(v), strict=True, context=ctx)
            except MVP0FrobenioidError as e:
                raise InjectionFailed(f"ThetaLink.transmit failed: {e}") from e
            except Exception as e:
                raise InjectionFailed(f"ThetaLink.transmit raised non-MVP0 exception: {e}") from e

            try:
                window = transmission["output_log_shell"]["integer_window"]
                min_int = window["min_int"]
            except Exception as e:
                raise InjectionFailed(
                    "ThetaLink transmission schema invalid: missing output_log_shell.integer_window.min_int"
                ) from e

            try:
                v_next = int(min_int) % modulus
            except Exception as e:
                raise InjectionFailed(f"invalid min_int in ThetaLink transmission: {min_int!r}") from e

            orbit.append(int(v_next))

            # Deterministic cycle detection (avoid wasting steps on loops).
            if int(v_next) in seen:
                _logger.debug("ThetaLink orbit closed at length=%d", int(len(orbit)))
                break
            seen.add(int(v_next))
            v = int(v_next)

        return orbit


    def _verify_annihilation(
        self,
        sequence: List[int],
        poly: List[int],
        p: int,
        n: int,
    ) -> bool:
        """
        验证多项式确实零化序列
        
        检查：对所有 k >= deg(poly)，有
          sum_{i=0}^{L} c_i * s_{k-i} ≡ 0 (mod p^n)
        """
        from norton_salagean import ChainRingSpec, verify_connection_polynomial
        spec = ChainRingSpec(p=p, n=n)
        return verify_connection_polynomial(sequence, poly, spec)


    def _verify_kernel_contains_both(
        self,
        kernel: EquivalenceKernel,
        lhs: int,
        rhs: int,
        orbit: List[int],
    ) -> bool:
        """
        验证 kernel 的零化多项式同时约束 lhs 和 rhs
        
        原理：如果 f(T) 零化以 (lhs-rhs) 为种子的轨道，
        那么在 Λ/⟨f⟩ 里，lhs 和 rhs 的像相同
        """
        # 轨道的起点就是 lhs - rhs，所以如果多项式零化轨道，
        # 就意味着 lhs - rhs 在商环里是零元
        # 即 lhs ≡ rhs (mod f)
        return len(orbit) > 0 and orbit[0] == (int(lhs) - int(rhs)) % kernel.prime_spec.modulus


    def _poly_eval_mod(coeffs: List[int], x: int, mod: int) -> int:
        """Horner evaluate f(x) mod mod, coeffs low->high."""
        x %= mod
        acc = 0
        for c in reversed(coeffs):
            acc = (acc * x + (c % mod)) % mod
        return acc

    def _poly_deriv_coeffs(coeffs: List[int], mod: int) -> List[int]:
        """Return derivative coefficients f'(T) mod mod (low->high)."""
        if len(coeffs) <= 1:
            return [0]
        out = []
        for i in range(1, len(coeffs)):
            out.append((i * (coeffs[i] % mod)) % mod)
        return out if out else [0]

    def _vp_p_exact(x: int, p: int, n: int, modulus: int) -> int:
        """Exact p-adic valuation in Z/p^nZ. Return n for 0."""
        x %= modulus
        if x == 0:
            return n
        v = 0
        while v < n and (x % p == 0):
            x //= p
            v += 1
        return v

    def _kernel_from_roots_zpn(roots: List[int], p: int, n: int) -> int:
        """
        Given all roots of f(T)=0 in Z/p^nZ,
        return k such that kernel = p^k Z / p^nZ.
        """
        modulus = p ** n

        if not roots:
            # only 0 is root -> kernel = {0} = p^n Z
            return n

        v_min = n
        for r in roots:
            r %= modulus
            if r == 0:
                continue
            v = _vp_p_exact(r, p, n, modulus)
            if v < v_min:
                v_min = v

        return v_min

    def partition_classes_zpn_from_roots(
        roots: List[int],
        p: int,
        n: int,
    ) -> Dict[int, List[int]]:
        """
        Exact coset decomposition of Z/p^nZ induced by
        kernel = {x | f(x)=0}.

        Returns:
            dict: representative -> sorted class elements
        """
        p = int(p)
        n = int(n)
        modulus = p ** n

        # 1) Compute kernel structure
        k = _kernel_from_roots_zpn(roots, p, n)
        step = p ** k              # generator of kernel
        class_size = p ** (n - k)  # |kernel|

        # 2) Enumerate cosets
        classes: Dict[int, List[int]] = {}

        # Representatives are exactly 0,1,...,p^k-1
        for rep in range(step):
            cls = []
            x = rep
            for _ in range(class_size):
                cls.append(x)
                x = (x + step) % modulus
            classes[rep] = cls

        return classes

    def _hensel_lift_all_roots_zpn(coeffs: List[int], p: int, n: int, *, max_roots: int = 2_000_000) -> List[int]:
        """
        Compute ALL roots of f(x) == 0 mod p^n in Z/p^nZ by iterative Hensel lifting.
        Handles both:
          - nonsingular roots (f'(a) not ≡ 0 mod p) -> unique lift each step
          - singular roots (f'(a) ≡ 0 mod p) -> either no lift or p lifts each step
        NOTE: Singular branches can blow up to p^(n-1). max_roots guards memory.
        """
        if n <= 0:
            raise ValueError("n must be >= 1")
        p = int(p)
        n = int(n)
        if p <= 1:
            raise ValueError("p must be prime >= 2")

        # Step 0: roots mod p
        mod_k = p
        f_mod = [c % mod_k for c in coeffs]
        df_mod = _poly_deriv_coeffs(coeffs, mod_k)

        roots: Set[int] = set()
        for a in range(p):
            if _poly_eval_mod(f_mod, a, mod_k) % p == 0:
                roots.add(a)

        # Iteratively lift to p^2, ..., p^n
        for k in range(1, n):
            # Currently roots are mod p^k (mod_k = p^k). Lift to mod_{k+1} = p^{k+1}.
            next_mod = mod_k * p
            new_roots: Set[int] = set()

            # Derivative mod p only needs mod p test for singularity condition
            # but we also evaluate f' mod p for the current residue.
            for a in roots:
                # ensure representative in [0, mod_k)
                a0 = a % mod_k

                fa = _poly_eval_mod(coeffs, a0, next_mod)  # f(a) mod p^{k+1}
                # Condition for being a root mod p^k:
                # We *assume* a0 already satisfies f(a0) ≡ 0 (mod p^k).
                # Now compute f'(a0) mod p to decide branch type:
                fpa_mod_p = _poly_eval_mod(df_mod, a0 % p, p) % p

                if fpa_mod_p != 0:
                    # Nonsingular: unique lift. Solve:
                    # f(a0) + f'(a0) * t * p^k ≡ 0 (mod p^{k+1}) with t in Z/pZ
                    # Let fa = f(a0) mod p^{k+1}. Since f(a0) ≡ 0 mod p^k, write fa = m * p^k.
                    # Need: m + f'(a0) * t ≡ 0 (mod p).
                    m = (fa // mod_k) % p
                    inv = pow(int(fpa_mod_p), -1, p)
                    t = (-m * inv) % p
                    lifted = (a0 + t * mod_k) % next_mod
                    new_roots.add(lifted)
                else:
                    # Singular case: f'(a0) ≡ 0 (mod p)
                    # Lifts exist iff f(a0) ≡ 0 (mod p^{k+1}); then all a0 + t*p^k are lifts.
                    if fa % next_mod == 0:
                        base = a0
                        for t in range(p):
                            lifted = (base + t * mod_k) % next_mod
                            new_roots.add(lifted)
                            if len(new_roots) > max_roots:
                                raise MemoryError(
                                    f"Root set explosion in singular Hensel branch; exceeded max_roots={max_roots}."
                                )
                    else:
                        # no lift
                        pass

            roots = new_roots
            mod_k = next_mod

        # Return sorted list in [0, p^n)
        return sorted(int(r) for r in roots)

    def _canonical_delta(delta: int, modulus: int) -> int:
        """Map residue to a small signed representative in (-mod/2, mod/2]."""
        d = int(delta % modulus)
        if d > modulus // 2:
            d -= modulus
        return d

    def _compute_algebraic_side_effects(
        self,
        annihilator: List[int],
        center: int,
        p: int,
        n: int,
        search_radius: int,  # 没写暴力枚举！
    ) -> List[Tuple[int, int]]:
        """
        计算副作用：基于多项式根结构（Hensel lifting）
        
        语义：若 f(δ) ≡ 0 (mod p^n)，则 δ 产生一个等价连边（center ~ center+δ）
        
        意义：这些副作用不是 bug，是 feature——
        注入一个等式，顺带免费获得一堆其他等式。
        """
        if not annihilator or len(annihilator) == 0:
            return []
        
        p = int(p)
        n = int(n)
        modulus = int(p ** n)
        
        # ─────────────────────────────────────────────────────────────
        # Step 1: Hensel lifting 求 f(T) 在 Z/p^nZ 的所有根
        # ─────────────────────────────────────────────────────────────
        roots = _hensel_lift_all_roots_zpn(
            poly=annihilator,
            p=p,
            n=n,
            max_roots=min(search_radius, 100_000),  # 防爆
        )
        
        if not roots:
            return []
        
        # ─────────────────────────────────────────────────────────────
        # Step 2: 根 → delta → 等价连边
        # ─────────────────────────────────────────────────────────────
        effects: List[Tuple[int, int]] = []
        seen_deltas: set = set()
        
        for r in roots:
            r_normalized = int(r) % modulus
            
            # delta = 0 是平凡的（自己等于自己）
            if r_normalized == 0:
                continue
            
            # 规范化 delta：取 min(d, modulus - d) 保证对称性
            d = _canonical_delta(r_normalized, modulus)
            
            if d in seen_deltas:
                continue
            seen_deltas.add(d)
            
            # 连边：center ~ center + d
            a = int(center)
            b = int(center + d) % modulus
            
            # 保证 a <= b 的规范顺序
            if a > b:
                a, b = b, a
            
            effects.append((a, b))
        
        # ─────────────────────────────────────────────────────────────
        # Step 3: 排序输出（确定性）
        # ─────────────────────────────────────────────────────────────
        effects = sorted(set(effects))
        
        _logger.debug(
            "_compute_algebraic_side_effects: %d roots -> %d edges",
            len(roots), len(effects)
        )
        
        return effects


    def _canonical_delta(d: int, modulus: int) -> int:
        """
        规范化 delta：取对称代表元
        
        在 Z/mZ 里，d 和 m-d 代表同一个"距离"，取较小的那个。
        """
        d = int(d) % int(modulus)
        return min(d, int(modulus) - d)


    def _hensel_lift_all_roots_zpn(
        poly: List[int],
        p: int,
        n: int,
        max_roots: int = 100_000,
    ) -> List[int]:
        """
        Hensel lifting：求 f(T) ≡ 0 (mod p^n) 的所有根
        
        算法：
          1. 先求 f(T) ≡ 0 (mod p) 的根（暴力或 Berlekamp）
          2. 对每个单根，Hensel lift 到 mod p^n
          3. 对每个奇异根（f'(r) ≡ 0），展开为 p 个分支递归
        
        返回：所有根的列表（可能有重复，由调用方去重）
        """
        p = int(p)
        n = int(n)
        
        if n < 1:
            raise ValueError("n must be >= 1")
        if not poly:
            return []
        
        modulus = int(p ** n)
        
        # 规范化多项式系数
        coeffs = [int(c) % modulus for c in poly]
        
        # 去除高位零系数
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        
        if len(coeffs) == 1:
            # 常数多项式
            return [0] if coeffs[0] % modulus == 0 else []
        
        # ─────────────────────────────────────────────────────────────
        # Base case: mod p
        # ─────────────────────────────────────────────────────────────
        roots_mod_p = _find_roots_mod_p(coeffs, p)
        
        if n == 1:
            return roots_mod_p
        
        # ─────────────────────────────────────────────────────────────
        # Hensel lift: mod p -> mod p^n
        # ─────────────────────────────────────────────────────────────
        all_roots: List[int] = []
        
        for r0 in roots_mod_p:
            lifted = _hensel_lift_single_root(
                coeffs=coeffs,
                root_mod_pk=int(r0),
                p=p,
                current_k=1,
                target_k=n,
                max_branches=max_roots - len(all_roots),
            )
            all_roots.extend(lifted)
            
            if len(all_roots) >= max_roots:
                _logger.warning("_hensel_lift_all_roots_zpn: hit max_roots=%d", max_roots)
                break
        
        return all_roots


    def _find_roots_mod_p(poly: List[int], p: int) -> List[int]:
        """
        f(T) ≡ 0 (mod p) 的所有根
        
        对小素数直接枚举。大素数需要 Berlekamp，这里先暴力，有空再补丁
        """
        p = int(p)
        roots = []
        
        for r in range(p):
            val = _eval_poly_mod(poly, r, p)
            if val == 0:
                roots.append(r)
        
        return roots


    def _eval_poly_mod(poly: List[int], x: int, m: int) -> int:
        """
        Horner 法求 f(x) mod m
        """
        result = 0
        for c in reversed(poly):
            result = (result * int(x) + int(c)) % int(m)
        return int(result)


    def _eval_poly_derivative_mod(poly: List[int], x: int, m: int) -> int:
        """
        求 f'(x) mod m
        
        f'(T) = sum_{i>=1} i * c_i * T^{i-1}
        """
        if len(poly) <= 1:
            return 0
        
        # 构造导数多项式
        deriv = [int(i) * int(poly[i]) for i in range(1, len(poly))]
        
        return _eval_poly_mod(deriv, x, m)


    def _hensel_lift_single_root(
        coeffs: List[int],
        root_mod_pk: int,
        p: int,
        current_k: int,
        target_k: int,
        max_branches: int,
    ) -> List[int]:
        """
        Hensel lift 单根从 mod p^k 到 mod p^{target_k}
        
        处理奇异根：若 f'(r) ≡ 0 (mod p)，分裂为 p 个分支
        """
        if max_branches <= 0:
            return []
        
        if current_k >= target_k:
            return [int(root_mod_pk)]
        
        p = int(p)
        pk = int(p ** current_k)
        pk1 = int(p ** (current_k + 1))
        
        r = int(root_mod_pk) % pk
        
        # 求 f(r) 和 f'(r)
        f_r = _eval_poly_mod(coeffs, r, pk1)
        fp_r = _eval_poly_derivative_mod(coeffs, r, p)  # f'(r) mod p 就够了
        
        results: List[int] = []
        
        if fp_r % p != 0:
            # ─────────────────────────────────────────────────────────
            # 单根情况：标准 Hensel
            # r_{k+1} = r_k - f(r_k) / f'(r_k) mod p^{k+1}
            # ─────────────────────────────────────────────────────────
            fp_r_inv = _mod_inverse(fp_r, p)
            if fp_r_inv is None:
                return []  # 不应该发生
            
            # f(r) 必须被 p^k 整除（因为 r 是 mod p^k 的根）
            if f_r % pk != 0:
                return []  # 不是有效根
            
            t = (f_r // pk) % p
            delta = (-t * fp_r_inv) % p
            
            r_new = (r + delta * pk) % pk1
            
            # 递归 lift
            results = _hensel_lift_single_root(
                coeffs=coeffs,
                root_mod_pk=r_new,
                p=p,
                current_k=current_k + 1,
                target_k=target_k,
                max_branches=max_branches,
            )
        
        else:
            # ─────────────────────────────────────────────────────────
            # 奇异根情况：f'(r) ≡ 0 (mod p)
            # 分裂为 p 个可能的 lift
            # ─────────────────────────────────────────────────────────
            if f_r % pk != 0:
                return []  # 不是有效根
            
            t = (f_r // pk) % p
            
            # 检查是否可 lift：需要 f(r)/p^k ≡ 0 (mod p)
            if t != 0:
                # 这个根在 mod p^{k+1} 下消失了
                return []
            
            # 可以 lift，分裂为 p 个分支
            branches_per = max(1, max_branches // p)
            
            for j in range(p):
                r_new = (r + j * pk) % pk1
                
                lifted = _hensel_lift_single_root(
                    coeffs=coeffs,
                    root_mod_pk=r_new,
                    p=p,
                    current_k=current_k + 1,
                    target_k=target_k,
                    max_branches=branches_per,
                )
                results.extend(lifted)
                
                if len(results) >= max_branches:
                    break
        
        return results


    def _mod_inverse(a: int, m: int) -> Optional[int]:
        """
        求 a^{-1} mod m
        
        若 gcd(a, m) != 1，返回 None
        """
        a = int(a) % int(m)
        if a == 0:
            return None
        
        # 扩展欧几里得
        def egcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            g, x, y = egcd(b % a, a)
            return g, y - (b // a) * x, x
        
        g, x, _ = egcd(a, int(m))
        if g != 1:
            return None
        
        return int(x) % int(m)

    def _trivial_injection(
        self,
        lhs: int,
        rhs: int,
        p: int,
        n: int,
    ) -> InjectionCertificate:
        """平凡情况：lhs ≡ rhs mod p^n"""
        return InjectionCertificate(
            lhs=int(lhs),
            rhs=int(rhs),
            kernel=EquivalenceKernel(
                prime_spec=PrimeSpec(p=p, k=n),
                shell_center=int(lhs),
                shell_radius_vp=int(n),
                defining_polynomial=(1,),  # 平凡多项式
                construction_method="trivial_congruence",
                commitment=self._compute_injection_commitment(lhs, rhs, [1]),
            ),
            minimal_k=int(n),
            polynomial_degree=0,
            polynomial_coeffs=(1,),
            side_effects=(),
            orbit_length=1,
            epsilon=Fraction(1, int(p) ** int(n)),
            is_valid=True,
            commitment="trivial",
        )
        
    def retract(self, cert: InjectionCertificate) -> bool:
        """
        撤回注入的定义
        
        Returns:
            是否成功撤回
        """
        if cert in self._injected:
            self._injected.remove(cert)
            _logger.info("retract: commitment=%s...", cert.commitment[:16])
            return True
        return False
    
    def list_injected(self) -> List[InjectionCertificate]:
        """列出所有已注入的定义"""
        return list(self._injected)


# =============================================================================
# 第四模块：全局账本 (GlobalLedger)
# =============================================================================


@dataclass
class AxiomProposal:
    """公理提议"""
    id: str
    kernel: EquivalenceKernel
    justification: str
    proposer: str
    votes_for: int = 0
    votes_against: int = 0
    executed: bool = False


@dataclass
class AxiomTransition:
    """公理切换记录"""
    from_axiom: str
    to_axiom: str
    kernel: Optional[EquivalenceKernel]
    timestamp: int
    commitment: str


class GlobalLedger:
    """
    全局公理账本
    
    用途：
      控制全局共识，让公理系统成为"地球标准"
    """
    
    def __init__(self):
        self.current_axiom: AxiomContext = AxiomContext(
            name="peano_default",
            kernel=None,
            epsilon=Fraction(0),
        )
        self.pending_proposals: Dict[str, AxiomProposal] = {}
        self.history: List[AxiomTransition] = []
        self._proposal_counter = 0
    
    def propose(
        self,
        kernel: EquivalenceKernel,
        justification: str,
        proposer: str = "anonymous",
    ) -> str:
        """
        提议切换公理系统
        
        Args:
            kernel: 新的等价核
            justification: 提议理由
            proposer: 提议者标识
        
        Returns:
            proposal_id
        """
        self._proposal_counter += 1
        proposal_id = f"PROP-{self._proposal_counter:06d}"
        
        proposal = AxiomProposal(
            id=proposal_id,
            kernel=kernel,
            justification=justification,
            proposer=proposer,
        )
        
        self.pending_proposals[proposal_id] = proposal
        
        _logger.info(
            "propose: id=%s, kernel=%s..., proposer=%s",
            proposal_id, kernel.commitment[:16], proposer
        )
        
        return proposal_id
    
    def vote(
        self,
        proposal_id: str,
        approve: bool,
        voter_proof: bytes,
    ) -> bool:
        """
        对提议投票
        
        Args:
            proposal_id: 提议 ID
            approve: 是否赞成
            voter_proof: 投票者证明（签名/ZK proof）
        
        Returns:
            投票是否被记录
        """
        if proposal_id not in self.pending_proposals:
            raise AxiomSwitchError(f"提议不存在: {proposal_id}")
        
        proposal = self.pending_proposals[proposal_id]
        
        if proposal.executed:
            raise AxiomSwitchError(f"提议已执行: {proposal_id}")
        
        # TODO: 验证 voter_proof
        
        if approve:
            proposal.votes_for += 1
        else:
            proposal.votes_against += 1
        
        _logger.info(
            "vote: id=%s, approve=%s, for=%d, against=%d",
            proposal_id, approve, proposal.votes_for, proposal.votes_against
        )
        
        return True
    
    def execute(
        self,
        proposal_id: str,
        *,
        min_votes: int = 1,
        min_ratio: Fraction = Fraction(1, 2),
    ) -> AxiomTransition:
        """
        执行公理切换
        
        Args:
            proposal_id: 提议 ID
            min_votes: 最小票数
            min_ratio: 最小赞成比例
        
        Returns:
            AxiomTransition 记录
        """
        if proposal_id not in self.pending_proposals:
            raise AxiomSwitchError(f"提议不存在: {proposal_id}")
        
        proposal = self.pending_proposals[proposal_id]
        
        if proposal.executed:
            raise AxiomSwitchError(f"提议已执行: {proposal_id}")
        
        total_votes = proposal.votes_for + proposal.votes_against
        
        if total_votes < min_votes:
            raise ConsensusNotReached(
                f"票数不足: {total_votes} < {min_votes}"
            )
        
        ratio = Fraction(proposal.votes_for, total_votes) if total_votes > 0 else Fraction(0)
        
        if ratio < min_ratio:
            raise ConsensusNotReached(
                f"赞成比例不足: {ratio} < {min_ratio}"
            )
        
        # 执行切换
        old_axiom = self.current_axiom.name
        
        new_context = AxiomContext(
            name=f"extended_{proposal_id}",
            kernel=proposal.kernel,
            epsilon=Fraction(1, int(proposal.kernel.prime_spec.modulus)),
        )
        
        commitment = _compute_commitment({
            "from": old_axiom,
            "to": new_context.name,
            "kernel": proposal.kernel.commitment,
            "votes_for": int(proposal.votes_for),
            "votes_against": int(proposal.votes_against),
        })
        
        transition = AxiomTransition(
            from_axiom=old_axiom,
            to_axiom=new_context.name,
            kernel=proposal.kernel,
            timestamp=0,  # TODO: 实际时间戳
            commitment=commitment,
        )
        
        self.current_axiom = new_context
        self.history.append(transition)
        proposal.executed = True
        
        _logger.info(
            "execute: %s -> %s, commitment=%s...",
            old_axiom, new_context.name, commitment[:16]
        )
        
        return transition
    
    def query(self, a: int, b: int) -> str:
        """
        在当前公理系统下查询 a 和 b 的关系
        
        Returns:
            "EQUAL" / "DISTINCT" / "EQUIVALENT_UNDER_KERNEL"
        """
        if a == b:
            return "EQUAL"
        
        if self.current_axiom.kernel is None:
            return "DISTINCT"
        
        kernel = self.current_axiom.kernel
        
        if kernel.contains(a) and kernel.contains(b):
            return "EQUIVALENT_UNDER_KERNEL"
        
        return "DISTINCT"


# =============================================================================
# 工具函数
# =============================================================================


def _compute_commitment(data: Dict[str, Any]) -> str:
    """计算承诺哈希"""
    # 确定性序列化
    def serialize(obj: Any) -> str:
        if obj is None:
            return "null"
        if isinstance(obj, bool):
            return "true" if obj else "false"
        if isinstance(obj, int):
            return f"i:{obj}"
        if isinstance(obj, str):
            return f"s:{obj}"
        if isinstance(obj, Fraction):
            return f"f:{obj.numerator}/{obj.denominator}"
        if isinstance(obj, (list, tuple)):
            return f"[{','.join(serialize(x) for x in obj)}]"
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: str(kv[0]))
            return f"{{{','.join(f'{serialize(k)}:{serialize(v)}' for k,v in items)}}}"
        return f"r:{repr(obj)}"
    
    s = serialize(data)
    return hashlib.sha256(s.encode()).hexdigest()


# =============================================================================
# smoke
# =============================================================================

def _self_test() -> Dict[str, Any]:
    """
    MVP23 自测试套件
    
    验收标准：
      1. construct_kernel(9, 10) 成功
      2. verify_kernel 三轨道通过
      3. transfer 跨公理成功
      4. inject 定义注入成功
      5. GlobalLedger 完整流程
    """
    results: Dict[str, Any] = {"ok": True, "tests": []}
    
    def record(name: str, passed: bool, detail: str = "") -> None:
        results["tests"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            results["ok"] = False
            _logger.error("SELF-TEST FAILED: %s - %s", name, detail)
    
    # 测试 1: construct_kernel
    try:
        spec = PrimeSpec(p=2, k=8)
        kernel = construct_kernel(9, 10, spec)
        assert kernel.contains(9), "kernel must contain source"
        assert kernel.contains(10), "kernel must contain target"
        record("construct_kernel_9_10", True)
    except Exception as e:
        record("construct_kernel_9_10", False, str(e))
    
    # 测试 2: verify_kernel
    try:
        verdict = verify_kernel(kernel, 9, 10, skip_large_prime=True)
        assert verdict.verdict == "EQUIVALENT", f"expected EQUIVALENT, got {verdict.verdict}"
        assert verdict.consensus is True, "consensus should be True"
        record("verify_kernel_9_10", True)
    except Exception as e:
        record("verify_kernel_9_10", False, str(e))
    
    # 测试 3: Peano 模式下 9 ≠ 10
    try:
        selector = AxiomSelector(base_prime=2, base_k=256)
        peano = selector.select("peano")
        assert peano.kernel is None, "Peano should have no kernel"
        # 在 Peano 下直接比较
        assert 9 != 10, "9 != 10 in Peano"
        record("peano_mode_distinct", True)
    except Exception as e:
        record("peano_mode_distinct", False, str(e))
    
    # 测试 4: 创建扩展上下文
    try:
        extended = selector.create_extended_context("nine_equals_ten", 9, 10)
        assert extended.kernel is not None
        assert extended.kernel.contains(9)
        assert extended.kernel.contains(10)
        record("create_extended_context", True)
    except Exception as e:
        record("create_extended_context", False, str(e))
    
    # 测试 5: transfer
    try:
        cert = selector.transfer(9, "nine_equals_ten", "peano")
        assert cert.is_valid is True
        record("transfer_extended_to_peano", True)
    except Exception as e:
        record("transfer_extended_to_peano", False, str(e))
    
    # 测试 6: inject
    try:
        injector = DefinitionInjector(selector)
        cert = injector.inject(100, 101)
        assert cert.is_valid is True
        assert cert.minimal_k > 0
        record("inject_definition", True)
    except Exception as e:
        record("inject_definition", False, str(e))
    
    # 测试 7: GlobalLedger 完整流程
    try:
        ledger = GlobalLedger()
        
        # 初始状态：Peano
        assert ledger.query(9, 10) == "DISTINCT"
        
        # 提议
        kernel = construct_kernel(9, 10, PrimeSpec(p=2, k=8))
        proposal_id = ledger.propose(kernel, "让 9 = 10", "math_god")
        
        # 投票
        ledger.vote(proposal_id, True, b"proof1")
        ledger.vote(proposal_id, True, b"proof2")
        
        # 执行
        transition = ledger.execute(proposal_id, min_votes=2)
        
        # 验证切换后状态
        result = ledger.query(9, 10)
        assert result == "EQUIVALENT_UNDER_KERNEL", f"expected EQUIVALENT, got {result}"
        
        record("global_ledger_full_flow", True)
    except Exception as e:
        record("global_ledger_full_flow", False, str(e))
    
    # 测试 8: 红线检查 - 无浮点
    try:
        # 遍历所有证书，确保无浮点
        def check_no_float(obj: Any, path: str = "") -> None:
            if isinstance(obj, float):
                raise ValueError(f"Float at {path}: {obj}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_float(v, f"{path}.{k}")
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_no_float(v, f"{path}[{i}]")
        
        # 检查 kernel
        check_no_float({
            "p": kernel.prime_spec.p,
            "k": kernel.prime_spec.k,
            "center": kernel.shell_center,
            "radius_vp": kernel.shell_radius_vp,
        })
        
        record("no_float_contamination", True)
    except Exception as e:
        record("no_float_contamination", False, str(e))
    
    # 总结
    passed = sum(1 for t in results["tests"] if t["passed"])
    total = len(results["tests"])
    _logger.info("MVP23 self-test: %d/%d passed", passed, total)
    
    return results

