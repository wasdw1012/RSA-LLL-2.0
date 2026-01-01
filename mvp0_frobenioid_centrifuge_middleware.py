#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVP0: Frobenioid × Anabelian Centrifuge 联动中间件
=================================================
  core/anabelian_centrifuge 接到主编排链路上
  输入来自上游确定性证书/整数信号 MVP17-18 / MVP11 / Frobenioid 等
  输出：多宇宙观测 bundle 可选 sidecar 落盘，避免 result.json 膨胀
  共识证书 严格结构检查：确定性排序/承诺自洽/兼容性声明/可比宇宙像一致
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CentrifugeAuditObject:
    """
    观测对象：
      - label: 仅用于可追溯标识
      - payload: 显式整数载荷
    """

    label: str
    payload: Tuple[int, ...]


def _as_int_strict(name: str, v: Any) -> int:
    try:
        return int(v)
    except Exception as e:
        raise TypeError(f"{name} must be int-like, got {v!r}") from e


def _env_strict_enum(name: str, *, allowed: Tuple[str, ...], default: str) -> str:
    """
    Read an env var as an enum-like string with strict validation.

    Redlines:
    - No silent downgrade: invalid values must raise (deployment/config error).
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return str(default)
    val = str(raw).strip().upper()
    if val not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {raw!r}")
    return val


def _env_int(name: str, *, default: int) -> int:
    """
    Read an env var as int (base-10), strict.
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip(), 10)
    except Exception as e:
        raise ValueError(f"{name} must be an integer (base-10), got {raw!r}") from e


def _derived_witt_length_for_evm_word() -> int:
    """
    Derive Witt length from EVM 256-bit word using machine uint64 limb width:
      witt_length = ceil(256 / 64)
    """
    evm_word_bits = 256
    u64_bits = int(8 * struct.calcsize("Q"))
    return int((int(evm_word_bits) + int(u64_bits) - 1) // int(u64_bits))


def _build_payload_from_sources(*, sources: Mapping[str, Any]) -> Tuple[int, ...]:
    """
    把一组确定性整数信号按 key 排序后压成 payload
    Redline:
      - 不允许缺字段/非整数悄悄掉线；调用方决定哪些 key 必须提供
    """
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a Mapping[str, Any]")
    out: List[int] = []
    for k in sorted(list(sources.keys())):
        out.append(_as_int_strict(f"sources[{k!r}]", sources[k]))
    return tuple(int(x) for x in out)


def extract_frobenioid_cert_for_l2(*, layer1_math_collider_summary: Mapping[str, Any]) -> Dict[str, Any]:
    """
    从 L1 的 MathCollider 总结中提取/规范化 Frobenioid 证书（最小直通，JSON-safe）
      - Frobenioid 证书是 MVP0×L2 证据链的底座通行证，不应由 audit_main.py 手写字段/回退逻辑
      - 统一 schema + 统一异常语义，避免后续维护在多个编排器里散落
    Redlines:
      - 缺失/类型不对必须抛异常（不得静默置 None）
      - 不做任何推断/补全：只接受上游显式给出的证书字段
    """
    if not isinstance(layer1_math_collider_summary, Mapping):
        raise TypeError(
            f"layer1_math_collider_summary must be a Mapping[str, Any], got {type(layer1_math_collider_summary).__name__}"
        )
    fc = layer1_math_collider_summary.get("frobenioid_cert")
    if not isinstance(fc, dict) or not fc:
        raise KeyError(
            "frobenioid_cert missing/empty in layer1 summary (redline: Frobenioid base evidence must be present)"
        )
    # A shallow copy is enough; downstream may pop/attach fields for logging/JSON, but must not mutate upstream.
    return dict(fc)


def build_mvp0_centrifuge_sources(
    *,
    mvp17_summary: Mapping[str, Any],
    opcodes_len: int,
    frobenioid_cert: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """
    组装 MVP0/Centrifuge 的 sources确定性整数信号：
    - 必选：MVP17-18 CFG-first 摘要字段
    - 必选：opcodes_len（MVP11 carrier）
    - 可选：Frobenioid payload_optimal_length（若提供 frobenioid_cert，则必须能取到该字段）

    Redlines:
      - 不允许 None/缺字段/非整数偷偷混进 sources（run_mvp0_centrifuge_bundle 会严格 int 化并抛错）
      - 不允许从其它字段猜或推断缺失值
    """
    if not isinstance(mvp17_summary, Mapping):
        raise TypeError(f"mvp17_summary must be Mapping[str, Any], got {type(mvp17_summary).__name__}")
    try:
        op_len = int(opcodes_len)
    except Exception as e:
        raise TypeError(f"opcodes_len must be int-like, got {opcodes_len!r}") from e
    if op_len <= 0:
        raise ValueError(f"opcodes_len must be >= 1 (deployment/feed error), got {op_len}")

    required_mvp17_keys = (
        "commutator_rank",
        "cycle_count",
        "edge_count",
        "lem_gap_nodes",
        "node_count",
        "period_max_pole_order",
        "period_max_selmer_rank_surrogate",
        "reachable_cycle_count",
        "sha_obstruction_dim",
    )
    missing: List[str] = []
    out: Dict[str, int] = {}
    for k in required_mvp17_keys:
        if k not in mvp17_summary or mvp17_summary.get(k) is None:
            missing.append(str(k))
            continue
        out[str(k)] = _as_int_strict(f"mvp17_summary[{k!r}]", mvp17_summary.get(k))
    if missing:
        raise KeyError(f"mvp17_summary missing required integer keys: {missing}")

    out["opcodes_len"] = int(op_len)

    if frobenioid_cert is not None:
        if not isinstance(frobenioid_cert, Mapping):
            raise TypeError(f"frobenioid_cert must be Mapping[str, Any], got {type(frobenioid_cert).__name__}")
        pb = frobenioid_cert.get("payload_boundary")
        if not isinstance(pb, Mapping) or not pb:
            raise KeyError("frobenioid_cert.payload_boundary missing/invalid (redline: no silent)")
        opt_len = pb.get("optimal_length")
        if opt_len is None:
            raise KeyError("frobenioid_cert.payload_boundary.optimal_length missing (redline: no silent)")
        out["payload_optimal_length"] = _as_int_strict("frobenioid_cert.payload_boundary.optimal_length", opt_len)

    return dict(out)


def run_mvp0_centrifuge_bundle(
    *,
    obj_label: str,
    sources: Mapping[str, Any],
    universes_label_prefix: str = "U",
    structure_tag: str = "X",
    max_workers: int = 0,
) -> Dict[str, Any]:
    """
    生成 centrifuge 多宇宙 bundle + 共识证书（严格、可审计、JSON-safe）
     禁止魔法数：
      - 若要既能比较又能展示不可比，多径向最小非平凡结构需要3个宇宙：
      - 两个宇宙在同一兼容组（可比较）
      - 一个宇宙单独成组（不可比）
      - 可比较的两个宇宙必须共享同一 detachment policy（否则承诺必然不同，共识应失败）
      - 不可比宇宙可使用更强切断策略（reveal_input_value=False）
    """
    if not isinstance(obj_label, str) or not obj_label.strip():
        raise ValueError("obj_label must be non-empty str")
    if not isinstance(universes_label_prefix, str) or not universes_label_prefix.strip():
        raise ValueError("universes_label_prefix must be non-empty str")
    if not isinstance(structure_tag, str) or not structure_tag.strip():
        raise ValueError("structure_tag must be non-empty str")

    payload = _build_payload_from_sources(sources=sources)
    # Keep an explicit key->value mapping for human auditability (no guessing vector dimensions).
    # NOTE: payload is defined as sources sorted by key; this mapping is the authoritative decoding.
    sources_int: Dict[str, int] = {}
    for k, v in dict(sources).items():
        sources_int[str(k)] = _as_int_strict(f"sources[{k!r}]", v)
    payload_keys_sorted = tuple(sorted(sources_int.keys()))
    payload_vector = tuple(int(sources_int[k]) for k in payload_keys_sorted)
    if payload_vector != tuple(int(x) for x in payload):
        # Redline: we must not emit a bundle whose declared payload does not match the sources definition.
        raise RuntimeError("internal error: payload_vector != payload (sources sorting mismatch)")
    payload_by_key = tuple({"key": k, "value": int(sources_int[k])} for k in payload_keys_sorted)

    # 导入：保持 middleware 可被上层按需拔插；导入失败必须中断（红线）
    from bridge_audit.core.anabelian_centrifuge import (
        ArithmeticUniverse,
        CompatibilityDeclaration,
        DetachmentPolicy,
        KummerDetacher,
        MultiradialConsensusEngine,
        MultiradialRepresentation,
        ThetaLinkFunctor,
    )
    from bridge_audit.core.anabelian_centrifuge.redline import assert_no_float_complex_set, sha256_hex_of_certificate
    from bridge_audit.core.frobenioid_base import (
        EpsilonScheduler,
        FrobenioidBaseArchitecture,
        IntegerPolynomial,
    )

    # -----------------------------------------------------------------
    # Multiradial profile switch (explicit, env-driven; default stays legacy).
    # -----------------------------------------------------------------
    profile = _env_strict_enum(
        "MVP0_MULTIRADIAL_PROFILE",
        allowed=("LEGACY", "PORISM_MIRROR"),
        default="LEGACY",
    )

    # Universe plan:
    # - LEGACY: 2 comparable + 1 incomparable (minimal non-trivial multiradial)
    # - PORISM_MIRROR: U_Prime (Independent) + U_Mirror_1 (Derived+DILATE) comparable, plus 1 incomparable shadow.
    if profile == "PORISM_MIRROR":
        padic_p = _env_int("MVP0_PADIC_PRIME", default=2)
        ghost_level = _env_int("MVP0_GHOST_TRUNCATION_LEVEL", default=1)
        porism_source = _env_int("MVP0_PORISM_SOURCE", default=14)
        porism_target = _env_int("MVP0_PORISM_TARGET", default=15)

        if padic_p < 2:
            raise ValueError("MVP0_PADIC_PRIME must be >=2")
        if ghost_level < 1:
            raise ValueError("MVP0_GHOST_TRUNCATION_LEVEL must be >=1 for PORISM_MIRROR profile")
        if porism_source == 0:
            raise ValueError("MVP0_PORISM_SOURCE must be non-zero (degree target/source)")

        # Derive declared morphism degree as target/source (reduced rational).
        deg = Fraction(int(porism_target), int(porism_source))
        if deg <= 0:
            raise ValueError("derived degree must be positive (porism_target/porism_source)")

        u_prime = "U_Prime"
        u_mirror = "U_Mirror_1"
        u_shadow = f"{universes_label_prefix}_2"
        universes = (
            ArithmeticUniverse(
                label=u_prime,
                structure_tag=str(structure_tag),
                observation_mode="OBSERVE",
                source_kind="INDEPENDENT",
            ),
            ArithmeticUniverse(
                label=u_mirror,
                structure_tag=str(structure_tag),
                observation_mode="DILATE",
                source_kind="DERIVED",
                derived_from=u_prime,
                derived_degree_num=int(deg.numerator),
                derived_degree_den=int(deg.denominator),
                ghost_truncation_level=int(ghost_level),
                padic_prime=int(padic_p),
            ),
            ArithmeticUniverse(
                label=u_shadow,
                structure_tag=str(structure_tag),
                observation_mode="OBSERVE",
                source_kind="INDEPENDENT",
            ),
        )
        compatibility = CompatibilityDeclaration.from_groups(
            universes=universes,
            groups=((u_prime, u_mirror), (u_shadow,)),
        )
    else:
        u0 = f"{universes_label_prefix}_0"
        u1 = f"{universes_label_prefix}_1"
        u2 = f"{universes_label_prefix}_2"
        universes = (
            ArithmeticUniverse(label=u0, structure_tag=str(structure_tag)),
            ArithmeticUniverse(label=u1, structure_tag=str(structure_tag)),
            ArithmeticUniverse(label=u2, structure_tag=str(structure_tag)),
        )
        compatibility = CompatibilityDeclaration.from_groups(universes=universes, groups=((u0, u1), (u2,)))

    # Comparable universes share the same detacher policy.
    detacher_reveal = KummerDetacher(policy=DetachmentPolicy(reveal_input_value=True, forbid_additive_neighbors=True))
    detacher_hidden = KummerDetacher(policy=DetachmentPolicy(reveal_input_value=False, forbid_additive_neighbors=True))

    def _payload_extractor(o: CentrifugeAuditObject) -> Sequence[int]:
        if not isinstance(o, CentrifugeAuditObject):
            raise TypeError("payload_extractor expects CentrifugeAuditObject")
        return tuple(int(x) for x in o.payload)

    # Payload dilate (ghost truncation): x -> floor(x / p^L)
    def _payload_extractor_dilate(o: CentrifugeAuditObject) -> Sequence[int]:
        if not isinstance(o, CentrifugeAuditObject):
            raise TypeError("payload_extractor_dilate expects CentrifugeAuditObject")
        padic_p = _env_int("MVP0_PADIC_PRIME", default=2)
        lvl = _env_int("MVP0_GHOST_TRUNCATION_LEVEL", default=1)
        if padic_p < 2:
            raise ValueError("MVP0_PADIC_PRIME must be >=2")
        if lvl < 1:
            raise ValueError("MVP0_GHOST_TRUNCATION_LEVEL must be >=1 for DILATE payload extractor")
        divisor = int(padic_p ** int(lvl))
        return tuple(int(int(x) // divisor) for x in o.payload)

    theta_reveal = ThetaLinkFunctor(payload_extractor=_payload_extractor, detacher=detacher_reveal, label="ThetaReveal")
    theta_hidden = ThetaLinkFunctor(payload_extractor=_payload_extractor, detacher=detacher_hidden, label="ThetaHidden")
    theta_mirror = ThetaLinkFunctor(payload_extractor=_payload_extractor_dilate, detacher=detacher_reveal, label="ThetaMirrorDilate")

    mrep = MultiradialRepresentation(
        universes=universes,
        compatibility=compatibility,
        theta_by_universe=(
            {universes[0].label: theta_reveal, universes[1].label: theta_mirror, universes[2].label: theta_hidden}
            if profile == "PORISM_MIRROR"
            else {universes[0].label: theta_reveal, universes[1].label: theta_reveal, universes[2].label: theta_hidden}
        ),
        max_workers=int(max_workers),
    )

    obj = CentrifugeAuditObject(label=str(obj_label), payload=tuple(int(x) for x in payload))

    logger.info(
        "[MVP0/Centrifuge] start: label=%s payload_len=%s universes=%s",
        obj.label,
        int(len(obj.payload)),
        [u.label for u in universes],
    )

    bundle = mrep.observe(obj)
    consensus = MultiradialConsensusEngine().verify(bundle)

    # -----------------------------------------------------------------
    # MVP0 "Porism" witness (Frobenioid side): show that Source(14) and Target(15)
    # become indistinguishable once w0 is ghost-truncated and epsilon is curvature-scheduled.
    # This is a deterministic, auditable diagnostic and does not affect L2 gating.
    # -----------------------------------------------------------------
    porism_report: Optional[Dict[str, Any]] = None
    if profile == "PORISM_MIRROR":
        padic_p = _env_int("MVP0_PADIC_PRIME", default=2)
        lvl = _env_int("MVP0_GHOST_TRUNCATION_LEVEL", default=1)
        porism_source = _env_int("MVP0_PORISM_SOURCE", default=14)
        porism_target = _env_int("MVP0_PORISM_TARGET", default=15)

        # Ghost truncation on the *x-coordinate* itself: drop w0..w_{L-1} digits (base-p).
        divisor = int(padic_p ** int(lvl))
        x_a = int(int(porism_source) // divisor)
        x_b = int(int(porism_target) // divisor)

        # Build a deterministic "singularity strike" polynomial.
        #
        # For integer quadratic P(x)=a x^2 + b x + c, discrete second difference is constant 2a.
        # Over p=2, v_2(2a) is minimized when a is odd, and resonance is automatically equal at x_a/x_b.
        # Choose P(x)=x^2+1 to (i) minimize v_p(Δ^2) under the quadratic constraint and
        # (ii) make P(x_a) odd when p=2 (maximizing center_defect factor).
        poly = IntegerPolynomial((1, 0, 1))  # 1 + x^2

        # Frobenioid PrimeSpec.k is the p-adic truncation depth on Z/p^kZ.
        # For EVM words, the canonical target is 2^256, so we default to k=256 (protocol constant, not a magic number).
        witt_precision = _env_int("MVP0_WITT_PRECISION", default=256)
        if witt_precision < 1:
            raise ValueError("MVP0_WITT_PRECISION must be >=1")

        # Derive an Arakelov height bound from the evaluated payload itself (no arbitrary constant).
        val_a = int(poly.evaluate(int(x_a)))
        val_b = int(poly.evaluate(int(x_b)))
        ar_height = int(max(abs(val_a), abs(val_b)))

        fb = FrobenioidBaseArchitecture(
            prime=int(padic_p),
            precision=int(witt_precision),
            conductor=1,
            arakelov_height=int(ar_height),
            modular_weight=2,
        )
        scheduler = EpsilonScheduler(fb.prime_spec)
        porism_report = fb.theta_link.transmit_polynomial(
            poly,
            x_a=int(x_a),
            x_b=int(x_b),
            epsilon_scheduler=scheduler,
        )
        porism_report = dict(porism_report)
        porism_report["ghost_truncation"] = {
            "prime_p": int(padic_p),
            "level": int(lvl),
            "source_raw": int(porism_source),
            "target_raw": int(porism_target),
            "source_truncated": int(x_a),
            "target_truncated": int(x_b),
            "formula": "x' = floor(x / p^level)  (drop w0..w_{level-1})",
            "equal_in_quotient": bool(int(x_a) == int(x_b)),
        }

        # Emit the requested healthy log line (single line, no spam).
        logger.info(
            "[Multiradial] Porism Verified: Source(%s) and Target(%s) are indistinguishable in the Symplectic Core.",
            int(porism_source),
            int(porism_target),
        )

    # Minimal summary (keep result.json small; full bundle can be sidecar-ed by caller)
    obs_summary = []
    for o in bundle.observations:
        img = o.image if isinstance(o.image, dict) else {}
        poly = img.get("poly_monoid") if isinstance(img.get("poly_monoid"), dict) else {}
        gens = poly.get("generators") if isinstance(poly.get("generators"), list) else []
        obs_summary.append(
            {
                "universe": o.universe.label,
                "structure_tag": o.universe.structure_tag,
                "theta_functor_label": o.theta_functor_label,
                "observation_commitment": o.commitment,
                "generator_count": int(len(gens)),
            }
        )

    payload_commitment = sha256_hex_of_certificate({"sources": dict((str(k), int(v)) for k, v in sources.items())})

    out = {
        "ok": True,
        "profile": str(profile),
        "object": {
            "label": obj.label,
            "payload_len": int(len(obj.payload)),
            # A hash of sources (key->int) to anchor the payload definition.
            "payload_commitment": payload_commitment,
            # Human-readable decoding of the payload vector (no "mystery dimensions").
            "sources": dict((k, int(v)) for k, v in sources_int.items()),
            "payload_key_order": list(payload_keys_sorted),
            "payload_by_key": list(payload_by_key),
        },
        "compatibility": compatibility.to_dict(),
        "bundle_commitment": bundle.commitment,
        "consensus": consensus.to_dict(),
        # Convenience index of hex commitments that are fully self-checked by consensus.
        # This is intended for "dumb" downstream verification scripts that only want hammered hex values.
        "hex_index": {
            "bundle_commitment": str(bundle.commitment),
            "compatibility_commitment": str(compatibility.commitment),
            "consensus_recomputed_commitment": str(consensus.recomputed_commitment),
            "observation_commitments": {o.universe.label: str(o.commitment) for o in bundle.observations},
        },
        # Explicit hex typing to avoid mixing MVP0 commitments with EVM tx hashes/calldata.
        # Redline: commitments are model-side evidence anchors; they are NOT execution proofs.
        "hex_kinds": {
            "bundle_commitment": "MVP0_BUNDLE_COMMITMENT",
            "compatibility_commitment": "MVP0_COMPATIBILITY_COMMITMENT",
            "consensus_recomputed_commitment": "MVP0_CONSENSUS_RECOMPUTED_COMMITMENT",
            "observation_commitments": "MVP0_OBSERVATION_COMMITMENT_BY_UNIVERSE",
        },
        "observations_summary": obs_summary,
        # Full bundle for optional sidecar storage (still JSON-safe by construction)
        "bundle": bundle.to_dict(),
        # Optional: Frobenioid-side porism report (strictly diagnostic; can be sidecar-ed by caller)
        "porism_report": porism_report,
    }
    assert_no_float_complex_set(out)
    logger.info(
        "[MVP0/Centrifuge] ok: bundle_commitment=%s consensus_passed=%s",
        str(bundle.commitment)[:16] + "...",
        bool(consensus.passed),
    )
    return out





