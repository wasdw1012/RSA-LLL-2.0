#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monoid Presentation (幺半群呈示) —— 为 Kummer Tower 加根关系准备的硬核层

在强化框架里，我们把 PolyMonoid 看作自由交换幺半群：
  - 它非常干净，但无法表达诸如 r^n = g 的 Kummer 关系。

因此引入呈示对象：
  - 生成元（generators）
  - 关系（relations）: 形式等式  LHS == RHS ，两边都是 Monomial

红线：
  - 本层不实现 word problem / 正规形求解（那会引入巨大工程债务）；
  - 只提供关系记录 + 证书承诺
      - 完整实现 Knuth-Bendix / Gröbner-like 正规化（非常难，但很强）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .errors import InputError
from .polymonoid import Monomial, PolyMonoid
from .redline import assert_no_float_complex_set, sha256_hex_of_certificate
from .types import Symbol


@dataclass(frozen=True)
class MonoidRelation:
    """
    A formal relation LHS == RHS in a monoid presentation.
    """

    lhs: Dict[str, Any]
    rhs: Dict[str, Any]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise InputError("MonoidRelation.reason must be non-empty str")
        assert_no_float_complex_set(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {"lhs": dict(self.lhs), "rhs": dict(self.rhs), "reason": self.reason}

    @property
    def relation_id(self) -> str:
        return sha256_hex_of_certificate(self.to_dict())


@dataclass(frozen=True)
class PresentedMonoid:
    """
    A monoid presented by generators and relations.

    Internally, we keep an underlying PolyMonoid as the free monoid on generators.
    Relations are stored as immutable records.
    """

    label: str
    free: PolyMonoid
    relations: Tuple[MonoidRelation, ...]
    commitment: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise InputError("PresentedMonoid.label must be non-empty str")
        if not isinstance(self.free, PolyMonoid):
            raise InputError("PresentedMonoid.free must be a PolyMonoid")
        if not isinstance(self.relations, tuple):
            raise InputError("PresentedMonoid.relations must be a tuple")
        if not isinstance(self.commitment, str) or not self.commitment:
            raise InputError("PresentedMonoid.commitment must be non-empty str")
        assert_no_float_complex_set(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "free": {"label": self.free.label, "generators": [g.key for g in self.free.generators]},
            "relations": [r.to_dict() for r in self.relations],
            "commitment": self.commitment,
        }

    @classmethod
    def from_generators_and_relations(
        cls,
        *,
        label: str,
        generators: Tuple[Symbol, ...],
        relations: Tuple[MonoidRelation, ...],
    ) -> "PresentedMonoid":
        free = PolyMonoid(label=f"Free({label})", generators=tuple(sorted(generators, key=lambda s: s.key)))
        body = {
            "label": str(label),
            "generators": [g.key for g in free.generators],
            "relations": [r.to_dict() for r in relations],
        }
        commitment = sha256_hex_of_certificate(body)
        return cls(label=str(label), free=free, relations=tuple(relations), commitment=commitment)


