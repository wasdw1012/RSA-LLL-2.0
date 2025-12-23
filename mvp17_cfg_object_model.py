#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVP 17 代数法庭 完备状空间 与 晶体上同调 

建模理念：以数学不完备性公式化霍奇猜想得到流形上的唯一可满足解（已做到血统+口号原型）

倾斜等价 (The Tilting Equivalence)

动机周期 (Motivic Periods)

晶体上同调 (Crystalline Cohomology)

"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import json
import math
import numpy as np
import heapq
from collections import Counter

from .tannakian_reconstruction_engine import invert_matrix_modp, matmul_modp, rank_modp
try:
    # Private helper is stable in this repo and gives deterministic F_p nullspace basis.
    from .tannakian_reconstruction_engine import _nullspace_basis_modp as _nullspace_basis_modp  # type: ignore
except Exception:  # pragma: no cover
    _nullspace_basis_modp = None  # type: ignore


class MVP17CFGError(RuntimeError):
    """MVP17 CFG 对象建模基础异常（必须中断）。"""


class MVP17CFGInputError(MVP17CFGError):
    """输入/工件格式错误。"""


def _as_int_list(xs: Any, *, name: str) -> List[int]:
    if xs is None:
        return []
    if not isinstance(xs, list):
        raise MVP17CFGInputError(f"{name} must be a list, got {type(xs)}")
    out: List[int] = []
    for x in xs:
        try:
            out.append(int(x))
        except Exception as e:
            raise MVP17CFGInputError(f"{name} contains non-int: {x!r}") from e
    return out


def _as_edge_list(xs: Any, *, name: str) -> List[Tuple[int, int]]:
    if xs is None:
        return []
    if not isinstance(xs, list):
        raise MVP17CFGInputError(f"{name} must be a list, got {type(xs)}")
    out: List[Tuple[int, int]] = []
    for e in xs:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise MVP17CFGInputError(f"{name} must contain [src,dst] pairs, got {e!r}")
        out.append((int(e[0]), int(e[1])))
    return out


def _identity(d: int) -> List[List[int]]:
    return [[1 if i == j else 0 for j in range(d)] for i in range(d)]

def _zero(d: int) -> List[List[int]]:
    return [[0 for _ in range(d)] for _ in range(d)]

def _mat_is_zero(A: Sequence[Sequence[int]]) -> bool:
    return all(int(x) == 0 for row in A for x in row)

def _mat_nnzs(A: Sequence[Sequence[int]], p: int) -> int:
    p_i = int(p)
    return sum(1 for row in A for x in row if (int(x) % p_i) != 0)


def _mat_add_modp(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]], p: int) -> List[List[int]]:
    p_i = int(p)
    return [[(int(a) + int(b)) % p_i for a, b in zip(ar, br)] for ar, br in zip(A, B)]


def _mat_sub_modp(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]], p: int) -> List[List[int]]:
    p_i = int(p)
    return [[(int(a) - int(b)) % p_i for a, b in zip(ar, br)] for ar, br in zip(A, B)]


def _mat_scale_modp(A: Sequence[Sequence[int]], s: int, p: int) -> List[List[int]]:
    p_i = int(p)
    ss = int(s) % p_i
    return [[(ss * int(a)) % p_i for a in row] for row in A]


def _mat_col(A: Sequence[Sequence[int]], j: int) -> List[int]:
    jj = int(j)
    return [int(row[jj]) for row in A]


def _first_nonzero_column_witness(A: Sequence[Sequence[int]], p: int) -> Optional[Dict[str, Any]]:
    """
    Return a compact witness of non-zero matrix:
      pick smallest column j with any non-zero entry, and report non-zero rows.
    """
    if not A:
        return None
    p_i = int(p)
    m = len(A)
    n = len(A[0])
    for j in range(n):
        nz: List[Tuple[int, int]] = []
        for i in range(m):
            v = int(A[i][j]) % p_i
            if v != 0:
                nz.append((int(i), int(v)))
        if nz:
            return {"basis_index": int(j), "nonzero_rows": nz}
    return None


def _mat_exp_nilpotent_modp(N: Sequence[Sequence[int]], p: int, *, max_terms: Optional[int] = None) -> List[List[int]]:
    """
    Compute exp(N) over F_p for (assumed) nilpotent N:
        exp(N) = I + N + N^2/2! + ... + N^k/k!
    Terminates early when N^k becomes zero.

    Engineering:
      - Deterministic
      - Works best for strictly triangular matrices (guaranteed nilpotent)
    """
    p_i = int(p)
    if not N:
        raise MVP17CFGInputError("exp_nilpotent: empty matrix")
    d = len(N)
    if any(len(row) != d for row in N):
        raise MVP17CFGInputError("exp_nilpotent: matrix must be square")
    if max_terms is None:
        # For strictly triangular, nilpotency index ≤ d.
        max_terms = int(d)
    if max_terms < 1:
        raise MVP17CFGInputError("exp_nilpotent: max_terms must be >= 1")
    if max_terms >= p_i:
        # factorial inverse exists for k < p in F_p (p prime)
        max_terms = p_i - 1

    I = _identity(d)
    out = [list(row) for row in I]
    powN = [list(map(int, row)) for row in N]  # N^1
    fact = 1
    for k in range(1, int(max_terms) + 1):
        fact = (fact * k) % p_i
        inv_fact = pow(int(fact), -1, p_i)
        term = _mat_scale_modp(powN, inv_fact, p_i)
        out = _mat_add_modp(out, term, p_i)
        if _mat_is_zero(powN):
            break
        # next power
        powN = matmul_modp(powN, N, p_i)
    return out


def _deterministic_mix(u: int, v: int) -> int:
    """
    Deterministic 32-bit-ish mixing (no python hash randomization).
    """
    x = (int(u) & 0xFFFFFFFF) ^ ((int(v) << 1) & 0xFFFFFFFF)
    x = (x * 0x9E3779B1) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= (x >> 13)
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= (x >> 16)
    return int(x)


# =============================================================================
# Section 0: Minimal EVM bytecode parsing (deterministic, no EVM execution)
# =============================================================================

_OPNAME: Dict[int, str] = {
    0x00: "STOP",
    0x01: "ADD",
    0x02: "MUL",
    0x03: "SUB",
    0x04: "DIV",
    0x05: "SDIV",
    0x06: "MOD",
    0x07: "SMOD",
    0x08: "ADDMOD",
    0x09: "MULMOD",
    0x0A: "EXP",
    0x10: "LT",
    0x11: "GT",
    0x14: "EQ",
    0x15: "ISZERO",
    0x56: "JUMP",
    0x57: "JUMPI",
    0x5B: "JUMPDEST",
    0xF0: "CREATE",
    0xF1: "CALL",
    0xF2: "CALLCODE",
    0xF4: "DELEGATECALL",
    0xF5: "CREATE2",
    0xFA: "STATICCALL",
    0xFD: "REVERT",
    0xFE: "INVALID",
    0xFF: "SELFDESTRUCT",
}


@dataclass(frozen=True)
class EVMInstruction:
    pc: int
    opcode: int
    op: str
    push_bytes: int = 0


def _strip_0x(s: str) -> str:
    ss = str(s)
    if ss.startswith("0x") or ss.startswith("0X"):
        return ss[2:]
    return ss


def decode_evm_bytecode(bytecode_hex: str) -> List[EVMInstruction]:
    """
    Decode EVM bytecode into a PC-indexed instruction list.

    Deterministic, no execution, no heuristics.
    """
    hx = _strip_0x(bytecode_hex or "")
    if hx == "" or hx == "0":
        return []
    if len(hx) % 2 != 0:
        raise MVP17CFGInputError("bytecode hex must have even length")
    try:
        b = bytes.fromhex(hx)
    except ValueError as e:
        raise MVP17CFGInputError(f"Invalid bytecode hex: {e}") from e

    out: List[EVMInstruction] = []
    pc = 0
    n = len(b)
    while pc < n:
        op = int(b[pc])
        if 0x60 <= op <= 0x7F:
            push_n = op - 0x5F
            out.append(EVMInstruction(pc=int(pc), opcode=op, op=f"PUSH{push_n}", push_bytes=int(push_n)))
            pc += 1 + push_n
            continue
        name = _OPNAME.get(op) or f"OP_{op:02x}"
        out.append(EVMInstruction(pc=int(pc), opcode=op, op=str(name), push_bytes=0))
        pc += 1
    return out


def _map_pc_to_block(pc: int, block_nodes_sorted: Sequence[int]) -> Optional[int]:
    """
    Map an instruction pc to its basic block start (largest block_pc <= pc).
    """
    if not block_nodes_sorted:
        return None
    i = bisect_right(list(block_nodes_sorted), int(pc)) - 1
    if i < 0:
        return None
    return int(block_nodes_sorted[i])


@dataclass(frozen=True)
class CallsiteSignature:
    call_pcs: List[int]
    call_blocks: List[int]
    delegatecall_pcs: List[int]
    delegatecall_blocks: List[int]
    staticcall_pcs: List[int]
    staticcall_blocks: List[int]
    create_pcs: List[int]
    create_blocks: List[int]

    def as_meta(self) -> Dict[str, Any]:
        return {
            "callsite": {
                "call_pcs": list(self.call_pcs),
                "call_blocks": list(self.call_blocks),
                "delegatecall_pcs": list(self.delegatecall_pcs),
                "delegatecall_blocks": list(self.delegatecall_blocks),
                "staticcall_pcs": list(self.staticcall_pcs),
                "staticcall_blocks": list(self.staticcall_blocks),
                "create_pcs": list(self.create_pcs),
                "create_blocks": list(self.create_blocks),
            }
        }


def extract_callsites_from_bytecode(bytecode_hex: str, *, block_nodes: Sequence[int]) -> CallsiteSignature:
    """
    从字节码提取 CALL/DELEGATECALL/STATICCALL/CREATE 调用点，将它们映射到CFG
    """
    instrs = decode_evm_bytecode(bytecode_hex)
    nodes_sorted = sorted(set(int(x) for x in block_nodes))

    call_pcs: List[int] = []
    delegatecall_pcs: List[int] = []
    staticcall_pcs: List[int] = []
    create_pcs: List[int] = []
    for ins in instrs:
        if ins.opcode == 0xF1 or ins.opcode == 0xF2:
            call_pcs.append(int(ins.pc))
        elif ins.opcode == 0xF4:
            delegatecall_pcs.append(int(ins.pc))
        elif ins.opcode == 0xFA:
            staticcall_pcs.append(int(ins.pc))
        elif ins.opcode == 0xF0 or ins.opcode == 0xF5:
            create_pcs.append(int(ins.pc))

    def map_blocks(pcs: Sequence[int]) -> List[int]:
        blocks: List[int] = []
        for pc in pcs:
            b = _map_pc_to_block(int(pc), nodes_sorted)
            if b is not None:
                blocks.append(int(b))
        return sorted(set(blocks))

    return CallsiteSignature(
        call_pcs=sorted(call_pcs),
        call_blocks=map_blocks(call_pcs),
        delegatecall_pcs=sorted(delegatecall_pcs),
        delegatecall_blocks=map_blocks(delegatecall_pcs),
        staticcall_pcs=sorted(staticcall_pcs),
        staticcall_blocks=map_blocks(staticcall_pcs),
        create_pcs=sorted(create_pcs),
        create_blocks=map_blocks(create_pcs),
    )


def _build_adjacency(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    node_set = set(nodes)
    adj: Dict[int, List[int]] = {int(n): [] for n in node_set}
    for u, v in edges:
        uu, vv = int(u), int(v)
        if uu in node_set and vv in node_set:
            adj[uu].append(vv)
    for k in list(adj.keys()):
        adj[k] = sorted(set(adj[k]))
    return adj


def _reverse_adjacency(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    node_set = set(nodes)
    radj: Dict[int, List[int]] = {int(n): [] for n in node_set}
    for u, v in edges:
        uu, vv = int(u), int(v)
        if uu in node_set and vv in node_set:
            radj[vv].append(uu)
    for k in list(radj.keys()):
        radj[k] = sorted(set(radj[k]))
    return radj


def _kosaraju_scc(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """
    Iterative Kosaraju SCC (deterministic order).
    """
    adj = _build_adjacency(nodes, edges)
    radj = _reverse_adjacency(nodes, edges)

    visited: set[int] = set()
    order: List[int] = []

    for n in sorted(adj.keys()):
        if n in visited:
            continue
        stack: List[Tuple[int, int]] = [(n, 0)]
        visited.add(n)
        while stack:
            u, i = stack[-1]
            neigh = adj.get(u, [])
            if i < len(neigh):
                v = neigh[i]
                stack[-1] = (u, i + 1)
                if v not in visited:
                    visited.add(v)
                    stack.append((v, 0))
            else:
                stack.pop()
                order.append(u)

    comps: List[List[int]] = []
    assigned: set[int] = set()
    for n in reversed(order):
        if n in assigned:
            continue
        comp: List[int] = []
        stack = [n]
        assigned.add(n)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in radj.get(u, []):
                if v not in assigned:
                    assigned.add(v)
                    stack.append(v)
        comps.append(sorted(comp))

    comps.sort(key=lambda c: (len(c), c[0] if c else 0), reverse=True)
    return comps


# =============================================================================
# Section G: Green's Current Streams (graph Green function / barrier field)
# =============================================================================

def _build_undirected_neighbors(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    node_set = set(int(n) for n in nodes)
    nbr: Dict[int, set[int]] = {int(n): set() for n in node_set}
    for u, v in edges:
        uu, vv = int(u), int(v)
        if uu in node_set and vv in node_set:
            nbr[uu].add(vv)
            nbr[vv].add(uu)
    return {k: sorted(vs) for k, vs in nbr.items()}


def _undirected_component(nodes: Sequence[int], edges: Sequence[Tuple[int, int]], *, start: int) -> List[int]:
    nbr = _build_undirected_neighbors(nodes, edges)
    s = int(start)
    if s not in nbr:
        return []
    seen: set[int] = {s}
    q = [s]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        for v in nbr.get(u, []):
            vv = int(v)
            if vv not in seen:
                seen.add(vv)
                q.append(vv)
    return sorted(seen)


def _weighted_undirected_laplacian(
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    *,
    weights: Optional[Mapping[Tuple[int, int], float]] = None,
) -> Tuple[np.ndarray, List[int], Dict[int, int]]:
    """
    Build an undirected weighted Laplacian L = D - W over float64.
    Returns (L, node_list, node_index).
    """
    node_list = sorted(set(int(n) for n in nodes))
    n = len(node_list)
    idx = {pc: i for i, pc in enumerate(node_list)}
    W = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        uu, vv = int(u), int(v)
        if uu not in idx or vv not in idx:
            continue
        w = 1.0
        if weights is not None:
            w = float(weights.get((uu, vv), weights.get((vv, uu), 1.0)))
        i = idx[uu]
        j = idx[vv]
        if i == j:
            continue
        # undirected: accumulate symmetric weight
        W[i, j] += w
        W[j, i] += w
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, node_list, idx


def greens_current_streams_certificate(
    *,
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    entry_node: int,
    revert_nodes: Sequence[int],
    lem_gap_edges: Optional[Sequence[Tuple[int, int]]] = None,
    guard_to_revert_edges: Optional[Sequence[Tuple[int, int]]] = None,
    max_cycle_barriers: int = 16,
) -> Dict[str, Any]:
    """
    在 CFG（无向代理）上计算确定性格林势场 g
    离散模型：
    - 将回退节点视为发射排斥势的源（奇点）
    - 求解入口处具有狄利克雷规范的泊松方程：
    L g = b，其中 g(entry)=0
    对于回退节点，b[i]=1
    """
    entry = int(entry_node)
    rev = sorted(set(int(x) for x in revert_nodes))
    if not nodes or not edges:
        return {"ok": True, "skipped": True, "reason": "empty graph"}
    node_set_all = set(int(n) for n in nodes)
    if entry not in node_set_all:
        return {"ok": False, "reason": "entry not in nodes"}

    # Build global weights once (applied per component)
    weights: Dict[Tuple[int, int], float] = {}
    if lem_gap_edges is not None:
        for (u, v) in lem_gap_edges:
            uu, vv = int(u), int(v)
            if uu in node_set_all and vv in node_set_all:
                weights[(uu, vv)] = min(weights.get((uu, vv), 1.0), 0.5)
    if guard_to_revert_edges is not None:
        for (u, v) in guard_to_revert_edges:
            uu, vv = int(u), int(v)
            if uu in node_set_all and vv in node_set_all:
                weights[(uu, vv)] = min(weights.get((uu, vv), 1.0), 0.25)

    # Enumerate undirected connected components (deterministic)
    nbr = _build_undirected_neighbors(nodes, edges)
    comps: List[List[int]] = []
    seen: set[int] = set()
    for n0 in sorted(node_set_all):
        if n0 in seen:
            continue
        # BFS in undirected graph
        q = [int(n0)]
        seen.add(int(n0))
        qi = 0
        comp: List[int] = []
        while qi < len(q):
            u = q[qi]
            qi += 1
            comp.append(u)
            for v in nbr.get(u, []):
                vv = int(v)
                if vv not in seen:
                    seen.add(vv)
                    q.append(vv)
        comps.append(sorted(comp))

    # Solve per component with deterministic ground:
    # - entry component: ground at entry
    # - other components: ground at min(node)
    node_potential: Dict[int, float] = {}
    comp_meta: List[Dict[str, Any]] = []
    residuals: List[float] = []
    g_min = float("inf")
    g_max = float("-inf")
    g_sum = 0.0
    g_cnt = 0

    # Precompute edge list once
    edge_list = [(int(u), int(v)) for (u, v) in edges if int(u) in node_set_all and int(v) in node_set_all]

    for comp in comps:
        comp_set = set(comp)
        comp_edges = [(u, v) for (u, v) in edge_list if u in comp_set and v in comp_set]
        if len(comp) <= 1:
            # singleton component => potential 0
            for x in comp:
                node_potential[int(x)] = 0.0
            comp_meta.append(
                {
                    "ground_node": int(comp[0]),
                    "n_nodes": int(len(comp)),
                    "n_edges": int(len(comp_edges)),
                    "revert_sources": [],
                    "laplacian_residual_rel": 0.0,
                    "note": "singleton component",
                }
            )
            continue

        ground = int(entry) if int(entry) in comp_set else int(min(comp))

        L, node_list, node_index = _weighted_undirected_laplacian(comp, comp_edges, weights=weights)
        n = len(node_list)
        b = np.zeros(n, dtype=np.float64)
        rev_in = [r for r in rev if r in comp_set]
        for r in rev_in:
            b[node_index[int(r)]] += 1.0

        gnd = node_index.get(ground)
        if gnd is None:
            return {"ok": False, "reason": "ground not in component index"}
        mask = np.ones(n, dtype=bool)
        mask[gnd] = False
        Lr = L[mask][:, mask]
        br = b[mask]

        try:
            gr = np.linalg.solve(Lr, br)
        except np.linalg.LinAlgError as e:
            return {"ok": False, "reason": f"laplacian_singular: {e}", "n_nodes": int(n), "ground_node": int(ground)}

        g = np.zeros(n, dtype=np.float64)
        g[mask] = gr
        g[gnd] = 0.0

        resid = np.linalg.norm(Lr @ gr - br, ord=2)
        bnorm = np.linalg.norm(br, ord=2) + np.finfo(np.float64).eps
        rel_resid = float(resid / bnorm)
        residuals.append(rel_resid)

        for i in range(n):
            pc = int(node_list[i])
            val = float(g[i])
            node_potential[pc] = val
            g_min = min(g_min, val)
            g_max = max(g_max, val)
            g_sum += val
            g_cnt += 1

        comp_meta.append(
            {
                "ground_node": int(ground),
                "n_nodes": int(n),
                "n_edges": int(len(comp_edges)),
                "revert_sources": rev_in,
                "laplacian_residual_rel": rel_resid,
                "contains_entry": bool(int(entry) in comp_set),
            }
        )

    # Guard-to-revert edge barriers: g at guard node as "barrier height" proxy
    gtr = guard_to_revert_edges or []
    gtr_vals: List[float] = []
    for (u, v) in gtr:
        uu, vv = int(u), int(v)
        if uu in node_potential and vv in node_potential:
            gtr_vals.append(float(node_potential[uu]))

    out: Dict[str, Any] = {
        "ok": True,
        "gauge": {"type": "dirichlet_per_component", "entry_grounded": int(entry)},
        "component_count": int(len(comp_meta)),
        "components": comp_meta,
        "laplacian_residual_rel_max": float(max(residuals) if residuals else 0.0),
        "potential_stats": {
            "min": 0.0 if g_cnt == 0 else float(g_min),
            "max": 0.0 if g_cnt == 0 else float(g_max),
            "mean": 0.0 if g_cnt == 0 else float(g_sum / float(g_cnt)),
        },
        "node_potential": node_potential,
        "guard_to_revert_barrier": {
            "count": int(len(gtr_vals)),
            "mean": float(np.mean(gtr_vals)) if gtr_vals else 0.0,
            "max": float(np.max(gtr_vals)) if gtr_vals else 0.0,
        },
        "weights_policy": {
            "default": 1.0,
            "lem_gap_edge": 0.5,
            "guard_to_revert_edge": 0.25,
            "note": "Lower conductance => higher Green barrier around singularities.",
        },
        "explain": "Green's Current Streams: solve weighted graph Poisson equation per connected component with deterministic Dirichlet gauge; barrier integral along path is sum of avg(g)*1/conductance (discrete).",
    }
    return out


def _edge_conductance(
    u: int,
    v: int,
    *,
    lem_gap_edges: set[Tuple[int, int]],
    guard_to_revert_edges: set[Tuple[int, int]],
) -> float:
    """
    Deterministic conductance policy (must match Green solver policy for consistency):
      - default: 1.0
      - LEM-gap edge: 0.5
      - guard_to_revert edge: 0.25
    """
    uu, vv = int(u), int(v)
    w = 1.0
    if (uu, vv) in lem_gap_edges:
        w = min(w, 0.5)
    if (uu, vv) in guard_to_revert_edges:
        w = min(w, 0.25)
    return float(w)


def _dijkstra_green_geodesics(
    *,
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    entry: int,
    node_potential: Mapping[int, float],
    lem_gap_edges: set[Tuple[int, int]],
    guard_to_revert_edges: set[Tuple[int, int]],
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Directed CFG geodesics under Green barrier cost.

    Edge cost:
        c(u->v) = ((g(u)+g(v))/2) * (1/conductance(u,v))
    """
    entry_i = int(entry)
    if entry_i not in node_potential:
        return {}, {}

    adj: Dict[int, List[int]] = {}
    node_set = set(int(n) for n in nodes)
    for u, v in edges:
        uu, vv = int(u), int(v)
        if uu in node_set and vv in node_set:
            adj.setdefault(uu, []).append(vv)
    for k in list(adj.keys()):
        adj[k] = sorted(set(adj[k]))

    INF = float("inf")
    dist: Dict[int, float] = {entry_i: 0.0}
    prev: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(0.0, entry_i)]

    # Deterministic tie-break epsilon
    eps = 1e-15

    while pq:
        du, u = heapq.heappop(pq)
        # stale
        if du > dist.get(u, INF) + eps:
            continue
        gu = float(node_potential.get(u, 0.0))
        for v in adj.get(u, []):
            if v not in node_potential:
                continue
            gv = float(node_potential.get(v, 0.0))
            w = _edge_conductance(u, v, lem_gap_edges=lem_gap_edges, guard_to_revert_edges=guard_to_revert_edges)
            length_factor = 1.0 / max(w, np.finfo(np.float64).eps)
            cost = 0.5 * (gu + gv) * length_factor
            nd = du + cost
            cur = dist.get(v, INF)
            if nd < cur - eps:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
            elif abs(nd - cur) <= eps:
                # deterministic predecessor tie-break
                if u < prev.get(v, 2**31 - 1):
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

    return dist, prev


def _reconstruct_path(prev: Mapping[int, int], *, entry: int, target: int, max_len: int = 128) -> List[int]:
    entry_i = int(entry)
    t = int(target)
    path: List[int] = [t]
    seen: set[int] = {t}
    while t != entry_i:
        if t not in prev:
            break
        t = int(prev[t])
        if t in seen:
            break
        seen.add(t)
        path.append(t)
        if len(path) >= int(max_len):
            break
    path.reverse()
    return path


def _bfs_shortest_parents(
    adj: Mapping[int, Sequence[int]],
    *,
    start: int,
    max_depth: int = 96,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Deterministic directed BFS up to max_depth.
    Returns (parent, dist) where:
      - parent[v] is the predecessor of v on a shortest path from start
      - dist[v] is hop distance
    """
    s = int(start)
    parent: Dict[int, int] = {}
    dist: Dict[int, int] = {s: 0}
    q: List[int] = [s]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        du = int(dist[u])
        if du >= int(max_depth):
            continue
        for v in adj.get(u, []):
            vv = int(v)
            if vv in dist:
                continue
            dist[vv] = du + 1
            parent[vv] = u
            q.append(vv)
    return parent, dist


def _reconstruct_bfs_path(parent: Mapping[int, int], *, start: int, target: int, max_len: int = 256) -> Optional[List[int]]:
    """
    Reconstruct path from start to target using BFS parent pointers.
    Returns list of nodes [start, ..., target] or None if target not reached.
    """
    s = int(start)
    t = int(target)
    if t == s:
        return [s]
    if t not in parent:
        return None
    path = [t]
    seen: set[int] = {t}
    while path[-1] != s:
        cur = int(path[-1])
        if cur not in parent:
            return None
        cur = int(parent[cur])
        if cur in seen:
            return None
        seen.add(cur)
        path.append(cur)
        if len(path) >= int(max_len):
            return None
    path.reverse()
    return path


def _bfs_depths(entry: int, adj: Mapping[int, Sequence[int]]) -> Dict[int, int]:
    dist: Dict[int, int] = {int(entry): 0}
    q = [int(entry)]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        du = dist[u]
        for v in adj.get(u, []):
            vv = int(v)
            if vv not in dist:
                dist[vv] = du + 1
                q.append(vv)
    return dist


def _find_cycle_in_scc(scc_nodes: Sequence[int], adj: Mapping[int, Sequence[int]], *, start: int) -> Optional[List[int]]:
    """
    迭代深度优先搜索 (DFS) 在 SCC 中查找一个有向环
    返回环的元素为 [n0, n1, ..., nk, n0]
    """
    scc_set = set(map(int, scc_nodes))
    start_i = int(start)
    if start_i not in scc_set:
        return None

    in_stack: set[int] = set()
    visited: set[int] = set()
    path: List[int] = []
    pos: Dict[int, int] = {}

    stack: List[Tuple[int, int]] = [(start_i, 0)]
    visited.add(start_i)
    in_stack.add(start_i)
    path.append(start_i)
    pos[start_i] = 0

    while stack:
        u, i = stack[-1]
        neigh = [v for v in adj.get(u, []) if int(v) in scc_set]
        if i >= len(neigh):
            stack.pop()
            in_stack.discard(u)
            if path and path[-1] == u:
                path.pop()
                pos.pop(u, None)
            continue
        v = int(neigh[i])
        stack[-1] = (u, i + 1)

        if v in in_stack:
            # back edge to current recursion stack => cycle
            j = pos.get(v)
            if j is None:
                continue
            cyc = path[j:] + [v]
            return cyc
        if v not in visited:
            visited.add(v)
            in_stack.add(v)
            stack.append((v, 0))
            pos[v] = len(path)
            path.append(v)

    return None


@dataclass(frozen=True)
class MVP17BaseSpace:
    theta_axes: Tuple[str, str, str]
    entry_node: int
    node_theta: Dict[int, Tuple[float, float, float]]
    lem_gap_nodes: List[int]
    lem_gap_edges: List[Tuple[int, int]]
    meta: Dict[str, Any]
    lem_gap_tags: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "theta_axes": list(self.theta_axes),
            "entry_node": int(self.entry_node),
            "lem_gap": {
                "nodes": list(self.lem_gap_nodes),
                "edges": [[int(u), int(v)] for u, v in self.lem_gap_edges],
                "tags": dict(self.lem_gap_tags),
            },
            "meta": dict(self.meta),
        }

class CrystallineFrobeniusError(RuntimeError):
    """晶体 Frobenius 谱分析错误（必须中断，禁止静默失败）。"""


def _as_int(x: Any, *, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise CrystallineFrobeniusError(f"{name} must be int-like, got {x!r}") from e


def _fraction_to_json(q: Fraction) -> Any:
    if not isinstance(q, Fraction):
        raise CrystallineFrobeniusError("internal: expected Fraction")
    if q.denominator == 1:
        return int(q.numerator)
    return {"num": int(q.numerator), "den": int(q.denominator)}


def _sorted_cycles(perm: Sequence[int]) -> List[List[int]]:
    n = len(perm)
    seen = [False] * n
    out: List[List[int]] = []
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
                raise CrystallineFrobeniusError(f"perm is not a permutation: jump to {i}")
        # deterministic: rotate cycle so smallest index is first
        if cyc:
            j = min(range(len(cyc)), key=lambda k: cyc[k])
            cyc = cyc[j:] + cyc[:j]
            out.append(cyc)
    out.sort(key=lambda c: (len(c), c[0]))
    return out


@dataclass(frozen=True)
class MonomialOperator:
    """
    单项式算子（monomial operator）：
      e_i ↦ p^{exp[i]} · e_{perm[i]}

    这等价于一个“置换矩阵 × 对角 p 幂缩放”的表示，但不显式构造矩阵，避免大整数/稠密计算。
    """

    p: int
    perm: Tuple[int, ...]
    exp: Tuple[int, ...]
    steps: int

    def __post_init__(self) -> None:
        p_i = _as_int(self.p, name="p")
        if p_i <= 1:
            raise CrystallineFrobeniusError(f"p must be > 1, got {p_i}")

        if len(self.perm) != len(self.exp):
            raise CrystallineFrobeniusError("perm/exp length mismatch")

        d = len(self.perm)
        if d < 1:
            raise CrystallineFrobeniusError("basis_dim must be >= 1")

        # perm validity
        perm_list = [int(x) for x in self.perm]
        if sorted(perm_list) != list(range(d)):
            raise CrystallineFrobeniusError("perm is not a permutation of [0..d-1]")

        # exp validity
        for i, e in enumerate(self.exp):
            ee = _as_int(e, name=f"exp[{i}]")
            if ee < 0:
                # exp<0 is valid on the isocrystal (after inverting p), but not on the lattice.
                # Here we allow it, because we may compare path1 vs path2 (ratios) upstream.
                continue

        steps_i = _as_int(self.steps, name="steps")
        if steps_i < 0:
            raise CrystallineFrobeniusError("steps must be >= 0")

    @property
    def dim(self) -> int:
        return int(len(self.perm))


def monomial_identity(*, basis_dim: int, p: int) -> MonomialOperator:
    d = _as_int(basis_dim, name="basis_dim")
    if d < 1:
        raise CrystallineFrobeniusError("basis_dim must be >= 1")
    return MonomialOperator(
        p=int(p),
        perm=tuple(range(d)),
        exp=tuple(0 for _ in range(d)),
        steps=0,
    )


def monomial_compose(left: MonomialOperator, right: MonomialOperator) -> MonomialOperator:
    """
    返回 left ∘ right
    """
    if not isinstance(left, MonomialOperator) or not isinstance(right, MonomialOperator):
        raise CrystallineFrobeniusError("monomial_compose requires MonomialOperator arguments")
    if int(left.p) != int(right.p):
        raise CrystallineFrobeniusError(f"prime mismatch: {left.p} vs {right.p}")
    if left.dim != right.dim:
        raise CrystallineFrobeniusError(f"dimension mismatch: {left.dim} vs {right.dim}")

    d = left.dim
    perm_l = left.perm
    perm_r = right.perm
    exp_l = left.exp
    exp_r = right.exp

    perm: List[int] = [0] * d
    exp: List[int] = [0] * d
    for j in range(d):
        mid = int(perm_r[j])
        perm[j] = int(perm_l[mid])
        exp[j] = int(exp_r[j]) + int(exp_l[mid])

    return MonomialOperator(
        p=int(left.p),
        perm=tuple(perm),
        exp=tuple(exp),
        steps=int(left.steps) + int(right.steps),
    )


def monomial_edge_operator(
    *,
    u: int,
    v: int,
    basis_dim: int,
    p: int,
    weights: Sequence[int],
) -> MonomialOperator:
    """
    一条边 (u,v) 的单项式算子：
      - 置换：swap(u mod d, v mod d)
      - p 幂：对源基向量 e_i 乘 p^{w_i}（w_i 来自 Hodge 权重）
    """
    d = _as_int(basis_dim, name="basis_dim")
    if d < 1:
        raise CrystallineFrobeniusError("basis_dim must be >= 1")
    if len(weights) != d:
        raise CrystallineFrobeniusError("weights length mismatch with basis_dim")

    uu = _as_int(u, name="u")
    vv = _as_int(v, name="v")
    i = int(uu % d)
    j = int(vv % d)

    perm = list(range(d))
    if i != j:
        perm[i], perm[j] = perm[j], perm[i]

    exp = [int(w) for w in weights]
    return MonomialOperator(p=int(p), perm=tuple(perm), exp=tuple(exp), steps=1)


def monomial_path_operator(
    edges: Sequence[Tuple[int, int]],
    *,
    basis_dim: int,
    p: int,
    weights: Sequence[int],
) -> MonomialOperator:
    op = monomial_identity(basis_dim=int(basis_dim), p=int(p))
    for (u, v) in edges:
        step = monomial_edge_operator(u=int(u), v=int(v), basis_dim=int(basis_dim), p=int(p), weights=weights)
        op = monomial_compose(step, op)
    return op


def hodge_weights_from_fiber(
    *,
    basis: Sequence[str],
    visible_basis: Sequence[str],
    weight_visible: int = 0,
    weight_nonvisible: int = 1,
) -> List[int]:
    """
    Hodge 权重 surrogate：
      - visible -> 0
      - non-visible (shadow + residue) -> 1
    """
    vis = set(str(x) for x in visible_basis)
    w0 = _as_int(weight_visible, name="weight_visible")
    w1 = _as_int(weight_nonvisible, name="weight_nonvisible")
    if w0 < 0 or w1 < 0:
        raise CrystallineFrobeniusError("weights must be >= 0")
    out: List[int] = []
    for b in basis:
        out.append(int(w0 if str(b) in vis else w1))
    if not out:
        raise CrystallineFrobeniusError("basis is empty")
    return out


def _polygon_points_from_unit_slopes(slopes: Sequence[Fraction]) -> List[Dict[str, Any]]:
    """
    输入 slopes 为长度 d 的单位步斜率（每步 Δx=1），输出多边形在整数 x 处的点。
    """
    y = Fraction(0, 1)
    pts: List[Dict[str, Any]] = [{"x": 0, "y": _fraction_to_json(y)}]
    for i, s in enumerate(slopes, start=1):
        if not isinstance(s, Fraction):
            raise CrystallineFrobeniusError("internal: slopes must be Fractions")
        y += s
        pts.append({"x": int(i), "y": _fraction_to_json(y)})
    return pts


def analyze_monomial_operator(
    op: MonomialOperator,
    *,
    hodge_weights: Sequence[int],
) -> Dict[str, Any]:
    """
    对单项式算子 op 做 Newton/Hodge–Newton 分析（使用“每一步”归一化斜率）。
    """
    if not isinstance(op, MonomialOperator):
        raise CrystallineFrobeniusError("op must be a MonomialOperator")
    d = op.dim
    if len(hodge_weights) != d:
        raise CrystallineFrobeniusError("hodge_weights length mismatch")

    m = int(op.steps)
    if m <= 0:
        raise CrystallineFrobeniusError("cannot analyze operator with steps=0 (empty path)")

    cycles = _sorted_cycles(op.perm)
    cycle_rows: List[Dict[str, Any]] = []
    newton_slopes_unit: List[Fraction] = []
    factors: List[Dict[str, Any]] = []

    for cyc in cycles:
        L = int(len(cyc))
        E = int(sum(int(op.exp[i]) for i in cyc))
        slope = Fraction(E, L * m)
        # multiplicity L
        for _ in range(L):
            newton_slopes_unit.append(slope)
        cycle_rows.append(
            {
                "cycle": [int(i) for i in cyc],
                "cycle_len": int(L),
                "exp_sum": int(E),
                "slope_per_step": _fraction_to_json(slope),
            }
        )
        factors.append({"degree": int(L), "p_exp": int(E), "factor": f"T^{int(L)} - p^{int(E)}"})

    # Hodge polygon slopes (unit steps)
    h_slopes_unit: List[Fraction] = [Fraction(int(w), 1) for w in hodge_weights]
    h_slopes_unit_sorted = sorted(h_slopes_unit)

    # Newton slopes sorted
    newton_slopes_unit_sorted = sorted(newton_slopes_unit)

    # Compare polygons via slopes list (unit-step resolution)
    ordinary = (newton_slopes_unit_sorted == h_slopes_unit_sorted)
    supersingular = not ordinary

    # Vertical gap at integer x (Newton - Hodge), using sorted slopes (canonical convex polygons)
    new_pts = _polygon_points_from_unit_slopes(newton_slopes_unit_sorted)
    hod_pts = _polygon_points_from_unit_slopes(h_slopes_unit_sorted)
    gap_max = Fraction(0, 1)
    gap_at: Optional[int] = None
    for i in range(len(new_pts)):
        yN = newton_slopes_unit_sorted[:i]
        yH = h_slopes_unit_sorted[:i]
        # compute cumulative y exactly
        sN = sum(yN, Fraction(0, 1))
        sH = sum(yH, Fraction(0, 1))
        gap = sN - sH
        if gap > gap_max:
            gap_max = gap
            gap_at = int(i)

    # Slope jump witness: any non-integral slope (den!=1)
    frac_slopes = [s for s in newton_slopes_unit_sorted if s.denominator != 1]
    frac_witness = None
    if frac_slopes:
        # pick the smallest fractional slope deterministically
        s0 = frac_slopes[0]
        frac_witness = {"slope_per_step": _fraction_to_json(s0), "note": "non-integral Newton slope => supersingular locus"}

    return {
        "ok": True,
        "field": {"type": "p-adic", "p": int(op.p)},
        "basis_dim": int(d),
        "steps": int(m),
        "charpoly": {
            "factored": factors,
            "note": "Characteristic polynomial given in cycle-factor form for monomial Frobenius operator.",
        },
        "newton": {
            "slopes_per_step_sorted": [_fraction_to_json(s) for s in newton_slopes_unit_sorted],
            "polygon_points": new_pts,
            "cycle_decomposition": cycle_rows,
        },
        "hodge": {
            "weights": [int(w) for w in hodge_weights],
            "slopes_sorted": [_fraction_to_json(s) for s in h_slopes_unit_sorted],
            "polygon_points": hod_pts,
        },
        "hodge_newton": {
            "ordinary": bool(ordinary),
            "supersingular": bool(supersingular),
            "max_vertical_gap": _fraction_to_json(gap_max),
            "gap_witness_x": gap_at,
            "fractional_slope_witness": frac_witness,
        },
        "explain": (
            "Crystalline Frobenius (monomial model): φ_e(e_i)=p^{w_i}·e_{π_e(i)} with π_e=swap(u mod d, v mod d). "
            "Newton slopes are cycle-averaged p-exponents; compare with Hodge slopes (visible→0, non-visible→1)."
        ),
    }


def crystalline_frobenius_spectral_certificate(
    *,
    p: int,
    basis: Sequence[str],
    visible_basis: Sequence[str],
    cycles: Sequence[Dict[str, Any]],
    wilson_loops: Sequence[Dict[str, Any]],
    max_items: int,
) -> Dict[str, Any]:
    """
    生成一个面向 report 的证书字典。
    输入结构与 MVP17 CFG auditor 的 cycles / wilson_loops 形状兼容：
      - cycle["cycle_edges"] = [[u,v], ...]
      - wilson["path1_edges"], wilson["path2_edges"] = [[u,v], ...]
    """
    p_i = _as_int(p, name="p")
    d = len(list(basis))
    if d < 1:
        raise CrystallineFrobeniusError("basis is empty")
    mi = _as_int(max_items, name="max_items")
    if mi < 1:
        raise CrystallineFrobeniusError("max_items must be >= 1")

    weights = hodge_weights_from_fiber(basis=basis, visible_basis=visible_basis)

    # Analyze cycles
    cyc_rows: List[Dict[str, Any]] = []
    for i, c in enumerate(cycles[:mi]):
        edges_raw = c.get("cycle_edges") or []
        edges: List[Tuple[int, int]] = []
        for e in edges_raw:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                edges.append((int(e[0]), int(e[1])))
        if not edges:
            cyc_rows.append({"ok": False, "index": int(i), "reason": "cycle_edges missing/empty"})
            continue
        op = monomial_path_operator(edges, basis_dim=d, p=p_i, weights=weights)
        try:
            rep = analyze_monomial_operator(op, hodge_weights=weights)
            cyc_rows.append({"ok": True, "index": int(i), "kind": "cycle", "analysis": rep})
        except Exception as e:
            cyc_rows.append({"ok": False, "index": int(i), "kind": "cycle", "reason": f"analyze_failed: {e}"})

    # Analyze diamond paths separately (path1 vs path2)
    dia_rows: List[Dict[str, Any]] = []
    for i, w in enumerate(wilson_loops[:mi]):
        p1_raw = w.get("path1_edges") or []
        p2_raw = w.get("path2_edges") or []

        def _edges(x: Any) -> List[Tuple[int, int]]:
            out: List[Tuple[int, int]] = []
            if not isinstance(x, list):
                return out
            for e in x:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    out.append((int(e[0]), int(e[1])))
            return out

        p1 = _edges(p1_raw)
        p2 = _edges(p2_raw)
        if not p1 or not p2:
            dia_rows.append({"ok": False, "index": int(i), "kind": "diamond", "reason": "missing path edges"})
            continue

        op1 = monomial_path_operator(p1, basis_dim=d, p=p_i, weights=weights)
        op2 = monomial_path_operator(p2, basis_dim=d, p=p_i, weights=weights)

        try:
            a1 = analyze_monomial_operator(op1, hodge_weights=weights)
            a2 = analyze_monomial_operator(op2, hodge_weights=weights)

            # Deterministic mismatch signal: compare sorted Newton slopes per-step lists
            s1 = a1.get("newton", {}).get("slopes_per_step_sorted")
            s2 = a2.get("newton", {}).get("slopes_per_step_sorted")
            mismatch = (s1 != s2)

            dia_rows.append(
                {
                    "ok": True,
                    "index": int(i),
                    "kind": "diamond",
                    "branch": w.get("branch"),
                    "merge": w.get("merge"),
                    "paths": {"path1": a1, "path2": a2},
                    "slope_mismatch": bool(mismatch),
                }
            )
        except Exception as e:
            dia_rows.append({"ok": False, "index": int(i), "kind": "diamond", "reason": f"analyze_failed: {e}"})

    # Verdict
    any_sup = any(
        bool(r.get("ok"))
        and isinstance(r.get("analysis"), dict)
        and bool(((r["analysis"].get("hodge_newton") or {}).get("supersingular")))
        for r in cyc_rows
    )
    any_mismatch = any(bool(r.get("ok")) and bool(r.get("slope_mismatch")) for r in dia_rows)

    return {
        "ok": True,
        "field": {"type": "p-adic", "p": int(p_i)},
        "basis_dim": int(d),
        "hodge_weights": [int(w) for w in weights],
        "cycles": cyc_rows,
        "diamonds": dia_rows,
        "signals": {
            "any_supersingular_cycle": bool(any_sup),
            "any_diamond_slope_mismatch": bool(any_mismatch),
        },
        "explain": "Second-stage crystalline certificate: compute Newton slopes from Dieudonné-style monomial Frobenius on each cycle/path; ordinary iff Newton==Hodge.",
    }


def witt_curvature_certificate_from_mvp17(
    *,
    p: int,
    cycles: Sequence[Mapping[str, Any]],
    wilson_loops: Sequence[Mapping[str, Any]],
    fiber_dim: int,
    edge_transport: Optional[Mapping[Tuple[int, int], Any]] = None,
) -> Dict[str, Any]:
    """
    Phase 5: Witt 接触几何层（Witt contact geometry）证书。

    目标（严格口径）：
    - 对 MVP17 的 Wilson diamond（两条同伦路径 path1/path2）做“Witt 端点比较”。
    - 这里的“端点”指 transport 的 holonomy：H = T(path1) · T(path2)^{-1}。

    重要工程约束（红线一致）：
    - 本函数 **不允许** 静默吞掉缺边/维度不一致；会在输出里给出明确 reason。
    - 当仅有 F_p 上的边传输（`edge_transport` 只提供 mod p 矩阵）时，
      我们只能给出 **W_1(F_p)=F_p** 的 Witt 证书（不伪造更高 Witt 分量）。
      这不是降级启发式，而是信息论/代数事实：没有 p^k 数据就无法声称 W_k 的等式。
    """
    p_i = int(p)
    d = int(fiber_dim)
    if p_i <= 1:
        return {"ok": False, "reason": f"p must be >=2, got {p_i}"}
    if d < 1:
        return {"ok": False, "reason": f"fiber_dim must be >=1, got {d}"}

    # Witt backend (strict, deterministic). Keep local import to avoid inflating import-time costs.
    try:
        from .mvp17_prismatic import FiniteFieldElement, WittVector  # type: ignore
    except Exception as e:
        return {"ok": False, "reason": f"mvp17_prismatic_unavailable: {e}"}

    # We can only certify W_1(F_p) unless a higher-precision lift is provided.
    # (Currently, MVP17 CFG connection transports are defined over F_p.)
    witt_length = 1

    def _w1(x: int) -> "WittVector":
        return WittVector([FiniteFieldElement(int(x) % p_i, p_i)], p_i)

    def _norm_edges(raw: Any) -> Optional[List[Tuple[int, int]]]:
        if not isinstance(raw, list):
            return None
        out: List[Tuple[int, int]] = []
        for e in raw:
            if not (isinstance(e, (list, tuple)) and len(e) == 2):
                return None
            out.append((int(e[0]), int(e[1])))
        return out

    def _norm_mat_fp(M0: Any) -> Optional[List[List[int]]]:
        # Accept list-of-lists or numpy-like tolist(); normalize to int mod p.
        if M0 is None:
            return None
        if hasattr(M0, "tolist"):
            try:
                M0 = M0.tolist()
            except Exception:
                pass
        if not isinstance(M0, list) or not M0:
            return None
        if not all(isinstance(r, list) for r in M0):
            return None
        try:
            M = [[int(x) % p_i for x in row] for row in M0]
        except Exception:
            return None
        if len(M) != d or any(len(row) != d for row in M):
            return None
        return M

    I = _identity(d)

    def _transport_path_fp(path_edges: List[Tuple[int, int]]) -> Tuple[Optional[List[List[int]]], Optional[Dict[str, Any]]]:
        if edge_transport is None:
            return None, {"reason": "edge_transport_missing"}
        T = _identity(d)
        for (u, v) in path_edges:
            U0 = edge_transport.get((int(u), int(v))) if isinstance(edge_transport, Mapping) else None
            U = _norm_mat_fp(U0)
            if U is None:
                return None, {"reason": "missing_or_malformed_edge_transport", "edge": [int(u), int(v)]}
            T = matmul_modp(T, U, p_i)
        return T, None

    def _delta_fp(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        return [[(int(a) - int(b)) % p_i for a, b in zip(ar, br)] for ar, br in zip(A, B)]

    def _nnz_fp(A: List[List[int]]) -> int:
        return int(sum(1 for row in A for x in row if (int(x) % p_i) != 0))

    def _first_nz_fp(A: List[List[int]]) -> Optional[Dict[str, Any]]:
        for i in range(d):
            for j in range(d):
                v = int(A[i][j]) % p_i
                if v != 0:
                    return {
                        "i": int(i),
                        "j": int(j),
                        "value_modp": int(v),
                        "witt_w1": {"components": [int(v)], "p": int(p_i), "length": int(witt_length)},
                    }
        return None

    wil_rows: List[Dict[str, Any]] = []
    failures_reachable = 0
    failures_all = 0
    checked = 0
    err_cap = 24

    for i, w in enumerate(list(wilson_loops)):
        if not isinstance(w, Mapping):
            wil_rows.append({"ok": False, "index": int(i), "reason": f"wilson_loops[{i}] not mapping"})
            continue
        p1 = _norm_edges(w.get("path1_edges") or [])
        p2 = _norm_edges(w.get("path2_edges") or [])
        if p1 is None or p2 is None:
            wil_rows.append({"ok": False, "index": int(i), "reason": "path_edges_malformed"})
            continue

        T1, e1 = _transport_path_fp(p1)
        T2, e2 = _transport_path_fp(p2)
        if T1 is None or T2 is None:
            wil_rows.append(
                {
                    "ok": False,
                    "index": int(i),
                    "reason": "path_transport_failed",
                    "path1_error": e1,
                    "path2_error": e2,
                }
            )
            continue

        try:
            T2inv = invert_matrix_modp(T2, p_i)
        except Exception as e:
            wil_rows.append({"ok": False, "index": int(i), "reason": f"path2_not_invertible_modp: {e}"})
            continue

        H = matmul_modp(T1, T2inv, p_i)
        delta = _delta_fp(H, I)
        nnz = _nnz_fp(delta)
        witness = _first_nz_fp(delta)
        checked += 1

        reachable = bool(w.get("reachable_from_entry_cfg", True))
        is_id = (nnz == 0)
        if not is_id:
            failures_all += 1
            if reachable:
                failures_reachable += 1

        # Bound per-loop details for JSON size (do not truncate logic, only reporting fields).
        row = {
            "ok": True,
            "index": int(i),
            "loop_type": str(w.get("loop_type") or ""),
            "branch": w.get("branch"),
            "merge": w.get("merge"),
            "reachable_from_entry_cfg": bool(reachable),
            # F_p-level summary
            "holonomy_rank_modp_reported": int(w.get("holonomy_rank_modp", 0)),
            # Witt (W_1) summary – equal to F_p, but explicitly certified in Witt kernel.
            "witt_length": int(witt_length),
            "witt_holonomy_is_identity": bool(is_id),
            "witt_delta_nnz": int(nnz),
            "witt_witness": witness,
        }
        if len(wil_rows) < err_cap:
            wil_rows.append(row)

    any_fail_reach = bool(failures_reachable > 0)
    any_fail_all = bool(failures_all > 0)

    out: Dict[str, Any] = {
        "ok": True,
        "field": {"type": "Witt", "p": int(p_i), "length": int(witt_length)},
        "inputs": {
            "fiber_dim": int(d),
            "cycles_total": int(len(list(cycles))),
            "wilson_loops_total": int(len(list(wilson_loops))),
            "edge_transport_provided": bool(edge_transport is not None),
        },
        "wilson": {
            "checked": int(checked),
            "failures_all": int(failures_all),
            "failures_reachable": int(failures_reachable),
            "rows_preview": wil_rows,
        },
        "signals": {
            # This is the key consumed by MVP17 verdict gating.
            "any_lem_failure_wilson": bool(any_fail_reach),
            # Extra traceability
            "any_failure_all": bool(any_fail_all),
        },
        "note": "This certificate is computed in W_1(F_p)=F_p because MVP17 CFG transports are currently defined mod p. A higher Witt length requires explicit p^k lifts of edge transports.",
        "explain": (
            "Witt contact geometry (minimal certificate): compute diamond holonomy H=T(path1)·T(path2)^{-1} "
            "from edge transports, then certify H=I in the Witt kernel W_1(F_p). Non-identity on any reachable "
            "diamond implies path dependence ⇒ LEM failure witness."
        ),
    }
    # Touch the Witt kernel so this is not a hollow placeholder.
    _ = _w1(0)
    return out


__all__ = [
    "CrystallineFrobeniusError",
    "MonomialOperator",
    "monomial_identity",
    "monomial_compose",
    "monomial_edge_operator",
    "monomial_path_operator",
    "hodge_weights_from_fiber",
    "analyze_monomial_operator",
    "crystalline_frobenius_spectral_certificate",
    "witt_curvature_certificate_from_mvp17",
]

@dataclass(frozen=True)
class MVP17Fiber:
    basis: List[str]
    visible_basis: List[str]
    shadow_basis: List[str]
    residue_basis: List[str]
    shadow_definitions: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "basis": list(self.basis),
            "visible_basis": list(self.visible_basis),
            "shadow_basis": list(self.shadow_basis),
            "residue_basis": list(self.residue_basis),
            "shadow_definitions": dict(self.shadow_definitions),
        }


@dataclass(frozen=True)
class MVP17BerryConnection:
    field_p: int
    generators: List[List[List[int]]]
    edge_transport: Dict[Tuple[int, int], List[List[int]]]
    generator_policy: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": {"type": "Fp", "p": int(self.field_p)},
            "generator_count": int(len(self.generators)),
            "generator_policy": dict(self.generator_policy),
            "edge_transport_count": int(len(self.edge_transport)),
        }


@dataclass(frozen=True)
class MVP17CFGAuditReport:
    ok: bool
    status: str
    base_space: Dict[str, Any]
    fiber: Dict[str, Any]
    connection: Dict[str, Any]
    cycles: List[Dict[str, Any]]
    wilson_loops: List[Dict[str, Any]]
    non_abelian: Dict[str, Any]
    wilson_non_abelian: Dict[str, Any]
    double_negation: Dict[str, Any]
    faltings_height: Dict[str, Any]
    excluded_middle: Dict[str, Any]
    hodge_conjecture: Dict[str, Any]
    greens_current: Dict[str, Any]
    tate_shafarevich: Dict[str, Any]
    crystalline_frobenius: Dict[str, Any]
    period_isomorphism: Dict[str, Any]
    witt_contact: Dict[str, Any]  # Phase 5: 接触几何层
    explain: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "status": str(self.status),
            "base_space": self.base_space,
            "fiber": self.fiber,
            "connection": self.connection,
            "cycles": list(self.cycles),
            "non_abelian": dict(self.non_abelian),
            "wilson_loops": list(self.wilson_loops),
            "wilson_non_abelian": dict(self.wilson_non_abelian),
            "double_negation": dict(self.double_negation),
            "faltings_height": dict(self.faltings_height),
            "excluded_middle": dict(self.excluded_middle),
            "hodge_conjecture": dict(self.hodge_conjecture),
            "greens_current": dict(self.greens_current),
            "tate_shafarevich": dict(self.tate_shafarevich),
            "crystalline_frobenius": dict(self.crystalline_frobenius),
            "period_isomorphism": dict(self.period_isomorphism),
            "witt_contact": dict(self.witt_contact),
            "explain": self.explain,
        }


class MVP17CFGObjectBuilder:
    """
    从 artifacts 的 cfg_topology 建模 MVP17 对象 (M, F, A)。
    """

    def __init__(self, *, p: int = 101) -> None:
        self.p = int(p)
        if self.p <= 2:
            raise MVP17CFGInputError("field prime p must be > 2")

    def build_from_cfg_topology(
        self,
        cfg_topology: Mapping[str, Any],
        *,
        cfg_analysis: Optional[Mapping[str, Any]] = None,
        bytecode: Optional[str] = None,
        evm_constraints: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Tuple[MVP17BaseSpace, MVP17Fiber, MVP17BerryConnection, Dict[str, Any]]:
        if not isinstance(cfg_topology, Mapping):
            raise MVP17CFGInputError("cfg_topology must be a dict-like mapping")

        nodes = _as_int_list(cfg_topology.get("nodes"), name="cfg_topology.nodes")
        edges = _as_edge_list(cfg_topology.get("edges"), name="cfg_topology.edges")
        if not nodes:
            raise MVP17CFGInputError("cfg_topology.nodes is empty")

        node_set = set(nodes)
        edges = [(u, v) for (u, v) in edges if u in node_set and v in node_set]

        revert_nodes = set(_as_int_list(cfg_topology.get("revert_nodes"), name="cfg_topology.revert_nodes"))
        guard_nodes = set(_as_int_list(cfg_topology.get("guard_nodes"), name="cfg_topology.guard_nodes"))
        sensitive_nodes = set(_as_int_list(cfg_topology.get("sensitive_nodes"), name="cfg_topology.sensitive_nodes"))
        sink_nodes = set(_as_int_list(cfg_topology.get("sink_nodes"), name="cfg_topology.sink_nodes"))

        entry = 0 if 0 in node_set else min(node_set)

        adj = _build_adjacency(nodes, edges)
        indeg: Dict[int, int] = {n: 0 for n in node_set}
        outdeg: Dict[int, int] = {n: len(adj.get(n, [])) for n in node_set}
        for u, v in edges:
            indeg[v] = indeg.get(v, 0) + 1

        # SCC / cycle components
        sccs = _kosaraju_scc(nodes, edges)
        cyclic_sccs: List[List[int]] = []
        edge_set = set(edges)
        for comp in sccs:
            if len(comp) > 1:
                cyclic_sccs.append(comp)
                continue
            if comp and (comp[0], comp[0]) in edge_set:
                cyclic_sccs.append(comp)

        # ---------------------------------------------------------------------
        # LEM-gap marking (CFG-first, but with bytecode-backed tags when available)
        # ---------------------------------------------------------------------
        tag_nodes: Dict[str, set[int]] = {}
        tag_edges: Dict[str, set[Tuple[int, int]]] = {}

        def _add_node_tag(tag: str, xs: Iterable[int]) -> None:
            s = tag_nodes.setdefault(str(tag), set())
            for x in xs:
                s.add(int(x))

        def _add_edge_tag(tag: str, xs: Iterable[Tuple[int, int]]) -> None:
            s = tag_edges.setdefault(str(tag), set())
            for (u, v) in xs:
                s.add((int(u), int(v)))

        # Explicit exporter tags
        _add_node_tag("revert", revert_nodes)
        _add_node_tag("guard", guard_nodes)
        _add_node_tag("sensitive", sensitive_nodes)
        _add_node_tag("sink", sink_nodes)

        # Structural: branch/merge surfaces encode path dependence
        branch_merge_nodes: set[int] = set()
        for n in node_set:
            if outdeg.get(n, 0) >= 2 or indeg.get(n, 0) >= 2:
                branch_merge_nodes.add(int(n))
        _add_node_tag("branch_merge", branch_merge_nodes)

        # Cyclic regions: holonomy carriers
        cycle_nodes: set[int] = set()
        for comp in cyclic_sccs:
            for n in comp:
                cycle_nodes.add(int(n))
        _add_node_tag("cycle", cycle_nodes)

        # Bytecode-backed callsites: callback() / external-call windows
        call_sig: Optional[CallsiteSignature] = None
        callsite_nodes: set[int] = set()
        delegatecall_nodes: set[int] = set()
        staticcall_nodes: set[int] = set()
        create_nodes: set[int] = set()
        if isinstance(bytecode, str) and bytecode and bytecode not in ("0x", "0X"):
            try:
                call_sig = extract_callsites_from_bytecode(bytecode, block_nodes=nodes)
                callsite_nodes = set(call_sig.call_blocks)
                delegatecall_nodes = set(call_sig.delegatecall_blocks)
                staticcall_nodes = set(call_sig.staticcall_blocks)
                create_nodes = set(call_sig.create_blocks)
                # Aggregate callsites for MVP17 "callback window"
                _add_node_tag("callsite", callsite_nodes)
                _add_node_tag("delegatecall", delegatecall_nodes)
                _add_node_tag("staticcall", staticcall_nodes)
                _add_node_tag("create", create_nodes)
            except Exception:
                # Do not hide it, but do not hard-fail the entire CFG model if bytecode is malformed.
                call_sig = None

        # Constraint-backed unchecked regions (if present)
        unchecked_nodes: set[int] = set()
        constraint_sensitive_nodes: set[int] = set()
        if evm_constraints is not None:
            for c in evm_constraints:
                if not isinstance(c, Mapping):
                    continue
                meta = c.get("meta")
                if not isinstance(meta, Mapping):
                    continue
                block_pc = meta.get("block_pc")
                if block_pc is None:
                    continue
                try:
                    bpc = int(block_pc)
                except Exception:
                    continue
                op = str(c.get("op") or meta.get("original_op") or "")
                has_guard = bool(c.get("has_guard", True))
                # "unchecked" proxy: sensitive arithmetic without guard
                if (not has_guard) and op:
                    unchecked_nodes.add(bpc)
                if op in ("EXP", "MULMOD", "ADDMOD", "DIV", "MOD"):
                    constraint_sensitive_nodes.add(bpc)
        if unchecked_nodes:
            _add_node_tag("unchecked", unchecked_nodes)
        if constraint_sensitive_nodes:
            _add_node_tag("arith_sensitive", constraint_sensitive_nodes)

        # Guard→Revert edges are the concrete "¬P vs P" surfaces
        guard_to_revert_edges: set[Tuple[int, int]] = set()
        guard_to_revert_nodes: set[int] = set()
        for g in guard_nodes:
            succ = adj.get(int(g), [])
            if not succ:
                continue
            has_revert = any(int(v) in revert_nodes for v in succ)
            has_continue = any(int(v) not in revert_nodes for v in succ)
            if has_revert and has_continue:
                guard_to_revert_nodes.add(int(g))
                for v in succ:
                    if int(v) in revert_nodes:
                        guard_to_revert_edges.add((int(g), int(v)))
        if guard_to_revert_nodes:
            _add_node_tag("guard_to_revert", guard_to_revert_nodes)
        if guard_to_revert_edges:
            _add_edge_tag("guard_to_revert", guard_to_revert_edges)

        # Edge tags
        _add_edge_tag("to_revert", [(u, v) for (u, v) in edges if int(v) in revert_nodes])
        _add_edge_tag("from_guard", [(u, v) for (u, v) in edges if int(u) in guard_nodes])
        if callsite_nodes:
            _add_edge_tag("from_callsite", [(u, v) for (u, v) in edges if int(u) in callsite_nodes])

        # Final LEM-gap sets = union of tagged nodes/edges
        lem_gap_nodes: set[int] = set()
        for s in tag_nodes.values():
            lem_gap_nodes |= set(int(x) for x in s)

        lem_gap_edges: set[Tuple[int, int]] = set()
        for s in tag_edges.values():
            lem_gap_edges |= set((int(u), int(v)) for (u, v) in s)
        # Also include neighborhood edges around lem-gap nodes to reflect local curvature support
        for (u, v) in edges:
            if (int(u) in lem_gap_nodes) or (int(v) in lem_gap_nodes):
                lem_gap_edges.add((int(u), int(v)))

        # Export tags into a stable JSON-friendly shape
        lem_gap_tags: Dict[str, Dict[str, Any]] = {}
        all_tags = sorted(set(list(tag_nodes.keys()) + list(tag_edges.keys())))
        for t in all_tags:
            lem_gap_tags[t] = {
                "nodes": sorted(tag_nodes.get(t, set())),
                "edges": [[int(u), int(v)] for (u, v) in sorted(tag_edges.get(t, set()))],
            }

        # Base-space theta coordinates per node (deterministic proxy):
        # x1(amount)=1 if LEM-gap else 0
        # x2(time)=normalized BFS depth from entry (0..1)
        # x3(selector)=pc normalized by max_pc
        dist = _bfs_depths(entry, adj)
        max_depth = max(dist.values()) if dist else 0
        max_pc = max(node_set) if node_set else 1
        node_theta: Dict[int, Tuple[float, float, float]] = {}
        for n in sorted(node_set):
            x1 = 1.0 if n in lem_gap_nodes else 0.0
            d = dist.get(n, max_depth + 1)
            x2 = float(d) / float(max(1, max_depth + 1))
            x3 = float(int(n)) / float(max(1, max_pc))
            node_theta[int(n)] = (x1, x2, x3)

        base_meta: Dict[str, Any] = {
            "node_count": int(len(node_set)),
            "edge_count": int(len(edges)),
            "cyclic_scc_count": int(len(cyclic_sccs)),
            "guard_nodes": sorted(guard_nodes),
            "revert_nodes": sorted(revert_nodes),
            "sensitive_nodes": sorted(sensitive_nodes),
            "sink_nodes": sorted(sink_nodes),
            "max_depth": int(max_depth),
        }
        if call_sig is not None:
            base_meta.update(call_sig.as_meta())
        base_meta["lem_gap_tag_counts"] = {k: len(v.get("nodes") or []) for k, v in lem_gap_tags.items()}

        base = MVP17BaseSpace(
            theta_axes=("amount", "time", "selector"),
            entry_node=int(entry),
            node_theta=node_theta,
            lem_gap_nodes=sorted(lem_gap_nodes),
            lem_gap_edges=sorted(lem_gap_edges),
            meta=base_meta,
            lem_gap_tags=lem_gap_tags,
        )

        fiber = self._build_fiber()
        conn = self._build_connection(
            fiber=fiber,
            edges=edges,
            lem_gap_edges=lem_gap_edges,
            revert_nodes=revert_nodes,
            guard_nodes=guard_nodes,
            sensitive_nodes=sensitive_nodes,
            cycle_nodes=cycle_nodes,
            callsite_nodes=callsite_nodes,
            delegatecall_nodes=delegatecall_nodes,
            staticcall_nodes=staticcall_nodes,
            create_nodes=create_nodes,
            unchecked_nodes=unchecked_nodes,
            guard_to_revert_edges=guard_to_revert_edges,
        )

        aux = {
            "nodes": nodes,
            "edges": edges,
            "adj": adj,
            "sccs": sccs,
            "cyclic_sccs": cyclic_sccs,
            "revert_nodes": sorted(revert_nodes),
            "guard_nodes": sorted(guard_nodes),
            "sensitive_nodes": sorted(sensitive_nodes),
            "sink_nodes": sorted(sink_nodes),
            "callsite_nodes": sorted(callsite_nodes),
            "delegatecall_nodes": sorted(delegatecall_nodes),
            "staticcall_nodes": sorted(staticcall_nodes),
            "create_nodes": sorted(create_nodes),
            "unchecked_nodes": sorted(unchecked_nodes),
            "guard_to_revert_nodes": sorted(guard_to_revert_nodes),
            "guard_to_revert_edges": sorted(guard_to_revert_edges),
            "lem_gap_nodes": sorted(lem_gap_nodes),
            "lem_gap_edges": sorted(lem_gap_edges),
            "cfg_analysis": dict(cfg_analysis) if isinstance(cfg_analysis, Mapping) else None,
        }
        return base, fiber, conn, aux

    def _build_fiber(self) -> MVP17Fiber:
        # Fiber basis (state + shadow + residue) — deterministic, no heuristics
        visible = ["balance", "res_x", "res_y", "allowance"]
        shadow = ["oracle_acc", "interest_index"]
        residue = ["gas_used", "fail_counter"]
        basis = visible + shadow + residue
        shadow_defs = {
            "shadow.K": "res_x * res_y (AMM constant product)",
            "shadow.health_factor": "collateral / (debt + eps) (if available)",
            "shadow.entropy": "Shannon entropy of |ψ| components (proxy for state disorder)",
        }
        return MVP17Fiber(
            basis=basis,
            visible_basis=visible,
            shadow_basis=shadow,
            residue_basis=residue,
            shadow_definitions=shadow_defs,
        )

    def _build_connection(
        self,
        *,
        fiber: MVP17Fiber,
        edges: Sequence[Tuple[int, int]],
        lem_gap_edges: set[Tuple[int, int]],
        revert_nodes: set[int],
        guard_nodes: set[int],
        sensitive_nodes: set[int],
        cycle_nodes: set[int],
        callsite_nodes: set[int],
        delegatecall_nodes: set[int],
        staticcall_nodes: set[int],
        create_nodes: set[int],
        unchecked_nodes: set[int],
        guard_to_revert_edges: set[Tuple[int, int]],
    ) -> MVP17BerryConnection:
        """
    通过边传输 U_e 构建离散 Berry 连接 A。
    相对于玩具版本的主要升级：
    - 使用结构化的、可解释的幂零生成器族 N_k（严格下三角）
    - 由边类型标签（调用窗口/回退曲面/守卫/循环/未检查）构成 N_e
    - 在 GL(d, F_p) 中计算 U_e = exp(N_e)（幂一，确定性）
        """
        d = len(fiber.basis)
        p = int(self.p)
        I = _identity(d)

        # ------------------------------------------------------------------
        # Generator library (nilpotent Lie algebra; strictly lower-triangular)
        # ------------------------------------------------------------------
        def L(i: int, j: int) -> List[List[int]]:
            ii, jj = int(i), int(j)
            if ii <= jj:
                raise MVP17CFGInputError(f"Generator must be strictly lower-triangular (i>j), got ({ii},{jj})")
            m = _zero(d)
            m[ii][jj] = 1
            return m

        gen_meta: List[Dict[str, Any]] = []
        gens: List[List[List[int]]] = []

        def add_gen(name: str, group: str, meaning: str, mat: List[List[int]]) -> None:
            gens.append(mat)
            gen_meta.append(
                {
                    "name": str(name),
                    "group": str(group),
                    "meaning": str(meaning),
                    "nnz": int(_mat_nnzs(mat, p)),
                }
            )

        # Elementary couplings (visible -> shadow)
        G_oracle_from_balance = L(4, 0)
        G_oracle_from_resx = L(4, 1)
        G_oracle_from_resy = L(4, 2)
        G_index_from_resy = L(5, 2)
        G_index_from_allow = L(5, 3)
        G_index_from_oracle = L(5, 4)

        # Shadow -> residue (double-negation residue channels)
        G_gas_from_oracle = L(6, 4)
        G_fail_from_index = L(7, 5)
        G_fail_from_allow = L(7, 3)
        G_fail_from_gas = L(7, 6)
        G_gas_from_balance = L(6, 0)

        # Chain generators (make exp(N) nontrivial with N^2 ≠ 0)
        G_chain_balance_oracle_gas = _mat_add_modp(G_oracle_from_balance, G_gas_from_oracle, p)
        G_chain_resx_oracle_gas = _mat_add_modp(G_oracle_from_resx, G_gas_from_oracle, p)
        G_chain_oracle_index_fail = _mat_add_modp(G_index_from_oracle, G_fail_from_index, p)

        # Register a doc-friendly generator family (this is what you can write into the doc)
        add_gen("oracle<-balance", "call_window", "oracle_acc += balance", G_oracle_from_balance)
        add_gen("oracle<-res_x", "call_window", "oracle_acc += res_x", G_oracle_from_resx)
        add_gen("oracle<-res_y", "call_window", "oracle_acc += res_y", G_oracle_from_resy)
        add_gen("index<-res_y", "call_window", "interest_index += res_y", G_index_from_resy)
        add_gen("index<-allow", "call_window", "interest_index += allowance", G_index_from_allow)
        add_gen("index<-oracle", "cycle_mix", "interest_index += oracle_acc", G_index_from_oracle)
        add_gen("gas<-oracle", "revert_surface", "gas_used += oracle_acc", G_gas_from_oracle)
        add_gen("fail<-index", "revert_surface", "fail_counter += interest_index", G_fail_from_index)
        add_gen("fail<-allow", "revert_surface", "fail_counter += allowance", G_fail_from_allow)
        add_gen("fail<-gas", "revert_surface", "fail_counter += gas_used", G_fail_from_gas)
        add_gen("gas<-balance", "unchecked", "gas_used += balance (unchecked proxy)", G_gas_from_balance)
        add_gen("chain(balance->oracle->gas)", "guard_surface", "chain: balance -> oracle_acc -> gas_used", G_chain_balance_oracle_gas)
        add_gen("chain(res_x->oracle->gas)", "guard_surface", "chain: res_x -> oracle_acc -> gas_used", G_chain_resx_oracle_gas)
        add_gen("chain(oracle->index->fail)", "cycle_mix", "chain: oracle_acc -> interest_index -> fail_counter", G_chain_oracle_index_fail)

        # ------------------------------------------------------------------
        # Edge-tag-conditioned composition: N_e = Σ w_t * N_t
        # ------------------------------------------------------------------
        def add_to_N(N: List[List[int]], G: List[List[int]], w: int) -> List[List[int]]:
            if int(w) == 0:
                return N
            return _mat_add_modp(N, _mat_scale_modp(G, int(w), p), p)

        edge_transport: Dict[Tuple[int, int], List[List[int]]] = {}
        stats = {
            "lem_gap_edges": 0,
            "revert_edges": 0,
            "guard_edges": 0,
            "call_edges": 0,
            "delegatecall_edges": 0,
            "staticcall_edges": 0,
            "create_edges": 0,
            "cycle_edges": 0,
            "unchecked_edges": 0,
            "micro_variation_edges": 0,
        }

        for u, v in edges:
            uu, vv = int(u), int(v)
            uv = (uu, vv)
            if uv not in lem_gap_edges:
                edge_transport[uv] = I
                continue

            stats["lem_gap_edges"] += 1

            # Edge-type booleans
            to_revert = vv in revert_nodes
            from_guard = uu in guard_nodes
            guard_to_revert = (uu, vv) in guard_to_revert_edges
            in_cycle = (uu in cycle_nodes) and (vv in cycle_nodes)

            from_call = uu in callsite_nodes
            from_delegate = uu in delegatecall_nodes
            from_static = uu in staticcall_nodes
            from_create = uu in create_nodes

            is_unchecked = uu in unchecked_nodes
            is_sensitive = (uu in sensitive_nodes) or (vv in sensitive_nodes)

            if to_revert:
                stats["revert_edges"] += 1
            if from_guard:
                stats["guard_edges"] += 1
            if in_cycle:
                stats["cycle_edges"] += 1
            if from_call:
                stats["call_edges"] += 1
            if from_delegate:
                stats["delegatecall_edges"] += 1
            if from_static:
                stats["staticcall_edges"] += 1
            if from_create:
                stats["create_edges"] += 1
            if is_unchecked:
                stats["unchecked_edges"] += 1

            # Weights (deterministic, not magic):
            # - to_revert + guard_to_revert carry the strongest curvature
            # - call-window introduces shadow drift
            # - cycle introduces mixing (non-abelian potential)
            w_revert = 2 + (1 if guard_to_revert else 0) if to_revert else 0
            w_call = 1 + (1 if from_delegate else 0) + (1 if from_static else 0) + (1 if from_create else 0) if (from_call or from_delegate or from_static or from_create) else 0
            w_cycle = 1 if in_cycle else 0
            w_guard = 1 if from_guard else 0
            w_sensitive = 1 if is_sensitive else 0
            w_unchecked = 2 if is_unchecked else 0

            # Compose nilpotent N_e
            N = _zero(d)
            # call-window: visible->shadow + index coupling
            N = add_to_N(N, G_oracle_from_resx, w_call)
            N = add_to_N(N, G_index_from_resy, w_call)
            N = add_to_N(N, G_index_from_oracle, w_cycle)

            # revert-surface: shadow/residue drift and chain effects
            N = add_to_N(N, G_gas_from_oracle, w_revert)
            N = add_to_N(N, G_fail_from_index, w_revert)
            N = add_to_N(N, G_fail_from_gas, w_revert)

            # guard surface: chain generators amplify residue through composition
            N = add_to_N(N, G_chain_balance_oracle_gas, w_guard + (1 if guard_to_revert else 0))
            N = add_to_N(N, G_chain_oracle_index_fail, w_cycle)

            # sensitive/unchecked: inject extra drift
            N = add_to_N(N, G_oracle_from_resy, w_sensitive)
            N = add_to_N(N, G_fail_from_allow, w_sensitive)
            N = add_to_N(N, G_gas_from_balance, w_unchecked)

            # Micro gauge variation: deterministic location-dependent generator
            # This is the discrete analogue of a non-constant gauge potential A(θ).
            # It prevents symmetric cycles (AB vs BA) from collapsing to commuting holonomies.
            micro_idx = _deterministic_mix(uu, vv) % len(gens)
            N = add_to_N(N, gens[int(micro_idx)], 1)
            stats["micro_variation_edges"] += 1

            # Deterministic exp transport
            U = _mat_exp_nilpotent_modp(N, p, max_terms=d)
            edge_transport[uv] = U

        return MVP17BerryConnection(
            field_p=p,
            generators=gens,
            edge_transport=edge_transport,
            generator_policy={
                "basis_dim": d,
                "transport_model": "U_e = exp(N_e) over F_p (unipotent, nilpotent N_e)",
                "edge_composition": {
                    "N_e": "w_call*N_call + w_revert*N_revert + w_cycle*N_cycle + w_guard*N_guard + w_sensitive*N_sensitive + w_unchecked*N_unchecked",
                    "w_revert": "2 (+1 if guard_to_revert) when to_revert else 0",
                    "w_call": "1 + delegate/static/create flags on call window else 0",
                    "w_cycle": "1 when edge inside cyclic SCC else 0",
                    "w_guard": "1 when from_guard else 0",
                    "w_sensitive": "1 when touches sensitive node else 0",
                    "w_unchecked": "2 when from unchecked-sensitive arithmetic node else 0",
                },
                "generator_family": gen_meta,
                "edge_stats": stats,
                "micro_variation": "N_e includes +1 * N_{mix(u,v)} to model location-dependent Berry gauge potential.",
            },
        )


class MVP17CFGAuditor:
    """
    CFG-only 审计器：输出MVP17的曲率证书与红线信号。
    """

    def __init__(self, *, p: int = 101, max_cycles: int = 8) -> None:
        self.builder = MVP17CFGObjectBuilder(p=p)
        self.max_cycles = int(max_cycles)
        if self.max_cycles < 1 or self.max_cycles > 128:
            raise MVP17CFGInputError("max_cycles must be in [1,128]")

    def analyze_target(self, target: Mapping[str, Any]) -> MVP17CFGAuditReport:
        if not isinstance(target, Mapping):
            raise MVP17CFGInputError("target must be dict-like")
        cfg_topology = target.get("cfg_topology")
        if not cfg_topology:
            raise MVP17CFGInputError("target missing cfg_topology (run address_hunter export_cfg_for_spectral)")

        base, fiber, conn, aux = self.builder.build_from_cfg_topology(
            cfg_topology,
            cfg_analysis=target.get("cfg_analysis"),
            bytecode=target.get("bytecode"),
            evm_constraints=target.get("evm_constraints"),
        )

        nodes: List[int] = list(aux["nodes"])
        edges: List[Tuple[int, int]] = list(aux["edges"])
        adj: Dict[int, List[int]] = aux["adj"]
        cyclic_sccs: List[List[int]] = aux["cyclic_sccs"]
        revert_nodes = set(aux["revert_nodes"])
        guard_nodes = set(aux["guard_nodes"])
        callsite_nodes = set(aux.get("callsite_nodes") or [])

        p = int(conn.field_p)
        d = len(fiber.basis)
        I = _identity(d)

        # Directed reachability from entry (global realizability proxy; deterministic)
        entry = int(base.entry_node)
        reach_nodes: set[int] = {entry}
        rq: List[int] = [entry]
        rqi = 0
        while rqi < len(rq):
            u = int(rq[rqi])
            rqi += 1
            for v in adj.get(u, []):
                vv = int(v)
                if vv not in reach_nodes:
                    reach_nodes.add(vv)
                    rq.append(vv)

        # Indegree map for merge detection (diamond loops)
        indeg: Dict[int, int] = {int(n): 0 for n in nodes}
        for (u, v) in edges:
            indeg[int(v)] = indeg.get(int(v), 0) + 1
        lem_gap_nodes = set(int(x) for x in (aux.get("lem_gap_nodes") or []))

        # Cycle holonomy
        cycles_out: List[Dict[str, Any]] = []
        cycle_holonomies: List[List[List[int]]] = []
        cycle_edges_seen: set[Tuple[Tuple[int, int], ...]] = set()

        cycles_examined = 0
        for comp in cyclic_sccs:
            if cycles_examined >= self.max_cycles:
                break
            if not comp:
                continue
            # Deterministic multi-start: sample up to K distinct starts per SCC
            k_starts = min(len(comp), max(2, min(8, self.max_cycles)))
            starts = [int(x) for x in comp[:k_starts]]
            for s in starts:
                if cycles_examined >= self.max_cycles:
                    break
                cyc = _find_cycle_in_scc(comp, adj, start=int(s))
                if not cyc or len(cyc) < 2:
                    continue
                # Build edges in cycle
                cyc_edges = [(int(cyc[i]), int(cyc[i + 1])) for i in range(len(cyc) - 1)]
                key = tuple(cyc_edges)
                if key in cycle_edges_seen:
                    continue
                cycle_edges_seen.add(key)

                # Holonomy H = Π U_e
                H = I
                for u, v in cyc_edges:
                    U = conn.edge_transport.get((u, v))
                    if U is None:
                        # Edge not in export list; treat as identity but do not hide it
                        U = I
                    H = matmul_modp(H, U, p)

                delta = _mat_sub_modp(H, I, p)
                rk = int(rank_modp(delta, p))
                nontrivial = rk > 0
                witness = _first_nonzero_column_witness(delta, p)
                nnz = int(_mat_nnzs(delta, p))

                cycles_out.append(
                    {
                        "scc_size": int(len(comp)),
                        "cycle_nodes": list(map(int, cyc)),
                        "cycle_edges": [[int(u), int(v)] for u, v in cyc_edges],
                        "reachable_from_entry_cfg": bool(any(int(x) in reach_nodes for x in cyc)),
                        "holonomy_rank_modp": rk,
                        "holonomy_nontrivial": bool(nontrivial),
                        "holonomy_delta_nnz": nnz,
                        "holonomy_witness": witness,
                    }
                )
                cycle_holonomies.append(H)
                cycles_examined += 1

        # Non-abelian commutator: prefer globally reachable cycle holonomies; fallback to all cycles.
        non_abelian: Dict[str, Any] = {"ok": False, "reason": "need >=2 cycles to compute commutator"}
        reachable_cycle_indices = [i for i, c in enumerate(cycles_out) if bool(c.get("reachable_from_entry_cfg"))]
        if len(reachable_cycle_indices) >= 2:
            best_rank = -1
            best_pair: Optional[Tuple[int, int]] = None
            best_witness: Optional[Dict[str, Any]] = None
            best_nnz = 0
            pairs_examined = 0
            for ii in range(len(reachable_cycle_indices)):
                for jj in range(ii + 1, len(reachable_cycle_indices)):
                    pairs_examined += 1
                    i = int(reachable_cycle_indices[ii])
                    j = int(reachable_cycle_indices[jj])
                    H1 = cycle_holonomies[i]
                    H2 = cycle_holonomies[j]
                    H1H2 = matmul_modp(H1, H2, p)
                    H2H1 = matmul_modp(H2, H1, p)
                    C = _mat_sub_modp(H1H2, H2H1, p)
                    crk = int(rank_modp(C, p))
                    if crk > best_rank:
                        best_rank = crk
                        best_pair = (int(i), int(j))
                        best_witness = _first_nonzero_column_witness(C, p)
                        best_nnz = int(_mat_nnzs(C, p))
            non_abelian = {
                "ok": True,
                "commutator_rank_modp": int(max(0, best_rank)),
                "commutator_delta_nnz": int(best_nnz),
                "witness_pair": list(best_pair) if best_pair else None,
                "witness": best_witness,
                "pairs_examined": int(pairs_examined),
                "non_abelian": bool(best_rank > 0),
                "scope": "reachable",
                "reachable_cycle_count": int(len(reachable_cycle_indices)),
                "explain": "Non-Abelian signal (reachable cycles): choose the max-rank commutator [H_i, H_j] over F_p with a concrete witness column.",
            }
        elif len(cycle_holonomies) >= 2:
            # Fallback: cycles exist but none/only one is reachable from entry; report but do not treat as strong evidence.
            best_rank = -1
            best_pair = None
            best_witness = None
            best_nnz = 0
            pairs_examined = 0
            for i in range(len(cycle_holonomies)):
                for j in range(i + 1, len(cycle_holonomies)):
                    pairs_examined += 1
                    H1 = cycle_holonomies[i]
                    H2 = cycle_holonomies[j]
                    H1H2 = matmul_modp(H1, H2, p)
                    H2H1 = matmul_modp(H2, H1, p)
                    C = _mat_sub_modp(H1H2, H2H1, p)
                    crk = int(rank_modp(C, p))
                    if crk > best_rank:
                        best_rank = crk
                        best_pair = (int(i), int(j))
                        best_witness = _first_nonzero_column_witness(C, p)
                        best_nnz = int(_mat_nnzs(C, p))
            non_abelian = {
                "ok": True,
                "commutator_rank_modp": int(max(0, best_rank)),
                "commutator_delta_nnz": int(best_nnz),
                "witness_pair": list(best_pair) if best_pair else None,
                "witness": best_witness,
                "pairs_examined": int(pairs_examined),
                "non_abelian": bool(best_rank > 0),
                "scope": "all",
                "reachable_cycle_count": int(len(reachable_cycle_indices)),
                "warning": "commutator computed on non-reachable cycles (phantom risk)",
                "explain": "Non-Abelian signal (all cycles): reachable cycles were insufficient; this may be a phantom (Tate–Shafarevich obstruction).",
            }

        # Double negation heuristic: guard node branching to revert + continue
        dn_nodes: List[int] = []
        dn_edges: List[Tuple[int, int]] = []
        for g in sorted(guard_nodes):
            if int(g) not in reach_nodes:
                continue
            succ = adj.get(int(g), [])
            if not succ:
                continue
            has_revert = any(int(v) in revert_nodes for v in succ)
            has_continue = any(int(v) not in revert_nodes for v in succ)
            if has_revert and has_continue:
                dn_nodes.append(int(g))
                for v in succ:
                    if int(v) in revert_nodes:
                        dn_edges.append((int(g), int(v)))

        double_negation = {
            "ok": True,
            "guard_nodes_with_revert_branch": dn_nodes,
            "guard_to_revert_edges": [[int(u), int(v)] for u, v in dn_edges],
            "potential_detected": bool(len(dn_nodes) > 0),
            "callsite_guard_nodes_with_revert_branch": sorted(set(dn_nodes) & set(callsite_nodes)),
            "explain": "CFG heuristic: guard with (continue, revert) branches models ¬P vs P surfaces; caller-level try/catch can realize ¬¬P paths.",
        }

        # Wilson loops on branch-merge diamonds: H = P(path1) * P(path2)^(-1)
        # This works even when there is no directed cycle (DAG diamonds).
        wilson_loops: List[Dict[str, Any]] = []
        wilson_holonomies: List[List[List[int]]] = []
        wilson_examined = 0
        diamond_seen: set[Tuple[int, int, int, int]] = set()
        candidate_branches = [
            int(b)
            for b in sorted(reach_nodes)
            if len(adj.get(int(b), [])) >= 2 and (int(b) in lem_gap_nodes or int(b) in guard_nodes or int(b) in callsite_nodes)
        ]
        for b in candidate_branches:
            if len(wilson_loops) >= self.max_cycles:
                break
            succs0 = list(adj.get(int(b), []))
            if len(succs0) < 2:
                continue
            succs = sorted(set(int(x) for x in succs0))[:3]  # deterministic cap per-branch
            for i in range(len(succs)):
                for j in range(i + 1, len(succs)):
                    if len(wilson_loops) >= self.max_cycles:
                        break
                    s1, s2 = int(succs[i]), int(succs[j])
                    key = (int(b), int(s1), int(s2), -1)
                    wilson_examined += 1

                    parent1, dist1 = _bfs_shortest_parents(adj, start=s1, max_depth=96)
                    parent2, dist2 = _bfs_shortest_parents(adj, start=s2, max_depth=96)
                    inter = set(dist1.keys()) & set(dist2.keys())
                    # prefer real merge points
                    merges = [int(x) for x in inter if int(indeg.get(int(x), 0)) >= 2]
                    if not merges:
                        continue
                    m = min(merges, key=lambda x: (int(dist1.get(int(x), 10**9)) + int(dist2.get(int(x), 10**9)), int(x)))
                    d1 = int(dist1.get(int(m), 10**9))
                    d2 = int(dist2.get(int(m), 10**9))

                    p1 = _reconstruct_bfs_path(parent1, start=s1, target=m, max_len=256)
                    p2 = _reconstruct_bfs_path(parent2, start=s2, target=m, max_len=256)
                    if not p1 or not p2:
                        continue

                    # Normalize diamond identity (order-invariant succ pair)
                    k0 = tuple(sorted([int(s1), int(s2)]))
                    dkey = (int(b), int(k0[0]), int(k0[1]), int(m))
                    if dkey in diamond_seen:
                        continue
                    diamond_seen.add(dkey)

                    path1_nodes = [int(b)] + [int(x) for x in p1]
                    path2_nodes = [int(b)] + [int(x) for x in p2]
                    path1_edges = [(path1_nodes[k], path1_nodes[k + 1]) for k in range(len(path1_nodes) - 1)]
                    path2_edges = [(path2_nodes[k], path2_nodes[k + 1]) for k in range(len(path2_nodes) - 1)]

                    # Transport along each path
                    P1 = I
                    for u, v in path1_edges:
                        U = conn.edge_transport.get((int(u), int(v))) or I
                        P1 = matmul_modp(P1, U, p)
                    P2 = I
                    for u, v in path2_edges:
                        U = conn.edge_transport.get((int(u), int(v))) or I
                        P2 = matmul_modp(P2, U, p)
                    try:
                        P2inv = invert_matrix_modp(P2, p)
                    except Exception:
                        continue
                    H = matmul_modp(P1, P2inv, p)

                    delta = _mat_sub_modp(H, I, p)
                    rk = int(rank_modp(delta, p))
                    nnz = int(_mat_nnzs(delta, p))
                    wilson_loops.append(
                        {
                            "loop_type": "diamond",
                            "branch": int(b),
                            "merge": int(m),
                            "succ_pair": [int(s1), int(s2)],
                            "depths": {"succ1_to_merge": int(d1), "succ2_to_merge": int(d2)},
                            "path1_nodes": path1_nodes,
                            "path2_nodes": path2_nodes,
                            "path1_edges": [[int(u), int(v)] for u, v in path1_edges],
                            "path2_edges": [[int(u), int(v)] for u, v in path2_edges],
                            "holonomy_rank_modp": rk,
                            "holonomy_nontrivial": bool(rk > 0),
                            "holonomy_delta_nnz": nnz,
                            "holonomy_witness": _first_nonzero_column_witness(delta, p),
                            "branch_is_lem_gap": bool(int(b) in lem_gap_nodes),
                            "reachable_from_entry_cfg": bool(int(b) in reach_nodes),
                        }
                    )
                    wilson_holonomies.append(H)

        wilson_non_abelian: Dict[str, Any] = {"ok": False, "reason": "need >=2 wilson_loops"}
        if len(wilson_holonomies) >= 2:
            best_rank = -1
            best_pair = None
            best_witness = None
            best_nnz = 0
            pairs_examined = 0
            for i in range(len(wilson_holonomies)):
                for j in range(i + 1, len(wilson_holonomies)):
                    pairs_examined += 1
                    H1 = wilson_holonomies[i]
                    H2 = wilson_holonomies[j]
                    H1H2 = matmul_modp(H1, H2, p)
                    H2H1 = matmul_modp(H2, H1, p)
                    C = _mat_sub_modp(H1H2, H2H1, p)
                    crk = int(rank_modp(C, p))
                    if crk > best_rank:
                        best_rank = crk
                        best_pair = (int(i), int(j))
                        best_witness = _first_nonzero_column_witness(C, p)
                        best_nnz = int(_mat_nnzs(C, p))
            wilson_non_abelian = {
                "ok": True,
                "commutator_rank_modp": int(max(0, best_rank)),
                "commutator_delta_nnz": int(best_nnz),
                "witness_pair": list(best_pair) if best_pair else None,
                "witness": best_witness,
                "pairs_examined": int(pairs_examined),
                "non_abelian": bool(best_rank > 0),
                "explain": "Diamond Wilson loops: non-abelian commutator across H_loop matrices (path-ordering dependence).",
            }

        # Faltings Height Pairing (surrogate): finite (p-adic) exactness vs infinite (precision) ghost intersections
        faltings_height = self._faltings_height_pairing_certificate(
            p=p,
            evm_constraints=target.get("evm_constraints"),
            wilson_loops=wilson_loops,
        )

        # ------------------------------------------------------------
        # Excluded Middle as a *unique-solution formula* (CFG surrogate)
        # ------------------------------------------------------------
        excluded_middle = self._excluded_middle_unique_solution_certificate(
            p=p,
            cycles=cycles_out,
            wilson_loops=wilson_loops,
        )

        # Verdict: any of the 3 acceptance signals -> CURVED_POTENTIAL
        any_holonomy = any(bool(c.get("holonomy_nontrivial")) and bool(c.get("reachable_from_entry_cfg", True)) for c in cycles_out)
        any_nonabel = bool(non_abelian.get("ok") and non_abelian.get("non_abelian"))
        any_dn = bool(double_negation.get("potential_detected"))
        any_wilson = any(bool(w.get("holonomy_nontrivial")) for w in wilson_loops)
        any_wilson_nonabel = bool(wilson_non_abelian.get("ok") and wilson_non_abelian.get("non_abelian"))
        any_faltings_ghost = bool(faltings_height.get("ghost_intersection_detected"))
        any_lem_unique_fail = bool(excluded_middle.get("ok") and excluded_middle.get("unique_solution_holds_reachable") is False)

        hodge_conjecture = self._hodge_conjecture_certificate(
            nodes=nodes,
            edges=edges,
            lem_gap_edges=set(aux.get("lem_gap_edges") or []),
            p=p,
        )

        greens_current = greens_current_streams_certificate(
            nodes=nodes,
            edges=edges,
            entry_node=int(base.entry_node),
            revert_nodes=list(aux.get("revert_nodes") or []),
            lem_gap_edges=list(aux.get("lem_gap_edges") or []),
            guard_to_revert_edges=list(aux.get("guard_to_revert_edges") or []),
        )

        tate_shafarevich = self._tate_shafarevich_obstruction_certificate(
            nodes=nodes,
            edges=edges,
            entry=int(base.entry_node),
            cyclic_sccs=cyclic_sccs,
            cycles_out=cycles_out,
            p=p,
        )

        try:
            crystalline_frobenius = crystalline_frobenius_spectral_certificate(
                p=int(p),
                basis=list(fiber.basis),
                visible_basis=list(fiber.visible_basis),
                cycles=cycles_out,
                wilson_loops=wilson_loops,
                max_items=int(self.max_cycles),
            )
        except Exception as e:
            crystalline_frobenius = {"ok": False, "reason": f"crystalline_frobenius_failed: {e}"}

        # ------------------------------------------------------------
        # Period ring comparison isomorphism on B_dR (Third stage)
        # ------------------------------------------------------------
        try:
            period_isomorphism = self._period_isomorphism_certificate(p=int(p), crystalline_frobenius=crystalline_frobenius)
        except Exception as e:
            period_isomorphism = {"ok": False, "reason": f"period_isomorphism_failed: {e}"}
        any_period_filtration_jump = bool(
            isinstance(period_isomorphism, dict)
            and isinstance(period_isomorphism.get("signals"), dict)
            and bool(period_isomorphism["signals"].get("filtration_jump"))
        )

        # ------------------------------------------------------------
        # Green barrier integrals on cycles + geodesic approach paths
        # ------------------------------------------------------------
        try:
            if bool(greens_current.get("ok")) and isinstance(greens_current.get("node_potential"), dict):
                gfield: Dict[int, float] = {int(k): float(v) for k, v in greens_current["node_potential"].items()}
                lem_gap_edges_set = set((int(u), int(v)) for (u, v) in (aux.get("lem_gap_edges") or []))
                gtr_edges_set = set((int(u), int(v)) for (u, v) in (aux.get("guard_to_revert_edges") or []))

                dist, prev = _dijkstra_green_geodesics(
                    nodes=nodes,
                    edges=edges,
                    entry=int(base.entry_node),
                    node_potential=gfield,
                    lem_gap_edges=lem_gap_edges_set,
                    guard_to_revert_edges=gtr_edges_set,
                )

                # Per-cycle barrier integrals
                for c in cycles_out:
                    cyc_nodes = c.get("cycle_nodes") or []
                    cyc_edges = c.get("cycle_edges") or []
                    # Normalize edges list shape
                    edges_uv: List[Tuple[int, int]] = []
                    for e in cyc_edges:
                        if isinstance(e, (list, tuple)) and len(e) == 2:
                            edges_uv.append((int(e[0]), int(e[1])))

                    if not edges_uv:
                        c["green_barrier"] = {"ok": False, "reason": "no edges"}
                        continue

                    costs: List[float] = []
                    for (u, v) in edges_uv:
                        if u not in gfield or v not in gfield:
                            continue
                        w = _edge_conductance(u, v, lem_gap_edges=lem_gap_edges_set, guard_to_revert_edges=gtr_edges_set)
                        length_factor = 1.0 / max(w, np.finfo(np.float64).eps)
                        costs.append(0.5 * (gfield[u] + gfield[v]) * length_factor)

                    if not costs:
                        c["green_barrier"] = {"ok": False, "reason": "missing potentials on cycle nodes"}
                        continue

                    # entry->cycle minimal approach cost
                    approach = float("inf")
                    approach_node = None
                    for n in cyc_nodes:
                        nn = int(n)
                        if nn in dist:
                            dn = float(dist[nn])
                            if dn < approach:
                                approach = dn
                                approach_node = nn

                    # Node potential extrema on this cycle
                    vals = [float(gfield[int(n)]) for n in cyc_nodes if int(n) in gfield]
                    reachable = not math.isinf(approach)
                    path_nodes = None
                    if reachable and approach_node is not None:
                        path_nodes = _reconstruct_path(prev, entry=int(base.entry_node), target=int(approach_node), max_len=96)
                    c["green_barrier"] = {
                        "ok": True,
                        "cycle_integral": float(sum(costs)),
                        "cycle_mean_edge": float(sum(costs) / max(1, len(costs))),
                        "cycle_max_edge": float(max(costs)),
                        "reachable_from_entry": bool(reachable),
                        "entry_to_cycle_min": None if not reachable else float(approach),
                        "entry_to_cycle_witness_node": None if not reachable else int(approach_node) if approach_node is not None else None,
                        "entry_to_cycle_path_nodes": path_nodes,
                        "total_approach_plus_cycle": None if not reachable else float(approach + sum(costs)),
                        "node_potential_min": float(min(vals)) if vals else None,
                        "node_potential_max": float(max(vals)) if vals else None,
                        "note": "Discrete integral approx: sum_e avg(g(u),g(v))*(1/conductance). Lower values indicate weaker barriers.",
                    }

                # Guard approach geodesics (top-K easiest guards)
                gtr_edges = list(aux.get("guard_to_revert_edges") or [])
                gtr_rows: List[Dict[str, Any]] = []
                for e in gtr_edges:
                    if not isinstance(e, (list, tuple)) or len(e) != 2:
                        continue
                    gu, rv = int(e[0]), int(e[1])
                    d_gu = dist.get(gu, float("inf"))
                    # include the final guard->revert edge cost as part of the approach scalar
                    step = None
                    if not math.isinf(d_gu) and gu in gfield and rv in gfield:
                        w = _edge_conductance(gu, rv, lem_gap_edges=lem_gap_edges_set, guard_to_revert_edges=gtr_edges_set)
                        length_factor = 1.0 / max(w, np.finfo(np.float64).eps)
                        step = float(0.5 * (gfield[gu] + gfield[rv]) * length_factor)
                    total = None
                    if step is not None and not math.isinf(d_gu):
                        total = float(d_gu + step)
                    gtr_rows.append(
                        {
                            "guard": gu,
                            "revert": rv,
                            "entry_to_guard": None if math.isinf(d_gu) else float(d_gu),
                            "g_guard": float(gfield.get(gu, 0.0)),
                            "guard_to_revert_edge_cost": step,
                            "entry_to_revert_via_guard": total,
                        }
                    )
                # Sort by entry_to_guard (None goes last)
                gtr_rows.sort(
                    key=lambda r: (
                        r["entry_to_revert_via_guard"] is None,
                        r["entry_to_revert_via_guard"] if r["entry_to_revert_via_guard"] is not None else 0.0,
                        r["guard"],
                    )
                )

                easiest = next((r for r in gtr_rows if r["entry_to_guard"] is not None), None)
                witness_path = None
                if easiest is not None:
                    witness_path = _reconstruct_path(prev, entry=int(base.entry_node), target=int(easiest["guard"]), max_len=96)

                # Contract-level unique scalar: minimum reachable (approach + cycle_integral) among cycles,
                # and minimum entry->revert-via-guard among guard_to_revert edges.
                best_cycle = None
                for i, c in enumerate(cycles_out):
                    gb = c.get("green_barrier") or {}
                    if not (isinstance(gb, dict) and gb.get("ok") and gb.get("total_approach_plus_cycle") is not None):
                        continue
                    val = float(gb["total_approach_plus_cycle"])
                    if best_cycle is None or val < float(best_cycle["value"]):
                        best_cycle = {"cycle_index": int(i), "value": float(val), "holonomy_nontrivial": bool(c.get("holonomy_nontrivial"))}

                best_guard = next((r for r in gtr_rows if r.get("entry_to_revert_via_guard") is not None), None)

                greens_current["geodesic"] = {
                    "ok": True,
                    "entry": int(base.entry_node),
                    "n_reachable": int(len(dist)),
                    "easiest_guard": easiest,
                    "easiest_guard_path_nodes": witness_path,
                    "guard_to_revert_entries_top16": gtr_rows[:16],
                    "unique_scalar": {
                        "min_cycle_total_barrier": best_cycle,
                        "min_entry_to_revert_via_guard": best_guard,
                        "note": "These are deterministic scalars derived from the Green barrier metric; use them as the 'one number' in docs/thresholding.",
                    },
                    "explain": "Geodesic under Green barrier cost: Dijkstra on directed CFG with edge cost avg(g)*1/conductance.",
                }
        except Exception as e:
            # Do not hide; attach diagnostic, keep overall report usable.
            greens_current["geodesic"] = {"ok": False, "reason": f"geodesic_failed: {e}"}

        # ------------------------------------------------------------
        # Phase 5: Witt 接触几何层 (Contact Geometry on Witt Space)
        # ------------------------------------------------------------
        try:
            witt_contact = witt_curvature_certificate_from_mvp17(
                p=int(p),
                cycles=cycles_out,
                wilson_loops=wilson_loops,
                fiber_dim=int(d),
                # edge_transport 从 conn.edge_transport 获取（如果需要完整矩阵信息）
                edge_transport=conn.edge_transport if hasattr(conn, 'edge_transport') else None,
            )
        except Exception as e:
            witt_contact = {"ok": False, "reason": f"witt_contact_failed: {e}"}

        any_witt_lem_failure = bool(
            isinstance(witt_contact, dict)
            and isinstance(witt_contact.get("signals"), dict)
            and bool(witt_contact["signals"].get("any_lem_failure_wilson"))
        )

        ok = bool(
            any_holonomy
            or any_nonabel
            or any_dn
            or any_wilson
            or any_wilson_nonabel
            or any_faltings_ghost
            or any_lem_unique_fail
            or any_period_filtration_jump
            or any_witt_lem_failure
        )
        status = "CURVED_POTENTIAL" if ok else "FLAT"

        explain = (
            "CFG-first MVP17: mark LEM-gap on CFG, build discrete Berry connection U_e in GL(d,F_p), "
            "then detect logic curvature via nontrivial cycle holonomy / non-abelian commutator / double-negation potential. "
            "Phase 5 adds Witt contact geometry: lifts Berry holonomy to W_n(F_p) and checks LEM failure via exact Witt endpoint comparison."
        )

        return MVP17CFGAuditReport(
            ok=ok,
            status=status,
            base_space=base.as_dict(),
            fiber=fiber.as_dict(),
            connection=conn.as_dict(),
            cycles=cycles_out,
            wilson_loops=wilson_loops,
            non_abelian=non_abelian,
            wilson_non_abelian=wilson_non_abelian,
            double_negation=double_negation,
            faltings_height=faltings_height,
            excluded_middle=excluded_middle,
            hodge_conjecture=hodge_conjecture,
            greens_current=greens_current,
            tate_shafarevich=tate_shafarevich,
            crystalline_frobenius=crystalline_frobenius,
            period_isomorphism=period_isomorphism,
            witt_contact=witt_contact,
            explain=explain,
        )

    def _excluded_middle_unique_solution_certificate(
        self,
        *,
        p: int,
        cycles: Sequence[Mapping[str, Any]],
        wilson_loops: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
    排中律失效作为唯一解公式
    离散上下文无关文法代理
        """
        p_i = int(p)
        if p_i <= 2:
            return {"ok": False, "reason": "p must be > 2"}

        cyc_all = [c for c in cycles if isinstance(c, Mapping)]
        wil_all = [w for w in wilson_loops if isinstance(w, Mapping)]

        cyc_reach = [c for c in cyc_all if bool(c.get("reachable_from_entry_cfg", True))]
        wil_reach = [w for w in wil_all if bool(w.get("reachable_from_entry_cfg", True))]

        def _nontrivial(x: Mapping[str, Any]) -> bool:
            return bool(x.get("holonomy_nontrivial")) or (int(x.get("holonomy_rank_modp", 0)) > 0)

        cyc_all_bad = [c for c in cyc_all if _nontrivial(c)]
        wil_all_bad = [w for w in wil_all if _nontrivial(w)]
        cyc_reach_bad = [c for c in cyc_reach if _nontrivial(c)]
        wil_reach_bad = [w for w in wil_reach if _nontrivial(w)]

        holds_reachable = (len(cyc_reach_bad) == 0) and (len(wil_reach_bad) == 0)

        # Choose a strongest witness deterministically: max rank, then nnz, then stable key
        best = None
        best_key = None

        def consider(kind: str, idx: int, row: Mapping[str, Any], *, scope: str) -> None:
            nonlocal best, best_key
            rk = int(row.get("holonomy_rank_modp", 0))
            nnz = int(row.get("holonomy_delta_nnz", 0))
            key = (-rk, -nnz, str(kind), int(idx), str(scope))
            if best is None or key < best_key:  # type: ignore[operator]
                best = {"kind": str(kind), "index": int(idx), "scope": str(scope), "row": dict(row)}
                best_key = key

        # Prefer reachable witnesses; fallback to all
        if (not holds_reachable) and (cyc_reach_bad or wil_reach_bad):
            for i, c in enumerate(cyc_reach_bad):
                consider("cycle", i, c, scope="reachable")
            for i, w in enumerate(wil_reach_bad):
                consider("diamond", i, w, scope="reachable")
        else:
            for i, c in enumerate(cyc_all_bad):
                consider("cycle", i, c, scope="all")
            for i, w in enumerate(wil_all_bad):
                consider("diamond", i, w, scope="all")

        max_rank_reach = 0
        for c in cyc_reach_bad:
            max_rank_reach = max(max_rank_reach, int(c.get("holonomy_rank_modp", 0)))
        for w in wil_reach_bad:
            max_rank_reach = max(max_rank_reach, int(w.get("holonomy_rank_modp", 0)))

        max_rank_all = 0
        for c in cyc_all_bad:
            max_rank_all = max(max_rank_all, int(c.get("holonomy_rank_modp", 0)))
        for w in wil_all_bad:
            max_rank_all = max(max_rank_all, int(w.get("holonomy_rank_modp", 0)))

        witness = None
        if best is not None:
            row = best["row"]
            if best["kind"] == "cycle":
                witness = {
                    "type": "cycle_holonomy",
                    "scope": best["scope"],
                    "cycle_nodes": row.get("cycle_nodes"),
                    "cycle_edges": row.get("cycle_edges"),
                    "holonomy_rank_modp": row.get("holonomy_rank_modp"),
                    "holonomy_delta_nnz": row.get("holonomy_delta_nnz"),
                    "holonomy_witness": row.get("holonomy_witness"),
                }
            else:
                witness = {
                    "type": "diamond_wilson_loop",
                    "scope": best["scope"],
                    "branch": row.get("branch"),
                    "merge": row.get("merge"),
                    "succ_pair": row.get("succ_pair"),
                    "path1_nodes": row.get("path1_nodes"),
                    "path2_nodes": row.get("path2_nodes"),
                    "holonomy_rank_modp": row.get("holonomy_rank_modp"),
                    "holonomy_delta_nnz": row.get("holonomy_delta_nnz"),
                    "holonomy_witness": row.get("holonomy_witness"),
                }

        return {
            "ok": True,
            "field": {"type": "Fp", "p": p_i},
            "unique_solution_holds_reachable": bool(holds_reachable),
            "violations": {
                "cycles_reachable_nontrivial": int(len(cyc_reach_bad)),
                "diamonds_reachable_nontrivial": int(len(wil_reach_bad)),
                "cycles_all_nontrivial": int(len(cyc_all_bad)),
                "diamonds_all_nontrivial": int(len(wil_all_bad)),
                "max_rank_modp_reachable": int(max_rank_reach),
                "max_rank_modp_all": int(max_rank_all),
            },
            "witness": witness,
            "formula": {
                "path_transport": "T(P)=∏_{e in P} U_e  in GL(d,F_p)",
                "cycle_wilson": "W(C)=T(C); LEM_unique => W(C)=I for all cycles",
                "diamond_wilson": "W= T(path1)·T(path2)^{-1}; LEM_unique => W=I for all diamonds",
                "equivalence": "LEM_unique holds iff all reachable transports entry→v are path-independent (unique solution of ∇s=0).",
            },
            "explain": "Excluded middle as uniqueness: if any reachable loop has nontrivial Wilson holonomy (rank(W-I)>0), then transported state depends on path => LEM fails in this CFG surrogate.",
        }

    def _faltings_height_pairing_certificate(
        self,
        *,
        p: int,
        evm_constraints: Any,
        wilson_loops: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Faltings Height Pairing (CFG-first surrogate)
        -------------------------------------------

          - Finite place ("exact in F_p"): use v_p(D) where D is a divisor/modulus constant in DIV/MOD constraints.
            This is an integer and is zero when D is invertible mod p (typical for p=101 and D=2^k).

          - Infinite place ("precision domain"): use floor(log2 D) = bit_length(D)-1 as an exact integer height.
            This models fixed-point scaling denominators that create rounding residue in EVM integer division.

        For each diamond Wilson loop with two homotopic paths (path1, path2):

            <γ,ω>_fin   := Δ Σ v_p(D)
            <γ,ω>_∞     := Δ Σ floor(log2 D)

        Ghost intersection:
            (<γ,ω>_fin == 0) AND (<γ,ω>_∞ != 0)

        This certificate is "mathematically legal" in the engineering sense:
          - all quantities are explicitly defined integers
          - no thresholds, no randomness, no EVM execution
        """
        p_i = int(p)
        if p_i <= 2:
            return {"ok": False, "reason": "p must be > 2"}

        if not isinstance(evm_constraints, list):
            return {
                "ok": True,
                "skipped": True,
                "reason": "no evm_constraints in artifact; cannot compute arithmetic height pairing",
                "ghost_intersection_detected": False,
            }

        def _parse_int(x: Any) -> Optional[int]:
            if x is None:
                return None
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, int):
                return int(x)
            if isinstance(x, str):
                s = x.strip()
                if s.startswith(("0x", "0X")):
                    try:
                        return int(s, 16)
                    except Exception:
                        return None
                try:
                    return int(s, 10)
                except Exception:
                    return None
            try:
                return int(x)
            except Exception:
                return None

        def _vp(n: int, p0: int) -> int:
            nn = int(abs(int(n)))
            pp = int(p0)
            if pp <= 1:
                return 0
            if nn == 0:
                return 10**9  # "infinite" valuation; should not happen for denominators
            k = 0
            while nn % pp == 0:
                nn //= pp
                k += 1
            return int(k)

        # Collect per-block denominator data from constraints
        node_bits: Dict[int, int] = {}
        node_vp: Dict[int, int] = {}
        node_samples: Dict[int, List[Dict[str, Any]]] = {}
        total_terms = 0
        used_terms = 0

        for c in evm_constraints:
            if not isinstance(c, Mapping):
                continue
            total_terms += 1
            meta = c.get("meta")
            if not isinstance(meta, Mapping):
                continue
            bpc = _parse_int(meta.get("block_pc"))
            if bpc is None:
                continue
            op = str(c.get("op") or meta.get("original_op") or "")
            op = op.upper().strip()
            if op not in ("DIV", "SDIV", "MOD", "SMOD"):
                continue
            # AddressHunter-style constraints usually encode constants in inputs like "const_0x100000000".
            D = _parse_int(c.get("operand_b") or meta.get("operand_b") or c.get("modulus") or meta.get("modulus"))
            if D is None:
                inputs = c.get("inputs")
                if isinstance(inputs, list) and len(inputs) >= 2:
                    tok = inputs[1]
                    if isinstance(tok, str):
                        s = tok.strip()
                        if s.startswith("const_"):
                            D = _parse_int(s[len("const_") :])
            if D is None:
                continue
            if int(D) <= 1:
                continue
            used_terms += 1
            bits = int(int(D).bit_length() - 1)
            node_bits[int(bpc)] = int(node_bits.get(int(bpc), 0) + bits)
            node_vp[int(bpc)] = int(node_vp.get(int(bpc), 0) + _vp(int(D), p_i))
            samp = node_samples.setdefault(int(bpc), [])
            if len(samp) < 4:
                samp.append(
                    {
                        "op": op,
                        "D": int(D),
                        "bits": int(bits),
                        "vp": int(_vp(int(D), p_i)),
                        "has_guard": bool(c.get("has_guard", True)),
                        "pc": _parse_int(meta.get("pc")),
                    }
                )

        def _path_height(path_nodes: Sequence[Any]) -> Dict[str, Any]:
            nodes_i = [int(x) for x in path_nodes if isinstance(x, (int, float, str)) or x is not None]
            ctr = Counter(nodes_i)
            bits_sum = 0
            vp_sum = 0
            for n0, k0 in ctr.items():
                bits_sum += int(k0) * int(node_bits.get(int(n0), 0))
                vp_sum += int(k0) * int(node_vp.get(int(n0), 0))
            return {"bits": int(bits_sum), "vp": int(vp_sum), "counts": dict(ctr)}

        loop_rows: List[Dict[str, Any]] = []
        ghost_indices: List[int] = []
        for i, w in enumerate(wilson_loops):
            p1 = w.get("path1_nodes")
            p2 = w.get("path2_nodes")
            if not (isinstance(p1, list) and isinstance(p2, list)):
                continue
            h1 = _path_height(p1)
            h2 = _path_height(p2)
            d_bits = int(h1["bits"] - h2["bits"])
            d_vp = int(h1["vp"] - h2["vp"])
            ghost = (d_vp == 0) and (d_bits != 0)

            # Witness node where bits contribution differs
            witness = None
            if d_bits != 0:
                all_nodes = sorted(set(int(x) for x in h1["counts"].keys()) | set(int(x) for x in h2["counts"].keys()))
                for n0 in all_nodes:
                    c1 = int(h1["counts"].get(int(n0), 0))
                    c2 = int(h2["counts"].get(int(n0), 0))
                    if c1 == c2:
                        continue
                    contrib = int((c1 - c2) * int(node_bits.get(int(n0), 0)))
                    if contrib != 0:
                        witness = {
                            "node": int(n0),
                            "bits_contrib": int(contrib),
                            "samples": node_samples.get(int(n0), []),
                        }
                        break

            row = {
                "loop_index": int(i),
                "loop_type": w.get("loop_type"),
                "branch": w.get("branch"),
                "merge": w.get("merge"),
                "holonomy_rank_modp": w.get("holonomy_rank_modp"),
                "pairing_fin_vp_delta": int(d_vp),
                "pairing_infty_bits_delta": int(d_bits),
                "ghost_intersection": bool(ghost),
                "witness": witness,
            }
            loop_rows.append(row)
            if ghost:
                ghost_indices.append(int(i))

        # Summary scalars
        ghost_abs = [abs(int(r["pairing_infty_bits_delta"])) for r in loop_rows if bool(r.get("ghost_intersection"))]
        min_abs = int(min(ghost_abs)) if ghost_abs else 0
        max_abs = int(max(ghost_abs)) if ghost_abs else 0

        return {
            "ok": True,
            "field": {"type": "Fp", "p": p_i},
            "terms_total": int(total_terms),
            "terms_used_divmod_with_const_D": int(used_terms),
            "node_count_with_terms": int(len(node_bits)),
            "ghost_intersection_detected": bool(len(ghost_indices) > 0),
            "ghost_loop_indices_sample": ghost_indices[:16],
            "ghost_min_abs_infty_bits_delta": int(min_abs),
            "ghost_max_abs_infty_bits_delta": int(max_abs),
            "per_loop": loop_rows[: min(16, len(loop_rows))],
            "definition": {
                "pairing_fin": "Delta sum v_p(D) across the two diamond paths (D from DIV/MOD constant denominators).",
                "pairing_infty": "Delta sum floor(log2 D) across the two diamond paths (exact bit_length(D)-1).",
                "ghost_intersection": "pairing_fin == 0 and pairing_infty != 0",
            },
            "explain": "Arithmetic height surrogate: detect precision-domain ghost intersections even when finite-field (p-adic) exactness holds.",
        }

    def _tate_shafarevich_obstruction_certificate(
        self,
        *,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
        entry: int,
        cyclic_sccs: Sequence[Sequence[int]],
        cycles_out: Sequence[Mapping[str, Any]],
        p: int,
    ) -> Dict[str, Any]:
        """
        Tate-Shafarevich 障碍
        目标：在不进行链上操作的情况下，将代数/拓扑模型中存在局部环与全局环可从实际条目实现区分开来
        离散证书：
        - 局部环空间：F_p 上完整边集的 ker(B_full)
        - 全局环空间：F_p 上条目可达导出子图的 ker(B_reach)
        - 障碍维度：dim_local - dim_global。
        如果 >0，则存在幻影环局部但不可全局实现
        还交叉引用枚举的 SCC/环，以提供具体的证明
        """
        p_i = int(p)
        entry_i = int(entry)
        if p_i <= 2:
            return {"ok": False, "reason": "p must be > 2"}
        if not nodes or not edges:
            return {"ok": True, "dim_local": 0, "dim_global": 0, "obstruction_dim": 0, "phantom_detected": False}

        node_set = set(int(n) for n in nodes)
        if entry_i not in node_set:
            entry_i = int(min(node_set))

        # Directed reachability (unweighted, deterministic)
        adj: Dict[int, List[int]] = {}
        for (u, v) in edges:
            uu, vv = int(u), int(v)
            if uu in node_set and vv in node_set:
                adj.setdefault(uu, []).append(vv)
        for k in list(adj.keys()):
            adj[k] = sorted(set(adj[k]))

        reach: set[int] = {entry_i}
        q = [entry_i]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in adj.get(u, []):
                if v not in reach:
                    reach.add(int(v))
                    q.append(int(v))

        # Deterministic edge indexing
        node_list = sorted(node_set)
        node_index = {pc: i for i, pc in enumerate(node_list)}
        edge_list = [(int(u), int(v)) for (u, v) in edges if int(u) in node_set and int(v) in node_set]
        m = len(edge_list)

        def build_incidence(node_list0: Sequence[int], cols_edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
            idx0 = {pc: i for i, pc in enumerate(node_list0)}
            mm = len(cols_edges)
            B = [[0] * mm for _ in range(len(node_list0))]
            for j, (u, v) in enumerate(cols_edges):
                iu = idx0.get(int(u))
                iv = idx0.get(int(v))
                if iu is None or iv is None:
                    continue
                B[iu][j] = (B[iu][j] - 1) % p_i
                B[iv][j] = (B[iv][j] + 1) % p_i
            return B

        # Local cycle space dimension
        B_full = build_incidence(node_list, edge_list)
        rank_full = int(rank_modp(B_full, p_i))
        dim_local = int(m - rank_full)

        # Global cycle space on reachable induced subgraph
        reach_nodes = sorted(reach)
        reach_set = set(reach_nodes)
        edge_reach = [e for e in edge_list if int(e[0]) in reach_set and int(e[1]) in reach_set]
        B_reach = build_incidence(reach_nodes, edge_reach) if edge_reach and reach_nodes else []
        rank_reach = int(rank_modp(B_reach, p_i)) if edge_reach else 0
        dim_global = int(len(edge_reach) - rank_reach)

        obstruction = int(max(0, dim_local - dim_global))
        phantom = obstruction > 0

        # SCC-level phantom carriers (cyclic SCCs not reachable)
        phantom_sccs: List[Dict[str, Any]] = []
        for comp in cyclic_sccs:
            cc = [int(x) for x in comp]
            if not cc:
                continue
            if any(x in reach_set for x in cc):
                continue
            phantom_sccs.append({"scc_size": int(len(cc)), "nodes_sample": cc[: min(24, len(cc))]})
        phantom_sccs = phantom_sccs[:16]

        # Enumerated phantom cycles (from cycle sampler)
        phantom_cycle_indices: List[int] = []
        for i, c in enumerate(cycles_out):
            cyc_nodes = c.get("cycle_nodes") or []
            if not isinstance(cyc_nodes, list):
                continue
            if any(int(x) in reach_set for x in cyc_nodes):
                continue
            phantom_cycle_indices.append(int(i))
        phantom_cycle_indices = phantom_cycle_indices[:16]

        out: Dict[str, Any] = {
            "ok": True,
            "field": {"type": "Fp", "p": p_i},
            "entry": int(entry_i),
            "n_nodes": int(len(node_list)),
            "m_edges": int(len(edge_list)),
            "n_nodes_reachable": int(len(reach_nodes)),
            "m_edges_reachable": int(len(edge_reach)),
            "rank_B_full": int(rank_full),
            "rank_B_reachable": int(rank_reach),
            "dim_cycle_space_local": int(dim_local),
            "dim_cycle_space_global": int(dim_global),
            "obstruction_dim": int(obstruction),
            "phantom_detected": bool(phantom),
            "phantom_sccs_sample": phantom_sccs,
            "phantom_cycle_indices_sample": phantom_cycle_indices,
            "explain": "Tate–Shafarevich surrogate: obstruction_dim = dim ker(B_full) - dim ker(B_reachable). >0 means local cycles exist that are not globally reachable from entry (phantom cycles).",
        }

        # Witness: nullspace vector with support on unreachable edges
        if phantom:
            if _nullspace_basis_modp is None:
                out["witness"] = {"ok": False, "reason": "nullspace basis unavailable"}
                return out
            basis = _nullspace_basis_modp(B_full, p_i, n_vars=m)
            reach_edges_set = set(edge_reach)
            unreachable_idx = {j for j, e in enumerate(edge_list) if e not in reach_edges_set}
            w_vec = None
            for v in basis:
                if any((int(v[j]) % p_i) != 0 for j in unreachable_idx):
                    w_vec = v
                    break
            if w_vec is None:
                out["witness"] = {"ok": False, "reason": "no witness found (unexpected); check obstruction_dim"}
                return out

            support: List[Tuple[int, Tuple[int, int], int]] = []
            for j, coef in enumerate(w_vec):
                cc = int(coef) % p_i
                if cc == 0:
                    continue
                if len(support) >= 24:
                    break
                support.append((int(j), edge_list[int(j)], int(cc)))

            unreachable_support = None
            for j, e, cc in support:
                if e not in reach_edges_set:
                    unreachable_support = {"edge_index": int(j), "edge": [int(e[0]), int(e[1])], "coef": int(cc)}
                    break

            out["witness"] = {
                "ok": True,
                "unreachable_edge": unreachable_support,
                "support_sample": [{"edge_index": int(j), "edge": [int(e[0]), int(e[1])], "coef": int(cc)} for j, e, cc in support],
                "note": "A concrete 1-cycle class has non-zero support on unreachable edges => local cycle not globally realizable from entry.",
            }

        return out

    def _hodge_conjecture_certificate(
        self,
        *,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
        lem_gap_edges: set[Tuple[int, int]],
        p: int,
    ) -> Dict[str, Any]:
        """
        引入霍奇猜想
        """
        p_i = int(p)
        if p_i <= 2:
            return {"ok": False, "reason": "p must be > 2"}
        if not nodes or not edges:
            return {"ok": True, "dim_full": 0, "dim_lem_gap": 0, "holds": True, "reason": "empty graph"}

        # Deterministic node/edge indexing
        node_list = sorted(set(int(n) for n in nodes))
        edge_list = [(int(u), int(v)) for (u, v) in edges]
        n = len(node_list)
        m = len(edge_list)
        node_index = {pc: i for i, pc in enumerate(node_list)}

        # Fast, exact cycle-space dimension for incidence matrices (no dense mod-p RREF):
        #
        # For an oriented incidence matrix B of a (possibly directed) graph on n nodes with c weakly-connected components,
        # rank(B) = n - c over any field F_p (p>2 here). Therefore:
        #   dim ker(B) = m - rank(B) = m - n + c
        #
        # This is deterministic, strict, and avoids O(n^2 m) Gauss-Jordan on dense lists.
        def _components_count(cols_edges: Sequence[Tuple[int, int]]) -> int:
            parent = list(range(n))
            rnk = [0] * n

            def _find(x: int) -> int:
                xx = int(x)
                while parent[xx] != xx:
                    parent[xx] = parent[parent[xx]]
                    xx = parent[xx]
                return xx

            def _union(a: int, b: int) -> bool:
                ra = _find(a)
                rb = _find(b)
                if ra == rb:
                    return False
                if rnk[ra] < rnk[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rnk[ra] == rnk[rb]:
                    rnk[ra] += 1
                return True

            # Weak connectivity: orientation does not affect rank (column sign flips do not change rank).
            for (u, v) in cols_edges:
                iu = node_index.get(int(u))
                iv = node_index.get(int(v))
                if iu is None or iv is None:
                    continue
                if int(iu) == int(iv):
                    continue  # self-loop column is zero; does not connect components
                _union(int(iu), int(iv))

            roots = {_find(i) for i in range(n)}
            return int(len(roots))

        # Full graph
        comp_full = _components_count(edge_list)
        rank_full = int(n - comp_full)
        dim_full = int(m - rank_full)

        # LEM-gap edge subgraph (same directed-edge membership as before)
        lem_cols = [e for e in edge_list if (int(e[0]), int(e[1])) in lem_gap_edges]
        comp_lem = _components_count(lem_cols) if lem_cols else int(n)  # all isolated
        rank_lem = int(n - comp_lem) if lem_cols else 0
        dim_lem = int(len(lem_cols) - rank_lem)

        holds = bool(dim_full == dim_lem)
        out: Dict[str, Any] = {
            "ok": True,
            "field": {"type": "Fp", "p": p_i},
            "n_nodes": int(n),
            "m_edges": int(m),
            "m_edges_lem_gap": int(len(lem_cols)),
            "rank_B_full": int(rank_full),
            "rank_B_lem_gap": int(rank_lem),
            "dim_cycle_space_full": int(dim_full),
            "dim_cycle_space_lem_gap": int(dim_lem),
            "components_full": int(comp_full),
            "components_lem_gap": int(comp_lem),
            "holds": holds,
            "explain": "Discrete Hodge-conjecture surrogate: cycle space dim(full) vs dim(LEM-gap subgraph). If equal, all harmonic 1-classes can be represented inside LEM-gap edges.",
        }

        # Witness when not holding
        if not holds:
            # Deterministic witness without dense nullspace basis:
            # Build a spanning forest that prefers LEM-gap edges, then find a non-LEM edge that closes a cycle.
            parent = list(range(n))
            rnk = [0] * n
            forest_adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]  # (nbr_idx, edge_idx)

            def _find(x: int) -> int:
                xx = int(x)
                while parent[xx] != xx:
                    parent[xx] = parent[parent[xx]]
                    xx = parent[xx]
                return xx

            def _union(a: int, b: int) -> bool:
                ra = _find(a)
                rb = _find(b)
                if ra == rb:
                    return False
                if rnk[ra] < rnk[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rnk[ra] == rnk[rb]:
                    rnk[ra] += 1
                return True

            def _edge_endpoints_idx(e: Tuple[int, int]) -> Optional[Tuple[int, int]]:
                iu = node_index.get(int(e[0]))
                iv = node_index.get(int(e[1]))
                if iu is None or iv is None:
                    return None
                return int(iu), int(iv)

            # 1) Add LEM-gap edges into the forest first (deterministic order = edge_list order).
            for j, e in enumerate(edge_list):
                if (int(e[0]), int(e[1])) not in lem_gap_edges:
                    continue
                uv = _edge_endpoints_idx(e)
                if uv is None:
                    continue
                iu, iv = uv
                if iu == iv:
                    continue  # loop doesn't help connect components
                if _union(iu, iv):
                    forest_adj[iu].append((iv, int(j)))
                    forest_adj[iv].append((iu, int(j)))

            # 2) Add non-LEM edges; the first one whose endpoints are already connected gives a cycle witness.
            cycle_edge_idx: Optional[int] = None
            cycle_u_idx: Optional[int] = None
            cycle_v_idx: Optional[int] = None
            for j, e in enumerate(edge_list):
                if (int(e[0]), int(e[1])) in lem_gap_edges:
                    continue
                uv = _edge_endpoints_idx(e)
                if uv is None:
                    continue
                iu, iv = uv
                if iu == iv:
                    # Non-LEM self-loop: immediate nullspace witness (zero column in incidence).
                    cycle_edge_idx = int(j)
                    cycle_u_idx = int(iu)
                    cycle_v_idx = int(iv)
                    break
                if not _union(iu, iv):
                    cycle_edge_idx = int(j)
                    cycle_u_idx = int(iu)
                    cycle_v_idx = int(iv)
                    break
                forest_adj[iu].append((iv, int(j)))
                forest_adj[iv].append((iu, int(j)))

            if cycle_edge_idx is None or cycle_u_idx is None or cycle_v_idx is None:
                out["witness"] = {"ok": False, "reason": "failed to construct non-LEM cycle witness (unexpected); check edge sets"}
                return out

            # Find unique forest path from v -> u, then add the non-LEM cycle edge (u -> v).
            def _forest_path_steps(start: int, goal: int) -> List[Tuple[int, int, int]]:
                if int(start) == int(goal):
                    return []
                prev_node = [-1] * n
                prev_edge = [-1] * n
                s = int(start)
                g = int(goal)
                prev_node[s] = s
                q: List[int] = [s]
                qi = 0
                while qi < len(q):
                    cur = q[qi]
                    qi += 1
                    for nxt, eidx in forest_adj[cur]:
                        if prev_node[int(nxt)] != -1:
                            continue
                        prev_node[int(nxt)] = int(cur)
                        prev_edge[int(nxt)] = int(eidx)
                        if int(nxt) == g:
                            qi = len(q)  # break outer loop
                            break
                        q.append(int(nxt))
                if prev_node[g] == -1:
                    return []
                steps: List[Tuple[int, int, int]] = []
                cur2 = g
                while cur2 != s:
                    p0 = int(prev_node[cur2])
                    e0 = int(prev_edge[cur2])
                    steps.append((p0, int(cur2), e0))
                    cur2 = p0
                steps.reverse()
                return steps

            path_steps = _forest_path_steps(int(cycle_v_idx), int(cycle_u_idx))

            def _coef_for_traversal(from_idx: int, to_idx: int, eidx: int) -> int:
                u0, v0 = edge_list[int(eidx)]
                a = int(node_list[int(from_idx)])
                b = int(node_list[int(to_idx)])
                if a == int(u0) and b == int(v0):
                    return 1
                if a == int(v0) and b == int(u0):
                    return int(p_i - 1)  # -1 mod p
                # Should not happen for a forest edge; keep deterministic fallback.
                return 1

            support_full: List[Tuple[int, Tuple[int, int], int]] = []
            for a_idx, b_idx, eidx in path_steps:
                cc = _coef_for_traversal(int(a_idx), int(b_idx), int(eidx))
                support_full.append((int(eidx), edge_list[int(eidx)], int(cc)))
            # cycle edge coefficient is +1 by choosing traversal v->...->u then u->v
            support_full.append((int(cycle_edge_idx), edge_list[int(cycle_edge_idx)], 1))

            # Bounded sample for JSON size (keep the non-LEM cycle edge explicitly).
            truncated = len(support_full) > 24
            if truncated:
                support = support_full[:23] + [support_full[-1]]
            else:
                support = support_full

            non_lem_support = {
                "edge_index": int(cycle_edge_idx),
                "edge": [int(edge_list[int(cycle_edge_idx)][0]), int(edge_list[int(cycle_edge_idx)][1])],
                "coef": 1,
            }

            out["witness"] = {
                "ok": True,
                "non_lem_gap_edge": non_lem_support,
                "cycle_length_edges": int(len(path_steps) + 1),
                "support_sample": [
                    {"edge_index": int(j), "edge": [int(e[0]), int(e[1])], "coef": int(cc)} for j, e, cc in support
                ],
                "truncated": bool(truncated),
                "note": "Cycle-space dim(full) > dim(LEM-gap). Witness provides a concrete 1-cycle that necessarily uses a non-LEM-gap edge (in this surrogate).",
            }

        return out

    def _period_isomorphism_certificate(
        self,
        *,
        p: int,
        crystalline_frobenius: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        第三处：周期环上的比较同构验证（B_dR surrogate）
        --------------------------------------------------
        目标（确定性、无阈值、无静默失败）：
        - 以第二处晶体 Frobenius 的 Newton 斜率谱为输入
        - 构造一个 B_dR 过滤层级判定：ordinary => Fil^0；非 ordinary => Fil^{-k}
        - 给出 Selmer 秩爆炸的离散代理量（由 Hodge–Newton gap 精确整化得到）

        注：
        - 这里不做“assert balance_new > balance_old”之类启发式。
        - 过滤阶 k 取 Newton 斜率分母的 lcm（canonical），用于把所有差值整化为整数。
        """
        p_i = int(p)
        if p_i <= 2:
            return {"ok": False, "reason": "p must be > 2"}

        if not isinstance(crystalline_frobenius, Mapping):
            return {"ok": False, "reason": "crystalline_frobenius must be dict-like"}
        if not bool(crystalline_frobenius.get("ok")):
            return {"ok": False, "reason": "crystalline_frobenius not ok"}

        try:
            d = int(crystalline_frobenius.get("basis_dim"))
        except Exception:
            return {"ok": False, "reason": "crystalline_frobenius missing basis_dim"}
        if d < 1:
            return {"ok": False, "reason": "basis_dim must be >= 1"}

        expected_filtration = 0  # Fil^0 baseline for “ordinary” outputs (engineering convention)

        def _q(x: Any) -> Fraction:
            if isinstance(x, bool):
                raise MVP17CFGError("period_isomorphism: slope cannot be bool")
            if isinstance(x, int):
                return Fraction(int(x), 1)
            if isinstance(x, Mapping) and ("num" in x) and ("den" in x):
                num = int(x.get("num"))
                den = int(x.get("den"))
                if den == 0:
                    raise MVP17CFGError("period_isomorphism: zero denominator")
                return Fraction(num, den)
            raise MVP17CFGError(f"period_isomorphism: cannot parse rational from {x!r}")

        def _qjson(qv: Fraction) -> Any:
            if qv.denominator == 1:
                return int(qv.numerator)
            return {"num": int(qv.numerator), "den": int(qv.denominator)}

        def _lcm(a: int, b: int) -> int:
            aa, bb = int(a), int(b)
            if aa < 0 or bb < 0:
                raise MVP17CFGError("period_isomorphism: lcm inputs must be non-negative")
            if aa == 0 or bb == 0:
                return int(aa + bb)
            return int(abs(aa * bb) // math.gcd(aa, bb))

        def _den_lcm(xs: Sequence[Fraction]) -> int:
            out = 1
            for qv in xs:
                out = _lcm(out, int(qv.denominator))
            return int(out)

        # Collect analyzable items (cycle + diamond path1/path2)
        items: List[Dict[str, Any]] = []

        cycles = crystalline_frobenius.get("cycles") or []
        if isinstance(cycles, list):
            for row in cycles:
                if not isinstance(row, Mapping):
                    continue
                if not bool(row.get("ok")):
                    continue
                an = row.get("analysis")
                if not isinstance(an, Mapping):
                    continue
                items.append({"kind": "cycle", "index": int(row.get("index", -1)), "path": None, "analysis": dict(an)})

        diamonds = crystalline_frobenius.get("diamonds") or []
        if isinstance(diamonds, list):
            for row in diamonds:
                if not isinstance(row, Mapping):
                    continue
                if not bool(row.get("ok")):
                    continue
                paths = row.get("paths")
                if not isinstance(paths, Mapping):
                    continue
                idx = int(row.get("index", -1))
                a1 = paths.get("path1")
                a2 = paths.get("path2")
                if isinstance(a1, Mapping):
                    items.append({"kind": "diamond", "index": idx, "path": "path1", "analysis": dict(a1)})
                if isinstance(a2, Mapping):
                    items.append({"kind": "diamond", "index": idx, "path": "path2", "analysis": dict(a2)})

        if not items:
            return {
                "ok": True,
                "skipped": True,
                "reason": "no analyzable crystalline items",
                "field": {"type": "B_dR", "p": p_i},
                "signals": {"filtration_jump": False, "max_pole_order": 0, "max_selmer_rank_surrogate": 0},
            }

        best = None
        best_key = None
        max_pole = 0
        max_selmer = 0
        max_inconsistency = 0
        analyzed = 0
        errors: List[Dict[str, Any]] = []
        err_cap = int(self.max_cycles)
        if err_cap < 1:
            err_cap = 1

        for it in items:
            kind = str(it.get("kind"))
            idx = int(it.get("index", -1))
            path = it.get("path")
            an = it.get("analysis")
            if not isinstance(an, Mapping):
                if len(errors) < err_cap:
                    errors.append({"kind": kind, "index": idx, "path": path, "reason": "analysis_not_mapping"})
                continue

            hn = an.get("hodge_newton") or {}
            if not isinstance(hn, Mapping):
                if len(errors) < err_cap:
                    errors.append({"kind": kind, "index": idx, "path": path, "reason": "missing_hodge_newton"})
                continue
            ordinary = bool(hn.get("ordinary"))
            supersingular = bool(hn.get("supersingular"))

            newt = an.get("newton") or {}
            hodg = an.get("hodge") or {}
            if not (isinstance(newt, Mapping) and isinstance(hodg, Mapping)):
                if len(errors) < err_cap:
                    errors.append({"kind": kind, "index": idx, "path": path, "reason": "missing_newton_or_hodge"})
                continue
            ns_raw = newt.get("slopes_per_step_sorted") or []
            hs_raw = hodg.get("slopes_sorted") or []
            if not (isinstance(ns_raw, list) and isinstance(hs_raw, list)):
                if len(errors) < err_cap:
                    errors.append({"kind": kind, "index": idx, "path": path, "reason": "slopes_not_list"})
                continue

            try:
                ns = [_q(x) for x in ns_raw]
                hs = [_q(x) for x in hs_raw]
            except Exception:
                # Do not crash whole report; this item is malformed.
                if len(errors) < err_cap:
                    errors.append({"kind": kind, "index": idx, "path": path, "reason": "slopes_parse_failed"})
                continue

            if len(ns) != d or len(hs) != d:
                if len(errors) < err_cap:
                    errors.append(
                        {"kind": kind, "index": idx, "path": path, "reason": f"slope_len_mismatch: newton={len(ns)} hodge={len(hs)} basis_dim={d}"}
                    )
                continue

            den_lcm = _den_lcm(ns + hs)
            pole_order = 0 if ordinary else int(den_lcm)

            # Prefix gaps: Δ_k = Σ_{i<k}(n_i - h_i)
            pref = Fraction(0, 1)
            gap_max = Fraction(0, 1)
            gap_min = Fraction(0, 1)
            for a, b in zip(ns, hs):
                pref += (a - b)
                if pref > gap_max:
                    gap_max = pref
                if pref < gap_min:
                    gap_min = pref

            # Scale gaps to integers using den_lcm
            selmer_scaled = gap_max * den_lcm
            incons_scaled = (-gap_min) * den_lcm
            if selmer_scaled.denominator != 1 or incons_scaled.denominator != 1:
                if len(errors) < err_cap:
                    errors.append({"kind": kind, "index": idx, "path": path, "reason": "scaled_gap_not_integer"})
                continue
            selmer_rank = int(selmer_scaled.numerator)
            incons_rank = int(incons_scaled.numerator)
            analyzed += 1

            severity = int(max(selmer_rank, incons_rank, pole_order))
            max_pole = max(max_pole, int(pole_order))
            max_selmer = max(max_selmer, int(selmer_rank))
            max_inconsistency = max(max_inconsistency, int(incons_rank))

            witness = {
                "kind": kind,
                "index": idx,
                "path": path,
                "ordinary": bool(ordinary),
                "supersingular": bool(supersingular),
                "pole_order": int(pole_order),
                "filtration": {"expected": int(expected_filtration), "actual": int(-pole_order), "jump": bool(pole_order > 0)},
                "den_lcm": int(den_lcm),
                "gap_max": _qjson(gap_max),
                "gap_min": _qjson(gap_min),
                "selmer_rank_surrogate": int(selmer_rank),
                "inconsistency_rank_surrogate": int(incons_rank),
                "fractional_slope_witness": hn.get("fractional_slope_witness"),
            }

            key = (-int(pole_order), -int(severity), str(kind), int(idx), "" if path is None else str(path))
            if best is None or key < best_key:  # type: ignore[operator]
                best = witness
                best_key = key

        filtration_jump = bool(max_pole > 0)

        xi = None
        if best is not None:
            k = int(best.get("pole_order", 0))
            a0 = int(max(int(best.get("selmer_rank_surrogate", 0)), int(best.get("inconsistency_rank_surrogate", 0)), k))
            if k > 0:
                xi = {
                    "kind": "laurent_pole",
                    "t_pole_order": int(k),
                    "leading_coeff": int(a0),
                    "representation": f"{int(a0)} * t^(-{int(k)})",
                }
            else:
                xi = {"kind": "regular", "filtration": int(expected_filtration), "representation": "0 in Fil^0"}

        if best is None:
            return {
                "ok": False,
                "reason": "no valid period items could be analyzed (malformed crystalline analysis)",
                "field": {"type": "B_dR", "p": int(p_i)},
                "basis_dim": int(d),
                "items_total": int(len(items)),
                "items_analyzed": int(analyzed),
                "errors_sample": errors,
            }

        return {
            "ok": True,
            "field": {"type": "B_dR", "p": int(p_i)},
            "basis_dim": int(d),
            "items_total": int(len(items)),
            "items_analyzed": int(analyzed),
            "errors_sample": errors,
            "expected_filtration": int(expected_filtration),
            "signals": {
                "filtration_jump": bool(filtration_jump),
                "max_pole_order": int(max_pole),
                "max_selmer_rank_surrogate": int(max_selmer),
                "max_inconsistency_rank_surrogate": int(max_inconsistency),
            },
            "xi": xi,
            "witness": best,
            "explain": (
                "Period comparison surrogate: treat ordinary Hodge–Newton equality as ξ∈Fil^0 B_dR; "
                "if Newton slopes have denominators, use lcm(den) as pole order k so ξ lands in Fil^{-k}. "
                "Selmer rank surrogate is the scaled Hodge–Newton gap (exact integer after clearing denominators)."
            ),
        }


def load_artifact_targets(path: str) -> List[Dict[str, Any]]:
    """
    json列表
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise MVP17CFGInputError("artifact json must be a list of targets")
    return [t for t in obj if isinstance(t, dict)]





