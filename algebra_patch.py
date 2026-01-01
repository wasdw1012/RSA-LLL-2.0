"""
- Garcia-Stichtenoth W1 塔构造和单点 AG 代码辅助函数
"""

from sage.all import GF, FunctionField, PolynomialRing, matrix, vector, ZZ


# --------------------------------------------------------------------------- #
# 0) Finite-field trace / norm (explicit Frobenius, exact)
# --------------------------------------------------------------------------- #

def ff_trace(a, q, r):
    """
    Tr_{F_{q^r}/F_q}(a) = sum_{i=0}^{r-1} a^{q^i}
    Exact (Frobenius powering), no blackbox.
    """
    F = a.parent()
    s = F.zero()
    x = a
    for _ in range(r):
        s += x
        x = x ** q
    return s


def ff_norm(a, q, r):
    """
    N_{F_{q^r}/F_q}(a) = product_{i=0}^{r-1} a^{q^i}
    Exact (Frobenius powering), no blackbox.
    """
    F = a.parent()
    p = F.one()
    x = a
    for _ in range(r):
        p *= x
        x = x ** q
    return p


def matrix_trace(M, q, r, K=None):
    """
    Coordinatewise trace on matrix over GF(q^r) -> GF(q).
    If K provided, coerce into K.
    """
    rows = []
    for i in range(M.nrows()):
        row = []
        for j in range(M.ncols()):
            t = ff_trace(M[i, j], q, r)
            row.append(K(t) if K is not None else t)
        rows.append(row)
    return matrix(rows)


# --------------------------------------------------------------------------- #
# 1) Local completion tools (exact residue evaluation)
# --------------------------------------------------------------------------- #

def _completion_pair(F, P, prec):
    """
    Return (K, phi) with phi: F -> K the completion embedding at P.
    Compatible with typical Sage completion() return forms.
    """
    cp = F.completion(P, prec)
    if isinstance(cp, tuple) and len(cp) == 2:
        return cp[0], cp[1]
    if hasattr(cp, "codomain"):
        return cp.codomain(), cp
    raise RuntimeError("Unsupported completion() return type in this Sage version.")


def eval_at_place_via_completion(F, P, f):
    """
    Exact evaluation f(P) (degree-1 place expected) via completion residue.
    """
    K, phi = _completion_pair(F, P, prec=1)
    lf = phi(f)
    if hasattr(lf, "residue"):
        return lf.residue()
    if hasattr(K, "residue_field"):
        k, red = K.residue_field()
        return red(lf)
    if hasattr(lf, "constant_coefficient"):
        return lf.constant_coefficient()
    if hasattr(lf, "coefficient"):
        return lf.coefficient(0)
    try:
        return lf[0]
    except Exception as e:
        raise RuntimeError(f"Cannot extract residue/constant term: {e}")


# --------------------------------------------------------------------------- #
# 2) Rational places and deterministic evaluation set selection
# --------------------------------------------------------------------------- #

def rational_places(F, exclude=set()):
    """
    List all degree-1 places except those in exclude.
    """
    out = []
    for P in F.places(1):
        if P not in exclude:
            out.append(P)
    return out


def pick_eval_places(F, Pinf, need_count, forbid=set()):
    """
    Deterministically pick 'need_count' rational places excluding Pinf and forbid.
    All returned places are degree-1.
    """
    ex = set(forbid)
    ex.add(Pinf)
    pts = rational_places(F, exclude=ex)
    if len(pts) < need_count:
        raise RuntimeError(
            "Not enough rational places available for certification/evaluation."
        )
    return pts[:need_count]


# --------------------------------------------------------------------------- #
# 3) Global pole support checks (no hidden poles)
# --------------------------------------------------------------------------- #

def pole_divisor_support(f):
    """
    Return set of places where f has a pole (support of divisor_of_poles).
    """
    Dp = f.divisor_of_poles()
    return set(Dp.support())


def pole_order_at(f, P):
    """
    Pole order at P: -v_P(f) if negative, else 0.
    """
    v = f.valuation(P)
    return ZZ(-v) if v < 0 else ZZ(0)


def assert_only_pole_at(f, Pinf):
    """
    Enforce: f has no poles except possibly at Pinf.
    """
    supp = pole_divisor_support(f)
    if len(supp) == 0:
        return
    if supp == set([Pinf]):
        return
    raise RuntimeError(f"Function has poles outside Pinf: support size={len(supp)}")


# --------------------------------------------------------------------------- #
# 4) Candidate enumeration by pole order (no riemann_roch_space)
# --------------------------------------------------------------------------- #

def enumerate_monomials_weighted(gens, weights, bound):
    """
    Enumerate monomials prod gens[i]^e_i with sum e_i*weights[i] <= bound.
    Deterministic DFS, no artificial caps besides weight bound.
    """
    n = len(gens)
    out = []

    def dfs(i, current, wsum):
        if i == n:
            out.append(current)
            return
        wi = weights[i]
        emax = (bound - wsum) // wi if wi > 0 else 0
        g = gens[i]
        p = current
        for e in range(emax + 1):
            dfs(i + 1, p, wsum + e * wi)
            p = p * g

    one = gens[0].parent()(1)
    dfs(0, one, 0)
    return out


def candidates_in_LmPinf(F, Pinf, gens_for_monomials, m):
    """
    Build a candidate pool subset of L(m Pinf) by:
      - compute weights w_i = pole order of gen_i at Pinf
      - enumerate all monomials with weighted pole order <= m
      - filter: monomial has no poles outside Pinf (global divisor_of_poles check)
    Returns candidates sorted by (pole_order, representation).
    """
    weights = []
    for g in gens_for_monomials:
        assert_only_pole_at(g, Pinf)
        w = pole_order_at(g, Pinf)
        if w <= 0:
            raise RuntimeError(
                "Generator has no pole at Pinf; cannot use as pole-weighted generator."
            )
        weights.append(w)

    mons = enumerate_monomials_weighted(gens_for_monomials, weights, ZZ(m))

    good = []
    for f in mons:
        assert_only_pole_at(f, Pinf)
        if pole_order_at(f, Pinf) <= m:
            good.append(f)

    good = sorted(good, key=lambda f: (pole_order_at(f, Pinf), f))
    return good


# --------------------------------------------------------------------------- #
# 5) Certified linear independence and basis selection by evaluations
# --------------------------------------------------------------------------- #

def eval_vectors(F, places, funcs):
    """
    Evaluate each f in funcs on 'places' by local completion residue; return matrix over const field.
    """
    K = F.constant_field()
    M = matrix(K, len(funcs), len(places))
    for i, f in enumerate(funcs):
        for j, P in enumerate(places):
            v = eval_at_place_via_completion(F, P, f)
            M[i, j] = K(v)
    return M


def certify_relation_in_LmPinf(F, Pinf, m, basis, target, eval_places):
    """
    Decide whether target is in span_K(basis) with an explicit certificate.

    We solve using evaluations at eval_places:
      E(B) * c = E(target)
    If solvable, set h = target - sum c_i B_i.
    h belongs to L(mPinf) because target and basis are in L(mPinf).
    If h vanishes at |eval_places| >= m+1 distinct rational places (excluding Pinf),
    then h must be 0 (since deg zeros <= m unless h=0).
    """
    K = F.constant_field()
    k = len(basis)

    if k == 0:
        return (False, None)

    if len(eval_places) < m + 1:
        raise RuntimeError(
            "Need >= m+1 evaluation places for a rigorous dependence certificate."
        )

    EB = eval_vectors(F, eval_places, basis)  # k x N
    Et = eval_vectors(F, eval_places, [target])  # 1 x N
    A = EB.transpose()
    b = Et.transpose()

    try:
        sol = A.solve_right(b)
    except Exception:
        return (False, None)

    coeffs = [K(sol[i, 0]) for i in range(k)]

    h = target
    for ci, fi in zip(coeffs, basis):
        h -= ci * fi

    for P in eval_places:
        if eval_at_place_via_completion(F, P, h) != 0:
            return (False, None)

    return (True, coeffs)


def basis_of_LmPinf(F, Pinf, m, gens_for_monomials, search_eval_places=None):
    """
    Construct a K-basis of L(m Pinf) without riemann_roch_space().
    """
    if search_eval_places is None:
        search_eval_places = pick_eval_places(F, Pinf, need_count=(m + 1), forbid=set())

    cands = candidates_in_LmPinf(F, Pinf, gens_for_monomials, m)

    basis = []
    for f in cands:
        if len(basis) == 0:
            basis.append(f)
            continue
        dep, _ = certify_relation_in_LmPinf(F, Pinf, m, basis, f, search_eval_places)
        if not dep:
            basis.append(f)

    return basis


def ag_code_generator_matrix_one_point(
    F, Pinf, m, gens_for_monomials, eval_places=None
):
    """
    One-point AG evaluation code:
      - L(mPinf) basis via basis_of_LmPinf()
      - Generator matrix by evaluating basis on eval_places (default: all rational places != Pinf)
    """
    if eval_places is None:
        eval_places = rational_places(F, exclude=set([Pinf]))

    if len(eval_places) < m + 1:
        raise RuntimeError(
            "Not enough eval places to certify basis construction (need >= m+1)."
        )

    cert_places = eval_places[: (m + 1)]
    basis = basis_of_LmPinf(
        F, Pinf, m, gens_for_monomials, search_eval_places=cert_places
    )

    K = F.constant_field()
    G = matrix(K, len(basis), len(eval_places))
    for i, f in enumerate(basis):
        for j, P in enumerate(eval_places):
            G[i, j] = K(eval_at_place_via_completion(F, P, f))
    return basis, eval_places, G


# --------------------------------------------------------------------------- #
# 6) Tower constructors with deterministic Pinf selection by lifting infinity
# --------------------------------------------------------------------------- #

def _base_infty_place(F0):
    infs = list(F0.places_infinite())
    if not infs:
        raise RuntimeError("No infinite places in base field.")
    return infs[0]


def _lift_place_choose_pole(Fext, Pbase, pole_test_func):
    """
    Lift a place Pbase in base field to a place in extension by selecting
    the place above where pole_test_func has a pole (negative valuation).
    Deterministic and checkable.
    """
    above = list(Fext.places_above(Pbase))
    if not above:
        raise RuntimeError("No places above base place (unexpected).")
    candidates = []
    for P in above:
        if pole_test_func.valuation(P) < 0:
            candidates.append(P)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        best = above[0]
        bestv = pole_test_func.valuation(best)
        for P in above[1:]:
            v = pole_test_func.valuation(P)
            if v < bestv:
                bestv = v
                best = P
        return best
    best = candidates[0]
    bestv = pole_test_func.valuation(best)
    for P in candidates[1:]:
        v = pole_test_func.valuation(P)
        if v < bestv:
            bestv = v
            best = P
    return best


# --------------------------------------------------------------------------- #
# 7) (A) Garcia–Stichtenoth tower W1 (Eq. 5.15)
# --------------------------------------------------------------------------- #

def gs_w1_tower(q, n_steps, base_field_degree=2):
    """
    Build W1 tower over F_{q^2}:
      L0 = F_{q^2}(t0)
      L_{i+1} = L_i(t_{i+1}),  (t_i^{q-1}+1)*(t_{i+1}^q + t_{i+1}) - t_i^q = 0

    Returns:
      (F, t_list, Pinf)
    with Pinf obtained by lifting the infinity place along the tower.
    """
    Fq2 = GF(q ** base_field_degree, name="a")
    F0 = FunctionField(Fq2["t0"].fraction_field())
    t0 = F0.gen()

    P0inf = _base_infty_place(F0)
    F = F0
    t_list = [t0]
    Pinf = P0inf

    for i in range(n_steps):
        ti = t_list[-1]
        R = PolynomialRing(F, names=("Y",))
        (Y,) = R.gens()
        f = (ti ** (q - 1) + 1) * (Y ** q + Y) - ti ** q
        F = F.extension(f, names=(f"t{i+1}",))
        t_next = F.gen()
        t_list.append(t_next)

        Pinf = _lift_place_choose_pole(F, Pinf, t0)

    assert_only_pole_at(t0, Pinf)
    return F, t_list, Pinf


def gs_w1_default_generators_for_one_point(F, t_list, Pinf):
    """
    Choose a deterministic generator set for candidate monomials in L(mPinf).
    We only keep those with poles supported on Pinf (checked).
    """
    gens = []
    for g in t_list:
        try:
            assert_only_pole_at(g, Pinf)
            if pole_order_at(g, Pinf) > 0:
                gens.append(g)
        except Exception:
            pass
    if not gens:
        raise RuntimeError("No suitable one-point generators found in W1 tower.")
    return gens


# --------------------------------------------------------------------------- #
# 8) (B) Drinfeld modular curve tower entry via W3 (Eq. 5.17)
# --------------------------------------------------------------------------- #

def drinfeld_w3_tower(q, n_steps, base_field_degree=2):
    """
    Build W3 subtower (Drinfeld modular) over F_{q^2}:
      F0 = F_{q^2}(x0)
      recurrence: (Y - 1)/Y^q = (X^q - 1)/X
      cleared: (Y - 1)*X - Y^q*(X^q - 1) = 0

    Returns:
      (F, x_list, Pinf) with Pinf lifted from base infinity.
    """
    Fq2 = GF(q ** base_field_degree, name="b")
    F0 = FunctionField(Fq2["x0"].fraction_field())
    x0 = F0.gen()

    P0inf = _base_infty_place(F0)
    F = F0
    x_list = [x0]
    Pinf = P0inf

    for i in range(n_steps):
        xi = x_list[-1]
        R = PolynomialRing(F, names=("Y",))
        (Y,) = R.gens()
        f = (Y - 1) * xi - (Y ** q) * (xi ** q - 1)
        F = F.extension(f, names=(f"x{i+1}",))
        x_next = F.gen()
        x_list.append(x_next)

        Pinf = _lift_place_choose_pole(F, Pinf, x0)

    assert_only_pole_at(x0, Pinf)
    return F, x_list, Pinf


def drinfeld_w3_default_generators_for_one_point(F, x_list, Pinf):
    """
    Deterministic generator set for L(mPinf) candidates in W3:
    keep those with poles only at Pinf and with positive pole order at Pinf.
    """
    gens = []
    for g in x_list:
        try:
            assert_only_pole_at(g, Pinf)
            if pole_order_at(g, Pinf) > 0:
                gens.append(g)
        except Exception:
            pass
    if not gens:
        raise RuntimeError("No suitable one-point generators found in W3 tower at this level.")
    return gens


def _w3_pole_profile(Pinf, symbols):
    """
    Compute pole orders of all symbols at Pinf.
    """
    return {name: pole_order_at(sym, Pinf) for name, sym in symbols}


def drinfeld_w3_genus_sequence(q, n_steps, base_field_degree=2):
    """
    Build the W3 tower while returning a per-level record of genus and pole orders.

    The pole-order bookkeeping addresses the subtle infinity adjustments of the
    W3 recursion: we lift the chosen infinity place deterministically and record
    the exact valuations of every generator, eliminating ambiguity about the
    divisor of poles that drives the genus growth.

    Returns:
        records: list of dictionaries with keys
            - level: tower level (0-based)
            - genus: exact genus of the level function field
            - pole_orders: mapping 'x{i}' -> v_{Pinf}(x_i) as positive pole orders
        final_field: top-level function field
        x_list: generators x0, x1, ..., x_n_steps
        Pinf: lifted infinity place at the top level
    """
    Fq2 = GF(q ** base_field_degree, name="b")
    F = FunctionField(Fq2["x0"].fraction_field())
    x0 = F.gen()

    Pinf = _base_infty_place(F)
    x_list = [x0]
    records = [
        {
            "level": 0,
            "genus": F.genus(),
            "pole_orders": _w3_pole_profile(Pinf, [("x0", x0)]),
        }
    ]

    for i in range(n_steps):
        xi = x_list[-1]
        R = PolynomialRing(F, names=("Y",))
        (Y,) = R.gens()
        f = (Y - 1) * xi - (Y ** q) * (xi ** q - 1)
        F = F.extension(f, names=(f"x{i+1}",))
        x_next = F.gen()
        x_list.append(x_next)

        Pinf = _lift_place_choose_pole(F, Pinf, x0)

        pole_orders = _w3_pole_profile(
            Pinf, [(f"x{j}", x_list[j]) for j in range(i + 2)]
        )
        records.append(
            {
                "level": i + 1,
                "genus": F.genus(),
                "pole_orders": pole_orders,
            }
        )

    assert_only_pole_at(x0, Pinf)
    return records, F, x_list, Pinf


# --------------------------------------------------------------------------- #
# 9) (C) Norm–Trace curve points + evaluation code + trace-code
# --------------------------------------------------------------------------- #

def norm_trace_points(q, r):
    """
    Enumerate affine points on Norm–Trace curve over F = GF(q^r):
      N(x) = Tr(y)
    """
    F = GF(q ** r, name="c")
    K, emb = F.subfield(q)

    buckets = {K(t): [] for t in K}
    for y in F:
        buckets[K(ff_trace(y, q, r))].append(y)

    pts = []
    for x in F:
        nx = K(ff_norm(x, q, r))
        for y in buckets[nx]:
            pts.append((x, y))
    return F, K, pts


def norm_trace_eval_generator_matrix(q, r, dx, dy):
    """
    Evaluation code from monomials x^i y^j with 0<=i<=dx, 0<=j<=dy
    on all affine rational points of Norm–Trace curve.
    """
    F, K, pts = norm_trace_points(q, r)
    mon_exps = [(i, j) for i in range(dx + 1) for j in range(dy + 1)]
    G = matrix(F, len(mon_exps), len(pts))
    for row, (i, j) in enumerate(mon_exps):
        for col, (x, y) in enumerate(pts):
            G[row, col] = (x ** i) * (y ** j)
    return F, pts, mon_exps, G


def norm_trace_trace_code_generator_matrix(q, r, dx, dy):
    """
    Build trace-code generator matrix over GF(q) from the extension-field eval matrix:
      GT[i,j] = Tr_{GF(q^r)/GF(q)}( G[i,j] )
    """
    F, pts, mon_exps, G = norm_trace_eval_generator_matrix(q, r, dx, dy)
    K = GF(q)
    Gt = matrix_trace(G, q, r, K=K)
    return F, K, pts, mon_exps, G, Gt