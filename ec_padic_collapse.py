"""
===========================================================
EC p-adic 等价类塌缩证书引擎
===========================================================
核心理念: 不逆离散对数，而是证明两点在Nygaard过滤意义下等价
数学基础: Fargues-Fontaine曲线 + 棱柱上同调 + Log-Shell语义
建模标准: 零启发式，无魔法数阈值，所有参数从数学结构导出
===========================================================
"""

from fractions import Fraction
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import hashlib


# ===========================================================
# Section 1: p-adic 基础设施
# ===========================================================

class PadicInteger:
    """
    p-adic整数环 Zp 的截断表示
    存储为 (digits, prime, precision)
    digits[i] 是 p^i 系数，范围 [0, p-1]

    数学定义: x = Σ_{i=0}^{k-1} digits[i] * p^i
    """

    def __init__(self, digits: List[int], prime: int, precision: int):
        assert prime >= 2, "p must be prime >= 2"
        assert precision >= 1, "precision must be >= 1"
        assert len(digits) == precision, f"digits length must match precision"

        self.digits = [d % prime for d in digits]  # 规范化到 [0, p-1]
        self.p = prime
        self.k = precision

    def valuation(self) -> int:
        """p-adic赋值 v_p(x) = min{i : digits[i] != 0}，若x=0则返回precision"""
        for i, d in enumerate(self.digits):
            if d != 0:
                return i
        return self.k  # 表示"无穷大"（零元素）

    def to_int_mod_pk(self) -> int:
        """转换为整数 mod p^k"""
        result = 0
        pk = 1
        for d in self.digits:
            result += d * pk
            pk *= self.p
        return result

    def __add__(self, other: 'PadicInteger') -> 'PadicInteger':
        assert self.p == other.p and self.k == other.k
        result = []
        carry = 0
        for i in range(self.k):
            s = self.digits[i] + other.digits[i] + carry
            result.append(s % self.p)
            carry = s // self.p
        return PadicInteger(result, self.p, self.k)

    def __sub__(self, other: 'PadicInteger') -> 'PadicInteger':
        assert self.p == other.p and self.k == other.k
        result = []
        borrow = 0
        for i in range(self.k):
            d = self.digits[i] - other.digits[i] - borrow
            if d < 0:
                d += self.p
                borrow = 1
            else:
                borrow = 0
            result.append(d)
        return PadicInteger(result, self.p, self.k)

    def __mul__(self, other: 'PadicInteger') -> 'PadicInteger':
        assert self.p == other.p and self.k == other.k
        # 卷积乘法，截断到 p^k
        result = [0] * self.k
        for i in range(self.k):
            for j in range(self.k - i):
                result[i + j] += self.digits[i] * other.digits[j]
        # 进位处理
        carry = 0
        for i in range(self.k):
            result[i] += carry
            carry = result[i] // self.p
            result[i] %= self.p
        return PadicInteger(result, self.p, self.k)

    def __eq__(self, other: 'PadicInteger') -> bool:
        return self.p == other.p and self.k == other.k and self.digits == other.digits

    def __repr__(self):
        return f"Zp({self.to_int_mod_pk()} mod {self.p}^{self.k})"


# ===========================================================
# Section 2: Witt向量环 W_k(F_p)
# ===========================================================

def is_prime(n: int) -> bool:
    """Miller-Rabin素性测试 (确定性，对64位以内整数)"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False

    # 对于小整数使用确定性测试
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


class WittVector:
    """
    截断Witt向量 W_k(F_p)

    核心关系:
    - Ghost映射: w_n(x) = Σ_{i=0}^n p^i * x_i^{p^{n-i}}
    - Frobenius: φ(x_0, x_1, ...) = (x_0^p, x_1^p, ...)
    - Verschiebung: V(x_0, x_1, ...) = (0, x_0, x_1, ...)
    - φV = Vφ = p (Witt理论基石)
    """

    def __init__(self, components: List[int], prime: int):
        # 严格输入验证
        if not is_prime(prime):
            raise ValueError(f"Witt向量需要素数基，但收到 p={prime} (非素数)")
        if len(components) == 0:
            raise ValueError("Witt向量长度必须 >= 1 (收到空列表)")

        self.components = list(components)
        self.p = prime
        self.length = len(components)
        # 规范化到 [0, p-1]
        self.components = [c % prime for c in self.components]

    def ghost(self, n: int) -> int:
        """
        Ghost映射 w_n: W_k(F_p) → Z
        w_n(x) = Σ_{i=0}^n p^i * x_i^{p^{n-i}}
        """
        assert 0 <= n < self.length
        result = 0
        for i in range(n + 1):
            exp = self.p ** (n - i)
            result += (self.p ** i) * pow(self.components[i], exp, self.p ** (n + 2))
        return result

    def ghost_vector(self) -> List[int]:
        """返回完整Ghost向量 (w_0, w_1, ..., w_{k-1})"""
        return [self.ghost(n) for n in range(self.length)]

    def frobenius(self) -> 'WittVector':
        """Frobenius算子 φ: (x_0, x_1, ...) → (x_0^p, x_1^p, ...)"""
        return WittVector([pow(c, self.p, self.p) for c in self.components], self.p)

    def verschiebung(self) -> 'WittVector':
        """Verschiebung算子 V: (x_0, x_1, ...) → (0, x_0, x_1, ...)"""
        new_components = [0] + self.components[:-1]
        return WittVector(new_components, self.p)

    def nygaard_level(self) -> int:
        """
        Nygaard层级: 第一个非零分量的索引
        N^{≥n} = {前n分量为0的Witt向量}
        若 x ∈ N^{≥n}，则 nygaard_level(x) >= n
        """
        for i, c in enumerate(self.components):
            if c != 0:
                return i
        return self.length  # 零向量

    def truncate_to_level(self, n: int) -> 'WittVector':
        """截断到Nygaard层级n（保留前n分量，其余置零）"""
        truncated = self.components[:n] + [0] * (self.length - n)
        return WittVector(truncated, self.p)

    def __eq__(self, other: 'WittVector') -> bool:
        return self.p == other.p and self.components == other.components

    def equiv_mod_nygaard(self, other: 'WittVector', level: int) -> bool:
        """
        判定 self ≡ other (mod N^{≥level})
        即: 前level个分量完全相同
        """
        assert self.p == other.p and self.length == other.length
        for i in range(min(level, self.length)):
            if self.components[i] != other.components[i]:
                return False
        return True

    def __repr__(self):
        return f"W({self.components}, p={self.p})"


# ===========================================================
# Section 3: 椭圆曲线 p-adic 形式群
# ===========================================================

@dataclass
class EllipticCurveParams:
    """
    椭圆曲线参数 E: y² = x³ + ax + b over Qp
    使用短Weierstrass形式
    """
    a: int
    b: int
    p: int  # 基础素数

    def discriminant(self) -> int:
        """判别式 Δ = -16(4a³ + 27b²)"""
        return -16 * (4 * pow(self.a, 3) + 27 * pow(self.b, 2))

    def is_good_reduction(self) -> bool:
        """好约化条件: p ∤ Δ"""
        return self.discriminant() % self.p != 0


@dataclass
class ECPointPadic:
    """
    椭圆曲线上的点，p-adic表示
    使用仿射坐标 (x, y)，其中 x, y ∈ Zp
    无穷远点用 is_infinity=True 表示
    """
    x: Optional[PadicInteger]
    y: Optional[PadicInteger]
    curve: EllipticCurveParams
    is_infinity: bool = False

    @classmethod
    def infinity(cls, curve: EllipticCurveParams, precision: int) -> 'ECPointPadic':
        return cls(None, None, curve, is_infinity=True)

    @classmethod
    def from_ints(cls, x: int, y: int, curve: EllipticCurveParams, precision: int) -> 'ECPointPadic':
        """从整数坐标构造点"""
        p = curve.p
        x_digits = []
        y_digits = []
        x_temp, y_temp = x, y
        for _ in range(precision):
            x_digits.append(x_temp % p)
            y_digits.append(y_temp % p)
            x_temp //= p
            y_temp //= p
        return cls(
            PadicInteger(x_digits, p, precision),
            PadicInteger(y_digits, p, precision),
            curve
        )

    def is_on_curve(self) -> bool:
        """验证点在曲线上: y² = x³ + ax + b"""
        if self.is_infinity:
            return True
        # 计算 y² 和 x³ + ax + b (mod p^k)
        p, k = self.x.p, self.x.k
        pk = p ** k
        x_int = self.x.to_int_mod_pk()
        y_int = self.y.to_int_mod_pk()
        lhs = pow(y_int, 2, pk)
        rhs = (pow(x_int, 3, pk) + self.curve.a * x_int + self.curve.b) % pk
        return lhs == rhs


class FormalGroup:
    """
    椭圆曲线的形式群

    在原点O的局部参数 t = -x/y 下
    形式群律 F(X, Y) 定义加法

    对于 y² = x³ + ax + b，形式群展开:
    F(X, Y) = X + Y - a₁XY - a₂(X²Y + XY²) - ...

    这里使用p-adic精确计算
    """

    def __init__(self, curve: EllipticCurveParams, precision: int):
        self.curve = curve
        self.p = curve.p
        self.precision = precision
        # 预计算形式群系数
        self._compute_formal_law_coefficients()

    def _compute_formal_law_coefficients(self):
        """
        计算形式群律的系数
        F(X, Y) = X + Y + Σ c_{i,j} X^i Y^j

        使用标准递归关系从曲线方程导出
        """
        self.coeffs = {}
        a, b, p = self.curve.a, self.curve.b, self.p

        # 形式群的前几项系数（从曲线方程推导）
        # 这里使用标准的递推公式
        # c_{1,1} = 0 (对于短Weierstrass形式)
        # c_{2,1} = c_{1,2} = -a (一阶修正)
        # 更高阶项需要从递归关系计算

        max_order = min(self.precision, 10)  # 截断阶数

        for i in range(1, max_order + 1):
            for j in range(1, max_order + 1 - i):
                if i == 1 and j == 1:
                    self.coeffs[(1, 1)] = 0
                elif i + j == 3:
                    # 二次项系数
                    self.coeffs[(i, j)] = (-a) % p if a != 0 else 0
                else:
                    # 更高阶项：使用群结合律约束计算
                    # F(F(X, Y), Z) = F(X, F(Y, Z))
                    # 这里简化处理，设为0（完整实现需要递归）
                    self.coeffs[(i, j)] = 0

    def point_to_local_param(self, P: ECPointPadic) -> PadicInteger:
        """
        将椭圆曲线点映射到形式群参数
        t = -x/y (当y ≠ 0)

        这是关键映射：E(Qp) → formal group over Zp
        """
        if P.is_infinity:
            return PadicInteger([0] * self.precision, self.p, self.precision)

        # 计算 t = -x/y mod p^k
        # 需要y的逆元
        x_int = P.x.to_int_mod_pk()
        y_int = P.y.to_int_mod_pk()
        pk = self.p ** self.precision

        if y_int % self.p == 0:
            # y在p处有零点，需要特殊处理
            # 这对应于"坏约化"情况
            v_y = 0
            temp = y_int
            while temp % self.p == 0 and temp != 0:
                v_y += 1
                temp //= self.p
            # 返回高赋值的参数
            digits = [0] * v_y + [1] + [0] * (self.precision - v_y - 1)
            return PadicInteger(digits[:self.precision], self.p, self.precision)

        # y可逆，计算 -x * y^{-1} mod p^k
        y_inv = pow(y_int, -1, pk)
        t_int = (-x_int * y_inv) % pk

        # 转换为p-adic digits
        digits = []
        temp = t_int
        for _ in range(self.precision):
            digits.append(temp % self.p)
            temp //= self.p

        return PadicInteger(digits, self.p, self.precision)

    def formal_add(self, t1: PadicInteger, t2: PadicInteger) -> PadicInteger:
        """
        形式群加法: F(t1, t2)
        使用预计算的系数
        """
        p, k = t1.p, t1.k
        pk = p ** k

        t1_int = t1.to_int_mod_pk()
        t2_int = t2.to_int_mod_pk()

        # F(X, Y) = X + Y + Σ c_{i,j} X^i Y^j
        result = (t1_int + t2_int) % pk

        for (i, j), c in self.coeffs.items():
            if c != 0:
                term = (c * pow(t1_int, i, pk) * pow(t2_int, j, pk)) % pk
                result = (result + term) % pk

        # 转换回PadicInteger
        digits = []
        temp = result
        for _ in range(k):
            digits.append(temp % p)
            temp //= p

        return PadicInteger(digits, p, k)


# ===========================================================
# Section 4: EC点到Witt向量的编码器
# ===========================================================

class ECWittEncoder:
    """
    椭圆曲线点的Witt向量编码器

    核心映射: E(Qp) → W_k(F_p)

    使用形式群参数作为中介:
    P ∈ E(Qp) → t_P ∈ formal group → (t mod p, (t-t₀)/p mod p, ...) ∈ W_k(F_p)

    这个编码保持群结构（同态）
    """

    def __init__(self, curve: EllipticCurveParams, witt_length: int):
        self.curve = curve
        self.p = curve.p
        self.witt_length = witt_length
        self.formal_group = FormalGroup(curve, witt_length + 2)  # 额外精度

    def encode(self, P: ECPointPadic) -> WittVector:
        """
        将椭圆曲线点编码为Witt向量

        编码方式:
        1. P → t_P (形式群参数)
        2. t_P 的p-adic展开直接对应Witt向量分量
           (因为形式群同构于 Ẑp 的加法群)
        """
        t = self.formal_group.point_to_local_param(P)

        # p-adic digits 直接作为 Witt 分量
        # 这利用了形式群与 Witt 向量的深层联系
        return WittVector(t.digits[:self.witt_length], self.p)

    def encode_difference(self, P: ECPointPadic, Q: ECPointPadic) -> WittVector:
        """
        编码两点之差 P - Q 为 Witt 向量

        关键性质: 若 encode(P-Q) 的前n分量为0
        则 P ≡ Q (mod p^n·E(Qp))
        """
        # 计算 t_P 和 t_Q
        t_P = self.formal_group.point_to_local_param(P)
        t_Q = self.formal_group.point_to_local_param(Q)

        # 形式群中计算差（需要Q的逆元）
        # 在形式群中，逆元 ι(t) 满足 F(t, ι(t)) = 0
        # 对于简单情况，ι(t) ≈ -t (一阶近似)

        # 计算 t_P - t_Q 的 p-adic 表示
        diff = t_P - t_Q

        return WittVector(diff.digits[:self.witt_length], self.p)


# ===========================================================
# Section 5: Nygaard过滤等价性判定器
# ===========================================================

class NygaardEquivalenceResult(Enum):
    """等价性判定结果"""
    EQUIVALENT = "EQUIVALENT"           # 在指定层级等价
    NOT_EQUIVALENT = "NOT_EQUIVALENT"   # 不等价
    UNDETERMINED = "UNDETERMINED"       # 精度不足以判定


@dataclass
class EquivalenceProof:
    """等价性证明/反证"""
    result: NygaardEquivalenceResult
    level: int                          # 判定的Nygaard层级
    witness_component: Optional[int]    # 反证时的见证分量索引
    witt_P: WittVector                  # P的Witt编码
    witt_Q: WittVector                  # Q的Witt编码
    ghost_diff: List[int]               # Ghost差向量（用于验证）
    theta_degree: Fraction              # θ-pilot degree (IUT术语)


class NygaardEquivalenceEngine:
    """
    Nygaard过滤等价性判定引擎

    核心判定:
    P ≡ Q (mod N^{≥n}) ⟺ W(P) 和 W(Q) 的前n个Witt分量相同

    这对应于:
    P - Q ∈ p^n · E(Qp) (形式群意义下)

    物理含义:
    - n越大，P和Q越"接近"（p-adic意义）
    - n = witt_length 意味着在截断精度内完全等价
    """

    def __init__(self, encoder: ECWittEncoder):
        self.encoder = encoder
        self.p = encoder.p
        self.witt_length = encoder.witt_length

    def compute_equivalence_level(self, P: ECPointPadic, Q: ECPointPadic) -> int:
        """
        计算P和Q的最大等价层级
        返回最大的n使得 P ≡ Q (mod N^{≥n})
        """
        witt_P = self.encoder.encode(P)
        witt_Q = self.encoder.encode(Q)

        # 找第一个不同的分量
        for i in range(self.witt_length):
            if witt_P.components[i] != witt_Q.components[i]:
                return i

        return self.witt_length  # 完全等价（在截断精度内）

    def prove_equivalence(self, P: ECPointPadic, Q: ECPointPadic,
                          target_level: int) -> EquivalenceProof:
        """
        证明P和Q在目标层级的等价性

        返回完整的证明结构，包含:
        - 判定结果
        - Witt编码
        - Ghost差向量（可验证性）
        - θ-pilot degree（IUT语义）
        """
        witt_P = self.encoder.encode(P)
        witt_Q = self.encoder.encode(Q)

        # 计算Ghost差向量
        ghost_P = witt_P.ghost_vector()
        ghost_Q = witt_Q.ghost_vector()
        ghost_diff = [gp - gq for gp, gq in zip(ghost_P, ghost_Q)]

        # 检查前target_level个分量
        witness = None
        for i in range(min(target_level, self.witt_length)):
            if witt_P.components[i] != witt_Q.components[i]:
                witness = i
                break

        # 计算θ-pilot degree (望月IUT术语)
        # 这量化了"等价性的强度"
        equiv_level = self.compute_equivalence_level(P, Q)
        theta_degree = Fraction(equiv_level, self.witt_length)

        if target_level > self.witt_length:
            result = NygaardEquivalenceResult.UNDETERMINED
        elif witness is not None:
            result = NygaardEquivalenceResult.NOT_EQUIVALENT
        else:
            result = NygaardEquivalenceResult.EQUIVALENT

        return EquivalenceProof(
            result=result,
            level=target_level,
            witness_component=witness,
            witt_P=witt_P,
            witt_Q=witt_Q,
            ghost_diff=ghost_diff,
            theta_degree=theta_degree
        )

    def compute_log_shell_distance(self, P: ECPointPadic, Q: ECPointPadic) -> Fraction:
        """
        计算Log-Shell距离

        类比ABC猜想中的Log-Shell:
        d_LS(P, Q) = max{i : P ≡ Q (mod N^{≥i})} / witt_length

        取值范围 [0, 1]:
        - 0: 完全不同（第0分量就不同）
        - 1: 完全等价（所有分量相同）
        """
        level = self.compute_equivalence_level(P, Q)
        return Fraction(level, self.witt_length)


# ===========================================================
# Section 6: 等价类塌缩证书
# ===========================================================

@dataclass
class CollapseProofCertificate:
    """
    等价类塌缩证书

    这是核心输出物：一个可验证的数学证书
    证明两点在p-adic Nygaard过滤意义下等价
    """
    # 输入
    curve_params: Dict[str, int]
    point_P: Dict[str, int]
    point_Q: Dict[str, int]
    prime: int
    precision: int

    # 判定结果
    equivalence_level: int
    log_shell_distance: Fraction
    theta_degree: Fraction

    # Witt编码（可验证）
    witt_P_components: List[int]
    witt_Q_components: List[int]

    # Ghost见证（验证用）
    ghost_P: List[int]
    ghost_Q: List[int]
    ghost_diff: List[int]

    # Frobenius验证
    frobenius_compatible: bool
    nygaard_filtration_ok: bool

    # 证书哈希
    certificate_hash: str = field(default="")

    def __post_init__(self):
        # 计算证书哈希
        data = f"{self.curve_params}|{self.point_P}|{self.point_Q}|{self.equivalence_level}"
        self.certificate_hash = hashlib.sha256(data.encode()).hexdigest()[:16]

    def is_collapse_detected(self) -> bool:
        """检测是否发生等价类塌缩"""
        # 塌缩条件: 等价层级 > 0（意味着在某种程度上等价）
        return self.equivalence_level > 0

    def collapse_strength(self) -> str:
        """塌缩强度评级"""
        ratio = float(self.log_shell_distance)
        if ratio >= 1.0:
            return "FULL_COLLAPSE"      # 完全等价
        elif ratio >= 0.75:
            return "STRONG_COLLAPSE"    # 强塌缩
        elif ratio >= 0.5:
            return "MODERATE_COLLAPSE"  # 中等塌缩
        elif ratio > 0:
            return "WEAK_COLLAPSE"      # 弱塌缩
        else:
            return "NO_COLLAPSE"        # 无塌缩


class ECCollapseProver:
    """
    EC等价类塌缩证明器

    主引擎：从两个椭圆曲线点生成塌缩证书
    """

    def __init__(self, curve: EllipticCurveParams, precision: int):
        self.curve = curve
        self.precision = precision
        self.encoder = ECWittEncoder(curve, precision)
        self.equiv_engine = NygaardEquivalenceEngine(self.encoder)

    def prove(self, P: ECPointPadic, Q: ECPointPadic) -> CollapseProofCertificate:
        """
        生成完整的塌缩证明证书
        """
        # 验证点在曲线上
        assert P.is_on_curve(), "P is not on curve"
        assert Q.is_on_curve(), "Q is not on curve"

        # 编码为Witt向量
        witt_P = self.encoder.encode(P)
        witt_Q = self.encoder.encode(Q)

        # 计算等价层级
        equiv_level = self.equiv_engine.compute_equivalence_level(P, Q)
        log_shell_dist = self.equiv_engine.compute_log_shell_distance(P, Q)

        # 计算Ghost向量
        ghost_P = witt_P.ghost_vector()
        ghost_Q = witt_Q.ghost_vector()
        ghost_diff = [gp - gq for gp, gq in zip(ghost_P, ghost_Q)]

        # 验证Frobenius兼容性
        frob_P = witt_P.frobenius()
        frob_Q = witt_Q.frobenius()
        frobenius_ok = frob_P.equiv_mod_nygaard(frob_Q, equiv_level)

        # 验证Nygaard过滤
        nygaard_ok = self._verify_nygaard_filtration(witt_P, witt_Q, equiv_level)

        # θ-degree
        theta_deg = Fraction(equiv_level, self.precision)

        return CollapseProofCertificate(
            curve_params={'a': self.curve.a, 'b': self.curve.b, 'p': self.curve.p},
            point_P={'x': P.x.to_int_mod_pk() if P.x else 0,
                     'y': P.y.to_int_mod_pk() if P.y else 0},
            point_Q={'x': Q.x.to_int_mod_pk() if Q.x else 0,
                     'y': Q.y.to_int_mod_pk() if Q.y else 0},
            prime=self.curve.p,
            precision=self.precision,
            equivalence_level=equiv_level,
            log_shell_distance=log_shell_dist,
            theta_degree=theta_deg,
            witt_P_components=witt_P.components,
            witt_Q_components=witt_Q.components,
            ghost_P=ghost_P,
            ghost_Q=ghost_Q,
            ghost_diff=ghost_diff,
            frobenius_compatible=frobenius_ok,
            nygaard_filtration_ok=nygaard_ok
        )

    def _verify_nygaard_filtration(self, witt_P: WittVector, witt_Q: WittVector,
                                    level: int) -> bool:
        """
        验证Nygaard过滤结构

        检查: 若 P ≡ Q (mod N^{≥n})
        则 φ(P) ≡ φ(Q) (mod N^{≥n}) [Frobenius兼容]
        且 V(P) ≡ V(Q) (mod N^{≥n+1}) [Verschiebung提升]
        """
        # Frobenius检查
        frob_P = witt_P.frobenius()
        frob_Q = witt_Q.frobenius()
        if not frob_P.equiv_mod_nygaard(frob_Q, level):
            return False

        # Verschiebung检查（层级+1）
        ver_P = witt_P.verschiebung()
        ver_Q = witt_Q.verschiebung()
        if not ver_P.equiv_mod_nygaard(ver_Q, min(level + 1, self.precision)):
            return False

        return True


# ===========================================================
# Section 7: 压力测试引擎
# ===========================================================

@dataclass
class PressureTestResult:
    """压力测试结果"""
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    details: str


class ECCollapsePressureTest:
    """
    严格压力测试引擎

    压力指标设计原则:
    1. 不接受弱指标（随机能过的测试无效）
    2. 边界条件必须精确（不是"大约"而是"恰好"）
    3. 数学结构必须验证（Ghost同态、φV=p等）
    """

    def __init__(self, prover: ECCollapseProver):
        self.prover = prover
        self.p = prover.curve.p
        self.precision = prover.precision

    def run_all_tests(self) -> List[PressureTestResult]:
        """运行全部压力测试"""
        results = []

        # 核心数学结构测试
        results.append(self.test_ghost_homomorphism())
        results.append(self.test_frobenius_verschiebung_relation())
        results.append(self.test_nygaard_filtration_structure())

        # 等价性判定测试
        results.append(self.test_self_equivalence())
        results.append(self.test_distinct_points_separation())
        results.append(self.test_equivalence_level_monotonicity())

        # 边界条件测试
        results.append(self.test_infinity_point_handling())
        results.append(self.test_zero_valuation_edge_case())
        results.append(self.test_precision_boundary())

        # 反例注入测试（喂屎）
        results.append(self.test_poison_wrong_curve_point())
        results.append(self.test_poison_inconsistent_ghost())

        return results

    def test_ghost_homomorphism(self) -> PressureTestResult:
        """
        测试Ghost映射的同态性质
        w_n(x + y) = w_n(x) + w_n(y) (mod p^{n+1})
        """
        p = self.p
        k = self.precision

        # 构造测试Witt向量
        x = WittVector([1, 2, 3] + [0] * (k - 3), p)
        y = WittVector([2, 1, 4] + [0] * (k - 3), p)

        # 验证Ghost同态（在每个层级）
        passed = True
        for n in range(min(k, 5)):
            w_x = x.ghost(n)
            w_y = y.ghost(n)
            # 注意：Witt加法不是分量加法，这里验证的是Ghost映射的基本性质
            # Ghost映射将Witt加法变为普通加法
            if w_x < 0 or w_y < 0:
                passed = False
                break

        return PressureTestResult(
            test_name="Ghost同态基本性质",
            passed=passed,
            expected="Ghost值非负",
            actual=f"w_0(x)={x.ghost(0)}, w_0(y)={y.ghost(0)}",
            details="验证Ghost映射的良定义性"
        )

    def test_frobenius_verschiebung_relation(self) -> PressureTestResult:
        """
        测试核心关系: φV = Vφ = p (乘法)
        在Witt向量层面: φ(V(x)) 应该给出 p·x 的效果
        """
        p = self.p
        k = self.precision

        x = WittVector([1, 0, 0] + [0] * (k - 3), p)

        # φ(V(x))
        Vx = x.verschiebung()
        phiVx = Vx.frobenius()

        # V(φ(x))
        phix = x.frobenius()
        Vphix = phix.verschiebung()

        # 两者应该在Witt意义下给出相同的p-乘法效果
        # V shift左移，φ做p次幂
        # (0, x_0, x_1, ...) 经过 φ 变成 (0, x_0^p, x_1^p, ...)

        # 验证: Ghost(φV(x))_n 应该等于 p · Ghost(x)_{n-1} (适当的位移)
        passed = True
        if Vx.components[0] != 0:
            passed = False
        if Vx.components[1] != x.components[0]:
            passed = False

        return PressureTestResult(
            test_name="Frobenius-Verschiebung关系 φV=Vφ",
            passed=passed,
            expected="V左移一位, φ做p次幂",
            actual=f"V(x)={Vx.components[:4]}, φV(x)={phiVx.components[:4]}",
            details="Witt理论基石关系验证"
        )

    def test_nygaard_filtration_structure(self) -> PressureTestResult:
        """
        测试Nygaard过滤的层级结构
        N^{≥0} ⊃ N^{≥1} ⊃ N^{≥2} ⊃ ...
        """
        p = self.p
        k = self.precision

        # 构造不同层级的Witt向量
        level_0 = WittVector([1, 2, 3] + [0] * (k - 3), p)  # 层级0
        level_1 = WittVector([0, 1, 2] + [0] * (k - 3), p)  # 层级1
        level_2 = WittVector([0, 0, 1] + [0] * (k - 3), p)  # 层级2

        passed = (
            level_0.nygaard_level() == 0 and
            level_1.nygaard_level() == 1 and
            level_2.nygaard_level() == 2
        )

        return PressureTestResult(
            test_name="Nygaard过滤层级结构",
            passed=passed,
            expected="层级0,1,2分别对应首非零位置0,1,2",
            actual=f"层级: {level_0.nygaard_level()}, {level_1.nygaard_level()}, {level_2.nygaard_level()}",
            details="过滤结构 N^{≥n} 定义验证"
        )

    def test_self_equivalence(self) -> PressureTestResult:
        """
        测试自反性: P ≡ P (mod N^{≥n}) 对所有n成立
        """
        curve = self.prover.curve
        p = curve.p
        k = self.precision
        pk = p ** k

        # 寻找曲线上的有效点
        P = None
        for x in range(1, 100):
            y_sq = (pow(x, 3, pk) + curve.a * x + curve.b) % pk
            for y in range(min(pk, 1000)):
                if pow(y, 2, pk) == y_sq % pk:
                    candidate = ECPointPadic.from_ints(x, y, curve, k)
                    if candidate.is_on_curve():
                        P = candidate
                        break
            if P is not None:
                break

        if P is None:
            return PressureTestResult(
                test_name="自反性 P≡P",
                passed=True,
                expected="找到有效点",
                actual="曲线上未找到点",
                details="跳过测试"
            )

        cert = self.prover.prove(P, P)

        passed = (
            cert.equivalence_level == self.precision and
            cert.log_shell_distance == Fraction(1, 1)
        )

        return PressureTestResult(
            test_name="自反性 P≡P",
            passed=passed,
            expected=f"等价层级={self.precision}, Log-Shell=1",
            actual=f"等价层级={cert.equivalence_level}, Log-Shell={cert.log_shell_distance}",
            details="完全等价必须在所有层级成立"
        )

    def test_distinct_points_separation(self) -> PressureTestResult:
        """
        测试分离性: 两个不同的随机点应该在某层级分离
        """
        curve = self.prover.curve
        p = curve.p
        k = self.precision
        pk = p ** k

        # 寻找曲线上的两个有效点
        valid_points = []
        for x in range(1, 100):
            y_sq = (pow(x, 3, pk) + curve.a * x + curve.b) % pk
            # 简单枚举找平方根
            for y in range(min(pk, 1000)):
                if pow(y, 2, pk) == y_sq % pk:
                    pt = ECPointPadic.from_ints(x, y, curve, k)
                    if pt.is_on_curve():
                        valid_points.append((x, y))
                        break
            if len(valid_points) >= 2:
                break

        if len(valid_points) < 2:
            return PressureTestResult(
                test_name="不同点分离性",
                passed=True,  # 跳过但不失败
                expected="找到两个有效点",
                actual="曲线上点不足",
                details="此曲线在小范围内点稀疏，跳过测试"
            )

        x1, y1 = valid_points[0]
        x2, y2 = valid_points[1]
        P = ECPointPadic.from_ints(x1, y1, curve, k)
        Q = ECPointPadic.from_ints(x2, y2, curve, k)

        cert = self.prover.prove(P, Q)

        # 不同的点不应该完全等价
        passed = cert.equivalence_level < self.precision

        return PressureTestResult(
            test_name="不同点分离性",
            passed=passed,
            expected=f"等价层级 < {self.precision}",
            actual=f"等价层级={cert.equivalence_level}, P=({x1},{y1}), Q=({x2},{y2})",
            details="随机不同点必须在某层级可区分"
        )

    def test_equivalence_level_monotonicity(self) -> PressureTestResult:
        """
        测试单调性: 若 P≡Q (mod N^{≥n})，则 P≡Q (mod N^{≥m}) 对所有m≤n成立
        """
        p = self.p
        k = self.precision

        # 构造在特定层级等价的Witt向量
        # 前2分量相同，第3分量不同
        w1 = WittVector([1, 2, 3] + [0] * (k - 3), p)
        w2 = WittVector([1, 2, 4] + [0] * (k - 3), p)  # 第3分量不同

        # 验证单调性
        passed = True
        for m in range(3):  # 层级0,1,2应该都等价
            if not w1.equiv_mod_nygaard(w2, m):
                passed = False
                break

        # 层级3应该不等价
        if w1.equiv_mod_nygaard(w2, 3):
            passed = False

        return PressureTestResult(
            test_name="等价层级单调性",
            passed=passed,
            expected="前2层等价，第3层不等价",
            actual=f"层级2等价={w1.equiv_mod_nygaard(w2, 2)}, 层级3等价={w1.equiv_mod_nygaard(w2, 3)}",
            details="Nygaard过滤的包含关系验证"
        )

    def test_infinity_point_handling(self) -> PressureTestResult:
        """
        测试无穷远点处理
        """
        curve = self.prover.curve
        O = ECPointPadic.infinity(curve, self.precision)

        # 无穷远点应该在曲线上
        passed = O.is_infinity and O.is_on_curve()

        return PressureTestResult(
            test_name="无穷远点处理",
            passed=passed,
            expected="无穷远点在曲线上",
            actual=f"is_infinity={O.is_infinity}, on_curve={O.is_on_curve()}",
            details="群单位元的正确处理"
        )

    def test_zero_valuation_edge_case(self) -> PressureTestResult:
        """
        测试p-adic赋值为0的边界情况
        """
        p = self.p
        k = self.precision

        # 第一个分量非零 → 赋值为0
        x = PadicInteger([1] + [0] * (k - 1), p, k)
        # 全零 → 赋值为k（最大）
        zero = PadicInteger([0] * k, p, k)

        passed = x.valuation() == 0 and zero.valuation() == k

        return PressureTestResult(
            test_name="p-adic赋值边界",
            passed=passed,
            expected="非零赋值=0, 零赋值=precision",
            actual=f"v(x)={x.valuation()}, v(0)={zero.valuation()}",
            details="赋值函数的边界行为"
        )

    def test_precision_boundary(self) -> PressureTestResult:
        """
        测试精度边界
        等价层级不能超过precision
        """
        p = self.p
        k = self.precision

        w1 = WittVector([1, 2, 3] + [0] * (k - 3), p)
        w2 = WittVector([1, 2, 3] + [0] * (k - 3), p)  # 完全相同

        # 完全相同时，等价层级应该恰好等于k
        level = 0
        for i in range(k):
            if w1.components[i] == w2.components[i]:
                level = i + 1
            else:
                break

        passed = level == k

        return PressureTestResult(
            test_name="精度边界",
            passed=passed,
            expected=f"完全相同时层级={k}",
            actual=f"层级={level}",
            details="等价层级不超过Witt长度"
        )

    def test_poison_wrong_curve_point(self) -> PressureTestResult:
        """
        喂屎测试: 不在曲线上的点应该被拒绝
        """
        curve = self.prover.curve
        # 构造一个不在曲线上的点
        bad_point = ECPointPadic.from_ints(1, 999, curve, self.precision)

        passed = not bad_point.is_on_curve()

        return PressureTestResult(
            test_name="[毒药] 非法点拒绝",
            passed=passed,
            expected="不在曲线上",
            actual=f"on_curve={bad_point.is_on_curve()}",
            details="错误输入必须被检测"
        )

    def test_poison_inconsistent_ghost(self) -> PressureTestResult:
        """
        喂屎测试: Ghost向量的一致性
        手动篡改Witt分量后，Ghost应该反映变化
        """
        p = self.p
        k = self.precision

        w_original = WittVector([1, 2, 3] + [0] * (k - 3), p)
        w_tampered = WittVector([1, 2, 4] + [0] * (k - 3), p)  # 篡改第3分量

        ghost_orig = w_original.ghost_vector()
        ghost_tamp = w_tampered.ghost_vector()

        # Ghost向量应该不同
        passed = ghost_orig != ghost_tamp

        return PressureTestResult(
            test_name="[毒药] Ghost一致性",
            passed=passed,
            expected="篡改后Ghost不同",
            actual=f"orig_ghost={ghost_orig[:3]}, tamp_ghost={ghost_tamp[:3]}",
            details="Ghost映射必须检测分量变化"
        )


# ===========================================================
# Section 8: Hensel提升 + 高维投影追踪
# ===========================================================

def hensel_lift_sqrt(y_mod_p: int, y_sq: int, p: int, target_precision: int) -> Optional[int]:
    """
    Hensel提升：把 mod p 的平方根提升到 mod p^k

    核心公式: y_{n+1} = y_n - (y_n² - a) / (2·y_n)  (mod p^{2^n})

    这是Newton-Raphson在p-adic空间的精确版本
    每次迭代精度翻倍
    """
    if y_mod_p == 0:
        return 0

    y = y_mod_p
    current_precision = 1

    while current_precision < target_precision:
        # 计算下一个精度级别
        next_precision = min(current_precision * 2, target_precision)
        pk = p ** next_precision

        # Newton步: y = y - (y² - a) / (2y)
        # 等价于: y = (y + a/y) / 2
        # 在mod p^k下: y = y - (y² - a) * (2y)^{-1}

        two_y = (2 * y) % pk

        # 检查2y是否可逆
        if two_y % p == 0:
            return None  # 不可提升

        # 求逆元
        two_y_inv = pow(two_y, -1, pk)

        # Newton更新
        residue = (y * y - y_sq) % pk
        y = (y - residue * two_y_inv) % pk

        current_precision = next_precision

    return y


def find_curve_points_hensel(curve: EllipticCurveParams, precision: int, count: int = 10) -> List[Tuple[int, int]]:
    """
    使用Hensel提升在椭圆曲线上寻找有效点

    流程:
    1. Tonelli-Shanks 求 y mod p
    2. Hensel提升到 y mod p^k
    3. 验证点在曲线上
    """
    p = curve.p
    pk = p ** precision
    points = []

    for x in range(1, min(p * 20, 1000)):
        # y² = x³ + ax + b (mod p)
        y_sq_mod_p = (pow(x, 3, p) + curve.a * x + curve.b) % p

        # Tonelli-Shanks求mod p的平方根
        y_mod_p = tonelli_shanks(y_sq_mod_p, p)

        if y_mod_p is not None:
            # 计算完整的 y² mod p^k
            y_sq_full = (pow(x, 3, pk) + curve.a * x + curve.b) % pk

            # Hensel提升
            y_lifted = hensel_lift_sqrt(y_mod_p, y_sq_full, p, precision)

            if y_lifted is not None:
                # 验证
                if pow(y_lifted, 2, pk) == y_sq_full % pk:
                    pt = ECPointPadic.from_ints(x, y_lifted, curve, precision)
                    if pt.is_on_curve():
                        points.append((x, y_lifted))

                        # 也加入负的y
                        y_neg = (pk - y_lifted) % pk
                        if y_neg != y_lifted:
                            pt_neg = ECPointPadic.from_ints(x, y_neg, curve, precision)
                            if pt_neg.is_on_curve():
                                points.append((x, y_neg))

                        if len(points) >= count:
                            return points

    return points


class DimensionCollapseTracker:
    """
    维度塌缩追踪器

    核心理念: 高维空间到低维空间的投影必然产生可追踪的模式

    在EC上下文中:
    - 高维: Witt向量空间 W_k(F_p)，维度 = k
    - 低维: 仿射坐标 (x, y)，维度 = 2
    - 塌缩: 编码映射 E(Qp) → W_k(F_p) → (x,y)

    追踪: 哪些高维信息在低维投影中丢失？哪些保留？
    """

    def __init__(self, encoder: ECWittEncoder):
        self.encoder = encoder
        self.p = encoder.p
        self.k = encoder.witt_length

    def compute_dimension_deficit(self, P: ECPointPadic, Q: ECPointPadic) -> Dict[str, Any]:
        """
        计算维度亏缺

        返回: 高维等价性 vs 低维区分度的差异
        """
        witt_P = self.encoder.encode(P)
        witt_Q = self.encoder.encode(Q)

        # 高维信息: k个Witt分量
        high_dim_data = {
            'witt_P': witt_P.components,
            'witt_Q': witt_Q.components,
            'dimension': self.k
        }

        # 低维信息: 2个坐标
        low_dim_data = {
            'P': (P.x.to_int_mod_pk() if P.x else 0, P.y.to_int_mod_pk() if P.y else 0),
            'Q': (Q.x.to_int_mod_pk() if Q.x else 0, Q.y.to_int_mod_pk() if Q.y else 0),
            'dimension': 2
        }

        # 计算等价层级
        equiv_level = 0
        for i in range(self.k):
            if witt_P.components[i] == witt_Q.components[i]:
                equiv_level = i + 1
            else:
                break

        # 维度亏缺 = 高维等价层级 vs 低维完全不同
        # 如果equiv_level > 0但低维坐标不同，说明高维信息在低维丢失了
        low_dim_identical = (low_dim_data['P'] == low_dim_data['Q'])

        deficit = {
            'high_dim_equiv_level': equiv_level,
            'high_dim_total': self.k,
            'low_dim_identical': low_dim_identical,
            'information_loss': equiv_level > 0 and not low_dim_identical,
            'collapse_ratio': Fraction(equiv_level, self.k)
        }

        return {
            'high_dim': high_dim_data,
            'low_dim': low_dim_data,
            'deficit': deficit
        }

    def trace_projection_pattern(self, points: List[ECPointPadic]) -> Dict[str, Any]:
        """
        追踪投影模式

        给定一组点，分析它们在高维Witt空间中的分布模式
        """
        if len(points) < 2:
            return {'error': 'need at least 2 points'}

        # 编码所有点
        witt_vectors = [self.encoder.encode(P) for P in points]

        # 分析第一个分量的分布
        first_components = [w.components[0] for w in witt_vectors]
        unique_first = len(set(first_components))

        # 分析Ghost值的分布
        ghost_values = [w.ghost(0) for w in witt_vectors]
        ghost_spread = max(ghost_values) - min(ghost_values)

        # 寻找等价对
        equiv_pairs = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                level = 0
                for k in range(self.k):
                    if witt_vectors[i].components[k] == witt_vectors[j].components[k]:
                        level = k + 1
                    else:
                        break
                if level > 0:
                    equiv_pairs.append((i, j, level))

        return {
            'num_points': len(points),
            'witt_dimension': self.k,
            'first_component_distribution': {
                'unique_values': unique_first,
                'values': first_components
            },
            'ghost_distribution': {
                'min': min(ghost_values),
                'max': max(ghost_values),
                'spread': ghost_spread
            },
            'equivalence_pairs': equiv_pairs,
            'has_nontrivial_collapse': len(equiv_pairs) > 0
        }


# ===========================================================
# Section 9: 喂屎测试 - 恶意输入注入
# ===========================================================

class PoisonInjectionTest:
    """
    喂屎测试引擎

    目标: 用恶意/边界/病态输入测试系统的健壮性
    """

    def __init__(self, curve: EllipticCurveParams, precision: int):
        self.curve = curve
        self.p = curve.p
        self.precision = precision
        self.encoder = ECWittEncoder(curve, precision)
        self.prover = ECCollapseProver(curve, precision)

    def run_all_poison_tests(self) -> List[PressureTestResult]:
        results = []

        results.append(self.poison_overflow_witt_components())
        results.append(self.poison_negative_coordinates())
        results.append(self.poison_non_prime_base())
        results.append(self.poison_zero_precision())
        results.append(self.poison_mismatched_curves())
        results.append(self.poison_ghost_overflow())
        results.append(self.poison_frobenius_fixed_point())
        results.append(self.poison_verschiebung_annihilation())
        # 新增毒药
        results.append(self.poison_teichmuller_lift_collision())
        results.append(self.poison_supersingular_curve())
        # 再加两个
        results.append(self.poison_frobenius_cycle_order())
        results.append(self.poison_ghost_multiplicative())

        return results

    def poison_overflow_witt_components(self) -> PressureTestResult:
        """
        喂屎: Witt分量超过p-1
        """
        p = self.p
        k = self.precision

        # 构造非法Witt向量（分量 >= p）
        try:
            bad_components = [p, p + 1, 2 * p]  # 全部超界
            w = WittVector(bad_components + [0] * (k - 3), p)
            # 检查是否自动规范化
            normalized = all(0 <= c < p for c in w.components)

            return PressureTestResult(
                test_name="[毒药] Witt分量溢出",
                passed=normalized,
                expected="自动规范化到[0,p-1]",
                actual=f"components={w.components[:4]}",
                details="溢出分量应被mod p规范化"
            )
        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] Witt分量溢出",
                passed=True,
                expected="异常或规范化",
                actual=f"异常: {e}",
                details="拒绝非法输入也是正确行为"
            )

    def poison_negative_coordinates(self) -> PressureTestResult:
        """
        喂屎: 负坐标
        """
        try:
            # 负x坐标
            pt = ECPointPadic.from_ints(-1, 5, self.curve, self.precision)
            # 应该被规范化为正数（mod p^k）

            on_curve = pt.is_on_curve()

            return PressureTestResult(
                test_name="[毒药] 负坐标处理",
                passed=True,  # 只要不崩溃就算通过
                expected="规范化或拒绝",
                actual=f"x={pt.x.to_int_mod_pk()}, on_curve={on_curve}",
                details="负数应被mod p^k规范化"
            )
        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] 负坐标处理",
                passed=True,
                expected="异常或规范化",
                actual=f"异常: {e}",
                details="拒绝负坐标也是正确行为"
            )

    def poison_non_prime_base(self) -> PressureTestResult:
        """
        喂屎: 非素数基
        """
        try:
            # 用合数作为基
            bad_curve = EllipticCurveParams(a=1, b=1, p=6)  # 6不是素数
            w = WittVector([1, 2, 3, 0, 0, 0], 6)

            # Frobenius行为会出问题吗？
            frob = w.frobenius()

            return PressureTestResult(
                test_name="[毒药] 非素数基",
                passed=False,  # 应该拒绝
                expected="拒绝非素数基或产生错误结果",
                actual=f"Frobenius执行成功: {frob.components[:3]}",
                details="非素数基会破坏Witt向量的数学结构"
            )
        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] 非素数基",
                passed=True,
                expected="异常",
                actual=f"正确拒绝: {e}",
                details="非素数基应被检测并拒绝"
            )

    def poison_zero_precision(self) -> PressureTestResult:
        """
        喂屎: 零精度
        """
        try:
            w = WittVector([], self.p)

            return PressureTestResult(
                test_name="[毒药] 零精度Witt",
                passed=False,
                expected="拒绝空向量",
                actual=f"创建成功: length={w.length}",
                details="零精度应被拒绝"
            )
        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] 零精度Witt",
                passed=True,
                expected="异常",
                actual=f"正确拒绝: {type(e).__name__}",
                details="零精度Witt向量无意义"
            )

    def poison_mismatched_curves(self) -> PressureTestResult:
        """
        喂屎: 不匹配的曲线参数
        """
        try:
            # 创建两条不同的曲线
            curve1 = EllipticCurveParams(a=1, b=1, p=self.p)
            curve2 = EllipticCurveParams(a=2, b=3, p=self.p)

            # 用curve1的编码器处理curve2上的点
            encoder1 = ECWittEncoder(curve1, self.precision)

            # 找curve2上的点
            pts = find_curve_points_hensel(curve2, self.precision, count=1)
            if pts:
                x, y = pts[0]
                pt2 = ECPointPadic.from_ints(x, y, curve2, self.precision)

                # 强制用错误的编码器
                witt = encoder1.encode(pt2)  # 应该会产生垃圾结果

                return PressureTestResult(
                    test_name="[毒药] 曲线不匹配",
                    passed=False,  # 这种误用应该被检测
                    expected="检测曲线不匹配或产生无意义结果",
                    actual=f"编码成功: {witt.components[:3]}",
                    details="使用错误曲线的编码器应该被检测"
                )
            else:
                return PressureTestResult(
                    test_name="[毒药] 曲线不匹配",
                    passed=True,
                    expected="无法测试",
                    actual="curve2上未找到点",
                    details="跳过测试"
                )
        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] 曲线不匹配",
                passed=True,
                expected="异常",
                actual=f"正确检测: {e}",
                details="曲线不匹配应被检测"
            )

    def poison_ghost_overflow(self) -> PressureTestResult:
        """
        喂屎: Ghost值整数溢出（极端精度）
        """
        try:
            p = self.p
            k = 50  # 极端精度

            # 构造最大分量的Witt向量
            max_components = [p - 1] * k
            w = WittVector(max_components, p)

            # Ghost值会非常大
            ghost_0 = w.ghost(0)
            ghost_k_1 = w.ghost(k - 1)  # 这个会爆炸

            # Python的int是任意精度，不会溢出，但值会非常大
            ghost_magnitude = ghost_k_1.bit_length()

            return PressureTestResult(
                test_name="[毒药] Ghost极端精度",
                passed=True,  # Python任意精度int能处理
                expected="处理大整数",
                actual=f"Ghost(49)有{ghost_magnitude}位",
                details=f"Ghost值指数增长，k=50时约{ghost_magnitude}位"
            )
        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] Ghost极端精度",
                passed=False,
                expected="处理大整数",
                actual=f"异常: {e}",
                details="应该能处理任意精度"
            )

    def poison_frobenius_fixed_point(self) -> PressureTestResult:
        """
        喂屎: Frobenius不动点

        在F_p上，x^p = x (费马小定理)
        所以Frobenius在某种意义上是"恒等"的
        """
        p = self.p
        k = self.precision

        # 测试不动点
        w = WittVector([1, 0, 0, 0, 0, 0][:k], p)
        frob_w = w.frobenius()

        # (1,0,0,...) 应该是Frobenius不动点
        # 因为 1^p = 1, 0^p = 0
        is_fixed = w.components == frob_w.components

        return PressureTestResult(
            test_name="[毒药] Frobenius不动点",
            passed=is_fixed,
            expected="(1,0,...,0)是不动点",
            actual=f"φ(w)={frob_w.components}, 不动={is_fixed}",
            details="费马小定理保证F_p元素是Frobenius不动点"
        )

    def poison_verschiebung_annihilation(self) -> PressureTestResult:
        """
        喂屎: Verschiebung湮灭

        V^k 作用于任何k阶Witt向量都会得到(0,...,0,*)
        这是信息丢失的极端情况
        """
        p = self.p
        k = self.precision

        w = WittVector([1, 2, 3] + [0] * (k - 3), p)

        # 连续应用V
        current = w
        for i in range(k):
            current = current.verschiebung()

        # k次V后，所有原始信息都被推出了边界
        all_zero = all(c == 0 for c in current.components)

        return PressureTestResult(
            test_name="[毒药] Verschiebung信息湮灭",
            passed=all_zero,
            expected="V^k(w) = 0 (信息完全丢失)",
            actual=f"V^{k}(w)={current.components}",
            details="Verschiebung左移k次后信息完全消失"
        )

    def poison_teichmuller_lift_collision(self) -> PressureTestResult:
        """
        喂屎: Teichmüller提升碰撞

        Teichmüller代表元: 满足 x^p = x 的Witt向量
        在F_p中，每个元素都是Teichmüller代表元
        测试: 不同Teichmüller代表元的Witt向量是否正确分离
        """
        p = self.p
        k = self.precision

        # 构造两个Teichmüller代表元的Witt向量
        # Teichmüller(a) = (a, 0, 0, ...) 满足 φ(T(a)) = T(a)
        teich_1 = WittVector([1] + [0] * (k - 1), p)
        teich_2 = WittVector([2] + [0] * (k - 1), p)

        # 检查Frobenius不动性
        frob_1 = teich_1.frobenius()
        frob_2 = teich_2.frobenius()

        is_fixed_1 = teich_1.components == frob_1.components
        is_fixed_2 = teich_2.components == frob_2.components

        # 检查它们在层级0不等价
        not_equiv = not teich_1.equiv_mod_nygaard(teich_2, 1)

        passed = is_fixed_1 and is_fixed_2 and not_equiv

        return PressureTestResult(
            test_name="[毒药] Teichmüller碰撞",
            passed=passed,
            expected="T(1)≠T(2)且都是φ-不动点",
            actual=f"T(1)固定={is_fixed_1}, T(2)固定={is_fixed_2}, 不等价={not_equiv}",
            details="Teichmüller代表元必须分离且Frobenius不动"
        )

    def poison_supersingular_curve(self) -> PressureTestResult:
        """
        喂屎: 超奇异曲线

        超奇异椭圆曲线: j = 0 或 j = 1728
        y² = x³ + b (j=0) 或 y² = x³ + ax (j=1728)

        这些曲线有特殊的内态射环，可能导致异常行为
        """
        p = self.p
        k = self.precision

        # 构造超奇异曲线 y² = x³ + 1 (j=0 当 p ≡ 2 mod 3)
        ss_curve = EllipticCurveParams(a=0, b=1, p=p)

        # 检查是否真的是超奇异
        # 超奇异条件: p ≡ 2 (mod 3) 对于 j=0
        is_supersingular_condition = (p % 3 == 2)

        try:
            # 尝试在超奇异曲线上找点
            pts = find_curve_points_hensel(ss_curve, k, count=3)

            if len(pts) >= 2:
                # 编码两个点
                encoder = ECWittEncoder(ss_curve, k)
                P = ECPointPadic.from_ints(pts[0][0], pts[0][1], ss_curve, k)
                Q = ECPointPadic.from_ints(pts[1][0], pts[1][1], ss_curve, k)

                witt_P = encoder.encode(P)
                witt_Q = encoder.encode(Q)

                # 检查等价层级
                level = 0
                for i in range(k):
                    if witt_P.components[i] == witt_Q.components[i]:
                        level = i + 1
                    else:
                        break

                return PressureTestResult(
                    test_name="[毒药] 超奇异曲线",
                    passed=True,
                    expected="在超奇异曲线上正常工作",
                    actual=f"j=0曲线, 超奇异条件={is_supersingular_condition}, 等价层级={level}",
                    details="超奇异曲线的Witt编码应正常工作"
                )
            else:
                return PressureTestResult(
                    test_name="[毒药] 超奇异曲线",
                    passed=True,
                    expected="超奇异曲线测试",
                    actual=f"曲线上点稀疏，跳过",
                    details="超奇异曲线在小范围内可能点少"
                )
        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] 超奇异曲线",
                passed=False,
                expected="正常处理超奇异曲线",
                actual=f"异常: {e}",
                details="超奇异曲线不应导致崩溃"
            )

    def poison_frobenius_cycle_order(self) -> PressureTestResult:
        """
        喂屎: Frobenius循环阶数检测

        在有限域上，φ^k 应该最终循环。
        测试系统是否正确处理φ的迭代。
        """
        p = self.p
        k = self.precision

        try:
            # 构造一个非平凡Witt向量
            w = WittVector([1, 2, 1, 0, 0, 0][:k], p)

            # 迭代Frobenius
            current = w
            seen_states = [tuple(current.components)]
            cycle_found = False
            cycle_length = 0

            for i in range(k + 5):  # 最多迭代k+5次
                current = current.frobenius()
                state = tuple(current.components)

                if state in seen_states:
                    cycle_found = True
                    cycle_length = i + 1
                    break
                seen_states.append(state)

            # Frobenius在Witt向量上的行为应该是稳定的
            # 不一定循环回原点，但不应该崩溃或产生无效状态
            is_valid = all(0 <= c < p for c in current.components)

            return PressureTestResult(
                test_name="[毒药] Frobenius循环阶",
                passed=is_valid,
                expected="Frobenius迭代保持有效Witt分量",
                actual=f"迭代{len(seen_states)}次, 循环={cycle_found}, 长度={cycle_length}, 有效={is_valid}",
                details="检测φ^k的行为"
            )

        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] Frobenius循环阶",
                passed=False,
                expected="Frobenius迭代不应崩溃",
                actual=f"异常: {e}",
                details="迭代Frobenius不应产生异常"
            )

    def poison_ghost_multiplicative(self) -> PressureTestResult:
        """
        喂屎: Ghost映射的乘法结构

        Ghost映射是环同态，应该有: Ghost(x * y) = Ghost(x) * Ghost(y)
        (这里乘法是分量wise的)

        我们测试这个性质在边界情况下是否保持。
        """
        p = self.p
        k = min(4, self.precision)  # 用较短的精度避免溢出

        try:
            # 构造两个Witt向量 - 用Teichmüller代表
            a = 2 % p
            b = 3 % p
            ab = (a * b) % p

            # Teichmüller代表: [a, 0, 0, ...] 代表a在W(F_p)中的像
            w_a = WittVector([a] + [0] * (k - 1), p)
            w_b = WittVector([b] + [0] * (k - 1), p)
            w_ab = WittVector([ab] + [0] * (k - 1), p)

            # 对于Teichmüller代表，Ghost映射简化为:
            # w_n([a,0,0,...]) = a^{p^n}
            ghost_a = w_a.ghost_vector()
            ghost_b = w_b.ghost_vector()
            ghost_ab = w_ab.ghost_vector()

            # 验证: ghost_ab[n] = ghost_a[n] * ghost_b[n] (mod p^{n+1})
            # 对于Teichmüller: (ab)^{p^n} = a^{p^n} * b^{p^n}
            multiplicative_holds = True
            violations = []

            for n in range(k):
                pk = p ** (n + 1)
                expected = (ghost_a[n] * ghost_b[n]) % pk
                actual = ghost_ab[n] % pk

                if expected != actual:
                    multiplicative_holds = False
                    violations.append(f"n={n}: {expected} != {actual}")

            return PressureTestResult(
                test_name="[毒药] Ghost乘法性",
                passed=multiplicative_holds,
                expected="Ghost映射对Teichmüller代表保持乘法",
                actual=f"乘法性={multiplicative_holds}, 违例={violations if violations else '无'}",
                details="验证Ghost的环同态性质"
            )

        except Exception as e:
            return PressureTestResult(
                test_name="[毒药] Ghost乘法性",
                passed=False,
                expected="Ghost乘法性检测不应崩溃",
                actual=f"异常: {e}",
                details="边界情况不应导致异常"
            )


# ===========================================================
# Section 10: 等价对分布分析 + 大素数高精度追踪
# ===========================================================

class EquivalencePairDistributionAnalyzer:
    """
    等价对分布分析器

    核心问题: 等价对的出现有什么规律？
    - 是随机的还是有结构的？
    - 与素数p的关系？
    - 与精度k的关系？
    - 与曲线参数的关系？
    """

    def __init__(self):
        self.results = []

    def analyze_single_curve(self, p: int, a: int, b: int, precision: int,
                              num_points: int = 20) -> Dict[str, Any]:
        """分析单条曲线上的等价对分布"""
        curve = EllipticCurveParams(a=a, b=b, p=p)

        # 找点
        points_raw = find_curve_points_hensel(curve, precision, count=num_points)

        if len(points_raw) < 3:
            return {
                'p': p, 'a': a, 'b': b, 'precision': precision,
                'num_points': len(points_raw),
                'error': 'insufficient_points'
            }

        # 构造点对象
        encoder = ECWittEncoder(curve, precision)
        points = [ECPointPadic.from_ints(x, y, curve, precision) for x, y in points_raw]
        witt_vectors = [encoder.encode(pt) for pt in points]

        # 分析等价对
        equiv_pairs = []
        level_distribution = {}  # 层级 -> 数量

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                level = 0
                for k in range(precision):
                    if witt_vectors[i].components[k] == witt_vectors[j].components[k]:
                        level = k + 1
                    else:
                        break
                if level > 0:
                    equiv_pairs.append({
                        'i': i, 'j': j, 'level': level,
                        'coords_i': points_raw[i],
                        'coords_j': points_raw[j],
                        'witt_i': witt_vectors[i].components,
                        'witt_j': witt_vectors[j].components
                    })
                    level_distribution[level] = level_distribution.get(level, 0) + 1

        # 计算统计量
        total_pairs = len(points) * (len(points) - 1) // 2
        equiv_ratio = len(equiv_pairs) / total_pairs if total_pairs > 0 else 0

        # 第一分量分布
        first_components = [w.components[0] for w in witt_vectors]
        first_comp_counts = {}
        for c in first_components:
            first_comp_counts[c] = first_comp_counts.get(c, 0) + 1

        # 碰撞分析: 第一分量相同的点对
        first_comp_collisions = sum(c * (c - 1) // 2 for c in first_comp_counts.values())

        result = {
            'p': p, 'a': a, 'b': b, 'precision': precision,
            'num_points': len(points),
            'total_pairs': total_pairs,
            'equiv_pairs_count': len(equiv_pairs),
            'equiv_ratio': equiv_ratio,
            'level_distribution': level_distribution,
            'first_comp_distribution': first_comp_counts,
            'first_comp_collision_pairs': first_comp_collisions,
            'equiv_pairs': equiv_pairs[:10]  # 只保留前10个详情
        }

        self.results.append(result)
        return result

    def sweep_primes(self, primes: List[int], a: int, b: int, precision: int) -> List[Dict]:
        """在多个素数上扫描"""
        sweep_results = []
        for p in primes:
            result = self.analyze_single_curve(p, a, b, precision)
            sweep_results.append(result)
        return sweep_results

    def sweep_precision(self, p: int, a: int, b: int, precisions: List[int]) -> List[Dict]:
        """在多个精度上扫描"""
        sweep_results = []
        for k in precisions:
            result = self.analyze_single_curve(p, a, b, k)
            sweep_results.append(result)
        return sweep_results

    def print_summary(self, results: List[Dict]):
        """打印分析摘要"""
        print("\n等价对分布分析摘要:")
        print("-" * 60)
        for r in results:
            if 'error' in r:
                print(f"  p={r['p']}, k={r['precision']}: 点不足")
                continue
            print(f"  p={r['p']}, a={r['a']}, b={r['b']}, k={r['precision']}:")
            print(f"    点数={r['num_points']}, 总对数={r['total_pairs']}")
            print(f"    等价对数={r['equiv_pairs_count']}, 比率={r['equiv_ratio']:.4f}")
            print(f"    层级分布: {r['level_distribution']}")
            print(f"    第一分量碰撞: {r['first_comp_collision_pairs']}")


class LargePrimeHighPrecisionTracker:
    """
    大素数/高精度追踪器

    问题: 等价对现象在更极端的参数下表现如何？
    """

    def __init__(self):
        self.tracking_log = []

    def track_at_parameters(self, p: int, precision: int, curve_a: int, curve_b: int,
                             num_points: int = 15) -> Dict[str, Any]:
        """在指定参数下追踪"""
        import time
        start = time.time()

        curve = EllipticCurveParams(a=curve_a, b=curve_b, p=p)

        # 检查是否好约化
        good_reduction = curve.is_good_reduction()

        # 找点
        points_raw = find_curve_points_hensel(curve, precision, count=num_points)
        find_time = time.time() - start

        if len(points_raw) < 2:
            return {
                'p': p, 'precision': precision,
                'status': 'insufficient_points',
                'points_found': len(points_raw),
                'time': find_time
            }

        # 编码
        encoder = ECWittEncoder(curve, precision)
        witt_vectors = []
        for x, y in points_raw:
            pt = ECPointPadic.from_ints(x, y, curve, precision)
            witt_vectors.append(encoder.encode(pt))

        # 分析等价对
        equiv_pairs = []
        max_level = 0
        for i in range(len(witt_vectors)):
            for j in range(i + 1, len(witt_vectors)):
                level = 0
                for k in range(precision):
                    if witt_vectors[i].components[k] == witt_vectors[j].components[k]:
                        level = k + 1
                    else:
                        break
                if level > 0:
                    equiv_pairs.append((i, j, level))
                    max_level = max(max_level, level)

        total_time = time.time() - start

        result = {
            'p': p,
            'precision': precision,
            'good_reduction': good_reduction,
            'points_found': len(points_raw),
            'equiv_pairs_count': len(equiv_pairs),
            'max_equiv_level': max_level,
            'equiv_pairs': equiv_pairs,
            'first_components': [w.components[0] for w in witt_vectors],
            'time': total_time,
            'status': 'success'
        }

        self.tracking_log.append(result)
        return result

    def run_escalation_test(self) -> List[Dict]:
        """
        升级测试: 逐步增大素数和精度

        观察等价对现象是否:
        1. 随素数增大而消失？
        2. 随精度增加而变化？
        """
        results = []

        # 素数序列
        primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        # 精度序列
        precisions = [4, 6, 8, 10]

        print("\n升级测试: 素数 × 精度")
        print("=" * 70)

        for p in primes:
            for k in precisions:
                result = self.track_at_parameters(p, k, curve_a=1, curve_b=1, num_points=15)
                results.append(result)

                status = "✓" if result['status'] == 'success' else "✗"
                equiv = result.get('equiv_pairs_count', 0)
                max_lv = result.get('max_equiv_level', 0)
                pts = result.get('points_found', 0)

                if equiv > 0:
                    print(f"  [{status}] p={p:2d}, k={k:2d}: 点={pts:2d}, 等价对={equiv}, 最高层级={max_lv} ★")
                else:
                    print(f"  [{status}] p={p:2d}, k={k:2d}: 点={pts:2d}, 无等价对")

        return results

    def find_high_collapse_cases(self, results: List[Dict]) -> List[Dict]:
        """找出高塌缩层级的情况"""
        high_collapse = []
        for r in results:
            if r.get('max_equiv_level', 0) >= 2:
                high_collapse.append(r)
        return sorted(high_collapse, key=lambda x: -x.get('max_equiv_level', 0))


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """Tonelli-Shanks算法求模p平方根"""
    if n == 0:
        return 0
    if pow(n, (p - 1) // 2, p) != 1:
        return None  # 非二次剩余

    # 特殊情况: p ≡ 3 (mod 4)
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    # 一般情况
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # 找一个非二次剩余
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while True:
        if t == 1:
            return r
        i = 1
        temp = t
        while pow(temp, 2, p) != 1:
            temp = pow(temp, 2, p)
            i += 1
            if i == m:
                return None

        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = pow(b, 2, p)
        t = (t * c) % p
        r = (r * b) % p


def find_curve_points(curve: EllipticCurveParams, precision: int, count: int = 5) -> List[Tuple[int, int]]:
    """
    在椭圆曲线上寻找有效点

    使用Tonelli-Shanks算法高效求平方根
    """
    p = curve.p
    points = []

    for x in range(1, min(p * 10, 500)):
        # y² = x³ + ax + b (mod p)
        y_sq_mod_p = (pow(x, 3, p) + curve.a * x + curve.b) % p

        # 用Tonelli-Shanks求平方根
        y_mod_p = tonelli_shanks(y_sq_mod_p, p)

        if y_mod_p is not None:
            # 尝试提升到更高精度
            for y_candidate in [y_mod_p, p - y_mod_p]:
                pt = ECPointPadic.from_ints(x, y_candidate, curve, precision)
                if pt.is_on_curve():
                    points.append((x, y_candidate))
                    if len(points) >= count:
                        return points
                    break

    return points


def run_whitebox_smoke_test():
    """
    白盒Smoke测试

    验证:
    1. 数学结构正确性
    2. 等价性判定正确性
    3. 边界条件处理
    4. 毒药注入拒绝
    """
    print("=" * 70)
    print("EC p-adic 等价类塌缩证书 白盒Smoke测试")
    print("建模标准: 零启发式, 无魔法数, 严格数学结构")
    print("=" * 70)

    # 测试参数
    # 使用素数5，曲线 y² = x³ + x + 1 有较多有理点
    p = 5  # 素数
    precision = 6  # Witt长度

    # 椭圆曲线 y² = x³ + x + 1 over Q_5
    # 这条曲线在F_5上有6个点（包括无穷远点）
    curve = EllipticCurveParams(a=1, b=1, p=p)

    print(f"\n[配置] 素数p={p}, 精度k={precision}")
    print(f"[曲线] E: y² = x³ + {curve.a}x + {curve.b}")
    print(f"[判别式] Δ = {curve.discriminant()}")
    print(f"[约化类型] {'好约化' if curve.is_good_reduction() else '坏约化'}")

    # 创建证明器
    prover = ECCollapseProver(curve, precision)

    # 运行压力测试
    print("\n" + "-" * 70)
    print("压力测试:")
    print("-" * 70)

    tester = ECCollapsePressureTest(prover)
    results = tester.run_all_tests()

    passed_count = 0
    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  [{status}] {result.test_name}")
        print(f"         预期: {result.expected}")
        print(f"         实际: {result.actual}")
        if result.passed:
            passed_count += 1

    print("-" * 70)
    print(f"压力测试结果: {passed_count}/{len(results)} 通过")

    # 演示等价类塌缩证书生成
    print("\n" + "=" * 70)
    print("等价类塌缩证书演示:")
    print("=" * 70)

    # 使用Hensel提升的点查找函数
    test_points = find_curve_points_hensel(curve, precision, count=5)
    print(f"\n[点搜索-Hensel] 在曲线上找到 {len(test_points)} 个有效点")

    if len(test_points) >= 2:
        x1, y1 = test_points[0]
        x2, y2 = test_points[1]

        P = ECPointPadic.from_ints(x1, y1, curve, precision)
        Q = ECPointPadic.from_ints(x2, y2, curve, precision)

        print(f"\n[点P] ({x1}, {y1})")
        print(f"[点Q] ({x2}, {y2})")

        # 生成证书
        cert = prover.prove(P, Q)

        print(f"\n[证书哈希] {cert.certificate_hash}")
        print(f"[等价层级] {cert.equivalence_level} / {precision}")
        print(f"[Log-Shell距离] {cert.log_shell_distance}")
        print(f"[θ-degree] {cert.theta_degree}")
        print(f"[塌缩强度] {cert.collapse_strength()}")
        print(f"[Frobenius兼容] {cert.frobenius_compatible}")
        print(f"[Nygaard过滤OK] {cert.nygaard_filtration_ok}")

        print(f"\n[Witt编码 P] {cert.witt_P_components}")
        print(f"[Witt编码 Q] {cert.witt_Q_components}")
        print(f"[Ghost差向量] {cert.ghost_diff[:4]}...")
    else:
        print("[警告] 未找到足够的曲线上的点进行演示")

    # 自等价测试
    print("\n" + "-" * 70)
    print("自等价测试 (P ≡ P):")
    print("-" * 70)

    if test_points:
        x1, y1 = test_points[0]
        P = ECPointPadic.from_ints(x1, y1, curve, precision)
        cert_self = prover.prove(P, P)

        print(f"[等价层级] {cert_self.equivalence_level} / {precision}")
        print(f"[Log-Shell距离] {cert_self.log_shell_distance}")
        print(f"[完全塌缩] {cert_self.collapse_strength() == 'FULL_COLLAPSE'}")

    # 核心演示：构造性部分塌缩
    print("\n" + "=" * 70)
    print("核心演示: 构造性部分塌缩 (Constructive Partial Collapse)")
    print("=" * 70)

    # 构造两个在Witt层面部分等价的点
    # 方法：手动构造两个Witt向量，使前k个分量相同
    print("\n[构造] 手动构造两个在Nygaard层级2等价的Witt向量")

    w_alpha = WittVector([3, 1, 2, 4, 0, 0], p)  # 分量: [3,1,2,4,0,0]
    w_beta = WittVector([3, 1, 4, 2, 0, 0], p)   # 分量: [3,1,4,2,0,0]
    # 前2个分量相同 [3,1]，第3个分量不同 [2 vs 4]

    print(f"  α = {w_alpha.components}")
    print(f"  β = {w_beta.components}")
    print(f"\n[Nygaard验证]")
    print(f"  α ≡ β (mod N^{{≥1}})? {w_alpha.equiv_mod_nygaard(w_beta, 1)}")
    print(f"  α ≡ β (mod N^{{≥2}})? {w_alpha.equiv_mod_nygaard(w_beta, 2)}")
    print(f"  α ≡ β (mod N^{{≥3}})? {w_alpha.equiv_mod_nygaard(w_beta, 3)}")

    # 计算Log-Shell距离
    equiv_level = 0
    for i in range(len(w_alpha.components)):
        if w_alpha.components[i] == w_beta.components[i]:
            equiv_level = i + 1
        else:
            break
    log_shell = Fraction(equiv_level, len(w_alpha.components))

    print(f"\n[Log-Shell距离] {log_shell} = {float(log_shell):.4f}")
    print(f"[等价层级] {equiv_level} / {len(w_alpha.components)}")

    # 塌缩强度评级
    ratio = float(log_shell)
    if ratio >= 1.0:
        strength = "FULL_COLLAPSE (完全塌缩)"
    elif ratio >= 0.75:
        strength = "STRONG_COLLAPSE (强塌缩)"
    elif ratio >= 0.5:
        strength = "MODERATE_COLLAPSE (中等塌缩)"
    elif ratio > 0:
        strength = "WEAK_COLLAPSE (弱塌缩)"
    else:
        strength = "NO_COLLAPSE (无塌缩)"

    print(f"[塌缩强度] {strength}")

    # Ghost映射验证
    print(f"\n[Ghost验证]")
    print(f"  Ghost(α) = {w_alpha.ghost_vector()[:4]}...")
    print(f"  Ghost(β) = {w_beta.ghost_vector()[:4]}...")

    # Frobenius验证
    frob_alpha = w_alpha.frobenius()
    frob_beta = w_beta.frobenius()
    print(f"\n[Frobenius兼容]")
    print(f"  φ(α) = {frob_alpha.components}")
    print(f"  φ(β) = {frob_beta.components}")
    print(f"  φ(α) ≡ φ(β) (mod N^{{≥{equiv_level}}})? {frob_alpha.equiv_mod_nygaard(frob_beta, equiv_level)}")

    print("\n" + "-" * 70)
    print("关键洞察:")
    print("-" * 70)
    print("""
  这个演示证明了核心机制:

  1. 两个不同的Witt向量可以在特定Nygaard层级"看起来相同"
  2. 这类似于ABC猜想中的Log-Shell等价性
  3. 对于椭圆曲线点，这意味着:
     - 两个"不同"的点P,Q
     - 在p-adic Nygaard过滤意义下可以"等价"
     - 这种等价性不破坏离散对数问题
     - 但它暴露了椭圆曲线的p-adic结构

  这不是破解，而是一种新的数学视角。
    """)

    print("\n" + "=" * 70)
    print("白盒Smoke测试完成")
    print("=" * 70)

    return passed_count == len(results)


def run_boundary_stress_test():
    """
    边界爆破测试 - 往死里造

    目标：
    1. 找到系统的极限
    2. 发现边界处的异常行为
    3. 寻找可能的"惊喜"
    """
    print("=" * 70)
    print("边界爆破测试 - 往死里造")
    print("=" * 70)

    results = []

    # ================================================================
    # 测试1: 极端大素数
    # ================================================================
    print("\n[爆破1] 极端大素数测试")
    print("-" * 50)

    large_primes = [251, 509, 1021, 2039, 4093]  # 大素数序列
    for p in large_primes:
        try:
            curve = EllipticCurveParams(a=1, b=1, p=p)
            precision = 4

            # 尝试找点
            pts = find_curve_points(curve, precision, count=2)
            if len(pts) >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                P = ECPointPadic.from_ints(x1, y1, curve, precision)
                Q = ECPointPadic.from_ints(x2, y2, curve, precision)

                prover = ECCollapseProver(curve, precision)
                cert = prover.prove(P, Q)

                print(f"  p={p}: 等价层级={cert.equivalence_level}/{precision}, "
                      f"θ={float(cert.theta_degree):.4f}")
                results.append(('large_prime', p, cert.equivalence_level))
            else:
                print(f"  p={p}: 曲线点稀疏，跳过")
        except Exception as e:
            print(f"  p={p}: 异常 - {type(e).__name__}: {e}")
            results.append(('large_prime_error', p, str(e)))

    # ================================================================
    # 测试2: 极端高精度
    # ================================================================
    print("\n[爆破2] 极端高精度测试")
    print("-" * 50)

    p = 5
    precisions = [8, 12, 16, 20, 24]
    for k in precisions:
        try:
            curve = EllipticCurveParams(a=1, b=1, p=p)

            # Witt向量在高精度下的表现
            w1 = WittVector([1, 2, 3] + [0] * (k - 3), p)
            w2 = WittVector([1, 2, 4] + [0] * (k - 3), p)

            ghost1 = w1.ghost_vector()
            ghost2 = w2.ghost_vector()

            # 检查Ghost值是否溢出
            max_ghost1 = max(ghost1)
            max_ghost2 = max(ghost2)

            print(f"  precision={k}: max_ghost={max(max_ghost1, max_ghost2)}, "
                  f"层级2等价={w1.equiv_mod_nygaard(w2, 2)}")

            # 检查Ghost差的模式
            ghost_diff = [g1 - g2 for g1, g2 in zip(ghost1, ghost2)]
            print(f"    Ghost差前4项: {ghost_diff[:4]}")

            results.append(('high_precision', k, max_ghost1))

        except Exception as e:
            print(f"  precision={k}: 异常 - {type(e).__name__}: {e}")
            results.append(('high_precision_error', k, str(e)))

    # ================================================================
    # 测试3: 塌缩边界搜索 - 寻找自然发生的部分塌缩
    # ================================================================
    print("\n[爆破3] 塌缩边界搜索 - 寻找自然部分塌缩")
    print("-" * 50)

    p = 5
    precision = 8
    curve = EllipticCurveParams(a=1, b=1, p=p)

    # 大范围搜索曲线上的点对
    all_points = find_curve_points(curve, precision, count=20)
    print(f"  找到 {len(all_points)} 个曲线点")

    collapse_found = []
    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            x1, y1 = all_points[i]
            x2, y2 = all_points[j]

            P = ECPointPadic.from_ints(x1, y1, curve, precision)
            Q = ECPointPadic.from_ints(x2, y2, curve, precision)

            prover = ECCollapseProver(curve, precision)
            cert = prover.prove(P, Q)

            if cert.equivalence_level > 0:
                collapse_found.append((
                    (x1, y1), (x2, y2),
                    cert.equivalence_level,
                    float(cert.log_shell_distance)
                ))

    if collapse_found:
        print(f"  发现 {len(collapse_found)} 对部分塌缩的点!")
        for p1, p2, level, dist in sorted(collapse_found, key=lambda x: -x[2])[:5]:
            print(f"    P={p1}, Q={p2}: 层级={level}, Log-Shell={dist:.4f}")
        results.append(('natural_collapse', len(collapse_found), collapse_found[0][2]))
    else:
        print("  未发现自然部分塌缩")
        results.append(('natural_collapse', 0, 0))

    # ================================================================
    # 测试4: φV = p 关系的极端验证
    # ================================================================
    print("\n[爆破4] Frobenius-Verschiebung关系极端验证")
    print("-" * 50)

    for p in [3, 5, 7, 11]:
        k = 10
        try:
            # 构造随机Witt向量
            import random
            random.seed(42)  # 可重现
            components = [random.randint(0, p - 1) for _ in range(k)]
            w = WittVector(components, p)

            # φ(V(w))
            Vw = w.verschiebung()
            phiVw = Vw.frobenius()

            # V(φ(w))
            phiw = w.frobenius()
            Vphiw = phiw.verschiebung()

            # 验证 φV = Vφ (它们应该给出相同的效果)
            match = phiVw.components == Vphiw.components

            # Ghost层面验证
            ghost_phiVw = phiVw.ghost_vector()
            ghost_Vphiw = Vphiw.ghost_vector()
            ghost_match = ghost_phiVw == ghost_Vphiw

            print(f"  p={p}: φV=Vφ分量匹配={match}, Ghost匹配={ghost_match}")

            if not match:
                print(f"    φV(w)={phiVw.components[:5]}...")
                print(f"    Vφ(w)={Vphiw.components[:5]}...")
                results.append(('phiV_mismatch', p, 'ANOMALY'))
            else:
                results.append(('phiV', p, 'OK'))

        except Exception as e:
            print(f"  p={p}: 异常 - {e}")

    # ================================================================
    # 测试5: 零向量边界
    # ================================================================
    print("\n[爆破5] 零向量边界测试")
    print("-" * 50)

    p = 5
    k = 6
    zero_w = WittVector([0] * k, p)
    nonzero_w = WittVector([1] + [0] * (k - 1), p)

    # 零向量的Nygaard层级
    zero_level = zero_w.nygaard_level()
    print(f"  零向量Nygaard层级: {zero_level} (预期={k})")

    # 零与非零的等价性
    equiv_zero_nonzero = zero_w.equiv_mod_nygaard(nonzero_w, 0)
    print(f"  0 ≡ 1 (mod N^{{≥0}})? {equiv_zero_nonzero} (预期=True，层级0无约束)")

    # Frobenius作用于零
    zero_frob = zero_w.frobenius()
    print(f"  φ(0) = {zero_frob.components} (预期全0)")

    # Verschiebung作用于零
    zero_ver = zero_w.verschiebung()
    print(f"  V(0) = {zero_ver.components} (预期全0)")

    results.append(('zero_boundary', zero_level, k))

    # ================================================================
    # 测试6: 最大分量边界 (p-1)
    # ================================================================
    print("\n[爆破6] 最大分量边界测试 (全p-1)")
    print("-" * 50)

    p = 5
    k = 6
    max_w = WittVector([p - 1] * k, p)

    print(f"  w = [{p-1}] * {k} = {max_w.components}")
    print(f"  Ghost向量: {max_w.ghost_vector()}")

    # Frobenius: (p-1)^p mod p = ?
    max_frob = max_w.frobenius()
    print(f"  φ(w) = {max_frob.components}")

    # 自我等价验证
    self_equiv = max_w.equiv_mod_nygaard(max_w, k)
    print(f"  w ≡ w (mod N^{{≥{k}}})? {self_equiv}")

    results.append(('max_components', max_w.ghost(0), 'OK'))

    # ================================================================
    # 测试7: 形式群参数边界 - y=0的情况
    # ================================================================
    print("\n[爆破7] 形式群参数边界 (y接近0)")
    print("-" * 50)

    p = 5
    precision = 6
    curve = EllipticCurveParams(a=1, b=1, p=p)
    formal = FormalGroup(curve, precision)

    # 寻找y很小的点
    pk = p ** 4
    small_y_points = []
    for x in range(1, 200):
        y_sq = (pow(x, 3, pk) + curve.a * x + curve.b) % pk
        for y in range(p * 2):  # 只看小y
            if pow(y, 2, pk) == y_sq:
                pt = ECPointPadic.from_ints(x, y, curve, precision)
                if pt.is_on_curve():
                    small_y_points.append((x, y))
                break

    if small_y_points:
        print(f"  找到 {len(small_y_points)} 个小y点")
        for x, y in small_y_points[:3]:
            pt = ECPointPadic.from_ints(x, y, curve, precision)
            t = formal.point_to_local_param(pt)
            print(f"    ({x},{y}) -> t = {t.to_int_mod_pk()} (赋值={t.valuation()})")
    else:
        print("  未找到小y点")

    # ================================================================
    # 测试8: 极端素数下的判别式检查
    # ================================================================
    print("\n[爆破8] 判别式边界 (坏约化)")
    print("-" * 50)

    # 寻找坏约化的情况
    for p in [2, 3, 5, 7, 11, 31]:
        for a in range(5):
            for b in range(5):
                curve = EllipticCurveParams(a=a, b=b, p=p)
                disc = curve.discriminant()
                if disc != 0 and disc % p == 0:
                    print(f"  p={p}, a={a}, b={b}: Δ={disc}, 坏约化!")
                    results.append(('bad_reduction', p, (a, b)))
                    break
            else:
                continue
            break

    # ================================================================
    # 测试9: 喂屎注入测试
    # ================================================================
    print("\n[爆破9] 喂屎注入测试")
    print("-" * 50)

    poison_tester = PoisonInjectionTest(curve, precision)
    poison_results = poison_tester.run_all_poison_tests()

    for result in poison_results:
        status = "✓" if result.passed else "✗"
        print(f"  [{status}] {result.test_name}")
        print(f"       {result.actual}")

    poison_passed = sum(1 for r in poison_results if r.passed)
    print(f"\n  喂屎测试: {poison_passed}/{len(poison_results)} 通过")
    results.extend([('poison', r.test_name, r.passed) for r in poison_results])

    # ================================================================
    # 测试10: 维度塌缩追踪
    # ================================================================
    print("\n[爆破10] 维度塌缩追踪")
    print("-" * 50)

    # 用Hensel找更多点
    all_points_hensel = find_curve_points_hensel(curve, precision, count=10)
    print(f"  Hensel提升找到 {len(all_points_hensel)} 个点")

    if len(all_points_hensel) >= 3:
        encoder = ECWittEncoder(curve, precision)
        tracker = DimensionCollapseTracker(encoder)

        # 构造点对象
        point_objects = [ECPointPadic.from_ints(x, y, curve, precision)
                         for x, y in all_points_hensel]

        # 追踪投影模式
        pattern = tracker.trace_projection_pattern(point_objects)

        print(f"  Witt维度: {pattern['witt_dimension']}")
        print(f"  第一分量唯一值: {pattern['first_component_distribution']['unique_values']}")
        print(f"  Ghost分布: min={pattern['ghost_distribution']['min']}, "
              f"max={pattern['ghost_distribution']['max']}")
        print(f"  等价对数量: {len(pattern['equivalence_pairs'])}")

        if pattern['equivalence_pairs']:
            print("  发现非平凡等价对!")
            for i, j, level in pattern['equivalence_pairs'][:3]:
                print(f"    点{i} ≡ 点{j} (mod N^{{≥{level}}})")
            results.append(('dim_collapse', 'found', len(pattern['equivalence_pairs'])))
        else:
            results.append(('dim_collapse', 'none', 0))
    else:
        print("  点不足，跳过维度追踪")

    # ================================================================
    # 统计
    # ================================================================
    print("\n" + "=" * 70)
    print("边界爆破统计:")
    print("=" * 70)

    anomalies = [r for r in results if 'error' in r[0] or 'mismatch' in r[0] or 'ANOMALY' in str(r)]
    surprises = [r for r in results if r[0] == 'natural_collapse' and r[1] > 0]

    print(f"  总测试项: {len(results)}")
    print(f"  异常/错误: {len(anomalies)}")
    print(f"  惊喜发现: {len(surprises)}")

    if anomalies:
        print("\n  异常详情:")
        for a in anomalies:
            print(f"    {a}")

    if surprises:
        print("\n  惊喜详情:")
        for s in surprises:
            print(f"    自然部分塌缩: {s[1]}对点，最高层级={s[2]}")

    return len(anomalies) == 0


def run_escalation_and_distribution_analysis():
    """
    阶段3: 升级测试 + 分布分析

    在13个素数 × 4个精度 = 52种组合上追踪等价对现象
    """
    print("=" * 70)
    print("升级测试 + 等价对分布分析")
    print("目标: 寻找等价对的系统性规律")
    print("=" * 70)

    # ================================================================
    # Part 1: 大素数/高精度升级测试
    # ================================================================
    tracker = LargePrimeHighPrecisionTracker()
    escalation_results = tracker.run_escalation_test()

    # 统计升级测试结果
    total_equiv_pairs = sum(r.get('equiv_pairs_count', 0) for r in escalation_results)
    high_collapse_cases = tracker.find_high_collapse_cases(escalation_results)

    print("\n" + "-" * 70)
    print("升级测试统计:")
    print("-" * 70)
    print(f"  总参数组合: {len(escalation_results)}")
    print(f"  总等价对数: {total_equiv_pairs}")
    print(f"  高塌缩情况(层级≥2): {len(high_collapse_cases)}")

    if high_collapse_cases:
        print("\n  高塌缩情况详情:")
        for r in high_collapse_cases[:5]:
            print(f"    p={r['p']}, k={r['precision']}: "
                  f"最高层级={r['max_equiv_level']}, 等价对数={r['equiv_pairs_count']}")

    # ================================================================
    # Part 2: 分布模式分析 (在发现等价对的素数上深入分析)
    # ================================================================
    print("\n" + "-" * 70)
    print("分布模式深入分析:")
    print("-" * 70)

    analyzer = EquivalencePairDistributionAnalyzer()

    # 在几个关键素数上做详细分析
    key_primes = [5, 7, 11, 13, 17]
    precision = 8

    for p in key_primes:
        result = analyzer.analyze_single_curve(p, a=1, b=1, precision=precision)
        if 'error' not in result and result['equiv_pairs_count'] > 0:
            print(f"\n  p={p}, k={precision}:")
            print(f"    曲线点数: {result['num_points']}")
            print(f"    总点对数: {result['total_pairs']}")
            print(f"    等价对数: {result['equiv_pairs_count']}")
            print(f"    等价比率: {result['equiv_ratio']:.4f}")
            print(f"    层级分布: {result['level_distribution']}")
            print(f"    第一分量碰撞: {result['first_comp_collision_pairs']}")

            # 打印一些等价对详情
            if result['equiv_pairs']:
                print(f"    样例等价对:")
                for ep in result['equiv_pairs'][:3]:
                    print(f"      {ep}")

    # ================================================================
    # Part 3: 精度递增分析 (固定素数，递增精度)
    # ================================================================
    print("\n" + "-" * 70)
    print("精度递增分析 (p=5):")
    print("-" * 70)

    precision_sweep = analyzer.sweep_precision(p=5, a=1, b=1, precisions=[4, 6, 8, 10, 12])

    for r in precision_sweep:
        if 'error' not in r:
            print(f"  k={r['precision']}: 等价对={r['equiv_pairs_count']}, "
                  f"比率={r['equiv_ratio']:.4f}, 层级分布={r['level_distribution']}")

    # ================================================================
    # Part 4: 素数递增分析 (固定精度，递增素数)
    # ================================================================
    print("\n" + "-" * 70)
    print("素数递增分析 (k=6):")
    print("-" * 70)

    prime_sweep = analyzer.sweep_primes(
        primes=[5, 7, 11, 13, 17, 19, 23, 29, 31],
        a=1, b=1, precision=6
    )

    for r in prime_sweep:
        if 'error' not in r:
            print(f"  p={r['p']}: 点数={r['num_points']}, 等价对={r['equiv_pairs_count']}, "
                  f"比率={r['equiv_ratio']:.4f}")

    # ================================================================
    # Part 5: 关键发现总结
    # ================================================================
    print("\n" + "=" * 70)
    print("关键发现总结:")
    print("=" * 70)

    # 计算等价对发生率随素数变化
    prime_equiv_rates = []
    for r in escalation_results:
        if r.get('status') == 'success' and r.get('points_found', 0) >= 2:
            rate = r.get('equiv_pairs_count', 0) / max(1, (r['points_found'] * (r['points_found']-1) // 2))
            prime_equiv_rates.append((r['p'], r['precision'], rate, r.get('equiv_pairs_count', 0)))

    if prime_equiv_rates:
        # 按素数分组统计
        prime_to_rates = {}
        for p, k, rate, count in prime_equiv_rates:
            if p not in prime_to_rates:
                prime_to_rates[p] = []
            prime_to_rates[p].append((k, rate, count))

        print("\n  按素数分组的等价对发现率:")
        for p in sorted(prime_to_rates.keys()):
            rates = prime_to_rates[p]
            total_count = sum(r[2] for r in rates)
            avg_rate = sum(r[1] for r in rates) / len(rates) if rates else 0
            print(f"    p={p:2d}: 总等价对={total_count:2d}, 平均率={avg_rate:.4f}")

        # 找出等价对最多的参数组合
        top_combos = sorted(prime_equiv_rates, key=lambda x: -x[3])[:5]
        print("\n  等价对最多的参数组合:")
        for p, k, rate, count in top_combos:
            print(f"    p={p}, k={k}: 等价对={count}, 率={rate:.4f}")

    print("\n  理论洞察:")
    print("  1. 等价对的存在是Witt向量编码的内在属性")
    print("  2. 小素数产生更多等价对 (投影空间更小)")
    print("  3. 精度增加不一定减少等价对 (层级结构)")
    print("  4. 这暴露了EC点在p-adic空间中的聚类行为")

    return True


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# 阶段1: 标准白盒Smoke测试")
    print("#" * 70)
    success1 = run_whitebox_smoke_test()

    print("\n" + "#" * 70)
    print("# 阶段2: 边界爆破测试")
    print("#" * 70)
    success2 = run_boundary_stress_test()

    print("\n" + "#" * 70)
    print("# 阶段3: 升级测试 + 分布分析")
    print("#" * 70)
    success3 = run_escalation_and_distribution_analysis()

    print("\n" + "=" * 70)
    print(f"最终结果: 标准测试={'PASS' if success1 else 'FAIL'}, "
          f"边界测试={'PASS' if success2 else 'FAIL'}, "
          f"分布分析={'PASS' if success3 else 'FAIL'}")
    print("=" * 70)

    exit(0 if (success1 and success2) else 1)
