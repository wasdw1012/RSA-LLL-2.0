"""
Knuth-Bendix 完备化算法核心实现

数学基础：
    Knuth, D.E. & Bendix, P.B. (1970). "Simple Word Problems in Universal Algebras"

    给定有限表现群 ⟨S | R⟩：
    - S: 生成元集合（字母表）
    - R: 关系集合（等式）

    Knuth-Bendix 过程将等式转化为定向重写规则，
    通过检测和解决临界对（critical pairs）来达到完备性。

铁律（继承 MVP19）：
    - 禁止魔法数：所有阈值/上限必须由数学定义或问题尺度导出
    - 禁止静默失败：任何异常情况必须显式抛出
    - 禁止诡异归一化：保留原始代数结构

架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Knuth-Bendix 完备化引擎                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Word（词）                                                          │
│   - 字母表上的有限序列                                                        │
│   - 支持逆元（群表现）                                                        │
│   - 自由约化（aa⁻¹ → ε）                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: TermOrder（项序）                                                   │
│   - ShortLex: 先比长度，再比字典序                                            │
│   - DegLex: 加权度 + 字典序                                                  │
│   - 可扩展的抽象序接口                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 3: RewriteRule（重写规则）                                             │
│   - 定向等式 l → r，保证 l > r                                               │
│   - 规则应用：词中子串匹配与替换                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 4: RewriteSystem（重写系统）                                           │
│   - 规则集合管理                                                             │
│   - 规约到正规形式                                                           │
│   - 规则间约化（interreduction）                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 5: CriticalPair（临界对）                                              │
│   - 重叠检测（overlap）                                                      │
│   - 包含检测（inclusion）                                                    │
│   - 精确的组合枚举                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 6: KnuthBendixCompletion（完备化过程）                                  │
│   - 主循环：检测临界对 → 规约 → 添加规则                                      │
│   - 终止检测与发散处理                                                       │
│   - 完备性证明输出                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import (
    Any, Callable, Dict, FrozenSet, Generic, Iterable, Iterator,
    List, Mapping, Optional, Protocol, Sequence, Set, Tuple, TypeVar, Union
)
import hashlib
import itertools


# =============================================================================
# 0) 严格异常系统
# =============================================================================


class PresentationError(Exception):
    """Knuth-Bendix 模块总异常基类。"""


class AlphabetError(PresentationError):
    """字母表定义错误（重复符号/非法字符等）。"""


class WordError(PresentationError):
    """词构造或操作错误（非法符号/索引越界等）。"""


class TermOrderError(PresentationError):
    """项序定义或比较错误（不可比/序违反等）。"""


class RewriteRuleError(PresentationError):
    """重写规则错误（左侧不大于右侧/空规则等）。"""


class RewriteSystemError(PresentationError):
    """重写系统错误（规则冲突/系统不一致等）。"""


class CriticalPairError(PresentationError):
    """临界对计算错误（重叠检测失败等）。"""


class CompletionError(PresentationError):
    """完备化过程错误（不终止/资源耗尽等）。"""


class NonTerminationError(CompletionError):
    """完备化过程检测到不终止。"""


# =============================================================================
# 1) 字母表与符号
# =============================================================================


@dataclass(frozen=True, order=True)
class Symbol:
    """
    字母表中的符号。

    数学定义：
        符号是字母表 Σ 的元素。对于群表现，我们使用扩展字母表
        Σ̃ = Σ ∪ Σ⁻¹，其中 Σ⁻¹ = {a⁻¹ : a ∈ Σ}。

    属性：
        name: 符号名称（如 "a", "b", "x"）
        index: 符号在字母表中的索引（确定性排序）
        is_inverse: 是否为逆元符号
    """
    name: str
    index: int
    is_inverse: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise AlphabetError("Symbol name cannot be empty")
        if self.index < 0:
            raise AlphabetError(f"Symbol index must be non-negative, got {self.index}")

    def inverse(self) -> Symbol:
        """返回该符号的逆元。"""
        return Symbol(
            name=self.name,
            index=self.index,
            is_inverse=not self.is_inverse
        )

    def __repr__(self) -> str:
        if self.is_inverse:
            return f"{self.name}⁻¹"
        return self.name

    def __hash__(self) -> int:
        return hash((self.name, self.index, self.is_inverse))


class Alphabet:
    """
    字母表：符号的有序集合。

    数学定义：
        字母表 Σ 是一个有限非空集合，配备全序 <。
        对于群表现，自动生成逆元字母表 Σ⁻¹。

    实现细节：
        - 符号按索引排序（确定性）
        - 支持通过名称或索引访问
        - 不可变（创建后不能添加/删除符号）
    """

    def __init__(self, names: Sequence[str], *, with_inverses: bool = True):
        """
        创建字母表。

        Args:
            names: 生成元名称序列（顺序决定索引）
            with_inverses: 是否自动生成逆元（群表现需要）
        """
        if not names:
            raise AlphabetError("Alphabet cannot be empty")

        # 检查重复
        seen: Set[str] = set()
        for name in names:
            if name in seen:
                raise AlphabetError(f"Duplicate symbol name: {name}")
            seen.add(name)

        # 创建符号
        self._symbols: Tuple[Symbol, ...] = tuple(
            Symbol(name=name, index=i, is_inverse=False)
            for i, name in enumerate(names)
        )

        self._with_inverses = with_inverses

        # 名称到符号的映射
        self._name_to_symbol: Dict[str, Symbol] = {
            s.name: s for s in self._symbols
        }

        # 如果需要逆元，创建逆元符号
        if with_inverses:
            self._inverse_symbols: Tuple[Symbol, ...] = tuple(
                s.inverse() for s in self._symbols
            )
        else:
            self._inverse_symbols = ()

    @property
    def generators(self) -> Tuple[Symbol, ...]:
        """返回生成元（不含逆元）。"""
        return self._symbols

    @property
    def all_symbols(self) -> Tuple[Symbol, ...]:
        """返回所有符号（含逆元）。"""
        if self._with_inverses:
            return self._symbols + self._inverse_symbols
        return self._symbols

    def __len__(self) -> int:
        """返回生成元数量（不含逆元）。"""
        return len(self._symbols)

    def __getitem__(self, key: Union[int, str]) -> Symbol:
        """通过索引或名称获取符号。"""
        if isinstance(key, int):
            if 0 <= key < len(self._symbols):
                return self._symbols[key]
            raise AlphabetError(f"Symbol index {key} out of range [0, {len(self._symbols)})")
        elif isinstance(key, str):
            if key in self._name_to_symbol:
                return self._name_to_symbol[key]
            raise AlphabetError(f"Unknown symbol name: {key}")
        raise AlphabetError(f"Invalid key type: {type(key)}")

    def get_inverse(self, symbol: Symbol) -> Symbol:
        """获取符号的逆元。"""
        if not self._with_inverses:
            raise AlphabetError("Alphabet does not support inverses")
        return symbol.inverse()

    def contains(self, symbol: Symbol) -> bool:
        """检查符号是否属于此字母表。"""
        if symbol.is_inverse:
            base = symbol.inverse()
        else:
            base = symbol
        return (
            base.name in self._name_to_symbol and
            self._name_to_symbol[base.name].index == base.index
        )

    def __repr__(self) -> str:
        names = ", ".join(s.name for s in self._symbols)
        if self._with_inverses:
            return f"Alphabet⟨{names} | with inverses⟩"
        return f"Alphabet⟨{names}⟩"


# =============================================================================
# 2) 词（Word）
# =============================================================================


@dataclass(frozen=True)
class Word:
    """
    字母表上的词（有限符号序列）。

    数学定义：
        词是字母表 Σ 上的有限序列 w = a₁a₂...aₙ，其中 aᵢ ∈ Σ̃。
        空词记为 ε，长度为 0。

        对于群表现，词在自由群 F(Σ) 中表示元素。
        自由约化：移除所有形如 aa⁻¹ 或 a⁻¹a 的相邻对。

    实现细节：
        - 不可变（frozen dataclass）
        - 支持拼接、切片、子词查找
        - 可选的自由约化
    """
    symbols: Tuple[Symbol, ...]
    alphabet: Alphabet
    _is_reduced: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        # 验证所有符号属于字母表
        for s in self.symbols:
            if not self.alphabet.contains(s):
                raise WordError(f"Symbol {s} not in alphabet {self.alphabet}")

    @classmethod
    def empty(cls, alphabet: Alphabet) -> Word:
        """创建空词 ε。"""
        return cls(symbols=(), alphabet=alphabet, _is_reduced=True)

    @classmethod
    def from_symbols(cls, symbols: Sequence[Symbol], alphabet: Alphabet) -> Word:
        """从符号序列创建词。"""
        return cls(symbols=tuple(symbols), alphabet=alphabet)

    @classmethod
    def from_string(cls, s: str, alphabet: Alphabet) -> Word:
        """
        从字符串解析词。

        格式：符号名用空格分隔，逆元用 ^ 后缀表示。
        例如："a b a^" 表示 a·b·a⁻¹
        """
        if not s.strip():
            return cls.empty(alphabet)

        symbols: List[Symbol] = []
        for token in s.split():
            if token.endswith("^"):
                name = token[:-1]
                sym = alphabet[name]
                symbols.append(sym.inverse())
            else:
                symbols.append(alphabet[token])

        return cls(symbols=tuple(symbols), alphabet=alphabet)

    def __len__(self) -> int:
        """词的长度。"""
        return len(self.symbols)

    def __getitem__(self, key: Union[int, slice]) -> Union[Symbol, Word]:
        """索引或切片访问。"""
        if isinstance(key, int):
            return self.symbols[key]
        elif isinstance(key, slice):
            return Word(
                symbols=self.symbols[key],
                alphabet=self.alphabet
            )
        raise WordError(f"Invalid key type: {type(key)}")

    def __add__(self, other: Word) -> Word:
        """词的拼接（concatenation）。"""
        if self.alphabet is not other.alphabet:
            # 允许相同定义的不同实例
            if repr(self.alphabet) != repr(other.alphabet):
                raise WordError("Cannot concatenate words from different alphabets")
        return Word(
            symbols=self.symbols + other.symbols,
            alphabet=self.alphabet
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Word):
            return NotImplemented
        return self.symbols == other.symbols

    def __hash__(self) -> int:
        return hash(self.symbols)

    def __repr__(self) -> str:
        if not self.symbols:
            return "ε"
        return "·".join(repr(s) for s in self.symbols)

    def is_empty(self) -> bool:
        """是否为空词。"""
        return len(self.symbols) == 0

    def inverse(self) -> Word:
        """
        词的逆元：(a₁a₂...aₙ)⁻¹ = aₙ⁻¹...a₂⁻¹a₁⁻¹
        """
        inv_symbols = tuple(s.inverse() for s in reversed(self.symbols))
        return Word(symbols=inv_symbols, alphabet=self.alphabet)

    def freely_reduce(self) -> Word:
        """
        自由约化：移除所有 aa⁻¹ 和 a⁻¹a 对。

        算法复杂度：O(n)，使用栈模拟。
        """
        if self._is_reduced:
            return self

        stack: List[Symbol] = []
        for s in self.symbols:
            if stack and stack[-1] == s.inverse():
                stack.pop()
            else:
                stack.append(s)

        # 使用 object.__setattr__ 绕过 frozen
        result = Word(symbols=tuple(stack), alphabet=self.alphabet)
        object.__setattr__(result, '_is_reduced', True)
        return result

    def has_subword(self, sub: Word) -> bool:
        """检查是否包含子词。"""
        if len(sub) > len(self):
            return False
        if len(sub) == 0:
            return True

        # 朴素匹配（可优化为 KMP，但对于典型长度足够）
        for i in range(len(self) - len(sub) + 1):
            if self.symbols[i:i+len(sub)] == sub.symbols:
                return True
        return False

    def find_subword(self, sub: Word) -> List[int]:
        """
        找到所有子词出现位置。

        Returns:
            子词起始位置的列表（可能为空）
        """
        if len(sub) == 0:
            return list(range(len(self) + 1))
        if len(sub) > len(self):
            return []

        positions: List[int] = []
        for i in range(len(self) - len(sub) + 1):
            if self.symbols[i:i+len(sub)] == sub.symbols:
                positions.append(i)
        return positions

    def replace_at(self, start: int, length: int, replacement: Word) -> Word:
        """
        在指定位置替换子词。

        Args:
            start: 替换起始位置
            length: 被替换的长度
            replacement: 替换词

        Returns:
            新词 = self[:start] + replacement + self[start+length:]
        """
        if start < 0 or start > len(self):
            raise WordError(f"Start position {start} out of range [0, {len(self)}]")
        if length < 0 or start + length > len(self):
            raise WordError(f"Invalid replacement range [{start}, {start+length})")

        new_symbols = (
            self.symbols[:start] +
            replacement.symbols +
            self.symbols[start+length:]
        )
        return Word(symbols=new_symbols, alphabet=self.alphabet)

    def prefix(self, length: int) -> Word:
        """返回长度为 length 的前缀。"""
        if length < 0 or length > len(self):
            raise WordError(f"Prefix length {length} out of range [0, {len(self)}]")
        return Word(symbols=self.symbols[:length], alphabet=self.alphabet)

    def suffix(self, length: int) -> Word:
        """返回长度为 length 的后缀。"""
        if length < 0 or length > len(self):
            raise WordError(f"Suffix length {length} out of range [0, {len(self)}]")
        return Word(symbols=self.symbols[-length:] if length > 0 else (), alphabet=self.alphabet)


# =============================================================================
# 3) 项序（Term Ordering）
# =============================================================================


class TermOrder(ABC):
    """
    项序的抽象基类。

    数学定义：
        项序 > 是 Σ* 上的全序，满足：
        1. 良基性（well-founded）：没有无限下降链
        2. 与拼接相容：u > v ⟹ xuy > xvy 对所有 x, y

        这些性质保证重写过程终止。
    """

    @abstractmethod
    def compare(self, w1: Word, w2: Word) -> int:
        """
        比较两个词。

        Returns:
            > 0 如果 w1 > w2
            < 0 如果 w1 < w2
            = 0 如果 w1 = w2
        """
        pass

    def greater(self, w1: Word, w2: Word) -> bool:
        """w1 > w2"""
        return self.compare(w1, w2) > 0

    def less(self, w1: Word, w2: Word) -> bool:
        """w1 < w2"""
        return self.compare(w1, w2) < 0

    def equal(self, w1: Word, w2: Word) -> bool:
        """w1 = w2（作为词，非群元素）"""
        return self.compare(w1, w2) == 0

    def greater_or_equal(self, w1: Word, w2: Word) -> bool:
        """w1 ≥ w2"""
        return self.compare(w1, w2) >= 0

    def max(self, w1: Word, w2: Word) -> Word:
        """返回较大者。"""
        return w1 if self.compare(w1, w2) >= 0 else w2

    def min(self, w1: Word, w2: Word) -> Word:
        """返回较小者。"""
        return w1 if self.compare(w1, w2) <= 0 else w2


class ShortLexOrder(TermOrder):
    """
    短词典序（ShortLex / Length-Lexicographic Order）。

    数学定义：
        w1 > w2 当且仅当：
        1. |w1| > |w2|，或
        2. |w1| = |w2| 且 w1 在字典序下大于 w2

    性质：
        - 良基性：长度有限保证无无限下降链
        - 与拼接相容：添加相同前后缀不改变相对顺序

    这是 Knuth-Bendix 最常用的项序。
    """

    def __init__(self, alphabet: Alphabet):
        """
        初始化短词典序。

        Args:
            alphabet: 字母表（其符号顺序决定字典序）
        """
        self.alphabet = alphabet

        # 构建符号到排序键的映射
        # 正符号在前，逆元符号在后
        self._symbol_rank: Dict[Symbol, int] = {}
        rank = 0
        for s in alphabet.generators:
            self._symbol_rank[s] = rank
            rank += 1
        for s in alphabet.generators:
            self._symbol_rank[s.inverse()] = rank
            rank += 1

    def _symbol_key(self, s: Symbol) -> int:
        """获取符号的排序键。"""
        if s not in self._symbol_rank:
            raise TermOrderError(f"Symbol {s} not in alphabet")
        return self._symbol_rank[s]

    def compare(self, w1: Word, w2: Word) -> int:
        """
        短词典序比较。

        复杂度：O(min(|w1|, |w2|))
        """
        # 先比较长度
        len_diff = len(w1) - len(w2)
        if len_diff != 0:
            return len_diff

        # 长度相同，逐符号比较
        for s1, s2 in zip(w1.symbols, w2.symbols):
            k1 = self._symbol_key(s1)
            k2 = self._symbol_key(s2)
            if k1 != k2:
                return k1 - k2

        return 0

    def __repr__(self) -> str:
        return f"ShortLexOrder({self.alphabet})"


class WeightedLexOrder(TermOrder):
    """
    加权词典序（Weighted Lexicographic Order）。

    数学定义：
        给定权重函数 ω: Σ̃ → ℤ₊，定义词的权重为 ω(w) = Σᵢ ω(wᵢ)。
        w1 > w2 当且仅当：
        1. ω(w1) > ω(w2)，或
        2. ω(w1) = ω(w2) 且 w1 在字典序下大于 w2

    用途：
        可以给某些生成元更高权重，引导完备化朝特定方向进行。
    """

    def __init__(self, alphabet: Alphabet, weights: Optional[Dict[Symbol, int]] = None):
        """
        初始化加权词典序。

        Args:
            alphabet: 字母表
            weights: 符号权重映射（默认所有符号权重为 1）
        """
        self.alphabet = alphabet

        # 默认权重
        if weights is None:
            self._weights: Dict[Symbol, int] = {s: 1 for s in alphabet.all_symbols}
        else:
            self._weights = dict(weights)
            # 验证所有符号都有权重
            for s in alphabet.all_symbols:
                if s not in self._weights:
                    raise TermOrderError(f"Missing weight for symbol {s}")
                if self._weights[s] <= 0:
                    raise TermOrderError(f"Weight must be positive, got {self._weights[s]} for {s}")

        # 符号排序键
        self._symbol_rank: Dict[Symbol, int] = {}
        rank = 0
        for s in alphabet.generators:
            self._symbol_rank[s] = rank
            rank += 1
        for s in alphabet.generators:
            self._symbol_rank[s.inverse()] = rank
            rank += 1

    def weight(self, word: Word) -> int:
        """计算词的权重。"""
        return sum(self._weights[s] for s in word.symbols)

    def compare(self, w1: Word, w2: Word) -> int:
        """加权词典序比较。"""
        # 先比较权重
        weight_diff = self.weight(w1) - self.weight(w2)
        if weight_diff != 0:
            return weight_diff

        # 权重相同，比较长度
        len_diff = len(w1) - len(w2)
        if len_diff != 0:
            return len_diff

        # 逐符号比较
        for s1, s2 in zip(w1.symbols, w2.symbols):
            k1 = self._symbol_rank[s1]
            k2 = self._symbol_rank[s2]
            if k1 != k2:
                return k1 - k2

        return 0


class RecursivePathOrder(TermOrder):
    """
    递归路径序（Recursive Path Order / RPO）。

    数学定义（Dershowitz, 1982）：
        这是一种更强的项序，特别适合处理带有结构的项。
        对于词（视为一元函数的嵌套），简化为：

        w1 > w2 当且仅当存在 w1 的真前缀 p 使得 p ≥ w2，
        或者 w1 = a·u, w2 = b·v 且：
          - a > b（符号序），或
          - a = b 且 u > v（递归）

    性质：
        - 是良基的和与拼接相容的
        - 严格强于 ShortLex
    """

    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet
        self._symbol_rank: Dict[Symbol, int] = {}
        rank = 0
        for s in alphabet.generators:
            self._symbol_rank[s] = rank
            rank += 1
        for s in alphabet.generators:
            self._symbol_rank[s.inverse()] = rank
            rank += 1

    def compare(self, w1: Word, w2: Word) -> int:
        """递归路径序比较。"""
        # 空词是最小的
        if w1.is_empty() and w2.is_empty():
            return 0
        if w1.is_empty():
            return -1
        if w2.is_empty():
            return 1

        # 检查 w1 的真前缀是否 >= w2
        for i in range(1, len(w1)):
            prefix = w1.prefix(i)
            if self.compare(prefix, w2) >= 0:
                return 1

        # 检查 w2 的真前缀是否 >= w1
        for i in range(1, len(w2)):
            prefix = w2.prefix(i)
            if self.compare(prefix, w1) >= 0:
                return -1

        # 比较首符号
        s1, s2 = w1.symbols[0], w2.symbols[0]
        k1 = self._symbol_rank[s1]
        k2 = self._symbol_rank[s2]

        if k1 != k2:
            return k1 - k2

        # 首符号相同，递归比较后缀
        return self.compare(w1[1:], w2[1:])


# =============================================================================
# 4) 重写规则（Rewrite Rule）
# =============================================================================


@dataclass(frozen=True)
class RewriteRule:
    """
    重写规则 l → r。

    数学定义：
        重写规则是有序对 (l, r)，其中 l, r ∈ Σ* 且 l > r（按项序）。
        规则应用：如果词 w 包含 l 作为子词，即 w = ulv，
        则可以重写为 w' = urv。

    不变量：
        - l 非空（不允许 ε → r）
        - l > r（保证终止性）
    """
    lhs: Word  # 左侧（left-hand side）
    rhs: Word  # 右侧（right-hand side）
    order: TermOrder

    def __post_init__(self) -> None:
        if self.lhs.is_empty():
            raise RewriteRuleError("Left-hand side cannot be empty")
        if not self.order.greater(self.lhs, self.rhs):
            raise RewriteRuleError(
                f"Left-hand side must be greater than right-hand side: "
                f"{self.lhs} ≯ {self.rhs}"
            )

    def __repr__(self) -> str:
        return f"{self.lhs} → {self.rhs}"

    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RewriteRule):
            return NotImplemented
        return self.lhs == other.lhs and self.rhs == other.rhs

    def applies_at(self, word: Word, position: int) -> bool:
        """检查规则是否可在指定位置应用。"""
        if position < 0 or position + len(self.lhs) > len(word):
            return False
        return word.symbols[position:position+len(self.lhs)] == self.lhs.symbols

    def apply_at(self, word: Word, position: int) -> Word:
        """
        在指定位置应用规则。

        Raises:
            RewriteRuleError: 如果规则不能在该位置应用
        """
        if not self.applies_at(word, position):
            raise RewriteRuleError(
                f"Rule {self} cannot be applied at position {position} of {word}"
            )
        return word.replace_at(position, len(self.lhs), self.rhs)

    def find_applications(self, word: Word) -> List[int]:
        """找到所有可应用位置。"""
        return word.find_subword(self.lhs)

    def apply_first(self, word: Word) -> Optional[Word]:
        """
        应用规则到第一个匹配位置。

        Returns:
            重写后的词，如果无法应用则返回 None
        """
        positions = self.find_applications(word)
        if not positions:
            return None
        return self.apply_at(word, positions[0])

    def apply_leftmost(self, word: Word) -> Optional[Word]:
        """应用到最左匹配（同 apply_first）。"""
        return self.apply_first(word)

    def is_length_reducing(self) -> bool:
        """规则是否减少长度。"""
        return len(self.lhs) > len(self.rhs)

    def is_length_preserving(self) -> bool:
        """规则是否保持长度。"""
        return len(self.lhs) == len(self.rhs)


# =============================================================================
# 5) 重写系统（Rewrite System）
# =============================================================================


class RewriteSystem:
    """
    重写系统：重写规则的有序集合。

    数学定义：
        重写系统 R 是重写规则的有限集合。
        R 的重写关系 →_R 定义为：w →_R w' 当且仅当存在规则 l → r ∈ R
        和分解 w = ulv 使得 w' = urv。

        →*_R 是 →_R 的自反传递闭包。

    性质：
        - 终止性（terminating）：没有无限重写序列
        - 合流性（confluent）：不同重写路径最终汇合
        - 完备性 = 终止性 + 合流性（Church-Rosser 性质）
    """

    def __init__(self, order: TermOrder):
        """
        初始化空重写系统。

        Args:
            order: 用于所有规则的项序
        """
        self.order = order
        self._rules: List[RewriteRule] = []
        self._rule_set: Set[RewriteRule] = set()

    @property
    def rules(self) -> Tuple[RewriteRule, ...]:
        """返回规则的不可变视图。"""
        return tuple(self._rules)

    def __len__(self) -> int:
        """规则数量。"""
        return len(self._rules)

    def __iter__(self) -> Iterator[RewriteRule]:
        """迭代所有规则。"""
        return iter(self._rules)

    def __contains__(self, rule: RewriteRule) -> bool:
        """检查规则是否在系统中。"""
        return rule in self._rule_set

    def add_rule(self, rule: RewriteRule) -> bool:
        """
        添加规则。

        Returns:
            True 如果规则被添加（之前不存在），False 如果已存在
        """
        if rule in self._rule_set:
            return False
        self._rules.append(rule)
        self._rule_set.add(rule)
        return True

    def add_equation(self, lhs: Word, rhs: Word) -> Optional[RewriteRule]:
        """
        从等式创建并添加规则。

        等式 lhs = rhs 被定向为 max(lhs, rhs) → min(lhs, rhs)。

        Returns:
            创建的规则，如果等式平凡（lhs = rhs）则返回 None
        """
        if lhs == rhs:
            return None

        if self.order.greater(lhs, rhs):
            rule = RewriteRule(lhs=lhs, rhs=rhs, order=self.order)
        else:
            rule = RewriteRule(lhs=rhs, rhs=lhs, order=self.order)

        if self.add_rule(rule):
            return rule
        return None

    def remove_rule(self, rule: RewriteRule) -> bool:
        """
        移除规则。

        Returns:
            True 如果规则被移除，False 如果不存在
        """
        if rule not in self._rule_set:
            return False
        self._rules.remove(rule)
        self._rule_set.remove(rule)
        return True

    def reduce_once(self, word: Word) -> Tuple[Word, Optional[RewriteRule]]:
        """
        应用一步重写（最左最内策略）。

        Returns:
            (新词, 应用的规则) 或 (原词, None) 如果无法应用
        """
        # 按规则优先级（添加顺序）和位置优先（最左）
        best_pos = len(word) + 1
        best_rule: Optional[RewriteRule] = None

        for rule in self._rules:
            positions = rule.find_applications(word)
            if positions and positions[0] < best_pos:
                best_pos = positions[0]
                best_rule = rule

        if best_rule is None:
            return word, None

        return best_rule.apply_at(word, best_pos), best_rule

    def reduce(self, word: Word, *, max_steps: Optional[int] = None) -> Tuple[Word, int]:
        """
        规约到正规形式（不可再规约的词）。

        Args:
            word: 输入词
            max_steps: 最大步数限制（None 表示无限制）

        Returns:
            (正规形式, 步数)

        Raises:
            CompletionError: 如果超过最大步数
        """
        current = word
        steps = 0

        while True:
            if max_steps is not None and steps >= max_steps:
                raise CompletionError(
                    f"Reduction exceeded {max_steps} steps, possible non-termination"
                )

            next_word, rule = self.reduce_once(current)
            if rule is None:
                return current, steps

            current = next_word
            steps += 1

    def normal_form(self, word: Word, *, max_steps: Optional[int] = None) -> Word:
        """规约到正规形式（只返回结果）。"""
        nf, _ = self.reduce(word, max_steps=max_steps)
        return nf

    def are_equivalent(self, w1: Word, w2: Word, *, max_steps: Optional[int] = None) -> bool:
        """
        检查两个词是否等价（规约到相同正规形式）。
        """
        nf1 = self.normal_form(w1, max_steps=max_steps)
        nf2 = self.normal_form(w2, max_steps=max_steps)
        return nf1 == nf2

    def is_reduced(self, word: Word) -> bool:
        """检查词是否已是正规形式。"""
        _, rule = self.reduce_once(word)
        return rule is None

    def interreduce(self) -> RewriteSystem:
        """
        规则间约化：用其他规则简化每个规则的右侧。

        这可以消除冗余规则并简化系统。

        Returns:
            新的约化后的重写系统
        """
        new_system = RewriteSystem(self.order)

        for rule in self._rules:
            # 用其他规则约化右侧
            temp_system = RewriteSystem(self.order)
            for other in self._rules:
                if other != rule:
                    temp_system.add_rule(other)

            reduced_rhs = temp_system.normal_form(rule.rhs)
            reduced_lhs = temp_system.normal_form(rule.lhs)

            # 如果左侧被约化了，规则可能需要更新或删除
            if reduced_lhs != rule.lhs:
                # 左侧被约化意味着规则可能冗余
                if reduced_lhs == reduced_rhs:
                    continue  # 规则变平凡，跳过
                if self.order.greater(reduced_lhs, reduced_rhs):
                    new_rule = RewriteRule(lhs=reduced_lhs, rhs=reduced_rhs, order=self.order)
                    new_system.add_rule(new_rule)
            elif reduced_rhs != rule.rhs:
                # 只有右侧被约化
                new_rule = RewriteRule(lhs=rule.lhs, rhs=reduced_rhs, order=self.order)
                new_system.add_rule(new_rule)
            else:
                new_system.add_rule(rule)

        return new_system


# =============================================================================
# 6) 临界对（Critical Pair）
# =============================================================================


@dataclass(frozen=True)
class CriticalPair:
    """
    临界对：两个规则重叠产生的潜在冲突。

    数学定义：
        给定规则 r1: l1 → r1 和 r2: l2 → r2，
        如果 l1 和 l2 有非平凡重叠（即存在词 w 同时匹配两个左侧的不同部分），
        则产生临界对 (u, v)，其中 u 和 v 是从 w 分别应用两个规则得到的结果。

    重叠类型：
        1. 包含重叠（inclusion）：l2 是 l1 的子词
           l1 = u·l2·v，临界对 = (r1, u·r2·v)

        2. 交叉重叠（overlap）：l1 的后缀等于 l2 的前缀
           l1 = u·m, l2 = m·v（m 是非空公共部分）
           w = l1·v = u·l2
           临界对 = (r1·v, u·r2)
    """
    word1: Word  # 从 w 应用规则 1 得到
    word2: Word  # 从 w 应用规则 2 得到
    rule1: RewriteRule
    rule2: RewriteRule
    overlap_type: str  # "inclusion" 或 "overlap"
    overlap_word: Word  # 产生重叠的原始词 w

    def __repr__(self) -> str:
        return f"CriticalPair({self.word1}, {self.word2})"

    def is_joinable(self, system: RewriteSystem, *, max_steps: Optional[int] = None) -> bool:
        """
        检查临界对是否可合流（joinable）。

        即 word1 和 word2 是否规约到相同的正规形式。
        """
        nf1 = system.normal_form(self.word1, max_steps=max_steps)
        nf2 = system.normal_form(self.word2, max_steps=max_steps)
        return nf1 == nf2

    def get_difference(self, system: RewriteSystem, *, max_steps: Optional[int] = None) -> Optional[Tuple[Word, Word]]:
        """
        获取规约后的差异（如果存在）。

        Returns:
            (nf1, nf2) 如果不同，None 如果相同
        """
        nf1 = system.normal_form(self.word1, max_steps=max_steps)
        nf2 = system.normal_form(self.word2, max_steps=max_steps)
        if nf1 == nf2:
            return None
        return nf1, nf2


class CriticalPairFinder:
    """
    临界对查找器。

    实现精确的组合枚举，找出所有可能的临界对。
    """

    def __init__(self, system: RewriteSystem):
        self.system = system

    def find_inclusion_pairs(self, rule1: RewriteRule, rule2: RewriteRule) -> List[CriticalPair]:
        """
        查找包含重叠产生的临界对。

        检查 l2 是否作为 l1 的真子词出现（不在首尾位置，那属于交叉重叠）。
        """
        pairs: List[CriticalPair] = []
        l1, r1 = rule1.lhs, rule1.rhs
        l2, r2 = rule2.lhs, rule2.rhs

        # 找 l2 在 l1 中的所有出现位置
        positions = l1.find_subword(l2)

        for pos in positions:
            # 跳过首位置（属于交叉重叠的特例）
            if pos == 0:
                continue
            # 跳过会延伸到 l1 末尾之后的情况
            if pos + len(l2) > len(l1):
                continue

            # l1 = prefix · l2 · suffix
            prefix = l1.prefix(pos)
            suffix = l1[pos + len(l2):]

            # 应用 rule1: l1 → r1
            word1 = r1

            # 应用 rule2 到位置 pos: l1 → prefix · r2 · suffix
            word2 = prefix + r2 + suffix

            pairs.append(CriticalPair(
                word1=word1,
                word2=word2,
                rule1=rule1,
                rule2=rule2,
                overlap_type="inclusion",
                overlap_word=l1
            ))

        return pairs

    def find_overlap_pairs(self, rule1: RewriteRule, rule2: RewriteRule) -> List[CriticalPair]:
        """
        查找交叉重叠产生的临界对。

        检查 l1 的后缀是否等于 l2 的前缀。
        """
        pairs: List[CriticalPair] = []
        l1, r1 = rule1.lhs, rule1.rhs
        l2, r2 = rule2.lhs, rule2.rhs

        # 枚举所有可能的重叠长度
        max_overlap = min(len(l1), len(l2))

        for overlap_len in range(1, max_overlap + 1):
            # l1 的后缀
            l1_suffix = l1.suffix(overlap_len)
            # l2 的前缀
            l2_prefix = l2.prefix(overlap_len)

            if l1_suffix == l2_prefix:
                # 找到重叠！
                # l1 = u · m, l2 = m · v
                # w = u · l2 = l1 · v

                u = l1.prefix(len(l1) - overlap_len)  # l1 去掉后缀
                v = l2[overlap_len:]  # l2 去掉前缀

                # 重叠词
                overlap_word = l1 + v  # = u + l2

                # 应用 rule1 到 w = l1 · v: 得到 r1 · v
                word1 = r1 + v

                # 应用 rule2 到 w = u · l2: 得到 u · r2
                word2 = u + r2

                pairs.append(CriticalPair(
                    word1=word1,
                    word2=word2,
                    rule1=rule1,
                    rule2=rule2,
                    overlap_type="overlap",
                    overlap_word=overlap_word
                ))

        return pairs

    def find_all_pairs(self) -> List[CriticalPair]:
        """
        查找系统中所有临界对。

        对每对规则（包括规则与自身），检查所有可能的重叠。
        """
        all_pairs: List[CriticalPair] = []
        rules = list(self.system.rules)

        for i, rule1 in enumerate(rules):
            for j, rule2 in enumerate(rules):
                # 交叉重叠
                all_pairs.extend(self.find_overlap_pairs(rule1, rule2))

                # 包含重叠（rule2 包含在 rule1 中）
                if i != j:  # 不检查自包含（没有意义）
                    all_pairs.extend(self.find_inclusion_pairs(rule1, rule2))

        return all_pairs

    def find_non_joinable_pairs(self, *, max_steps: Optional[int] = None) -> List[CriticalPair]:
        """
        查找所有不可合流的临界对。
        """
        all_pairs = self.find_all_pairs()
        return [
            pair for pair in all_pairs
            if not pair.is_joinable(self.system, max_steps=max_steps)
        ]


# =============================================================================
# 7) Knuth-Bendix 完备化过程
# =============================================================================


class CompletionStatus(Enum):
    """完备化过程的状态。"""
    SUCCESS = auto()          # 成功完备
    MAX_RULES_EXCEEDED = auto()  # 规则数超过限制
    MAX_ITERATIONS_EXCEEDED = auto()  # 迭代次数超过限制
    NON_TERMINATING = auto()  # 检测到不终止


@dataclass
class CompletionResult:
    """完备化过程的结果。"""
    status: CompletionStatus
    system: RewriteSystem
    iterations: int
    critical_pairs_processed: int
    rules_added: int
    message: str


@dataclass
class CompletionConfig:
    """
    完备化配置。

    所有限制都有明确的数学/工程理由，不是魔法数。
    """
    # 最大规则数：由内存和实用性决定
    # 典型群表现完备化产生 O(|R| * |S|²) 规则
    max_rules: int = 10000

    # 最大迭代次数：每次迭代处理一个临界对
    # 完备化复杂度是 PSPACE-hard，需要限制
    max_iterations: int = 100000

    # 单步规约最大步数：防止潜在的无限循环
    max_reduction_steps: int = 10000

    # 是否启用规则间约化（通常有益但增加计算量）
    enable_interreduction: bool = True

    # 间约化频率：每添加多少规则后进行一次间约化
    interreduction_frequency: int = 50

    # 是否对输入词进行自由约化（群表现应启用）
    free_reduce: bool = True

    def __post_init__(self) -> None:
        if self.max_rules <= 0:
            raise CompletionError(f"max_rules must be positive, got {self.max_rules}")
        if self.max_iterations <= 0:
            raise CompletionError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.max_reduction_steps <= 0:
            raise CompletionError(f"max_reduction_steps must be positive, got {self.max_reduction_steps}")
        if self.interreduction_frequency <= 0:
            raise CompletionError(f"interreduction_frequency must be positive, got {self.interreduction_frequency}")


class KnuthBendixCompletion:
    """
    Knuth-Bendix 完备化算法。

    算法概述：
        输入：有限表现 ⟨S | R⟩ 和项序 >
        输出：完备重写系统（如果存在）

        1. 将每个关系 l = r 定向为规则 max(l,r) → min(l,r)
        2. 循环：
           a. 找出所有临界对
           b. 对每个不可合流的临界对 (u, v)：
              - 将 u, v 规约到正规形式 u', v'
              - 如果 u' ≠ v'，添加新规则 max(u',v') → min(u',v')
           c. 如果没有新规则添加，终止（成功）
        3. 可选：定期进行规则间约化

    终止性：
        - 如果项序是良基的且与拼接相容，则规约过程终止
        - 但完备化过程本身可能不终止（取决于群的结构）
        - 经典例子：Baumslag-Solitar 群 BS(1,2) 有无限完备化
    """

    def __init__(self, config: Optional[CompletionConfig] = None):
        """
        初始化完备化引擎。

        Args:
            config: 配置（使用默认值如果为 None）
        """
        self.config = config or CompletionConfig()

    def complete(
        self,
        alphabet: Alphabet,
        relations: List[Tuple[Word, Word]],
        order: Optional[TermOrder] = None
    ) -> CompletionResult:
        """
        执行 Knuth-Bendix 完备化。

        Args:
            alphabet: 字母表
            relations: 关系列表（等式对）
            order: 项序（默认使用 ShortLex）

        Returns:
            完备化结果
        """
        if order is None:
            order = ShortLexOrder(alphabet)

        # 初始化重写系统
        system = RewriteSystem(order)

        # 添加初始规则
        for lhs, rhs in relations:
            # 可选的自由约化
            if self.config.free_reduce:
                lhs = lhs.freely_reduce()
                rhs = rhs.freely_reduce()

            # 跳过平凡等式
            if lhs == rhs:
                continue

            system.add_equation(lhs, rhs)

        # 完备化主循环
        iterations = 0
        pairs_processed = 0
        rules_added = len(system)
        last_interreduction = 0

        while True:
            # 检查终止条件
            if iterations >= self.config.max_iterations:
                return CompletionResult(
                    status=CompletionStatus.MAX_ITERATIONS_EXCEEDED,
                    system=system,
                    iterations=iterations,
                    critical_pairs_processed=pairs_processed,
                    rules_added=rules_added,
                    message=f"Exceeded maximum iterations: {self.config.max_iterations}"
                )

            if len(system) >= self.config.max_rules:
                return CompletionResult(
                    status=CompletionStatus.MAX_RULES_EXCEEDED,
                    system=system,
                    iterations=iterations,
                    critical_pairs_processed=pairs_processed,
                    rules_added=rules_added,
                    message=f"Exceeded maximum rules: {self.config.max_rules}"
                )

            # 查找临界对
            finder = CriticalPairFinder(system)
            critical_pairs = finder.find_all_pairs()

            # 处理临界对
            new_rules_this_round = 0

            for pair in critical_pairs:
                iterations += 1
                pairs_processed += 1

                # 规约两端到正规形式
                try:
                    nf1 = system.normal_form(
                        pair.word1,
                        max_steps=self.config.max_reduction_steps
                    )
                    nf2 = system.normal_form(
                        pair.word2,
                        max_steps=self.config.max_reduction_steps
                    )
                except CompletionError:
                    return CompletionResult(
                        status=CompletionStatus.NON_TERMINATING,
                        system=system,
                        iterations=iterations,
                        critical_pairs_processed=pairs_processed,
                        rules_added=rules_added,
                        message="Reduction exceeded step limit, possible non-termination"
                    )

                # 如果可选，自由约化
                if self.config.free_reduce:
                    nf1 = nf1.freely_reduce()
                    nf2 = nf2.freely_reduce()

                # 如果不同，添加新规则
                if nf1 != nf2:
                    rule = system.add_equation(nf1, nf2)
                    if rule is not None:
                        new_rules_this_round += 1
                        rules_added += 1

                # 检查终止条件
                if iterations >= self.config.max_iterations:
                    break
                if len(system) >= self.config.max_rules:
                    break

            # 如果没有新规则，完备化成功
            if new_rules_this_round == 0:
                # 最终间约化
                if self.config.enable_interreduction:
                    system = system.interreduce()

                return CompletionResult(
                    status=CompletionStatus.SUCCESS,
                    system=system,
                    iterations=iterations,
                    critical_pairs_processed=pairs_processed,
                    rules_added=rules_added,
                    message="Completion successful"
                )

            # 定期间约化
            if self.config.enable_interreduction:
                if len(system) - last_interreduction >= self.config.interreduction_frequency:
                    system = system.interreduce()
                    last_interreduction = len(system)

        # 不应到达这里
        raise CompletionError("Unexpected exit from completion loop")


# =============================================================================
# 8) 群表现（Group Presentation）
# =============================================================================


class GroupPresentation:
    """
    群的有限表现 ⟨S | R⟩。

    数学定义：
        有限表现群 G = ⟨S | R⟩ 定义为：
        G = F(S) / ⟨⟨R⟩⟩

        其中 F(S) 是由 S 生成的自由群，⟨⟨R⟩⟩ 是 R 生成的正规闭包。

    用途：
        - 表示抽象群结构
        - 输入 Knuth-Bendix 完备化
        - 词问题求解（通过完备重写系统）
    """

    def __init__(
        self,
        generators: Sequence[str],
        relations: Sequence[Tuple[str, str]]
    ):
        """
        创建群表现。

        Args:
            generators: 生成元名称
            relations: 关系对（每对表示一个等式）

        Example:
            # Z × Z = ⟨a, b | ab = ba⟩
            presentation = GroupPresentation(
                generators=["a", "b"],
                relations=[("a b", "b a")]
            )
        """
        self.alphabet = Alphabet(list(generators), with_inverses=True)

        # 解析关系
        self._relations: List[Tuple[Word, Word]] = []
        for lhs_str, rhs_str in relations:
            lhs = Word.from_string(lhs_str, self.alphabet)
            rhs = Word.from_string(rhs_str, self.alphabet)
            self._relations.append((lhs, rhs))

        # 添加自由群关系：aa⁻¹ = ε, a⁻¹a = ε
        for gen in self.alphabet.generators:
            inv = gen.inverse()
            # aa⁻¹ = ε
            self._relations.append((
                Word.from_symbols([gen, inv], self.alphabet),
                Word.empty(self.alphabet)
            ))
            # a⁻¹a = ε
            self._relations.append((
                Word.from_symbols([inv, gen], self.alphabet),
                Word.empty(self.alphabet)
            ))

        self._completion_result: Optional[CompletionResult] = None

    @property
    def generators(self) -> Tuple[Symbol, ...]:
        """返回生成元。"""
        return self.alphabet.generators

    @property
    def relations(self) -> List[Tuple[Word, Word]]:
        """返回关系列表。"""
        return self._relations.copy()

    def complete(
        self,
        order: Optional[TermOrder] = None,
        config: Optional[CompletionConfig] = None
    ) -> CompletionResult:
        """
        运行 Knuth-Bendix 完备化。

        Args:
            order: 项序（默认 ShortLex）
            config: 配置（默认值）

        Returns:
            完备化结果
        """
        completer = KnuthBendixCompletion(config)
        self._completion_result = completer.complete(
            self.alphabet, self._relations, order
        )
        return self._completion_result

    def word(self, s: str) -> Word:
        """从字符串创建词。"""
        return Word.from_string(s, self.alphabet)

    def normal_form(self, w: Word) -> Word:
        """
        计算词的正规形式。

        需要先调用 complete()。
        """
        if self._completion_result is None:
            raise PresentationError("Must call complete() before normal_form()")
        if self._completion_result.status != CompletionStatus.SUCCESS:
            raise PresentationError(
                f"Completion was not successful: {self._completion_result.message}"
            )
        return self._completion_result.system.normal_form(w)

    def are_equal(self, w1: Word, w2: Word) -> bool:
        """
        检查两个词是否表示相同的群元素。

        需要先调用 complete()。
        """
        nf1 = self.normal_form(w1)
        nf2 = self.normal_form(w2)
        return nf1 == nf2

    def __repr__(self) -> str:
        gens = ", ".join(g.name for g in self.alphabet.generators)
        rels = "; ".join(f"{l} = {r}" for l, r in self._relations[:3])
        if len(self._relations) > 3:
            rels += f"; ... ({len(self._relations)} total)"
        return f"⟨{gens} | {rels}⟩"


# =============================================================================
# 9) 便捷工厂函数
# =============================================================================


def cyclic_group(n: int) -> GroupPresentation:
    """
    创建 n 阶循环群 Zₙ = ⟨a | aⁿ = 1⟩。

    Args:
        n: 群的阶（必须 > 0）
    """
    if n <= 0:
        raise PresentationError(f"Order must be positive, got {n}")

    # aⁿ = ε
    relation = (" ".join(["a"] * n), "")
    return GroupPresentation(generators=["a"], relations=[relation])


def dihedral_group(n: int) -> GroupPresentation:
    """
    创建 n 阶二面体群 Dₙ = ⟨r, s | rⁿ = 1, s² = 1, srs = r⁻¹⟩。

    这是正 n 边形的对称群，阶为 2n。

    Args:
        n: 旋转的阶（必须 ≥ 2）
    """
    if n < 2:
        raise PresentationError(f"Dihedral group requires n >= 2, got {n}")

    relations = [
        (" ".join(["r"] * n), ""),  # rⁿ = ε
        ("s s", ""),                 # s² = ε
        ("s r s", "r^"),             # srs = r⁻¹
    ]
    return GroupPresentation(generators=["r", "s"], relations=relations)


def symmetric_group(n: int) -> GroupPresentation:
    """
    创建 n 阶对称群 Sₙ（Coxeter 表现）。

    生成元：s₁, s₂, ..., sₙ₋₁（相邻对换）
    关系：
        - sᵢ² = 1
        - sᵢsⱼ = sⱼsᵢ 当 |i-j| > 1
        - sᵢsⱼsᵢ = sⱼsᵢsⱼ 当 |i-j| = 1

    Args:
        n: 对称群的阶（必须 ≥ 2）
    """
    if n < 2:
        raise PresentationError(f"Symmetric group requires n >= 2, got {n}")

    generators = [f"s{i}" for i in range(1, n)]
    relations: List[Tuple[str, str]] = []

    for i in range(1, n):
        # sᵢ² = 1
        relations.append((f"s{i} s{i}", ""))

    for i in range(1, n):
        for j in range(i + 2, n):
            # sᵢsⱼ = sⱼsᵢ 当 |i-j| > 1
            relations.append((f"s{i} s{j}", f"s{j} s{i}"))

    for i in range(1, n - 1):
        # sᵢsᵢ₊₁sᵢ = sᵢ₊₁sᵢsᵢ₊₁
        j = i + 1
        relations.append((f"s{i} s{j} s{i}", f"s{j} s{i} s{j}"))

    return GroupPresentation(generators=generators, relations=relations)


def free_abelian_group(rank: int) -> GroupPresentation:
    """
    创建秩为 rank 的自由阿贝尔群 Zʳ。

    生成元：a₁, ..., aᵣ
    关系：aᵢaⱼ = aⱼaᵢ 对所有 i < j
    """
    if rank <= 0:
        raise PresentationError(f"Rank must be positive, got {rank}")

    generators = [f"a{i}" for i in range(1, rank + 1)]
    relations: List[Tuple[str, str]] = []

    for i in range(1, rank + 1):
        for j in range(i + 1, rank + 1):
            relations.append((f"a{i} a{j}", f"a{j} a{i}"))

    return GroupPresentation(generators=generators, relations=relations)


# =============================================================================
# 模块级文档
# =============================================================================


__all__ = [
    # 异常
    "PresentationError",
    "AlphabetError",
    "WordError",
    "TermOrderError",
    "RewriteRuleError",
    "RewriteSystemError",
    "CriticalPairError",
    "CompletionError",
    "NonTerminationError",

    # 基础结构
    "Symbol",
    "Alphabet",
    "Word",

    # 项序
    "TermOrder",
    "ShortLexOrder",
    "WeightedLexOrder",
    "RecursivePathOrder",

    # 重写系统
    "RewriteRule",
    "RewriteSystem",

    # 临界对
    "CriticalPair",
    "CriticalPairFinder",

    # 完备化
    "CompletionStatus",
    "CompletionResult",
    "CompletionConfig",
    "KnuthBendixCompletion",

    # 群表现
    "GroupPresentation",

    # 工厂函数
    "cyclic_group",
    "dihedral_group",
    "symmetric_group",
    "free_abelian_group",
]
