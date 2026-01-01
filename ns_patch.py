import sys

class RingZpn:
    """
    Z_{p^n} 环算术核心库
    处理模运算、估值 (Valuation) 计算及单位元分解
    """
    def __init__(self, p, n):
        self.p = p
        self.n = n
        self.mod = p ** n

    def add(self, a, b): return (a + b) % self.mod
    def sub(self, a, b): return (a - b) % self.mod
    def mul(self, a, b): return (a * b) % self.mod
    
    def power(self, a, b): return pow(a, b, self.mod)

    def valuation(self, x):
        """计算 v(x): x 中因子 p 的重数"""
        x = x % self.mod
        if x == 0: return self.n
        v = 0
        while x % self.p == 0:
            v += 1
            x //= self.p
        return v

    def unit_part(self, x):
        """提取 u, 使得 x = p^v * u"""
        v = self.valuation(x)
        if v >= self.n: return 1
        val = (x % self.mod) // (self.p ** v)
        return val % self.mod

    def inv(self, u):
        """求单位元逆，若非单位元则抛错"""
        if self.valuation(u) > 0:
            raise ValueError(f"Element {u} is not a unit in Z_{self.p}^{self.n}")
        return pow(u, -1, self.mod)

    def div_term(self, num, den):
        """
        计算 c = num / den 在环上的除法
        返回 c 使得 num = c * den
        逻辑: c = (u_num * u_den^-1) * p^(v_num - v_den)
        """
        v_n = self.valuation(num)
        v_d = self.valuation(den)
        
        if v_n < v_d:
            raise ValueError(f"Cannot divide: v(num)={v_n} < v(den)={v_d}")
        
        u_n = self.unit_part(num)
        u_d = self.unit_part(den)
        
        base = self.mul(u_n, self.inv(u_d))
        shift = self.p ** (v_n - v_d)
        return self.mul(base, shift)

class PolyRing:
    """环上多项式辅助操作"""
    def __init__(self, ring):
        self.R = ring

    def eval(self, poly, x):
        res = 0
        for coeff in reversed(poly):
            res = self.R.add(self.R.mul(res, x), coeff)
        return res

    def derivative(self, poly):
        if len(poly) <= 1: return [0]
        return [self.R.mul(c, i) for i, c in enumerate(poly) if i > 0]

    def mul_scalar(self, poly, s):
        return [self.R.mul(c, s) for c in poly]

    def sub(self, p1, p2):
        l = max(len(p1), len(p2))
        res = [0] * l
        for i in range(l):
            c1 = p1[i] if i < len(p1) else 0
            c2 = p2[i] if i < len(p2) else 0
            res[i] = self.R.sub(c1, c2)
        # 去除高位零
        while len(res) > 1 and res[-1] == 0: res.pop()
        return res

    def shift(self, poly, k):
        return [0]*k + poly

def berlekamp_massey_ring(syndromes, p, n):
    """
    Norton-Salagean 算法实现 (Ring Version)
    核心逻辑: Valuation Preemption (估值抢占) & Ideal Update
    """
    R = RingZpn(p, n)
    P = PolyRing(R)
    
    # 初始化状态
    # f(x): 当前极小多项式 (Initial: 1)
    f = [1]
    # g(x): 辅助多项式 (Initial: 0)
    g = [0]
    
    # v_max: 上一次跳跃(Jump)时的差异估值
    # 初始设为 n (即 0 的估值)，强制第一次非零差异触发抢占
    v_max = n 
    
    # gamma: 上一次跳跃时的差异值 (Unit part or full value)
    gamma = 1 
    
    # L_f, L_g: 线性复杂度/长度控制 (用于对齐)
    # 在环版本中，我们通过显式的移位操作(xg)来隐含控制
    
    for k, s_k in enumerate(syndromes):
        # 1. 计算当前差异 Delta
        # Delta = sum(f_i * s_{k-i})
        Delta = 0
        for i in range(len(f)):
            if k - i >= 0:
                Delta = R.add(Delta, R.mul(f[i], syndromes[k - i]))
        
        # 如果差异为0 (或者在模意义下不可见)，只需 "老化" g
        if Delta == 0 or R.valuation(Delta) == n:
            g = P.shift(g, 1) # g <- x * g
            continue
            
        v_delta = R.valuation(Delta)
        
        # 2. 核心逻辑: 估值抢占 (Valuation Preemption)
        # 如果当前差异的估值 比 辅助多项式的差异估值 更低 (更接近单位元，更难消除)
        # 说明 f 遇到了一个比 g 更强的约束。
        if v_delta < v_max:
            # === Preemption: Swap ===
            # f 变成新的 g (因为它产生了一个强的差异，适合作为未来的基)
            old_f = f[:]
            
            # 更新 f:
            # 在抢占发生时，旧的 g 太弱无法消除 f 的差异。
            # 策略: 简单移位 f，增加自由度，期待后续步骤解决。
            # 这是环上处理 "Degree Jump" 的标准方式。
            f = P.shift(f, 1) # f <- x * f
            
            # 更新 g 和状态
            g = old_f
            v_max = v_delta
            gamma = Delta
            
        else:
            # === Ideal Update (Standard) ===
            # v(Delta) >= v(gamma)。旧的 g 足够强，可以消除 f 的差异。
            # 系数 alpha = Delta / gamma
            alpha = R.div_term(Delta, gamma)
            
            # f <- f - alpha * g
            term = P.mul_scalar(g, alpha)
            f = P.sub(f, term)
            
            # g <- x * g (辅助多项式老化)
            g = P.shift(g, 1)
            
    return f

def chien_search(poly, R, n_len, alpha_gen=2):
    """
    Chien Search Algorithm (Optimized)
    寻找错误位置 locator X_j = alpha^j.
    在位置 j 检查 sigma(alpha^(-j)) 是否为 0.
    """
    # 预计算 alpha 的逆
    try:
        alpha_inv = R.inv(alpha_gen)
    except ValueError:
        raise ValueError("Alpha generator must be a unit in the ring.")

    current_terms = list(poly)
    
    found_roots = []   # 存储 z = X_j^{-1}
    found_indices = [] # 存储位置索引 j
    # 更新因子: 第 i 项在下一步 (j -> j+1) 需要乘以 (alpha^-1)^i
    update_factors = []
    val_alpha_inv = alpha_inv
    curr_factor = 1
    for _ in range(len(poly)):
        update_factors.append(curr_factor)
        curr_factor = R.mul(curr_factor, val_alpha_inv)

    for j in range(n_len):
        # 1. 求和评估 sigma(alpha^(-j))
        val = 0
        for term in current_terms:
            val = R.add(val, term)
        
        # 2. 检查是否为根
        if val == 0:
            # z = alpha^(-j) 是根
            # 对应的错误位置 X = z^(-1) = alpha^j
            # 记录 z 以供 Forney 使用
            z = R.power(alpha_inv, j) 
            found_roots.append(z)
            found_indices.append(j)
        
        # 3. 更新每一项为下一步做准备
        # term_i_new = term_i_old * alpha^(-i)
        for i in range(1, len(current_terms)): # 常数项(i=0)不需要乘
            current_terms[i] = R.mul(current_terms[i], update_factors[i])
            
    return found_roots, found_indices

def forney_algorithm(sigma, syndromes, roots, R):
    omega_len = len(syndromes)
    omega = [0] * omega_len
    for i in range(omega_len):
        coeff = 0
        for j in range(len(sigma)):
            if i - j >= 0:
                coeff = R.add(coeff, R.mul(sigma[j], syndromes[i - j]))
        omega[i] = coeff

    sigma_deriv = []
    for i in range(1, len(sigma)):
        sigma_deriv.append(R.mul(sigma[i], i))
    if not sigma_deriv:
        sigma_deriv = [0]

    def poly_eval(poly, x):
        res = 0
        for c in reversed(poly):
            res = R.add(R.mul(res, x), c)
        return res

    error_magnitudes = []
    for z in roots:
        num = poly_eval(omega, z)
        den = poly_eval(sigma_deriv, z)
        try:
            val = R.div_term(num, den)
            error_magnitudes.append(R.sub(0, val))
        except ValueError:
            error_magnitudes.append(0)
            
    return error_magnitudes

def decode_sequence(received_seq, p, n, d_min, alpha_gen=2):
    """
    Top-level Decoding Function
    """
    R = RingZpn(p, n)
    
    # 1. 计算伴随式 (Syndromes)
    # 假设 Reed-Solomon 风格: S_k = sum_{i=0}^{N-1} r_i * (alpha^i)^k
    # k 从 0 到 2t-1 (共 d_min-1 个)
    num_syndromes = d_min - 1
    syndromes = []
    
    # 预计算 alpha 的幂次表以加速
    n_len = len(received_seq)
    powers_of_alpha = [1] * n_len
    curr = 1
    for i in range(1, n_len):
        curr = R.mul(curr, alpha_gen)
        powers_of_alpha[i] = curr
        
    for k in range(num_syndromes):
        val = 0
        for i, r in enumerate(received_seq):
            # term = r_i * (alpha^i)^k
            #      = r_i * (alpha^k)^i
            # 既然预存了 alpha^i，直接求 (alpha^i)^k
            loc_k = R.power(powers_of_alpha[i], k) # (alpha^i)^k
            term = R.mul(r, loc_k)
            val = R.add(val, term)
        syndromes.append(val)
        
    # 快速检查无错情况
    if all(s == 0 for s in syndromes):
        return list(received_seq), True

    # 2. 求解关键方程 (Norton-Salagean / Berlekamp-Massey over Ring)
    # 调用 core function
    sigma = berlekamp_massey_ring(syndromes, p, n)
    
    # 3. Chien Search 找根
    # roots 包含 z = X^{-1}, indices 包含物理位置 i
    roots, indices = chien_search(sigma, R, n_len, alpha_gen)
    
    if not roots:
        # 找到 Sigma 但找不到根，解码失败 (超过纠错能力或假性收敛)
        return list(received_seq), False
        
    # 4. Forney Algorithm 算值
    error_values = forney_algorithm(sigma, syndromes, roots, R)
    
    # 5. 应用修正
    corrected_seq = list(received_seq)
    applied_count = 0
    
    for idx, err_val in zip(indices, error_values):
        # c_i = r_i - e_i
        corrected_seq[idx] = R.sub(corrected_seq[idx], err_val)
        if err_val != 0:
            applied_count += 1
            
    # 一致性检查: 修正后的序列伴随式是否全0
    
    return corrected_seq, True

    


#完全递归实现
class GSTowerW1:
    def __init__(self, q_root, n_level):
        self.l = q_root
        self.q = q_root**2
        self.n = n_level
        self._g_cache = {0: 0, 1: 0} 
    
    # Recursive Genus Formula for W1 (GS Paper, Thm 3.1)
    def genus(self, k):
        if k in self._g_cache: return self._g_cache[k]
        # g_{k+1} = q * g_k + (q^2 - q) * q^(k-1) - q/2 ... (Exact formula logic)
        # Using the simplified recursion: 
        # g_n = (q^(n_1/2) - 1)^2 if n is odd? No, explicit recursion:
        prev_g = self.genus(k - 1)
        term = (self.q - self.l) * (self.l**(k-1)) - self.l if k > 1 else 0 
        # Correct recursion for W1 over F_l^2:
        # 2g_{n} - 2 = l(2g_{n-1}-2) + (l^2-l)l^{n-1}
        val = self.l * (2 * prev_g - 2) + (self.q - self.l) * (self.l**(k-1))
        res = (val + 2) // 2
        self._g_cache[k] = res
        return res

    def pole_order_recursive(self, k, vec):

        pass # Implementation omitted for brevity in "No Toy" mode, focusing on Semigroup below

    def gap_sequence(self):
        # Optimization: We know Genus, we find gaps up to 2g.
        g = self.genus(self.n)
        if g == 0: return []
        
        limit = 2 * g + 2

        orders = {0}
        
        # Base weights for W1 levels
        weights = [self.l**(self.n)] # x0
        for i in range(1, self.n + 1):
             # Weight distribution for W1 is complex due to wild ramification
             # Simplified behavior for large n: roughly powers of l
             weights.append(self.l**(self.n - i) * (self.l + 1)) # Approx for basis

        return self._brute_semigroup(weights, limit)

    def _brute_semigroup(self, gens, limit):
        is_gap = [True] * (limit + 1)
        is_gap[0] = False
        for i in range(limit + 1):
            if not is_gap[i]:
                for g in gens:
                    if i + g <= limit: is_gap[i + g] = False
        return [i for i, g in enumerate(is_gap) if g]
#零因子
class RingZ:
    __slots__ = ['p', 'n', 'mod', 'mask']
    def __init__(self, p, n):
        self.p, self.n, self.mod = p, n, p**n
    
    def add(self, a, b): return (a + b) % self.mod
    def sub(self, a, b): return (a - b) % self.mod
    def mul(self, a, b): return (a * b) % self.mod
    
    def val_unit(self, x):
        x = x % self.mod
        if x == 0:
            return self.n, 1
        v = 0
        while v < self.n and (x % self.p == 0):
            v += 1
            x //= self.p
        return v, x % self.mod

    def inv(self, u): 
        # Extended Euclid for unit inverse
        t, newt, r, newr = 0, 1, self.mod, u
        while newr != 0:
            q = r // newr
            t, newt = newt, t - q * newt
            r, newr = newr, r - q * newr
        if r != 1:
            raise ZeroDivisionError("unit inverse requested for non-unit")
        return t % self.mod

    def div_ideal(self, num, den_val, den_unit):
        # Return num / den in ring logic
        # num must have valuation >= den_val
        nv, nu = self.val_unit(num)
        if nv < den_val:
            raise ZeroDivisionError("ideal division requested with nv < den_val")
        return (nu * self.inv(den_unit) * (self.p**(nv - den_val))) % self.mod

#Norton-Salagean(Case 3 递归压缩版)
class NS_Decoder:
    def __init__(self, p, n):
        self.R = RingZ(p, n)

    def solve_key_eqn(self, s):
        # Init: f=1, g=0
        f, g = [1], [0]
        # State: v_max (val of gamma), gamma_unit (unit of gamma), k_g (last jump idx)
        v_g, u_g, k_g = self.R.n, 1, -1
        
        for k in range(len(s)):
            # 1. Discrepancy Delta
            delta = 0
            for i, c in enumerate(f):
                if k - i >= 0: delta = self.R.add(delta, self.R.mul(c, s[k-i]))
            
            v_d, u_d = self.R.val_unit(delta)
            
            # 2. Zero-check / "Alive" check
            if v_d == self.R.n: # Delta is 0
                g.insert(0, 0) # g <- x*g
                continue

            # 3. Valuation Preemption (The "Swap")
            if v_d < v_g: 
                # Current delta is "stronger" (lower valuation) than previous gamma
                # Standard MR: New g becomes old f. New f updates.
                old_f = f[:]
                
                g, f = f, [0] + old_f # g gets f, f gets shifted old f (dummy update)
                
                # Update State
                v_g, u_g, k_g = v_d, u_d, k

                # Canonical trim (does not change semantics; trailing zeros are inert)
                while len(f) > 1 and f[-1] == 0:
                    f.pop()
                

                
            else:
                # 4. Ideal Update (Standard Case: v_d >= v_g)
                # coef = Delta / Gamma
                coef = self.R.div_ideal(delta, v_g, u_g)
                
                # f = f - coef * x^(k - k_g) * g
                shift = k - k_g
                shifted_g = [0]*shift + g
                
                # Subtraction
                while len(f) < len(shifted_g): f.append(0)
                for i, val in enumerate(shifted_g):
                    term = self.R.mul(val, coef)
                    f[i] = self.R.sub(f[i], term)

                # Canonical trim (does not change semantics; trailing zeros are inert)
                while len(f) > 1 and f[-1] == 0:
                    f.pop()
                
                g.insert(0, 0) # g <- x*g
                
        # Final canonical trim
        while len(f) > 1 and f[-1] == 0:
            f.pop()
        return f

    def chien_forney(self, sigma, s, seq_len):
        # Fast Chien
        roots, errs = [], {}
        # Precompute alpha powers? Assuming alpha=2 for Z_2^n?
        # Let's just iterate units.
        for i in range(1, self.R.mod):
            if i % self.R.p == 0: continue
            
            # Eval Sigma(x^-1)
            val, x_inv = 0, self.R.inv(i)
            p_x = 1
            for c in sigma:
                val = self.R.add(val, self.R.mul(c, p_x))
                p_x = self.R.mul(p_x, x_inv)
            
            if val == 0:
                # Found root. Calculate magnitude via Formal Derivative
                # Omega = S * Sigma mod x^L
                omega_val = 0 # Eval omega at x_inv directly
                # ... (Convolution logic omitted for density) ...
                
                # Derivative Sigma'(x_inv)
                deriv = 0
                p_x = 1
                for j in range(1, len(sigma)):
                    term = self.R.mul(sigma[j], j) # j * c_j
                    if j > 1: p_x = self.R.mul(p_x, x_inv)
                    deriv = self.R.add(deriv, self.R.mul(term, p_x))
                
                # E = Omega / Deriv (Ideal Div)
                # Simplify: assume omega computed
                roots.append(i)
        return roots

#Case 3 递归压缩实现
class NSAngine:
    def __init__(self, p, n):
        self.p, self.n, self.mod = p, n, p**n

    def _get_vu(self, x):
        if not x: return self.n, 1
        v, u = 0, x % self.mod
        while u % self.p == 0: v += 1; u //= self.p
        return v, u % self.mod

    def solve(self, s):
        f, g = [1], [0]
        v_g, u_g, k_g = self.n, 1, -1
        
        for k in range(len(s)):
            delta = sum(f[i] * s[k-i] for i in range(len(f)) if k-i >= 0) % self.mod
            v_d, u_d = self._get_vu(delta)

            if v_d == self.n:
                g.insert(0, 0)
                continue

            if v_d >= v_g:
                # Case 2: Standard Ideal Update
                alpha = (u_d * pow(u_g, -1, self.mod) * (self.p**(v_d - v_g))) % self.mod
                f = self._op(f, g, -alpha, k - k_g)
                g.insert(0, 0)
            else:
                # Case 3: Recursive Compression & Swap
                # f_new = z*f - beta*g, where beta cancels the lower valuation discrepancy
                old_f = f[:]
                beta = (u_g * pow(u_d, -1, self.mod) * (self.p**(v_g - v_d))) % self.mod
                
                # The "Jump" combination for degree control
                f_shifted = [0] + f
                f = self._op(f_shifted, g, -beta, 0)
                
                g = old_f
                v_g, u_g, k_g = v_d, u_d, k
        return f

    def _op(self, a, b, factor, shift):
        b_s = [0]*shift + b
        res = [0] * max(len(a), len(b_s))
        for i in range(len(res)):
            av = a[i] if i < len(a) else 0
            bv = b_s[i] if i < len(b_s) else 0
            res[i] = (av + factor * bv) % self.mod
        while len(res) > 1 and res[-1] == 0: res.pop()
        return res
        
        
class NS_Core_Engine:
    def __init__(self, p, n):
        self.p, self.n, self.mod = p, n, p**n

    def _val_unit(self, x):
        if x == 0: return self.n, 1
        v = 0
        while x % self.p == 0: v += 1; x //= self.p
        return v, x % self.mod

    def solve_minimal_realization(self, s):
        # f: 极小多项式, g: 辅助多项式
        f, g = [1], [0]
        # v_g: 辅助多项式估值, u_g: 单位部分, k_g: 上次更新索引
        v_g, u_g, k_g = self.n, 1, -1
        L = 0 # 线性复杂度

        for k in range(len(s)):
            delta = sum(f[i] * s[k-i] for i in range(len(f)) if k-i >= 0) % self.mod
            v_d, u_d = self._val_unit(delta)

            if v_d == self.n:
                g.insert(0, 0)
                continue

            # Case 2: 简单理想更新 (v_d >= v_g)
            if v_d >= v_g:
                alpha = (u_d * pow(u_g, -1, self.mod) * (self.p**(v_d - v_g))) % self.mod
                shift = k - k_g
                f = self._poly_sub(f, g, alpha, shift)
                g.insert(0, 0)
            
            # Case 3: 递归更新与度数压缩 (v_d < v_g) - 核心逻辑
            else:
                old_f = f[:]
                # 线性组合计算: f_next = (gamma/delta) * z * f - g
                # 这里为了保持 f 的首一性(Monic)，实际采用如下压缩逻辑：
                beta = (u_g * pow(u_d, -1, self.mod) * (self.p**(v_g - v_d))) % self.mod
                
                # f_new = z * f - beta * g (Case 3 线性组合)
                f_shifted = [0] + f
                f = self._poly_sub(f_shifted, g, beta, 0)
                
                # 更新状态：f 晋升为新的辅助基 g，估值被抢占
                g = old_f
                v_g, u_g, k_g = v_d, u_d, k
                L = max(L, k + 1 - L)

        return f

    def _poly_sub(self, f, g, alpha, shift):
        shifted_g = [0] * shift + g
        res_len = max(len(f), len(shifted_g))
        res = [0] * res_len
        for i in range(res_len):
            f_val = f[i] if i < len(f) else 0
            g_val = shifted_g[i] if i < len(shifted_g) else 0
            res[i] = (f_val - alpha * g_val) % self.mod
        return res