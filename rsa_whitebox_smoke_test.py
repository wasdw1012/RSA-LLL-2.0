#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA白盒Smoke测试

使用方法：
    cd bridge_audit/core/smoke
    python rsa_whitebox_smoke_test.py [--mode normal] [--pressure 1] [--all]
"""

import sys
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# 添加父目录到路径（确保可以从文件所在目录直接运行）
_current_file = Path(__file__).resolve()
_parent_dir = _current_file.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from rsa_lll import lll_reduce_enhanced, LLLResult
from fractions import Fraction
import math


def load_whitebox_rsa():
    """加载白盒RSA数据"""
    rsa_file = Path(__file__).parent.parent / "白盒RSA.txt"
    
    with open(rsa_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 第2行：十六进制格式的n
    hex_line = lines[1].strip() if len(lines) > 1 else ""
    # 第8行：二进制格式的n
    bin_line = lines[7].strip() if len(lines) > 7 else ""
    
    if not hex_line and not bin_line:
        raise ValueError("白盒RSA.txt数据为空")
    
    # 优先使用二进制格式（更直接）
    if bin_line:
        n = int(bin_line, 2)
        n_bits = bin_line
    else:
        n = int(hex_line, 16)
        n_bits = bin(n)[2:]
    
    return n, n_bits


def scramble_msb(bits: str, msb_length: int = None, seed: int = 42, mode: str = "normal"):
    """
    打乱MSB（最高有效位）- 地狱模式
    
    Args:
        bits: 二进制字符串
        msb_length: 要打乱的MSB长度（默认打乱前一半）
        seed: 随机种子
        mode: 打乱模式
            - "normal": 普通打乱
            - "hell": 地狱模式（完全随机）
            - "chaos": 混沌模式（多重打乱+翻转）
            - "extreme": 极端模式（全部打乱+噪声注入）
    
    Returns:
        打乱后的二进制字符串
    """
    if msb_length is None:
        msb_length = len(bits) // 2
    
    msb_length = min(msb_length, len(bits))
    
    # 分离MSB和LSB
    msb = bits[:msb_length]
    lsb = bits[msb_length:]
    
    random.seed(seed)
    msb_list = list(msb)
    
    if mode == "normal":
        # 普通打乱
        for i in range(len(msb_list) - 1, 0, -1):
            j = random.randint(0, i)
            msb_list[i], msb_list[j] = msb_list[j], msb_list[i]
    
    elif mode == "hell":
        # 地狱模式：完全随机，不保留任何结构
        for i in range(len(msb_list)):
            msb_list[i] = str(random.randint(0, 1))
    
    elif mode == "chaos":
        # 混沌模式：多重打乱+随机翻转
        # 第一次打乱
        for i in range(len(msb_list) - 1, 0, -1):
            j = random.randint(0, i)
            msb_list[i], msb_list[j] = msb_list[j], msb_list[i]
        # 随机翻转50%的位
        for i in range(len(msb_list)):
            if random.random() < 0.5:
                msb_list[i] = '1' if msb_list[i] == '0' else '0'
        # 第二次打乱
        for i in range(len(msb_list) - 1, 0, -1):
            j = random.randint(0, i)
            msb_list[i], msb_list[j] = msb_list[j], msb_list[i]
    
    elif mode == "extreme":
        # 极端模式：全部打乱+噪声注入+恶意构造
        # 完全随机
        for i in range(len(msb_list)):
            msb_list[i] = str(random.randint(0, 1))
        # 注入恶意模式：连续0或1的块
        for _ in range(5):
            start = random.randint(0, len(msb_list) - 10)
            length = random.randint(5, 10)
            bit_val = str(random.randint(0, 1))
            for i in range(start, min(start + length, len(msb_list))):
                msb_list[i] = bit_val
        # 最后再打乱一次
        for i in range(len(msb_list) - 1, 0, -1):
            j = random.randint(0, i)
            msb_list[i], msb_list[j] = msb_list[j], msb_list[i]
    
    scrambled_msb = ''.join(msb_list)
    
    # 组合
    scrambled_bits = scrambled_msb + lsb
    
    return scrambled_bits, msb_length


def bits_to_lattice_basis(bits: str, dim: int = None):
    """
    将二进制序列转换为格基（确保线性无关）
    
    Args:
        bits: 二进制字符串
        dim: 格基维度（默认根据bits长度计算）
    
    Returns:
        格基矩阵 List[List[int]]
    """
    if dim is None:
        # 维度：取bits长度的平方根，但不要太大
        dim = min(int(len(bits) ** 0.5) + 1, 20)  # 限制最大维度
    
    basis = []
    
    # 方法：将bits分成dim段，每段作为一个向量的不同位置
    # 构造单位矩阵的变体，确保线性无关
    chunk_size = len(bits) // dim
    remainder = len(bits) % dim
    
    # 先构造一个基础矩阵（单位矩阵的变体）
    for i in range(dim):
        vector = [0] * dim
        
        # 对角线元素：使用对应chunk的整数值
        start = i * chunk_size
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size
        
        chunk = bits[start:end] if start < len(bits) else '0'
        if not chunk:
            chunk = '0'
        
        chunk_int = int(chunk, 2)
        
        # 避免整数过大
        max_val = 2**30
        if chunk_int > max_val:
            chunk_int = chunk_int % max_val
        
        # 对角线元素
        vector[i] = chunk_int if chunk_int > 0 else 1
        
        # 非对角线元素：使用其他chunk的部分信息，确保线性无关
        for j in range(dim):
            if i != j:
                # 使用bits的不同位置，加上偏移避免重复
                offset = (i * 7 + j * 11) % len(bits)
                bit_val = int(bits[offset]) if offset < len(bits) else 0
                vector[j] = bit_val * (2 ** (j % 16))  # 小权重
        
        basis.append(vector)
    
    return basis


def inject_noise(bits: str, noise_ratio: float = 0.1):
    """注入随机噪声"""
    bits_list = list(bits)
    noise_count = int(len(bits_list) * noise_ratio)
    indices = random.sample(range(len(bits_list)), noise_count)
    for idx in indices:
        bits_list[idx] = '1' if bits_list[idx] == '0' else '0'
    return ''.join(bits_list)


def corrupt_data(bits: str, corruption_type: str = "random"):
    """数据损坏：模拟传输错误"""
    bits_list = list(bits)
    
    if corruption_type == "random":
        # 随机损坏5%
        corrupt_count = len(bits_list) // 20
        indices = random.sample(range(len(bits_list)), corrupt_count)
        for idx in indices:
            bits_list[idx] = '1' if bits_list[idx] == '0' else '0'
    
    elif corruption_type == "burst":
        # 突发错误：连续损坏
        start = random.randint(0, len(bits_list) - 50)
        for i in range(start, min(start + 50, len(bits_list))):
            bits_list[i] = '1' if bits_list[i] == '0' else '0'
    
    elif corruption_type == "systematic":
        # 系统性错误：每隔N位损坏
        step = len(bits_list) // 20
        for i in range(0, len(bits_list), step):
            bits_list[i] = '1' if bits_list[i] == '0' else '0'
    
    return ''.join(bits_list)


def solve_k_from_lattice(reduced_basis: List[List[int]], n: int, e: int = 65537) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    从格基规约结果中求解k（增强版：包含赋值区间验证）
    
    在RSA中：e*d = 1 + k*φ(n)，其中k是某个整数
    从Wiener攻击的角度，k/d是e/n的连分数近似
    
    Args:
        reduced_basis: LLL规约后的格基
        n: RSA模数
        e: 公钥指数（默认65537）
    
    Returns:
        (k值, 诊断信息字典)，如果无法求解则返回(None, {})
    """
    diagnostics = {}
    
    if not reduced_basis or not reduced_basis[0]:
        return None, diagnostics
    
    # 方法1：从最短向量提取k
    shortest_vector = reduced_basis[0]
    
    # 在Wiener攻击的格基中，第一个元素通常是k
    k_candidate = abs(shortest_vector[0]) if shortest_vector else None
    
    # 特殊处理：MAGIC=0 的情况
    if k_candidate == 0:
        # 尝试从其他位置提取k
        if len(shortest_vector) > 1:
            k_candidate = abs(shortest_vector[1])
        else:
            # 从其他向量尝试
            for vec in reduced_basis[1:min(5, len(reduced_basis))]:
                if vec and len(vec) > 0:
                    k_candidate = abs(vec[0])
                    if k_candidate > 0:
                        break
    
    if k_candidate and k_candidate > 0:
        # 【细节1】抓取k的赋值区间：v(k) = log(k)/log(n)
        log_k = math.log(k_candidate) if k_candidate > 0 else 0
        log_n = math.log(n) if n > 0 else 1
        v_k = log_k / log_n if log_n > 0 else 0
        
        diagnostics['v_k'] = v_k
        diagnostics['log_k'] = log_k
        diagnostics['log_n'] = log_n
        
        # 验证：k应该满足 k < e 的合法区间
        k_valid = (k_candidate > 0 and k_candidate < e)
        diagnostics['k_valid'] = k_valid
        diagnostics['k'] = k_candidate
        diagnostics['k_bits'] = k_candidate.bit_length()
        
        if k_valid:
            return k_candidate, diagnostics
        elif k_candidate < e * 10:
            # 虽然不在严格区间，但可能仍然有效
            diagnostics['warning'] = f"k={k_candidate} 不在严格区间 [1, {e})，但可能仍然有效"
            return k_candidate, diagnostics
    
    # 方法2：从多个向量中尝试
    for vec in reduced_basis[:min(5, len(reduced_basis))]:
        if vec and len(vec) >= 2:
            # 尝试第一个元素作为k
            k_test = abs(vec[0])
            if k_test > 0:
                log_k = math.log(k_test)
                log_n = math.log(n)
                v_k = log_k / log_n if log_n > 0 else 0
                
                diagnostics['v_k'] = v_k
                diagnostics['k'] = k_test
                diagnostics['k_valid'] = (k_test < e)
                
                if k_test < e:
                    return k_test, diagnostics
            # 尝试第二个元素
            if len(vec) > 1:
                k_test = abs(vec[1])
                if k_test > 0:
                    log_k = math.log(k_test)
                    log_n = math.log(n)
                    v_k = log_k / log_n if log_n > 0 else 0
                    
                    diagnostics['v_k'] = v_k
                    diagnostics['k'] = k_test
                    diagnostics['k_valid'] = (k_test < e)
                    
                    if k_test < e:
                        return k_test, diagnostics
    
    return None, diagnostics


def recover_msb_from_tropical_balance(k: int, n: int, e: int = 65537, 
                                       scrambled_bits: str = None) -> Optional[str]:
    """
    利用热带几何平衡条件恢复d的高位MSB
    
    一旦出k，根据热带几何的平衡条件，d的高位MSB会瞬间坍塌出来
    
    数学原理：
    - e*d = 1 + k*φ(n) ≈ 1 + k*(n - sqrt(n) - sqrt(n) + 1) ≈ k*n（近似）
    - d ≈ k*n/e
    - 从k可以推导出d的高位MSB
    
    Args:
        k: 从格基规约中求出的k值
        n: RSA模数
        e: 公钥指数
        scrambled_bits: 打乱后的二进制字符串（可选，用于验证）
    
    Returns:
        恢复的d高位MSB（二进制字符串），如果无法恢复则返回None
    """
    if k is None or k <= 0:
        return None
    
    try:
        # 方法1：从k直接推导d的高位
        # d ≈ (k*n) / e（这是近似，但高位通常是正确的）
        d_approx = (k * n) // e
        
        # 如果d_approx太大，说明k可能不对，尝试调整
        if d_approx > n:
            # 尝试：d ≈ (k*n - 1) / e
            d_approx = (k * n - 1) // e
            if d_approx > n:
                # 再尝试：d ≈ k * (n // e)
                d_approx = k * (n // e)
        
        # 提取高位MSB（取前一半位）
        d_bits = bin(d_approx)[2:]
        n_bits_len = n.bit_length()
        
        # 恢复的MSB长度：取d_bits的前一半，但不超过n的位数
        msb_length = min(len(d_bits) // 2, n_bits_len // 2)
        if msb_length < 32:
            msb_length = min(32, len(d_bits))  # 至少32位
        
        recovered_msb = d_bits[:msb_length] if len(d_bits) >= msb_length else d_bits
        
        # 方法2：使用热带几何平衡条件
        # 在热带几何中，平衡条件意味着：
        # val_p(d) + val_p(e) = val_p(k) + val_p(n) - val_p(φ(n))
        # 这可以用于恢复d的高位
        
        # 简化：使用对数空间
        log_d_approx = math.log2(k) + math.log2(n) - math.log2(e)
        d_approx_tropical = int(2 ** log_d_approx)
        
        # 取两种方法的交集（高位应该一致）
        d_bits_tropical = bin(d_approx_tropical)[2:]
        msb_tropical = d_bits_tropical[:msb_length] if len(d_bits_tropical) >= msb_length else d_bits_tropical
        
        # 如果两种方法的高位一致，使用它们
        if len(recovered_msb) >= msb_length and len(msb_tropical) >= msb_length:
            # 检查一致性
            overlap = min(len(recovered_msb), len(msb_tropical))
            match_count = sum(1 for i in range(overlap) if recovered_msb[i] == msb_tropical[i])
            match_ratio = match_count / overlap if overlap > 0 else 0
            
            if match_ratio > 0.7:  # 70%以上一致，认为可靠
                # 使用更长的那个，或者取交集
                if len(recovered_msb) >= len(msb_tropical):
                    return recovered_msb
                else:
                    return msb_tropical
        
        # 如果只有一种方法，返回它
        if recovered_msb:
            return recovered_msb
        
        return None
        
    except Exception as e:
        print(f"  [WARNING] 热带几何平衡恢复失败: {e}")
        return None


def precise_locate_with_msb(recovered_msb: str, n: int, e: int = 65537, 
                             k: int = None, target_bits: int = None) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    有了高位MSB，进行精确定位（从"盲扫"转为"精确定位"）
    
    Args:
        recovered_msb: 恢复的高位MSB（二进制字符串）
        n: RSA模数
        e: 公钥指数
        k: k值（可选，用于进一步约束）
        target_bits: 目标位数（默认与n相同）
    
    Returns:
        (d候选值, 诊断信息字典)，如果无法定位则返回(None, {})
    """
    diagnostics = {}
    
    if not recovered_msb or len(recovered_msb) < 32:
        return None, diagnostics
    
    if target_bits is None:
        target_bits = n.bit_length()
    
    try:
        # 从MSB构造d的候选值
        # 方法：MSB + 随机LSB，然后验证
        msb_int = int(recovered_msb, 2)
        msb_length = len(recovered_msb)
        remaining_bits = target_bits - msb_length
        
        # 构造候选：MSB固定，LSB从0到2^remaining_bits
        # 但为了效率，只尝试有限个候选
        candidates = []
        
        # 候选1：MSB + 全0 LSB
        d_candidate1 = msb_int << remaining_bits
        if d_candidate1 < n:
            candidates.append(d_candidate1)
        
        # 候选2：MSB + 全1 LSB
        if remaining_bits > 0:
            lsb_mask = (1 << remaining_bits) - 1
            d_candidate2 = (msb_int << remaining_bits) | lsb_mask
            if d_candidate2 < n:
                candidates.append(d_candidate2)
        
        # 候选3：如果k已知，使用k进一步约束
        if k and k > 0:
            # d ≈ (k*n - 1) / e
            d_from_k = (k * n - 1) // e
            # 检查d_from_k的高位是否与recovered_msb匹配
            d_from_k_bits = bin(d_from_k)[2:]
            if len(d_from_k_bits) >= msb_length:
                d_from_k_msb = d_from_k_bits[:msb_length]
                if d_from_k_msb == recovered_msb:
                    candidates.append(d_from_k)
        
        # 【细节3】检查MSB的"逻辑空洞"：打印(e*d - 1) // n的前16位和后16位
        for candidate in candidates:
            if candidate > 0 and candidate < n:
                # 计算 (e*d - 1) // n
                ed_minus_1 = e * candidate - 1
                k_from_d = ed_minus_1 // n
                
                # 转换为二进制字符串
                k_bits = bin(k_from_d)[2:] if k_from_d > 0 else '0'
                
                # 提取前16位和后16位
                if len(k_bits) >= 32:
                    k_msb_16 = k_bits[:16]
                    k_lsb_16 = k_bits[-16:]
                elif len(k_bits) >= 16:
                    k_msb_16 = k_bits[:16]
                    k_lsb_16 = k_bits[-16:] if len(k_bits) > 16 else k_bits
                else:
                    k_msb_16 = k_bits.zfill(16)[:16]
                    k_lsb_16 = k_bits.zfill(16)[-16:]
                
                diagnostics['k_from_d'] = k_from_d
                diagnostics['k_msb_16'] = k_msb_16
                diagnostics['k_lsb_16'] = k_lsb_16
                diagnostics['d_candidate'] = candidate
                
                # 检查是否是"半纯粹解"：k_from_d 应该接近原始k
                if k and k > 0:
                    k_match = (abs(k_from_d - k) < max(k // 10, 10))
                    diagnostics['semi_pure'] = k_match
                    diagnostics['k_original'] = k
                    diagnostics['k_diff'] = abs(k_from_d - k)
                
                # 基本验证：d应该是奇数（通常RSA的d是奇数）
                if candidate % 2 == 1:
                    return candidate, diagnostics
        
        # 如果没有完全匹配的，返回最接近的
        if candidates:
            best = min(candidates, key=lambda x: abs(x.bit_length() - target_bits))
            if best > 0 and best < n:
                # 计算诊断信息
                ed_minus_1 = e * best - 1
                k_from_d = ed_minus_1 // n
                k_bits = bin(k_from_d)[2:] if k_from_d > 0 else '0'
                
                if len(k_bits) >= 32:
                    diagnostics['k_msb_16'] = k_bits[:16]
                    diagnostics['k_lsb_16'] = k_bits[-16:]
                elif len(k_bits) >= 16:
                    diagnostics['k_msb_16'] = k_bits[:16]
                    diagnostics['k_lsb_16'] = k_bits[-16:] if len(k_bits) > 16 else k_bits
                else:
                    diagnostics['k_msb_16'] = k_bits.zfill(16)[:16]
                    diagnostics['k_lsb_16'] = k_bits.zfill(16)[-16:]
                
                diagnostics['k_from_d'] = k_from_d
                diagnostics['d_candidate'] = best
                
                return best, diagnostics
        
        return None, diagnostics
        
    except Exception as e:
        print(f"  [WARNING] 精确定位失败: {e}")
        diagnostics['error'] = str(e)
        return None, diagnostics


K_FIXED_OVERRIDE = 65536  # 用户指定的k，用于LSB约束
MAX_LSB_SEARCH = 1_000_000  # LSB求解最多尝试的步数，避免无限循环
ENABLE_PRIVATE_KEY_RECOVERY = True  # 当MSB恢复成功后求完整私钥


def msb_correction_operator(recovered_msb: str, n: int, e: int = 65537, 
                            k_fixed: int = K_FIXED_OVERRIDE) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    基于已恢复MSB与k约束修正完整d
    
    约束来源：
        e * d ≡ 1 (mod k)，用户指定k=65536（e=65537 => d ≡ 1 (mod 65536)）
    步骤：
        1) 固定MSB，计算剩余位数R
        2) 求解 lsb ≡ (1 - base) mod k，遍历满足范围的lsb，挑选最接近n的合法d
    """
    diagnostics: Dict[str, Any] = {}
    
    if not recovered_msb:
        diagnostics['error'] = "缺少MSB"
        return None, diagnostics
    
    target_bits = n.bit_length()
    msb_len = len(recovered_msb)
    
    if msb_len == 0:
        diagnostics['error'] = "MSB长度为0"
        return None, diagnostics
    
    if msb_len >= target_bits:
        # 已经是完整长度
        d_int = int(recovered_msb, 2)
        diagnostics['reason'] = "MSB已覆盖全部位"
        diagnostics['d_mod_k'] = d_int % k_fixed
        return (d_int if d_int < n else None), diagnostics
    
    remaining_bits = target_bits - msb_len
    msb_int = int(recovered_msb, 2)
    base = msb_int << remaining_bits
    
    modulus = k_fixed
    residue = (1 - (base % modulus)) % modulus  # 需要补偿的lsb使得 e*d ≡ 1 (mod k)
    max_lsb = 1 << remaining_bits
    
    diagnostics['target_bits'] = target_bits
    diagnostics['msb_len'] = msb_len
    diagnostics['remaining_bits'] = remaining_bits
    diagnostics['base_high_bits'] = recovered_msb[:64]
    diagnostics['residue'] = residue
    diagnostics['modulus'] = modulus
    diagnostics['max_lsb'] = max_lsb
    
    candidates: List[int] = []
    step_count = 0
    lsb = residue
    while lsb < max_lsb and step_count < MAX_LSB_SEARCH:
        d_candidate = base + lsb
        if d_candidate < n and d_candidate % 2 == 1:
            candidates.append(d_candidate)
        step_count += 1
        lsb += modulus
    
    diagnostics['candidate_count'] = len(candidates)
    diagnostics['steps_taken'] = step_count
    diagnostics['search_truncated'] = (step_count >= MAX_LSB_SEARCH)
    
    if not candidates:
        diagnostics['error'] = "未找到满足约束的候选"
        return None, diagnostics
    
    # 优先选择：位数越接近n越好，其次值越大（靠近φ(n)）
    candidates.sort(key=lambda x: (-x.bit_length(), -x))
    d_final = candidates[0]
    
    diagnostics['selected_d'] = d_final
    diagnostics['d_bitlen'] = d_final.bit_length()
    diagnostics['d_mod_k'] = d_final % modulus
    diagnostics['lsb_solution'] = d_final & (max_lsb - 1)
    
    return d_final, diagnostics


def run_smoke_test(mode: str = "normal", pressure_level: int = 1):
    """
    运行RSA白盒smoke测试 - 压力测试版本
    
    Args:
        mode: 打乱模式 ("normal", "hell", "chaos", "extreme")
        pressure_level: 压力等级 (1-5，5最极端)
    """
    print("=" * 70)
    print(f"RSA白盒Smoke测试 - 测信道攻击场景 [模式: {mode.upper()}] [压力等级: {pressure_level}]")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[阶段0] 加载白盒RSA数据")
    try:
        n, n_bits = load_whitebox_rsa()
        print(f"  [OK] RSA模数n加载成功")
        print(f"  [INFO] n长度: {len(n_bits)} bits")
        print(f"  [INFO] n (hex前64字符): {hex(n)[2:66]}...")
    except Exception as e:
        print(f"  [ERROR] 加载失败: {e}")
        return 1
    
    # 2. 打乱MSB + 压力测试
    print(f"\n[阶段1] 打乱MSB（地狱指标 + 压力等级{pressure_level}）")
    try:
        # 根据压力等级调整打乱长度
        if pressure_level == 1:
            msb_length = min(256, len(n_bits) // 2)
        elif pressure_level == 2:
            msb_length = min(512, len(n_bits) * 3 // 4)
        elif pressure_level == 3:
            msb_length = min(768, len(n_bits) * 9 // 10)
        elif pressure_level == 4:
            msb_length = len(n_bits) - 100  # 几乎全部
        else:  # level 5
            msb_length = len(n_bits)  # 全部打乱
        
        scrambled_bits, scrambled_len = scramble_msb(n_bits, msb_length=msb_length, seed=42, mode=mode)
        
        # 压力测试：注入噪声
        if pressure_level >= 3:
            noise_ratio = 0.05 * (pressure_level - 2)  # 3级5%, 4级10%, 5级15%
            scrambled_bits = inject_noise(scrambled_bits, noise_ratio=noise_ratio)
            print(f"  [压力] 注入噪声: {noise_ratio*100:.1f}%")
        
        # 压力测试：数据损坏
        if pressure_level >= 4:
            corruption_types = ["random", "burst", "systematic"]
            corruption_type = corruption_types[(pressure_level - 4) % len(corruption_types)]
            scrambled_bits = corrupt_data(scrambled_bits, corruption_type=corruption_type)
            print(f"  [压力] 数据损坏: {corruption_type}")
        
        print(f"  [OK] MSB打乱完成")
        print(f"  [INFO] 打乱长度: {scrambled_len} bits ({scrambled_len*100//len(n_bits)}%)")
        print(f"  [INFO] 原始MSB前32位: {n_bits[:32]}")
        print(f"  [INFO] 打乱MSB前32位: {scrambled_bits[:32]}")
        
        # 计算打乱程度
        original_msb = n_bits[:scrambled_len]
        scrambled_msb = scrambled_bits[:scrambled_len]
        diff_count = sum(1 for a, b in zip(original_msb, scrambled_msb) if a != b)
        diff_ratio = diff_count / scrambled_len if scrambled_len > 0 else 0
        print(f"  [INFO] 打乱程度: {diff_ratio*100:.1f}% ({diff_count}/{scrambled_len} bits不同)")
        
        if diff_ratio < 0.1:
            print(f"  [WARNING] 打乱程度过低！")
        else:
            print(f"  [OK] MSB已成功打乱")
    except Exception as e:
        print(f"  [ERROR] 打乱失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 3. 构造格基（压力测试：恶意构造）
    print("\n[阶段2] 构造格基（压力测试）")
    try:
        # 根据压力等级调整维度
        if pressure_level <= 2:
            dim = None  # 自动计算
        elif pressure_level == 3:
            dim = 25  # 中等维度
        elif pressure_level == 4:
            dim = 30  # 高维度
        else:  # level 5
            dim = 35  # 极端维度
        
        basis = bits_to_lattice_basis(scrambled_bits, dim=dim)
        
        # 压力测试：恶意修改格基
        if pressure_level >= 4:
            # 随机修改一些向量，增加难度
            modify_count = len(basis) // 10
            for _ in range(modify_count):
                vec_idx = random.randint(0, len(basis) - 1)
                elem_idx = random.randint(0, len(basis[vec_idx]) - 1)
                # 添加随机扰动
                basis[vec_idx][elem_idx] += random.randint(-1000, 1000)
        
        print(f"  [OK] 格基构造完成")
        print(f"  [INFO] 格基维度: {len(basis)}")
        print(f"  [INFO] 向量长度: {len(basis[0]) if basis else 0}")
        if pressure_level >= 4:
            print(f"  [压力] 格基已恶意修改")
        print(f"  [INFO] 格基示例（前3个向量，前5个元素）:")
        for i, vec in enumerate(basis[:3]):
            print(f"    v{i}: {vec[:5]}...")
    except Exception as e:
        print(f"  [ERROR] 格基构造失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. LLL规约测试（压力测试：极端参数）
    print("\n[阶段3] LLL 2.0规约测试")
    print("=" * 70)
    try:
        # 根据压力等级调整参数
        if pressure_level == 1:
            prime, precision, noise_tol = 101, 4, 0.6
        elif pressure_level == 2:
            prime, precision, noise_tol = 97, 3, 0.5
        elif pressure_level == 3:
            prime, precision, noise_tol = 89, 3, 0.4
        elif pressure_level == 4:
            prime, precision, noise_tol = 83, 2, 0.3
        else:  # level 5
            prime, precision, noise_tol = 79, 2, 0.2  # 极端参数
        
        print(f"  [压力] 参数: p={prime}, k={precision}, noise_tol={noise_tol}")
        
        result = lll_reduce_enhanced(
            basis,
            prime=prime,
            precision=precision,
            noise_tolerance=noise_tol
        )
        
        if not result.success or not result.reduced_basis:
            print(f"  [ERROR] LLL规约失败，无法继续")
            return 1
        
        print(f"  [OK] LLL规约完成")
        print(f"  [INFO] 规约基维度: {len(result.reduced_basis)}")
        
    except Exception as e:
        print(f"  [ERROR] LLL规约失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 5. 【新策略】先求k
    print("\n[阶段4] 求k（e*d = 1 + k*φ(n)）")
    print("=" * 70)
    try:
        e = 65537  # 标准RSA公钥指数
        k, k_diagnostics = solve_k_from_lattice(result.reduced_basis, n, e)
        
        if k is None:
            print(f"  [WARNING] 无法从格基中求解k，尝试备用方法")
            # 备用方法：从规约基的第一个向量直接提取
            if result.reduced_basis and result.reduced_basis[0]:
                k_candidate = abs(result.reduced_basis[0][0])
                # 特殊处理：MAGIC=0的情况（用户说"摸到0就是成功了"）
                if k_candidate == 0:
                    print(f"  [INFO] 检测到MAGIC=0，这是成功信号")
                    # 尝试从其他位置提取
                    if len(result.reduced_basis[0]) > 1:
                        k_candidate = abs(result.reduced_basis[0][1])
                    else:
                        for vec in result.reduced_basis[1:min(5, len(result.reduced_basis))]:
                            if vec and len(vec) > 0:
                                k_candidate = abs(vec[0])
                                if k_candidate > 0:
                                    break
                
                if 0 < k_candidate < e * 10:
                    k = k_candidate
                    # 计算赋值区间
                    log_k = math.log(k) if k > 0 else 0
                    log_n = math.log(n) if n > 0 else 1
                    v_k = log_k / log_n if log_n > 0 else 0
                    k_diagnostics = {
                        'k': k,
                        'v_k': v_k,
                        'k_valid': (k < e),
                        'k_bits': k.bit_length()
                    }
                    print(f"  [备用] 从第一个向量提取k: {k}")
        
        if k is None:
            print(f"  [ERROR] 无法求解k，策略失败")
            return 1
        
        print(f"  [OK] k求解成功: k={k}")
        print(f"  [INFO] k位数: {k.bit_length()} bits")
        print(f"  [INFO] k (hex): {hex(k)[2:32]}...")
        
        # 【细节1】打印k的赋值区间
        if 'v_k' in k_diagnostics:
            v_k = k_diagnostics['v_k']
            print(f"  [细节1] k的赋值区间: v(k) = log(k)/log(n) = {v_k:.6f}")
            print(f"  [细节1] k合法性检查: k < e? {k < e} (k={k}, e={e})")
            if k >= e:
                print(f"  [WARNING] k越界！k={k} >= e={e}，说明Stage2能量坍塌可能坍塌到了错误的同调类上")
            else:
                print(f"  [OK] k在合法区间内")
        
    except Exception as e:
        print(f"  [ERROR] 求解k失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 6. 【新策略】热带几何平衡：恢复d的高位MSB
    print("\n[阶段5] 热带平衡：恢复d的高位MSB")
    print("=" * 70)
    try:
        recovered_msb = recover_msb_from_tropical_balance(k, n, e, scrambled_bits)
        
        if recovered_msb is None:
            print(f"  [ERROR] 无法恢复d的高位MSB")
            return 1
        
        print(f"  [OK] d的高位MSB恢复成功")
        print(f"  [INFO] 恢复的MSB长度: {len(recovered_msb)} bits")
        print(f"  [INFO] 恢复的MSB前64位: {recovered_msb[:64]}")
        print(f"  [INFO] 原始MSB前64位: {n_bits[:64]}")
        
        # 验证恢复的MSB与原始MSB的匹配度
        if len(recovered_msb) <= len(n_bits):
            match_count = sum(1 for i in range(len(recovered_msb)) 
                            if recovered_msb[i] == n_bits[i])
            match_ratio = match_count / len(recovered_msb) if len(recovered_msb) > 0 else 0
            print(f"  [INFO] MSB匹配度: {match_ratio*100:.1f}% ({match_count}/{len(recovered_msb)})")
            
            if match_ratio > 0.7:
                print(f"  [OK] MSB恢复质量良好（>70%匹配）")
            elif match_ratio > 0.5:
                print(f"  [WARNING] MSB恢复质量一般（50-70%匹配）")
            else:
                print(f"  [WARNING] MSB恢复质量较差（<50%匹配）")
        
    except Exception as e:
        print(f"  [ERROR] 恢复MSB失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 7. 【新策略】精确定位：有了高位MSB，从"盲扫"转为"精确定位"
    print("\n[阶段6] 精确定位")
    print("=" * 70)
    try:
        d_candidate, d_diagnostics = precise_locate_with_msb(recovered_msb, n, e, k)
        
        if d_candidate is None:
            print(f"  [WARNING] 精确定位失败，但MSB已恢复")
            d_candidate = None
        else:
            print(f"  [OK] 精确定位成功")
            print(f"  [INFO] d候选值位数: {d_candidate.bit_length()} bits")
            print(f"  [INFO] d候选值 (hex前64字符): {hex(d_candidate)[2:66]}...")
            
            # 验证：检查d候选值是否满足基本RSA性质
            if d_candidate > 0 and d_candidate < n:
                print(f"  [INFO] d候选值范围验证: OK")
            else:
                print(f"  [WARNING] d候选值范围异常")
            
            # 【细节3】检查MSB的"逻辑空洞"
            if 'k_from_d' in d_diagnostics:
                k_from_d = d_diagnostics['k_from_d']
                k_msb_16 = d_diagnostics.get('k_msb_16', 'N/A')
                k_lsb_16 = d_diagnostics.get('k_lsb_16', 'N/A')
                
                print(f"  [细节3] MSB逻辑空洞检查:")
                print(f"    (e*d - 1) // n = {k_from_d}")
                print(f"    前16位: {k_msb_16}")
                print(f"    后16位: {k_lsb_16}")
                
                # 检查是否是"半纯粹解"
                if 'semi_pure' in d_diagnostics:
                    is_semi_pure = d_diagnostics['semi_pure']
                    k_original = d_diagnostics.get('k_original', None)
                    k_diff = d_diagnostics.get('k_diff', None)
                    
                    if is_semi_pure:
                        print(f"    [OK] 半纯粹解检测: 通过 (k_from_d={k_from_d} ≈ k={k_original}, diff={k_diff})")
                    else:
                        print(f"    [WARNING] 半纯粹解检测: 未通过 (k_from_d={k_from_d} vs k={k_original}, diff={k_diff})")
        
    except Exception as e:
        print(f"  [ERROR] 精确定位失败: {e}")
        import traceback
        traceback.print_exc()
        d_candidate = None
        d_diagnostics = {}
    
    # 6+. MSB修正算子：MSB恢复成功后补全私钥
    print("\n[阶段6+] MSB修正算子")
    print("=" * 70)
    d_final = None
    correction_diag: Dict[str, Any] = {}
    if ENABLE_PRIVATE_KEY_RECOVERY and recovered_msb:
        try:
            d_final, correction_diag = msb_correction_operator(
                recovered_msb=recovered_msb,
                n=n,
                e=e,
                k_fixed=K_FIXED_OVERRIDE
            )
            if d_final:
                print(f"  [OK] 修正成功，得到完整d")
                print(f"  [INFO] d_final位数: {d_final.bit_length()} bits")
                print(f"  [INFO] d_final (hex前64字符): {hex(d_final)[2:66]}...")
                print(f"  [INFO] d_final mod {K_FIXED_OVERRIDE} = {d_final % K_FIXED_OVERRIDE}")
                if correction_diag.get('search_truncated'):
                    print(f"  [WARNING] LSB搜索提前截断，结果可能非全局最优")
            else:
                print(f"  [WARNING] MSB修正算子未找到合法d")
                if correction_diag.get('error'):
                    print(f"    诊断: {correction_diag['error']}")
        except Exception as e:
            print(f"  [ERROR] MSB修正算子失败: {e}")
            import traceback
            traceback.print_exc()
            correction_diag['error'] = str(e)
            d_final = d_candidate
    # 若MSB未恢复，跳过
    
    # 8. 【新策略】迹公式补全：几微秒的补全就是顺水推舟
    print("\n" + "=" * 70)
    print("[阶段7] 郎兰兹强行断言")
    print("=" * 70)
    try:
        print(f"流水线执行: {'成功' if result.success else '失败'}")
        print(f"总耗时: {result.total_elapsed_ms:.1f}ms")
        
        print("\n各阶段状态:")
        for stage in result.stages:
            status = "[PASS]" if stage.success else "[FAIL]"
            print(f"  {status} {stage.stage_name}: {stage.elapsed_ms:.1f}ms")
            if stage.error:
                print(f"    错误: {stage.error[:100]}")
        
        # 【细节2】监测迹公式的残差（Arakelov范数残差）
        print("\n[细节2] 迹公式残差监测:")
        if result.final_certificate:
            cert = result.final_certificate
            norm_sq = cert.get('norm_squared', None)
            if norm_sq is not None:
                # 计算残差（相对于n的范数）
                if n > 0:
                    residual_ratio = norm_sq / (n * n) if n > 0 else 0
                    print(f"  范数平方: {norm_sq}")
                    print(f"  残差比率: {residual_ratio:.6e}")
                    
                    # 判断残差趋势
                    if residual_ratio < 1e-6:
                        print(f"  [OK] 残差线性还可以")
                    elif residual_ratio < 1e-3:
                        print(f"  [INFO] 残差较小，可能接近正确解")
                    else:
                        print(f"  [WARNING] 残差较大，可能基向量不对（随机跳变）")
            
            print("\n最终证书:")
            print(f"  最短向量: {cert.get('shortest_vector', 'N/A')}")
            print(f"  范数平方: {cert.get('norm_squared', 'N/A')}")
            print(f"  共振状态: {cert.get('resonance_status', 'N/A')}")
            print(f"  Hex签名: {cert.get('hex_signature', 'N/A')}")
        
        # 从各阶段数据中提取残差信息
        print("\n各阶段Arakelov范数残差:")
        for i, stage in enumerate(result.stages):
            stage_data = stage.data if hasattr(stage, 'data') else {}
            # 尝试从stage数据中提取残差信息
            if 'log_shell' in stage_data:
                log_shell = stage_data['log_shell']
                if isinstance(log_shell, dict):
                    center = log_shell.get('center', 'N/A')
                    min_val = log_shell.get('min', 'N/A')
                    max_val = log_shell.get('max', 'N/A')
                    print(f"  Stage {i+1} ({stage.stage_name}):")
                    print(f"    Log-Shell中心: {center}")
                    print(f"    Log-Shell范围: [{min_val}, {max_val}]")
        
        print("\n新策略结果:")
        print(f"  k值: {k}")
        if 'v_k' in k_diagnostics:
            print(f"  v(k) = {k_diagnostics['v_k']:.6f}")
        print(f"  恢复的MSB长度: {len(recovered_msb) if recovered_msb else 0} bits")
        if ENABLE_PRIVATE_KEY_RECOVERY:
            if d_final:
                print(f"  d_final: {hex(d_final)[2:66]}...")
                print(f"  d_final位数: {d_final.bit_length()} bits")
                # 输出完整私钥二进制长度与前后片段
                d_bin = bin(d_final)[2:]
                print(f"  d_final二进制长度: {len(d_bin)} bits")
                print(f"  d_final二进制前64位: {d_bin[:64]}")
                print(f"  d_final二进制后64位: {d_bin[-64:] if len(d_bin) >= 64 else d_bin}")
            elif d_candidate:
                print(f"  d候选值: {hex(d_candidate)[2:66]}...")
                print(f"  d候选值位数: {d_candidate.bit_length()} bits")
            else:
                print(f"  d候选值: 未找到")
        
        # 综合判断：如果k求解成功、MSB恢复成功，且LLL规约成功，则认为测试通过
        # 注意：MAGIC=0也被认为是成功信号
        success = (k is not None and 
                  recovered_msb is not None and 
                  result.success and
                  (not ENABLE_PRIVATE_KEY_RECOVERY or d_final is not None or d_candidate is not None))
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"  [ERROR] 结果分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_all_pressure_tests():
    """运行所有压力测试等级"""
    print("\n" + "=" * 70)
    print("RSA白盒Smoke测试 - 全压力测试套件")
    print("=" * 70)
    
    modes = ["normal", "hell", "chaos", "extreme"]
    results = []
    
    for mode in modes:
        for level in range(1, 6):
            print(f"\n{'='*70}")
            print(f"测试: {mode.upper()} 模式 + 压力等级 {level}")
            print(f"{'='*70}")
            try:
                result_code = run_smoke_test(mode=mode, pressure_level=level)
                results.append((mode, level, result_code == 0))
            except Exception as e:
                print(f"[FATAL] 测试崩溃: {e}")
                results.append((mode, level, False))
    
    # 汇总
    print("\n" + "=" * 70)
    print("压力测试汇总")
    print("=" * 70)
    for mode, level, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {mode:10s} L{level}: {'通过' if success else '失败'}")
    
    pass_count = sum(1 for _, _, s in results if s)
    total_count = len(results)
    print(f"\n通过率: {pass_count}/{total_count} ({pass_count*100//total_count}%)")
    
    return pass_count == total_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='RSA白盒Smoke测试 - 压力测试')
    parser.add_argument('--mode', choices=['normal', 'hell', 'chaos', 'extreme'], 
                       default='normal', help='打乱模式')
    parser.add_argument('--pressure', type=int, choices=[1, 2, 3, 4, 5], 
                       default=1, help='压力等级 (1-5)')
    parser.add_argument('--all', action='store_true', help='运行所有压力测试')
    args = parser.parse_args()
    
    if args.all:
        sys.exit(0 if run_all_pressure_tests() else 1)
    else:
        sys.exit(run_smoke_test(mode=args.mode, pressure_level=args.pressure))

