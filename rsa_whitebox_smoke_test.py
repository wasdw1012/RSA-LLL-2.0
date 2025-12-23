#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA 直接攻击 smoke（确定性、无交互）

本 smoke 只做一件事：用 `rsa_direct_attack.wiener_attack()` 跑一遍内置
公钥参数，验证脚本可运行、日志健康输出、失败策略明确（不静默）。
"""
from rsa_direct_attack import wiener_attack

# 默认公钥（来自你原脚本里 embedded 的公开参数）
PUBLIC_KEY_N = 147004412277921838680869025013592551987344186594935222797512997046358586799421960015983436424475673494712754193384589284928836482051596899986862524771250138856271015997223731800253202794745001453488564519971178376789260933485597039961213953127729514973034706299250888361513076798942263376704534236697921673073
PUBLIC_KEY_E = 65537


def main() -> int:
    res = wiener_attack(PUBLIC_KEY_N, PUBLIC_KEY_E, verbose=True)
    if res.success:
        print(f"[SMOKE] OK d_bits={res.d.bit_length()} p_bits={res.p.bit_length()} q_bits={res.q.bit_length()}")
        return 0
    print(f"[SMOKE] FAIL: {res.error}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
