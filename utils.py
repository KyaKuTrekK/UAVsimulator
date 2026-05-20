import numpy as np
from scipy.special import erfinv

# 模型名称映射
MODEL_LABELS = {
    'free_space': '自由空间',
    'probabilistic': '概率信道 (LoS/NLoS)',
    'rician': '莱斯 (简化)',
    'rician_fading': '莱斯衰落 (随机)',
    'nakagami_fading': 'Nakagami‑m 衰落',
    'height_dependent': '高度依赖 (GBS-UAV)',
    'angle_dependent': '角度依赖 (仰角θ)',
    'prob_los_angle': '概率LoS (角度)',
    'shadowing': '阴影衰落 (正弦波叠加)',
    'gaussian_shadow': '高斯阴影',
    '3gpp_uma': '3GPP 混合信道 (UMa)',
    'custom': '自定义公式'
}

ENV_LABELS = {'open': '空旷', 'suburban': '郊区', 'urban': '城市'}

# 每个模型对应的默认自定义参数
DEFAULT_PARAMS = {
    'free_space': [
        {'name': 'Pt', 'type': 'number', 'value': 30, 'min': 0, 'max': 50, 'step': 0.5},
        {'name': 'Gt', 'type': 'number', 'value': 3, 'min': 0, 'max': 20, 'step': 0.1},
        {'name': 'Gr', 'type': 'number', 'value': 3, 'min': 0, 'max': 20, 'step': 0.1},
        {'name': 'Bandwidth', 'type': 'number', 'value': 20, 'min': 1, 'max': 100, 'step': 1},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7, 'min': 0, 'max': 15, 'step': 0.1},
    ],
    'probabilistic': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
    ],
    'rician': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
    ],
    'rician_fading': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'K', 'type': 'number', 'value': 10, 'min': 0.1, 'max': 100, 'step': 0.5},
        {'name': 'alpha', 'type': 'number', 'value': 2.0, 'min': 2, 'max': 6, 'step': 0.1},
        {'name': 'seed', 'type': 'number', 'value': 123, 'min': 0, 'max': 9999, 'step': 1},
    ],
    'nakagami_fading': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'm', 'type': 'number', 'value': 2, 'min': 0.5, 'max': 20, 'step': 0.1},
        {'name': 'alpha', 'type': 'number', 'value': 2.0, 'min': 2, 'max': 6, 'step': 0.1},
        {'name': 'seed', 'type': 'number', 'value': 456, 'min': 0, 'max': 9999, 'step': 1},
    ],
    'height_dependent': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'p1', 'type': 'number', 'value': 4.5, 'min': 2, 'max': 10, 'step': 0.1},
        {'name': 'p2', 'type': 'number', 'value': 0.8, 'min': 0.1, 'max': 2, 'step': 0.1},
    ],
    'angle_dependent': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'A', 'type': 'number', 'value': -12, 'min': -30, 'max': 0, 'step': 0.1},
        {'name': 'B', 'type': 'number', 'value': 15, 'min': 1, 'max': 50, 'step': 0.5},
        {'name': 'theta0', 'type': 'number', 'value': 10, 'min': 0, 'max': 90, 'step': 1},
        {'name': 'eta0', 'type': 'number', 'value': 2, 'min': 0, 'max': 20, 'step': 0.1},
        {'name': 'a_sig', 'type': 'number', 'value': 0.3, 'min': 0, 'max': 2, 'step': 0.1},
        {'name': 'sigma0', 'type': 'number', 'value': 4, 'min': 0, 'max': 15, 'step': 0.1},
    ],
    'prob_los_angle': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'alpha', 'type': 'number', 'value': 2.7, 'min': 2, 'max': 6, 'step': 0.1},
        {'name': 'kappa', 'type': 'number', 'value': 0.5, 'min': 0.1, 'max': 0.9, 'step': 0.1},
        {'name': 'a_prob', 'type': 'number', 'value': 9.6, 'min': 1, 'max': 50, 'step': 0.1},
        {'name': 'b_prob', 'type': 'number', 'value': 0.28, 'min': 0.01, 'max': 1, 'step': 0.01},
        {'name': 'theta0_prob', 'type': 'number', 'value': 15, 'min': 0, 'max': 90, 'step': 1},
    ],
    'shadowing': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'alpha', 'type': 'number', 'value': 2.0, 'min': 2, 'max': 6, 'step': 0.1},
        {'name': 'sigma_shadow', 'type': 'number', 'value': 8, 'min': 1, 'max': 20, 'step': 0.5},
        {'name': 'N_sin', 'type': 'number', 'value': 16, 'min': 4, 'max': 50, 'step': 1},
        {'name': 'seed', 'type': 'number', 'value': 42, 'min': 0, 'max': 9999, 'step': 1},
    ],
    'gaussian_shadow': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'alpha', 'type': 'number', 'value': 2.0, 'min': 2, 'max': 6, 'step': 0.1},
        {'name': 'sigma_shadow', 'type': 'number', 'value': 8, 'min': 1, 'max': 20, 'step': 0.5},
        {'name': 'dc', 'type': 'number', 'value': 30, 'min': 5, 'max': 200, 'step': 1},
        {'name': 'N_sin', 'type': 'number', 'value': 16, 'min': 4, 'max': 50, 'step': 1},
        {'name': 'seed', 'type': 'number', 'value': 42, 'min': 0, 'max': 9999, 'step': 1},
    ],
    '3gpp_uma': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
        {'name': 'scenario', 'type': 'select', 'value': 'UMa', 'options': ['UMa', 'RMa']},
        {'name': 'h_BS', 'type': 'number', 'value': 25, 'min': 10, 'max': 100, 'step': 1},
        {'name': 'h_UT', 'type': 'number', 'value': 1.5, 'min': 1, 'max': 10, 'step': 0.5},
        {'name': 'sigma_SF', 'type': 'number', 'value': 4, 'min': 0, 'max': 12, 'step': 0.5},
        {'name': 'K_factor', 'type': 'number', 'value': 10, 'min': 0, 'max': 100, 'step': 0.5},
        {'name': 'fast_fading', 'type': 'select', 'value': 'rician', 'options': ['rician', 'rayleigh']},
        {'name': 'seed', 'type': 'number', 'value': 123, 'min': 0, 'max': 9999, 'step': 1},
    ],
    'custom': [
        {'name': 'Pt', 'type': 'number', 'value': 30}, {'name': 'Gt', 'type': 'number', 'value': 3},
        {'name': 'Gr', 'type': 'number', 'value': 3}, {'name': 'Bandwidth', 'type': 'number', 'value': 20},
        {'name': 'NoiseFigure', 'type': 'number', 'value': 7},
    ]
}

# 确定性伪随机数生成器
def mulberry32(seed):
    s = int(seed)
    def rand():
        nonlocal s
        s += 0x6D2B79F5
        t = (s ^ (s >> 15)) * (1 | s)
        t ^= t + (t ^ (t >> 7)) * (61 | t)
        return ((t ^ (t >> 14)) >> 0) / 4294967296
    return rand

# Box-Muller 法生成标准正态
def randn(rand_func):
    u1 = rand_func()
    u2 = rand_func()
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

# Gamma 分布 (Marsaglia-Tsang)
def gamma_rand(shape, scale, rand_func):
    if shape < 1:
        u = rand_func()
        return gamma_rand(1 + shape, scale, rand_func) * u ** (1 / shape)
    d = shape - 1 / 3
    c = 1 / np.sqrt(9 * d)
    while True:
        x = randn(rand_func)
        v = (1 + c * x) ** 3
        if v <= 0:
            continue
        u = rand_func()
        if u < 1 - 0.0331 * x**4:
            return d * v * scale
        if np.log(u) < 0.5 * x**2 + d * (1 - v + np.log(v)):
            return d * v * scale

# 自由空间路径损耗 (dB)
def free_space_loss(distance, freq_ghz):
    lam = 3e8 / (freq_ghz * 1e9)
    return 20 * np.log10(4 * np.pi * distance / lam)

# 仰角 (度)
def elevation(height, distance):
    return np.degrees(np.arctan2(height, distance))