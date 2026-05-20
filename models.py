import numpy as np
from utils import (free_space_loss, elevation, randn, gamma_rand, mulberry32)

def compute_pathloss(model, height, distance, env, freq_ghz, custom_params):
    """
    返回: path_loss_dB
    """
    fspl = free_space_loss(distance, freq_ghz)
    theta = elevation(height, distance)

    # 简单 LoS 概率
    if env == 'open':
        pLos = 1.0
    elif env == 'suburban':
        pLos = min(1, 1 / (1 + 0.15 * np.exp(-0.005 * (height - 30))))
    else:  # urban
        pLos = min(1, 1 / (1 + 0.3 * np.exp(-0.003 * (height - 50))))

    if model == 'free_space':
        return fspl
    elif model == 'probabilistic':
        excess = {'open': 0, 'suburban': 12, 'urban': 25}[env]
        return pLos * fspl + (1 - pLos) * (fspl + excess)
    elif model == 'rician':
        return fspl + 2
    elif model == 'rician_fading':
        K = custom_params.get('K', 10)
        alpha = custom_params.get('alpha', 2.0)
        pl_fs = 10 * alpha * np.log10(distance) + free_space_loss(distance, freq_ghz)  # 实际上直接用fspl重算
        # 更准确：自由空间指数alpha
        pl_fs = 10 * alpha * np.log10(distance) + 20 * np.log10(4 * np.pi / (3e8 / (freq_ghz * 1e9)))
        seed = custom_params.get('seed', 123) + int(distance * 1000)
        rand = mulberry32(seed)
        s = np.sqrt(2 * K / (K + 1))
        sigma = np.sqrt(1 / (2 * (K + 1)))
        real = s + sigma * randn(rand)
        imag = sigma * randn(rand)
        r = np.sqrt(real**2 + imag**2)
        return pl_fs - 20 * np.log10(r)
    elif model == 'nakagami_fading':
        m_val = custom_params.get('m', 2)
        alpha = custom_params.get('alpha', 2.0)
        pl_fs = 10 * alpha * np.log10(distance) + 20 * np.log10(4 * np.pi / (3e8 / (freq_ghz * 1e9)))
        seed = custom_params.get('seed', 456) + int(distance * 1000) + 999
        rand = mulberry32(seed)
        g = gamma_rand(m_val, 1 / m_val, rand)
        r = np.sqrt(max(g, 1e-15))
        return pl_fs - 20 * np.log10(r)
    elif model == 'height_dependent':
        p1 = custom_params.get('p1', 4.5)
        p2 = custom_params.get('p2', 0.8)
        a = max(p1 - p2 * np.log10(height), 2)
        return 10 * a * np.log10(distance) + 20 * np.log10(4 * np.pi / (3e8 / (freq_ghz * 1e9)))
    elif model == 'angle_dependent':
        A = custom_params.get('A', -12)
        B = custom_params.get('B', 15)
        th0 = custom_params.get('theta0', 10)
        eta0 = custom_params.get('eta0', 2)
        delta = theta - th0
        eta = A * delta * np.exp(-delta / B) + eta0
        return fspl + eta
    elif model == 'prob_los_angle':
        alpha = custom_params.get('alpha', 2.7)
        kappa = custom_params.get('kappa', 0.5)
        a_prob = custom_params.get('a_prob', 9.6)
        b_prob = custom_params.get('b_prob', 0.28)
        th0_prob = custom_params.get('theta0_prob', 15)
        PLos = 1 / (1 + a_prob * np.exp(-b_prob * (theta - th0_prob)))
        beta0_dB = 20 * np.log10(4 * np.pi / (3e8 / (freq_ghz * 1e9)))
        return beta0_dB + 10 * alpha * np.log10(distance) - 10 * np.log10(PLos + kappa * (1 - PLos))
    elif model == 'shadowing':
        # 简化版正弦波叠加阴影
        alpha = custom_params.get('alpha', 2.0)
        sigma = custom_params.get('sigma_shadow', 8)
        N = int(custom_params.get('N_sin', 16))
        seed = custom_params.get('seed', 42)
        rand = mulberry32(seed)
        v = 0
        cn = np.sqrt(2 / N)
        for n in range(N):
            sn = (n + 1) * 0.05
            phase = rand() * 2 * np.pi
            v += cn * np.cos(2 * np.pi * sn * distance + phase)
        shadow = sigma * v
        pl_fs = 10 * alpha * np.log10(distance) + 20 * np.log10(4 * np.pi / (3e8 / (freq_ghz * 1e9)))
        return pl_fs + shadow
    elif model == 'gaussian_shadow':
        alpha = custom_params.get('alpha', 2.0)
        sigma = custom_params.get('sigma_shadow', 8)
        dc = custom_params.get('dc', 30)
        N = int(custom_params.get('N_sin', 16))
        seed = custom_params.get('seed', 42)
        rand = mulberry32(seed)
        v = 0
        for n in range(N):
            fn = (n + 1) * 0.5 / dc
            sn = fn
            S = sigma**2 * dc * np.sqrt(np.pi) * np.exp(-(np.pi * dc * fn)**2)
            amp = np.sqrt(2 * S * (0.5 / dc))
            phase = rand() * 2 * np.pi
            v += amp * np.cos(2 * np.pi * sn * distance + phase)
        pl_fs = 10 * alpha * np.log10(distance) + 20 * np.log10(4 * np.pi / (3e8 / (freq_ghz * 1e9)))
        return pl_fs + v
    elif model == '3gpp_uma':
        scenario = custom_params.get('scenario', 'UMa')
        hBS = custom_params.get('h_BS', 25)
        hUT = custom_params.get('h_UT', 1.5)
        sigmaSF = custom_params.get('sigma_SF', 4)
        fast = custom_params.get('fast_fading', 'rician')
        Kfac = custom_params.get('K_factor', 10)
        seed = custom_params.get('seed', 123)
        d2D = distance
        d3D = np.sqrt(d2D**2 + (hBS - hUT)**2)
        fc = freq_ghz

        # 3GPP UMa LoS概率
        if scenario == 'UMa':
            if d2D <= 18:
                PLos = 1.0
            else:
                C = ((hUT - 13) / 10)**1.5 if hUT > 13 else 0
                PLos = (18 / d2D + np.exp(-d2D / 63) * (1 - 18 / d2D)) * (1 + C * 1.25 * (d2D / 100)**3 * np.exp(-d2D / 150))
        else:  # RMa
            if d2D <= 10:
                PLos = 1.0
            else:
                PLos = np.exp(-(d2D - 10) / 1000)
        PLos = min(1, max(0, PLos))

        # 路径损耗
        if scenario == 'UMa':
            hE = 1.0
            dBP = 4 * (hBS - hE) * (hUT - hE) * fc / 0.3
            if d2D <= dBP:
                plLos = 28.0 + 22 * np.log10(d3D) + 20 * np.log10(fc)
            else:
                plLos = 28.0 + 40 * np.log10(d3D) + 20 * np.log10(fc) - 9 * np.log10(dBP**2 + (hBS - hUT)**2)
            plNlos = 13.54 + 39.08 * np.log10(d3D) + 20 * np.log10(fc) - 0.6 * (hUT - 1.5)
        else:
            plLos = 20 * np.log10(40 * np.pi * d3D * fc / 3) + min(0.03 * 5**1.72, 10) * np.log10(d3D) - min(0.044 * 5**1.72, 14.77) + 0.002 * np.log10(5) * d3D
            plNlos = 161.04 - 7.1 * np.log10(20) + 7.5 * np.log10(5) - (24.37 - 3.7 * (5 / hBS)**2) * np.log10(hBS) + (43.42 - 3.1 * np.log10(hBS)) * (np.log10(d3D) - 3) + 20 * np.log10(fc) - (3.2 * (np.log10(11.75 * hUT))**2 - 4.97)

        pathloss = PLos * plLos + (1 - PLos) * plNlos

        # 阴影
        rand = mulberry32(seed + int(d2D))
        shadow = sigmaSF * randn(rand)
        pathloss += shadow

        # 快衰落
        if fast == 'rician':
            s = np.sqrt(2 * Kfac / (Kfac + 1))
            sigma = np.sqrt(1 / (2 * (Kfac + 1)))
            real = s + sigma * randn(rand)
            imag = sigma * randn(rand)
            r = np.sqrt(real**2 + imag**2)
            pathloss -= 20 * np.log10(r)
        else:
            re = randn(rand) / np.sqrt(2)
            im = randn(rand) / np.sqrt(2)
            r = np.sqrt(re**2 + im**2)
            pathloss -= 20 * np.log10(r)
        return pathloss
    else:
        raise ValueError(f"Unknown model: {model}")