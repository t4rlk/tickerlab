# -*- coding: utf-8 -*-
"""
Tests de robustesse avances pour la VaR GARCH.

Berkowitz (2001)           — PIT / LR test densite predictive
Engle-Manganelli (2004)    — Dynamic Quantile (DQ) test
Diebold-Mariano (1995)     — comparaison methodes via tick loss Giacomini-Komunjer (2005)
Sign Bias (Engle-Ng 1993)  — re-exporte sur le modele retenu
"""
import math
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm, chi2, t as t_dist


# ── Helpers internes ──────────────────────────────────────────────────────────

def _oos_data(rendements, garch_final, config):
    """Retourne (serie_train, y_oos, vol_oos) alignees."""
    split_ratio = config.get('backtest', {}).get('split_ratio', 0.70)
    serie = rendements.dropna()
    T = len(serie)
    T_train = int(np.floor(split_ratio * T))
    serie_vals = serie.values
    y_oos  = serie_vals[T_train:]
    vol_all = garch_final.conditional_volatility.values
    # Aligner : vol_all peut avoir une longueur differente de T si dropna differe
    start = max(0, len(vol_all) - (T - T_train))
    vol_oos = vol_all[start:]
    T_oos = min(len(y_oos), len(vol_oos))
    return serie_vals[:T_train], y_oos[:T_oos], vol_oos[:T_oos]


def _q_z(garch_final, alpha):
    """Quantile z de la distribution estimee pour le niveau alpha."""
    dist_name = garch_final.model.distribution.name.lower()
    nu = float(garch_final.params.get('nu', 4.0))
    if math.isnan(nu) or nu <= 2:
        nu = 4.0
    if dist_name == 'normal':
        return float(norm.ppf(1 - alpha))
    elif dist_name == 't':
        return float(t_dist.ppf(1 - alpha, df=nu))
    else:
        resid_std = (garch_final.resid / garch_final.conditional_volatility).dropna().values
        return float(np.quantile(resid_std, 1 - alpha))


def _pit(z_oos, garch_final):
    """PIT u_t = F(z_t) selon la distribution du modele."""
    dist_name = garch_final.model.distribution.name.lower()
    nu = float(garch_final.params.get('nu', 4.0))
    if math.isnan(nu) or nu <= 2:
        nu = 4.0
    if dist_name == 'normal':
        u = norm.cdf(z_oos)
    elif dist_name == 't':
        u = t_dist.cdf(z_oos, df=nu)
    else:
        resid_std = np.sort(
            (garch_final.resid / garch_final.conditional_volatility).dropna().values
        )
        idx = np.searchsorted(resid_std, z_oos, side='right')
        u = idx / len(resid_std)
    return np.clip(u, 1e-8, 1 - 1e-8)


# ── Berkowitz (2001) ──────────────────────────────────────────────────────────

def berkowitz_pit(rendements, garch_final, config) -> dict:
    """
    Test LR de Berkowitz (2001) sur la densite predictive GARCH.

    Transforme les rendements OOS en PIT u_t = F(y_t | Omega_{t-1}),
    puis en normales standard xi_t = Phi^{-1}(u_t).
    H0 : xi_t ~ iid N(0,1) — mu=0, sigma^2=1, rho=0.
    LR_joint ~ chi2(3).

    Ref : Berkowitz, J. (2001). Testing density forecasts, with applications
    to risk management. JBES 19(4), 465-474.
    """
    nan_r = {k: float('nan') for k in ('LR', 'p_value', 'mu_hat', 'sig2_hat', 'rho_hat')}
    nan_r['verdict'] = 'N/A'; nan_r['T_oos'] = 0
    try:
        serie_train, y_oos, vol_oos = _oos_data(rendements, garch_final, config)
        mu_fit = float(garch_final.params.get('mu', float(np.mean(serie_train))))
        z_oos  = (y_oos - mu_fit) / vol_oos
        xi     = norm.ppf(_pit(z_oos, garch_final))
        n      = len(xi)

        # H0 : logL sous N(0,1)
        L0 = float(-0.5 * n * math.log(2 * math.pi) - 0.5 * np.sum(xi**2))

        # H1 : xi_t = mu + rho * xi_{t-1} + eps, eps ~ N(0, sigma2)
        xi1 = xi[:-1]; xi2 = xi[1:]; m = len(xi2)
        mu2, mu1 = xi2.mean(), xi1.mean()
        var1 = np.sum((xi1 - mu1) ** 2)
        rho_hat = float(np.sum((xi2 - mu2) * (xi1 - mu1)) / var1) if var1 > 1e-12 else 0.0
        mu_hat  = float(mu2 - rho_hat * mu1)
        resid1  = xi2 - mu_hat - rho_hat * xi1
        sig2    = max(float(np.var(resid1)), 1e-12)
        L1 = float(-0.5 * m * math.log(2 * math.pi * sig2)
                   - 0.5 * np.sum(resid1**2) / sig2)

        LR    = max(2.0 * (L1 - L0), 0.0)
        p_val = float(1 - chi2.cdf(LR, df=3))
        return {
            'LR': LR, 'p_value': p_val, 'T_oos': n,
            'mu_hat': mu_hat, 'sig2_hat': sig2, 'rho_hat': rho_hat,
            'verdict': 'OK' if p_val >= 0.05 else 'REJET',
        }
    except Exception as e:
        warnings.warn(f'berkowitz_pit: {e}')
        return nan_r


# ── DQ test (Engle-Manganelli 2004) ──────────────────────────────────────────

def dq_test(rendements, garch_final, config, n_lags: int = 4) -> dict:
    """
    Dynamic Quantile (DQ) test d'Engle & Manganelli (2004, CAViaR).

    Hit_t = I(y_t < VaR_t) - (1-alpha)
    Regression : Hit_t = b0 + b1*Hit_{t-1} + ... + bK*Hit_{t-K} + b_{K+1}*VaR_t
    H0 jointe : tous les coefficients nuls. Wald ~ chi2(K+2).

    Ref : Engle, R.F. & Manganelli, S. (2004). CAViaR. JBES 22(4), 367-381.
    """
    niveaux = config.get('backtest', {}).get('niveaux_test', [0.95, 0.99])
    results = {}
    try:
        serie_train, y_oos, vol_oos = _oos_data(rendements, garch_final, config)
        mu_fit = float(garch_final.params.get('mu', float(np.mean(serie_train))))

        for alpha in niveaux:
            niv = f'{int(alpha * 100)}%'
            try:
                var_oos = mu_fit + _q_z(garch_final, alpha) * vol_oos
                Hit     = (y_oos < var_oos).astype(float) - (1 - alpha)
                T_oos   = len(Hit)
                s       = n_lags
                if T_oos <= s + 10:
                    results[niv] = {'DQ': float('nan'), 'p_value': float('nan'), 'verdict': 'N/A'}
                    continue
                H = Hit[s:]
                X = np.column_stack([
                    np.ones(T_oos - s),
                    *[Hit[s - j - 1: T_oos - j - 1] for j in range(n_lags)],
                    var_oos[s:],
                ])
                XtX_inv = np.linalg.pinv(X.T @ X)
                dq_stat = float(max((H @ X @ XtX_inv @ X.T @ H) / ((1 - alpha) * alpha), 0))
                p_val   = float(1 - chi2.cdf(dq_stat, df=X.shape[1]))
                results[niv] = {
                    'DQ': dq_stat, 'p_value': p_val,
                    'verdict': 'OK' if p_val >= 0.05 else 'REJET',
                }
            except Exception as e:
                warnings.warn(f'dq_test [{niv}]: {e}')
                results[niv] = {'DQ': float('nan'), 'p_value': float('nan'), 'verdict': 'N/A'}
    except Exception as e:
        warnings.warn(f'dq_test: {e}')
    return results


# ── Diebold-Mariano tick loss (Giacomini-Komunjer 2005) ──────────────────────

def diebold_mariano_var(rendements, garch_final, config) -> pd.DataFrame:
    """
    Test Diebold-Mariano (1995) avec la tick loss de Giacomini & Komunjer (2005).

    rho_alpha(u) = u * (alpha - I(u < 0))
    DM_ij = sqrt(T) * mean(d) / std(d) -> N(0,1) sous H0 precision egale.
    d_t = L(m1_t) - L(m2_t)

    Ref : Giacomini, R. & Komunjer, I. (2005). Evaluation and Combination of
    Conditional Quantile Forecasts. JBES 23(4), 416-431.
    """
    niveaux = config.get('backtest', {}).get('niveaux_test', [0.95, 0.99])
    rows = []
    try:
        serie_train, y_oos, vol_oos = _oos_data(rendements, garch_final, config)
        mu_fit  = float(garch_final.params.get('mu', float(np.mean(serie_train))))
        mu_tr   = float(np.mean(serie_train))
        sig_tr  = float(np.std(serie_train))
        nu_t, mu_t, sig_t = t_dist.fit(serie_train, floc=mu_tr)

        def tick(y, q, a):
            u = y - q
            return u * (a - (u < 0).astype(float))

        for alpha in niveaux:
            niv = f'{int(alpha * 100)}%'
            qz  = _q_z(garch_final, alpha)
            T   = len(y_oos)

            methodes = {
                'Historique': np.full(T, float(np.quantile(serie_train, 1 - alpha))),
                'Normale':    np.full(T, float(norm.ppf(1 - alpha, loc=mu_tr, scale=sig_tr))),
                'Student':    np.full(T, float(t_dist.ppf(1 - alpha, df=nu_t, loc=mu_t, scale=sig_t))),
                'GARCH dyn.': mu_fit + qz * vol_oos,
            }
            noms = list(methodes.keys())
            for i, m1 in enumerate(noms):
                for m2 in noms[i + 1:]:
                    n = min(T, len(methodes[m1]), len(methodes[m2]))
                    d = (tick(y_oos[:n], methodes[m1][:n], 1 - alpha)
                         - tick(y_oos[:n], methodes[m2][:n], 1 - alpha))
                    sd = float(np.std(d, ddof=1))
                    dm = float(np.mean(d)) * math.sqrt(n) / max(sd, 1e-12)
                    pv = float(2 * (1 - norm.cdf(abs(dm))))
                    rows.append({
                        'Niveau': niv, 'M1': m1, 'M2': m2,
                        'DM': round(dm, 3), 'p_value': round(pv, 4),
                        'Favori': m1 if dm > 0 else m2,
                        'Sig.': '***' if pv < 0.01 else '**' if pv < 0.05 else '*' if pv < 0.10 else '',
                    })
    except Exception as e:
        warnings.warn(f'diebold_mariano_var: {e}')
    return pd.DataFrame(rows)


# ── Wrapper principal ─────────────────────────────────────────────────────────

def tester_robustesse(rendements, garch_final, config) -> dict:
    """
    Point d'entree — execute les 4 tests et retourne un dict de resultats.

    Returns
    -------
    dict avec cles :
      'berkowitz'       : dict (LR, p_value, verdict, ...)
      'dq'              : dict par niveau (DQ, p_value, verdict)
      'diebold_mariano' : pd.DataFrame des paires
      'sign_bias'       : dict (sign_stat, sign_pval, joint_f, joint_pval, ...)
    """
    berk = berkowitz_pit(rendements, garch_final, config)
    dq   = dq_test(rendements, garch_final, config)
    dm   = diebold_mariano_var(rendements, garch_final, config)
    try:
        from tickerlab.core.rapport._stats import sign_bias_test
        resid_std = (garch_final.resid / garch_final.conditional_volatility).dropna().values
        sbt = sign_bias_test(resid_std)
    except Exception:
        sbt = {}
    return {'berkowitz': berk, 'dq': dq, 'diebold_mariano': dm, 'sign_bias': sbt}
