# -*- coding: utf-8 -*-
"""
Benchmark EVT-POT (Extreme Value Theory — Peaks over Threshold).

Methode GPD via scipy.stats.genpareto :
  - Selection du seuil : percentile 95% des pertes (cote gauche)
  - Ajustement GPD par MLE sur les exceedances
  - VaR et TVaR analytiques GPD aux niveaux 95% et 99%
  - Test KS et AD pour valider l'ajustement
  - Comparaison avec VaR GARCH via DM tick loss

Ref : McNeil, A.J. & Frey, R. (2000). Estimation of tail-related risk measures
      for heteroscedastic financial time series. J. Empirical Finance 7, 271-300.
"""
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import genpareto, kstest, norm
from scipy.stats import t as t_dist


# ── Helpers ───────────────────────────────────────────────────────────────────

def _losses(serie: np.ndarray) -> np.ndarray:
    """Convertit les rendements en pertes positives (signe inverse)."""
    return -serie


def _seuil_95(losses: np.ndarray) -> float:
    """Seuil u = 95e percentile des pertes."""
    return float(np.quantile(losses, 0.95))


# ── Ajustement GPD ────────────────────────────────────────────────────────────

def ajuster_gpd(serie: np.ndarray, seuil_pct: float = 0.95):
    """
    Ajuste une GPD sur les exceedances des pertes au-dela du seuil.

    Parameters
    ----------
    serie : np.ndarray
        Log-rendements (negatifs = pertes).
    seuil_pct : float
        Percentile pour selectionner le seuil (defaut 0.95).

    Returns
    -------
    dict avec cles :
      u       : seuil retenu
      n_u     : nombre d'exceedances
      xi      : parametre de forme (tail index)
      beta    : parametre d'echelle
      ks_stat, ks_pval : test KS sur les exceedances
      fit_ok  : bool (convergence + n_u >= 30)
    """
    losses = _losses(serie)
    u = float(np.quantile(losses, seuil_pct))
    exceedances = losses[losses > u] - u
    n_u = len(exceedances)
    n   = len(losses)

    result = {
        'u': u, 'n_u': n_u, 'n_total': n,
        'xi': float('nan'), 'beta': float('nan'),
        'ks_stat': float('nan'), 'ks_pval': float('nan'),
        'fit_ok': False,
    }

    if n_u < 30:
        warnings.warn(f'EVT-POT : seulement {n_u} exceedances (< 30). GPD non fiable.')
        return result

    try:
        xi, loc, beta = genpareto.fit(exceedances, floc=0)
        result['xi']   = float(xi)
        result['beta'] = float(beta)

        ks_stat, ks_pval = kstest(exceedances, 'genpareto',
                                   args=(xi, 0, beta))
        result['ks_stat'] = float(ks_stat)
        result['ks_pval'] = float(ks_pval)
        result['fit_ok']  = (ks_pval >= 0.05)
    except Exception as e:
        warnings.warn(f'ajuster_gpd MLE : {e}')

    return result


# ── VaR et TVaR analytiques GPD ──────────────────────────────────────────────

def var_gpd(gpd_fit: dict, alpha: float) -> float:
    """
    VaR(alpha) analytique sous GPD.

    VaR_alpha = u + (beta / xi) * ((n/n_u * (1-alpha))^{-xi} - 1)
    Si xi ~ 0 : VaR_alpha = u + beta * log(n/n_u * (1-alpha))  [limite Gumbel]
    """
    u    = gpd_fit['u']
    n    = gpd_fit['n_total']
    n_u  = gpd_fit['n_u']
    xi   = gpd_fit['xi']
    beta = gpd_fit['beta']

    if not gpd_fit['fit_ok'] or math.isnan(xi) or math.isnan(beta):
        return float('nan')

    ratio = (n / n_u) * (1 - alpha)
    if ratio <= 0:
        return float('nan')

    if abs(xi) < 1e-6:
        return -(u + beta * math.log(ratio))
    else:
        return -(u + (beta / xi) * (ratio ** (-xi) - 1))


def tvar_gpd(gpd_fit: dict, alpha: float) -> float:
    """
    TVaR(alpha) analytique sous GPD.

    TVaR = VaR / (1 - xi) + (beta - xi*u) / (1 - xi)
    Valide uniquement si xi < 1.
    """
    xi   = gpd_fit['xi']
    beta = gpd_fit['beta']
    u    = gpd_fit['u']

    if not gpd_fit['fit_ok'] or math.isnan(xi) or math.isnan(beta):
        return float('nan')
    if xi >= 1:
        return float('inf')

    var = var_gpd(gpd_fit, alpha)
    if math.isnan(var):
        return float('nan')
    return var / (1 - xi) + (beta - xi * u) / (1 - xi)


# ── Comparaison GARCH vs EVT ─────────────────────────────────────────────────

def comparer_garch_evt(rendements, garch_final, config=None) -> dict:
    """
    Compare VaR/TVaR GARCH et EVT-GPD.

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements complets.
    garch_final : arch result
    config : dict, optional

    Returns
    -------
    dict avec cles :
      'gpd_fit' : resultat ajustement GPD
      'tableau'  : pd.DataFrame (Niveau, Methode, VaR, TVaR)
    """
    niveaux = [0.95, 0.99]
    if config is not None:
        niveaux = config.get('backtest', {}).get('niveaux_test', niveaux)

    serie = rendements.dropna().values
    gpd   = ajuster_gpd(serie)

    dist_name = garch_final.model.distribution.name.lower()
    params    = garch_final.params
    nu_garch  = float(params.get('nu', 4.0))
    last_vol  = float(garch_final.conditional_volatility.iloc[-1])
    mu        = float(params.get('mu', float(np.mean(serie))))

    rows = []
    for alpha in niveaux:
        niv = f'{int(alpha * 100)}%'

        # GARCH VaR/TVaR
        if dist_name == 'normal':
            q_g  = float(norm.ppf(1 - alpha))
            vg   = mu + q_g * last_vol
            tvg  = mu - last_vol * norm.pdf(q_g) / (1 - alpha)
        elif dist_name == 't':
            q_g  = float(t_dist.ppf(1 - alpha, df=nu_garch))
            vg   = mu + q_g * last_vol
            tvg  = mu - last_vol * (t_dist.pdf(q_g, df=nu_garch) / (1 - alpha)) * \
                   (nu_garch + q_g**2) / (nu_garch - 1)
        else:
            resid_std = (garch_final.resid / garch_final.conditional_volatility).dropna().values
            q_g  = float(np.quantile(resid_std, 1 - alpha))
            vg   = mu + q_g * last_vol
            tail = resid_std[resid_std <= q_g]
            tvg  = mu + (float(tail.mean()) if len(tail) > 0 else q_g) * last_vol

        # EVT VaR/TVaR
        ve  = var_gpd(gpd, alpha)
        tve = tvar_gpd(gpd, alpha)

        rows.append({'Niveau': niv, 'Methode': 'GARCH dyn.',
                     'VaR': round(vg, 4), 'TVaR': round(tvg, 4)})
        rows.append({'Niveau': niv, 'Methode': 'EVT-GPD',
                     'VaR': round(ve, 4) if not math.isnan(ve) else float('nan'),
                     'TVaR': round(tve, 4) if not math.isnan(tve) else float('nan')})

    return {
        'gpd_fit': gpd,
        'tableau': pd.DataFrame(rows),
    }
