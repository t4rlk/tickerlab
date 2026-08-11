# -*- coding: utf-8 -*-
"""
Monitoring dynamique de la VaR.

Deux indicateurs :
  1. CUSUM standardise des violations (Hawkins & Olwell 1998, chap. 2).
     Increment standardise : (Hit_t - p) / sigma_hit  ou sigma_hit = sqrt(p(1-p)).
     S_t = max(0, S_{t-1} + (Hit_t - p)/sigma_hit - k_std),
       k_std = delta_detect / 2  (reference shift, Page 1954).
     Alerte si S_t > h_std (seuil critique standard, defaut 5.0).
     Ref : Hawkins, D.M. & Olwell, D.H. (1998). Cumulative Sum Charts and
           Charting for Quality Improvement. Springer, p. 27.

  2. Ratio glissant d'exceptions : taux observe sur fenetre mobile de 250 j.
     Alerte si ratio > 2 * (1 - alpha).
"""
import numpy as np
import pandas as pd


def _build_hits(rendements: pd.Series, garch_final, alpha: float,
                config: dict) -> pd.Series:
    """
    Construit la serie Hit_t = I(y_t < VaR_t) sur la periode complete.

    Hit = 1 si violation, 0 sinon.
    """
    from scipy.stats import norm
    from scipy.stats import t as t_dist

    serie = rendements.dropna()
    vol   = garch_final.conditional_volatility
    vol   = vol.reindex(serie.index).dropna()
    serie = serie.reindex(vol.index)

    mu        = float(garch_final.params.get('mu', float(serie.mean())))
    dist_name = garch_final.model.distribution.name.lower()
    nu        = float(garch_final.params.get('nu', 4.0))

    if dist_name == 'normal':
        q_z = float(norm.ppf(1 - alpha))
    elif dist_name == 't':
        q_z = float(t_dist.ppf(1 - alpha, df=nu))
    else:
        resid_std = (garch_final.resid / garch_final.conditional_volatility).dropna().values
        q_z = float(np.quantile(resid_std, 1 - alpha))

    var_t = mu + q_z * vol.values
    hits  = (serie.values < var_t).astype(float)
    return pd.Series(hits, index=serie.index, name=f'Hit_{int(alpha*100)}pct')


def cusum_violations(rendements: pd.Series, garch_final, config: dict,
                     alpha: float = 0.99) -> dict:
    """
    CUSUM standardise des violations VaR (Hawkins & Olwell 1998).

    L'increment est standardise par sigma_hit = sqrt(p*(1-p)) de sorte que
    chaque terme (Hit_t - p)/sigma_hit est comparable a une variable N(0,1).
    Le seuil critique h_std = 5.0 correspond aux tables de Page (1954) pour
    un ARL in-control ~ 370 observations (standard industrie).

    Parameters
    ----------
    rendements : pd.Series
    garch_final : arch result
    config : dict
        Lit monitoring.cusum_delta_detect (defaut 1.0) et
        monitoring.cusum_h_std (defaut 5.0).
    alpha : float
        Niveau de confiance (defaut 0.99).

    Returns
    -------
    dict avec :
      'cusum'         : pd.Series CUSUM standardise S_t
      'alertes'       : pd.DatetimeIndex dates ou S_t > h_std
      'n_alertes'     : int
      'h'             : float seuil critique h_std (pour compatibilite _sections.py)
      'k'             : float seuil de reference k_std
      'delta_detect'  : float magnitude de shift en sigma
    """
    mon_cfg      = config.get('monitoring', {})
    delta_detect = float(mon_cfg.get('cusum_delta_detect', 1.0))
    h_std        = float(mon_cfg.get('cusum_h_std', 5.0))

    hits      = _build_hits(rendements, garch_final, alpha, config)
    p         = 1 - alpha
    sigma_hit = float(np.sqrt(p * (1 - p)))
    # k_std = delta/2 : point de reference optimal Page (1954) pour detecter
    # un shift de delta sigma dans la moyenne des hits.
    k_std     = 0.5 * delta_detect

    cusum = np.zeros(len(hits))
    for i in range(1, len(hits)):
        increment  = (float(hits.iloc[i]) - p) / sigma_hit - k_std
        cusum[i]   = max(0.0, cusum[i - 1] + increment)

    cusum_s = pd.Series(cusum, index=hits.index, name='CUSUM')
    alertes = hits.index[cusum > h_std]

    return {
        'cusum':        cusum_s,
        'alertes':      alertes,
        'n_alertes':    len(alertes),
        'h':            h_std,    # compatible avec _sections.py (cu.get('h'))
        'k':            k_std,
        'delta_detect': delta_detect,
    }


def rolling_exception_ratio(rendements: pd.Series, garch_final, config: dict,
                             alpha: float = 0.99, window: int = 250) -> dict:
    """
    Ratio glissant d'exceptions sur fenetre de `window` jours.

    Parameters
    ----------
    window : int
        Taille de la fenetre glissante en jours (defaut 250).

    Returns
    -------
    dict avec :
      'ratio'     : pd.Series ratio glissant
      'seuil'     : float seuil d'alerte = 2 * (1-alpha)
      'alertes'   : pd.DatetimeIndex
      'n_alertes' : int
    """
    mon_cfg  = config.get('monitoring', {})
    window   = int(mon_cfg.get('rolling_window', window))
    mult_al  = float(mon_cfg.get('ratio_alerte_mult', 2.0))

    hits   = _build_hits(rendements, garch_final, alpha, config)
    ratio  = hits.rolling(window=window, min_periods=window // 2).mean()
    seuil  = mult_al * (1 - alpha)
    alertes = ratio.index[ratio > seuil]

    return {
        'ratio':    ratio.rename('Ratio glissant'),
        'seuil':    seuil,
        'alertes':  alertes,
        'n_alertes': len(alertes),
        'window':   window,
    }


def monitorer_var(rendements: pd.Series, garch_final, config: dict) -> dict:
    """
    Point d'entree : execute le monitoring pour 95% et 99%.

    Returns
    -------
    dict avec cles '95%' et '99%', chacun contenant :
      'cusum'  : resultat cusum_violations()
      'ratio'  : resultat rolling_exception_ratio()
    """
    results = {}
    for alpha in [0.95, 0.99]:
        niv = f'{int(alpha * 100)}%'
        try:
            cu = cusum_violations(rendements, garch_final, config, alpha)
        except Exception as e:
            import warnings
            warnings.warn(f'monitoring CUSUM {niv} : {e}')
            cu = {}
        try:
            rr = rolling_exception_ratio(rendements, garch_final, config, alpha)
        except Exception as e:
            import warnings
            warnings.warn(f'monitoring rolling {niv} : {e}')
            rr = {}
        results[niv] = {'cusum': cu, 'ratio': rr}
    return results
