# -*- coding: utf-8 -*-
"""
Métriques réglementaires FRTB / Basel IV (BCBS d457, janvier 2023).

Ce module implémente les métriques de risque utilisées dans le cadre
de l'Internal Models Approach (IMA) du FRTB :
  - sVaR  : Stressed Value-at-Risk (fenêtre de 12 mois la pire)
  - ES    : Expected Shortfall à 97.5% (métrique FRTB officielle)
  - Capital requirement IMA
  - VaR multi-horizons par simulation GARCH exacte (sans règle sqrt(H))

Toutes les fonctions lisent les paramètres dans config['frtb'] et
exposent des valeurs par défaut prudentes.

References
----------
BCBS (2023). Minimum capital requirements for market risk (d457).
    Basel Committee on Banking Supervision, January 2023.
BCBS (2019). Explanatory note on the minimum capital requirements for
    market risk (d457 revision). January 2019.
Acerbi, C. & Szekely, B. (2014). Back-testing expected shortfall.
    Risk Magazine, November 2014.
Christoffersen, P.F. (2012). Elements of Financial Risk Management,
    2nd ed. Academic Press, Chapter 4.
"""
import logging
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

_log = logging.getLogger(__name__)

# ── Helpers internes ──────────────────────────────────────────────────────────

def _es_z_empirique(resid_std: np.ndarray, alpha: float) -> float:
    """
    ES des innovations standardisées par méthode empirique.

    ES_z(α) = E[z | z < quantile(z, 1-α)]

    Paramètre interne commun à expected_shortfall_frtb et acerbi_szekely.
    """
    arr = np.asarray(resid_std)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float('nan')
    q   = float(np.quantile(arr, 1.0 - alpha))
    sub = arr[arr < q]
    return float(np.mean(sub)) if len(sub) > 0 else q


def _es_z_analytique(dist_name: str, nu: float, alpha: float) -> float:
    """
    ES des innovations standardisées — formule analytique pour N et t.

    Pour les distributions non-standard (skewt, ged), retourne NaN
    (appelant doit fallback vers empirique).
    """
    p = 1.0 - alpha       # probabilité d'exceedance
    if dist_name == 'normal':
        q    = norm.ppf(p)
        return float(-norm.pdf(q) / p)  # négatif (perte)
    elif dist_name == 't':
        nu_  = nu if nu > 2.0 else 4.0
        q    = t_dist.ppf(p, df=nu_)
        pdf_q = t_dist.pdf(q, df=nu_)
        # Formule analytique Student-t (McNeil et al. 2015, eq. 2.71)
        return float(-(nu_ / (nu_ - 1.0)) * pdf_q / p
                     * (nu_ + q ** 2) / nu_)
    return float('nan')


# ── Stressed VaR ─────────────────────────────────────────────────────────────

def stressed_var(
    returns: pd.Series,
    garch_spec: dict,
    confidence: float = 0.99,
    window_months: int = 12,
    config: Optional[dict] = None,
) -> dict:
    """
    Stressed Value-at-Risk (sVaR) FRTB.

    Identifie la fenêtre glissante de ``window_months`` mois qui maximise
    la VaR conditionnelle GARCH. Correspond à la période de stress historique
    maximale, requise par le FRTB IMA (BCBS d457 §181-187).

    Parameters
    ----------
    returns : pd.Series
        Log-rendements complets (série avec index datetime).
    garch_spec : dict
        Spécification du modèle GARCH retenu :
        {'vol': str, 'p': int, 'o': int, 'q': int, 'dist': str}.
    confidence : float
        Niveau de confiance VaR (défaut 0.99 = FRTB standard).
    window_months : int
        Taille de la fenêtre de stress en mois calendaires (défaut 12).
    config : dict, optional
        Lit config['frtb'] pour les paramètres si fourni.

    Returns
    -------
    dict avec :
      'svar'         : float  sVaR au quantile confidence sur la fenêtre stress.
      'window_start' : date   Début de la fenêtre de stress optimale.
      'window_end'   : date   Fin de la fenêtre de stress optimale.
      'var_series'   : pd.Series  VaR glissante sur toutes les fenêtres.

    References
    ----------
    BCBS (2023). Minimum capital requirements for market risk, §181-187.
    """
    from arch import arch_model

    if config is not None:
        frtb_cfg      = config.get('frtb', {})
        confidence    = float(frtb_cfg.get('confidence_var', confidence))
        window_months = int(frtb_cfg.get('stressed_window_months', window_months))

    serie     = returns.dropna()
    n         = len(serie)
    window_d  = window_months * 21          # ~21 jours ouvrables/mois

    vol    = garch_spec.get('vol', 'Garch')
    p      = int(garch_spec.get('p', 1))
    o      = int(garch_spec.get('o', 1))
    q      = int(garch_spec.get('q', 1))
    dist   = garch_spec.get('dist', 'skewt')

    if window_d >= n:
        _log.warning('stressed_var: fenetre %d > n=%d, reduction a n//2', window_d, n)
        window_d = n // 2

    var_vals   = []
    win_starts = []

    try:
        # Fit once on full series for efficiency
        fit_full = arch_model(serie, vol=vol, p=p, o=o, q=q, dist=dist
                              ).fit(disp='off', show_warning=False)
        mu    = float(fit_full.params.get('mu', float(serie.mean())))
        nu    = float(fit_full.params.get('nu', 4.0))
        resid_std = np.asarray(
            fit_full.resid / fit_full.conditional_volatility
        )
        resid_std = resid_std[np.isfinite(resid_std)]
        q_z   = float(np.quantile(resid_std, 1.0 - confidence))
        vol_cv = np.asarray(fit_full.conditional_volatility)

        # Fenêtre glissante sur la vol conditionnelle (proxy pour sVaR)
        for start in range(0, n - window_d):
            end      = start + window_d
            vol_sub  = vol_cv[start:end]
            var_t    = mu + q_z * float(np.mean(vol_sub))
            var_vals.append(var_t)
            win_starts.append(serie.index[start])

        var_series = pd.Series(var_vals, index=win_starts, name='VaR_rolling')
        worst_idx  = int(np.argmin(var_vals))    # VaR la plus négative = pire
        svar_val   = float(var_vals[worst_idx])
        w_start    = win_starts[worst_idx]
        w_end      = serie.index[worst_idx + window_d - 1]

    except Exception as exc:
        _log.warning('stressed_var: estimation GARCH echouee: %s', exc)
        svar_val   = float('nan')
        w_start    = w_end = None
        var_series = pd.Series(dtype=float)

    return {
        'svar':         svar_val,
        'window_start': w_start,
        'window_end':   w_end,
        'var_series':   var_series,
        'confidence':   confidence,
        'window_months': window_months,
    }


# ── Expected Shortfall FRTB ───────────────────────────────────────────────────

def expected_shortfall_frtb(
    returns: pd.Series,
    garch_fit,
    confidence: float = 0.975,
    config: Optional[dict] = None,
) -> dict:
    """
    Expected Shortfall (ES) conditionnel à 97.5% — métrique FRTB officielle.

    La VaR à 99% actuelle est remplacée par l'ES à 97.5% depuis FRTB 2023.
    L'ES conditionnel est calculé comme :
        ES_t = mu + sigma_t · ES_z(α)
    où ES_z(α) = E[z | z < quantile_z(1-α)] sur les innovations standardisées.

    Pour les distributions analytiques (normal, t), la formule exacte est
    utilisée. Pour skewt/ged, estimation empirique sur les résidus standardisés.

    Parameters
    ----------
    returns : pd.Series
        Log-rendements.
    garch_fit
        Résultat arch (ARCHModelResult) : contient resid, conditional_volatility,
        params, model.distribution.
    confidence : float
        Niveau de confiance ES (défaut 0.975 — FRTB standard).
    config : dict, optional
        Lit config['frtb']['confidence_es'] si fourni.

    Returns
    -------
    dict avec :
      'es_series'  : pd.Series  ES conditionnel jour par jour.
      'es_mean'    : float      ES moyen sur la période.
      'es_current' : float      ES au dernier jour disponible.
      'confidence' : float      Niveau utilisé.
      'methode'    : str        'analytique' ou 'empirique'.

    References
    ----------
    BCBS (2023). d457 §189. ES à 97.5% sur horizon 10j.
    McNeil, A.J., Frey, R. & Embrechts, P. (2015). Quantitative Risk
        Management, 2nd ed., Princeton University Press, §2.3.
    """
    if config is not None:
        confidence = float(config.get('frtb', {}).get('confidence_es', confidence))

    serie    = returns.dropna()
    mu       = float(garch_fit.params.get('mu', float(serie.mean())))
    nu       = float(garch_fit.params.get('nu', 4.0))
    if math.isnan(nu) or nu <= 2.0:
        nu = 4.0
    dist_name = garch_fit.model.distribution.name.lower()

    # Innovations standardisées
    resid_std = np.asarray(garch_fit.resid / garch_fit.conditional_volatility)
    resid_std = resid_std[np.isfinite(resid_std)]

    # ES_z : analytique si possible, empirique sinon
    es_z_ana = _es_z_analytique(dist_name, nu, confidence)
    if math.isfinite(es_z_ana):
        es_z   = es_z_ana
        methode = 'analytique'
    else:
        es_z   = _es_z_empirique(resid_std, confidence)
        methode = 'empirique'

    vol_cv = garch_fit.conditional_volatility.reindex(serie.index)
    es_series = mu + es_z * vol_cv
    es_series.name = f'ES_{int(confidence*100)}pct'

    return {
        'es_series':   es_series,
        'es_mean':     float(es_series.mean()),
        'es_current':  float(es_series.dropna().iloc[-1]) if len(es_series.dropna()) > 0 else float('nan'),
        'confidence':  confidence,
        'methode':     methode,
        'es_z':        es_z,
    }


# ── Capital Requirement IMA ───────────────────────────────────────────────────

def capital_requirement_ima(
    var_99_10d: float,
    svar_99_10d: float,
    mult: float = 3.0,
    traffic_light: str = 'green',
    config: Optional[dict] = None,
) -> dict:
    """
    Capital réglementaire IMA selon FRTB (BCBS d457 §189).

    Formule :
        MRC = max(VaR_{t-1}, mult_k · avg60j(VaR)) + IRC
    Simplifié (sans IRC) :
        Capital = max(VaR_99_10d · mult, sVaR_99_10d · 3.0)

    Le multiplicateur mult_k est ajusté selon le feu tricolore Basel :
        Vert   : mult_k = 3.0
        Jaune  : mult_k = 3.4 → 3.85 selon le nombre de violations
        Rouge  : mult_k = 4.0

    Parameters
    ----------
    var_99_10d  : float  VaR 99% 10j courante.
    svar_99_10d : float  sVaR 99% 10j (pire fenêtre historique).
    mult        : float  Multiplicateur de base (défaut 3.0 = vert).
    traffic_light : str  'green' | 'yellow' | 'red' — feu tricolore.
    config : dict, optional
        Lit config['frtb'] pour les multiplicateurs.

    Returns
    -------
    dict avec :
      'capital'     : float  Capital IMA en unités de rendements (%).
      'composante_var'  : float
      'composante_svar' : float
      'multiplicateur'  : float
      'feu'         : str

    References
    ----------
    BCBS (2023). d457 §189 — Internal models approach capital requirement.
    """
    if config is not None:
        frtb_cfg = config.get('frtb', {})
        mult_green  = float(frtb_cfg.get('capital_multiplier_green',      3.0))
        mult_yellow = float(frtb_cfg.get('capital_multiplier_yellow_max', 3.85))
        mult_red    = float(frtb_cfg.get('capital_multiplier_red',        4.0))
        mult_map    = {'green': mult_green, 'yellow': mult_yellow, 'red': mult_red}
        mult        = mult_map.get(traffic_light.lower(), mult_green)

    comp_var  = abs(var_99_10d)  * mult
    comp_svar = abs(svar_99_10d) * 3.0      # sVaR toujours mult × 3
    capital   = max(comp_var, comp_svar)

    return {
        'capital':           capital,
        'composante_var':    comp_var,
        'composante_svar':   comp_svar,
        'multiplicateur':    mult,
        'feu':               traffic_light,
        'var_99_10d':        var_99_10d,
        'svar_99_10d':       svar_99_10d,
    }


# ── Multi-horizons GARCH exact (simulation Monte-Carlo) ──────────────────────

def multi_horizon_garch_exact(
    returns: pd.Series,
    garch_fit,
    horizons: list[int] | None = None,
    n_sim: int = 10_000,
    confidence: float = 0.99,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    VaR multi-horizons GARCH par simulation Monte-Carlo exacte.

    Remplace l'approximation sqrt(H) par des trajectoires GARCH complètes.
    Pour chaque horizon H, simule n_sim trajectoires H-step via arch et
    calcule le quantile empirique de la distribution H-step.

    P&L H-step = Σ_{h=1}^{H} r_t+h  (rendements cumulés, log-approximation).

    Parameters
    ----------
    returns   : pd.Series   Log-rendements.
    garch_fit               Résultat arch.
    horizons  : list[int]   Horizons en jours ouvrables (défaut [1,5,10,22]).
    n_sim     : int         Nombre de trajectoires Monte-Carlo (défaut 10 000).
    confidence: float       Niveau de confiance VaR (défaut 0.99).
    config    : dict        Lit config['frtb'] ou config['var'] si fourni.

    Returns
    -------
    pd.DataFrame colonnes : ['H', 'VaR_exact', 'VaR_sqrtH', 'ratio']
        Indexé par H (horizon).

    References
    ----------
    Christoffersen, P.F. (2012). Elements of Financial Risk Management,
        2nd ed. Academic Press, Chapter 4 — Multi-period VaR.
    BCBS (2023). d457 §§180-187 — Scaling horizons in IMA.
    """
    if horizons is None:
        if config is not None:
            horizons = config.get('var', {}).get('horizons', [1, 5, 10, 22])
        if horizons is None:
            horizons = [1, 5, 10, 22]
    if config is not None:
        n_sim = int(config.get('var', {}).get('n_simulations_horizons', n_sim))

    H_max = max(horizons)
    rows  = []

    try:
        # Simulation H_max pas en avant depuis le dernier état du modèle
        fc = garch_fit.forecast(horizon=H_max, method='simulation',
                                simulations=n_sim, reindex=False)
        # fc.simulations.values : shape (nobs, n_sim, H_max)
        # On prend le dernier état (dernière obs)
        sim_paths = fc.simulations.values[-1, :, :]   # (n_sim, H_max)

        # VaR à 1j (référence)
        var_1j = float(np.quantile(sim_paths[:, 0], 1.0 - confidence))

        for H in horizons:
            cum_returns = sim_paths[:, :H].sum(axis=1)   # P&L cumulé H jours
            var_H_exact = float(np.quantile(cum_returns, 1.0 - confidence))
            var_H_sqrt  = var_1j * math.sqrt(H)           # règle sqrt(H)
            ratio       = (var_H_exact / var_H_sqrt
                           if abs(var_H_sqrt) > 1e-10 else float('nan'))
            rows.append({
                'H':          H,
                'VaR_exact':  round(var_H_exact, 6),
                'VaR_sqrtH':  round(var_H_sqrt,  6),
                'ratio':      round(ratio, 4),
            })

    except Exception as exc:
        _log.warning('multi_horizon_garch_exact: %s', exc)
        for H in horizons:
            rows.append({'H': H, 'VaR_exact': float('nan'),
                         'VaR_sqrtH': float('nan'), 'ratio': float('nan')})

    df = pd.DataFrame(rows).set_index('H')
    return df
