# -*- coding: utf-8 -*-
"""
Bloc D — Calculs statistiques et tests econometriques pour le PDF.

Toutes les fonctions retournent des dicts (pas de Flowables) :
les sections les consomment et formatent via _helpers._tableau_eviews().
"""
import logging
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_log = logging.getLogger('tickerlab.rapport._stats')

warnings.filterwarnings('ignore')


# ── Statistiques descriptives ────────────────────────────────────────────────

def stats_desc(serie) -> dict:
    """
    Statistiques descriptives completes (format EViews).

    Returns
    -------
    dict
        Cles : n, mean, median, max, min, std, skew, kurt_exc, jb_stat, jb_pval.
        kurt_exc = kurtosis de Fisher (excess kurtosis, =0 pour normale).
    """
    s = pd.Series(serie).dropna().values
    n = len(s)
    if n < 4:
        return {k: float('nan') for k in
                ('n', 'mean', 'median', 'max', 'min', 'std',
                 'skew', 'kurt_exc', 'jb_stat', 'jb_pval')}

    jb_stat, jb_pval = sp_stats.jarque_bera(s)
    return {
        'n':        n,
        'mean':     float(np.mean(s)),
        'median':   float(np.median(s)),
        'max':      float(np.max(s)),
        'min':      float(np.min(s)),
        'std':      float(np.std(s, ddof=1)),
        'skew':     float(sp_stats.skew(s)),
        'kurt_exc': float(sp_stats.kurtosis(s, fisher=True)),   # excess
        'jb_stat':  float(jb_stat),
        'jb_pval':  float(jb_pval),
    }


# ── Tests de stationnarite ────────────────────────────────────────────────────

def _adf_une_spec(serie, regression: str, max_lag: int) -> dict:
    """ADF pour une specification ('n', 'c', 'ct')."""
    from statsmodels.tsa.stattools import adfuller
    try:
        res = adfuller(serie.dropna(), maxlag=max_lag,
                       autolag='BIC', regression=regression)
        adf_stat, pval, used_lag, _, crit, _ = res
        return {
            'stat':     float(adf_stat),
            'pval':     float(pval),
            'lags':     int(used_lag),
            'crit_1':   float(crit['1%']),
            'crit_5':   float(crit['5%']),
            'crit_10':  float(crit['10%']),
        }
    except Exception as e:
        _log.warning('[WARN] ADF (%s) : %s', regression, e)
        return {k: float('nan') for k in ('stat', 'pval', 'lags', 'crit_1', 'crit_5', 'crit_10')}


def adf_complet(serie) -> dict:
    """
    ADF test sur 3 specifications : 'n' (aucune), 'c' (constante), 'ct' (const+trend).

    Lag max auto : floor(12 * (n/100)^0.25)  — Schwert (1989).
    Selection automatique : BIC (= SIC dans EViews).

    Returns
    -------
    dict avec cles 'n', 'c', 'ct' — chacun un dict de stats.
    """
    s = pd.Series(serie).dropna()
    n = len(s)
    max_lag = int(np.floor(12 * (n / 100) ** 0.25))
    return {spec: _adf_une_spec(s, spec, max_lag) for spec in ('n', 'c', 'ct')}


def _pp_une_spec(serie, trend: str) -> dict:
    """Phillips-Perron pour une specification ('n', 'c', 'ct')."""
    try:
        from arch.unitroot import PhillipsPerron
        pp = PhillipsPerron(serie.dropna(), trend=trend)
        crit = pp.critical_values
        return {
            'stat':    float(pp.stat),
            'pval':    float(pp.pvalue),
            'lags':    int(pp.lags),
            'crit_1':  float(crit.get('1%', float('nan'))),
            'crit_5':  float(crit.get('5%', float('nan'))),
            'crit_10': float(crit.get('10%', float('nan'))),
        }
    except Exception as e:
        _log.warning('[WARN] PP (%s) : %s', trend, e)
        return {k: float('nan') for k in ('stat', 'pval', 'lags', 'crit_1', 'crit_5', 'crit_10')}


def pp_complet(serie) -> dict:
    """
    Phillips-Perron (1988) sur 3 specifications : 'n', 'c', 'ct'.

    Returns
    -------
    dict avec cles 'n', 'c', 'ct'.
    """
    s = pd.Series(serie).dropna()
    return {spec: _pp_une_spec(s, spec) for spec in ('n', 'c', 'ct')}


def _kpss_une_spec(serie, regression: str) -> dict:
    """KPSS pour une specification ('c' ou 'ct')."""
    from statsmodels.tsa.stattools import kpss
    try:
        stat, pval, lags, crit = kpss(serie.dropna(), regression=regression, nlags='auto')
        return {
            'stat':    float(stat),
            'pval':    float(pval),
            'lags':    int(lags),
            'crit_1':  float(crit.get('1%', float('nan'))),
            'crit_5':  float(crit.get('5%', float('nan'))),
            'crit_10': float(crit.get('10%', float('nan'))),
        }
    except Exception as e:
        _log.warning('[WARN] KPSS (%s) : %s', regression, e)
        return {k: float('nan') for k in ('stat', 'pval', 'lags', 'crit_1', 'crit_5', 'crit_10')}


def kpss_complet(serie) -> dict:
    """
    KPSS (1992) sur 2 specifications : 'c' (niveau) et 'ct' (tendance).
    H0 KPSS : serie stationnaire (inverse d'ADF/PP).

    Returns
    -------
    dict avec cles 'c', 'ct'.
    """
    s = pd.Series(serie).dropna()
    return {spec: _kpss_une_spec(s, spec) for spec in ('c', 'ct')}


# ── Test ARCH-LM (Engle 1982) ─────────────────────────────────────────────────

def arch_lm(resid, lags=None) -> list:
    """
    Test ARCH-LM d'Engle (1982) sur les residus ARIMA.

    Parameters
    ----------
    resid : array-like
        Residus de l'ARIMA (fit.resid).
    lags : list of int, optional
        Ordres testes. Defaut : [1, 4, 8, 12].

    Returns
    -------
    list of dict
        Chaque dict : {lag, f_stat, f_pval, lm_stat, lm_pval}.
    """
    from statsmodels.stats.diagnostic import het_arch
    if lags is None:
        lags = [1, 4, 8, 12]
    r = np.asarray(resid, dtype=float)
    r = r[~np.isnan(r)]
    resultats = []
    for lag in lags:
        try:
            lm_stat, lm_pval, f_stat, f_pval = het_arch(r, nlags=lag)
            resultats.append({
                'lag':     lag,
                'f_stat':  float(f_stat),
                'f_pval':  float(f_pval),
                'lm_stat': float(lm_stat),
                'lm_pval': float(lm_pval),
            })
        except Exception as e:
            _log.warning('[WARN] ARCH-LM (lag=%d) : %s', lag, e)
            resultats.append({
                'lag': lag,
                'f_stat': float('nan'), 'f_pval': float('nan'),
                'lm_stat': float('nan'), 'lm_pval': float('nan'),
            })
    return resultats


# ── Test de biais de signe Engle-Ng (1993) ───────────────────────────────────

def sign_bias_test(z) -> dict:
    """
    Test de biais de signe d'Engle & Ng (1993) sur les residus standardises GARCH.

    Quatre tests :
    - Biais de signe   : z²_t ~ a + b*S-_{t-1}
    - Biais taille neg : z²_t ~ a + b*S-_{t-1}*eps_{t-1}
    - Biais taille pos : z²_t ~ a + b*S+_{t-1}*eps_{t-1}
    - Joint (F-test)   : z²_t ~ a + b1*S- + b2*S-*eps + b3*S+*eps

    Parameters
    ----------
    z : array-like
        Residus standardises z_t = eps_t / sigma_t.

    Returns
    -------
    dict
        Cles : sign_stat, sign_pval, neg_stat, neg_pval, pos_stat, pos_pval,
               joint_f, joint_pval.
    """
    import statsmodels.api as sm

    z = np.asarray(z, dtype=float)
    z = z[~np.isnan(z)]
    T = len(z)
    if T < 10:
        return {k: float('nan') for k in
                ('sign_stat', 'sign_pval', 'neg_stat', 'neg_pval',
                 'pos_stat', 'pos_pval', 'joint_f', 'joint_pval')}

    z_sq   = z[1:] ** 2         # variable dependante : z²_t
    eps    = z[:-1]              # innovations laggees : eps_{t-1}
    S_neg  = (eps < 0).astype(float)
    S_pos  = 1.0 - S_neg
    const  = np.ones(T - 1)

    def _ols_tstat(X, y):
        """Retourne t-stat et p-val du dernier coefficient."""
        try:
            res = sm.OLS(y, X).fit()
            idx = X.shape[1] - 1
            return float(res.tvalues[idx]), float(res.pvalues[idx])
        except Exception:
            return float('nan'), float('nan')

    # Tests individuels (t-stat sur le regresseur d'interet)
    s_t,  s_p  = _ols_tstat(np.column_stack([const, S_neg]),          z_sq)
    sn_t, sn_p = _ols_tstat(np.column_stack([const, S_neg * eps]),    z_sq)
    sp_t, sp_p = _ols_tstat(np.column_stack([const, S_pos * eps]),    z_sq)

    # Test conjoint (F-test sur b1, b2, b3)
    try:
        X_joint = np.column_stack([const, S_neg, S_neg * eps, S_pos * eps])
        res_j   = sm.OLS(z_sq, X_joint).fit()
        f_joint = res_j.f_test('x1=0, x2=0, x3=0')
        jf, jp  = float(f_joint.fvalue), float(f_joint.pvalue)
    except Exception:
        jf, jp = float('nan'), float('nan')

    return {
        'sign_stat':  s_t,  'sign_pval':  s_p,
        'neg_stat':   sn_t, 'neg_pval':   sn_p,
        'pos_stat':   sp_t, 'pos_pval':   sp_p,
        'joint_f':    jf,   'joint_pval': jp,
    }


# ── Test McNeil-Frey (2000) pour l'ES ────────────────────────────────────────

def mcneil_frey_test(
    r,
    var_dyn,
    es_dyn,
    sigma: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Test McNeil & Frey (2000) de validite de l'Expected Shortfall.

    z_t = (r_t - ES_t) / sigma_t   pour t : r_t < VaR_t

    Sous H0 (modele correct) : E[z_t] = 0.
    P-value bilaterale par bootstrap.

    Reference : McNeil & Frey (2000), JEF 7(3-4), eq. (10).
    # TODO: Acerbi-Szekely (2014) Test Z2 en v2

    Parameters
    ----------
    r : array-like
        Log-rendements observes (OOS).
    var_dyn : array-like
        VaR dynamique (meme longueur que r).
    es_dyn : array-like
        Expected Shortfall dynamique (meme longueur que r).
    sigma : array-like, optional
        Volatilite conditionnelle sigma_t. Si None, utilise std(r_exc).
    n_boot : int
        Nombre de replications bootstrap (defaut 1000).
    seed : int
        Graine numpy (defaut 42, pour reproductibilite des resultats).

    Returns
    -------
    dict
        Cles : z_bar, n_exc, p_value, verdict.
    """
    rng = np.random.default_rng(seed)

    r_arr   = np.asarray(r,       dtype=float)
    var_arr = np.asarray(var_dyn, dtype=float)
    es_arr  = np.asarray(es_dyn,  dtype=float)

    # Aligner les longueurs (prise en compte du split OOS eventuel)
    n = min(len(r_arr), len(var_arr), len(es_arr))
    r_arr   = r_arr[:n]
    var_arr = var_arr[:n]
    es_arr  = es_arr[:n]

    mask  = r_arr < var_arr   # depassements VaR
    r_exc = r_arr[mask]
    es_exc = es_arr[mask]
    n_exc = int(mask.sum())

    if n_exc < 5:
        return {
            'z_bar':   float('nan'),
            'n_exc':   n_exc,
            'p_value': float('nan'),
            'verdict': 'N/A (trop peu de depassements)',
        }

    if sigma is not None:
        sig_arr = np.asarray(sigma, dtype=float)[:n]
        sig_exc = sig_arr[mask]
        sig_exc = np.where(sig_exc > 0, sig_exc, np.std(r_exc))
    else:
        sig_exc = np.full(n_exc, np.std(r_exc) if np.std(r_exc) > 0 else 1.0)

    z      = (r_exc - es_exc) / sig_exc
    z_bar  = float(np.mean(z))

    # Bootstrap bilateral sous H0 : z ~ distribution centree sur 0
    z_c    = z - z_bar                    # centrer pour simuler H0
    boot   = np.array([
        np.mean(rng.choice(z_c, size=n_exc, replace=True))
        for _ in range(n_boot)
    ])
    p_value = float(np.mean(np.abs(boot) >= abs(z_bar)))

    return {
        'z_bar':   z_bar,
        'n_exc':   n_exc,
        'p_value': p_value,
        'verdict': 'OK' if p_value > 0.05 else 'Rejet H0 (ES sous-estime)',
    }


# ── Persistance GARCH (dispatcher par modele) ─────────────────────────────────

def _kappa_aparch(delta: float, gamma_asym: float) -> float:
    """
    E[(|z| - gamma*z)^delta] sous z ~ N(0,1).

    Derivation analytique (symetrie de N(0,1)) :
        E[(|z| - gamma*z)^delta]
        = [(1-gamma)^delta + (1+gamma)^delta]
          * 2^(delta/2 - 1) * Gamma((delta+1)/2) / sqrt(pi)

    Reference : Ding, Granger & Engle (1993), JEF 5(1), eq. 3.7.
    Valide pour delta > 0 et gamma in (-1, 1).
    """
    from scipy.special import gamma as _sp_gamma
    if math.isnan(delta) or math.isnan(gamma_asym):
        return float('nan')
    d = float(delta)
    g = max(-1.0 + 1e-9, min(1.0 - 1e-9, float(gamma_asym)))   # guard [-1,1]
    try:
        kappa = ((1 - g) ** d + (1 + g) ** d) * 2 ** (d / 2 - 1) * _sp_gamma((d + 1) / 2) / math.sqrt(math.pi)
        return float(kappa)
    except Exception:
        return float('nan')


def persistance_garch(modele: str, params) -> float:
    """
    Calcule la persistance du modele GARCH selon sa famille.

    Somme tous les lags (multi-lag correct) :
    - GARCH(p,q)     : sum_i alpha[i] + sum_j beta[j]  (Bollerslev 1986)
    - GJR-GARCH(p,q) : sum_i alpha[i] + sum_j beta[j] + 0.5*sum_k gamma[k]
    - EGARCH(p,o,q)  : sum_j beta[j]  (Nelson 1991, eq. 10)
    - TGARCH/APARCH  : sum_j beta[j] + sum_i alpha[i]*kappa(delta, gamma[i])
                       kappa = E[(|z|-gamma*z)^delta] sous N(0,1)
                       (Ding, Granger & Engle 1993, eq. 3.7)
    - FIGARCH        : 1.0 par construction (memoire longue).

    Parameters
    ----------
    modele : str
        Nom du modele (ex: 'GJR-GARCH').
    params : dict or pd.Series
        Parametres estimes du modele.
    """
    try:
        p = params

        def _sum_keys(prefix):
            return sum(float(v) for k, v in p.items()
                       if str(k).startswith(prefix))

        if modele == 'GARCH':
            return _sum_keys('alpha[') + _sum_keys('beta[')

        if modele == 'GJR-GARCH':
            return (_sum_keys('alpha[')
                    + _sum_keys('beta[')
                    + 0.5 * _sum_keys('gamma['))

        if modele == 'EGARCH':
            return _sum_keys('beta[')

        if modele in ('TGARCH', 'APARCH'):
            delta_val = p.get('delta', None)
            if delta_val is None:
                return float('nan')
            delta = float(delta_val)
            if math.isnan(delta):
                return float('nan')
            beta_sum = _sum_keys('beta[')
            alpha_kappa = 0.0
            i = 1
            while True:
                av = p.get(f'alpha[{i}]', None)
                if av is None:
                    break
                gv = float(p.get(f'gamma[{i}]', 0.0))
                kappa = _kappa_aparch(delta, gv)
                if math.isnan(kappa):
                    return float('nan')
                alpha_kappa += float(av) * kappa
                i += 1
            return beta_sum + alpha_kappa

        if modele == 'FIGARCH':
            return 1.0

    except Exception as exc:
        _log.warning('persistance_garch(%s) : calcul échoué, retour NaN : %s', modele, exc)
    return float('nan')
