# -*- coding: utf-8 -*-
"""
Comparaison statistique des methodes VaR — DM (1995) + GK (2005).

Tests implementes :
    DM  : Diebold & Mariano (1995, JBES 13:3) — egalite de perte esperee
           inconditionnelle. HAC Newey-West, p-valeur bilaterale.
    GK  : Giacomini & Komunjer (2005, JBES 23:4) — egalite de perte
           esperee CONDITIONNELLE aux instruments Z_t = [1, d_{t-1}].
           Statistique chi2(q) unilatérale droite (toujours positive
           par construction). La DIRECTION du gain se lit dans signe(d_bar)
           du DM associe.

Choix de GK 2005 vs GW 2006 :
    La méthodologie retient explicitement Giacomini & Komunjer (2005,
    JBES 23:4) — test concu pour la comparaison de previsions de quantiles
    (= VaR directement), avec la perte tick comme scoring rule.
    Note : avec Z_t = [1, d_{t-1}] uniquement, GK 2005 est asymptotiquement
    equivalent a GW 2006 (Giacomini & White, Econometrica 2006) applique a la
    meme perte tick (cf. Patton & Timmermann 2007, JoE).

Paires comparees :
    Section 8.6 du PDF : 3 paires structurantes (config : paires_section_principale).
    Dict de retour complet : toutes les paires disponibles dans var_series_dict.
    Avec 7 methodes (Historique, Normale, Student, CF, MC, GARCH dyn., FHS),
    cela donne C(7,2) = 21 paires — matrice exhaustive pour l'annexe.

Verdicts (5) :
    'model1_wins'              — DM p < alpha_test ET d_bar < 0
    'model2_wins'              — DM p < alpha_test ET d_bar > 0
    'egalite_stable'           — DM p >= alpha_test ET GK p >= alpha_test
    'egalite_avec_instabilite' — DM p >= alpha_test ET GK p < alpha_test
                                 (= pas de gagnant moyen, mais profil
                                  temporel instable — investiguer regime)
    'non_discriminable'        — test degenere (d_t ≈ 0 ou Omega singuliere) ;
                                 pas une preuve d'equivalence (limite methodologique)

References
----------
Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.
Giacomini, R. & Komunjer, I. (2005). Evaluation and Combination of
    Conditional Quantile Forecasts. Journal of Business & Economic
    Statistics, 23(4), 416-431.
Gonzalez-Rivera, G., Lee, T.H. & Mishra, S. (2004). Forecasting Volatility:
    A Reality Check Based on Option Pricing, Utility Function, Value-at-Risk,
    and Predictive Likelihood. International Journal of Forecasting, 20(4),
    629-645.
Patton, A.J. & Timmermann, A. (2007). Testing Forecast Optimality Under
    Unknown Loss. Journal of the American Statistical Association, 102,
    1172-1184.
"""
import logging
import math
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.stats import norm, chi2

__etape_version__ = '1'
_log = logging.getLogger(__name__)

_SQRT2_OV_PI = math.sqrt(2.0 / math.pi)

# Paires structurantes par defaut (section PDF 8.6)
_PAIRES_PRINCIPALES_DEFAULT = ['GARCH dyn._vs_FHS',
                                'Historique_vs_FHS',
                                'GARCH dyn._vs_Historique']

def find_pair(dm_gk_alp: dict, m1: str, m2: str) -> dict:
    """
    Recupere le resultat d'une paire (m1, m2) depuis le dict alpha
    en essayant les deux ordres (alphabetique canonical + inverse).

    Les cles sont toujours canoniques (ordre alphabetique), mais les
    paires_section_principale dans la config peuvent utiliser l'ordre inverse.
    """
    k1 = f"{m1}_vs_{m2}"
    k2 = f"{m2}_vs_{m1}"
    return dm_gk_alp.get(k1) or dm_gk_alp.get(k2)


VERDICTS_LABELS = {
    'model1_wins':               'model1 predit mieux (DM, p<alpha, d_bar<0)',
    'model2_wins':               'model2 predit mieux (DM, p<alpha, d_bar>0)',
    'egalite_stable':            'egalite stable (DM et GK non rejetes)',
    'egalite_avec_instabilite':  'egalite moyenne, instabilite temporelle (GK rejete)',
    'non_discriminable':         'test degenere — differentiel nul ou Omega singuliere',
}


# ── Fonctions de base ─────────────────────────────────────────────────────────

def tick_loss(r: np.ndarray, var: np.ndarray, alpha: float) -> np.ndarray:
    """
    Perte tick/quantile (Gonzalez-Rivera et al. 2004) :
        L_alpha(u_t) = u_t * (alpha - 1{u_t < 0}),  u_t = r_t - VaR_t

    Asymetrique : penalise davantage les violations (r_t < VaR_t).
    Nulle si u_t = 0 exactement (violation a la frontiere, non a l'interieur).
    """
    r   = np.asarray(r,   dtype=float)
    var = np.asarray(var, dtype=float)
    u   = r - var
    return u * (alpha - (u < 0).astype(float))


def _newey_west_var(d: np.ndarray, lags: int) -> float:
    """HAC variance scalaire Newey-West — pour DM (d_t scalaire)."""
    T   = len(d)
    d_c = d - d.mean()
    S   = float(d_c @ d_c) / T
    for l in range(1, lags + 1):
        w     = 1.0 - l / (lags + 1)          # noyau Bartlett
        gamma = float(d_c[l:] @ d_c[:T - l]) / T
        S    += 2.0 * w * gamma
    return max(S, 1e-14)


def _newey_west_matrix(g: np.ndarray, lags: int) -> np.ndarray:
    """
    HAC matrice Newey-West — pour GK (g_t = Z_t * d_t, dim T x q).

    Estime S = Var(sqrt(T) * g_bar) = Gamma(0) + sum_l w_l*(Gamma(l)+Gamma(l)').
    """
    T = len(g)
    S = g.T @ g / T                             # Gamma(0)
    for l in range(1, lags + 1):
        w       = 1.0 - l / (lags + 1)
        Gamma_l = g[l:].T @ g[:T - l] / T
        S       = S + w * (Gamma_l + Gamma_l.T)
    return S


def _auto_lags(T: int) -> int:
    """Lags HAC automatiques Andrews (1991) : int(T^(1/3))."""
    return max(1, int(T ** (1.0 / 3.0)))


# ── Tests principaux ──────────────────────────────────────────────────────────

def dm_test(var1: np.ndarray, var2: np.ndarray, r: np.ndarray,
            alpha: float, hac_lags='auto', alpha_test: float = 0.05) -> dict:
    """
    Test de Diebold & Mariano (1995) — precision predictive inconditionnelle.

    H0 : E[L_alpha(r - VaR1) - L_alpha(r - VaR2)] = 0.
    Stat : S_DM = d_bar / sqrt(Var_HAC(d) / T) ~ N(0,1) asymptotiquement.
    p-valeur bilaterale.

    d_bar < 0 => model1 a une perte esperee plus faible (model1 gagne).
    d_bar > 0 => model2 gagne.

    Parameters
    ----------
    var1, var2 : np.ndarray  Series VaR OOS.
    r          : np.ndarray  Rendements OOS (meme longueur).
    alpha      : float       Niveau de confiance VaR.
    hac_lags   : int|'auto'  Lags Newey-West. 'auto' = int(T^(1/3)).
    alpha_test : float       Seuil bilateral (defaut 0.05).

    Returns
    -------
    dict : stat, pval, loss_diff_mean, T_eff, hac_lags_used.
    """
    r    = np.asarray(r,    dtype=float)
    var1 = np.asarray(var1, dtype=float)
    var2 = np.asarray(var2, dtype=float)
    T    = min(len(r), len(var1), len(var2))
    r, var1, var2 = r[:T], var1[:T], var2[:T]

    d     = tick_loss(r, var1, alpha) - tick_loss(r, var2, alpha)
    d_bar = float(d.mean())

    # Détection sur la CAUSE (séries quasi identiques), pas sur le symptôme
    # (variance plancher). Le floor 1e-14 de _newey_west_var rend inatteignable
    # la détection via se < 1e-10 pour T < 1e6.
    if np.max(np.abs(d)) < 1e-12:
        lags = _auto_lags(T) if hac_lags == 'auto' else int(hac_lags)
        return {
            'stat': 0.0, 'pval': 1.0,
            'loss_diff_mean': d_bar, 'T_eff': T, 'hac_lags_used': lags,
            'degenerate': True,
        }

    lags = _auto_lags(T) if hac_lags == 'auto' else int(hac_lags)
    if lags > T // 4:
        _log.warning(
            'dm_test: HAC lags=%d > T/4=%d — variance HAC instable '
            '(serie trop courte). Utiliser hac_lags="auto" recommande.',
            lags, T // 4,
        )

    var_d = _newey_west_var(d, lags)
    se    = math.sqrt(var_d / T)

    if se < 1e-10:
        return {
            'stat': 0.0, 'pval': 1.0,
            'loss_diff_mean': d_bar, 'T_eff': T, 'hac_lags_used': lags,
            'degenerate': True,
        }

    stat = d_bar / se
    pval = float(2.0 * (1.0 - norm.cdf(abs(stat))))

    return {
        'stat':           round(stat, 4),
        'pval':           round(pval, 4),
        'loss_diff_mean': round(d_bar, 8),
        'T_eff':          T,
        'hac_lags_used':  lags,
        'degenerate':     False,
    }


def gk_test(var1: np.ndarray, var2: np.ndarray, r: np.ndarray,
            alpha: float, n_instruments: int = 2, hac_lags='auto',
            alpha_test: float = 0.05) -> dict:
    """
    Test de Giacomini & Komunjer (2005, JBES 23:4) — precision predictive
    conditionnelle.

    H0 : E[Z_t * (L_alpha(r_t - VaR1_t) - L_alpha(r_t - VaR2_t))] = 0
    Stat : GK = T_eff * g_bar' * Omega_hat^{-1} * g_bar ~ chi2(q) sous H0.

    La statistique est TOUJOURS positive par construction (forme quadratique).
    p-valeur unilatérale droite chi2. La DIRECTION du gain se lit dans
    le signe de d_bar (DM associe), pas dans la statistique GK.

    Instruments Z_t (q = n_instruments) :
        n_instruments=1 : Z_t = [1]              (GK reduit a DM^2)
        n_instruments=2 : Z_t = [1, d_{t-1}]     (standard, plus puissant)

    Note : avec Z_t = [1, d_{t-1}] uniquement, GK est asymptotiquement
    equivalent a GW 2006 applique a la meme perte tick (Patton & Timmermann 2007).

    Parameters
    ----------
    n_instruments : int  Dimension q de Z_t (1 ou 2).
    hac_lags      : int|'auto'  Lags HAC matrice Omega.
    alpha_test    : float        Seuil chi2 (defaut 0.05).

    Returns
    -------
    dict : stat, pval, df, reject.
    """
    r    = np.asarray(r,    dtype=float)
    var1 = np.asarray(var1, dtype=float)
    var2 = np.asarray(var2, dtype=float)
    T    = min(len(r), len(var1), len(var2))
    r, var1, var2 = r[:T], var1[:T], var2[:T]

    d      = tick_loss(r, var1, alpha) - tick_loss(r, var2, alpha)
    lags_z = n_instruments - 1
    lags_h = _auto_lags(T) if hac_lags == 'auto' else int(hac_lags)

    if lags_z > 0:
        # Z_t = [1, d_{t-1}] — instrument strictement predetermine a t-1
        Z     = np.column_stack([np.ones(T - lags_z), d[:T - lags_z]])
        d_eff = d[lags_z:]
    else:
        Z     = np.ones((T, 1))
        d_eff = d

    T_eff = len(d_eff)
    q     = Z.shape[1]

    # Conditions de moments : g_t = Z_t * d_t (d scalaire → multiplication simple)
    g     = Z * d_eff[:, None]
    g_bar = g.mean(axis=0)

    Omega = _newey_west_matrix(g, lags_h)

    _degenerate = False
    _reliable   = True

    try:
        cond = np.linalg.cond(Omega)
        if cond > 1e10:
            _log.warning('gk_test: Omega mal conditionnee (cond=%.1e) — passage a pinv', cond)
            Omega_inv   = np.linalg.pinv(Omega)
            _degenerate = True
            _reliable   = False
        else:
            Omega_inv = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        Omega_inv   = np.linalg.pinv(Omega)
        _degenerate = True
        _reliable   = False

    stat = float(T_eff * g_bar @ Omega_inv @ g_bar)
    stat = max(stat, 0.0)                         # forme quadratique >= 0

    if _degenerate:
        pval   = float('nan')
        reject = False
    else:
        pval   = float(1.0 - chi2.cdf(stat, df=q))
        reject = pval < alpha_test

    return {
        'stat':       round(stat, 4),
        'pval':       float('nan') if _degenerate else round(pval, 4),
        'df':         q,
        'reject':     reject,
        'degenerate': _degenerate,
        'reliable':   _reliable,
    }


def _determine_verdict(d_bar: float, dm_pval: float, gk_pval: float,
                        alpha_test: float = 0.05,
                        dm_degenerate: bool = False,
                        gk_degenerate: bool = False) -> str:
    """
    5 verdicts bases sur DM + GK :

    d_bar < 0 → model1 perte esperee plus faible → model1 gagne si DM rejette.
    d_bar > 0 → model2 gagne si DM rejette.
    GK rejette sans DM → instabilite temporelle (pas de gagnant moyen).
    Degenere → 'non_discriminable' (priorite absolue, pas une preuve d'equivalence).
    """
    if dm_degenerate or gk_degenerate:
        return 'non_discriminable'
    if dm_pval < alpha_test:
        return 'model1_wins' if d_bar < 0 else 'model2_wins'
    if gk_pval < alpha_test:
        return 'egalite_avec_instabilite'
    return 'egalite_stable'


# ── Construction des series VaR OOS ──────────────────────────────────────────

def _garch_vol_oos_manual(garch_final, serie_test: np.ndarray) -> np.ndarray:
    """
    Volatilite conditionnelle OOS par recursion GARCH manuelle.

    Utilise les parametres estimes sur le TRAIN (garch_final) et applique
    la recursion sur les rendements OOS. Aucun appel a arch.fix() — robuste
    aux changements d'API arch.
    """
    from .fhs import _extract_garch_state

    state = _extract_garch_state(garch_final)
    T_oos = len(serie_test)
    mu      = state['mu']
    omega   = state['omega']
    alpha_p = state['alpha']
    gamma_p = state['gamma']
    beta_p  = state['beta']
    o       = state['o']
    last_v2 = state['last_vol2']
    last_e  = state['last_eps']

    vol_oos = np.zeros(T_oos)

    if state['vol_class'] == 'EGARCH':
        log_h  = math.log(max(last_v2, 1e-12))
        z_prev = last_e / max(math.sqrt(last_v2), 1e-12)
        for i in range(T_oos):
            log_h   = (omega + alpha_p * (abs(z_prev) - _SQRT2_OV_PI)
                       + gamma_p * z_prev + beta_p * log_h)
            sigma2  = math.exp(log_h)
            vol_oos[i] = math.sqrt(max(sigma2, 1e-12))
            eps     = float(serie_test[i]) - mu
            z_prev  = eps / max(vol_oos[i], 1e-12)
    else:                                          # GARCH / GJR-GARCH
        sigma2 = last_v2
        eps    = last_e
        for i in range(T_oos):
            if o > 0:                              # GJR
                ind    = float(eps < 0)
                sigma2 = omega + (alpha_p + gamma_p * ind) * eps ** 2 + beta_p * sigma2
            else:
                sigma2 = omega + alpha_p * eps ** 2 + beta_p * sigma2
            sigma2     = max(sigma2, 1e-12)
            vol_oos[i] = math.sqrt(sigma2)
            eps        = float(serie_test[i]) - mu

    return vol_oos


def build_var_series(rendements, garch_final, fhs_result: Optional[dict],
                     config: dict, T_train: Optional[int] = None) -> dict:
    """
    Construit le dict var_series_dict requis par comparer_methodes_var.

    Pour chaque methode disponible, calcule la serie VaR OOS de longueur T_eff.
    Methodes statiques (Historique, Normale, Student, CF, MC) : VaR constante
    repliee en array(T_eff,). Methodes dynamiques (GARCH dyn., FHS) : serie
    one-step-ahead (T_eff,).

    Le pool de residus FHS provient STRICTEMENT du TRAIN (anti look-ahead bias).

    Parameters
    ----------
    rendements  : pd.Series  Log-rendements complets (train + test).
    garch_final : ARCHModelResult  Estime sur le TRAIN uniquement.
    fhs_result  : dict|None  Resultat de calculer_var_fhs.
    config      : dict  Config globale (lit backtest.split_ratio, var.niveaux).
    T_train     : int|None  Si None, recalcule depuis split_ratio.

    Returns
    -------
    dict : {methode_name: {alpha: np.ndarray(T_eff,)}}
           alpha ∈ niveaux (ex. {0.95: array, 0.99: array}).

    Contrat — colonnes df_bt equivalentes lues :
        Methode | Niveau | var_series | viol_series
    (var_series et viol_series sont les arrays internes equivalents stockes ici
     sous forme de dict — pas dans df_bt pour eviter toute modification de
     backtest.py, conformement a l'exigence d'isolation.)
    """
    from scipy.stats import norm as sp_norm, t as sp_t
    from .fhs import fhs_var_oos
    from .var_engine import var_cornish_fisher

    split_ratio = float(config.get('backtest', {}).get('split_ratio', 0.70))
    niveaux     = list(config.get('var', {}).get('niveaux', [0.95, 0.99]))

    serie_full = rendements.dropna()
    n          = len(serie_full)
    if T_train is None:
        T_train = int(math.floor(split_ratio * n))

    serie_train = serie_full.iloc[:T_train]
    serie_test  = serie_full.iloc[T_train:]
    T_oos       = len(serie_test)

    if T_oos < 50:
        _log.warning('build_var_series: T_oos=%d tres court — resultats peu fiables.', T_oos)

    # ── Volatilite OOS (recursion manuelle GARCH) ─────────────────────────────
    vol_oos = _garch_vol_oos_manual(garch_final, serie_test.values)
    params  = garch_final.params
    mu_bt   = float(params.get('mu', float(serie_train.mean())))
    nu_bt   = float(params.get('nu', float('nan')))

    # Distribution du meilleur modele GARCH
    dist_name = type(garch_final.model.distribution).__name__.lower()

    var_series: dict = {}

    # ── GARCH dyn. (one-step-ahead parametrique) ──────────────────────────────
    for alp in niveaux:
        if 'normal' in dist_name:
            q_z = float(sp_norm.ppf(1.0 - alp))
        elif 't' in dist_name and not math.isnan(nu_bt):
            q_z = float(sp_t.ppf(1.0 - alp, df=nu_bt))
        else:
            z_raw = (garch_final.resid / garch_final.conditional_volatility).dropna().values
            z_fin = z_raw[np.isfinite(z_raw)]
            q_z   = float(np.quantile(z_fin, 1.0 - alp)) if len(z_fin) > 0 else float(sp_norm.ppf(1.0 - alp))
        var_series.setdefault('GARCH dyn.', {})[alp] = mu_bt + q_z * vol_oos

    # ── FHS one-step-ahead (pool residus TRAIN strictement) ──────────────────
    # z_pool = residus standardises estimes sur le TRAIN — AUCUN residu post-train.
    # Cf. Barone-Adesi, Giannopoulos & Vosper (1999), Section 3.
    # Limite H=1 : GARCH dyn. et FHS partagent vol_oos, mu_bt et le pool z.
    # A H=1, q_empirique(z_pool) ≈ q_parametrique => differentiels d_t ≈ 0 =>
    # DM peut etre degenere => verdict 'non_discriminable' (documente, non un bug).
    if fhs_result is not None:
        z_raw   = (garch_final.resid / garch_final.conditional_volatility).dropna().values
        z_pool  = z_raw[np.isfinite(z_raw)]
        if len(z_pool) > 100:
            fhs_oos = fhs_var_oos(z_pool, vol_oos, mu_bt, niveaux)
            for alp in niveaux:
                if alp in fhs_oos:
                    var_series.setdefault('FHS', {})[alp] = fhs_oos[alp]['var']

    # ── Methodes statiques ────────────────────────────────────────────────────
    mu_tr    = float(serie_train.mean())
    sig_tr   = float(serie_train.std())
    S_tr     = float(serie_train.skew())
    K_tr     = float(serie_train.kurt())
    nu_tr, mu_t_tr, sig_t_tr = sp_t.fit(serie_train.values, floc=mu_tr)

    for alp in niveaux:
        statics = {
            'Historique':     float(np.quantile(serie_train.values, 1.0 - alp)),
            'Normale':        float(sp_norm.ppf(1.0 - alp, loc=mu_tr, scale=sig_tr)),
            'Student':        float(sp_t.ppf(1.0 - alp, df=nu_tr,
                                             loc=mu_t_tr, scale=sig_t_tr)),
            'Cornish-Fisher': float(var_cornish_fisher(mu_tr, sig_tr, S_tr, K_tr, alp)),
        }
        for name, val in statics.items():
            arr = np.full(T_oos, val)
            var_series.setdefault(name, {})[alp] = arr

    return var_series


# ── Comparaison toutes paires ─────────────────────────────────────────────────

def comparer_methodes_var(
    var_series_dict: dict,
    r_oos: np.ndarray,
    niveaux: list,
    config: dict,
) -> dict:
    """
    Applique DM + GK (Giacomini & Komunjer 2005) a toutes les paires de methodes.

    Contrat d'entree (var_series_dict)
    -----------------------------------
    Cles   : noms de methode (str), ex. 'GARCH dyn.', 'FHS', 'Historique'.
    Valeurs: dict {alpha (float): np.ndarray(T_eff,)} — serie VaR OOS.

    Equivalent colonnes df_bt :
        Methode | Niveau | var_alpha_0.95 | var_alpha_0.99
        (stockes ici sous forme de dict pour eviter modification de backtest.py)

    Verdicts possibles :
        'model1_wins'              — DM rejette, model1 predit mieux
        'model2_wins'              — DM rejette, model2 predit mieux
        'egalite_stable'           — DM et GK non rejetes
        'egalite_avec_instabilite' — DM non rejete, GK rejete (instabilite)

    Returns
    -------
    dict : {
      'paires_principales': list[str],
      'alpha_test': float,
      alpha_0: {paire_key: {model1, model2, dm_stat, dm_pval, gk_stat,
                             gk_pval, gk_df, loss_diff_mean, verdict, T_eff}},
      alpha_1: {...},
      ...
    }
    La paire_key = f"{model1}_vs_{model2}" (ordre alphabetique).
    """
    dm_gk_cfg  = config.get('dm_gk', {})
    alpha_test = float(dm_gk_cfg.get('alpha_test', 0.05))
    hac_lags   = dm_gk_cfg.get('hac_lags', 'auto')
    n_inst     = int(dm_gk_cfg.get('n_instruments_gk', 2))
    paires_pp  = dm_gk_cfg.get('paires_section_principale', _PAIRES_PRINCIPALES_DEFAULT)

    T_oos = len(r_oos)
    if T_oos < 250:
        _log.warning(
            'comparer_methodes_var: T_oos=%d < 250 — puissance statistique '
            'insuffisante a alpha=99%% (recommandation T_oos >= 500).',
            T_oos,
        )
    elif T_oos < 500:
        alpha_eff = dm_gk_cfg.get('alpha_test_petit_echantillon', 0.10)
        _log.info(
            'comparer_methodes_var: T_oos=%d < 500 — alpha_test effectif '
            'eleve a %.2f (config dm_gk.alpha_test_petit_echantillon).',
            T_oos, alpha_eff,
        )

    methodes = sorted(var_series_dict.keys())
    pairs    = list(combinations(methodes, 2))    # ordre alphabetique stable
    # Alias de recherche order-insensitif : paire_key canonique = alphabetique
    # Permet de retrouver une paire via "A_vs_B" ou "B_vs_A" dans le retour.

    result: dict = {
        'paires_principales': paires_pp,
        'alpha_test':         alpha_test,
        'n_methodes':         len(methodes),
        'n_paires':           len(pairs),
        'methodes':           methodes,
    }

    for alp in niveaux:
        result[alp] = {}
        for m1, m2 in pairs:
            v1_d = var_series_dict.get(m1, {})
            v2_d = var_series_dict.get(m2, {})
            if alp not in v1_d or alp not in v2_d:
                continue

            v1   = np.asarray(v1_d[alp], dtype=float)
            v2   = np.asarray(v2_d[alp], dtype=float)
            T_eff = min(len(v1), len(v2), T_oos)
            r_eff = r_oos[:T_eff]
            v1, v2 = v1[:T_eff], v2[:T_eff]

            dm = dm_test(v1, v2, r_eff, alp,
                         hac_lags=hac_lags, alpha_test=alpha_test)
            gk = gk_test(v1, v2, r_eff, alp,
                         n_instruments=n_inst, hac_lags=hac_lags,
                         alpha_test=alpha_test)

            verdict = _determine_verdict(
                dm['loss_diff_mean'], dm['pval'], gk['pval'], alpha_test,
                dm_degenerate=dm.get('degenerate', False),
                gk_degenerate=gk.get('degenerate', False),
            )

            key = f"{m1}_vs_{m2}"
            result[alp][key] = {
                'model1':          m1,
                'model2':          m2,
                'dm_stat':         dm['stat'],
                'dm_pval':         dm['pval'],
                'gk_stat':         gk['stat'],
                'gk_pval':         gk['pval'],
                'gk_df':           gk['df'],
                'loss_diff_mean':  dm['loss_diff_mean'],
                'verdict':         verdict,
                'T_eff':           T_eff,
                'hac_lags_used':   dm['hac_lags_used'],
            }

    return result
