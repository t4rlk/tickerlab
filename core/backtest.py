"""Backtesting OOS de la VaR : Kupiec (LR_UC), Christoffersen (LR_IND), conjoint (LR_CC).

Extensions FRTB / Basel IV (BCBS d457 janvier 2023) :
  - Dynamic Quantile (DQ) : Engle & Manganelli (2004, JBES 22:4)
  - ES backtest Z1/Z2    : Acerbi & Székely (2014, Risk Magazine)
  - Lopez loss           : Lopez (1999, FRBNY EP Review 5:2)
  - Fissler-Ziegel FZ0   : Fissler & Ziegel (2016, Ann. Stat. 44:4)
  - Basel traffic light  : BCBS (1996 / FRTB d457)
"""
import logging
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t as t_dist
from arch import arch_model

from .var_engine import var_cornish_fisher
from .fhs import fhs_var_oos

__etape_version__ = '1'
_log = logging.getLogger(__name__)


# â”€â”€ Tests statistiques â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def kupiec_test(N, T, alpha):
    """
    Test LR_UC de Kupiec (1995) â€” couverture inconditionnelle.

    H0 : taux de violation = 1 - alpha. Statistique chi2(1).

    Parameters
    ----------
    N : int
        Nombre de violations observÃ©es.
    T : int
        Taille de l'Ã©chantillon OOS.
    alpha : float
        Niveau de confiance de la VaR.

    Returns
    -------
    tuple
        (LR : float, p_value : float).
    """
    p = 1 - alpha
    if N == 0:
        return 0.0, 1.0
    if N == T:
        return np.inf, 0.0
    try:
        LR = -2 * ((T-N)*np.log(1-p) + N*np.log(p)
                 - (T-N)*np.log(1-N/T) - N*np.log(N/T))
        return float(LR), float(1 - chi2.cdf(LR, df=1))
    except Exception:
        return np.nan, np.nan


def christoffersen_test(viol):
    """
    Test LR_IND de Christoffersen (1998) â€” indÃ©pendance des violations.

    H0 : les violations sont indÃ©pendantes. Statistique chi2(1).

    Parameters
    ----------
    viol : array-like of int
        SÃ©quence binaire des violations (1 = violation, 0 = non-violation).

    Returns
    -------
    tuple
        (LR : float, p_value : float).
    """
    I   = np.asarray(viol, dtype=int)
    T00 = int(np.sum((I[:-1]==0) & (I[1:]==0)))
    T01 = int(np.sum((I[:-1]==0) & (I[1:]==1)))
    T10 = int(np.sum((I[:-1]==1) & (I[1:]==0)))
    T11 = int(np.sum((I[:-1]==1) & (I[1:]==1)))
    eps  = 1e-12
    pi01 = T01 / max(T00+T01, 1)
    pi11 = T11 / max(T10+T11, 1)
    pi   = (T01+T11) / max(len(I)-1, 1)
    try:
        L0 = (1-pi+eps)**(T00+T10) * (pi+eps)**(T01+T11)
        L1 = ((1-pi01+eps)**T00 * (pi01+eps)**T01 *
               (1-pi11+eps)**T10 * (pi11+eps)**T11)
        LR = -2 * np.log(L0/L1)
        return float(LR), float(1 - chi2.cdf(LR, df=1))
    except Exception:
        return np.nan, np.nan


def _build_row(methode, alpha, N_viol, T_eff, lr_uc, p_uc, lr_ind, p_ind):
    if not (np.isnan(lr_uc) or np.isnan(lr_ind)):
        lr_cc = lr_uc + lr_ind
        p_cc  = float(1 - chi2.cdf(lr_cc, df=2))
    else:
        lr_cc, p_cc = np.nan, np.nan

    fmt = lambda v: round(v, 4) if not np.isnan(v) else 'n/a'
    return {
        'Methode':    methode,
        'Niveau':     f'{int(alpha*100)}%',
        'N viol.':    N_viol,
        'Taux obs.':  round(N_viol / T_eff, 4),
        'Taux theo.': round(1 - alpha, 4),
        'LR_UC':      fmt(lr_uc),  'p_UC':  fmt(p_uc),
        'LR_IND':     fmt(lr_ind), 'p_IND': fmt(p_ind),
        'LR_CC':      fmt(lr_cc),  'p_CC':  fmt(p_cc),
        'Verdict UC': 'OK'  if (not isinstance(p_uc, str) and p_uc  > 0.05) else 'NON',
        'Verdict CC': 'OK'  if (not isinstance(p_cc, str) and p_cc  > 0.05) else 'NON',
    }


# â”€â”€ Backtest principal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _rnd4(x):
    """Arrondi defensif a 4 decimales, transparent aux NaN/None."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return round(xf, 4) if math.isfinite(xf) else xf


def _detail_backtest_oos(serie_oos, mu_bt, nu_bt, dist, vol_oos_vals, garch_bt,
                         mu_tr, sigma_tr, T_eff_dyn, config_frtb=None):
    """Bloc de detail enrichi pour l'API v1 : les 6 tests de backtest sur la
    fenetre OOS du GARCH dynamique.

    Reutilise EXCLUSIVEMENT des fonctions existantes (aucune econometrie nouvelle) :
      - kupiec_test / christoffersen_test    (ce module)
      - run_frtb_backtests -> DQ, Acerbi-Szekely, Fissler-Ziegel, Lopez, feu Basel
      - garch_selector.berkowitz_test        (PIT LR sur residus standardises OOS)
      - var_engine.tvar_normale / tvar_student (multiplicateur ES standardise)
      - dm_gk._newey_west_var / _auto_lags   (cœur HAC pour le DM de FZ0)

    Convention pertes : VaR/ES negatifs (queue gauche), identique a backtest_oos.
    """
    from .var_engine import tvar_normale, tvar_student, var_student
    from .garch_selector import berkowitz_test
    from .dm_gk import _newey_west_var, _auto_lags

    r   = np.asarray(serie_oos,    dtype=float)[:T_eff_dyn]
    sig = np.asarray(vol_oos_vals, dtype=float)[:T_eff_dyn]
    alpha_v, alpha_e = 0.99, 0.975

    # Quantile standardise VaR 99 % + multiplicateur ES 97.5 % (meme dist que la boucle)
    resid_std = (garch_bt.resid / garch_bt.conditional_volatility).dropna().values
    resid_std = resid_std[np.isfinite(resid_std)]
    if dist == 'normal':
        q99 = norm.ppf(1 - alpha_v)
        esz = tvar_normale(0.0, 1.0, alpha_e)
    elif dist == 't' and not math.isnan(nu_bt):
        # resid_std est standardise : quantile Student standardise lui aussi.
        q99 = var_student(0.0, 1.0, nu_bt, alpha_v)
        esz = tvar_student(0.0, 1.0, nu_bt, alpha_e)
    else:                                    # skewt / ged / empirique
        q99 = float(np.quantile(resid_std, 1 - alpha_v))
        thr = float(np.quantile(resid_std, 1 - alpha_e))
        tail = resid_std[resid_std <= thr]
        esz = float(tail.mean()) if len(tail) else thr

    var99 = mu_bt + q99 * sig                 # negatif (queue de perte)
    es975 = mu_bt + esz * sig                 # negatif (esz < 0)

    # Kupiec + Christoffersen sur GARCH dynamique 99 %
    viol = (r < var99).astype(int)
    N = int(viol.sum())
    lr_uc,  p_uc  = kupiec_test(N, T_eff_dyn, alpha_v)
    lr_ind, p_ind = christoffersen_test(viol)

    # FRTB : DQ (Engle-Manganelli), Acerbi-Szekely Z1/Z2, Fissler-Ziegel, Lopez, feu
    frtb = run_frtb_backtests(r, var99, es975, alpha_var=alpha_v, alpha_es=alpha_e,
                              config=config_frtb)

    # Berkowitz PIT sur residus standardises OOS
    z_oos = (r - mu_bt) / sig
    berk  = berkowitz_test(z_oos, dist, {'nu': nu_bt})

    # ── Fissler-Ziegel : verdict par Diebold-Mariano vs benchmark Normal statique ─
    # FZ0 est un SCORE de comparaison, PAS un test isole. Le verdict accepte/rejete
    # provient d'un DM (HAC Newey-West, cœur reutilise de dm_gk) du score FZ0 du GARCH
    # dynamique contre un benchmark VaR/ES Normal STATIQUE (mu, sigma du train,
    # constants sur l'OOS). H0 : precision predictive egale. Benchmark nomme dans la
    # sortie. `accepte` = GARCH non significativement PIRE que le benchmark.
    var99_bench = np.full(T_eff_dyn, mu_tr + norm.ppf(1 - alpha_v) * sigma_tr)
    es975_bench = np.full(T_eff_dyn, mu_tr + tvar_normale(0.0, 1.0, alpha_e) * sigma_tr)
    fz_g = fissler_ziegel_loss(r, var99,       es975,       alpha=alpha_v)
    fz_b = fissler_ziegel_loss(r, var99_bench, es975_bench, alpha=alpha_v)
    fz_g_ser = fz_g['score_series'].values
    fz_b_ser = fz_b['score_series'].values
    n_dm = min(len(fz_g_ser), len(fz_b_ser))
    dm_stat = dm_pval = float('nan')
    if n_dm >= 10:
        d = fz_g_ser[:n_dm] - fz_b_ser[:n_dm]
        d = d[np.isfinite(d)]
        Td = len(d)
        if Td >= 10 and float(np.std(d)) > 1e-14:
            var_d = _newey_west_var(d, _auto_lags(Td))
            if var_d > 0:
                dm_stat = float(np.mean(d) / math.sqrt(var_d / Td))
                dm_pval = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))

    return {
        'alpha_var': alpha_v, 'alpha_es': alpha_e, 'T_eff_dyn': int(T_eff_dyn),
        'n_exceptions_99': N,
        'kupiec':         {'stat': _rnd4(lr_uc),  'pvalue': _rnd4(p_uc)},
        'christoffersen': {'stat': _rnd4(lr_ind), 'pvalue': _rnd4(p_ind)},
        'frtb': frtb,                          # dict {dq, acerbi_szekely, lopez, fissler_ziegel, traffic_light}
        'berkowitz': berk,                     # dict {LR_stat, pvalue, ...}
        'fissler_ziegel_dm': {
            'benchmark':       'Normale statique (mu, sigma du train)',
            'score_garch':     fz_g['score_moyen'],
            'score_benchmark': fz_b['score_moyen'],
            'dm_stat':         _rnd4(dm_stat),
            'dm_pvalue':       _rnd4(dm_pval),
        },
    }


def backtest_oos(rendements, best, split_ratio=0.70,
                 niveaux_test=None, n_simulations_mc=50_000,
                 retour_detail=False, config_frtb=None, **_):
    """
    Backtesting VaR out-of-sample avec split chronologique.

    Les paramÃ¨tres GARCH sont estimÃ©s sur le train et les prÃ©visions
    one-step-ahead sont produites via arch_model.fix() sur la sÃ©rie complÃ¨te.

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements en pourcentage (sÃ©rie complÃ¨te).
    best : dict
        ParamÃ¨tres du meilleur GARCH (clÃ©s : vol, p, o, q, dist).
    split_ratio : float, optional
        Fraction train (dÃ©faut 0.70).
    niveaux_test : list of float, optional
        Niveaux VaR Ã  tester (dÃ©faut [0.95, 0.99]).
    n_simulations_mc : int, optional
        Simulations Monte Carlo pour la VaR MC (dÃ©faut 50 000).
    **_ :
        ParamÃ¨tres supplÃ©mentaires ignorÃ©s.

    Returns
    -------
    tuple
        (df_bt : pd.DataFrame, T_train : int, T_eff_dyn : int).
    """
    if niveaux_test is None:
        niveaux_test = [0.95, 0.99]

    serie_full = rendements.dropna()
    T_full     = len(serie_full)
    T_train    = int(np.floor(split_ratio * T_full))
    T_eff_dyn  = T_full - T_train

    serie_train = serie_full.iloc[:T_train]
    serie_test  = serie_full.iloc[T_train:]

    vol_opt = best['vol']
    p_opt   = int(best['p'])
    o_opt   = int(best['o'])
    q_opt   = int(best['q'])
    dist    = best['dist']

    garch_bt = arch_model(
        serie_train, vol=vol_opt, p=p_opt, o=o_opt, q=q_opt, dist=dist
    ).fit(disp='off')

    mu_bt = float(garch_bt.params.get('mu', float(serie_train.mean())))
    nu_bt = float(garch_bt.params.get('nu', math.nan))

    vol_oos = arch_model(
        serie_full, vol=vol_opt, p=p_opt, o=o_opt, q=q_opt, dist=dist
    ).fix(garch_bt.params).conditional_volatility.iloc[T_train:]

    mu_tr    = float(serie_train.mean())
    sigma_tr = float(serie_train.std())
    S_tr     = float(serie_train.skew())
    K_tr     = float(serie_train.kurt())
    nu_tr, mu_t_tr, sig_t_tr = t_dist.fit(serie_train.values, floc=mu_tr)

    mc_bt_sim = garch_bt.forecast(horizon=1, method='simulation',
                                  simulations=n_simulations_mc)
    mc_ret_bt = mc_bt_sim.simulations.values[-1, :, 0]

    serie_oos = serie_test.values[:T_eff_dyn]
    rows = []

    # ── FHS one-step-ahead — pool de résidus STRICTEMENT train ───────────────
    # z_pool_bt = résidus standardisés GARCH estimé sur le TRAIN uniquement,
    # strictement {z_τ : τ ≤ T_train − 1}. AUCUN résidu post-train.
    # Cf. Barone-Adesi, Giannopoulos & Vosper (1999), Section 3.
    # Frontière explicite — anti look-ahead bias (erreur classique des implémentations naïves).
    _z_fhs_raw  = (garch_bt.resid / garch_bt.conditional_volatility).dropna().values
    z_pool_bt   = _z_fhs_raw[np.isfinite(_z_fhs_raw)]
    _fhs_ok     = len(z_pool_bt) > 100
    if not _fhs_ok:
        _log.warning("FHS backtest: pool trop petit (%d résidus) — méthode FHS ignorée.",
                     len(z_pool_bt))
    _fhs_oos = (fhs_var_oos(z_pool_bt, vol_oos.values[:T_eff_dyn], mu_bt, niveaux_test)
                if _fhs_ok else {})

    for alpha in niveaux_test:
        if dist == 'normal':
            q_z_bt = norm.ppf(1 - alpha)
        elif dist == 't':
            q_z_bt = t_dist.ppf(1 - alpha, df=nu_bt) if not math.isnan(nu_bt) \
                     else norm.ppf(1 - alpha)
        else:
            resid_std_bt = (garch_bt.resid / garch_bt.conditional_volatility).dropna()
            q_z_bt = float(np.quantile(resid_std_bt.values, 1 - alpha))

        var_garch_oos = (mu_bt + q_z_bt * vol_oos.values)[:T_eff_dyn]

        methodes_statiques = {
            'Historique':     np.quantile(serie_train.values, 1 - alpha),
            'Normale':        norm.ppf(1 - alpha, loc=mu_tr, scale=sigma_tr),
            'Student':        t_dist.ppf(1 - alpha, df=nu_tr, loc=mu_t_tr, scale=sig_t_tr),
            'Cornish-Fisher': var_cornish_fisher(mu_tr, sigma_tr, S_tr, K_tr, alpha),
            'Monte Carlo':    float(np.quantile(mc_ret_bt, 1 - alpha)),
        }

        for methode, var_val in methodes_statiques.items():
            viol     = (serie_oos < var_val).astype(int)
            N        = int(viol.sum())
            lr_uc,  p_uc  = kupiec_test(N, T_eff_dyn, alpha)
            lr_ind, p_ind = christoffersen_test(viol)
            rows.append(_build_row(methode, alpha, N, T_eff_dyn, lr_uc, p_uc, lr_ind, p_ind))

        viol_g   = (serie_oos < var_garch_oos).astype(int)
        N_g      = int(viol_g.sum())
        lr_uc_g,  p_uc_g  = kupiec_test(N_g, T_eff_dyn, alpha)
        lr_ind_g, p_ind_g = christoffersen_test(viol_g)
        rows.append(_build_row('GARCH dyn.', alpha, N_g, T_eff_dyn,
                               lr_uc_g, p_uc_g, lr_ind_g, p_ind_g))

        # FHS one-step-ahead (déterministe, équivalent n_boot → ∞ pour H=1)
        # z_pool_bt ici contient UNIQUEMENT les résidus train — cf. commentaire supra.
        if _fhs_ok and alpha in _fhs_oos:
            var_fhs_vec  = _fhs_oos[alpha]['var']
            viol_fhs     = (serie_oos < var_fhs_vec).astype(int)
            N_fhs        = int(viol_fhs.sum())
            lr_uc_f,  p_uc_f  = kupiec_test(N_fhs, T_eff_dyn, alpha)
            lr_ind_f, p_ind_f = christoffersen_test(viol_fhs)
            rows.append(_build_row('FHS', alpha, N_fhs, T_eff_dyn,
                                   lr_uc_f, p_uc_f, lr_ind_f, p_ind_f))

    df = pd.DataFrame(rows)
    if not retour_detail:
        return df, T_train, T_eff_dyn

    # Detail enrichi (les 6 tests OOS pour l'API v1). Un echec ici ne doit PAS
    # perdre le backtest de base : on journalise avec contexte et on renvoie un
    # detail marque en erreur (jamais un silence, jamais un except: pass).
    try:
        detail = _detail_backtest_oos(
            serie_oos, mu_bt, nu_bt, dist, vol_oos.values, garch_bt,
            mu_tr, sigma_tr, T_eff_dyn, config_frtb=config_frtb,
        )
    except Exception as exc:                     # noqa: BLE001 — journalise + degrade
        _log.exception('backtest_oos: calcul du detail enrichi en echec '
                       '(dist=%s, T_eff_dyn=%s) : %s', dist, T_eff_dyn, exc)
        detail = {'erreur': str(exc)}
    return df, T_train, T_eff_dyn, detail


# ── Backtests FRTB-grade ──────────────────────────────────────────────────────

def engle_manganelli_dq(
    returns: np.ndarray,
    var: np.ndarray,
    lags: int = 4,
    alpha: float = 0.99,
) -> dict:
    """
    Dynamic Quantile (DQ) test — Engle & Manganelli (2004).

    Régresse Hit_t = I(r_t < VaR_{α,t}) − (1−α) sur ses propres lags et VaR_t.
    Statistique : DQ = Ψ′X(X′X)⁻¹X′Ψ / (α(1−α)) ~ χ²(lags+2) sous H₀.

    H₀ : la VaR est correctement spécifiée (couverture + indépendance).

    Parameters
    ----------
    returns : np.ndarray
        Rendements out-of-sample.
    var : np.ndarray
        VaR_{α,t} dynamique (même longueur que returns).
    lags : int
        Nombre de lags du hit (défaut 4, valeur du papier original).
    alpha : float
        Niveau de confiance (défaut 0.99).

    Returns
    -------
    dict avec 'DQ_stat', 'p_value', 'df', 'verdict'.

    References
    ----------
    Engle, R.F. & Manganelli, S. (2004). CAViaR: Conditional Autoregressive
        Value at Risk by Regression Quantiles. Journal of Business & Economic
        Statistics, 22(4), 367–381.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var,     dtype=float)
    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]
    T = len(r)
    if T <= lags + 2:
        return {'DQ_stat': float('nan'), 'p_value': float('nan'),
                'df': lags + 2, 'verdict': 'n/a'}

    hit = (r < v).astype(float) - (1.0 - alpha)   # indicateur centré
    start = lags
    h_dep = hit[start:]                             # variable dépendante

    cols = [np.ones(T - start)]
    for lag in range(1, lags + 1):
        cols.append(hit[start - lag: T - lag])
    cols.append(v[start:])
    X = np.column_stack(cols)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        DQ = float(h_dep @ X @ XtX_inv @ X.T @ h_dep) / (alpha * (1.0 - alpha))
        df = lags + 2
        pv = float(1.0 - chi2.cdf(DQ, df=df))
    except np.linalg.LinAlgError:
        _log.warning('engle_manganelli_dq: matrice singulière')
        DQ, pv, df = float('nan'), float('nan'), lags + 2

    return {
        'DQ_stat': round(DQ, 4) if math.isfinite(DQ) else DQ,
        'p_value': round(pv, 4) if math.isfinite(pv) else pv,
        'df':      lags + 2,
        'verdict': 'OK' if (math.isfinite(pv) and pv > 0.05) else 'NON',
    }


def acerbi_szekely_z_test(
    returns: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float = 0.975,
    n_sim: int = 10_000,
) -> dict:
    """
    Test ES d'Acerbi & Székely — statistiques Z1 et Z2.

    Z1 = (1/(N_T·|ē|)) · Σ_t I_t·r_t + 1     (ES moyen, conditionnel)
    Z2 = (1/(T·p))     · Σ_t I_t·r_t/|e_t| + 1  (ES ponctuel, inconditionnel)

    Sous H₀ (ES correctement spécifié), E[Z1] = E[Z2] = 0.
    Z < 0 indique une sous-estimation de l'ES (risque sous-estimé).

    Les dénominateurs prennent la MAGNITUDE de l'ES : les entrées sont en
    convention pertes (négatives), et sans valeur absolue le rapport
    Σ I·r / ES change de signe, plaçant Z autour de +2 sous H₀ au lieu de 0.
    Z1 étant conditionnel aux violations, il est normalisé par leur nombre
    observé N_T et vaut nan en leur absence ; Z2, inconditionnel, vaut alors
    exactement 1.0, qui est sa valeur maximale.
    p-valeurs par bootstrap paramétrique sous H₀.

    Parameters
    ----------
    returns : np.ndarray
        Rendements out-of-sample (négatifs = pertes).
    var : np.ndarray
        VaR_{α,t} (négatif, convention pertes).
    es : np.ndarray
        ES_{α,t} (négatif, convention pertes).
    alpha : float
        Niveau de confiance (défaut 0.975 — norme FRTB).
    n_sim : int
        Simulations bootstrap pour les p-valeurs (défaut 10 000).

    Returns
    -------
    dict avec 'Z1', 'p_Z1', 'Z2', 'p_Z2', 'verdict_Z1', 'verdict_Z2'.

    References
    ----------
    Acerbi, C. & Székely, B. (2014). Back-testing expected shortfall.
        Risk Magazine, November 2014, pp. 76–81.
    """
    r    = np.asarray(returns, dtype=float)
    v    = np.asarray(var,     dtype=float)
    e    = np.asarray(es,      dtype=float)
    mask = np.isfinite(r) & np.isfinite(v) & np.isfinite(e)
    r, v, e = r[mask], v[mask], e[mask]
    T = len(r)
    p = 1.0 - alpha
    hits = (r < v).astype(float)

    n_viol = float(hits.sum())
    e_bar  = float(np.mean(e))

    # L'ES entre en convention pertes (negatif) mais la statistique compare une
    # perte a une MAGNITUDE d'ES : sans la valeur absolue le rapport change de
    # signe et Z vaut ~ +2 sous H0 au lieu de ~ 0, ce qui inverse la lecture
    # « Z negatif = risque sous-estime ».
    # Z1 est conditionnel aux violations : normalise par leur nombre OBSERVE, et
    # indefini (nan) en leur absence — a ne pas confondre avec 0, qui signifierait
    # « ES correctement specifie ».
    Z1 = (float(np.dot(hits, r)) / (n_viol * abs(e_bar)) + 1.0
          if n_viol > 0 and abs(e_bar) > 1e-12 else float('nan'))

    abs_e = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        z2_terms = np.where(abs_e > 1e-12, hits * r / abs_e, 0.0)
    Z2 = float(z2_terms.sum()) / (T * p) + 1.0

    # Violations empiriques (r_t, e_t) pour bootstrap sous H₀
    viol_mask = r < v
    viol_r = r[viol_mask]
    viol_e = e[viol_mask]

    if len(viol_r) == 0 or n_sim == 0:
        return {
            'Z1': Z1, 'p_Z1': float('nan'),
            'Z2': Z2, 'p_Z2': float('nan'),
            'verdict_Z1': 'n/a', 'verdict_Z2': 'n/a',
        }

    rng = np.random.default_rng(0)
    N_s = rng.binomial(T, p, size=n_sim)          # nb violations par sim
    n_viol_obs = len(viol_r)

    Z1_null = np.zeros(n_sim)
    Z2_null = np.zeros(n_sim)

    for s in range(n_sim):
        n_v = int(N_s[s])
        if n_v == 0:
            Z1_null[s] = float('nan')     # indefini, comme la statistique observee
            Z2_null[s] = 1.0
            continue
        idx   = rng.choice(n_viol_obs, size=n_v, replace=True)
        r_s   = viol_r[idx]
        e_s   = viol_e[idx]
        Z1_null[s] = (r_s.sum() / (n_v * abs(e_bar)) + 1.0
                      if abs(e_bar) > 1e-12 else float('nan'))
        ae_s = np.abs(e_s)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z2_null[s] = (np.where(ae_s > 1e-12, r_s / ae_s, 0.0).sum()
                          / (T * p) + 1.0)

    # Z1_null peut contenir des nan (tirages sans violation) : les exclure plutot
    # que de les laisser compter comme des non-depassements.
    fini1 = np.isfinite(Z1_null)
    p_Z1 = (float(np.mean(Z1_null[fini1] <= Z1))
            if math.isfinite(Z1) and fini1.any() else float('nan'))
    p_Z2 = float(np.mean(Z2_null <= Z2)) if math.isfinite(Z2) else float('nan')

    return {
        'Z1':        round(Z1,   4) if math.isfinite(Z1)   else Z1,
        'p_Z1':      round(p_Z1, 4) if math.isfinite(p_Z1) else p_Z1,
        'Z2':        round(Z2,   4) if math.isfinite(Z2)   else Z2,
        'p_Z2':      round(p_Z2, 4) if math.isfinite(p_Z2) else p_Z2,
        'verdict_Z1': 'OK' if (math.isfinite(p_Z1) and p_Z1 > 0.05) else 'NON',
        'verdict_Z2': 'OK' if (math.isfinite(p_Z2) and p_Z2 > 0.05) else 'NON',
    }


def lopez_loss(returns: np.ndarray, var: np.ndarray) -> dict:
    """
    Fonction de perte de Lopez (1999) — pénalise l'amplitude des violations.

    L_t = 1 + (r_t − VaR_t)² si r_t < VaR_t, sinon 0.
    Score total = Σ_t L_t (plus faible = meilleur).

    Parameters
    ----------
    returns : np.ndarray
        Rendements out-of-sample.
    var : np.ndarray
        VaR dynamique (même longueur que returns).

    Returns
    -------
    dict avec 'score_total', 'n_violations', 'score_moyen', 'loss_series'.

    References
    ----------
    Lopez, J.A. (1999). Methods for Evaluating Value-at-Risk Estimates.
        Federal Reserve Bank of New York Economic Policy Review, 5(2), 3–17.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var,     dtype=float)
    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]

    viol  = r < v
    loss  = np.where(viol, 1.0 + (r - v) ** 2, 0.0)
    total = float(loss.sum())

    return {
        'score_total':  round(total, 4),
        'n_violations': int(viol.sum()),
        'score_moyen':  round(total / max(len(r), 1), 6),
        'loss_series':  pd.Series(loss, name='Lopez_loss'),
    }


def fissler_ziegel_loss(
    returns: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float = 0.99,
) -> dict:
    """
    Score FZ0 de Fissler & Ziegel — strictement consistant pour (VaR_α, ES_α).

    Formule FZ0 (convention pertes négatives, Nolde & Ziegel 2017 eq. 3.6) :
        S_t = [(p − I_t)·v_t + I_t·r_t] / e_t + p·log(−e_t)
    où p = 1−α, I_t = I(r_t < v_t), e_t = ES_t < 0.

    Score total plus faible = meilleur modèle (comparaison relative).

    Parameters
    ----------
    returns : np.ndarray
        Rendements out-of-sample (négatifs = pertes).
    var : np.ndarray
        VaR_{α,t} (négatif, convention pertes).
    es : np.ndarray
        ES_{α,t} (négatif, convention pertes).
    alpha : float
        Niveau de confiance (défaut 0.99).

    Returns
    -------
    dict avec 'score_total', 'score_moyen', 'score_series'.

    References
    ----------
    Fissler, T. & Ziegel, J.F. (2016). Higher order elicitability and
        Osband's principle. Annals of Statistics, 44(4), 1680–1707.
    Nolde, N. & Ziegel, J.F. (2017). Elicitability and backtesting:
        Perspectives for banking regulation. Annals of Statistics,
        45(4), 1597–1638.
    """
    r    = np.asarray(returns, dtype=float)
    v    = np.asarray(var,     dtype=float)
    e    = np.asarray(es,      dtype=float)
    mask = np.isfinite(r) & np.isfinite(v) & np.isfinite(e) & (e < -1e-12)
    r, v, e = r[mask], v[mask], e[mask]

    hits  = (r < v).astype(float)
    p     = 1.0 - alpha
    score_t = ((p - hits) * v + hits * r) / e + p * np.log(-e)
    total   = float(score_t.sum())

    return {
        'score_total':  round(total, 4),
        'score_moyen':  round(total / max(len(r), 1), 6),
        'score_series': pd.Series(score_t, name='FZ_loss'),
    }


def basel_traffic_light(
    returns: np.ndarray,
    var_99: np.ndarray,
    window: int = 250,
) -> dict:
    """
    Feu tricolore Basel sur fenêtre glissante de 250 jours (BCBS 1996 / FRTB).

    Comptage des violations sur les 250 derniers jours ouvrables :
        0–4  violations : zone verte  (multiplicateur 3.0)
        5–9  violations : zone jaune  (multiplicateur 3.40–3.85)
        ≥10  violations : zone rouge  (multiplicateur 4.0)

    Parameters
    ----------
    returns : np.ndarray
        Rendements out-of-sample.
    var_99 : np.ndarray
        VaR à 99% (même longueur que returns).
    window : int
        Fenêtre glissante en jours ouvrables (défaut 250 — standard Basel).

    Returns
    -------
    dict avec 'zone', 'n_violations_250j', 'multiplicateur', 'rolling_violations'.

    References
    ----------
    BCBS (1996). Supervisory framework for the use of backtesting in
        conjunction with the internal models approach to market risk
        capital requirements. Basel Committee on Banking Supervision.
    BCBS (2023). Minimum capital requirements for market risk (d457),
        §§325–336 — Traffic light approach.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_99,  dtype=float)
    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]
    T = len(r)

    hits = (r < v).astype(int)

    # Rolling count on last 'window' observations
    roll_viol = (pd.Series(hits)
                 .rolling(window=window, min_periods=min(window, T))
                 .sum()
                 .values)

    # Zone selon les 250 derniers jours (ou toute la période si T < 250)
    n_last = int(roll_viol[-1]) if T > 0 and np.isfinite(roll_viol[-1]) else 0

    if n_last <= 4:
        zone = 'vert'
        mult = 3.0
    elif n_last <= 9:
        zone = 'jaune'
        # Interpolation linéaire entre 3.40 (5 viol.) et 3.85 (9 viol.)
        mult = round(3.40 + (n_last - 5) * (3.85 - 3.40) / 4, 2)
    else:
        zone = 'rouge'
        mult = 4.0

    _log.info('Basel traffic light : %d violations / %d j → zone %s (mult=%.2f)',
              n_last, min(T, window), zone, mult)

    return {
        'zone':               zone,
        'n_violations_250j':  n_last,
        'multiplicateur':     mult,
        'rolling_violations': pd.Series(roll_viol, name='rolling_violations'),
    }


def run_frtb_backtests(
    returns_oos: np.ndarray,
    var_garch_oos: np.ndarray,
    es_garch_oos: np.ndarray,
    alpha_var: float = 0.99,
    alpha_es: float = 0.975,
    config: dict | None = None,
) -> dict:
    """
    Exécute l'ensemble des tests FRTB-grade sur la période out-of-sample.

    Regroupe : DQ (Engle-Manganelli), Z1/Z2 (Acerbi-Székely), Lopez,
    Fissler-Ziegel et feu tricolore Basel sur les prévisions GARCH dynamiques.

    Parameters
    ----------
    returns_oos   : np.ndarray  Rendements OOS.
    var_garch_oos : np.ndarray  VaR GARCH 99% OOS.
    es_garch_oos  : np.ndarray  ES GARCH 97.5% OOS.
    alpha_var     : float       Niveau VaR (défaut 0.99).
    alpha_es      : float       Niveau ES FRTB (défaut 0.975).
    config        : dict        Lit config['backtest']['tests_frtb'] si fourni.

    Returns
    -------
    dict avec clés 'dq', 'acerbi_szekely', 'lopez', 'fissler_ziegel',
    'traffic_light'.
    """
    if config is not None:
        frtb_bt = config.get('backtest', {}).get('tests_frtb', {})
        n_sim    = int(frtb_bt.get('n_sim_as', 10_000))
        lags_dq  = int(frtb_bt.get('lags_dq',  4))
    else:
        n_sim   = 10_000
        lags_dq = 4

    _log.info('[FRTB] DQ test (lags=%d, alpha=%.2f)…', lags_dq, alpha_var)
    dq = engle_manganelli_dq(returns_oos, var_garch_oos, lags=lags_dq, alpha=alpha_var)

    _log.info('[FRTB] Acerbi-Székely Z1/Z2 (n_sim=%d)…', n_sim)
    as_ = acerbi_szekely_z_test(returns_oos, var_garch_oos, es_garch_oos,
                                 alpha=alpha_es, n_sim=n_sim)

    _log.info('[FRTB] Lopez loss…')
    lop = lopez_loss(returns_oos, var_garch_oos)

    _log.info('[FRTB] Fissler-Ziegel FZ0…')
    fz  = fissler_ziegel_loss(returns_oos, var_garch_oos, es_garch_oos, alpha=alpha_var)

    _log.info('[FRTB] Feu tricolore Basel…')
    tl  = basel_traffic_light(returns_oos, var_garch_oos)

    return {
        'dq':             dq,
        'acerbi_szekely': as_,
        'lopez':          lop,
        'fissler_ziegel': fz,
        'traffic_light':  tl,
    }
