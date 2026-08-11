# -*- coding: utf-8 -*-
"""
Filtered Historical Simulation (FHS) — Barone-Adesi, Bourgoin & Giannopoulos (1998).

FHS hybride : volatilite conditionnelle GARCH (parametrique) + distribution
empirique des chocs (non-parametrique). Avantages par rapport aux methodes pures :

  * Versus HS simple   : capte le clustering de volatilite via GARCH (σ_t variable)
  * Versus paramétrique: capture la queue empirique reelle (non-gaussienne)
  * Versus règle √H    : VaR multi-periodes coherente avec la dynamique GARCH et les
                         queues epaisses reelles — diverge significativement a H >= 5

Algorithme (Barone-Adesi & Vosper 1999, Section 3) :
  1. Filtrer z_t = e_t / s_t depuis le GARCH retenu
  2. Pour i = 1..n_boot : tirer z_{t_1},...,z_{t_H} avec remise depuis l'historique
  3. Propager la variance GARCH :
       s²_{T+k} = f(theta, s²_{T+k-1}, z_{T+k-1})  — récurrence GARCH stochastique
       r_{T+k}  = mu + z_{t_k} * s_{T+k}
  4. VaR_alpha(H) = quantile(1-alpha) de sum_k r_{T+k}
     ES_alpha(H)  = moyenne de la queue <= VaR

Vectorisation : boucle externe sur H (sequentiel par construction GARCH),
mise a jour interne vectorisee sur n_boot chemins simultanes (axe 0).

References
----------
Barone-Adesi, G., Bourgoin, F. & Giannopoulos, K. (1998). Don't Look Back.
    Risk, 11(8), 100–103.
Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR Without Correlations
    for Portfolios of Derivative Securities. Journal of Futures Markets, 19(5), 583–602.
Nelson, D.B. (1991). Conditional Heteroskedasticity in Asset Returns.
    Econometrica, 59(2), 347–370.
"""
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

__etape_version__ = '2'
logger = logging.getLogger(__name__)

# E[|z|] pour z ~ N(0,1) — convention arch EGARCH (Nelson 1991)
_SQRT2_OV_PI: float = math.sqrt(2.0 / math.pi)


# ── Extraction de l'état GARCH ────────────────────────────────────────────────

def _extract_garch_state(garch_final) -> dict:
    """
    Extrait les paramètres GARCH et le dernier état (σ²_T, ε_T) pour la propagation FHS.

    Prend en charge GARCH (symétrique et GJR asymétrique), EGARCH, et
    APARCH/TGARCH (approximé en GARCH(1,1) avec avertissement).
    Seuls les termes d'ordre 1 sont utilisés pour la propagation multi-etapes ;
    pour p > 1 ou q > 1 un avertissement est emis.

    Returns
    -------
    dict avec :
        vol_class : str    'GARCH', 'EGARCH', 'APARCH'
        o         : int    ordre asymétrique (0 = symétrique)
        omega, alpha, gamma, beta : float   paramètres premier ordre
        mu        : float  paramètre de tendance
        last_vol2 : float  σ²_T (dernière variance conditionnelle)
        last_eps  : float  ε_T  (dernier résidu)
    """
    params    = garch_final.params
    vol_model = garch_final.model.volatility
    vol_class = type(vol_model).__name__   # 'GARCH', 'EGARCH', 'APARCH'

    try:
        p = int(vol_model.p)
        o = int(vol_model.o)
        q = int(vol_model.q)
    except AttributeError:
        p, o, q = 1, 0, 1

    if p > 1 or q > 1:
        logger.warning(
            "FHS: modele d'ordre p=%d, q=%d — seuls alpha[1]/beta[1] utilises pour "
            "la propagation multi-etapes (approximation ordre 1).", p, q,
        )

    if vol_class == 'APARCH':
        logger.warning(
            "FHS: modele APARCH/TGARCH approxi en GARCH(1,1) pour la propagation "
            "multi-etapes. Resultats H>1 indicatifs uniquement."
        )

    omega = float(params.get('omega',    1e-5))
    alpha = float(params.get('alpha[1]', 0.05))
    gamma = float(params.get('gamma[1]', 0.0))
    beta  = float(params.get('beta[1]',  0.90))
    mu    = float(params.get('mu',       0.0))

    last_vol  = float(garch_final.conditional_volatility.iloc[-1])
    last_vol2 = max(last_vol**2, 1e-12)
    last_eps  = float(garch_final.resid.iloc[-1])

    return {
        'vol_class': vol_class,
        'o':         o,
        'omega':     omega,
        'alpha':     alpha,
        'gamma':     gamma,
        'beta':      beta,
        'mu':        mu,
        'last_vol2': last_vol2,
        'last_eps':  last_eps,
    }


# ── Propagation vectorisée ────────────────────────────────────────────────────

def _propagate_paths(z_sampled: np.ndarray, state: dict) -> np.ndarray:
    """
    Propagation vectorisée de la variance GARCH pour la simulation FHS multi-etapes.

    La boucle externe est sur H (sequentielle — récurrence GARCH intrinsèque) ;
    la mise à jour interne est vectorisée sur les n_boot chemins simultanément.
    Complexité : O(H) itérations × O(n_boot) par étape.

    ❌ NON :
        for i in range(n_boot):
            for k in range(H): sigma2[i, k] = ...
    ✅ OUI :
        for k in range(H): sigma2[:, k] = ...   # vectorisé sur n_boot

    Parameters
    ----------
    z_sampled : np.ndarray, shape (n_boot, H)
        Innovations standardisées tirées avec remise depuis z_pool.
    state : dict
        Retour de _extract_garch_state().

    Returns
    -------
    np.ndarray, shape (n_boot, H)
        Rendements simulés r_{T+k}, k = 1..H.
    """
    n_boot, H = z_sampled.shape
    vol_class = state['vol_class']
    omega     = state['omega']
    alpha     = state['alpha']
    gamma     = state['gamma']
    beta      = state['beta']
    mu        = state['mu']
    o         = state['o']
    last_vol2 = state['last_vol2']
    last_eps  = state['last_eps']

    returns = np.empty((n_boot, H), dtype=float)

    if vol_class == 'EGARCH':
        # Récurrence log-variance (convention arch / Nelson 1991) :
        #   log(h_t) = ω + α·(|z_{t-1}| − √(2/π)) + γ·z_{t-1} + β·log(h_{t-1})
        # k=0 : z_prev = z_T connu (scalaire, même pour tous les chemins)
        z_prev     = np.full(n_boot, last_eps / math.sqrt(last_vol2))
        log_h_prev = np.full(n_boot, math.log(last_vol2))  # log(σ²_T)

        for k in range(H):
            # Propagation : log(σ²_{T+k+1}) à partir de z_{T+k} = z_prev
            log_h_k = (omega
                       + alpha * (np.abs(z_prev) - _SQRT2_OV_PI)
                       + gamma * z_prev
                       + beta  * log_h_prev)
            sigma_k          = np.exp(0.5 * log_h_k)
            returns[:, k]    = mu + z_sampled[:, k] * sigma_k
            # Mise à jour : z_{T+k+1} = z tiré depuis le pool
            z_prev     = z_sampled[:, k]
            log_h_prev = log_h_k

    else:
        # GARCH / GJR-GARCH (et APARCH approximé en GARCH) :
        #   σ²_{t+1} = ω + (α + γ·I(ε_t<0))·ε²_t + β·σ²_t
        # Premier pas : σ²_{T+1} déterministe (identique pour tous les chemins)
        if o > 0:
            ind_last  = float(last_eps < 0)
            sigma2_t1 = omega + (alpha + gamma * ind_last) * last_eps**2 + beta * last_vol2
        else:
            sigma2_t1 = omega + alpha * last_eps**2 + beta * last_vol2

        sigma2_prev = np.full(n_boot, max(sigma2_t1, 1e-12))

        for k in range(H):
            sigma_k       = np.sqrt(sigma2_prev)
            returns[:, k] = mu + z_sampled[:, k] * sigma_k
            eps_k         = z_sampled[:, k] * sigma_k   # ε_{T+k+1}

            # Propagation σ²_{T+k+2} depuis σ²_{T+k+1} et ε_{T+k+1}
            if o > 0:
                # GJR : indicateur signe sur l'innovation simulée
                ind         = (z_sampled[:, k] < 0).astype(float)
                sigma2_prev = omega + (alpha + gamma * ind) * eps_k**2 + beta * sigma2_prev
            else:
                sigma2_prev = omega + alpha * eps_k**2 + beta * sigma2_prev
            sigma2_prev = np.maximum(sigma2_prev, 1e-12)

    return returns


# ── Diagnostics iid ───────────────────────────────────────────────────────────

def _iid_diagnostics(z_pool: np.ndarray, lags: int = 10) -> dict:
    """
    Tests d'adéquation iid sur les résidus standardisés z_pool.

    FHS suppose z_t iid. Deux tests :
    1. Ljung-Box sur z²   (Box & Pierce 1970) — autocorrélation d'ordre 2 = ARCH résiduel
    2. Engle-Ng joint (1993)                  — biais de signe résiduel

    Si l'un rejette (p < 0.05) → residus_iid_ok = False.
    Avertissement PDF section 8 : "clustering résiduel non capté → VaR FHS H>1 peut
    être sous-estimée."

    References
    ----------
    Box, G.E.P. & Pierce, D.A. (1970). JASA, 65:332.
    Engle, R.F. & Ng, V.K. (1993). Journal of Finance, 48(5), 1749–1778.
    """
    lb_z2_pval    = 1.0
    engle_ng_pval = 1.0

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_res     = acorr_ljungbox(z_pool**2, lags=[lags], return_df=True)
        lb_z2_pval = float(lb_res['lb_pvalue'].iloc[-1])
    except Exception as exc:
        logger.debug("FHS iid LB(z2) echoue : %s", exc)

    try:
        from tickerlab.core.rapport._stats import sign_bias_test
        sbt           = sign_bias_test(z_pool)
        engle_ng_pval = float(sbt.get('joint_pval', 1.0))
    except Exception as exc:
        logger.debug("FHS iid Engle-Ng echoue : %s", exc)

    return {
        'residus_iid_ok': bool((lb_z2_pval >= 0.05) and (engle_ng_pval >= 0.05)),
        'lb_z2_pval':     lb_z2_pval,
        'engle_ng_pval':  engle_ng_pval,
    }


# ── Estimation principale (mode production) ───────────────────────────────────

def calculer_var_fhs(
    rendements: pd.Series,
    garch_final,
    config: dict,
    niveaux: Optional[list] = None,
) -> Optional[dict]:
    """
    Filtered Historical Simulation — VaR & ES multi-horizons.

    Triggering : retourne None si fhs.enabled = false dans config.yaml.

    La propagation GARCH est vectorisée sur l'axe n_boot ; la boucle séquentielle
    reste sur H (récurrence temporelle intrinsèque). Avec n_boot=10 000 et H=22 :
    temps typique 3–5 s (vs 30–60 s en boucle Python pure).

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements (non utilisé directement — résidus extraits de garch_final).
    garch_final : ARCHModelResult
        GARCH estimé sur la série complète (résidus = série complète).
    config : dict
        Configuration pipeline — lit la section ``fhs``.
    niveaux : list of float, optional
        Niveaux de confiance. Défaut : config['var']['niveaux'] ou [0.90, 0.95, 0.99].

    Returns
    -------
    dict or None
        None si fhs.enabled = false.
        Clés (quand non None) :

        ``var_fhs_par_horizon``
            {H: {alpha: VaR}} — VaR FHS par horizon et niveau.
        ``es_fhs_par_horizon``
            {H: {alpha: ES}} — ES FHS.
        ``n_boot``
            Nombre de simulations utilisées.
        ``horizons``
            Liste des horizons simulés.
        ``residus_iid_ok``
            bool — True si LB(z²) et Engle-Ng ne rejettent pas à 5%.
        ``engle_ng_pval``
            p-valeur Engle-Ng joint.
        ``lb_z2_pval``
            p-valeur LB(z²) sur `lags` retards.
        ``fenetre_residus``
            dict de traçabilité — {date_debut, date_fin, n_residus, z_min, z_max}.

    References
    ----------
    Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999).
        Journal of Futures Markets, 19(5), 583–602.
    """
    fhs_cfg = config.get('fhs', {})
    if not fhs_cfg.get('enabled', False):
        logger.debug("FHS desactive (fhs.enabled=false).")
        return None

    n_boot   = int(fhs_cfg.get('n_boot',    10_000))
    horizons = list(fhs_cfg.get('horizons', [1, 5, 10, 22]))
    seed_raw = fhs_cfg.get('seed', 42)
    seed     = int(seed_raw) if seed_raw is not None else None

    if niveaux is None:
        niveaux = config.get('var', {}).get('niveaux', [0.90, 0.95, 0.99])

    # ── Résidus standardisés — SERIE COMPLETE (mode production) ──────────────
    try:
        resid  = np.asarray(garch_final.resid,                  dtype=float)
        vol    = np.asarray(garch_final.conditional_volatility,  dtype=float)
        z      = resid / vol
        mask   = np.isfinite(z)
        z_pool = z[mask]
    except Exception as exc:
        logger.warning("FHS: residus inaccessibles : %s", exc)
        return None

    assert len(z_pool) > 100, (
        f"FHS: pool trop petit ({len(z_pool)} résidus <= 100) — estimation impossible."
    )

    # ── Fenêtre de résidus (traçabilité audit) ────────────────────────────────
    try:
        dates      = garch_final.conditional_volatility.index[mask]
        date_debut = str(dates[0].date())  if hasattr(dates[0],  'date') else str(dates[0])
        date_fin   = str(dates[-1].date()) if hasattr(dates[-1], 'date') else str(dates[-1])
    except Exception:
        date_debut = date_fin = 'n/a'

    fenetre_residus = {
        'date_debut': date_debut,
        'date_fin':   date_fin,
        'n_residus':  int(len(z_pool)),
        'z_min':      float(np.min(z_pool)),
        'z_max':      float(np.max(z_pool)),
    }

    # ── Diagnostics iid ───────────────────────────────────────────────────────
    diag = _iid_diagnostics(z_pool)
    if not diag['residus_iid_ok']:
        logger.warning(
            "FHS: residus_iid_ok=False (Engle-Ng p=%.4f, LB(z2) p=%.4f). "
            "Clustering residuel non capte — VaR FHS H>1 peut etre sous-estimee.",
            diag['engle_ng_pval'], diag['lb_z2_pval'],
        )

    # ── État GARCH et simulation ──────────────────────────────────────────────
    state = _extract_garch_state(garch_final)
    H_max = max(horizons)
    rng   = np.random.default_rng(seed)

    # Tirage avec remise : (n_boot, H_max) — toutes les innovations d'un coup
    z_sampled = rng.choice(z_pool, size=(n_boot, H_max), replace=True)

    logger.info(
        "FHS: n_boot=%d, H_max=%d, horizons=%s, seed=%s, n_pool=%d, model=%s",
        n_boot, H_max, horizons, seed, len(z_pool), state['vol_class'],
    )

    # Propagation : boucle sur H, vectorisé sur n_boot
    returns_matrix = _propagate_paths(z_sampled, state)   # (n_boot, H_max)

    # ── VaR & ES par horizon et niveau ────────────────────────────────────────
    var_par_horizon: dict = {}
    es_par_horizon:  dict = {}

    for H in horizons:
        cum_ret = returns_matrix[:, :H].sum(axis=1)   # (n_boot,) — rendement H-jours
        var_par_horizon[H] = {}
        es_par_horizon[H]  = {}

        for alpha in niveaux:
            q      = float(np.quantile(cum_ret, 1.0 - alpha))
            tail   = cum_ret[cum_ret <= q]
            es_val = float(tail.mean()) if len(tail) > 0 else q
            var_par_horizon[H][alpha] = round(q,      6)
            es_par_horizon[H][alpha]  = round(es_val, 6)

    return {
        'var_fhs_par_horizon': var_par_horizon,
        'es_fhs_par_horizon':  es_par_horizon,
        'n_boot':              n_boot,
        'horizons':            horizons,
        **diag,
        'fenetre_residus':     fenetre_residus,
    }


# ── One-step-ahead FHS — mode backtest OOS ───────────────────────────────────

def fhs_var_oos(
    z_pool: np.ndarray,
    vol_oos: np.ndarray,
    mu: float,
    niveaux: list,
) -> dict:
    """
    VaR & ES FHS one-step-ahead — déterministe, équivalent n_boot → ∞.

    Pour H=1, la distribution simulée de r_{t+1} = μ + z_τ·σ_{t+1} avec
    z_τ ~ z_pool est entièrement déterminée par les quantiles empiriques de z_pool
    (σ_{t+1} est scalaire et connu). La VaR exacte est donc :

        VaR_α(t+1) = μ + σ_{t+1} · Q_{1−α}(z_pool)
        ES_α(t+1)  = μ + σ_{t+1} · mean(z_pool[z_pool ≤ Q_{1−α}(z_pool)])

    Vectorisé sur T_eff_dyn dates simultanément — O(T_eff_dyn · |niveaux|).

    Parameters
    ----------
    z_pool : np.ndarray
        # z_pool ici = résidus standardisés du GARCH estimé sur le TRAIN,
        # strictement {z_τ : τ ≤ T_train − 1}. AUCUN résidu post-train.
        # Cf. Barone-Adesi, Giannopoulos & Vosper (1999), Section 3.
        # Frontière train/test respectée — anti look-ahead bias.
    vol_oos : np.ndarray, shape (T_eff_dyn,)
        Volatilité conditionnelle OOS σ_{t+1} (one-step-ahead, déjà calculée).
    mu : float
        Paramètre de tendance μ estimé sur le train.
    niveaux : list of float
        Niveaux de confiance α.

    Returns
    -------
    dict : {alpha: {'var': np.ndarray, 'es': np.ndarray}} par niveau.
    """
    result: dict = {}
    for alpha in niveaux:
        q_z    = float(np.quantile(z_pool, 1.0 - alpha))
        tail_z = z_pool[z_pool <= q_z]
        es_z   = float(tail_z.mean()) if len(tail_z) > 0 else q_z

        # Vecteurs OOS de longueur T_eff_dyn
        result[alpha] = {
            'var': mu + q_z  * vol_oos,
            'es':  mu + es_z * vol_oos,
        }
    return result
