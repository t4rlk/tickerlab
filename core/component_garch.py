# -*- coding: utf-8 -*-
"""
Component GARCH (Engle & Lee, 1999) — décomposition permanente/transitoire.

The CGARCH model decomposes the conditional variance σ²_t into:
  - Permanent component  q_t        (long-run trend, driven by ρ)
  - Transitory component σ²_t - q_t (short-run deviation, driven by α, β)

Recursion (Engle & Lee 1999, eq. 7, p. 482) :
  q_t   = ω + ρ · q_{t-1} + φ · (ε²_{t-1} − σ²_{t-1})
  σ²_t  = q_t + α · (ε²_{t-1} − q_{t-1}) + β · (σ²_{t-1} − q_{t-1})

Constraints (non-negotiable for economic interpretability) :
  (C1)  α + β < 1       stationarity of the transitory component
  (C2)  0 < ρ < 1       stationarity of the permanent component
  (C3)  α + β < ρ       component separation — without this the model loses
                         its permanent/transitory interpretation

Estimation : MLE with Student-t innovations (ν estimated, ν > 2).
Optimizer  : SLSQP (scipy.optimize.minimize) — required for non-box
             inequality constraints (C1)–(C3).

Triggering logic (config.component_garch) :
  - enabled = false  → never estimate
  - enabled = true AND (force_estimation = true OR near_igarch = true)
    → estimate

References
----------
Engle, R.F. & Lee, G.G.J. (1999). A Permanent and Transitory Component Model
    of Stock Return Volatility. In R.F. Engle & H. White (Eds.), Cointegration,
    Causality and Forecasting: A Festschrift in Honour of Clive W.J. Granger
    (pp. 475–497). Oxford University Press.
"""
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as _t_dist

# '2' : ajout de la detection de degenerescence (cles degenerate/degeneracies).
# Les entrees ecrites en version '1' ne portent pas ces cles et produiraient un
# rapport sans alerte : le bump de version les rend inatteignables.
__etape_version__ = '2'
logger = logging.getLogger(__name__)

_PARAM_NAMES = ('omega', 'rho', 'phi', 'alpha', 'beta', 'nu')

# ── Seuils de dégénérescence ──────────────────────────────────────────────────
# Une solution collée à une borne du domaine satisfait formellement (C1)-(C3)
# sans constituer une décomposition permanente/transitoire interprétable.
# Ces seuils sont DISTINCTS de saturation_warning, qui ne surveille que la
# proximité de la frontière C3 (|α+β − ρ| < 1e-3).
_BORNE_BETA_BASSE = 1e-6     # β sur sa borne basse (bounds : 1e-8)
_BORNE_RHO_HAUTE  = 0.9995   # ρ sur sa borne haute (bounds : 0.9999)
_BORNE_NU_BASSE   = 2.5      # ν sur sa borne basse (bounds : 2.01)


# ── Recursion ──────────────────────────────────────────────────────────────────

def _recursion(
    omega: float, rho: float, phi: float,
    alpha: float, beta: float,
    eps: np.ndarray,
) -> tuple:
    """
    Component GARCH variance recursion.

    Parameters
    ----------
    omega, rho, phi, alpha, beta : float
        Model parameters.
    eps : np.ndarray, shape (T,)
        Residuals ε_t.

    Returns
    -------
    q      : np.ndarray, shape (T,)   Permanent component q_t.
    sigma2 : np.ndarray, shape (T,)   Conditional variance σ²_t.
    """
    T = len(eps)
    eps2 = eps ** 2
    q = np.empty(T)
    sigma2 = np.empty(T)

    # Initialisation à la variance inconditionnelle
    q0 = omega / max(1.0 - rho, 1e-10)
    q0 = max(q0, 1e-8)
    q[0] = q0
    sigma2[0] = q0

    for t in range(1, T):
        q[t] = omega + rho * q[t - 1] + phi * (eps2[t - 1] - sigma2[t - 1])
        sigma2[t] = (q[t]
                     + alpha * (eps2[t - 1] - q[t - 1])
                     + beta  * (sigma2[t - 1] - q[t - 1]))
        q[t]      = max(q[t],      1e-8)
        sigma2[t] = max(sigma2[t], 1e-8)

    return q, sigma2


# ── Log-vraisemblance ─────────────────────────────────────────────────────────

def _neg_loglik(params: np.ndarray, eps: np.ndarray) -> float:
    """
    Negative Student-t log-likelihood for Component GARCH.

    l_t = log_c − ½ log σ²_t − (ν+1)/2 · log(1 + ε²_t / (σ²_t · (ν−2)))
    log_c = ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln(π · (ν−2))

    Returns 1e10 on any numerical failure.
    """
    omega, rho, phi, alpha, beta, nu = params
    if nu <= 2.0:
        return 1e10

    _, sigma2 = _recursion(omega, rho, phi, alpha, beta, eps)
    eps2 = eps ** 2

    try:
        log_c = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                 - 0.5 * math.log(math.pi * (nu - 2.0)))
    except Exception:
        return 1e10

    ll = 0.0
    for t in range(len(eps)):
        s2 = sigma2[t]
        if s2 <= 0.0:
            return 1e10
        z2 = eps2[t] / s2
        val = log_c - 0.5 * math.log(s2) - (nu + 1.0) / 2.0 * math.log(1.0 + z2 / (nu - 2.0))
        if not math.isfinite(val):
            return 1e10
        ll += val

    return -ll


# ── Warmstart ─────────────────────────────────────────────────────────────────

def _warmstart(garch_final) -> np.ndarray:
    """
    Derive a feasible initial guess from the GARCH(p,q) result.

    Strategy
    --------
    - α₀, β₀   : from GARCH α[1]/β[1] (clipped to safe range)
    - ρ₀       : α₀ + β₀ + 0.04 (strictly above α+β → satisfies C3)
    - ω₀       : preserves approximate unconditional variance
    - φ₀       : 0.01 (small, neutral)
    - ν₀       : from GARCH ν if Student-t, else 8.0 (clipped ≥ 4)

    Returns
    -------
    np.ndarray, shape (6,)  — (omega, rho, phi, alpha, beta, nu)
    """
    params = garch_final.params
    idx = list(params.index)

    alpha0 = float(params['alpha[1]']) if 'alpha[1]' in idx else 0.05
    beta0  = float(params['beta[1]'])  if 'beta[1]'  in idx else 0.85
    nu0    = float(params['nu'])        if 'nu'       in idx else 8.0
    omega0 = float(params.get('omega', params.iloc[1] if len(params) > 1 else 0.01))

    alpha0 = float(np.clip(alpha0, 0.01, 0.25))
    beta0  = float(np.clip(beta0,  0.10, 0.90))
    nu0    = max(nu0, 4.0)

    ab = alpha0 + beta0
    if ab >= 0.98:
        alpha0, beta0 = 0.05, 0.85
        ab = 0.90

    # C3 must hold at warmstart (rho0 > alpha0+beta0 strictly).
    # Projection explicite : rho0 = max(ab + 0.01, 0.97) pour garantir que
    # la composante permanente est plus persistante que la transitoire, puis
    # on laisse l'optimiseur affiner. Le min(., 0.9999) respecte la borne haute.
    rho0 = min(max(ab + 0.01, 0.97), 0.9999)                             # C3: ρ > α+β
    omega0_cg = max(float(omega0) * (1.0 - rho0) / max(1.0 - ab, 1e-6), 1e-6)
    phi0 = 0.01

    return np.array([omega0_cg, rho0, phi0, alpha0, beta0, nu0])


# ── Estimation principale ─────────────────────────────────────────────────────

def estimer_component_garch(
    rendements: pd.Series,
    garch_final,
    config: dict,
    near_igarch: bool = False,
) -> Optional[dict]:
    """
    Estimate Component GARCH (Engle & Lee 1999) — permanent/transitory decomposition.

    Triggering logic
    ----------------
    - ``config.component_garch.enabled = False`` → return None immediately.
    - ``enabled = True`` AND (``force_estimation = True`` OR ``near_igarch = True``)
      → estimate and return results dict.
    - ``enabled = True`` AND ``force_estimation = False`` AND ``near_igarch = False``
      → return None (mean-reverting series, no need to decompose).

    Parameters
    ----------
    rendements : pd.Series
        Log-return series (same as used for GARCH estimation).
    garch_final : ARCHModelResult
        Fitted GARCH model — used for residuals and warmstart.
    config : dict
        Global pipeline configuration. Reads ``component_garch`` sub-dict.
    near_igarch : bool, optional
        True if IGARCH diagnostic flagged ``near_igarch`` or ``igarch_strict``.
        Controls conditional triggering when ``force_estimation = False``.

    Returns
    -------
    dict or None
        Keys (when not None) :

        ``omega``, ``rho``, ``phi``, ``alpha``, ``beta``, ``nu``
            Point estimates.
        ``std_errors``
            dict — keys same as params, values are asymptotic standard errors
            (NaN if Hessian non-invertible).
        ``pvalues``
            dict — two-sided t-test p-values (NaN if std_err unavailable).
        ``constraints_ok``
            dict with booleans : ``alpha_plus_beta_lt_1``, ``rho_in_01``,
            ``separation`` (C3). ``separation`` exige en outre une composante
            transitoire non dégénérée : avec β ≈ 0 l'inégalité α+β < ρ est
            obtenue par effondrement, et ``separation`` vaut False.
        ``aic``, ``bic``, ``loglik``
            Information criteria and log-likelihood.
        ``phi_warning``
            bool — True if |φ̂| > 0.5 (Engle & Lee 1999 warn of instability).
        ``saturation_warning``
            bool — True if |α̂+β̂ − ρ̂| < 1e-3 (constraint C3 nearly active).
        ``degenerate``
            bool — True si au moins un paramètre est écrasé contre une borne
            du domaine. Un modèle dégénéré ne doit PAS être présenté comme une
            décomposition permanente/transitoire valide.
        ``degeneracies``
            list of str — libellés explicites des dégénérescences détectées
            (vide si aucune).
        ``beta_degenerate``, ``rho_at_bound``, ``nu_at_bound``
            bool — critères individuels : β < 1e-6 (composante transitoire
            disparue), ρ > 0.9995, ν < 2.5.
        ``q_t``
            pd.Series — estimated permanent component (same index as garch_final).
        ``sigma2_t``
            pd.Series — estimated conditional variance σ²_t.
        ``n_obs``
            int — number of observations used.
        ``converged``
            bool — SLSQP convergence flag.

    References
    ----------
    Engle, R.F. & Lee, G.G.J. (1999). A Permanent and Transitory Component Model
        of Stock Return Volatility. In R.F. Engle & H. White (Eds.),
        Cointegration, Causality and Forecasting (pp. 475–497). Oxford Univ. Press.
    """
    cfg = config.get('component_garch', {})
    if not cfg.get('enabled', False):
        logger.debug("Component GARCH desactive (component_garch.enabled=false).")
        return None

    force = cfg.get('force_estimation', False)
    if not force and not near_igarch:
        logger.debug(
            "Component GARCH: near_igarch=%s, force_estimation=%s — ignore.",
            near_igarch, force,
        )
        return None

    # ── Résidus GARCH ────────────────────────────────────────────────────────
    try:
        eps = np.asarray(garch_final.resid, dtype=float)
        eps = eps[np.isfinite(eps)]
    except Exception as exc:
        logger.warning("Component GARCH: residus inaccessibles : %s", exc)
        return None

    n_obs = len(eps)
    if n_obs < 100:
        logger.warning("Component GARCH: serie trop courte (n=%d < 100).", n_obs)
        return None

    # ── Point de départ ───────────────────────────────────────────────────────
    theta0 = _warmstart(garch_final)

    # ── Contraintes SLSQP ────────────────────────────────────────────────────
    # idx  : 0=omega  1=rho  2=phi  3=alpha  4=beta  5=nu
    # (C1) α + β < 1  →  1 − α − β > 0
    # (C2) 0 < ρ < 1  →  ρ > 0  (borne basse dans bounds) + ρ < 1 (borne haute)
    # (C3) α + β < ρ  →  ρ − α − β > 0   (NON-NÉGOCIABLE)
    # (C4) ν > 2      →  ν − 2 > 0
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[3] - x[4] - 1e-6},   # (C1)
        {'type': 'ineq', 'fun': lambda x: x[1] - x[3] - x[4] - 1e-6},  # (C3)
        {'type': 'ineq', 'fun': lambda x: x[5] - 2.0 - 1e-6},           # (C4)
    ]
    bounds = [
        (1e-8,  None),    # omega > 0
        (1e-4,  0.9999),  # 0 < ρ < 1  (C2)
        (None,  None),    # phi — peut être négatif (asymétrie cyclique)
        (1e-8,  None),    # alpha ≥ 0
        (1e-8,  None),    # beta  ≥ 0
        (2.01,  200.0),   # ν > 2
    ]

    logger.info(
        "Component GARCH: SLSQP (n=%d, theta0=[ω=%.2e, ρ=%.3f, φ=%.3f, "
        "α=%.3f, β=%.3f, ν=%.1f])...",
        n_obs, *theta0,
    )

    try:
        res = minimize(
            _neg_loglik,
            theta0,
            args=(eps,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False},
        )
    except Exception as exc:
        logger.warning("Component GARCH: echec SLSQP : %s", exc)
        return None

    # ── Multi-start inconditionnel : 3 points déterministes ─────────────────
    # Couvre trois bassins d'attraction distincts :
    # (A) rho=0.99, ab=warmstart  → actifs avec forte composante transitoire (^GSPC)
    # (B) rho=0.97, ab=warmstart  → point de secours intermédiaire
    # (C) rho=0.99, ab=0.03       → actifs FX/low-shock (EURUSD), composante transitoire faible
    _extra_starts = [
        np.array([theta0[0], 0.9900, 0.010, theta0[3],  theta0[4],  theta0[5]]),  # (A)
        np.array([theta0[0], 0.9700, 0.020, theta0[3],  theta0[4],  theta0[5]]),  # (B)
        np.array([theta0[0], 0.9900, 0.010, 0.020,      0.010,      theta0[5]]),  # (C)
    ]
    for theta_p in _extra_starts:
        # Garantir C3 et ν > 2 sur chaque point de départ additionnel
        ab_p = float(theta_p[3] + theta_p[4])
        if theta_p[1] <= ab_p:
            theta_p[1] = min(ab_p + 0.02, 0.9999)
        theta_p[5] = max(float(theta_p[5]), 4.0)
        try:
            res2 = minimize(
                _neg_loglik, theta_p, args=(eps,),
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False},
            )
            if res2.fun < res.fun:
                res = res2
                logger.info(
                    "Component GARCH: multi-start ameliore la solution "
                    "(fun=%.6g, rho=%.4f, phi=%.4f).",
                    res.fun, res.x[1], res.x[2],
                )
        except Exception as exc_r:
            logger.debug("Component GARCH: point additionnel echoue : %s", exc_r)

    if not res.success:
        logger.warning("Component GARCH: SLSQP non converge : %s", res.message)

    omega_h, rho_h, phi_h, alpha_h, beta_h, nu_h = res.x
    loglik   = -res.fun
    n_params = 6
    aic      = -2.0 * loglik + 2.0 * n_params
    bic      = -2.0 * loglik + n_params * math.log(n_obs)

    # ── Erreurs standard (Hessien numérique — différences finies centrées) ───
    se_dict  = {k: float('nan') for k in _PARAM_NAMES}
    pval_dict = {k: float('nan') for k in _PARAM_NAMES}
    try:
        h = 1e-4
        n_p = len(res.x)
        hess = np.zeros((n_p, n_p))
        for i in range(n_p):
            for j in range(i, n_p):
                ei = np.zeros(n_p); ei[i] = h
                ej = np.zeros(n_p); ej[j] = h
                val = (
                    _neg_loglik(res.x + ei + ej, eps)
                    - _neg_loglik(res.x + ei - ej, eps)
                    - _neg_loglik(res.x - ei + ej, eps)
                    + _neg_loglik(res.x - ei - ej, eps)
                ) / (4.0 * h * h)
                hess[i, j] = val
                hess[j, i] = val

        cov = np.linalg.inv(hess)
        se  = np.sqrt(np.maximum(np.diag(cov), 0.0))

        df_resid = max(n_obs - n_params, 1)
        for i, name in enumerate(_PARAM_NAMES):
            se_dict[name]   = float(se[i])
            if se[i] > 0.0:
                t_stat          = float(res.x[i]) / se[i]
                pval_dict[name] = float(2.0 * _t_dist.sf(abs(t_stat), df=df_resid))
    except Exception as exc:
        logger.debug("Component GARCH: Hessien non inversible : %s", exc)

    # ── Détection des solutions dégénérées ───────────────────────────────────
    # Distincte de saturation_warning : on ne teste pas la proximité de C3 mais
    # l'écrasement de la solution contre une borne du domaine. Une telle
    # solution n'est PAS une décomposition permanente/transitoire valide, même
    # lorsque (C1)-(C3) sont formellement satisfaites.
    ab = alpha_h + beta_h

    beta_degenerate = bool(beta_h < _BORNE_BETA_BASSE)
    rho_at_bound    = bool(rho_h  > _BORNE_RHO_HAUTE)
    nu_at_bound     = bool(nu_h   < _BORNE_NU_BASSE)

    degeneracies = []
    if beta_degenerate:
        degeneracies.append(
            f"beta = {beta_h:.3e} sur sa borne basse (< {_BORNE_BETA_BASSE:g}) : "
            f"composante transitoire disparue, le modele degenere en GARCH simple"
        )
    if rho_at_bound:
        degeneracies.append(
            f"rho = {rho_h:.6f} sur sa borne haute (> {_BORNE_RHO_HAUTE:g}) : "
            f"composante permanente contre le bord du domaine"
        )
    if nu_at_bound:
        degeneracies.append(
            f"nu = {nu_h:.4f} proche de sa borne basse (< {_BORNE_NU_BASSE:g}) : "
            f"degres de liberte a la limite"
        )
    degenerate = bool(degeneracies)

    # ── Vérification des contraintes ─────────────────────────────────────────
    # C3 : la séparation suppose une composante transitoire NON dégénérée.
    # Avec β ≈ 0, l'inégalité α+β < ρ est obtenue par effondrement de la
    # composante transitoire, pas par une séparation réelle : on ne valide pas.
    constraints_ok = {
        'alpha_plus_beta_lt_1': bool(ab < 1.0),
        'rho_in_01':            bool(0.0 < rho_h < 1.0),
        'separation':           bool(ab < rho_h and not beta_degenerate),
    }

    # ── Avertissements ───────────────────────────────────────────────────────
    phi_warning        = bool(abs(phi_h) > 0.5)
    saturation_warning = bool(abs(ab - rho_h) < 1e-3)

    for _lib in degeneracies:
        logger.warning("Component GARCH: solution degeneree — %s", _lib)
    if degenerate:
        logger.warning(
            "Component GARCH: la decomposition permanente/transitoire n'est PAS "
            "interpretable pour cette serie (%d critere(s) de degenerescence).",
            len(degeneracies),
        )

    if phi_warning:
        logger.warning(
            "Component GARCH: |phi|=%.4f > 0.5 — potentielle instabilite "
            "(Engle & Lee 1999).", phi_h,
        )
    if saturation_warning:
        logger.warning(
            "Component GARCH: |alpha+beta - rho|=%.2e < 1e-3 — "
            "contrainte de separation C3 saturee.", abs(ab - rho_h),
        )

    # ── Séries q_t et σ²_t ────────────────────────────────────────────────────
    q_arr, sigma2_arr = _recursion(omega_h, rho_h, phi_h, alpha_h, beta_h, eps)

    try:
        dates = garch_final.conditional_volatility.index[-n_obs:]
    except Exception:
        dates = pd.RangeIndex(n_obs)

    q_t      = pd.Series(q_arr,      index=dates, name='q_t')
    sigma2_t = pd.Series(sigma2_arr, index=dates, name='sigma2_t')

    return {
        'omega':              float(omega_h),
        'rho':                float(rho_h),
        'phi':                float(phi_h),
        'alpha':              float(alpha_h),
        'beta':               float(beta_h),
        'nu':                 float(nu_h),
        'std_errors':         se_dict,
        'pvalues':            pval_dict,
        'constraints_ok':     constraints_ok,
        'aic':                float(aic),
        'bic':                float(bic),
        'loglik':             float(loglik),
        'phi_warning':        phi_warning,
        'saturation_warning': saturation_warning,
        'degenerate':         degenerate,
        'degeneracies':       degeneracies,
        'beta_degenerate':    beta_degenerate,
        'rho_at_bound':       rho_at_bound,
        'nu_at_bound':        nu_at_bound,
        'q_t':                q_t,
        'sigma2_t':           sigma2_t,
        'n_obs':              n_obs,
        'converged':          bool(res.success),
    }
