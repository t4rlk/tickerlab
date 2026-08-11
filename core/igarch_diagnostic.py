# -*- coding: utf-8 -*-
"""
Diagnostic IGARCH post-hoc sur le modèle GARCH estimé.

Approche : au lieu d'ajouter IGARCH à la grille de sélection (approche initiale
de la sous-tâche 1.2), ce module diagnostique si le modèle DÉJÀ RETENU est à la
frontière IGARCH. Ce changement de cap est justifié par Mikosch & Stărică (2004) :
l'apparence IGARCH est souvent un artefact de ruptures structurelles dans la variance,
et non une propriété intrinsèque du processus. Estimer un IGARCH séparé et le
sélectionner par AIC conduirait à un faux positif systématique sur des matières
premières à crises récurrentes. Le diagnostic post-hoc fournit une alerte sans biaiser
la sélection de modèle.

References
----------
Engle, R.F. & Bollerslev, T. (1986). Modelling the persistence of conditional variances.
    Econometric Reviews, 5(1), 1–50.
Lamoureux, C.G. & Lastrapes, W.D. (1990). Persistence in variance, structural change,
    and the GARCH model. Journal of Business & Economic Statistics, 8(2), 225–234.
Mikosch, T. & Stărică, C. (2004). Nonstationarities in financial time series, the
    long-range dependence, and the IGARCH effects. Review of Economics and Statistics,
    86(1), 378–390.
Nelson, D.B. (1991). Conditional heteroskedasticity in asset returns: A new approach.
    Econometrica, 59(2), 347–370.
"""
import math
import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _s_chi2

from tickerlab.core.rapport._stats import persistance_garch

__etape_version__ = '1'
logger = logging.getLogger(__name__)

# ── Conversion fréquence → jours calendaires ─────────────────────────────────
# Daily : trading days (5/week) → ×(7/5)=1.4 pour obtenir jours calendaires.
# Weekly, monthly : conversion directe.
_FREQ_TO_JOURS_CAL: dict[str, float] = {
    'daily':   7 / 5,    # marché 5j/semaine → ×1.4 (weekends absents dans les données)
    'weekly':  7.0,
    'monthly': 30.44,
}

_FREQ_LABEL: dict[str, str] = {
    'daily':   'jours',
    'weekly':  'semaines',
    'monthly': 'mois',
}


# ── Gradients analytiques de la persistance ──────────────────────────────────

def _gradient_analytique(params: pd.Series, vol_type: str) -> np.ndarray:
    """
    Gradient analytique de la persistance par rapport aux paramètres du modèle.

    La persistance est définie par famille (Bollerslev 1986, Nelson 1991,
    Glosten–Jagannathan–Runkle 1993) :

    - GARCH      : Σ α_i + Σ β_j       → ∂/∂α_i = 1, ∂/∂β_j = 1
    - GJR-GARCH  : Σ α_i + Σ β_j + ½Σ γ_i → ∂/∂γ_i = 0.5
    - EGARCH     : Σ β_j seulement (Nelson 1991, eq.10) → ∂/∂α_i = 0, ∂/∂γ_i = 0

    Parameters
    ----------
    params : pd.Series
        Paramètres estimés (index = noms, valeurs = estimations MLE).
    vol_type : str
        Famille du modèle volatilité.

    Returns
    -------
    np.ndarray
        Vecteur gradient de même taille que ``params``.
    """
    keys = list(params.index)
    g = np.zeros(len(keys))
    for i, k in enumerate(keys):
        if vol_type == 'EGARCH':
            if k.startswith('beta['):
                g[i] = 1.0
        elif vol_type == 'GJR-GARCH':
            if k.startswith('alpha['):
                g[i] = 1.0
            elif k.startswith('beta['):
                g[i] = 1.0
            elif k.startswith('gamma['):
                g[i] = 0.5
        else:  # GARCH standard
            if k.startswith('alpha['):
                g[i] = 1.0
            elif k.startswith('beta['):
                g[i] = 1.0
    return g


def _gradient_numerique(
    params: pd.Series,
    vol_type: str,
    delta_fd: float = 1e-5,
) -> np.ndarray:
    """
    Gradient numérique (différences finies centrées) pour TGARCH/APARCH.

    Nécessaire car kappa(δ, γ) dans la formule de persistance APARCH (Ding,
    Granger & Engle 1993, eq. 3.7) n'a pas de forme analytique simple.

    Parameters
    ----------
    params : pd.Series
        Paramètres estimés.
    vol_type : str
        Famille TGARCH ou APARCH.
    delta_fd : float
        Pas des différences finies (défaut 1e-5).

    Returns
    -------
    np.ndarray
        Vecteur gradient approché.
    """
    params_dict = dict(params)
    g = np.zeros(len(params))
    for i, k in enumerate(params.index):
        p_plus = dict(params_dict)
        p_plus[k] += delta_fd
        p_minus = dict(params_dict)
        p_minus[k] -= delta_fd
        pers_plus  = persistance_garch(vol_type, p_plus)
        pers_minus = persistance_garch(vol_type, p_minus)
        if not (math.isnan(pers_plus) or math.isnan(pers_minus)):
            g[i] = (pers_plus - pers_minus) / (2 * delta_fd)
    return g


# ── Diagnostic principal ──────────────────────────────────────────────────────

def diagnostiquer_igarch(
    garch_final,
    vol_type: str,
    seuil_igarch: float = 0.98,
    frequence_serie: str = 'daily',
) -> dict:
    """
    Diagnostic IGARCH post-hoc sur le modèle GARCH retenu.

    Calcule persistance, demi-vie (en périodes natives ET en jours calendaires),
    test de Wald H₀: persistance = 1 (delta méthode) et classification.

    Note sur la demi-vie en jours calendaires : pour une série quotidienne de
    marché, les weekends sont absents des données → la conversion utilise le
    facteur (7/5) = 1.4 et non 1.0. Pour du hebdomadaire, le facteur est 7.

    Parameters
    ----------
    garch_final : ARCHModelResult
        Résultat de l'estimation GARCH (arch package).
    vol_type : str
        Famille du modèle : ``'GARCH'``, ``'GJR-GARCH'``, ``'EGARCH'``,
        ``'TGARCH'``, ``'APARCH'``.
    seuil_igarch : float, optional
        Seuil de persistance pour flag near-IGARCH (défaut 0.98, zone
        Engle & Bollerslev 1986 / Lamoureux & Lastrapes 1990).
    frequence_serie : str, optional
        Fréquence de la série : ``'daily'``, ``'weekly'``, ``'monthly'``.
        Utilisée pour la conversion de la demi-vie en jours calendaires.

    Returns
    -------
    dict
        Clés :

        - ``persistance``       : float — persistance du modèle
        - ``half_life_periodes``: float|None — demi-vie en périodes natives ;
          ``None`` si persistance ≥ 1 (non-stationnaire)
        - ``half_life_jours_cal``: float|None — demi-vie en jours calendaires ;
          ``None`` si non-stationnaire
        - ``frequence_serie``  : str — traçabilité de la fréquence utilisée
        - ``near_igarch``      : bool — True si code ≠ 'mean_reverting'
        - ``wald_stat``        : float — statistique de Wald χ²(1) ; NaN si indisponible
        - ``wald_pval``        : float — p-valeur du test de Wald ; NaN si indisponible
        - ``code``             : str — 'igarch_strict'|'near_igarch'|'mean_reverting'
        - ``message_pedagogique``: str — interprétation formatée pour le rapport

    References
    ----------
    Engle, R.F. & Bollerslev, T. (1986). Modelling the persistence of conditional
        variances. Econometric Reviews, 5(1), 1–50.
    Lamoureux, C.G. & Lastrapes, W.D. (1990). Persistence in variance, structural
        change, and the GARCH model. Journal of Business & Economic Statistics,
        8(2), 225–234.
    Mikosch, T. & Stărică, C. (2004). Nonstationarities in financial time series,
        the long-range dependence, and the IGARCH effects. Review of Economics and
        Statistics, 86(1), 378–390.
    """
    params = garch_final.params
    pers = persistance_garch(vol_type, params)

    # ── Demi-vie ──────────────────────────────────────────────────────────────
    mult_cal = _FREQ_TO_JOURS_CAL.get(frequence_serie, 1.0)
    if math.isnan(pers) or pers >= 1.0:
        half_life_periodes  = None   # sentinel JSON/LaTeX-safe
        half_life_jours_cal = None
    else:
        half_life_periodes  = math.log(0.5) / math.log(pers)
        half_life_jours_cal = half_life_periodes * mult_cal

    # ── Classification ────────────────────────────────────────────────────────
    if math.isnan(pers) or pers >= 1.0:
        code = 'igarch_strict'
    elif pers >= seuil_igarch:
        code = 'near_igarch'
    else:
        code = 'mean_reverting'

    # ── Test de Wald H₀: persistance = 1 (delta méthode) ─────────────────────
    wald_stat = float('nan')
    wald_pval = float('nan')
    try:
        cov = garch_final.param_cov
        if cov is None:
            raise ValueError("param_cov is None")
        cov_vals = cov.values if hasattr(cov, 'values') else np.asarray(cov)
        if np.any(np.isnan(cov_vals)):
            raise ValueError("param_cov contient des NaN")

        if vol_type in ('TGARCH', 'APARCH'):
            g = _gradient_numerique(params, vol_type)
        else:
            g = _gradient_analytique(params, vol_type)

        var_pers = float(g @ cov_vals @ g)
        if var_pers <= 0:
            raise ValueError(f"Var(persistance) ≤ 0 : {var_pers:.6g}")

        wald_stat = float((pers - 1.0) ** 2 / var_pers)
        wald_pval = float(_s_chi2.sf(wald_stat, df=1))
    except Exception as exc:
        logger.debug("Test Wald IGARCH non calculable : %s", exc)

    # ── Message pédagogique ───────────────────────────────────────────────────
    freq_label = _FREQ_LABEL.get(frequence_serie, 'périodes')

    def _hl_str() -> str:
        if half_life_periodes is None:
            return 'non-stationnaire (chocs permanents)'
        return (
            f'{half_life_periodes:.1f} {freq_label} '
            f'(≈{half_life_jours_cal:.0f} j calendaires)'
        )

    def _wald_str() -> str:
        if math.isnan(wald_pval):
            return ''
        return f', Wald χ²(1) p = {wald_pval:.4f}'

    if code == 'igarch_strict':
        message = (
            f'Persistance = {pers:.4f} ≥ 1 — modèle {vol_type} à la frontière IGARCH '
            '(Engle & Bollerslev 1986). La variance conditionnelle est non-stationnaire : '
            'les chocs de volatilité sont permanents. Ce comportement est souvent un '
            'artefact de ruptures structurelles dans la variance plutôt qu\'une propriété '
            'intrinsèque du processus (Mikosch & Stărică 2004, Lamoureux & Lastrapes 1990).'
        )
    elif code == 'near_igarch':
        message = (
            f'Persistance = {pers:.4f} > {seuil_igarch} — zone near-IGARCH{_wald_str()}. '
            f'Demi-vie des chocs de volatilité : {_hl_str()}. '
            'La convergence vers la variance inconditionnelle est très lente ; '
            'les épisodes de volatilité élevée perdurent longtemps après le choc initial '
            '(Lamoureux & Lastrapes 1990). Vérifier l\'absence de ruptures structurelles '
            'non capturées (Mikosch & Stărică 2004).'
        )
    else:
        message = (
            f'Persistance = {pers:.4f} — volatilité mean-reverting. '
            f'Demi-vie : {_hl_str()}.'
        )

    return {
        'persistance':         pers,
        'half_life_periodes':  half_life_periodes,
        'half_life_jours_cal': half_life_jours_cal,
        'frequence_serie':     frequence_serie,
        'near_igarch':         code != 'mean_reverting',
        'wald_stat':           wald_stat,
        'wald_pval':           wald_pval,
        'code':                code,
        'message_pedagogique': message,
    }
