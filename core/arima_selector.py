"""Sélection automatique ARIMA par Box-Jenkins (grille AIC + significativité)."""
import logging

__etape_version__ = '1'

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


def _determiner_interpretation(
    motif: str,
    p_opt: int,
    q_opt: int,
    fit_params: dict,
    lb_pval_rendements: float,
    n_obs: int,
) -> dict:
    """
    Détermine l'interprétation économique de la sélection ARIMA.

    Taxonomie (4 catégories mutuellement exclusives) :

    - ``serie_predictible``          : coefficients significatifs et amplitude > 0.10
    - ``bruit_faible``               : coefficients significatifs mais amplitude ≤ 0.10
      (Lo & MacKinlay, 1988)
    - ``martingale_marche_efficient``: aucun modèle n'améliore le random walk
      (Fama, 1970)
    - ``incertain``                  : échantillon trop court ou cas ambigu

    Parameters
    ----------
    motif : str
        Motif de sélection issu de selectionner_arima() :
        ``'tous_sig + parcimonie'``, ``'tous_sig + AIC'``,
        ``'AIC_seul_AUCUN_SIG'``, ``'random_walk_BA'``.
    p_opt : int
        Ordre AR optimal retenu.
    q_opt : int
        Ordre MA optimal retenu.
    fit_params : dict
        Paramètres estimés du modèle ARIMA retenu (clés : ``'ar.L1'``, ``'ma.L1'``,
        etc.). Dict vide si p_opt == q_opt == 0.
    lb_pval_rendements : float
        P-valeur du test de Ljung-Box (10 lags) sur les rendements bruts.
        Utilisée pour discriminer martingale vs incertain dans le cas
        ``AIC_seul_AUCUN_SIG``.
    n_obs : int
        Nombre d'observations de la série.

    Returns
    -------
    dict
        ``{'code': str, 'message': str}`` où ``code`` est l'une des 4 catégories.

    References
    ----------
    Fama, E.F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work.
        Journal of Finance, 25(2), 383–417.
    Lo, A.W. & MacKinlay, A.C. (1988). Stock market prices do not follow random walks:
        evidence from a simple specification test. Review of Financial Studies, 1(1), 41–66.
    Burnham, K.P. & Anderson, D.R. (2002). Model Selection and Multimodel Inference (2e éd.).
        Springer. Table 2.5.
    """
    # ── Garde : série trop courte ─────────────────────────────────────────────
    if n_obs < 100:
        logger.warning(
            "Échantillon trop court (n=%d < 100) : interprétation ARIMA forcée à 'incertain'.",
            n_obs,
        )
        return {
            'code': 'incertain',
            'message': (
                f"Échantillon trop petit (n={n_obs} < 100). "
                "La puissance des tests de significativité est insuffisante pour "
                "discriminer les catégories. Résultat : non concluant."
            ),
        }

    # ── Magnitude maximale des coefficients AR/MA ─────────────────────────────
    coefs_ar_ma = [v for k, v in fit_params.items() if k.startswith(('ar.', 'ma.'))]
    max_magnitude = max(abs(c) for c in coefs_ar_ma) if coefs_ar_ma else 0.0

    # ── Cas 1 : modèle avec coefficients tous significatifs ───────────────────
    if motif.startswith('tous_sig'):
        if max_magnitude > 0.10:
            return {
                'code': 'serie_predictible',
                'message': (
                    f"Le modèle ARIMA retenu présente des coefficients statistiquement "
                    f"significatifs et économiquement non négligeables "
                    f"(|φ̂, θ̂|_max = {max_magnitude:.3f} > 0.10). "
                    "La série contient une structure dynamique exploitable pour la "
                    "prévision à court terme."
                ),
            }
        # Lo & MacKinlay (1988) : significatif mais amplitude négligeable
        return {
            'code': 'bruit_faible',
            'message': (
                f"Les coefficients sont statistiquement significatifs mais "
                f"économiquement négligeables (|φ̂, θ̂|_max = {max_magnitude:.3f} ≤ 0.10). "
                "Conformément à Lo & MacKinlay (1988), la dépendance linéaire existe "
                "mais son ampleur est trop faible pour générer un avantage prédictif "
                "après frais de transaction."
            ),
        }

    # ── Cas 2 : random walk B&A ou LB non rejeté → efficience faible ─────────
    if motif == 'random_walk_BA' or (
        motif == 'AIC_seul_AUCUN_SIG' and lb_pval_rendements > 0.10
    ):
        return {
            'code': 'martingale_marche_efficient',
            'message': (
                "Aucun modèle ARIMA n'améliore significativement la prévision par rapport "
                "à une marche aléatoire (Burnham & Anderson 2002, ΔAIC < 4 ; "
                f"test de Ljung-Box rendements bruts : p = {lb_pval_rendements:.3f}). "
                "Cette série est compatible avec l'hypothèse d'efficience faible "
                "(Fama, 1970) : les rendements passés ne contiennent pas "
                "d'information prédictive exploitable."
            ),
        }

    # ── Cas résiduel : AIC_seul_AUCUN_SIG avec LB rejeté → ambigu ────────────
    return {
        'code': 'incertain',
        'message': (
            "Aucun modèle ARIMA ne présente de coefficients significatifs, "
            "mais le test de Ljung-Box sur les rendements bruts rejette l'indépendance "
            f"(p = {lb_pval_rendements:.3f} ≤ 0.10). "
            "Une structure dynamique semble présente mais n'est pas capturée par "
            "les modèles ARIMA(p,q) testés. Résultat : non concluant."
        ),
    }


def grid_search_arima(serie, d, p_max=6, q_max=6, seuil_sig=0.5):
    """
    Grille ARIMA(p,d,q) triée par AIC (format EViews) avec filtre de significativité.

    Parameters
    ----------
    serie : array-like
        Série temporelle à modéliser.
    d : int
        Ordre de différenciation.
    p_max : int, optional
        Ordre AR maximal (défaut 6).
    q_max : int, optional
        Ordre MA maximal (défaut 6).
    seuil_sig : float, optional
        Seuil de p-valeur pour la significativité des coefficients (défaut 0.10).

    Returns
    -------
    pd.DataFrame
        Colonnes : p, d, q, AIC, BIC, logL, tous_sig.
        AIC et BIC sont normalisés au format EViews. Trié par AIC croissant.
    """
    resultats = []
    for p in range(0, p_max + 1):
        for q in range(0, q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(serie, order=(p, d, q)).fit()
                n = fit.nobs
                k = len(fit.params)
                logL = fit.llf

                pvals = fit.pvalues
                coefs = [c for c in pvals.index if c.startswith(('ar.', 'ma.'))]
                tous_sig = bool(all(pvals[c] < seuil_sig for c in coefs)) if coefs else False

                aic_eviews = -2 * (logL / n) + 2 * (k / n)
                bic_eviews = -2 * (logL / n) + k * np.log(n) / n

                resultats.append({
                    'p': p, 'd': d, 'q': q,
                    'AIC': round(aic_eviews, 6),
                    'BIC': round(bic_eviews, 6),
                    'logL': round(logL, 4),
                    'tous_sig': tous_sig,
                })
            except Exception as exc:
                logger.warning('Estimation ARIMA(%s,%s,%s) échouée : %s', p, d, q, exc)
    if not resultats:
        return pd.DataFrame(columns=['p', 'd', 'q', 'AIC', 'BIC', 'logL', 'tous_sig'])
    return pd.DataFrame(resultats).sort_values('AIC').reset_index(drop=True)


def selectionner_arima(serie_arima, d_arima, p_max=4, q_max=4,
                       seuil_significativite=0.5, preference_parcimonie=True,
                       tolerance_aic_parcimonie=10):
    """
    Sélectionne le meilleur ARIMA selon la hiérarchie Box-Jenkins stricte :
      1. Filtre dur : tous_sig=True
      2. Parcimonie : complexité minimale (p+q)
      3. AIC : départage à égalité

    Si aucun modèle n'est significatif, applique le critère Burnham & Anderson
    (2002) pour comparer le meilleur AIC global avec ARIMA(0,d,0) (random walk).

    Parameters
    ----------
    serie_arima : array-like
        Série temporelle à modéliser.
    d_arima : int
        Ordre de différenciation détecté.
    p_max : int, optional
        Ordre AR maximal (défaut 4).
    q_max : int, optional
        Ordre MA maximal (défaut 4).
    seuil_significativite : float, optional
        Seuil p-valeur pour la significativité (défaut 0.10).
    preference_parcimonie : bool, optional
        Active le test B&A random walk dans le fallback (défaut True).
    tolerance_aic_parcimonie : float, optional
        Si le modèle le plus parcimonieux a un AIC supérieur de plus de cette
        valeur au meilleur AIC parmi les significatifs, retenir le meilleur AIC
        (défaut 10). Évite de sacrifier trop d'AIC pour la parcimonie.

    Returns
    -------
    dict
        Clés : df, p_opt, d_opt, q_opt, aic, aic_norm,
               motif_selection, fiabilite.
    """
    df = grid_search_arima(serie_arima, d_arima, p_max, q_max, seuil_significativite)

    # Grille vide (toutes les estimations ont échoué) → fallback immédiat RW
    if df.empty:
        logger.warning("Grille ARIMA vide : aucun modèle estimable — retour random walk.")
        lb_pval_rendements = float('nan')
        try:
            lb_res = acorr_ljungbox(serie_arima, lags=[10], return_df=True)
            lb_pval_rendements = float(lb_res['lb_pvalue'].iloc[-1])
        except Exception as exc:
            logger.debug('Ljung-Box (grille vide) échoué, pval=NaN : %s', exc)
        return {
            'df':                  df,
            'p_opt':               0,
            'd_opt':               d_arima,
            'q_opt':               0,
            'aic':                 float('nan'),
            'aic_norm':            float('nan'),
            'motif_selection':     'grille_vide',
            'fiabilite':           False,
            'interpretation':      'incertain',
            'message_pedagogique': (
                "Aucun modèle ARIMA n'a pu être estimé sur cette série "
                "(problème de convergence ou série trop courte)."
            ),
            'lb_pval_rendements':  lb_pval_rendements,
            'max_magnitude_coef':  0.0,
        }

    df_sig = df[df['tous_sig']].copy()

    if len(df_sig) > 0:
        # Hiérarchie : parcimonie (p+q minimal) → AIC
        df_sig['complexite'] = df_sig['p'] + df_sig['q']
        df_sig = df_sig.sort_values(['complexite', 'AIC']).reset_index(drop=True)
        best_parc = df_sig.iloc[0]
        best_aic  = df_sig.sort_values('AIC').iloc[0]
        # Garde-fou : ne pas sacrifier trop d'AIC pour la parcimonie
        if float(best_parc['AIC']) - float(best_aic['AIC']) > tolerance_aic_parcimonie:
            best  = best_aic
            motif = 'tous_sig + AIC'
        else:
            best  = best_parc
            motif = 'tous_sig + parcimonie'
    else:
        # Fallback : comparer meilleur AIC global avec random walk (B&A)
        best_global = df.iloc[0]
        best        = best_global
        motif       = 'AIC_seul_AUCUN_SIG'
        if preference_parcimonie:
            try:
                arima_rw = ARIMA(serie_arima, order=(0, d_arima, 0)).fit()
                n_rw     = arima_rw.nobs
                k_rw     = len(arima_rw.params)
                aic_rw   = -2 * (arima_rw.llf / n_rw) + 2 * (k_rw / n_rw)
                bic_rw   = -2 * (arima_rw.llf / n_rw) + k_rw * np.log(n_rw) / n_rw
                # Burnham & Anderson (2002) : ΔAIC < 4 → modèles équivalents
                if aic_rw - float(best_global['AIC']) < 4:
                    best = pd.Series({
                        'p': 0, 'd': d_arima, 'q': 0,
                        'AIC': round(aic_rw, 6),
                        'BIC': round(bic_rw, 6),
                        'logL': round(arima_rw.llf, 4),
                        'tous_sig': False,
                    })
                    motif = 'random_walk_BA'
            except Exception as exc:
                logger.debug('Estimation random-walk BA échouée, motif inchangé : %s', exc)

    # ── Phase 1.1 — Interprétation pédagogique ────────────────────────────────
    # Ljung-Box (10 lags) sur les rendements bruts — discrimine martingale/incertain
    try:
        lb_res = acorr_ljungbox(serie_arima, lags=[10], return_df=True)
        lb_pval_rendements = float(lb_res['lb_pvalue'].iloc[-1])
    except Exception:
        lb_pval_rendements = float('nan')
        logger.warning("Calcul Ljung-Box (rendements bruts) échoué.")

    # Re-estimation du modèle retenu pour extraire les paramètres AR/MA
    p_best = int(best['p'])
    q_best = int(best['q'])
    fit_params: dict = {}
    if p_best > 0 or q_best > 0:
        try:
            _fit = ARIMA(serie_arima, order=(p_best, d_arima, q_best)).fit()
            fit_params = dict(_fit.params)
        except Exception:
            logger.warning(
                "Re-estimation ARIMA(%d,%d,%d) pour interprétation échouée.",
                p_best, d_arima, q_best,
            )

    interp = _determiner_interpretation(
        motif=motif,
        p_opt=p_best,
        q_opt=q_best,
        fit_params=fit_params,
        lb_pval_rendements=lb_pval_rendements,
        n_obs=len(serie_arima),
    )
    max_magnitude = max(
        (abs(v) for k, v in fit_params.items() if k.startswith(('ar.', 'ma.'))),
        default=0.0,
    )

    return {
        'df':                  df,
        'p_opt':               int(best['p']),
        'd_opt':               int(best['d']),
        'q_opt':               int(best['q']),
        'aic':                 float(best['AIC']),
        'aic_norm':            float(best['AIC']),   # format EViews = déjà normalisé/obs
        'motif_selection':     motif,
        'fiabilite':           motif != 'AIC_seul_AUCUN_SIG',
        # ── Phase 1.1 ──────────────────────────────────────────────────────────
        'interpretation':      interp['code'],
        'message_pedagogique': interp['message'],
        'lb_pval_rendements':  lb_pval_rendements,
        'max_magnitude_coef':  max_magnitude,
    }
