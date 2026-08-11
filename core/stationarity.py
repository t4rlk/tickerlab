"""Tests de stationnarité : ADF, Phillips-Perron, KPSS."""
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron


def tester_stationnarite(serie, nom=''):
    """
    Vote majoritaire ADF + PP + KPSS pour décider la stationnarité.

    Parameters
    ----------
    serie : array-like
        Série temporelle à tester.
    nom : str, optional
        Nom de la série (pour logs éventuels).

    Returns
    -------
    tuple
        (est_stationnaire : bool, dict_pvalues : dict).
        dict_pvalues contient les clés 'adf_p', 'pp_p', 'kpss_p'.
    """
    adf_stat, adf_p, *_, adf_cv, _ = adfuller(serie, autolag='AIC')
    pp = PhillipsPerron(serie)
    ks, kp, _, kcv = kpss(serie, regression='c', nlags='auto')
    votes_ns = sum([adf_p >= 0.05, pp.pvalue >= 0.05, kp < 0.05])
    return (votes_ns < 2), {'adf_p': adf_p, 'pp_p': pp.pvalue, 'kpss_p': kp}


def analyser_stationnarite(prix, rendements):
    """
    Détermine l'ordre de différenciation d sur log(prix).

    Si d == 1, la série ARIMA est alignée sur les rendements (cohérence
    d'échelle avec le GARCH). Sinon, log(prix) est utilisé directement.

    Parameters
    ----------
    prix : pd.DataFrame
        DataFrame avec colonne 'prix' (sortie de telecharger_prix).
    rendements : pd.Series
        Log-rendements (sortie de calculer_rendements).

    Returns
    -------
    tuple
        (d : int, SERIE_ARIMA : pd.Series, D_ARIMA : int).
    """
    log_prix = np.log(prix['prix'].dropna()).dropna()  # supprime NaN si prix <= 0 (ex: CL=F)
    est_stat, _ = tester_stationnarite(log_prix, 'log(prix)')
    d = 0
    serie_diff = log_prix.copy()
    while not est_stat and d < 3:
        d += 1
        serie_diff = serie_diff.diff().dropna()
        est_stat, _ = tester_stationnarite(serie_diff, f'log(prix) diff {d}x')

    if d == 1:
        return d, rendements, 0
    return d, log_prix, d
