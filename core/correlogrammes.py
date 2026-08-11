# -*- coding: utf-8 -*-
"""Corrélogrammes numériques style EViews en LaTeX (ACF, PACF, Q-stat, Prob)."""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

_log = logging.getLogger('tickerlab.correlogrammes')


def _ecrire(contenu: str, nom: str, dossier):
    p = Path(dossier)
    p.mkdir(parents=True, exist_ok=True)
    (p / nom).write_text(contenu, encoding='utf-8')
    _log.info('  OK -> %s', p / nom)


def _table_correlogramme(serie: pd.Series, lags: int,
                          caption: str, label: str) -> str:
    s = serie.dropna()
    n = len(s)

    ac_vals  = acf(s,  nlags=lags, fft=True)[1:]
    pac_vals = pacf(s, nlags=lags)[1:]

    lb = acorr_ljungbox(s, lags=list(range(1, lags + 1)), return_df=True)
    q_stats = lb['lb_stat'].values
    q_pvals = lb['lb_pvalue'].values

    header = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{' + caption + '}\n'
        r'\label{' + label + '}\n'
        r'\begin{adjustbox}{max width=\textwidth}' + '\n'
        r'\begin{tabular}{crrrrr}' + '\n'
        r'\toprule' + '\n'
        r'Retard & AC & PAC & Q-Stat (Ljung-Box) & Prob. \\' + '\n'
        r'\midrule' + '\n'
    )

    rows = []
    for k in range(lags):
        lag  = k + 1
        ac   = ac_vals[k]
        pa   = pac_vals[k]
        q    = q_stats[k]
        pv   = q_pvals[k]
        pv_str = f'{pv:.4f}'
        if pv < 0.05:
            pv_str = r'\textbf{' + pv_str + '}'
        rows.append(
            f'  {lag} & {ac:.3f} & {pa:.3f} & {q:.4f} & {pv_str} \\\\'
        )

    footer = (
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{Les p-values en \textbf{gras} sont significatives au seuil 5\,\%.}' + '\n'
        r'\end{table}' + '\n'
    )

    return header + '\n'.join(rows) + '\n' + footer


def generer_correlogrammes(rendements, arima_result: dict, config: dict):
    """Produit 4 tables de corrélogramme : LBRENT, DLBRENT, résidus ARIMA, résidus² ARIMA."""
    import numpy as np

    ticker  = config['data'].get('ticker', '')
    lags    = config.get('sorties_etendues', {}).get('correlogramme_lags', 36)
    dossier = Path(config['output']['dossier_resultats']) / 'latex'

    rend_s = (rendements.iloc[:, 0] if isinstance(rendements, pd.DataFrame)
              else rendements).dropna()

    # LBRENT et DLBRENT sont déduits du pipeline (rendements = log-diff)
    # On recrée lbrent à partir de rend_s (cumsum + constante arbitraire)
    # En pratique, l'ACF de DLBRENT = ACF de rendements (même série)
    dlbrent = rend_s
    lbrent  = rend_s.cumsum()

    p_opt = arima_result.get('p_opt', 1)
    d_opt = arima_result.get('d_opt', 0)
    q_opt = arima_result.get('q_opt', 1)
    try:
        arima_fit   = ARIMA(rend_s, order=(p_opt, d_opt, q_opt)).fit(
            method_kwargs={'warn_convergence': False})
        resid_arima = arima_fit.resid.dropna()
    except Exception:
        resid_arima = rend_s.copy()
    resid2_arima = (resid_arima ** 2).rename('Résidus² ARIMA')

    specs = [
        (lbrent,      'tab_corr_lbrent.tex',
         f'Corrélogramme --- LBRENT (log-prix {ticker})', 'tab:corr_lbrent'),
        (dlbrent,     'tab_corr_dlbrent.tex',
         f'Corrélogramme --- DLBRENT (log-rendements {ticker})', 'tab:corr_dlbrent'),
        (resid_arima, 'tab_corr_resid_arima.tex',
         f'Corrélogramme --- Résidus ARIMA({p_opt},{d_opt},{q_opt})', 'tab:corr_resid_arima'),
        (resid2_arima,'tab_corr_resid2_arima.tex',
         f'Corrélogramme --- Résidus² ARIMA (effets ARCH)', 'tab:corr_resid2_arima'),
    ]

    for serie, nom, caption, label in specs:
        contenu = _table_correlogramme(serie, lags, caption, label)
        _ecrire(contenu, nom, dossier)
