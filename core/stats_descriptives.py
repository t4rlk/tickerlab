# -*- coding: utf-8 -*-
"""Statistiques descriptives formatées en LaTeX — 3 tableaux : BRENT, LBRENT, DLBRENT."""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from statsmodels.stats.stattools import jarque_bera

_log = logging.getLogger('tickerlab.stats_descriptives')


def _ecrire(contenu: str, nom: str, dossier):
    p = Path(dossier)
    p.mkdir(parents=True, exist_ok=True)
    (p / nom).write_text(contenu, encoding='utf-8')
    _log.info('  OK -> %s', p / nom)


def _table_stats(serie: pd.Series, caption: str, label: str, decimales: int = 4) -> str:
    s = serie.dropna()
    n = len(s)
    moyenne = float(s.mean())
    mediane = float(s.median())
    maxi    = float(s.max())
    mini    = float(s.min())
    ecart   = float(s.std())
    skew    = float(sp_stats.skew(s.values))
    kurt    = float(sp_stats.kurtosis(s.values, fisher=False))
    jb, jb_p, _, _ = jarque_bera(s.values)

    fmt = f'.{decimales}f'
    if jb_p < 0.0001:
        prob_str = f'{jb_p:.4e}'
    else:
        prob_str = f'{jb_p:{fmt}}'

    lignes = [
        ('Observations',      f'{n}'),
        ('Moyenne',           f'{moyenne:{fmt}}'),
        (r'M\'{e}diane',      f'{mediane:{fmt}}'),
        ('Maximum',           f'{maxi:{fmt}}'),
        ('Minimum',           f'{mini:{fmt}}'),
        (r'\'{E}cart-type',   f'{ecart:{fmt}}'),
        ('Skewness',          f'{skew:{fmt}}'),
        ('Kurtosis',          f'{kurt:{fmt}}'),
        ('Jarque-Bera',       f'{jb:{fmt}}'),
        (r'Probabilit\'{e} (JB)', prob_str),
    ]

    rows = '\n'.join(f'  {lab} & {val} \\\\' for lab, val in lignes)

    return (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{' + caption + '}\n'
        r'\label{' + label + '}\n'
        r'\centering' + '\n'
        r'\begin{tabular}{lr}' + '\n'
        r'\toprule' + '\n'
        r'Statistique & Valeur \\' + '\n'
        r'\midrule' + '\n'
        + rows + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{table}' + '\n'
    )


def generer_stats_descriptives(prix, rendements, config: dict):
    """Génère tab_stats_brent.tex, tab_stats_lbrent.tex, tab_stats_dlbrent.tex."""
    ticker    = config['data'].get('ticker', '')
    decimales = config.get('sorties_etendues', {}).get('stats_descriptives_decimales', 4)
    dossier   = Path(config['output']['dossier_resultats']) / 'latex'

    if isinstance(prix, pd.DataFrame):
        prix_s = prix.iloc[:, 0]
    else:
        prix_s = prix

    lbrent  = np.log(prix_s).rename('LBRENT')
    dlbrent = lbrent.diff().dropna().rename('DLBRENT')
    rend_s  = (rendements.iloc[:, 0] if isinstance(rendements, pd.DataFrame)
               else rendements).dropna()

    specs = [
        (prix_s,  f'tab_stats_brent.tex',
         f'Statistiques descriptives --- {ticker} (niveau)',
         'tab:stats_brent'),
        (lbrent,  'tab_stats_lbrent.tex',
         f'Statistiques descriptives --- LBRENT (log-prix)',
         'tab:stats_lbrent'),
        (dlbrent, 'tab_stats_dlbrent.tex',
         f'Statistiques descriptives --- DLBRENT (log-rendements)',
         'tab:stats_dlbrent'),
    ]

    for serie, nom, caption, label in specs:
        contenu = _table_stats(serie, caption, label, decimales)
        _ecrire(contenu, nom, dossier)
