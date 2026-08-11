# -*- coding: utf-8 -*-
"""Tableau des dépassements annuels de la VaR (OOS), par modèle GARCH et niveau."""
import logging
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, t as t_dist
from arch import arch_model

_log = logging.getLogger('tickerlab.depassements_annuels')

from .var_engine import var_cornish_fisher

warnings.filterwarnings('ignore')


def _ecrire(contenu: str, nom: str, dossier):
    p = Path(dossier)
    p.mkdir(parents=True, exist_ok=True)
    (p / nom).write_text(contenu, encoding='utf-8')
    _log.info('  OK -> %s', p / nom)


def _var_oos_serie(rendements: pd.Series, model_dict: dict,
                   split_ratio: float, niveaux: list):
    """Retourne un dict {alpha: pd.Series(violation, index=dates_oos)}."""
    serie_full  = rendements.dropna()
    T_full      = len(serie_full)
    T_train     = int(np.floor(split_ratio * T_full))
    serie_train = serie_full.iloc[:T_train]
    serie_test  = serie_full.iloc[T_train:]

    vol_opt = model_dict['vol']
    p_opt   = int(model_dict['p'])
    o_opt   = int(model_dict['o'])
    q_opt   = int(model_dict['q'])
    dist    = model_dict.get('dist', 't')

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

    resultats = {}
    for alpha in niveaux:
        if dist == 'normal':
            q_z = norm.ppf(1 - alpha)
        elif dist == 't':
            q_z = t_dist.ppf(1 - alpha, df=nu_bt) if not math.isnan(nu_bt) \
                  else norm.ppf(1 - alpha)
        else:
            resid_std = (garch_bt.resid / garch_bt.conditional_volatility).dropna()
            q_z = float(np.quantile(resid_std.values, 1 - alpha))

        var_dyn = pd.Series(mu_bt + q_z * vol_oos.values,
                            index=serie_test.index[:len(vol_oos)])
        viol_dyn = (serie_test.iloc[:len(vol_oos)] < var_dyn).astype(int)
        resultats[alpha] = viol_dyn

    return resultats, serie_test


def generer_depassements_annuels(rendements: pd.Series, best_dict: dict,
                                  df_garch: pd.DataFrame, config: dict):
    """Produit tab_depassements_annuels.tex — violations OOS par année et modèle."""
    split_ratio = config.get('backtest', {}).get('split_ratio', 0.70)
    niveaux     = [0.95, 0.99]
    dossier     = Path(config['output']['dossier_resultats']) / 'latex'

    crises_cfg = {
        str(e['annee']): e['label']
        for e in config.get('events', {}).get('crises', [])
    }

    # Sélectionner les 2 modèles : meilleur GJR-t et meilleur EGARCH-t
    modeles = []
    for nom_mod, label in [('GJR-GARCH', 'GJR-GARCH(t)'), ('EGARCH', 'EGARCH(t)')]:
        sub = df_garch[(df_garch['modele'] == nom_mod) & (df_garch['dist'] == 't')]
        if sub.empty:
            sub = df_garch[df_garch['modele'] == nom_mod]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            row['dist'] = 't'
            modeles.append((row, label))

    # Fallback si un seul modèle dans la grille
    if not modeles:
        modeles = [(best_dict, best_dict.get('modele', 'GARCH'))]

    # Calculer les violations par modèle
    all_viols = {}
    all_tests = {}
    for m_dict, m_label in modeles:
        _log.info('    Calcul OOS %s...', m_label)
        try:
            viols, serie_test = _var_oos_serie(
                rendements, m_dict, split_ratio, niveaux)
            all_viols[m_label] = viols
            all_tests[m_label] = serie_test
        except Exception as e:
            _log.warning('    ERREUR %s: %s', m_label, e)

    if not all_viols:
        _log.warning('    Aucune donnee OOS disponible.')
        return

    # Grouper par année
    serie_test_ref = list(all_tests.values())[0]
    annees = sorted(serie_test_ref.index.year.unique())

    # Construction du tableau
    header_models = ' & '.join(
        r'\multicolumn{4}{c}{' + lbl + '}'
        for lbl in all_viols.keys()
    )
    n_mods = len(all_viols)
    col_fmt = 'llr' + 'rrrr' * n_mods

    sub_hdr_parts = ['Année', 'Événement', 'N obs']
    for _ in all_viols:
        sub_hdr_parts += [r'Dép. 95\%', r'Tx 95\%', r'Dép. 99\%', r'Tx 99\%']
    sub_hdr = ' & '.join(sub_hdr_parts) + r' \\'

    # Lignes de cmidrule
    cmidrule = ' '.join(
        r'\cmidrule(lr){' + f'{4+4*i}-{7+4*i}' + '}'
        for i in range(n_mods)
    )

    rows = []
    totaux = {lbl: {0.95: 0, 0.99: 0} for lbl in all_viols}
    total_obs = 0

    for an in annees:
        mask_yr = (serie_test_ref.index.year == an)
        n_obs   = int(mask_yr.sum())
        total_obs += n_obs
        ev_label = crises_cfg.get(str(an), '')

        # Annoter les crises
        if ev_label:
            an_str = f'{an} \\textit{{({ev_label[:15]})}}'
        else:
            an_str = str(an)

        row = [f'  {an_str}', ev_label[:0], str(n_obs)]
        for lbl, viols in all_viols.items():
            for alpha in niveaux:
                mask_an = viols[alpha].index.year == an
                n_dep   = int(viols[alpha][mask_an].sum())
                taux    = n_dep / n_obs if n_obs > 0 else 0
                totaux[lbl][alpha] += n_dep
                row += [str(n_dep), f'{taux:.3f}']
        rows.append(' & '.join(row) + ' \\\\')

    # Ligne TOTAL
    row_tot = [r'  \textbf{TOTAL}', '', str(total_obs)]
    for lbl in all_viols:
        for alpha in niveaux:
            n_tot  = totaux[lbl][alpha]
            tx_tot = n_tot / total_obs if total_obs > 0 else 0
            row_tot += [f'\\textbf{{{n_tot}}}', f'\\textbf{{{tx_tot:.3f}}}']
    rows.append(r'\midrule')
    rows.append(' & '.join(row_tot) + ' \\\\')

    multicol_line = '  & & & ' + header_models + r' \\'

    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Dépassements annuels de la VaR OOS par modèle GARCH}' + '\n'
        r'\label{tab:depassements_annuels}' + '\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{' + col_fmt + '}\n'
        r'\toprule' + '\n'
        + multicol_line + '\n'
        + cmidrule + '\n'
        + sub_hdr + '\n'
        r'\midrule' + '\n'
        + '\n'.join(rows) + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{Dép. = nombre de dépassements ; Tx = taux observé. '
        r'Taux théorique : 5\,\% (95\,\%) et 1\,\% (99\,\%).}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, 'tab_depassements_annuels.tex', dossier)
