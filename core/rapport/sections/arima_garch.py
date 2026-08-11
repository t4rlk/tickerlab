# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging as _logging

import tickerlab.core.rapport._helpers as _H
from reportlab.platypus import PageBreak as _PageBreak
from tickerlab.core.rapport._helpers import (
    _h1, _h2, _p, _caption, _spacer, _embed_figure,
    _tableau_eviews, _correlogramme_eviews,
    _fmt, _fmt_pval, _fmt_signif, _hex, _warn,
    LARG_U, nom_actif,
)
from tickerlab.core.rapport._eviews import (
    bloc_eviews_estimation,
    eviews_dist_label,
)
from tickerlab.core.rapport._stats import (
    stats_desc, adf_complet, pp_complet, kpss_complet,
)
from tickerlab.core.rapport.sections._common import (
    _sections_log, _log, _th, _is_nan, _int_or_na, _sq,
    _encadre_pedagogique, _page_break_if, _lags_rapport,
    _max_lignes_tableau, _split_tableau, _extra_pval_bold,
)
from tickerlab.core.rapport.sections.stationarite import _tab_stats_desc



# =============================================================================
# Helpers prives sections 4-6
# =============================================================================

def _fmt_arima_param(name: str) -> str:
    """Rend lisibles les noms de parametres statsmodels ARIMA."""
    if name == 'const':  return 'Constante'
    if name == 'sigma2': return 'Variance (sigma2)'
    if name.startswith('ar.L'): return f'AR({name[4:]})'
    if name.startswith('ma.L'): return f'MA({name[4:]})'
    return name




def _fmt_garch_param(name: str) -> str:
    """Rend lisibles les noms de parametres arch GARCH."""
    mapping = {
        'mu':     'mu (cste)',
        'omega':  'omega',
        'delta':  'delta (puissance)',
        'nu':     'nu (dof/shape)',
        'lambda': 'lambda (asym.)',
    }
    return mapping.get(name, name)




def _tab_arima_coefs(fit, label: str) -> list:
    """Tableau de coefficients ARIMA style EViews."""
    th = _th()
    params = fit.params
    bse    = fit.bse
    tvals  = fit.tvalues
    pvals  = fit.pvalues

    lignes   = []
    extra    = []
    row_base = 2   # titre=0, header=1

    for i, name in enumerate(params.index):
        est = float(params[name])
        se  = float(bse[name])
        tv  = float(tvals[name])
        pv  = float(pvals[name])
        sig = _fmt_signif(pv)
        lignes.append([
            _fmt_arima_param(name),
            _fmt(est, 6),
            _fmt(se,  6),
            _fmt(tv,  4),
            _fmt_pval(pv),
            sig,
        ])
        if pv < 0.05:
            extra += [
                ('FONTNAME',  (4, row_base + i), (4, row_base + i), 'Courier-Bold'),
                ('TEXTCOLOR', (4, row_base + i), (4, row_base + i), th['warn']),
            ]

    # Footer EViews : Log L, AIC, BIC dans le tableau (ligne separatrice)
    from reportlab.lib import colors as _rlc
    n_obs = int(fit.nobs)
    k     = len(fit.params)
    logL  = float(fit.llf)
    aic_e = -2 * logL / n_obs + 2 * k / n_obs
    bic_e = -2 * logL / n_obs + k * math.log(n_obs) / n_obs

    sep_row = row_base + len(fit.params)
    lignes.append(['Log L',        _fmt(logL,  4), '', '', '', ''])
    lignes.append(['AIC (EViews)', _fmt(aic_e, 6), '', '', '', ''])
    lignes.append(['BIC (EViews)', _fmt(bic_e, 6), '', '', '', ''])
    extra.append(('LINEABOVE', (0, sep_row), (-1, sep_row), 0.8, _rlc.black))

    cw = [LARG_U*0.24, LARG_U*0.15, LARG_U*0.15, LARG_U*0.14, LARG_U*0.16, LARG_U*0.16]
    return _tableau_eviews(
        titre=f'Estimation ARIMA -- {label}',
        colonnes=['Parametre', 'Estim.', 'Std. Err.', 't-stat', 'Prob.', 'Sig.'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='*** p<0.01, ** p<0.05, * p<0.10.',
    )




def _tab_garch_coefs(fit_result, label: str) -> list:
    """Tableau de coefficients GARCH style EViews (arch ARCHModelResult)."""
    th     = _th()
    params = fit_result.params
    bse    = fit_result.std_err
    tvals  = fit_result.tvalues
    pvals  = fit_result.pvalues

    lignes   = []
    extra    = []
    row_base = 2

    for i, name in enumerate(params.index):
        est = float(params[name])
        se  = float(bse[name])  if not _is_nan(bse[name])  else float('nan')
        tv  = float(tvals[name]) if not _is_nan(tvals[name]) else float('nan')
        pv  = float(pvals[name]) if not _is_nan(pvals[name]) else float('nan')
        sig = _fmt_signif(pv)
        lignes.append([
            _fmt_garch_param(name),
            _fmt(est, 6),
            _fmt(se,  6),
            _fmt(tv,  4),
            _fmt_pval(pv),
            sig,
        ])
        if not _is_nan(pv) and pv < 0.05:
            extra += [
                ('FONTNAME',  (4, row_base + i), (4, row_base + i), 'Courier-Bold'),
                ('TEXTCOLOR', (4, row_base + i), (4, row_base + i), th['warn']),
            ]

    # Footer EViews : Log L, AIC, BIC dans le tableau (ligne separatrice)
    from reportlab.lib import colors as _rlc
    n_obs = int(fit_result.nobs)
    logL  = float(fit_result.loglikelihood)
    aic_e = fit_result.aic / n_obs
    bic_e = fit_result.bic / n_obs

    sep_row = row_base + len(params)
    lignes.append(['Log L',        _fmt(logL,  4), '', '', '', ''])
    lignes.append(['AIC (EViews)', _fmt(aic_e, 6), '', '', '', ''])
    lignes.append(['BIC (EViews)', _fmt(bic_e, 6), '', '', '', ''])
    extra.append(('LINEABOVE', (0, sep_row), (-1, sep_row), 0.8, _rlc.black))

    cw = [LARG_U*0.24, LARG_U*0.15, LARG_U*0.15, LARG_U*0.14, LARG_U*0.16, LARG_U*0.16]
    return _tableau_eviews(
        titre=f'Estimation GARCH -- {label}',
        colonnes=['Parametre', 'Estim.', 'Std. Err.', 't-stat', 'Prob.', 'Sig.'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='*** p<0.01, ** p<0.05, * p<0.10.',
    )




def _tab_grille_arima(df_grid: pd.DataFrame, p_opt: int, d_opt: int, q_opt: int,
                      n_top: int = 10) -> list:
    """Grille ARIMA top-N avec surlignage du modele retenu."""
    from reportlab.lib.colors import HexColor
    df_show  = df_grid.head(n_top).reset_index(drop=True)
    lignes   = []
    extra    = []
    sel_color = HexColor('#FFF3CD')   # ambre clair, neutre sur tous les themes

    for enum_idx, (_, row) in enumerate(df_show.iterrows()):
        tag    = f"ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])})"
        sig    = 'Oui' if row.get('tous_sig', False) else 'Non'
        logL_val = row.get('logL', float('nan'))
        lignes.append([tag, _fmt(row['AIC'], 6), _fmt(row['BIC'], 6),
                        _fmt(logL_val, 2), sig])
        if (int(row['p']) == p_opt and int(row['d']) == d_opt
                and int(row['q']) == q_opt):
            ri = enum_idx + 2   # +2 = titre + header
            extra.append(('BACKGROUND', (0, ri), (-1, ri), sel_color))

    cw = [LARG_U*0.31, LARG_U*0.17, LARG_U*0.17, LARG_U*0.18, LARG_U*0.17]
    return _tableau_eviews(
        titre=f'Grille ARIMA -- Top {n_top} par AIC (format EViews)',
        colonnes=['Modele', 'AIC', 'BIC', 'log L', 'Coefs sig.'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='AIC = -2*logL/n + 2*k/n (EViews). Fond jaune = modele retenu. Sig. = tous coefs p < seuil.',
    )




def _tab_grille_garch(df_garch: pd.DataFrame, modele_sel: str, p_sel: int,
                      q_sel: int, dist_sel: str, n_top: int = 15,
                      o_sel: int = 0) -> list:
    """Grille GARCH top-N avec colonne delta (APARCH/TGARCH) et surlignage."""
    from reportlab.lib.colors import HexColor
    sel_color = HexColor('#FFF3CD')
    df_show   = df_garch.head(n_top).reset_index(drop=True)
    lignes    = []
    extra     = []

    for enum_idx, (_, row) in enumerate(df_show.iterrows()):
        o_row = int(row.get('o', 0))
        spec  = (f"({int(row['p'])},{o_row},{int(row['q'])})"
                 if o_row > 0 else f"({int(row['p'])},{int(row['q'])})")
        sig   = 'Oui' if row.get('tous_sig', False) else 'Non'
        pw    = row.get('power', float('nan'))
        delta = _fmt(pw, 3) if not _is_nan(pw) else '--'
        dist  = str(row.get('dist', ''))
        lignes.append([str(row.get('modele', '')), spec, dist,
                        _fmt(row.get('AIC_norm', row.get('AIC')), 4),
                        _fmt(row.get('BIC_norm', row.get('BIC')), 4),
                        _fmt(row.get('max_pval_vol'), 4), sig, delta])
        if (str(row.get('modele', '')) == modele_sel
                and int(row['p']) == p_sel
                and int(row.get('o', 0)) == o_sel
                and int(row['q']) == q_sel
                and str(row.get('dist', '')) == dist_sel):
            ri = enum_idx + 2
            extra.append(('BACKGROUND', (0, ri), (-1, ri), sel_color))

    cw = [LARG_U*0.20, LARG_U*0.08, LARG_U*0.10, LARG_U*0.12,
          LARG_U*0.12, LARG_U*0.12, LARG_U*0.08, LARG_U*0.18]
    return _tableau_eviews(
        titre=f'Grille GARCH -- Top {n_top} par BIC (format EViews)',
        colonnes=['Modele', '(p,q)', 'Dist.', 'AIC', 'BIC', 'p-max', 'Sig.', 'Delta'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='AIC/BIC normalises par obs. Sig. = coefs variance tous significatifs. Delta = puissance APARCH/TGARCH.',
    )




def _tab_arch_lm(results: list) -> list:
    """Tableau tests ARCH-LM (Engle 1982) multi-ordres."""
    th = _th()
    lignes = []
    extra  = []
    row_base = 2

    for i, r in enumerate(results):
        pf = float(r.get('f_pval',   float('nan')))
        pm = float(r.get('lm_pval',  float('nan')))
        lignes.append([
            str(int(r['lag'])),
            _fmt(r.get('f_stat'),  4),
            _fmt_pval(pf),
            _fmt(r.get('lm_stat'), 4),
            _fmt_pval(pm),
        ])
        # Bold LM pval (col 4) if significant
        if not _is_nan(pm) and pm < 0.05:
            extra += [
                ('FONTNAME',  (4, row_base + i), (4, row_base + i), 'Courier-Bold'),
                ('TEXTCOLOR', (4, row_base + i), (4, row_base + i), th['warn']),
            ]

    cw = [LARG_U*0.12, LARG_U*0.18, LARG_U*0.18, LARG_U*0.18, LARG_U*0.18]
    # total manque 0.16 — ajuster col0
    cw[0] = LARG_U * 0.16
    return _tableau_eviews(
        titre='Tests ARCH-LM -- Engle (1982)',
        colonnes=['Lag', 'F-stat', 'Prob. F', 'LM-stat', 'Prob. LM'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='H0 : pas d\'effet ARCH. Prob. en gras = rejet au seuil 5%.',
    )




def _tab_sign_bias(sbt: dict) -> list:
    """Tableau test de biais de signe Engle-Ng (1993)."""
    th = _th()
    rows_data = [
        ('Biais de signe',    sbt.get('sign_stat'),  sbt.get('sign_pval')),
        ('Biais taille neg.', sbt.get('neg_stat'),   sbt.get('neg_pval')),
        ('Biais taille pos.', sbt.get('pos_stat'),   sbt.get('pos_pval')),
        ('F-test conjoint',   sbt.get('joint_f'),    sbt.get('joint_pval')),
    ]
    lignes = []
    extra  = []
    row_base = 2

    for i, (lbl, stat, pval) in enumerate(rows_data):
        pv = float(pval) if pval is not None and not _is_nan(pval) else float('nan')
        lignes.append([lbl, _fmt(stat, 4), _fmt_pval(pv), _fmt_signif(pv)])
        if not _is_nan(pv) and pv < 0.05:
            extra += [
                ('FONTNAME',  (2, row_base + i), (2, row_base + i), 'Courier-Bold'),
                ('TEXTCOLOR', (2, row_base + i), (2, row_base + i), th['warn']),
            ]

    cw = [LARG_U*0.42, LARG_U*0.20, LARG_U*0.22, LARG_U*0.16]
    return _tableau_eviews(
        titre='Test de biais de signe -- Engle & Ng (1993)',
        colonnes=['Test', 't/F stat.', 'Prob.', 'Sig.'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='H0 individuel : pas de biais (t-test). H0 conjoint : F-test sur b1, b2, b3.',
    )




def _tab_persistance(modele_nom: str, params, nobs: int) -> list:
    """Tableau caracteristiques du modele GARCH final."""
    from tickerlab.core.rapport._stats import persistance_garch
    pers = persistance_garch(modele_nom, params)

    if not _is_nan(pers) and 0 < pers < 1:
        halflife = round(math.log(0.5) / math.log(pers), 1)
        halflife_str = f'{halflife} jours'
        stationnaire = 'Oui (< 1)'
    elif not _is_nan(pers) and abs(pers - 1.0) < 1e-9:
        halflife_str = 'Infinie (memoire longue)'
        stationnaire = 'N/A (FIGARCH)'
    elif not _is_nan(pers):
        halflife_str = 'N/A (non-stationnaire)'
        stationnaire = 'Non (>= 1)'
    else:
        halflife_str = 'N/A'
        stationnaire = 'N/A (formule a implementer)'
        pers = float('nan')

    lignes = [
        ['Observations', str(int(nobs))],
        ['Persistance',  _fmt(pers, 4)],
        ['Demi-vie',     halflife_str],
        ['Stationnaire', stationnaire],
    ]
    cw = [LARG_U * 0.55, LARG_U * 0.45]
    return _tableau_eviews(
        titre=f'Caracteristiques -- {modele_nom}',
        colonnes=['Indicateur', 'Valeur'],
        lignes=lignes, col_widths=cw,
        note=(
            'Persistance : sum(alpha_i)+sum(beta_j) (GARCH/GJR) ; '
            'sum(beta_j) tous lags (EGARCH, Nelson 1991 eq.10) ; '
            'sum(beta_j)+sum(alpha_i*kappa) (APARCH/TGARCH). '
            'Demi-vie : log(0.5)/log(pers.) en jours.'
        ),
    )




def _fig_rendements(rendements: pd.Series, config: dict, nom: str = 'l\'actif') -> plt.Figure:
    """Rendements journaliers avec bande +/- 2 ecart-types."""
    th  = _th()
    std = float(rendements.std())

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(rendements.index, rendements.values,
            color=_hex(th['accent']), lw=0.55, alpha=0.85)
    ax.axhline(0,        color='black', lw=0.4)
    ax.axhline( 2 * std, color=_hex(th['warn']), lw=0.6, ls='--', alpha=0.6)
    ax.axhline(-2 * std, color=_hex(th['warn']), lw=0.6, ls='--', alpha=0.6)

    crises = config.get('events', {}).get('crises', [])
    trans  = ax.get_xaxis_transform()
    for cr in crises:
        annee = cr.get('annee')
        dates = [d for d in rendements.index if hasattr(d, 'year') and d.year == annee]
        if dates:
            ax.axvline(dates[0], color=_hex(th['warn']),
                       lw=0.6, ls=':', alpha=0.5, ymin=0, ymax=0.93)
            ax.text(dates[0], 0.95, cr.get('label', ''), transform=trans,
                    fontsize=5.8, rotation=90, va='top', ha='right',
                    color=_hex(th['warn']), alpha=0.75)

    ax.set_title(f'Rendements journaliers -- {nom} (log-differences)', fontsize=9, fontweight='bold')
    ax.set_ylabel('r_t = ln(P_t/P_{t-1})', fontsize=8)
    ax.set_xlabel('Date', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.15, linestyle=':')
    fig.tight_layout()
    return fig




def _fig_volatilite(sigma_t, dates, config: dict) -> plt.Figure:
    """Volatilite conditionnelle sigma_t avec evenements."""
    th     = _th()
    crises = config.get('events', {}).get('crises', [])

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(dates, sigma_t, color=_hex(th['accent']), lw=0.75, alpha=0.9)

    trans = ax.get_xaxis_transform()
    for cr in crises:
        annee = cr.get('annee')
        date_list = [d for d in dates if hasattr(d, 'year') and d.year == annee]
        if date_list:
            ax.axvline(date_list[0], color=_hex(th['warn']),
                       lw=0.6, ls='--', alpha=0.6, ymin=0, ymax=0.93)
            ax.text(date_list[0], 0.95, cr.get('label', ''), transform=trans,
                    fontsize=5.8, rotation=90, va='top', ha='right',
                    color=_hex(th['warn']), alpha=0.80)

    ax.set_title('Volatilite conditionnelle sigma_t', fontsize=9, fontweight='bold')
    ax.set_ylabel('sigma_t', fontsize=8)
    ax.set_xlabel('Date', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.15, linestyle=':')
    fig.tight_layout()
    return fig




def _fig_residus_std(z_t, dates) -> plt.Figure:
    """Residus standardises z_t avec bandes +/- 1.96."""
    th = _th()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(dates, z_t, color=_hex(th['accent']), lw=0.55, alpha=0.80)
    ax.axhline( 1.96, color=_hex(th['warn']), lw=0.7, ls='--', alpha=0.65)
    ax.axhline(-1.96, color=_hex(th['warn']), lw=0.7, ls='--', alpha=0.65)
    ax.axhline(0,     color='black', lw=0.35)
    ax.set_title('Residus standardises z_t = eps_t / sigma_t', fontsize=9, fontweight='bold')
    ax.set_ylabel('z_t', fontsize=8)
    ax.set_xlabel('Date', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.15, linestyle=':')
    fig.tight_layout()
    return fig


# =============================================================================
# Section 4 — Rendements journaliers
# =============================================================================



# =============================================================================
# Section 4 — Rendements journaliers
# =============================================================================

def section_4(rendements: pd.Series, config: dict, ticker: str = '') -> list:
    """
    Section 4 : Rendements journaliers r_t = ln(P_t / P_{t-1}).
    Contenu : evolution, stats descriptives, correlogrammes r et r^2.
    """
    story = []
    nom   = nom_actif(ticker) if ticker else 'l\'actif'
    lags  = config.get('sorties_etendues', {}).get('correlogramme_lags', 36)
    rend  = rendements.dropna()

    story.append(_h1('4. Rendements journaliers'))
    story.append(_p(
        'Les rendements logarithmiques r_t = ln(P_t / P_{t-1}) constituent '
        'la serie stationnaire derivee du log-prix. Ils presentent les '
        'proprietes stylisees caracteristiques des series financieres : '
        'distribution leptokurtique, volatilite agregee, faible autocorrelation '
        'en niveau mais forte en carre (effet ARCH).'
    ))

    # 4.1 Evolution
    story.append(_h2('4.1 Evolution des rendements'))
    try:
        fig = _fig_rendements(rend, config, nom=nom)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=4.2))
        story.append(_caption(
            f'Figure 4.1 -- Rendements journaliers -- {nom}. '
            'Traits pointilles : bandes +/- 2 ecart-types.'
        ))
    except Exception as e:
        _warn(f'Section 4 figure rendements : {e}')

    story.append(_spacer(0.5))

    # 4.2 Stats descriptives
    story.append(_h2('4.2 Statistiques descriptives'))
    try:
        sd = stats_desc(rend)
        story.extend(_tab_stats_desc(sd, f'Rendements journaliers -- {nom}'))
        story.append(_p(
            'Un kurtosis en exces positif et un test Jarque-Bera significatif '
            'confirment la leptokurtose des rendements, incompatible avec '
            'l\'hypothese de normalite -- justifiant les distributions t, '
            'skew-t ou GED dans la modelisation GARCH.'
        ))
    except Exception as e:
        _warn(f'Section 4 stats_desc : {e}')

    story.append(_spacer(0.5))

    # 4.3 Correlogramme rendements
    story.append(_h2('4.3 Correlogramme ACF/PACF -- Rendements'))
    story.append(_p(
        f'Le correlogramme des rendements sur {lags} retards doit presenter '
        'peu d\'autocorrelation significative, confirmant l\'absence de '
        'predictibilite lineaire de la moyenne conditionnelle.'
    ))
    story.extend(_correlogramme_eviews(
        rend, lags=lags, titre='Correlogramme -- Rendements journaliers',
    ))

    story.append(_spacer(0.4))

    # 4.4 Correlogramme rendements^2
    story.append(_h2('4.4 Correlogramme ACF/PACF -- Rendements carres'))
    story.append(_p(
        'Les autocorrelations significatives dans r_t^2 revelent la dependance '
        'temporelle de la variance conditionnelle (effet ARCH/GARCH), '
        'justifiant le passage a une modelisation heteroscedastique.'
    ))
    rend_sq = rend ** 2
    rend_sq.name = 'r2'
    story.extend(_correlogramme_eviews(
        rend_sq, lags=lags, titre='Correlogramme -- Rendements carres (r_t^2)',
    ))

    return story


# =============================================================================
# Section 5 — ARIMA Box-Jenkins
# =============================================================================



# =============================================================================
# Section 5 — ARIMA Box-Jenkins
# =============================================================================

def section_5(prix: pd.Series, arima_result: dict, config: dict,
              rendements: pd.Series = None) -> list:
    """
    Section 5 : Modelisation ARIMA -- methode Box-Jenkins.
    arima_result : dict {df, p_opt, d_opt, q_opt, aic} issu de selectionner_arima().
    Le modele final est re-estime en interne pour extraire residus et diagnostics.
    rendements : log-rendements en % passes depuis l'orchestrateur.
                 Quand d_opt==0, le pipeline a estime l'ARIMA sur les rendements
                 (series deja stationnaires apres d=1 en stationnarite) — pas sur le
                 log-prix. On re-estime donc sur la meme serie que main.py.
    """
    prix  = _sq(prix)
    story = []
    lags   = config.get('sorties_etendues', {}).get('correlogramme_lags', 36)
    p_opt  = int(arima_result.get('p_opt', 0))
    d_opt  = int(arima_result.get('d_opt', 1))
    q_opt  = int(arima_result.get('q_opt', 0))
    df_grid = arima_result.get('df', pd.DataFrame())

    log_prix = np.log(prix.dropna())

    # Serie sur laquelle l'ARIMA a ete ajuste dans le pipeline principal :
    # d_opt==0 + rendements disponibles → log-prix est I(1), ARIMA sur rendements
    # d_opt >0 ou pas de rendements   → ARIMA sur log-prix avec d diff. internes
    if d_opt == 0 and rendements is not None:
        serie_arima = _sq(rendements).dropna()
    else:
        serie_arima = log_prix

    spec_str = f'ARIMA({p_opt},{d_opt},{q_opt})'

    story.append(_h1('5. Modelisation ARIMA -- Methode Box-Jenkins'))
    story.append(_p(
        'La methode Box-Jenkins (1976) proceede par trois etapes : '
        '(1) identification de l\'ordre (p,d,q) via la grille AIC, '
        '(2) estimation par maximum de vraisemblance, '
        '(3) validation sur les residus (bruit blanc, normalite, absence d\'effet ARCH).'
    ))

    # 5.0 Encadre pedagogique — interpretation economique (Phase 1.1)
    msg_ped  = arima_result.get('message_pedagogique', '')
    code_int = arima_result.get('interpretation', '')
    if msg_ped:
        story.extend(_encadre_pedagogique(msg_ped, code_int, config))

    # 5.1 Grille AIC
    story.append(_h2(f'5.1 Grille AIC normalisee -- Top 10'))
    story.append(_p(
        'AIC normalise = -2*logL/n + 2*k/n (format EViews). '
        f'Modele retenu : {spec_str} (fond jaune).'
    ))
    _log(f'[PDF] Grille ARIMA ({len(df_grid)} modeles)...')
    try:
        story.extend(_tab_grille_arima(df_grid, p_opt, d_opt, q_opt, n_top=10))
    except Exception as e:
        _warn(f'Section 5 grille ARIMA : {e}')

    story.append(_spacer(0.5))

    # 5.2 Modele retenu — re-estimation
    story.append(_h2(f'5.2 Modele retenu : {spec_str}'))
    _log(f'[PDF] Re-estimation {spec_str}...')
    fit_arima = None
    residus   = None
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fit_arima = ARIMA(serie_arima, order=(p_opt, d_opt, q_opt)).fit()
        residus   = fit_arima.resid.dropna()
        _tk = (config.get('data', {}).get('ticker', 'ACTIF')
               .replace('=F', '').replace('=X', '').replace('^', '').upper()[:8])
        story.extend(bloc_eviews_estimation(
            fit_arima, dep_var=f'DL{_tk}',
            method=f'Least Squares -- {spec_str}'))
    except Exception as e:
        _warn(f'Section 5 re-estimation ARIMA : {e}')
        story.append(_p(f'Estimation {spec_str} non disponible : {e}'))

    story.append(_spacer(0.5))

    # 5.3 Diagnostics residus
    story.append(_h2('5.3 Diagnostics des residus'))

    story.append(_h2('5.3.1 Correlogramme des residus'))
    story.append(_p(
        'Les residus d\'un ARIMA(p,d,q) correctement specifie doivent etre '
        'un bruit blanc : absence d\'autocorrelation significative.'
    ))
    if residus is not None and len(residus) > lags + 5:
        story.extend(_correlogramme_eviews(
            residus, lags=lags, titre=f'Correlogramme residus -- {spec_str}',
        ))
    else:
        story.append(_p('Residus non disponibles.'))

    story.append(_spacer(0.3))

    story.append(_h2('5.3.2 Correlogramme des residus carres'))
    story.append(_p(
        'Les autocorrelations significatives dans eps_t^2 trahissent un '
        'effet ARCH non capture par le modele ARIMA, justifiant '
        'l\'introduction d\'une equation de variance conditionnelle (GARCH).'
    ))
    if residus is not None and len(residus) > lags + 5:
        resid_sq = (residus ** 2)
        resid_sq.name = 'resid2'
        story.extend(_correlogramme_eviews(
            resid_sq, lags=lags, titre=f'Correlogramme residus carres -- {spec_str}',
        ))
    else:
        story.append(_p('Residus non disponibles.'))

    story.append(_spacer(0.4))

    # 5.4 ARCH-LM
    story.append(_h2('5.4 Tests ARCH-LM -- Engle (1982)'))
    story.append(_p(
        'H0 : absence d\'effet ARCH. Un rejet confirme la dependance '
        'temporelle de la variance et justifie la modelisation GARCH.'
    ))
    _log('[PDF] Calcul ARCH-LM...')
    if residus is not None and len(residus) > 15:
        try:
            from tickerlab.core.rapport._stats import arch_lm
            alm = arch_lm(residus, lags=[1, 4, 8, 12])
            story.extend(_tab_arch_lm(alm))
        except Exception as e:
            _warn(f'Section 5 ARCH-LM : {e}')
    else:
        story.append(_p('Residus non disponibles.'))

    story.append(_spacer(0.4))

    # 5.5 Normalite residus
    story.append(_h2('5.5 Normalite des residus'))
    if residus is not None:
        try:
            sd_resid = stats_desc(residus)
            story.extend(_tab_stats_desc(sd_resid, f'Residus {spec_str}'))
            jb_pv = float(sd_resid.get('jb_pval', 1.0))
            if jb_pv < 0.05:
                story.append(_p(
                    f'Jarque-Bera rejette la normalite des residus (p = {_fmt_pval(jb_pv)}). '
                    'Ce resultat, combine a l\'effet ARCH detecte, motive '
                    'l\'utilisation d\'une distribution leptokurtique (Student, GED) '
                    'dans le modele GARCH.'
                ))
            else:
                story.append(_p(
                    f'Jarque-Bera ne rejette pas la normalite des residus (p = {_fmt_pval(jb_pv)}).'
                ))
        except Exception as e:
            _warn(f'Section 5 normalite residus : {e}')

    return story


# =============================================================================
# Section 6 — Modelisation GARCH
# =============================================================================



# =============================================================================
# Section 6 — Modelisation GARCH
# =============================================================================

def _tab_component_garch(cg: dict) -> list:
    """
    Tableau Component GARCH (Engle & Lee 1999) — 6 lignes de paramètres.

    Colonnes : Paramètre | Estimé | Std. Err. | p-value | Contrainte Engle-Lee

    Parameters
    ----------
    cg : dict
        Résultat de ``estimer_component_garch()`` — jamais None ici.

    Returns
    -------
    list of Flowable
    """
    from reportlab.lib import colors as _rlc

    th   = _th()
    ck   = cg.get('constraints_ok', {})
    se   = cg.get('std_errors', {})
    pv   = cg.get('pvalues', {})
    ok   = '✓'
    nok  = '✗'

    # Contrainte associée à chaque paramètre
    _constraint_label = {
        'omega': 'omega > 0',
        'rho':   '0 < rho < 1 (C2)',
        'phi':   '',
        'alpha': 'alpha >= 0',
        'beta':  'alpha+beta < 1 (C1)',
        'nu':    'nu > 2',
    }
    _constraint_check = {
        'omega': True,
        'rho':   ck.get('rho_in_01', True),
        'phi':   True,
        'alpha': True,
        'beta':  ck.get('alpha_plus_beta_lt_1', True),
        'nu':    True,
    }

    lignes = []
    extra  = []
    row_base = 2   # titre + header

    _labels = [
        ('omega', 'omega (long-run)'),
        ('rho',   'rho (permanence)'),
        ('phi',   'phi (feedback)'),
        ('alpha', 'alpha (ARCH)'),
        ('beta',  'beta (GARCH)'),
        ('nu',    'nu (Student-t)'),
    ]

    for i, (key, label) in enumerate(_labels):
        est  = cg.get(key, float('nan'))
        s    = se.get(key, float('nan'))
        p    = pv.get(key, float('nan'))
        c_ok = _constraint_check.get(key, True)
        c_lbl= _constraint_label.get(key, '')

        c_str = f'{ok} {c_lbl}' if c_ok else f'{nok} {c_lbl}'
        row = [label, _fmt(est, 6), _fmt(s, 6), _fmt_pval(p), c_str]
        lignes.append(row)

        # Surligne la cellule contrainte en rouge si violée
        if not c_ok:
            extra += [
                ('FONTNAME',  (4, row_base + i), (4, row_base + i), 'Courier-Bold'),
                ('TEXTCOLOR', (4, row_base + i), (4, row_base + i), th['warn']),
            ]

    # Ligne de séparation puis critères d'information
    sep_row = row_base + len(_labels)
    loglik  = cg.get('loglik', float('nan'))
    aic     = cg.get('aic',    float('nan'))
    bic     = cg.get('bic',    float('nan'))
    n_obs   = cg.get('n_obs',  0)
    aic_n   = aic / n_obs if n_obs > 0 else float('nan')
    bic_n   = bic / n_obs if n_obs > 0 else float('nan')

    lignes.append(['Log L',        _fmt(loglik, 4), '', '', ''])
    lignes.append(['AIC (norm.)',   _fmt(aic_n,  6), '', '', ''])
    lignes.append(['BIC (norm.)',   _fmt(bic_n,  6), '', '', ''])
    extra.append(('LINEABOVE', (0, sep_row), (-1, sep_row), 0.8, _rlc.black))

    # Séparation C3 (contrainte α+β < ρ)
    ab  = cg.get('alpha', 0.0) + cg.get('beta', 0.0)
    rho = cg.get('rho', 1.0)
    c3_ok  = ck.get('separation', True)
    c3_str = (f'{ok} alpha+beta ({ab:.4f}) < rho ({rho:.4f}) — separation OK'
              if c3_ok
              else f'{nok} VIOLATION : alpha+beta ({ab:.4f}) >= rho ({rho:.4f})')

    notes = [
        'Engle & Lee (1999, eq.7, p.482). Opt. SLSQP. Student-t innovations.',
        f'Contrainte C3 (separation) : {c3_str}.',
    ]
    if cg.get('phi_warning'):
        notes.append('ALERTE : |phi| > 0.5 — verifier la stationnarite (Engle & Lee 1999).')
    if cg.get('saturation_warning'):
        notes.append('ALERTE : contrainte C3 quasi-saturee (|alpha+beta - rho| < 1e-3).')
    if not cg.get('converged', True):
        notes.append('Note : SLSQP non converge — resultats indicatifs.')

    cw = [LARG_U * 0.30, LARG_U * 0.14, LARG_U * 0.14, LARG_U * 0.14, LARG_U * 0.28]
    return _tableau_eviews(
        titre='Component GARCH (Engle & Lee 1999) — decomposition permanente/transitoire',
        colonnes=['Parametre', 'Estim.', 'Std. Err.', 'p-value', 'Contrainte'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note=' | '.join(notes),
    )




def _fig_component_decomposition(cg: dict, config: dict):
    """
    Figure superposée σ_t (vol. cond.) et √q_t (composante permanente).

    Deux courbes sur un axe unique :
    - σ_t = √σ²_t : fin, gris moyen — volatilité conditionnelle totale.
    - √q_t : épais, rouge bordeaux — composante permanente (long terme).

    Annotations des crises si leurs dates tombent dans la série (config.events.crises).

    Parameters
    ----------
    cg     : dict   Résultat de estimer_component_garch().
    config : dict   Configuration globale (pour config.events.crises).

    Returns
    -------
    matplotlib.figure.Figure
    """
    q_t      = cg.get('q_t')
    sigma2_t = cg.get('sigma2_t')
    if q_t is None or sigma2_t is None:
        raise ValueError("q_t ou sigma2_t absent du resultat Component GARCH")

    dates   = sigma2_t.index
    vol_t   = np.sqrt(np.maximum(sigma2_t.values, 0.0))   # σ_t
    q_vol_t = np.sqrt(np.maximum(q_t.values,      0.0))   # √q_t

    th = _th()
    fig, ax = plt.subplots(figsize=(14, 3.8))

    ax.plot(dates, vol_t,   color='#999999', linewidth=0.7, alpha=0.85,
            label=r'$\sigma_t$ (vol. conditionnelle totale)')
    ax.plot(dates, q_vol_t, color='#8B0000', linewidth=1.8, alpha=0.90,
            label=r'$\sqrt{q_t}$ (composante permanente)')

    # ── Annotations crises ───────────────────────────────────────────────────
    try:
        date_min = pd.Timestamp(dates[0])
        date_max = pd.Timestamp(dates[-1])
        for ev in config.get('events', {}).get('crises', []):
            annee = int(ev.get('annee', 0))
            label = str(ev.get('label', ''))
            ts = pd.Timestamp(f'{annee}-01-01')
            if date_min <= ts <= date_max:
                ax.axvline(ts, color='#CC4444', linewidth=0.8, linestyle='--', alpha=0.5)
                ax.text(ts, ax.get_ylim()[1] * 0.92, label,
                        fontsize=5.5, color='#CC4444', rotation=90, va='top', ha='right')
    except Exception as exc:
        _sections_log.debug('_fig_component_decomposition : tracé événements historiques ignoré : %s', exc)

    ax.set_ylabel('Volatilite (memes unites que r_t)', fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    plt.tight_layout(pad=0.4)
    return fig




def _encadre_component_garch(cg: dict, config: dict) -> list:
    """
    Encadré synthétique Component GARCH (Phase 1.3).

    Affiche le tableau paramètres + figure superposée σ_t / √q_t.
    Retourne [] si cg est None ou vide.

    Parameters
    ----------
    cg     : dict   Résultat de estimer_component_garch() (peut être None/empty).
    config : dict   Configuration globale.
    """
    if not cg:
        return []

    story = []
    story.append(_h2('6.2bis Component GARCH (Engle & Lee, 1999) — decomposition permanente/transitoire'))
    story.append(_p(
        'Le modele Component GARCH decompose la variance conditionnelle en une '
        'composante permanente q_t (tendance de long terme, pilotee par rho) et '
        'une composante transitoire (sigma_t^2 - q_t, pilotee par alpha+beta). '
        'Contrainte de separation (C3) : alpha+beta < rho — non-negociable pour '
        'l\'interpretabilite economique (Engle & Lee 1999, eq. 7, p. 482).'
    ))
    try:
        story.extend(_tab_component_garch(cg))
    except Exception as e:
        _warn(f'Section 6.2bis tableau Component GARCH : {e}')

    story.append(_spacer(0.3))

    # ── Figure σ_t / √q_t superposés ─────────────────────────────────────────
    try:
        fig = _fig_component_decomposition(cg, config)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=4.2))
        story.append(_caption(
            'Figure 6.0bis -- Decomposition Component GARCH. '
            'Gris : volatilite conditionnelle sigma_t. '
            'Rouge bordeaux (epais) : composante permanente sqrt(q_t). '
            'Traits pointilles : dates de crises majeures (config.events.crises).'
        ))
    except Exception as e:
        _warn(f'Section 6.2bis figure Component GARCH : {e}')

    story.append(_spacer(0.3))
    return story




def _encadre_igarch(igarch_diagnostic: dict, config: dict) -> list:
    """
    Encadré d'alerte IGARCH thème-adaptatif (Phase 1.2).

    Apparaît uniquement si ``code in ('near_igarch', 'igarch_strict')``.
    Pour ``'mean_reverting'``, retourne une seule ligne de texte (pas de box).

    Parameters
    ----------
    igarch_diagnostic : dict
        Résultat de diagnostiquer_igarch() : clés code, persistance,
        half_life_periodes, half_life_jours_cal, wald_pval, message_pedagogique.
    config : dict
        Configuration globale (thème).

    Returns
    -------
    list of Flowable
    """
    from reportlab.platypus import Table, TableStyle, Paragraph

    if not igarch_diagnostic:
        return []

    code    = igarch_diagnostic.get('code', 'mean_reverting')
    message = igarch_diagnostic.get('message_pedagogique', '')
    pers    = igarch_diagnostic.get('persistance', float('nan'))

    hl_p = igarch_diagnostic.get('half_life_periodes')
    hl_j = igarch_diagnostic.get('half_life_jours_cal')
    freq = igarch_diagnostic.get('frequence_serie', 'daily')
    freq_label = {'daily': 'jours', 'weekly': 'semaines', 'monthly': 'mois'}.get(freq, 'périodes')

    # Pour mean_reverting : ligne simple, pas de box
    if code == 'mean_reverting':
        import math
        if hl_p is not None and not math.isnan(hl_p):
            hl_str = f'{hl_p:.1f} {freq_label} (≈{hl_j:.0f} j cal.)'
        else:
            hl_str = 'N/A'
        return [_p(
            f'<b>Persistance :</b> {pers:.4f} (mean-reverting, demi-vie : {hl_str}).'
        ), _spacer(0.2)]

    # near_igarch / igarch_strict : encadré
    if not message:
        return []

    theme = _th()
    bg_color     = theme['entete_fond']
    border_color = theme['warn']   # couleur d'alerte (rouge/accent selon thème)

    _titres = {
        'near_igarch':  'Alerte : Persistance elevee — zone near-IGARCH',
        'igarch_strict': 'Alerte : Persistance unitaire — frontiere IGARCH',
    }
    titre_box = _titres.get(code, 'Diagnostic IGARCH')

    styles    = _H._get_styles()
    title_par = Paragraph(f'<b>{titre_box}</b>', styles['Heading2'])
    msg_par   = Paragraph(message, styles['Body'])

    t = Table([[title_par], [msg_par]], colWidths=[LARG_U])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), bg_color),
        ('BOX',           (0, 0), (-1, -1), 1.5, border_color),
        ('LINEBELOW',     (0, 0), (-1, 0),  0.5, border_color),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return [t, _spacer(0.3)]




def section_6(rendements: pd.Series, df_garch: pd.DataFrame,
              garch_final, modele_nom: str, config: dict,
              igarch_diagnostic: dict = None,
              component_garch_result: dict = None) -> list:
    """
    Section 6 : Modelisation GARCH (grille etendue + modele final).

    Parameters
    ----------
    rendements             : pd.Series   Log-rendements.
    df_garch               : DataFrame   Grille GARCH (sortie grid_search_garch).
    garch_final            : ARCHModelResult  Modele final estime.
    modele_nom             : str         Nom du modele retenu (ex: 'GJR-GARCH').
    config                 : dict
    igarch_diagnostic      : dict, optional  Resultat diagnostiquer_igarch() (Phase 1.2).
    component_garch_result : dict, optional  Resultat estimer_component_garch() (Phase 1.3).
    """
    story = []
    lags  = config.get('sorties_etendues', {}).get('correlogramme_lags', 36)

    # Infos sur le modele selectionne — extraites depuis garch_final.params
    # pour garantir la coherence titre/tableau/persistance (pas de filtrage df_garch).
    try:
        pnames = list(garch_final.params.index)
        p_sel  = sum(1 for k in pnames if k.startswith('alpha['))
        o_sel  = sum(1 for k in pnames if k.startswith('gamma['))
        q_sel  = sum(1 for k in pnames if k.startswith('beta['))
        pset   = set(pnames)
        if 'lambda' in pset:
            dist_sel = 'skewt'
        elif 'eta' in pset:
            dist_sel = 'ged'
        elif 'nu' in pset:
            dist_sel = 't'
        else:
            dist_sel = 'normal'
    except Exception:
        p_sel, o_sel, q_sel, dist_sel = 1, 0, 1, ''

    if o_sel > 0:
        spec_str = f'{modele_nom}({p_sel},{o_sel},{q_sel}) [{dist_sel}]'
    else:
        spec_str = f'{modele_nom}({p_sel},{q_sel}) [{dist_sel}]'

    story.append(_h1('6. Modelisation GARCH'))
    story.append(_p(
        'La grille GARCH etendue couvre cinq familles de modeles '
        '(GARCH, GJR-GARCH, EGARCH, TGARCH, APARCH) et quatre distributions '
        'des innovations (normale, t, skew-t, GED), soit jusqu\'a '
        '80 specifications par ordre (p,q). La selection scientifique repose '
        'sur trois etapes : (1) filtre de validite — convergence, '
        'significativite des parametres de variance, stationnarite '
        '(persistance < 1) ; (2) tests de specification simultanes sur les '
        'residus standardises z_t (Ljung-Box ordre 10 sur z_t et z_t^2, '
        'Engle-Ng 1993) — un modele doit passer les trois simultanement ; '
        '(3) selection par BIC avec preference de parcimonie dans la fenetre '
        'Burnham-Anderson (delta BIC < 2).'
    ))

    # 6.1 Grille comparative
    story.append(_h2(f'6.1 Grille comparative -- Top 15 par BIC'))
    story.append(_p(
        f'Modele retenu : {spec_str} (fond jaune). '
        'Colonne Sig. = 1 si tous les parametres de variance sont significatifs '
        'au seuil retenu (filtre etape 1). '
        'Colonne Delta = puissance delta estimee (APARCH/TGARCH uniquement).'
    ))
    _log(f'[PDF] Affichage grille GARCH ({len(df_garch)} modeles)...')
    try:
        story.extend(_tab_grille_garch(
            df_garch, modele_nom, p_sel, q_sel, dist_sel, n_top=15, o_sel=o_sel,
        ))
    except Exception as e:
        _warn(f'Section 6 grille GARCH : {e}')

    # Avertissement fallback aic_global (aucun modele tous_sig=True)
    try:
        has_sig = bool(df_garch['tous_sig'].any())
    except Exception:
        has_sig = True
    if not has_sig:
        story.append(_p(
            '<b>Note :</b> aucun modele de la grille n\'a tous ses coefficients '
            'de variance significatifs au seuil retenu. Le modele affiche est '
            'le meilleur par AIC global (regle de fallback). Cette situation est '
            'frequente lorsque alpha[1] converge vers zero (effet ARCH faible) '
            'ou que beta[1] sature pres de 1 (persistance elevee).'
        ))

    story.append(_spacer(0.5))

    # 6.2 Estimation du modele final
    story.append(_h2(f'6.2 Estimation du modele retenu : {spec_str}'))
    _log(f'[PDF] Tableau coefficients {spec_str}...')
    try:
        _tk = (config.get('data', {}).get('ticker', 'ACTIF')
               .replace('=F', '').replace('=X', '').replace('^', '').upper()[:8])
        _dist_lbl = eviews_dist_label(type(garch_final.model.distribution).__name__)
        story.extend(bloc_eviews_estimation(
            garch_final, dep_var=f'DL{_tk}',
            method=f'ML ARCH -- {_dist_lbl} distribution',
            nom_loi=_dist_lbl))
    except Exception as e:
        _warn(f'Section 6 coefs GARCH : {e}')

    # ── Diagnostic IGARCH (Phase 1.2) ─────────────────────────────────────────
    if igarch_diagnostic:
        story.extend(_encadre_igarch(igarch_diagnostic, config))
    else:
        story.append(_spacer(0.5))

    # ── Component GARCH (Phase 1.3) ───────────────────────────────────────────
    if component_garch_result:
        story.extend(_encadre_component_garch(component_garch_result, config))

    # 6.3 Diagnostics post-estimation
    story.append(_h2('6.3 Diagnostics post-estimation'))

    # Extraire z_t et sigma_t
    z_t     = None
    sigma_t = None
    dates   = None
    try:
        sigma_t = garch_final.conditional_volatility
        z_t     = (garch_final.resid / sigma_t).dropna()
        dates   = garch_final.conditional_volatility.index
    except Exception as e:
        _warn(f'Section 6 extraction z_t : {e}')

    # 6.3.1 Residus standardises
    story.append(_h2('6.3.1 Residus standardises z_t'))
    if z_t is not None:
        try:
            fig = _fig_residus_std(z_t.values, z_t.index)
            story.append(_embed_figure(fig, width_cm=15.5, height_cm=3.8))
            story.append(_caption(
                'Figure 6.1 -- Residus standardises z_t = eps_t/sigma_t. '
                'Traits pointilles : bandes +/- 1.96 (quantiles normaux a 5%).'
            ))
        except Exception as e:
            _warn(f'Section 6 figure z_t : {e}')

        story.append(_spacer(0.3))

        story.append(_h2('6.3.2 Correlogramme z_t'))
        story.extend(_correlogramme_eviews(
            z_t, lags=lags, titre=f'Correlogramme z_t -- {spec_str}',
        ))

        story.append(_spacer(0.3))

        story.append(_h2('6.3.3 Correlogramme z_t carres'))
        story.append(_p(
            'L\'absence d\'autocorrelation dans z_t^2 valide que '
            'le modele GARCH capture correctement la structure ARCH.'
        ))
        z_sq = z_t ** 2
        z_sq.name = 'z2'
        story.extend(_correlogramme_eviews(
            z_sq, lags=lags, titre=f'Correlogramme z_t^2 -- {spec_str}',
        ))
    else:
        story.append(_p('Residus standardises non disponibles.'))

    story.append(_spacer(0.4))

    # 6.3.4 Test biais de signe
    story.append(_h2('6.3.4 Test de biais de signe -- Engle & Ng (1993)'))
    story.append(_p(
        'H0 : les innovations negatives et positives ont un impact '
        'symetrique sur la variance conditionnelle.'
    ))
    _log('[PDF] Calcul test biais de signe...')
    if z_t is not None:
        try:
            from tickerlab.core.rapport._stats import sign_bias_test
            sbt = sign_bias_test(z_t.values)
            story.extend(_tab_sign_bias(sbt))
        except Exception as e:
            _warn(f'Section 6 biais de signe : {e}')

    story.append(_spacer(0.4))

    # 6.4 Persistance
    story.append(_h2('6.4 Persistance'))
    try:
        nobs = int(garch_final.nobs)
        story.extend(_tab_persistance(modele_nom, garch_final.params, nobs))
    except Exception as e:
        _warn(f'Section 6 persistance : {e}')

    # sigma_t et figure volatilite conditionnelle : section 7

    return story


# =============================================================================
# Helpers prives sections 7-11
# =============================================================================

