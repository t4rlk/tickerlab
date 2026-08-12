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
from tickerlab.core.rapport._stats import (
    stats_desc, adf_complet, pp_complet, kpss_complet,
)
from tickerlab.core.rapport.sections._common import (
    _sections_log, _log, _th, _is_nan, _int_or_na, _sq,
    _encadre_pedagogique, _page_break_if, _lags_rapport,
    _max_lignes_tableau, _split_tableau, _extra_pval_bold,
)
from tickerlab.core.rapport.sections.stationarite import _tab_stats_desc
from tickerlab.core.rapport.sections.arima_garch import _fig_volatilite



# =============================================================================
# Helpers prives sections 7-11
# =============================================================================

def _fmt_lr(val) -> str:
    """Formate une statistique LR (float ou string 'n/a')."""
    if isinstance(val, str):
        return val
    try:
        return _fmt(float(val), 4)
    except Exception:
        return 'N/A'




def _fmt_pval_bt(val) -> str:
    """Formate une p-value de backtest (float ou string 'n/a')."""
    if isinstance(val, str):
        return val
    try:
        return _fmt_pval(float(val))
    except Exception:
        return 'N/A'




def _verdict_style(verdict: str, row_idx: int, col_idx: int, th: dict) -> list:
    """Styles conditionnels pour les colonnes Verdict (OK=accent, NON=warn)."""
    if verdict == 'OK':
        return [('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), th['accent'])]
    if verdict == 'NON':
        return [
            ('FONTNAME',  (col_idx, row_idx), (col_idx, row_idx), 'Courier-Bold'),
            ('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), th['warn']),
        ]
    return []




def _tab_var_statique(df_vt: pd.DataFrame, label: str = 'VaR') -> list:
    """Tableau VaR ou TVaR multi-methodes x multi-niveaux."""
    if label == 'VaR':
        cols  = ['VaR Historique', 'VaR Normale', 'VaR Student',
                 'VaR Cornish-Fisher', 'VaR GARCH', 'VaR Monte Carlo (1j)']
        short = ['Histor.', 'Normale', 'Student', 'C-F', 'GARCH', 'MC']
    else:
        cols  = ['TVaR Historique', 'TVaR Normale', 'TVaR Student',
                 'TVaR CF semi-empirique', 'TVaR GARCH', 'TVaR Monte Carlo (1j)']
        short = ['Histor.', 'Normale', 'Student', 'C-F', 'GARCH', 'MC']

    _cf_cols = {'VaR Cornish-Fisher', 'TVaR CF semi-empirique'}
    lignes = []
    for niveau in df_vt.index:
        row_data = [str(niveau)]
        for col in cols:
            try:
                v = float(df_vt.loc[niveau, col])
                if math.isnan(v) and col in _cf_cols:
                    row_data.append('N/A (CF non monotone)')
                else:
                    row_data.append(_fmt(v, 4))
            except Exception:
                row_data.append('N/A')
        lignes.append(row_data)

    w0    = LARG_U * 0.10
    w_col = (LARG_U - w0) / len(short)
    cw    = [w0] + [w_col] * len(short)
    return _tableau_eviews(
        titre=f'{label} statique -- comparaison des methodes (en % des rendements)',
        colonnes=['Niveau'] + short,
        lignes=lignes, col_widths=cw,
        note=f'{label} en %. Signe negatif = perte. C-F = Cornish-Fisher (expansion Edgeworth).',
    )




def _tab_backtest(df_bt: pd.DataFrame, niveau: str) -> list:
    """Tableau backtest Kupiec & Christoffersen pour un niveau donne."""
    th  = _th()
    sub = df_bt[df_bt['Niveau'] == niveau].reset_index(drop=True)
    if len(sub) == 0:
        return [_p(f'Aucune donnee backtest pour le niveau {niveau}.')]

    lignes   = []
    extra    = []
    row_base = 2

    for i, (_, r) in enumerate(sub.iterrows()):
        vuc   = str(r.get('Verdict UC', ''))
        vcc   = str(r.get('Verdict CC', ''))
        row_i = row_base + i
        lignes.append([
            str(r.get('Methode', '')),
            str(int(r.get('N viol.', 0))),
            _fmt(r.get('Taux obs.'),  4),
            _fmt(r.get('Taux theo.'), 4),
            _fmt_lr(r.get('LR_UC')),
            _fmt_pval_bt(r.get('p_UC')),
            _fmt_lr(r.get('LR_IND')),
            _fmt_pval_bt(r.get('p_IND')),
            vuc,
            vcc,
        ])
        extra += _verdict_style(vuc, row_i, 8, th)
        extra += _verdict_style(vcc, row_i, 9, th)

    cw = [LARG_U * v for v in [0.17, 0.08, 0.09, 0.09, 0.09, 0.09, 0.10, 0.09, 0.10, 0.10]]
    return _tableau_eviews(
        titre=f'Backtesting VaR {niveau} -- Kupiec & Christoffersen (OOS)',
        colonnes=['Methode', 'N viol.', 'Tx obs.', 'Tx theo.',
                  'LR_UC', 'p_UC', 'LR_IND', 'p_IND', 'Verd.UC', 'Verd.CC'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='LR_UC : Kupiec (1995). LR_IND : Christoffersen (1998). H0 : calibration correcte. OK si p > 5%.',
    )




def _tab_violations_annuelles(df_vol_95, df_vol_99, rendements) -> list:
    """Violations VaR GARCH par annee -- 95% et 99%."""
    col_95 = 'VaR 95% GARCH (%)'
    col_99 = 'VaR 99% GARCH (%)'
    th     = _th()

    rend_s       = rendements.dropna()
    idx99        = df_vol_99.index
    rend_aligned = rend_s.reindex(idx99)

    annees   = sorted({d.year for d in idx99 if hasattr(d, 'year')})
    lignes   = []
    extra    = []
    row_base = 2

    avail_95 = col_95 in df_vol_95.columns
    avail_99 = col_99 in df_vol_99.columns

    for enum_i, an in enumerate(annees):
        mask99 = pd.Series([d.year == an for d in idx99], index=idx99)
        n_an   = int(mask99.sum())
        r_an   = rend_aligned[mask99]

        # VaR 95%
        if avail_95:
            idx95    = df_vol_95.index
            common95 = idx99[mask99].intersection(idx95)
            if len(common95) > 0:
                r95   = rend_s.reindex(common95).values
                v95   = df_vol_95.loc[common95, col_95].values
                nv95  = int(np.sum(r95 < v95))
                tx95  = nv95 / n_an if n_an > 0 else 0.0
            else:
                nv95, tx95 = 0, 0.0
        else:
            nv95, tx95 = 0, 0.0

        # VaR 99%
        if avail_99:
            r99_vals = r_an.dropna().values
            v99_vals = df_vol_99.loc[r_an.dropna().index, col_99].values
            nv99     = int(np.sum(r99_vals < v99_vals))
            tx99     = nv99 / n_an if n_an > 0 else 0.0
        else:
            nv99, tx99 = 0, 0.0

        row_i = row_base + enum_i
        lignes.append([str(an), str(n_an),
                       str(nv95), f'{tx95*100:.2f}%',
                       str(nv99), f'{tx99*100:.2f}%'])

        # Alert 95% : hors [3%, 7%] si assez d'obs
        if n_an >= 50 and (tx95 > 0.07 or tx95 < 0.03):
            extra += [('TEXTCOLOR', (3, row_i), (3, row_i), th['warn']),
                      ('FONTNAME',  (3, row_i), (3, row_i), 'Courier-Bold')]
        # Alert 99% : > 2%
        if n_an >= 50 and tx99 > 0.02:
            extra += [('TEXTCOLOR', (5, row_i), (5, row_i), th['warn']),
                      ('FONTNAME',  (5, row_i), (5, row_i), 'Courier-Bold')]

    cw = [LARG_U * v for v in [0.12, 0.10, 0.14, 0.14, 0.14, 0.14]]
    # ajustement residuel
    cw[-1] += LARG_U - sum(cw)
    return _tableau_eviews(
        titre='Violations annuelles VaR GARCH -- 95% et 99%',
        colonnes=['Annee', 'N obs.', 'N viol 95%', 'Tx 95%', 'N viol 99%', 'Tx 99%'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='Tx theorique : 5% (95%) et 1% (99%). En rouge : ecart significatif de la cible (>2pp).',
    )




def _tab_ratio_tvar_var(df_vt: pd.DataFrame) -> list:
    """Ratio TVaR/VaR par methode et niveau."""
    methodes = [
        ('Historique',    'VaR Historique',       'TVaR Historique'),
        ('Normale',       'VaR Normale',           'TVaR Normale'),
        ('Student',       'VaR Student',           'TVaR Student'),
        ('Cornish-Fisher','VaR Cornish-Fisher',    'TVaR CF semi-empirique'),
        ('GARCH',         'VaR GARCH',             'TVaR GARCH'),
        ('Monte Carlo',   'VaR Monte Carlo (1j)',  'TVaR Monte Carlo (1j)'),
    ]
    niveaux = list(df_vt.index)
    lignes  = []

    for nm, vc, tc in methodes:
        row_data = [nm]
        for niv in niveaux:
            try:
                v_var  = float(df_vt.loc[niv, vc])
                v_tvar = float(df_vt.loc[niv, tc])
                ratio  = v_tvar / v_var if v_var != 0 else float('nan')
                row_data.append(_fmt(ratio, 3))
            except Exception:
                row_data.append('N/A')
        lignes.append(row_data)

    w0    = LARG_U * 0.24
    w_col = (LARG_U - w0) / len(niveaux)
    cw    = [w0] + [w_col] * len(niveaux)
    return _tableau_eviews(
        titre='Ratio TVaR/VaR par methode et niveau',
        colonnes=['Methode'] + list(niveaux),
        lignes=lignes, col_widths=cw,
        note='Ratio TVaR/VaR > 1 : queue plus epaisse que sous hypothese normale. Attendu : 1.10-1.50.',
    )




def _tab_mcneil_frey(mf_95: dict, mf_99: dict) -> list:
    """Tableau test McNeil & Frey aux niveaux 95% et 99%."""
    th       = _th()
    lignes   = []
    extra    = []
    row_base = 2

    for i, (niv, mf) in enumerate([('95%', mf_95), ('99%', mf_99)]):
        pv      = mf.get('p_value', float('nan'))
        verdict = str(mf.get('verdict', 'N/A'))
        lignes.append([
            niv,
            str(int(mf.get('n_exc', 0))),
            _fmt(mf.get('z_bar'), 4),
            _fmt_pval(pv),
            verdict,
        ])
        if not _is_nan(pv) and pv < 0.05:
            row_i = row_base + i
            extra += [
                ('FONTNAME',  (3, row_i), (3, row_i), 'Courier-Bold'),
                ('TEXTCOLOR', (3, row_i), (3, row_i), th['warn']),
            ]

    cw = [LARG_U * v for v in [0.10, 0.12, 0.14, 0.14, 0.50]]
    return _tableau_eviews(
        titre="Test McNeil & Frey (2000) -- Validite de l'Expected Shortfall",
        colonnes=['Niveau', 'N exc.', 'z-bar', 'p-value', 'Verdict'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note="H0 : ES correctement estime (E[z_t]=0). Bootstrap bilateral B=1000, seed=42. Rejet si p < 5%.",
    )




def _fig_vol_rendements(rendements, garch_final, config, ticker: str) -> plt.Figure:
    """Figure double axe : rendements (gauche, gris) + sigma_t (droite, couleur)."""
    th     = _th()
    rend   = rendements.dropna()
    sigma  = garch_final.conditional_volatility
    dates  = sigma.index
    rend_a = rend.reindex(dates)
    crises = config.get('events', {}).get('crises', [])

    fig, ax1 = plt.subplots(figsize=(14, 4.2))
    ax2 = ax1.twinx()

    ax1.plot(dates, rend_a.values, color='#999999', lw=0.50, alpha=0.65, label='r_t')
    ax1.axhline(0, color='black', lw=0.30)
    ax1.set_ylabel('Rendement (%)', fontsize=8, color='#555555')
    ax1.tick_params(axis='y', labelsize=7, labelcolor='#555555')

    ax2.plot(dates, sigma.values, color=_hex(th['accent']), lw=0.80, alpha=0.90, label='sigma_t')
    ax2.set_ylabel('sigma_t (%)', fontsize=8, color=_hex(th['accent']))
    ax2.tick_params(axis='y', labelsize=7, labelcolor=_hex(th['accent']))

    trans = ax1.get_xaxis_transform()
    for cr in crises:
        annee = cr.get('annee')
        dlist = [d for d in dates if hasattr(d, 'year') and d.year == annee]
        if dlist:
            ax1.axvline(dlist[0], color=_hex(th['warn']), lw=0.55, ls='--', alpha=0.50)
            ax1.text(dlist[0], 0.96, cr.get('label', ''), transform=trans,
                     fontsize=5.5, rotation=90, va='top', ha='right',
                     color=_hex(th['warn']), alpha=0.75)

    ax1.set_title(f'Rendements et volatilite conditionnelle -- {ticker}',
                  fontsize=9, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=8)
    ax1.tick_params(axis='x', labelsize=7)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.12, linestyle=':')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='upper right', fontsize=7)
    fig.tight_layout()
    return fig




def _fig_var_dynamique_dual(df_vol_95, df_vol_99, rendements, config, ticker: str) -> plt.Figure:
    """VaR dynamique 95% (orange) + 99% (rouge) avec depassements et bandes crises."""
    th     = _th()
    col_95 = 'VaR 95% GARCH (%)'
    col_99 = 'VaR 99% GARCH (%)'
    crises = config.get('events', {}).get('crises', [])

    rend_s   = rendements.dropna()
    idx99    = df_vol_99.index
    rend_a   = rend_s.reindex(idx99).values

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(idx99, rend_a, color='#aaaaaa', lw=0.45, alpha=0.70, label='Rendement')

    if col_95 in df_vol_95.columns:
        var95 = df_vol_95[col_95].reindex(idx99).values
        ax.plot(idx99, var95, color='darkorange', lw=0.75, ls='--', alpha=0.85,
                label='VaR 95% GARCH')
        mask95 = np.isfinite(rend_a) & np.isfinite(var95) & (rend_a < var95)
        if mask95.any():
            ax.scatter(idx99[mask95], rend_a[mask95], color='darkorange',
                       s=8, alpha=0.60, zorder=4,
                       label=f'Dep. 95% ({mask95.sum()})')

    if col_99 in df_vol_99.columns:
        var99 = df_vol_99[col_99].values
        ax.plot(idx99, var99, color='red', lw=0.85, alpha=0.90, label='VaR 99% GARCH')
        mask99 = np.isfinite(rend_a) & np.isfinite(var99) & (rend_a < var99)
        if mask99.any():
            ax.scatter(idx99[mask99], rend_a[mask99], color='red',
                       s=10, zorder=5, label=f'Dep. 99% ({mask99.sum()})')

    trans = ax.get_xaxis_transform()
    for cr in crises:
        an   = cr.get('annee')
        xmin = pd.Timestamp(f'{an}-01-01')
        xmax = pd.Timestamp(f'{an}-12-31')
        if xmin < idx99[-1] and xmax > idx99[0]:
            ax.axvspan(xmin, xmax, alpha=0.07, color='gold', zorder=0)
            ax.text(xmin, 0.02, f' {cr.get("label","")[:14]}', transform=trans,
                    fontsize=5.8, color='#886600', rotation=90, va='bottom', ha='left')

    ax.axhline(0, color='black', lw=0.30, ls='--', alpha=0.50)
    ax.set_title(f'{ticker} -- VaR dynamique GARCH 95%/99% avec depassements',
                 fontsize=9, fontweight='bold')
    ax.set_ylabel('%', fontsize=8)
    ax.set_xlabel('Date', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle=':')
    fig.tight_layout()
    return fig




def _fig_tvar_dynamique(df_vol_99, rendements, config, ticker: str) -> plt.Figure:
    """VaR + TVaR dynamiques 99% avec depassements."""
    th       = _th()
    col_var  = 'VaR 99% GARCH (%)'
    col_tvar = 'TVaR 99% GARCH (%)'
    crises   = config.get('events', {}).get('crises', [])

    rend_s = rendements.dropna()
    idx99  = df_vol_99.index
    rend_a = rend_s.reindex(idx99).values

    fig, ax = plt.subplots(figsize=(14, 5.0))
    ax.plot(idx99, rend_a, color='#aaaaaa', lw=0.45, alpha=0.65, label='Rendement')

    if col_var in df_vol_99.columns:
        var99 = df_vol_99[col_var].values
        ax.plot(idx99, var99, color='red', lw=0.80, label='VaR 99% GARCH')
        mask = np.isfinite(rend_a) & np.isfinite(var99) & (rend_a < var99)
        if mask.any():
            ax.scatter(idx99[mask], rend_a[mask], color='red',
                       s=10, zorder=5, label=f'Dep. VaR ({mask.sum()})')

    if col_tvar in df_vol_99.columns:
        tvar99 = df_vol_99[col_tvar].values
        ax.plot(idx99, tvar99, color='darkred', lw=0.80, ls='--', label='TVaR 99% GARCH')

    trans = ax.get_xaxis_transform()
    for cr in crises:
        an   = cr.get('annee')
        xmin = pd.Timestamp(f'{an}-01-01')
        xmax = pd.Timestamp(f'{an}-12-31')
        if xmin < idx99[-1] and xmax > idx99[0]:
            ax.axvspan(xmin, xmax, alpha=0.07, color='gold', zorder=0)
            ax.text(xmin, 0.02, f' {cr.get("label","")[:14]}', transform=trans,
                    fontsize=5.8, color='#886600', rotation=90, va='bottom', ha='left')

    ax.axhline(0, color='black', lw=0.30, ls='--', alpha=0.50)
    ax.set_title(f'{ticker} -- VaR et TVaR dynamiques GARCH 99%',
                 fontsize=9, fontweight='bold')
    ax.set_ylabel('%', fontsize=8)
    ax.set_xlabel('Date', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle=':')
    fig.tight_layout()
    return fig


# =============================================================================
# Section 7 — Volatilite conditionnelle
# =============================================================================



# =============================================================================
# Section 7 — Volatilite conditionnelle
# =============================================================================

def section_7(rendements: pd.Series, garch_final, modele_nom: str,
              config: dict, ticker: str = 'BZ=F') -> list:
    """
    Section 7 : Volatilite conditionnelle sigma_t.
    Evolution, stats descriptives, representation superposee rendements/sigma_t.
    """
    story = []

    story.append(_h1('7. Volatilite conditionnelle'))
    story.append(_p(
        f'La volatilite conditionnelle sigma_t issue du modele {modele_nom} '
        'capture la dynamique temporelle du risque : elle est elevee pendant les '
        'crises et se contracte en periode calme. Ce phenomene d\'agregation '
        'de la volatilite est une propriete stylisee fondamentale des series financieres.'
    ))

    # 7.1 Evolution sigma_t
    story.append(_h2('7.1 Evolution de la volatilite conditionnelle sigma_t'))
    _log('[PDF] Figure sigma_t...')
    try:
        sigma_t = garch_final.conditional_volatility
        fig     = _fig_volatilite(sigma_t.values, sigma_t.index, config)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=4.2))
        story.append(_caption(
            f'Figure 7.1 -- Volatilite conditionnelle sigma_t ({modele_nom}). '
            'Les pics correspondent aux episodes de turbulence majeurs du marche petrolier.'
        ))
    except Exception as e:
        _warn(f'Section 7 figure sigma_t : {e}')

    story.append(_spacer(0.5))

    # 7.2 Stats descriptives sigma_t
    story.append(_h2('7.2 Statistiques descriptives -- sigma_t'))
    try:
        sd = stats_desc(garch_final.conditional_volatility.values)
        story.extend(_tab_stats_desc(sd, f'Volatilite conditionnelle sigma_t -- {modele_nom}'))
    except Exception as e:
        _warn(f'Section 7 stats sigma_t : {e}')

    story.append(_spacer(0.5))

    # 7.3 Dual-axis : rendements vs sigma_t
    story.append(_h2('7.3 Rendements et volatilite -- representation superposee'))
    story.append(_p(
        'La superposition des rendements et de la volatilite conditionnelle '
        'illustre l\'aggregation de la volatilite : les grands chocs (positifs '
        'ou negatifs) se concentrent dans les episodes de haute volatilite.'
    ))
    _log('[PDF] Figure dual-axis rendements/sigma_t...')
    try:
        fig = _fig_vol_rendements(rendements, garch_final, config, ticker)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=4.5))
        story.append(_caption(
            'Figure 7.2 -- Rendements r_t (axe gauche, gris) et sigma_t (axe droit, couleur). '
            'L\'agregation de la volatilite est clairement visible.'
        ))
    except Exception as e:
        _warn(f'Section 7 figure dual-axis : {e}')

    return story


# =============================================================================
# Section 8 — Value-at-Risk
# =============================================================================



# =============================================================================
# Section 8 — Value-at-Risk
# =============================================================================

def section_8(rendements: pd.Series, garch_final, df_var_tvar: pd.DataFrame,
              df_bt: pd.DataFrame, config: dict, ticker: str = 'BZ=F',
              df_violations: pd.DataFrame = None,
              df_params_drift: pd.DataFrame = None,
              stats_rolling: dict = None,
              fhs_result: dict = None,
              dm_gk_result: dict = None) -> list:
    """
    Section 8 : Value-at-Risk.
    VaR statique multi-methodes, figure dynamique 95%/99%, violations annuelles,
    backtesting Kupiec & Christoffersen, horizons Bale,
    derive des parametres rolling (§8.6), backtest rolling (§8.7),
    comparaison statistique DM+GK (§8.8).

    Parameters
    ----------
    df_var_tvar : DataFrame  Sortie de calculer_var_tvar(), index = ['90%','95%','99%'].
    df_bt       : DataFrame  Sortie de backtest_oos().
    """
    from tickerlab.core.var_engine import construire_df_vol
    story = []

    story.append(_h1('8. Value-at-Risk (VaR)'))
    story.append(_p(
        'La Value-at-Risk au niveau alpha designe la perte maximale non depassee '
        'avec probabilite alpha sur un horizon d\'un jour. Six methodes sont '
        'comparees : historique, normale, Student, Cornish-Fisher, GARCH dynamique '
        'et Monte Carlo (50 000 simulations).'
    ))

    # 8.1 VaR statique
    story.append(_h2('8.1 VaR statique -- comparaison des methodes'))
    _log('[PDF] Tableau VaR multi-methodes...')
    try:
        story.extend(_tab_var_statique(df_var_tvar, 'VaR'))
    except Exception as e:
        _warn(f'Section 8 VaR statique : {e}')

    # Bootstrap CI express (actif si express=true, meme si enabled=false)
    try:
        from tickerlab.core.var_engine import calculer_bootstrap_ci_var
        df_ci = calculer_bootstrap_ci_var(rendements, garch_final, config)
        if not df_ci.empty:
            boot_cfg = config.get('bootstrap', {}) if config else {}
            enabled  = bool(boot_cfg.get('enabled', False))
            express  = bool(boot_cfg.get('express', False))
            if enabled:
                ci_level  = float(boot_cfg.get('ci_level', 0.95))
                n_reps    = int(boot_cfg.get('n_boot', 500))
            else:
                niv_ic    = boot_cfg.get('niveaux_ic', [0.95])
                ci_level  = float(niv_ic[0]) if niv_ic else 0.95
                n_reps    = int(boot_cfg.get('n_replications', 200))
            bloc = int(boot_cfg.get('block_length' if express and not enabled else 'block_size', 10))
            lignes_ci = []
            for niv, row in df_ci.iterrows():
                lignes_ci.append([
                    str(niv),
                    _fmt(row['VaR GARCH'], 4) + ' %',
                    _fmt(row['CI lower'],  4) + ' %',
                    _fmt(row['CI upper'],  4) + ' %',
                ])
            story.append(_spacer(0.2))
            story.extend(_tableau_eviews(
                titre=f'VaR GARCH -- IC {int(ci_level*100)}% bootstrap stationnaire conditionnel',
                colonnes=['Niveau', 'VaR GARCH',
                          f'IC {int((1-ci_level)/2*100)}%',
                          f'IC {int((1-(1-ci_level)/2)*100)}%'],
                lignes=lignes_ci,
                col_widths=[LARG_U * v for v in [0.18, 0.27, 0.27, 0.28]],
                note=(
                    f'Bootstrap stationnaire conditionnel (Politis-Romano 1994), '
                    f'B={n_reps} replications, bloc={bloc}. '
                    'Parametres GARCH fixes : incertitude liee a l\'echantillonnage '
                    'des residus, non parametrique (Pascual-Romo-Ruiz 2006).'
                ),
            ))
    except Exception as e:
        _warn(f'Section 8 bootstrap CI : {e}')

    story.extend(_page_break_if(config))

    # 8.2 VaR dynamique -- figure 95% + 99%
    story.append(_h2('8.2 VaR dynamique GARCH -- figure 95% et 99%'))
    story.append(_p(
        'La VaR dynamique est recalculee a chaque date a partir de la volatilite '
        'conditionnelle sigma_t. Les depassements (rendement < VaR) sont marques '
        'par des points de couleur.'
    ))
    _log('[PDF] Figure VaR dynamique 95/99...')
    df_vol_95 = None
    df_vol_99 = None
    try:
        df_vol_95 = construire_df_vol(rendements, garch_final, alpha=0.95)
        df_vol_99 = construire_df_vol(rendements, garch_final, alpha=0.99)
        fig = _fig_var_dynamique_dual(df_vol_95, df_vol_99, rendements, config, ticker)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=5.5))
        story.append(_caption(
            'Figure 8.1 -- VaR dynamique GARCH 95% (orange, pointille) et 99% (rouge) '
            'sur la periode complete. Depassements = points de couleur.'
        ))
    except Exception as e:
        _warn(f'Section 8 figure VaR dynamique : {e}')

    story.extend(_page_break_if(config))

    # 8.3 Violations annuelles
    story.append(_h2('8.3 Violations annuelles'))
    story.append(_p(
        'Le decompte annuel des depassements permet de detecter les periodes '
        'de sous-estimation ou surestimation systematique du risque.'
    ))
    _log('[PDF] Tableau violations annuelles...')
    if df_vol_95 is not None and df_vol_99 is not None:
        try:
            story.extend(_tab_violations_annuelles(df_vol_95, df_vol_99, rendements))
        except Exception as e:
            _warn(f'Section 8 violations annuelles : {e}')
    else:
        story.append(_p('Donnees de violation non disponibles.'))

    story.extend(_page_break_if(config))

    # 8.4 Backtesting
    story.append(_h2('8.4 Backtesting OOS -- Kupiec & Christoffersen'))
    story.append(_p(
        'Backtesting out-of-sample sur l\'echantillon de test (30% final). '
        'Kupiec (1995) : couverture inconditionnelle (LR_UC ~ chi2(1)). '
        'Christoffersen (1998) : independance des violations (LR_IND ~ chi2(1)). '
        'Conjoint : LR_CC = LR_UC + LR_IND ~ chi2(2).'
    ))
    _log('[PDF] Tableaux backtest 95% et 99%...')
    for niv in ['95%', '99%']:
        try:
            story.extend(_tab_backtest(df_bt, niv))
            story.append(_spacer(0.3))
        except Exception as e:
            _warn(f'Section 8 backtest {niv} : {e}')

    # 8.5 Horizons multiples (Bale)
    story.extend(_page_break_if(config))
    story.append(_h2('8.5 VaR 99% multi-horizons -- regle Bale vs simulation'))
    story.append(_p(
        'Le cadre Bale III impose une VaR 99% sur 10 jours ouvrables. '
        'Deux approches sont comparees : (1) la regle de la racine carree '
        'VaR_H = VaR_1 * sqrt(H), valable sous i.i.d. normaux ; '
        '(2) la simulation directe H-step du modele GARCH, qui tient compte '
        'de la dynamique de la variance conditionnelle. '
        'L\'ecart relatif mesure le biais de la regle racine carree.'
    ))
    try:
        from tickerlab.core.var_engine import calculer_var_multi_horizon
        df_mh = calculer_var_multi_horizon(garch_final, config)
        lignes_mh = []
        for h, row in df_mh.iterrows():
            vs  = row['VaR sqrt(H)']
            vsm = row['VaR simulation']
            ec  = row['Ecart rel. (%)']
            lignes_mh.append([
                str(int(h)),
                _fmt(vs,  4) + ' %',
                _fmt(vsm, 4) + ' %' if not math.isnan(vsm) else 'N/A',
                (_fmt(ec, 2) + ' %' if not math.isnan(ec) else 'N/A'),
            ])
        story.extend(_tableau_eviews(
            titre='VaR GARCH 99% par horizon -- sqrt(H) vs simulation directe',
            colonnes=['Horizon (j)', 'VaR sqrt(H)', 'VaR simulation', 'Ecart rel.'],
            lignes=lignes_mh,
            col_widths=[LARG_U * v for v in [0.20, 0.28, 0.28, 0.24]],
            note=(
                'VaR sqrt(H) = VaR_1j * sqrt(H) (hypothese i.i.d.). '
                'Simulation = somme cumulee sur H pas via forecast GARCH (10 000 trajectoires). '
                'Bale III : horizon 10 jours, niveau 99%.'
            ),
        ))
    except Exception as e:
        _warn(f'Section 8.5 multi-horizon : {e}')
        story.append(_p(f'Horizons multiples indisponibles : {e}'))

    # ── 8.5bis FHS multi-horizons vs regle sqrt(H) ──────────────────────────
    story.extend(_page_break_if(config))
    story.append(_h2(
        '8.5bis VaR FHS 99% multi-horizons -- Filtered Historical Simulation'
    ))
    story.append(_p(
        'La FHS (Barone-Adesi, Giannopoulos & Vosper 1999) hybride la volatilite '
        'conditionnelle GARCH avec la distribution empirique des innovations '
        'standardisees z_t = e_t / s_t. La VaR multi-periodes est obtenue par '
        'simulation : a chaque chemin, les z_t sont tires avec remise depuis '
        'l\'historique et la variance GARCH est propagee de facon stochastique. '
        'La divergence FHS vs regle sqrt(H) mesure l\'impact des queues epaisses '
        'sur l\'horizon : si FHS < sqrt(H), la CLT compresse les queues '
        '(convergence vers le normal) ; si FHS > sqrt(H), le clustering de '
        'volatilite amplifie le risque multi-periodes.'
    ))
    _fhs = fhs_result or {}
    if _fhs.get('var_fhs_par_horizon'):
        try:
            from tickerlab.core.var_engine import calculer_var_multi_horizon
            df_mh2 = calculer_var_multi_horizon(garch_final, config)
            horizons_fhs = _fhs.get('horizons', [1, 5, 10, 22])
            alpha_ref    = 0.99

            lignes_fhs = []
            for H in horizons_fhs:
                vs  = df_mh2.loc[H, 'VaR sqrt(H)']   if H in df_mh2.index else float('nan')
                vsm = df_mh2.loc[H, 'VaR simulation'] if H in df_mh2.index else float('nan')
                vf  = _fhs['var_fhs_par_horizon'].get(H, {}).get(alpha_ref, float('nan'))

                ec_sqrt = ((vf - vs) / abs(vs) * 100
                           if not (math.isnan(vf) or math.isnan(vs) or abs(vs) < 1e-12)
                           else float('nan'))

                lignes_fhs.append([
                    str(int(H)),
                    (_fmt(vs,  4) + ' %') if not math.isnan(vs)  else 'N/A',
                    (_fmt(vsm, 4) + ' %') if not math.isnan(vsm) else 'N/A',
                    (_fmt(vf,  4) + ' %') if not math.isnan(vf)  else 'N/A',
                    (_fmt(ec_sqrt, 2) + ' %') if not math.isnan(ec_sqrt) else 'N/A',
                ])

            n_boot_str = str(_fhs.get('n_boot', '?'))
            note_fhs = (
                f'FHS : {n_boot_str} simulations, seed={_fhs.get("seed","?") if "seed" in _fhs else "42"}. '
                'sqrt(H) = VaR_1j * sqrt(H) (hypothese i.i.d.). '
                'Delta = (FHS - sqrt(H)) / |sqrt(H)| -- divergence due aux queues epaisses.'
            )
            if not _fhs.get('residus_iid_ok', True):
                eng_p = _fhs.get('engle_ng_pval', float('nan'))
                lb_p  = _fhs.get('lb_z2_pval',   float('nan'))
                note_fhs += (
                    f' ATTENTION : residus_iid_ok=False '
                    f'(Engle-Ng p={eng_p:.3f}, LB(z2) p={lb_p:.3f}). '
                    'Clustering residuel non capte par le GARCH — '
                    'VaR FHS aux horizons longs peut etre sous-estimee.'
                )

            story.extend(_tableau_eviews(
                titre='VaR FHS 99% par horizon -- FHS vs sqrt(H) vs simulation GARCH',
                colonnes=['Horizon (j)', 'VaR sqrt(H)', 'VaR GARCH sim', 'VaR FHS', 'Delta FHS-sqrt(H)'],
                lignes=lignes_fhs,
                col_widths=[LARG_U * v for v in [0.16, 0.21, 0.21, 0.21, 0.21]],
                note=note_fhs,
            ))

            if not _fhs.get('residus_iid_ok', True):
                story.append(_spacer(0.15))
                story.append(_p(
                    'AVERTISSEMENT : les diagnostics de specification suggèrent un '
                    'clustering residuel non capte par le GARCH retenu '
                    f'(Engle-Ng p={_fhs.get("engle_ng_pval", float("nan")):.3f}, '
                    f'LB(z2) p={_fhs.get("lb_z2_pval", float("nan")):.3f}). '
                    'La VaR FHS aux horizons longs (H >= 5) peut etre sous-estimee. '
                    'Envisager un modele GARCH de specification superieure ou une '
                    'specification avec clustering asymetrique.'
                ))
        except Exception as exc:
            _warn(f'Section 8.5bis FHS : {exc}')
            story.append(_p(f'FHS multi-horizons indisponible : {exc}'))
    else:
        story.append(_p(
            'FHS non active (fhs.enabled: false dans config.yaml) ou '
            'resultats non disponibles. Activer pour comparer avec la regle sqrt(H).'
        ))

    # ── 8.6 Derive des parametres GARCH (rolling) ────────────────────────────
    _rolling_active = (
        df_params_drift is not None
        and not (df_params_drift.empty if hasattr(df_params_drift, 'empty') else True)
    )
    story.extend(_page_break_if(config))
    story.append(_h2('8.6 Derive des parametres GARCH (rolling re-estimations)'))
    if not _rolling_active:
        story.append(_p(
            'Backtesting rolling non active (rolling_backtest.enabled: false). '
            'Activer dans config.yaml pour visualiser la stabilite des coefficients.'
        ))
    else:
        story.append(_p(
            'Les parametres principaux du modele GARCH sont re-estimes a chaque '
            'cycle mensuel sur la fenetre glissante. La derive dans le temps est '
            'un indicateur de stabilite structurelle du modele.'
        ))
        try:
            drift = df_params_drift.copy()
            # Colonnes d'interet : alpha, beta, gamma (ou omega1/alpha1/beta1 selon modele)
            param_cols = [c for c in drift.columns
                          if any(k in c.lower() for k in
                                 ['alpha', 'beta', 'gamma', 'omega', 'eta', 'lambda'])
                          and c != 'mu'][:6]  # max 6 parametres
            if not param_cols:
                param_cols = [c for c in drift.columns if c != 'mu'][:6]

            if param_cols:
                th  = _th()
                fig, axes = plt.subplots(len(param_cols), 1,
                                         figsize=(14, 2.0 * len(param_cols)),
                                         sharex=True)
                if len(param_cols) == 1:
                    axes = [axes]
                colors = [_hex(th['accent']), _hex(th['bordure']), _hex(th['warn']),
                          '#4a90d9', '#7b68ee', '#20b2aa']
                for ax, col, col_c in zip(axes, param_cols, colors):
                    vals = drift[col].dropna()
                    ax.plot(vals.index, vals.values, color=col_c, lw=1.0)
                    ax.axhline(vals.mean(), color=col_c, lw=0.6, ls='--', alpha=0.5)
                    ax.set_ylabel(col, fontsize=7)
                    ax.tick_params(labelsize=6)
                    ax.grid(True, alpha=0.15, linestyle=':')
                axes[-1].set_xlabel('Date re-estimation', fontsize=7)
                axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                fig.suptitle('Derive des parametres GARCH — rolling window',
                             fontsize=9, fontweight='bold')
                fig.tight_layout()
                story.append(_embed_figure(fig, width_cm=15.5,
                                           height_cm=2.0 * len(param_cols)))
                story.append(_caption(
                    f'Figure 8.2 -- Derive des {len(param_cols)} parametres '
                    'principaux du modele GARCH a chaque re-estimation mensuelle. '
                    'Les tirets = moyenne sur la periode.'
                ))
                plt.close(fig)
        except Exception as e:
            _warn(f'Section 8.6 figure derive : {e}')
            story.append(_p(f'Figure derive indisponible : {e}'))

    story.append(_spacer(0.4))

    # ── 8.7 Backtest OOS rolling window ─────────────────────────────────────
    story.append(_h2('8.7 Backtesting OOS rolling window'))
    if not _rolling_active or not stats_rolling:
        story.append(_p(
            'Backtesting rolling non active. '
            'Resultat disponible apres activation de rolling_backtest.enabled.'
        ))
    else:
        story.append(_p(
            'Backtest out-of-sample sur la periode hors fenetre initiale. '
            'La VaR est calculee avec les parametres de la derniere re-estimation. '
            'Tests : Kupiec (LR_UC), Christoffersen (LR_CC), DQ (Engle-Manganelli).'
        ))
        rb_cfg = config.get('rolling_backtest', {})
        story.append(_p(
            f'Fenetre : {rb_cfg.get("window_size",1000)} jours | '
            f'Re-estimation : tous les {rb_cfg.get("refit_every",22)} jours.'
        ))
        try:
            lignes_roll = []
            for niv in ['95%', '99%']:
                s = stats_rolling.get(niv, {})
                if not s:
                    continue
                t_eff  = s.get('T_eff', 0)
                n_viol = s.get('N_viol', 0)
                taux   = s.get('Taux_obs', float('nan'))
                p_uc   = s.get('p_UC', float('nan'))
                p_cc   = s.get('p_CC', float('nan'))
                dq_p   = s.get('DQ_pval', float('nan'))
                v_uc   = s.get('Verdict_UC', 'N/A')
                v_cc   = s.get('Verdict_CC', 'N/A')
                v_dq   = s.get('Verdict_DQ', 'N/A')
                lignes_roll.append([
                    niv,
                    str(n_viol), str(t_eff),
                    _fmt(taux, 4),
                    _fmt(p_uc,  4), v_uc,
                    _fmt(p_cc,  4), v_cc,
                    _fmt(dq_p,  4), v_dq,
                ])
            if lignes_roll:
                th_r = _th()
                extra_r = []
                _col_ok  = _hex(th_r['accent'])
                _col_non = _hex(th_r['warn'])
                for i, row in enumerate(lignes_roll):
                    row_idx = i + 2
                    for col_idx, verdict in [(5, row[5]), (7, row[7]), (9, row[9])]:
                        color = _col_ok if verdict == 'OK' else _col_non
                        extra_r.append(('BACKGROUND', (col_idx, row_idx),
                                        (col_idx, row_idx), color))
                cw_r = [LARG_U * v for v in [0.09, 0.08, 0.08, 0.08,
                                              0.10, 0.08, 0.10, 0.08, 0.10, 0.08]]
                story.extend(_tableau_eviews(
                    titre='Backtest rolling -- Kupiec, Christoffersen, DQ',
                    colonnes=['Niveau', 'N viol.', 'T eff.',
                              'Taux obs.', 'p_UC', 'UC', 'p_CC', 'CC', 'p_DQ', 'DQ'],
                    lignes=lignes_roll,
                    col_widths=cw_r,
                    extra_styles=extra_r,
                    note=(
                        'Rolling window : re-estimation mensuelle (22j) sur '
                        'fenetre fixe. UC = Kupiec, CC = Christoffersen conjoint, '
                        'DQ = Engle-Manganelli. OK = p > 5%.'
                    ),
                ))
        except Exception as e:
            _warn(f'Section 8.7 rolling table : {e}')
            story.append(_p(f'Table rolling indisponible : {e}'))

    # ── 8.8 Comparaison statistique DM + GK─────────────────────
    _dmgk = dm_gk_result or {}
    try:
        story.extend(_page_break_if(config))
        story.append(_h2('8.8 Comparaison statistique des methodes VaR '
                         '(Diebold-Mariano 1995 + Giacomini-Komunjer 2005)'))
        if not _dmgk or not any(isinstance(k, float) for k in _dmgk):
            story.append(_p(
                'Module DM-GK non active (dm_gk.enabled: false dans config.yaml) '
                'ou aucune paire disponible.'
            ))
        else:
            # Description methodologique
            story.append(_p(
                'La perte tick L_alpha(u_t) = u_t * (alpha - 1{u_t < 0}) '
                'ou u_t = r_t - VaR_t quantifie l\'erreur de prevision de '
                'chaque methode (Gonzalez-Rivera et al. 2004). '
                'Le test DM (Diebold & Mariano 1995) teste H0 : egalite de perte '
                'esperee inconditionnelle (stat N(0,1) bilateral, HAC Newey-West). '
                'Le test GK (Giacomini & Komunjer 2005, JBES 23:4) teste H0 '
                'conditionnellement aux instruments Z_t = [1, d_{t-1}] '
                '(stat chi2(q) unilateral droit -- la direction se lit dans DM). '
                'Quatre verdicts : model1_wins, model2_wins, egalite_stable, '
                'egalite_avec_instabilite (DM non rejete + GK rejete).'
            ))
            story.append(_spacer())

            alpha_test = _dmgk.get('alpha_test', 0.05)
            paires_pp  = _dmgk.get('paires_section_principale',
                                    ['GARCH dyn._vs_FHS',
                                     'Historique_vs_FHS',
                                     'GARCH dyn._vs_Historique'])
            niveaux_disp = [k for k in _dmgk if isinstance(k, float)]
            niveaux_disp = sorted(niveaux_disp, reverse=True)[:2]  # 99% puis 95%

            for alp in niveaux_disp:
                paires_alp = _dmgk.get(alp, {})
                if not paires_alp:
                    continue
                story.append(_p(f'<b>Niveau alpha = {int(alp*100)}% '
                                f'(seuil test = {alpha_test:.0%})</b>'))
                story.append(_spacer(0.15))

                # 3 paires structurantes (lookup order-insensitif : A_vs_B ou B_vs_A)
                from tickerlab.core.dm_gk import find_pair as _find_pair
                lignes = [['Paire', 'DM stat', 'p DM', 'GK stat', 'p GK chi2', 'Verdict']]
                for pp_key in paires_pp:
                    parts = pp_key.split('_vs_')
                    info = _find_pair(paires_alp, parts[0], parts[1]) if len(parts) == 2 else None
                    if info is None:
                        continue
                    label = pp_key.replace('_vs_', ' vs ')
                    verdict_short = {
                        'model1_wins':               'model1 gagne',
                        'model2_wins':               'model2 gagne',
                        'egalite_stable':            'egalite stable',
                        'egalite_avec_instabilite':  'egalite + instabilite',
                    }.get(info['verdict'], info['verdict'])
                    lignes.append([
                        label,
                        f"{info['dm_stat']:+.3f}",
                        f"{info['dm_pval']:.3f}",
                        f"{info['gk_stat']:.3f}",
                        f"{info['gk_pval']:.3f}",
                        verdict_short,
                    ])

                colonnes_dmgk = lignes[0]
                data_dmgk     = lignes[1:]
                if data_dmgk:
                    story.extend(_tableau_eviews(
                        titre=f'DM+GK -- paires structurantes -- alpha={int(alp*100)}%',
                        colonnes=colonnes_dmgk,
                        lignes=data_dmgk,
                        col_widths=[LARG_U * v for v in [0.30, 0.12, 0.10, 0.12, 0.12, 0.24]],
                        note=(
                            f'DM : H0 egalite perte esperee inconditionnelle, '
                            f'bilateral, HAC Newey-West. '
                            f'GK : H0 egalite conditionnelle, chi2(2). '
                            f'Seuil = {alpha_test:.0%}. '
                            f'd_bar < 0 : model1 perd moins en moyenne. '
                            f'egalite_avec_instabilite : pas de gagnant moyen '
                            f'mais profil temporel instable -- investiguer regime-dependance.'
                        ),
                    ))
                story.append(_spacer(0.3))

    except Exception as e:
        _warn(f'Section 8.8 DM-GK : {e}')
        story.append(_p(f'Section 8.8 indisponible : {e}'))

    return story


# =============================================================================
# Section 9 — TVaR / Expected Shortfall
# =============================================================================



# =============================================================================
# Section 9 — TVaR / Expected Shortfall
# =============================================================================

def section_9(rendements: pd.Series, garch_final, df_var_tvar: pd.DataFrame,
              config: dict, ticker: str = 'BZ=F') -> list:
    """
    Section 9 : TVaR / Expected Shortfall.
    TVaR statique, ratio TVaR/VaR, figure dynamique 99%, test McNeil-Frey.
    """
    from tickerlab.core.var_engine import construire_df_vol
    from tickerlab.core.rapport._stats import mcneil_frey_test
    story = []

    story.append(_h1('9. TVaR -- Expected Shortfall'))
    story.append(_p(
        'La TVaR (Tail Value-at-Risk), aussi appelee Expected Shortfall (ES) ou CVaR, '
        'mesure la perte moyenne conditionnelle au depassement de la VaR : '
        'ES_alpha = E[r_t | r_t < VaR_alpha]. '
        'Coherente au sens d\'Artzner et al. (1999), elle satisfait la '
        'sous-additivite et capture mieux le risque de queue extreme.'
    ))

    # 9.1 TVaR statique
    story.append(_h2('9.1 TVaR statique -- comparaison des methodes'))
    _log('[PDF] Tableau TVaR multi-methodes...')
    try:
        story.extend(_tab_var_statique(df_var_tvar, 'TVaR'))
    except Exception as e:
        _warn(f'Section 9 TVaR statique : {e}')

    story.append(_spacer(0.5))

    # 9.2 Ratio TVaR/VaR
    story.append(_h2('9.2 Ratio TVaR/VaR'))
    story.append(_p(
        'Le ratio TVaR/VaR mesure l\'epaisseur relative de la queue de distribution. '
        'Pour une loi normale a 95% : ratio ~ 1.15. Un ratio plus eleve traduit '
        'une queue plus epaisse (leptokurtose) que ne le suppose la normalite.'
    ))
    try:
        story.extend(_tab_ratio_tvar_var(df_var_tvar))
    except Exception as e:
        _warn(f'Section 9 ratio TVaR/VaR : {e}')

    story.append(_spacer(0.5))

    # 9.3 Figure TVaR dynamique 99%
    story.append(_h2('9.3 VaR et TVaR dynamiques GARCH 99%'))
    _log('[PDF] Figure TVaR dynamique 99%...')
    df_vol_99 = None
    try:
        df_vol_99 = construire_df_vol(rendements, garch_final, alpha=0.99)
        fig = _fig_tvar_dynamique(df_vol_99, rendements, config, ticker)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=5.2))
        story.append(_caption(
            'Figure 9.1 -- VaR 99% (rouge plein) et TVaR 99% (pointille) dynamiques GARCH. '
            'La TVaR est toujours inferieure a la VaR : elle capte la profondeur des pertes extremes.'
        ))
    except Exception as e:
        _warn(f'Section 9 figure TVaR : {e}')

    story.append(_spacer(0.5))

    # 9.4 Test McNeil-Frey
    story.append(_h2("9.4 Test McNeil & Frey (2000) -- Validite de l'ES"))
    story.append(_p(
        "H0 : l'Expected Shortfall est correctement estime -- "
        "E[(r_t - ES_t)/sigma_t | r_t < VaR_t] = 0. "
        "Le test est applique aux deux niveaux 95% et 99%."
    ))
    _log('[PDF] McNeil-Frey 95% et 99%...')

    _mf_nan = {'z_bar': float('nan'), 'n_exc': 0,
                'p_value': float('nan'), 'verdict': 'N/A'}
    mf_95 = dict(_mf_nan)
    mf_99 = dict(_mf_nan)

    try:
        df_vol_95_mf = construire_df_vol(rendements, garch_final, alpha=0.95)
        df_vol_99_mf = df_vol_99 if df_vol_99 is not None else \
                       construire_df_vol(rendements, garch_final, alpha=0.99)
        sigma_vec = garch_final.conditional_volatility
        rend_s    = rendements.dropna()

        for alpha_val, df_vol_a, col_var, col_tvar, mf_ref in [
            (0.95, df_vol_95_mf, 'VaR 95% GARCH (%)', 'TVaR 95% GARCH (%)', '95'),
            (0.99, df_vol_99_mf, 'VaR 99% GARCH (%)', 'TVaR 99% GARCH (%)', '99'),
        ]:
            try:
                idx        = df_vol_a.index
                r_arr      = rend_s.reindex(idx).values
                var_arr    = df_vol_a[col_var].values  if col_var  in df_vol_a.columns \
                             else np.full(len(idx), np.nan)
                es_arr     = df_vol_a[col_tvar].values if col_tvar in df_vol_a.columns \
                             else np.full(len(idx), np.nan)
                sigma_arr  = sigma_vec.reindex(idx).values

                mf = mcneil_frey_test(r_arr, var_arr, es_arr,
                                      sigma=sigma_arr, n_boot=1000, seed=42)
                if mf_ref == '95':
                    mf_95 = mf
                else:
                    mf_99 = mf
            except Exception as e_inner:
                _warn(f'McNeil-Frey alpha={alpha_val} : {e_inner}')

    except Exception as e:
        _warn(f'Section 9 McNeil-Frey setup : {e}')

    try:
        story.extend(_tab_mcneil_frey(mf_95, mf_99))
    except Exception as e:
        _warn(f'Section 9 tableau McNeil-Frey : {e}')

    story.append(_spacer(0.5))

    # 9.5 EVT-POT benchmark
    story.append(_h2('9.5 Benchmark EVT-POT (Peaks over Threshold)'))
    story.append(_p(
        'La theorie des valeurs extremes (EVT) modelise directement la queue gauche '
        'de la distribution des pertes via une loi de Pareto generalisee (GPD). '
        'Le seuil est fixe au 95e percentile des pertes. '
        'Les VaR et TVaR analytiques GPD sont comparees aux estimations GARCH. '
        'Test KS pour valider l\'ajustement (H0 : exceedances ~ GPD).'
    ))
    try:
        from tickerlab.core.benchmark_evt import comparer_garch_evt, ajuster_gpd
        evt_res = comparer_garch_evt(rendements, garch_final, config)
        gpd     = evt_res['gpd_fit']
        df_evt  = evt_res['tableau']

        # Tableau ajustement GPD
        xi_v   = gpd.get('xi', float('nan'))
        beta_v = gpd.get('beta', float('nan'))
        u_v    = gpd.get('u', float('nan'))
        n_u    = gpd.get('n_u', 0)
        n_tot  = gpd.get('n_total', 0)
        ks_s   = gpd.get('ks_stat', float('nan'))
        ks_p   = gpd.get('ks_pval', float('nan'))
        fit_ok = gpd.get('fit_ok', False)
        lignes_gpd = [
            ['Seuil u (95e pct pertes)',  _fmt(u_v, 4) + ' %'],
            ['Exceedances n_u / n',       f'{n_u} / {n_tot}'],
            ['Parametre de forme xi',     _fmt(xi_v, 4)],
            ['Parametre d\'echelle beta', _fmt(beta_v, 4)],
            ['KS stat.',                  _fmt(ks_s, 4)],
            ['KS p-value',                _fmt(ks_p, 4)],
            ['Ajustement (KS, seuil 5%)', 'OK' if fit_ok else 'REJET'],
        ]
        story.extend(_tableau_eviews(
            titre='Ajustement GPD -- parametres et test KS',
            colonnes=['Indicateur', 'Valeur'],
            lignes=lignes_gpd,
            col_widths=[LARG_U * 0.65, LARG_U * 0.35],
            note='MLE sur les exceedances (pertes > seuil u). xi > 0 = queue Pareto epaisse.',
        ))

        # Tableau comparatif GARCH vs EVT
        story.append(_spacer(0.3))
        if not df_evt.empty:
            lignes_cmp = []
            for _, row in df_evt.iterrows():
                var_v  = row['VaR']
                tvar_v = row['TVaR']
                lignes_cmp.append([
                    row['Niveau'],
                    row['Methode'],
                    _fmt(var_v,  4) + ' %' if not (isinstance(var_v,  float) and math.isnan(var_v))  else 'N/A',
                    _fmt(tvar_v, 4) + ' %' if not (isinstance(tvar_v, float) and math.isnan(tvar_v)) else 'N/A',
                ])
            story.extend(_tableau_eviews(
                titre='VaR et TVaR -- GARCH dynamique vs EVT-GPD',
                colonnes=['Niveau', 'Methode', 'VaR', 'TVaR'],
                lignes=lignes_cmp,
                col_widths=[LARG_U * v for v in [0.15, 0.28, 0.28, 0.29]],
                note='EVT-GPD = estimation inconditionnelle sur la queue historique complete.',
            ))
    except Exception as e:
        _warn(f'Section 9.5 EVT-POT : {e}')
        story.append(_p(f'EVT-POT indisponible : {e}'))

    return story


# =============================================================================
# Section 10 — Synthese et conclusions
# =============================================================================

