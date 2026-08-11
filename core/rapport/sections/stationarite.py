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



# ── Decisions stationnarite ───────────────────────────────────────────────────

def _dec_adf_pp(r: dict) -> str:
    """ADF/PP : H0 racine unitaire, rejet si stat < VC (valeurs negatives)."""
    try:
        s   = float(r['stat'])
        c1  = float(r['crit_1'])
        c5  = float(r['crit_5'])
        c10 = float(r['crit_10'])
        if s < c1:  return 'Stationnaire***'
        if s < c5:  return 'Stationnaire**'
        if s < c10: return 'Stationnaire*'
        return 'Non-stationnaire'
    except Exception:
        return 'N/A'




def _dec_kpss(r: dict) -> str:
    """KPSS : H0 stationnaire, rejet si stat > VC."""
    try:
        s, c5 = float(r['stat']), float(r['crit_5'])
        return 'Non-stationnaire' if s > c5 else 'Stationnaire'
    except Exception:
        return 'N/A'




# ── Tables stationnarite compactes (3 specs en colonnes) ─────────────────────

def _tab_adf(res: dict, label: str, note_nc: str = '') -> list:
    rn, rc, rct = res.get('n', {}), res.get('c', {}), res.get('ct', {})
    cw = [LARG_U * 0.28] + [LARG_U * 0.24] * 3
    lignes = [
        ['t-stat ADF',  _fmt(rn.get('stat'), 4),  _fmt(rc.get('stat'), 4),  _fmt(rct.get('stat'), 4)],
        ['p-value',     _fmt_pval(rn.get('pval')),_fmt_pval(rc.get('pval')),_fmt_pval(rct.get('pval'))],
        ['Lags (BIC)',  _int_or_na(rn.get('lags')),_int_or_na(rc.get('lags')),_int_or_na(rct.get('lags'))],
        ['VC 1%',       _fmt(rn.get('crit_1'), 3), _fmt(rc.get('crit_1'), 3), _fmt(rct.get('crit_1'), 3)],
        ['VC 5%',       _fmt(rn.get('crit_5'), 3), _fmt(rc.get('crit_5'), 3), _fmt(rct.get('crit_5'), 3)],
        ['VC 10%',      _fmt(rn.get('crit_10'),3), _fmt(rc.get('crit_10'),3), _fmt(rct.get('crit_10'),3)],
        ['Decision',    _dec_adf_pp(rn),           _dec_adf_pp(rc),           _dec_adf_pp(rct)],
    ]
    # row 3 = p-value (titre=0, header=1, t-stat=2, p-value=3)
    extra = _extra_pval_bold([rn, rc, rct], row_pval=3)
    note_base = 'H0 : racine unitaire. Lags : BIC, max = floor(12*(n/100)^0.25). *** 1%, ** 5%, * 10%.'
    note = note_base + (f' | {note_nc}' if note_nc else '')
    return _tableau_eviews(
        titre=f'ADF -- {label}',
        colonnes=['', 'Aucune', 'Constante', 'Const+Trend'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note=note,
    )




def _tab_pp(res: dict, label: str, note_nc: str = '') -> list:
    rn, rc, rct = res.get('n', {}), res.get('c', {}), res.get('ct', {})
    cw = [LARG_U * 0.28] + [LARG_U * 0.24] * 3
    lignes = [
        ['t-stat PP',   _fmt(rn.get('stat'), 4),  _fmt(rc.get('stat'), 4),  _fmt(rct.get('stat'), 4)],
        ['p-value',     _fmt_pval(rn.get('pval')),_fmt_pval(rc.get('pval')),_fmt_pval(rct.get('pval'))],
        ['Bandwidth (NW)', _int_or_na(rn.get('lags')),_int_or_na(rc.get('lags')),_int_or_na(rct.get('lags'))],
        ['VC 1%',       _fmt(rn.get('crit_1'), 3), _fmt(rc.get('crit_1'), 3), _fmt(rct.get('crit_1'), 3)],
        ['VC 5%',       _fmt(rn.get('crit_5'), 3), _fmt(rc.get('crit_5'), 3), _fmt(rct.get('crit_5'), 3)],
        ['VC 10%',      _fmt(rn.get('crit_10'),3), _fmt(rc.get('crit_10'),3), _fmt(rct.get('crit_10'),3)],
        ['Decision',    _dec_adf_pp(rn),           _dec_adf_pp(rc),           _dec_adf_pp(rct)],
    ]
    extra = _extra_pval_bold([rn, rc, rct], row_pval=3)
    note_base = 'H0 : racine unitaire. Bandwidth = noyau Newey-West (Andrews 1991).'
    note = note_base + (f' | {note_nc}' if note_nc else '')
    return _tableau_eviews(
        titre=f'Phillips-Perron -- {label}',
        colonnes=['', 'Aucune', 'Constante', 'Const+Trend'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note=note,
    )




def _tab_kpss(res: dict, label: str) -> list:
    rc, rct = res.get('c', {}), res.get('ct', {})
    cw = [LARG_U * 0.40] + [LARG_U * 0.30] * 2
    lignes = [
        ['stat KPSS', _fmt(rc.get('stat'), 4),   _fmt(rct.get('stat'), 4)],
        ['p-value',   _fmt_pval(rc.get('pval')),  _fmt_pval(rct.get('pval'))],
        ['Lags auto', _int_or_na(rc.get('lags')), _int_or_na(rct.get('lags'))],
        ['VC 1%',     _fmt(rc.get('crit_1'), 4),  _fmt(rct.get('crit_1'), 4)],
        ['VC 5%',     _fmt(rc.get('crit_5'), 4),  _fmt(rct.get('crit_5'), 4)],
        ['VC 10%',    _fmt(rc.get('crit_10'),4),  _fmt(rct.get('crit_10'),4)],
        ['Decision',  _dec_kpss(rc),              _dec_kpss(rct)],
    ]
    extra = _extra_pval_bold([rc, rct], row_pval=3)
    return _tableau_eviews(
        titre=f'KPSS -- {label}',
        colonnes=['', 'Niveau (c)', 'Tendance (ct)'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='H0 : stationnaire (inverse ADF/PP). Rejet si stat > VC.',
    )




def _tab_synthese(adf_lp, pp_lp, kpss_lp, adf_rd, pp_rd, kpss_rd) -> list:
    """Tableau de synthese stationnarite (spec. 'Constante' comme reference)."""

    def _conc(adf_r, pp_r, kpss_r) -> str:
        try:
            adf_s  = float(adf_r.get('c', {}).get('stat',  1)) < float(adf_r.get('c', {}).get('crit_5',  0))
            pp_s   = float(pp_r.get('c',  {}).get('stat',  1)) < float(pp_r.get('c',  {}).get('crit_5',  0))
            kpss_s = float(kpss_r.get('c',{}).get('stat',  0)) < float(kpss_r.get('c',{}).get('crit_5',  1))
            n_stat = int(adf_s) + int(pp_s) + int(kpss_s)
            if n_stat == 0: return 'I(1)'
            if n_stat == 3: return 'I(0)'
            return 'Ambigu'
        except Exception:
            return 'N/A'

    cw = [LARG_U * 0.22] + [LARG_U * 0.195] * 4
    lignes = [
        ['log(prix)',
         _dec_adf_pp(adf_lp.get('c', {})), _dec_adf_pp(pp_lp.get('c', {})),
         _dec_kpss(kpss_lp.get('c', {})),  _conc(adf_lp, pp_lp, kpss_lp)],
        ['d(log-prix)',
         _dec_adf_pp(adf_rd.get('c', {})), _dec_adf_pp(pp_rd.get('c', {})),
         _dec_kpss(kpss_rd.get('c', {})),  _conc(adf_rd, pp_rd, kpss_rd)],
    ]
    return _tableau_eviews(
        titre='Synthese des tests de stationnarite',
        colonnes=['Serie', 'ADF (c)', 'PP (c)', 'KPSS (c)', 'Conclusion'],
        lignes=lignes, col_widths=cw,
        note='Spec. "Constante" retenue comme reference. *** 1%, ** 5%, * 10%.',
    )


# ── Table statistiques descriptives ──────────────────────────────────────────



# ── Table statistiques descriptives ──────────────────────────────────────────

def _tab_stats_desc(sd: dict, label: str) -> list:
    """Table descriptive 2 colonnes : Statistique | Valeur."""
    jb_pval_str = _fmt_pval(sd.get('jb_pval')) + '  ' + _fmt_signif(sd.get('jb_pval', 1.0))
    n_val = _int_or_na(sd.get('n'))
    # Kurtosis : excès (Fisher, standard académique) + brute entre crochets
    # (convention EViews, normale = 3) pour lever toute ambiguïté de lecture.
    _ke = sd.get('kurt_exc')
    if isinstance(_ke, (int, float)) and _ke == _ke:  # non-NaN
        _kurt_str = f'{_fmt(_ke, 4)}  [brute {_fmt(_ke + 3.0, 4)}]'
    else:
        _kurt_str = _fmt(_ke, 4)
    lignes = [
        ['Observations',    n_val],
        ['Minimum',         _fmt(sd.get('min'),      4)],
        ['Maximum',         _fmt(sd.get('max'),      4)],
        ['Moyenne',         _fmt(sd.get('mean'),     4)],
        ['Mediane',         _fmt(sd.get('median'),   4)],
        ['Ecart-type',      _fmt(sd.get('std'),      4)],
        ['Skewness',        _fmt(sd.get('skew'),     4)],
        ['Kurtosis (exc.)', _kurt_str],
        ['Jarque-Bera',     _fmt(sd.get('jb_stat'),  4)],
        ['Prob. JB',        jb_pval_str],
    ]
    # Prob. JB = ligne 11 (titre=0, header=1, lignes data 2..11)
    th = _th()
    extra = []
    try:
        if float(sd.get('jb_pval', 1.0)) < 0.05:
            extra += [
                ('FONTNAME',  (1, 11), (1, 11), 'Courier-Bold'),
                ('TEXTCOLOR', (1, 11), (1, 11), th['warn']),
            ]
    except Exception as exc:
        _sections_log.debug('_tab_stats_desc : style JB-pval ignoré : %s', exc)
    cw = [LARG_U * 0.55, LARG_U * 0.45]
    return _tableau_eviews(
        titre=f'Statistiques descriptives -- {label}',
        colonnes=['Statistique', 'Valeur'],
        lignes=lignes, col_widths=cw, extra_styles=extra,
        note='Jarque-Bera : H0 = normalite. *** p<0.01, ** p<0.05, * p<0.10.',
    )


# ── Figure cours temporel ─────────────────────────────────────────────────────



# ── Figure cours temporel ─────────────────────────────────────────────────────

def _fig_cours(serie: pd.Series, titre: str, ylabel: str, config: dict) -> plt.Figure:
    """Cours temporel avec evenements/crises en axvline + texte axes-transform."""
    th    = _th()
    crises = config.get('events', {}).get('crises', [])

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(serie.index, serie.values, color=_hex(th['accent']), lw=0.85, alpha=0.92)

    trans = ax.get_xaxis_transform()   # x en data coords, y en axes [0,1]
    for cr in crises:
        annee = cr.get('annee')
        lbl   = cr.get('label', '')
        dates = [d for d in serie.index if hasattr(d, 'year') and d.year == annee]
        if dates:
            ax.axvline(dates[0], color=_hex(th['warn']),
                       lw=0.7, ls='--', alpha=0.65, ymin=0, ymax=0.93)
            ax.text(dates[0], 0.95, lbl, transform=trans,
                    fontsize=6.2, rotation=90, va='top', ha='right',
                    color=_hex(th['warn']), alpha=0.85)

    ax.set_title(titre, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel('Date', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.18, linestyle=':')
    fig.tight_layout()
    return fig


# ── Sections publiques ────────────────────────────────────────────────────────



# ── Sections publiques ────────────────────────────────────────────────────────

def section_1(prix: pd.Series, config: dict, ticker: str, prix_stats=None) -> list:
    """
    Section 1 : Prix de l'actif -- donnees brutes.
    Contenu : evolution du cours, stats descriptives, correlogramme.
    prix_stats : serie resampled a la frequence d'analyse (si different du daily).
                 Utilise pour 1.2/1.3 ; le graphique 1.1 garde la serie journaliere complete.
    """
    prix  = _sq(prix)
    freq  = config.get('data', {}).get('frequency', 'daily')
    nom   = nom_actif(ticker)
    prix_for_stats = _sq(prix_stats) if prix_stats is not None else prix
    story = []
    lags  = config.get('sorties_etendues', {}).get('correlogramme_lags', 36)

    story.append(_h1(f'1. Prix de {nom} ({ticker}) -- Donnees brutes'))
    story.append(_p(
        f'Cette section presente les donnees brutes de prix de {nom}. '
        'La serie couvre la periode complete retenue pour l\'analyse, avec identification '
        'des principaux chocs exogenes.'
    ))

    # 1.1 Cours
    story.append(_h2('1.1 Evolution du cours'))
    try:
        fig = _fig_cours(prix, f'Prix -- {nom} ({ticker})',
                         'Prix', config)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=4.5))
        story.append(_caption(
            f'Figure 1.1 -- Prix de {nom} ({ticker}). '
            'Les traits pointilles signalent les principaux evenements.'
        ))
        if prix_stats is not None and freq != 'daily':
            n_daily = len(prix.dropna())
            n_stats = len(prix_for_stats.dropna())
            story.append(_p(
                f'<i><font size="8">Note : graphique en donnees journalieres '
                f'({n_daily} obs.). Statistiques calculees sur la serie '
                f'{freq} ({n_stats} obs.) pour coherence avec l\'analyse.</font></i>'
            ))
    except Exception as e:
        _warn(f'Section 1 figure cours : {e}')

    story.append(_spacer(0.5))

    # 1.2 Stats descriptives (serie a la frequence d'analyse)
    story.append(_h2('1.2 Statistiques descriptives'))
    try:
        sd = stats_desc(prix_for_stats)
        story.extend(_tab_stats_desc(sd, f'Prix -- {nom} ({ticker})'))
    except Exception as e:
        _warn(f'Section 1 stats_desc : {e}')

    story.append(_spacer(0.5))

    # 1.3 Correlogramme (serie a la frequence d'analyse)
    story.append(_h2('1.3 Correlogramme ACF/PACF -- Prix brut'))
    story.append(_p(
        f'Correlogramme sur {lags} retards. Une autocorrelation positive et '
        'lentement decroissante est caracteristique d\'un processus I(1).'
    ))
    story.extend(_correlogramme_eviews(
        prix_for_stats, lags=lags, titre=f'Correlogramme -- Prix ({ticker})',
    ))

    return story




def section_2(prix: pd.Series, config: dict, ticker: str = '') -> list:
    """
    Section 2 : Log-prix.
    Contenu : transformation, evolution, stats descriptives, correlogramme.
    """
    prix  = _sq(prix)
    nom   = nom_actif(ticker) if ticker else 'l\'actif'
    story = []
    lags    = config.get('sorties_etendues', {}).get('correlogramme_lags', 36)
    log_prix = np.log(prix.dropna())
    log_prix.name = 'log_prix'

    story.append(_h1('2. Log-prix -- Transformation logarithmique'))
    story.append(_p(
        'La transformation logarithmique stabilise la variance conditionnelle et '
        'permet d\'interpreter les differences premieres comme des rendements continus : '
        'r_t = ln(P_t/P_{t-1}). Le log-prix constitue la serie de base pour '
        'les tests de stationnarite et la modelisation ARIMA-GARCH.'
    ))

    # 2.1 Cours log-prix
    story.append(_h2('2.1 Evolution du log-prix'))
    try:
        fig = _fig_cours(log_prix, f'Log-prix -- {nom}', 'ln(Prix)', config)
        story.append(_embed_figure(fig, width_cm=15.5, height_cm=4.5))
        story.append(_caption(
            f'Figure 2.1 -- Log-prix de {nom}. '
            'La tendance stochastique (marche aleatoire) reste apparente en niveau.'
        ))
    except Exception as e:
        _warn(f'Section 2 figure log-prix : {e}')

    story.append(_spacer(0.5))

    # 2.2 Stats descriptives
    story.append(_h2('2.2 Statistiques descriptives'))
    try:
        sd = stats_desc(log_prix)
        story.extend(_tab_stats_desc(sd, f'Log-prix -- {nom}'))
    except Exception as e:
        _warn(f'Section 2 stats_desc : {e}')

    story.append(_spacer(0.5))

    # 2.3 Correlogramme
    story.append(_h2('2.3 Correlogramme ACF/PACF -- Log-prix'))
    story.append(_p(
        'Le profil ACF decroissant hyperboliquement du log-prix est '
        'caracteristique d\'un processus integre d\'ordre 1 (I(1)).'
    ))
    story.extend(_correlogramme_eviews(
        log_prix, lags=lags, titre=f'Correlogramme -- Log-prix ({nom})',
    ))

    return story




def section_3(prix: pd.Series, rendements: pd.Series, config: dict) -> list:
    """
    Section 3 : Tests de stationnarite ADF / PP / KPSS.
    6 tableaux compacts (3 specs en colonnes) + synthese.
    """
    prix  = _sq(prix)
    story = []
    log_prix = np.log(prix.dropna())
    log_prix.name = 'log_prix'
    rend = rendements.dropna()

    story.append(_h1('3. Analyse de stationnarite'))
    story.append(_p(
        'La stationnarite est un pre-requis pour la validite des tests '
        'econometriques. Trois tests complementaires sont appliques -- '
        'ADF (Dickey-Fuller augmente, 1979), Phillips-Perron (1988) et '
        'KPSS (Kwiatkowski et al., 1992) -- sur le log-prix et les '
        'rendements journaliers. Les trois specifications usuelles sont '
        'presentees simultanement (sans derive, avec constante, constante + tendance).'
    ))

    # ── 3.1 ADF ──────────────────────────────────────────────────────────
    story.append(_h2('3.1 Test ADF -- Dickey-Fuller augmente (1979)'))
    story.append(_p(
        'H0 : presence d\'une racine unitaire (non-stationnarite). '
        'Selection automatique des retards par BIC '
        '(lag max = floor(12 * (n/100)^0.25), Schwert 1989).'
    ))
    mu_lp = float(log_prix.mean())
    _note_nc_lp = (
        f'Spec. "Aucune" suppose mu=0 — non pertinente ici '
        f'(mu(log-prix) = {mu_lp:.2f}). Interpreter uniquement "Constante" et "Const+Trend".'
    )

    _log('[PDF] Calcul ADF log-prix...')
    try:
        adf_lp = adf_complet(log_prix)
        story.extend(_tab_adf(adf_lp, 'log(prix)', note_nc=_note_nc_lp))
    except Exception as e:
        _warn(f'ADF log-prix : {e}')
        adf_lp = {'n': {}, 'c': {}, 'ct': {}}

    story.append(_spacer(0.4))
    _log('[PDF] Calcul ADF rendements...')
    try:
        adf_rd = adf_complet(rend)
        story.extend(_tab_adf(adf_rd, 'd(log-prix) -- rendements'))
    except Exception as e:
        _warn(f'ADF rendements : {e}')
        adf_rd = {'n': {}, 'c': {}, 'ct': {}}

    story.append(_spacer(0.6))

    # ── 3.2 PP ───────────────────────────────────────────────────────────
    story.append(_h2('3.2 Test Phillips-Perron (1988)'))
    story.append(_p(
        'H0 : racine unitaire. Correction non-parametrique de Newey-West '
        'pour l\'heteroscedasticite et l\'autocorrelation des residus.'
    ))
    _log('[PDF] Calcul PP log-prix...')
    try:
        pp_lp = pp_complet(log_prix)
        story.extend(_tab_pp(pp_lp, 'log(prix)', note_nc=_note_nc_lp))
    except Exception as e:
        _warn(f'PP log-prix : {e}')
        pp_lp = {'n': {}, 'c': {}, 'ct': {}}

    story.append(_spacer(0.4))
    _log('[PDF] Calcul PP rendements...')
    try:
        pp_rd = pp_complet(rend)
        story.extend(_tab_pp(pp_rd, 'd(log-prix) -- rendements'))
    except Exception as e:
        _warn(f'PP rendements : {e}')
        pp_rd = {'n': {}, 'c': {}, 'ct': {}}

    story.append(_spacer(0.6))

    # ── 3.3 KPSS ─────────────────────────────────────────────────────────
    story.append(_h2('3.3 Test KPSS (Kwiatkowski et al., 1992)'))
    story.append(_p(
        'H0 : serie stationnaire -- hypothese inverse de l\'ADF et du PP. '
        'Un rejet conjoint par ADF/PP ET une non-stationnarite selon KPSS '
        'confirment robustement la conclusion I(1).'
    ))
    _log('[PDF] Calcul KPSS log-prix...')
    try:
        kpss_lp = kpss_complet(log_prix)
        story.extend(_tab_kpss(kpss_lp, 'log(prix)'))
    except Exception as e:
        _warn(f'KPSS log-prix : {e}')
        kpss_lp = {'c': {}, 'ct': {}}

    story.append(_spacer(0.4))
    _log('[PDF] Calcul KPSS rendements...')
    try:
        kpss_rd = kpss_complet(rend)
        story.extend(_tab_kpss(kpss_rd, 'd(log-prix) -- rendements'))
    except Exception as e:
        _warn(f'KPSS rendements : {e}')
        kpss_rd = {'c': {}, 'ct': {}}

    story.append(_spacer(0.6))

    # ── 3.4 Synthese ──────────────────────────────────────────────────────
    story.append(_h2('3.4 Synthese'))
    try:
        story.extend(_tab_synthese(adf_lp, pp_lp, kpss_lp, adf_rd, pp_rd, kpss_rd))
    except Exception as e:
        _warn(f'Synthese stationnarite : {e}')

    story.append(_spacer(0.5))

    # ── 3.5 Ruptures structurelles ────────────────────────────────────────
    story.append(_h2('3.5 Ruptures structurelles'))
    story.append(_p(
        'CUSUM-OLS (Brown, Durbin & Evans 1975) et test de Chow sequentiel '
        'detectent les instabilites de la serie. '
        'Le CUSUM est standardise : rejet de la stabilite si max|S_t| > 1.358 '
        '(seuil Ploberger-Kramer 1992, 5%). '
        'Le test de Chow cherche la rupture unique maximisant la statistique F '
        'sur les points interieurs de la serie (trim = 15%).'
    ))
    try:
        from tickerlab.core.structural_breaks import detecter_ruptures
        brk = detecter_ruptures(prix, rendements, config)

        # Resume CUSUM
        cu_px = brk.get('cusum_prix', {})
        cu_rd = brk.get('cusum_rend', {})
        # Resume Chow
        ch_px = brk.get('chow_prix', {})
        ch_rd = brk.get('chow_rend', {})

        def _date_str(d):
            if d is None:
                return 'N/A'
            try:
                return str(d)[:10]
            except Exception:
                return 'N/A'

        lignes_brk = [
            ['CUSUM-OLS prix',
             _fmt(cu_px.get('stat', float('nan')), 3),
             _fmt(cu_px.get('seuil_5pct', float('nan')), 3),
             'Rejet' if cu_px.get('rejet_5pct', False) else 'OK',
             _date_str(cu_px.get('rupture'))],
            ['CUSUM-OLS rendements',
             _fmt(cu_rd.get('stat', float('nan')), 3),
             _fmt(cu_rd.get('seuil_5pct', float('nan')), 3),
             'Rejet' if cu_rd.get('rejet_5pct', False) else 'OK',
             _date_str(cu_rd.get('rupture'))],
            ['Chow seq. prix',
             _fmt(ch_px.get('f_stat', float('nan')), 3),
             'p=' + _fmt(ch_px.get('p_value', float('nan')), 4),
             str(ch_px.get('verdict', 'N/A'))[:20],
             _date_str(ch_px.get('rupture'))],
            ['Chow seq. rendements',
             _fmt(ch_rd.get('f_stat', float('nan')), 3),
             'p=' + _fmt(ch_rd.get('p_value', float('nan')), 4),
             str(ch_rd.get('verdict', 'N/A'))[:20],
             _date_str(ch_rd.get('rupture'))],
        ]
        story.extend(_tableau_eviews(
            titre='Detection de ruptures structurelles',
            colonnes=['Test', 'Stat.', 'Seuil/p', 'Verdict', 'Rupture estimee'],
            lignes=lignes_brk,
            col_widths=[LARG_U * v for v in [0.28, 0.15, 0.18, 0.20, 0.19]],
            note=(
                'CUSUM-OLS : seuil 5% = 1.358 (Ploberger-Kramer 1992). '
                'Chow : F-stat max sur points interieurs, p-value F(k, n-2k).'
            ),
        ))
    except Exception as e:
        _warn(f'Section 3.5 ruptures : {e}')
        story.append(_p(f'Ruptures structurelles indisponibles : {e}'))

    return story


# =============================================================================
# Helpers prives sections 4-6
# =============================================================================

