# -*- coding: utf-8 -*-
"""
Blocs PDF format EViews — Annexe B.

5 fonctions publiques, retournent list[Flowable].
Style : Courier N&B, aucun _THEME, double filet === (LINEABOVE + LINEBELOW 1.2 pt),
filet simple --- (LINEBELOW 0.5 pt), 6 decimales.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import List

import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from tickerlab.core.rapport._helpers import LARG_U, _correlogramme_eviews

_log = logging.getLogger('tickerlab.rapport._eviews')

# ── Constantes de mise en page ────────────────────────────────────────────────
_FS    = 7.5     # taille police Courier
_LEAD  = 10.5    # interligne
_THICK = 1.2     # épaisseur double filet ====
_THIN  = 0.5     # épaisseur filet simple ----


# ── Styles Paragraph Courier N&B ─────────────────────────────────────────────
def _mkst(align=TA_LEFT, bold=False, italic=False, size=None) -> ParagraphStyle:
    face = ('Courier-BoldOblique' if bold and italic
            else 'Courier-Bold'   if bold
            else 'Courier-Oblique' if italic
            else 'Courier')
    return ParagraphStyle(
        f'EV_{align}_{face}_{size or _FS}',
        fontName=face,
        fontSize=size or _FS,
        leading=_LEAD,
        textColor=colors.black,
        alignment=align,
        spaceAfter=0, spaceBefore=0,
    )

_SL  = _mkst(TA_LEFT)
_SR  = _mkst(TA_RIGHT)
_SC  = _mkst(TA_CENTER)
_SLB = _mkst(TA_LEFT,   bold=True)
_SCB = _mkst(TA_CENTER, bold=True)
_SCI = _mkst(TA_CENTER, italic=True)

def _cell(txt, style=None) -> Paragraph:
    return Paragraph(str(txt), style or _SL)

def _sp(h: float = 0.15) -> Spacer:
    from reportlab.lib.units import cm
    return Spacer(1, h * cm)


# ── Formatage numérique ───────────────────────────────────────────────────────

def _f6(v) -> str:
    try:
        f = float(v)
        return 'N/A' if (math.isnan(f) or math.isinf(f)) else f'{f:.6f}'
    except Exception:
        return 'N/A'

def _f2(v) -> str:
    try:
        f = float(v)
        return 'N/A' if (math.isnan(f) or math.isinf(f)) else f'{f:.2f}'
    except Exception:
        return 'N/A'


# ── Libellé de distribution EViews ───────────────────────────────────────────

def eviews_dist_label(dist_cls: str) -> str:
    """
    Retourne le libellé EViews à partir du nom de classe de distribution arch.

    Ordre critique : 'Skew' testé avant 'Student' pour éviter que
    'SkewStudent' soit classé 'Student'.
    """
    if 'Skew' in dist_cls or 'skew' in dist_cls:
        return 'Skew-Student'
    if 'Student' in dist_cls or 'student' in dist_cls:
        return 'Student'
    if 'GED' in dist_cls or 'Generalized' in dist_cls or 'generalized' in dist_cls:
        return 'GED'
    return 'Normal'


# ── Mapping paramètres arch/statsmodels → libellés EViews ────────────────────

def _param_section_label(name: str, model_upper: str,
                          nom_loi: str = 'Normal') -> tuple[str, str]:
    """Retourne (section, eviews_label) pour un paramètre arch.

    Gère les ordres > 1 : alpha[k]/gamma[k]/beta[k] → RESID(-k)^2 / asym / GARCH(-k).
    `delta` (puissance APARCH/TGARCH) est un terme de l'équation de variance.
    """
    if name == 'mu':
        return 'mean', 'C'
    if name == 'omega':
        return 'variance', 'C'
    # alpha[k] / gamma[k] / beta[k] — tout ordre k, pas seulement 1
    m = re.match(r'^(alpha|gamma|beta)\[(\d+)\]$', name)
    if m:
        fam, k = m.group(1), m.group(2)
        is_egarch = 'EGARCH' in model_upper
        if fam == 'alpha':
            if is_egarch:
                return 'variance', f'|RESID(-{k})/SQRT(GARCH(-{k}))|'
            return 'variance', f'RESID(-{k})^2'
        if fam == 'gamma':
            if is_egarch:
                return 'variance', f'RESID(-{k})/SQRT(GARCH(-{k}))'
            return 'variance', f'RESID(-{k})^2*(RESID(-{k})<0)'
        # beta[k] : EViews nomme le terme de variance retardé EGARCH(-k) en
        # spécification log-variance, GARCH(-k) sinon.
        return 'variance', (f'EGARCH(-{k})' if is_egarch else f'GARCH(-{k})')
    # Puissance APARCH/TGARCH (param 'delta' ou 'delta[1]') → variance, PAS mean
    if name == 'delta' or name.startswith('delta'):
        return 'variance', 'POWER (delta)'
    if name == 'nu':
        if 'GED' in nom_loi:
            return 'dist', 'GED PARAMETER'
        return 'dist', 'T-DIST. DOF'
    if name == 'eta':
        return 'dist', 'T-DIST. DOF'
    if name == 'lambda':
        return 'dist', 'SKEWNESS PARAMETER'
    if any(name.startswith(p) for p in ('alpha', 'gamma', 'beta', 'omega')):
        return 'variance', name.upper()
    return 'mean', name.upper()


# ── Builder Table EViews ──────────────────────────────────────────────────────

_BASE_CMDS = [
    ('FONTNAME',      (0, 0), (-1, -1), 'Courier'),
    ('FONTSIZE',      (0, 0), (-1, -1), _FS),
    ('LEADING',       (0, 0), (-1, -1), _LEAD),
    ('TEXTCOLOR',     (0, 0), (-1, -1), colors.black),
    ('BACKGROUND',    (0, 0), (-1, -1), colors.white),
    ('LEFTPADDING',   (0, 0), (-1, -1), 3),
    ('RIGHTPADDING',  (0, 0), (-1, -1), 3),
    ('TOPPADDING',    (0, 0), (-1, -1), 1),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
]


def _build_table(rows_with_type: list, col_widths: list,
                 extra_cmds: list = None) -> Table:
    """
    Construit une Table EViews à partir de (row_data, row_type).

    row_type :
        'header'  → col labels, === above + below
        'section' → section label spanning, --- before
        'data'    → données, label gauche, chiffres droite
        'last'    → comme 'data' + === below
        'sep'     → ligne vide, --- above
    """
    cmds = list(_BASE_CMDS)
    data = []

    for i, (row, rtype) in enumerate(rows_with_type):
        data.append(row)
        if rtype == 'header':
            cmds += [
                ('ALIGN',     (0, i), (-1, i), 'CENTER'),
                ('FONTNAME',  (0, i), (-1, i), 'Courier-Bold'),
                ('LINEABOVE', (0, i), (-1, i), _THICK, colors.black),
                ('LINEBELOW', (0, i), (-1, i), _THICK, colors.black),
            ]
        elif rtype == 'section':
            n = len(col_widths)
            cmds += [
                ('SPAN',      (0, i), (n - 1, i)),
                ('ALIGN',     (0, i), (-1, i), 'CENTER'),
                ('FONTNAME',  (0, i), (-1, i), 'Courier-Oblique'),
                ('LINEABOVE', (0, i), (-1, i), _THIN, colors.black),
            ]
        elif rtype == 'sep':
            n = len(col_widths)
            cmds += [
                ('SPAN',      (0, i), (n - 1, i)),
                ('LINEABOVE', (0, i), (-1, i), _THIN, colors.black),
            ]
        elif rtype in ('data', 'last'):
            cmds += [
                ('ALIGN', (0, i), (0, i),  'LEFT'),
                ('ALIGN', (1, i), (-1, i), 'RIGHT'),
            ]
            if rtype == 'last':
                cmds += [('LINEBELOW', (0, i), (-1, i), _THICK, colors.black)]

    if extra_cmds:
        cmds += extra_cmds

    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(cmds))
    return t


# =============================================================================
# Rendu 1 — Estimation (GARCH arch ou ARIMA statsmodels)
# =============================================================================

def bloc_eviews_estimation(fit, dep_var: str, method: str,
                            nom_loi: str = 'Normal') -> List:
    """
    Bloc EViews d'estimation GARCH (arch) ou ARIMA (statsmodels).

    Parameters
    ----------
    fit :
        ARCHModelResult ou SARIMAXResults.
    dep_var : str
        Variable dépendante (ex. 'DLBRENT').
    method : str
        Ex. 'ML ARCH - Normal distribution' ou 'Least Squares'.
    nom_loi : str
        Nom lisible de la distribution, pour mapping params.
    """
    try:
        is_arch = hasattr(fit, 'loglikelihood')
        params  = fit.params
        stderr  = fit.std_err if is_arch else fit.bse
        tvals   = fit.tvalues
        pvals   = fit.pvalues
        nobs    = int(getattr(fit, 'nobs', len(params)))

        loglik  = float(fit.loglikelihood if is_arch else fit.llf)
        aic_raw = float(fit.aic)
        bic_raw = float(fit.bic)
        aic_n   = aic_raw / nobs if nobs > 0 else float('nan')
        bic_n   = bic_raw / nobs if nobs > 0 else float('nan')
        avg_ll  = loglik  / nobs if nobs > 0 else float('nan')

        stat_lbl = 'z-Statistic' if is_arch else 't-Statistic'
        # Détecte la famille de volatilité (EGARCH/GARCH/APARCH) depuis l'objet
        # arch — le libellé `method` affiché ne porte pas le nom du modèle, donc
        # on ne peut pas s'y fier pour router les labels d'équation de variance.
        model_up = method.upper()
        if is_arch:
            try:
                model_up += ' ' + type(fit.model.volatility).__name__.upper()
            except Exception:
                pass

        def _pval4(nm):
            try:
                return _f6(pvals[nm])
            except Exception:
                return 'N/A'

        def _se(nm):
            try:
                return _f6(stderr[nm])
            except Exception:
                return 'N/A'

        def _tv(nm):
            try:
                return _f6(tvals[nm])
            except Exception:
                return 'N/A'

        # ── Métadonnées ────────────────────────────────────────────────────────
        now = datetime.now()
        story: List = [
            _cell(f'Dependent Variable: {dep_var}'),
            _cell(f'Method: {method}'),
            _cell(f'Date: {now.strftime("%m/%d/%y")}   Time: {now.strftime("%H:%M")}'),
            _cell(f'Included observations: {nobs} after adjustments'),
            _sp(0.1),
        ]

        # ── Largeurs colonnes ──────────────────────────────────────────────────
        cw = [LARG_U * 0.34, LARG_U * 0.165,
              LARG_U * 0.165, LARG_U * 0.165, LARG_U * 0.165]

        hdr_row = (
            [_cell(c, _SCB) for c in
             ['Variable', 'Coefficient', 'Std. Error', stat_lbl, 'Prob.']],
            'header',
        )

        rows = [hdr_row]

        if is_arch:
            # Sections Mean / Variance / Distribution
            for section_key, label in (
                ('mean',     'Mean Equation'),
                ('variance', 'Variance Equation'),
                ('dist',     'Distribution Parameters'),
            ):
                section_params = [
                    (nm, _param_section_label(nm, model_up, nom_loi))
                    for nm in params.index
                    if _param_section_label(nm, model_up, nom_loi)[0] == section_key
                ]
                if not section_params:
                    continue
                rows.append(([label, '', '', '', ''], 'section'))
                for nm, (_, ev_lbl) in section_params:
                    rows.append((
                        [_cell(ev_lbl), _cell(_f6(params[nm]), _SR),
                         _cell(_se(nm),   _SR), _cell(_tv(nm), _SR),
                         _cell(_pval4(nm), _SR)],
                        'data',
                    ))
        else:
            # ARIMA : section unique, mapping statsmodels → EViews
            for nm in params.index:
                if nm == 'const':
                    ev_lbl = 'C'
                elif nm == 'sigma2':
                    continue
                elif nm.startswith('ar.L'):
                    ev_lbl = f'AR({nm.split("L")[-1]})'
                elif nm.startswith('ma.L'):
                    ev_lbl = f'MA({nm.split("L")[-1]})'
                else:
                    ev_lbl = nm.upper()
                rows.append((
                    [_cell(ev_lbl), _cell(_f6(params[nm]), _SR),
                     _cell(_se(nm), _SR), _cell(_tv(nm), _SR),
                     _cell(_pval4(nm), _SR)],
                    'data',
                ))

        # Dernière ligne de données → === below
        if len(rows) > 1:
            last_d, _ = rows[-1]
            rows[-1] = (last_d, 'last')

        story.append(_build_table(rows, cw))
        story.append(_sp(0.05))

        # ── Pied de stats façon EViews (6 lignes, 2 paires) ────────────────────
        k_params = (int(getattr(fit, 'num_params', len(params)))
                    if is_arch else len(params))
        y_dep = (getattr(getattr(fit, 'model', None), '_y', None) if is_arch
                 else getattr(getattr(fit, 'model', None), 'endog', None))
        resid_v = getattr(fit, 'resid', None)

        def _clean(x):
            try:
                a = np.asarray(x, dtype=float).ravel()
                return a[~np.isnan(a)]
            except Exception:
                return None

        yv, ev = _clean(y_dep), _clean(resid_v)
        nan = float('nan')
        mean_dep = sd_dep = se_reg = ssr = r2 = adj_r2 = dw = hq_n = nan
        try:
            if yv is not None and yv.size:
                mean_dep = float(np.mean(yv))
                sd_dep = float(np.std(yv, ddof=1))
            if ev is not None and ev.size:
                ssr = float(np.sum(ev ** 2))
                se_reg = math.sqrt(ssr / max(nobs - k_params, 1)) if ssr >= 0 else nan
                dw = float(np.sum(np.diff(ev) ** 2) / ssr) if ssr > 0 else nan
            if yv is not None and yv.size and not math.isnan(ssr):
                tss = float(np.sum((yv - np.mean(yv)) ** 2))
                if tss > 0:
                    r2 = 1.0 - ssr / tss
                    adj_r2 = 1.0 - (1.0 - r2) * (nobs - 1) / max(nobs - k_params, 1)
            if nobs > 2:
                hq_n = (-2.0 * loglik + 2.0 * k_params * math.log(math.log(nobs))) / nobs
        except Exception:
            pass

        cw4 = [LARG_U * 0.27, LARG_U * 0.23, LARG_U * 0.27, LARG_U * 0.23]
        foot_rows = [
            [_cell('R-squared'),          _cell(_f6(r2),       _SR),
             _cell('Mean dependent var'), _cell(_f6(mean_dep), _SR)],
            [_cell('Adjusted R-squared'), _cell(_f6(adj_r2),   _SR),
             _cell('S.D. dependent var'), _cell(_f6(sd_dep),   _SR)],
            [_cell('S.E. of regression'), _cell(_f6(se_reg),   _SR),
             _cell('Akaike info criterion'), _cell(_f6(aic_n), _SR)],
            [_cell('Sum squared resid'),  _cell(_f2(ssr),      _SR),
             _cell('Schwarz criterion'),  _cell(_f6(bic_n),    _SR)],
            [_cell('Log likelihood'),     _cell(_f2(loglik),   _SR),
             _cell('Hannan-Quinn criter.'), _cell(_f6(hq_n),   _SR)],
            [_cell('Durbin-Watson stat'), _cell(_f6(dw),       _SR),
             _cell('Avg. log likelihood'), _cell(_f6(avg_ll),  _SR)],
        ]
        foot_cmds = list(_BASE_CMDS) + [
            ('ALIGN',     (0, 0), (0, -1), 'LEFT'),
            ('ALIGN',     (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN',     (2, 0), (2, -1), 'LEFT'),
            ('ALIGN',     (3, 0), (3, -1), 'RIGHT'),
            ('LINEABOVE', (0, 0), (-1, 0),  _THIN, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), _THICK, colors.black),
        ]
        t_foot = Table(foot_rows, colWidths=cw4)
        t_foot.setStyle(TableStyle(foot_cmds))
        story.append(t_foot)
        story.append(_sp(0.35))

        return story

    except Exception as exc:
        _log.warning('bloc_eviews_estimation(%s) : %s', dep_var, exc)
        return []


# =============================================================================
# Rendu 2 — Test de racine unitaire (ADF / PP / KPSS)
# =============================================================================

def bloc_eviews_unitroot(res: dict, serie_label: str,
                          exo: str = 'Constant',
                          test: str = 'ADF') -> List:
    """
    Bloc EViews test de racine unitaire.

    Parameters
    ----------
    res : dict
        Clés : stat, pval, lags, crit_1, crit_5, crit_10.
    serie_label : str
        Nom de la série (ex. 'LBRENT').
    exo : str
        Exogène ('Constant', 'Constant, Linear Trend', 'None').
    test : str
        'ADF', 'PP' ou 'KPSS'.
    """
    try:
        stat    = res.get('stat',   float('nan'))
        pval    = res.get('pval',   float('nan'))
        lags    = res.get('lags',   '?')
        crit_1  = res.get('crit_1', float('nan'))
        crit_5  = res.get('crit_5', float('nan'))
        crit_10 = res.get('crit_10', float('nan'))
        t_up    = test.upper()
        is_kpss = t_up == 'KPSS'

        if is_kpss:
            h0        = f'{serie_label} is stationary'
            stat_hdr  = 'LM-Stat.'
            stat_name = 'KPSS test statistic'
            note      = '*Kwiatkowski-Phillips-Schmidt-Shin (1992, Table 1).'
        elif t_up == 'PP':
            h0        = f'{serie_label} has a unit root'
            stat_hdr  = 't-Statistic'
            stat_name = 'Phillips-Perron test statistic'
            note      = '*MacKinnon (1996) one-sided p-values.'
        else:
            h0        = f'{serie_label} has a unit root'
            stat_hdr  = 't-Statistic'
            stat_name = 'Augmented Dickey-Fuller test statistic'
            note      = '*MacKinnon (1996) one-sided p-values.'

        story: List = [
            _cell(f'Null Hypothesis: {h0}'),
            _cell(f'Exogenous: {exo}'),
            _cell(f'Lag Length: {lags} (Automatic - based on SIC)'),
            _sp(0.05),
        ]

        cw3 = [LARG_U * 0.62, LARG_U * 0.19, LARG_U * 0.19]

        rows = [
            ([_cell(''), _cell(stat_hdr, _SCB), _cell('Prob.*', _SCB)], 'header'),
            ([_cell(stat_name), _cell(_f6(stat), _SR), _cell(_f6(pval), _SR)], 'data'),
            ([_cell(f'Test critical values:   1% level'),
              _cell(_f6(crit_1),  _SR), _cell('')], 'data'),
            ([_cell(f'                        5% level'),
              _cell(_f6(crit_5),  _SR), _cell('')], 'data'),
            ([_cell(f'                       10% level'),
              _cell(_f6(crit_10), _SR), _cell('')], 'last'),
        ]

        story.append(_build_table(rows, cw3))
        story.append(_cell(note))
        story.append(_sp(0.25))
        return story

    except Exception as exc:
        _log.warning('bloc_eviews_unitroot(%s, %s) : %s', serie_label, test, exc)
        return []


# =============================================================================
# Rendu 3 — Statistiques descriptives
# =============================================================================

def bloc_eviews_stats(stats: dict, serie_label: str) -> List:
    """
    Bloc EViews de statistiques descriptives.

    Parameters
    ----------
    stats : dict
        Sortie de stats_desc() : n, mean, median, max, min, std, skew,
        kurt_exc, jb_stat, jb_pval.
    serie_label : str
        Nom de la série (ex. 'DLBRENT').
    """
    try:
        kurt_ev = stats.get('kurt_exc', float('nan'))
        try:
            kurt_ev = float(kurt_ev) + 3.0   # EViews = kurtosis non-excédentaire
        except Exception:
            kurt_ev = float('nan')

        story: List = [
            _cell(f'Series: {serie_label}'),
            _sp(0.05),
        ]

        cw2 = [LARG_U * 0.55, LARG_U * 0.45]

        rows = [
            ([_cell('Mean'),       _cell(_f6(stats.get('mean',   float('nan'))), _SR)], 'data'),
            ([_cell('Median'),     _cell(_f6(stats.get('median', float('nan'))), _SR)], 'data'),
            ([_cell('Maximum'),    _cell(_f6(stats.get('max',    float('nan'))), _SR)], 'data'),
            ([_cell('Minimum'),    _cell(_f6(stats.get('min',    float('nan'))), _SR)], 'data'),
            ([_cell('Std. Dev.'),  _cell(_f6(stats.get('std',    float('nan'))), _SR)], 'data'),
            ([_cell('Skewness'),   _cell(_f6(stats.get('skew',   float('nan'))), _SR)], 'data'),
            ([_cell('Kurtosis'),   _cell(_f6(kurt_ev),                           _SR)], 'data'),
            ([_cell(''), _cell('')], 'sep'),
            ([_cell('Jarque-Bera'), _cell(_f6(stats.get('jb_stat', float('nan'))), _SR)], 'data'),
            ([_cell('Probability'), _cell(_f6(stats.get('jb_pval', float('nan'))), _SR)], 'data'),
            ([_cell(''), _cell('')], 'sep'),
            ([_cell('Observations'), _cell(str(int(stats.get('n', 0))), _SR)], 'last'),
        ]

        # Ajouter LINEABOVE sur la première ligne
        extra = [('LINEABOVE', (0, 0), (-1, 0), _THICK, colors.black)]

        story.append(_build_table(rows, cw2, extra_cmds=extra))
        story.append(_sp(0.25))
        return story

    except Exception as exc:
        _log.warning('bloc_eviews_stats(%s) : %s', serie_label, exc)
        return []


# =============================================================================
# Rendu 4 — Corrélogramme (wrapper sur _correlogramme_eviews existant)
# =============================================================================

def bloc_eviews_correlogram(serie, lags: int = 36, titre: str = '') -> List:
    """
    Corrélogramme format EViews (AC / PAC / Q-Stat / Prob + barres).
    Réutilise _correlogramme_eviews de _helpers.py.
    """
    try:
        return _correlogramme_eviews(serie, lags=lags, titre=titre)
    except Exception as exc:
        _log.warning('bloc_eviews_correlogram(%s) : %s', titre, exc)
        return []


# =============================================================================
# Rendu 5 — Test ARCH-LM (Heteroskedasticity Test: ARCH)
# =============================================================================

def bloc_eviews_archlm(archlm_res) -> List:
    """
    Bloc EViews « Heteroskedasticity Test: ARCH ».

    Parameters
    ----------
    archlm_res : list[dict] | dict
        Sortie de core.rapport._stats.arch_lm().
        Clés : lag, f_stat, f_pval, lm_stat, lm_pval.
    """
    try:
        if isinstance(archlm_res, dict):
            archlm_res = [archlm_res]
        if not archlm_res:
            return []

        story: List = [_cell('Heteroskedasticity Test: ARCH'), _sp(0.05)]

        cw4 = [LARG_U * 0.30, LARG_U * 0.20, LARG_U * 0.30, LARG_U * 0.20]

        for item in archlm_res:
            q    = int(item.get('lag',    0))
            fst  = item.get('f_stat',  float('nan'))
            fpv  = item.get('f_pval',  float('nan'))
            obs_r2 = item.get('lm_stat', float('nan'))   # n·R² = LM stat
            chi_pv = item.get('lm_pval', float('nan'))

            story.append(_cell(f'ARCH — lag q = {q}'))

            tbl_rows = [
                [_cell('F-statistic'),    _cell(_f6(fst),    _SR),
                 _cell(f'Prob. F({q},...)'),   _cell(_f6(fpv),  _SR)],
                [_cell('Obs*R-squared'),  _cell(_f6(obs_r2), _SR),
                 _cell(f'Prob. Chi-Square({q})'), _cell(_f6(chi_pv), _SR)],
            ]
            cmds = list(_BASE_CMDS) + [
                ('ALIGN',     (0, 0), (0, -1), 'LEFT'),
                ('ALIGN',     (1, 0), (1, -1), 'RIGHT'),
                ('ALIGN',     (2, 0), (2, -1), 'LEFT'),
                ('ALIGN',     (3, 0), (3, -1), 'RIGHT'),
                ('LINEABOVE', (0, 0), (-1, 0),  _THICK, colors.black),
                ('LINEBELOW', (0, -1), (-1, -1), _THICK, colors.black),
            ]
            t = Table(tbl_rows, colWidths=cw4)
            t.setStyle(TableStyle(cmds))
            story.append(t)
            story.append(_sp(0.2))

        return story

    except Exception as exc:
        _log.warning('bloc_eviews_archlm : %s', exc)
        return []
