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
    stats_desc, adf_complet, pp_complet, kpss_complet, arch_lm,
)
from tickerlab.core.rapport._eviews import (
    bloc_eviews_estimation,
    bloc_eviews_unitroot,
    bloc_eviews_stats,
    bloc_eviews_correlogram,
    bloc_eviews_archlm,
    eviews_dist_label,
)
from tickerlab.core.rapport.sections._common import (
    _sections_log, _log, _th, _is_nan, _int_or_na, _sq,
    _encadre_pedagogique, _page_break_if, _lags_rapport,
    _max_lignes_tableau, _split_tableau, _extra_pval_bold,
)



# =============================================================================
# Annexes — Methodologie et references
# =============================================================================

def section_annexes(config: dict, trace_garch: list = None,
                    rendements=None, prix=None,
                    arima_result: dict = None, garch_final=None) -> list:
    """
    Annexes : notes methodologiques, references bibliographiques,
    et sorties format EViews (si rendements fournis).
    """
    from tickerlab.core.rapport._helpers import _pagebreak
    story = []
    story.append(_pagebreak())

    story.append(_h1('Annexes'))

    # A. Notes methodologiques
    story.append(_h2('A. Notes methodologiques'))

    notes = [
        ('<b>A.1 Tests de stationnarite.</b> '
         'ADF (Dickey-Fuller augmente, 1979) : selection des retards par BIC '
         '(Schwert 1989, lag max = floor(12*(n/100)^0.25)). '
         'PP (Phillips-Perron 1988) : correction Newey-West, bandwidth Andrews (1991). '
         'KPSS (Kwiatkowski et al. 1992) : H0 inverse -- serie stationnaire.'),

        ('<b>A.2 Selection ARIMA.</b> '
         'Grille exhaustive p in [0,p_max], q in [0,q_max], d fixe par ADF. '
         'Critere : AIC normalise = -2*logL/n + 2*k/n (format EViews). '
         'Contrainte de parcimonie optionnelle.'),

        ('<b>A.3 Grille GARCH.</b> '
         'Familles : GARCH (Bollerslev 1986), GJR-GARCH (Glosten et al. 1993), '
         'EGARCH (Nelson 1991), TGARCH, APARCH (Ding et al. 1993). '
         'Distributions : normale, t-Student, skew-t, GED. '
         'Selection hierarchique : (1) filtre significativite des parametres de variance ; '
         '(2) tri par AIC croissant ; (3) garde-fou de parcimonie Burnham-Anderson (2002) : '
         'dans la fenetre [AIC_min, AIC_min + 10], preference au modele le plus parcimonieux ; '
         '(4) test Engle-Ng (1993) sur z_t : rejet si p < 0,05 (effet de levier residuel). '
         'Fallback : meilleur AIC global si aucun candidat ne passe Engle-Ng.'),

        ('<b>A.4 Persistance GARCH.</b> '
         'GARCH : alpha+beta. '
         'GJR-GARCH : alpha+beta+0.5*gamma. '
         'EGARCH : |beta[1]| (stationnarite si < 1). '
         'APARCH/TGARCH : beta + alpha*kappa(delta,gamma), '
         'kappa = E[(|z|-gamma*z)^delta] sous N(0,1) '
         '= [(1-gamma)^delta + (1+gamma)^delta] * 2^(delta/2-1) * Gamma((delta+1)/2) / sqrt(pi) '
         '(Ding, Granger & Engle 1993, eq. 3.7). '
         'Demi-vie : log(0.5)/log(persistance) jours.'),

        ('<b>A.5 VaR et TVaR.</b> '
         'Historique : quantile empirique. Normale et Student : parametriques. '
         'Cornish-Fisher : expansion Edgeworth ordre 2 (skewness + kurtosis). '
         'GARCH dynamique : quantile de la loi conditionnelle. '
         'Monte Carlo : 50 000 simulations path-dependent. '
         'TVaR = E[r | r < VaR] (coherente, Artzner et al. 1999).'),

        ('<b>A.6 Backtesting.</b> '
         'Split chronologique : 70% train, 30% test OOS. '
         'Kupiec (1995) : LR_UC ~ chi2(1), H0 taux de violation = 1-alpha. '
         'Christoffersen (1998) : LR_IND ~ chi2(1), H0 independance des violations. '
         'Conjoint LR_CC = LR_UC + LR_IND ~ chi2(2). Verdict OK si p > 5%.'),

        ('<b>A.7 Test McNeil-Frey (2000).</b> '
         'Residus z_t = (r_t - ES_t)/sigma_t pour t : r_t < VaR_t. '
         'H0 : E[z_t] = 0 (ES correctement estime). '
         'P-value bilaterale par bootstrap (B = 1 000, seed = 42). '
         'Reference : McNeil & Frey (2000), JEF 7(3-4), eq. (10).'),
    ]
    for note in notes:
        story.append(_p(note))
        story.append(_spacer(0.15))

    # A.3bis — Procedure de selection du modele GARCH (trace tests simultanees)
    if trace_garch:
        story.append(_spacer(0.4))
        story.append(_h2('A.3bis. Procedure de selection GARCH — tests de specification'))

        # Detecter le chemin utilise (scientifique vs legacy)
        _has_lb = any(
            not (isinstance(e.get('lb_z_pval'), float) and math.isnan(e.get('lb_z_pval', float('nan'))))
            for e in trace_garch
        )
        if _has_lb:
            story.append(_p(
                'Le tableau ci-dessous retrace la procedure de selection scientifique. '
                'Pour chaque candidat valide (convergence + significativite + stationnarite), '
                'trois tests de specification sont appliques simultanement sur les residus '
                'standardises z_t : Ljung-Box(10) sur z_t (autocorrelation), '
                'Ljung-Box(10) sur z_t^2 (effet ARCH residuel) et Engle-Ng (1993) '
                '(asymetrie residuelle). Un modele passe si les trois p-valeurs depassent 5%. '
                'Parmi les modeles valides, le meilleur BIC est retenu, avec preference '
                'pour la parcimonie dans la fenetre Burnham-Anderson (delta BIC < 2).'
            ))
        else:
            story.append(_p(
                'Le tableau ci-dessous retrace la procedure de selection sequentielle (chemin '
                'de compatibilite). Le test Engle-Ng (1993) est applique sur les residus '
                'standardises z_t ; le premier candidat dont p >= 0,05 est retenu.'
            ))
        story.append(_spacer(0.2))

        from reportlab.lib import colors as _rl_colors
        from reportlab.lib.units import cm as _cm
        from reportlab.platypus import Paragraph, Table, TableStyle
        from tickerlab.core.rapport._helpers import _get_styles
        styles = _get_styles()

        def _cell(txt, bold=False):
            s = styles['TH'] if bold else styles['Note']
            return Paragraph(str(txt), s)

        def _fmt_p(val):
            if isinstance(val, float) and math.isnan(val):
                return 'N/A'
            try:
                return f"{float(val):.3f}"
            except Exception:
                return 'N/A'

        def _fmt_pers(val):
            if isinstance(val, float) and math.isnan(val):
                return 'N/A'
            try:
                return f"{float(val):.4f}"
            except Exception:
                return 'N/A'

        header = [
            _cell('Rang',       bold=True),
            _cell('Modele',     bold=True),
            _cell('Spec.',      bold=True),
            _cell('Dist.',      bold=True),
            _cell('BIC',        bold=True),
            _cell('Pers.',      bold=True),
            _cell('LB z p',     bold=True),
            _cell('LB z2 p',    bold=True),
            _cell('EngNg p',    bold=True),
            _cell('Spec OK',    bold=True),
            _cell('Fen. B&A',   bold=True),
            _cell('Compl.',     bold=True),
        ]
        rows = [header]
        for entry in trace_garch:
            spec   = f"({entry['p']},{entry['o']},{entry['q']})"
            bic_v  = entry.get('BIC', float('nan'))
            bic_s  = f"{float(bic_v):.2f}" if not (isinstance(bic_v, float) and math.isnan(bic_v)) else 'N/A'
            passe  = entry.get('tous_passent', entry.get('verdict', '') == 'OK')
            rows.append([
                _cell(str(entry['rang'])),
                _cell(entry['modele']),
                _cell(spec),
                _cell(entry['dist']),
                _cell(bic_s),
                _cell(_fmt_pers(entry.get('persistance', float('nan')))),
                _cell(_fmt_p(entry.get('lb_z_pval',     float('nan')))),
                _cell(_fmt_p(entry.get('lb_z2_pval',    float('nan')))),
                _cell(_fmt_p(entry.get('engle_ng_pval', entry.get('engle_ng_p', float('nan'))))),
                _cell('Oui' if passe else 'Non'),
                _cell('Oui' if entry.get('dans_fenetre') else 'Non'),
                _cell(str(entry.get('complexite', ''))),
            ])

        col_w = [0.6, 1.8, 1.0, 0.9, 1.8, 1.2, 1.0, 1.0, 1.0, 1.0, 1.2, 0.8]
        tbl = Table(rows, colWidths=[w * _cm for w in col_w], repeatRows=1)

        tbl_style = [
            ('BACKGROUND',  (0, 0), (-1, 0),  _rl_colors.HexColor('#2C3E50')),
            ('TEXTCOLOR',   (0, 0), (-1, 0),  _rl_colors.white),
            ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',    (0, 0), (-1, -1), 7),
            ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [_rl_colors.HexColor('#F5F5F5'), _rl_colors.white]),
            ('GRID',        (0, 0), (-1, -1), 0.4, _rl_colors.grey),
            ('TOPPADDING',  (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]
        # Surligner en vert le modele retenu (rang 1 parmi ceux qui passent)
        retenu_idx = None
        for i, entry in enumerate(trace_garch, start=1):
            passe = entry.get('tous_passent', entry.get('verdict', '') == 'OK')
            if passe and entry.get('dans_fenetre', False) and retenu_idx is None:
                retenu_idx = i
        if retenu_idx is None:
            for i, entry in enumerate(trace_garch, start=1):
                passe = entry.get('tous_passent', entry.get('verdict', '') == 'OK')
                if passe and retenu_idx is None:
                    retenu_idx = i
        if retenu_idx is not None:
            tbl_style.append(('BACKGROUND', (0, retenu_idx), (-1, retenu_idx),
                               _rl_colors.HexColor('#D5F5E3')))
            tbl_style.append(('FONTNAME', (0, retenu_idx), (-1, retenu_idx), 'Helvetica-Bold'))

        tbl.setStyle(TableStyle(tbl_style))
        story.append(tbl)
        story.append(_spacer(0.2))
        story.append(_p(
            '<i>Note : LB z p = p-valeur Ljung-Box(10) sur z_t ; LB z2 p = p-valeur '
            'Ljung-Box(10) sur z_t^2 ; EngNg p = p-valeur Engle-Ng (1993) conjoint. '
            'Spec OK = Oui si les trois p-valeurs >= 5% simultanement. '
            'Fen. B&A = modele dans la fenetre delta BIC < 2 (Burnham-Anderson 2002). '
            'Pers. = persistance de la variance conditionnelle. '
            'La ligne verte indique le modele retenu.</i>'
        ))

    story.append(_spacer(0.6))

    # B. References bibliographiques
    story.append(_h2('B. References bibliographiques'))

    refs = [
        'Andrews, D.W.K. (1991). Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation. <i>Econometrica</i>, 59(3), 817-858.',
        'Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). Coherent Measures of Risk. <i>Mathematical Finance</i>, 9(3), 203-228.',
        'Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. <i>Journal of Econometrics</i>, 31(3), 307-327.',
        'Box, G.E.P., Jenkins, G.M. (1976). <i>Time Series Analysis: Forecasting and Control</i>. Holden-Day, San Francisco.',
        'Burnham, K.P., Anderson, D.R. (2002). <i>Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach</i> (2nd ed.). Springer, New York.',
        'Christoffersen, P.F. (1998). Evaluating Interval Forecasts. <i>International Economic Review</i>, 39(4), 841-862.',
        'Dickey, D.A., Fuller, W.A. (1979). Distribution of the Estimators for Autoregressive Time Series With a Unit Root. <i>Journal of the American Statistical Association</i>, 74(366), 427-431.',
        'Ding, Z., Granger, C.W.J., Engle, R.F. (1993). A Long Memory Property of Stock Market Returns and a New Model. <i>Journal of Empirical Finance</i>, 1(1), 83-106.',
        'Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. <i>Econometrica</i>, 50(4), 987-1007.',
        'Engle, R.F., Ng, V.K. (1993). Measuring and Testing the Impact of News on Volatility. <i>Journal of Finance</i>, 48(5), 1749-1778.',
        'Glosten, L.R., Jagannathan, R., Runkle, D.E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. <i>Journal of Finance</i>, 48(5), 1779-1801.',
        'Kupiec, P.H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. <i>Journal of Derivatives</i>, 3(2), 73-84.',
        'Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., Shin, Y. (1992). Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root. <i>Journal of Econometrics</i>, 54(1-3), 159-178.',
        'McNeil, A.J., Frey, R. (2000). Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series: an Extreme Value Approach. <i>Journal of Empirical Finance</i>, 7(3-4), 271-300.',
        'Nelson, D.B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. <i>Econometrica</i>, 59(2), 347-370.',
        'Phillips, P.C.B., Perron, P. (1988). Testing for a Unit Root in Time Series Regression. <i>Biometrika</i>, 75(2), 335-346.',
        'Schwert, G.W. (1989). Tests for Unit Roots: A Monte Carlo Investigation. <i>Journal of Business & Economic Statistics</i>, 7(2), 147-159.',
    ]
    for ref in refs:
        story.append(_p(ref))
        story.append(_spacer(0.10))

    # ── Annexe C : Sorties format EViews ─────────────────────────────────────
    if rendements is not None:
        story.append(_spacer(0.6))
        story.append(_h2('C. Sorties format EViews'))
        story.append(_p(
            'Les tableaux suivants reproduisent le format de sortie du logiciel '
            'EViews pour les principales estimations et tests econometriques. '
            'Ils completent les analyses des sections precedentes.'
        ))

        ticker_raw  = config.get('data', {}).get('ticker', 'ACTIF')
        ticker_lbl  = (ticker_raw.replace('=F', '').replace('=X', '')
                       .replace('^', '').upper()[:8])
        nom_l       = f'L{ticker_lbl}'
        nom_dl      = f'DL{ticker_lbl}'
        rend        = _sq(rendements).dropna()
        corr_lags   = int(config.get('sorties_etendues', {}).get('correlogramme_lags', 36))

        # C.1 Statistiques descriptives
        story.append(_spacer(0.4))
        story.append(_h2('C.1 Statistiques descriptives'))
        try:
            sd = stats_desc(rend)
            story.extend(bloc_eviews_stats(sd, nom_dl))
        except Exception as exc:
            _warn(f'Annexe C.1 stats EViews : {exc}')

        # C.2 Correlogramme des rendements
        story.append(_spacer(0.4))
        story.extend(_page_break_if(config))
        story.append(_h2('C.2 Correlogramme des rendements'))
        try:
            story.extend(bloc_eviews_correlogram(
                rend, lags=corr_lags, titre=f'Correlogramme -- {nom_dl}'))
        except Exception as exc:
            _warn(f'Annexe C.2 correlogramme EViews : {exc}')

        # C.3 Tests de stationnarite (spec constante — ADF et PP)
        story.append(_spacer(0.4))
        story.extend(_page_break_if(config))
        story.append(_h2('C.3 Tests de stationnarite'))
        if prix is not None:
            log_prix_c3 = np.log(_sq(prix).dropna())
            try:
                adf_lp = adf_complet(log_prix_c3)
                story.extend(bloc_eviews_unitroot(
                    adf_lp.get('c', {}), serie_label=nom_l,
                    exo='Constant', test='ADF'))
            except Exception as exc:
                _warn(f'Annexe C.3 ADF log-prix : {exc}')
            try:
                adf_rd = adf_complet(rend)
                story.extend(bloc_eviews_unitroot(
                    adf_rd.get('c', {}), serie_label=nom_dl,
                    exo='Constant', test='ADF'))
            except Exception as exc:
                _warn(f'Annexe C.3 ADF rendements : {exc}')
            try:
                pp_rd = pp_complet(rend)
                story.extend(bloc_eviews_unitroot(
                    pp_rd.get('c', {}), serie_label=nom_dl,
                    exo='Constant', test='PP'))
            except Exception as exc:
                _warn(f'Annexe C.3 PP rendements : {exc}')
        else:
            story.append(_p('(prix non fourni — tests de stationnarite omis)'))

        # C.4 Test ARCH-LM sur les rendements
        story.append(_spacer(0.4))
        story.extend(_page_break_if(config))
        story.append(_h2('C.4 Test ARCH-LM (Engle 1982)'))
        try:
            archlm_res = arch_lm(rend, lags=[1, 4, 8, 12])
            story.extend(bloc_eviews_archlm(archlm_res))
        except Exception as exc:
            _warn(f'Annexe C.4 ARCH-LM rendements : {exc}')

        # C.5 Estimation ARIMA (re-estimation sur la meme serie que main.py)
        if arima_result is not None:
            story.append(_spacer(0.4))
            story.extend(_page_break_if(config))
            story.append(_h2('C.5 Estimation ARIMA'))
            try:
                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                p_opt = int(arima_result.get('p_opt', 0))
                d_opt = int(arima_result.get('d_opt', 1))
                q_opt = int(arima_result.get('q_opt', 0))
                if d_opt == 0 and rendements is not None:
                    serie_arima = rend
                elif prix is not None:
                    serie_arima = np.log(_sq(prix).dropna())
                else:
                    serie_arima = rend
                fit_arima = _ARIMA(serie_arima, order=(p_opt, d_opt, q_opt)).fit()
                spec_str  = f'ARIMA({p_opt},{d_opt},{q_opt})'
                story.extend(bloc_eviews_estimation(
                    fit_arima, dep_var=nom_dl,
                    method=f'Least Squares -- {spec_str}'))
                try:
                    archlm_arima = arch_lm(fit_arima.resid.dropna(), lags=[1, 4, 8])
                    story.extend(bloc_eviews_archlm(archlm_arima))
                except Exception as exc2:
                    _warn(f'Annexe C.5 ARCH-LM residus ARIMA : {exc2}')
            except Exception as exc:
                _warn(f'Annexe C.5 ARIMA EViews : {exc}')

        # C.6 Estimation GARCH (modele retenu)
        if garch_final is not None:
            story.append(_spacer(0.4))
            story.extend(_page_break_if(config))
            story.append(_h2('C.6 Estimation GARCH'))
            try:
                dist_lbl = 'Normal'
                try:
                    dist_cls = type(garch_final.model.distribution).__name__
                    dist_lbl = eviews_dist_label(dist_cls)
                except Exception:
                    pass
                story.extend(bloc_eviews_estimation(
                    garch_final, dep_var=nom_dl,
                    method=f'ML ARCH -- {dist_lbl} distribution',
                    nom_loi=dist_lbl))
            except Exception as exc:
                _warn(f'Annexe C.6 GARCH EViews : {exc}')

    return story


# ── Section FRTB (Tâche 4 + 3) ───────────────────────────────────────────────



# ── Section FRTB (Tâche 4 + 3) ───────────────────────────────────────────────

def section_frtb_resume(frtb_results: dict, config: dict) -> list:
    """
    Page de résumé FRTB / Basel IV (BCBS d457 janvier 2023).

    Affiche : ES 97.5%, sVaR 99%, capital IMA, feu tricolore Basel,
    et résultats des backtests FRTB-grade (DQ, Acerbi-Székely, Lopez, FZ0)
    lorsque les données sont disponibles.

    Appelée depuis le rapport si config.frtb.enabled = true et frtb_results non vide.

    Parameters
    ----------
    frtb_results : dict
        Résultats agrégés des modules frtb_metrics et backtest.run_frtb_backtests.
        Clés attendues (toutes optionnelles) :
          'es'            : dict de expected_shortfall_frtb()
          'svar'          : dict de stressed_var()
          'capital'       : dict de capital_requirement_ima()
          'backtest_frtb' : dict de run_frtb_backtests()
    config : dict
        Configuration pipeline complète.

    Returns
    -------
    list[Flowable]

    References
    ----------
    BCBS (2023). Minimum capital requirements for market risk (d457),
        §§181-189 — IMA Expected Shortfall et sVaR.
    Acerbi, C. & Székely, B. (2014). Back-testing expected shortfall.
        Risk Magazine, November 2014.
    Engle, R.F. & Manganelli, S. (2004). CAViaR. JBES 22(4), 367-381.
    Lopez, J.A. (1999). FRBNY Economic Policy Review, 5(2), 3-17.
    Fissler, T. & Ziegel, J.F. (2016). Annals of Statistics, 44(4), 1680-1707.
    """
    story = []
    story.append(_h1('FRTB / Basel IV — Métriques de risque réglementaires'))
    story.append(_p(
        'Le cadre FRTB (Fundamental Review of the Trading Book, BCBS d457 janvier 2023) '
        'remplace la VaR 99% par l\'Expected Shortfall (ES) à 97.5% comme mesure centrale '
        'du risque de marché dans l\'Internal Models Approach (IMA). '
        'Le capital réglementaire est le maximum de VaR×mult et sVaR×3.'
    ))

    # ── ES 97.5% ──────────────────────────────────────────────────────────────
    es_r = frtb_results.get('es', {})
    story.append(_h2('ES 97.5% conditionnel (FRTB IMA officiel)'))
    if es_r:
        conf_es = float(es_r.get('confidence', 0.975))
        lignes_es = [
            ['ES moyen (série complète)',  _fmt(es_r.get('es_mean',    float('nan')), 4) + ' %'],
            ['ES actuel (dernier jour)',   _fmt(es_r.get('es_current', float('nan')), 4) + ' %'],
            ['Niveau de confiance',        f'{int(conf_es*100)} %'],
            ['Méthode',                    str(es_r.get('methode', 'n/a'))],
        ]
        story.extend(_tableau_eviews(
            titre=f'Expected Shortfall à {int(conf_es*100)}% (BCBS d457 §189)',
            colonnes=['Indicateur', 'Valeur'],
            lignes=lignes_es,
            col_widths=[LARG_U * 0.60, LARG_U * 0.40],
            note=(
                'ES = E[r_t | r_t < VaR_{α,t}]. Formule analytique pour normal/Student, '
                'empirique sur z_t pour skewt/ged.'
            ),
        ))
    else:
        story.append(_p('ES non calculé (frtb.enabled: false).'))

    story.extend(_page_break_if(config))

    # ── sVaR ──────────────────────────────────────────────────────────────────
    sv_r = frtb_results.get('svar', {})
    story.append(_h2('sVaR 99% — fenêtre de stress 12 mois'))
    if sv_r:
        w_start = sv_r.get('window_start', None)
        w_end   = sv_r.get('window_end',   None)
        lignes_sv = [
            ['sVaR 99% (pire fenêtre 12 mois)', _fmt(sv_r.get('svar', float('nan')), 4) + ' %'],
            ['Début fenêtre de stress', str(w_start)[:10] if w_start else 'n/a'],
            ['Fin fenêtre de stress',   str(w_end)[:10]   if w_end   else 'n/a'],
            ['Durée fenêtre',           f'{sv_r.get("window_months", 12)} mois'],
        ]
        story.extend(_tableau_eviews(
            titre='Stressed VaR (sVaR) — BCBS d457 §§181-187',
            colonnes=['Indicateur', 'Valeur'],
            lignes=lignes_sv,
            col_widths=[LARG_U * 0.60, LARG_U * 0.40],
            note='sVaR = VaR maximale sur la fenêtre glissante de 12 mois identifiée sur la série historique.',
        ))
    else:
        story.append(_p('sVaR non calculé (frtb.enabled: false).'))

    story.append(_spacer(0.4))

    # ── Capital IMA ───────────────────────────────────────────────────────────
    cap_r = frtb_results.get('capital', {})
    story.append(_h2('Capital réglementaire IMA'))
    if cap_r:
        feu_map = {'vert': 'Vert (0-4 violations)', 'jaune': 'Jaune (5-9 violations)',
                   'rouge': 'Rouge (≥10 violations)'}
        feu_label = feu_map.get(str(cap_r.get('feu', '')).lower(), str(cap_r.get('feu', 'n/a')))
        lignes_cap = [
            ['Capital IMA = max(VaR×mult, sVaR×3)',
             _fmt(cap_r.get('capital', float('nan')), 4) + ' %'],
            ['Composante VaR×mult',
             _fmt(cap_r.get('composante_var', float('nan')), 4) + ' %'],
            ['Composante sVaR×3',
             _fmt(cap_r.get('composante_svar', float('nan')), 4) + ' %'],
            ['Multiplicateur Basel (mult)',  str(cap_r.get('multiplicateur', 3.0))],
            ['Feu tricolore',                feu_label],
        ]
        story.extend(_tableau_eviews(
            titre='Capital IMA — BCBS d457 §189',
            colonnes=['Indicateur', 'Valeur'],
            lignes=lignes_cap,
            col_widths=[LARG_U * 0.60, LARG_U * 0.40],
            note=(
                'Capital = max(VaR_{t-1}×mult_k, sVaR×3). '
                'Feu tricolore selon les violations sur 250 jours (BCBS 1996).'
            ),
        ))
    else:
        story.append(_p('Capital IMA non calculé.'))

    story.extend(_page_break_if(config))

    # ── Backtests FRTB-grade ──────────────────────────────────────────────────
    bt_r = frtb_results.get('backtest_frtb', {})
    story.append(_h2('Backtests FRTB-grade'))
    if bt_r:
        dq  = bt_r.get('dq',  {})
        as_ = bt_r.get('acerbi_szekely', {})
        lop = bt_r.get('lopez', {})
        fz  = bt_r.get('fissler_ziegel', {})
        tl  = bt_r.get('traffic_light', {})

        lignes_bt = [
            ['DQ (Engle-Manganelli 2004)', 'p-value',
             _fmt_pval(dq.get('p_value', float('nan'))),
             str(dq.get('verdict', 'n/a'))],
            ['Acerbi-Székely Z1 (2014)', 'p-value',
             _fmt_pval(as_.get('p_Z1', float('nan'))),
             str(as_.get('verdict_Z1', 'n/a'))],
            ['Acerbi-Székely Z2 (2014)', 'p-value',
             _fmt_pval(as_.get('p_Z2', float('nan'))),
             str(as_.get('verdict_Z2', 'n/a'))],
            ['Lopez loss (1999)', 'Score total',
             _fmt(lop.get('score_total', float('nan')), 4), '—'],
            ['Fissler-Ziegel FZ0 (2016)', 'Score total',
             _fmt(fz.get('score_total', float('nan')), 4), '—'],
            ['Feu tricolore Basel', 'Zone',
             str(tl.get('zone', 'n/a')).capitalize(),
             f'{tl.get("n_violations_250j", "n/a")} viol. / 250j'],
        ]
        story.extend(_tableau_eviews(
            titre='Backtests FRTB-grade (BCBS d457 2023)',
            colonnes=['Test', 'Statistique', 'Valeur', 'Verdict / Zone'],
            lignes=lignes_bt,
            col_widths=[LARG_U * v for v in [0.32, 0.18, 0.20, 0.30]],
            note=(
                'DQ : H0 VaR correctement specifiee (chi2(6)). '
                'Z1/Z2 : H0 ES correctement specifie (p-value bootstrap). '
                'FZ0 : score consistant — plus bas = meilleur. '
                'Feu tricolore : 0-4 vert, 5-9 jaune, ≥10 rouge (BCBS 1996).'
            ),
        ))
    else:
        story.append(_p(
            'Tests FRTB-grade non calculés. '
            'Activer config.backtest.tests_frtb.enabled: true pour les produire.'
        ))

    return story
