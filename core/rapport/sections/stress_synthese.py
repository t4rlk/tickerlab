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
from tickerlab.core.rapport.sections.var_backtest import _verdict_style



# =============================================================================
# Section 10 — Synthese et conclusions
# =============================================================================

def section_10(rendements: pd.Series, garch_final, modele_nom: str,
               df_var_tvar: pd.DataFrame, df_bt: pd.DataFrame,
               config: dict, ticker: str = '') -> list:
    """
    Section 10 : Synthese et conclusions.
    Caracteristiques du modele, VaR/TVaR 99%, verdict backtest global.
    """
    from tickerlab.core.rapport._stats import persistance_garch
    story = []
    nom = nom_actif(ticker) if ticker else 'l\'actif'

    story.append(_h1('10. Synthese et conclusions'))
    story.append(_p(
        'Cette section recapitule les resultats cles de la modelisation '
        f'ARIMA-GARCH appliquee aux rendements de {nom}, et les implications '
        'en termes de quantification du risque de marche (VaR/TVaR).'
    ))

    # 10.1 Modele retenu
    story.append(_h2('10.1 Modele GARCH retenu -- caracteristiques'))
    try:
        nobs = int(garch_final.nobs)
        pers = persistance_garch(modele_nom, garch_final.params)
        if not _is_nan(pers) and 0 < pers < 1:
            hl_str = f'{round(math.log(0.5)/math.log(pers), 1)} jours'
        else:
            hl_str = 'N/A'

        try:
            var99  = float(df_var_tvar.loc['99%', 'VaR GARCH'])
            tvar99 = float(df_var_tvar.loc['99%', 'TVaR GARCH'])
        except Exception:
            var99, tvar99 = float('nan'), float('nan')

        lignes = [
            ['Modele GARCH retenu',  modele_nom],
            ['Ticker',               ticker],
            ['Observations',         str(nobs)],
            ['Persistance',          _fmt(pers, 4)],
            ['Demi-vie',             hl_str],
            ['VaR 99% GARCH',        _fmt(var99,  4) + ' %'],
            ['TVaR 99% GARCH',       _fmt(tvar99, 4) + ' %'],
        ]
        cw = [LARG_U * 0.55, LARG_U * 0.45]
        story.extend(_tableau_eviews(
            titre='Caracteristiques du modele GARCH retenu',
            colonnes=['Indicateur', 'Valeur'],
            lignes=lignes, col_widths=cw,
        ))
    except Exception as e:
        _warn(f'Section 10 modele : {e}')

    story.append(_spacer(0.5))

    # 10.2 Comparaison VaR/TVaR 99%
    story.append(_h2('10.2 VaR et TVaR 99% -- synthese toutes methodes'))
    try:
        if '99%' in df_var_tvar.index:
            row99 = df_var_tvar.loc['99%']
            methodes_99 = [
                ('Historique',    'VaR Historique',       'TVaR Historique'),
                ('Normale',       'VaR Normale',           'TVaR Normale'),
                ('Student',       'VaR Student',           'TVaR Student'),
                ('Cornish-Fisher','VaR Cornish-Fisher',    'TVaR CF semi-empirique'),
                ('GARCH',         'VaR GARCH',             'TVaR GARCH'),
                ('Monte Carlo',   'VaR Monte Carlo (1j)',  'TVaR Monte Carlo (1j)'),
            ]
            lignes = []
            _cf_nm = {'Cornish-Fisher'}
            for nm, vc, tc in methodes_99:
                try:
                    vf = float(row99[vc] if vc in row99.index else float('nan'))
                    tf = float(row99[tc] if tc in row99.index else float('nan'))
                    na_cf = nm in _cf_nm
                    v = ('N/A (CF non monotone)' if na_cf and math.isnan(vf)
                         else _fmt(vf, 4) + ' %')
                    t = ('N/A (CF non monotone)' if na_cf and math.isnan(tf)
                         else _fmt(tf, 4) + ' %')
                except Exception:
                    v, t = 'N/A', 'N/A'
                lignes.append([nm, v, t])
            cw = [LARG_U * 0.34, LARG_U * 0.33, LARG_U * 0.33]
            story.extend(_tableau_eviews(
                titre='VaR et TVaR 99% -- synthese toutes methodes',
                colonnes=['Methode', 'VaR 99%', 'TVaR 99%'],
                lignes=lignes, col_widths=cw,
                note='En %. Signe negatif = perte. C-F = Cornish-Fisher.',
            ))
    except Exception as e:
        _warn(f'Section 10 tableau VaR/TVaR : {e}')

    story.append(_spacer(0.5))

    # 10.3 Synthese backtest
    story.append(_h2('10.3 Synthese du backtesting'))
    try:
        th       = _th()
        lignes   = []
        extra    = []
        row_base = 2

        for i, niv_pct in enumerate(['95%', '99%']):
            sub    = df_bt[df_bt['Niveau'] == niv_pct]
            total  = len(sub)
            ok_uc  = int((sub['Verdict UC'] == 'OK').sum())
            non_uc = total - ok_uc
            ok_cc  = int((sub['Verdict CC'] == 'OK').sum())
            non_cc = total - ok_cc
            vg     = 'OK' if ok_cc >= total * 0.7 else \
                     'Partiel' if ok_cc >= total * 0.4 else 'NON'
            lignes.append([niv_pct,
                           f'{ok_uc}/{total}', f'{non_uc}/{total}',
                           f'{ok_cc}/{total}', f'{non_cc}/{total}', vg])
            row_i = row_base + i
            extra += _verdict_style(vg, row_i, 5, th)

        cw = [LARG_U * v for v in [0.12, 0.18, 0.18, 0.18, 0.18, 0.16]]
        story.extend(_tableau_eviews(
            titre='Synthese backtesting -- verdicts UC et CC',
            colonnes=['Niveau', 'UC OK', 'UC NON', 'CC OK', 'CC NON', 'Global'],
            lignes=lignes, col_widths=cw, extra_styles=extra,
            note='UC = couverture inconditionnelle (Kupiec). CC = conjoint. Seuil : p > 5%.',
        ))
    except Exception as e:
        _warn(f'Section 10 synthese backtest : {e}')

    story.append(_spacer(0.4))

    # 10.4 Tests de robustesse avances
    story.append(_h2('10.4 Tests de robustesse avances'))
    story.append(_p(
        'Quatre tests econometriques complementaires evaluent la qualite '
        'predictive du modele GARCH retenu au-dela du backtesting classique.'
    ))
    try:
        from tickerlab.core.tests_robustesse import tester_robustesse
        rob = tester_robustesse(rendements, garch_final, config)

        # — Berkowitz (2001) ——————————————————————————————————————————————
        story.append(_p(
            '<b>Test de Berkowitz (2001)</b> — densite predictive complète. '
            'Transforme les rendements OOS en PIT u_t = F(y_t | Omega_{t-1}), '
            'puis xi_t = Phi^{-1}(u_t). H0 : xi_t ~ iid N(0,1). LR ~ chi2(3).'
        ))
        try:
            bk = rob.get('berkowitz', {})
            bk_lr   = bk.get('LR', float('nan'))
            bk_pv   = bk.get('p_value', float('nan'))
            bk_verd = bk.get('verdict', 'N/A')
            bk_n    = bk.get('T_oos', 0)
            story.extend(_tableau_eviews(
                titre='Berkowitz (2001) — LR test densite predictive',
                colonnes=['Statistique', 'Valeur'],
                lignes=[
                    ['LR (chi2(3))', _fmt(bk_lr, 3)],
                    ['p-value',      _fmt(bk_pv, 4)],
                    ['Observations OOS', str(bk_n)],
                    ['Verdict (seuil 5%)', bk_verd],
                ],
                col_widths=[LARG_U * 0.55, LARG_U * 0.45],
                note='H0 : densite predictive correctement specifiee. OK = non-rejet.',
            ))
        except Exception as e:
            _warn(f'S10 Berkowitz table : {e}')
            story.append(_p(f'Berkowitz : calcul indisponible ({e}).'))

        # — DQ test (Engle-Manganelli 2004) ——————————————————————————————
        story.append(_spacer(0.3))
        story.append(_p(
            '<b>Test DQ d\'Engle & Manganelli (2004)</b> — quantile dynamique. '
            'Teste si les violations de la VaR sont predictibles via leurs lags '
            'et la VaR elle-meme. H0 : Hit_t non autocorrele. Wald ~ chi2(K+2).'
        ))
        try:
            dq = rob.get('dq', {})
            lignes_dq = []
            for niv, res in dq.items():
                dq_s = res.get('DQ', float('nan'))
                dq_p = res.get('p_value', float('nan'))
                dq_v = res.get('verdict', 'N/A')
                lignes_dq.append([niv, _fmt(dq_s, 3), _fmt(dq_p, 4), dq_v])
            if lignes_dq:
                story.extend(_tableau_eviews(
                    titre='DQ test — Engle & Manganelli (2004)',
                    colonnes=['Niveau', 'DQ (Wald)', 'p-value', 'Verdict'],
                    lignes=lignes_dq,
                    col_widths=[LARG_U * 0.18, LARG_U * 0.28, LARG_U * 0.28, LARG_U * 0.26],
                    note='H0 : violations non autocorrelees ni predictibles. OK = non-rejet.',
                ))
        except Exception as e:
            _warn(f'S10 DQ table : {e}')
            story.append(_p(f'DQ test : calcul indisponible ({e}).'))

        # — Diebold-Mariano tick loss (Giacomini-Komunjer 2005) ———————————
        story.append(_spacer(0.3))
        story.append(_p(
            '<b>Test Diebold-Mariano (1995) avec tick loss</b> — comparaison de '
            'methodes (Giacomini & Komunjer 2005). DM ~ N(0,1) sous H0 de '
            'precision egale. Un DM positif favorise M1. *** p<1%, ** p<5%, * p<10%.'
        ))
        try:
            dm = rob.get('diebold_mariano', pd.DataFrame())
            if isinstance(dm, pd.DataFrame) and not dm.empty:
                for niv_grp, sub in dm.groupby('Niveau'):
                    lignes_dm = []
                    for _, row in sub.iterrows():
                        lignes_dm.append([
                            row['M1'], row['M2'],
                            _fmt(row['DM'], 3), _fmt(row['p_value'], 4),
                            str(row.get('Favori', '')), str(row.get('Sig.', '')),
                        ])
                    story.extend(_tableau_eviews(
                        titre=f'Diebold-Mariano tick loss — VaR {niv_grp}',
                        colonnes=['M1', 'M2', 'DM', 'p-value', 'Favori', 'Sig.'],
                        lignes=lignes_dm,
                        col_widths=[LARG_U * v for v in [0.22, 0.22, 0.14, 0.14, 0.18, 0.10]],
                        note='rho_alpha(u) = u * (alpha - I(u<0)). DM > 0 : M1 surperforme M2.',
                    ))
        except Exception as e:
            _warn(f'S10 DM table : {e}')
            story.append(_p(f'Diebold-Mariano : calcul indisponible ({e}).'))

        # — Sign Bias (Engle-Ng 1993) ————————————————————————————————————
        story.append(_spacer(0.3))
        story.append(_p(
            '<b>Test de Sign Bias d\'Engle & Ng (1993)</b> — effet asymetrique '
            'des chocs sur la variance conditionnelle. '
            'Un rejet du test joint signale une asymetrie non capturee par le modele.'
        ))
        try:
            sbt = rob.get('sign_bias', {})
            if sbt:
                lignes_sb = [
                    ['Sign Bias (t-stat)',     _fmt(sbt.get('sign_stat',   float('nan')), 3)],
                    ['Sign Bias (p-value)',     _fmt(sbt.get('sign_pval',   float('nan')), 4)],
                    ['Neg. Sign Bias (t-stat)', _fmt(sbt.get('neg_stat',    float('nan')), 3)],
                    ['Neg. Sign Bias (p)',       _fmt(sbt.get('neg_pval',    float('nan')), 4)],
                    ['Pos. Sign Bias (t-stat)', _fmt(sbt.get('pos_stat',    float('nan')), 3)],
                    ['Pos. Sign Bias (p)',       _fmt(sbt.get('pos_pval',    float('nan')), 4)],
                    ['Test joint F (p-value)',  _fmt(sbt.get('joint_pval',  float('nan')), 4)],
                    ['Verdict (seuil 5%)',      str(sbt.get('verdict', 'N/A'))],
                ]
                story.extend(_tableau_eviews(
                    titre='Sign Bias — Engle & Ng (1993)',
                    colonnes=['Statistique', 'Valeur'],
                    lignes=lignes_sb,
                    col_widths=[LARG_U * 0.65, LARG_U * 0.35],
                    note='H0 : pas d\'effet de signe sur la variance conditionnelle.',
                ))
            else:
                story.append(_p('Sign Bias : test non disponible.'))
        except Exception as e:
            _warn(f'S10 Sign Bias table : {e}')
            story.append(_p(f'Sign Bias : calcul indisponible ({e}).'))

    except Exception as e:
        _warn(f'Section 10.4 tests robustesse : {e}')
        story.append(_p(f'Tests de robustesse indisponibles : {e}'))

    story.append(_spacer(0.4))

    # 10.5 Monitoring dynamique
    story.append(_h2('10.5 Monitoring dynamique de la VaR'))
    story.append(_p(
        'Deux indicateurs de surveillance continue permettent de detecter '
        'precocement une derive du modele : le CUSUM des violations (Jorion 2007) '
        'et le ratio glissant d\'exceptions sur 250 jours ouvrables.'
    ))
    try:
        from tickerlab.core.monitoring import monitorer_var
        mon = monitorer_var(rendements, garch_final, config)
        lignes_mon = []
        for niv in ['95%', '99%']:
            res = mon.get(niv, {})
            cu  = res.get('cusum', {})
            rr  = res.get('ratio', {})
            n_cu  = cu.get('n_alertes', 'N/A')
            n_rr  = rr.get('n_alertes', 'N/A')
            h_v   = cu.get('h', float('nan'))
            fen   = rr.get('window', 250)
            lignes_mon.append([
                niv,
                str(n_cu),
                _fmt(h_v, 4) if not isinstance(h_v, str) and not math.isnan(h_v) else 'N/A',
                str(n_rr),
                str(fen),
            ])
        story.extend(_tableau_eviews(
            titre='Monitoring CUSUM et ratio glissant des violations',
            colonnes=['Niveau', 'Alertes CUSUM', 'Seuil h', 'Alertes ratio', 'Fenetre (j)'],
            lignes=lignes_mon,
            col_widths=[LARG_U * v for v in [0.14, 0.22, 0.18, 0.22, 0.24]],
            note=(
                'CUSUM : S_t = max(0, S_{t-1} + Hit_t - k), alerte si S_t > h. '
                'Ratio : taux exceptions sur fenetre glissante, alerte si > 2*(1-alpha).'
            ),
        ))
    except Exception as e:
        _warn(f'Section 10.5 monitoring : {e}')
        story.append(_p(f'Monitoring indisponible : {e}'))

    story.append(_spacer(0.4))
    story.append(_p(
        'La modelisation ARIMA-GARCH confirme les proprietes stylisees des '
        f'rendements de {nom} : non-normalite, heteroscedasticite conditionnelle '
        'et persistance elevee de la variance. La VaR dynamique GARCH produit des '
        'estimations plus calibrees que les methodes statiques, notamment lors '
        'des episodes de turbulence extreme (crises financiere, COVID, Ukraine).'
    ))

    return story


# =============================================================================
# Section 11 — Journal d'execution
# =============================================================================



# =============================================================================
# Section 11 — Journal d'execution
# =============================================================================

def section_11(meta: dict, config: dict) -> list:
    """
    Section 11 : Journal d'execution.
    Versions logicielles, parametres de grille, timing, avertissements.

    Parameters
    ----------
    meta : dict  Cles optionnelles : ticker, n_obs, modele_garch, dist_garch,
                 arima_spec, n_garch_evalues, n_arima_evalues, elapsed_s,
                 warnings_list.
    """
    import sys
    import importlib
    story = []

    story.append(_h1("11. Journal d'execution"))
    story.append(_p(
        'Cette section documente les conditions de production du rapport : '
        'versions logicielles, parametres de selection, timing et '
        'eventuels avertissements emis durant l\'execution.'
    ))

    # 11.1 Environnement logiciel
    story.append(_h2('11.1 Environnement logiciel'))
    try:
        import platform
        py_ver = (f'{sys.version_info.major}.{sys.version_info.minor}'
                  f'.{sys.version_info.micro}')
        pkgs   = ['numpy', 'pandas', 'scipy', 'statsmodels', 'arch', 'reportlab']
        lignes = [['Python',      py_ver],
                  ['Plateforme',  platform.system() + ' ' + platform.release()]]
        for pkg in pkgs:
            try:
                mod = importlib.import_module(pkg)
                ver = getattr(mod, '__version__', '?')
            except Exception:
                ver = 'N/A'
            lignes.append([pkg, ver])
        cw = [LARG_U * 0.45, LARG_U * 0.55]
        story.extend(_tableau_eviews(
            titre='Versions logicielles',
            colonnes=['Package', 'Version'],
            lignes=lignes, col_widths=cw,
        ))
    except Exception as e:
        _warn(f'Section 11 environnement : {e}')

    story.append(_spacer(0.4))

    # 11.2 Parametres d'execution
    story.append(_h2("11.2 Parametres d'execution"))
    try:
        elapsed = meta.get('elapsed_s', float('nan'))
        elapsed_str = f'{elapsed:.1f} s' if not _is_nan(elapsed) else 'N/A'
        ticker_val  = meta.get('ticker',
                       config.get('data', {}).get('ticker', 'N/A'))
        lignes = [
            ['Ticker',                  str(ticker_val)],
            ['Observations',            str(meta.get('n_obs',           'N/A'))],
            ['Spec. ARIMA retenue',     str(meta.get('arima_spec',      'N/A'))],
            ['Modele GARCH retenu',     str(meta.get('modele_garch',    'N/A'))],
            ['Distribution GARCH',      str(meta.get('dist_garch',      'N/A'))],
            ['N modeles ARIMA evalues', str(meta.get('n_arima_evalues', 'N/A'))],
            ['N modeles GARCH evalues', str(meta.get('n_garch_evalues', 'N/A'))],
            ['Temps total execution',   elapsed_str],
        ]
        cw = [LARG_U * 0.55, LARG_U * 0.45]
        story.extend(_tableau_eviews(
            titre="Parametres d'execution",
            colonnes=['Parametre', 'Valeur'],
            lignes=lignes, col_widths=cw,
        ))
    except Exception as e:
        _warn(f'Section 11 parametres : {e}')

    story.append(_spacer(0.4))

    # 11.3 Avertissements
    story.append(_h2('11.3 Avertissements emis'))
    warnings_list = list(meta.get('warnings_list', []))
    try:
        from tickerlab.core.rapport._helpers import _WARNINGS_LOG
        if _WARNINGS_LOG:
            warnings_list = list(_WARNINGS_LOG) + [
                w for w in warnings_list if w not in _WARNINGS_LOG
            ]
    except Exception as exc:
        _sections_log.debug('section_annexes : _WARNINGS_LOG indisponible, liste inchangée : %s', exc)

    if warnings_list:
        lignes = [[str(i + 1), str(w)] for i, w in enumerate(warnings_list)]
        cw     = [LARG_U * 0.07, LARG_U * 0.93]
        story.extend(_tableau_eviews(
            titre=f'Avertissements ({len(warnings_list)} emis)',
            colonnes=['#', 'Message'],
            lignes=lignes, col_widths=cw,
        ))
    else:
        story.append(_p('Aucun avertissement emis lors de l\'execution.'))

    return story


# =============================================================================
# Annexes — Methodologie et references
# =============================================================================

