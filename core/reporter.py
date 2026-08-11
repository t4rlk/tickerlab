# -*- coding: utf-8 -*-
"""Generation du rapport recapitulatif et exports Excel."""
import logging
import subprocess
import shutil
import pandas as pd
from pathlib import Path

_log = logging.getLogger('tickerlab.reporter')


def generer_rapport(dossier_out, prix, rendements, arima_result, best,
                    garch_final, df_var, df_bt, df_vol, config, T_train, T_eff_dyn,
                    skip_graphiques=False):
    """
    Affiche le recapitulatif en console et sauvegarde les resultats en Excel.

    Parameters
    ----------
    dossier_out : str
        Repertoire de sortie.
    prix : pd.DataFrame
        Serie des prix bruts.
    rendements : pd.Series
        Log-rendements.
    arima_result : dict
        Sortie de selectionner_arima() (cles : p_opt, d_opt, q_opt, aic).
    best : dict
        Parametres du meilleur modele GARCH.
    garch_final : arch.univariate.base.ARCHModelResult
        Resultat GARCH estime sur la serie complete.
    df_var : pd.DataFrame
        Tableau VaR/TVaR toutes methodes.
    df_bt : pd.DataFrame
        Tableau de backtesting OOS.
    df_vol : pd.DataFrame
        Sortie de construire_df_vol() (volatilite conditionnelle).
    config : dict
        Configuration YAML complete.
    T_train : int
        Taille de l'echantillon d'estimation.
    T_eff_dyn : int
        Taille de la fenetre OOS.
    skip_graphiques : bool, optional
        Si True, desactive la compilation LaTeX/PDF (defaut False).
        Utilise quand pdf_unique=True pour eviter une double generation PDF.
    """
    ticker = config['data']['ticker']
    start  = config['data']['start_date']
    end    = config['data']['end_date']
    T_full = len(rendements.dropna())

    params = garch_final.params
    nu_fit = float(params.get('nu', float('nan')))

    try:
        from tickerlab.core.rapport._stats import persistance_garch as _pers_fn
        persistance = _pers_fn(best.get('modele', best.get('vol', '')), params)
        label_pers  = 'Persistance'
    except Exception:
        persistance = float('nan')
        label_pers  = 'Persistance'

    sep = '=' * 65
    _log.info('%s', sep)
    _log.info('  RECAPITULATIF - %s  (%s -> %s)', ticker, start, end)
    _log.info('%s', sep)
    _log.info('  Observations totales : %d  (train=%d | OOS=%d)', T_full, T_train, T_eff_dyn)
    _log.info('  Modele ARIMA         : ARIMA(%d,%d,%d)  AIC=%.6f',
              arima_result['p_opt'], arima_result['d_opt'],
              arima_result['q_opt'], arima_result['aic'])
    _log.info('  Modele GARCH         : %s(%d,%d,%d) dist=%s  AIC=%.2f  (norm=%.6f)',
              best['modele'], int(best['p']), int(best['o']), int(best['q']),
              best['dist'], best['AIC'], best.get('AIC_norm', float('nan')))
    if not pd.isna(nu_fit):
        _log.info('  nu Student           : %.4f', nu_fit)
    if not pd.isna(persistance):
        _log.info('  %-27s: %.4f', label_pers, persistance)
    _log.info('')
    _log.info('  VaR & TVaR principales (%%) :')
    cols_show = [c for c in ['VaR Historique', 'TVaR Historique',
                              'VaR GARCH', 'TVaR GARCH',
                              'VaR Monte Carlo (1j)', 'TVaR Monte Carlo (1j)']
                 if c in df_var.columns]
    _log.info('%s', df_var[cols_show].to_string())
    _log.info('')
    _log.info('  Backtesting OUT-OF-SAMPLE (verdict conjoint CC) :')
    bt_display = df_bt[['Methode', 'Niveau', 'N viol.', 'Taux obs.',
                         'p_CC', 'Verdict CC']].copy()
    _log.info('%s', bt_display.to_string(index=False))
    _log.info('%s', sep)

    # ── Export Excel (toujours actif) ─────────────────────────────────────────
    if config.get('output', {}).get('export_excel'):
        out = Path(dossier_out)
        out.mkdir(parents=True, exist_ok=True)
        safe_ticker = ticker.replace('=', '_').replace('/', '_')
        excel_path  = out / f'resultats_{safe_ticker}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_var.to_excel(writer, sheet_name='VaR_TVaR')
            df_bt.to_excel(writer, sheet_name='Backtest_OOS', index=False)
            if df_vol is not None:
                df_vol.to_excel(writer, sheet_name='Volatilite_conditionnelle')
        _log.info('  Excel -> %s', excel_path)

    # ── Compilation LaTeX (desactivee quand skip_graphiques=True) ────────────
    if not skip_graphiques and config.get('output', {}).get('export_latex'):
        latex_dir = Path(dossier_out) / 'latex'
        main_tex  = latex_dir / 'main.tex'
        if main_tex.exists() and shutil.which('pdflatex'):
            try:
                for _ in range(2):
                    subprocess.run(
                        ['pdflatex', '--enable-installer',
                         '-interaction=nonstopmode', 'main.tex'],
                        cwd=latex_dir, capture_output=True
                    )
                pdf_path = latex_dir / 'main.pdf'
                if pdf_path.exists():
                    _log.info('  PDF   -> %s', pdf_path)
            except Exception as e:
                _log.warning('  PDF   : echec compilation (%s)', e)
