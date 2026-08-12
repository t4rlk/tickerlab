# -*- coding: utf-8 -*-
"""
Pipeline TickerLab complet - Orchestration end-to-end.

Usage :
    python main.py                       # config.yaml par defaut
    python main.py --config custom.yaml  # config personnalisee
    python main.py --ticker CL=F         # override du ticker (WTI)

Etapes :
    1. Telechargement des donnees
    2. Stationnarite + ARIMA
    3. Grid search GARCH
    4. VaR & TVaR
    5. Graphique volatilite conditionnelle
    6. Backtesting OOS
    7. Exports LaTeX (tableaux)
    8. Rapport + Excel
    9. (Optionnel) Redaction IA + compilation PDF
"""
import argparse
import datetime
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
import yaml
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tickerlab.core.config_schema  import valider_config_structure
from tickerlab.core.data_loader    import telecharger_prix, calculer_rendements, resampler_prix, valider_donnees
from tickerlab.core.stationarity   import analyser_stationnarite
from tickerlab.core.arima_selector import selectionner_arima
from tickerlab.core.garch_selector import grid_search_garch, selectionner_meilleur, estimer_final, verdict_specification
from tickerlab.core.var_engine     import calculer_var_tvar, construire_df_vol
from tickerlab.core.backtest         import backtest_oos
from tickerlab.core.backtest_rolling import backtest_rolling_var
from tickerlab.core.reporter                import generer_rapport
from tickerlab.core.sorties_complementaires import generer_toutes_sorties
from tickerlab.core.stats_descriptives      import generer_stats_descriptives
from tickerlab.core.correlogrammes          import generer_correlogrammes
from tickerlab.core.depassements_annuels    import generer_depassements_annuels
from tickerlab.utils.latex_export           import exporter_tous_tableaux
from tickerlab.utils.plotter                import (setup_style, tracer_rendements_vol,
                                                        tracer_var_dynamique_breaches,
                                                        tracer_garch_comparaison)
from tickerlab.utils.ai_writer              import construire_contexte, rediger_toutes_sections
from tickerlab.utils.cache                  import charger_cache, sauvegarder_cache
from tickerlab.core.rapport_pdf_unique      import generer_pdf_unique as _generer_pdf_unique
from tickerlab.core.igarch_diagnostic       import diagnostiquer_igarch
from tickerlab.core.component_garch         import estimer_component_garch
from tickerlab.core.fhs                     import calculer_var_fhs
from tickerlab.core.dm_gk                   import build_var_series, comparer_methodes_var
from tickerlab.core._logging                import setup_tickerlab_logger, get_logger
from tickerlab.core.config_validation       import config_hash, valider_config
from tickerlab.core.pipeline_result         import PipelineResult
from tickerlab.core.exceptions              import PipelineError
from tickerlab.core.cache_v2                import CacheEtapes


def slugify_ticker(ticker: str) -> str:
    """BZ=F -> BZ_F  |  ^GSPC -> GSPC  |  EURUSD=X -> EURUSD_X  |  BTC-USD -> BTC_USD"""
    return ticker.replace('=', '_').replace('^', '').replace('-', '_')


def _diagnostic_oos_final(df_bt: 'pd.DataFrame', modele_nom: str, config: dict) -> None:
    """Diagnostic post-selection OOS — informatif uniquement."""
    _log = get_logger('main')
    oos_cfg = config.get('garch', {}).get('validation_oos', {})
    if not oos_cfg.get('active', True):
        return

    seuil_kup  = float(oos_cfg.get('seuil_kupiec_p',         0.05))
    seuil_chr  = float(oos_cfg.get('seuil_christoffersen_p', 0.05))

    rows_garch = df_bt[df_bt['Methode'] == 'GARCH dyn.']
    if rows_garch.empty:
        return

    alertes = []
    for _, row in rows_garch.iterrows():
        niveau = row['Niveau']
        p_uc   = row.get('p_UC',  'n/a')
        p_cc   = row.get('p_CC',  'n/a')
        if isinstance(p_uc, (int, float)) and p_uc < seuil_kup:
            alertes.append(f'    VaR {niveau} : Kupiec p={p_uc:.4f} < {seuil_kup} (couverture incorrecte)')
        if isinstance(p_cc, (int, float)) and p_cc < seuil_chr:
            alertes.append(f'    VaR {niveau} : Christoffersen p={p_cc:.4f} < {seuil_chr} (regroupement violations)')

    if alertes:
        _log.warning('  [OOS-DIAG] ALERTE backtest GARCH dyn. (%s) :', modele_nom)
        for a in alertes:
            _log.warning('%s', a)
        _log.warning('  [OOS-DIAG] Diagnostic informatif — selection inchangee.\n')
    else:
        _log.info('  [OOS-DIAG] %s : VaR dynamique validee (Kupiec + Christoffersen OK)\n',
                  modele_nom)


def _nettoyer_anciennes_sorties(config: dict) -> None:
    """Supprime graphiques/, latex/ et latex_doc/ si force_clean=True."""
    _log = get_logger('main')
    if not config.get('output', {}).get('force_clean', False):
        return
    dossier = Path(config['output']['dossier_resultats'])
    for sous_dir in ('graphiques', 'latex', 'latex_doc'):
        p = dossier / sous_dir
        if p.exists():
            shutil.rmtree(p)
            _log.info('  [clean] %s supprime', p)


def run_pipeline(config: dict, verbosity: str = 'normal',
                 force_refresh: bool = False) -> PipelineResult:
    """Orchestre le pipeline ARIMA-GARCH-VaR complet.

    Parameters
    ----------
    config : dict
        Configuration YAML chargée.
    verbosity : {'normal', 'quiet'}
        'quiet' → seuls les WARNING sont émis (utilisé par le runner multi-tickers).
    force_refresh : bool
        True → ignore les lectures du cache (écriture seulement).
    """
    import logging as _logging
    _level = _logging.WARNING if verbosity == 'quiet' else _logging.INFO
    setup_tickerlab_logger(level=_level)
    logger = get_logger('main')

    t_debut    = time.time()
    _t: dict   = {}
    setup_style()
    ticker     = config['data']['ticker']
    pdf_unique = config.get('output', {}).get('pdf_unique', False)
    reuse      = config.get('_reuse_cache', False)
    sep        = '=' * 65
    logger.info('\n%s\n  PIPELINE TICKERLAB - %s\n%s\n', sep, ticker, sep)

    _nettoyer_anciennes_sorties(config)

    # Résolution en absolu dès l'entrée du pipeline — évite les doublons de
    # chemin quand CWD != racine du projet (ex: tickerlab/resultats/ → doublons).
    dossier_resultats = str(Path(config['output']['dossier_resultats']).resolve())
    if config.get('output', {}).get('sous_dossier_par_ticker', True):
        dossier_resultats = str(Path(dossier_resultats) / slugify_ticker(ticker))
    config['output']['dossier_resultats'] = dossier_resultats
    logger.info('  Sorties -> %s', dossier_resultats)
    Path(dossier_resultats).mkdir(parents=True, exist_ok=True)

    # ── Cache versionne par etape────────────────────────────────
    _cache_v2_enabled = config.get('cache_v2', {}).get('enabled', True)
    _cv2 = CacheEtapes(dossier_resultats, config) if _cache_v2_enabled else None
    _meta_cache: dict = {}

    # Ancien cache monolithique : uniquement si cache_v2 desactive (mode legacy)
    if not _cache_v2_enabled and reuse:
        _old = charger_cache(dossier_resultats, config)
        if _old is not None:
            logger.info('  [Cache legacy] Resultats charges (cache_v2 desactive).\n')
            prix         = _old['prix']
            rendements   = _old['rendements']
            arima_result = _old['arima_result']
            df_garch     = _old['df_garch']
            best         = _old['best']
            garch_final  = _old['garch_final']
            df_var       = _old['df_var']
            df_bt        = _old['df_bt']
            df_vol       = _old['df_vol']
            T_train      = _old['T_train']
            T_eff_dyn    = _old['T_eff_dyn']
            trace_garch  = _old.get('trace_garch', [])
            df_violations   = _old.get('df_violations',   None)
            df_params_drift = _old.get('df_params_drift', None)
            stats_rolling   = _old.get('stats_rolling',   {})
            igarch_diag       = _old.get('igarch_diagnostic', {})
            component_garch_r = _old.get('component_garch_result', None)
            fhs_r             = _old.get('fhs_result', None)
            dm_gk_r           = _old.get('dm_gk_result', None)
            freq       = config['data'].get('frequency', 'daily')
            prix_stats = resampler_prix(prix, freq=freq)
            rb_cfg = config.get('rolling_backtest', {})
            if rb_cfg.get('enabled', False) and df_violations is None:
                logger.info('  [Rolling] Non cache — calcul...')
                try:
                    best_d = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
                    df_violations, df_params_drift, stats_rolling = backtest_rolling_var(
                        rendements, best_d, config
                    )
                except Exception as e:
                    logger.warning('  [Rolling] ECHEC : %s', e)
            # skip all computation below
            motif = locals().get('motif', '')
        else:
            _old = None
    else:
        _old = None

    sb_result = None
    if _old is None:
        # Avertir si ancien cache present mais cache_v2 actif
        import os as _os
        _old_pkl = _os.path.join(dossier_resultats, 'pipeline_state.pkl')
        if _cache_v2_enabled and _os.path.exists(_old_pkl):
            logger.info('  [Cache] Ancien cache pipeline_state.pkl detecte — '
                        'ignoré (cache_v2 actif). Peut être supprime.')

        # ── 1. Donnees ───────────────────────────────────────────────────────
        _t0 = time.time()
        _cached = _cv2.get('download') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            prix, rendements, prix_stats = _cached
            _meta_cache['download'] = 'HIT'
            logger.info('  [Cache] download HIT\n')
        else:
            prix       = telecharger_prix(ticker=config['data']['ticker'],
                                          start_date=config['data']['start_date'],
                                          end_date=config['data']['end_date'],
                                          auto_adjust=config['data'].get('auto_adjust', True))
            rendements = calculer_rendements(prix, freq=config['data']['frequency'])
            freq       = config['data'].get('frequency', 'daily')
            prix_stats = resampler_prix(prix, freq=freq)
            if _cv2:
                _cv2.set('download', (prix, rendements, prix_stats))
            _meta_cache['download'] = 'MISS'
        freq = config['data'].get('frequency', 'daily')
        _t['donnees'] = time.time() - _t0
        logger.info('%d prix | %d rendements | %d prix %s\n',
                    len(prix), len(rendements), len(prix_stats), freq)
        valider_config(config, len(rendements.dropna()))
        valider_donnees(prix, rendements, config)

        # ── 2. Stationnarite + ARIMA ─────────────────────────────────────────
        _t0 = time.time()
        _cached = _cv2.get('arima') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            arima_result, d, serie_arima, d_arima = _cached
            _meta_cache['arima'] = 'HIT'
            logger.info('  [Cache] arima HIT\n')
        else:
            d, serie_arima, d_arima = analyser_stationnarite(prix, rendements)
            arima_result = selectionner_arima(serie_arima, d_arima, **config['arima'])
            if _cv2:
                _cv2.set('arima', (arima_result, d, serie_arima, d_arima))
            _meta_cache['arima'] = 'MISS'
        _t['arima'] = time.time() - _t0
        logger.info('>>> ARIMA(%d,%d,%d)  AIC=%.6f  motif=%s\n',
                    arima_result['p_opt'], arima_result['d_opt'],
                    arima_result['q_opt'], arima_result['aic'],
                    arima_result['motif_selection'])
        if not arima_result.get('fiabilite', True):
            seuil_pct = config['arima'].get('seuil_significativite', 0) * 100
            logger.warning('  >>> WARNING : aucun ARIMA significatif au seuil %.0f%%'
                           ' — modele retenu par AIC seul (validation a completer).\n',
                           seuil_pct)

        # ── 3. GARCH ─────────────────────────────────────────────────────────
        # CRITIQUE : tuple stocke APRES selectionner_meilleur + estimer_final
        # pour que df_garch contienne les colonnes spec enrichies in-place.
        _t0 = time.time()
        _cached = _cv2.get('garch') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            df_garch, best, motif, trace_garch, garch_final = _cached
            _meta_cache['garch'] = 'HIT'
            logger.info('  [Cache] garch HIT\n')
        else:
            df_garch, _z_cache = grid_search_garch(rendements, build_z_cache=True,
                                                    **config['garch'])
            best, motif, trace_garch = selectionner_meilleur(
                df_garch, rendements, config, z_cache=_z_cache
            )
            garch_final = estimer_final(rendements, **best.to_dict())
            if _cv2:
                _cv2.set('garch', (df_garch, best, motif, trace_garch, garch_final))
            _meta_cache['garch'] = 'MISS'
        logger.info('>>> %s(%d,%d,%d) [%s]  AIC=%.2f  (AIC/n=%.6f)  (%s)\n',
                    best['modele'], int(best['p']), int(best['o']), int(best['q']),
                    best['dist'], best['AIC'], best.get('AIC_norm', float('nan')), motif)
        _t['garch'] = time.time() - _t0

        # ── 3b. Diagnostic IGARCH post-hoc───────────────────────
        _t0 = time.time()
        _cached = _cv2.get('igarch_diag') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            igarch_diag = _cached
            _meta_cache['igarch_diag'] = 'HIT'
        else:
            igarch_diag = diagnostiquer_igarch(
                garch_final,
                vol_type=str(best.get('modele', 'GARCH')),
                seuil_igarch=config.get('garch', {}).get('seuil_igarch', 0.98),
                frequence_serie=config.get('data', {}).get('frequency', 'daily'),
            )
            if _cv2:
                _cv2.set('igarch_diag', igarch_diag)
            _meta_cache['igarch_diag'] = 'MISS'
        _t['igarch_diag'] = time.time() - _t0

        # ── 3b-bis. Ruptures structurelles ICSS (Condition 3) ────────────────
        _t0 = time.time()
        try:
            from tickerlab.core.structural_breaks import analyser_icss_pipeline as _icss_fn
            sb_result = _icss_fn(rendements, garch_final, config, garch_best=best)
        except Exception as _sb_e:
            logger.warning('  [ICSS] Echec : %s', _sb_e)
        _t['structural_breaks'] = time.time() - _t0
        if sb_result and sb_result.get('n_breaks', 0) > 0:
            logger.info('  [ICSS] %d rupture(s) | mode=%s | L&L_warning=%s\n',
                        sb_result['n_breaks'], sb_result['mode'],
                        sb_result.get('warning_lamoureux', False))

        # ── 3c. Component GARCH──────────────────────────────────
        _t0 = time.time()
        _cached = _cv2.get('component_garch') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            component_garch_r = _cached
            _meta_cache['component_garch'] = 'HIT'
        else:
            component_garch_r = estimer_component_garch(
                rendements, garch_final, config,
                near_igarch=igarch_diag.get('near_igarch', False),
            )
            if _cv2:
                _cv2.set('component_garch', component_garch_r)
            _meta_cache['component_garch'] = 'MISS'
        _t['component_garch'] = time.time() - _t0
        if component_garch_r is not None:
            logger.info('  [CompGARCH] rho=%.4f alpha+beta=%.4f nu=%.2f AIC=%.2f\n',
                        component_garch_r['rho'],
                        component_garch_r['alpha'] + component_garch_r['beta'],
                        component_garch_r['nu'], component_garch_r['aic'])

        # ── 3d. FHS — Filtered Historical Simulation─────────────
        _split_ratio = config.get('backtest', {}).get('split_ratio', 0.70)
        T_train      = int(len(rendements.dropna()) * _split_ratio)
        T_eff_dyn    = len(rendements.dropna()) - T_train

        _t0 = time.time()
        _cached = _cv2.get('fhs') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            fhs_r = _cached
            _meta_cache['fhs'] = 'HIT'
        else:
            fhs_r = calculer_var_fhs(rendements, garch_final, config)
            if _cv2:
                _cv2.set('fhs', fhs_r)
            _meta_cache['fhs'] = 'MISS'
        _t['fhs'] = time.time() - _t0
        if fhs_r is not None:
            v1  = fhs_r['var_fhs_par_horizon'].get(1,  {}).get(0.99, float('nan'))
            v22 = fhs_r['var_fhs_par_horizon'].get(22, {}).get(0.99, float('nan'))
            logger.info('  [FHS] H=1 VaR_99=%.4f%%  H=22 VaR_99=%.4f%%  iid_ok=%s  n_boot=%s\n',
                        v1, v22, fhs_r['residus_iid_ok'], fhs_r['n_boot'])

        # ── 3e. DM + GK — Comparaison statistique VaR───────────
        # Dependances : garch + fhs (pas backtest — dm_gk s'execute avant backtest_oos)
        _t0 = time.time()
        _cached = _cv2.get('dm_gk') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            dm_gk_r = _cached
            _meta_cache['dm_gk'] = 'HIT'
        else:
            dm_gk_r = None
            if config.get('dm_gk', {}).get('enabled', True):
                try:
                    _var_s = build_var_series(rendements, garch_final, fhs_r, config,
                                              T_train=T_train)
                    _r_oos = rendements.dropna().values[T_train:T_train + T_eff_dyn]
                    _niv   = config.get('var', {}).get('niveaux', [0.95, 0.99])
                    dm_gk_r = comparer_methodes_var(_var_s, _r_oos, _niv, config)
                    from tickerlab.core.dm_gk import find_pair as _fp
                    _pp = _fp(dm_gk_r.get(0.99, {}), 'GARCH dyn.', 'FHS') or {}
                    if _pp:
                        logger.info(
                            '  [DM-GK] GARCH dyn. vs FHS | alpha=99%% | '
                            'DM stat=%.3f p=%.3f | GK stat=%.3f p=%.3f | verdict=%s\n',
                            _pp['dm_stat'], _pp['dm_pval'],
                            _pp['gk_stat'], _pp['gk_pval'], _pp['verdict'])
                except Exception as _e:
                    logger.warning('  [DM-GK] ECHEC : %s\n', _e)
            if _cv2:
                _cv2.set('dm_gk', dm_gk_r)
            _meta_cache['dm_gk'] = 'MISS'
        _t['dm_gk'] = time.time() - _t0

        # ── 4. VaR & TVaR + vol ──────────────────────────────────────────────
        _t0 = time.time()
        _cached = _cv2.get('var') if (_cv2 and not force_refresh) else None
        if _cached is not None:
            df_var, df_vol = _cached
            _meta_cache['var'] = 'HIT'
            logger.info('  [Cache] var HIT\n')
        else:
            var_cfg = config.get('var', {})
            with warnings.catch_warnings(record=True) as _caught_w:
                warnings.simplefilter('always', UserWarning)
                df_var = calculer_var_tvar(
                    rendements, garch_final,
                    niveaux=var_cfg.get('niveaux', None),
                    n_simulations_mc=int(var_cfg.get('n_simulations_mc', 50_000)),
                )
            for _w in _caught_w:
                logger.warning('%s', _w.message)
            df_vol = construire_df_vol(rendements, garch_final, alpha=0.99)
            if _cv2:
                _cv2.set('var', (df_var, df_vol))
            _meta_cache['var'] = 'MISS'
        _t['var'] = time.time() - _t0

        # ── 4b. Bootstrap express IC VaR─────────────────────────
        boot_cfg = config.get('bootstrap', {})
        if bool(boot_cfg.get('express', False)) or bool(boot_cfg.get('enabled', False)):
            try:
                from tickerlab.core.var_engine import calculer_bootstrap_ci_var
                _df_ci = calculer_bootstrap_ci_var(rendements, garch_final, config)
                if not _df_ci.empty:
                    for _niv, _row in _df_ci.iterrows():
                        logger.info('  [Bootstrap] VaR GARCH %s = %.4f%%  IC 95%% [%.4f%%, %.4f%%]\n',
                                    _niv, _row['VaR GARCH'], _row['CI lower'], _row['CI upper'])
            except Exception as _eb:
                logger.warning('  [Bootstrap] ECHEC : %s\n', _eb)

        # ── 5. Backtesting OOS ───────────────────────────────────────────────
        _t0 = time.time()
        # Detail enrichi (6 tests FRTB/Berkowitz pour l'API v1) : opt-in via
        # config['backtest']['detail_frtb']. Quand demande, on court-circuite le
        # cache backtest (le detail n'y est pas stocke) pour toujours l'obtenir.
        _want_detail = bool(config.get('backtest', {}).get('detail_frtb', False))
        backtest_detail = None
        _cached = (_cv2.get('backtest')
                   if (_cv2 and not force_refresh and not _want_detail) else None)
        if _cached is not None:
            df_bt, T_train, T_eff_dyn = _cached
            _meta_cache['backtest'] = 'HIT'
            logger.info('  [Cache] backtest HIT\n')
        else:
            _bt_kwargs = {k: v for k, v in config['backtest'].items()
                          if k != 'detail_frtb'}
            _res = backtest_oos(
                rendements, best.to_dict(),
                **_bt_kwargs,
                n_simulations_mc=int(config.get('var', {}).get('n_simulations_mc', 50_000)),
                retour_detail=_want_detail,
                config_frtb=(config if _want_detail else None),
            )
            if _want_detail:
                df_bt, T_train, T_eff_dyn, backtest_detail = _res
            else:
                df_bt, T_train, T_eff_dyn = _res
            if _cv2 and not _want_detail:
                _cv2.set('backtest', (df_bt, T_train, T_eff_dyn))
            _meta_cache['backtest'] = 'MISS'
        _t['backtest'] = time.time() - _t0

        # ── 5b. Diagnostic OOS post-selection (informatif) ───────────────────
        modele_nom = (f"{best['modele']}({int(best['p'])},{int(best['o'])},"
                      f"{int(best['q'])})[{best['dist']}]")
        _diagnostic_oos_final(df_bt, modele_nom, config)

        # ── 5c. Rolling backtest (si active — non cache) ──────────────────────
        df_violations = df_params_drift = None
        stats_rolling = {}
        rb_cfg = config.get('rolling_backtest', {})
        if rb_cfg.get('enabled', False):
            logger.info('  [Rolling] window=%dj refit=/%dj',
                        rb_cfg.get('window_size', 1000), rb_cfg.get('refit_every', 22))
            try:
                df_violations, df_params_drift, stats_rolling = backtest_rolling_var(
                    rendements, best.to_dict(), config
                )
                n_ev = len(df_violations) if df_violations is not None else 0
                n_re = len(df_params_drift) if df_params_drift is not None else 0
                logger.info('  [Rolling] %d previsions | %d re-estimations OK', n_ev, n_re)
            except Exception as e:
                logger.warning('  [Rolling] ECHEC : %s', e)

        if _meta_cache:
            _hits = sum(1 for v in _meta_cache.values() if v == 'HIT')
            _total = len(_meta_cache)
            logger.info('  [Cache v2] %d/%d etapes en cache (HIT)\n', _hits, _total)

    # ── Mode metriques seules (sans generation de fichiers) ──────────────────
    if config.get('_metrics_only', False):
        best_d_m = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
        spec_v_m = verdict_specification(best_d_m, df_garch, config)
        _duree = time.time() - t_debut
        if _t:
            for _k, _v in _t.items():
                logger.info('  [PERF] %-20s %.1fs', _k, _v)
            logger.info('  [PERF] %-20s %.1fs  (TOTAL)', 'TOTAL', _duree)
        return PipelineResult(
            prix=prix, rendements=rendements,
            arima=arima_result, garch_best=best,
            garch_final=garch_final,
            var=df_var, backtest=df_bt, vol=df_vol,
            igarch_diag=igarch_diag,
            component_garch=component_garch_r,
            fhs=fhs_r,
            dm_gk=dm_gk_r,
            T_train=T_train,
            T_eff_dyn=T_eff_dyn,
            df_garch=df_garch,
            spec_verdict=spec_v_m,
            motif_selection=locals().get('motif', ''),
            structural_breaks=sb_result,
            backtest_detail=backtest_detail,
            meta={
                'ticker': ticker,
                'version': '6.3.1',
                'config_hash': config_hash(config),
                'cache': _meta_cache,
                'duree_totale_s': round(_duree, 2),
                'timings': {k: round(v, 2) for k, v in _t.items()},
                'icss_breaks': sb_result,
                'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            },
        )

    # ── Sorties ───────────────────────────────────────────────────────────────
    best_dict     = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
    dossier_graph = str(Path(dossier_resultats) / 'graphiques')

    if pdf_unique:
        # Mode PDF unique : Excel seul + PDF consolidé (pas de graphiques standalone)
        generer_rapport(
            dossier_resultats,
            prix, rendements, arima_result, best_dict, garch_final,
            df_var, df_bt, df_vol, config, T_train, T_eff_dyn,
        )
        _generer_pdf_unique(
            prix, rendements, arima_result, garch_final, df_garch,
            df_var, df_vol, df_bt, T_train, T_eff_dyn,
            config, ticker,
            t_debut=t_debut,
            best=best,
            trace_garch=trace_garch,
            df_violations=df_violations,
            df_params_drift=df_params_drift,
            stats_rolling=stats_rolling,
            igarch_diagnostic=igarch_diag,
            component_garch_result=component_garch_r,
            fhs_result=fhs_r,
            dm_gk_result=dm_gk_r,
            prix_stats=prix_stats,
        )

    else:
        # Mode classique : graphiques standalone + LaTeX + IA
        tracer_rendements_vol(df_vol, ticker, best_dict, alpha=0.99,
                              dossier=dossier_graph)

        if config['output'].get('export_latex'):
            exporter_tous_tableaux(
                dossier_resultats,
                df_arima=arima_result['df'], df_garch=df_garch,
                df_var=df_var, df_bt=df_bt, df_vol=df_vol,
                arima_result=arima_result,
                garch_best=best, garch_motif=motif,
            )

        generer_rapport(
            dossier_resultats,
            prix, rendements, arima_result, best_dict, garch_final,
            df_var, df_bt, df_vol, config, T_train, T_eff_dyn,
        )

        generer_stats_descriptives(prix, rendements, config)
        generer_correlogrammes(rendements, arima_result, config)
        generer_toutes_sorties(prix, rendements, arima_result, garch_final,
                               df_garch, df_var, best_dict, config)
        generer_depassements_annuels(rendements, best_dict, df_garch, config)
        tracer_var_dynamique_breaches(df_vol, rendements, ticker, config,
                                      dossier=dossier_graph)
        tracer_garch_comparaison(rendements, ticker, best_dict, df_garch,
                                 dossier=dossier_graph)
        _deployer_templates(config)

    # ── Verdict de specification (utile pour l'IA et le PipelineResult) ─────
    spec_verdict = verdict_specification(best_dict, df_garch, config)

    # ── Etape IA (rédaction + PDF LaTeX IA) — pdf_unique ou classique ────────
    cfg_ai = config.get('ai', {})
    if cfg_ai.get('enabled', False):
        _etape_ia(config, prix, rendements, arima_result, best_dict,
                  garch_final, df_var, df_bt, df_vol, T_train, T_eff_dyn,
                  spec_verdict=spec_verdict, igarch_diag=igarch_diag)

    # ── Export academique PDF+Excel (dossier persistant) ─────────────────────
    ea_cfg = config.get('export_academique', {})
    if ea_cfg.get('enabled', False):
        from tickerlab.core.export_academique import generer_export_academique
        generer_export_academique(
            prix=prix, rendements=rendements,
            arima_result=arima_result, garch_final=garch_final,
            df_garch=df_garch, df_var=df_var, df_bt=df_bt,
            best=best, config=config,
            T_train=T_train, T_eff_dyn=T_eff_dyn,
        )

    motif_sel = locals().get('motif', '')
    _duree = time.time() - t_debut
    if _t:
        for _k, _v in _t.items():
            logger.info('  [PERF] %-20s %.1fs', _k, _v)
        logger.info('  [PERF] %-20s %.1fs  (TOTAL)', 'TOTAL', _duree)

    return PipelineResult(
        prix=prix, rendements=rendements,
        arima=arima_result, garch_best=best,
        garch_final=garch_final,
        var=df_var, backtest=df_bt, vol=df_vol,
        igarch_diag=igarch_diag,
        component_garch=component_garch_r,
        fhs=fhs_r,
        dm_gk=dm_gk_r,
        T_train=T_train,
        T_eff_dyn=T_eff_dyn,
        df_garch=df_garch,
        spec_verdict=spec_verdict,
        motif_selection=motif_sel,
        structural_breaks=sb_result,
        backtest_detail=backtest_detail,
        meta={
            'ticker': ticker,
            'version': '6.3.1',
            'config_hash': config_hash(config),
            'cache': _meta_cache,
            'duree_totale_s': round(_duree, 2),
            'timings': {k: round(v, 2) for k, v in _t.items()},
            'icss_breaks': sb_result,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        },
    )


def _deployer_templates(config):
    """Copie les squelettes LaTeX dans resultats/latex_doc/ (sans ecraser)."""
    from tickerlab.utils.ai_writer import _nom_sous_jacent
    dossier_out   = Path(config['output']['dossier_resultats'])
    output_dir    = dossier_out / 'latex_doc'
    templates_dir = Path(__file__).parent / 'templates' / 'latex_doc'

    if not templates_dir.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for src in templates_dir.iterdir():
        dst = output_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    # Remplacer %%TICKER_NAME%% dans main.tex une seule fois
    ticker      = config['data']['ticker']
    ticker_name = _nom_sous_jacent(ticker)
    main_tex    = output_dir / 'main.tex'
    if main_tex.exists():
        contenu = main_tex.read_text(encoding='utf-8')
        if '%%TICKER_NAME%%' in contenu:
            main_tex.write_text(
                contenu.replace('%%TICKER_NAME%%', ticker_name),
                encoding='utf-8'
            )

    get_logger('main').info('  OK squelettes LaTeX -> %s', output_dir)


def _etape_ia(config, prix, rendements, arima_result, best, garch_final,
              df_var, df_bt, df_vol, T_train, T_eff_dyn,
              spec_verdict=None, igarch_diag=None):
    """Etape 8 : redaction des sections par Claude + compilation PDF."""
    import time
    sep = '=' * 65

    _log = get_logger('main')
    from tickerlab.utils.llm_providers import fabriquer_provider
    try:
        provider = fabriquer_provider(config)
    except Exception as exc:
        _log.warning('\n%s', sep)
        _log.warning('  ETAPE IA : impossible de creer le provider LLM : %s', exc)
        _log.warning('  Verifier config[ai_writer] et la variable d\'env correspondante.')
        _log.warning('%s\n', sep)
        return

    # ai.enabled = porte d'activation ; ai_writer.* = configuration du writer
    cfg_writer   = config.get('ai_writer', {})
    cfg_ai       = config.get('ai', {})
    max_tokens   = cfg_writer.get('max_tokens', cfg_ai.get('max_tokens', 4000))
    # sections_a_rediger : lu depuis ai_writer en priorité, fallback sur ai (compat)
    sections     = cfg_writer.get('sections_a_rediger',
                                  cfg_ai.get('sections_a_rediger', []))
    dossier_out  = Path(config['output']['dossier_resultats'])

    output_dir    = dossier_out / 'latex_doc'
    templates_dir = Path(__file__).parent / 'templates' / 'latex_doc'
    prompts_dir   = Path(__file__).parent / 'prompts'

    # Copier les templates (sans ecraser le contenu IA existant)
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in templates_dir.iterdir():
        dst = output_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    # Remplacer le placeholder %%TICKER_NAME%% dans main.tex
    from tickerlab.utils.ai_writer import _nom_sous_jacent
    ticker       = config['data']['ticker']
    ticker_name  = _nom_sous_jacent(ticker)
    main_tex     = output_dir / 'main.tex'
    contenu_main = main_tex.read_text(encoding='utf-8')
    contenu_main = contenu_main.replace('%%TICKER_NAME%%', ticker_name)
    main_tex.write_text(contenu_main, encoding='utf-8')

    _log.info('\n%s', sep)
    _log.info('  ETAPE IA - Redaction de %d section(s) | prompts_dir=%s',
              len(sections), prompts_dir)
    _log.info('%s', sep)

    # Construction du contexte numerique
    contexte = construire_contexte(
        ticker, prix, rendements, arima_result, best, garch_final,
        df_var, df_bt, df_vol, config, T_train, T_eff_dyn,
        spec_verdict=spec_verdict, igarch_diag=igarch_diag,
    )

    # Redaction section par section
    t0 = time.time()
    stats = rediger_toutes_sections(
        contexte, output_dir, provider, sections, prompts_dir, max_tokens
    )
    elapsed = time.time() - t0

    n_ok    = stats['sections']
    n_echec = stats.get('sections_echec', 0)
    _log.info('\n  %d section(s) redigees, %d echec(s) en %.0fs',
              n_ok, n_echec, elapsed)
    _log.info('  Tokens totaux consommes : %d', stats['tokens_total'])

    # Compilation PDF avec latexmk — seulement si toutes les sections ont reussi
    if n_echec > 0:
        _log.warning(
            '  %d section(s) en echec : compilation LaTeX annulee. '
            'Corriger la cause (cle API ? paquet manquant ?) puis relancer.',
            n_echec,
        )
        return

    pdf_ok = _compiler_latex(output_dir, _log)
    if not pdf_ok:
        _log.warning('  PDF IA non compile (voir main.log).')

    _log.info('%s', sep)


def _compiler_latex(output_dir, _log) -> bool:
    """Compile main.tex en PDF. Essaie latexmk, bascule sur pdflatex si besoin.

    latexmk est un script Perl : sous Windows/MiKTeX sans Perl il echoue sans
    produire de PDF ni de main.log. On bascule alors sur pdflatex (toujours
    present avec une distribution TeX) en enchainant les passes + bibtex.

    Returns True si main.pdf a ete produit.
    """
    pdf_path = output_dir / 'main.pdf'

    def _passes_pdflatex(pdflatex: str) -> None:
        bibtex = shutil.which('bibtex')
        # 1re passe — genere .aux. nonstopmode : ne bloque jamais sur une erreur.
        # MiKTeX : --enable-installer evite tout prompt d'installation de paquet.
        args = [pdflatex, '-interaction=nonstopmode']
        if 'miktex' in pdflatex.lower():
            args.append('--enable-installer')
        for passe in range(3):
            if passe == 1 and bibtex and (output_dir / 'main.aux').exists():
                subprocess.run([bibtex, 'main'], cwd=output_dir,
                               capture_output=True, text=True, timeout=120)
            subprocess.run(args + ['main.tex'], cwd=output_dir,
                           capture_output=True, text=True, timeout=300)

    # 1) latexmk si fonctionnel (Linux/CI avec Perl)
    latexmk = shutil.which('latexmk')
    if latexmk:
        _log.info('\n  Compilation PDF (latexmk)...')
        try:
            subprocess.run([latexmk, '-pdf', '-interaction=nonstopmode', 'main.tex'],
                           cwd=output_dir, capture_output=True, text=True, timeout=600)
        except (subprocess.TimeoutExpired, OSError):
            pass
        if pdf_path.exists():
            _log.info('  PDF final -> %s', pdf_path)
            return True
        _log.info('  latexmk inoperant — bascule sur pdflatex.')

    # 2) pdflatex direct (fallback robuste)
    pdflatex = shutil.which('pdflatex')
    if not pdflatex:
        _log.warning('  Ni latexmk ni pdflatex disponibles — PDF non compile.')
        return False
    _log.info('  Compilation PDF (pdflatex)...')
    try:
        _passes_pdflatex(pdflatex)
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log.warning('  pdflatex interrompu : %s', exc)
    if pdf_path.exists():
        _log.info('  PDF final -> %s', pdf_path)
        return True

    _log.warning('  Compilation echouee - consulter main.log')
    log_path = output_dir / 'main.log'
    if log_path.exists():
        lignes = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
        for ligne in [l for l in lignes if l.startswith('!')][:15]:
            _log.warning('    %s', ligne)
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline TickerLab')
    parser.add_argument('--config', default='config.yaml',
                        help='Chemin vers le fichier de configuration YAML')
    parser.add_argument('--ticker', default=None,
                        help='Override du ticker (ex : CL=F pour le WTI)')
    parser.add_argument('--reuse-cache', action='store_true',
                        help='Reutilise les resultats pickle si empreinte config identique')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path

    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    valider_config_structure(cfg)

    if args.ticker:
        cfg['data']['ticker'] = args.ticker

    cfg['_reuse_cache'] = args.reuse_cache

    run_pipeline(cfg)
