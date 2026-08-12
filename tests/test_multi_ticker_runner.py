# -*- coding: utf-8 -*-
"""
Tests — Runner multi-tickers.

4 tests < 30s total (run_pipeline mocke au niveau du runner).

Test 1 : run_unique pipeline ok    -> erreur vide, metriques extraites
Test 2 : run_unique pipeline crash  -> erreur capturee, erreur.txt cree
Test 3 : colonnes CSV obligatoires  -> 33 colonnes, parseable
Test 4 : append CSV deux runs       -> 2 lignes, tickers corrects, sans collision
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_multi_ticker import (
    run_unique, _extraire_metriques, _metriques_vides,
    _append_csv, _COLONNES_CSV, _slug_ticker,
)


# ── Fake result (evite un vrai appel GARCH) ───────────────────────────────────

def _fake_rendements(n=400, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.5, n)
    return pd.Series(r, index=pd.bdate_range('2020-01-02', periods=n))


def _fake_pipeline_result(ticker='BZ=F', n=400) -> dict:
    """Dict minimal compatible avec _extraire_metriques — sans ModelResult."""
    rend = _fake_rendements(n=n)
    T_train = int(n * 0.70)

    df_var = pd.DataFrame({
        'Niveau':              ['95%', '99%'],
        'VaR GARCH':           [-1.21, -1.88],
        'TVaR GARCH':          [-1.55, -2.30],
        'VaR Historique':      [-1.10, -1.75],
        'TVaR Historique':     [-1.40, -2.10],
    }).set_index('Niveau')

    df_bt = pd.DataFrame([
        {'Methode': 'GARCH dyn.', 'Niveau': '95%', 'p_UC': 0.32, 'p_CC': 0.28},
        {'Methode': 'GARCH dyn.', 'Niveau': '99%', 'p_UC': 0.41, 'p_CC': 0.35},
        {'Methode': 'FHS',        'Niveau': '95%', 'p_UC': 0.55, 'p_CC': 0.48},
        {'Methode': 'FHS',        'Niveau': '99%', 'p_UC': 0.62, 'p_CC': 0.58},
    ])

    return {
        'prix':       rend.to_frame('prix'),
        'rendements': rend,
        'T_train':    T_train,
        'T_eff_dyn':  n - T_train,
        'arima': {
            'p_opt': 1, 'd_opt': 0, 'q_opt': 1,
            'aic': -120.5,
            'interpretation': 'martingale_marche_efficient',
        },
        'garch_best': pd.Series({
            'modele': 'GARCH', 'p': 1, 'o': 0, 'q': 1,
            'dist': 'normal', 'AIC': -210.3,
        }),
        'garch_final': None,
        'var':         df_var,
        'backtest':    df_bt,
        'vol':         pd.DataFrame(),
        'igarch_diag': {
            'persistance': 0.93,
            'near_igarch': False,
            'half_life_periodes': 9.5,
            'code': 'mean_reverting',
        },
        'component_garch': None,
        'fhs': {
            'var_fhs_par_horizon': {1: {0.95: -1.18, 0.99: -1.82}},
            'residus_iid_ok': True,
            'n_boot': 200,
        },
        'dm_gk': {
            0.99: {
                'GARCH dyn._vs_FHS': {
                    'dm_stat': 0.42, 'dm_pval': 0.67,
                    'gk_stat': 0.18, 'gk_pval': 0.83,
                    'verdict': 'equivalent',
                }
            }
        },
        'spec_verdict': {
            'spec_ok': True,
            'gravite': 'aucune',
            'all_candidates_failed_spec': False,
            'message': '',
        },
        'motif_selection': 'spec_OK + AIC + parcimonie',
        'df_garch': pd.DataFrame(),
    }


def _cfg_minimal():
    """Config minimal : seulement les cles que _build_config attend."""
    return {
        'data': {'ticker': 'BZ=F', 'start_date': '2020-01-01',
                 'end_date': '2022-01-01', 'frequency': 'daily'},
        'arima': {'p_max': 1, 'q_max': 1, 'seuil_significativite': 0.05,
                  'preference_parcimonie': True, 'tolerance_aic_parcimonie': 10},
        'garch': {
            'modeles': ['GARCH'], 'distributions': ['normal'],
            'p_max': 1, 'q_max': 1,
            'seuil_significativite': 0.05,
            'critere_information': 'AIC',
            'tolerance_delta_critere_brut': 4.0,
            'seuil_igarch': 0.98,
            'score_composite': {'enabled': False},
        },
        'var':      {'niveaux': [0.95, 0.99], 'n_simulations_mc': 1000},
        'backtest': {'split_ratio': 0.70, 'niveaux_test': [0.95, 0.99],
                     'seuil_p_value': 0.05},
        'fhs':      {'enabled': True, 'n_boot': 200, 'horizons': [1],
                     'seed': 42, 'fenetre_residus': None},
        'dm_gk':    {'enabled': True, 'test_type': 'gk', 'alpha_test': 0.05,
                     'alpha_test_petit_echantillon': 0.10, 'hac_lags': 'auto',
                     'n_instruments_gk': 2},
        'component_garch': {'enabled': False},
        'rolling_backtest': {'enabled': False},
        'output':   {'dossier_resultats': '', 'pdf_unique': False,
                     'export_latex': False, 'export_excel': False,
                     'format_graphiques': [], 'dpi': 72, 'force_clean': False},
        'bootstrap': {'express': False, 'enabled': False,
                      'n_replications': 50, 'block_length': 5,
                      'niveaux_ic': [0.95], 'seed': 42, 'inclure_tvar': False},
        'rapport':  {'theme': 'academique', 'var_niveaux_graphique': [0.95, 0.99],
                     'une_page_un_message': False, 'max_lignes_par_tableau': 12,
                     'lags_ljung_box_rapport': [1, 5, 10]},
        'sorties_etendues': {'correlogramme_lags': 10, 'ljung_box_lags': 5,
                             'stats_descriptives_decimales': 4},
        'ai':       {'enabled': False},
        '_reuse_cache': False,
    }


# ── Test 1 : run_unique pipeline ok ──────────────────────────────────────────

def test_run_unique_wrapper_ok(tmp_path):
    """
    run_pipeline mocke -> retourne _fake_pipeline_result.
    run_unique doit : erreur=='', garch_modele non vide, n_obs > 0, T_train > 0.
    """
    from unittest.mock import patch

    fake = _fake_pipeline_result(ticker='BZ=F', n=400)

    with patch('scripts.run_multi_ticker.run_pipeline', return_value=fake):
        metrics = run_unique('BZ=F', 'daily', _cfg_minimal(), str(tmp_path))

    assert metrics['erreur'] == '', (
        f"Erreur inattendue : {metrics['erreur']}"
    )
    assert metrics['garch_modele'] not in ('', float('nan'), None), (
        'garch_modele vide'
    )
    assert metrics['n_obs'] > 0
    assert metrics['T_train'] > 0
    assert metrics['ticker'] == 'BZ=F'
    assert metrics['frequence'] == 'daily'
    assert float(metrics['persistance']) == pytest.approx(0.93, abs=1e-6)
    assert metrics['spec_gravite'] == 'aucune'
    assert metrics['var99_garch'] == pytest.approx(-1.88, abs=1e-6)
    assert metrics['var99_fhs_h1'] == pytest.approx(-1.82, abs=1e-6)
    assert metrics['kupiec_99_pval'] == pytest.approx(0.41, abs=1e-6)


# ── Test 2 : run_unique crash pipeline ────────────────────────────────────────

def test_run_unique_pipeline_exception(tmp_path):
    """
    run_pipeline leve RuntimeError -> run_unique capture l'erreur,
    retourne erreur != '', cree erreur.txt dans le run_dir.
    """
    from unittest.mock import patch

    with patch('scripts.run_multi_ticker.run_pipeline',
               side_effect=RuntimeError('GARCH fit impossible — serie trop courte')):
        metrics = run_unique('BZ=F', 'daily', _cfg_minimal(), str(tmp_path))

    assert metrics['erreur'] != '', (
        "run_unique aurait du capturer l'erreur"
    )
    assert 'GARCH fit impossible' in metrics['erreur'] or 'RuntimeError' in metrics['erreur']

    slug = _slug_ticker('BZ=F')
    erreur_txt = tmp_path / f'{slug}_daily' / 'erreur.txt'
    assert erreur_txt.exists(), (
        f'erreur.txt doit etre cree dans {tmp_path / f"{slug}_daily"}'
    )
    contenu = erreur_txt.read_text(encoding='utf-8')
    assert 'RuntimeError' in contenu or 'GARCH' in contenu


# ── Test 3 : colonnes CSV ─────────────────────────────────────────────────────

def test_csv_colonnes_obligatoires(tmp_path):
    """
    Trois rows (2 vides + 1 avec metriques) -> CSV avec exactement les 33 colonnes.
    """
    csv_path = tmp_path / 'summary.csv'

    rows = [_metriques_vides('BZ=F', 'weekly'),
            _metriques_vides('XXXXXXXX', 'daily')]

    row3 = _metriques_vides('^GSPC', 'weekly')
    row3['n_obs']        = 500
    row3['garch_modele'] = 'GARCH(1,0,1)[t]'
    row3['spec_gravite'] = 'aucune'
    row3['near_igarch']  = False
    rows.append(row3)

    for r in rows:
        _append_csv(r, csv_path)

    assert csv_path.exists(), 'CSV absent apres _append_csv'
    df = pd.read_csv(csv_path)
    assert len(df) == 3, f'3 lignes attendues, {len(df)} lues'

    for col in _COLONNES_CSV:
        assert col in df.columns, f"Colonne '{col}' absente du CSV"

    assert set(df.columns) == set(_COLONNES_CSV), (
        f'Colonnes supplementaires ou manquantes : '
        f'{set(df.columns).symmetric_difference(set(_COLONNES_CSV))}'
    )


# ── Test 4 : append CSV deux runs sans collision ──────────────────────────────

def test_csv_append_deux_runs(tmp_path):
    """
    Deux _append_csv successifs -> 2 lignes, tickers distincts,
    toutes les colonnes presentes, pas de collision.
    Valide que l'ecriture incrementale (mode 'a') fonctionne correctement.
    """
    csv_path = tmp_path / 'multi.csv'

    row1 = _metriques_vides('BZ=F', 'weekly')
    row1['garch_modele'] = 'GARCH(1,0,1)[normal]'
    row1['spec_gravite'] = 'aucune'
    row1['erreur']       = ''

    row2 = _metriques_vides('^GSPC', 'daily')
    row2['garch_modele']  = 'GJR-GARCH(1,1,1)[t]'
    row2['spec_gravite']  = 'mineure'
    row2['near_igarch']   = True
    row2['persistance']   = 0.991
    row2['erreur']        = ''

    _append_csv(row1, csv_path)
    _append_csv(row2, csv_path)

    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2, f'2 lignes attendues, {len(df)} lues'

    tickers = set(df['ticker'].tolist())
    assert tickers == {'BZ=F', '^GSPC'}, f'Tickers incorrects : {tickers}'

    for col in _COLONNES_CSV:
        assert col in df.columns, f"Colonne '{col}' absente"

    row_gspc = df[df['ticker'] == '^GSPC'].iloc[0]
    assert row_gspc['spec_gravite'] == 'mineure'
    assert float(row_gspc['persistance']) == pytest.approx(0.991, abs=1e-6)
    assert bool(row_gspc['near_igarch']) is True
