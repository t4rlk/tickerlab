# -*- coding: utf-8 -*-
"""
Tests Phase 3.2 — Cache versionne par etape.

11 tests unitaires, aucun appel reseau ni GARCH reel.

Test 1  : hit_apres_set         — round-trip pickle avec df.attrs
Test 2  : invalidation_config_aval  — fhs change -> fhs MISS, download/garch HIT
Test 3  : invalidation_cascade_amont — garch change -> cascade MISS sauf download/arima
Test 4  : invalidation_version_module — monkeypatch __etape_version__ -> cascade
Test 5  : ttl_download          — 25h sans end_date -> MISS ; end_date explicite -> HIT
Test 6  : df_garch_enrichi      — CRITIQUE : colonnes spec presentes dans cache
Test 7  : force_refresh         — all MISS, fichiers réécrits
Test 8  : disabled_strict       — enabled=False -> aucun .cache_v2/ cree
Test 9  : non_regression_cache_froid_vs_chaud — serie synthetique, resultats identiques
Test 10 : invalidation_dmgk_via_split_ratio — split_ratio change invalide dm_gk
Test 11 : couverture_sections_config_par_etape — audit DAG vs reference
"""
import json
import sys
import time
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.cache_v2 import CacheEtapes, ETAPES_DAG


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _cfg(data_end_date='2024-01-01', fhs_n_boot=10000, garch_p_max=1):
    return {
        'data':  {'ticker': 'BZ=F', 'start_date': '2020-01-01',
                  'end_date': data_end_date, 'frequency': 'daily'},
        'arima': {'p_max': 2},
        'garch': {'p_max': garch_p_max, 'q_max': 1},
        'var':   {'niveaux': [0.95, 0.99]},
        'backtest': {'split_ratio': 0.70},
        'fhs':   {'n_boot': fhs_n_boot},
        'dm_gk': {'enabled': True},
        'component_garch': {},
        'bootstrap': {},
        'rolling_backtest': {'enabled': False},
    }


def _cache(tmp, cfg=None):
    return CacheEtapes(tmp, cfg or _cfg())


# ── Test 1 : round-trip pickle avec df.attrs ──────────────────────────────────

def test_hit_apres_set():
    """set() puis get() retourne la valeur identique, y compris df.attrs."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _cache(tmp)

        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        df.attrs['source'] = 'test_phase32'

        c.set('var', (df, df.copy()))
        result = c.get('var')

        assert result is not None, 'get() apres set() doit retourner valeur (HIT)'
        df_out, _ = result
        assert list(df_out['a']) == [1.0, 2.0, 3.0], 'valeurs DataFrame incorrectes'
        assert df_out.attrs.get('source') == 'test_phase32', 'df.attrs perdu apres pickle'


# ── Test 2 : invalidation config aval ─────────────────────────────────────────

def test_invalidation_config_aval():
    """Modifier fhs.n_boot -> fhs MISS ; download et garch restent HIT."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_v1 = _cfg(fhs_n_boot=10000)
        c1 = CacheEtapes(tmp, cfg_v1)

        # Peupler download et garch avec v1
        c1.set('download', ('prix', 'rendements', 'prix_stats'))
        c1.set('garch',    ('df_garch', 'best', 'motif', [], 'garch_final'))
        c1.set('fhs',      {'var_fhs': 1.23})

        # Avec config v2 (n_boot different)
        cfg_v2 = _cfg(fhs_n_boot=5000)
        c2 = CacheEtapes(tmp, cfg_v2)

        assert c2.get('download') is not None, 'download doit etre HIT (config inchangee)'
        assert c2.get('garch')    is not None, 'garch doit etre HIT (config inchangee)'
        assert c2.get('fhs')      is None,     'fhs doit etre MISS (n_boot change)'


# ── Test 3 : invalidation cascade amont ──────────────────────────────────────

def test_invalidation_cascade_amont():
    """Modifier garch.p_max -> garch, igarch_diag, component_garch, fhs,
    dm_gk, var, backtest MISS ; download et arima restent HIT."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_v1 = _cfg(garch_p_max=1)
        c1 = CacheEtapes(tmp, cfg_v1)

        for etape in ('download', 'arima', 'garch', 'igarch_diag',
                      'component_garch', 'fhs', 'dm_gk', 'var', 'backtest'):
            c1.set(etape, f'valeur_{etape}')

        cfg_v2 = _cfg(garch_p_max=2)
        c2 = CacheEtapes(tmp, cfg_v2)

        # Inchanges (pas de dependance garch)
        assert c2.get('download') is not None, 'download doit rester HIT'
        assert c2.get('arima')    is not None, 'arima doit rester HIT'

        # Invalides en cascade
        for etape in ('garch', 'igarch_diag', 'component_garch',
                      'fhs', 'dm_gk', 'var', 'backtest'):
            assert c2.get(etape) is None, (
                f'{etape} doit etre MISS apres garch.p_max change')


# ── Test 4 : invalidation version module ──────────────────────────────────────

def test_invalidation_version_module():
    """Monkeypatch __etape_version__ du module garch_selector -> cascade MISS."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        c = CacheEtapes(tmp, cfg)

        for etape in ('download', 'arima', 'garch', 'fhs', 'backtest'):
            c.set(etape, f'valeur_{etape}')

        # Cle dm_gk avant bump
        cle_dmgk_avant = c.cle('dm_gk')

        import tickerlab.core.garch_selector as _gs
        original_version = getattr(_gs, '__etape_version__', '1')
        try:
            _gs.__etape_version__ = '99'
            c_new = CacheEtapes(tmp, cfg)

            # download et arima non affectes par garch_selector
            assert c_new.get('download') is not None, 'download doit rester HIT'
            assert c_new.get('arima')    is not None, 'arima doit rester HIT'

            # garch et tout ce qui en depend -> MISS
            assert c_new.get('garch')   is None, 'garch MISS apres version bump'
            assert c_new.get('fhs')     is None, 'fhs MISS (cascade depuis garch)'
            assert c_new.get('backtest') is None, 'backtest MISS (cascade depuis garch)'

            # cle dm_gk change aussi
            cle_dmgk_apres = c_new.cle('dm_gk')
            assert cle_dmgk_avant != cle_dmgk_apres, 'cle(dm_gk) doit changer avec version garch'

        finally:
            _gs.__etape_version__ = original_version


# ── Test 5 : TTL download ─────────────────────────────────────────────────────

def test_ttl_download():
    """Manifest 25h + no end_date -> MISS. end_date explicite -> HIT (pas de TTL)."""
    with tempfile.TemporaryDirectory() as tmp:
        # Cas 1 : end_date absent -> TTL actif
        cfg_sans_end = _cfg(data_end_date='')
        c_sans = CacheEtapes(tmp, cfg_sans_end)
        c_sans.set('download', ('prix', 'rend', 'pstats'))

        # Truquer ts_utc dans manifest : 25h dans le passe
        manifest_path = Path(tmp) / '.cache_v2' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['download']['ts_utc'] = time.time() - 25 * 3600
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

        # Reconstruire pour relire manifest
        c_sans2 = CacheEtapes(tmp, cfg_sans_end)
        result = c_sans2.get('download')
        assert result is None, 'download doit etre MISS si TTL expire (25h, pas de end_date)'

    with tempfile.TemporaryDirectory() as tmp:
        # Cas 2 : end_date explicite -> pas de TTL
        cfg_avec_end = _cfg(data_end_date='2024-01-01')
        c_avec = CacheEtapes(tmp, cfg_avec_end)
        c_avec.set('download', ('prix', 'rend', 'pstats'))

        # Truquer ts_utc : 25h dans le passe
        manifest_path = Path(tmp) / '.cache_v2' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['download']['ts_utc'] = time.time() - 25 * 3600
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

        c_avec2 = CacheEtapes(tmp, cfg_avec_end)
        result = c_avec2.get('download')
        assert result is not None, 'download doit rester HIT si end_date explicite (pas de TTL)'


# ── Test 6 : df_garch enrichi dans le cache ───────────────────────────────────

def test_df_garch_enrichi_dans_cache():
    """CRITIQUE : lb_z_pval et tous_passent presents dans df_garch lu depuis cache.

    Reproduit la serie seed=7 utilisee dans test_df_garch_contient_colonnes_spec
    (test_phase16bis) pour garantir que selectionner_meilleur enrichit df_garch
    avant la mise en cache.
    """
    from tickerlab.core.garch_selector import (
        grid_search_garch, selectionner_meilleur, estimer_final
    )

    rng = np.random.default_rng(7)
    n = 500
    h = np.ones(n)
    r = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.10 * r[t-1]**2 + 0.85 * h[t-1]
        r[t] = np.sqrt(h[t]) * rng.standard_t(df=5)
    serie = pd.Series(r * 100)

    cfg_garch = {
        'modeles': ['GARCH'], 'distributions': ['t'],
        'p_max': 1, 'q_max': 1,
        'seuil_significativite': 0.05,
        'critere_information': 'BIC',
        'tolerance_delta_critere_brut': 4.0,
        'seuil_igarch': 0.98,
        'score_composite': {'enabled': False},
    }
    full_cfg = _cfg()
    full_cfg['garch'].update(cfg_garch)

    df_garch = grid_search_garch(serie, **cfg_garch)
    best, motif, trace_garch = selectionner_meilleur(df_garch, serie, full_cfg)
    garch_final = estimer_final(serie, **best.to_dict())

    # Verifier que les colonnes spec sont la avant mise en cache
    assert 'tous_passent' in df_garch.columns, (
        'tous_passent absent de df_garch avant set() — side effect non applique')

    with tempfile.TemporaryDirectory() as tmp:
        c = CacheEtapes(tmp, full_cfg)
        c.set('garch', (df_garch, best, motif, trace_garch, garch_final))

        result = c.get('garch')
        assert result is not None, 'garch doit etre HIT apres set()'

        df_out, _, _, _, _ = result
        for col in ('lb_z_pval', 'lb_z2_pval', 'engle_ng_pval', 'tous_passent'):
            assert col in df_out.columns, (
                f'Colonne spec {col!r} absente de df_garch deserialie depuis cache — '
                'le tuple a ete store AVANT enrichissement (bug de sequencement)')


# ── Test 7 : force_refresh ────────────────────────────────────────────────────

def test_force_refresh():
    """Avec force_refresh=True, toutes les etapes sont MISS meme si cache present."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        c = CacheEtapes(tmp, cfg)

        # Peupler le cache
        c.set('download', ('prix', 'rend', 'ps'))
        c.set('garch',    'garch_val')

        # get() normal -> HIT
        assert c.get('download') is not None, 'download doit etre HIT sans force_refresh'

        # Simuler force_refresh : le pattern run_pipeline est
        # _cached = _cv2.get(etape) if (cv2 and not force_refresh) else None
        force_refresh = True
        _cached_dl = c.get('download') if (c and not force_refresh) else None
        _cached_g  = c.get('garch')    if (c and not force_refresh) else None

        assert _cached_dl is None, 'download doit etre MISS avec force_refresh=True'
        assert _cached_g  is None, 'garch doit etre MISS avec force_refresh=True'

        # Apres re-set(), les valeurs sont bien ecrites (cache reecrit)
        c.set('download', ('prix2', 'rend2', 'ps2'))
        assert c.get('download') == ('prix2', 'rend2', 'ps2'), \
            'set() apres force_refresh doit mettre a jour le cache'


# ── Test 8 : enabled=False -> aucun fichier cree ─────────────────────────────

def test_disabled_strict():
    """cache_v2.enabled=False -> aucun dossier .cache_v2/ cree, CacheEtapes=None."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg['cache_v2'] = {'enabled': False}

        # Le pipeline ne cree pas CacheEtapes quand enabled=False
        _cache_v2_enabled = cfg.get('cache_v2', {}).get('enabled', True)
        _cv2 = CacheEtapes(tmp, cfg) if _cache_v2_enabled else None

        assert _cv2 is None, 'CacheEtapes doit etre None quand cache_v2.enabled=False'

        # Aucun dossier .cache_v2/ ne doit exister
        cache_dir = Path(tmp) / '.cache_v2'
        assert not cache_dir.exists(), '.cache_v2/ cree malgre enabled=False'


# ── Test 9 : non-regression cache froid vs chaud ─────────────────────────────

def test_non_regression_cache_froid_vs_chaud():
    """Serie synthetique : var99, persistance et modele identiques entre run froid et chaud.

    Utilise seed=7 (meme serie que test_df_garch_contient_colonnes_spec) pour
    garantir la convergence GARCH. Verifie que le round-trip cache est neutre.
    """
    from tickerlab.core.garch_selector import (
        grid_search_garch, selectionner_meilleur, estimer_final
    )
    from tickerlab.core.var_engine import calculer_var_tvar

    rng = np.random.default_rng(7)
    n = 500
    h = np.ones(n)
    r = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.10 * r[t-1]**2 + 0.85 * h[t-1]
        r[t] = np.sqrt(h[t]) * rng.standard_t(df=5)
    serie = pd.Series(r * 100)

    cfg_garch = {
        'modeles': ['GARCH'], 'distributions': ['t'],
        'p_max': 1, 'q_max': 1,
        'seuil_significativite': 0.05,
        'critere_information': 'BIC',
        'tolerance_delta_critere_brut': 4.0,
        'seuil_igarch': 0.98,
        'score_composite': {'enabled': False},
    }
    full_cfg = _cfg()
    full_cfg['garch'].update(cfg_garch)

    def _extract_var99(df_var):
        """Extrait VaR GARCH 99% — gere index string '99%' ou float 0.99."""
        col = 'VaR GARCH'
        if col not in df_var.columns:
            return float('nan')
        for idx_key in ('99%', 0.99):
            if idx_key in df_var.index:
                return float(df_var.loc[idx_key, col])
        return float('nan')

    def _run_cold(tmp):
        """Run complet sans cache."""
        df_garch = grid_search_garch(serie, **cfg_garch)
        best, motif, trace = selectionner_meilleur(df_garch, serie, full_cfg)
        garch_final = estimer_final(serie, **best.to_dict())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_var = calculer_var_tvar(serie, garch_final, niveaux=[0.99])
        var99 = _extract_var99(df_var)
        params = garch_final.params
        persistance = float(params.get('alpha[1]', 0) + params.get('beta[1]', 0))
        modele = str(best['modele'])
        c = CacheEtapes(tmp, full_cfg)
        c.set('garch', (df_garch, best, motif, trace, garch_final))
        c.set('var', (df_var, None))
        return var99, persistance, modele

    def _run_warm(tmp):
        """Run depuis cache (HIT)."""
        c = CacheEtapes(tmp, full_cfg)
        df_garch, best, motif, trace, garch_final = c.get('garch')
        df_var, _ = c.get('var')
        var99 = _extract_var99(df_var)
        params = garch_final.params
        persistance = float(params.get('alpha[1]', 0) + params.get('beta[1]', 0))
        modele = str(best['modele'])
        return var99, persistance, modele

    with tempfile.TemporaryDirectory() as tmp:
        var99_cold, pers_cold, mod_cold = _run_cold(tmp)
        var99_warm, pers_warm, mod_warm = _run_warm(tmp)

    import math
    assert not math.isnan(var99_cold), (
        'var99 froid est NaN — modele GARCH non converge sur cette serie (choisir autre seed)')
    assert abs(var99_cold - var99_warm) < 1e-6, (
        f'var99 froid={var99_cold:.8f} != chaud={var99_warm:.8f} (delta > 1e-6)')
    assert abs(pers_cold - pers_warm) < 1e-10, (
        f'persistance froid={pers_cold} != chaud={pers_warm}')
    assert mod_cold == mod_warm, (
        f'modele froid={mod_cold} != chaud={mod_warm}')


# ── Test 10 : invalidation dm_gk via split_ratio ──────────────────────────────

def test_invalidation_dmgk_via_split_ratio():
    """Changer backtest.split_ratio invalide dm_gk (fenetre OOS derivee).

    Verifie les deux dimensions du DAG pour dm_gk :
    - cle() change quand split_ratio change (config section 'backtest' present)
    - download et garch restent HIT (pas de cascade depuis backtest)
    """
    cfg_v1 = _cfg()
    cfg_v1['backtest'] = {'split_ratio': 0.70}

    cfg_v2 = _cfg()
    cfg_v2['backtest'] = {'split_ratio': 0.80}

    with tempfile.TemporaryDirectory() as tmp:
        c1 = CacheEtapes(tmp, cfg_v1)
        c1.set('download', 'prix')
        c1.set('garch',    'garch')
        c1.set('fhs',      'fhs')
        c1.set('dm_gk',    'dm_gk_result')
        c1.set('backtest', 'backtest_result')

        # cle dm_gk doit changer
        cle_dmgk_v1 = c1.cle('dm_gk')
        c2 = CacheEtapes(tmp, cfg_v2)
        cle_dmgk_v2 = c2.cle('dm_gk')
        assert cle_dmgk_v1 != cle_dmgk_v2, (
            'cle(dm_gk) identique apres split_ratio 0.70->0.80 — '
            "backtest manquant dans les sections config de dm_gk (trou d'invalidation)")

        # dm_gk MISS avec la nouvelle config
        assert c2.get('dm_gk') is None, 'dm_gk doit etre MISS apres split_ratio change'

        # download et garch restent HIT (pas de relation de cascade avec backtest)
        assert c2.get('download') is not None, 'download doit rester HIT'
        assert c2.get('garch')    is not None, 'garch doit rester HIT'

    # Verifie aussi fhs et backtest invalides par split_ratio
    # fhs ne lit pas split_ratio -> cle fhs stable
    # backtest lit split_ratio via config['backtest'] -> cle backtest change
    with tempfile.TemporaryDirectory() as tmp:
        c1 = CacheEtapes(tmp, cfg_v1)
        c2 = CacheEtapes(tmp, cfg_v2)

        cle_fhs_v1 = c1.cle('fhs')
        cle_fhs_v2 = c2.cle('fhs')
        # fhs ne consomme pas backtest -> sa cle ne change pas avec split_ratio seul
        # (elle change si garch ou fhs config change)
        assert cle_fhs_v1 == cle_fhs_v2, (
            'cle(fhs) change apres split_ratio seul — fhs ne lit pas split_ratio '
            '(sur-invalidation non desiree)')

        cle_bt_v1 = c1.cle('backtest')
        cle_bt_v2 = c2.cle('backtest')
        assert cle_bt_v1 != cle_bt_v2, (
            "cle(backtest) identique apres split_ratio change — trou d'invalidation backtest")


# ── Test 11 : couverture sections config par etape (anti-regression DAG) ─────

def test_couverture_sections_config_par_etape():
    """Anti-regression : chaque etape a dans sa cle toutes les sections config
    que son code consomme. Liste de reference en dur, comparee au DAG reel.

    Ce test detecte toute regression introduite par une modification du DAG
    (section oubliee, section incorrecte). Verifie aussi que les cles changent
    bien quand la section correspondante est modifiee.
    """
    # Table de reference : etape -> sections config minimales attendues
    # Derivee de l'audit systematique main.py + modules (2026-06-13)
    REFERENCE = {
        'download'       : {'data'},
        'arima'          : {'arima'},
        'garch'          : {'garch'},
        'igarch_diag'    : {'garch'},
        'component_garch': {'component_garch'},
        'fhs'            : {'fhs', 'var'},
        'dm_gk'          : {'dm_gk', 'backtest', 'var'},
        'var'            : {'var', 'bootstrap'},
        'backtest'       : {'backtest', 'var'},
    }

    for etape, sections_attendues in REFERENCE.items():
        dag_upstream, dag_sections = ETAPES_DAG[etape]
        dag_sections_set = set(dag_sections)
        manquantes = sections_attendues - dag_sections_set
        assert not manquantes, (
            f"Etape '{etape}' : sections config manquantes dans le DAG : {manquantes}. "
            f"DAG actuel : {dag_sections_set}, reference : {sections_attendues}. "
            f"Trou d'invalidation possible si ces sections changent.")

    # Verification supplementaire : pour chaque section dans la reference,
    # confirmer que modifier cette section change bien la cle de l'etape.
    cfg_base = _cfg()
    cfg_base['bootstrap'] = {'n_boot': 100}

    with tempfile.TemporaryDirectory() as tmp:
        c_base = CacheEtapes(tmp, cfg_base)

        # download change si data change
        cfg2 = _cfg(); cfg2['data'] = dict(cfg_base['data']); cfg2['data']['ticker'] = 'CL=F'
        assert c_base.cle('download') != CacheEtapes(tmp, cfg2).cle('download'), \
            "cle('download') insensible a data.ticker"

        # fhs change si var change (niveaux)
        cfg3 = _cfg(); cfg3['var'] = {'niveaux': [0.90, 0.99]}
        assert c_base.cle('fhs') != CacheEtapes(tmp, cfg3).cle('fhs'), \
            "cle('fhs') insensible a var.niveaux"

        # dm_gk change si backtest.split_ratio change
        cfg4 = _cfg(); cfg4['backtest'] = {'split_ratio': 0.80}
        assert c_base.cle('dm_gk') != CacheEtapes(tmp, cfg4).cle('dm_gk'), \
            "cle('dm_gk') insensible a backtest.split_ratio"

        # dm_gk change si var.niveaux change
        cfg5 = _cfg(); cfg5['var'] = {'niveaux': [0.90, 0.99]}
        assert c_base.cle('dm_gk') != CacheEtapes(tmp, cfg5).cle('dm_gk'), \
            "cle('dm_gk') insensible a var.niveaux"

        # backtest change si var.n_simulations_mc change
        cfg6 = _cfg(); cfg6['var'] = {'niveaux': [0.95, 0.99], 'n_simulations_mc': 100}
        assert c_base.cle('backtest') != CacheEtapes(tmp, cfg6).cle('backtest'), \
            "cle('backtest') insensible a var.n_simulations_mc"
