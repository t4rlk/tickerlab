# -*- coding: utf-8 -*-
"""
Tests — Intégrité : déterminisme prouvé + atomicité cache.

Tests :
  B1 : FHS VaR reproductible — run1 == run2 (np.array_equal / ==)
  B2 : Bootstrap CI reproductible — run1 == run2 (DataFrame.equals)
  B3 : Warning émis quand bootstrap.seed=null
  A1 : Cache atomicité — interruption pickle.dump => .pkl final intact/absent
  A2 : Cache set() réussi => aucun .pkl.tmp résiduel
  A3 : Cache get() supprime un .pkl.tmp résiduel silencieusement
"""
import io
import logging
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.cache_v2 import CacheEtapes
from tickerlab.core.fhs import calculer_var_fhs
from tickerlab.core.var_engine import calculer_bootstrap_ci_var


# ── Fixture GARCH partagée (module scope pour ne fitter qu'une fois) ──────────

_garch_cache: dict = {}


def _get_garch_fit():
    """Retourne (garch_final, rendements) — fitte une seule fois par session."""
    if 'fit' not in _garch_cache:
        from arch import arch_model
        rng     = np.random.default_rng(99)
        returns = pd.Series(rng.standard_normal(400) * 0.01)
        fit     = arch_model(returns, vol='Garch', p=1, q=1, dist='t').fit(disp='off')
        _garch_cache['fit']     = fit
        _garch_cache['returns'] = returns
    return _garch_cache['fit'], _garch_cache['returns']


def _fhs_cfg():
    return {
        'fhs': {
            'enabled': True,
            'n_boot':  500,
            'horizons': [1, 5],
            'seed':    42,
        },
        'var': {'niveaux': [0.95, 0.99]},
    }


def _boot_cfg(seed=42):
    return {
        'bootstrap': {
            'express': True,
            'seed':    seed,
        }
    }


def _minimal_cache_config():
    return {'data': {}, 'garch': {}, 'var': {}, 'bootstrap': {}}


# ── B1 : FHS reproductible ────────────────────────────────────────────────────

def test_fhs_reproductible():
    """B1 : deux runs FHS avec seed=42 => VaR et ES identiques (np.array_equal)."""
    garch_final, rendements = _get_garch_fit()
    cfg = _fhs_cfg()
    r1  = calculer_var_fhs(rendements, garch_final, cfg)
    r2  = calculer_var_fhs(rendements, garch_final, cfg)
    assert r1 is not None and r2 is not None, 'FHS a retourne None'
    for H in cfg['fhs']['horizons']:
        for alp in [0.95, 0.99]:
            v1 = r1['var_fhs_par_horizon'][H][alp]
            v2 = r2['var_fhs_par_horizon'][H][alp]
            assert v1 == v2, (
                f'FHS VaR non reproductible H={H} alpha={alp}: {v1} != {v2}'
            )
            e1 = r1['es_fhs_par_horizon'][H][alp]
            e2 = r2['es_fhs_par_horizon'][H][alp]
            assert e1 == e2, (
                f'FHS ES non reproductible H={H} alpha={alp}: {e1} != {e2}'
            )


# ── B2 : Bootstrap CI reproductible ──────────────────────────────────────────

def test_bootstrap_ci_reproductible():
    """B2 : deux runs bootstrap CI avec seed=42 => DataFrame.equals (bit-exact)."""
    garch_final, rendements = _get_garch_fit()
    cfg = _boot_cfg(seed=42)
    df1 = calculer_bootstrap_ci_var(rendements, garch_final, cfg)
    df2 = calculer_bootstrap_ci_var(rendements, garch_final, cfg)
    assert not df1.empty, 'Bootstrap CI a retourne un DataFrame vide'
    assert df1.equals(df2), (
        f'Bootstrap CI non reproductible :\nrun1=\n{df1}\nrun2=\n{df2}'
    )


# ── B3 : Warning si bootstrap.seed=null ──────────────────────────────────────

def test_bootstrap_warning_si_seed_null():
    """B3 : calculer_bootstrap_ci_var emet un WARNING si bootstrap.seed=null.

    Utilise un handler StreamHandler sur le logger directement pour rester
    robuste quelle que soit la configuration logging du reste de la suite.
    """
    garch_final, rendements = _get_garch_fit()
    cfg = _boot_cfg(seed=None)

    buf     = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    logger  = logging.getLogger('tickerlab.core.var_engine')
    logger.addHandler(handler)
    saved_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        calculer_bootstrap_ci_var(rendements, garch_final, cfg)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved_level)

    output = buf.getvalue()
    assert 'NON reproductible' in output, (
        f'Warning attendu absent. Sortie log capturee : {repr(output)}'
    )


# ── A1 : Atomicité — interruption => pas de .pkl corrompu ────────────────────

def test_cache_atomique_interruption():
    """A1 : pickle.dump qui leve => .pkl final absent, .pkl.tmp nettoyé."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg   = _minimal_cache_config()
        cache = CacheEtapes(tmp_dir, cfg)

        with patch('tickerlab.core.cache_v2.pickle.dump',
                   side_effect=OSError('disque simule plein')):
            with pytest.raises(OSError):
                cache.set('var', {'test': 42})

        cache_dir = Path(tmp_dir) / '.cache_v2'

        # .pkl final : absent (aucune écriture partielle)
        pkls = list(cache_dir.glob('var_*.pkl'))
        assert not pkls, f'Fichier pickle corrompu present apres echec : {pkls}'

        # .pkl.tmp : nettoyé par le bloc except
        tmps = list(cache_dir.glob('var_*.pkl.tmp'))
        assert not tmps, f'.pkl.tmp residuel apres echec : {tmps}'

        # Manifest : inchangé (pas de mise à jour après dump raté)
        assert 'var' not in cache._manifest, (
            'Manifest mis a jour alors que le dump a echoue'
        )


# ── A2 : Set réussi => pas de .pkl.tmp résiduel ──────────────────────────────

def test_cache_set_pas_de_tmp_residuel():
    """A2 : set() réussi => aucun .pkl.tmp ne subsiste dans le répertoire."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg   = _minimal_cache_config()
        cache = CacheEtapes(tmp_dir, cfg)
        cache.set('var', {'valeur': 123})

        cache_dir = Path(tmp_dir) / '.cache_v2'
        tmps      = list(cache_dir.glob('*.pkl.tmp'))
        assert not tmps, f'.pkl.tmp residuel apres set() reussi : {tmps}'

        # Le .pkl final doit exister
        pkls = list(cache_dir.glob('var_*.pkl'))
        assert pkls, 'Aucun .pkl cree apres set() reussi'

        # get() doit retrouver la valeur
        val = cache.get('var')
        assert val == {'valeur': 123}, f'Valeur incorrecte : {val}'


# ── A3 : get() supprime un .pkl.tmp résiduel ─────────────────────────────────

def test_cache_get_supprime_tmp_residuel():
    """A3 : get() supprime silencieusement un .pkl.tmp laissé par un crash précédent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg   = _minimal_cache_config()
        cache = CacheEtapes(tmp_dir, cfg)

        cache_dir = Path(tmp_dir) / '.cache_v2'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Simuler un .pkl.tmp résiduel (crash d'un run précédent)
        cle12 = cache.cle('var')
        tmp_path = cache_dir / f'var_{cle12}.pkl.tmp'
        tmp_path.write_bytes(b'fichier corrompu simule')
        assert tmp_path.exists(), 'Setup du test incorrect'

        # get() doit ignorer/supprimer le .pkl.tmp et retourner None (MISS)
        result = cache.get('var')
        assert result is None, f'get() doit retourner None (MISS) : {result}'
        assert not tmp_path.exists(), '.pkl.tmp residuel non supprime par get()'
