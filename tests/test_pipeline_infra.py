# -*- coding: utf-8 -*-
"""
Tests — API industrialisee.

8 tests unitaires, aucun appel reseau ni GARCH reel.

Test 1 : hierarchie exceptions            -> 5 classes, champs ticker/etape/contexte
Test 2 : config_hash cles scientifiques   -> hash stable, exclut output/cache/ai
Test 3 : valider_config OK                -> aucune exception
Test 4 : valider_config raise             -> ValidationError si window >= n_obs
Test 5 : PipelineResult dict-like         -> [] get keys in, champ meta present
Test 6 : PipelineResult retrocompat       -> acces comme dict
Test 7 : setup_tickerlab_logger idempotent -> 1 handler apres 2 appels, niveau ajuste
Test 8 : run_multi_ticker 35 colonnes     -> _COLONNES_CSV contient config_hash
"""
import sys
import warnings
import logging
from pathlib import Path

import pytest

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.exceptions import (
    PipelineError, DataError, ModelError, ValidationError, ExportError,
)
from tickerlab.core.config_validation import config_hash, valider_config
from tickerlab.core.pipeline_result import PipelineResult
from tickerlab.core._logging import setup_tickerlab_logger


# ── Fixtures ──────────────────────────────────────────────────────────────────

_CFG_BASE = {
    'data':    {'ticker': 'BZ=F', 'start_date': '2015-01-01', 'end_date': '2024-01-01',
                'frequency': 'daily'},
    'arima':   {'p_max': 3, 'q_max': 3},
    'garch':   {'p_max': 1, 'q_max': 1},
    'var':     {'niveaux': [0.95, 0.99]},
    'backtest': {'split_ratio': 0.70},
    'output':  {'dossier_resultats': '/tmp/test'},
    'cache':   {'enabled': False},
    'ai':      {'enabled': False},
    '_metrics_only': True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Hiérarchie exceptions
# ─────────────────────────────────────────────────────────────────────────────

def test_hierarchie_exceptions():
    """5 classes, champs corrects, __str__ inclut ticker et etape."""
    assert issubclass(DataError,       PipelineError)
    assert issubclass(ModelError,      PipelineError)
    assert issubclass(ValidationError, PipelineError)
    assert issubclass(ExportError,     PipelineError)

    err = ModelError('convergence GARCH echouee', ticker='BZ=F',
                     etape='garch', contexte={'p': 1, 'q': 1})
    assert err.ticker == 'BZ=F'
    assert err.etape  == 'garch'
    assert err.contexte == {'p': 1, 'q': 1}
    assert 'BZ=F' in str(err)
    assert 'garch' in str(err)

    # PipelineError sans champs optionnels
    base = PipelineError('message simple')
    assert base.ticker  is None
    assert base.etape   is None
    assert base.contexte == {}
    assert str(base) == 'message simple'


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — config_hash : clés scientifiques seulement
# ─────────────────────────────────────────────────────────────────────────────

def test_config_hash_cles_scientifiques():
    """Hash stable sur les cles scientifiques ; exclut output/cache/ai/_metrics_only."""
    cfg1 = dict(_CFG_BASE)
    cfg2 = dict(_CFG_BASE)

    # Modifier output/cache/ai/_metrics_only -> hash identique
    cfg2['output'] = {'dossier_resultats': '/autre/dossier'}
    cfg2['cache']  = {'enabled': True}
    cfg2['ai']     = {'enabled': True}
    cfg2['_metrics_only'] = False
    assert config_hash(cfg1) == config_hash(cfg2), \
        'Hash doit ignorer output/cache/ai/_metrics_only'

    # Modifier une cle scientifique (garch) -> hash different
    cfg3 = dict(_CFG_BASE)
    cfg3['garch'] = {'p_max': 2, 'q_max': 2}
    assert config_hash(cfg1) != config_hash(cfg3), \
        'Hash doit changer si parametres GARCH changent'

    # Modifier component_garch -> hash different (correctif 3.1)
    cfg4 = dict(_CFG_BASE)
    cfg4['component_garch'] = {'seuil_persistance': 0.95}
    assert config_hash(cfg1) != config_hash(cfg4), \
        'Hash doit changer si component_garch change (correctif 3.1)'

    # Modifier stress_testing -> hash different (correctif 3.1)
    cfg5 = dict(_CFG_BASE)
    cfg5['stress_testing'] = {'enabled': True, 'scenarios': ['covid']}
    assert config_hash(cfg1) != config_hash(cfg5), \
        'Hash doit changer si stress_testing change (correctif 3.1)'

    # Format 12 hex
    h = config_hash(cfg1)
    assert len(h) == 12
    assert all(c in '0123456789abcdef' for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — valider_config OK (aucune exception)
# ─────────────────────────────────────────────────────────────────────────────

def test_valider_config_ok():
    """Configuration valide : aucune exception, aucun warning critique."""
    cfg = dict(_CFG_BASE)
    cfg['backtest'] = {'split_ratio': 0.70}
    cfg['rolling_backtest'] = {'enabled': False}
    # n_obs = 500, split_ratio=0.70 -> T_oos = 150 >= 50 : OK
    valider_config(cfg, n_obs=500)   # ne doit pas lever


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — valider_config lève ValidationError
# ─────────────────────────────────────────────────────────────────────────────

def test_valider_config_raise_validation_error():
    """window_size >= n_obs -> ValidationError typee avec champs."""
    cfg = dict(_CFG_BASE)
    cfg['rolling_backtest'] = {'enabled': True, 'window_size': 600}
    with pytest.raises(ValidationError) as exc_info:
        valider_config(cfg, n_obs=500)
    err = exc_info.value
    assert err.etape == 'config_validation'
    assert err.contexte['window_size'] == 600
    assert err.contexte['n_obs'] == 500
    assert issubclass(type(err), PipelineError)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — PipelineResult : interface dict-like
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_result_dict_like():
    """PipelineResult expose [], get, keys, in, meta."""
    import pandas as pd

    pr = PipelineResult(
        prix=None,
        rendements=pd.Series([1.0, 2.0]),
        T_train=100,
        meta={'ticker': 'BZ=F', 'version': '6.3.1', 'config_hash': 'abc123'},
    )
    # Acces par cle
    assert pr['T_train'] == 100
    assert isinstance(pr['rendements'], pd.Series)
    assert pr['meta']['version'] == '6.3.1'
    assert pr['meta']['config_hash'] == 'abc123'

    # get() avec default
    assert pr.get('T_train') == 100
    assert pr.get('inexistant', 'fallback') == 'fallback'

    # keys()
    ks = pr.keys()
    assert 'prix' in ks
    assert 'meta' in ks
    assert 'T_train' in ks

    # __contains__
    assert 'T_train' in pr
    assert 'inexistant' not in pr

    # KeyError sur cle absente
    with pytest.raises(KeyError):
        _ = pr['inexistant']


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — PipelineResult : rétrocompatibilité dict
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_result_retrocompat():
    """Les 17 cles sont toutes accessibles via [] comme un dict."""
    import pandas as pd

    _fake_df = pd.DataFrame()
    pr = PipelineResult(
        prix=_fake_df, rendements=_fake_df,
        arima={'p_opt': 1, 'd_opt': 0, 'q_opt': 1},
        garch_best=None, garch_final=None,
        var=_fake_df, backtest=_fake_df, vol=_fake_df,
        igarch_diag={}, component_garch=None, fhs=None, dm_gk=None,
        T_train=700, T_eff_dyn=300, df_garch=_fake_df,
        spec_verdict={}, motif_selection='AIC',
        meta={'ticker': 'BZ=F'},
    )
    cles_phase2 = [
        'prix', 'rendements', 'arima', 'garch_best', 'garch_final',
        'var', 'backtest', 'vol', 'igarch_diag', 'component_garch',
        'fhs', 'dm_gk', 'T_train', 'T_eff_dyn', 'df_garch',
        'spec_verdict', 'motif_selection',
    ]
    for k in cles_phase2:
        assert k in pr, f'Cle manquante : {k}'
        _ = pr[k]   # ne doit pas lever


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — setup_tickerlab_logger : idempotence + niveau
# ─────────────────────────────────────────────────────────────────────────────

def test_setup_tickerlab_logger_idempotent():
    """Deux appels successifs : toujours 1 StreamHandler, niveau ajuste."""
    pkg_logger = logging.getLogger('tickerlab')

    # Reset handlers pour le test (isole)
    pkg_logger.handlers.clear()

    setup_tickerlab_logger(level=logging.INFO)
    n_handlers_after_1 = len(pkg_logger.handlers)
    assert n_handlers_after_1 == 1

    setup_tickerlab_logger(level=logging.WARNING)
    n_handlers_after_2 = len(pkg_logger.handlers)
    assert n_handlers_after_2 == 1, \
        f'Un seul handler attendu, {n_handlers_after_2} trouves'
    assert pkg_logger.level == logging.WARNING

    setup_tickerlab_logger(level=logging.INFO)
    assert len(pkg_logger.handlers) == 1
    assert pkg_logger.level == logging.INFO

    assert not pkg_logger.propagate


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — runner : 35 colonnes CSV
# ─────────────────────────────────────────────────────────────────────────────

def test_runner_35_colonnes():
    """_COLONNES_CSV doit contenir exactement 35 colonnes dont config_hash."""
    from scripts.run_multi_ticker import _COLONNES_CSV
    assert len(_COLONNES_CSV) == 35, \
        f'{len(_COLONNES_CSV)} colonnes trouvees, 35 attendues'
    assert 'config_hash' in _COLONNES_CSV
    # Colonnes critiques toujours presentes
    for col in ('ticker', 'var99_garch', 'var99_garch_floored',
                 'garch_modele', 'persistance', 'erreur'):
        assert col in _COLONNES_CSV, f'Colonne manquante : {col}'
