# -*- coding: utf-8 -*-
"""Tests — backtest en fenêtre glissante + garde-fou de configuration runtime.

backtest_rolling_var : fenêtre impossible → ValueError ; fenêtre trop proche de
n_total → auto-ajustement + UserWarning. config_validation.valider_config : un
split_ratio laissant moins de 50 observations hors-échantillon → UserWarning.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _serie_courte(n=300, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, n) * 0.5,
                     index=pd.bdate_range('2020-01-02', periods=n))


def _best_garch():
    return {'vol': 'Garch', 'p': 1, 'o': 0, 'q': 1, 'dist': 'normal',
            'modele': 'GARCH'}


def _cfg_rolling(window=1000, enabled=True):
    return {
        'rolling_backtest': {
            'enabled':    enabled,
            'window_size': window,
            'refit_every': 22,
            'niveaux_test': [0.95, 0.99],
        }
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rolling_window_impossible_raises():
    """window_size >= n_total -> ValueError."""
    from tickerlab.core.backtest_rolling import backtest_rolling_var
    serie = _serie_courte(n=300)
    with pytest.raises(ValueError, match='rolling_backtest impossible'):
        backtest_rolling_var(serie, _best_garch(), _cfg_rolling(window=350))


def test_rolling_window_auto_ajustement():
    """window trop proche de n_total -> UserWarning + auto-ajustement (pas d'erreur)."""
    from tickerlab.core.backtest_rolling import backtest_rolling_var
    serie = _serie_courte(n=400)
    cfg   = _cfg_rolling(window=370)  # 400 - 370 = 30 < min_oos=50 -> auto-ajustement
    with pytest.warns(UserWarning, match='Auto-ajustement'):
        df_viol, df_params, stats = backtest_rolling_var(serie, _best_garch(), cfg)
    assert isinstance(df_viol, pd.DataFrame), "df_violations doit etre un DataFrame"


def test_validation_config_runtime_split_trop_petit():
    """split_ratio laissant < 50 OOS -> UserWarning (via config_validation.valider_config)."""
    from tickerlab.core.config_validation import valider_config
    cfg = {'backtest': {'split_ratio': 0.99}, 'rolling_backtest': {'enabled': False}}
    with pytest.warns(UserWarning, match='split_ratio'):
        valider_config(cfg, n_obs=200)
