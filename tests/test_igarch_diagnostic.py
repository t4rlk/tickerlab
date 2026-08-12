# -*- coding: utf-8 -*-
"""Tests — Diagnostic IGARCH post-hoc.

Couvre :
- diagnostiquer_igarch() : 3 codes + demi-vie + Wald
- Fallback param_cov=None : pas de crash, wald_stat=NaN
- Gradient analytique EGARCH : ∂pers/∂β=1, tous les autres=0
- Persistance ≥ 1 : code igarch_strict, half_life=None
- Rétrocompatibilité : best_dict.get('igarch_diagnostic', {}) ne crash pas
"""
import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tickerlab.core.igarch_diagnostic import (
    diagnostiquer_igarch,
    _gradient_analytique,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _MockGarchResult:
    """Mock minimal d'un ARCHModelResult pour les tests unitaires."""

    def __init__(self, params_dict: dict, cov_matrix=None):
        self.params = pd.Series(params_dict)
        if cov_matrix is not None:
            self.param_cov = pd.DataFrame(
                cov_matrix,
                index=self.params.index,
                columns=self.params.index,
            )
        else:
            self.param_cov = None


def _garch_params(alpha: float, beta: float) -> dict:
    return {'mu': 0.0, 'omega': 0.01, 'alpha[1]': alpha, 'beta[1]': beta}


# ── Tests ─────────────────────────────────────────────────────────────────────

# 1. mean_reverting — persistance < 0.98
def test_mean_reverting():
    """Persistance = 0.85 < 0.98 → mean_reverting."""
    mock = _MockGarchResult(_garch_params(alpha=0.10, beta=0.75))
    result = diagnostiquer_igarch(mock, 'GARCH', seuil_igarch=0.98)
    assert result['code'] == 'mean_reverting', result
    assert result['near_igarch'] is False
    assert result['half_life_periodes'] is not None
    assert result['half_life_periodes'] > 0
    # Demi-vie daily → half_life_jours_cal = periodes × (7/5)
    expected_jcal = result['half_life_periodes'] * (7 / 5)
    assert abs(result['half_life_jours_cal'] - expected_jcal) < 1e-6


# 2. near_igarch — 0.98 ≤ persistance < 1.0
def test_near_igarch():
    """Persistance = 0.99 ≥ 0.98 → near_igarch."""
    mock = _MockGarchResult(_garch_params(alpha=0.05, beta=0.94))
    result = diagnostiquer_igarch(mock, 'GARCH', seuil_igarch=0.98)
    assert result['code'] == 'near_igarch', result
    assert result['near_igarch'] is True
    assert result['half_life_periodes'] is not None
    assert result['half_life_jours_cal'] is not None
    assert 'Lamoureux' in result['message_pedagogique'] or 'near-IGARCH' in result['message_pedagogique']


# 3. igarch_strict — persistance ≥ 1.0
def test_igarch_strict_persistance_unite():
    """Persistance ≥ 1.0 → igarch_strict, half_life=None (sentinel JSON-safe)."""
    mock = _MockGarchResult(_garch_params(alpha=0.10, beta=0.90))
    # alpha + beta = 1.0 → IGARCH exact
    result = diagnostiquer_igarch(mock, 'GARCH', seuil_igarch=0.98)
    assert result['code'] == 'igarch_strict', result
    assert result['near_igarch'] is True
    assert result['half_life_periodes'] is None     # sentinel None, pas float('inf')
    assert result['half_life_jours_cal'] is None


# 4. Wald fallback param_cov=None — pas de crash, wald=NaN
def test_wald_fallback_param_cov_none():
    """param_cov=None → wald_stat=NaN, wald_pval=NaN, near_igarch calculé par seuil."""
    mock = _MockGarchResult(_garch_params(alpha=0.05, beta=0.94), cov_matrix=None)
    result = diagnostiquer_igarch(mock, 'GARCH', seuil_igarch=0.98)
    assert math.isnan(result['wald_stat']), result
    assert math.isnan(result['wald_pval']), result
    # near_igarch toujours calculé par seuil, pas de crash
    assert isinstance(result['near_igarch'], bool)
    assert result['code'] == 'near_igarch'


# 5. Gradient EGARCH — seuls les β_j contribuent (Nelson 1991)
def test_egarch_gradient_specifique():
    """Pour EGARCH, ∂persistance/∂β=1 et ∂/∂α=∂/∂γ=0 (Nelson 1991, eq. 10)."""
    params = pd.Series({
        'mu': 0.0,
        'omega': -0.50,
        'alpha[1]': 0.10,
        'gamma[1]': 0.05,
        'beta[1]': 0.95,
        'eta': 1.50,
        'lambda': -0.10,
    })
    g = _gradient_analytique(params, 'EGARCH')
    idx = list(params.index)

    assert g[idx.index('beta[1]')]  == 1.0, "beta doit contribuer"
    assert g[idx.index('alpha[1]')] == 0.0, "alpha ne contribue pas pour EGARCH"
    assert g[idx.index('gamma[1]')] == 0.0, "gamma ne contribue pas pour EGARCH"
    assert g[idx.index('mu')]       == 0.0
    assert g[idx.index('omega')]    == 0.0
    assert g[idx.index('eta')]      == 0.0
    assert g[idx.index('lambda')]   == 0.0


# 6. Wald stat disponible quand param_cov non-None
def test_wald_stat_calculable():
    """Avec param_cov non-NaN, wald_stat et wald_pval sont des scalaires finis."""
    n_params = 4  # mu, omega, alpha[1], beta[1]
    cov = np.eye(n_params) * 1e-4   # covariance diagonale simple
    params_dict = _garch_params(alpha=0.05, beta=0.94)
    mock = _MockGarchResult(params_dict, cov_matrix=cov)
    result = diagnostiquer_igarch(mock, 'GARCH', seuil_igarch=0.98)
    assert not math.isnan(result['wald_stat']), "wald_stat doit être calculable"
    assert not math.isnan(result['wald_pval']), "wald_pval doit être calculable"
    assert result['wald_stat'] >= 0


# 7. Rétrocompatibilité — best_dict.get('igarch_diagnostic', {}) ne crash pas
def test_retrocompatibilite_best_dict():
    """Un best_dict sans 'igarch_diagnostic' retourne {} sans erreur."""
    best_dict = {'modele': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'skewt'}
    diag = best_dict.get('igarch_diagnostic', {})
    # La box PDF doit accepter un dict vide sans crash
    import tickerlab.core.rapport._helpers as _H
    _H._set_theme('academique')
    from tickerlab.core.rapport._sections import _encadre_igarch
    flowables = _encadre_igarch(diag, {})
    assert flowables == []   # dict vide → aucun flowable
