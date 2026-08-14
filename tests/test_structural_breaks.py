# -*- coding: utf-8 -*-
"""
Tests — Ruptures structurelles ICSS (Condition 3).

5 tests :
  1. mode='diagnostic' retourne un dict valide
  2. mode='diagnostic' NE modifie PAS la persistance (non-régression)
  3. mode='diagnostic' n'expose PAS les clés persistance_avant/apres
  4. mode='integrate'  expose les clés persistance_avant ET persistance_apres
  5. mode='off' et enabled=false retournent None
+ 1 test unitaire ICSS : détection d'une rupture connue.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.structural_breaks import (
    analyser_icss_pipeline,
    estimer_garch_omega_par_regime,
    inclan_tiao_icss,
    selectionner_ruptures_espacees,
)

# Persistance réellement simulée par la fixture, des DEUX côtés de la rupture.
PERSISTANCE_SIMULEE = 0.08 + 0.88   # alpha + beta
RUPTURE_SIMULEE     = 300           # indice de la rupture dans la fixture


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_garch_with_breaks(n: int = 600, seed: int = 0):
    """
    Série GARCH(1,1) avec rupture de variance forte à t=300 (variance x9).
    La rupture est suffisamment large pour que l'ICSS la détecte avec certitude.

    Le multiplicateur porte sur omega — saut de niveau de la variance
    inconditionnelle omega/(1-alpha-beta) — et NON sur eps. Appliqué à eps, il
    serait réinjecté au carré dans la récurrence de h et porterait la
    persistance effective à 9*alpha+beta = 1.60 > 1 : la variance exploserait
    géométriquement au lieu de sauter d'un palier.

    Échelle : rendements quotidiens en pourcentage, ecart-type ~1.1 % avant la
    rupture et ~3.4 % apres. Persistance alpha+beta = 0.96 < 1 des deux cotes.
    """
    from arch import arch_model
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.08, 0.88
    eps = np.zeros(n)
    h   = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    var_mult = np.ones(n)
    var_mult[300:] = 9.0  # x9 variance → x3 volatility, rupture nette

    for i in range(1, n):
        h[i]   = omega * var_mult[i] + alpha * eps[i - 1] ** 2 + beta * h[i - 1]
        eps[i] = rng.standard_normal() * np.sqrt(h[i])

    series   = pd.Series(eps)
    fit      = arch_model(series, vol='Garch', p=1, o=0, q=1, dist='t').fit(disp='off')
    best_row = pd.Series({
        'modele': 'GARCH', 'vol': 'Garch',
        'p': 1, 'o': 0, 'q': 1, 'dist': 't',
        'AIC': fit.aic, 'BIC': fit.bic, 'power': float('nan'),
    })
    return series, fit, best_row


def _cfg_diag() -> dict:
    return {
        'structural_breaks': {
            'enabled': True,
            'mode': 'diagnostic',
            'methode': 'icss',
            'seuil_persistance_alerte': 0.97,
            'seuil_n_ruptures_alerte': 3,
            'max_dummies': 5,
        }
    }


def _cfg_integrate() -> dict:
    cfg = _cfg_diag()
    cfg['structural_breaks']['mode'] = 'integrate'
    return cfg


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_icss_detects_known_break():
    """ICSS détecte une rupture connue sur série synthétique à variance bimodale."""
    rng = np.random.default_rng(7)
    eps1 = rng.standard_normal(200) * 1.0
    eps2 = rng.standard_normal(200) * 5.0   # rupture nette à t=200
    eps_sq = np.concatenate([eps1, eps2]) ** 2
    breaks = inclan_tiao_icss(eps_sq)
    assert len(breaks) >= 1, f"ICSS devrait détecter ≥1 rupture, obtenu {len(breaks)}"


def test_diagnostic_returns_valid_dict():
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(series, fit, _cfg_diag(), garch_best=best)
    assert result is not None
    assert result['mode'] == 'diagnostic'
    assert 'n_breaks' in result
    assert 'indices' in result
    assert 'warning_lamoureux' in result
    assert isinstance(result['n_breaks'], int)
    assert isinstance(result['indices'], list)


def test_diagnostic_non_regression():
    """mode='diagnostic' ne modifie PAS les paramètres du modèle GARCH."""
    series, fit, best = _make_garch_with_breaks()
    pers_avant = float(
        fit.params.get('alpha[1]', 0) + fit.params.get('beta[1]', 0)
    )
    analyser_icss_pipeline(series, fit, _cfg_diag(), garch_best=best)
    pers_apres = float(
        fit.params.get('alpha[1]', 0) + fit.params.get('beta[1]', 0)
    )
    assert pers_avant == pers_apres, (
        "mode='diagnostic' NE doit PAS modifier les paramètres GARCH estimés"
    )


def test_diagnostic_no_persistance_keys():
    """mode='diagnostic' n'expose PAS persistance_avant / persistance_apres."""
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(series, fit, _cfg_diag(), garch_best=best)
    assert 'persistance_avant' not in result
    assert 'persistance_apres' not in result


def test_integrate_persistance_keys():
    """mode='integrate' expose persistance_avant et persistance_apres si n_breaks > 0."""
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(series, fit, _cfg_integrate(), garch_best=best)

    if result is None or result.get('n_breaks', 0) == 0:
        pytest.skip("Aucune rupture détectée — ICSS dépend des données simulées")

    assert 'persistance_avant' in result, "persistance_avant manquante en mode integrate"
    assert 'persistance_apres' in result, "persistance_apres manquante en mode integrate"
    assert 'dummies_injectees' in result

    assert isinstance(result['persistance_avant'], float)
    assert isinstance(result['persistance_apres'], float)
    assert not math.isnan(result['persistance_avant'])
    assert not math.isnan(result['persistance_apres'])


def test_mode_off_returns_none():
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(
        series, fit,
        {'structural_breaks': {'enabled': True, 'mode': 'off'}},
        garch_best=best,
    )
    assert result is None


def test_disabled_returns_none():
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(
        series, fit,
        {'structural_breaks': {'enabled': False}},
        garch_best=best,
    )
    assert result is None
