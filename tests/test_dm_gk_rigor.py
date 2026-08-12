# -*- coding: utf-8 -*-
"""
Tests — Rigueur DM-GK.

Conditions testees :
  C1 : dm_test retourne degenerate=True si var1=var2
  C1 : dm_test retourne degenerate=False sur series distinctes
  C1 : _determine_verdict retourne 'non_discriminable' si dm_degenerate=True
  C1 : _determine_verdict retourne 'non_discriminable' si gk_degenerate=True
  C1 : 'non_discriminable' prime sur tous les autres verdicts
  C1 : 'non_discriminable' present dans VERDICTS_LABELS
  C1 : comparer_methodes_var retourne 'non_discriminable' pour series identiques
  C2 : gk_test retourne reliable=False, pval=nan, reject=False si Omega singuliere
  C2 : gk_test retourne reliable=True, pval finie si Omega bien conditionnee
  C4 : build_var_series contient le commentaire H=1
"""
import inspect
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.dm_gk import (
    build_var_series,
    comparer_methodes_var,
    dm_test,
    gk_test,
    _determine_verdict,
    VERDICTS_LABELS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_config(alpha_test: float = 0.05) -> dict:
    return {
        'dm_gk': {
            'enabled': True,
            'alpha_test': alpha_test,
            'alpha_test_petit_echantillon': 0.10,
            'hac_lags': 'auto',
            'n_instruments_gk': 2,
            'paires_section_principale': ['A_vs_B'],
        },
        'var': {'niveaux': [0.95]},
    }


# ── C1 : degenerate dans dm_test ──────────────────────────────────────────────

def test_dm_degenerate_si_se_quasi_nul():
    """C1 : dm_test retourne degenerate=True quand se < 1e-10 (var HAC quasi-nulle).

    _newey_west_var est patchee pour retourner 1e-25 (< 1e-20 => se < 1e-10),
    simulant le cas ou var1=var2 avec un T >> 1e6 (limite documentee H=1).
    """
    r   = np.zeros(500) - 2.0
    var = np.full(500, -2.33)
    with patch('tickerlab.core.dm_gk._newey_west_var', return_value=1e-25):
        res = dm_test(var, var, r, alpha=0.99)
    assert res.get('degenerate') is True, (
        f"dm_test doit retourner degenerate=True si se < 1e-10, obtenu {res}"
    )
    assert res['stat'] == pytest.approx(0.0, abs=1e-8)
    assert res['pval'] == pytest.approx(1.0, abs=1e-8)


def test_dm_degenerate_series_identiques_reel():
    """C1 : dm_test retourne degenerate=True pour var1==var2, SANS patcher _newey_west_var.

    Vérifie la détection directe sur le différentiel de perte (cause réelle),
    non via le seuil se < 1e-10 (inatteignable pour T < 1e6 à cause du floor 1e-14).
    """
    rng = np.random.default_rng(0)
    r   = rng.standard_normal(500) - 2.0
    var = np.full(500, -2.33)
    res = dm_test(var, var, r, alpha=0.99)
    assert res.get('degenerate') is True, (
        f"dm_test doit retourner degenerate=True si var1=var2 (sans patch), obtenu {res}"
    )
    assert res['stat'] == pytest.approx(0.0, abs=1e-8)
    assert res['pval'] == pytest.approx(1.0, abs=1e-8)


def test_dm_not_degenerate_si_var_differentes():
    """C1 : dm_test retourne degenerate=False sur series distinctes."""
    rng = np.random.default_rng(1)
    r   = rng.standard_normal(500) - 2.0
    res = dm_test(np.full(500, -1.65), np.full(500, -2.33), r, alpha=0.99)
    assert res.get('degenerate') is False


# ── C1 : non_discriminable dans _determine_verdict ───────────────────────────

def test_verdict_non_discriminable_si_dm_degenerate():
    """C1 : _determine_verdict retourne 'non_discriminable' si dm_degenerate=True."""
    v = _determine_verdict(d_bar=-0.001, dm_pval=1.0, gk_pval=0.8,
                           alpha_test=0.05, dm_degenerate=True, gk_degenerate=False)
    assert v == 'non_discriminable', f"Attendu 'non_discriminable', obtenu '{v}'"


def test_verdict_non_discriminable_si_gk_degenerate():
    """C1 : _determine_verdict retourne 'non_discriminable' si gk_degenerate=True."""
    v = _determine_verdict(d_bar=0.5, dm_pval=0.01, gk_pval=0.03,
                           alpha_test=0.05, dm_degenerate=False, gk_degenerate=True)
    assert v == 'non_discriminable', f"Attendu 'non_discriminable', obtenu '{v}'"


def test_verdict_non_discriminable_priorite_absolue():
    """C1 : 'non_discriminable' prime sur les 4 verdicts existants (meme si DM rejette)."""
    v = _determine_verdict(d_bar=-1.0, dm_pval=0.001, gk_pval=0.001,
                           alpha_test=0.05, dm_degenerate=True, gk_degenerate=True)
    assert v == 'non_discriminable'


def test_verdict_non_discriminable_dans_labels():
    """C1 : VERDICTS_LABELS contient la cle 'non_discriminable'."""
    assert 'non_discriminable' in VERDICTS_LABELS, (
        f"'non_discriminable' absent de VERDICTS_LABELS : {list(VERDICTS_LABELS)}"
    )


def test_verdict_standard_si_non_degenere():
    """C1 : les 4 verdicts habituels sont toujours retournes si degenerate=False."""
    v = _determine_verdict(d_bar=-0.5, dm_pval=0.01, gk_pval=0.5,
                           alpha_test=0.05, dm_degenerate=False, gk_degenerate=False)
    assert v == 'model1_wins'


# ── C1 : comparer_methodes_var retourne non_discriminable ────────────────────

def test_comparer_non_discriminable_si_series_identiques():
    """C1+C4 : comparer_methodes_var retourne 'non_discriminable' si A=B."""
    rng   = np.random.default_rng(7)
    T     = 500
    r_oos = rng.standard_normal(T) - 2.0
    var_identique = np.full(T, -2.33)
    var_series = {
        'A': {0.95: var_identique},
        'B': {0.95: var_identique},
    }
    cfg = _minimal_config()
    res = comparer_methodes_var(var_series, r_oos, [0.95], cfg)
    verdict = res[0.95]['A_vs_B']['verdict']
    assert verdict == 'non_discriminable', (
        f"Attendu 'non_discriminable' pour A=B, obtenu '{verdict}'"
    )


# ── C2 : gk_test reliable/degenerate ─────────────────────────────────────────

def test_gk_non_reliable_si_omega_singuliere():
    """C2 : gk_test retourne reliable=False, pval=nan, reject=False si var1=var2."""
    rng = np.random.default_rng(5)
    T   = 300
    r   = rng.standard_normal(T) - 2.0
    var = np.full(T, -2.33)
    res = gk_test(var, var, r, alpha=0.99)
    assert res.get('reliable') is False, (
        f"reliable doit etre False pour var1=var2, obtenu {res}"
    )
    assert res.get('degenerate') is True
    assert math.isnan(res['pval']), f"pval doit etre nan, obtenu {res['pval']}"
    assert res['reject'] is False
    assert res['stat'] >= 0.0
    assert np.isfinite(res['stat'])


def test_gk_reliable_si_omega_bien_conditionnee():
    """C2 : gk_test retourne reliable=True, pval finie, si series distinctes."""
    rng  = np.random.default_rng(3)
    T    = 500
    r    = rng.standard_normal(T) * 2.0 - 3.0
    var1 = np.full(T, -1.65)
    var2 = np.full(T, -2.33)
    res  = gk_test(var1, var2, r, alpha=0.99)
    assert res.get('reliable') is True, (
        f"reliable doit etre True pour series distinctes, obtenu {res}"
    )
    assert res.get('degenerate') is False
    assert math.isfinite(res['pval']), f"pval doit etre finie, obtenu {res['pval']}"
    assert 0.0 <= res['pval'] <= 1.0


# ── C4 : commentaire H=1 dans build_var_series ───────────────────────────────

def test_build_var_series_commentaire_h1():
    """C4 : build_var_series documente la limite H=1 (non_discriminable GARCH dyn. vs FHS)."""
    src = inspect.getsource(build_var_series)
    assert 'H=1' in src, (
        "build_var_series doit contenir le commentaire sur la limite H=1 "
        "(verdict 'non_discriminable' pour GARCH dyn. vs FHS)"
    )
