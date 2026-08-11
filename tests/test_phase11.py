# -*- coding: utf-8 -*-
"""Tests Phase 1.1 — Interprétation pédagogique ARIMA.

Couvre :
- _determiner_interpretation() : les 4 catégories + garde n < 100
- selectionner_arima()          : rétrocompatibilité des clés du dict retourné
- _encadre_pedagogique()        : génération de flowables ReportLab
"""
import sys
import os

import numpy as np
import pytest

# Assure la résolution du package depuis la racine du dépôt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tickerlab.core.arima_selector import _determiner_interpretation, selectionner_arima


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rng(seed: int = 42):
    return np.random.default_rng(seed)


# =============================================================================
# 1. serie_predictible — AR(1) φ=0.5, n=500
# =============================================================================

def test_interpretation_serie_predictible():
    """Coefficients significatifs et amplitude > 0.10 → serie_predictible."""
    fit_params = {'ar.L1': 0.5, 'const': 0.01}
    result = _determiner_interpretation(
        motif='tous_sig + parcimonie',
        p_opt=1, q_opt=0,
        fit_params=fit_params,
        lb_pval_rendements=0.001,
        n_obs=500,
    )
    assert result['code'] == 'serie_predictible', result
    assert 'Lo' not in result['message']      # pas la branche bruit_faible
    assert 'Fama' not in result['message']    # pas la branche martingale


# =============================================================================
# 2. bruit_faible — AR(1) φ=0.05, n=5000
# =============================================================================

def test_interpretation_bruit_faible():
    """Coefficients significatifs mais amplitude ≤ 0.10 → bruit_faible (Lo & MacKinlay 1988)."""
    fit_params = {'ar.L1': 0.05, 'const': 0.0}
    result = _determiner_interpretation(
        motif='tous_sig + parcimonie',
        p_opt=1, q_opt=0,
        fit_params=fit_params,
        lb_pval_rendements=0.02,
        n_obs=5000,
    )
    assert result['code'] == 'bruit_faible', result
    assert 'MacKinlay' in result['message'] or 'Lo' in result['message']


# =============================================================================
# 3. martingale_marche_efficient — bruit blanc, LB non rejeté
# =============================================================================

def test_interpretation_martingale_par_LB():
    """AIC_seul_AUCUN_SIG + lb_pval_brut > 0.10 → martingale_marche_efficient."""
    result = _determiner_interpretation(
        motif='AIC_seul_AUCUN_SIG',
        p_opt=1, q_opt=0,
        fit_params={'ar.L1': 0.03},
        lb_pval_rendements=0.45,   # > 0.10 : pas de structure détectée
        n_obs=500,
    )
    assert result['code'] == 'martingale_marche_efficient', result
    assert 'Fama' in result['message']


# =============================================================================
# 4. incertain — série trop courte (n < 100)
# =============================================================================

def test_interpretation_incertain_serie_courte():
    """n < 100 → force 'incertain' indépendamment des autres indicateurs."""
    result = _determiner_interpretation(
        motif='tous_sig + parcimonie',
        p_opt=1, q_opt=0,
        fit_params={'ar.L1': 0.8},   # gros coefficient — mais n trop petit
        lb_pval_rendements=0.001,
        n_obs=50,
    )
    assert result['code'] == 'incertain', result
    assert '50' in result['message']


# =============================================================================
# 5. Rétrocompatibilité — selectionner_arima() enrichit sans casser les clés
# =============================================================================

def test_interpretation_retrocompatibilite():
    """selectionner_arima() enrichit le dict sans supprimer les clés existantes."""
    rng = _rng(0)
    # AR(1) φ=0.3 : assure qu'au moins un modèle ARIMA converge dans la grille
    eps = rng.normal(size=300)
    x = np.zeros(300)
    for i in range(1, 300):
        x[i] = 0.3 * x[i - 1] + eps[i]
    serie = x.tolist()

    result = selectionner_arima(serie, d_arima=0, p_max=2, q_max=2)

    # Anciennes clés présentes et typées correctement
    assert 'p_opt' in result and isinstance(result['p_opt'], int)
    assert 'd_opt' in result and isinstance(result['d_opt'], int)
    assert 'q_opt' in result and isinstance(result['q_opt'], int)
    assert 'aic' in result and isinstance(result['aic'], float)
    assert 'motif_selection' in result and isinstance(result['motif_selection'], str)
    assert 'fiabilite' in result and isinstance(result['fiabilite'], bool)

    # Nouvelles clés Phase 1.1 présentes et non vides
    assert 'interpretation' in result
    assert 'message_pedagogique' in result
    assert 'lb_pval_rendements' in result
    assert 'max_magnitude_coef' in result

    assert result['interpretation'] in {
        'serie_predictible', 'bruit_faible',
        'martingale_marche_efficient', 'incertain',
    }
    assert isinstance(result['message_pedagogique'], str)
    assert len(result['message_pedagogique']) > 0
    # .get() avec défaut ne plante pas
    assert result.get('interpretation', '') != ''


# =============================================================================
# 6. Flowables — _encadre_pedagogique retourne une liste non vide
# =============================================================================

def test_rapport_contient_encadre_pedagogique():
    """_encadre_pedagogique retourne des flowables ReportLab pour un message non vide."""
    import tickerlab.core.rapport._helpers as _H
    _H._set_theme('academique')

    from tickerlab.core.rapport._sections import _encadre_pedagogique

    config = {'rapport': {'theme': 'academique'}}
    flowables = _encadre_pedagogique(
        message='Test pédagogique.',
        code='serie_predictible',
        config=config,
    )
    assert len(flowables) > 0, "Aucun flowable généré — encadré vide."

    # Message vide → liste vide (pas de crash)
    flowables_empty = _encadre_pedagogique(message='', code='incertain', config=config)
    assert flowables_empty == []
