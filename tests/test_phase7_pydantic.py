# -*- coding: utf-8 -*-
"""
Tests Phase 7 — Validation Pydantic structurelle.

Tests :
  P0 : config.yaml de production passe sans erreur (test critique)
  P1 : seuil_significativite='0.05' (string) lève ValidationError
  P2 : split_ratio=1.5 (hors borne) lève ValidationError
  P3 : frequency='hourly' (hors Literal) lève ValidationError
  P4 : clé inconnue dans garch (extra='forbid') lève ValidationError
  P5 : end_date < start_date lève ValidationError
  P6 : non-régression — valeurs prod inchangées après passage Pydantic
"""
import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.config_schema import valider_config_structure
from tickerlab.core.exceptions import ValidationError


# ── Fixture config prod ───────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / 'config.yaml'
    with open(cfg_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ── P0 : config prod passe sans erreur ───────────────────────────────────────

def test_config_prod_passe_sans_erreur():
    """P0 (CRITIQUE) : config.yaml de production passe la validation Pydantic."""
    cfg = _load_config()
    result = valider_config_structure(cfg)
    assert result is cfg, 'valider_config_structure doit retourner le dict original'


# ── P1 : mauvais type float ───────────────────────────────────────────────────

def test_reject_seuil_significativite_string():
    """P1 : seuil_significativite='0.05' (string) doit lever ValidationError."""
    cfg = _load_config()
    cfg['garch']['seuil_significativite'] = '0.05'
    with pytest.raises(ValidationError) as exc_info:
        valider_config_structure(cfg)
    msg = str(exc_info.value)
    assert 'garch' in msg or 'seuil_significativite' in msg, (
        f"Message d'erreur doit mentionner le champ problematique : {msg}"
    )


# ── P2 : valeur hors borne ────────────────────────────────────────────────────

def test_reject_split_ratio_hors_borne():
    """P2 : split_ratio=1.5 (>= 1.0) doit lever ValidationError."""
    cfg = _load_config()
    cfg['backtest']['split_ratio'] = 1.5
    with pytest.raises(ValidationError):
        valider_config_structure(cfg)


# ── P3 : valeur hors Literal ─────────────────────────────────────────────────

def test_reject_frequency_invalide():
    """P3 : frequency='hourly' (hors Literal daily/weekly/monthly) doit lever ValidationError."""
    cfg = _load_config()
    cfg['data']['frequency'] = 'hourly'
    with pytest.raises(ValidationError) as exc_info:
        valider_config_structure(cfg)
    msg = str(exc_info.value)
    assert 'frequency' in msg or 'hourly' in msg, (
        f"Message doit mentionner frequency ou la valeur rejetee : {msg}"
    )


# ── P4 : clé inconnue (extra='forbid') ───────────────────────────────────────

def test_reject_cle_inconnue_garch():
    """P4 : clé inconnue dans garch (extra='forbid') doit lever ValidationError."""
    cfg = _load_config()
    cfg['garch']['parametre_inexistant_xyz'] = 42
    with pytest.raises(ValidationError) as exc_info:
        valider_config_structure(cfg)
    msg = str(exc_info.value)
    assert 'parametre_inexistant_xyz' in msg or 'garch' in msg, (
        f"Message doit mentionner la cle invalide : {msg}"
    )


# ── P5 : dates incohérentes ───────────────────────────────────────────────────

def test_reject_end_date_avant_start_date():
    """P5 : end_date < start_date doit lever ValidationError."""
    cfg = _load_config()
    cfg['data']['start_date'] = '2026-01-01'
    cfg['data']['end_date'] = '2006-01-01'
    with pytest.raises(ValidationError) as exc_info:
        valider_config_structure(cfg)
    msg = str(exc_info.value)
    assert 'start_date' in msg or 'end_date' in msg or 'anterieure' in msg, (
        f"Message doit mentionner le probleme d'ordre des dates : {msg}"
    )


# ── P6 : non-régression valeurs ──────────────────────────────────────────────

def test_non_regression_valeurs_apres_validation():
    """P6 : les valeurs clés de config.yaml sont inchangées après passage Pydantic."""
    cfg = _load_config()
    cfg_copy = copy.deepcopy(cfg)
    result = valider_config_structure(cfg)

    # Vérifier que le dict retourné est le même objet (pas une copie)
    assert result is cfg

    # Vérifier que les valeurs numériques et structurelles sont préservées
    assert result['garch']['seuil_significativite'] == pytest.approx(0.05)
    assert result['backtest']['split_ratio'] == pytest.approx(0.70)
    assert result['garch']['modeles'] == ['GARCH', 'GJR-GARCH', 'EGARCH', 'TGARCH', 'APARCH']
    assert result['fhs']['seed'] == 42
    assert result['bootstrap']['seed'] == 42
    assert result['data']['ticker'] == 'BZ=F'
    assert result['data']['frequency'] == 'weekly'
    assert result['structural_breaks']['mode'] == 'diagnostic'
    assert result['structural_breaks']['injecter_dans_garch'] is False
    assert result['bootstrap']['enabled'] is False
    assert result['fhs']['horizons'] == [1, 5, 10, 22]

    # Vérifier que le dict original n'a pas été muté
    assert cfg == cfg_copy
