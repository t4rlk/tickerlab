# -*- coding: utf-8 -*-
"""
Tests — Robustesse d'entrée et rééchantillonnage (data_loader).

Robustesse (valider_donnees) :
  R1 : série trop courte (< min_observations) => DataError
  R2 : trop de NaN dans les prix (> max_nan_ratio) => DataError
  R3 : prix <= 0 détectés => DataError
  R4 : série quasi constante (std ~0) => DataError
  R5 : non-régression — série propre de 800 obs ne lève rien
  R6 : config.yaml avec nouvelles clés data passe le schéma Pydantic
  R7 : telecharger_prix lève DataError (et non ValueError) sur DataFrame vide

Rééchantillonnage (resampler_prix / calculer_rendements) :
  cohérence n_obs prix/rendements en weekly, invariance en daily,
  réduction attendue en weekly/monthly.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.data_loader import valider_donnees, resampler_prix
from tickerlab.core.exceptions import DataError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prix(n: int, values=None, nan_indices=None) -> pd.DataFrame:
    """Crée un DataFrame de prix avec index DatetimeIndex."""
    dates = pd.date_range('2010-01-01', periods=n, freq='B')
    if values is None:
        rng = np.random.default_rng(0)
        values = np.cumprod(1 + rng.standard_normal(n) * 0.005) * 100.0
    s = pd.Series(values, index=dates, name='prix', dtype=float)
    if nan_indices is not None:
        s.iloc[nan_indices] = np.nan
    return pd.DataFrame({'prix': s})


def _make_rendements(prix: pd.DataFrame) -> pd.Series:
    """Calcule les log-rendements (en %, dropna)."""
    return np.log(prix['prix'] / prix['prix'].shift(1)).dropna() * 100


def _minimal_config(ticker='TEST', min_obs=250, max_nan=0.10) -> dict:
    return {
        'data': {
            'ticker': ticker,
            'min_observations': min_obs,
            'max_nan_ratio': max_nan,
        }
    }


# ── R1 : série trop courte ────────────────────────────────────────────────────

def test_reject_serie_trop_courte():
    """R1 : 40 obs < min_observations=250 => DataError."""
    prix = _make_prix(40)
    rendements = _make_rendements(prix)
    with pytest.raises(DataError) as exc_info:
        valider_donnees(prix, rendements, _minimal_config())
    msg = str(exc_info.value)
    assert '39' in msg or '40' in msg or '250' in msg, (
        f"Message doit mentionner le nombre d'obs ou le seuil : {msg}"
    )
    assert exc_info.value.etape == 'data_validation'


def test_reject_serie_trop_courte_seuil_custom():
    """R1bis : seuil custom min_observations=100 => 40 obs encore rejeté."""
    prix = _make_prix(40)
    rendements = _make_rendements(prix)
    cfg = _minimal_config(min_obs=100)
    with pytest.raises(DataError):
        valider_donnees(prix, rendements, cfg)


def test_accept_serie_exactement_au_seuil():
    """R1ter : n == min_observations => pas d'erreur (borne incluse)."""
    prix = _make_prix(251)  # 251 prix => 250 rendements
    rendements = _make_rendements(prix)
    cfg = _minimal_config(min_obs=250)
    valider_donnees(prix, rendements, cfg)  # ne doit pas lever


# ── R2 : trop de NaN ─────────────────────────────────────────────────────────

def test_reject_trop_de_nan():
    """R2 : 30% de NaN > max_nan_ratio=10% => DataError.

    Utilise n=1000 pour que les 300 NaN laissent ~699 rendements valides
    (>> 250) et que le check b) s'active avant le check a).
    """
    n = 1000
    nan_idx = list(range(0, 300))  # 30% de NaN au début
    prix = _make_prix(n, nan_indices=nan_idx)
    rendements = _make_rendements(prix)
    with pytest.raises(DataError) as exc_info:
        valider_donnees(prix, rendements, _minimal_config())
    assert exc_info.value.etape == 'data_validation'
    ctx = exc_info.value.contexte
    assert ctx['n_nan'] == 300
    assert ctx['part_nan'] > 0.10


def test_accept_nan_sous_seuil():
    """R2bis : 5% de NaN < max_nan_ratio=10% => pas d'erreur."""
    n = 300
    nan_idx = list(range(0, 15))  # 5% de NaN
    prix = _make_prix(n, nan_indices=nan_idx)
    rendements = _make_rendements(prix)
    valider_donnees(prix, rendements, _minimal_config())  # ne doit pas lever


# ── R3 : prix non positifs ────────────────────────────────────────────────────

def test_reject_prix_negatif():
    """R3 : un prix < 0 => DataError (log-rendements impossibles)."""
    n = 300
    values = np.cumprod(1 + np.random.default_rng(1).standard_normal(n) * 0.005) * 100
    values[50] = -5.0
    prix = _make_prix(n, values=values)
    rendements = _make_rendements(prix)
    with pytest.raises(DataError) as exc_info:
        valider_donnees(prix, rendements, _minimal_config())
    assert exc_info.value.etape == 'data_validation'
    assert exc_info.value.contexte['n_prix_non_positifs'] >= 1


def test_reject_prix_zero():
    """R3bis : un prix == 0 => DataError."""
    n = 300
    values = np.cumprod(1 + np.random.default_rng(2).standard_normal(n) * 0.005) * 100
    values[100] = 0.0
    prix = _make_prix(n, values=values)
    rendements = _make_rendements(prix)
    with pytest.raises(DataError):
        valider_donnees(prix, rendements, _minimal_config())


# ── R4 : série quasi constante ───────────────────────────────────────────────

def test_reject_serie_constante():
    """R4 : std(rendements) ~ 0 => DataError (ticker gelé/suspendu)."""
    n = 300
    values = np.ones(n) * 42.0
    prix = _make_prix(n, values=values)
    rendements = _make_rendements(prix)  # tous zéros => std = 0
    with pytest.raises(DataError) as exc_info:
        valider_donnees(prix, rendements, _minimal_config())
    assert exc_info.value.etape == 'data_validation'
    assert exc_info.value.contexte['std_rendements'] < 1e-8


# ── R5 : non-régression — série propre ───────────────────────────────────────

def test_non_regression_serie_propre():
    """R5 : série propre de 800 obs => valider_donnees ne lève rien."""
    rng = np.random.default_rng(42)
    values = np.cumprod(1 + rng.standard_normal(800) * 0.01) * 100.0
    prix = _make_prix(800, values=values)
    rendements = _make_rendements(prix)
    cfg = _minimal_config(min_obs=250, max_nan=0.10)
    valider_donnees(prix, rendements, cfg)  # must NOT raise


# ── R6 : nouvelles clés config passent le schéma Pydantic ────────────────────

def test_config_nouvelles_cles_passe_pydantic():
    """R6 : config.yaml avec min_observations/max_nan_ratio passe valider_config_structure."""
    from tickerlab.core.config_schema import valider_config_structure
    cfg_path = Path(__file__).parent.parent / 'config.yaml'
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    result = valider_config_structure(cfg)
    assert result is cfg
    # Les nouvelles clés sont bien présentes et ont les bonnes valeurs
    assert cfg['data']['min_observations'] == 250
    assert cfg['data']['max_nan_ratio'] == pytest.approx(0.10)


# ── R7 : DataError sur DataFrame vide (ex-ValueError) ────────────────────────

def test_telecharger_prix_leve_dataerror_sur_vide():
    """R7 : telecharger_prix lève DataError (pas ValueError) si yfinance retourne vide."""
    from tickerlab.core.data_loader import telecharger_prix

    empty_df = pd.DataFrame()
    with patch('tickerlab.core.data_loader.yf.download', return_value=empty_df):
        with pytest.raises(DataError) as exc_info:
            telecharger_prix('FAKE', '2020-01-01', '2021-01-01')
    assert exc_info.value.etape == 'download'
    assert exc_info.value.ticker == 'FAKE'


# ── Rééchantillonnage de fréquence (resampler_prix) ──────────────────────────

def _prix_daily(n=1000, start='2018-01-02'):
    idx = pd.bdate_range(start=start, periods=n)
    vals = 50 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, n))
    return pd.DataFrame({'prix': vals}, index=idx)


def test_coherence_n_obs_freq_weekly():
    """
    Avec freq='weekly', resampler_prix reduit le DataFrame de prix ;
    len(prix_stats) doit etre compatible avec len(rendements) calcules en weekly.
    """
    from tickerlab.core.data_loader import calculer_rendements

    prix = _prix_daily(n=1000)
    prix_stats  = resampler_prix(prix, freq='weekly')
    rendements  = calculer_rendements(prix, freq='weekly')

    # rendements = diff(prix_stats) -> len(rendements) == len(prix_stats) - 1
    assert len(prix_stats) > 0, "resampler_prix retourne DataFrame vide"
    assert abs(len(prix_stats) - len(rendements)) <= 1, (
        f"Incoherence : {len(prix_stats)} prix stats vs {len(rendements)} rendements weekly "
        f"(ecart > 1 inattendu)"
    )


def test_resampler_daily_ne_change_rien():
    """freq='daily' ne modifie pas le DataFrame."""
    prix = _prix_daily(n=200)
    result = resampler_prix(prix, freq='daily')
    assert len(result) == len(prix), "resampler_prix daily a modifie la taille"
    assert result is prix, "resampler_prix daily devrait retourner l'objet original"


def test_resampler_weekly_reduit_obs():
    """freq='weekly' produit environ n//5 observations (semaines de travail)."""
    prix = _prix_daily(n=500)
    result = resampler_prix(prix, freq='weekly')
    # 500 jours ouvres ~ 100 semaines; on accepte entre 80 et 115
    assert 80 <= len(result) <= 115, (
        f"resampler_prix weekly : {len(result)} obs inattendu pour 500 jours ouvres"
    )


def test_resampler_monthly_reduit_obs():
    """freq='monthly' produit environ n//22 observations."""
    prix = _prix_daily(n=500)
    result = resampler_prix(prix, freq='monthly')
    assert 18 <= len(result) <= 26, (
        f"resampler_prix monthly : {len(result)} obs inattendu pour 500 jours ouvres"
    )
