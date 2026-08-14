# -*- coding: utf-8 -*-
"""Schéma Pydantic v2 pour la validation structurelle de config.yaml.

Valide : types, bornes numériques, clés inconnues (extra='forbid' sur blocs
scientifiques), ordre des dates.  Ne valide PAS les contraintes croisées
config×données (rôle de valider_config() dans config_validation.py).

Verdicts:
  - extra='forbid' + strict=True sur blocs scientifiques → attrape typos + mauvais types
  - extra='allow' sur blocs infra et racine → compatibilité _reuse_cache, legacy keys, etc.
"""
from __future__ import annotations

import re
from typing import Any, List, Literal, Optional, Union

import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import ValidationError as PipelineValidationError

# ── Config partagée pour blocs scientifiques ──────────────────────────────────
# strict=True : rejette str→float, int→bool, etc. (YAML donne déjà les bons types)
# extra='forbid' : toute clé non déclarée lève une ValidationError

_SCIENTIFIC = ConfigDict(extra='forbid', strict=True)


# ── Blocs scientifiques ───────────────────────────────────────────────────────

class DataConfig(BaseModel):
    model_config = _SCIENTIFIC
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    frequency: Optional[Literal['daily', 'weekly', 'monthly', 'annual']] = None
    min_observations: Optional[int] = Field(default=None, ge=50)
    max_nan_ratio: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    # Type de prix (configurateur web) : 'close' = clôture brute,
    # 'adjclose' = clôture ajustée dividendes/splits. auto_adjust en découle.
    price: Optional[Literal['close', 'adjclose']] = None
    auto_adjust: Optional[bool] = None

    @field_validator('start_date', 'end_date', mode='before')
    @classmethod
    def _date_format(cls, v: Any) -> Any:
        if v is not None and not re.match(r'^\d{4}-\d{2}-\d{2}$', str(v)):
            raise ValueError(f"format attendu 'YYYY-MM-DD', recu '{v}'")
        return v

    @model_validator(mode='after')
    def _date_order(self) -> 'DataConfig':
        if (self.start_date and self.end_date
                and self.start_date >= self.end_date):
            raise ValueError(
                f"start_date ({self.start_date}) doit etre anterieure "
                f"a end_date ({self.end_date})"
            )
        return self


class ArimaConfig(BaseModel):
    model_config = _SCIENTIFIC
    p_max: Optional[int] = Field(default=None, ge=1, le=10)
    q_max: Optional[int] = Field(default=None, ge=1, le=10)
    seuil_significativite: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    preference_parcimonie: Optional[bool] = None
    tolerance_aic_parcimonie: Optional[float] = Field(default=None, ge=0.0)  # LEGACY


class GarchConfig(BaseModel):
    model_config = _SCIENTIFIC
    modeles: Optional[List[Literal['GARCH', 'GJR-GARCH', 'EGARCH', 'TGARCH', 'APARCH']]] = None
    distributions: Optional[List[Literal['normal', 't', 'skewt', 'ged']]] = None
    p_max: Optional[int] = Field(default=None, ge=1, le=10)
    q_max: Optional[int] = Field(default=None, ge=1, le=10)
    seuil_significativite: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    inclure_figarch: Optional[bool] = None
    critere_significativite: Optional[str] = None
    tolerance_aic_parcimonie: Optional[float] = Field(default=None, ge=0.0)   # LEGACY
    seuil_engle_ng: Optional[float] = Field(default=None, gt=0.0, lt=1.0)    # LEGACY
    critere_information: Optional[Literal['AIC', 'BIC', 'HQIC']] = None
    tolerance_delta_critere_brut: Optional[float] = Field(default=None, ge=0.0)
    specification_tests: Optional[Any] = None
    validation_oos: Optional[Any] = None
    seuil_igarch: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    score_composite: Optional[Any] = None


class FhsConfig(BaseModel):
    model_config = _SCIENTIFIC
    enabled: Optional[bool] = None
    n_boot: Optional[int] = Field(default=None, ge=100)
    n_boot_backtest: Optional[int] = Field(default=None, ge=1)
    horizons: Optional[List[int]] = None
    seed: Optional[int] = None


class DmGkConfig(BaseModel):
    model_config = _SCIENTIFIC
    enabled: Optional[bool] = None
    test_type: Optional[str] = None
    alpha_test: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    alpha_test_petit_echantillon: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    hac_lags: Optional[Union[int, str]] = None
    n_instruments_gk: Optional[int] = Field(default=None, ge=1, le=5)
    paires_section_principale: Optional[List[str]] = None


class ComponentGarchConfig(BaseModel):
    model_config = _SCIENTIFIC
    enabled: Optional[bool] = None
    force_estimation: Optional[bool] = None
    seuil_persistance: Optional[float] = Field(default=None, gt=0.0, lt=1.0)


class VarConfig(BaseModel):
    model_config = _SCIENTIFIC
    niveaux: Optional[List[float]] = None
    n_simulations_mc: Optional[int] = Field(default=None, ge=1)
    horizons: Optional[List[int]] = None
    n_simulations_horizons: Optional[int] = Field(default=None, ge=1)


class BacktestConfig(BaseModel):
    model_config = _SCIENTIFIC
    split_ratio: Optional[float] = Field(default=None, ge=0.5, lt=1.0)
    niveaux_test: Optional[List[float]] = None
    seuil_p_value: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    tests_frtb: Optional[Any] = None
    # Detail enrichi (6 tests FRTB/Berkowitz) demande par l'API v1 (opt-in).
    detail_frtb: Optional[bool] = None


class StructuralBreaksConfig(BaseModel):
    model_config = _SCIENTIFIC
    enabled: Optional[bool] = None
    mode: Optional[Literal['diagnostic', 'integrate', 'off']] = None
    methode: Optional[str] = None
    seuil_persistance_alerte: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    seuil_n_ruptures_alerte: Optional[int] = Field(default=None, ge=1)
    # Mode 'integrate' : selection des ruptures pour l'estimation omega/regime.
    max_regimes: Optional[int] = Field(default=None, ge=2)
    min_obs_regime: Optional[int] = Field(default=None, ge=2)
    # DEPRECATED — comptait des ruptures a l'epoque des dummies ; interprete
    # comme (max_dummies + 1) regimes si max_regimes est absent.
    max_dummies: Optional[int] = Field(default=None, ge=1)
    trim: Optional[float] = None           # LEGACY
    icss: Optional[Any] = None             # LEGACY
    zivot_andrews: Optional[Any] = None    # LEGACY
    injecter_dans_garch: Optional[bool] = None  # DEPRECATED


class BootstrapConfig(BaseModel):
    model_config = _SCIENTIFIC
    enabled: Optional[bool] = None
    n_boot: Optional[int] = Field(default=None, ge=1)
    ci_level: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    block_size: Optional[int] = Field(default=None, ge=1)
    express: Optional[bool] = None
    n_replications: Optional[int] = Field(default=None, ge=1)
    block_length: Optional[int] = Field(default=None, ge=1)
    niveaux_ic: Optional[List[float]] = None
    inclure_tvar: Optional[bool] = None
    seed: Optional[int] = None


class RollingBacktestConfig(BaseModel):
    model_config = _SCIENTIFIC
    enabled: Optional[bool] = None
    window_size: Optional[int] = Field(default=None, ge=1)
    refit_every: Optional[int] = Field(default=None, ge=1)
    niveaux_test: Optional[List[float]] = None


# ── Racine ────────────────────────────────────────────────────────────────────
# extra='allow' : accepte _reuse_cache, _metrics_only, cache_v2, blocs infra
# (output, events, rapport, sorties_etendues, ai, ai_writer, monitoring,
#  export_academique, frtb) sans les valider structurellement.

class TickerLabConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    data: DataConfig
    arima: Optional[ArimaConfig] = None
    garch: Optional[GarchConfig] = None
    fhs: Optional[FhsConfig] = None
    dm_gk: Optional[DmGkConfig] = None
    component_garch: Optional[ComponentGarchConfig] = None
    var: Optional[VarConfig] = None
    backtest: Optional[BacktestConfig] = None
    structural_breaks: Optional[StructuralBreaksConfig] = None
    bootstrap: Optional[BootstrapConfig] = None
    rolling_backtest: Optional[RollingBacktestConfig] = None


# ── Point d'entrée public ─────────────────────────────────────────────────────

def _format_errors(e: pydantic.ValidationError) -> str:
    lines = [f'Config invalide — {e.error_count()} erreur(s) :']
    for err in e.errors():
        loc = '.'.join(str(x) for x in err['loc']) if err['loc'] else '(racine)'
        inp = err.get('input')
        msg = err['msg']
        lines.append(f'  * {loc} = {inp!r} : {msg}')
    return '\n'.join(lines)


def valider_config_structure(raw: dict) -> dict:
    """Valide la structure de config via Pydantic (types, bornes, clés inconnues).

    Lève ValidationError (exceptions.py) avec message exploitable si invalide.
    Retourne le dict original inchangé si valide — les defaults Pydantic ne
    sont pas propagés pour ne pas court-circuiter les .get() du code aval.
    """
    try:
        TickerLabConfig.model_validate(raw)
    except pydantic.ValidationError as exc:
        message = _format_errors(exc)
        raise PipelineValidationError(
            message,
            etape='config_schema',
            contexte={'n_erreurs': exc.error_count()},
        ) from exc
    return raw
