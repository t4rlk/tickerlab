# -*- coding: utf-8 -*-
"""PipelineResult : conteneur rétrocompatible des sorties run_pipeline()."""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineResult:
    """Sortie structurée de run_pipeline().

    Interface dict-like ([], get, keys, in) pour la rétrocompatiblité avec
    tout code qui utilisait le dict retourné auparavant.
    Le champ `meta` expose les métadonnées de reproductibilité.
    """
    # Résultats scientifiques (identiques au dict d'origine)
    prix: Any = None
    rendements: Any = None
    arima: Any = None
    garch_best: Any = None
    garch_final: Any = None
    var: Any = None
    backtest: Any = None
    vol: Any = None
    igarch_diag: Any = None
    component_garch: Any = None
    fhs: Any = None
    dm_gk: Any = None
    T_train: Optional[int] = None
    T_eff_dyn: Optional[int] = None
    df_garch: Any = None
    spec_verdict: Any = None
    motif_selection: str = ''
    structural_breaks: Any = None
    # Detail backtest enrichi (FRTB + Berkowitz + series OOS) — peuple sur demande
    # (retour_detail dans backtest_oos). None par defaut : aucune regression sur les
    # sorties numeriques existantes.
    backtest_detail: Any = None
    # Métadonnées reproductibilité
    meta: dict = field(default_factory=dict)

    # ── Interface dict-like ──────────────────────────────────────────────────
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def keys(self):
        return list(self.__dataclass_fields__.keys())

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
