# -*- coding: utf-8 -*-
"""Registre minimal des sorties econometriques exposables au configurateur.

Catalogue unique des cles `outputs` acceptees par l'API v1 / le configurateur web
et par la CLI. Deux usages :

  1. Validation stricte : rejeter toute cle inconnue (traduite en 422 par l'API).
  2. Ordre de section : l'ordre de CLIC du configurateur est preserve tel quel
     (`config['sorties_ordre']`, cf. web/mapping.py) ; ce registre fournit l'ordre
     CANONIQUE de repli et le libelle humain de chaque sortie.

Volontairement leger : un dict + quelques helpers, pas un framework. Les cles sont
alignees sur `OUTS` du front (web/static/index.html) : garch, var, backtest,
breaks, report, charts. Etendre ICI (source de verite unique) quand une nouvelle
sortie est exposee.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SortieEcon:
    """Description d'une sortie exposable."""
    cle: str
    libelle: str
    ordre: int          # ordre canonique de section (repli si non pilote par le clic)
    intrinseque: bool   # True = toujours calcule par le pipeline (non desactivable)


# Ordre canonique = ordre de lecture naturel du rapport.
_REGISTRE: "dict[str, SortieEcon]" = {
    'garch':    SortieEcon('garch',    'Modele GARCH (variance conditionnelle)',          10, True),
    'var':      SortieEcon('var',      'VaR & Expected Shortfall',                         20, True),
    'backtest': SortieEcon('backtest', 'Backtesting (Kupiec, Christoffersen, DQ, FRTB)',   30, True),
    'breaks':   SortieEcon('breaks',   'Ruptures structurelles (ICSS, Zivot-Andrews)',     40, False),
    'charts':   SortieEcon('charts',   'Graphiques & annexes',                             50, True),
    'report':   SortieEcon('report',   'Rapport redige (PDF)',                             60, False),
}


def cles_valides() -> "frozenset[str]":
    """Ensemble des cles `outputs` reconnues."""
    return frozenset(_REGISTRE)


def valider_outputs(outputs) -> "list[str]":
    """Valide une liste de cles `outputs` en preservant l'ordre de clic.

    Retourne la liste dedupliquee (ordre d'apparition conserve). Leve ValueError
    sur toute cle inconnue — la couche appelante (API) la traduit en 422.
    """
    vu: "set[str]" = set()
    ordonne: "list[str]" = []
    for o in outputs or []:
        if o not in _REGISTRE:
            raise ValueError(
                f"Sortie inconnue : {o!r}. Cles valides : "
                f"{', '.join(sorted(_REGISTRE))}."
            )
        if o not in vu:
            vu.add(o)
            ordonne.append(o)
    return ordonne


def libelle(cle: str) -> str:
    """Libelle humain d'une cle (ou la cle elle-meme si inconnue)."""
    s = _REGISTRE.get(cle)
    return s.libelle if s else cle


def ordre_canonique(cle: str) -> int:
    """Rang de section canonique d'une cle (999 si inconnue)."""
    s = _REGISTRE.get(cle)
    return s.ordre if s else 999
