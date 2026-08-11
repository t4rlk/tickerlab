# -*- coding: utf-8 -*-
"""Estimation du nombre d'observations depuis (dates, frequence).

Source de verite serveur pour les gardes de l'API v1 :
  - < 250 obs estimees  -> refus 422 SERIE_TROP_COURTE
  - 250 a 500 obs       -> acceptation + warning SERIE_FRAGILE

Estimation volontairement simple (pas de calendrier de bourse reel) : facteurs
moyens jours ouvres / semaines / mois sur la periode. Suffisant pour une GARDE ;
la validation fine reste faite par core/data_loader.valider_donnees a l'execution.
"""
from __future__ import annotations

from datetime import date

SEUIL_REFUS: int = 250      # en-deca : serie trop courte (refus)
SEUIL_FRAGILE: int = 500    # 250-500 : serie fragile (warning)

_OBS_PAR_AN = {'daily': 252.0, 'weekly': 52.0, 'monthly': 12.0}


def _parse(d: str) -> date:
    return date.fromisoformat(str(d)[:10])


def estimer_observations(date_from: str, date_to: str, freq: str) -> int:
    """Estime le nombre d'observations sur [date_from, date_to] pour `freq`.

    Retourne 0 si les dates sont invalides ou inversees (l'appelant tranche).
    """
    try:
        d0, d1 = _parse(date_from), _parse(date_to)
    except (ValueError, TypeError):
        return 0
    if d1 <= d0:
        return 0
    annees = (d1 - d0).days / 365.25
    facteur = _OBS_PAR_AN.get(str(freq), 252.0)
    return int(annees * facteur)


def evaluer_longueur(date_from: str, date_to: str, freq: str) -> tuple[int, str | None]:
    """Retourne (n_obs_estime, statut) ou statut in {None, 'REFUS', 'FRAGILE'}.

    - 'REFUS'   : n_obs < SEUIL_REFUS  -> l'API renvoie 422 SERIE_TROP_COURTE
    - 'FRAGILE' : SEUIL_REFUS <= n_obs < SEUIL_FRAGILE -> warning SERIE_FRAGILE
    - None      : n_obs suffisant
    """
    n = estimer_observations(date_from, date_to, freq)
    if n < SEUIL_REFUS:
        return n, 'REFUS'
    if n < SEUIL_FRAGILE:
        return n, 'FRAGILE'
    return n, None
