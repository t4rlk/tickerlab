# -*- coding: utf-8 -*-
"""
Reverse stress testing.

Pour un actif unique : identifie le choc minimal (au sens de Mahalanobis)
qui produit une perte cible donnee.

Pour le cas multi-actifs (portefeuille) : extension future.

References :
  Studer G. (1997) — Maximum loss for measurement of market risk
  Berkowitz J. (2000) — A coherent framework for stress testing
  BCBS d365 §44 — Reverse stress testing
"""
import numpy as np
from scipy import stats


def _proba_queue_t_standardisee(z: float, nu: float) -> float:
    """P(Z <= z) sous t standardisee (variance=1, convention arch). Voir stress_scenarios.py."""
    return float(stats.t.cdf(z * np.sqrt(nu / (nu - 2.0)), df=nu))


def reverse_stress(
    perte_cible: float,
    sigma_t: float,
    dist: str = 'normal',
    nu: float = None,
    horizon_jours: int = 1,
) -> dict:
    """
    Identifie le scenario minimal (distance Mahalanobis minimale) qui produit
    exactement perte_cible sur un actif unique.

    Pour un actif unique, la resolution est analytique :
      s*       = perte_cible  (choc direct = perte cible, position longue)
      sigma_H  = sigma_t * sqrt(H)
      d        = |s*| / sigma_H  (distance de Mahalanobis univariee)
      P        = F(s* / sigma_H) sous la distribution conditionnelle

    Parameters
    ----------
    perte_cible   : perte en rendement (negatif, ex: -0.30 pour -30%)
    sigma_t       : volatilite conditionnelle H=1 du modele GARCH
    dist          : 'normal' ou 't'
    nu            : degres de liberte (requis si dist='t')
    horizon_jours : horizon du scenario (scaling sigma_H = sigma_t * sqrt(H))

    Returns
    -------
    dict avec :
      choc_requis          : s* = perte_cible
      sigma_H              : volatilite a l'horizon H
      distance_mahalanobis : |s*| / sigma_H
      proba_sous_modele    : P(r <= s*) sous la distribution conditionnelle
      quantile_equivalent  : niveau de quantile correspondant (ex: 0.001 = top 0.1%)
      interpreteur         : description lisible

    Raises
    ------
    ValueError si perte_cible >= 0 ou sigma_t <= 0.
    """
    if perte_cible >= 0:
        raise ValueError(
            f'perte_cible doit etre negative (recu {perte_cible:.4f}). '
            'La perte est exprimee en rendement negatif, ex: -0.30 pour -30%.'
        )
    if sigma_t <= 0:
        raise ValueError(f'sigma_t doit etre > 0 (recu {sigma_t})')
    if horizon_jours < 1:
        raise ValueError(f'horizon_jours doit etre >= 1 (recu {horizon_jours})')

    s = perte_cible
    sigma_H = sigma_t * np.sqrt(horizon_jours)
    z = s / sigma_H          # z-score standardise (negatif)
    d = abs(z)               # distance de Mahalanobis (positive)

    # P(r <= s*) sous la distribution conditionnelle (perte_cible < 0, queue inferieure)
    if dist == 't' and nu is not None and nu > 2:
        proba = _proba_queue_t_standardisee(z, nu)
        dist_label = f't_std(nu={nu:.1f})'
    else:
        proba = float(stats.norm.cdf(z))
        dist_label = 'N(0,1)'

    # Niveau de quantile equivalent (pour lire comme "evenement 1-en-X periodes")
    if proba > 0:
        periode_retour = 1.0 / proba
    else:
        periode_retour = float('inf')

    return {
        'perte_cible':           perte_cible,
        'choc_requis':           s,
        'horizon_jours':         horizon_jours,
        'sigma_t':               sigma_t,
        'sigma_H':               round(sigma_H, 6),
        'distance_mahalanobis':  round(d, 4),
        'proba_sous_modele':     round(proba, 10),
        'quantile_equivalent':   round(proba, 10),
        'periode_retour_periodes': round(periode_retour, 1) if np.isfinite(periode_retour) else float('inf'),
        'interpreteur': (
            f"Perte de {abs(perte_cible):.1%} sur H={horizon_jours}j "
            f"= choc de {d:.2f}sigma_H sous {dist_label} "
            f"[P={proba:.6%}, 1 evenement tous les {periode_retour:.0f} periodes]"
            if np.isfinite(periode_retour) else
            f"Perte de {abs(perte_cible):.1%} sur H={horizon_jours}j "
            f"= choc de {d:.2f}sigma_H sous {dist_label} [P~0, evenement quasiment impossible]"
        ),
    }
