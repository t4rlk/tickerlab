# -*- coding: utf-8 -*-
"""Generateur de signaux directionnels (endpoint GET /api/v1/signaux).

# SPEC PROVISOIRE — methodologie a rediger (section VII du site)
============================================================================
La regle ci-dessous est PROVISOIRE et destinee a une demonstration pedagogique.
Elle n'est PAS un conseil en investissement.

Regle (documentee) : on part de la volatilite conditionnelle sigma_t du dernier
GARCH(1,1) Student-t estime pour le symbole. Pour chaque seance :
    sigma_bar_t = moyenne mobile 20 seances de sigma
    z_t = (sigma_t - sigma_bar_t) / sigma_bar_t          (ecart relatif de sigma)
    - z_t <= -SEUIL  -> 'achat'  (risque bas relativement a la tendance)
    - z_t >= +SEUIL  -> 'vente'  (risque eleve)
    - sinon          -> 'neutre'
    sigma_pct  = sigma_t  (en %)
    var99_pct  = -(mu + q01_std * sigma_t)   (VaR de PERTE, queue GAUCHE, en %),
                 q01_std = quantile 1% des residus standardises estimes (skew-t
                 respectee) -- MEME convention de queue que backtest_oos.
    confiance_pct = min(50 + PENTE*|z_t|, PLAFOND)   (pente douce, gradient lisible)
============================================================================
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

_log = logging.getLogger('tickerlab.web')

# Parametres de la regle provisoire (documentes, unique source de verite).
SEUIL: float = 0.10       # declenchement achat/vente sur |z|
MA_FENETRE: int = 20      # fenetre de la moyenne mobile de sigma
CONF_BASE: float = 50.0
CONF_PENTE: float = 150.0  # pente douce : |z|=0.10 -> ~65 %, |z|=0.30 -> 95 %
CONF_PLAFOND: float = 95.0

AVERTISSEMENT = ('Demonstration pedagogique — ne constitue pas un conseil en '
                 'investissement.')
MODELE_SOURCE = 'GARCH(1,1) Student-t (SPEC PROVISOIRE)'

# Memoisation legere en-process : evite de re-estimer le meme jour.
_CACHE: dict = {}


def regle_signal(z: float) -> str:
    """Traduit l'ecart relatif de sigma z_t en signal (regle provisoire)."""
    if z <= -SEUIL:
        return 'achat'
    if z >= SEUIL:
        return 'vente'
    return 'neutre'


def _confiance(z: float) -> int:
    return int(round(min(CONF_BASE + CONF_PENTE * abs(z), CONF_PLAFOND)))


def lignes_signaux(dates, sigma_pct, mu: float, q01_std: float,
                   jours: int) -> list[dict]:
    """Calcule les `jours` dernieres lignes de signaux. FONCTION PURE / DETERMINISTE.

    Parameters
    ----------
    dates : sequence de dates (len == N)
    sigma_pct : array-like des volatilites conditionnelles en % (len == N)
    mu : float          Moyenne conditionnelle (meme echelle que sigma).
    q01_std : float     Quantile 1% des residus standardises (negatif, queue gauche).
    jours : int         Nombre de dernieres seances a restituer.
    """
    sigma = np.asarray(sigma_pct, dtype=float)
    ma = pd.Series(sigma).rolling(MA_FENETRE, min_periods=MA_FENETRE).mean().values
    N = len(sigma)
    debut = max(0, N - int(jours))
    lignes: list[dict] = []
    for i in range(debut, N):
        s = float(sigma[i])
        m = float(ma[i]) if (i < len(ma) and math.isfinite(ma[i]) and ma[i] != 0) else float('nan')
        z = (s - m) / m if math.isfinite(m) else 0.0
        var99 = -(mu + q01_std * s)          # queue gauche : magnitude de perte
        lignes.append({
            'date':          str(pd.Timestamp(dates[i]).date()),
            'signal':        regle_signal(z),
            'sigma_pct':     round(s, 4),
            'var99_pct':     round(float(var99), 4),
            'confiance_pct': _confiance(z),
        })
    return lignes


def _estimer_sigma(symbol: str, config: Optional[dict]) -> tuple:
    """Telecharge ~5 ans de donnees, estime un GARCH(1,1)-t, renvoie
    (dates, sigma_pct, mu, q01_std). Reutilise data_loader ; leve DataError sur
    symbole inconnu / periode vide (propagee au handler HTTP)."""
    from arch import arch_model
    from tickerlab.core.data_loader import telecharger_prix, calculer_rendements

    fin = _dt.date.today()
    debut = fin - _dt.timedelta(days=5 * 365)
    auto_adjust = True
    if config:
        auto_adjust = bool(config.get('data', {}).get('auto_adjust', True))

    prix = telecharger_prix(symbol, debut.isoformat(), fin.isoformat(),
                            auto_adjust=auto_adjust)
    rendements = calculer_rendements(prix, freq='daily').dropna()

    res = arch_model(rendements, mean='Constant', vol='GARCH', p=1, o=0, q=1,
                     dist='t').fit(disp='off')
    sigma = res.conditional_volatility
    mu = float(res.params.get('mu', float(rendements.mean())))
    z_resid = (res.resid / res.conditional_volatility).dropna().values
    z_resid = z_resid[np.isfinite(z_resid)]
    q01_std = float(np.quantile(z_resid, 0.01)) if len(z_resid) else float('nan')
    return sigma.index, sigma.values, mu, q01_std


def generer_signaux(symbol: str, jours: int = 30,
                    config: Optional[dict] = None) -> dict:
    """Genere la reponse complete de GET /api/v1/signaux pour `symbol`.

    Memoisation par (symbol, jours, jour_courant) : deux appels le meme jour
    renvoient exactement les memes lignes (determinisme a cache identique).
    """
    symbol = str(symbol).strip()
    jours = max(1, min(int(jours), 365))
    cle = (symbol, jours, _dt.date.today().isoformat())
    if cle in _CACHE:
        return _CACHE[cle]

    dates, sigma, mu, q01 = _estimer_sigma(symbol, config)
    lignes = lignes_signaux(dates, sigma, mu, q01, jours)

    reponse = {
        'symbol':        symbol,
        'genere_le':     (_dt.datetime.now(_dt.timezone.utc)
                          .replace(tzinfo=None, microsecond=0).isoformat() + 'Z'),
        'modele_source': MODELE_SOURCE,
        'avertissement': AVERTISSEMENT,
        'lignes':        lignes,
    }
    _CACHE[cle] = reponse
    return reponse
