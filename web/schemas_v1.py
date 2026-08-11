# -*- coding: utf-8 -*-
"""Schemas Pydantic v2 de l'API v1 — contrats stricts (`extra='forbid'`).

Source de verite des validations scientifiques cote serveur : le front envoie ces
memes champs et toute cle inconnue est rejetee (422). Un adaptateur traduit la
requete v1 vers le payload interne consomme par web/mapping.construire_config.

Aucune cle API, aucun secret ici. Les valeurs de `price`/`freq`/`module` sont des
Literals fermes : toute autre valeur => 422.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tickerlab.core.econ_registry import valider_outputs

Module = Literal['univarie', 'var', 'ruptures']
Freq   = Literal['daily', 'weekly', 'monthly']
Price  = Literal['adj_close', 'close']


class AnalyseRequest(BaseModel):
    """Requete POST /api/v1/analyses (contrat strict, `extra='forbid'`)."""
    model_config = ConfigDict(extra='forbid')

    symbol:    str
    module:    Module = 'univarie'
    date_from: str
    date_to:   str
    freq:      Freq  = 'daily'
    price:     Price = 'adj_close'
    outputs:   list[str] = Field(default_factory=list)

    @field_validator('symbol')
    @classmethod
    def _symbol_non_vide(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError('symbol vide')
        return str(v).strip()

    @field_validator('outputs')
    @classmethod
    def _outputs_connus(cls, v):
        # Leve ValueError (=> 422) sur toute cle inconnue ; conserve l'ordre de clic.
        return valider_outputs(v)

    def to_internal_payload(self) -> dict:
        """Traduit la requete v1 vers le payload interne (web/mapping)."""
        return {
            'ticker':  self.symbol,
            'from':    self.date_from,
            'to':      self.date_to,
            'freq':    self.freq,
            'price':   'adjclose' if self.price == 'adj_close' else 'close',
            'outputs': list(self.outputs),
            'module':  self.module,
        }

    def hash_scientifique(self) -> str:
        """Hash stable des cles scientifiques normalisees (idempotence du job_id).

        Deux requetes identiques sur ces cles => meme hash => meme job_id tant que
        le job est en memoire. `outputs` est trie : l'ordre de clic ne change pas
        l'identite scientifique de l'analyse.
        """
        cle = {
            'symbol':    self.symbol,
            'module':    self.module,
            'date_from': self.date_from,
            'date_to':   self.date_to,
            'freq':      self.freq,
            'price':     self.price,
            'outputs':   sorted(self.outputs),
        }
        blob = json.dumps(cle, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]
