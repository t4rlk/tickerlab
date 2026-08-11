# -*- coding: utf-8 -*-
"""Abstraction multi-providers LLM (Phase 3.3).

SECURITE : aucune cle API n'est jamais stockee dans ce module.
config['ai_writer']['env_key'] contient uniquement le NOM de la variable
d'environnement a lire — jamais sa valeur.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

_log = logging.getLogger('tickerlab.llm_providers')


@dataclass
class LLMResponse:
    contenu: str
    tokens_in: int
    tokens_out: int
    provider: str
    modele: str


@runtime_checkable
class LLMProvider(Protocol):
    def generer(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        ...


# Specificites par fournisseur (comportements non-standard de l'API)
_QUIRKS: dict[str, dict] = {
    'anthropic': {},
    'groq':      {'retry_after_header': True},   # Groq renvoie Retry-After sur 429
    'gemini':    {'usage_fallback': True},         # Gemini: usage parfois None
    'ollama':    {'dummy_api_key': 'ollama'},      # Ollama n'a pas besoin de vraie cle
    'openai':    {},
}

_BASE_URLS: dict[str, str] = {
    'groq':   'https://api.groq.com/openai/v1',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/',
    'ollama': 'http://localhost:11434/v1',
    'openai': 'https://api.openai.com/v1',
}

# Plafond du backoff entre tentatives. Au-dela (typiquement un quota journalier
# qui renvoie Retry-After de plusieurs minutes), on echoue VITE plutot que de
# geler le thread : attendre ne debloque pas un quota epuise dans la requete.
_BACKOFF_CAP_S = 30


def _statut_http(exc) -> Optional[int]:
    """Code HTTP d'une exception SDK OpenAI/Anthropic (None si indisponible)."""
    code = getattr(exc, 'status_code', None)
    if code is None:
        resp = getattr(exc, 'response', None)
        code = getattr(resp, 'status_code', None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_s(exc) -> Optional[int]:
    """Valeur de l'en-tete Retry-After en secondes (None si absente/illisible)."""
    resp = getattr(exc, 'response', None)
    headers = getattr(resp, 'headers', None)
    if not headers:
        return None
    try:
        ra = headers.get('Retry-After')
        return int(float(ra)) if ra is not None else None
    except (TypeError, ValueError):
        return None


class AnthropicProvider:
    """Provider Anthropic via SDK officiel (comportement Phase 3.2 preserve)."""

    def __init__(self, modele: str, env_key: str = 'ANTHROPIC_API_KEY') -> None:
        self._modele = modele
        self._env_key = env_key

    def generer(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        # Verifier la cle AVANT l'import pour que l'erreur soit explicite
        # meme quand le package anthropic n'est pas installe.
        api_key = os.environ.get(self._env_key)
        if not api_key:
            raise RuntimeError(
                f"Variable d'env {self._env_key!r} non definie. "
                f"Definir avant le lancement, jamais dans le code."
            )
        import anthropic  # import local : evite ModuleNotFoundError au chargement
        client = anthropic.Anthropic(api_key=api_key)

        for tentative in range(3):
            try:
                resp = client.messages.create(
                    model=self._modele,
                    max_tokens=max_tokens,
                    temperature=0,
                    system=system,
                    messages=[{'role': 'user', 'content': user}],
                )
                return LLMResponse(
                    contenu=resp.content[0].text,
                    tokens_in=resp.usage.input_tokens,
                    tokens_out=resp.usage.output_tokens,
                    provider='anthropic',
                    modele=self._modele,
                )
            except Exception as exc:
                attente = 2 ** tentative
                _log.warning('  [Anthropic] Erreur (t.%d/3): %s — attente %ds',
                             tentative + 1, exc, attente)
                time.sleep(attente)
        raise RuntimeError('[Anthropic] Echec definitif apres 3 tentatives.')


class OpenAICompatibleProvider:
    """Provider OpenAI-compatible : Groq, Gemini, Ollama, OpenAI."""

    def __init__(self, fournisseur: str, modele: str,
                 base_url: str, env_key: str) -> None:
        self._fournisseur = fournisseur
        self._modele = modele
        self._base_url = base_url
        self._env_key = env_key
        self._quirks = _QUIRKS.get(fournisseur, {})

    def generer(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        from openai import OpenAI

        dummy_key = self._quirks.get('dummy_api_key')
        api_key = os.environ.get(self._env_key) or dummy_key
        if not api_key:
            raise RuntimeError(
                f"Variable d'env {self._env_key!r} non definie. "
                f"Definir avant le lancement, jamais dans le code."
            )

        client = OpenAI(api_key=api_key, base_url=self._base_url)

        for tentative in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self._modele,
                    max_tokens=max_tokens,
                    temperature=0,
                    messages=[
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': user},
                    ],
                )
                usage = resp.usage
                if self._quirks.get('usage_fallback') and usage is None:
                    tok_in, tok_out = 0, 0
                else:
                    tok_in  = usage.prompt_tokens if usage else 0
                    tok_out = usage.completion_tokens if usage else 0

                return LLMResponse(
                    contenu=resp.choices[0].message.content,
                    tokens_in=tok_in,
                    tokens_out=tok_out,
                    provider=self._fournisseur,
                    modele=self._modele,
                )
            except Exception as exc:
                statut = _statut_http(exc)
                ra = _retry_after_s(exc) if self._quirks.get('retry_after_header') else None

                # Quota/limite atteinte avec attente longue : echec RAPIDE.
                # Inutile (et bloquant) de dormir plusieurs minutes dans la requete.
                if statut == 429 and ra is not None and ra > _BACKOFF_CAP_S:
                    raise RuntimeError(
                        f'[{self._fournisseur}] Limite API atteinte (429). '
                        f'Reessayer dans ~{ra}s (quota). Detail : {exc}'
                    ) from exc

                attente = min(2 ** tentative, _BACKOFF_CAP_S)
                if ra is not None:
                    attente = min(max(attente, ra), _BACKOFF_CAP_S)
                _log.warning('  [%s] Erreur (t.%d/3): %s — attente %ds',
                             self._fournisseur, tentative + 1, exc, attente)
                time.sleep(attente)
        raise RuntimeError(
            f'[{self._fournisseur}] Echec definitif apres 3 tentatives.'
        )


def fabriquer_provider(config: dict) -> LLMProvider:
    """Instancie le bon provider depuis config['ai_writer'].

    config['ai_writer'] attendu (exemple pour Groq) :
        fournisseur: groq
        modele: llama-3.3-70b-versatile
        env_key: GROQ_API_KEY          # nom de la variable, jamais la valeur
        max_tokens: 4000
    """
    cfg = config.get('ai_writer', {})
    # Fallback retrocompatible : config['ai']['model'] existait en Phase 2
    fournisseur = str(cfg.get('fournisseur', 'anthropic')).lower()

    if fournisseur == 'anthropic':
        modele_defaut = config.get('ai', {}).get('model', 'claude-sonnet-4-6')
        return AnthropicProvider(
            modele=str(cfg.get('modele', modele_defaut)),
            env_key=str(cfg.get('env_key', 'ANTHROPIC_API_KEY')),
        )

    if fournisseur not in _BASE_URLS:
        raise ValueError(
            f"Fournisseur inconnu : {fournisseur!r}. "
            f"Valides : anthropic, {', '.join(_BASE_URLS)}"
        )

    return OpenAICompatibleProvider(
        fournisseur=fournisseur,
        modele=str(cfg.get('modele', '')),
        base_url=str(cfg.get('base_url', _BASE_URLS[fournisseur])),
        env_key=str(cfg.get('env_key', f'{fournisseur.upper()}_API_KEY')),
    )
