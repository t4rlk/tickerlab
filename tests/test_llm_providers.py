# -*- coding: utf-8 -*-
"""Tests Phase 3.3-A : abstraction LLM providers + defense anti-hallucination.

Tests :
  1. fabriquer_provider retourne AnthropicProvider par defaut
  2. fabriquer_provider retourne OpenAICompatibleProvider pour groq
  3. Fournisseur inconnu leve ValueError
  4. AnthropicProvider.generer() sans cle env leve RuntimeError (sans appel reseau)
  5. _substituer_variables : cle absente -> '[DONNEE MANQUANTE]'
"""
import os
import pytest

from tickerlab.utils.llm_providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    fabriquer_provider,
    LLMProvider,
)
from tickerlab.utils.ai_writer import _substituer_variables


# ── Test 1 : provider par defaut = Anthropic ─────────────────────────────────

def test_fabriquer_provider_defaut_anthropic():
    config = {}  # pas de section ai_writer -> defaut anthropic
    provider = fabriquer_provider(config)
    assert isinstance(provider, AnthropicProvider)


# ── Test 2 : provider groq ────────────────────────────────────────────────────

def test_fabriquer_provider_groq():
    config = {
        'ai_writer': {
            'fournisseur': 'groq',
            'modele': 'llama-3.3-70b-versatile',
            'env_key': 'GROQ_API_KEY',
        }
    }
    provider = fabriquer_provider(config)
    assert isinstance(provider, OpenAICompatibleProvider)
    # Verifie que le provider satisfait le Protocol LLMProvider
    assert isinstance(provider, LLMProvider)


# ── Test 3 : fournisseur inconnu leve ValueError ──────────────────────────────

def test_fabriquer_provider_inconnu():
    config = {'ai_writer': {'fournisseur': 'xyzzy'}}
    with pytest.raises(ValueError, match="Fournisseur inconnu"):
        fabriquer_provider(config)


# ── Test 4 : AnthropicProvider sans cle env leve RuntimeError ────────────────
# La cle est verifiee AVANT l'import anthropic — le test passe meme si le
# package anthropic n'est pas installe.

def test_anthropic_provider_sans_cle_env():
    provider = AnthropicProvider(modele='claude-sonnet-4-6', env_key='_PEA_TEST_ABSENT_KEY_')
    os.environ.pop('_PEA_TEST_ABSENT_KEY_', None)
    with pytest.raises(RuntimeError, match="_PEA_TEST_ABSENT_KEY_"):
        provider.generer(system='sys', user='hello', max_tokens=10)


# ── Test 6 : rate-limit 429 avec attente longue -> echec RAPIDE ──────────────

def test_openai_compatible_rate_limit_echec_rapide(monkeypatch):
    """Un 429 avec Retry-After long ne doit PAS geler le thread : echec immediat.

    Regression : avant, le provider dormait la duree du Retry-After (ex. 1881s),
    gelant le worker web. Desormais il leve une RuntimeError 'Limite API atteinte'.
    """
    import time as _time
    import openai

    class _FakeResp:
        status_code = 429
        headers = {'Retry-After': '1881'}

    class _FakeErr(Exception):
        status_code = 429
        response = _FakeResp()

    class _FakeClient:
        def __init__(self, **kw):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise _FakeErr('rate limit reached (tokens per day)')

    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: _FakeClient())
    monkeypatch.setenv('GROQ_API_KEY', 'gsk_fake_test')
    # Garde-fou : aucun sommeil long ne doit etre declenche.
    monkeypatch.setattr(_time, 'sleep',
                        lambda s: (_ for _ in ()).throw(AssertionError(f'sleep({s}) interdit')))

    provider = OpenAICompatibleProvider(
        fournisseur='groq', modele='llama-3.3-70b-versatile',
        base_url='https://api.groq.com/openai/v1', env_key='GROQ_API_KEY',
    )
    import time as _t
    t0 = _t.time()
    with pytest.raises(RuntimeError, match='Limite API atteinte'):
        provider.generer(system='s', user='u', max_tokens=10)
    assert _t.time() - t0 < 2.0, 'le provider a gele au lieu d echouer vite'


# ── Test 5 : cle absente dans contexte -> '[DONNEE MANQUANTE]' ───────────────

def test_contexte_donnee_manquante_devient_placeholder():
    """Defense structurelle anti-hallucination : substitution explicite."""
    contexte = {'TICKER': 'BZ=F', 'N_OBS': '500'}
    texte = "Le ticker est {{TICKER}} avec {{N_OBS}} obs et AIC={{GARCH_AIC}}."
    resultat = _substituer_variables(texte, contexte)

    assert 'BZ=F' in resultat
    assert '500' in resultat
    # La cle absente GARCH_AIC doit produire exactement le placeholder
    assert '[DONNÉE MANQUANTE]' in resultat
    # Ne doit PAS contenir une version mal formee ou vide
    assert '{{GARCH_AIC}}' not in resultat
    assert 'MANQUANT:' not in resultat
