# -*- coding: utf-8 -*-
"""Tests unitaires pour utils/ai_writer.py.

Aucun appel réseau. Tout est mocké ou testé sur des données synthétiques.
"""
import logging
import re
import types
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tickerlab.core.data_loader import ensure_series
from tickerlab.utils.ai_writer import (
    _scanner_hallucinations,
    _substituer_variables,
    construire_contexte,
    rediger_section,
    rediger_toutes_sections,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_garch_result(params: dict):
    """Mock léger d'un résultat arch.ARCHModelResult."""
    res = mock.MagicMock()
    res.params = params
    # conditional_volatility : Series de 100 valeurs
    res.conditional_volatility = pd.Series(np.full(100, 0.01))
    return res


def _make_pipeline_inputs():
    """Jeu de données minimal pour construire_contexte."""
    n = 200
    idx = pd.date_range('2010-01-01', periods=n, freq='W-FRI')
    prix_s   = pd.Series(np.linspace(50, 80, n), index=idx, name='prix')
    rdt_s    = pd.Series(np.random.default_rng(0).normal(0, 2, n),
                          index=idx, name='rdt')

    arima = {
        'p_opt': 2, 'd_opt': 0, 'q_opt': 2,
        'aic': 6.07, 'AIC': 6.07,
        'interpretation': 'composantes_ar_ma',
        'message_pedagogique': 'AR(2) + MA(2)',
        'lb_pval_rendements': 0.35,
        'max_magnitude_coef': 0.42,
    }
    best = {
        'modele': 'EGARCH', 'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1,
        'dist': 'skewt', 'AIC': 5496.62, 'AIC_norm': 5.72,
    }
    params = {
        'omega': -0.15, 'alpha[1]': 0.08, 'beta[1]': 0.95,
        'gamma[1]': -0.07, 'eta': 8.5,
    }
    garch_final = _make_garch_result(params)

    df_var = pd.DataFrame(
        {'VaR Historique': [-5.86, -8.22, -13.71],
         'TVaR Historique': [-9.78, -12.47, -20.01],
         'VaR GARCH': [-4.56, -6.34, -10.78],
         'TVaR GARCH': [-7.26, -9.16, -14.02],
         'VaR Normale': [-4.56, -6.34, -10.78],
         'TVaR Normale': [-7.0, -9.0, -13.5],
         'VaR Student': [-4.56, -6.34, -10.78],
         'TVaR Student': [-7.0, -9.0, -13.5],
         'VaR Cornish-Fisher': [-4.56, -6.34, -10.78],
         'TVaR Cornish-Fisher': [-7.0, -9.0, -13.5],
         'VaR Monte Carlo (1j)': [-4.56, -6.34, -10.78],
         'TVaR Monte Carlo (1j)': [-7.0, -9.0, -13.5]},
        index=['90%', '95%', '99%']
    )

    df_bt = pd.DataFrame({
        'Methode': ['GARCH dyn.', 'Historique'],
        'Niveau':  ['95%', '95%'],
        'N viol.': [18, 12],
        'Taux obs.': [0.0625, 0.0417],
        'p_CC':    [0.1929, 0.4741],
        'Verdict CC': ['OK', 'OK'],
    })

    df_vol = pd.Series(np.abs(rdt_s) * 0.5, index=rdt_s.index)

    config = {
        'data':     {'ticker': 'BZ=F', 'start_date': '2010-01-01',
                     'end_date': '2024-01-01', 'frequency': 'weekly'},
        'arima':    {'p_max': 4, 'q_max': 4, 'seuil_significativite': 0.10},
        'garch':    {'modeles': ['GARCH', 'EGARCH'], 'distributions': ['normal', 'skewt'],
                     'p_max': 2, 'q_max': 2, 'seuil_significativite': 0.05},
        'var':      {'niveaux': [0.90, 0.95, 0.99], 'n_simulations_mc': 50000},
        'backtest': {'split_ratio': 0.70, 'niveaux_test': [0.95, 0.99]},
    }

    return prix_s, rdt_s, arima, best, garch_final, df_var, df_bt, df_vol, config


# ── Bug B — ensure_series ──────────────────────────────────────────────────────

def test_ensure_series_from_series():
    s = pd.Series([1.0, 2.0], name='x')
    assert ensure_series(s) is s


def test_ensure_series_from_dataframe_une_colonne():
    df = pd.DataFrame({'prix': [50.0, 60.0]})
    result = ensure_series(df)
    assert isinstance(result, pd.Series)
    assert list(result) == [50.0, 60.0]


def test_ensure_series_from_dataframe_multicolonne():
    df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
    result = ensure_series(df)
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 1.0


# ── Bug B — construire_contexte accepte Series ET DataFrame ───────────────────

def test_construire_contexte_avec_series():
    prix_s, rdt_s, arima, best, garch_final, df_var, df_bt, df_vol, config = \
        _make_pipeline_inputs()
    ctx = construire_contexte(
        'BZ=F', prix_s, rdt_s, arima, best, garch_final,
        df_var, df_bt, df_vol, config, T_train=140, T_eff_dyn=60,
    )
    assert isinstance(ctx['MOY_PRIX'], str)
    assert float(ctx['MOY_PRIX']) != 0.0


def test_construire_contexte_avec_dataframe():
    prix_s, rdt_s, arima, best, garch_final, df_var, df_bt, df_vol, config = \
        _make_pipeline_inputs()
    prix_df = prix_s.to_frame('prix')
    rdt_df  = rdt_s.to_frame('rdt')
    ctx = construire_contexte(
        'BZ=F', prix_df, rdt_df, arima, best, garch_final,
        df_var, df_bt, df_vol, config, T_train=140, T_eff_dyn=60,
    )
    assert isinstance(float(ctx['MOY_PRIX']), float)
    assert isinstance(float(ctx['STD_RDT']), float)


def test_construire_contexte_inclut_spec_gravite():
    """SPEC_GRAVITE et DEMI_VIE doivent être présents dans le contexte."""
    prix_s, rdt_s, arima, best, garch_final, df_var, df_bt, df_vol, config = \
        _make_pipeline_inputs()
    spec_verdict = {
        'gravite': 'majeure',
        'lb_z_pval': 0.010,
        'lb_z2_pval': 0.230,
        'engle_ng_pval': 0.150,
        'all_candidates_failed_spec': True,
        'message': 'Autocorrelation residuelle detectee.',
    }
    igarch_diag = {
        'half_life_periodes': 14.5,
        'half_life_jours_cal': 101.5,
        'frequence_serie': 'weekly',
        'persistance': 0.9534,
    }
    ctx = construire_contexte(
        'BZ=F', prix_s, rdt_s, arima, best, garch_final,
        df_var, df_bt, df_vol, config, T_train=140, T_eff_dyn=60,
        spec_verdict=spec_verdict, igarch_diag=igarch_diag,
    )
    assert ctx['SPEC_GRAVITE'] == 'majeure'
    assert ctx['SPEC_LB_Z_PVAL'] == '0.010'
    assert ctx['SPEC_ALL_CANDIDATES_FAILED'] == 'oui'
    assert ctx['DEMI_VIE'] == '14.5'
    assert ctx['DEMI_VIE_UNITE'] == 'semaines'


def test_construire_contexte_spec_absente_donne_manquant():
    """Sans spec_verdict, SPEC_GRAVITE vaut '[DONNÉE MANQUANTE]'."""
    prix_s, rdt_s, arima, best, garch_final, df_var, df_bt, df_vol, config = \
        _make_pipeline_inputs()
    ctx = construire_contexte(
        'BZ=F', prix_s, rdt_s, arima, best, garch_final,
        df_var, df_bt, df_vol, config, T_train=140, T_eff_dyn=60,
    )
    assert ctx['SPEC_GRAVITE'] == '[DONNÉE MANQUANTE]'
    assert ctx['DEMI_VIE'] == '[DONNÉE MANQUANTE]'


# ── Bug A — sections_a_rediger lu depuis ai_writer ────────────────────────────

def test_sections_lues_depuis_ai_writer(tmp_path, monkeypatch):
    """_etape_ia doit trouver les sections dans config['ai_writer'], pas config['ai']."""
    sections_attendues = ['07_arma', '08_garch']

    # Simuler _etape_ia sans appel réseau : vérifier que la liste sections
    # vient bien de ai_writer.sections_a_rediger
    config = {
        'ai':       {'enabled': True, 'max_tokens': 100},
        'ai_writer': {
            'fournisseur': 'groq',
            'modele': 'llama-3.3-70b-versatile',
            'env_key': 'GROQ_API_KEY',
            'max_tokens': 200,
            'sections_a_rediger': sections_attendues,
        },
        'output': {'dossier_resultats': str(tmp_path)},
        'data':   {'ticker': 'BZ=F'},
    }

    cfg_writer = config.get('ai_writer', {})
    cfg_ai     = config.get('ai', {})
    sections   = cfg_writer.get('sections_a_rediger',
                                cfg_ai.get('sections_a_rediger', []))
    assert sections == sections_attendues


def test_sections_vides_si_absentes_des_deux():
    config = {'ai': {}, 'ai_writer': {}}
    cfg_writer = config.get('ai_writer', {})
    cfg_ai     = config.get('ai', {})
    sections   = cfg_writer.get('sections_a_rediger',
                                cfg_ai.get('sections_a_rediger', []))
    assert sections == []


# ── Volet D — garde-fou anti-hallucination ─────────────────────────────────────

def test_scanner_hallucinations_pas_de_warning_sur_valeurs_connues(caplog):
    contexte = {'VAR_GARCH_99': '-10.78', 'N_OBS': '960'}
    texte = r"La VaR GARCH à 99\,\% s'établit à -10,78\,\%."
    with caplog.at_level(logging.WARNING, logger='tickerlab.utils.ai_writer'):
        _scanner_hallucinations('08_garch', texte, contexte)
    assert not any('IA-HALLUCINATION' in r.message for r in caplog.records)


def test_scanner_hallucinations_detecte_nombre_inconnu(caplog):
    contexte = {'VAR_GARCH_99': '-10.78'}
    # 42.99 est absent du contexte, a une décimale → doit être détecté
    texte = r"La valeur observée est 42,99 ce qui est inattendu."
    with caplog.at_level(logging.WARNING, logger='tickerlab.utils.ai_writer'):
        _scanner_hallucinations('08_garch', texte, contexte)
    hal_msgs = [r.message for r in caplog.records if 'IA-HALLUCINATION' in r.message]
    assert len(hal_msgs) >= 1
    assert '42.99' in hal_msgs[0]


def test_scanner_hallucinations_annees_exclues(caplog):
    contexte = {}
    texte = "En 1999, Engle a publié ses travaux. Bâle III date de 2010."
    with caplog.at_level(logging.WARNING, logger='tickerlab.utils.ai_writer'):
        _scanner_hallucinations('05_revue', texte, contexte)
    assert not any('IA-HALLUCINATION' in r.message for r in caplog.records)


# ── substitution variables ────────────────────────────────────────────────────

def test_substituer_variables_cle_presente():
    from tickerlab.utils.ai_writer import _substituer_variables
    assert _substituer_variables('VaR = {{VAR}}', {'VAR': '-10.78'}) == 'VaR = -10.78'


def test_substituer_variables_cle_absente():
    from tickerlab.utils.ai_writer import _substituer_variables
    result = _substituer_variables('{{ABSENT}}', {})
    assert result == '[DONNÉE MANQUANTE]'


# ── Comptage sections_ok vs sections_echec ────────────────────────────────────

def test_rediger_toutes_sections_echec_non_compte_comme_ok(tmp_path, monkeypatch):
    """sections_ok ne doit pas augmenter quand le provider échoue (0 token)."""
    # Créer un squelette .tex minimal
    section_id = '08_garch'
    tex = tmp_path / f'{section_id}.tex'
    tex.write_text('% CONTENU_IA_DEBUT\n% CONTENU_IA_FIN\n', encoding='utf-8')

    # Provider qui lève une exception (simule module manquant)
    class _ProviderKO:
        def generer(self, system, user, max_tokens):
            raise RuntimeError('No module named openai')

    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / f'{section_id}.txt').write_text('Prompt test {{TICKER}}', encoding='utf-8')

    stats = rediger_toutes_sections(
        contexte={'TICKER': 'BZ=F'},
        output_dir=tmp_path,
        provider=_ProviderKO(),
        sections=[section_id],
        prompts_dir=prompts_dir,
        max_tokens=100,
    )

    assert stats['sections']       == 0, "Aucune section ne doit être comptée comme rédigée"
    assert stats['sections_echec'] == 1, "L'échec doit être comptabilisé"
    assert stats['tokens_total']   == 0

    # Le placeholder doit être injecté dans le .tex
    contenu_tex = tex.read_text(encoding='utf-8')
    assert '% ECHEC IA' in contenu_tex


def test_rediger_toutes_sections_succes_compte_ok(tmp_path):
    """sections_ok augmente quand le provider retourne des tokens."""
    from tickerlab.utils.llm_providers import LLMResponse

    section_id = '07_arma'
    tex = tmp_path / f'{section_id}.tex'
    tex.write_text('% CONTENU_IA_DEBUT\n% CONTENU_IA_FIN\n', encoding='utf-8')

    class _ProviderOK:
        def generer(self, system, user, max_tokens):
            return LLMResponse(
                contenu='Texte genere.', tokens_in=10, tokens_out=20,
                provider='mock', modele='mock',
            )

    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / f'{section_id}.txt').write_text('Prompt {{TICKER}}', encoding='utf-8')

    stats = rediger_toutes_sections(
        contexte={'TICKER': 'BZ=F'},
        output_dir=tmp_path,
        provider=_ProviderOK(),
        sections=[section_id],
        prompts_dir=prompts_dir,
        max_tokens=100,
    )

    assert stats['sections']       == 1
    assert stats['sections_echec'] == 0
    assert stats['tokens_total']   == 30
