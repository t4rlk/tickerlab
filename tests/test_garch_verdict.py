# -*- coding: utf-8 -*-
"""Tests — verdict de spécification GARCH (garch_selector.verdict_specification).

Double-verdict du résumé exécutif : gravité aucune/mineure/majeure, persistance
des colonnes de spécification dans df_garch, et promotion mineure→majeure quand
tous les candidats échouent la spécification.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _best_dict_avec_pvals(lb_z=0.8, lb_z2=0.7, engle=0.6,
                          aic=1000.0, p=1, o=0, q=1,
                          modele='GARCH', dist='t'):
    return {
        'lb_z_pval':     lb_z,
        'lb_z2_pval':    lb_z2,
        'engle_ng_pval': engle,
        'AIC':           aic,
        'p': p, 'o': o, 'q': q,
        'modele': modele, 'dist': dist,
    }


def _df_garch_simple(best_aic=1000.0, n=3, tous_passent=True):
    rows = [{'AIC': best_aic + i, 'tous_passent': tous_passent} for i in range(n)]
    return pd.DataFrame(rows)


def _cfg():
    return {'garch': {'seuil_significativite': 0.05, 'tolerance_delta_critere_brut': 4.0}}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_verdict_specification_spec_ok():
    """Tous les p-values > 0.05 -> gravite='aucune'."""
    from tickerlab.core.garch_selector import verdict_specification
    best = _best_dict_avec_pvals(lb_z=0.80, lb_z2=0.70, engle=0.60)
    df   = _df_garch_simple()
    v    = verdict_specification(best, df, _cfg())
    assert v['spec_ok'] is True
    assert v['gravite'] == 'aucune'


def test_verdict_specification_alerte_majeure():
    """
    Test hybride (remplace le test unitaire precedent, faux positif).

    Partie 1 — Integration : utilise la meme serie que test_df_garch_contient_colonnes_spec
    (seed=7, connue pour passer Etage 1). Verifie que :
      a. df_garch contient 'tous_passent' apres selectionner_meilleur (fix principal)
      b. all_candidates_failed_spec coherent avec df_garch reel (lecture reelle)
      c. gravite logiquement coherente avec spec_ok

    Partie 2 — Classification directe : verifie la logique majeure/ALERTE
    sur des p-values injectees directement (independant de la serie).
    Les deux parties ensemble eliminent le faux-positif de l'ancien test.
    """
    from tickerlab.core.garch_selector import (
        grid_search_garch, selectionner_meilleur, verdict_specification
    )

    # ── Partie 1 : integration (meme serie que test_df_garch_contient_colonnes_spec) ─
    rng7 = np.random.default_rng(7)
    n = 500
    h7 = np.ones(n)
    r7 = np.zeros(n)
    for t in range(1, n):
        h7[t] = 0.05 + 0.10 * r7[t-1]**2 + 0.85 * h7[t-1]
        r7[t] = np.sqrt(h7[t]) * rng7.standard_t(df=5)
    serie = pd.Series(r7 * 100)

    cfg_integ = {
        'garch': {
            'modeles': ['GARCH'],
            'distributions': ['t'],
            'p_max': 1, 'q_max': 1,
            'seuil_significativite': 0.05,
            'critere_information': 'BIC',
            'tolerance_delta_critere_brut': 4.0,
            'seuil_igarch': 0.98,
            'score_composite': {'enabled': False},
        }
    }

    df_garch = grid_search_garch(serie, **cfg_integ['garch'])
    best, _, _ = selectionner_meilleur(df_garch, serie, cfg_integ)

    assert 'tous_passent' in df_garch.columns, (
        "df_garch ne contient pas 'tous_passent' apres selectionner_meilleur — "
        "persistance colonnes spec non appliquee (fix latent manquant)"
    )

    best_d = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
    v = verdict_specification(best_d, df_garch, cfg_integ)

    n_pass = int(df_garch['tous_passent'].sum())
    all_failed_in_df = (n_pass == 0)
    assert v['all_candidates_failed_spec'] == all_failed_in_df, (
        f"all_candidates_failed_spec incoherent : verdict={v['all_candidates_failed_spec']}, "
        f"df_garch n_pass={n_pass} — lecture non reelle de la colonne"
    )
    if v['spec_ok']:
        assert v['gravite'] == 'aucune'
    else:
        assert v['gravite'] in ('mineure', 'majeure')

    # ── Partie 2 : classification majeure sur p-values directes ─────────────
    best_maj = _best_dict_avec_pvals(lb_z=0.010, lb_z2=0.003, engle=0.002)
    df_maj   = pd.DataFrame([{'AIC': 1000.0, 'tous_passent': False}])
    v_maj    = verdict_specification(best_maj, df_maj, _cfg())
    assert v_maj['spec_ok'] is False
    assert v_maj['gravite'] == 'majeure'
    assert v_maj['all_candidates_failed_spec'] is True
    assert 'autocorrelation' in v_maj['message'].lower()


def test_df_garch_contient_colonnes_spec():
    """
    Regression : apres grid_search_garch + selectionner_meilleur, df_garch doit
    contenir lb_z_pval, lb_z2_pval, engle_ng_pval, tous_passent.
    Ce test detecte toute regression sur la persistance des colonnes spec.
    """
    from tickerlab.core.garch_selector import (
        grid_search_garch, selectionner_meilleur
    )
    rng = np.random.default_rng(7)
    n = 500
    h = np.ones(n)
    r = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.10 * r[t-1]**2 + 0.85 * h[t-1]
        r[t] = np.sqrt(h[t]) * rng.standard_t(df=5)
    serie = pd.Series(r * 100)

    cfg_reg = {
        'garch': {
            'modeles': ['GARCH'],
            'distributions': ['t'],
            'p_max': 1,
            'q_max': 1,
            'seuil_significativite': 0.05,
            'critere_information': 'BIC',
            'tolerance_delta_critere_brut': 4.0,
            'seuil_igarch': 0.98,
            'score_composite': {'enabled': False},
        }
    }

    df_garch = grid_search_garch(serie, **cfg_reg['garch'])
    selectionner_meilleur(df_garch, serie, cfg_reg)

    for col in ('lb_z_pval', 'lb_z2_pval', 'engle_ng_pval', 'tous_passent'):
        assert col in df_garch.columns, (
            f"df_garch manque la colonne '{col}' apres selectionner_meilleur"
        )


def test_verdict_specification_alerte_mineure():
    """LB z_t p=0.032 (> 0.01 mais < 0.05) -> gravite='mineure'.
    Au moins 1 candidat passe la spec -> pas de promotion majeure."""
    from tickerlab.core.garch_selector import verdict_specification
    best = _best_dict_avec_pvals(lb_z=0.032, lb_z2=0.028, engle=0.041)
    # Un candidat passe (tous_passent=True) -> all_candidates_failed=False -> pas de promotion
    df = pd.DataFrame([
        {'AIC': 1000.0, 'tous_passent': False},
        {'AIC': 1001.0, 'tous_passent': True},
        {'AIC': 1002.0, 'tous_passent': False},
    ])
    v    = verdict_specification(best, df, _cfg())
    assert v['spec_ok'] is False
    assert v['all_candidates_failed_spec'] is False
    assert v['gravite'] == 'mineure'
    assert 'marginal' in v['message']


def test_verdict_specification_colonnes_defensives():
    """Colonnes spec non standard -> all_candidates_failed_spec=False (pas d'erreur)."""
    from tickerlab.core.garch_selector import verdict_specification
    best = _best_dict_avec_pvals(lb_z=0.010, lb_z2=0.003, engle=0.002)
    df   = pd.DataFrame([{'AIC': 1000.0, 'colonne_inconnue': True}])
    v    = verdict_specification(best, df, _cfg())
    assert v['all_candidates_failed_spec'] is False


def test_verdict_promotion_mineure_to_majeure_si_all_failed():
    """
    Sur p_min marginal (>0.01 donc 'mineure' par la regle p-value),
    mais all_candidates_failed_spec=True (tous les candidats B&A echouent),
    la gravite est promue 'majeure' avec message d'asset class non modelisable.

    Scenario reproduit BZ=F ou p_min=0.0103 et 5/5 candidats echouent.
    """
    from tickerlab.core.garch_selector import verdict_specification
    best = _best_dict_avec_pvals(lb_z=0.0103, lb_z2=0.20, engle=0.83)
    df = pd.DataFrame([
        {'AIC': 1000.0, 'tous_passent': False},
        {'AIC': 1001.0, 'tous_passent': False},
        {'AIC': 1002.5, 'tous_passent': False},
        {'AIC': 1003.5, 'tous_passent': False},
    ])
    v = verdict_specification(best, df, _cfg())
    assert v['all_candidates_failed_spec'] is True
    assert v['gravite'] == 'majeure', (
        f"p_min=0.0103 (mineure isolement) MAIS tous candidats echouent -> "
        f"doit etre promu 'majeure', obtenu '{v['gravite']}'"
    )
    assert ('asset class' in v['message'].lower()
            or 'systematique' in v['message'].lower()
            or 'systematique' in v['message'])


def test_verdict_pas_de_promotion_si_seul_modele_echoue():
    """
    Si p_min marginal ('mineure') ET au moins 1 candidat passe la spec
    (all_candidates_failed_spec=False), la gravite reste 'mineure'.
    Pas de promotion abusive.
    """
    from tickerlab.core.garch_selector import verdict_specification
    best = _best_dict_avec_pvals(lb_z=0.032, lb_z2=0.028, engle=0.041)
    df = pd.DataFrame([
        {'AIC': 1000.0, 'tous_passent': False},
        {'AIC': 1001.0, 'tous_passent': True},
        {'AIC': 1002.0, 'tous_passent': False},
    ])
    v = verdict_specification(best, df, _cfg())
    assert v['all_candidates_failed_spec'] is False
    assert v['gravite'] == 'mineure'
