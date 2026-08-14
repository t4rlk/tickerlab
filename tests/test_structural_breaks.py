# -*- coding: utf-8 -*-
"""
Tests — Ruptures structurelles ICSS (Condition 3).

Aiguillage des modes :
  1. mode='diagnostic' retourne un dict valide
  2. mode='diagnostic' NE modifie PAS la persistance (non-régression)
  3. mode='diagnostic' n'expose PAS les clés persistance_avant/apres
  4. mode='integrate'  expose les clés persistance_avant ET persistance_apres
  5. mode='off' et enabled=false retournent None

ICSS : détection et localisation d'une rupture connue.

Estimation à omega par régime (Hillebrand 2005) : récupération de la
persistance simulée, effet Lamoureux-Lastrapes sur le GARCH global, et
dégénérescence d'un régime trop court.

Garde-fou : échelle de la série produite par la fixture.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.structural_breaks import (
    analyser_icss_pipeline,
    estimer_garch_omega_par_regime,
    inclan_tiao_icss,
    selectionner_ruptures_espacees,
)

# Persistance réellement simulée par la fixture, des DEUX côtés de la rupture.
PERSISTANCE_SIMULEE = 0.08 + 0.88   # alpha + beta
RUPTURE_SIMULEE     = 300           # indice de la rupture dans la fixture


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_garch_with_breaks(n: int = 600, seed: int = 0):
    """
    Série GARCH(1,1) avec rupture de variance forte à t=300 (variance x9).
    La rupture est suffisamment large pour que l'ICSS la détecte avec certitude.

    Le multiplicateur porte sur omega — saut de niveau de la variance
    inconditionnelle omega/(1-alpha-beta) — et NON sur eps. Appliqué à eps, il
    serait réinjecté au carré dans la récurrence de h et porterait la
    persistance effective à 9*alpha+beta = 1.60 > 1 : la variance exploserait
    géométriquement au lieu de sauter d'un palier.

    Échelle : rendements quotidiens en pourcentage, ecart-type ~1.1 % avant la
    rupture et ~3.4 % apres. Persistance alpha+beta = 0.96 < 1 des deux cotes.
    """
    from arch import arch_model
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.08, 0.88
    eps = np.zeros(n)
    h   = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    var_mult = np.ones(n)
    var_mult[300:] = 9.0  # x9 variance → x3 volatility, rupture nette

    for i in range(1, n):
        h[i]   = omega * var_mult[i] + alpha * eps[i - 1] ** 2 + beta * h[i - 1]
        eps[i] = rng.standard_normal() * np.sqrt(h[i])

    series   = pd.Series(eps)
    fit      = arch_model(series, vol='Garch', p=1, o=0, q=1, dist='t').fit(disp='off')
    best_row = pd.Series({
        'modele': 'GARCH', 'vol': 'Garch',
        'p': 1, 'o': 0, 'q': 1, 'dist': 't',
        'AIC': fit.aic, 'BIC': fit.bic, 'power': float('nan'),
    })
    return series, fit, best_row


def _cfg_diag() -> dict:
    return {
        'structural_breaks': {
            'enabled': True,
            'mode': 'diagnostic',
            'methode': 'icss',
            'seuil_persistance_alerte': 0.97,
            'seuil_n_ruptures_alerte': 3,
            'max_regimes': 6,
            'min_obs_regime': 50,
        }
    }


def _cfg_integrate() -> dict:
    cfg = _cfg_diag()
    cfg['structural_breaks']['mode'] = 'integrate'
    return cfg


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_icss_detects_known_break():
    """ICSS détecte une rupture connue ET la localise correctement.

    La tolérance de +/-50 observations est large au regard de la précision
    réelle de l'algorithme sur ce jeu (rupture localisée en 199 pour une
    rupture simulée en 200) : elle absorbe les variations de version de
    numpy/scipy sans vider l'assertion de son contenu — sans borne de
    localisation, une rupture détectée n'importe où validerait le test.
    """
    rng = np.random.default_rng(7)
    eps1 = rng.standard_normal(200) * 1.0
    eps2 = rng.standard_normal(200) * 5.0   # rupture nette à t=200
    eps_sq = np.concatenate([eps1, eps2]) ** 2
    breaks = inclan_tiao_icss(eps_sq)
    assert len(breaks) >= 1, f"ICSS devrait détecter ≥1 rupture, obtenu {len(breaks)}"

    ecart = min(abs(b - 200) for b in breaks)
    assert ecart <= 50, (
        f"rupture la plus proche a {ecart} observations de t=200 "
        f"(ruptures detectees : {breaks})"
    )


def test_fixture_echelle_realiste():
    """La fixture doit produire des rendements quotidiens d'échelle plausible.

    Garde-fou : appliquer le multiplicateur de variance a eps plutot qu'a omega
    le reinjecterait au carre dans la recurrence de h, portant la persistance
    effective a 9*alpha+beta = 1.60 > 1 et l'echelle de la serie a ~1e44. Cette
    assertion empeche que la correction soit defaite silencieusement.
    """
    series, _, _ = _make_garch_with_breaks()
    ecart_type = float(series.std())
    assert 0.5 < ecart_type < 5.0, (
        f"ecart-type = {ecart_type:.4g} % hors de [0.5, 5] : la fixture ne "
        f"produit plus des rendements quotidiens plausibles"
    )


# ── Sélection des ruptures par espacement ────────────────────────────────────

def test_selection_ruptures_espacement_minimal():
    """Toute rupture retenue laisse >= min_obs_regime observations de chaque côté.

    Cas realiste : ruptures agglutinees en debut de serie, comme l'ICSS en
    produit sur donnees reelles. Une troncature par rang retiendrait les cinq
    premieres et creerait des regimes de 5 a 20 observations.
    """
    bps = [5, 12, 18, 40, 60, 300, 305, 900, 1180]
    n   = 1200
    ret = selectionner_ruptures_espacees(bps, n, min_obs_regime=50, max_regimes=6)

    bornes = [0] + ret + [n]
    tailles = [bornes[i + 1] - bornes[i] for i in range(len(bornes) - 1)]
    assert all(t >= 50 for t in tailles), f"regimes trop courts : {tailles}"
    assert 1180 not in ret, "une rupture a 20 obs de la fin doit etre ecartee"
    assert ret == sorted(ret)


def test_max_dummies_deprecie_converti_avec_avertissement():
    """'max_dummies' est lu comme alias déprécié et converti en régimes.

    Ancienne sémantique : nombre de RUPTURES. Nouvelle : nombre de RÉGIMES.
    max_dummies=5 doit donc valoir max_regimes=6, et le signaler.
    Ce chemin est exercé ICI uniquement — les autres fixtures utilisent la clé
    courante, afin qu'une dépréciation ne devienne pas silencieuse à force
    d'être partout.

    Le handler est attaché directement au logger du module plutôt que via
    `caplog` : la configuration de journalisation du projet coupe la
    propagation vers la racine, et `caplog` ne verrait alors rien dès qu'un
    autre test a initialisé cette configuration.
    """
    import logging

    series, fit, best = _make_garch_with_breaks()
    cfg = {'structural_breaks': {'enabled': True, 'mode': 'integrate',
                                 'seuil_persistance_alerte': 0.97,
                                 'seuil_n_ruptures_alerte': 3,
                                 'max_dummies': 5}}

    captures: list = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captures.append(record.getMessage())

    logger  = logging.getLogger('tickerlab.core.structural_breaks')
    handler = _Capture(level=logging.WARNING)
    niveau  = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        res_alias = analyser_icss_pipeline(series, fit, cfg, garch_best=best)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(niveau)

    messages = ' | '.join(captures)
    assert 'max_dummies' in messages and 'deprecie' in messages, (
        f"l'alias doit emettre un WARNING explicite, journal : {messages}"
    )
    assert '6 regimes' in messages, f"conversion 5 -> 6 non annoncee : {messages}"

    # Résultat identique à celui obtenu avec la clé courante.
    cfg_neuf = {'structural_breaks': {'enabled': True, 'mode': 'integrate',
                                      'seuil_persistance_alerte': 0.97,
                                      'seuil_n_ruptures_alerte': 3,
                                      'max_regimes': 6}}
    res_neuf = analyser_icss_pipeline(series, fit, cfg_neuf, garch_best=best)
    assert res_alias['n_regimes'] == res_neuf['n_regimes']
    assert res_alias.get('ruptures_retenues') == res_neuf.get('ruptures_retenues')


def test_selection_ruptures_plafond_regimes():
    """Le plafond porte sur le nombre de RÉGIMES : au plus max_regimes-1 ruptures."""
    bps = list(range(100, 2000, 100))          # 19 ruptures largement espacees
    ret = selectionner_ruptures_espacees(bps, 2500, min_obs_regime=50, max_regimes=4)
    assert len(ret) == 3, f"4 regimes => 3 ruptures au plus, obtenu {ret}"


def test_selection_ruptures_coherente_avec_degenerescence():
    """Les ruptures retenues ne produisent aucun régime jugé dégénéré.

    C'est l'invariant qui lie le seuil de selection a celui de degenerescence :
    selectionner une rupture aussitot declaree degeneree n'aurait pas de sens.
    """
    series, _, _ = _make_garch_with_breaks()
    n = len(series)
    bps = [3, 8, 15, 120, 128, 300, 560, 598]
    ret = selectionner_ruptures_espacees(bps, n, min_obs_regime=50, max_regimes=6)

    est = estimer_garch_omega_par_regime(series, ret, dist='normal')
    assert not any('regime' in d and 'taille' in d for d in est['degeneracies']), (
        f"selection incoherente avec la detection de degenerescence : "
        f"{est['degeneracies']}"
    )


# ── Estimation à omega par régime (Hillebrand 2005) ──────────────────────────

def test_omega_par_regime_retrouve_persistance():
    """L'estimation à omega par régime retrouve la persistance réellement simulée.

    La fixture simule alpha+beta = 0.96 des DEUX cotes d'une rupture de niveau
    de variance. Un omega par regime doit ramener la persistance pres de cette
    valeur, nettement en dessous de celle du GARCH global.
    """
    series, fit, _ = _make_garch_with_breaks()
    a = float(fit.params['alpha[1]'])
    b = float(fit.params['beta[1]'])
    pers_globale = a + b

    est = estimer_garch_omega_par_regime(
        series, [RUPTURE_SIMULEE], dist='normal', alpha0=a, beta0=b)

    assert est['converged'],      f"SLSQP non converge : {est['degeneracies']}"
    assert not est['degenerate'], f"degenerescences : {est['degeneracies']}"
    assert est['n_regimes'] == 2
    assert len(est['omega']) == 2

    assert est['persistance'] < pers_globale - 0.01, (
        f"persistance corrigee {est['persistance']:.4f} pas assez inferieure "
        f"a la persistance globale {pers_globale:.4f}"
    )
    assert abs(est['persistance'] - PERSISTANCE_SIMULEE) < 0.05, (
        f"persistance corrigee {est['persistance']:.4f} loin de la valeur "
        f"simulee {PERSISTANCE_SIMULEE:.4f}"
    )
    # La rupture multiplie la variance inconditionnelle par 9 : omega doit suivre.
    assert est['omega'][1] > 3.0 * est['omega'][0], (
        f"omega par regime = {est['omega']} — le saut de niveau n'est pas capte"
    )
    assert math.isfinite(est['loglik']) and math.isfinite(est['aic'])


def test_garch_global_surestime_persistance():
    """Effet Lamoureux-Lastrapes : le GARCH global gonfle la persistance.

    C'est la raison d'etre du mode 'integrate' — sans quoi la correction serait
    sans objet.
    """
    series, fit, _ = _make_garch_with_breaks()
    a = float(fit.params['alpha[1]'])
    b = float(fit.params['beta[1]'])
    pers_globale = a + b

    est = estimer_garch_omega_par_regime(
        series, [RUPTURE_SIMULEE], dist='normal', alpha0=a, beta0=b)

    assert pers_globale > PERSISTANCE_SIMULEE, (
        f"persistance globale {pers_globale:.4f} devrait exceder la valeur "
        f"simulee {PERSISTANCE_SIMULEE:.4f} (rupture non modelisee)"
    )
    assert pers_globale > est['persistance'], (
        f"globale {pers_globale:.4f} vs omega/regime {est['persistance']:.4f}"
    )


def test_regime_trop_court_signale_degenerescence():
    """Un régime de moins de 50 observations est signalé comme dégénéré."""
    series, _, _ = _make_garch_with_breaks()

    est = estimer_garch_omega_par_regime(series, [10], dist='normal')

    assert est['degenerate'] is True
    assert any('regime 0' in d for d in est['degeneracies']), (
        f"la taille du regime 0 (10 obs) doit etre signalee : {est['degeneracies']}"
    )


def test_regime_trop_court_pas_de_persistance_fausse(monkeypatch):
    """Mode 'integrate' : aucune persistance_apres n'est renvoyée si dégénéré.

    Mieux vaut une clé absente qu'une persistance fausse presentee comme
    corrigee.
    """
    import tickerlab.core.structural_breaks as SB

    series, fit, best = _make_garch_with_breaks()
    monkeypatch.setattr(SB, 'inclan_tiao_icss', lambda *a, **k: [10])

    result = SB.analyser_icss_pipeline(series, fit, _cfg_integrate(), garch_best=best)

    assert result is not None
    assert result['n_breaks'] == 1
    assert result['persistance_apres_degenere'] is True
    assert 'persistance_apres' not in result, (
        "une persistance degeneree ne doit pas etre exposee"
    )
    assert 'persistance_avant' in result
    assert result['degeneracies_apres']


def test_diagnostic_returns_valid_dict():
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(series, fit, _cfg_diag(), garch_best=best)
    assert result is not None
    assert result['mode'] == 'diagnostic'
    assert 'n_breaks' in result
    assert 'indices' in result
    assert 'warning_lamoureux' in result
    assert isinstance(result['n_breaks'], int)
    assert isinstance(result['indices'], list)


def test_diagnostic_non_regression():
    """mode='diagnostic' ne modifie PAS les paramètres du modèle GARCH."""
    series, fit, best = _make_garch_with_breaks()
    pers_avant = float(
        fit.params.get('alpha[1]', 0) + fit.params.get('beta[1]', 0)
    )
    analyser_icss_pipeline(series, fit, _cfg_diag(), garch_best=best)
    pers_apres = float(
        fit.params.get('alpha[1]', 0) + fit.params.get('beta[1]', 0)
    )
    assert pers_avant == pers_apres, (
        "mode='diagnostic' NE doit PAS modifier les paramètres GARCH estimés"
    )


def test_diagnostic_no_persistance_keys():
    """mode='diagnostic' n'expose PAS persistance_avant / persistance_apres."""
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(series, fit, _cfg_diag(), garch_best=best)
    assert 'persistance_avant' not in result
    assert 'persistance_apres' not in result


def test_integrate_persistance_keys():
    """mode='integrate' : contrat de sortie, selon que l'estimation dégénère ou non.

    ASSERTION MODIFIEE. L'ancienne version exigeait inconditionnellement
    'persistance_apres'. Le mode omet desormais cette cle lorsque l'estimation
    a omega par regime degenere — mieux vaut une cle absente qu'une persistance
    fausse presentee comme corrigee. Le test verifie donc les DEUX branches du
    contrat plutot qu'une seule.

    Le pytest.skip sur n_breaks == 0 est remplace par une assertion : sur la
    fixture corrigee l'ICSS detecte la rupture, et un echec de detection doit
    etre rouge, pas silencieusement ignore.
    """
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(series, fit, _cfg_integrate(), garch_best=best)

    assert result is not None
    assert result['n_breaks'] > 0, "ICSS ne detecte plus la rupture de la fixture"

    assert 'persistance_avant' in result, "persistance_avant manquante en mode integrate"
    assert isinstance(result['persistance_avant'], float)
    assert not math.isnan(result['persistance_avant'])

    assert result['methode'] == 'omega_par_regime'
    assert result['n_regimes'] >= 2
    assert len(result['omega_par_regime']) == result['n_regimes']
    assert 'dummies_injectees' not in result, (
        "cle supprimee : plus aucune dummy n'est injectee depuis la refonte"
    )

    if result['persistance_apres_degenere']:
        assert 'persistance_apres' not in result, (
            "une persistance degeneree ne doit pas etre exposee"
        )
        assert result['degeneracies_apres'], "le motif de degenerescence doit etre donne"
    else:
        assert 'persistance_apres' in result
        assert isinstance(result['persistance_apres'], float)
        assert not math.isnan(result['persistance_apres'])


def test_mode_off_returns_none():
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(
        series, fit,
        {'structural_breaks': {'enabled': True, 'mode': 'off'}},
        garch_best=best,
    )
    assert result is None


def test_disabled_returns_none():
    series, fit, best = _make_garch_with_breaks()
    result = analyser_icss_pipeline(
        series, fit,
        {'structural_breaks': {'enabled': False}},
        garch_best=best,
    )
    assert result is None
