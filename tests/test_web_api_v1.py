# -*- coding: utf-8 -*-
"""Tests de l'API v1 (contrats, mapping du result, signaux, idempotence) + une
intégration bout-en-bout HORS LIGNE (pipeline réel sur DGP GARCH synthétique).

Les tests de contrat n'exécutent JAMAIS le pipeline (le 422 est renvoyé avant la
création du job) : ils sont rapides et sans réseau. L'intégration est marquée
`slow` et monkeypatche le téléchargement de données (aucun réseau).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def app_module():
    return pytest.importorskip('tickerlab.web.app')


@pytest.fixture
def client(app_module):
    """Client réel (les 422 de contrat ne déclenchent aucun job)."""
    return TestClient(app_module.app)


@pytest.fixture
def client_no_run(app_module, monkeypatch):
    """Client dont l'exécuteur de job est neutralisé : les POST valides créent un
    job qui reste 'queued' (pas d'exécution pipeline / réseau)."""
    monkeypatch.setattr(app_module, '_executer_job_v1', lambda job: None)
    return TestClient(app_module.app)


def _prix_garch_synthetique(n=2600, seed=7):
    """DataFrame prix (colonne 'prix') issu d'un DGP GARCH(1,1)-t NON constant."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta, nu = 0.02, 0.07, 0.90, 7.0
    sig2 = np.empty(n); eps = np.empty(n)
    sig2[0] = omega / (1 - alpha - beta)
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))
    eps[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, n):
        sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
    prix = 100.0 * np.exp(np.cumsum(eps / 100.0))
    idx = pd.bdate_range('2014-01-01', periods=n)
    return pd.DataFrame({'prix': prix}, index=idx)


# ── 1. Contrats (validation stricte, aucun job lancé) ─────────────────────────

def _req(**over):
    base = {'symbol': 'BZ=F', 'date_from': '2006-01-02', 'date_to': '2024-12-31',
            'freq': 'daily', 'price': 'adj_close'}
    base.update(over)
    return base


def test_champ_inconnu_422(client):
    r = client.post('/api/v1/analyses', json=_req(foo=1))
    assert r.status_code == 422  # extra='forbid'


def test_dates_inversees_422(client):
    r = client.post('/api/v1/analyses', json=_req(date_from='2024-12-31', date_to='2006-01-02'))
    assert r.status_code == 422
    assert r.json().get('code') == 'DATES_INCOHERENTES'


def test_serie_trop_courte_422(client):
    r = client.post('/api/v1/analyses', json=_req(date_from='2024-06-01', date_to='2024-12-31'))
    assert r.status_code == 422
    body = r.json()
    assert body.get('code') == 'SERIE_TROP_COURTE'
    assert body.get('n_observations_estime', 999) < 250


def test_module_var_indisponible_422(client):
    r = client.post('/api/v1/analyses', json=_req(module='var'))
    assert r.status_code == 422
    assert r.json().get('code') == 'MODULE_INDISPONIBLE'


def test_outputs_inconnu_422(client):
    r = client.post('/api/v1/analyses', json=_req(outputs=['garch', 'inexistant']))
    assert r.status_code == 422  # valider_outputs -> ValueError -> 422


def test_serie_fragile_warning_202(client_no_run):
    # ~250-500 obs estimées => accepté + warning SERIE_FRAGILE (pas de refus).
    r = client_no_run.post('/api/v1/analyses',
                           json=_req(date_from='2023-01-01', date_to='2024-06-30'))
    assert r.status_code == 202
    body = r.json()
    assert body['status'] == 'queued'
    assert body['warning'] == 'SERIE_FRAGILE'


# ── 2. Idempotence du job_id ──────────────────────────────────────────────────

def test_idempotence_job_id(client_no_run):
    payload = _req()
    a = client_no_run.post('/api/v1/analyses', json=payload).json()
    b = client_no_run.post('/api/v1/analyses', json=payload).json()
    assert a['job_id'] == b['job_id']         # même requête => même job


def test_job_inconnu_404(client):
    assert client.get('/api/v1/analyses/inexistant0000').status_code == 404


# ── 3. Mapping du result : clés exactes + 6 tests ─────────────────────────────

def _fake_pipeline_result():
    """PipelineResult minimal + backtest_detail réel (calculé hors ligne)."""
    from tickerlab.core.backtest import backtest_oos
    from tickerlab.core.pipeline_result import PipelineResult

    prix = _prix_garch_synthetique(n=1800)
    rendements = np.log(prix['prix'] / prix['prix'].shift(1)).dropna() * 100
    best = {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 't',
            'modele': 'GARCH', 'model': 'GARCH'}
    _, _, T_eff, detail = backtest_oos(rendements, best, split_ratio=0.70,
                                       niveaux_test=[0.95, 0.99],
                                       n_simulations_mc=1500, retour_detail=True)
    df_var = pd.DataFrame({'VaR GARCH': {'99%': -4.5}, 'TVaR GARCH': {'99%': -5.9}})
    garch_final = type('GF', (), {'params': pd.Series({'omega': 0.02,
                                                       'alpha[1]': 0.07,
                                                       'beta[1]': 0.90})})()
    return PipelineResult(rendements=rendements, garch_best=best,
                          garch_final=garch_final, var=df_var,
                          backtest_detail=detail)


def test_result_cles_exactes_et_six_tests():
    from tickerlab.web.resultat import construire_result

    res = _fake_pipeline_result()
    out = construire_result(res, {}, rapport_pdf_url='/api/v1/analyses/x/rapport')

    for cle in ('modele_retenu', 'distribution', 'persistance',
                'taux_exception_var99', 'ratio_tvar_var', 'backtesting',
                'rapport_pdf_url', 'n_observations'):
        assert cle in out, cle

    assert out['modele_retenu'] == 'GARCH(1,1)'
    assert out['distribution'] == 'Student-t'
    assert out['persistance'] == pytest.approx(0.97, abs=1e-6)   # 0.07 + 0.90
    assert out['ratio_tvar_var'] == pytest.approx(5.9 / 4.5, abs=1e-3)

    detail = out['backtesting']['detail']
    assert set(detail) == {'kupiec', 'christoffersen', 'dq', 'acerbi_szekely',
                           'fissler_ziegel', 'berkowitz_pit'}
    for t in detail.values():
        assert 'verdict' in t and t['verdict'] in ('accepte', 'rejete', 'n/a')
        assert 'convention' in t
    assert detail['fissler_ziegel'].get('benchmark')          # benchmark nommé
    assert out['backtesting']['total'] == 6
    assert 0 <= out['backtesting']['valides'] <= 6


# ── 4. Signaux : schéma, bornes, déterminisme ─────────────────────────────────

def test_signaux_regle_et_bornes():
    from tickerlab.web import signaux
    dates = pd.bdate_range('2024-01-01', periods=60)
    sigma = np.linspace(1.0, 3.0, 60)
    lignes = signaux.lignes_signaux(dates, sigma, mu=0.02, q01_std=-2.6, jours=30)
    assert len(lignes) == 30
    for l in lignes:
        assert l['signal'] in ('achat', 'vente', 'neutre')
        assert 0 <= l['confiance_pct'] <= 95          # pente douce plafonnée
        assert l['sigma_pct'] > 0
        assert l['var99_pct'] > 0                      # magnitude de perte (queue gauche)


def test_signaux_endpoint_deterministe(app_module, monkeypatch):
    from tickerlab.web import signaux
    signaux._CACHE.clear()
    dates = pd.bdate_range('2020-01-01', periods=400)
    sigma = 1.5 + 0.5 * np.sin(np.linspace(0, 20, 400))
    monkeypatch.setattr(signaux, '_estimer_sigma',
                        lambda symbol, config: (dates, sigma, 0.0, -2.5))
    client = TestClient(app_module.app)
    r1 = client.get('/api/v1/signaux', params={'symbol': 'BZ=F', 'days': 20})
    r2 = client.get('/api/v1/signaux', params={'symbol': 'BZ=F', 'days': 20})
    assert r1.status_code == 200
    j1 = r1.json()
    assert set(j1) == {'symbol', 'genere_le', 'modele_source', 'avertissement', 'lignes'}
    assert len(j1['lignes']) == 20
    assert r1.json() == r2.json()                       # déterminisme à cache identique


# ── 5. Non-régression backtest_oos (3-uple par défaut) ────────────────────────

def test_backtest_oos_non_regression_3uple():
    from tickerlab.core.backtest import backtest_oos
    prix = _prix_garch_synthetique(n=1200)
    rendements = np.log(prix['prix'] / prix['prix'].shift(1)).dropna() * 100
    best = {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 't', 'modele': 'GARCH'}
    out = backtest_oos(rendements, best, split_ratio=0.70,
                       niveaux_test=[0.95, 0.99], n_simulations_mc=1000)
    assert isinstance(out, tuple) and len(out) == 3    # signature historique intacte


# ── 6. Intégration bout-en-bout HORS LIGNE (pipeline réel, sans réseau) ───────

@pytest.mark.slow
def test_e2e_pipeline_offline(app_module, monkeypatch, tmp_path):
    """POST -> polling -> result -> rapport, sur données synthétiques (aucun réseau).

    Monkeypatche telecharger_prix : le pipeline RÉEL s'exécute hors ligne. Couvre
    le cycle de vie, les clés du result et le PDF (%PDF, > 10 Ko)."""
    import tickerlab.main as _main
    monkeypatch.setattr(_main, 'telecharger_prix',
                        lambda ticker, start_date, end_date, auto_adjust=True:
                        _prix_garch_synthetique(n=2600))

    client = TestClient(app_module.app)
    payload = _req(symbol='SYNTH=F', date_from='2014-01-02', date_to='2024-12-31',
                   outputs=['garch', 'var', 'backtest', 'charts'])
    r = client.post('/api/v1/analyses', json=payload)
    assert r.status_code == 202
    job_id = r.json()['job_id']

    etat = None
    for _ in range(240):                               # <= 240 s
        etat = client.get(f'/api/v1/analyses/{job_id}').json()
        if etat['status'] in ('done', 'error'):
            break
        time.sleep(1.0)

    assert etat is not None and etat['status'] == 'done', etat
    result = etat['result']
    for cle in ('modele_retenu', 'distribution', 'persistance',
                'taux_exception_var99', 'ratio_tvar_var', 'backtesting',
                'n_observations'):
        assert cle in result, cle
    assert set(result['backtesting']['detail']) == {
        'kupiec', 'christoffersen', 'dq', 'acerbi_szekely',
        'fissler_ziegel', 'berkowitz_pit'}

    rr = client.get(f'/api/v1/analyses/{job_id}/rapport')
    assert rr.status_code == 200
    assert rr.headers['content-type'] == 'application/pdf'
    assert rr.content[:4] == b'%PDF'
    assert len(rr.content) > 10_000
