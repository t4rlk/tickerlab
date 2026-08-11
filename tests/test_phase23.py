# -*- coding: utf-8 -*-
"""
Tests Phase 2.3 — Stress testing scenariste + reverse stress.

9 tests < 5s (pas d'appel reseau ni GARCH reel).

Test 1 : scenario_application_correct    — choc +30% BZ=F oil_shock_2022
Test 2 : reverse_stress_converge         — perte_cible=-30%, sigma_t=0.10
Test 3 : scenarios_documentes            — 4 scenarios, champs obligatoires
Test 4 : perte_proportionnelle_horizon   — meme choc, H=1 vs H=5 -> d coherente
Test 5 : couverture_actifs_phase21       — BZ=F, ^GSPC, EURUSD=X chacun >=1 scenario
Test 6 : t_standardisee_vs_standard      — t standardisee (variance=1) vs scipy.t direct
Test 7 : proba_queue_choc_positif        — directionnelle : queue sup pour gain, inf pour perte
Test 8 : sigma_H_exact_vs_sqrt           — recursion GARCH/IGARCH vs regle sqrt(H)
Test 9 : btc001_floor                    — VaR GARCH < -99.9% => floor + warning
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stress_scenarios import (
    SCENARIOS_BASE,
    appliquer_scenario,
    appliquer_tous_scenarios,
    scenarios_pour_ticker,
)
from core.reverse_stress import reverse_stress


# ── Test 1 : application scenario correcte ───────────────────────────────────

def test_scenario_application_correct():
    """
    oil_shock_2022 sur BZ=F (choc = +30%, position longue = gain).
    Verification :
      - choc_direct = +0.30
      - sens = 'gain'
      - distance_mahalanobis = 0.30 / (0.03 * sqrt(5)) coherente
      - multiple_var99_h1 = 0.30 / 0.08 = 3.75
      - proba > 0.5 (choc positif est dans la queue superieure)
    """
    result = appliquer_scenario(
        'oil_shock_2022',
        ticker='BZ=F',
        sigma_t=0.03,
        dist='t',
        nu=5.0,
        var99_h1=-0.08,
    )

    assert result is not None, "oil_shock_2022 doit etre applicable a BZ=F"
    assert result['choc_direct'] == pytest.approx(+0.30, abs=1e-9)
    assert result['sens'] == 'gain', "Position longue + choc positif = gain"
    assert result['horizon_jours'] == 5

    # sigma_H = 0.03 * sqrt(5) ~ 0.0671
    # distance = 0.30 / 0.0671 ~ 4.47
    assert result['distance_mahalanobis'] == pytest.approx(0.30 / (0.03 * np.sqrt(5)), rel=1e-4)

    # multiple_var99 = 0.30 / 0.08 = 3.75
    assert result['multiple_var99_h1'] == pytest.approx(3.75, rel=1e-4)

    # choc positif -> queue superieure (proba_queue = P(r >= +30%)) -> petit nombre
    assert 'proba_queue' in result, "Champ 'proba_queue' requis (ex-'proba_unilaterale')"
    assert 'proba_unilaterale' not in result, "Ancien champ supprime"
    assert result['proba_queue'] < 0.5  # upper tail d'un choc extreme

    # methode_sigma_H present
    assert 'methode_sigma_H' in result

    # interpreteur non vide
    assert len(result['interpreteur']) > 10


# ── Test 2 : reverse stress converge ─────────────────────────────────────────

def test_reverse_stress_converge():
    """
    Perte cible de -30% avec sigma_t=0.10 (N(0,1), H=1).
    Verification :
      - choc_requis = -0.30
      - distance_mahalanobis = 3.0 exactement
      - proba ~ norm.cdf(-3) ~ 0.00135
      - periode_retour ~ 741 periodes
    """
    from scipy import stats

    result = reverse_stress(
        perte_cible=-0.30,
        sigma_t=0.10,
        dist='normal',
        horizon_jours=1,
    )

    assert result['choc_requis'] == pytest.approx(-0.30, abs=1e-9)
    assert result['sigma_H'] == pytest.approx(0.10, abs=1e-9)
    assert result['distance_mahalanobis'] == pytest.approx(3.0, rel=1e-4)

    expected_proba = stats.norm.cdf(-3.0)
    assert result['proba_sous_modele'] == pytest.approx(expected_proba, rel=1e-6)
    assert result['proba_sous_modele'] < 0.01   # evenement rare
    assert result['periode_retour_periodes'] > 700

    # avec dist='t', nu=5 : probabilite plus elevee (queues plus epaisses)
    result_t = reverse_stress(
        perte_cible=-0.30,
        sigma_t=0.10,
        dist='t',
        nu=5.0,
        horizon_jours=1,
    )
    # t(5) a des queues plus epaisses -> P(X <= -3) > P(N(0,1) <= -3)
    assert result_t['proba_sous_modele'] > result['proba_sous_modele']

    # ValueError si perte_cible >= 0
    with pytest.raises(ValueError):
        reverse_stress(perte_cible=0.10, sigma_t=0.10)

    # ValueError si sigma_t <= 0
    with pytest.raises(ValueError):
        reverse_stress(perte_cible=-0.10, sigma_t=0.0)


# ── Test 3 : tous les scenarios sont documentes ───────────────────────────────

def test_scenarios_documentes():
    """
    Chaque scenario dans SCENARIOS_BASE a les champs obligatoires :
    description, reference, shocks (non vide), horizon_jours >= 1.
    """
    assert len(SCENARIOS_BASE) >= 4, (
        f"Au moins 4 scenarios attendus, {len(SCENARIOS_BASE)} trouves"
    )

    champs_obligatoires = ['description', 'reference', 'shocks', 'horizon_jours']

    for nom, sc in SCENARIOS_BASE.items():
        for champ in champs_obligatoires:
            assert champ in sc, (
                f"Scenario '{nom}' : champ obligatoire '{champ}' manquant"
            )
        assert len(sc['description']) > 10, (
            f"Scenario '{nom}' : description trop courte"
        )
        assert len(sc['reference']) > 5, (
            f"Scenario '{nom}' : reference trop courte"
        )
        assert isinstance(sc['shocks'], dict) and len(sc['shocks']) >= 1, (
            f"Scenario '{nom}' : shocks doit etre un dict non vide"
        )
        assert sc['horizon_jours'] >= 1, (
            f"Scenario '{nom}' : horizon_jours doit etre >= 1"
        )
        # chaque choc doit etre un float dans [-1, +1] (rendement simple)
        for ticker, choc in sc['shocks'].items():
            assert -1.0 <= choc <= 1.0, (
                f"Scenario '{nom}', ticker '{ticker}' : "
                f"choc {choc:.2f} hors plage [-1, +1] (rendement simple)"
            )


# ── Test 4 : coherence horizon (perte proportionnelle) ───────────────────────

def test_perte_proportionnelle_horizon():
    """
    Pour un meme choc (ex: -10%), l'horizon plus long implique une sigma_H plus grande
    et donc une distance de Mahalanobis plus petite (le choc est moins "surprenant"
    relativement a la volatilite sur l'horizon).
    Verifie : d(H=1) > d(H=5) pour un meme choc de taille fixe.
    """
    sigma_t = 0.02   # 2% daily

    # Simuler un scenario H=1 et H=5 avec meme choc
    # fed_hike_2022 : ^GSPC -8%, H=1
    # geopolitique_taiwan : ^GSPC -15%, H=5
    # On utilise les vrais scenarios pour le test de coherence generique

    # Test generique : meme choc absolu (-10%), H=1 vs H=5
    # Calcul manuel : d(H=1) = 0.10/0.02 = 5.0, d(H=5) = 0.10/(0.02*sqrt(5)) ~ 2.24
    from core.reverse_stress import reverse_stress as rs

    r_h1 = rs(perte_cible=-0.10, sigma_t=sigma_t, dist='normal', horizon_jours=1)
    r_h5 = rs(perte_cible=-0.10, sigma_t=sigma_t, dist='normal', horizon_jours=5)

    assert r_h1['distance_mahalanobis'] > r_h5['distance_mahalanobis'], (
        "Distance Mahalanobis doit diminuer quand H augmente "
        "(sigma_H = sigma_t * sqrt(H) augmente)"
    )
    assert r_h1['proba_sous_modele'] < r_h5['proba_sous_modele'], (
        "P(perte | H=1) < P(perte | H=5) car la volatilite est plus haute a H=5"
    )

    # Verification numerique : d(H=1) / d(H=5) = sqrt(5)
    ratio = r_h1['distance_mahalanobis'] / r_h5['distance_mahalanobis']
    assert ratio == pytest.approx(np.sqrt(5), rel=1e-3)

    # Les multiples FHS ne doivent pas etre nan quand var99_fhs fourni
    r_avec_fhs = appliquer_scenario(
        'fed_hike_2022', '^GSPC',
        sigma_t=0.012,
        dist='normal',
        var99_h1=-0.020,
        var99_fhs_hH=-0.018,
    )
    assert not np.isnan(r_avec_fhs['multiple_var99_fhs'])
    assert r_avec_fhs['multiple_var99_fhs'] == pytest.approx(0.08 / 0.018, rel=1e-3)


# ── Test 5 : couverture des actifs Phase 2.1 ─────────────────────────────────

def test_couverture_actifs_phase21():
    """
    Les 3 tickers de reference Phase 2.1 ont chacun au moins 1 scenario applicable.
    BZ=F : oil_shock_2022 et covid_march_2020 attendus.
    ^GSPC : covid_march_2020, fed_hike_2022, geopolitique_taiwan attendus.
    EURUSD=X : fed_hike_2022 attendu (seul scenario FX inclus).
    """
    tickers_attendus = {
        'BZ=F':     {'oil_shock_2022', 'covid_march_2020'},
        '^GSPC':    {'covid_march_2020', 'fed_hike_2022', 'geopolitique_taiwan'},
        'EURUSD=X': {'fed_hike_2022'},
    }

    for ticker, scenarios_attendus in tickers_attendus.items():
        sc_list = set(scenarios_pour_ticker(ticker))
        assert sc_list, f"Aucun scenario trouve pour {ticker}"
        for sc_attendu in scenarios_attendus:
            assert sc_attendu in sc_list, (
                f"Scenario '{sc_attendu}' attendu pour {ticker} mais absent "
                f"de scenarios_pour_ticker('{ticker}') = {sc_list}"
            )

    # Verifier que appliquer_scenario renvoie None pour un ticker non defini
    result_na = appliquer_scenario(
        'geopolitique_taiwan', 'BZ=F',   # BZ=F n'est pas dans geopolitique_taiwan
        sigma_t=0.03, dist='normal',
    )
    assert result_na is None, (
        "appliquer_scenario doit retourner None pour un ticker absent du scenario"
    )

    # appliquer_tous_scenarios sur EURUSD=X doit retourner exactement 1 ligne
    df = appliquer_tous_scenarios('EURUSD=X', sigma_t=0.005, dist='normal')
    assert len(df) == 1, (
        f"EURUSD=X doit avoir exactement 1 scenario (fed_hike_2022), "
        f"{len(df)} trouve(s)"
    )
    assert df.iloc[0]['scenario'] == 'fed_hike_2022'


# ── Test 6 : t standardisee vs scipy.t direct ────────────────────────────────

def test_t_standardisee_vs_standard():
    """
    t standardisee (variance=1, convention arch) : P(Z <= z) != scipy.t.cdf(z, nu).

    Pour z < 0 :
      z_scipy = z * sqrt(nu/(nu-2)) est plus negatif que z (|z_scipy| > |z|)
      => P(t_scipy <= z_scipy) < P(t_scipy <= z)
      => t standardisee donne une probabilite PLUS PETITE pour le meme z negatif.
      Interpretation : le meme z-score est un evenement PLUS EXTREME sous t_std.

    Pour nu -> infini : t_std -> N(0,1).
    """
    from core.stress_scenarios import _proba_queue_t_standardisee
    from scipy import stats as sp

    nu = 5.0
    z = -2.0

    p_scipy_direct = float(sp.t.cdf(z, df=nu))
    p_standardisee = _proba_queue_t_standardisee(z, nu)

    # z_scipy = -2 * sqrt(5/3) ~ -2.582 -> plus negatif -> P plus petite
    assert p_standardisee < p_scipy_direct, (
        "t standardisee (variance=1) donne P(Z<=z) PLUS PETITE que scipy.t direct "
        "pour z negatif (z_scipy plus extreme)"
    )

    # Verification numerique : z_scipy = z * sqrt(nu/(nu-2))
    z_scipy_expected = z * np.sqrt(nu / (nu - 2.0))
    assert p_standardisee == pytest.approx(float(sp.t.cdf(z_scipy_expected, df=nu)), rel=1e-9)

    # Convergence vers N(0,1) quand nu -> infini
    nu_large = 1000.0
    z_test = -1.96
    p_large = _proba_queue_t_standardisee(z_test, nu_large)
    p_norm = float(sp.norm.cdf(z_test))
    assert abs(p_large - p_norm) < 0.001

    # reverse_stress utilise aussi la t standardisee
    from core.reverse_stress import _proba_queue_t_standardisee as rs_prob
    assert rs_prob(z, nu) == pytest.approx(p_standardisee, rel=1e-9)


# ── Test 7 : probabilite de queue directionnelle ──────────────────────────────

def test_proba_queue_choc_positif():
    """
    Choc positif (gain) -> proba_queue = queue superieure P(r >= choc).
    Choc negatif (perte) -> proba_queue = queue inferieure P(r <= choc).
    Les deux sont < 0.5 pour des chocs extremes.
    Champ 'proba_queue' present, 'proba_unilaterale' absent.
    """
    sigma_t = 0.03

    # Choc positif +30% (oil_shock_2022, BZ=F) -> gain -> upper tail
    r_pos = appliquer_scenario('oil_shock_2022', 'BZ=F', sigma_t=sigma_t, dist='normal')
    assert r_pos is not None
    assert 'proba_queue' in r_pos, "Champ 'proba_queue' requis"
    assert 'proba_unilaterale' not in r_pos, "Ancien champ 'proba_unilaterale' supprime"
    assert r_pos['sens'] == 'gain'
    assert r_pos['proba_queue'] < 0.5, "Upper tail d'un choc extreme doit etre < 0.5"

    # Choc negatif -65% (covid_march_2020, BZ=F) -> perte -> lower tail
    r_neg = appliquer_scenario('covid_march_2020', 'BZ=F', sigma_t=sigma_t, dist='normal')
    assert r_neg is not None
    assert r_neg['sens'] == 'perte'
    assert r_neg['proba_queue'] < 0.5, "Lower tail d'un choc extreme doit etre < 0.5"

    # Symetrie sous normale : choc +x et choc -x (meme amplitude) -> meme proba_queue
    # oil_shock_2022 : choc=+0.30, H=5 -> sigma_H = 0.03*sqrt(5)
    sigma_H = sigma_t * np.sqrt(5)
    z = 0.30 / sigma_H
    from scipy import stats as sp
    p_upper_expected = float(sp.norm.sf(z))    # P(N > z)
    p_lower_expected = float(sp.norm.cdf(-z))  # P(N < -z) = P(N > z) par symetrie
    assert p_upper_expected == pytest.approx(p_lower_expected, rel=1e-9)
    # proba_queue du choc positif == p_upper_expected
    # tolerance 2% : round(proba_queue, 8) perd de la precision pour des valeurs ~1e-6
    assert r_pos['proba_queue'] == pytest.approx(p_upper_expected, rel=2e-2)


# ── Test 8 : sigma_H exact via recursion GARCH ───────────────────────────────

def test_sigma_H_exact_vs_sqrt():
    """
    _sigma_horizon_exact() via recursion analytique GARCH/IGARCH.

    GARCH stationnaire (sigma_T < sigma_bar) : mean-reversion vers le haut
      -> sigma_H_exact > sqrt(H) * sigma_T.
    Near-IGARCH (rho >= 1) : variance croit lineairement
      -> sigma_H_exact > sqrt(H) * sigma_T.
    EGARCH : fallback sqrt(H) (recursion non analytique).
    H=1 : sigma_H = sigma_T (identite).
    """
    from core.stress_scenarios import _sigma_horizon_exact

    sigma_t = 0.03
    H = 22
    sigma_H_sqrt = sigma_t * np.sqrt(H)

    # Mock minimal d'un ARCHModelResult : seuls model.volatility.__name__ et params.get() requis
    class _MockVol:
        pass

    class _MockModel:
        def __init__(self, vname):
            self.volatility = _MockVol()
            type(self.volatility).__name__ = property(lambda self: vname)

    # Classe mock avec __name__ fixe plus proprement
    class _MkResult:
        def __init__(self, vol_name, omega, alpha, beta, gamma=0.0):
            class _V:
                pass
            _V.__name__ = vol_name
            class _M:
                pass
            _M.volatility = _V()
            self.model = _M()
            _p = {'omega': omega, 'alpha[1]': alpha, 'beta[1]': beta, 'gamma[1]': gamma}
            class _P:
                def get(s, k, d=0.0):
                    return _p.get(k, d)
            self.params = _P()

    # Cas 1 : GARCH stationnaire, sigma_T < sigma_bar (mean-reversion vers le haut)
    # rho=0.90, omega=0.0009 -> sigma_bar = sqrt(0.0009/0.10) = sqrt(0.009) ~ 9.5%
    # sigma_T=3% < sigma_bar => V_H/H > sigma_T^2 => sigma_H_exact > sqrt(H)*sigma_T
    mock_stat = _MkResult('GARCH', omega=0.0009, alpha=0.05, beta=0.85)
    sigma_H_stat, methode_stat = _sigma_horizon_exact(mock_stat, sigma_t, H)
    assert methode_stat == 'garch_exact'
    assert sigma_H_stat > sigma_H_sqrt, (
        "GARCH stationnaire (sigma_T < sigma_bar) : mean-reversion vers le haut "
        "-> sigma_H_exact > sqrt(H)*sigma_T"
    )

    # Cas 2 : near-IGARCH (rho >= 1) — variance croit lineairement
    # alpha + 0.5*gamma + beta = 0.05 + 0 + 0.95 = 1.0 -> IGARCH
    mock_ig = _MkResult('GARCH', omega=0.0001, alpha=0.05, beta=0.95)
    sigma_H_ig, methode_ig = _sigma_horizon_exact(mock_ig, sigma_t, H)
    assert methode_ig == 'igarch_exact'
    assert sigma_H_ig > sigma_H_sqrt, (
        "IGARCH (rho=1) : variance croit lineairement -> sigma_H_exact > sqrt(H)*sigma_T"
    )
    ecart_pct = (sigma_H_ig - sigma_H_sqrt) / sigma_H_sqrt * 100
    assert ecart_pct > 0.5, f"Ecart IGARCH vs sqrt(H) : {ecart_pct:.2f}% trop petit"

    # Cas 3 : EGARCH -> fallback sqrt(H)
    mock_eg = _MkResult('EGARCH', omega=0.0, alpha=0.05, beta=0.97)
    sigma_H_eg, methode_eg = _sigma_horizon_exact(mock_eg, sigma_t, H)
    assert 'EGARCH' in methode_eg
    assert sigma_H_eg == pytest.approx(sigma_H_sqrt, rel=1e-6)

    # Cas 4 : H=1 -> sigma_H = sigma_T (early return)
    sigma_H_h1, _ = _sigma_horizon_exact(mock_ig, sigma_t, 1)
    assert sigma_H_h1 == pytest.approx(sigma_t, rel=1e-9)

    # Cas 5 : garch_final=None -> fallback sqrt(H)
    sigma_H_none, methode_none = _sigma_horizon_exact(None, sigma_t, H)
    assert sigma_H_none == pytest.approx(sigma_H_sqrt, rel=1e-9)
    assert methode_none == 'sqrt_H'


# ── Test 9 : floor BTC-001 ────────────────────────────────────────────────────

def test_btc001_floor():
    """
    _apply_btc001_floor() : VaR GARCH < -99.9% => floor a -99.9% + warning [BTC-001].
    Cas reel : BTC-USD weekly near-IGARCH + t(nu~3) + sigma_T eleve => VaR ~ -357%.
    """
    import warnings as _warnings
    from core.var_engine import _apply_btc001_floor

    # Cas nominal : VaR > -99.9% -> pas de modification
    val, floored = _apply_btc001_floor(-50.0)
    assert not floored
    assert val == pytest.approx(-50.0)

    # Cas BTC-001 : VaR << -99.9% -> floor + UserWarning
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter('always')
        val_btc, floored_btc = _apply_btc001_floor(-357.0, ticker='BTC-USD')

    assert floored_btc, "VaR = -357% doit etre floore"
    assert val_btc == pytest.approx(-99.9, abs=1e-9)
    assert len(w) == 1
    assert 'BTC-001' in str(w[0].message)
    assert 'BTC-USD' in str(w[0].message)
    assert issubclass(w[0].category, UserWarning)

    # Limite exacte : -99.9 -> pas de floor (plancher strict)
    val_limit, floored_limit = _apply_btc001_floor(-99.9)
    assert not floored_limit
    assert val_limit == pytest.approx(-99.9)

    # En dessous de la limite : -99.91 -> floor
    val_below, floored_below = _apply_btc001_floor(-99.91)
    assert floored_below
    assert val_below == pytest.approx(-99.9, abs=1e-9)

    # Sans ticker : warning toujours emis
    with _warnings.catch_warnings(record=True) as w2:
        _warnings.simplefilter('always')
        _apply_btc001_floor(-200.0)
    assert len(w2) == 1
    assert 'BTC-001' in str(w2[0].message)
