"""
NIVEAU 4 - Invariants analytiques.

Aucune donnee externe : ces tests sont deterministes (ou a graine fixee) et
comparent tickerlab a des verites mathematiques, pas a une autre
implementation. Ce sont eux qui attrapent les erreurs de signe, de queue et de
normalisation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from . import adapter
from .conftest import require

ALPHAS = [0.05, 0.025, 0.01, 0.005]


# =============================================================================
# 1. Formes fermees VaR / ES
# =============================================================================


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("sigma", [1.0, 0.017])
def test_var_es_normal_closed_form(alpha, sigma):
    """VaR = -sigma*z_a, ES = sigma*phi(z_a)/a pour une gaussienne centree."""
    got = require(adapter.var_es_normal(0.0, sigma, alpha), "var_es_normal")
    z = stats.norm.ppf(alpha)
    exp_var = -sigma * z
    exp_es = sigma * stats.norm.pdf(z) / alpha
    assert np.isclose(got[0], exp_var, rtol=1e-10), "VaR gaussienne"
    assert np.isclose(got[1], exp_es, rtol=1e-10), "ES gaussienne"


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("nu", [4.0, 5.0, 8.0])
def test_var_es_student_vs_numerical_integration(alpha, nu):
    """Student STANDARDISEE (variance unitaire) : la forme fermee doit egaler
    l'integration numerique de E[X | X <= q_a].

    Piege classique : oublier le facteur sqrt(nu/(nu-2)) et renvoyer la VaR de
    la t brute, qui surestime le risque de ~22 % a nu = 5.
    """
    got = require(adapter.var_es_student(0.0, 1.0, nu, alpha), "var_es_student")
    s = np.sqrt(nu / (nu - 2.0))
    q_std = stats.t.ppf(alpha, nu) / s
    pdf_std = lambda x: stats.t.pdf(x * s, nu) * s
    tail, _ = integrate.quad(lambda x: x * pdf_std(x), -np.inf, q_std)
    assert np.isclose(got[0], -q_std, rtol=1e-8), "VaR Student"
    assert np.isclose(got[1], -tail / alpha, rtol=1e-6), "ES Student"


def test_student_converges_to_normal():
    """Quand nu -> inf, la t standardisee doit converger vers la gaussienne."""
    t = require(adapter.var_es_student(0.0, 1.0, 500.0, 0.01), "var_es_student")
    n = require(adapter.var_es_normal(0.0, 1.0, 0.01), "var_es_normal")
    assert np.allclose(t, n, rtol=5e-3)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_location_scale_equivariance(alpha):
    """VaR(mu, sigma) = -mu + sigma * VaR(0,1) : verifie que mu n'est pas
    absorbe ou ignore."""
    base = require(adapter.var_es_normal(0.0, 1.0, alpha), "var_es_normal")
    mu, sigma = 0.0004, 0.013
    got = adapter.var_es_normal(mu, sigma, alpha)
    assert np.isclose(got[0], -mu + sigma * base[0], rtol=1e-10)
    assert np.isclose(got[1], -mu + sigma * base[1], rtol=1e-10)


# =============================================================================
# 2. Invariants d'ordre
# =============================================================================


@pytest.mark.parametrize("alpha", ALPHAS)
def test_es_dominates_var(alpha):
    """ES >= VaR, toujours, pour toute loi continue."""
    for got in (adapter.var_es_normal(0.0, 1.0, alpha),
                adapter.var_es_student(0.0, 1.0, 5.0, alpha)):
        got = require(got, "var_es_normal / var_es_student")
        assert got[1] > got[0]


def test_monotonicity_in_alpha():
    """VaR et ES croissent quand la probabilite de queue diminue."""
    vars_, ess = [], []
    for a in sorted(ALPHAS, reverse=True):
        v, e = require(adapter.var_es_normal(0.0, 1.0, a), "var_es_normal")
        vars_.append(v)
        ess.append(e)
    assert np.all(np.diff(vars_) > 0)
    assert np.all(np.diff(ess) > 0)


def test_fat_tails_increase_deep_tail_risk():
    """A variance egale, la Student doit dominer la gaussienne dans la queue
    profonde -- et lui etre INFERIEURE en VaR a 5 % (les queues epaisses se
    paient par un centre plus resserre)."""
    n_deep = require(adapter.var_es_normal(0.0, 1.0, 0.01), "var_es_normal")
    t_deep = require(adapter.var_es_student(0.0, 1.0, 4.0, 0.01), "var_es_student")
    n_shal = adapter.var_es_normal(0.0, 1.0, 0.05)
    t_shal = adapter.var_es_student(0.0, 1.0, 4.0, 0.05)
    assert t_deep[0] > n_deep[0] and t_deep[1] > n_deep[1]
    assert t_shal[0] < n_shal[0]


def test_es_is_subadditive_and_var_is_not(rng):
    """ES coherente (sous-additive) ; contre-exemple explicite de non
    sous-additivite de la VaR sur deux obligations a defaut independant.

    Deux titres, defaut de proba 4 %, perte 100. A 95 % : VaR individuelle = 0
    (le defaut est dans les 4 % extremes), mais le portefeuille defaille avec
    proba 1 - 0.96^2 = 7.84 % > 5 % -> VaR du portefeuille = 100 > 0 + 0.
    """
    alpha = 0.05
    n = 400_000
    d1 = (rng.random(n) < 0.04).astype(float)
    d2 = (rng.random(n) < 0.04).astype(float)
    r1, r2 = -100.0 * d1, -100.0 * d2

    v1, e1 = require(adapter.var_es_empirical(r1, alpha), "var_es_empirical")
    v2, e2 = adapter.var_es_empirical(r2, alpha)
    vp, ep = adapter.var_es_empirical(r1 + r2, alpha)

    assert vp > v1 + v2 + 1e-9, "la VaR devrait violer la sous-additivite ici"
    assert ep <= e1 + e2 + 1e-6, "l'ES doit rester sous-additive"


# =============================================================================
# 3. Elicitabilite (Fissler-Ziegel) -- le test qui prouve que le score est le bon
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("alpha", [0.05, 0.025])
def test_fz0_is_minimised_at_true_var_es(alpha, rng):
    """Propriete definissante du score FZ0 : l'esperance du score est minimale
    au couple (VaR, ES) veritable, et nulle part ailleurs.

    C'est plus fort qu'une comparaison de formules : cela verifie que la
    fonction implementee est bien un score strictement consistant pour la paire
    (VaR, ES) -- ce que ne garantit aucun test de valeur ponctuelle.
    """
    x = rng.standard_normal(2_000_000)
    z = stats.norm.ppf(alpha)
    v_true, e_true = -z, stats.norm.pdf(z) / alpha   # pertes positives

    grid = np.linspace(-0.30, 0.30, 25)
    best, best_loss = None, np.inf
    for dv in grid:
        for de in grid:
            v, e = v_true + dv, e_true + de
            if e <= v:                     # domaine admissible : ES > VaR
                continue
            loss = np.mean(require(adapter.fz0_loss(x, v, e, alpha), "fz0_loss"))
            if loss < best_loss:
                best_loss, best = loss, (v, e)

    step = grid[1] - grid[0]
    assert abs(best[0] - v_true) <= step, f"argmin VaR {best[0]:.4f} vs {v_true:.4f}"
    assert abs(best[1] - e_true) <= step, f"argmin ES  {best[1]:.4f} vs {e_true:.4f}"


# =============================================================================
# 4. Recuperation des parametres GARCH (consistance de l'estimateur)
# =============================================================================


@pytest.mark.slow
def test_garch_parameter_recovery():
    """Sur un long echantillon simule a partir de parametres CONNUS,
    l'estimateur doit les retrouver. Teste simultanement la vraisemblance, la
    recursion de variance et l'optimiseur.
    """
    true = dict(mu=0.02, omega=0.01, alpha=0.09, beta=0.89)
    n = 50_000

    r = adapter.simulate_garch11(**true, n=n, seed=12345)
    if r is None:                     # DGP de secours : package `arch`
        arch = pytest.importorskip("arch")
        from arch.univariate import ConstantMean, GARCH, Normal
        am = ConstantMean(np.zeros(n))
        am.volatility = GARCH(1, 0, 1)
        am.distribution = Normal(seed=12345)
        params = np.array([true["mu"], true["omega"], true["alpha"], true["beta"]])
        r = am.simulate(params, n)["data"].values

    fit = require(adapter.fit_garch11(r, backcast=float(np.var(r, ddof=1))),
                  "fit_garch11")

    # Tolerances calibrees sur l'erreur-type de MLE a n = 50 000
    assert abs(fit["alpha"] - true["alpha"]) < 0.015
    assert abs(fit["beta"] - true["beta"]) < 0.020
    assert abs(fit["omega"] - true["omega"]) < 0.005
    assert abs(fit["mu"] - true["mu"]) < 0.02

    uncond = fit["omega"] / (1 - fit["alpha"] - fit["beta"])
    assert abs(uncond - np.var(r, ddof=1)) / uncond < 0.20


# =============================================================================
# 5. Feu tricolore balois
# =============================================================================


@pytest.mark.parametrize("n_exc,zone", [
    (0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"),
    (10, "red"), (25, "red"),
])
def test_basel_traffic_light(n_exc, zone):
    """Seuils reglementaires : vert <= 4, jaune 5-9, rouge >= 10 exceptions sur
    250 jours a 99 %."""
    got = require(adapter.basel_traffic_light(n_exc, 250, 0.01),
                  "basel_traffic_light")
    assert str(got).lower() == zone
