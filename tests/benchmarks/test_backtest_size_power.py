"""
NIVEAU 2 - Taille et puissance des backtests (Monte-Carlo).

Il n'existe pas de "valeur publiee" universelle a reproduire pour Kupiec,
Christoffersen, DQ ou Berkowitz : la validation correcte consiste a verifier
que, sous H0, le taux de rejet egale le niveau nominal (TAILLE), et que sous
une alternative realiste le test rejette effectivement (PUISSANCE).

Un test qui ne rejette jamais passe la taille et echoue la puissance ; un test
au mauvais nombre de degres de liberte echoue la taille. Les deux moities sont
necessaires -- une suite qui ne teste que la taille est verte avec
`return 1.0` en dur.

Nombre de replications : variable d'environnement BENCH_REPS
(defaut 2000 ; CI rapide : 400 ; publication : 20000).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from . import adapter
from .conftest import _mc_reps, require

LEVEL = 0.05          # niveau nominal des tests backtestes
N_OBS = 1000


def rejection_rate(pvalues, level=LEVEL):
    p = np.asarray([x for x in pvalues if np.isfinite(x)], dtype=float)
    assert p.size > 0, "aucune p-value exploitable"
    return float(np.mean(p < level))


def assert_size(pvalues, reps, expected=LEVEL, model_tol=0.012, name=""):
    """Compare le taux de rejet empirique a sa valeur ATTENDUE.

    L'intervalle d'acceptation est l'intervalle binomial exact a 99.9 %
    (bruit Monte-Carlo pur), elargi de `model_tol` pour l'erreur d'arrondi de
    l'approximation asymptotique. Aucun test ne doit dependre d'une graine
    chanceuse : les graines sont derivees du nom du test (cf. conftest).

    `expected` n'est pas toujours 5 % : certains tests ont une distorsion de
    taille documentee en echantillon fini (cf. appels ci-dessous).
    """
    r = rejection_rate(pvalues)
    lo_c, hi_c = stats.binom.interval(0.999, reps, expected)
    lo, hi = lo_c / reps - model_tol, hi_c / reps + model_tol
    assert lo < r < hi, (
        f"{name}: taille empirique {r:.3f} hors de [{lo:.3f}, {hi:.3f}] "
        f"(attendu ~{expected:.3f} sur {reps} replications)")
    return r


# =============================================================================
# Degres de liberte - verification deterministe
# =============================================================================


def test_degrees_of_freedom_are_correct(rng):
    """Chaque p-value doit etre exactement la survie d'un chi2 au BON nombre de
    degres de liberte, verifie sur la statistique renvoyee.

    Test instantane et deterministe : il attrape une erreur de ddl que les
    tests de taille Monte-Carlo ne detectent que marginalement (un chi2(2) a la
    place d'un chi2(1) ne fait "que" ramener le taux de rejet de 5 % a 1.4 %).

    Convention DQ : regression de Hit_t sur [constante, L retards, VaR_t],
    soit L + 2 degres de liberte.
    """
    n = N_OBS
    hits = (rng.random(n) < LEVEL).astype(float)
    var = np.full(n, -stats.norm.ppf(LEVEL))
    lags = 4

    cases = [
        ("Kupiec", adapter.kupiec_pof(hits, LEVEL), 1),
        ("Christoffersen ind", adapter.christoffersen_ind(hits), 1),
        ("Christoffersen cc", adapter.christoffersen_cc(hits, LEVEL), 2),
        ("DQ", adapter.dq_test(hits, var, LEVEL, lags), lags + 2),
        ("Berkowitz", adapter.berkowitz(rng.random(n)), 3),
    ]
    for name, res, df in cases:
        res = require(res, name)
        stat, pval = float(res[0]), float(res[1])
        assert np.isclose(pval, stats.chi2.sf(stat, df), rtol=1e-6, atol=1e-12), (
            f"{name}: p-value {pval:.6g} incompatible avec un chi2({df}) "
            f"evalue en {stat:.6g} (ddl attendu : {df})")


# =============================================================================
# Kupiec (1995) - couverture inconditionnelle, chi2(1)
# =============================================================================


def test_kupiec_size(rng):
    """Taille nominale respectee : ~5.0 % a n = 1000, alpha = 5 %."""
    reps = max(_mc_reps(2000), 1000)
    hits = rng.random((reps, N_OBS)) < LEVEL
    pv = [require(adapter.kupiec_pof(h.astype(float), LEVEL), "kupiec_pof")[1]
          for h in hits]
    assert_size(pv, reps, expected=0.050, model_tol=0.008, name="Kupiec")


def test_kupiec_power_against_undercoverage(rng):
    """Modele annonce a 5 % mais violant 10 % du temps : doit etre rejete."""
    reps = _mc_reps(2000)
    hits = rng.random((reps, N_OBS)) < 0.10
    pv = [require(adapter.kupiec_pof(h.astype(float), LEVEL), "kupiec_pof")[1]
          for h in hits]
    assert rejection_rate(pv) > 0.90


def test_kupiec_asymptotic_distribution(rng):
    """La statistique doit suivre un chi2(1) : on compare les quantiles, pas
    seulement le taux de rejet (attrape un mauvais nombre de ddl)."""
    reps = max(_mc_reps(2000), 2000)   # quantiles a 99 % : plancher necessaire
    hits = rng.random((reps, N_OBS)) < LEVEL
    stat = np.array([require(adapter.kupiec_pof(h.astype(float), LEVEL),
                             "kupiec_pof")[0] for h in hits])
    for q in (0.50, 0.90, 0.95, 0.99):
        emp, theo = np.quantile(stat, q), stats.chi2.ppf(q, df=1)
        assert abs(emp - theo) < 0.35 + 0.15 * theo, (
            f"quantile {q}: empirique {emp:.3f} vs chi2(1) {theo:.3f}")


# =============================================================================
# Christoffersen (1998) - independance chi2(1), couverture conditionnelle chi2(2)
# =============================================================================


def _markov_hits(rng, reps, n, p01, p11):
    """Chaine de Markov d'ordre 1 sur les violations (clustering)."""
    out = np.zeros((reps, n), dtype=float)
    state = (rng.random(reps) < p01 / (1 + p01 - p11)).astype(float)
    for t in range(n):
        p = np.where(state == 1, p11, p01)
        state = (rng.random(reps) < p).astype(float)
        out[:, t] = state
    return out


def test_christoffersen_ind_size(rng):
    """Distorsion de taille CONNUE : a n = 1000 et alpha = 5 %, le LR
    d'independance rejette ~8 % du temps sous H0, pas 5 %. La statistique est
    tres discrete (peu de transitions 1->1 observees) et l'approximation
    chi2(1) sur-rejette. On teste donc la valeur de reference mesuree, pas le
    niveau nominal : un test qui "corrigerait" ce 8 % en 5 % ne serait plus le
    test de Christoffersen.
    """
    reps = max(_mc_reps(2000), 500)
    hits = (rng.random((reps, N_OBS)) < LEVEL).astype(float)
    pv = [require(adapter.christoffersen_ind(h), "christoffersen_ind")[1]
          for h in hits]
    assert_size(pv, reps, expected=0.080, model_tol=0.020,
                name="Christoffersen ind")


def test_christoffersen_ind_power_against_clustering(rng):
    """Violations groupees a couverture inconditionnelle CORRECTE (5 %) :
    Kupiec est aveugle, l'independance doit voir."""
    reps = _mc_reps(1000)
    hits = _markov_hits(rng, reps, N_OBS, p01=0.02632, p11=0.50)
    assert abs(hits.mean() - LEVEL) < 0.01, "calibration de la chaine"

    pv_ind = [require(adapter.christoffersen_ind(h), "christoffersen_ind")[1]
              for h in hits]
    pv_uc = [require(adapter.kupiec_pof(h, LEVEL), "kupiec_pof")[1] for h in hits]

    r_ind, r_uc = rejection_rate(pv_ind), rejection_rate(pv_uc)
    assert r_ind > 0.90, f"aveugle au clustering (rejet {r_ind:.2f})"
    # Kupiec sur-rejette un peu (le clustering gonfle la variance du nombre
    # d'exceptions) mais reste tres loin derriere : c'est exactement la raison
    # d'etre du test d'independance.
    assert r_uc < 0.45 and r_ind > 2.0 * r_uc, (
        f"rejet independance {r_ind:.2f} vs Kupiec {r_uc:.2f}")


def test_christoffersen_cc_size_and_dof(rng):
    reps = max(_mc_reps(2000), 2000)
    hits = (rng.random((reps, N_OBS)) < LEVEL).astype(float)
    res = [require(adapter.christoffersen_cc(h, LEVEL), "christoffersen_cc")
           for h in hits]
    stat = np.array([r[0] for r in res])
    assert_size([r[1] for r in res], reps, expected=0.055,
                name="Christoffersen cc")
    for q in (0.50, 0.90, 0.95):
        emp, theo = np.quantile(stat, q), stats.chi2.ppf(q, df=2)
        assert abs(emp - theo) < 0.40 + 0.15 * theo, (
            f"CC quantile {q}: {emp:.3f} vs chi2(2) {theo:.3f} "
            "(mauvais nombre de degres de liberte ?)")


# =============================================================================
# Engle & Manganelli (2004) - Dynamic Quantile test
# =============================================================================


def _garch_path(rng, n, omega=0.02, a=0.08, b=0.90):
    e = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / (1 - a - b)
    for t in range(1, n):
        s2[t] = omega + a * e[t - 1] ** 2 + b * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return e, np.sqrt(s2)


def test_dq_size(rng):
    """Modele correct : VaR conditionnelle exacte sur un DGP GARCH."""
    reps = max(_mc_reps(600), 500)
    q = -stats.norm.ppf(LEVEL)
    pv = []
    for _ in range(reps):
        e, s = _garch_path(rng, N_OBS)
        var = q * s
        hits = (-e > var).astype(float)
        pv.append(require(adapter.dq_test(hits, var, LEVEL, 4), "dq_test")[1])
    assert_size(pv, reps, expected=0.050, name="DQ")


def test_dq_power_against_static_var(rng):
    """VaR constante appliquee a un risque variable : les violations se
    groupent ET dependent du niveau de VaR -> le DQ doit rejeter la ou Kupiec
    peut rester muet."""
    reps = _mc_reps(300)
    n = 1500
    q = -stats.norm.ppf(LEVEL)
    pv_dq, pv_uc = [], []
    for _ in range(reps):
        e, s = _garch_path(rng, n, omega=0.03, a=0.13, b=0.85)
        var = np.full(n, q * s.mean())
        hits = (-e > var).astype(float)
        pv_dq.append(require(adapter.dq_test(hits, var, LEVEL, 4), "dq_test")[1])
        pv_uc.append(require(adapter.kupiec_pof(hits, LEVEL), "kupiec_pof")[1])
    r_dq, r_uc = rejection_rate(pv_dq), rejection_rate(pv_uc)
    assert r_dq > 0.75, f"puissance DQ insuffisante ({r_dq:.2f})"
    # La couverture inconditionnelle reste correcte : Kupiec ne voit rien.
    assert r_uc < 0.35 and r_dq > 3.0 * r_uc


# =============================================================================
# Berkowitz (2001) - PIT, chi2(3)
# =============================================================================


def test_berkowitz_size(rng):
    reps = max(_mc_reps(400), 400)
    pv = [require(adapter.berkowitz(rng.random(N_OBS)), "berkowitz")[1]
          for _ in range(reps)]
    assert_size(pv, reps, expected=0.048, name="Berkowitz")


def test_berkowitz_power_against_scale_misspecification(rng):
    """Volatilite sous-estimee de 25 % : c'est l'alternative pour laquelle
    Berkowitz est concu (moyenne / variance du z-score). Puissance ~ 1."""
    reps = _mc_reps(300)
    pv = [require(adapter.berkowitz(stats.norm.cdf(rng.standard_normal(N_OBS) * 1.25)),
                  "berkowitz")[1]
          for _ in range(reps)]
    assert rejection_rate(pv) > 0.90


def test_berkowitz_has_limited_power_against_tail_shape(rng):
    """Limite CONNUE, encodee comme telle : contre une mauvaise forme de queue
    a variance correcte (Student(4) standardisee jugee gaussienne), Berkowitz
    ne rejette qu'environ un tiers du temps a n = 1000. Si ce test devient tres
    puissant, c'est que l'implementation ne teste pas ce qu'elle annonce."""
    reps = _mc_reps(300)
    scale = np.sqrt(4.0 / 2.0)
    pv = [require(adapter.berkowitz(stats.norm.cdf(rng.standard_t(4.0, N_OBS) / scale)),
                  "berkowitz")[1]
          for _ in range(reps)]
    r = rejection_rate(pv)
    assert 0.15 < r < 0.60, f"taux de rejet {r:.2f} inattendu"


# =============================================================================
# Acerbi & Szekely (2014) - Z1 / Z2
# =============================================================================


def _normal_var_es(alpha):
    z = stats.norm.ppf(alpha)
    return -z, stats.norm.pdf(z) / alpha


def test_as_z2_equals_one_without_exception():
    """Invariant exact : sans aucune violation de VaR, Z2 vaut 1.0, qui est
    aussi sa valeur maximale. Un ecart signale une erreur de normalisation."""
    alpha = 0.025
    v, e = _normal_var_es(alpha)
    r = np.full(250, 0.5)                      # aucune perte
    z2 = require(adapter.acerbi_szekely_z2(r, np.full(250, v), np.full(250, e),
                                           alpha), "acerbi_szekely_z2")
    assert np.isclose(float(z2), 1.0, atol=1e-12)


def test_as_z1_undefined_without_exception(rng):
    """Z1 est conditionnel aux violations : sans violation, il est indefini
    (nan), et surtout PAS 0."""
    alpha = 0.025
    v, e = _normal_var_es(alpha)
    ctrl = require(adapter.acerbi_szekely_z1(rng.standard_normal(2000),
                                             np.full(2000, v), np.full(2000, e),
                                             alpha), "acerbi_szekely_z1")
    assert np.isfinite(float(ctrl)), "Z1 devrait etre defini en presence de violations"

    r = np.full(250, 0.5)
    z1 = adapter.acerbi_szekely_z1(r, np.full(250, v), np.full(250, e), alpha)
    assert z1 is not None and np.isnan(float(z1)), (
        f"sans violation, Z1 doit valoir nan (obtenu {z1})")


def test_as_centred_under_h0(rng):
    """Sous le modele correct, Z1 et Z2 sont centres sur 0."""
    alpha = 0.025
    v, e = _normal_var_es(alpha)
    n, reps = 1000, _mc_reps(400)
    z1s, z2s = [], []
    for _ in range(reps):
        r = rng.standard_normal(n)
        z1s.append(require(adapter.acerbi_szekely_z1(r, np.full(n, v),
                                                     np.full(n, e), alpha),
                           "acerbi_szekely_z1"))
        z2s.append(adapter.acerbi_szekely_z2(r, np.full(n, v), np.full(n, e),
                                             alpha))
    assert abs(np.nanmedian(z1s)) < 0.03
    assert abs(np.median(z2s)) < 0.05


def test_as_negative_under_risk_underestimation(rng):
    """Les deux tests sont unilateraux : une statistique NEGATIVE signale une
    sous-estimation du risque. On evalue un VaR/ES gaussien sur des rendements
    Student(4) : les pertes en queue depassent l'ES annonce."""
    alpha = 0.025
    v, e = _normal_var_es(alpha)
    n = 5000
    r = rng.standard_t(4, n) / np.sqrt(2.0)
    z1 = require(adapter.acerbi_szekely_z1(r, np.full(n, v), np.full(n, e),
                                           alpha), "acerbi_szekely_z1")
    z2 = adapter.acerbi_szekely_z2(r, np.full(n, v), np.full(n, e), alpha)
    assert float(z1) < -0.05 and float(z2) < -0.05


@pytest.mark.slow
def test_as_z2_threshold_matches_literature(rng):
    """Reproduction d'un ordre de grandeur publie : sur 250 jours a 2.5 % sous
    hypothese gaussienne, le quantile a 5 % de Z2 vaut environ -0.70, seuil
    cite par Acerbi & Szekely (2014) et repris dans la litterature.

    Z1 est plus sensible au protocole de simulation ; on se contente d'un
    encadrement large.
    """
    alpha, n = 0.025, 250
    v, e = _normal_var_es(alpha)
    reps = _mc_reps(5000)
    z1s, z2s = [], []
    for _ in range(reps):
        r = rng.standard_normal(n)
        z1s.append(require(adapter.acerbi_szekely_z1(r, np.full(n, v),
                                                     np.full(n, e), alpha),
                           "acerbi_szekely_z1"))
        z2s.append(adapter.acerbi_szekely_z2(r, np.full(n, v), np.full(n, e),
                                             alpha))
    q_z2 = float(np.quantile(z2s, 0.05))
    q_z1 = float(np.nanquantile(z1s, 0.05))
    assert -0.78 < q_z2 < -0.62, f"quantile 5 % de Z2 = {q_z2:.3f}, attendu ~ -0.70"
    assert -0.25 < q_z1 < -0.06, f"quantile 5 % de Z1 = {q_z1:.3f}"
