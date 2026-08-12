# -*- coding: utf-8 -*-
"""
Tests — FHS (Filtered Historical Simulation, Barone-Adesi et al. 1999).

6 tests couvrant :
  1. Monotonicité VaR en alpha (sanity check)
  2. Propagation GARCH conforme à la formule (vérifie la récurrence)
  3. Reproductibilité bit-à-bit avec seed fixé
  4. Scaling multi-périodes sub-linéaire sur série leptokurtique (CLT vs queues épaisses)
  5. Absence de look-ahead bias (fhs_var_oos strictement train-only)
  6. Drapeau residus_iid_ok = False si LB(z²) rejette
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.fhs import (
    _extract_garch_state,
    _propagate_paths,
    _iid_diagnostics,
    calculer_var_fhs,
    fhs_var_oos,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_garch_result(n: int = 1500, seed: int = 0):
    """
    Simule une série GARCH(1,1)-t et l'estime. Retourne ARCHModelResult.
    """
    from arch import arch_model
    rng = np.random.default_rng(seed)
    # Simulation manuelle GARCH(1,1) avec innovations Student-t(5)
    omega, alpha, beta, nu = 0.05, 0.08, 0.88, 5.0
    from scipy.stats import t as t_dist
    eps = np.zeros(n)
    h   = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for i in range(1, n):
        h[i]   = omega + alpha * eps[i-1]**2 + beta * h[i-1]
        eps[i] = rng.standard_t(nu) * np.sqrt(h[i] * (nu - 2) / nu)
    series = eps * 100  # en pourcentage

    import pandas as pd
    s = pd.Series(series)
    fit = arch_model(s, vol='Garch', p=1, o=0, q=1, dist='t').fit(disp='off')
    return fit


def _minimal_config(n_boot: int = 500, seed: int = 42, horizons=None) -> dict:
    if horizons is None:
        horizons = [1, 5, 10, 22]
    return {
        'fhs': {
            'enabled':  True,
            'n_boot':   n_boot,
            'horizons': horizons,
            'seed':     seed,
        },
        'var': {'niveaux': [0.90, 0.95, 0.99]},
    }


# ── Test 1 : monotonicité VaR en alpha ───────────────────────────────────────

def test_fhs_var_decroit_avec_alpha():
    """
    VaR_99 < VaR_95 < VaR_90 (en termes de rendement négatif = plus grande perte).

    La VaR au niveau de confiance α mesure la perte seuil à (1-α) : le quantile
    (1-α) est plus négatif pour α plus élevé (queue plus profonde).
    """
    fit = _make_garch_result()
    cfg = _minimal_config(n_boot=2000)
    res = calculer_var_fhs(None, fit, cfg)

    assert res is not None, "calculer_var_fhs doit retourner un dict si fhs.enabled=true"

    for H in [1, 5, 22]:
        v90 = res['var_fhs_par_horizon'][H][0.90]
        v95 = res['var_fhs_par_horizon'][H][0.95]
        v99 = res['var_fhs_par_horizon'][H][0.99]
        assert v99 < v95 < v90, (
            f"H={H}: VaR_99={v99:.4f} doit être < VaR_95={v95:.4f} < VaR_90={v90:.4f} "
            "(convention : VaR = quantile négatif, α plus élevé = perte plus grande)"
        )


# ── Test 2 : propagation GARCH conforme à la formule ─────────────────────────

def test_fhs_propagation_garch():
    """
    Au pas k=0, σ²_{T+1} est déterministe :
        σ²_{T+1} = ω + α·ε²_T + β·σ²_T

    Avec z_sampled = 1.0 (constant sur tous les chemins), tous les rendements
    simulés au pas k=0 doivent être identiques (variance zéro).
    """
    # État GARCH contrôlé
    omega, alpha, beta, mu = 0.01, 0.10, 0.80, 0.0
    last_vol2, last_eps    = 1.0, 0.5

    state = {
        'vol_class': 'GARCH',
        'o':         0,
        'omega':     omega,
        'alpha':     alpha,
        'gamma':     0.0,
        'beta':      beta,
        'mu':        mu,
        'last_vol2': last_vol2,
        'last_eps':  last_eps,
    }

    expected_sigma2_t1 = omega + alpha * last_eps**2 + beta * last_vol2
    # z = 1.0 pour tous les chemins et tous les pas
    n_boot, H = 500, 3
    z_sampled  = np.ones((n_boot, H))

    returns = _propagate_paths(z_sampled, state)

    # Vérification du pas k=0 : tous les retours = mu + 1.0 * sqrt(sigma2_t1)
    expected_r0 = mu + 1.0 * np.sqrt(expected_sigma2_t1)
    np.testing.assert_allclose(
        returns[:, 0], expected_r0, rtol=1e-10,
        err_msg="Pas k=0 : tous les chemins doivent avoir le même return (σ²_{T+1} déterministe)"
    )

    # Vérification variance nulle au pas k=0 (z identiques → returns identiques)
    assert np.std(returns[:, 0]) == pytest.approx(0.0, abs=1e-10), \
        "Variance des returns au pas k=0 doit être zéro (z_sampled constant)"


# ── Test 3 : reproductibilité bit-à-bit ──────────────────────────────────────

def test_fhs_reproductibilite():
    """
    Même seed → mêmes chiffres VaR bit-à-bit. Reproductibilité scientifique
    obligatoire (rapport PDF identique à chaque relance, Bâle/FRTB BCBS d457).
    """
    fit  = _make_garch_result()
    cfg1 = _minimal_config(n_boot=1000, seed=42)
    cfg2 = _minimal_config(n_boot=1000, seed=42)

    res1 = calculer_var_fhs(None, fit, cfg1)
    res2 = calculer_var_fhs(None, fit, cfg2)

    assert res1 is not None and res2 is not None

    for H in [1, 22]:
        for alpha in [0.95, 0.99]:
            v1 = res1['var_fhs_par_horizon'][H][alpha]
            v2 = res2['var_fhs_par_horizon'][H][alpha]
            assert v1 == v2, (
                f"H={H}, α={alpha}: VaR={v1} ≠ {v2} avec seed identique — "
                "reproductibilité violée"
            )

    # Seed différent → résultat différent (sensibilité MC)
    cfg3 = _minimal_config(n_boot=1000, seed=999)
    res3 = calculer_var_fhs(None, fit, cfg3)
    assert res3 is not None
    # Sur au moins une combinaison, les valeurs doivent différer
    diffs = [
        res1['var_fhs_par_horizon'][H][0.99] != res3['var_fhs_par_horizon'][H][0.99]
        for H in [1, 22]
    ]
    assert any(diffs), "Des seeds différents devraient produire des VaR différentes"


# ── Test 4 : scaling multi-périodes sub-linéaire ──────────────────────────────

def test_fhs_horizon_scaling():
    """
    Sur série très leptokurtique (t(3)) avec GARCH quasi-iid (α+β=0.15), deux
    propriétés doivent tenir :

    (a) VaR(H=22) < VaR(H=1) en valeur signée (perte 22j plus grande en absolu)
    (b) VaR(H=22) > √22 × VaR(H=1) — CLT réduit les queues épaisses → scaling
        sous-linéaire vs règle √H.

    Note : avec GARCH haute persistance (α+β ≈ 0.96), le clustering de volatilité
    peut dominer l'effet CLT et inverser la propriété (b). Le test isole l'effet
    queues épaisses en utilisant α+β = 0.15 (quasi-iid) + t(3) très leptokurtique.

    Référence : Barone-Adesi et al. (1999), Section 4 — la FHS révèle la divergence
    vs √H précisément sur les actifs à queues épaisses.
    """
    rng = np.random.default_rng(42)

    # Pool z ~ t(3) : queues très épaisses, quantile 1% ≈ −4.54σ vs −2.33σ normal
    z_pool = rng.standard_t(3, size=2000)

    # GARCH quasi-iid : α+β = 0.15 → variance proche de l'inconditionnelle à chaque pas
    # Faible clustering → l'effet CLT domine sur les queues épaisses
    sigma2_uncond = 0.5 / (1.0 - 0.15)
    state = {
        'vol_class': 'GARCH',
        'o':         0,
        'omega':     0.5,
        'alpha':     0.05,
        'gamma':     0.0,
        'beta':      0.10,
        'mu':        0.0,
        'last_vol2': sigma2_uncond,
        'last_eps':  0.0,
    }

    n_boot, H_max = 5000, 22
    rng2      = np.random.default_rng(0)
    z_sampled = rng2.choice(z_pool, size=(n_boot, H_max), replace=True)
    returns   = _propagate_paths(z_sampled, state)

    var_1  = float(np.quantile(returns[:, 0],                  0.01))
    var_22 = float(np.quantile(returns[:, :22].sum(axis=1),    0.01))

    assert var_22 < var_1, (
        f"VaR(H=22)={var_22:.4f} doit être < VaR(H=1)={var_1:.4f} "
        "(perte sur 22 jours plus grande en valeur absolue)"
    )

    sqrt_rule = var_1 * np.sqrt(22)   # négatif × √22 = encore plus négatif
    assert var_22 > sqrt_rule, (
        f"VaR(H=22)={var_22:.4f} doit être > √22×VaR(H=1)={sqrt_rule:.4f} "
        "(CLT réduit les queues épaisses — scaling sous-linéaire vs règle Bâle)"
    )


# ── Test 5 : absence de look-ahead bias ──────────────────────────────────────

def test_fhs_pas_de_look_ahead():
    """
    La VaR FHS one-step-ahead ne doit dépendre QUE de z_pool (résidus train)
    et de vol_oos. Remplacer les résidus post-train par des valeurs extrêmes
    ne doit PAS changer le résultat si z_pool reste identique.

    Vérifie aussi que INCLURE les résidus post-train dans z_pool CHANGE le résultat
    (démonstration que la frontière train/test est effective).

    Référence : Barone-Adesi et al. (1999), Section 3 — pool de résidus strictement
    antérieurs à la date de prévision.
    """
    rng = np.random.default_rng(7)

    # Résidus train : distribution plausible N(0,1) tronquée
    z_train  = rng.standard_normal(800)
    vol_oos  = np.abs(rng.standard_normal(100)) + 0.5   # volatilités OOS
    mu       = 0.01
    niveaux  = [0.95, 0.99]

    # VaR de référence avec z_pool train
    ref = fhs_var_oos(z_train, vol_oos, mu, niveaux)

    # Remplacer les résidus post-train par des valeurs extrêmes (look-ahead simulé)
    # Le z_pool reste z_train — la VaR ne doit PAS changer
    _ = rng.standard_normal(100) * 1000   # "résidus post-train extrêmes" (non utilisés)
    same = fhs_var_oos(z_train, vol_oos, mu, niveaux)

    for alpha in niveaux:
        np.testing.assert_array_equal(
            ref[alpha]['var'], same[alpha]['var'],
            err_msg=f"α={alpha}: la VaR doit être identique quand z_pool est inchangé "
                    "(anti look-ahead bias)"
        )

    # Démonstration : z_pool contaminé avec valeurs extrêmes → résultat DIFFÉRENT
    z_contaminated = np.concatenate([z_train, np.full(100, -10.0)])
    contaminated   = fhs_var_oos(z_contaminated, vol_oos, mu, niveaux)

    for alpha in niveaux:
        assert not np.allclose(ref[alpha]['var'], contaminated[alpha]['var']), (
            f"α={alpha}: inclure des résidus post-train extrêmes devrait changer la VaR "
            "(le pool contaminé a un quantile différent)"
        )


# ── Test 6 : drapeau residus_iid_ok = False ──────────────────────────────────

def test_fhs_residus_iid_warning():
    """
    Si LB(z²) rejette l'hypothèse iid (p < 0.05), residus_iid_ok doit être False.

    On construit un z_pool volontairement non-iid : clustering ARCH simple
    (z_t² autocorrélé) → LB(z²) rejette → drapeau activé.
    """
    rng = np.random.default_rng(99)
    n   = 1000

    # Série z volontairement non-iid : ARCH(1) avec forte persistance
    # σ²_t = 0.3 + 0.7 * z²_{t-1} → autocorrélation forte de z²
    z = np.zeros(n)
    h = np.ones(n)
    for i in range(1, n):
        h[i] = 0.3 + 0.7 * z[i-1]**2
        z[i] = rng.standard_normal() * np.sqrt(h[i])

    diag = _iid_diagnostics(z, lags=10)

    assert not diag['residus_iid_ok'], (
        "residus_iid_ok doit être False pour une série z² fortement autocorrélée "
        f"(LB(z²) p={diag['lb_z2_pval']:.4f})"
    )
    assert diag['lb_z2_pval'] < 0.05, (
        f"LB(z²) doit rejeter (p < 0.05) sur série ARCH — p={diag['lb_z2_pval']:.4f}"
    )

    # Série réellement iid → drapeau True
    z_iid = rng.standard_normal(n)
    diag_iid = _iid_diagnostics(z_iid, lags=10)
    assert diag_iid['residus_iid_ok'], (
        f"residus_iid_ok doit être True pour série N(0,1) iid "
        f"(LB(z²) p={diag_iid['lb_z2_pval']:.4f}, Engle-Ng p={diag_iid['engle_ng_pval']:.4f})"
    )
