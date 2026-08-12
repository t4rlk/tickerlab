# -*- coding: utf-8 -*-
"""
Tests d'intégration cross-modules (smoke test pré-merge).

Distinct des test_phase1X.py qui testent chaque sous-tâche isolément.
Ce fichier valide les ENCHAÎNEMENTS : 1.1→1.2, 1.2→1.3, 1.4→1.5.

Exécution : pytest tests/test_phase1.py -v  → 8/8 pass, < 90s.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings('ignore')


# ── Helpers simulation ────────────────────────────────────────────────────────

def _sim_garch11(omega, alpha, beta, n, seed, dist='normal'):
    """Simulate GARCH(1,1) returns in natural units."""
    rng = np.random.default_rng(seed)
    h = np.ones(n)
    r = np.zeros(n)
    h[0] = omega / max(1.0 - alpha - beta, 1e-6)
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        h[t] = max(h[t], 1e-8)
        if dist == 't':
            r[t] = np.sqrt(h[t]) * float(rng.standard_t(df=5))
        else:
            r[t] = np.sqrt(h[t]) * rng.standard_normal()
    return r


def _to_series(arr, start='2018-01-01'):
    return pd.Series(arr, index=pd.bdate_range(start, periods=len(arr)))


# ── Config helpers ────────────────────────────────────────────────────────────

def _cfg_garch_minimal():
    return dict(
        modeles=['GARCH'],
        distributions=['normal'],
        p_max=1,
        q_max=1,
        seuil_significativite=0.05,
        critere_information='BIC',
        tolerance_delta_critere_brut=4.0,
        seuil_igarch=0.98,
        score_composite={'enabled': False},
    )


def _cfg_pipeline_minimal():
    return {
        'garch': _cfg_garch_minimal(),
        'backtest': {'split_ratio': 0.70},
        'var':     {'niveaux': [0.95, 0.99]},
        'dm_gk':   {'alpha_test': 0.05, 'hac_lags': 'auto', 'n_instruments_gk': 2},
    }


# ── Fixtures module-scope (shared across tests) ───────────────────────────────

@pytest.fixture(scope='module')
def serie_bruit_blanc():
    """500 obs iid N(0,1), seed=42 — martingale."""
    rng = np.random.default_rng(42)
    return _to_series(rng.standard_normal(500))


@pytest.fixture(scope='module')
def serie_near_igarch():
    """GARCH(1,1) ω=0.05 α=0.10 β=0.89, seed=2 → fitted pers≈0.99."""
    r = _sim_garch11(omega=0.05, alpha=0.10, beta=0.89, n=600, seed=2)
    return _to_series(r)


@pytest.fixture(scope='module')
def serie_garch_simple():
    """GARCH(1,1) ω=0.20 α=0.05 β=0.90, seed=42 → pers≈0.95."""
    r = _sim_garch11(omega=0.20, alpha=0.05, beta=0.90, n=500, seed=42)
    return _to_series(r)


@pytest.fixture(scope='module')
def pipeline_near_igarch(serie_near_igarch):
    """(best_d, garch_final, config) fitted once for near-IGARCH series."""
    from tickerlab.core.garch_selector import (
        grid_search_garch, selectionner_meilleur, estimer_final,
    )
    cfg_g  = _cfg_garch_minimal()
    config = _cfg_pipeline_minimal()
    df_garch = grid_search_garch(serie_near_igarch, **cfg_g)
    best, _, _ = selectionner_meilleur(df_garch, serie_near_igarch, config)
    best_d = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
    garch_final = estimer_final(serie_near_igarch, **best_d)
    return best_d, garch_final, config


@pytest.fixture(scope='module')
def pipeline_garch_simple(serie_garch_simple):
    """(best_d, garch_final, config) fitted once for simple GARCH series."""
    from tickerlab.core.garch_selector import (
        grid_search_garch, selectionner_meilleur, estimer_final,
    )
    cfg_g  = _cfg_garch_minimal()
    config = _cfg_pipeline_minimal()
    df_garch = grid_search_garch(serie_garch_simple, **cfg_g)
    best, _, _ = selectionner_meilleur(df_garch, serie_garch_simple, config)
    best_d = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
    garch_final = estimer_final(serie_garch_simple, **best_d)
    return best_d, garch_final, config


# ── Test 1 : bruit blanc → martingale ───────────────────────────

def test_arima_interpretation_martingale(serie_bruit_blanc):
    """
    Pipeline 1.1 : série iid N(0,1) → interpretation == 'martingale_marche_efficient'.

    Vérifie que le sélecteur ARIMA reconnaît l'absence de structure
    prédictive et produit le bon code d'interpretation sur données vivantes
    (pas de mock — distinctif des tests unitaires test_phase11.py).
    """
    from tickerlab.core.arima_selector import selectionner_arima

    result = selectionner_arima(
        serie_bruit_blanc,
        d_arima=0,
        p_max=2,
        q_max=2,
        seuil_significativite=0.05,
        preference_parcimonie=True,
        tolerance_aic_parcimonie=10,
    )

    assert 'interpretation' in result, "selectionner_arima doit retourner 'interpretation'"
    # bruit_faible : coefficients significatifs mais |coef| ≤ 0.10 → pas de prédictibilité exploitable
    # martingale_marche_efficient : aucun modèle n'améliore le random walk
    # Les deux sont compatibles avec une série iid (faux positif type I à 5% par hasard)
    NON_PREDICTIBLE = {'martingale_marche_efficient', 'bruit_faible'}
    assert result['interpretation'] in NON_PREDICTIBLE, (
        f"Série iid N(0,1) devrait donner 'martingale_marche_efficient' ou 'bruit_faible', "
        f"obtenu '{result['interpretation']}' (motif={result.get('motif_selection')}). "
        f"Une interprétation 'serie_predictible' sur du bruit blanc serait anormale."
    )


# ── Test 2 : diagnostic IGARCH post-hoc ─────────────────────────

def test_igarch_diagnostic_post_hoc(pipeline_near_igarch):
    """
    Diagnostic post-hoc : diagnostiquer_igarch() sur série near-IGARCH
    retourne near_igarch=True + code str + half_life_periodes float > 0.

    L'IGARCH n'est plus dans la grille (diagnostic post-hoc).
    Ce test vérifie le contrat API de diagnostiquer_igarch().
    """
    from tickerlab.core.igarch_diagnostic import diagnostiquer_igarch

    best_d, garch_final, _ = pipeline_near_igarch

    diag = diagnostiquer_igarch(
        garch_final,
        vol_type=str(best_d.get('modele', 'GARCH')),
        seuil_igarch=0.98,
        frequence_serie='daily',
    )

    assert isinstance(diag, dict),          "diagnostiquer_igarch doit retourner un dict"
    assert 'near_igarch'       in diag,     "clé 'near_igarch' absente"
    assert 'code'              in diag,     "clé 'code' absente"
    assert 'half_life_periodes' in diag,    "clé 'half_life_periodes' absente"
    assert isinstance(diag['near_igarch'], bool), "'near_igarch' doit être bool"
    assert isinstance(diag['code'],        str),  "'code' doit être str"

    assert diag['near_igarch'] is True, (
        f"Série ω=0.05 α=0.10 β=0.89 seed=2 (pers_fit≈0.99) devrait donner "
        f"near_igarch=True. Obtenu code='{diag['code']}'. "
        f"Vérifier seuil_igarch=0.98 et la persistance estimée."
    )
    hl = diag['half_life_periodes']
    assert hl is not None and float(hl) > 0, (
        f"half_life_periodes doit être > 0 pour code='near_igarch', obtenu {hl}"
    )


# ── Test 3 : Component GARCH convergence ─────────────────────────

def test_component_garch_convergence():
    """
    Série simulée Component GARCH (ω=0.1, ρ=0.97, φ=0.05,
    α=0.05, β=0.85, ν=8) → ρ estimé dans [0.97 ± 25%] ET C3 respectée (α+β < ρ).

    Tolérance 25% : multi-start SLSQP + 500 obs → convergence locale possible.
    Seuls ρ et C3 sont testés — α, β, φ, ν ont une variance d'estimation élevée.
    """
    from arch import arch_model
    from tickerlab.core.component_garch import estimer_component_garch

    OMEGA, RHO, PHI, ALPHA, BETA, NU = 0.10, 0.97, 0.05, 0.05, 0.85, 8.0
    n   = 500
    rng = np.random.default_rng(42)

    # Simulation Component GARCH manuelle (Engle & Lee 1999)
    q = np.ones(n) * OMEGA / max(1.0 - RHO, 1e-6)
    h = np.ones(n) * q[0]
    e = np.zeros(n)
    for t in range(1, n):
        z    = float(rng.standard_t(df=NU))
        q[t] = OMEGA + RHO * q[t - 1] + PHI * (e[t - 1] ** 2 - h[t - 1])
        q[t] = max(q[t], 1e-8)
        h[t] = q[t] + ALPHA * (e[t - 1] ** 2 - q[t - 1]) + BETA * (h[t - 1] - q[t - 1])
        h[t] = max(h[t], 1e-8)
        e[t] = np.sqrt(h[t]) * z

    serie = _to_series(e)

    # Pré-estimation GARCH(1,1) requise par estimer_component_garch
    fit = arch_model(serie, vol='Garch', p=1, q=1, dist='t').fit(
        disp='off', show_warning=False
    )

    cfg_comp = {
        'component_garch': {
            'enabled':          True,
            'force_estimation': True,
            'n_starts':         3,
        }
    }
    result = estimer_component_garch(serie, fit, cfg_comp, near_igarch=True)

    assert result is not None, (
        "estimer_component_garch a retourné None (force_estimation=True). "
        "Vérifier enabled=True et les paramètres d'optimisation."
    )
    assert result.get('converged') is True, (
        f"Component GARCH n'a pas convergé. Résultat : {result}"
    )

    rho_est = float(result['rho'])
    tol     = 0.25
    rho_lo  = RHO * (1.0 - tol)
    rho_hi  = min(RHO * (1.0 + tol), 0.9999)
    assert rho_lo <= rho_est <= rho_hi, (
        f"ρ estimé={rho_est:.4f} hors tolérance 25% autour de {RHO} "
        f"(attendu [{rho_lo:.4f}, {rho_hi:.4f}]). "
        f"Multi-start SLSQP sur 500 obs — vérifier seed=42 ou n_starts."
    )

    alpha_est = float(result['alpha'])
    beta_est  = float(result['beta'])
    assert alpha_est + beta_est < rho_est, (
        f"Contrainte C3 violée : α+β={alpha_est + beta_est:.4f} >= ρ={rho_est:.4f}. "
        f"La décomposition permanente/transitoire est invalide (Engle & Lee 1999)."
    )


# ── Test 4 : FHS vs GARCH paramétrique ──────────────────────────

def test_fhs_vs_garch_parametrique(pipeline_garch_simple, serie_garch_simple):
    """
    VaR FHS et GARCH paramétrique du même ordre à H=1.

    |VaR_FHS - VaR_GARCH| / |VaR_GARCH| < 30% aux niveaux 95% et 99%.
    Vérifie la cohérence entre simulation bootstrap et VaR conditionnelle.
    """
    from tickerlab.core.fhs import calculer_var_fhs
    from scipy.stats import norm

    best_d, garch_final, _ = pipeline_garch_simple

    cfg_fhs = {
        'fhs': {
            'enabled':       True,
            'n_boot':        2000,
            'horizons':      [1],
            'seed':          42,
            'fenetre_residus': None,
        },
        'var': {'niveaux': [0.95, 0.99]},
    }

    fhs_result = calculer_var_fhs(
        serie_garch_simple, garch_final, cfg_fhs, niveaux=[0.95, 0.99]
    )
    assert fhs_result is not None, (
        "calculer_var_fhs a retourné None. Vérifier fhs.enabled=True dans cfg_fhs."
    )

    var_fhs = fhs_result['var_fhs_par_horizon'][1]  # H=1

    last_vol = float(garch_final.conditional_volatility.iloc[-1])
    mu       = float(garch_final.params.get('mu', float(serie_garch_simple.mean())))

    # VaR GARCH paramétrique (distribution normale — pipeline_garch_simple utilise dist='normal')
    var_garch = {
        0.95: mu + float(norm.ppf(1.0 - 0.95)) * last_vol,
        0.99: mu + float(norm.ppf(1.0 - 0.99)) * last_vol,
    }

    for alpha in [0.95, 0.99]:
        vf = float(var_fhs[alpha])
        vg = float(var_garch[alpha])
        assert vf < 0 and vg < 0, (
            f"VaR à {alpha:.0%} doit être négative — FHS={vf:.4f} GARCH={vg:.4f}"
        )
        ecart_rel = abs(vf - vg) / max(abs(vg), 1e-8)
        assert ecart_rel < 0.30, (
            f"À {alpha:.0%} : VaR_FHS={vf:.4f} vs VaR_GARCH={vg:.4f} "
            f"(écart relatif={ecart_rel:.1%} > 30%). "
            f"Bootstrap empirique vs normal paramétrique — divergence anormale."
        )


# ── Test 5 : symétrie DM ─────────────────────────────────────────

def test_dm_gk_symmetrie(serie_garch_simple):
    """
    DM(A, B) = -DM(B, A) par construction (d_t = L(A) - L(B)).

    Vérifie |dm_AB + dm_BA| < 0.01 (tolérance numérique HAC Newey-West).
    Tolérance 0.01 (pas 1e-6) : le bandwidth HAC est calculé indépendamment
    pour chaque appel, pouvant introduire un écart numérique résiduel.
    """
    from tickerlab.core.dm_gk import dm_test

    T_oos = 150
    rng   = np.random.default_rng(99)

    r_oos = serie_garch_simple.values[-T_oos:]

    # Deux séries VaR distinctes sur la période OOS
    base  = float(np.quantile(serie_garch_simple.values[:-T_oos], 1.0 - 0.95))
    var_a = np.full(T_oos, base)
    var_b = var_a * 1.20 + rng.normal(0, 0.02, T_oos)

    dm_ab = dm_test(var_a, var_b, r_oos, alpha=0.95)
    dm_ba = dm_test(var_b, var_a, r_oos, alpha=0.95)

    stat_ab = float(dm_ab['stat'])
    stat_ba = float(dm_ba['stat'])

    assert abs(stat_ab + stat_ba) < 0.01, (
        f"DM(A,B)={stat_ab:.6f} + DM(B,A)={stat_ba:.6f} = {stat_ab + stat_ba:.2e} "
        f"(|somme| ≥ 0.01 — asymétrie anormale de l'implémentation DM)."
    )
    assert abs(abs(stat_ab) - abs(stat_ba)) < 0.01, (
        f"|DM(A,B)|={abs(stat_ab):.6f} ≠ |DM(B,A)|={abs(stat_ba):.6f} "
        f"(diff={abs(abs(stat_ab) - abs(stat_ba)):.2e} ≥ 0.01)."
    )


# ── Test 6 : bootstrap express IC contient VaR GARCH ─────────────

def test_bootstrap_express_taille_ic():
    """
    Bootstrap express : IC_inf ≤ VaR_GARCH_ponctuel ≤ IC_sup aux niveaux
    95% et 99%, sur 3 séries (vol forte, vol faible, queues épaisses).

    Bootstrap conditionnel Pascual-Romo-Ruiz (2006) : l'estimateur ponctuel
    est par construction dans l'IC puisqu'on fixe les params GARCH.
    """
    from arch import arch_model
    from tickerlab.core.var_engine import calculer_bootstrap_ci_var

    cfg_express = {
        'bootstrap': {
            'express':       True,
            'n_replications': 200,
            'block_length':  10,
            'niveaux_ic':    [0.95, 0.99],
            'seed':          42,
        }
    }

    n   = 400
    series = {
        'vol_forte':      _to_series(_sim_garch11(0.50, 0.08, 0.88, n, seed=1)),
        'vol_moderee':    _to_series(_sim_garch11(0.15, 0.04, 0.90, n, seed=4)),
        'queues_epaisses': _to_series(_sim_garch11(0.20, 0.06, 0.88, n, seed=3, dist='t')),
    }

    for nom, s in series.items():
        fit = arch_model(s, vol='Garch', p=1, q=1, dist='normal').fit(
            disp='off', show_warning=False
        )
        df_ic = calculer_bootstrap_ci_var(s, fit, cfg_express)

        assert not df_ic.empty, (
            f"Bootstrap express vide pour '{nom}' — vérifier config express=True."
        )

        for niv_str in df_ic.index:
            row    = df_ic.loc[niv_str]
            var_pt = float(row['VaR GARCH'])
            ci_lo  = float(row['CI lower'])
            ci_hi  = float(row['CI upper'])

            assert ci_lo < ci_hi, (
                f"IC dégénéré pour '{nom}' niveau {niv_str} : "
                f"lower={ci_lo:.4f} >= upper={ci_hi:.4f}"
            )
            assert ci_lo <= var_pt <= ci_hi, (
                f"VaR GARCH ponctuel hors IC pour '{nom}' niveau {niv_str} : "
                f"IC=[{ci_lo:.4f}, {ci_hi:.4f}], VaR={var_pt:.4f}. "
                f"Bootstrap conditionnel — l'estimateur ponctuel doit être dans l'IC."
            )


# ── Test 7 — Intégration 1.2→1.3 : near-IGARCH → Component GARCH ────────────

def test_igarch_diagnostic_declenche_component_garch(
    pipeline_near_igarch, serie_near_igarch
):
    """
    Pipeline 1.2 → 1.3 : si near_igarch=True détecté par diagnostiquer_igarch(),
    Component GARCH est estimé automatiquement (force_estimation=False).

    Anti-régression contre découplage silencieux du pipeline 1.2/1.3.
    Vérifie aussi la contrainte Engle-Lee C3 : α+β < ρ.
    """
    from tickerlab.core.igarch_diagnostic import diagnostiquer_igarch
    from tickerlab.core.component_garch import estimer_component_garch

    best_d, garch_final, _ = pipeline_near_igarch

    # Étape 1 : diagnostic IGARCH (réutilise le pipeline déjà fitté)
    diag = diagnostiquer_igarch(
        garch_final,
        vol_type=str(best_d.get('modele', 'GARCH')),
        seuil_igarch=0.98,
        frequence_serie='daily',
    )
    assert diag['near_igarch'] is True, (
        f"Pré-condition : near_igarch doit être True (série pers≈0.99). "
        f"Obtenu code='{diag['code']}' — précondition du test non satisfaite."
    )

    # Étape 2 : Component GARCH déclenché via near_igarch=True, force_estimation=False
    cfg_comp = {
        'component_garch': {
            'enabled':          True,
            'force_estimation': False,   # contrat : near_igarch=True seul suffit
            'n_starts':         3,
        }
    }
    result = estimer_component_garch(
        serie_near_igarch,
        garch_final,
        cfg_comp,
        near_igarch=True,   # transmis depuis diag['near_igarch']
    )

    assert result is not None, (
        "estimer_component_garch a retourné None malgré near_igarch=True. "
        "Contrat rompu : enabled=True AND near_igarch=True doit déclencher l'estimation "
        "(indépendamment de force_estimation)."
    )

    # Étape 3 : contrainte Engle-Lee C3 (α+β < ρ)
    alpha_est = float(result['alpha'])
    beta_est  = float(result['beta'])
    rho_est   = float(result['rho'])
    assert alpha_est + beta_est < rho_est, (
        f"Contrainte C3 violée sur données near-IGARCH : "
        f"α+β={alpha_est + beta_est:.4f} >= ρ={rho_est:.4f}. "
        f"La composante transitoire ne peut pas dominer la composante permanente."
    )


# ── Test 8 — Intégration 1.4→1.5 : FHS dans DM-GK ───────────────────────────

def test_fhs_apparait_dans_dm_gk(pipeline_garch_simple, serie_garch_simple):
    """
    Pipeline 1.4 → 1.5 : FHS produit une ligne dans df_bt, DM-GK l'inclut
    dans sa matrice de comparaison.

    Anti-régression contre 'FHS sort du tableau silencieusement'.
    FHS est calculé automatiquement dans backtest_oos() — pas d'injection manuelle.
    """
    from tickerlab.core.backtest import backtest_oos
    from tickerlab.core.dm_gk import build_var_series, comparer_methodes_var

    best_d, garch_final, config = pipeline_garch_simple

    # Étape 1 : backtest OOS (FHS calculé automatiquement via fhs_var_oos interne)
    df_bt, T_train, T_eff_dyn = backtest_oos(
        serie_garch_simple,
        best_d,
        split_ratio=0.70,
        niveaux_test=[0.95, 0.99],
        n_simulations_mc=1000,   # réduit pour vitesse smoke test
    )

    # Étape 2 : FHS présent dans df_bt
    assert 'Methode' in df_bt.columns, "df_bt doit avoir une colonne 'Methode'"
    methodes_bt = df_bt['Methode'].values
    assert 'FHS' in methodes_bt, (
        f"'FHS' absent de df_bt['Methode']. Méthodes trouvées : {sorted(set(methodes_bt))}. "
        f"backtest_oos() doit calculer FHS automatiquement (fhs_var_oos interne)."
    )

    fhs_rows = df_bt[df_bt['Methode'] == 'FHS']
    assert not fhs_rows.empty, "Aucune ligne FHS dans df_bt"
    assert (fhs_rows['N viol.'] >= 0).all(), "Violations FHS négatives impossible"

    # Étape 3 : build_var_series inclut FHS via garch_final résidus (anti look-ahead)
    # fhs_result={} (non-None) active le chemin FHS dans build_var_series
    var_series = build_var_series(
        serie_garch_simple,
        garch_final,
        fhs_result={},   # non-None déclenche le calcul FHS
        config=config,
        T_train=T_train,
    )
    assert 'FHS' in var_series, (
        f"'FHS' absent de var_series. Méthodes : {sorted(var_series.keys())}. "
        f"build_var_series doit inclure FHS quand fhs_result est non-None."
    )

    for alpha in [0.95, 0.99]:
        assert alpha in var_series['FHS'], f"Niveau {alpha} absent de var_series['FHS']"
        fhs_arr = np.asarray(var_series['FHS'][alpha], dtype=float)
        assert len(fhs_arr) > 0,                    "var_series FHS vide"
        assert np.all(np.isfinite(fhs_arr)),         "VaR FHS contient des non-finis"
        assert np.all(fhs_arr < 0), (
            f"VaR FHS doit être négative (perte) pour alpha={alpha}, "
            f"obtenu max={fhs_arr.max():.4f}"
        )

    # Étape 4 : DM-GK reçoit FHS et compare au moins une paire
    r_oos = serie_garch_simple.values[T_train: T_train + T_eff_dyn]
    dm_result = comparer_methodes_var(
        var_series, r_oos, niveaux=[0.95, 0.99], config=config
    )

    assert 'methodes' in dm_result, "dm_result manque la clé 'methodes'"
    assert 'FHS' in dm_result['methodes'], (
        f"'FHS' absent de dm_result['methodes'] : {dm_result['methodes']}. "
        f"build_var_series n'a pas fourni FHS à comparer_methodes_var."
    )

    paires_fhs_95 = [k for k in dm_result.get(0.95, {}) if 'FHS' in k]
    assert len(paires_fhs_95) >= 1, (
        f"Aucune paire impliquant FHS dans dm_result[0.95]. "
        f"Paires disponibles : {list(dm_result.get(0.95, {}).keys())}."
    )
