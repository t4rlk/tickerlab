# -*- coding: utf-8 -*-
"""
Tests Phase 1.5 — DM (Diebold-Mariano 1995) + GK (Giacomini-Komunjer 2005).

6 tests couvrant :
  1. tick_loss nulle a la frontiere exacte (u_t = 0)
  2. dm_test retourne stat=0 si var1 = var2 (perte differentielle nulle)
  3. dm_test detecte le meilleur modele sur serie simulee
  4. gk_test non-rejet si d_t iid (instruments non-informatifs)
  5. Structure dict complet de comparer_methodes_var
  6. CRITIQUE — gk_test detecte l'instabilite temporelle (DM non rejete, GK rejete)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tickerlab.core.dm_gk import (
    tick_loss,
    dm_test,
    gk_test,
    _determine_verdict,
    comparer_methodes_var,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_oos(T: int = 500, seed: int = 0) -> tuple:
    """Serie OOS simple : r ~ N(0,1), var = cte correctement calibree."""
    rng = np.random.default_rng(seed)
    r   = rng.standard_normal(T)
    return r


def _minimal_config(alpha_test: float = 0.05) -> dict:
    return {
        'dm_gk': {
            'enabled':                   True,
            'alpha_test':                alpha_test,
            'alpha_test_petit_echantillon': 0.10,
            'hac_lags':                  'auto',
            'n_instruments_gk':          2,
            'paires_section_principale': ['A_vs_B'],
        },
        'var': {'niveaux': [0.95, 0.99]},
    }


# ── Test 1 : tick_loss nulle a la frontiere ───────────────────────────────────

def test_tick_loss_zero_a_la_frontiere():
    """
    u_t = r_t - VaR_t = 0 exactement (violation a la frontiere, pas a l'interieur)
    => L_alpha(0) = 0 * (alpha - 1{0 < 0}) = 0 * alpha = 0.

    Note : u_t = 0 signifie que le retour tombe exactement sur la VaR — cas limite
    qui n'est ni une violation (r < VaR) ni un non-evenement strict (r > VaR).
    La perte tick est nulle par definition continue de la perte quantile.
    """
    r   = np.zeros(100)
    var = np.zeros(100)    # u_t = r_t - VaR_t = 0 partout

    for alpha in [0.90, 0.95, 0.99]:
        loss = tick_loss(r, var, alpha)
        np.testing.assert_array_equal(
            loss, np.zeros(100),
            err_msg=f"alpha={alpha}: tick_loss doit etre 0 quand u_t=0 partout"
        )


# ── Test 2 : dm_test stat=0 si var1 = var2 ───────────────────────────────────

def test_dm_stat_zero_si_var_identiques():
    """
    Si var1 = var2 exactement, la perte differentielle d_t = 0 partout.
    => stat DM = 0, p-valeur = 1.0.
    """
    rng = np.random.default_rng(1)
    r   = rng.standard_normal(600)
    var = rng.standard_normal(600) * 0.5 - 1.5   # VaR quelconque

    res = dm_test(var, var, r, alpha=0.99)

    assert res['stat'] == pytest.approx(0.0, abs=1e-8), (
        f"DM stat={res['stat']} doit etre 0 si var1=var2"
    )
    assert res['pval'] == pytest.approx(1.0, abs=1e-8), (
        f"DM pval={res['pval']} doit etre 1.0 si var1=var2"
    )
    assert res['loss_diff_mean'] == pytest.approx(0.0, abs=1e-8)


# ── Test 3 : dm_test detecte le meilleur modele ───────────────────────────────

def test_dm_detecte_meilleur_modele():
    """
    Modele 1 (var1) est exact : VaR = vrai quantile de r.
    Modele 2 (var2) est systematiquement biaise vers 0 (sous-estime la perte).

    => var2 produit plus de violations => perte tick plus elevee pour var2.
    => d_bar = L(var1) - L(var2) < 0 => DM doit rejeter (model1_wins).

    Convention : d_bar < 0 => model1 a perte plus faible => model1 gagne.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(42)
    T   = 2000
    r   = rng.standard_normal(T)
    alpha = 0.99

    # var1 = vrai quantile N(0,1) au niveau alpha
    q_true = float(norm.ppf(1.0 - alpha))
    var1   = np.full(T, q_true)

    # var2 = quantile sous-estime (biais positif = moins prudent)
    var2 = np.full(T, q_true + 1.0)   # VaR trop haute -> moins de violations -> mauvais

    # Avec var2 trop haute (moins de violations), perte tick est plus elevee
    # car les violations manquees sont penalisees.
    # Mais attendez — avec alpha=0.99 et L_alpha(u) = u*(alpha - 1{u<0}) :
    # Si VaR est trop haute (var2 > var1), alors r_t < var2 moins souvent
    # => moins de violations => mais les non-violations ont u_t = r_t - var2 < 0
    # => penalite alpha * u_t > 0 (car u_t > 0 pour non-violations)
    # En fait, var2 > var1 => u_t = r_t - var2 < r_t - var1
    # Pour r_t > var1 (non-violation pour var1) : u > 0 => L = alpha * u < alpha*(r-var1)
    # Sous-estimation penalise plus que surestimation sous tick loss.
    # Utiliser biais negatif pour var2 (surestimation de la perte = trop prudent).

    # Reconcevoir : var2 = quantile sur-estime (trop prudent)
    var2 = np.full(T, q_true - 1.0)   # VaR trop basse -> plus de violations

    res  = dm_test(var1, var2, r, alpha=alpha, alpha_test=0.05)
    loss1 = tick_loss(r, var1, alpha).mean()
    loss2 = tick_loss(r, var2, alpha).mean()

    # var1 est exact => perte plus faible que var2 sur-conservative
    assert loss1 < loss2, (
        f"var1 exact doit avoir perte tick inferieure a var2 sur-conservative : "
        f"L(var1)={loss1:.6f} vs L(var2)={loss2:.6f}"
    )

    # d_bar = L(var1) - L(var2) < 0 => model1 gagne
    assert res['loss_diff_mean'] < 0, (
        f"d_bar={res['loss_diff_mean']:.6f} doit etre < 0 (var1 meilleur)"
    )
    # DM doit rejeter sur T=2000
    assert res['pval'] < 0.05, (
        f"DM pval={res['pval']:.4f} doit rejeter (modele 1 systematiquement meilleur "
        f"sur T={T} observations)"
    )


# ── Test 4 : gk_test non-rejet si d_t iid ────────────────────────────────────

def test_gk_non_rejet_si_iid():
    """
    Si la perte differentielle d_t est iid (aucune autocorrelation), les instruments
    Z_t = [1, d_{t-1}] ne sont pas informatifs sur d_t => GK ne doit pas rejeter.

    On simule d_t ~ N(0,1) iid (directement — sans passer par var1/var2).
    On verifie que GK produit une p-valeur uniforme sous H0 sur de nombreuses replications.
    Un seul test peut echouer par hasard ; on teste la p-valeur directement sur 1 rep.
    """
    rng = np.random.default_rng(77)
    T   = 1000

    # d_t iid => simuler des VaR qui donnent une difference d_t ~ N(0,1)
    # Methode : var1 = var2 + bruit iid => d_t est asymptotiquement iid
    r    = rng.standard_normal(T) * 2.0 - 3.0
    base = np.full(T, -2.33)
    var1 = base + rng.standard_normal(T) * 0.1
    var2 = base + rng.standard_normal(T) * 0.1

    res = gk_test(var1, var2, r, alpha=0.99, n_instruments=2, alpha_test=0.05)

    # Pas de rejet attendu sous H0 (par construction d_t approximativement iid)
    # Note : test peut echouer par chance (~5%), mais avec T=1000 et bruit N(0,1)
    # l'autocorrelation de d_t doit etre proche de 0 -> pas de rejet typiquement
    # On verifie uniquement que la statistique est finie et positive
    assert np.isfinite(res['stat']), f"GK stat non finie : {res['stat']}"
    assert res['stat'] >= 0.0, f"GK stat negative (forme quadratique impossible) : {res['stat']}"
    assert 0.0 <= res['pval'] <= 1.0, f"GK pval hors [0,1] : {res['pval']}"
    # Sur plusieurs seeds, GK ne doit pas rejeter systematiquement
    rejections = 0
    for seed in range(20):
        rng_s = np.random.default_rng(seed * 1000 + 7)
        r_s    = rng_s.standard_normal(T) * 2.0 - 3.0
        v1_s   = base + rng_s.standard_normal(T) * 0.1
        v2_s   = base + rng_s.standard_normal(T) * 0.1
        r_s_gk = gk_test(v1_s, v2_s, r_s, alpha=0.99, n_instruments=2)
        if r_s_gk['pval'] < 0.05:
            rejections += 1
    # Sous H0, taux de rejet ~ 5% — sur 20 replications, espere 1 rejet.
    # On tolere jusqu'a 6 rejets (99.9% CI binomial(20, 0.05)).
    assert rejections <= 6, (
        f"GK rejette sur {rejections}/20 series iid (attendu ~1/20 sous H0)"
    )


# ── Test 5 : structure dict complet comparer_methodes_var ────────────────────

def test_structure_retour_comparer():
    """
    comparer_methodes_var doit retourner un dict avec :
    - 'paires_principales', 'alpha_test', 'n_methodes', 'n_paires', 'methodes'
    - Pour chaque alpha : dict de paires avec cles obligatoires
    """
    rng = np.random.default_rng(5)
    T   = 300
    r_oos = rng.standard_normal(T) - 2.0

    var_series = {
        'A': {0.95: np.full(T, -1.65), 0.99: np.full(T, -2.33)},
        'B': {0.95: np.full(T, -1.80), 0.99: np.full(T, -2.50)},
    }
    cfg = _minimal_config()
    res = comparer_methodes_var(var_series, r_oos, [0.95, 0.99], cfg)

    # Cles meta
    assert 'paires_principales' in res, "Cle 'paires_principales' manquante"
    assert 'alpha_test' in res, "Cle 'alpha_test' manquante"
    assert 'n_methodes' in res and res['n_methodes'] == 2
    assert 'n_paires'   in res and res['n_paires']   == 1   # C(2,2)=1

    # Cles par alpha
    for alp in [0.95, 0.99]:
        assert alp in res, f"Alpha {alp} manquant dans le retour"
        paire_key = 'A_vs_B'
        assert paire_key in res[alp], f"Paire '{paire_key}' manquante pour alpha={alp}"
        info = res[alp][paire_key]
        cles_attendues = ['model1', 'model2', 'dm_stat', 'dm_pval', 'gk_stat',
                          'gk_pval', 'gk_df', 'loss_diff_mean', 'verdict', 'T_eff']
        for cle in cles_attendues:
            assert cle in info, f"Cle '{cle}' manquante dans la paire A_vs_B, alpha={alp}"

    # Verdict dans les 5 valeurs autorisees
    verdicts_ok = {'model1_wins', 'model2_wins', 'egalite_stable',
                   'egalite_avec_instabilite', 'non_discriminable'}
    for alp in [0.95, 0.99]:
        v = res[alp]['A_vs_B']['verdict']
        assert v in verdicts_ok, f"Verdict inconnu '{v}' — doit etre dans {verdicts_ok}"


# ── Test 6 : CRITIQUE — GK detecte l'instabilite temporelle ─────────────────

def test_gk_detecte_instabilite_temporelle():
    """
    Sur serie a regime changeant a mi-parcours :
    - [0, T/2] : modele 1 bat modele 2 (d_t < 0)
    - [T/2, T] : modele 2 bat modele 1 (d_t > 0)

    => d_bar moyen ≈ 0 => DM NE DOIT PAS rejeter (perte moyenne egale).
    => d_{t-1} est informatif sur d_t (changement de signe) => GK DOIT rejeter.

    C'est le test qui justifie l'ajout de GK au pipeline.
    Sans ce resultat, GK n'apporte rien de plus que DM.

    Reference : Giacomini & Komunjer (2005, JBES 23:4), Section 5 — le test GK
    est concu pour detecter ce type d'instabilite que DM manque systematiquement.
    """
    rng = np.random.default_rng(314)
    T   = 1000
    alpha = 0.99

    r = rng.standard_normal(T) * 2.0 - 3.0   # rendements negatifs ~ pertes

    # Construire var1, var2 tels que d_t change de signe a mi-parcours
    half = T // 2
    # Premiere moitie : var1 exactement calibree, var2 trop haute (sous-penalisee)
    # => L(var1) < L(var2) => d_t = L(var1) - L(var2) < 0
    var1 = np.empty(T)
    var2 = np.empty(T)
    from scipy.stats import norm as sp_norm
    q_exact = float(sp_norm.ppf(1.0 - alpha))
    var1[:half] = q_exact * 2.0           # var1 bien calibree (x2 pour amplifier)
    var2[:half] = q_exact * 2.0 + 2.0    # var2 trop haute (moins de violations)

    # Deuxieme moitie : inversion — var2 exacte, var1 trop haute
    var2[half:] = q_exact * 2.0
    var1[half:] = q_exact * 2.0 + 2.0

    # Verification : d_bar doit etre proche de 0 (symetrie exacte)
    d = tick_loss(r, var1, alpha) - tick_loss(r, var2, alpha)
    d_bar = float(d.mean())

    # DM : ne doit pas rejeter (d_bar ≈ 0)
    res_dm = dm_test(var1, var2, r, alpha=alpha, alpha_test=0.05)
    assert res_dm['pval'] >= 0.05, (
        f"DM doit accepter H0 sur serie symetrique (d_bar={d_bar:.4f}, "
        f"pval={res_dm['pval']:.4f}) — DM ne voit pas l'instabilite"
    )

    # GK : doit rejeter (d_{t-1} est fortement informatif sur signe de d_t)
    res_gk = gk_test(var1, var2, r, alpha=alpha, n_instruments=2, alpha_test=0.05)
    assert res_gk['pval'] < 0.05, (
        f"GK doit rejeter sur serie a regime changeant "
        f"(GK stat={res_gk['stat']:.4f}, pval={res_gk['pval']:.4f}) — "
        f"l'instrument d_{{t-1}} doit predire le changement de signe"
    )

    # Verdict doit etre egalite_avec_instabilite
    verdict = _determine_verdict(d_bar, res_dm['pval'], res_gk['pval'], alpha_test=0.05)
    assert verdict == 'egalite_avec_instabilite', (
        f"Verdict attendu 'egalite_avec_instabilite', obtenu '{verdict}' "
        f"(DM pval={res_dm['pval']:.4f}, GK pval={res_gk['pval']:.4f})"
    )
