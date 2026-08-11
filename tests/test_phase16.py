# -*- coding: utf-8 -*-
"""
Tests Phase 1.6 -- Bootstrap stationnaire express IC VaR conditionnel.

5 tests :
  1. Reproductibilite : meme seed -> IC identiques
  2. Rapidite express : 200 reps < 30 s
  3. Isolation RNG : seed FHS=42 non perturbe apres bootstrap
  4. IC grandit avec horizon (direction) : largeur(IC_b=20) > largeur(IC_b=1) * 1.10 sur AR(1)
  5. Format DataFrame correct : colonnes, index, valeurs coherentes
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arch import arch_model
from tickerlab.core.var_engine import calculer_bootstrap_ci_var


# ── Fixture : GARCH(1,1)-t sur serie synthetique ─────────────────────────────

def _serie_garch(n=800, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=5, size=n)
    h = np.ones(n)
    r = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.10 * r[t-1]**2 + 0.85 * h[t-1]
        r[t] = np.sqrt(h[t]) * z[t]
    return pd.Series(r * 100)


def _serie_garch_cluster(n=1200, seed=1):
    """GARCH(1,1) a forte persistance (alpha+beta=0.95). Bien calibre, sans *100."""
    rng = np.random.default_rng(seed)
    h = np.ones(n)
    r = np.zeros(n)
    for t in range(1, n):
        h[t] = max(0.02 + 0.10 * r[t-1]**2 + 0.85 * h[t-1], 1e-8)
        r[t] = np.sqrt(h[t]) * rng.standard_t(df=5)
    return pd.Series(r)


def _fit_garch(serie):
    m = arch_model(serie, vol='Garch', p=1, o=0, q=1, dist='t')
    return m.fit(disp='off')


def _config_express(seed=42, n_rep=200, block=10):
    return {
        'bootstrap': {
            'enabled':       False,
            'express':       True,
            'n_replications': n_rep,
            'block_length':  block,
            'niveaux_ic':    [0.95],
            'inclure_tvar':  False,
            'seed':          seed,
        }
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_bootstrap_reproductibilite():
    """Meme seed -> IC identiques (bootstrap deterministique)."""
    serie = _serie_garch()
    fit   = _fit_garch(serie)
    cfg   = _config_express(seed=42)

    df1 = calculer_bootstrap_ci_var(serie, fit, cfg)
    df2 = calculer_bootstrap_ci_var(serie, fit, cfg)

    assert not df1.empty, "DataFrame vide inattendu"
    assert df1.equals(df2), "IC non reproductibles avec seed fixe"


def test_bootstrap_express_rapide():
    """200 replications express doivent terminer en moins de 30 s."""
    serie = _serie_garch(n=600)
    fit   = _fit_garch(serie)
    cfg   = _config_express(n_rep=200)

    t0 = time.time()
    df = calculer_bootstrap_ci_var(serie, fit, cfg)
    elapsed = time.time() - t0

    assert not df.empty, "DataFrame vide inattendu"
    assert elapsed < 30.0, f"Bootstrap express trop lent : {elapsed:.1f} s (> 30 s)"


def test_bootstrap_rng_isolation():
    """Le bootstrap ne perturbe pas np.random global (isolation FHS seed=42)."""
    np.random.seed(42)
    ref = np.random.random(10)   # etat reference avant bootstrap

    np.random.seed(42)           # reinitialiser pour avoir le meme etat
    serie = _serie_garch()
    fit   = _fit_garch(serie)
    cfg   = _config_express(seed=99)
    calculer_bootstrap_ci_var(serie, fit, cfg)

    post = np.random.random(10)  # etat apres bootstrap

    np.testing.assert_array_equal(ref, post,
        err_msg="Bootstrap a perturbe np.random global — RNG non isole")


def test_bootstrap_ic_grandit_avec_horizon():
    """
    Sur une serie GARCH(1,1) a forte persistance (alpha+beta=0.95), un bloc plus
    large (b=20) preserve les clusters de volatilite -> distribution bootstrap plus
    etalee -> IC plus large que b=1 (reechantillonnage iid).

    Serie : GARCH seed=1, n=1200. Bootstrap seed=42, n_rep=400.
    Assertion : largeur(IC_b=20) > largeur(IC_b=1) * 1.10
    """
    serie = _serie_garch_cluster(n=1200, seed=1)
    fit   = _fit_garch(serie)

    cfg_b1  = _config_express(block=1,  n_rep=400, seed=42)
    cfg_b20 = _config_express(block=20, n_rep=400, seed=42)

    df1  = calculer_bootstrap_ci_var(serie, fit, cfg_b1)
    df20 = calculer_bootstrap_ci_var(serie, fit, cfg_b20)

    assert '95%' in df1.index,  "Niveau 95% absent (b=1)"
    assert '95%' in df20.index, "Niveau 95% absent (b=20)"

    largeur_b1  = df1.loc['95%', 'CI upper']  - df1.loc['95%', 'CI lower']
    largeur_b20 = df20.loc['95%', 'CI upper'] - df20.loc['95%', 'CI lower']

    assert largeur_b20 > largeur_b1 * 1.10, (
        f"IC bloc=20 ({largeur_b20:.4f}) pas suffisamment plus large "
        f"que IC bloc=1 ({largeur_b1:.4f}) x 1.10 = {largeur_b1*1.10:.4f}"
    )


def test_bootstrap_format_dataframe():
    """Structure DataFrame : index correct, colonnes presentes, CI coheent."""
    serie = _serie_garch()
    fit   = _fit_garch(serie)
    cfg   = _config_express(seed=0)

    df = calculer_bootstrap_ci_var(serie, fit, cfg)

    assert not df.empty, "DataFrame vide"
    assert list(df.columns) == ['VaR GARCH', 'CI lower', 'CI upper'], \
        f"Colonnes incorrectes : {list(df.columns)}"
    assert df.index.name == 'Niveau', f"Index name incorrect : {df.index.name}"
    assert '95%' in df.index, "Niveau 95% absent"

    row = df.loc['95%']
    assert row['CI lower'] <= row['VaR GARCH'], \
        "CI lower > VaR GARCH (incoherent)"
    assert row['VaR GARCH'] <= row['CI upper'], \
        "VaR GARCH > CI upper (incoherent)"
    assert row['CI lower'] < row['CI upper'], \
        "IC degenere (lower >= upper)"


def test_bootstrap_desactive_retourne_vide():
    """Sans express ni enabled, retourne DataFrame vide."""
    serie = _serie_garch()
    fit   = _fit_garch(serie)
    cfg   = {'bootstrap': {'enabled': False, 'express': False}}

    df = calculer_bootstrap_ci_var(serie, fit, cfg)
    assert df.empty, "DataFrame non vide alors que bootstrap desactive"
