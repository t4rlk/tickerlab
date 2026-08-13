"""
NIVEAU 1 - Reproduction du benchmark FCP (GARCH(1,1) sur DEM/GBP).

Reference
---------
Fiorentini, G., Calzolari, G. & Panattoni, L. (1996), "Analytic Derivatives and
the Computation of GARCH Estimates", Journal of Applied Econometrics 11,
399-417. Benchmark propose par McCullough & Renfro (1999) pour noter les
logiciels econometriques, repris par Brooks, Burke & Persand (2001) puis par
Hill & McCullough (2019) sur les packages R.

Donnees : Bollerslev & Ghysels (1996), rendements quotidiens en pourcentage du
taux DEM/GBP, 02/01/1984 - 31/12/1991.

Modele : r_t = mu + e_t,  e_t = sigma_t z_t,  z_t ~ N(0,1)
         sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

Initialisation
--------------
Les estimations ne sont reproductibles qu'a initialisation controlee : la page
Econometric Benchmarks publie d'ailleurs trois jeux de valeurs, un par option
d'initialisation du presample. On impose ici h_0 = e_0^2 = variance
d'echantillon des rendements. Avec le backcast par defaut du package `arch`
(moyenne exponentiellement ponderee), omega devie de ~8 % : ce n'est pas un
bug, c'est une convention differente -- mais elle doit etre EXPLICITE.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import adapter
from .conftest import require

# Estimations de reference, 6 chiffres significatifs (FCP, modele I)
FCP = {"mu": -0.00619041, "omega": 0.0107613, "alpha": 0.153134, "beta": 0.805974}

# Tolerances : absolues sur mu (parametre de moyenne, mal identifie sur 1974
# points -> les references divergent des la 3e decimale), relatives sur les
# parametres de variance qui sont, eux, tres precisement determines.
TOL = {"mu": ("abs", 1e-4), "omega": ("rel", 2e-3),
       "alpha": ("rel", 1e-3), "beta": ("rel", 1e-3)}


@pytest.fixture(scope="module")
def fcp_fit(dem2gbp):
    backcast = float(np.var(dem2gbp, ddof=1))
    res = require(adapter.fit_garch11(dem2gbp, backcast=backcast), "fit_garch11")
    return res


def test_dataset_shape(dem2gbp):
    """Garde-fou : bonne serie, bonne echelle (pourcentage, pas fraction)."""
    assert dem2gbp.size == 1974
    assert 0.4 < dem2gbp.std(ddof=1) < 0.5   # ~0.4702 en % quotidien
    assert abs(dem2gbp.mean()) < 0.02


@pytest.mark.parametrize("name", ["mu", "omega", "alpha", "beta"])
def test_fcp_coefficients(fcp_fit, name):
    """Chaque coefficient doit reproduire la valeur publiee par FCP."""
    got, ref = float(fcp_fit[name]), FCP[name]
    kind, tol = TOL[name]
    err = abs(got - ref) if kind == "abs" else abs(got - ref) / abs(ref)
    assert err < tol, (
        f"{name}: obtenu {got:.8f}, reference FCP {ref:.8f} "
        f"(ecart {kind} {err:.2e} > {tol:.0e})"
    )


def test_fcp_persistence(fcp_fit):
    """alpha + beta ~ 0.959 : stationnarite en covariance, et ordre de grandeur
    attendu sur des donnees de change quotidiennes."""
    persistence = float(fcp_fit["alpha"] + fcp_fit["beta"])
    assert persistence < 1.0
    assert abs(persistence - (FCP["alpha"] + FCP["beta"])) < 5e-3


def test_unconditional_variance_matches_sample(fcp_fit, dem2gbp):
    """omega / (1 - alpha - beta) doit approcher la variance empirique."""
    uncond = fcp_fit["omega"] / (1.0 - fcp_fit["alpha"] - fcp_fit["beta"])
    sample = float(np.var(dem2gbp, ddof=1))
    assert abs(uncond - sample) / sample < 0.25


@pytest.mark.slow
def test_backcast_sensitivity_is_documented(dem2gbp):
    """Le choix d'initialisation doit changer les estimations de facon
    marginale mais non nulle : si ce test passe avec un ecart nul, c'est que le
    parametre `backcast` est ignore par l'implementation."""
    var = float(np.var(dem2gbp, ddof=1))
    a = require(adapter.fit_garch11(dem2gbp, backcast=var), "fit_garch11")
    b = require(adapter.fit_garch11(dem2gbp, backcast=2.0 * var), "fit_garch11")
    delta = abs(a["omega"] - b["omega"]) / a["omega"]
    assert delta > 1e-8, "le parametre backcast semble ignore"
    assert delta < 0.20, "sensibilite anormale a l'initialisation"
    # Ordre de grandeur attendu : ~9 % sur omega pour un backcast double.
