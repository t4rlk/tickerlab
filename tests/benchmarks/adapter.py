"""
Couche d'adaptation entre tickerlab et la suite de benchmarks.

C'EST LE SEUL FICHIER A EDITER. Chaque entree vaut None par defaut : les tests
correspondants sont alors SKIP (jamais silencieusement verts).

------------------------------------------------------------------------------
CONVENTIONS DE SIGNE (source n°1 de bugs -- a respecter strictement)
------------------------------------------------------------------------------
* `returns`  : rendements (log ou simples), PAS de multiplication par 100 sauf
               mention explicite (le jeu FCP est deja en pourcentage).
* `alpha`    : probabilite de QUEUE. alpha = 0.01  <=>  VaR 99 %.
* `var`,`es` : magnitudes de PERTE, donc POSITIVES pour une VaR usuelle.
               VaR_0.01 d'une N(0,1) = +2.3263, ES = +2.6652.
* `hits`     : vecteur 0/1, hit_t = 1 si perte_t > VaR_t (exception).
* `pit`      : Probability Integral Transform, u_t = F_t(r_t) dans [0,1].

Si tickerlab utilise une autre convention, la conversion se fait ICI (et nulle
part ailleurs) : c'est precisement le role de ce fichier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tickerlab.core.backtest import (
    acerbi_szekely_z_test,
    basel_traffic_light as _tl_zone,
    christoffersen_test,
    engle_manganelli_dq,
    fissler_ziegel_loss,
    kupiec_test,
)
from tickerlab.core.garch_selector import berkowitz_test, estimer_final
from tickerlab.core.var_engine import (
    tvar_historique, tvar_normale, tvar_student,
    var_historique, var_normale, var_student,
)

# ---------------------------------------------------------------------------
# Conversions communes (les DEUX seules divergences de convention systematiques)
# ---------------------------------------------------------------------------


def _conf(alpha: float) -> float:
    """Probabilite de QUEUE (suite) -> NIVEAU DE CONFIANCE (tickerlab).

    alpha = 0.01 (queue) <=> 0.99 (niveau de confiance tickerlab).
    """
    return 1.0 - float(alpha)


def _perte(x):
    """Valeur signee tickerlab (negative) -> magnitude de perte (positive).

    Involutive : sert aussi dans l'autre sens (entree suite -> entree tickerlab).
    """
    return -np.asarray(x, dtype=float) if np.ndim(x) else -float(x)


# =============================================================================
# 1. ESTIMATION GARCH
# =============================================================================


def fit_garch11(returns: np.ndarray, backcast: float) -> dict:
    """GARCH(1,1) gaussien a moyenne constante, estime par MLE.

    `backcast` est la valeur d'initialisation imposee pour h_0 ET e_0^2.
    Elle DOIT etre respectee : le benchmark FCP n'est reproductible qu'avec une
    initialisation controlee (cf. README, section "Initialisation").

    Retour : {"mu": float, "omega": float, "alpha": float, "beta": float,
              "loglik": float}
    """
    res = estimer_final(
        pd.Series(np.asarray(returns, dtype=float)),
        vol='Garch', o=0, p=1, q=1, dist='normal', power=None,
        backcast=float(backcast),
        # rescale=False : sans quoi arch pourrait rechelonner la serie et les
        # coefficients ne seraient plus comparables aux valeurs publiees.
        rescale=False,
    )
    par = res.params
    return {
        "mu":     float(par['mu']),
        "omega":  float(par['omega']),
        "alpha":  float(par['alpha[1]']),
        "beta":   float(par['beta[1]']),
        "loglik": float(res.loglikelihood),
    }


def simulate_garch11(mu, omega, alpha, beta, n, seed) -> np.ndarray:
    """Simulation d'un GARCH(1,1) gaussien. Optionnel : si None, la suite
    utilise le simulateur du package `arch` comme DGP de reference."""
    return None


# =============================================================================
# 2. VaR / ES PARAMETRIQUES
# =============================================================================


def var_es_normal(mu: float, sigma: float, alpha: float):
    """-> (VaR, ES), magnitudes de perte positives."""
    conf = _conf(alpha)
    return (_perte(var_normale(mu, sigma, conf)),
            _perte(tvar_normale(mu, sigma, conf)))


def var_es_student(mu: float, sigma: float, nu: float, alpha: float):
    """Student t STANDARDISEE (variance unitaire), pas la t brute.
    -> (VaR, ES), magnitudes de perte positives.

    sigma = ecart-type, contrat unique de var_student/tvar_student.
    """
    conf = _conf(alpha)
    return (_perte(var_student(mu, sigma, nu, conf)),
            _perte(tvar_student(mu, sigma, nu, conf)))


def var_es_empirical(returns: np.ndarray, alpha: float):
    """VaR/ES historiques (non parametriques). -> (VaR, ES) positifs."""
    r = np.asarray(returns, dtype=float)
    conf = _conf(alpha)
    return (_perte(var_historique(r, conf)),
            _perte(tvar_historique(r, conf)))


# =============================================================================
# 3. BACKTESTS
# =============================================================================


def kupiec_pof(hits: np.ndarray, alpha: float):
    """Kupiec (1995), test de couverture inconditionnelle. -> (stat, p_value).
    Loi asymptotique attendue : chi2(1)."""
    h = np.asarray(hits, dtype=float)
    return kupiec_test(N=int(h.sum()), T=int(h.size), alpha=_conf(alpha))


def christoffersen_ind(hits: np.ndarray):
    """Christoffersen (1998), independance. -> (stat, p_value). chi2(1)."""
    return christoffersen_test(np.asarray(hits, dtype=float))


def christoffersen_cc(hits: np.ndarray, alpha: float):
    """Christoffersen (1998), couverture conditionnelle. -> (stat, p_value).
    chi2(2)."""
    # tickerlab n'expose pas de CC autonome : LR_cc = LR_uc + LR_ind, la
    # decomposition meme de l'article (les deux LR sont independants sous H0).
    h = np.asarray(hits, dtype=float)
    lr_uc, _ = kupiec_test(N=int(h.sum()), T=int(h.size), alpha=_conf(alpha))
    lr_ind, _ = christoffersen_test(h)
    stat = float(lr_uc) + float(lr_ind)
    return stat, float(stats.chi2.sf(stat, 2))


def dq_test(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int = 4):
    """Engle & Manganelli (2004), Dynamic Quantile test.
    -> (stat, p_value). chi2(lags + 2) pour une regression
    [constante, L lags de hits, VaR_t]."""
    # engle_manganelli_dq prend des RENDEMENTS, pas des hits : on les synthetise
    # de sorte que son indicateur interne I(r < v) reproduise exactement `hits`.
    h = np.asarray(hits, dtype=float)
    v = _perte(var)                              # perte positive -> VaR negative
    r = np.where(h > 0, v - 1.0, v + 1.0)
    d = engle_manganelli_dq(r, v, lags=int(lags), alpha=_conf(alpha))
    return d['DQ_stat'], d['p_value']


def berkowitz(pit: np.ndarray):
    """Berkowitz (2001), version complete (moyenne, variance, AR(1)).
    -> (stat, p_value). chi2(3)."""
    # berkowitz_test prend des residus STANDARDISES, pas un PIT : on applique
    # la transformation normale inverse (le test la re-passe en CDF en interne).
    u = np.clip(np.asarray(pit, dtype=float), 1e-9, 1.0 - 1e-9)
    d = berkowitz_test(stats.norm.ppf(u), 'normal', {})
    return d['LR_stat'], d['pvalue']


def _acerbi_szekely(returns, var, es, alpha) -> dict:
    """Appel commun a Z1 et Z2.

    n_sim=0 est IMPERATIF : le bootstrap par defaut fait 10 000 iterations en
    boucle Python et cette fonction est appelee des centaines de fois par la
    suite. Les p-valeurs bootstrap ne sont de toute facon pas utilisees ici.
    """
    return acerbi_szekely_z_test(
        np.asarray(returns, dtype=float),
        _perte(var),
        _perte(es),
        alpha=_conf(alpha),
        n_sim=0,
    )


def acerbi_szekely_z1(returns, var, es, alpha):
    """Test 1 d'Acerbi & Szekely (2014), conditionnel. -> float (nan si aucune
    exception). Negatif = sous-estimation du risque."""
    return _acerbi_szekely(returns, var, es, alpha)['Z1']


def acerbi_szekely_z2(returns, var, es, alpha):
    """Test 2 d'Acerbi & Szekely (2014), inconditionnel. -> float.
    Vaut exactement 1.0 en l'absence d'exception (valeur maximale)."""
    return _acerbi_szekely(returns, var, es, alpha)['Z2']


def fz0_loss(returns, var, es, alpha):
    """Score de Fissler-Ziegel 0-homogene (forme FZ0 de Nolde & Ziegel).
    -> array des scores par observation, ou scalaire (moyenne)."""
    # `var` et `es` peuvent etre scalaires (balayage de grille) : on diffuse a
    # la taille de `returns`, que fissler_ziegel_loss attend en vecteurs.
    r = np.asarray(returns, dtype=float)
    v = -np.broadcast_to(np.asarray(var, dtype=float), r.shape)
    e = -np.broadcast_to(np.asarray(es,  dtype=float), r.shape)
    # 'score_series' et non 'score_total' : la serie n'est pas arrondie.
    return fissler_ziegel_loss(r, v, e, alpha=_conf(alpha))['score_series']


def basel_traffic_light(n_exceptions: int, n_obs: int = 250, alpha: float = 0.01):
    """Zones du feu tricolore balois. -> "green" | "yellow" | "red"."""
    # La fonction tickerlab prend des series, pas un comptage : on fabrique le
    # vecteur minimal produisant exactement n_exceptions violations (r < v).
    # `alpha` n'intervient pas : les zones balloises sont definies par des
    # SEUILS DE COMPTAGE sur 250 jours, deja cables a 99 % cote tickerlab.
    n = int(n_obs)
    v = np.full(n, -1.0)
    r = np.zeros(n)
    r[:int(n_exceptions)] = -2.0
    zone = _tl_zone(r, v, window=n)['zone']
    return {'vert': 'green', 'jaune': 'yellow', 'rouge': 'red'}[zone]


# =============================================================================
# Introspection (ne pas modifier)
# =============================================================================

_ENTRIES = [
    "fit_garch11", "simulate_garch11", "var_es_normal", "var_es_student",
    "var_es_empirical", "kupiec_pof", "christoffersen_ind", "christoffersen_cc",
    "dq_test", "berkowitz", "acerbi_szekely_z1", "acerbi_szekely_z2",
    "fz0_loss", "basel_traffic_light",
]


"""Le mecanisme de skip est gere par `require()` dans conftest.py : toute
entree renvoyant None declenche un SKIP explicite, jamais un PASS."""
