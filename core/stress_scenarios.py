# -*- coding: utf-8 -*-
"""
Stress testing scenariste — Phase 2.3.

Applique des scenarios de stress hypothetiques et historiques sur un actif unique.
Pour le stress test multi-actifs (portefeuille), voir Phase 4 (DCC-GARCH).

References :
  FRTB d457 §181-185 — stress scenario methodology
  Berkowitz J. (2000) — A coherent framework for stress testing
  Studer G. (1997) — Maximum loss for measurement of market risk
"""
import numpy as np
import pandas as pd
from scipy import stats

# ── Scenarios de base ─────────────────────────────────────────────────────────

SCENARIOS_BASE = {
    'oil_shock_2022': {
        'description': 'Choc petrolier Ukraine 2022 — Brent +30% en 1 semaine',
        'shocks': {'BZ=F': +0.30, 'CL=F': +0.28, '^GSPC': -0.05},
        'horizon_jours': 5,
        'reference': 'Mars 2022, invasion Ukraine',
        'fenetre_historique': ('2022-02-21', '2022-03-07'),
    },
    'covid_march_2020': {
        'description': 'Krach COVID — equity -34% / oil -65% en 5 semaines',
        'shocks': {'^GSPC': -0.34, 'BZ=F': -0.65, 'GC=F': +0.10},
        'horizon_jours': 22,
        'reference': '20 fev - 23 mars 2020',
        'fenetre_historique': ('2020-02-20', '2020-03-23'),
    },
    'fed_hike_2022': {
        'description': 'Resserrement Fed agressif — taux +75bp surprise',
        'shocks': {'^GSPC': -0.08, 'EURUSD=X': -0.03, 'GC=F': -0.05},
        'horizon_jours': 1,
        'reference': 'Juin 2022, +75bp surprise',
        'fenetre_historique': ('2022-06-13', '2022-06-14'),
    },
    'geopolitique_taiwan': {
        'description': 'Choc geopolitique majeur — fuite vers valeurs refuges',
        'shocks': {'^GSPC': -0.15, 'GC=F': +0.12, 'BTC-USD': -0.25},
        'horizon_jours': 5,
        'reference': 'Hypothetique — calibre sur Crimee 2014 x 2',
        'fenetre_historique': None,
    },
}


# ── Helpers probabilistes ─────────────────────────────────────────────────────

def _proba_queue_t_standardisee(z: float, nu: float) -> float:
    """
    P(Z <= z) sous t de Student STANDARDISEE (variance=1, convention arch/rugarch).

    Le modele ARCH utilise une t standardisee avec E[Z]=0, Var[Z]=1.
    scipy.stats.t a variance nu/(nu-2). La conversion est :
      P(Z_std <= z) = P(scipy.t <= z * sqrt(nu/(nu-2)))

    Parameters
    ----------
    z   : z-score standardise (ratio choc / sigma_H)
    nu  : degres de liberte (doit etre > 2)

    Returns
    -------
    float : probabilite cumulee P(Z_std <= z)
    """
    z_scipy = z * np.sqrt(nu / (nu - 2.0))
    return float(stats.t.cdf(z_scipy, df=nu))


def _sigma_horizon_exact(garch_final, sigma_t: float, H: int) -> tuple:
    """
    sigma_H = sqrt(sum_{h=1}^H E[sigma_{t+h}^2 | F_t]) via recursion analytique GARCH.

    Pour GARCH(1,1) stationnaire (rho < 1) :
      E[sigma_{t+h}^2] = sigma_bar^2 + (sigma_T^2 - sigma_bar^2) * rho^h
      V_H = H * sigma_bar^2 + (sigma_T^2 - sigma_bar^2) * rho * (1-rho^H)/(1-rho)

    Pour near-IGARCH (rho >= 1) :
      E[sigma_{t+h}^2] = sigma_T^2 + h * omega
      V_H = H * sigma_T^2 + omega * H * (H+1) / 2

    Pour GJR-GARCH : rho = alpha + gamma/2 + beta (persistance effective).
    Pour EGARCH : fallback sqrt(H) (recursion non analytique).

    Parameters
    ----------
    garch_final : ARCHModelResult ou None
    sigma_t     : volatilite conditionnelle H=1 (en unites coherentes avec sigma_H)
    H           : horizon en periodes

    Returns
    -------
    (sigma_H, methode_sigma_H) : sigma_H en memes unites que sigma_t,
        methode in {'garch_exact', 'igarch_exact', 'sqrt_H', 'sqrt_H_EGARCH', 'sqrt_H_fallback'}
    """
    if garch_final is None or H <= 1:
        return float(sigma_t * np.sqrt(H)), 'sqrt_H'

    try:
        vol = garch_final.model.volatility
        vol_name = type(vol).__name__.upper()

        if 'EGARCH' in vol_name:
            return float(sigma_t * np.sqrt(H)), 'sqrt_H_EGARCH'

        params = garch_final.params
        omega = float(params.get('omega', 0.0))
        alpha = float(params.get('alpha[1]', 0.0))
        beta  = float(params.get('beta[1]', 0.0))
        gamma = float(params.get('gamma[1]', 0.0))  # GJR-GARCH asymmetry
        rho = alpha + 0.5 * gamma + beta

        sigma_T2 = float(sigma_t) ** 2

        if rho >= 1.0:
            # IGARCH : variance croit lineairement avec h
            V_H = H * sigma_T2 + omega * H * (H + 1) / 2.0
            methode = 'igarch_exact'
        else:
            # GARCH stationnaire : mean-reversion vers sigma_bar
            sigma_bar2 = omega / (1.0 - rho)
            rho_sum = rho * (1.0 - rho ** H) / (1.0 - rho)
            V_H = H * sigma_bar2 + (sigma_T2 - sigma_bar2) * rho_sum
            methode = 'garch_exact'

        sigma_H = float(np.sqrt(max(V_H, 1e-30)))
        return sigma_H, methode

    except Exception:
        return float(sigma_t * np.sqrt(H)), 'sqrt_H_fallback'


# ── Fonction principale ───────────────────────────────────────────────────────

def appliquer_scenario(
    scenario_name: str,
    ticker: str,
    sigma_t: float,
    dist: str = 'normal',
    nu: float = None,
    var99_h1: float = None,
    var99_fhs_hH: float = None,
    garch_final=None,
) -> dict | None:
    """
    Calcule les metriques de stress pour un couple (ticker, scenario).

    Parameters
    ----------
    scenario_name : cle dans SCENARIOS_BASE
    ticker        : code Yahoo Finance du sous-jacent
    sigma_t       : volatilite conditionnelle H=1 a la derniere date du modele
                    (en unites de rendement coherentes avec les chocs, ex: 0.02 = 2%)
    dist          : distribution conditionnelle GARCH ('normal' ou 't')
    nu            : degres de liberte (requis si dist='t')
    var99_h1      : VaR99 baseline H=1 (valeur negative, memes unites que sigma_t)
    var99_fhs_hH  : VaR99 FHS a l'horizon H du scenario (valeur negative)
    garch_final   : ARCHModelResult — si fourni, sigma_H est calcule par recursion
                    GARCH exacte plutot que par la regle sqrt(H)

    Returns
    -------
    dict ou None si le ticker n'est pas dans le scenario.

    Metriques cles :
      choc_direct        : rendement du scenario pour ce ticker
      sens               : 'perte' (position longue) ou 'gain'
      distance_mahalanobis : |choc| / sigma_H  (Studer 1997)
      proba_queue        : P(r <= choc) pour perte, P(r >= choc) pour gain
                           sous la distribution conditionnelle t standardisee ou normale
      methode_sigma_H    : methode utilisee pour sigma_H
      multiple_var99_h1  : |choc| / |VaR99 H=1|  (combien de fois la VaR)
      multiple_var99_fhs : |choc| / |VaR99 FHS H|
    """
    if scenario_name not in SCENARIOS_BASE:
        raise KeyError(f"Scenario inconnu : '{scenario_name}'")

    scenario = SCENARIOS_BASE[scenario_name]
    if ticker not in scenario['shocks']:
        return None

    choc = scenario['shocks'][ticker]
    H = scenario['horizon_jours']

    # sigma a l'horizon H
    if garch_final is not None and H > 1 and sigma_t > 0:
        sigma_H, methode_sigma_H = _sigma_horizon_exact(garch_final, sigma_t, H)
    elif sigma_t > 0:
        sigma_H = float(sigma_t * np.sqrt(H))
        methode_sigma_H = 'sqrt_H'
    else:
        sigma_H = float('nan')
        methode_sigma_H = 'nan'

    z = choc / sigma_H if (sigma_H > 0 and not np.isnan(sigma_H)) else float('nan')
    d = abs(z) if not np.isnan(z) else float('nan')

    # Probabilite de queue directionnelle sous distribution conditionnelle
    if not np.isnan(z):
        if dist == 't' and nu is not None and nu > 2:
            cdf_z = _proba_queue_t_standardisee(z, nu)
        else:
            cdf_z = float(stats.norm.cdf(z))
        # Directionnelle : queue inferieure pour perte, superieure pour gain
        proba_queue = cdf_z if choc < 0 else (1.0 - cdf_z)
    else:
        proba_queue = float('nan')

    # Multiples de VaR
    if var99_h1 is not None and var99_h1 != 0:
        multiple_h1 = round(abs(choc) / abs(var99_h1), 3)
    else:
        multiple_h1 = float('nan')

    if var99_fhs_hH is not None and var99_fhs_hH != 0:
        multiple_fhs = round(abs(choc) / abs(var99_fhs_hH), 3)
    else:
        multiple_fhs = float('nan')

    sens = 'perte' if choc < 0 else 'gain'
    return {
        'scenario':              scenario_name,
        'ticker':                ticker,
        'description':           scenario['description'],
        'horizon_jours':         H,
        'reference':             scenario['reference'],
        'choc_direct':           choc,
        'sens':                  sens,
        'sigma_t':               sigma_t,
        'sigma_H':               round(sigma_H, 6) if not np.isnan(sigma_H) else float('nan'),
        'methode_sigma_H':       methode_sigma_H,
        'distance_mahalanobis':  round(d, 3) if not np.isnan(d) else float('nan'),
        'proba_queue':           round(proba_queue, 8) if not np.isnan(proba_queue) else float('nan'),
        'multiple_var99_h1':     multiple_h1,
        'multiple_var99_fhs':    multiple_fhs,
        'interpreteur': (
            f"{abs(choc):.1%} de {sens} sur {ticker} / H={H}j = "
            f"{d:.2f}s [P_queue={proba_queue:.4%}] [{methode_sigma_H}]"
            if not np.isnan(d) else f"{abs(choc):.1%} de {sens} sur {ticker} (sigma_t=0)"
        ),
    }


def appliquer_tous_scenarios(
    ticker: str,
    sigma_t: float,
    dist: str = 'normal',
    nu: float = None,
    var99_h1: float = None,
    fhs_result: dict = None,
    garch_final=None,
) -> pd.DataFrame:
    """
    Applique tous les scenarios SCENARIOS_BASE au ticker donne.
    Ignore les scenarios ou le ticker n'est pas defini.

    Parameters
    ----------
    fhs_result : dict issu de run_pipeline() — cle 'var_fhs_par_horizon'
                 ex: {1: {0.95: -1.18, 0.99: -1.82}, 22: {0.99: ...}}
    garch_final : ARCHModelResult pour sigma_H exact (optionnel)

    Returns
    -------
    pd.DataFrame — 1 ligne par scenario applicable.
    """
    rows = []
    for nom_sc, sc in SCENARIOS_BASE.items():
        if ticker not in sc['shocks']:
            continue

        H = sc['horizon_jours']
        var99_fhs_hH = None
        if fhs_result:
            horizons = fhs_result.get('var_fhs_par_horizon', {})
            vhH = horizons.get(H, {})
            var99_fhs_hH = vhH.get(0.99)

        r = appliquer_scenario(
            nom_sc, ticker, sigma_t, dist, nu,
            var99_h1=var99_h1,
            var99_fhs_hH=var99_fhs_hH,
            garch_final=garch_final,
        )
        if r is not None:
            rows.append(r)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def scenarios_pour_ticker(ticker: str) -> list:
    """Retourne la liste des noms de scenarios applicables au ticker."""
    return [nom for nom, sc in SCENARIOS_BASE.items() if ticker in sc['shocks']]
