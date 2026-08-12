# -*- coding: utf-8 -*-
"""
Validation — FHS sur 3 tickers (BZ=F, ^GSPC, EURUSD=X).

Table :
  Ticker | n_boot | H=1 VaR_99 FHS | H=22 VaR_99 FHS | H=22 VaR_99 sqrt(H) |
  Δ FHS-sqrt(H) | residus_iid_ok | Kupiec FHS 99% (backtest, p-val)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arch import arch_model
from tickerlab.core.fhs import calculer_var_fhs, fhs_var_oos
from tickerlab.core.backtest import kupiec_test, christoffersen_test


TICKERS    = ["BZ=F", "^GSPC", "EURUSD=X"]
START      = "2010-01-01"
END        = "2026-01-01"
SPLIT      = 0.70
ALPHA      = 0.99
N_BOOT     = 5000
HORIZONS   = [1, 5, 10, 22]
SEED       = 42

CONFIG_FHS = {
    'fhs': {'enabled': True, 'n_boot': N_BOOT, 'horizons': HORIZONS, 'seed': SEED},
    'var': {'niveaux': [0.90, 0.95, 0.99]},
}


def download(ticker: str) -> pd.Series:
    import yfinance as yf
    raw = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    closes = raw["Close"].squeeze().dropna()
    rets   = np.log(closes / closes.shift(1)).dropna() * 100
    return rets


def fit_garch(series: pd.Series):
    m = arch_model(series, vol='Garch', p=1, o=0, q=1, dist='t')
    return m.fit(disp='off')


def validate_ticker(ticker: str) -> dict:
    print(f"  [{ticker}] téléchargement...")
    rets = download(ticker)
    n    = len(rets)
    T_tr = int(n * SPLIT)

    train = rets.iloc[:T_tr]
    test  = rets.iloc[T_tr:]

    print(f"  [{ticker}] GARCH(1,1)-t sur {T_tr} obs train...")
    fit = fit_garch(train)

    # ── FHS multi-horizons (train-only) ──────────────────────────────────────
    fhs_r = calculer_var_fhs(None, fit, CONFIG_FHS)
    var_h1  = fhs_r['var_fhs_par_horizon'][1][ALPHA]
    var_h22 = fhs_r['var_fhs_par_horizon'][22][ALPHA]
    iid_ok  = fhs_r['residus_iid_ok']
    lb_pval = fhs_r['lb_z2_pval']

    # ── sqrt(H) baseline à partir du VaR H=1 ─────────────────────────────────
    var_sqrt22 = var_h1 * np.sqrt(22)   # négatif × √22 = encore plus négatif

    delta = var_h22 - var_sqrt22        # FHS − sqrt(H) : >0 = FHS moins prudent

    # ── Backtest OOS one-step (H=1, 99%) ─────────────────────────────────────
    print(f"  [{ticker}] backtest OOS {len(test)} obs...")
    z_raw = (fit.resid / fit.conditional_volatility).dropna().values
    z_pool_bt = z_raw[np.isfinite(z_raw)]

    # Volatilité OOS : filtrage manuel GARCH(1,1) avec paramètres train fixés
    # (récurrence σ²_{t+1} = ω + α·ε²_t + β·σ²_t appliquée sur les retours OOS)
    p = fit.params
    omega_p = float(p['omega'])
    alpha_p = float(p['alpha[1]'])
    beta_p  = float(p['beta[1]'])
    mu_bt   = float(p.get('mu', 0.0))

    last_vol2 = float(fit.conditional_volatility.iloc[-1] ** 2)
    last_eps  = float(fit.resid.iloc[-1])

    T_oos  = len(test)
    vol_oos = np.zeros(T_oos)
    for i in range(T_oos):
        sigma2 = omega_p + alpha_p * last_eps**2 + beta_p * last_vol2
        vol_oos[i] = np.sqrt(max(sigma2, 1e-12))
        last_eps  = float(test.values[i]) - mu_bt
        last_vol2 = sigma2

    fhs_oos = fhs_var_oos(z_pool_bt, vol_oos, mu_bt, [ALPHA])

    var_vec_bt = fhs_oos[ALPHA]['var']
    serie_oos  = test.values
    viol = (serie_oos < var_vec_bt).astype(int)
    N_viol = int(viol.sum())
    _, p_uc = kupiec_test(N_viol, T_oos, ALPHA)
    _, p_ind = christoffersen_test(viol)

    return {
        'ticker':       ticker,
        'n':            n,
        'T_train':      T_tr,
        'T_test':       T_oos,
        'T_oos':        T_oos,
        'n_boot':       N_BOOT,
        'var_h1':       var_h1,
        'var_h22':      var_h22,
        'var_sqrt22':   var_sqrt22,
        'delta':        delta,
        'iid_ok':       iid_ok,
        'lb_z2_pval':   lb_pval,
        'N_viol':       N_viol,
        'taux_viol':    N_viol / T_oos,
        'p_kupiec':     p_uc,
        'p_christo':    p_ind,
    }


def print_table(results: list[dict]):
    print()
    print("=" * 105)
    print("VALIDATION FHS -- Phase 1.4 (Barone-Adesi, Giannopoulos & Vosper 1999)")
    print(f"  GARCH(1,1)-t | n_boot={N_BOOT} | seed={SEED} | split={SPLIT:.0%} train | H=[1,22] | alpha=99%")
    print("=" * 105)
    hdr = (
        f"{'Ticker':<12} {'H=1 VaR99':>11} {'H=22 FHS':>10} {'H=22 sqH':>10} "
        f"{'D FHS-sqH':>10} {'iid_ok':>7} {'LB(z2) p':>9} "
        f"{'N_viol':>7} {'Taux%':>6} {'p_Kupiec':>9} {'p_Christo':>10}"
    )
    print(hdr)
    print("-" * 105)
    for r in results:
        flag = "OK " if r['iid_ok'] else "WARN"
        scaling = "sub-lin" if r['var_h22'] > r['var_sqrt22'] else "super-lin"
        print(
            f"{r['ticker']:<12} {r['var_h1']:>11.4f} {r['var_h22']:>10.4f} {r['var_sqrt22']:>10.4f} "
            f"{r['delta']:>+10.4f} {flag:>7} {r['lb_z2_pval']:>9.4f} "
            f"{r['N_viol']:>7} {r['taux_viol']*100:>6.2f} {r['p_kupiec']:>9.4f} {r['p_christo']:>10.4f}"
            f"  [{scaling}]"
        )
    print("=" * 105)
    print()
    print("Légende :")
    print("  H=1 VaR99   : VaR FHS à 1 jour, niveau 99% (valeur négative = perte)")
    print("  H=22 FHS    : VaR FHS cumulee sur 22 jours (simulation GARCH path)")
    print("  H=22 sqrtH  : regle Bale sqrt(22) x VaR(H=1) -- sous-estime si queues epaisses")
    print("  D FHS-sqrtH : >0 = FHS moins prudent que sqrtH (inattendu) | <0 = FHS plus prudent")
    print("  iid_ok      : LB(z^2) + Engle-Ng -- OK = residus standardises bien modelises")
    print("  Taux%       : taux d'exceptions OOS (cible theorique = 1.00%)")
    print("  p_Kupiec    : Kupiec 1995 UC -- rejet si < 0.05 (modele mal calibre)")
    print("  p_Christo   : Christoffersen 1998 -- rejet si < 0.05 (clusters de violations)")
    print()


if __name__ == '__main__':
    print("\nValidation FHS — Phase 1.4\n")
    results = []
    for ticker in TICKERS:
        try:
            r = validate_ticker(ticker)
            results.append(r)
            print(f"  [{ticker}] OK — VaR_99 H=1={r['var_h1']:.4f}  H=22={r['var_h22']:.4f}  "
                  f"iid_ok={r['iid_ok']}  N_viol={r['N_viol']}/{r['T_test']}\n")
        except Exception as e:
            print(f"  [{ticker}] ERREUR : {e}\n")
    if results:
        print_table(results)
