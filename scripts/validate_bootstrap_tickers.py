# -*- coding: utf-8 -*-
"""
Validation — Bootstrap stationnaire express IC VaR (3 tickers).

Tableau attendu :
  Ticker | Niveau | VaR GARCH | IC lower | IC upper | Largeur | T_eff | Temps (s)

Criteres de succes :
  [OK] 3/3 tickers : IC non degenere (lower < upper)
  [OK] 3/3 tickers : VaR GARCH in [IC lower, IC upper]
  [OK] Temps total < 60 s (express = rapide)
  [NOTE] Largeur IC >= 0.01% (signal minimal de variabilite)
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arch import arch_model
from tickerlab.core.var_engine import calculer_bootstrap_ci_var

TICKERS = ["BZ=F", "^GSPC", "EURUSD=X"]
START   = "2010-01-01"
END     = "2026-01-01"
SPLIT   = 0.70

CONFIG = {
    'bootstrap': {
        'enabled':        False,
        'express':        True,
        'n_replications': 200,
        'block_length':   10,
        'niveaux_ic':     [0.95],
        'inclure_tvar':   False,
        'seed':           42,
    }
}


def download(ticker):
    import yfinance as yf
    raw    = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    closes = raw["Close"].squeeze().dropna()
    return np.log(closes / closes.shift(1)).dropna() * 100


def fit_garch(series):
    m = arch_model(series, vol='Garch', p=1, o=0, q=1, dist='t')
    return m.fit(disp='off')


def run_ticker(ticker):
    print(f"  [{ticker}] download + GARCH(1,1)-t + bootstrap express...")
    rets = download(ticker)
    n    = len(rets)
    T_tr = int(n * SPLIT)
    fit  = fit_garch(rets.iloc[:T_tr])

    t0  = time.time()
    df  = calculer_bootstrap_ci_var(rets.iloc[:T_tr], fit, CONFIG)
    dur = time.time() - t0

    rows = []
    for niv, row in df.iterrows():
        rows.append({
            'Ticker':    ticker,
            'Niveau':    niv,
            'VaR GARCH': row['VaR GARCH'],
            'IC lower':  row['CI lower'],
            'IC upper':  row['CI upper'],
            'Largeur':   round(row['CI upper'] - row['CI lower'], 4),
            'T_eff':     T_tr,
            'Temps (s)': round(dur, 2),
        })
    return rows


if __name__ == '__main__':
    print("\nValidation Bootstrap Express -- Phase 1.6\n")
    all_rows  = []
    t_total   = time.time()

    for ticker in TICKERS:
        try:
            r = run_ticker(ticker)
            all_rows.extend(r)
            print(f"  [{ticker}] OK -- {len(r)} niveau(x)  "
                  f"temps={r[0]['Temps (s)']:.2f} s\n")
        except Exception as e:
            print(f"  [{ticker}] ERREUR : {e}\n")

    elapsed_total = time.time() - t_total

    if not all_rows:
        print("Aucun resultat.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print("=" * 100)
    print("VALIDATION BOOTSTRAP EXPRESS -- Phase 1.6 (GARCH(1,1)-t, express=200 reps, bloc=10)")
    print("=" * 100)
    for _, row in df.iterrows():
        ok_ic   = 'OK' if row['IC lower'] < row['IC upper'] else 'FAIL'
        ok_cont = 'OK' if row['IC lower'] <= row['VaR GARCH'] <= row['IC upper'] else 'WARN'
        print(
            f"{row['Ticker']:<12} {row['Niveau']:<5} | "
            f"VaR={row['VaR GARCH']:+7.4f}%  "
            f"IC=[{row['IC lower']:+7.4f}%, {row['IC upper']:+7.4f}%]  "
            f"Larg={row['Largeur']:.4f}%  "
            f"T={row['T_eff']:4d}  t={row['Temps (s)']:.2f}s  "
            f"IC:{ok_ic}  Cont:{ok_cont}"
        )
    print("=" * 100)
    print(f"Temps total : {elapsed_total:.1f} s")
    print()

    # Criteres de succes
    ok_ic    = all(r['IC lower'] < r['IC upper'] for r in all_rows)
    ok_cont  = all(r['IC lower'] <= r['VaR GARCH'] <= r['IC upper'] for r in all_rows)
    ok_temps = elapsed_total < 60.0
    ok_larg  = all(r['Largeur'] >= 0.01 for r in all_rows)

    print("Criteres de succes :")
    print(f"  [{'OK' if ok_ic   else 'FAIL'}] IC non degenere (lower < upper) : "
          f"{sum(r['IC lower'] < r['IC upper'] for r in all_rows)}/{len(all_rows)}")
    print(f"  [{'OK' if ok_cont  else 'WARN'}] VaR in IC : "
          f"{sum(r['IC lower'] <= r['VaR GARCH'] <= r['IC upper'] for r in all_rows)}/{len(all_rows)}")
    print(f"  [{'OK' if ok_temps else 'FAIL'}] Temps total < 60 s : {elapsed_total:.1f} s")
    print(f"  [{'OK' if ok_larg  else 'NOTE'}] Largeur IC >= 0.01% : "
          f"{sum(r['Largeur'] >= 0.01 for r in all_rows)}/{len(all_rows)}")
    print()
