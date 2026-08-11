# -*- coding: utf-8 -*-
"""
Validation Phase 1.5 -- DM + GK sur 3 tickers (BZ=F, ^GSPC, EURUSD=X).

Tableau attendu :
  Ticker | Paire | alpha | DM stat | DM p-val | GK stat | GK p-val | Verdict
Pour paires : GARCH dyn. vs FHS (H=1) et Historique vs FHS (H=1)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arch import arch_model
from tickerlab.core.fhs import calculer_var_fhs
from tickerlab.core.dm_gk import build_var_series, comparer_methodes_var, find_pair

TICKERS  = ["BZ=F", "^GSPC", "EURUSD=X"]
START    = "2010-01-01"
END      = "2026-01-01"
SPLIT    = 0.70
NIVEAUX  = [0.95, 0.99]
N_BOOT   = 2000
SEED     = 42

CONFIG = {
    'backtest': {'split_ratio': SPLIT},
    'var':      {'niveaux': NIVEAUX},
    'fhs':      {'enabled': True, 'n_boot': N_BOOT, 'horizons': [1, 5, 10, 22], 'seed': SEED},
    'dm_gk': {
        'enabled':                   True,
        'alpha_test':                0.05,
        'alpha_test_petit_echantillon': 0.10,
        'hac_lags':                  'auto',
        'n_instruments_gk':          2,
        'paires_section_principale': ['GARCH dyn._vs_FHS', 'Historique_vs_FHS'],
    },
}

PAIRES_FOCUS = ['GARCH dyn._vs_FHS', 'Historique_vs_FHS', 'GARCH dyn._vs_Historique']


def download(ticker):
    import yfinance as yf
    raw    = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    closes = raw["Close"].squeeze().dropna()
    return np.log(closes / closes.shift(1)).dropna() * 100


def fit_garch(series):
    m = arch_model(series, vol='Garch', p=1, o=0, q=1, dist='t')
    return m.fit(disp='off')


def run_ticker(ticker):
    print(f"  [{ticker}] download + GARCH(1,1)-t...")
    rets   = download(ticker)
    n      = len(rets)
    T_tr   = int(n * SPLIT)

    fit    = fit_garch(rets.iloc[:T_tr])
    fhs_r  = calculer_var_fhs(None, fit, CONFIG)

    r_oos  = rets.values[T_tr:]
    T_oos  = len(r_oos)

    var_s  = build_var_series(rets, fit, fhs_r, CONFIG, T_train=T_tr)
    dm_gk  = comparer_methodes_var(var_s, r_oos, NIVEAUX, CONFIG)

    rows = []
    for alp in NIVEAUX:
        paires_alp = dm_gk.get(alp, {})
        for paire_key in PAIRES_FOCUS:
            parts = paire_key.split('_vs_')
            info = find_pair(paires_alp, parts[0], parts[1]) if len(parts) == 2 else None
            if info is None:
                continue
            rows.append({
                'Ticker':  ticker,
                'Paire':   paire_key.replace('_vs_', ' vs '),
                'alpha':   f"{int(alp*100)}%",
                'DM stat': info['dm_stat'],
                'DM p':    info['dm_pval'],
                'GK stat': info['gk_stat'],
                'GK p':    info['gk_pval'],
                'T_eff':   info['T_eff'],
                'Verdict': info['verdict'],
            })
    return rows


if __name__ == '__main__':
    print("\nValidation DM-GK -- Phase 1.5\n")
    all_rows = []
    for ticker in TICKERS:
        try:
            r = run_ticker(ticker)
            all_rows.extend(r)
            print(f"  [{ticker}] OK -- {len(r)} paires x niveaux\n")
        except Exception as e:
            print(f"  [{ticker}] ERREUR : {e}\n")

    if not all_rows:
        print("Aucun resultat.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print("=" * 110)
    print("VALIDATION DM-GK -- Phase 1.5 (GARCH(1,1)-t, H=1, split=70%)")
    print("=" * 110)
    for _, row in df.iterrows():
        stars_dm = "**" if row["DM p"] < 0.05 else ("*" if row["DM p"] < 0.10 else "  ")
        stars_gk = "**" if row["GK p"] < 0.05 else ("*" if row["GK p"] < 0.10 else "  ")
        print(
            f"{row['Ticker']:<12} {row['Paire']:<30} alpha={row['alpha']} | "
            f"DM={row['DM stat']:+6.3f} p={row['DM p']:.3f}{stars_dm} | "
            f"GK={row['GK stat']:6.3f} p={row['GK p']:.3f}{stars_gk} | "
            f"T={row['T_eff']:4d} | {row['Verdict']}"
        )
    print("=" * 110)
    print("** p<0.05  * p<0.10")
    print()

    # Criteres de succes
    instab = df[df['Verdict'] == 'egalite_avec_instabilite']
    wins   = df[df['Verdict'].isin(['model1_wins', 'model2_wins'])]
    hist_fhs = df[df['Paire'] == 'Historique vs FHS']

    print("Criteres de succes :")
    c1 = not (instab.empty and wins.empty)
    print(f"  [{'OK' if c1 else 'FAIL'}] Au moins 1 verdict non-egalite_stable : "
          f"{len(instab)} egalite_avec_instabilite + {len(wins)} model_wins")

    c2 = not instab.empty
    print(f"  [{'OK' if c2 else 'NOTE'}] Verdict egalite_avec_instabilite present "
          f"(valide GK empiriquement) : {len(instab)} cas")
    if not c2:
        print("    -> GK non valide empiriquement sur ces donnees -- investiguer puissance")

    c3 = not (hist_fhs[hist_fhs['Verdict'] != 'egalite_stable'].empty)
    print(f"  [{'OK' if c3 else 'NOTE'}] Historique vs FHS : verdict non-egalite_stable "
          f"sur au moins 1 ticker/niveau")
    print()
