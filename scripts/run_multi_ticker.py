# -*- coding: utf-8 -*-
"""
Runner multi-tickers Phase 2.1 — validation croisee des actifs.

Modes :
  --metrics-only (defaut)  Metriques uniquement, sans generation de fichiers.
                           Usage : app investissement, tableaux de bord.
  --full-output            Pipeline complet : graphiques, Excel, LaTeX.
                           Usage : app analyse, rapport academique.

Usage :
    python scripts/run_multi_ticker.py                      # 12 runs metriques (6 tickers x 2 freq)
    python scripts/run_multi_ticker.py --quick              # 3 runs : BZ=F, ^GSPC, EURUSD=X weekly
    python scripts/run_multi_ticker.py --full-output        # 12 runs complets avec tous les fichiers
    python scripts/run_multi_ticker.py --workers 2          # parallelisme (2 processus)
    python scripts/run_multi_ticker.py --tickers BZ=F CL=F --frequencies daily
    python scripts/run_multi_ticker.py --config config.yaml --out dev/multi_ticker_summary.csv

Colonnes CSV : 35 colonnes — voir _COLONNES_CSV.
"""
import argparse
import copy
import datetime
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout, as_completed
from pathlib import Path

import pandas as pd
import numpy as np

# ── Import pipeline ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

import yaml
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from main import run_pipeline, slugify_ticker as _slug_ticker
from tickerlab.core.exceptions import PipelineError
from tickerlab.core.config_validation import config_hash as _config_hash

# ── Constantes ────────────────────────────────────────────────────────────────

TICKERS_DEFAUT = ['BZ=F', 'CL=F', '^GSPC', 'GC=F', 'EURUSD=X', 'BTC-USD']
FREQUENCES_DEFAUT = ['daily', 'weekly']
TICKERS_QUICK = ['BZ=F', '^GSPC', 'EURUSD=X']
FREQUENCES_QUICK = ['weekly']

# Grid GARCH reduit Phase 2.1 (3 modeles x 2 distributions x p,q=1 = 9 fits/run)
_GARCH_OVERRIDE = {
    'modeles':       ['GARCH', 'GJR-GARCH', 'EGARCH'],
    'distributions': ['normal', 't'],
    'p_max': 1,
    'q_max': 1,
}

_COLONNES_CSV = [
    'ticker', 'frequence', 'date_run', 'n_obs', 'duree_s',
    'T_train', 'T_eff_dyn',
    'arima_p', 'arima_d', 'arima_q', 'arima_interpretation',
    'motif_selection',
    'garch_modele', 'garch_aic', 'persistance', 'near_igarch',
    'all_candidates_failed_spec', 'spec_gravite',
    'var95_garch', 'var99_garch', 'var99_garch_floored', 'tvar99_garch',
    'var95_fhs_h1', 'var99_fhs_h1',
    'kupiec_95_pval', 'kupiec_99_pval',
    'christoffersen_95_pval', 'christoffersen_99_pval',
    'dm_garch_vs_fhs_stat', 'dm_garch_vs_fhs_pval',
    'component_garch_estime', 'component_garch_rho',
    'bootstrap_ic_largeur_pct',
    'config_hash',
    'erreur',
]


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _val(df: 'pd.DataFrame', filtre: dict, col: str):
    """Extrait df[col] apres filtrage multi-cle, renvoie nan si absent."""
    mask = pd.Series([True] * len(df), index=df.index)
    for k, v in filtre.items():
        if k in df.columns:
            mask &= (df[k] == v)
    rows = df[mask]
    if rows.empty or col not in rows.columns:
        return float('nan')
    val = rows.iloc[0][col]
    if val == 'n/a':
        return float('nan')
    return val


# ── Config ────────────────────────────────────────────────────────────────────

def _build_config(ticker: str, freq: str, base_cfg: dict, output_root: Path,
                  metrics_only: bool = True) -> dict:
    """Deep copy du template + overrides specifiques au run."""
    cfg = copy.deepcopy(base_cfg)
    slug = _slug_ticker(ticker)
    run_dir = str(output_root / f'{slug}_{freq}')

    cfg['data']['ticker'] = ticker
    cfg['data']['frequency'] = freq
    cfg['data']['start_date'] = '2015-01-01'

    cfg.setdefault('output', {})
    cfg['output']['dossier_resultats'] = run_dir
    cfg['output']['sous_dossier_par_ticker'] = False  # run_dir inclut deja le slug
    cfg['output']['pdf_unique'] = False
    cfg['output']['export_latex'] = False

    cfg.setdefault('rolling_backtest', {})
    cfg['rolling_backtest']['enabled'] = False

    cfg.setdefault('ai', {})
    cfg['ai']['enabled'] = False

    cfg['_reuse_cache'] = False
    cfg['_metrics_only'] = metrics_only

    # Grid GARCH reduit
    cfg.setdefault('garch', {})
    cfg['garch'].update(_GARCH_OVERRIDE)

    # Budget MC reduit
    cfg.setdefault('var', {})
    cfg['var']['n_simulations_mc'] = 10_000
    cfg.setdefault('fhs', {})
    cfg['fhs']['n_boot'] = 2_000

    # Bootstrap desactive en mode multi-tickers
    cfg.setdefault('bootstrap', {})
    cfg['bootstrap']['express'] = False
    cfg['bootstrap']['enabled'] = False

    return cfg


# ── Extraction metriques ──────────────────────────────────────────────────────

def _metriques_vides(ticker: str, freq: str) -> dict:
    row = {c: float('nan') for c in _COLONNES_CSV}
    row['ticker'] = ticker
    row['frequence'] = freq
    row['date_run'] = datetime.datetime.now().isoformat()
    row['erreur'] = ''
    row['all_candidates_failed_spec'] = False
    row['near_igarch'] = False
    row['component_garch_estime'] = False
    row['var99_garch_floored'] = False
    row['config_hash'] = ''
    return row


def _extraire_metriques(ticker: str, freq: str, result, t0: float, cfg: dict = None) -> dict:
    """Extrait les 35 colonnes CSV depuis le PipelineResult retourne par run_pipeline()."""
    import time
    from tickerlab.core.dm_gk import find_pair

    m = _metriques_vides(ticker, freq)
    m['duree_s'] = round(time.time() - t0, 1)

    # ── Observations ──────────────────────────────────────────────────────────
    rend = result.get('rendements')
    if rend is not None:
        m['n_obs'] = int(len(rend.dropna()))
    m['T_train']   = result.get('T_train', float('nan'))
    m['T_eff_dyn'] = result.get('T_eff_dyn', float('nan'))

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    arima = result.get('arima') or {}
    m['arima_p'] = arima.get('p_opt', float('nan'))
    m['arima_d'] = arima.get('d_opt', float('nan'))
    m['arima_q'] = arima.get('q_opt', float('nan'))
    m['arima_interpretation'] = arima.get('interpretation', '')

    # ── GARCH ─────────────────────────────────────────────────────────────────
    best = result.get('garch_best')
    if best is not None:
        bd = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
        m['garch_modele'] = (f"{bd.get('modele','?')}({int(bd.get('p',0))},"
                             f"{int(bd.get('o',0))},{int(bd.get('q',0))})"
                             f"[{bd.get('dist','?')}]")
        m['garch_aic'] = bd.get('AIC', float('nan'))

    m['motif_selection'] = result.get('motif_selection', '')

    igarch = result.get('igarch_diag') or {}
    m['persistance'] = igarch.get('persistance', float('nan'))
    m['near_igarch'] = bool(igarch.get('near_igarch', False))

    spec = result.get('spec_verdict') or {}
    m['all_candidates_failed_spec'] = bool(spec.get('all_candidates_failed_spec', False))
    m['spec_gravite'] = spec.get('gravite', '')

    # ── VaR ───────────────────────────────────────────────────────────────────
    df_var = result.get('var')
    if df_var is not None and not df_var.empty:
        for niv_str, key95, key99 in [
            ('VaR GARCH',  'var95_garch',  'var99_garch'),
            ('TVaR GARCH', None,           'tvar99_garch'),
        ]:
            try:
                if key95 and '95%' in df_var.index:
                    m[key95] = float(df_var.loc['95%', niv_str])
                if '99%' in df_var.index:
                    m[key99] = float(df_var.loc['99%', niv_str])
            except (KeyError, TypeError):
                pass
        m['var99_garch_floored'] = bool(df_var.attrs.get('var99_garch_floored', False))

    # ── FHS ───────────────────────────────────────────────────────────────────
    fhs = result.get('fhs') or {}
    horizons = (fhs.get('var_fhs_par_horizon') or {})
    h1 = horizons.get(1) or {}
    m['var95_fhs_h1'] = h1.get(0.95, float('nan'))
    m['var99_fhs_h1'] = h1.get(0.99, float('nan'))

    # ── Backtest ──────────────────────────────────────────────────────────────
    df_bt = result.get('backtest')
    if df_bt is not None and not df_bt.empty:
        for niv, k_kup, k_chr in [
            ('95%', 'kupiec_95_pval', 'christoffersen_95_pval'),
            ('99%', 'kupiec_99_pval', 'christoffersen_99_pval'),
        ]:
            m[k_kup] = _val(df_bt, {'Methode': 'GARCH dyn.', 'Niveau': niv}, 'p_UC')
            m[k_chr] = _val(df_bt, {'Methode': 'GARCH dyn.', 'Niveau': niv}, 'p_CC')

    # ── DM-GK ─────────────────────────────────────────────────────────────────
    dm_gk = result.get('dm_gk') or {}
    try:
        pair99 = find_pair(dm_gk.get(0.99, {}), 'GARCH dyn.', 'FHS') or {}
        m['dm_garch_vs_fhs_stat'] = pair99.get('dm_stat', float('nan'))
        m['dm_garch_vs_fhs_pval'] = pair99.get('dm_pval', float('nan'))
    except Exception:
        pass

    # ── Component GARCH ───────────────────────────────────────────────────────
    cgarch = result.get('component_garch')
    m['component_garch_estime'] = cgarch is not None
    if cgarch is not None:
        m['component_garch_rho'] = cgarch.get('rho', float('nan'))

    # ── Bootstrap IC largeur ──────────────────────────────────────────────────
    # Non calcule en mode multi-tickers (bootstrap desactive) -> nan par defaut

    # ── Config hash ───────────────────────────────────────────────────────────
    meta = result.get('meta') or {}
    m['config_hash'] = meta.get('config_hash', _config_hash(cfg) if cfg else '')

    return m


# ── run_unique ────────────────────────────────────────────────────────────────

def run_unique(ticker: str, freq: str, base_cfg: dict, output_root_str: str,
               metrics_only: bool = True) -> dict:
    """
    Execute run_pipeline pour un ticker x frequence.
    Retourne toujours un dict de metriques (erreur isole au niveau run).
    Doit etre picklable (definie au niveau module) pour ProcessPoolExecutor.
    metrics_only=True (defaut) : pas de generation de fichiers, ~2x plus rapide.
    metrics_only=False         : pipeline complet avec graphiques/Excel/LaTeX.
    """
    import time
    output_root = Path(output_root_str)
    t0 = time.time()
    cfg = _build_config(ticker, freq, base_cfg, output_root, metrics_only=metrics_only)
    run_dir = Path(cfg['output']['dossier_resultats'])

    try:
        result = run_pipeline(cfg, verbosity='quiet')
        metrics = _extraire_metriques(ticker, freq, result, t0, cfg=cfg)
        metrics['erreur'] = ''
    except PipelineError as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (run_dir / 'erreur.txt').write_text(tb, encoding='utf-8')
        metrics = _metriques_vides(ticker, freq)
        metrics['erreur'] = f'PipelineError({exc.etape}): {str(exc)[:280]}'
        metrics['duree_s'] = round(time.time() - t0, 1)
        print(f'  [ERREUR PipelineError] {ticker} {freq} : {str(exc)[:120]}')
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (run_dir / 'erreur.txt').write_text(tb, encoding='utf-8')
        metrics = _metriques_vides(ticker, freq)
        metrics['erreur'] = repr(exc)[:300]
        metrics['duree_s'] = round(time.time() - t0, 1)
        print(f'  [ERREUR] {ticker} {freq} : {repr(exc)[:120]}')

    return metrics


# ── CSV ───────────────────────────────────────────────────────────────────────

def _append_csv(metrics: dict, csv_path: Path) -> None:
    """Append incremental : ecrit l'en-tete si le fichier n'existe pas encore."""
    row = {c: metrics.get(c, float('nan')) for c in _COLONNES_CSV}
    # 'erreur' vide ecrit comme cellule vide -> pandas relit comme NaN.
    # On utilise 'none' comme sentinel succes pour que fillna('') fonctionne
    # de facon coherente, et que la colonne reste lisible.
    if row.get('erreur') == '':
        row['erreur'] = 'none'
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        row_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row_df.to_csv(csv_path, index=False)


def _is_success(erreur_val) -> bool:
    """True si le run a reussi (erreur vide, 'none', ou NaN)."""
    if pd.isna(erreur_val):
        return True
    return str(erreur_val).strip().lower() in ('', 'none')


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Runner multi-tickers Phase 2.1')
    p.add_argument('--quick',  action='store_true',
                   help='3 runs : BZ=F, ^GSPC, EURUSD=X weekly')
    p.add_argument('--full-output', action='store_true', dest='full_output',
                   help='Pipeline complet (graphiques/Excel/LaTeX). Defaut : metriques seules.')
    p.add_argument('--tickers', nargs='+', default=None,
                   help='Liste de tickers (defaut : 6 tickers Phase 2)')
    p.add_argument('--frequencies', nargs='+', default=None,
                   help='Frequences : daily weekly monthly (defaut : daily weekly)')
    p.add_argument('--workers', type=int, default=1,
                   help='Nombre de processus paralleles (defaut : 1 = sequentiel)')
    p.add_argument('--config', default='config.yaml',
                   help='Chemin config YAML template (defaut : config.yaml)')
    p.add_argument('--out', default='dev/multi_ticker_summary.csv',
                   help='Chemin CSV de sortie')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # ── Config template ───────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    with open(cfg_path, encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    # ── Liste des runs ────────────────────────────────────────────────────────
    metrics_only = not args.full_output

    if args.quick:
        tickers = TICKERS_QUICK
        frequencies = FREQUENCES_QUICK
        mode_label = 'metriques' if metrics_only else 'complet'
        print(f'\n[MODE --quick] 3 runs Phase 1 weekly (BZ=F, ^GSPC, EURUSD=X) [{mode_label}]\n')
    else:
        tickers    = args.tickers    or TICKERS_DEFAUT
        frequencies = args.frequencies or FREQUENCES_DEFAUT

    runs = [(t, f) for t in tickers for f in frequencies]
    csv_path = Path(args.out)
    if not csv_path.is_absolute():
        csv_path = _ROOT / csv_path

    output_root = _ROOT / 'dev' / 'resultats' / 'multi'
    output_root_str = str(output_root)

    mode_str = 'metriques-only' if metrics_only else 'full-output'
    print(f'[Phase 2.1] {len(runs)} run(s) | workers={args.workers} | mode={mode_str} | CSV -> {csv_path}')
    print('-' * 65)

    # ── Execution ─────────────────────────────────────────────────────────────
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_unique, t, f, base_cfg, output_root_str, metrics_only): (t, f)
                for t, f in runs
            }
            completed = 0
            for fut in as_completed(futures):
                t, f = futures[fut]
                try:
                    metrics = fut.result(timeout=240)
                except FutureTimeout:
                    metrics = _metriques_vides(t, f)
                    metrics['erreur'] = 'TIMEOUT: depasse 240s'
                    print(f'  [TIMEOUT] {t} {f} depasse 240s — skip')
                except Exception as exc:
                    metrics = _metriques_vides(t, f)
                    metrics['erreur'] = repr(exc)[:300]
                _append_csv(metrics, csv_path)
                completed += 1
                ok = 'OK' if not metrics['erreur'] else 'FAIL'
                print(f'  [{completed}/{len(runs)}] {ok} {t} {f} '
                      f'({metrics.get("duree_s","??")}s)')
    else:
        for i, (t, f) in enumerate(runs, 1):
            print(f'  [{i}/{len(runs)}] {t} {f} ...', end=' ', flush=True)
            metrics = run_unique(t, f, base_cfg, output_root_str, metrics_only)
            _append_csv(metrics, csv_path)
            ok = 'OK' if not metrics['erreur'] else f'ERREUR ({metrics["erreur"][:60]})'
            print(f'{ok} ({metrics.get("duree_s","??")}s)')

    print('-' * 65)
    print(f'[Phase 2.1] Termine. CSV -> {csv_path}')

    # Resume terminal
    try:
        df_sum = pd.read_csv(csv_path)
        masque_ok  = df_sum['erreur'].apply(_is_success)
        n_ok  = int(masque_ok.sum())
        n_err = len(df_sum) - n_ok
        print(f'            Succes : {n_ok}/{len(df_sum)} | Erreurs : {n_err}')
        if n_ok > 0:
            print('\nAperu (spec_gravite + kupiec_99_pval) :')
            cols_resume = ['ticker', 'frequence', 'garch_modele',
                           'spec_gravite', 'kupiec_99_pval', 'near_igarch']
            cols_ok = [c for c in cols_resume if c in df_sum.columns]
            print(df_sum[masque_ok][cols_ok].to_string(index=False))
    except Exception:
        pass
