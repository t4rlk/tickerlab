# -*- coding: utf-8 -*-
"""Point d'entree CLI tickerlab.

Sous-commandes :
  tickerlab analyze  --ticker BZ=F --frequency weekly [--config ...] [--no-ai] [--force-refresh]
  tickerlab batch    --config ... [--output-root ...]
  tickerlab stress   --ticker BZ=F --frequency weekly --scenario covid_crash [--config ...]
  tickerlab --version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tickerlab import __version__


# ── Helpers ──────────────────────────────────────────────────────────────────

def _charger_config(chemin: str) -> dict:
    import yaml
    p = Path(chemin)
    if not p.exists():
        print(f"[ERREUR] Fichier config introuvable : {chemin}", file=sys.stderr)
        sys.exit(1)
    with p.open(encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _config_defaut() -> dict:
    """Charge config.yaml depuis la racine du projet ou le repertoire courant."""
    for candidat in [Path('config.yaml'), Path(__file__).parent / 'config.yaml']:
        if candidat.exists():
            import yaml
            with candidat.open(encoding='utf-8') as fh:
                return yaml.safe_load(fh)
    print('[ERREUR] Aucun config.yaml trouve. Specifier --config <chemin>.', file=sys.stderr)
    sys.exit(1)


# ── Runner interne partage (analyze / run) ────────────────────────────────────

def _executer_pipeline(config: dict, verbosity: str = 'normal',
                       force_refresh: bool = False):
    """Runner interne unique des sous-commandes `analyze` et `run`.

    Meme moteur (tickerlab.main.run_pipeline) : seule la CONSTRUCTION de la config
    differe d'une sous-commande a l'autre. `run` est donc un alias fin d'`analyze`
    (aucun doublon de logique d'execution).
    """
    from tickerlab.main import run_pipeline

    result = run_pipeline(config, verbosity=verbosity, force_refresh=force_refresh)
    dossier = config.get('output', {}).get('dossier_resultats', 'resultats')
    print(f'\n[OK] Pipeline termine — resultats dans {dossier}')
    meta = getattr(result, 'meta', {}) or {}
    if meta.get('cache'):
        hits  = sum(1 for v in meta['cache'].values() if v == 'HIT')
        total = len(meta['cache'])
        print(f'     Cache : {hits}/{total} etapes en HIT')
    return result


# ── Sous-commande : analyze ───────────────────────────────────────────────────

def cmd_analyze(args: argparse.Namespace) -> None:
    """Execute le pipeline complet pour un ticker."""
    config = _charger_config(args.config) if args.config else _config_defaut()
    config['data']['ticker']    = args.ticker
    config['data']['frequency'] = args.frequency
    if args.ai_provider:
        config.setdefault('ai_writer', {})['fournisseur'] = args.ai_provider
        config.setdefault('ai', {})['enabled'] = True
    if args.no_ai:  # --no-ai prioritaire sur --ai-provider
        config.setdefault('ai', {})['enabled'] = False

    _executer_pipeline(config, verbosity=args.verbosity,
                       force_refresh=args.force_refresh)


# ── Sous-commande : run (alignee sur le configurateur web) ────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """Analyse alignee sur la commande affichee par le configurateur web.

    Alias fin d'`analyze` : reutilise web.mapping.construire_config (source unique
    de verite front <-> CLI) puis le runner interne partage. `--module var` produit
    une erreur propre (ModuleUnavailableError), jamais un crash.
    """
    from tickerlab.web.mapping import construire_config
    from tickerlab.core.exceptions import ModuleUnavailableError
    from tickerlab.core.econ_registry import valider_outputs

    outputs_bruts = [o.strip() for o in (args.out.split(',') if args.out else []) if o.strip()]
    try:
        outputs = valider_outputs(outputs_bruts)
    except ValueError as exc:
        print(f'[ERREUR] {exc}', file=sys.stderr)
        sys.exit(2)

    payload = {
        'ticker':  args.symbole,
        'from':    args.date_from,
        'to':      args.date_to,
        'freq':    args.freq,
        'price':   'adjclose' if args.price == 'adj_close' else 'close',
        'outputs': outputs,
        'module':  args.module,
    }
    dossier = args.output or 'resultats'
    try:
        config = construire_config(payload, dossier, detail_frtb=True)
    except ModuleUnavailableError as exc:
        print(f'[ERREUR] {exc}', file=sys.stderr)
        sys.exit(2)

    _executer_pipeline(config, verbosity=args.verbosity,
                       force_refresh=args.force_refresh)


# ── Sous-commande : serve (serveur web) ───────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> None:
    """Lance le serveur web (API v1 + configurateur). Necessite l'extra [web]."""
    try:
        from tickerlab.web.app import main as _serve
    except ImportError as exc:
        print(f'[ERREUR] Couche web indisponible ({exc}). '
              f'Installer : pip install "tickerlab[web]"', file=sys.stderr)
        sys.exit(1)
    print(f'[serve] Demarrage sur http://{args.host}:{args.port}  '
          f'(Ctrl+C pour arreter)')
    _serve(host=args.host, port=args.port)


# ── Sous-commande : batch ─────────────────────────────────────────────────────

def _import_run_multi_ticker_module():
    """Charge scripts/run_multi_ticker.py sans l'importer comme package."""
    import importlib.util
    p = Path(__file__).parent / 'scripts' / 'run_multi_ticker.py'
    if not p.exists():
        print(f'[ERREUR] Script introuvable : {p}', file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location('run_multi_ticker', p)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_batch(args: argparse.Namespace) -> None:
    """Execute le pipeline sur plusieurs tickers via run_multi_ticker."""
    run_unique = _import_run_multi_ticker_module().run_unique
    config = _charger_config(args.config) if args.config else _config_defaut()
    output_root = args.output_root or config.get('output', {}).get('dossier_resultats', 'resultats')

    tickers    = config.get('batch', {}).get('tickers', [])
    frequences = config.get('batch', {}).get('frequences', ['weekly'])

    if not tickers:
        print('[ERREUR] Aucun ticker dans config[batch][tickers].', file=sys.stderr)
        sys.exit(1)

    print(f'[batch] {len(tickers)} tickers x {len(frequences)} frequences')
    for ticker in tickers:
        for freq in frequences:
            print(f'  -> {ticker} / {freq} ...', end='', flush=True)
            result = run_unique(ticker, freq, config, output_root, metrics_only=False)
            status = 'OK' if not result.get('erreur') else f'ERR: {result["erreur"]}'
            print(f' {status}')


# ── Sous-commande : stress ────────────────────────────────────────────────────

def cmd_stress(args: argparse.Namespace) -> None:
    """Execute le pipeline puis applique un scenario de stress.

    run_pipeline() est toujours appele en premier — un HIT cache le rend rapide.
    """
    from tickerlab.main import run_pipeline
    from tickerlab.core.stress_scenarios import appliquer_scenario

    config = _charger_config(args.config) if args.config else _config_defaut()
    config['data']['ticker']    = args.ticker
    config['data']['frequency'] = args.frequency

    # Etape 1 : pipeline complet (HIT cache si disponible)
    print(f'[stress] Execution pipeline {args.ticker}/{args.frequency}...')
    result = run_pipeline(config, verbosity='quiet')

    garch_final = result.garch_final
    best        = result.garch_best

    if garch_final is None or best is None:
        print('[ERREUR] Pipeline n\'a pas produit de resultat GARCH.', file=sys.stderr)
        sys.exit(1)

    # Extraction des entrees du scenario
    try:
        sigma_t = float(garch_final.conditional_volatility.iloc[-1])
    except Exception as exc:
        print(f'[ERREUR] Impossible d\'extraire sigma_t : {exc}', file=sys.stderr)
        sys.exit(1)

    dist = str(best.get('dist', best['dist']) if hasattr(best, 'get') else best['dist'])
    p = garch_final.params
    nu_raw = p.get('nu', p.get('eta'))   # Student/GED: nu ; skewt: eta
    nu = float(nu_raw) if nu_raw is not None else None
    # appliquer_scenario ne connait que 'normal' et 't' — skewt est traite comme 't'
    dist_stress = 't' if dist in ('t', 'skewt') and nu is not None else 'normal'

    # Extraction VaR99 depuis le resultat pipeline
    var99_h1 = None
    if result.var is not None:
        try:
            df_var = result.var
            idx = '99%' if '99%' in df_var.index else 0.99
            col = 'VaR GARCH' if 'VaR GARCH' in df_var.columns else df_var.columns[0]
            var99_h1 = float(df_var.loc[idx, col])
        except Exception:
            pass

    # Etape 2 : application du scenario
    scenarios = args.scenario if isinstance(args.scenario, list) else [args.scenario]
    for sc in scenarios:
        print(f'\n[stress] Scenario : {sc}')
        res = appliquer_scenario(
            scenario_name=sc,
            ticker=args.ticker,
            sigma_t=sigma_t,
            dist=dist_stress,
            nu=nu,
            var99_h1=var99_h1,
            garch_final=garch_final,
        )
        if res is None:
            print(f'  Ticker {args.ticker!r} non present dans le scenario {sc!r}.')
            continue
        print(f'  Choc direct        : {res.get("choc_direct", "N/A"):.4f}')
        print(f'  Sens               : {res.get("sens", "N/A")}')
        print(f'  Distance Mahal.    : {res.get("distance_mahalanobis", float("nan")):.2f}')
        print(f'  Proba queue        : {res.get("proba_queue", float("nan")):.6f}')
        print(f'  Methode sigma_H    : {res.get("methode_sigma_H", "N/A")}')


# ── Parser principal ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='tickerlab',
        description='Pipeline de risque de marche',
    )
    parser.add_argument('--version', action='version', version=f'tickerlab {__version__}')

    sub = parser.add_subparsers(dest='commande', metavar='<commande>')

    # analyze
    p_analyze = sub.add_parser('analyze', help='Pipeline complet pour un ticker')
    p_analyze.add_argument('--ticker', required=True, help='Code Yahoo Finance (ex: BZ=F)')
    p_analyze.add_argument('--frequency', default='weekly',
                           choices=['daily', 'weekly'], help='Frequence des donnees')
    p_analyze.add_argument('--config', default=None, help='Chemin vers config.yaml')
    p_analyze.add_argument('--no-ai', action='store_true',
                           help='Desactive la redaction IA')
    p_analyze.add_argument('--force-refresh', action='store_true',
                           help='Ignore le cache (re-execute tout)')
    p_analyze.add_argument('--verbosity', default='normal',
                           choices=['quiet', 'normal', 'verbose'])
    # TODO(dette-technique): --ai-provider ecrit dans ai.enabled ET ai_writer.fournisseur
    # car cmd_analyze() depend encore de config['ai']['enabled'] pour la porte d'entree.
    p_analyze.add_argument('--ai-provider',
                           choices=['anthropic', 'groq', 'gemini', 'ollama', 'openai'],
                           default=None,
                           help='Fournisseur LLM (ecrase ai_writer.fournisseur et active ai)')

    # batch
    p_batch = sub.add_parser('batch', help='Pipeline sur plusieurs tickers')
    p_batch.add_argument('--config', default=None, help='Chemin vers config.yaml')
    p_batch.add_argument('--output-root', default=None, help='Dossier racine des resultats')

    # stress
    p_stress = sub.add_parser('stress', help='Scenario de stress post-pipeline')
    p_stress.add_argument('--ticker', required=True, help='Code Yahoo Finance (ex: BZ=F)')
    p_stress.add_argument('--frequency', default='weekly',
                          choices=['daily', 'weekly'])
    p_stress.add_argument('--scenario', required=True, nargs='+',
                          help='Nom(s) de scenario (ex: covid_crash ukraine_war)')
    p_stress.add_argument('--config', default=None, help='Chemin vers config.yaml')

    # run (aligne sur la commande affichee par le configurateur web)
    p_run = sub.add_parser('run', help='Analyse alignee sur le configurateur web')
    p_run.add_argument('symbole', help='Code Yahoo Finance (ex: BZ=F)')
    p_run.add_argument('--module', default='univarie',
                       choices=['univarie', 'var', 'ruptures'],
                       help='Module d\'analyse (defaut: univarie)')
    p_run.add_argument('--from', dest='date_from', required=True,
                       metavar='AAAA-MM-JJ', help='Date de debut')
    p_run.add_argument('--to', dest='date_to', required=True,
                       metavar='AAAA-MM-JJ', help='Date de fin')
    p_run.add_argument('--freq', default='daily',
                       choices=['daily', 'weekly', 'monthly'])
    p_run.add_argument('--price', default='adj_close',
                       choices=['adj_close', 'close'])
    p_run.add_argument('--out', default=None,
                       help='Sorties (CSV): garch,var,backtest,breaks,report,charts')
    p_run.add_argument('--output', default=None,
                       help='Dossier resultats (defaut: resultats)')
    p_run.add_argument('--verbosity', default='normal',
                       choices=['quiet', 'normal', 'verbose'])
    p_run.add_argument('--force-refresh', action='store_true',
                       help='Ignore le cache (re-execute tout)')

    # serve (serveur web : API v1 + configurateur)
    p_serve = sub.add_parser('serve', help='Lance le serveur web (API + configurateur)')
    p_serve.add_argument('--host', default='127.0.0.1', help='Hote d\'ecoute')
    p_serve.add_argument('--port', type=int, default=8742, help='Port d\'ecoute')

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.commande == 'analyze':
        cmd_analyze(args)
    elif args.commande == 'run':
        cmd_run(args)
    elif args.commande == 'serve':
        cmd_serve(args)
    elif args.commande == 'batch':
        cmd_batch(args)
    elif args.commande == 'stress':
        cmd_stress(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
