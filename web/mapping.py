# -*- coding: utf-8 -*-
"""Traduction du payload du configurateur web → dict de config pipeline.

Le configurateur envoie :
    { ticker, from, to, freq, price, outputs: [...] }

On part de config.yaml (tous les défauts scientifiques préservés) et on n'écrase
QUE les champs pilotés par le front. La logique économétrique n'est jamais touchée.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from tickerlab.core.exceptions import ModuleUnavailableError

# Racine du dépôt = parent du dossier web/
_RACINE = Path(__file__).resolve().parent.parent
_CONFIG_YAML = _RACINE / 'config.yaml'


def activer_ia(config: dict, actif: bool) -> None:
    """Unifie l'activation de la rédaction IA en un seul point.

    Historiquement, l'activation était éclatée entre config['ai']['enabled']
    (la porte d'entrée) et config['ai_writer'] (la configuration du writer).
    Cette fonction est l'unique endroit qui bascule la porte ; la configuration
    du provider (fournisseur, modèle, env_key) reste celle de config.yaml.

    SÉCURITÉ : aucune clé API n'est manipulée ici. config['ai_writer']['env_key']
    ne contient que le NOM de la variable d'environnement, jamais sa valeur.
    """
    config.setdefault('ai', {})['enabled'] = bool(actif)
    config.setdefault('ai_writer', {})  # garantit la présence du bloc writer


def charger_config_base() -> dict:
    """Charge config.yaml et en retourne une copie profonde (jamais mutée en place)."""
    with _CONFIG_YAML.open(encoding='utf-8') as fh:
        return copy.deepcopy(yaml.safe_load(fh))


_MODULES_CONNUS = {'univarie', 'var', 'ruptures'}


def construire_config(payload: dict, dossier_sortie: str,
                      detail_frtb: bool = False) -> dict:
    """Construit le dict de config pipeline depuis le payload du configurateur.

    Parameters
    ----------
    payload : dict
        { ticker, from, to, freq, price, outputs: [...], module? }
    dossier_sortie : str
        Dossier racine isolé pour ce job (ex. web_runs/{job_id}).
    detail_frtb : bool
        Active le calcul du détail enrichi de backtest (6 tests FRTB/Berkowitz)
        exposé par l'API v1. Faux pour la surface legacy /api/run (inchangée).

    Returns
    -------
    dict
        Config prête pour run_pipeline (et valider_config_structure).

    Raises
    ------
    ModuleUnavailableError
        Si `module='var'` (module multivarié non disponible sur cette branche).
    """
    # ── Module d'analyse ─────────────────────────────────────────────────────
    # univarie (défaut) : pipeline ARIMA-GARCH-VaR-backtesting.
    # ruptures          : idem + détection de ruptures (ICSS + Zivot-Andrews) activée.
    # var               : module multivarié — indisponible ici (autre branche).
    module = str(payload.get('module', 'univarie') or 'univarie')
    if module == 'var':
        raise ModuleUnavailableError(
            "Le module multivarié « var » (Granger, IRF, FEVD) n'est pas "
            "disponible sur ce déploiement. Modules disponibles : univarie, ruptures.",
            etape='mapping',
        )
    if module not in _MODULES_CONNUS:
        raise ModuleUnavailableError(
            f"Module inconnu : {module!r}. Attendu : univarie, var, ruptures.",
            etape='mapping',
        )

    config = charger_config_base()

    # ── Bloc data : tout ce que le front pilote ──────────────────────────────
    data: dict[str, Any] = config.setdefault('data', {})
    data['ticker']     = str(payload['ticker']).strip()
    data['start_date'] = str(payload['from'])
    data['end_date']   = str(payload['to'])
    data['frequency']  = str(payload['freq'])
    price = str(payload.get('price', 'adjclose'))
    data['price']       = price
    # close → clôture brute (auto_adjust=False) ; adjclose → ajustée (True)
    data['auto_adjust'] = (price == 'adjclose')

    outputs_bruts = payload.get('outputs', []) or []
    outs = set(outputs_bruts)  # tests d'appartenance ci-dessous (ordre non pertinent ici)

    # ── Ordre d'affichage (Phase ordering) ────────────────────────────────────
    # Le front envoie 'outputs' déjà trié par ordre de clic utilisateur (voir
    # analyse.html / selectedOutputs). On le préserve tel quel dans la config,
    # dédupliqué, sans le retrier ni le filtrer ici : c'est à l'orchestrateur de
    # rapport (core/rapport/_orchestrateur.py) de décider, module par module, ce
    # qu'il fait de cet ordre quand il le consommera (chantier à venir — cf.
    # generalisation du registre core/econ_registry.py aux sorties univariees).
    # Tant que ce cablage n'existe pas, ce champ est un pass-through inoffensif :
    # aucune section actuelle du rapport ne le lit encore.
    vu: set[str] = set()
    ordre_clic: list[str] = []
    for o in outputs_bruts:
        if o not in vu:
            vu.add(o)
            ordre_clic.append(o)
    config['sorties_ordre'] = ordre_clic

    # ── Ruptures structurelles : opt-in via la sortie 'breaks' ───────────────
    # mode='diagnostic' = défaut sûr (ne modifie AUCUN résultat numérique).
    sb = config.setdefault('structural_breaks', {})
    sb['enabled'] = ('breaks' in outs) or (module == 'ruptures')

    # ── Rolling backtest : désactivé pour les runs web ───────────────────────
    # Non exposé par le configurateur (la sortie 'backtest' = backtest OOS
    # standard). Le défaut de config.yaml (window_size=800) est calibré pour la
    # série de prod et échouerait sur les périodes plus courtes choisies au front
    # (valider_config rejette si window_size >= n_obs).
    config.setdefault('rolling_backtest', {})['enabled'] = False

    # ── Rédaction IA : opt-in via la sortie 'report' ─────────────────────────
    report_demande = ('report' in outs)
    activer_ia(config, report_demande)

    # garch / var / backtest / charts sont intrinsèques au pipeline et au PDF :
    # ils sont toujours calculés, aucun mapping nécessaire.

    # ── Mode de sortie ───────────────────────────────────────────────────────
    # 'report' demande => mode classique : genere les figures + tableaux LaTeX
    #   standalone dont le rapport IA (main.tex) a besoin, puis compile main.pdf.
    # 'report' absent  => pdf_unique : PDF ReportLab consolide, rapide, sans IA.
    out = config.setdefault('output', {})
    out['pdf_unique']             = (not report_demande)
    out['sous_dossier_par_ticker'] = True
    out['dossier_resultats']      = dossier_sortie

    # ── Détail enrichi de backtest (6 tests FRTB/Berkowitz) pour l'API v1 ─────
    # Opt-in : la surface legacy /api/run ne le demande pas (comportement inchangé).
    config.setdefault('backtest', {})['detail_frtb'] = bool(detail_frtb)

    return config
