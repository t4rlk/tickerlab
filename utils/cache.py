# -*- coding: utf-8 -*-
"""Cache intermediaire du pipeline TickerLab (pickle)."""
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional

_log = logging.getLogger('tickerlab.cache')

_CACHE_VERSION = "1.0"


def _empreinte_config(config: dict) -> str:
    """Hash MD5 de ticker + start_date + end_date + version -> cle de cache."""
    data = config.get('data', {})
    cle = (
        f"{data.get('ticker', '')}|"
        f"{data.get('start_date', '')}|"
        f"{data.get('end_date', '')}|"
        f"{data.get('frequency', 'daily')}|"
        f"{_CACHE_VERSION}"
    )
    return hashlib.md5(cle.encode()).hexdigest()


def sauvegarder_cache(dossier: str, config: dict, **resultats) -> None:
    """
    Sauvegarde les resultats lourds du pipeline dans un fichier pickle.

    Parameters
    ----------
    dossier : str
        Dossier resultats (ex: 'tickerlab/resultats/').
    config : dict
        Configuration du pipeline (utilisee pour la cle de cache).
    **resultats
        Donnees a sauvegarder : prix, rendements, arima_result, garch_final,
        df_garch, df_var, df_vol, df_bt, T_train, T_eff_dyn.
    """
    cache_dir = Path(dossier) / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / 'pipeline_state.pkl'
    payload = {
        'version':   _CACHE_VERSION,
        'empreinte': _empreinte_config(config),
        'resultats': resultats,
    }
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        _log.info('  OK cache -> %s', cache_path)
    except Exception as e:
        _log.warning('  [WARN] Sauvegarde cache echouee : %s', e)


def charger_cache(dossier: str, config: dict) -> Optional[dict]:
    """
    Charge le cache si l'empreinte correspond. Retourne None sinon.

    Parameters
    ----------
    dossier : str
        Dossier resultats.
    config : dict
        Configuration du pipeline.

    Returns
    -------
    dict or None
        Dictionnaire des resultats si cache valide, None sinon.
    """
    cache_path = Path(dossier) / '.cache' / 'pipeline_state.pkl'
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        _log.warning('  [WARN] Cache corrompu, ignore : %s', e)
        return None

    if payload.get('version') != _CACHE_VERSION:
        _log.info('  Cache version %s != %s - recalcul.',
                  payload.get('version'), _CACHE_VERSION)
        return None
    if payload.get('empreinte') != _empreinte_config(config):
        _log.info('  Cache invalide (ticker/dates differents) - recalcul.')
        return None

    _log.info('  Cache valide charge depuis %s', cache_path)
    return payload['resultats']
