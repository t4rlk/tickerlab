# -*- coding: utf-8 -*-
"""Cache versionne par etape avec invalidation en cascade (Phase 3.2).

Chaque etape du pipeline possede une cle independante calculee recursivement
depuis le DAG — modifer un parametre amont invalide automatiquement tous les
descendants, sans toucher les branches non affectees.

Storage : {dossier}/.cache_v2/{etape}_{cle12}.pkl + manifest.json
"""
import hashlib
import importlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger('tickerlab.cache_v2')

# ── DAG ──────────────────────────────────────────────────────────────────────
# Format: etape -> (upstream_etapes, config_sections_affectees)
# upstream_etapes : etapes dont la cle entre dans la cle de cette etape
# config_sections : cles du dict config dont la valeur impacte cette etape

# Tableau de couverture : sections config reellement consommees par chaque etape
# (verifiees par audit main.py + modules, 2026-06-13)
#
# Etape          Sections config consommees             Source
# -----------    ------------------------------------   --------------------------------
# download       data                                   main.py: config['data'][...]
# arima          arima                                  main.py: **config['arima']
# garch          garch                                  main.py: **config['garch']
# igarch_diag    garch (+ data.frequency via cascade)   main.py: config['garch']['seuil_igarch']
#                                                        main.py: config['data']['frequency']
#                                                        → couvert par download→garch cascade
# component_garch component_garch                       module: config.get('component_garch',{})
# fhs            fhs, var                               fhs.py:314: config['var']['niveaux']
# dm_gk          dm_gk, backtest, var                   dm_gk.py:384: config['backtest']['split_ratio']
#                                                        dm_gk.py:385: config['var']['niveaux']
#                                                        Note: dm_gk s'execute AVANT backtest_oos
#                                                        — 'backtest' est une dep config, pas ordre
# var            var, bootstrap                         main.py: config['var'], config['bootstrap']
# backtest       backtest, var                          main.py: config['backtest']
#                                                        main.py:405: config['var']['n_simulations_mc']

ETAPES_DAG: dict[str, tuple[list[str], list[str]]] = {
    'download'       : ([],                       ['data']),
    'arima'          : (['download'],              ['arima']),
    'garch'          : (['download'],              ['garch']),
    'igarch_diag'    : (['garch'],                 ['garch']),
    'component_garch': (['garch', 'igarch_diag'],  ['component_garch']),
    'fhs'            : (['garch'],                 ['fhs', 'var']),
    'dm_gk'          : (['garch', 'fhs'],          ['dm_gk', 'backtest', 'var']),
    'var'            : (['garch'],                 ['var', 'bootstrap']),
    'backtest'       : (['garch', 'fhs'],          ['backtest', 'var']),
}

# Module portant __etape_version__ pour chaque etape
_ETAPE_MODULE: dict[str, str] = {
    'download'       : 'tickerlab.core.data_loader',
    'arima'          : 'tickerlab.core.arima_selector',
    'garch'          : 'tickerlab.core.garch_selector',
    'igarch_diag'    : 'tickerlab.core.igarch_diagnostic',
    'component_garch': 'tickerlab.core.component_garch',
    'fhs'            : 'tickerlab.core.fhs',
    'dm_gk'          : 'tickerlab.core.dm_gk',
    'var'            : 'tickerlab.core.var_engine',
    'backtest'       : 'tickerlab.core.backtest',
}


def _version(etape: str) -> str:
    """Lit __etape_version__ du module associe, retourne '0' si absent."""
    try:
        mod = importlib.import_module(_ETAPE_MODULE[etape])
        return str(getattr(mod, '__etape_version__', '0'))
    except (ImportError, KeyError):
        return '0'


class CacheEtapes:
    """Cache par etape avec invalidation automatique en cascade.

    Parameters
    ----------
    dossier : str | Path
        Dossier resultats du pipeline (ex. resultats/BZ=F_weekly).
    config : dict
        Configuration complete du pipeline.
    ttl_download_heures : float
        TTL en heures pour l'etape download quand end_date est absent.
        Defaut : 24h.
    """

    SOUS_DOSSIER = '.cache_v2'

    def __init__(self, dossier: Any, config: dict,
                 ttl_download_heures: float = 24.0) -> None:
        self._dir = Path(dossier) / self.SOUS_DOSSIER
        self._config = config
        self._ttl_h = float(
            config.get('cache_v2', {}).get('ttl_download_heures', ttl_download_heures)
        )
        self._stats: dict[str, str] = {}
        self._manifest: dict = self._lire_manifest()

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self._dir / 'manifest.json'

    def _lire_manifest(self) -> dict:
        p   = self._manifest_path()
        tmp = p.with_name(p.name + '.tmp')
        if tmp.exists():
            _log.debug('  [Cache] suppression manifest.json.tmp residuel au demarrage.')
            tmp.unlink(missing_ok=True)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _sauver_manifest(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        p   = self._manifest_path()
        tmp = p.with_name(p.name + '.tmp')
        try:
            tmp.write_text(
                json.dumps(self._manifest, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
            os.replace(tmp, p)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    # ── Calcul de cle (recursif, pur) ────────────────────────────────────────

    def cle(self, etape: str, _vu: Optional[set] = None) -> str:
        """Calcule la cle de l'etape de maniere recursive et pure.

        La cle ne depend que du config et du DAG — jamais du manifest ni du
        disque. Ainsi le calcul est deterministe independamment de l'ordre
        d'execution et de l'etat du cache.

        Protege contre les cycles via _vu (ne devrait jamais se produire avec
        un DAG acyclique).
        """
        if _vu is None:
            _vu = set()
        if etape in _vu:
            return '000000000000'
        _vu = _vu | {etape}

        upstream_etapes, config_sections = ETAPES_DAG[etape]

        config_deps = {k: self._config.get(k) for k in config_sections}
        upstream_keys = [self.cle(e, _vu) for e in upstream_etapes]
        version = _version(etape)

        payload = {
            'config': config_deps,
            'upstream': upstream_keys,
            'version': version,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(serialized).hexdigest()[:12]

    # ── API publique ──────────────────────────────────────────────────────────

    def get(self, etape: str) -> Optional[Any]:
        """Retourne la valeur cachee ou None (MISS).

        Pour 'download' sans end_date explicite, verifie le TTL.
        """
        cle12 = self.cle(etape)
        pkl   = self._dir / f'{etape}_{cle12}.pkl'
        tmp   = pkl.with_name(pkl.name + '.tmp')
        if tmp.exists():
            _log.debug('  [Cache] suppression tmp residuel au chargement : %s', tmp.name)
            tmp.unlink(missing_ok=True)

        if not pkl.exists():
            self._stats[etape] = 'MISS'
            return None

        if etape == 'download' and self._ttl_expire(cle12):
            self._stats[etape] = 'MISS'
            _log.info('  [Cache] download : TTL expire, re-telechargement.')
            return None

        try:
            with pkl.open('rb') as fh:
                valeur = pickle.load(fh)
            self._stats[etape] = 'HIT'
            return valeur
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            _log.warning('  [Cache] lecture echec %s : %s', pkl.name, exc)
            self._stats[etape] = 'MISS'
            return None

    def set(self, etape: str, valeur: Any) -> None:
        """Serialise valeur dans le cache et met a jour le manifest.

        Ecriture atomique : dump dans un .pkl.tmp puis os.replace() vers le .pkl
        final. En cas d'interruption, le .pkl final reste intact (ou absent) —
        jamais tronque. Le manifest n'est mis a jour qu'apres le replace reussi.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        cle12 = self.cle(etape)
        pkl   = self._dir / f'{etape}_{cle12}.pkl'
        tmp   = pkl.with_name(pkl.name + '.tmp')

        self._supprimer_anciens_pkls(etape, cle12)

        try:
            with tmp.open('wb') as fh:
                pickle.dump(valeur, fh, protocol=5)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, pkl)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        entry: dict = {'cle12': cle12}
        if etape == 'download':
            entry['ts_utc'] = time.time()
        self._manifest[etape] = entry
        self._sauver_manifest()
        self._stats[etape] = 'MISS'

    def invalider(self, etape: Optional[str] = None) -> int:
        """Supprime les fichiers pickle de l'etape (ou de toutes si etape=None).

        Returns
        -------
        int
            Nombre de fichiers supprimes.
        """
        if not self._dir.exists():
            return 0
        supprime = 0
        etapes = [etape] if etape else list(ETAPES_DAG)
        for e in etapes:
            for f in self._dir.glob(f'{e}_*.pkl'):
                f.unlink(missing_ok=True)
                supprime += 1
            if e in self._manifest:
                del self._manifest[e]
        if supprime:
            self._sauver_manifest()
        return supprime

    def stats(self) -> dict[str, str]:
        """Retourne le dict {etape -> 'HIT'|'MISS'|'DISABLED'} apres le run."""
        return dict(self._stats)

    # ── Helpers prives ────────────────────────────────────────────────────────

    def _ttl_expire(self, cle12_actuelle: str) -> bool:
        """True si le cache download est trop vieux ET que end_date est absent."""
        end_date = self._config.get('data', {}).get('end_date', '')
        if end_date:
            return False
        entry = self._manifest.get('download', {})
        if entry.get('cle12') != cle12_actuelle:
            return True
        ts = entry.get('ts_utc')
        if ts is None:
            return True
        age_h = (time.time() - float(ts)) / 3600.0
        return age_h > self._ttl_h

    def _supprimer_anciens_pkls(self, etape: str, cle12_garde: str) -> None:
        """Supprime les .pkl dont la cle differe + tous les .pkl.tmp residuels."""
        for f in self._dir.glob(f'{etape}_*.pkl'):
            if f.stem != f'{etape}_{cle12_garde}':
                f.unlink(missing_ok=True)
        for f in self._dir.glob(f'{etape}_*.pkl.tmp'):
            _log.debug('  [Cache] suppression tmp residuel : %s', f.name)
            f.unlink(missing_ok=True)
