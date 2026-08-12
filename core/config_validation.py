# -*- coding: utf-8 -*-
"""Validation centralisée de la configuration et calcul d'empreinte scientifique."""
import hashlib
import json
import warnings

# Clés dont la valeur modifie les résultats numériques.
# output, cache, ai, _metrics_only, _reuse_cache : infrastructure seulement.
_CLES_SCIENTIFIQUES = {
    'data', 'arima', 'garch', 'var', 'backtest',
    'rolling_backtest', 'bootstrap', 'dm_gk', 'fhs',
    'component_garch', 'stress_testing',
}


def config_hash(config: dict) -> str:
    """Empreinte SHA-256 (12 hex) des clés scientifiques uniquement.

    Les clés d'infrastructure (output, cache, ai, _metrics_only…) sont exclues
    pour que deux runs produisant les mêmes chiffres partagent le même hash.
    """
    scientific = {k: v for k, v in config.items() if k in _CLES_SCIENTIFIQUES}
    serialized = json.dumps(scientific, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def valider_config(config: dict, n_obs: int) -> None:
    """Valide la configuration runtime (après téléchargement des données).

    Lève ValidationError si un paramètre est physiquement incompatible avec
    la série. Émet UserWarning pour les cas dégradés mais acceptables.

    Remplace _valider_config_runtime() de main.py.
    """
    from .exceptions import ValidationError

    rb_cfg = config.get('rolling_backtest', {})
    if rb_cfg.get('enabled', False):
        window = int(rb_cfg.get('window_size', 1000))
        if window >= n_obs:
            raise ValidationError(
                f'rolling_backtest.window_size={window} >= n_obs={n_obs}. '
                'Désactiver rolling_backtest ou utiliser une série plus longue '
                f'(minimum recommandé : window_size + 100 obs).',
                etape='config_validation',
                contexte={'window_size': window, 'n_obs': n_obs},
            )
        if window >= int(0.9 * n_obs):
            warnings.warn(
                f'[CONFIG] rolling_backtest.window_size={window} laisse '
                f'{n_obs - window} prédictions OOS — très peu pour des statistiques '
                f'robustes. Recommandé : window_size <= {int(0.7 * n_obs)}.',
                UserWarning, stacklevel=2,
            )

    bt_cfg = config.get('backtest', {})
    split_ratio = float(bt_cfg.get('split_ratio', 0.70))
    T_oos = n_obs - int(n_obs * split_ratio)
    if T_oos < 50:
        warnings.warn(
            f'[CONFIG] backtest.split_ratio={split_ratio} laisse {T_oos} obs OOS '
            '(recommandé >= 50). Backtesting peu robuste.',
            UserWarning, stacklevel=2,
        )
