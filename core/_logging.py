# -*- coding: utf-8 -*-
"""Setup centralisé du logger tickerlab."""
import logging

_PKG = 'tickerlab'


def setup_tickerlab_logger(level: int = logging.INFO) -> None:
    """Idempotent : ajoute un StreamHandler si absent, ajuste le niveau à chaque appel."""
    logger = logging.getLogger(_PKG)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger tickerlab.<name> (hérite du niveau du parent)."""
    return logging.getLogger(f'{_PKG}.{name}')
