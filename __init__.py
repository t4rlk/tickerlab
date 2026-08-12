# -*- coding: utf-8 -*-
"""Pipeline ARIMA-GARCH-VaR/TVaR."""
__version__ = "6.3.1"
__author__  = "TickerLab"


def _bloquer_sous_espaces_non_publics():
    """Empeche dev/, scripts/, resultats/ d'etre importables comme tickerlab.*.

    Probleme inherent au layout root-IS-package : tickerlab.__path__ pointe
    sur la racine du projet, ce qui rend tous ses sous-repertoires visibles comme
    packages namespace. Corriger en inserant None dans sys.modules pour les noms
    non publics sans __init__.py (namespace packages parasites seulement).
    Les sous-repertoires avec __init__.py (core, utils, tests) ne sont pas bloques.
    """
    import sys
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    for nom in os.listdir(root):
        if nom.startswith(('.', '_')):
            continue
        chemin = os.path.join(root, nom)
        if not os.path.isdir(chemin):
            continue
        # Ne bloquer que les repertoires sans __init__.py (namespace packages)
        if os.path.exists(os.path.join(chemin, '__init__.py')):
            continue
        bloque = 'tickerlab.' + nom
        if bloque not in sys.modules:
            sys.modules[bloque] = None


_bloquer_sous_espaces_non_publics()
del _bloquer_sous_espaces_non_publics

from tickerlab.core.data_loader    import telecharger_prix, calculer_rendements
from tickerlab.core.stationarity   import analyser_stationnarite, tester_stationnarite
from tickerlab.core.arima_selector import selectionner_arima, grid_search_arima
from tickerlab.core.garch_selector import (grid_search_garch, selectionner_meilleur,
                                           estimer_final)
from tickerlab.core.var_engine     import (calculer_var_tvar, construire_df_vol,
                                           var_normale, var_student, var_historique,
                                           tvar_historique, tvar_normale,
                                           tvar_student, var_cornish_fisher)
from tickerlab.core.backtest       import backtest_oos, kupiec_test, christoffersen_test
from tickerlab.core.reporter       import generer_rapport

__all__ = [
    '__version__', '__author__',
    'telecharger_prix', 'calculer_rendements',
    'analyser_stationnarite', 'tester_stationnarite',
    'selectionner_arima', 'grid_search_arima',
    'grid_search_garch', 'selectionner_meilleur', 'estimer_final',
    'calculer_var_tvar', 'construire_df_vol',
    'var_normale', 'var_student', 'var_historique',
    'tvar_historique', 'tvar_normale', 'tvar_student', 'var_cornish_fisher',
    'backtest_oos', 'kupiec_test', 'christoffersen_test',
    'generer_rapport',
]
