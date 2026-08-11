# -*- coding: utf-8 -*-
"""conftest.py racine -- correction de sys.path pour les tests.

Probleme : quand pytest est lance depuis la racine du projet, '' (CWD) est le
premier element de sys.path. PathFinder trouve alors tickerlab/ (dossier
de resultats, sans __init__.py) comme namespace package AVANT que le
_EditableFinder de l'install editable puisse trouver le vrai __init__.py.

Correction : retirer '' de sys.path et ajouter le chemin absolu du projet.
"""
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))

# Retirer CWD generique ('') qui laisse PathFinder trouver tickerlab/ output
if '' in sys.path:
    sys.path.remove('')

# Ajouter le chemin absolu si absent (utile pour les invocations hors install)
if _here not in sys.path:
    sys.path.insert(0, _here)

# Invalider le cache si tickerlab a deja ete charge comme namespace package
if 'tickerlab' in sys.modules:
    mod = sys.modules['tickerlab']
    if getattr(mod, '__file__', None) is None:
        # C'est le namespace package de l'output dir -- le supprimer pour reimport
        del sys.modules['tickerlab']
