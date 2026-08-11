# -*- coding: utf-8 -*-
"""Couche web tickerlab : API FastAPI + configurateur statique.

Cette couche est une fine façade au-dessus du pipeline économétrique existant
(tickerlab.main.run_pipeline). Elle ne réimplémente AUCUNE logique de calcul :
elle traduit la configuration du front en dict de config, lance le pipeline
dans un thread, et expose la progression + le rapport PDF.
"""
