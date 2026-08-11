# -*- coding: utf-8 -*-
"""Hiérarchie d'exceptions typées pour le pipeline TickerLab."""


class PipelineError(Exception):
    """Erreur de base du pipeline. Toutes les exceptions métier en héritent."""

    def __init__(self, message: str, ticker: str = None, etape: str = None,
                 contexte: dict = None):
        super().__init__(message)
        self.ticker = ticker
        self.etape = etape
        self.contexte = contexte or {}

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.ticker:
            parts.append(f'ticker={self.ticker}')
        if self.etape:
            parts.append(f'etape={self.etape}')
        return ' | '.join(parts)


class DataError(PipelineError):
    """Erreur liée au téléchargement ou à la préparation des données."""


class ModelError(PipelineError):
    """Erreur d'estimation ou de convergence de modèle (ARIMA, GARCH, FHS…)."""


class ValidationError(PipelineError):
    """Erreur de validation de la configuration ou des paramètres d'entrée."""


class ExportError(PipelineError):
    """Erreur lors de la génération des sorties (PDF, LaTeX, Excel, CSV…)."""


class ModuleUnavailableError(PipelineError):
    """Module d'analyse demandé mais non disponible sur ce déploiement.

    Levée par la couche web/CLI quand un `module` est reconnu mais non câblé sur
    la branche courante (ex. le module multivarié `var` vit sur une autre branche).
    Doit produire une erreur explicite — jamais un crash ni un silence.
    """
