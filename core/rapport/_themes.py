# -*- coding: utf-8 -*-
"""Themes visuels pour le PDF — palette de couleurs par theme."""
import logging as _logging
from reportlab.lib import colors

_log = _logging.getLogger('tickerlab.rapport._themes')

# ── Themes disponibles ────────────────────────────────────────────────────────
# Chaque theme definit la palette complete. Pour ajouter un theme, copier un
# bloc existant et modifier les valeurs hex. Selection via config.yaml -> rapport.theme.

THEMES = {
    'academique': {
        # Style rapport academique — sobre, bleu-gris fonce
        'titre_fond':       '#2C3E50',
        'titre_texte':      '#FFFFFF',
        'entete_fond':      '#ECF0F1',
        'ligne_paire':      '#F5F6FA',
        'ligne_impaire':    '#FFFFFF',
        'bordure':          '#BDC3C7',
        'accent':           '#2980B9',
        'warn':             '#E74C3C',
        'texte_secondaire': '#555555',
        'h2_texte':         '#34495E',
        'note':             '#666666',
    },
    'monochrome': {
        # Noir et blanc pur — proche EViews / impression laser
        'titre_fond':       '#000000',
        'titre_texte':      '#FFFFFF',
        'entete_fond':      '#E0E0E0',
        'ligne_paire':      '#F8F8F8',
        'ligne_impaire':    '#FFFFFF',
        'bordure':          '#888888',
        'accent':           '#000000',
        'warn':             '#000000',
        'texte_secondaire': '#333333',
        'h2_texte':         '#000000',
        'note':             '#555555',
    },
    'eviews': {
        # Style EViews authentique — fond bleu marine, accents indigo
        'titre_fond':       '#1F3864',
        'titre_texte':      '#FFFFFF',
        'entete_fond':      '#D9E1F2',
        'ligne_paire':      '#FFF8E7',
        'ligne_impaire':    '#FFFFFF',
        'bordure':          '#8FAADC',
        'accent':           '#1F3864',
        'warn':             '#C00000',
        'texte_secondaire': '#1F3864',
        'h2_texte':         '#1F3864',
        'note':             '#404040',
    },
    'minimal': {
        # Tres epure, gris sur blanc — presentation moderne
        'titre_fond':       '#FAFAFA',
        'titre_texte':      '#1A1A1A',
        'entete_fond':      '#FFFFFF',
        'ligne_paire':      '#FAFAFA',
        'ligne_impaire':    '#FFFFFF',
        'bordure':          '#E0E0E0',
        'accent':           '#0066CC',
        'warn':             '#CC0000',
        'texte_secondaire': '#666666',
        'h2_texte':         '#333333',
        'note':             '#888888',
    },
    'sombre': {
        # Accents bleu-indigo profonds — presentation slides
        'titre_fond':       '#1A237E',
        'titre_texte':      '#FFFFFF',
        'entete_fond':      '#E8EAF6',
        'ligne_paire':      '#F5F5F5',
        'ligne_impaire':    '#FFFFFF',
        'bordure':          '#9FA8DA',
        'accent':           '#3F51B5',
        'warn':             '#D32F2F',
        'texte_secondaire': '#424242',
        'h2_texte':         '#283593',
        'note':             '#616161',
    },
}

THEME_DEFAUT = 'academique'


def charger_theme(nom: str = None) -> dict:
    """
    Retourne le dict de couleurs ReportLab pour un theme donne.

    Parameters
    ----------
    nom : str, optional
        Nom du theme. Si None ou inconnu, retourne THEME_DEFAUT.

    Returns
    -------
    dict
        {nom_couleur: reportlab.lib.colors.HexColor}
    """
    nom = nom or THEME_DEFAUT
    if nom not in THEMES:
        _log.warning('[WARN] Theme inconnu "%s" — bascule sur "%s".', nom, THEME_DEFAUT)
        nom = THEME_DEFAUT
    return {k: colors.HexColor(v) for k, v in THEMES[nom].items()}


def lister_themes() -> list:
    """Liste les noms de themes disponibles."""
    return list(THEMES.keys())
