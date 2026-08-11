# -*- coding: utf-8 -*-
"""Helpers transverses pour la génération PDF — commun à tous les sous-modules sections/."""
import math
import pandas as pd
import logging as _logging

import tickerlab.core.rapport._helpers as _H
from reportlab.platypus import PageBreak as _PageBreak
from tickerlab.core.rapport._helpers import LARG_U, _spacer, _tableau_eviews

_sections_log = _logging.getLogger('tickerlab.rapport._sections')



def _log(msg: str) -> None:
    _sections_log.info('%s', msg)

from tickerlab.core.rapport._helpers import (
    _h1, _h2, _p, _caption, _spacer,
    _embed_figure, _tableau_eviews, _correlogramme_eviews,
    _fmt, _fmt_pval, _fmt_signif, _hex, _warn,
    LARG_U, nom_actif,
)
from tickerlab.core.rapport._stats import (
    stats_desc, adf_complet, pp_complet, kpss_complet,
)


# ── Acces theme courant (toujours a jour apres _set_theme) ───────────────────



# ── Acces theme courant (toujours a jour apres _set_theme) ───────────────────

def _th() -> dict:
    return _H._THEME


# ── Micro-utilitaires ─────────────────────────────────────────────────────────



# ── Micro-utilitaires ─────────────────────────────────────────────────────────

def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return True




def _int_or_na(x) -> str:
    return 'N/A' if _is_nan(x) else str(int(float(x)))




def _sq(x) -> pd.Series:
    """Garantit une pd.Series — squeeze si DataFrame une colonne (prix yfinance)."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0].squeeze()
    if not isinstance(x, pd.Series):
        return pd.Series(x)
    return x


# ── Helpers pagination v6 ─────────────────────────────────────────────────────



# ── Helpers pagination v6 ─────────────────────────────────────────────────────

def _encadre_pedagogique(message: str, code: str, config: dict) -> list:
    """
    Encadré pédagogique thème-adaptatif pour l'interprétation économique ARIMA.

    Fond = ``entete_fond`` du thème courant, bordure = ``accent``.

    Parameters
    ----------
    message : str
        Texte pédagogique issu de ``arima_result['message_pedagogique']``.
    code : str
        Code d'interprétation : ``'serie_predictible'``, ``'bruit_faible'``,
        ``'martingale_marche_efficient'`` ou ``'incertain'``.
    config : dict
        Configuration globale (non utilisée directement, mais cohérente avec le reste).

    Returns
    -------
    list of Flowable
        ``[Table, Spacer]`` si message non vide, sinon ``[]``.
    """
    from reportlab.platypus import Table, TableStyle, Paragraph

    if not message:
        return []

    theme = _th()
    bg_color     = theme['entete_fond']
    border_color = theme['accent']

    _titres = {
        'serie_predictible':           'Interpretation : Serie previsible',
        'bruit_faible':                'Interpretation : Bruit faible (Lo & MacKinlay, 1988)',
        'martingale_marche_efficient': (
            'Interpretation : Marche aleatoire — Efficience faible (Fama, 1970)'
        ),
        'incertain':                   'Interpretation : Resultat non concluant',
    }
    titre_box = _titres.get(code, 'Interpretation ARIMA')

    styles    = _H._get_styles()
    title_par = Paragraph(f'<b>{titre_box}</b>', styles['Heading2'])
    msg_par   = Paragraph(message, styles['Body'])

    t = Table([[title_par], [msg_par]], colWidths=[LARG_U])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), bg_color),
        ('BOX',           (0, 0), (-1, -1), 1.5, border_color),
        ('LINEBELOW',     (0, 0), (-1, 0),  0.5, border_color),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return [t, _spacer(0.3)]




def _page_break_if(config: dict) -> list:
    """
    Retourne [PageBreak()] si config.rapport.une_page_un_message est True,
    sinon [_spacer(0.5)] (comportement legacy).
    """
    if config.get('rapport', {}).get('une_page_un_message', True):
        return [_PageBreak()]
    return [_spacer(0.5)]




def _lags_rapport(config: dict) -> list:
    """
    Liste de lags LB a afficher dans le rapport PDF.
    Lit config.rapport.lags_ljung_box_rapport (defaut [1,2,3,4,5,6,10,12,15,20,24,36]).
    """
    lags = config.get('rapport', {}).get(
        'lags_ljung_box_rapport', [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 36]
    )
    return [int(l) for l in lags]




def _max_lignes_tableau(config: dict) -> int:
    """
    Limite de lignes par tableau (config.rapport.max_lignes_par_tableau, defaut 12).
    """
    return int(config.get('rapport', {}).get('max_lignes_par_tableau', 12))




def _split_tableau(
    titre: str,
    colonnes: list,
    lignes: list,
    max_rows: int = 12,
    **kw,
) -> list:
    """
    Decoupe un tableau long en sous-tableaux de max_rows lignes chacun.

    Utile pour les tableaux depassant une page A4.
    Note : extra_styles avec indices absolus est supprime sur les tables splittees
    (les colorations cellules dependent des indices de ligne locaux).

    Parameters
    ----------
    titre     : str   Titre du tableau (suffixe (n/N) ajoute si multi-parties).
    colonnes  : list  En-tetes de colonnes.
    lignes    : list  Lignes de donnees (list of list).
    max_rows  : int   Lignes max par partie (defaut 12).
    **kw              Transmis a _tableau_eviews sauf extra_styles (retire sur split).
    """
    if len(lignes) <= max_rows:
        return _tableau_eviews(titre, colonnes, lignes, **kw)

    kw.pop('extra_styles', None)   # indices absolus invalides apres split
    result = []
    n_parts = math.ceil(len(lignes) / max_rows)
    for i in range(n_parts):
        chunk = lignes[i * max_rows: (i + 1) * max_rows]
        label = f'{titre} ({i + 1}/{n_parts})'
        result.extend(_tableau_eviews(label, colonnes, chunk, **kw))
        if i < n_parts - 1:
            result.append(_PageBreak())
    return result


# ── Decisions stationnarite ───────────────────────────────────────────────────



def _extra_pval_bold(specs_list: list, row_pval: int) -> list:
    """Extra TableStyle : gras + couleur warn pour p-values < 0.05."""
    th = _th()
    extra = []
    for col_idx, r in enumerate(specs_list, start=1):
        try:
            if float(r.get('pval', 1.0)) < 0.05:
                extra += [
                    ('FONTNAME',  (col_idx, row_pval), (col_idx, row_pval), 'Courier-Bold'),
                    ('TEXTCOLOR', (col_idx, row_pval), (col_idx, row_pval), th['warn']),
                ]
        except Exception as exc:
            _sections_log.debug('_extra_pval_bold : style pval ignoré : %s', exc)
    return extra


# ── Tables stationnarite compactes (3 specs en colonnes) ─────────────────────

