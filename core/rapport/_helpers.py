# -*- coding: utf-8 -*-
"""
Helpers bas niveau pour la generation PDF ReportLab.

Blocs A (infrastructure), B (tableau EViews), C (correlogramme EViews).
Couleurs chargees dynamiquement via _THEME — cf. _themes.py et _set_theme().
"""
import math
import warnings
from io import BytesIO
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, Spacer, Table, TableStyle,
)

from tickerlab.core.rapport._themes import charger_theme

warnings.filterwarnings('ignore')


# ── Mise en page A4 ───────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4          # 595.28 x 841.89 pt
MARGE_G  = 2.5 * cm
MARGE_D  = 2.5 * cm
MARGE_H  = 2.0 * cm
MARGE_B  = 2.0 * cm
LARG_U   = PAGE_W - MARGE_G - MARGE_D   # largeur utile ~ 454 pt


# ── Theme actif (modifiable via _set_theme) ───────────────────────────────────
_THEME: dict = charger_theme()   # defaut : 'academique'
_STYLES: Optional[dict] = None   # regenere a chaque changement de theme


def _set_theme(nom: str) -> None:
    """
    Change le theme actif. Appeler une seule fois avant de construire le PDF.
    Invalide le cache de styles pour forcer la regeneration avec les nouvelles couleurs.
    """
    global _THEME, _STYLES
    _THEME  = charger_theme(nom)
    _STYLES = None


def _hex(color) -> str:
    """Convertit un objet ReportLab Color en chaine hex '#RRGGBB' pour matplotlib."""
    if isinstance(color, str):
        return color
    r = int(round(color.red   * 255))
    g = int(round(color.green * 255))
    b = int(round(color.blue  * 255))
    return f'#{r:02X}{g:02X}{b:02X}'


# ── Log global ────────────────────────────────────────────────────────────────
_WARNINGS_LOG: List[str] = []


def _warn(msg: str) -> None:
    """Log un avertissement via logging et dans _WARNINGS_LOG."""
    import logging as _logging
    _logging.getLogger('tickerlab.rapport').warning('[WARN] %s', msg)
    _WARNINGS_LOG.append(msg)


# ── Styles ReportLab (singleton invalide lors de _set_theme) ─────────────────

def _get_styles() -> dict:
    """Initialise et retourne le dictionnaire de styles (singleton par theme)."""
    global _STYLES
    if _STYLES is not None:
        return _STYLES

    base = getSampleStyleSheet()

    def _s(name, **kw):
        return ParagraphStyle(name, parent=base['Normal'], **kw)

    _STYLES = {
        # ── Headings (lies au TOC via afterFlowable) ──────────────────────
        'Heading1': _s(
            'Heading1',
            fontSize=13, fontName='Helvetica-Bold',
            spaceAfter=6, spaceBefore=14,
            textColor=_THEME['titre_fond'], keepWithNext=True,
        ),
        'Heading2': _s(
            'Heading2',
            fontSize=11, fontName='Helvetica-Bold',
            spaceAfter=4, spaceBefore=9,
            textColor=_THEME['h2_texte'], keepWithNext=True,
        ),
        # ── Corps de texte ─────────────────────────────────────────────────
        'Body': _s(
            'Body',
            fontSize=10, fontName='Helvetica',
            spaceAfter=4, leading=14,
        ),
        'BodyItalic': _s(
            'BodyItalic',
            fontSize=10, fontName='Helvetica-Oblique',
            spaceAfter=4, leading=14,
        ),
        # ── Legendes et notes ──────────────────────────────────────────────
        'Caption': _s(
            'Caption',
            fontSize=8, fontName='Helvetica-Oblique',
            alignment=TA_CENTER, spaceAfter=6,
            textColor=_THEME['texte_secondaire'],
        ),
        'Note': _s(
            'Note',
            fontSize=8, fontName='Helvetica',
            spaceAfter=8, textColor=_THEME['note'],
        ),
        # ── Monospace ─────────────────────────────────────────────────────
        'Code': _s(
            'Code',
            fontSize=8, fontName='Courier',
            spaceAfter=2, leading=10,
        ),
        # ── Tableau ───────────────────────────────────────────────────────
        'TH': _s(
            'TH',
            fontSize=9, fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            textColor=_THEME['titre_texte'],
        ),
        'TH_left': _s(
            'TH_left',
            fontSize=9, fontName='Helvetica-Bold',
            alignment=TA_LEFT,
        ),
        # ── TOC ───────────────────────────────────────────────────────────
        'TOCEntry1': _s(
            'TOCEntry1',
            fontSize=11, fontName='Helvetica-Bold',
            spaceBefore=3, spaceAfter=2, leftIndent=0,
        ),
        'TOCEntry2': _s(
            'TOCEntry2',
            fontSize=10, fontName='Helvetica',
            spaceBefore=1, spaceAfter=1, leftIndent=20,
        ),
        # ── Page de garde ─────────────────────────────────────────────────
        'Titre': _s(
            'Titre',
            fontSize=22, fontName='Helvetica-Bold',
            alignment=TA_CENTER, spaceAfter=10,
            textColor=_THEME['titre_fond'],
        ),
        'SousTitre': _s(
            'SousTitre',
            fontSize=14, fontName='Helvetica',
            alignment=TA_CENTER, spaceAfter=6,
            textColor=_THEME['texte_secondaire'],
        ),
        'MetaInfo': _s(
            'MetaInfo',
            fontSize=11, fontName='Helvetica',
            alignment=TA_CENTER, spaceAfter=4,
        ),
    }
    return _STYLES


# ── Utilitaires de formatage ──────────────────────────────────────────────────

def _safe_ticker(ticker: str) -> str:
    """Remplace les caracteres invalides d'un ticker Yahoo Finance par '_'."""
    for c in ('=', '^', '/', '\\', ' ', ':', '?', '*', '<', '>', '|'):
        ticker = ticker.replace(c, '_')
    return ticker


_NOMS_ACTIFS = {
    'BZ=F':     'Brent Crude',
    'CL=F':     'WTI Crude',
    'GC=F':     'Gold',
    'SI=F':     'Silver',
    'NG=F':     'Natural Gas',
    'HG=F':     'Copper',
    'BTC-USD':  'Bitcoin',
    'EURUSD=X': 'EUR/USD',
    '^GSPC':    'S&P 500',
    'AAPL':     'Apple',
}


def nom_actif(ticker: str) -> str:
    """Nom lisible de l'actif pour un ticker Yahoo Finance.

    Retombe sur le ticker brut si absent du mapping.
    """
    return _NOMS_ACTIFS.get(ticker, ticker)


def _fmt(x, dec: int = 4) -> str:
    """Formate un nombre pour affichage tableau. NaN/None -> 'N/A'."""
    if x is None:
        return 'N/A'
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(f) or math.isinf(f):
        return 'N/A'
    if f != 0 and abs(f) < 10 ** (-dec):
        return f'{f:.4e}'
    return f'{f:.{dec}f}'


def _fmt_pval(p) -> str:
    """Formate une p-valeur. Notation scientifique si < 0.0001."""
    try:
        f = float(p)
    except (TypeError, ValueError):
        return 'N/A'
    if math.isnan(f):
        return 'N/A'
    if f < 0.0001:
        return f'{f:.4e}'
    return f'{f:.4f}'


def _fmt_signif(p) -> str:
    """Etoiles de significativite : *** 1%, ** 5%, * 10%."""
    try:
        f = float(p)
    except (TypeError, ValueError):
        return ''
    if f < 0.01:
        return '***'
    if f < 0.05:
        return '**'
    if f < 0.10:
        return '*'
    return ''


# ── Helpers de mise en page ───────────────────────────────────────────────────

def _embed_figure(fig, width_cm: float = 15.5, height_cm: float = 8.0) -> Image:
    """
    Convertit une figure matplotlib en Image ReportLab (PNG BytesIO).
    Jamais ecrit sur disque. plt.close(fig) appele apres sauvegarde.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def _h1(text: str) -> Paragraph:
    """Titre de section niveau 1 (notifie le TOC via afterFlowable)."""
    return Paragraph(text, _get_styles()['Heading1'])


def _h2(text: str) -> Paragraph:
    """Titre de sous-section niveau 2."""
    return Paragraph(text, _get_styles()['Heading2'])


def _p(text: str) -> Paragraph:
    """Paragraphe de corps de texte courant."""
    return Paragraph(text, _get_styles()['Body'])


def _caption(text: str) -> Paragraph:
    """Legende de figure ou tableau (italique centre)."""
    return Paragraph(f'<i>{text}</i>', _get_styles()['Caption'])


def _note(text: str) -> Paragraph:
    """Note de bas de tableau (petit, couleur secondaire)."""
    return Paragraph(text, _get_styles()['Note'])


def _spacer(h_cm: float = 0.4) -> Spacer:
    return Spacer(1, h_cm * cm)


def _pagebreak() -> PageBreak:
    return PageBreak()


# ── Bloc B — Tableau EViews ───────────────────────────────────────────────────

def _tableau_eviews(
    titre: str,
    colonnes: List[str],
    lignes: List[list],
    note: str = None,
    col_widths: List[float] = None,
    extra_styles: List[tuple] = None,
    col0_is_label: bool = True,
    sous_titre: str = '',
) -> List:
    """
    Cree un tableau style EViews avec titre, en-tetes, donnees et note optionnelle.

    Parameters
    ----------
    titre : str
        Titre du tableau (premiere ligne, pleine largeur, fond theme).
    colonnes : list of str
        En-tetes des colonnes.
    lignes : list of list
        Donnees : chaque element est une liste de str ou Paragraph.
    note : str, optional
        Note de bas de tableau.
    col_widths : list of float, optional
        Largeurs en points. Auto-calculees si None.
    extra_styles : list of tuple, optional
        Commandes TableStyle supplementaires (bold p-values, cellules colorees...).
    col0_is_label : bool
        Si True (defaut), col 0 = Helvetica gauche. Sinon tout Courier droite.
    sous_titre : str
        Texte optionnel en plus petite police sous le titre.

    Returns
    -------
    list of Flowable
        [Table] ou [Table, Paragraph(note)].
    """
    styles   = _get_styles()
    n_cols   = len(colonnes)
    n_lignes = len(lignes)

    # ── Titre ──────────────────────────────────────────────────────────────
    titre_contenu = f'<b>{titre}</b>'
    if sous_titre:
        titre_contenu += f'<br/><font size="8">{sous_titre}</font>'

    data = [
        [Paragraph(titre_contenu, styles['TH'])] + [''] * (n_cols - 1),
        [Paragraph(c, styles['TH']) for c in colonnes],
    ]
    data.extend(lignes)

    # ── Largeurs de colonnes ───────────────────────────────────────────────
    if col_widths is None:
        if n_cols == 2:
            col_widths = [LARG_U * 0.55, LARG_U * 0.45]
        elif n_cols == 3:
            col_widths = [LARG_U * 0.40] + [LARG_U * 0.30] * 2
        else:
            first_w  = LARG_U * 0.28
            other_w  = (LARG_U - first_w) / (n_cols - 1)
            col_widths = [first_w] + [other_w] * (n_cols - 1)

    # ── Commandes de style ─────────────────────────────────────────────────
    cmds = [
        # Titre : span complet, fond theme, texte theme
        ('SPAN',          (0, 0),   (-1, 0)),
        ('BACKGROUND',    (0, 0),   (-1, 0),  _THEME['titre_fond']),
        ('TEXTCOLOR',     (0, 0),   (-1, 0),  _THEME['titre_texte']),
        ('ALIGN',         (0, 0),   (-1, 0),  'CENTER'),
        ('VALIGN',        (0, 0),   (-1, 0),  'MIDDLE'),
        ('TOPPADDING',    (0, 0),   (-1, 0),  6),
        ('BOTTOMPADDING', (0, 0),   (-1, 0),  6),

        # En-tetes colonnes
        ('BACKGROUND',    (0, 1),   (-1, 1),  _THEME['entete_fond']),
        ('ALIGN',         (0, 1),   (-1, 1),  'CENTER'),
        ('FONTNAME',      (0, 1),   (-1, 1),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 1),   (-1, 1),  9),
        ('TEXTCOLOR',     (0, 1),   (-1, 1),  colors.black),
        ('LINEBELOW',     (0, 1),   (-1, 1),  2.0, colors.black),   # effet double ligne

        # Donnees
        ('FONTSIZE',      (0, 2),   (-1, -1), 9),
        ('TOPPADDING',    (0, 2),   (-1, -1), 2),
        ('BOTTOMPADDING', (0, 2),   (-1, -1), 2),

        # Bordures externes
        ('LINEABOVE',     (0, 0),   (-1, 0),  1.0, _THEME['bordure']),
        ('LINEBELOW',     (0, -1),  (-1, -1), 1.0, _THEME['bordure']),
        ('LINEBEFORE',    (0, 0),   (0, -1),  0.5, _THEME['bordure']),
        ('LINEAFTER',     (-1, 0),  (-1, -1), 0.5, _THEME['bordure']),

        # Grille interieure fine
        ('INNERGRID',     (0, 1),   (-1, -1), 0.25, _THEME['bordure']),

        # Padding global
        ('LEFTPADDING',   (0, 0),   (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0),   (-1, -1), 5),
    ]

    # Alignement et police colonnes donnees
    if col0_is_label:
        cmds += [
            ('FONTNAME', (0, 2), (0, -1),  'Helvetica'),
            ('ALIGN',    (0, 2), (0, -1),  'LEFT'),
            ('FONTNAME', (1, 2), (-1, -1), 'Courier'),
            ('ALIGN',    (1, 2), (-1, -1), 'RIGHT'),
        ]
    else:
        cmds += [
            ('FONTNAME', (0, 2), (-1, -1), 'Courier'),
            ('ALIGN',    (0, 2), (-1, -1), 'RIGHT'),
        ]

    # Couleurs alternees sur les lignes de donnees (boucle explicite)
    for i in range(n_lignes):
        if i % 2 == 1:
            cmds.append(('BACKGROUND', (0, i + 2), (-1, i + 2), _THEME['ligne_paire']))

    if extra_styles:
        cmds.extend(extra_styles)

    t = Table(data, colWidths=col_widths, style=TableStyle(cmds), repeatRows=1)

    flowables = [t]
    if note:
        flowables.append(_note(f'Note : {note}'))
    return flowables


# ── Bloc C — Correlogramme EViews ─────────────────────────────────────────────

def _correlogramme_eviews(serie, lags: int = 36, titre: str = '') -> List:
    """
    Produit un correlogramme EViews complet : figure ACF/PACF + tableau numerique.

    Parameters
    ----------
    serie : pd.Series
        Serie temporelle (log-prix, rendements, residus...).
    lags : int
        Nombre de retards (defaut 36).
    titre : str
        Titre affiche dans la legende de la figure.

    Returns
    -------
    list of Flowable
        [Image, Spacer, Table, Note] ou [] si echec.
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError as e:
        _warn(f'statsmodels absent pour correlogramme : {e}')
        return []

    try:
        s = serie.dropna()
        n = len(s)
        if n < lags + 5:
            _warn(f'Serie trop courte pour correlogramme ({n} obs, {lags} lags).')
            return []

        conf     = 1.96 / np.sqrt(n)
        ac_vals  = acf(s, nlags=lags, fft=True)[1:]
        pac_vals = pacf(s, nlags=lags)[1:]
        lb       = acorr_ljungbox(s, lags=list(range(1, lags + 1)), return_df=True)
        q_stats  = lb['lb_stat'].values
        pvals    = lb['lb_pvalue'].values

        # ── Figure barres ACF + PACF ──────────────────────────────────────
        warn_hex = _hex(_THEME['warn'])
        c_ac  = ['steelblue'  if abs(v) < conf else warn_hex for v in ac_vals]
        c_pac = ['darkorange' if abs(v) < conf else warn_hex for v in pac_vals]

        lags_arr = np.arange(1, lags + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.8))
        if titre:
            fig.suptitle(titre, fontsize=9, fontweight='bold')

        ax1.bar(lags_arr, ac_vals,  color=c_ac,  width=0.7, alpha=0.85)
        ax1.axhline( conf, ls='--', color='steelblue',  lw=0.8, alpha=0.5)
        ax1.axhline(-conf, ls='--', color='steelblue',  lw=0.8, alpha=0.5)
        ax1.axhline(0, color='black', lw=0.4)
        ax1.set_title('Autocorrelation (ACF)', fontsize=9)
        ax1.set_xlabel('Retard', fontsize=8)
        ax1.set_ylabel('AC', fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.2, linestyle=':')

        ax2.bar(lags_arr, pac_vals, color=c_pac, width=0.7, alpha=0.85)
        ax2.axhline( conf, ls='--', color='darkorange', lw=0.8, alpha=0.5)
        ax2.axhline(-conf, ls='--', color='darkorange', lw=0.8, alpha=0.5)
        ax2.axhline(0, color='black', lw=0.4)
        ax2.set_title('Autocorrelation partielle (PACF)', fontsize=9)
        ax2.set_xlabel('Retard', fontsize=8)
        ax2.set_ylabel('PAC', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(True, alpha=0.2, linestyle=':')

        fig.tight_layout()
        img = _embed_figure(fig, width_cm=15.5, height_cm=4.5)

        # ── Table numerique Lag | AC | PAC | Q-Stat | Prob ───────────────
        entetes = ['Retard', 'AC', 'PAC', 'Q-Stat', 'Prob.']
        col_w   = [
            LARG_U * 0.10,
            LARG_U * 0.16,
            LARG_U * 0.16,
            LARG_U * 0.30,
            LARG_U * 0.28,
        ]

        lignes_tab = []
        bold_cmds  = []
        for k in range(lags):
            pv     = pvals[k]
            pv_str = _fmt_pval(pv)
            lignes_tab.append([
                str(k + 1),
                f'{ac_vals[k]:.3f}',
                f'{pac_vals[k]:.3f}',
                f'{q_stats[k]:.4f}',
                pv_str,
            ])
            if pv < 0.05:
                row_idx = k + 2   # +2 : ligne titre + ligne en-tetes
                bold_cmds.append(('FONTNAME',  (4, row_idx), (4, row_idx), 'Courier-Bold'))
                bold_cmds.append(('TEXTCOLOR', (4, row_idx), (4, row_idx), _THEME['warn']))

        tab_flowables = _tableau_eviews(
            titre        = 'Correlogramme numerique (Ljung-Box)',
            colonnes     = entetes,
            lignes       = lignes_tab,
            col_widths   = col_w,
            extra_styles = bold_cmds,
            col0_is_label = False,
            note = (f'Bandes de confiance +/-1.96/sqrt({n}) = {conf:.3f}. '
                    f'Prob. en gras = significative au seuil 5%.'),
        )

        plt.close('all')
        return [img, _spacer(0.2)] + tab_flowables

    except Exception as e:
        _warn(f'Erreur calcul correlogramme : {e}')
        plt.close('all')
        return []
