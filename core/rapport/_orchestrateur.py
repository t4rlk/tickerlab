# -*- coding: utf-8 -*-
"""
Orchestrateur du PDF unique TickerLab.

Construit la story Flowable section par section, gere le TOC,
la page de garde, le timing global et les warnings accumules.
"""
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

_log = logging.getLogger('tickerlab.rapport._orchestrateur')

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents

import tickerlab.core.rapport._helpers as _H
from tickerlab.core.rapport._helpers import (
    _set_theme, _safe_ticker, _WARNINGS_LOG,
    _h1, _h2, _p, _spacer, _pagebreak, _get_styles,
    MARGE_G, MARGE_D, MARGE_H, MARGE_B,
)
from tickerlab.core.rapport import _sections as SEC

warnings.filterwarnings('ignore')


# ── Document custom avec TOC + pied de page ───────────────────────────────────

class _RapportDocTemplate(BaseDocTemplate):
    """BaseDocTemplate avec hook afterFlowable pour TOC et numero de page."""

    def __init__(self, filename, **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        frame = Frame(
            MARGE_G, MARGE_B,
            A4[0] - MARGE_G - MARGE_D,
            A4[1] - MARGE_H - MARGE_B,
            id='normal',
        )
        self.addPageTemplates([
            PageTemplate(id='all', frames=frame, onPage=self._draw_footer),
        ])

    def afterFlowable(self, flowable):
        """Enregistre les Heading1/2 dans le TOC."""
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        text  = flowable.getPlainText()
        if style == 'Heading1':
            self.notify('TOCEntry', (0, text, self.page))
        elif style == 'Heading2':
            self.notify('TOCEntry', (1, text, self.page))

    @staticmethod
    def _draw_footer(canvas, doc):
        """Pied de page : label gauche + numero de page droite."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillGray(0.45)
        generated = time.strftime('%d/%m/%Y')
        canvas.drawString(
            MARGE_G, 0.8 * cm,
            f'Note technique automatisee — generee le {generated}',
        )
        canvas.drawRightString(
            A4[0] - MARGE_D, 0.8 * cm,
            f'Page {doc.page}',
        )
        canvas.restoreState()


# ── Page de garde ─────────────────────────────────────────────────────────────

def _page_garde(ticker: str, config: dict) -> list:
    styles = _get_styles()
    data   = config.get('data', {})
    start  = data.get('start_date', 'N/A')
    end    = data.get('end_date',   'N/A')
    freq   = data.get('frequency',  'daily')
    today  = time.strftime('%d/%m/%Y')

    return [
        Spacer(1, 4.5 * cm),
        Paragraph(
            'Note technique',
            styles['Titre'],
        ),
        Paragraph(
            'Analyse economique du risque de marche',
            styles['Titre'],
        ),
        Spacer(1, 0.8 * cm),
        Paragraph(f'<i>Ticker : {ticker}</i>',           styles['SousTitre']),
        Paragraph(f'Periode : {start} – {end} ({freq})', styles['SousTitre']),
        Spacer(1, 3.5 * cm),
        Paragraph(
            f'<font size="9">TickerLab — version generee le {today}</font>',
            styles['MetaInfo'],
        ),
        PageBreak(),
    ]


# ── Resume executif (page 2) ─────────────────────────────────────────────────

def _resume_executif(ticker: str, best, garch_final, df_var: pd.DataFrame,
                     df_bt: pd.DataFrame, config: dict,
                     spec_verdict: Optional[dict] = None) -> list:
    """
    Page de resume executif (5 lignes max) inseree entre la page de garde
    et la table des matieres.
    spec_verdict : sortie de verdict_specification(), ou None.
    """
    styles = _get_styles()
    today  = time.strftime('%d/%m/%Y')
    items  = []

    # ── Titre ────────────────────────────────────────────────────────────────
    items.append(Paragraph('Resume executif', styles['Heading1']))
    items.append(Spacer(1, 0.4 * cm))

    # ── Ligne 1 : Modele retenu ───────────────────────────────────────────────
    try:
        d = best if isinstance(best, dict) else best.to_dict()
        mod_str = (f"{d['modele']}({int(d['p'])},{int(d['o'])},{int(d['q'])}) "
                   f"[{d['dist']}]")
    except Exception:
        mod_str = 'N/A'
    items.append(Paragraph(f'<b>Modele retenu :</b> {mod_str}', styles['Body']))
    items.append(Spacer(1, 0.15 * cm))

    # ── Ligne 2 : VaR / TVaR 99% 1j ─────────────────────────────────────────
    try:
        var99  = df_var.loc['99%', 'VaR GARCH']
        tvar99 = df_var.loc['99%', 'TVaR GARCH']
        items.append(Paragraph(
            f'<b>VaR / TVaR 99% 1j :</b> {var99:.4f} % / {tvar99:.4f} %',
            styles['Body'],
        ))
    except Exception:
        items.append(Paragraph('<b>VaR / TVaR 99% 1j :</b> N/A', styles['Body']))
    items.append(Spacer(1, 0.15 * cm))

    # ── Ligne 3 : VaR 99% 10j Bale ───────────────────────────────────────────
    try:
        from tickerlab.core.var_engine import calculer_var_multi_horizon
        df_mh = calculer_var_multi_horizon(garch_final)
        var10j = float(df_mh.loc[10, 'VaR simulation'])
        var10j_str = f'{var10j:.4f} %'
    except Exception:
        var10j_str = 'voir §8.5'
    items.append(Paragraph(
        f'<b>VaR 99% 10j (Bale) :</b> {var10j_str} (simulation GARCH)',
        styles['Body'],
    ))
    items.append(Spacer(1, 0.15 * cm))

    # ── Ligne 4 : Backtesting ─────────────────────────────────────────────────
    try:
        garch_rows = df_bt[df_bt['Methode'] == 'GARCH dyn.']
        verdicts   = garch_rows['Verdict CC'].tolist()
        if all(v == 'OK' for v in verdicts):
            bt_verdict = 'OK — VaR dynamique valide (Kupiec + Christoffersen)'
        elif any(v == 'OK' for v in verdicts):
            bt_verdict = 'ATTENTION — validation partielle'
        else:
            bt_verdict = 'KO — violations anormales detectees'
    except Exception:
        bt_verdict = 'N/A'
    items.append(Paragraph(
        f'<b>Backtesting global :</b> {bt_verdict}',
        styles['Body'],
    ))
    items.append(Spacer(1, 0.15 * cm))

    # ── Ligne 5a : Specification (si verdict disponible) ─────────────────────
    if spec_verdict is not None:
        gravite = spec_verdict.get('gravite', 'aucune')
        spec_msg = spec_verdict.get('message', 'N/A')
        gravite_label = {'aucune': 'OK', 'mineure': 'ATTENTION', 'majeure': 'ALERTE'}.get(
            gravite, gravite.upper()
        )
        items.append(Paragraph(
            f'<b>Specification :</b> {gravite_label} — {spec_msg}',
            styles['Body'],
        ))
        items.append(Spacer(1, 0.15 * cm))

    # ── Ligne 5b : Recommandation (logique 5 cas Backtest x Spec) ────────────
    bt_ok   = 'OK' in bt_verdict and 'KO' not in bt_verdict
    bt_att  = 'ATTENTION' in bt_verdict
    bt_ko   = 'KO' in bt_verdict and 'OK' not in bt_verdict or bt_verdict == 'N/A'
    sp_ok   = spec_verdict is None or spec_verdict.get('gravite', 'aucune') == 'aucune'
    sp_min  = spec_verdict is not None and spec_verdict.get('gravite') == 'mineure'
    sp_maj  = spec_verdict is not None and spec_verdict.get('gravite') == 'majeure'

    if bt_ok and sp_ok:
        reco = 'Usage standard — modele calibre et valide (backtest + specification OK).'
    elif bt_ok and sp_min:
        reco = ('Utilisation possible avec surveillance — backtest OK mais '
                'blanchiment marginal ; recalibrer sur fenetre recente.')
    elif bt_ok and sp_maj:
        reco = ('ATTENTION : backtest valide mais specification defaillante — '
                'la VaR peut etre biaisee ; audit du modele recommande.')
    elif bt_att:
        reco = 'Surveillance accrue — validation partielle ; re-calibrer sur fenetre recente.'
    else:
        reco = 'Rejet — le modele sous-estime le risque ; ne pas utiliser en production.'
    items.append(Paragraph(f'<b>Recommandation :</b> {reco}', styles['Body']))
    items.append(Spacer(1, 0.3 * cm))

    items.append(Paragraph(
        f'<font size="8"><i>Genere le {today} — TickerLab</i></font>',
        styles['MetaInfo'],
    ))
    items.append(PageBreak())
    return items


# ── Table des matieres ────────────────────────────────────────────────────────

def _table_matieres() -> list:
    styles = _get_styles()
    toc = TableOfContents()
    toc.levelStyles = [styles['TOCEntry1'], styles['TOCEntry2']]
    return [
        Paragraph('<b>Table des matières</b>', styles['Heading1']),
        Spacer(1, 0.4 * cm),
        toc,
        PageBreak(),
    ]


# ── Helper : extraire le nom du modele retenu ─────────────────────────────────

def _nom_modele(df_garch: pd.DataFrame, best=None) -> str:
    """Deduit le nom du modele GARCH retenu."""
    if best is not None:
        try:
            d = best if isinstance(best, dict) else best.to_dict()
            return str(d.get('modele', 'GARCH'))
        except Exception as exc:
            _log.debug('_nom_modele : lecture best.to_dict() échouée, fallback : %s', exc)
    # Fallback : re-appliquer la logique de selectionner_meilleur sur df_garch
    try:
        from tickerlab.core.garch_selector import selectionner_meilleur as _sm
        best_row, *_ = _sm(df_garch)
        return str(best_row.get('modele', 'GARCH'))
    except Exception as exc:
        _log.debug('_nom_modele : selectionner_meilleur échoué, retour GARCH : %s', exc)
    try:
        return str(df_garch.iloc[0].get('modele', 'GARCH'))
    except Exception:
        return 'GARCH'


# ── Fonction principale ───────────────────────────────────────────────────────

def generer_pdf_unique(
    prix: pd.Series,
    rendements: pd.Series,
    arima_result: dict,
    garch_final,
    df_garch: pd.DataFrame,
    df_var: pd.DataFrame,
    df_vol: pd.DataFrame,
    df_bt: Optional[pd.DataFrame],
    T_train: int,
    T_eff_dyn: int,
    config: dict,
    ticker: str,
    t_debut: Optional[float] = None,
    best=None,
    trace_garch: Optional[list] = None,
    df_violations: Optional[pd.DataFrame] = None,
    df_params_drift: Optional[pd.DataFrame] = None,
    stats_rolling: Optional[dict] = None,
    igarch_diagnostic: Optional[dict] = None,
    component_garch_result: Optional[dict] = None,
    fhs_result: Optional[dict] = None,
    dm_gk_result: Optional[dict] = None,
    prix_stats=None,
) -> Path:
    """
    Genere le PDF unique TickerLab dans le dossier resultats/.

    Chaque section est wrappee en try/except : un echec n'interrompt pas
    les sections suivantes (un placeholder est insere a la place).

    Returns
    -------
    Path
        Chemin absolu du PDF genere.
    """
    if t_debut is None:
        t_debut = time.time()

    # 1. Reset warnings, theme, verbosity
    _WARNINGS_LOG.clear()
    theme = config.get('rapport', {}).get('theme', 'academique')
    _set_theme(theme)
    SEC._VERBOSE = False   # silencieux en prod — True pour debug

    # 2. Chemin de sortie
    dossier = Path(config['output']['dossier_resultats'])
    dossier.mkdir(parents=True, exist_ok=True)
    pdf_path = dossier / f'Rapport_{_safe_ticker(ticker)}.pdf'

    # 3. Modele GARCH retenu
    modele_nom = _nom_modele(df_garch, best)

    # Prix a la frequence d'analyse (pour stats sections 1.2/2.x/3.x)
    _prix_stats = prix_stats if prix_stats is not None else prix

    # Verdict specification
    _spec_verdict = None
    if best is not None:
        try:
            from tickerlab.core.garch_selector import verdict_specification as _vs
            _best_d = best if isinstance(best, dict) else best.to_dict()
            _spec_verdict = _vs(_best_d, df_garch, config)
        except Exception as _ev:
            _H._warn(f'verdict_specification : {_ev}')

    # 4. df_bt : garantir un DataFrame meme si absent
    if df_bt is None:
        df_bt = pd.DataFrame(columns=[
            'Methode', 'Niveau', 'N viol.', 'Taux obs.', 'Taux theo.',
            'LR_UC', 'p_UC', 'LR_IND', 'p_IND', 'LR_CC', 'p_CC',
            'Verdict UC', 'Verdict CC',
        ])

    # 5. Document ReportLab
    doc = _RapportDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGE_G,  rightMargin=MARGE_D,
        topMargin=MARGE_H,   bottomMargin=MARGE_B,
        title=f'Note technique — Analyse economique du risque de marche — {ticker}',
        author='TickerLab',
    )

    # 6. Construction de la story
    story = []

    # Page de garde
    try:
        story.extend(_page_garde(ticker, config))
    except Exception as e:
        _H._warn(f'Page de garde : {e}')
        story.append(PageBreak())

    # Resume executif (page 2)
    try:
        story.extend(_resume_executif(ticker, best, garch_final, df_var, df_bt, config,
                                      spec_verdict=_spec_verdict))
    except Exception as e:
        _H._warn(f'Resume executif : {e}')
        story.append(PageBreak())

    # Table des matieres
    try:
        story.extend(_table_matieres())
    except Exception as e:
        _H._warn(f'Table des matieres : {e}')

    # Helper isolation : remplace une section defaillante par un placeholder
    def _safe(label, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except Exception as e:
            _H._warn(f'Section {label} : {type(e).__name__} -- {e}')
            return [
                _h1(f'{label} — Section indisponible'),
                _p(f'Échec de la génération : {type(e).__name__} — {e}. '
                   'Consulter la section 11 (Journal d\'exécution).'),
                _pagebreak(),
            ]

    story.extend(_safe('1', SEC.section_1, prix, config, ticker, prix_stats=_prix_stats))
    story.extend(_safe('2', SEC.section_2, _prix_stats, config, ticker))
    story.extend(_safe('3', SEC.section_3, _prix_stats, rendements, config))
    story.extend(_safe('4', SEC.section_4, rendements, config, ticker))
    story.extend(_safe('5', SEC.section_5, prix, arima_result, config, rendements))
    story.extend(_safe('6', SEC.section_6, rendements, df_garch,
                        garch_final, modele_nom, config,
                        igarch_diagnostic=igarch_diagnostic or {},
                        component_garch_result=component_garch_result or {}))
    story.extend(_safe('7', SEC.section_7, rendements, garch_final,
                        modele_nom, config, ticker))
    story.extend(_safe('8', SEC.section_8, rendements, garch_final,
                        df_var, df_bt, config, ticker,
                        df_violations=(df_violations if df_violations is not None
                                       else pd.DataFrame()),
                        df_params_drift=(df_params_drift if df_params_drift is not None
                                         else pd.DataFrame()),
                        stats_rolling=(stats_rolling if stats_rolling is not None
                                       else {}),
                        fhs_result=fhs_result or {},
                        dm_gk_result=dm_gk_result or {}))
    story.extend(_safe('9', SEC.section_9, rendements, garch_final,
                        df_var, config, ticker))
    story.extend(_safe('10', SEC.section_10, rendements, garch_final,
                        modele_nom, df_var, df_bt, config, ticker))

    # Section 11 : meta enrichie
    elapsed = time.time() - t_debut
    try:
        n_arima = len(arima_result.get('df', []))
    except Exception:
        n_arima = 0
    meta = {
        'ticker':          ticker,
        'n_obs':           int(getattr(garch_final, 'nobs', len(rendements))),
        'modele_garch':    modele_nom,
        'dist_garch':      str(df_garch.iloc[0].get('dist', '')) if len(df_garch) else '',
        'arima_spec':      (f"ARIMA({arima_result.get('p_opt', 0)},"
                            f"{arima_result.get('d_opt', 0)},"
                            f"{arima_result.get('q_opt', 0)})"),
        'n_arima_evalues': n_arima,
        'n_garch_evalues': len(df_garch),
        'elapsed_s':       elapsed,
        'warnings_list':   list(_WARNINGS_LOG),
        'T_train':         T_train,
        'T_eff_dyn':       T_eff_dyn,
    }
    story.extend(_safe('11', SEC.section_11, meta, config))
    story.extend(_safe('annexes', SEC.section_annexes, config,
                       trace_garch=trace_garch or [],
                       rendements=rendements,
                       prix=prix,
                       arima_result=arima_result,
                       garch_final=garch_final))

    # 7. Compilation (multiBuild requis pour le TOC)
    _log.info('  [PDF] Compilation %s...', pdf_path.name)
    doc.multiBuild(story)

    # 8. Bilan
    elapsed_final = time.time() - t_debut
    n_warn = len(_WARNINGS_LOG)
    _log.info('  [PDF] OK -> %s', pdf_path)
    _log.info('  [PDF] Duree : %.1fs | Warnings : %d', elapsed_final, n_warn)
    if n_warn:
        for w in _WARNINGS_LOG:
            _log.warning('    [WARN] %s', w)

    return pdf_path
