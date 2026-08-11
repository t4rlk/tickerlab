"""Export des DataFrames en code LaTeX prêt à compiler avec pdflatex/latexmk."""
import logging
import re
import pandas as pd
from pathlib import Path

_log = logging.getLogger('tickerlab.latex_export')


def _echapper_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne une copie du DataFrame dont les noms de colonnes ont les
    caractères spéciaux LaTeX échappés (_, %, &, #, $, ^, ~, {, })."""
    def _esc(s: str) -> str:
        s = str(s)
        s = s.replace('_', r'\_')
        s = s.replace('%', r'\%')
        s = s.replace('&', r'\&')
        s = s.replace('#', r'\#')
        s = s.replace('$', r'\$')
        s = s.replace('^', r'\^{}')
        return s
    df2 = df.copy()
    df2.columns = [_esc(c) for c in df2.columns]
    if df2.index.name:
        df2.index.name = _esc(df2.index.name)
    return df2


def _echapper_valeurs(df: pd.DataFrame) -> pd.DataFrame:
    """Échappe % dans les cellules string et dans l'index."""
    def _esc_val(v):
        s = str(v)
        s = s.replace('%', r'\%')
        s = s.replace('_', r'\_')
        return s

    df2 = df.copy()
    for col in df2.select_dtypes(include='object').columns:
        df2[col] = df2[col].apply(
            lambda v: _esc_val(v) if isinstance(v, str) else v
        )
    # Echapper aussi les valeurs de l'index
    if isinstance(df2.index, pd.MultiIndex):
        df2.index = pd.MultiIndex.from_tuples(
            [tuple(_esc_val(l) for l in idx) for idx in df2.index],
            names=df2.index.names
        )
    else:
        df2.index = [_esc_val(v) for v in df2.index]
    return df2


def df_to_latex(df: pd.DataFrame, caption: str, label: str,
                output_path: Path = None, **kwargs) -> str:
    """Convertit un DataFrame en tableau LaTeX (style booktabs).

    Les underscores et % dans les noms de colonnes sont automatiquement
    échappés pour éviter les erreurs de compilation.
    """
    df_safe = _echapper_valeurs(_echapper_colonnes(df))
    col_fmt = 'l' + 'r' * len(df_safe.columns)
    latex = df_safe.to_latex(
        caption=caption, label=label,
        position='htbp', escape=False,
        column_format=col_fmt, **kwargs
    )
    # Remplacer \hline par les commandes booktabs
    hline_count = latex.count('\\hline')
    if hline_count >= 2:
        latex = latex.replace('\\hline', '\\toprule', 1)
        latex = latex.replace('\\hline', '\\midrule', hline_count - 1)
        last = latex.rfind('\\midrule')
        latex = latex[:last] + '\\bottomrule' + latex[last + len('\\midrule'):]
    # Envelopper tabular dans adjustbox pour eviter les debordements
    latex = latex.replace(
        r'\begin{tabular}',
        r'\begin{adjustbox}{max width=\linewidth}' + '\n' + r'\begin{tabular}'
    )
    latex = latex.replace(
        r'\end{tabular}',
        r'\end{tabular}' + '\n' + r'\end{adjustbox}'
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(latex, encoding='utf-8')
        _log.info('  OK LaTeX -> %s', output_path)
    return latex


# ── Helpers tableaux de sélection ────────────────────────────────────────────

def _bold_row(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Enveloppe toutes les valeurs de la ligne idx dans \\textbf{…}."""
    df = df.copy()
    for col in df.columns:
        v = df.at[idx, col]
        df.at[idx, col] = f'\\textbf{{{v}}}'
    return df


def _exporter_tab_arima(df_arima: pd.DataFrame, out_path: Path,
                        arima_result: dict = None):
    """
    Exporte tab_arima.tex avec colonne Complexité, tri hiérarchique
    (tous_sig DESC, complexite ASC, AIC ASC) et ligne retenue en gras.
    """
    df = df_arima.copy()
    df['complexite'] = (df['p'] + df['q']).astype(int)
    df = (df.sort_values(
              ['tous_sig', 'complexite', 'AIC'],
              ascending=[False, True, True])
          .head(10)
          .reset_index(drop=True))

    # Construction du DataFrame d'affichage (tout en strings)
    disp = pd.DataFrame({
        'p':          df['p'].astype(int).astype(str).values,
        'd':          df['d'].astype(int).astype(str).values,
        'q':          df['q'].astype(int).astype(str).values,
        'AIC':        [f'{v:.6f}' for v in df['AIC']],
        'BIC':        [f'{v:.6f}' for v in df['BIC']],
        'Sig.':       df['tous_sig'].map({True: 'Oui', False: 'Non'}).values,
        'Complexite': df['complexite'].astype(str).values,
    })

    # Identification et mise en gras de la ligne retenue
    caption_base = r'S\'election ARIMA --- Top 10 (tous\_sig $\downarrow$, complexit\'e $\uparrow$, AIC $\uparrow$)'
    footnote = ''
    if arima_result:
        p_opt = arima_result.get('p_opt', -1)
        d_opt = arima_result.get('d_opt', -1)
        q_opt = arima_result.get('q_opt', -1)
        motif = arima_result.get('motif_selection', '?')
        for i, row in df.iterrows():
            if int(row['p']) == p_opt and int(row['d']) == d_opt and int(row['q']) == q_opt:
                disp = _bold_row(disp, i)
                break
        footnote = (fr'\par\footnotesize{{Mod\`ele retenu : '
                    fr'ARIMA({p_opt},{d_opt},{q_opt}) --- motif : {motif}.}}')

    df_to_latex(disp, caption_base + footnote, 'tab:arima', out_path, index=False)


def _exporter_tab_garch(df_garch: pd.DataFrame, out_path: Path,
                        garch_best=None, garch_motif: str = None):
    """
    Exporte tab_garch.tex avec colonne Complexité (p+o+q), tri hiérarchique
    (tous_sig DESC, complexite ASC, AIC ASC) et ligne retenue en gras.
    """
    df = df_garch.copy()
    col_sig = ('tous_sig_vol' if 'tous_sig_vol' in df.columns else 'tous_sig')
    df['complexite'] = (df['p'].astype(int)
                        + df['o'].astype(int)
                        + df['q'].astype(int))
    df = (df.sort_values(
              [col_sig, 'complexite', 'AIC'],
              ascending=[False, True, True])
          .head(10)
          .reset_index(drop=True))

    cols_base = ['modele', 'p', 'o', 'q', 'dist']
    cols_base = [c for c in cols_base if c in df.columns]

    disp = pd.DataFrame(index=range(len(df)))
    for c in cols_base:
        disp[c] = df[c].astype(str).values
    disp['AIC']        = [f'{v:.2f}' for v in df['AIC']]
    disp['AIC/n']      = [f'{v:.6f}' for v in df.get('AIC_norm', df['AIC'])]
    disp['Sig.']       = df[col_sig].map({True: 'Oui', False: 'Non'}).values
    disp['Complexite'] = df['complexite'].astype(str).values

    caption_base = (r'S\'election GARCH $\times$ Distribution --- Top 10 '
                    r'(tous\_sig $\downarrow$, complexit\'e $\uparrow$, AIC $\uparrow$)')
    footnote = ''
    if garch_best is not None:
        bd = garch_best if isinstance(garch_best, dict) else garch_best.to_dict()
        m_nom = bd.get('modele', '?')
        p_g   = int(bd.get('p', 0))
        o_g   = int(bd.get('o', 0))
        q_g   = int(bd.get('q', 0))
        dist  = bd.get('dist', '?')
        motif = garch_motif or '?'
        for i, row in df.iterrows():
            if (str(row.get('modele')) == m_nom
                    and int(row['p']) == p_g
                    and int(row['o']) == o_g
                    and int(row['q']) == q_g
                    and str(row.get('dist')) == dist):
                disp = _bold_row(disp, i)
                break
        footnote = (fr'\par\footnotesize{{Mod\`ele retenu : '
                    fr'{m_nom}({p_g},{o_g},{q_g}) [{dist}] --- motif : {motif}.}}')

    df_to_latex(disp, caption_base + footnote, 'tab:garch', out_path, index=False)


# ── Point d'entrée principal ──────────────────────────────────────────────────

def exporter_tous_tableaux(dossier_out: str, df_arima=None, df_garch=None,
                            df_var=None, df_bt=None, df_vol=None,
                            arima_result=None, garch_best=None, garch_motif=None):
    """Exporte tous les tableaux clés en .tex pour compilation locale."""
    out = Path(dossier_out) / 'latex'
    if df_arima is not None:
        _exporter_tab_arima(df_arima, out / 'tab_arima.tex',
                            arima_result=arima_result)
    if df_garch is not None:
        _exporter_tab_garch(df_garch, out / 'tab_garch.tex',
                            garch_best=garch_best, garch_motif=garch_motif)
    if df_var is not None:
        df_to_latex(df_var,
                    'VaR et TVaR --- Toutes méthodes $\\times$ Niveaux', 'tab:var',
                    out / 'tab_var.tex')
    if df_bt is not None:
        df_to_latex(df_bt,
                    'Backtesting OOS --- Kupiec, Christoffersen', 'tab:backtest',
                    out / 'tab_backtest.tex', index=False)
