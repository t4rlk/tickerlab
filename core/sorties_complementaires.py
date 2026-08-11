# -*- coding: utf-8 -*-
"""Sorties complémentaires : graphiques EViews + tableaux GARCH benchmarks,
comparaisons de distributions, effet de levier, ratios TVaR/VaR, validation résidus."""
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

_log = logging.getLogger('tickerlab.sorties_complementaires')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy import stats as sp_stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import re

warnings.filterwarnings('ignore')

_DIST_LABEL = {'normal': 'Normale', 't': r'Student-$t$', 'ged': 'GED'}
_DIST_LIST  = ['normal', 't', 'ged']


# ─── Helpers I/O ──────────────────────────────────────────────────────────────

def _mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def _sauvegarder(fig, nom_base: str, dossier, dpi: int = 300):
    _mkdir(dossier)
    for ext in ('png', 'pdf'):
        chemin = Path(dossier) / f'{nom_base}.{ext}'
        fig.savefig(chemin, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        _log.info('  OK -> %s', chemin)
    plt.close(fig)


def _ecrire(contenu: str, nom: str, dossier):
    _mkdir(dossier)
    chemin = Path(dossier) / nom
    chemin.write_text(contenu, encoding='utf-8')
    _log.info('  OK -> %s', chemin)


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def _signif(p: float) -> str:
    if p < 0.01:  return '***'
    if p < 0.05:  return '**'
    if p < 0.10:  return '*'
    return ''


# ─── Graphiques de séries ─────────────────────────────────────────────────────

def tracer_serie(serie, titre: str, ylabel: str, nom_base: str, dossier):
    s = _to_series(serie)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(s.index, s.values, color='steelblue', linewidth=0.8, alpha=0.9)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_title(titre, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, alpha=0.25, linestyle=':')
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    fig.tight_layout()
    _sauvegarder(fig, nom_base, dossier)


# ─── Histogramme style EViews ─────────────────────────────────────────────────

def tracer_histogramme(serie, nom_serie: str, nom_base: str, dossier):
    s = _to_series(serie).dropna()
    n = len(s)
    moyenne = float(s.mean())
    mediane = float(s.median())
    maxi    = float(s.max())
    mini    = float(s.min())
    ecart   = float(s.std())
    skew    = float(sp_stats.skew(s.values))
    kurt    = float(sp_stats.kurtosis(s.values, fisher=False))
    jb, jb_p, _, _ = jarque_bera(s.values)

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], figure=fig)

    ax  = fig.add_subplot(gs[0])
    n_bins = min(30, int(np.sqrt(n)))
    ax.hist(s, bins=n_bins, density=True, color='steelblue',
            edgecolor='white', linewidth=0.4, alpha=0.85)
    x_range = np.linspace(mini - ecart, maxi + ecart, 300)
    ax.plot(x_range, sp_stats.norm.pdf(x_range, moyenne, ecart),
            color='red', linewidth=1.4)
    ax.set_title(nom_serie, fontsize=10, fontweight='bold')
    ax.set_ylabel('Densité', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle=':')
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    lignes = [
        ('Mean',         f'{moyenne:.6f}'),
        ('Median',       f'{mediane:.6f}'),
        ('Maximum',      f'{maxi:.6f}'),
        ('Minimum',      f'{mini:.6f}'),
        ('Std. Dev.',    f'{ecart:.6f}'),
        ('Skewness',     f'{skew:.6f}'),
        ('Kurtosis',     f'{kurt:.6f}'),
        ('',             ''),
        ('Jarque-Bera',  f'{jb:.4f}'),
        ('Probability',  f'{jb_p:.4f}'),
        ('',             ''),
        ('Observations', f'{n}'),
    ]
    y0, dy = 0.97, 0.075
    box = FancyBboxPatch((0, 0.02), 1.0, 0.96, boxstyle='round,pad=0.01',
                         linewidth=0.8, edgecolor='#555555', facecolor='#f9f9f9',
                         transform=ax2.transAxes, zorder=0)
    ax2.add_patch(box)
    for i, (lab, val) in enumerate(lignes):
        y = y0 - i * dy
        ax2.text(0.04, y, lab, transform=ax2.transAxes,
                 fontsize=8, ha='left', va='top', color='#333333')
        ax2.text(0.96, y, val, transform=ax2.transAxes,
                 fontsize=8, ha='right', va='top', color='#111111', fontweight='bold')

    fig.tight_layout(pad=1.5)
    _sauvegarder(fig, nom_base, dossier)


# ─── Corrélogramme graphique style EViews ─────────────────────────────────────

def tracer_correlogramme(serie, lags: int, titre: str, nom_base: str, dossier):
    s  = _to_series(serie).dropna()
    n  = len(s)
    ci = 1.96 / np.sqrt(n)
    ac_vals  = acf(s, nlags=lags, fft=True)[1:]
    pac_vals = pacf(s, nlags=lags)[1:]
    lag_arr  = np.arange(1, lags + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    for ax, vals, label, color in zip(
            axes,
            [ac_vals, pac_vals],
            ['Autocorrélation (AC)', 'Autocorrélation partielle (PAC)'],
            ['steelblue', 'darkorange']):
        ax.bar(lag_arr, vals, color=color, alpha=0.8, width=0.5)
        ax.axhline(+ci, color='red', linestyle='--', linewidth=0.9, alpha=0.8)
        ax.axhline(-ci, color='red', linestyle='--', linewidth=0.9, alpha=0.8)
        ax.axhline(0,   color='black', linewidth=0.5)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis='y', alpha=0.2, linestyle=':')
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    axes[1].set_xlabel('Lag', fontsize=8)
    axes[1].set_xticks(lag_arr[::2])
    fig.suptitle(titre, fontsize=10, fontweight='bold', y=1.02)
    fig.tight_layout()
    _sauvegarder(fig, nom_base, dossier)


# ─── Tests de stationnarité ───────────────────────────────────────────────────

def _adf(serie, reg='c'):
    res = adfuller(serie.dropna(), regression=reg, autolag='AIC')
    return res[0], res[1], int(res[2])


def _pp(serie, trend='c'):
    try:
        from arch.unitroot import PhillipsPerron
        pp = PhillipsPerron(serie.dropna(), trend=trend)
        return pp.stat, pp.pvalue
    except Exception:
        return float('nan'), float('nan')


def _kpss_test(serie, reg='c'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stat, pv, _, _ = kpss(serie.dropna(), regression=reg, nlags='auto')
    return stat, pv


def tableau_tests_stationnarite(serie, nom_serie: str, nom: str, dossier):
    specs = [
        ('Sans constante ni tendance', 'n',  'c'),
        ('Avec constante',             'c',  'c'),
        ('Avec constante et tendance', 'ct', 'ct'),
    ]
    l_adf, l_pp, l_kpss = [], [], []
    for label, reg_adf, reg_kpss in specs:
        t_a, p_a, lag = _adf(serie, reg_adf)
        t_p, p_p      = _pp(serie, trend='n' if reg_adf == 'n' else reg_adf)
        t_k, p_k      = _kpss_test(serie, reg_kpss)
        l_adf.append(f'  {label} & {t_a:.4f} & {p_a:.4f}{_signif(p_a)} & {lag} \\\\')
        l_pp.append(f'  {label} & {t_p:.4f} & {p_p:.4f}{_signif(p_p)} \\\\')
        l_kpss.append(f'  {label} & {t_k:.4f} & {p_k:.4f} \\\\')

    label_id = nom.replace('.tex', '').replace('tab_', '')
    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Tests de racine unitaire --- ' + nom_serie + '}\n'
        r'\label{tab:statio_' + label_id + '}\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{lrrr}' + '\n'
        r'\toprule' + '\n'
        r'\multicolumn{4}{c}{\textbf{Test ADF (Augmented Dickey--Fuller)}} \\' + '\n'
        r'\midrule' + '\n'
        r'Spécification & Stat. & $p$-val. & Retards \\' + '\n'
        r'\midrule' + '\n'
        + '\n'.join(l_adf) + '\n'
        r'\midrule' + '\n'
        r'\multicolumn{4}{c}{\textbf{Test PP (Phillips--Perron)}} \\' + '\n'
        r'\midrule' + '\n'
        r'Spécification & Stat. & $p$-val. & \\' + '\n'
        r'\midrule' + '\n'
        + '\n'.join(l_pp) + '\n'
        r'\midrule' + '\n'
        r'\multicolumn{4}{c}{\textbf{Test KPSS ($H_0$ : stationnaire)}} \\' + '\n'
        r'\midrule' + '\n'
        r'Spécification & Stat. & $p$-val. & \\' + '\n'
        r'\midrule' + '\n'
        + '\n'.join(l_kpss) + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{*** $p<0{,}01$ ; ** $p<0{,}05$ ; * $p<0{,}10$.}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, nom, dossier)


def tableau_synthese_stationnarite(serie_niv, serie_diff, nom: str, dossier):
    def _row(s, label, d_order):
        t, p, _ = _adf(s, 'c')
        tp, pp  = _pp(s, 'c')
        tk, pk  = _kpss_test(s, 'c')
        adf_c  = 'NS' if p  > 0.05 else 'S'
        pp_c   = 'NS' if pp > 0.05 else 'S'
        kpss_c = 'NS' if pk < 0.05 else 'S'
        return (f'  {label} & {t:.4f} ({adf_c}) & {tp:.4f} ({pp_c}) '
                f'& {tk:.4f} ({kpss_c}) & $I({d_order})$ \\\\')

    row1 = _row(serie_niv,  r'LBRENT (niveau)',           1)
    row2 = _row(serie_diff, r'DLBRENT (1\`ere diff.)',    0)
    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Synthèse des tests de stationnarité}' + '\n'
        r'\label{tab:statio_synthese}' + '\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{lllll}' + '\n'
        r'\toprule' + '\n'
        r'Série & ADF & PP & KPSS & Ordre \\' + '\n'
        r'\midrule' + '\n'
        + row1 + '\n' + row2 + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{S = Stationnaire, NS = Non stationnaire au seuil 5\,\%.}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, nom, dossier)


# ─── ARIMA ────────────────────────────────────────────────────────────────────

def tableau_arima_estim(rendements, p: int, d: int, q: int, nom: str, dossier):
    s = _to_series(rendements).dropna()
    try:
        res = ARIMA(s, order=(p, d, q)).fit(method_kwargs={'warn_convergence': False})
    except Exception:
        res = ARIMA(s, order=(p, d, q)).fit()

    lignes = []
    for name, coef, se, t_v, pv in zip(
            res.params.index, res.params, res.bse, res.tvalues, res.pvalues):
        lignes.append(
            f'  {name} & {coef:.6f} & {se:.6f} & {t_v:.4f} & {pv:.4f}{_signif(pv)} \\\\'
        )

    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Estimation ARIMA(' + f'{p},{d},{q}' + r')}' + '\n'
        r'\label{tab:arima_estim}' + '\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{lrrrr}' + '\n'
        r'\toprule' + '\n'
        r'Paramètre & Coeff. & Std. Err. & $t$ & $p$-valeur \\' + '\n'
        r'\midrule' + '\n'
        + '\n'.join(lignes) + '\n'
        r'\midrule' + '\n'
        r'\multicolumn{5}{l}{\footnotesize AIC = ' + f'{res.aic:.4f}'
        + r' \quad BIC = ' + f'{res.bic:.4f}' + r'} \\' + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{*** $p<0{,}01$ ; ** $p<0{,}05$ ; * $p<0{,}10$.}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, nom, dossier)


# ─── ARCH-LM ─────────────────────────────────────────────────────────────────

def tableau_arch_lm(resid: pd.Series, lags_list: list, nom: str, dossier):
    from statsmodels.stats.diagnostic import het_arch
    lignes = []
    for lag in lags_list:
        try:
            lm_s, lm_p, f_s, f_p = het_arch(resid.dropna(), nlags=lag)
            lignes.append(
                f'  {lag} & {lm_s:.4f} & {lm_p:.4f}{_signif(lm_p)} '
                f'& {f_s:.4f} & {f_p:.4f} \\\\'
            )
        except Exception:
            lignes.append(f'  {lag} & --- & --- & --- & --- \\\\')

    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Test ARCH-LM sur les résidus ARIMA}' + '\n'
        r'\label{tab:arch_lm}' + '\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{lrrrr}' + '\n'
        r'\toprule' + '\n'
        r'Retards & LM stat. & $p$-val. (LM) & $F$ stat. & $p$-val. ($F$) \\' + '\n'
        r'\midrule' + '\n'
        + '\n'.join(lignes) + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{$H_0$ : pas d\'effet ARCH. *** $p<0{,}01$ ; ** $p<0{,}05$ ; * $p<0{,}10$.}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, nom, dossier)


# ─── GARCH benchmarks (tableaux individuels) ──────────────────────────────────

def _fit_garch(serie: pd.Series, vol: str, p: int, o: int, q: int, dist: str):
    am = arch_model(serie.dropna() * 100, vol=vol, p=p, o=o, q=q,
                    dist=dist, mean='Constant', rescale=False)
    return am.fit(disp='off', show_warning=False)


def _latex_garch_table_3dist(resultats: dict, titre: str, label: str) -> str:
    all_params = set()
    for r in resultats.values():
        if r is not None:
            all_params.update(r.params.index.tolist())

    ordre = ['Const', 'mu', 'omega',
             'alpha[1]', 'alpha[2]', 'alpha[3]', 'alpha[4]',
             'beta[1]', 'gamma[1]', 'nu', 'lambda', 'eta']
    params_ord = [p for p in ordre if p in all_params]
    params_ord += [p for p in sorted(all_params) if p not in params_ord]

    n_dist = sum(1 for d in _DIST_LIST if d in resultats and resultats[d] is not None)
    col_fmt = 'l' + 'rr' * n_dist

    mc_parts = ' & '.join(
        r'\multicolumn{2}{c}{' + _DIST_LABEL[d] + '}'
        for d in _DIST_LIST if d in resultats and resultats[d] is not None
    )
    sub_hdr = ' & '.join(
        r'Coeff. & Std.Err.'
        for d in _DIST_LIST if d in resultats and resultats[d] is not None
    )
    cmidrule = ' '.join(
        r'\cmidrule(lr){' + f'{2+2*i}-{3+2*i}' + '}'
        for i, d in enumerate(d for d in _DIST_LIST
                               if d in resultats and resultats[d] is not None)
    )

    param_rows = []
    for param in params_ord:
        row = [f'  {param}']
        for d in _DIST_LIST:
            if d not in resultats or resultats[d] is None:
                continue
            r = resultats[d]
            if param not in r.params.index:
                row += ['---', '---']
            else:
                coef = r.params[param]
                try:
                    se  = r.std_err[param]
                    pv  = r.pvalues[param]
                    row += [f'{coef:.6f}{_signif(pv)}', f'({se:.6f})']
                except Exception:
                    row += [f'{coef:.6f}', '(---)']
        param_rows.append(' & '.join(row) + ' \\\\')

    crit_rows = []
    for crit, attr in [('AIC', 'aic'), ('BIC', 'bic'),
                       ('Log-vrais.', 'loglikelihood')]:
        row = [f'  {crit}']
        for d in _DIST_LIST:
            if d not in resultats or resultats[d] is None:
                continue
            r = resultats[d]
            val = getattr(r, attr)
            row += [f'{val:.4f}', '']
        crit_rows.append(' & '.join(row) + ' \\\\')

    return (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{' + titre + '}\n'
        r'\label{' + label + '}\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{' + col_fmt + '}\n'
        r'\toprule' + '\n'
        + mc_parts + r' \\' + '\n'
        + cmidrule + '\n'
        r'Paramètre & ' + sub_hdr + r' \\' + '\n'
        r'\midrule' + '\n'
        + '\n'.join(param_rows) + '\n'
        r'\midrule' + '\n'
        + '\n'.join(crit_rows) + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{*** $p<0{,}01$ ; ** $p<0{,}05$ ; * $p<0{,}10$. Std.Err. entre parenthèses.}' + '\n'
        r'\end{table}' + '\n'
    )


def _generer_table_famille(serie, vol: str, p: int, o: int, q: int,
                           titre: str, label: str, nom: str, dossier):
    resultats = {}
    for dist in _DIST_LIST:
        try:
            resultats[dist] = _fit_garch(serie, vol, p, o, q, dist)
            _log.info('    %s(%d,%d,%d) [%s] OK', vol, p, o, q, dist)
        except Exception as e:
            _log.warning('    %s(%d,%d,%d) [%s] ERREUR: %s', vol, p, o, q, dist, e)
            resultats[dist] = None
    contenu = _latex_garch_table_3dist(resultats, titre, label)
    _ecrire(contenu, nom, dossier)


def tableau_arch4(serie, nom: str, dossier):
    _generer_table_famille(serie, 'ARCH', 4, 0, 0,
                           r'ARCH(4) --- Normale, Student-$t$, GED',
                           'tab:arch4', nom, dossier)


def tableau_garch11(serie, nom: str, dossier):
    _generer_table_famille(serie, 'Garch', 1, 0, 1,
                           r'GARCH(1,1) --- Normale, Student-$t$, GED',
                           'tab:garch11', nom, dossier)


def tableau_gjr11(serie, nom: str, dossier):
    _generer_table_famille(serie, 'GARCH', 1, 1, 1,
                           r'GJR-GARCH(1,1) --- Normale, Student-$t$, GED',
                           'tab:gjr11', nom, dossier)


def tableau_egarch11(serie, nom: str, dossier):
    _generer_table_famille(serie, 'EGARCH', 1, 1, 1,
                           r'EGARCH(1,1) --- Normale, Student-$t$, GED',
                           'tab:egarch11', nom, dossier)


# ─── Priorité 3 : comparaison des distributions pour un modèle donné ──────────

def _best_ordre_par_modele(df_garch: pd.DataFrame, nom_modele: str):
    sub = df_garch[df_garch['modele'] == nom_modele]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {'vol': row['vol'], 'p': int(row['p']), 'o': int(row['o']), 'q': int(row['q'])}


def tableau_comparaison_distributions(serie: pd.Series, df_garch: pd.DataFrame,
                                      best_dict: dict, dossier):
    """Génère 3 tables de comparaison Normal/Student/GED :
    - tab_garch_comparaison_dist.tex  (meilleur modèle global)
    - tab_tgarch_comparaison_dist.tex (meilleur GJR-GARCH)
    - tab_egarch_comparaison_dist.tex (meilleur EGARCH)
    """
    def _build(spec_dict, nom_humain, nom_fichier, label):
        if spec_dict is None:
            _log.info('    Aucun modele %s dans la grille, ignore.', nom_humain)
            return
        vol = spec_dict['vol']
        p, o, q = spec_dict['p'], spec_dict['o'], spec_dict['q']
        modele_str = f'{nom_humain}({p},{q})'
        titre = f'Comparaison distributions --- {modele_str}'
        _generer_table_famille(serie, vol, p, o, q, titre, label, nom_fichier, dossier)

    # Meilleur modèle global
    m_best = {k: best_dict[k] for k in ('vol', 'p', 'o', 'q')}
    m_best['p'] = int(m_best['p'])
    m_best['o'] = int(m_best['o'])
    m_best['q'] = int(m_best['q'])
    nom_best = best_dict.get('modele', 'GARCH')
    _build(m_best,
           nom_best,
           'tab_garch_comparaison_dist.tex',
           'tab:garch_comp_dist')

    # Meilleur GJR-GARCH
    gjr = _best_ordre_par_modele(df_garch, 'GJR-GARCH')
    _build(gjr, 'GJR-GARCH', 'tab_tgarch_comparaison_dist.tex', 'tab:tgarch_comp_dist')

    # Meilleur EGARCH
    egarch = _best_ordre_par_modele(df_garch, 'EGARCH')
    _build(egarch, 'EGARCH', 'tab_egarch_comparaison_dist.tex', 'tab:egarch_comp_dist')


# ─── Priorité 5 : effet de levier ─────────────────────────────────────────────

def tableau_effet_levier(garch_final, best_dict: dict, nom: str, dossier):
    """Tableau de l'effet de levier pour EGARCH ou GJR-GARCH."""
    vol      = best_dict.get('vol', '')
    modele   = best_dict.get('modele', '')
    params   = garch_final.params
    pvalues  = garch_final.pvalues

    if vol == 'EGARCH':
        alpha = float(params.get('alpha[1]', float('nan')))
        gamma = float(params.get('gamma[1]', float('nan')))
        beta  = float(params.get('beta[1]',  float('nan')))
        p_gam = float(pvalues.get('gamma[1]', float('nan')))

        # Impact sur ln(σ²) à z=+1 vs z=-1 (simplifié)
        impact_pos = alpha + gamma
        impact_neg = -alpha + gamma

        lignes = [
            ('Coefficient $\\alpha$ (impact choc)', f'{alpha:.6f}', ''),
            ('Coefficient $\\gamma$ (asymétrie)',   f'{gamma:.6f}',
             f'{p_gam:.4f}{_signif(p_gam)}'),
            ('Coefficient $\\beta$ (persistance)',  f'{beta:.6f}', ''),
            ('',                                    '',            ''),
            ('Choc positif ($z=+1$) : $\\alpha+\\gamma$',
             f'{impact_pos:.6f}', ''),
            ('Choc négatif ($z=-1$) : $-\\alpha+\\gamma$',
             f'{impact_neg:.6f}', ''),
            ('Ratio |négatif / positif|',
             f'{abs(impact_neg/impact_pos):.4f}x' if abs(impact_pos) > 1e-10 else '---', ''),
        ]
        titre = 'Effet de levier --- EGARCH'
        label = 'tab:effet_levier_egarch'

    elif modele == 'GJR-GARCH' or (vol == 'Garch' and int(best_dict.get('o', 0)) > 0):
        alpha = float(params.get('alpha[1]', float('nan')))
        gamma = float(params.get('gamma[1]', float('nan')))
        beta  = float(params.get('beta[1]',  float('nan')))
        p_gam = float(pvalues.get('gamma[1]', float('nan')))

        impact_pos = alpha
        impact_neg = alpha + gamma

        lignes = [
            ('Coefficient $\\alpha$ (impact choc positif)', f'{alpha:.6f}', ''),
            ('Coefficient $\\gamma$ (surcroît choc négatif)', f'{gamma:.6f}',
             f'{p_gam:.4f}{_signif(p_gam)}'),
            ('Coefficient $\\beta$ (persistance)',  f'{beta:.6f}', ''),
            ('',                                    '',            ''),
            ('Choc positif : $\\alpha$',            f'{impact_pos:.6f}', ''),
            ('Choc négatif : $\\alpha+\\gamma$',    f'{impact_neg:.6f}', ''),
            ('Ratio négatif / positif',
             f'{impact_neg/impact_pos:.4f}x' if abs(impact_pos) > 1e-10 else '---', ''),
        ]
        titre = 'Effet de levier --- GJR-GARCH'
        label = 'tab:effet_levier_gjr'
    else:
        _log.info('    Modele %s sans effet de levier — tab_effet_levier.tex ignore.', modele)
        return

    rows = '\n'.join(
        f'  {l} & {v} & {p} \\\\' for l, v, p in lignes
    )
    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{' + titre + '}\n'
        r'\label{' + label + '}\n'
        r'\begin{tabular}{lrl}' + '\n'
        r'\toprule' + '\n'
        r'Type de choc / Paramètre & Valeur & $p$-valeur \\' + '\n'
        r'\midrule' + '\n'
        + rows + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\footnotesize{*** $p<0{,}01$ ; ** $p<0{,}05$ ; * $p<0{,}10$.}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, nom, dossier)


# ─── Priorité 6 : ratio TVaR/VaR ─────────────────────────────────────────────

def tableau_ratio_tvar_var(df_var: pd.DataFrame, nom: str, dossier):
    """Tableau VaR 95%/99% + TVaR + ratio TVaR/VaR par méthode."""
    methodes_var  = ['VaR Historique', 'VaR Normale', 'VaR Student',
                     'VaR Cornish-Fisher', 'VaR GARCH', 'VaR Monte Carlo (1j)']
    methodes_tvar = ['TVaR Historique', 'TVaR Normale', 'TVaR Student',
                     'TVaR CF semi-empirique', 'TVaR GARCH', 'TVaR Monte Carlo (1j)']
    labels = ['Historique', 'Normale', 'Student', 'Cornish-Fisher',
              'GARCH dyn.', 'Monte Carlo']

    def _val(col): return df_var[col] if col in df_var.columns else pd.Series([float('nan')] * len(df_var), index=df_var.index)

    lignes = []
    for lab, cv, ct in zip(labels, methodes_var, methodes_tvar):
        row = [f'  {lab}']
        for niv in ['95%', '99%']:
            if niv not in df_var.index:
                row += ['---', '---', '---']
                continue
            v  = float(_val(cv).loc[niv])
            tv = float(_val(ct).loc[niv])
            r  = tv / v if abs(v) > 1e-10 else float('nan')
            row += [f'{v:.4f}', f'{tv:.4f}', f'{r:.4f}']
        lignes.append(' & '.join(row) + ' \\\\')

    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Ratio TVaR/VaR par méthode et niveau de confiance}' + '\n'
        r'\label{tab:ratio_tvar_var}' + '\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{lrrrrrr}' + '\n'
        r'\toprule' + '\n'
        r' & \multicolumn{3}{c}{95\,\%} & \multicolumn{3}{c}{99\,\%} \\' + '\n'
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\n'
        r'Méthode & VaR & TVaR & TVaR/VaR & VaR & TVaR & TVaR/VaR \\' + '\n'
        r'\midrule' + '\n'
        + '\n'.join(lignes) + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{VaR et TVaR exprimées en \%. Ratio TVaR/VaR $>1$ indique une queue épaisse.}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, nom, dossier)


# ─── Priorité 7 : validation résidus GARCH ────────────────────────────────────

def tableau_validation_residus_garch(garch_final, config: dict, nom: str, dossier):
    """Tableau de validation des résidus standardisés du modèle GARCH retenu."""
    lags = config.get('sorties_etendues', {}).get('ljung_box_lags', 10)

    resid_raw = garch_final.resid
    vol_cond  = garch_final.conditional_volatility
    z = (resid_raw / vol_cond).dropna()

    moyenne  = float(z.mean())
    std      = float(z.std())
    skew_z   = float(sp_stats.skew(z.values))
    kurt_z   = float(sp_stats.kurtosis(z.values, fisher=True))
    jb, jb_p, _, _ = jarque_bera(z.values)

    lb    = acorr_ljungbox(z,    lags=[lags], return_df=True)
    lb_sq = acorr_ljungbox(z**2, lags=[lags], return_df=True)
    lb_q  = float(lb['lb_stat'].iloc[-1])
    lb_p  = float(lb['lb_pvalue'].iloc[-1])
    lb2_q = float(lb_sq['lb_stat'].iloc[-1])
    lb2_p = float(lb_sq['lb_pvalue'].iloc[-1])

    def _concl_norm(p): return 'Normalité' if p > 0.05 else 'Non-normalité'
    def _concl_no(p):   return 'OK' if p > 0.05 else 'Rejet'

    lignes = [
        ('Moyenne',                    f'{moyenne:.6f}',  '',             ''),
        ('Écart-type',                 f'{std:.6f}',      '',             ''),
        ('Skewness',                   f'{skew_z:.6f}',   '',             ''),
        ('Kurtosis (excès)',           f'{kurt_z:.6f}',   '',             ''),
        ('Jarque-Bera',                f'{jb:.4f}',       f'{jb_p:.4f}',  _concl_norm(jb_p)),
        (f'Ljung-Box({lags}) résidus', f'{lb_q:.4f}',    f'{lb_p:.4f}',  _concl_no(lb_p)),
        (f'Ljung-Box({lags}) résidus²', f'{lb2_q:.4f}',  f'{lb2_p:.4f}', _concl_no(lb2_p)),
    ]

    rows = '\n'.join(
        f'  {l} & {v} & {p} & {c} \\\\' for l, v, p, c in lignes
    )
    contenu = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Validation des résidus standardisés --- modèle GARCH retenu}' + '\n'
        r'\label{tab:valid_resid_garch}' + '\n'
        r'\begin{adjustbox}{max width=\linewidth}' + '\n'
        r'\begin{tabular}{lrrl}' + '\n'
        r'\toprule' + '\n'
        r'Test / Statistique & Valeur & $p$-valeur & Conclusion \\' + '\n'
        r'\midrule' + '\n'
        + rows + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{adjustbox}' + '\n'
        r'\footnotesize{Résidus standardisés : $z_t = \hat{\varepsilon}_t / \hat{\sigma}_t$.}' + '\n'
        r'\end{table}' + '\n'
    )
    _ecrire(contenu, nom, dossier)


# ─── Orchestrateur principal ──────────────────────────────────────────────────

def generer_toutes_sorties(prix, rendements, arima_result: dict,
                           garch_final, df_garch: pd.DataFrame,
                           df_var: pd.DataFrame, best_dict: dict, config: dict):
    sep = '-' * 60
    _log.info('\n%s', sep)
    _log.info('  SORTIES COMPLEMENTAIRES')
    _log.info('%s', sep)

    dossier_res   = Path(config['output']['dossier_resultats'])
    dossier_graph = dossier_res / 'graphiques'
    dossier_latex = dossier_res / 'latex'

    prix_s  = _to_series(prix)
    rend_s  = _to_series(rendements).dropna()
    lbrent  = np.log(prix_s);   lbrent.name  = 'LBRENT'
    dlbrent = lbrent.diff().dropna(); dlbrent.name = 'DLBRENT'

    p_opt = arima_result.get('p_opt', 1)
    d_opt = arima_result.get('d_opt', 0)
    q_opt = arima_result.get('q_opt', 1)
    try:
        arima_fit   = ARIMA(rend_s, order=(p_opt, d_opt, q_opt)).fit(
            method_kwargs={'warn_convergence': False})
        resid_arima = arima_fit.resid.dropna()
    except Exception:
        resid_arima = rend_s.copy()
    resid_arima.name  = 'Résidus ARIMA'
    resid2_arima      = (resid_arima ** 2); resid2_arima.name = 'Résidus² ARIMA'

    ticker = config['data'].get('ticker', '')
    lags   = config.get('sorties_etendues', {}).get('correlogramme_lags', 36)

    # ── A. Séries temporelles ─────────────────────────────────────────────────
    _log.info('\n  [A] Graphiques series temporelles')
    tracer_serie(prix_s,  f'Prix du {ticker} (USD/baril)', 'USD/baril', 'fig_brent',      dossier_graph)
    tracer_serie(lbrent,  f'Log-prix {ticker}',            'log(USD)',  'fig_lbrent',     dossier_graph)
    tracer_serie(dlbrent, f'Log-rendements {ticker}',      '%',         'fig_dlbrent',    dossier_graph)
    tracer_serie(rend_s,  f'Rendements {ticker} (pipeline)', '%',       'fig_rendements', dossier_graph)

    # ── B. Histogrammes ───────────────────────────────────────────────────────
    _log.info('\n  [B] Histogrammes style EViews')
    tracer_histogramme(prix_s,      f'BRENT (niveau)',            'fig_hist_brent',      dossier_graph)
    tracer_histogramme(lbrent,      f'LBRENT (log-prix)',         'fig_hist_lbrent',     dossier_graph)
    tracer_histogramme(dlbrent,     f'DLBRENT (log-rendements)',  'fig_hist_dlbrent',    dossier_graph)
    tracer_histogramme(rend_s,      f'Rendements (pipeline)',     'fig_hist_rendements', dossier_graph)
    tracer_histogramme(resid_arima, f'Résidus ARIMA({p_opt},{d_opt},{q_opt})',
                       'fig_hist_resid_arima', dossier_graph)

    # ── C. Corrélogrammes graphiques ──────────────────────────────────────────
    _log.info('\n  [C] Correlogrammes ACF/PACF (graphiques)')
    tracer_correlogramme(lbrent,       lags, f'Corrélogramme LBRENT',                         'fig_corr_lbrent',      dossier_graph)
    tracer_correlogramme(dlbrent,      lags, f'Corrélogramme DLBRENT',                        'fig_corr_dlbrent',     dossier_graph)
    tracer_correlogramme(rend_s,       lags, f'Corrélogramme Rendements',                     'fig_corr_rendements',  dossier_graph)
    tracer_correlogramme(resid_arima,  lags, f'Corrélogramme Résidus ARIMA({p_opt},{d_opt},{q_opt})', 'fig_corr_resid_arima', dossier_graph)
    tracer_correlogramme(resid2_arima, lags, f'Corrélogramme Résidus² ARIMA (ARCH)',           'fig_corr_resid2_arima', dossier_graph)

    # ── D. Tests de stationnarité ─────────────────────────────────────────────
    _log.info('\n  [D] Tables stationnarite')
    tableau_tests_stationnarite(lbrent,  'LBRENT',  'tab_statio_lbrent.tex',  dossier_latex)
    tableau_tests_stationnarite(dlbrent, 'DLBRENT', 'tab_statio_dlbrent.tex', dossier_latex)
    tableau_synthese_stationnarite(lbrent, dlbrent, 'tab_statio_synthese.tex', dossier_latex)

    # ── E. ARIMA ──────────────────────────────────────────────────────────────
    _log.info('\n  [E] Table ARIMA estimation')
    tableau_arima_estim(rend_s, p_opt, d_opt, q_opt, 'tab_arima_estim.tex', dossier_latex)

    # ── F. ARCH-LM ────────────────────────────────────────────────────────────
    _log.info('\n  [F] Table ARCH-LM')
    tableau_arch_lm(resid_arima, [1, 2, 4, 8, 12], 'tab_arch_lm.tex', dossier_latex)

    # ── G. GARCH benchmarks individuels ──────────────────────────────────────
    _log.info('\n  [G] Tables GARCH benchmarks')
    tableau_arch4(   resid_arima, 'tab_arch4.tex',    dossier_latex)
    tableau_garch11( resid_arima, 'tab_garch11.tex',  dossier_latex)
    tableau_gjr11(   resid_arima, 'tab_gjr11.tex',    dossier_latex)
    tableau_egarch11(resid_arima, 'tab_egarch11.tex', dossier_latex)

    # ── H. Comparaisons par distribution (P3) ────────────────────────────────
    _log.info('\n  [H] Tables comparaison distributions')
    tableau_comparaison_distributions(resid_arima, df_garch, best_dict, dossier_latex)

    # ── I. Effet de levier (P5) ───────────────────────────────────────────────
    _log.info('\n  [I] Table effet de levier')
    tableau_effet_levier(garch_final, best_dict, 'tab_effet_levier.tex', dossier_latex)

    # ── J. Ratio TVaR/VaR (P6) ───────────────────────────────────────────────
    _log.info('\n  [J] Table ratio TVaR/VaR')
    tableau_ratio_tvar_var(df_var, 'tab_ratio_tvar_var.tex', dossier_latex)

    # ── K. Validation résidus GARCH (P7) ─────────────────────────────────────
    _log.info('\n  [K] Table validation residus GARCH')
    tableau_validation_residus_garch(
        garch_final, config, 'tab_validation_residus_garch.tex', dossier_latex)

    _log.info('\n%s', sep)
    _log.info('  Sorties graphiques : %s', dossier_graph)
    _log.info('  Sorties LaTeX      : %s', dossier_latex)
    _log.info('%s', sep)
