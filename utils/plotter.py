"""Génération et sauvegarde des graphiques en haute résolution."""
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from arch import arch_model

warnings.filterwarnings('ignore')

_log = logging.getLogger('tickerlab.plotter')


def setup_style():
    """Style cohérent professionnel (steelblue / darkorange)."""
    plt.rcParams.update({
        'figure.dpi':        100,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'font.size':         10,
        'axes.titlesize':    11,
        'axes.titleweight':  'bold',
        'axes.grid':         True,
        'grid.alpha':        0.3,
    })


def sauver_figure(fig, nom: str, dossier: str = 'resultats/graphiques',
                  formats=('png', 'pdf')):
    """
    Sauvegarde une figure aux formats spécifiés en 300 dpi.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure à sauvegarder.
    nom : str
        Nom de base du fichier (sans extension).
    dossier : str, optional
        Répertoire de destination (créé si absent).
    formats : tuple of str, optional
        Formats de sortie (défaut ('png', 'pdf')).
    """
    out = Path(dossier)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out / f'{nom}.{fmt}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        _log.info('  OK graphique -> %s', path)


def tracer_rendements_vol(df_vol, ticker, best, alpha=0.99, dossier=None):
    """
    Graphique 3 panneaux : rendements / volatilite conditionnelle / VaR-TVaR.

    Parameters
    ----------
    df_vol : pd.DataFrame
        Sortie de construire_df_vol() — colonnes Rendement (%),
        Vol. conditionnelle (%), VaR {pct}% GARCH (%), TVaR {pct}% GARCH (%).
    ticker : str
        Identifiant du sous-jacent (ex: 'BZ=F').
    best : dict
        Dictionnaire du meilleur modèle GARCH (clés: modele, p, o, q, dist).
    alpha : float, optional
        Niveau de confiance (défaut 0.99).
    dossier : str or Path, optional
        Répertoire de sauvegarde. Si None, la figure n'est pas sauvegardée.

    Returns
    -------
    matplotlib.figure.Figure
        Figure générée (fermée après sauvegarde).
    """
    pct      = int(alpha * 100)
    col_var  = f'VaR {pct}% GARCH (%)'
    col_tvar = f'TVaR {pct}% GARCH (%)'
    p        = int(best['p'])
    o        = int(best['o'])
    q        = int(best['q'])
    dist     = best.get('dist', '')
    modele   = best.get('modele', 'GARCH')

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f'{ticker} -- {modele}({p},{o},{q}) [{dist}]',
                 fontsize=13, fontweight='bold')

    axes[0].plot(df_vol.index, df_vol['Rendement (%)'], lw=0.6, color='steelblue')
    axes[0].axhline(0, color='black', lw=0.4, ls='--')
    axes[0].set_ylabel('Rendement (%)')
    axes[0].set_title('Rendements journaliers')

    axes[1].fill_between(df_vol.index, df_vol['Vol. conditionnelle (%)'],
                         alpha=0.4, color='darkorange')
    axes[1].plot(df_vol.index, df_vol['Vol. conditionnelle (%)'],
                 lw=0.7, color='darkorange')
    axes[1].set_ylabel('Vol. (%)')
    axes[1].set_title('Volatilite conditionnelle')

    if col_var in df_vol.columns and col_tvar in df_vol.columns:
        axes[2].plot(df_vol.index, df_vol[col_var],
                     lw=0.8, color='red', label=f'VaR {pct}%')
        axes[2].plot(df_vol.index, df_vol[col_tvar],
                     lw=0.8, color='darkred', ls='--', label=f'TVaR {pct}%')
    axes[2].plot(df_vol.index, df_vol['Rendement (%)'],
                 lw=0.4, color='steelblue', alpha=0.5, label='Rendement')
    axes[2].axhline(0, color='black', lw=0.4, ls='--')
    axes[2].set_ylabel('%')
    axes[2].set_title(f'VaR & TVaR {pct}% dynamiques vs Rendements')
    axes[2].legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    if dossier:
        sauver_figure(fig, 'rendements_vol_var', dossier)
    plt.close(fig)
    return fig


def tracer_var_dynamique_breaches(df_vol, rendements, ticker: str,
                                  config: dict, alpha_95: float = 0.95,
                                  alpha_99: float = 0.99, dossier=None):
    """
    Graphique plein-largeur : rendements + courbes VaR 95%/99% + points de dépassement
    + bandes verticales pour les périodes de crise.
    """
    col_var99 = 'VaR 99% GARCH (%)'
    if isinstance(rendements, pd.DataFrame):
        rend_s = rendements.iloc[:, 0]
    else:
        rend_s = rendements

    # Aligner rendements sur l'index df_vol
    rend_aligned = rend_s.reindex(df_vol.index)
    var99 = df_vol[col_var99] if col_var99 in df_vol.columns else None

    crises = config.get('events', {}).get('crises', [])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_vol.index, rend_aligned, color='#aaaaaa', lw=0.5, alpha=0.7, label='Rendement')

    if var99 is not None:
        ax.plot(df_vol.index, var99, color='red', lw=0.9, label='VaR 99% GARCH')
        breach99 = df_vol.index[rend_aligned < var99]
        if len(breach99):
            ax.scatter(breach99, rend_aligned.loc[breach99],
                       color='red', s=12, zorder=5, label=f'Dép. 99% ({len(breach99)})')

    # Bandes de crise
    for ev in crises:
        an   = ev['annee']
        lbl  = ev.get('label', str(an))
        xmin = pd.Timestamp(f'{an}-01-01')
        xmax = pd.Timestamp(f'{an}-12-31')
        if xmin < df_vol.index[-1] and xmax > df_vol.index[0]:
            ax.axvspan(xmin, xmax, alpha=0.08, color='gold', zorder=0)
            ax.text(xmin, ax.get_ylim()[0] if ax.get_ylim() else -5,
                    f' {lbl[:12]}', fontsize=6.5, color='#666600',
                    rotation=90, va='bottom', ha='left')

    ax.axhline(0, color='black', lw=0.4, ls='--', alpha=0.6)
    ax.set_title(f'{ticker} — VaR dynamique GARCH (OOS complet) avec dépassements',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('%', fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle=':')
    fig.tight_layout()
    if dossier:
        sauver_figure(fig, 'fig_var_dynamique_breaches', dossier)
    plt.close(fig)
    return fig


def tracer_garch_comparaison(rendements, ticker: str,
                              best_dict: dict, df_garch: pd.DataFrame,
                              dossier=None):
    """
    Graphique 3×1 : volatilité conditionnelle GARCH(1,1) / GJR(1,1) / EGARCH(1,1)
    avec distribution Student-t, mêmes échelles.
    """
    if isinstance(rendements, pd.DataFrame):
        rend_s = rendements.iloc[:, 0].dropna()
    else:
        rend_s = rendements.dropna()

    specs = [
        ('GARCH(1,1) Student', 'Garch',  1, 0, 1),
        ('GJR-GARCH(1,1) Student', 'Garch', 1, 1, 1),
        ('EGARCH(1,1) Student',  'EGARCH', 1, 1, 1),
    ]
    colors = ['steelblue', 'darkorange', 'crimson']

    fits = []
    for label, vol, p, o, q in specs:
        try:
            fit = arch_model(rend_s, vol=vol, p=p, o=o, q=q, dist='t').fit(disp='off')
            fits.append((label, fit))
            _log.info('    %s OK', label)
        except Exception as e:
            fits.append((label, None))
            _log.warning('    %s ERREUR: %s', label, e)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f'{ticker} — Comparaison volatilités conditionnelles (Student-t)',
                 fontsize=11, fontweight='bold')

    y_max = max(
        (fit.conditional_volatility.max() for _, fit in fits if fit is not None),
        default=10
    )

    for ax, (label, fit), color in zip(axes, fits, colors):
        if fit is not None:
            vol_cond = fit.conditional_volatility
            ax.fill_between(vol_cond.index, vol_cond, alpha=0.3, color=color)
            ax.plot(vol_cond.index, vol_cond, lw=0.7, color=color)
        ax.set_ylabel('Vol. (%)', fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, y_max * 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linestyle=':')
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

    axes[-1].set_xlabel('Date', fontsize=8)
    fig.tight_layout()
    if dossier:
        sauver_figure(fig, 'fig_garch_comparaison', dossier)
    plt.close(fig)
    return fig
