# -*- coding: utf-8 -*-
"""
Detection de ruptures structurelles — sans package externe (scipy + statsmodels).

Methodes implementees :

1. CUSUM-OLS (Brown, Durbin & Evans 1975)
   Ref : Ploberger, W. & Kramer, W. (1992). Econometrica 60(2), 271-285.

2. Chow test sequentiel simplifie (Bai-Perron mono-break)
   Ref : Chow, G.C. (1960). Econometrica 28(3), 591-605.

3. ICSS — Iterated Cumulative Sums of Squares (Inclan & Tiao 1994)
   Detect automatiquement les ruptures de variance dans eps^2.
   Ref : Inclan, C. & Tiao, G.C. (1994). Use of cumulative sums of squares
         for retrospective detection of changes of variance. JASA 89(427),
         913-923.
   Application GARCH : Hillebrand, E. (2005). Neglecting parameter changes
         in GARCH models. Journal of Econometrics 129(1-2), 121-138.

4. Zivot-Andrews (1992) — test de racine unitaire avec rupture endogene
   Wraps statsmodels.tsa.stattools.zivot_andrews.
   Ref : Zivot, E. & Andrews, D.W.K. (1992). Further evidence on the great
         crash, the oil-price shock, and the unit-root hypothesis. JBES
         10(3), 251-270.
"""
import logging
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, norm

_log = logging.getLogger(__name__)


# ── CUSUM-OLS ─────────────────────────────────────────────────────────────────

def cusum_ols(serie: pd.Series, trim: float = 0.15) -> dict:
    """
    CUSUM-OLS de Brown, Durbin & Evans (1975).

    Residus recursifs OLS d'une regression de y_t sur une constante
    (et optionnellement une tendance).

    Parameters
    ----------
    serie : pd.Series
        Serie temporelle (prix ou rendements).
    trim : float
        Fraction d'observations exclues en debut et fin (defaut 0.15).

    Returns
    -------
    dict avec :
      'cusum'      : pd.Series  CUSUM standardise
      'rupture'    : date (ou None)  date du max |CUSUM|
      'rupture_idx': int
      'stat'       : float  max |CUSUM|
      'seuil_5pct' : float  seuil critique 5% (approximation Ploberger-Kramer)
      'rejet_5pct' : bool
      'n'          : int
    """
    y = serie.dropna().values
    n = len(y)
    if n < 20:
        return {'cusum': pd.Series(dtype=float), 'rupture': None,
                'rupture_idx': None, 'stat': float('nan'),
                'seuil_5pct': float('nan'), 'rejet_5pct': False, 'n': n}

    # Residus recursifs OLS (constante uniquement)
    k = 1  # nombre de regresseurs (constante)
    cusum_vals = []
    sigma2_hat = None
    for t in range(k, n):
        sub = y[:t]
        mu_t = float(np.mean(sub))
        e_t  = y[t] - mu_t
        cusum_vals.append(e_t)

    # Standardisation : sigma de la serie entiere
    sigma = float(np.std(y, ddof=1))
    if sigma < 1e-12:
        sigma = 1.0

    cusum_std  = np.cumsum(cusum_vals) / (sigma * math.sqrt(n))
    idx_start  = k
    idx_dates  = serie.dropna().index[idx_start: idx_start + len(cusum_std)]
    cusum_s    = pd.Series(cusum_std, index=idx_dates, name='CUSUM-OLS')

    max_abs    = float(np.max(np.abs(cusum_std)))
    rupture_i  = int(np.argmax(np.abs(cusum_std))) + idx_start
    rupture_dt = serie.dropna().index[rupture_i] if rupture_i < len(serie.dropna()) else None

    # Seuil Ploberger-Kramer 5% : c_alpha * sqrt(n) / sqrt(n) ~= 1.358 (approximation)
    seuil_5pct = 1.358  # valeur tabulee Ploberger-Kramer 1992, 5%

    return {
        'cusum':       cusum_s,
        'rupture':     rupture_dt,
        'rupture_idx': rupture_i,
        'stat':        max_abs,
        'seuil_5pct':  seuil_5pct,
        'rejet_5pct':  max_abs > seuil_5pct,
        'n':           n,
    }


# ── Chow test sequentiel (une rupture) ───────────────────────────────────────

def chow_test_seq(serie: pd.Series, trim: float = 0.15) -> dict:
    """
    Test de Chow sequentiel : cherche la rupture unique maximisant la stat F.

    H0 : pas de rupture
    H1 : une rupture en tau (constante + tendance)

    Parameters
    ----------
    serie : pd.Series
    trim : float
        Fraction min d'observations avant et apres le point de rupture.

    Returns
    -------
    dict avec :
      'rupture'    : date (ou None)
      'rupture_idx': int
      'f_stat'     : float  statistique F au point optimal
      'p_value'    : float
      'verdict'    : str ('Rupture detectee (5%)' | 'Pas de rupture')
      'n'          : int
      'f_series'   : pd.Series  profil F sur les candidats
    """
    y = serie.dropna().values
    n = len(y)
    k = 1  # regresseur : constante

    t_min = max(k + 1, int(trim * n))
    t_max = n - t_min
    candidates = np.arange(t_min, t_max)

    if len(candidates) < 2:
        return {'rupture': None, 'rupture_idx': None,
                'f_stat': float('nan'), 'p_value': float('nan'),
                'verdict': 'N/A', 'n': n,
                'f_series': pd.Series(dtype=float)}

    rss_total = float(np.sum((y - y.mean()) ** 2))
    f_vals    = np.full(len(candidates), float('nan'))

    for i, tau in enumerate(candidates):
        y1, y2 = y[:tau], y[tau:]
        rss1 = float(np.sum((y1 - y1.mean()) ** 2)) if len(y1) > 1 else 0.0
        rss2 = float(np.sum((y2 - y2.mean()) ** 2)) if len(y2) > 1 else 0.0
        rss_r = rss1 + rss2
        denom = rss_r / max(n - 2 * k, 1)
        numer = (rss_total - rss_r) / k
        f_vals[i] = numer / max(denom, 1e-12)

    idx_best  = int(np.nanargmax(f_vals))
    tau_best  = candidates[idx_best]
    f_best    = float(f_vals[idx_best])
    df1, df2  = k, n - 2 * k
    p_val     = float(1 - f_dist.cdf(f_best, df1, max(df2, 1)))

    rupture_dt = serie.dropna().index[tau_best] if tau_best < len(serie.dropna()) else None

    idx_dates  = serie.dropna().index[candidates]
    f_series   = pd.Series(f_vals, index=idx_dates, name='F-stat Chow')

    return {
        'rupture':     rupture_dt,
        'rupture_idx': tau_best,
        'f_stat':      f_best,
        'p_value':     p_val,
        'verdict':     'Rupture detectee (5%)' if p_val < 0.05 else 'Pas de rupture',
        'n':           n,
        'f_series':    f_series,
    }


# ── ICSS — Inclán & Tiao (1994) ──────────────────────────────────────────────

def inclan_tiao_icss(
    eps_squared: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> list[int]:
    """
    Algorithme ICSS de détection de ruptures de variance (Inclán & Tiao 1994).

    Appliqué aux résidus au carré d'un modèle GARCH, il identifie les
    changements de niveau de variance non modélisés par le processus GARCH.
    Ces ruptures, si ignorées, gonflent artificiellement la persistance α+β
    (Hillebrand 2005).

    Algorithme récursif :
      1. Ck = Σ eps²[0:k], Cn = Σ eps²
      2. Dk = Ck/Cn − k/n   (ICSS process)
      3. IT = sqrt(n/2) · max|Dk|
      4. Rejet H0 si IT > crit_val ; breakpoint k* = argmax|Dk|
      5. Récurse sur [0, k*-1] et [k*+1, n-1]

    Parameters
    ----------
    eps_squared : array-like
        Résidus au carré (eps_t²) ou rendements au carré.
    alpha : float
        Seuil de significativité. Valeurs critiques simulées :
        0.10 → 1.224 | 0.05 → 1.358 | 0.01 → 1.628 (Table 1, Inclán & Tiao).

    Returns
    -------
    list of int
        Indices des breakpoints détectés (base 0), triés par ordre croissant.

    References
    ----------
    Inclán, C. & Tiao, G.C. (1994). Use of cumulative sums of squares for
        retrospective detection of changes of variance. JASA 89(427), 913-923.
    Hillebrand, E. (2005). Neglecting parameter changes in GARCH models.
        Journal of Econometrics 129(1-2), 121-138.
    """
    _CRIT = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}
    crit_val = _CRIT.get(alpha, 1.358)

    arr = np.asarray(eps_squared, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]

    def _recurse(sub: np.ndarray, offset: int) -> list[int]:
        m = len(sub)
        if m < 15:
            return []
        Cm  = sub.cumsum()
        Cn  = Cm[-1]
        if Cn < 1e-14:
            return []
        k   = np.arange(1, m + 1, dtype=float)
        Dk  = Cm / Cn - k / m
        IT  = float(np.sqrt(m / 2.0) * np.max(np.abs(Dk)))
        if IT <= crit_val:
            return []
        k_star = int(np.argmax(np.abs(Dk)))       # 0-indexed dans sub
        breaks = [offset + k_star]
        if k_star > 10:
            breaks += _recurse(sub[:k_star], offset)
        if m - k_star - 1 > 10:
            breaks += _recurse(sub[k_star + 1:], offset + k_star + 1)
        return breaks

    raw = _recurse(arr, 0)
    return sorted(set(raw))


# ── Zivot-Andrews (1992) ──────────────────────────────────────────────────────

def zivot_andrews_test(
    series: pd.Series,
    regression: str = 'ct',
) -> dict:
    """
    Test de racine unitaire avec rupture structurelle endogène (Zivot-Andrews 1992).

    Wraps ``statsmodels.tsa.stattools.zivot_andrews``. La date de rupture est
    choisie endogènement comme celle qui minimise la statistique t de la
    racine unitaire (maximise le rejet).

    Parameters
    ----------
    series : pd.Series
        Série temporelle (log-prix ou rendements).
    regression : str
        'c'  — rupture dans la constante seulement.
        't'  — rupture dans la tendance seulement.
        'ct' — rupture dans constante et tendance (défaut, modèle C de ZA).

    Returns
    -------
    dict avec :
      stat       : float  Statistique ZA (négative sous H1).
      pvalue     : float  p-value asymptotique.
      cv_1/5/10  : float  Valeurs critiques 1%, 5%, 10%.
      bp_date    : date   Date de rupture optimale (ou None).
      bp_idx     : int    Indice de la rupture dans la série.
      rejet_5pct : bool   Rejet de H0 (racine unitaire) à 5%.

    References
    ----------
    Zivot, E. & Andrews, D.W.K. (1992). Further evidence on the great crash,
        the oil-price shock, and the unit-root hypothesis. Journal of Business
        & Economic Statistics 10(3), 251-270.
    """
    from statsmodels.tsa.stattools import zivot_andrews as _za

    y = series.dropna()
    try:
        zastat, pvalue, cvdict, baselag, bp = _za(y.values, regression=regression,
                                                   autolag='AIC')
        bp_date = y.index[int(bp)] if int(bp) < len(y) else None
        cv5     = float(cvdict.get('5%', float('nan')))
        return {
            'stat':       float(zastat),
            'pvalue':     float(pvalue),
            'cv_1':       float(cvdict.get('1%',  float('nan'))),
            'cv_5':       cv5,
            'cv_10':      float(cvdict.get('10%', float('nan'))),
            'bp_date':    bp_date,
            'bp_idx':     int(bp),
            'baselag':    int(baselag),
            'rejet_5pct': float(zastat) < cv5,
            'regression': regression,
        }
    except Exception as exc:
        _log.warning('zivot_andrews_test : %s', exc)
        return {
            'stat': float('nan'), 'pvalue': float('nan'),
            'cv_1': float('nan'), 'cv_5':   float('nan'), 'cv_10': float('nan'),
            'bp_date': None, 'bp_idx': None, 'baselag': None,
            'rejet_5pct': False, 'regression': regression,
        }


# ── Génération des dummies de rupture ────────────────────────────────────────

def _build_break_dummies(
    n: int,
    breakpoints: list[int],
    index=None,
) -> pd.DataFrame:
    """
    Construit une matrice de variables dummy à partir des breakpoints ICSS.

    Chaque colonne D_k vaut 0 avant le breakpoint k et 1 à partir de k.
    Ces dummies peuvent être injectées dans l'équation de moyenne d'un ARX.

    Note : arch 8.0.0 ne supporte pas GARCHX (régresseurs dans la variance).
    L'injection se fait donc via l'équation de moyenne (ARX), ce qui capture
    les sauts de niveau plutôt que les changements de variance directs.
    Pour une injection dans la variance, utiliser arch >= 7.x avec GARCHX
    ou implémenter une sous-classe custom de GARCH.

    Parameters
    ----------
    n          : int       Longueur de la série.
    breakpoints: list[int] Indices des ruptures (base 0).
    index      : pd.Index  Index temporel optionnel.

    Returns
    -------
    pd.DataFrame  shape (n, len(breakpoints)), colonnes 'break_0', 'break_1', ...
    """
    data = {}
    for i, bp in enumerate(breakpoints):
        col = np.zeros(n, dtype=float)
        col[bp:] = 1.0
        data[f'break_{i}'] = col
    df = pd.DataFrame(data, index=index if index is not None else np.arange(n))
    return df


# ── Wrapper principal ─────────────────────────────────────────────────────────

def detecter_ruptures_brent(
    prix: pd.Series,
    rendements: pd.Series,
    config: dict,
    residuals_squared: np.ndarray | None = None,
) -> dict:
    """
    Détection de ruptures structurelles — version production.

    Combine quatre méthodes complémentaires :
      - CUSUM-OLS (Brown et al. 1975) sur prix et rendements
      - Chow séquentiel (1960) sur prix et rendements
      - ICSS (Inclán & Tiao 1994) sur les résidus² ou rendements²
      - Zivot-Andrews (1992) sur log(prix)

    Parameters
    ----------
    prix               : pd.Series   Série de prix.
    rendements         : pd.Series   Log-rendements.
    config             : dict        Configuration pipeline.
    residuals_squared  : array-like  Résidus² du GARCH final (si disponibles).
                         Si None, utilise rendements² comme proxy.

    Returns
    -------
    dict avec :
      'cusum_prix', 'cusum_rend', 'chow_prix', 'chow_rend'  — v5 (inchangés)
      'icss_breakpoints'  : list[int]  Indices ICSS sur eps² (base 0)
      'icss_dates'        : list       Dates des ruptures ICSS
      'icss_n_breaks'     : int
      'zivot_andrews'     : dict       Résultat ZA sur log(prix)
      'dummies'           : pd.DataFrame  Variables dummy pour injection ARX
    """
    sb_cfg = config.get('structural_breaks', {})
    trim   = float(sb_cfg.get('trim', 0.15))

    # ── Méthodes v5 (CUSUM-OLS + Chow) ──────────────────────────────────────
    results: dict = {}
    for label, fn, s in [
        ('cusum_prix',  cusum_ols,     prix),
        ('cusum_rend',  cusum_ols,     rendements),
        ('chow_prix',   chow_test_seq, prix),
        ('chow_rend',   chow_test_seq, rendements),
    ]:
        try:
            results[label] = fn(s, trim=trim)
        except Exception as exc:
            warnings.warn(f'detecter_ruptures [{label}] : {exc}')
            results[label] = {}

    # ── ICSS sur résidus² ────────────────────────────────────────────────────
    icss_cfg    = sb_cfg.get('icss', {})
    icss_active = bool(icss_cfg.get('enabled', True))
    icss_alpha  = float(icss_cfg.get('alpha', 0.05))

    rend_clean  = rendements.dropna()
    eps2_proxy  = (np.asarray(residuals_squared)
                   if residuals_squared is not None
                   else rend_clean.values ** 2)

    if icss_active:
        try:
            bps_idx   = inclan_tiao_icss(eps2_proxy, alpha=icss_alpha)
            bps_dates = [rend_clean.index[i] for i in bps_idx if i < len(rend_clean)]
            _log.info('ICSS : %d ruptures détectées sur %d obs', len(bps_idx), len(eps2_proxy))
        except Exception as exc:
            warnings.warn(f'ICSS : {exc}')
            bps_idx, bps_dates = [], []
    else:
        bps_idx, bps_dates = [], []

    results['icss_breakpoints'] = bps_idx
    results['icss_dates']       = bps_dates
    results['icss_n_breaks']    = len(bps_idx)
    results['dummies']          = _build_break_dummies(
        len(rend_clean), bps_idx, index=rend_clean.index,
    )

    # ── Zivot-Andrews sur log(prix) ──────────────────────────────────────────
    za_cfg    = sb_cfg.get('zivot_andrews', {})
    za_active = bool(za_cfg.get('enabled', True))
    za_reg    = str(za_cfg.get('regression', 'ct'))

    if za_active:
        try:
            log_prix = np.log(prix.dropna().clip(lower=1e-6))
            results['zivot_andrews'] = zivot_andrews_test(log_prix, regression=za_reg)
        except Exception as exc:
            warnings.warn(f'Zivot-Andrews : {exc}')
            results['zivot_andrews'] = {}
    else:
        results['zivot_andrews'] = {}

    return results


def analyser_icss_pipeline(rendements, garch_final, config: dict,
                           garch_best=None) -> dict | None:
    """
    Détection ICSS post-GARCH des ruptures de variance (Inclán & Tiao 1994).

    Applique l'algorithme ICSS aux résidus² du modèle GARCH sélectionné
    pour identifier des changements de régime de variance non modélisés.
    Si ignorés, ces ruptures gonflent artificiellement la persistance α+β
    (Lamoureux & Lastrapes 1990 ; Hillebrand 2005).

    Modes (config['structural_breaks']['mode']) :
      'diagnostic' (défaut) — détecte, avertit, n'altère PAS l'estimation.
      'integrate'           — ré-estime le meilleur modèle avec dummies en
                              mean model et compare la persistance avant/après.
      'off'                 — aucun calcul (équivaut à enabled: false).

    Parameters
    ----------
    rendements  : pd.Series        Log-rendements en pourcentage.
    garch_final : ARCHModelResult  Résultat de l'estimation finale.
    config      : dict             Configuration pipeline.
    garch_best  : pd.Series | dict Ligne df_garch du modèle sélectionné.
                  Requis pour mode='integrate' ; ignoré sinon.

    Returns
    -------
    dict avec clés : n_breaks, indices, dates, mode, warning_lamoureux.
    Si mode='integrate' et n_breaks > 0 : + persistance_avant, persistance_apres,
    dummies_injectees.
    None si désactivé (enabled=False ou mode='off').
    """
    sb_cfg  = config.get('structural_breaks', {})
    enabled = bool(sb_cfg.get('enabled', True))
    mode    = str(sb_cfg.get('mode', 'diagnostic')).lower()

    if not enabled or mode == 'off':
        return None

    seuil_pers  = float(sb_cfg.get('seuil_persistance_alerte', 0.97))
    seuil_n     = int(sb_cfg.get('seuil_n_ruptures_alerte', 3))
    max_dummies = int(sb_cfg.get('max_dummies', 5))

    # ── ICSS sur eps² ─────────────────────────────────────────────────────────
    try:
        rend_clean  = rendements.dropna()
        eps_sq      = garch_final.resid.dropna().values ** 2
        breakpoints = inclan_tiao_icss(eps_sq)
    except Exception as exc:
        _log.warning('[ICSS] échec détection : %s', exc)
        return None

    dates = []
    try:
        dates = [str(rend_clean.index[i]) for i in breakpoints if i < len(rend_clean)]
    except Exception as exc:
        _log.debug('[ICSS] construction des dates de rupture échouée : %s', exc)

    n_breaks = len(breakpoints)

    # ── Alerte Lamoureux-Lastrapes ────────────────────────────────────────────
    pers = float('nan')
    try:
        from tickerlab.core.rapport._stats import persistance_garch as _pers_fn
        nom_mod = 'GARCH'
        if garch_best is not None:
            bd = garch_best.to_dict() if hasattr(garch_best, 'to_dict') else dict(garch_best)
            nom_mod = str(bd.get('modele', 'GARCH'))
        pers = float(_pers_fn(nom_mod, garch_final.params))
    except Exception as exc:
        _log.debug('[ICSS] calcul persistance Lamoureux-Lastrapes échoué (pers=NaN) : %s', exc)

    warning_ll = (not math.isnan(pers) and pers > seuil_pers and n_breaks >= seuil_n)
    if warning_ll:
        _log.warning(
            '[ICSS] %d rupture(s) de variance détectée(s) — pers=%.4f > %.2f. '
            'La persistance est potentiellement gonflée par des changements de '
            'régime non modélisés (Lamoureux & Lastrapes 1990, JoE).',
            n_breaks, pers, seuil_pers,
        )
    else:
        _log.info('[ICSS] %d rupture(s) | pers=%.4f | alerte L&L: NON',
                  n_breaks, pers)

    result: dict = {
        'n_breaks':         n_breaks,
        'indices':          breakpoints,
        'dates':            dates,
        'mode':             mode,
        'warning_lamoureux': warning_ll,
    }

    if mode != 'integrate' or garch_best is None or n_breaks == 0:
        return result

    # ── mode='integrate' — ré-estimation avec dummies ─────────────────────────
    try:
        from arch import arch_model as _arch_model
        from tickerlab.core.rapport._stats import persistance_garch as _pers_fn

        bd = garch_best.to_dict() if hasattr(garch_best, 'to_dict') else dict(garch_best)
        nom_mod    = str(bd['modele'])
        pers_avant = float(_pers_fn(nom_mod, garch_final.params))

        # Dummies plafonnées
        bps_cap = breakpoints[:max_dummies]
        dummies = _build_break_dummies(len(rend_clean), bps_cap, index=rend_clean.index)

        kw = {
            'vol':  str(bd['vol']),
            'p':    int(bd['p']),
            'o':    int(bd.get('o', 0)),
            'q':    int(bd['q']),
            'dist': str(bd['dist']),
        }
        power = bd.get('power', None)
        if power is not None:
            try:
                pv = float(power)
                if not math.isnan(pv):
                    kw['power'] = pv
            except (TypeError, ValueError) as exc:
                _log.debug('[ICSS] power ignoré (non convertible en float) : %s', exc)

        fit_avec   = _arch_model(rend_clean, x=dummies, **kw).fit(disp='off')
        pers_apres = float(_pers_fn(nom_mod, fit_avec.params))

        result['persistance_avant'] = round(pers_avant,  6)
        result['persistance_apres'] = round(pers_apres,  6)
        result['dummies_injectees'] = len(bps_cap)

        _log.info(
            '[ICSS integrate] pers. avant=%.4f → après=%.4f (Δ=%.4f, %d dummy(ies))',
            pers_avant, pers_apres, pers_apres - pers_avant, len(bps_cap),
        )
    except Exception as exc_int:
        _log.warning('[ICSS integrate] ré-estimation échouée : %s', exc_int)

    return result


def detecter_ruptures(prix: pd.Series, rendements: pd.Series, config: dict) -> dict:
    """
    Alias de compatibilité v5 → appelle detecter_ruptures_brent.

    Pour les nouveaux appelants, utiliser detecter_ruptures_brent directement
    avec residuals_squared pour alimenter l'ICSS sur les résidus GARCH.
    """
    return detecter_ruptures_brent(prix, rendements, config, residuals_squared=None)
