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
from scipy.optimize import minimize
from scipy.special import gammaln
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


# ── GARCH à omega par régime (Hillebrand 2005) ───────────────────────────────

_PENALITE_NLL   = 1e10   # valeur retournée par la log-vraisemblance en échec
_BORNE_BASSE_AB = 1e-6   # alpha/beta jugés écrasés sur leur borne en deçà
_MIN_OBS_REGIME = 50     # taille minimale d'un régime exploitable
_MAX_REGIMES_DEFAUT = 6  # plafond d'identifiabilité (1 paramètre omega/régime)


def selectionner_ruptures_espacees(
    breakpoints,
    n: int,
    min_obs_regime: int = _MIN_OBS_REGIME,
    max_regimes: int = _MAX_REGIMES_DEFAUT,
) -> list:
    """
    Sélectionne les ruptures exploitables pour une estimation à omega par régime.

    Critère d'ESPACEMENT, non de rang chronologique. Les ruptures sont
    parcourues dans l'ordre et retenues seulement si elles laissent au moins
    ``min_obs_regime`` observations depuis la rupture précédemment retenue —
    le début de série jouant ce rôle pour la première — ET jusqu'à la fin de
    la série.

    Motivation : une troncature par rang (``breakpoints[:k]``) retient les
    ruptures les plus PRÉCOCES. Sur des séries réelles où l'ICSS en détecte
    plusieurs dizaines, souvent agglutinées en début d'échantillon, elle
    produit des régimes de quelques dizaines d'observations et rend
    l'estimation systématiquement dégénérée.

    Cohérence des seuils : le défaut est ``_MIN_OBS_REGIME``, celui-là même en
    deçà duquel ``estimer_garch_omega_par_regime`` déclare un régime dégénéré.
    Les deux doivent rester alignés — retenir une rupture qui produirait un
    régime aussitôt jugé dégénéré n'aurait aucun sens. Le test de sélection est
    ``>= min_obs_regime`` et celui de dégénérescence ``< min_obs_regime`` : un
    régime d'exactement ``min_obs_regime`` observations est donc accepté par
    les deux.

    Plafond ``max_regimes`` : chaque régime consomme un paramètre omega libre.
    Le plafond borne la dimension du problème d'estimation et protège
    l'identifiabilité des omega ; il ne dérive PAS de l'ancien mécanisme
    d'injection de dummies, qui n'existe plus.

    Parameters
    ----------
    breakpoints : list of int
        Indices candidats (base 0), dans un ordre quelconque.
    n : int
        Longueur de la série.
    min_obs_regime : int
        Taille minimale d'un régime, en observations (défaut 50).
    max_regimes : int
        Nombre maximal de régimes (défaut 6), soit au plus
        ``max_regimes - 1`` ruptures retenues.

    Returns
    -------
    list of int
        Ruptures retenues, triées, espacées d'au moins ``min_obs_regime``.
    """
    if n <= 0 or int(max_regimes) < 2:
        return []

    min_obs      = max(int(min_obs_regime), 1)
    max_ruptures = int(max_regimes) - 1

    retenues: list = []
    precedent = 0                       # borne gauche du régime courant
    for b in sorted({int(x) for x in breakpoints}):
        if len(retenues) >= max_ruptures:
            break
        if b <= 0 or b >= n:
            continue
        if b - precedent < min_obs:     # régime courant trop court
            continue
        if n - b < min_obs:             # dernier régime trop court
            continue
        retenues.append(b)
        precedent = b

    return retenues


def _indices_regimes(n: int, breakpoints) -> tuple:
    """Indice de régime k(t) pour chaque t, et liste des ruptures retenues.

    Les ruptures hors de ]0, n[ et les doublons sont écartés : une rupture en 0
    ou en n ne sépare aucun régime.
    """
    bps = sorted({int(b) for b in breakpoints if 0 < int(b) < n})
    k = np.zeros(n, dtype=np.int64)
    for j, b in enumerate(bps, start=1):
        k[b:] = j
    return k, bps


def _recursion_omega_regime(omegas, alpha, beta, eps2, regime, sigma2_0):
    """Récursion sigma2[t] = omega[k(t)] + alpha·eps2[t-1] + beta·sigma2[t-1].

    Renvoie None dès qu'une valeur non finie ou non strictement positive
    apparaît, plutôt que de propager.

    Le contrôle de finitude est EXPLICITE et non un plancher `max(x, 1e-8)` :
    `max` ne protège pas contre un nan, toute comparaison avec nan étant
    fausse, et le nan traverserait alors silencieusement toute la récursion
    (leçon de core/component_garch.py).
    """
    if not (math.isfinite(sigma2_0) and sigma2_0 > 0.0):
        return None
    T = len(eps2)
    sigma2 = np.empty(T, dtype=float)
    sigma2[0] = sigma2_0
    for t in range(1, T):
        s2 = omegas[regime[t]] + alpha * eps2[t - 1] + beta * sigma2[t - 1]
        if not math.isfinite(s2) or s2 <= 0.0:
            return None
        sigma2[t] = s2
    return sigma2


def _nll_omega_regime(params, eps2, regime, n_regimes, sigma2_0, dist) -> float:
    """Log-vraisemblance négative du GARCH à omega par régime.

    Renvoie ``_PENALITE_NLL`` sur tout échec numérique — jamais un nan, que
    l'optimiseur ne saurait pas exploiter.
    """
    if not np.all(np.isfinite(params)):
        return _PENALITE_NLL

    omegas = params[:n_regimes]
    alpha  = float(params[n_regimes])
    beta   = float(params[n_regimes + 1])
    nu     = float(params[n_regimes + 2]) if dist == 't' else None

    if dist == 't' and nu <= 2.0:
        return _PENALITE_NLL

    sigma2 = _recursion_omega_regime(omegas, alpha, beta, eps2, regime, sigma2_0)
    if sigma2 is None:
        return _PENALITE_NLL

    if dist == 't':
        # Student STANDARDISÉE (variance unitaire) — même convention que
        # core.var_engine.var_student : ecart-type = echelle × sqrt(nu/(nu-2)).
        log_c = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                 - 0.5 * math.log(math.pi * (nu - 2.0)))
        ll = float(np.sum(log_c - 0.5 * np.log(sigma2)
                          - (nu + 1.0) / 2.0
                          * np.log1p(eps2 / (sigma2 * (nu - 2.0)))))
    else:
        ll = float(np.sum(-0.5 * math.log(2.0 * math.pi)
                          - 0.5 * np.log(sigma2)
                          - 0.5 * eps2 / sigma2))

    return -ll if math.isfinite(ll) else _PENALITE_NLL


def estimer_garch_omega_par_regime(
    rendements,
    breakpoints,
    dist: str = 'normal',
    nu=None,
    alpha0=None,
    beta0=None,
) -> dict:
    """
    GARCH(1,1) à omega propre à chaque régime (spécification de Hillebrand 2005).

        sigma2[t] = omega[k(t)] + alpha · eps2[t-1] + beta · sigma2[t-1]

    où k(t) est l'indice du régime auquel appartient t. ``alpha`` et ``beta``
    sont COMMUNS à tous les régimes ; il y a ``len(breakpoints) + 1``
    paramètres omega.

    Motivation : la persistance alpha+beta mesurée par un GARCH global est
    gonflée lorsqu'un omega unique est imposé à des régimes de niveaux de
    variance différents (Lamoureux & Lastrapes 1990 ; Hillebrand 2005). En
    libérant omega par régime, alpha+beta retombe vers sa valeur réelle.

    Conventions
    -----------
    - Moyenne : les rendements sont centrés sur leur moyenne empirique, qui
      n'est pas ré-estimée et n'est pas comptée dans les paramètres libres.
    - **Initialisation de la récursion** : ``sigma2[0]`` est fixé à la variance
      empirique du PREMIER régime (et non à la variance de toute la série, qui
      mélangerait les régimes, ni à la variance inconditionnelle implicite, qui
      dépendrait des paramètres estimés).
    - Loi de Student STANDARDISÉE à variance unitaire, même convention que
      ``core.var_engine`` — l'écart-type vaut échelle × sqrt(nu/(nu-2)).

    Parameters
    ----------
    rendements : pd.Series or np.ndarray
        Log-rendements (en pourcentage, comme le reste du pipeline).
    breakpoints : list of int
        Indices de rupture (base 0). Les valeurs hors ]0, n[ sont ignorées.
        Une liste vide donne un GARCH ordinaire à un seul régime.
    dist : {'normal', 't'}
        Loi des innovations.
    nu : float, optional
        Valeur initiale des degrés de liberté (``dist='t'``). Défaut 8.
    alpha0, beta0 : float, optional
        Valeurs initiales issues d'une estimation globale. À défaut, 0.08/0.88.

    Returns
    -------
    dict
        ``omega`` (list, un par régime), ``alpha``, ``beta``, ``nu``,
        ``persistance`` (alpha+beta), ``n_regimes``, ``loglik``, ``aic``,
        ``bic``, ``converged``, ``degenerate``, ``degeneracies``.

        En cas de terminaison sur la pénalité, ``loglik``/``aic``/``bic`` valent
        nan — jamais une valeur finie dénuée de sens — et ``degeneracies`` le
        dit explicitement.

    References
    ----------
    Hillebrand, E. (2005). Neglecting parameter changes in GARCH models.
        Journal of Econometrics 129(1-2), 121-138.
    Lamoureux, C.G. & Lastrapes, W.D. (1990). Persistence in variance,
        structural change, and the GARCH model. JBES 8(2), 225-234.
    """
    serie = (rendements.dropna() if hasattr(rendements, 'dropna')
             else pd.Series(np.asarray(rendements, dtype=float)).dropna())
    r = np.asarray(serie, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)

    dist = 't' if str(dist).lower().startswith('t') else 'normal'

    if T < 2 * _MIN_OBS_REGIME:
        return {
            'omega': [], 'alpha': float('nan'), 'beta': float('nan'), 'nu': None,
            'persistance': float('nan'), 'n_regimes': 0,
            'loglik': float('nan'), 'aic': float('nan'), 'bic': float('nan'),
            'converged': False, 'degenerate': True,
            'degeneracies': [f'serie trop courte (n={T} < {2 * _MIN_OBS_REGIME})'],
        }

    eps  = r - float(np.mean(r))
    eps2 = eps ** 2
    regime, bps = _indices_regimes(T, breakpoints)
    n_regimes   = len(bps) + 1

    tailles = [int(np.sum(regime == k)) for k in range(n_regimes)]
    var_k   = [float(np.mean(eps2[regime == k])) if tailles[k] > 0 else float(np.mean(eps2))
               for k in range(n_regimes)]
    sigma2_0 = var_k[0] if var_k[0] > 0 else float(np.mean(eps2))

    # ── Point de départ ──────────────────────────────────────────────────────
    a0 = 0.08 if alpha0 is None or not math.isfinite(float(alpha0)) else float(alpha0)
    b0 = 0.88 if beta0 is None or not math.isfinite(float(beta0)) else float(beta0)
    a0 = float(np.clip(a0, 0.01, 0.30))
    b0 = float(np.clip(b0, 0.10, 0.95))
    if a0 + b0 >= 0.99:                       # faisabilité de la contrainte
        a0, b0 = 0.08, 0.88
    nu0 = 8.0 if nu is None or not math.isfinite(float(nu)) else float(nu)
    nu0 = float(np.clip(nu0, 4.0, 200.0))

    omega0 = [max(v * (1.0 - a0 - b0), 1e-8) for v in var_k]
    theta0 = np.array(omega0 + [a0, b0] + ([nu0] if dist == 't' else []), dtype=float)

    # ── Bornes de boîte + contrainte d'inégalité (schéma component_garch) ────
    bounds = ([(1e-8, None)] * n_regimes
              + [(1e-8, 1.0), (1e-8, 1.0)]
              + ([(2.01, 200.0)] if dist == 't' else []))
    constraints = [
        # Stationnarité en covariance : alpha + beta < 1
        {'type': 'ineq',
         'fun': lambda x, m=n_regimes: 1.0 - x[m] - x[m + 1] - 1e-6},
    ]

    args = (eps2, regime, n_regimes, sigma2_0, dist)
    try:
        res = minimize(_nll_omega_regime, theta0, args=args, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-9, 'disp': False})
    except Exception as exc:
        _log.warning('[omega/regime] echec SLSQP : %s', exc)
        return {
            'omega': [], 'alpha': float('nan'), 'beta': float('nan'), 'nu': None,
            'persistance': float('nan'), 'n_regimes': n_regimes,
            'loglik': float('nan'), 'aic': float('nan'), 'bic': float('nan'),
            'converged': False, 'degenerate': True,
            'degeneracies': [f'exception SLSQP : {exc}'],
        }

    omega_h = [float(v) for v in res.x[:n_regimes]]
    alpha_h = float(res.x[n_regimes])
    beta_h  = float(res.x[n_regimes + 1])
    nu_h    = float(res.x[n_regimes + 2]) if dist == 't' else None

    # ── Dégénérescences ──────────────────────────────────────────────────────
    sur_penalite = bool(res.fun >= _PENALITE_NLL * (1.0 - 1e-9))
    degeneracies = []

    if sur_penalite:
        degeneracies.append(
            "terminaison sur la penalite de vraisemblance : aucune solution "
            "exploitable, loglik/aic/bic non definis"
        )
    if not res.success:
        degeneracies.append(f"SLSQP non converge : {res.message}")
    if alpha_h < _BORNE_BASSE_AB:
        degeneracies.append(
            f"alpha = {alpha_h:.3e} sur sa borne basse (< {_BORNE_BASSE_AB:g}) : "
            f"pas d'effet ARCH identifie"
        )
    if beta_h < _BORNE_BASSE_AB:
        degeneracies.append(
            f"beta = {beta_h:.3e} sur sa borne basse (< {_BORNE_BASSE_AB:g}) : "
            f"pas de persistance identifiee"
        )
    for k, taille in enumerate(tailles):
        if taille < _MIN_OBS_REGIME:
            degeneracies.append(
                f"regime {k} de taille {taille} < {_MIN_OBS_REGIME} observations : "
                f"omega_{k} non identifiable de facon fiable"
            )

    degenerate = bool(degeneracies)

    if sur_penalite:
        loglik = aic = bic = float('nan')
    else:
        loglik   = float(-res.fun)
        n_params = n_regimes + 2 + (1 if dist == 't' else 0)
        aic      = float(-2.0 * loglik + 2.0 * n_params)
        bic      = float(-2.0 * loglik + n_params * math.log(T))

    for lib in degeneracies:
        _log.warning('[omega/regime] %s', lib)

    return {
        'omega':        omega_h,
        'alpha':        alpha_h,
        'beta':         beta_h,
        'nu':           nu_h,
        'persistance':  alpha_h + beta_h,
        'n_regimes':    n_regimes,
        'loglik':       loglik,
        'aic':          aic,
        'bic':          bic,
        'converged':    bool(res.success),
        'degenerate':   degenerate,
        'degeneracies': degeneracies,
    }


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
      'integrate'           — ré-estime un GARCH à omega PAR RÉGIME (Hillebrand
                              2005) et compare la persistance avant/après.
      'off'                 — aucun calcul (équivaut à enabled: false).

    Mode 'integrate' — changement de méthode
    ----------------------------------------
    Les versions antérieures injectaient des dummies de rupture dans l'équation
    de MOYENNE (``arch_model(..., x=dummies)``), arch 8.0.0 ne supportant pas
    GARCHX. Ces dummies captent des sauts de niveau de rendement et laissent la
    persistance α+β inchangée : le mode n'accomplissait pas ce qu'il annonçait.
    L'estimation repose désormais sur ``estimer_garch_omega_par_regime`` — un
    omega libre par régime, α et β communs.
    Si l'estimation échoue ou dégénère, ``persistance_apres`` est ABSENTE du
    dict (jamais une valeur fausse) et le motif est journalisé en WARNING.

    Alerte Lamoureux-Lastrapes — critère
    ------------------------------------
    ``warning_lamoureux`` se déclenche dès UNE rupture avec une persistance
    supérieure à ``seuil_persistance_alerte`` : une rupture de variance franche
    gonfle la persistance au moins autant que plusieurs petites. Le paramètre
    ``seuil_n_ruptures_alerte`` ne commande plus le déclenchement mais le
    niveau renforcé, exposé séparément par ``warning_lamoureux_renforce``.

    Parameters
    ----------
    rendements  : pd.Series        Log-rendements en pourcentage.
    garch_final : ARCHModelResult  Résultat de l'estimation finale.
    config      : dict             Configuration pipeline.
    garch_best  : pd.Series | dict Ligne df_garch du modèle sélectionné.
                  Requis pour mode='integrate' ; ignoré sinon.

    Returns
    -------
    dict avec clés : n_breaks, indices, dates, mode, warning_lamoureux,
    warning_lamoureux_renforce.
    Si mode='integrate' et n_breaks > 0 : + persistance_avant, methode,
    n_regimes, omega_par_regime, ruptures_retenues,
    persistance_apres_degenere, et persistance_apres UNIQUEMENT si l'estimation
    n'a pas dégénéré — sinon degeneracies_apres en donne le motif.
    None si désactivé (enabled=False ou mode='off').
    """
    sb_cfg  = config.get('structural_breaks', {})
    enabled = bool(sb_cfg.get('enabled', True))
    mode    = str(sb_cfg.get('mode', 'diagnostic')).lower()

    if not enabled or mode == 'off':
        return None

    seuil_pers  = float(sb_cfg.get('seuil_persistance_alerte', 0.97))
    seuil_n     = int(sb_cfg.get('seuil_n_ruptures_alerte', 3))

    # Sélection des ruptures pour le mode 'integrate'. `max_dummies` est un
    # alias déprécié : il comptait des RUPTURES (soit n_regimes - 1) à l'époque
    # où le mode injectait des dummies.
    min_obs_regime = int(sb_cfg.get('min_obs_regime', _MIN_OBS_REGIME))
    if sb_cfg.get('max_regimes') is not None:
        max_regimes = int(sb_cfg['max_regimes'])
    elif sb_cfg.get('max_dummies') is not None:
        max_regimes = int(sb_cfg['max_dummies']) + 1
        _log.warning(
            "[ICSS] 'max_dummies' est deprecie (plus aucune dummy n'est injectee) "
            "— utiliser 'max_regimes'. Valeur %d interpretee comme %d regimes.",
            int(sb_cfg['max_dummies']), max_regimes,
        )
    else:
        max_regimes = _MAX_REGIMES_DEFAUT

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

    # Alerte dès UNE rupture : une rupture de variance franche gonfle la
    # persistance au moins autant que plusieurs petites. `seuil_n` ne commande
    # plus le déclenchement mais le niveau RENFORCÉ de l'alerte.
    pers_elevee         = (not math.isnan(pers)) and pers > seuil_pers
    warning_ll          = bool(pers_elevee and n_breaks >= 1)
    warning_ll_renforce = bool(pers_elevee and n_breaks >= seuil_n)

    if warning_ll:
        _log.warning(
            '[ICSS] %d rupture(s) de variance détectée(s) — pers=%.4f > %.2f%s. '
            'La persistance est potentiellement gonflée par des changements de '
            'régime non modélisés (Lamoureux & Lastrapes 1990, JoE).',
            n_breaks, pers, seuil_pers,
            f' — ALERTE RENFORCEE (>= {seuil_n} ruptures)' if warning_ll_renforce else '',
        )
    else:
        _log.info('[ICSS] %d rupture(s) | pers=%.4f | alerte L&L: NON',
                  n_breaks, pers)

    result: dict = {
        'n_breaks':         n_breaks,
        'indices':          breakpoints,
        'dates':            dates,
        'mode':             mode,
        'warning_lamoureux':          warning_ll,
        'warning_lamoureux_renforce': warning_ll_renforce,
    }

    if mode != 'integrate' or garch_best is None or n_breaks == 0:
        return result

    # ── mode='integrate' — ré-estimation à omega par régime ───────────────────
    # Spécification de Hillebrand (2005) : un omega propre à chaque régime,
    # alpha et beta communs. Remplace l'injection de dummies en équation de
    # MOYENNE, qui captait des sauts de niveau de rendement et laissait la
    # persistance inchangée — arch 8.0.0 ne supporte pas GARCHX.
    try:
        from tickerlab.core.rapport._stats import persistance_garch as _pers_fn

        bd = garch_best.to_dict() if hasattr(garch_best, 'to_dict') else dict(garch_best)
        nom_mod    = str(bd['modele'])
        pers_avant = float(_pers_fn(nom_mod, garch_final.params))

        # Sélection par ESPACEMENT : une troncature par rang retiendrait les
        # ruptures les plus précoces et produirait des régimes trop courts pour
        # que les omega soient identifiables (cf. selectionner_ruptures_espacees).
        bps_cap = selectionner_ruptures_espacees(
            breakpoints, len(rend_clean),
            min_obs_regime=min_obs_regime, max_regimes=max_regimes,
        )
        result['ruptures_retenues'] = list(bps_cap)

        if not bps_cap:
            _log.warning(
                '[ICSS integrate] aucune rupture ne laisse %d observations de '
                'part et d\'autre (%d rupture(s) detectee(s)) — pas de regime a '
                'estimer, persistance_apres NON renseignee.',
                min_obs_regime, n_breaks,
            )
            result['methode']                    = 'omega_par_regime'
            result['n_regimes']                  = 1
            result['omega_par_regime']           = []
            result['persistance_apres_degenere'] = True
            result['degeneracies_apres']         = [
                f'aucune rupture espacee d au moins {min_obs_regime} observations'
            ]
            result['persistance_avant'] = round(pers_avant, 6)
            return result

        params_g = garch_final.params
        est = estimer_garch_omega_par_regime(
            rend_clean, bps_cap,
            dist=str(bd.get('dist', 'normal')),
            nu=float(params_g['nu']) if 'nu' in list(params_g.index) else None,
            alpha0=float(params_g.get('alpha[1]', float('nan'))),
            beta0=float(params_g.get('beta[1]',  float('nan'))),
        )

        result['persistance_avant']          = round(pers_avant, 6)
        result['methode']                    = 'omega_par_regime'
        result['n_regimes']                  = int(est['n_regimes'])
        result['omega_par_regime']           = [round(v, 8) for v in est['omega']]
        result['persistance_apres_degenere'] = bool(est['degenerate'])

        if est['degenerate'] or not math.isfinite(est['persistance']):
            # Repli : mieux vaut PAS de persistance_apres qu'une fausse.
            _log.warning(
                '[ICSS integrate] estimation omega/regime degeneree ou non '
                'exploitable — persistance_apres NON renseignee. Motifs : %s',
                ' | '.join(est['degeneracies']) or 'persistance non finie',
            )
            result['degeneracies_apres'] = est['degeneracies']
        else:
            result['persistance_apres'] = round(float(est['persistance']), 6)
            _log.info(
                '[ICSS integrate] pers. avant=%.4f → après=%.4f (Δ=%.4f, '
                '%d régime(s), omega=%s)',
                pers_avant, est['persistance'], est['persistance'] - pers_avant,
                est['n_regimes'],
                '[' + ', '.join(f'{v:.4g}' for v in est['omega']) + ']',
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
