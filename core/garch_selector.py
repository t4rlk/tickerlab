# -*- coding: utf-8 -*-
"""Selection et estimation des modeles GARCH (grille etendue)."""
import logging
import re
import math
import warnings
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm as _s_norm, chi2 as _s_chi2, t as _s_t_dist
from scipy.stats import rankdata as _s_rankdata

warnings.filterwarnings('ignore')

from tickerlab.core.rapport._stats import sign_bias_test, persistance_garch

__etape_version__ = '1'
_log = logging.getLogger(__name__)


# ── Catalogue des modeles ─────────────────────────────────────────────────────
# power_init : None  = pas de parametre power (Garch/EGARCH)
#              float = valeur initiale du parametre delta (APARCH/TGARCH)
# Note TGARCH : Zakoian (1994) definit TGARCH avec delta=1 fixe. Dans arch,
# delta est estime librement ; power_init=1.0 est la valeur de depart MLE.
MODELES_DISPO = {
    'GARCH':     {'vol': 'Garch',  'o': 0, 'power_init': None},
    'GJR-GARCH': {'vol': 'Garch',  'o': 1, 'power_init': None},
    'EGARCH':    {'vol': 'EGARCH', 'o': 1, 'power_init': None},
    'TGARCH':    {'vol': 'APARCH', 'o': 1, 'power_init': 1.0},
    'APARCH':    {'vol': 'APARCH', 'o': 1, 'power_init': 2.0},
}

# Modeles APARCH : necessitent options={'maxiter':200} et ont un parametre delta
_MODELES_APARCH = {'TGARCH', 'APARCH'}

# Regex parametres par categorie
_RE_VOL_PARAMS  = re.compile(r'^(omega|alpha\[|beta\[|gamma\[|delta)')
_RE_DIST_PARAMS = re.compile(r'^(nu|eta|lambda)')   # Student/GED: nu ; skewt: eta, lambda
_RE_MEAN_PARAMS = re.compile(r'^mu$')


# ── Grille principale ─────────────────────────────────────────────────────────

def grid_search_garch(rendements, modeles, distributions,
                      p_max=3, q_max=3, seuil_significativite=0.05,
                      inclure_figarch=False, critere_significativite='variance_seule',
                      tolerance_aic_parcimonie=10.0, seuil_engle_ng=0.05,
                      critere_information='BIC', tolerance_delta_critere_brut=2.0,
                      specification_tests=None, validation_oos=None,
                      build_z_cache=False, **kwargs):
    """
    Grille GARCH(p,o,q) x distribution triee par AIC (grille etendue).

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements en pourcentage.
    modeles : list of str
        Noms des modeles parmi MODELES_DISPO.
    distributions : list of str
        Distributions : 'normal', 't', 'skewt', 'ged'.
    p_max : int
        Ordre ARCH maximal (defaut 3).
    q_max : int
        Ordre GARCH maximal (defaut 3).
    seuil_significativite : float
        Seuil p-valeur pour la significativite des coefs.
    inclure_figarch : bool
        Si True, estime FIGARCH(1,d,1) en complement (hors grille principale).
    critere_significativite : str
        Ignore ici (utilise dans selectionner_meilleur). Parametre accepte
        pour compatibilite avec **config['garch'].

    Returns
    -------
    pd.DataFrame
        Colonnes : modele, vol, o, p, q, dist, AIC, BIC, AIC_norm, BIC_norm,
        max_pval_vol, max_pval_dist, max_pval_mean, max_pval_global,
        tous_sig, tous_sig_vol, tous_sig_global, power.
        Trie par AIC croissant.
    """
    n = len(rendements.dropna())
    resultats = []
    z_cache: dict = {}

    for nom_mod in modeles:
        if nom_mod not in MODELES_DISPO:
            _log.warning('  [WARN] Modele inconnu ignore : %s', nom_mod)
            continue

        m = MODELES_DISPO[nom_mod]
        vol        = m['vol']
        o          = m['o']
        power_init = m['power_init']
        est_aparch = nom_mod in _MODELES_APARCH

        for dist in distributions:
            for p in range(1, p_max + 1):
                for q in range(1, q_max + 1):
                    try:
                        model_kwargs = {
                            'vol': vol, 'p': p, 'o': o, 'q': q, 'dist': dist
                        }
                        if power_init is not None:
                            model_kwargs['power'] = power_init

                        fit_kwargs = {'disp': 'off'}
                        if est_aparch:
                            fit_kwargs['options'] = {'maxiter': 200}

                        fit = arch_model(
                            rendements, **model_kwargs
                        ).fit(**fit_kwargs)

                        pv_vol  = {k: float(v) for k, v in fit.pvalues.items()
                                   if _RE_VOL_PARAMS.match(k)}
                        pv_dist = {k: float(v) for k, v in fit.pvalues.items()
                                   if _RE_DIST_PARAMS.match(k)}
                        pv_mean = {k: float(v) for k, v in fit.pvalues.items()
                                   if _RE_MEAN_PARAMS.match(k)}

                        max_p_vol  = max(pv_vol.values())  if pv_vol  else 0.0
                        max_p_dist = max(pv_dist.values()) if pv_dist else 0.0
                        max_p_mean = max(pv_mean.values()) if pv_mean else 0.0
                        max_p_glob = max(max_p_vol, max_p_dist, max_p_mean)

                        power_est = (
                            float(fit.params.get('delta', math.nan))
                            if est_aparch else math.nan
                        )

                        pers = persistance_garch(nom_mod, fit.params)

                        seuil = seuil_significativite
                        resultats.append({
                            'modele':          nom_mod,
                            'vol':             vol,
                            'o':               o,
                            'p':               p,
                            'q':               q,
                            'dist':            dist,
                            'AIC':             fit.aic,
                            'BIC':             fit.bic,
                            'AIC_norm':        round(fit.aic / n, 6),
                            'BIC_norm':        round(fit.bic / n, 6),
                            'max_pval_vol':    round(max_p_vol,  4),
                            'max_pval_dist':   round(max_p_dist, 4),
                            'max_pval_mean':   round(max_p_mean, 4),
                            'max_pval_global': round(max_p_glob, 4),
                            'tous_sig':        max_p_vol  < seuil,
                            'tous_sig_vol':    max_p_vol  < seuil,
                            'tous_sig_global': max_p_glob < seuil,
                            'power':           power_est,
                            'persistance':     round(pers, 6) if not math.isnan(pers) else math.nan,
                        })
                        if build_z_cache:
                            _z_bc = (fit.resid / fit.conditional_volatility).dropna().values
                            z_cache[(nom_mod, dist, p, q, o)] = (_z_bc, dict(fit.params))

                    except Exception as exc:
                        _log.warning(
                            'Estimation GARCH échouée — spec=%s(%s,%s,%s) dist=%s : %s',
                            nom_mod, p, q, o, dist, exc,
                        )

    df = pd.DataFrame(resultats).sort_values('AIC').reset_index(drop=True)

    # FIGARCH hors-grille : une seule specification (p=1, q=1)
    if inclure_figarch:
        rows_fig = []
        for dist in distributions:
            row = _estimer_figarch(rendements, p=1, q=1, dist=dist,
                                   seuil=seuil_significativite)
            if row is not None:
                rows_fig.append({k: v for k, v in row.items() if k != 'fit_obj'})
        if rows_fig:
            df_fig = pd.DataFrame(rows_fig)
            df = df_fig if df.empty else pd.concat([df, df_fig], ignore_index=True)
            df = df.sort_values('AIC').reset_index(drop=True)

    if build_z_cache:
        return df, z_cache
    return df


# ── FIGARCH standalone ────────────────────────────────────────────────────────

def _estimer_figarch(rendements, p=1, q=1, dist='t', max_iter=200, seuil=0.05):
    """
    Estimation FIGARCH(p,d,q) isolee — non appelee par grid_search_garch
    sauf si inclure_figarch=True.

    Retourne un dict de resultats ou None si echec/non-convergence.
    Cle 'fit_obj' contient l'objet ARCHModelResult pour usage ulterieur.
    """
    n = len(rendements.dropna())
    try:
        fit = arch_model(
            rendements, vol='FIGARCH', p=p, q=q, dist=dist
        ).fit(disp='off', options={'maxiter': max_iter})

        pv_vol  = {k: float(v) for k, v in fit.pvalues.items()
                   if k in {'omega', 'd'} or k.startswith(('phi[', 'beta['))}
        pv_dist = {k: float(v) for k, v in fit.pvalues.items()
                   if _RE_DIST_PARAMS.match(k)}
        pv_mean = {k: float(v) for k, v in fit.pvalues.items()
                   if _RE_MEAN_PARAMS.match(k)}

        max_p_vol  = max(pv_vol.values())  if pv_vol  else 0.0
        max_p_dist = max(pv_dist.values()) if pv_dist else 0.0
        max_p_mean = max(pv_mean.values()) if pv_mean else 0.0
        max_p_glob = max(max_p_vol, max_p_dist, max_p_mean)

        return {
            'modele':          'FIGARCH',
            'vol':             'FIGARCH',
            'o':               0,
            'p':               p,
            'q':               q,
            'dist':            dist,
            'AIC':             fit.aic,
            'BIC':             fit.bic,
            'AIC_norm':        round(fit.aic / n, 6),
            'BIC_norm':        round(fit.bic / n, 6),
            'max_pval_vol':    round(max_p_vol,  4),
            'max_pval_dist':   round(max_p_dist, 4),
            'max_pval_mean':   round(max_p_mean, 4),
            'max_pval_global': round(max_p_glob, 4),
            'tous_sig':        max_p_vol  < seuil,
            'tous_sig_vol':    max_p_vol  < seuil,
            'tous_sig_global': max_p_glob < seuil,
            'power':           math.nan,
            'fit_obj':         fit,
        }
    except Exception as e:
        _log.warning('  [WARN] FIGARCH(%d,d,%d) [%s] echec : %s', p, q, dist, e)
        return None


# ── Selection ────────────────────────────────────────────────────────────────

def filtrer_validite_statistique(df_garch: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Etage 1 : filtre les modeles valides selon trois criteres hierarchiques.

    1. Convergence     : AIC non-NaN (garde-fou explicite).
    2. Significativite : tous_sig_vol == True (params variance p < seuil).
    3. Stationnarite   : persistance < 0.9999 (condition necessaire covariance-stationnarite).

    Parameters
    ----------
    df_garch : pd.DataFrame   Grille complete (sortie grid_search_garch).
    config   : dict           Configuration pipeline (section 'garch').

    Returns
    -------
    pd.DataFrame   Sous-ensemble des modeles valides, ordre AIC conserve.
    """
    # 1. Convergence
    df = df_garch[df_garch['AIC'].notna()].copy()
    n_conv = len(df)

    # 2. Significativite variance
    col_sig = 'tous_sig_vol' if 'tous_sig_vol' in df.columns else 'tous_sig'
    df = df[df[col_sig] == True].copy()
    n_sig = len(df)

    # 3. Stationnarite conditionnelle
    if 'persistance' in df.columns:
        df = df[df['persistance'].apply(
            lambda x: not (isinstance(x, float) and math.isnan(x)) and float(x) < 0.9999
        )].copy()
    n_stat = len(df)

    _log.info('  [FILTRE] %d converges | %d sig_vol=True | %d stationnaires (pers<1)',
              n_conv, n_sig, n_stat)
    return df


def _estimer_z(rendements, row):
    """Estime le modele et retourne les residus standardises z_t."""
    kw = {
        'vol':  row['vol'],
        'p':    int(row['p']),
        'o':    int(row['o']),
        'q':    int(row['q']),
        'dist': row['dist'],
    }
    fit_kw = {'disp': 'off'}
    power_val = row.get('power', None)
    if row['vol'] == 'APARCH':
        try:
            pv = float(power_val)
        except (TypeError, ValueError):
            pv = 2.0
        kw['power'] = pv if not math.isnan(pv) else 2.0
        fit_kw['options'] = {'maxiter': 500}
    fit = arch_model(rendements, **kw).fit(**fit_kw)
    z = (fit.resid / fit.conditional_volatility).dropna().values
    return z


def _estimer_z_et_params(rendements, row):
    """Estime le modele et retourne (z, params_dict) pour usage ulterieur."""
    kw = {
        'vol':  row['vol'],
        'p':    int(row['p']),
        'o':    int(row['o']),
        'q':    int(row['q']),
        'dist': row['dist'],
    }
    fit_kw = {'disp': 'off'}
    power_val = row.get('power', None)
    if row['vol'] == 'APARCH':
        try:
            pv = float(power_val)
        except (TypeError, ValueError):
            pv = 2.0
        kw['power'] = pv if not math.isnan(pv) else 2.0
        fit_kw['options'] = {'maxiter': 500}
    fit = arch_model(rendements, **kw).fit(**fit_kw)
    z = (fit.resid / fit.conditional_volatility).dropna().values
    return z, dict(fit.params)


def _engle_ng_sur_modele(rendements, row):
    """[LEGACY] Retourne p_joint Engle-Ng. Conserve pour compatibilite."""
    try:
        z = _estimer_z(rendements, row)
        sbt = sign_bias_test(z)
        return float(sbt.get('joint_pval', 1.0))
    except Exception as e:
        nom = f"{row.get('modele','?')}({row.get('p','?')},{row.get('o','?')},{row.get('q','?')})"
        _log.warning('  [WARN] Engle-Ng sur %s : %s', nom, e)
        return float('nan')


def berkowitz_test(
    z: np.ndarray,
    dist_name: str,
    params: dict,
) -> dict:
    """
    Test LR de Berkowitz (2001) sur la distribution des innovations GARCH.

    Transforme les résidus standardisés z_t via la CDF du modèle (PIT),
    puis applique la transformation normale inverse : x_t = Φ⁻¹(F(z_t)).
    Sous H₀ (distribution correctement spécifiée), x_t ~ N(0,1) i.i.d.
    Le test régresse x_t sur x_{t-1} (AR(1)) et teste μ=ρ=0, σ²=1
    conjointement via LR ~ χ²(3).

    Parameters
    ----------
    z : np.ndarray
        Résidus standardisés z_t = ε_t/σ_t.
    dist_name : str
        Nom de la distribution GARCH ('normal', 't', 'skewt', 'ged').
    params : dict
        Paramètres estimés du modèle (doit contenir 'nu' pour Student/skewt).

    Returns
    -------
    dict avec 'LR_stat', 'pvalue', 'mu_hat', 'rho_hat', 'sigma2_hat'.

    References
    ----------
    Berkowitz, J. (2001). Testing density forecasts, with applications to
        risk management. Journal of Business & Economic Statistics,
        19(4), 465–474.
    """
    z_arr = np.asarray(z, dtype=float)
    z_arr = z_arr[np.isfinite(z_arr)]
    T = len(z_arr)
    if T < 20:
        return {'LR_stat': float('nan'), 'pvalue': float('nan'),
                'mu_hat': float('nan'), 'rho_hat': float('nan'),
                'sigma2_hat': float('nan')}

    nu = float(params.get('nu', 4.0))
    if math.isnan(nu) or nu <= 2.0:
        nu = 4.0

    # Étape 1 : transformation PIT u_t = F(z_t)
    dist_lc = dist_name.lower()
    if dist_lc == 'normal':
        u = _s_norm.cdf(z_arr)
    elif dist_lc == 't':
        u = _s_t_dist.cdf(z_arr, df=nu)
    else:
        # Empirique pour skewt/ged (rang uniforme)
        ranks = _s_rankdata(z_arr)
        u = ranks / (T + 1.0)

    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    # Étape 2 : quantile normal x_t = Φ⁻¹(u_t)
    x = _s_norm.ppf(u)

    # Étape 3 : AR(1) par MCO
    x1 = x[1:]
    x0 = x[:-1]
    T1 = len(x1)
    X_reg = np.column_stack([np.ones(T1), x0])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_reg, x1, rcond=None)
        mu_hat, rho_hat = float(beta[0]), float(beta[1])
    except np.linalg.LinAlgError:
        return {'LR_stat': float('nan'), 'pvalue': float('nan'),
                'mu_hat': float('nan'), 'rho_hat': float('nan'),
                'sigma2_hat': float('nan')}

    resid   = x1 - mu_hat - rho_hat * x0
    sig2_h  = float(np.mean(resid ** 2))
    if sig2_h <= 0.0:
        return {'LR_stat': float('nan'), 'pvalue': float('nan'),
                'mu_hat': round(mu_hat, 4), 'rho_hat': round(rho_hat, 4),
                'sigma2_hat': float('nan')}

    # Étape 4 : statistique LR ~ χ²(3)
    ll_h0 = -0.5 * T1 * math.log(2 * math.pi) - 0.5 * float(np.sum(x1 ** 2))
    ll_h1 = (-0.5 * T1 * math.log(2 * math.pi)
             - 0.5 * T1 * math.log(sig2_h)
             - 0.5 / sig2_h * float(np.sum(resid ** 2)))
    LR  = 2.0 * (ll_h1 - ll_h0)
    pv  = float(1.0 - _s_chi2.cdf(LR, df=3)) if math.isfinite(LR) else float('nan')

    return {
        'LR_stat':    round(LR,      4) if math.isfinite(LR)  else LR,
        'pvalue':     round(pv,      4) if math.isfinite(pv)  else pv,
        'mu_hat':     round(mu_hat,  4),
        'rho_hat':    round(rho_hat, 4),
        'sigma2_hat': round(sig2_h,  4),
    }


def tester_specification(rendements, row, config: dict, z_cache=None) -> dict:
    """
    Etage 2 : tests de specification simultanes sur les residus standardises z_t.

    Applique trois tests independants et retourne un verdict booleen conjoint.
    Aucune logique sequentielle — chaque modele est evalue en une seule passe.

    Si config['garch']['score_composite']['enabled'] est True, calcule aussi
    le test de Berkowitz (2001) sur la meme estimation (sans surcoût).

    Tests appliques :
    - Ljung-Box sur z_t     : H0 absence d'autocorrelation (Box-Pierce 1970)
    - Ljung-Box sur z_t^2   : H0 absence d'effet ARCH residuel (Engle 1982)
    - Engle-Ng joint        : H0 absence d'asymetrie residuelle (Engle-Ng 1993)
    - Berkowitz PIT LR      : H0 distribution correctement specifiee (optionnel)

    Parameters
    ----------
    rendements : pd.Series   Log-rendements.
    row        : pd.Series   Ligne du df_garch (vol, p, o, q, dist, power).
    config     : dict        Configuration pipeline.

    Returns
    -------
    dict avec cles : lb_z_pval, lb_z2_pval, engle_ng_pval, tous_passent,
                     berkowitz_pval (NaN si score_composite desactive).

    References
    ----------
    Box, G.E.P. & Pierce, D.A. (1970). JASA 65:332.
    Engle, R.F. & Ng, V.K. (1993). Journal of Finance, 48(5), 1749-1778.
    Berkowitz, J. (2001). JBES 19(4), 465-474.
    """
    nan_result = {
        'lb_z_pval': float('nan'), 'lb_z2_pval': float('nan'),
        'engle_ng_pval': float('nan'), 'tous_passent': False,
        'berkowitz_pval': float('nan'),
    }
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        gcfg     = config.get('garch', {})
        spec_cfg = gcfg.get('specification_tests', {})
        lb_z_cfg  = spec_cfg.get('ljung_box_z',  {'lags': 10, 'seuil_p': 0.05})
        lb_z2_cfg = spec_cfg.get('ljung_box_z2', {'lags': 10, 'seuil_p': 0.05})
        eng_cfg   = spec_cfg.get('engle_ng',     {'seuil_p': 0.05})

        lags_z   = int(lb_z_cfg.get('lags',   10))
        lags_z2  = int(lb_z2_cfg.get('lags',  10))
        seuil_z  = float(lb_z_cfg.get('seuil_p',  0.05))
        seuil_z2 = float(lb_z2_cfg.get('seuil_p', 0.05))
        seuil_ng = float(eng_cfg.get('seuil_p',   0.05))

        sc_enabled = bool(gcfg.get('score_composite', {}).get('enabled', False))

        _cache_key = (str(row.get('modele', '')), str(row.get('dist', '')),
                      int(row['p']), int(row['q']), int(row['o']))
        _cached_z = z_cache.get(_cache_key) if z_cache else None

        if sc_enabled:
            if _cached_z is not None:
                z, fit_params = _cached_z
            else:
                z, fit_params = _estimer_z_et_params(rendements, row)
            berk = berkowitz_test(z, str(row.get('dist', 'normal')), fit_params)
            berk_pval = float(berk.get('pvalue', float('nan')))
        else:
            if _cached_z is not None:
                z, _ = _cached_z
            else:
                z = _estimer_z(rendements, row)
            berk_pval = float('nan')

        lb_z_p  = float(acorr_ljungbox(z,    lags=[lags_z],  return_df=True)['lb_pvalue'].iloc[-1])
        lb_z2_p = float(acorr_ljungbox(z**2, lags=[lags_z2], return_df=True)['lb_pvalue'].iloc[-1])
        eng_p   = float(sign_bias_test(z).get('joint_pval', 1.0))

        tous = (lb_z_p >= seuil_z) and (lb_z2_p >= seuil_z2) and (eng_p >= seuil_ng)
        return {
            'lb_z_pval':     lb_z_p,
            'lb_z2_pval':    lb_z2_p,
            'engle_ng_pval': eng_p,
            'tous_passent':  tous,
            'berkowitz_pval': berk_pval,
        }
    except Exception as e:
        nom = f"{row.get('modele','?')}({row.get('p','?')},{row.get('o','?')},{row.get('q','?')})"
        _log.warning('  [WARN] tester_specification %s : %s', nom, e)
        return nan_result


def selectionner_meilleur(df_garch, rendements=None, config=None,
                           tolerance_aic_parcimonie=10.0,
                           seuil_engle_ng=0.05,
                           critere='variance_seule',
                           z_cache=None):
    """
    Selection du meilleur modele GARCH.

    Deux chemins selon la presence de config :

    config fourni (chemin scientifique) :
        Etage 1 : filtrer_validite_statistique (convergence + sig_vol + stationnarite)
        Etage 2 : tester_specification simultanee (LB z_t, LB z_t^2, Engle-Ng)
        Etage 3 : tri par IC (BIC par defaut), fenetre B&A brute, puis complexite
        Motif : 'spec_OK + BIC + parcimonie'

    config=None (chemin legacy, backward compat) :
        Filtre sig_vol=True, tri AIC, fenetre B&A, Engle-Ng sequentielle.

    Parameters
    ----------
    df_garch : pd.DataFrame
        Grille GARCH complete (sortie grid_search_garch).
    rendements : pd.Series or None
        Log-rendements (requis pour les tests de specification).
    config : dict or None
        Configuration pipeline complete. Si None, chemin legacy.
    tolerance_aic_parcimonie : float
        [LEGACY] Fenetre B&A en unites AIC (chemin legacy uniquement).
    seuil_engle_ng : float
        [LEGACY] Seuil Engle-Ng sequentiel (chemin legacy uniquement).
    critere : str
        [LEGACY] 'variance_seule' | 'global' (chemin legacy uniquement).

    Returns
    -------
    tuple : (best_row : pd.Series, motif : str, trace : list)
    """
    if config is not None:
        return _selectionner_scientifique(df_garch, rendements, config, z_cache=z_cache)

    # ── Chemin LEGACY ────────────────────────────────────────────────────────
    if critere == 'global' and 'tous_sig_global' in df_garch.columns:
        col_sig = 'tous_sig_global'
    elif 'tous_sig_vol' in df_garch.columns:
        col_sig = 'tous_sig_vol'
    else:
        col_sig = 'tous_sig'

    trace = []
    df_sig = df_garch[df_garch[col_sig] == True].copy()

    if len(df_sig) == 0:
        _log.warning('  [WARN] Aucun modele sig_vol=True : fallback AIC global brut')
        return df_garch.iloc[0], 'AIC_seul_AUCUN_SIG', trace

    if rendements is None:
        df_sig['complexite'] = (df_sig['p'].astype(int)
                                + df_sig['o'].astype(int)
                                + df_sig['q'].astype(int))
        df_sig = df_sig.sort_values(['complexite', 'AIC']).reset_index(drop=True)
        return df_sig.iloc[0], 'tous_sig + parcimonie', trace

    df_sig = df_sig.sort_values('AIC').reset_index(drop=True)
    aic_min = float(df_sig['AIC'].iloc[0])

    fenetre = df_sig[df_sig['AIC'] <= aic_min + tolerance_aic_parcimonie].copy()
    fenetre['complexite'] = (fenetre['p'].astype(int)
                             + fenetre['o'].astype(int)
                             + fenetre['q'].astype(int))
    fenetre = fenetre.sort_values(['complexite', 'AIC'], ascending=[True, True])

    candidats_fenetre = fenetre.index.tolist()
    candidats_hors    = [i for i in df_sig.index if i not in set(candidats_fenetre)]
    ordre_test        = candidats_fenetre + candidats_hors

    _log.info('  [SELECT] %d modeles %s=True | fenetre B&A (tol=%s) : %d candidats',
              len(df_sig), col_sig, tolerance_aic_parcimonie, len(fenetre))

    for rang, idx in enumerate(ordre_test):
        candidat = df_sig.loc[idx]
        nom = (f"{candidat['modele']}"
               f"({int(candidat['p'])},{int(candidat['o'])},{int(candidat['q'])})"
               f"[{candidat['dist']}]")
        p_eng   = _engle_ng_sur_modele(rendements, candidat)
        verdict = 'OK' if (not math.isnan(p_eng) and p_eng >= seuil_engle_ng) else 'REJET'
        _log.info('  [SELECT] #%2d %-45s AIC=%.2f | Engle-Ng p=%.4f | %s',
                  rang + 1, nom, candidat['AIC'], p_eng, verdict)
        trace.append({
            'rang':         rang + 1,
            'modele':       str(candidat['modele']),
            'p':            int(candidat['p']),
            'o':            int(candidat['o']),
            'q':            int(candidat['q']),
            'dist':         str(candidat['dist']),
            'AIC':          float(candidat['AIC']),
            'BIC':          float(candidat.get('BIC', float('nan'))),
            'persistance':  float(candidat.get('persistance', float('nan'))),
            'complexite':   int(candidat['p']) + int(candidat['o']) + int(candidat['q']),
            'dans_fenetre': idx in set(candidats_fenetre),
            'lb_z_pval':    float('nan'),
            'lb_z2_pval':   float('nan'),
            'engle_ng_pval': p_eng,
            'tous_passent': not math.isnan(p_eng) and p_eng >= seuil_engle_ng,
            'verdict':      verdict,
        })

        if not math.isnan(p_eng) and p_eng >= seuil_engle_ng:
            motif = ('parcimonie + Engle-Ng OK'
                     if idx in set(candidats_fenetre)
                     else 'B&A + Engle-Ng OK')
            return candidat, motif, trace

    _log.warning('  [WARN] Aucun modele sig_vol=True ne passe Engle-Ng — fallback AIC min')
    return df_sig.iloc[0], 'AIC_seul_engle_ng_echoue', trace


def _selectionner_scientifique(df_garch, rendements, config, z_cache=None):
    """
    Algorithme de selection scientifique (chemin config fourni).

    Etage 1 : validite statistique (convergence + sig_vol + stationnarite)
    Etage 2 : tests de specification simultanees (LB z_t, LB z_t^2, Engle-Ng,
              Berkowitz si score_composite.enabled)
    Etage 3a (legacy)        : tri IC, fenetre B&A, parcimonie
    Etage 3b (score composite): rang pondere AIC+LBz2+EngNg+Berkowitz+complexite
              Active via config['garch']['score_composite']['enabled'] = true.
    """
    gcfg        = config.get('garch', {})
    critere_ic  = gcfg.get('critere_information', 'BIC')
    tol_delta   = float(gcfg.get('tolerance_delta_critere_brut', 2.0))
    sc_cfg      = gcfg.get('score_composite', {})
    sc_enabled  = bool(sc_cfg.get('enabled', False))
    w_aic       = float(sc_cfg.get('w_aic',       0.30))
    w_lb_z2     = float(sc_cfg.get('w_lb_z2',     0.25))
    w_eng       = float(sc_cfg.get('w_engle_ng',  0.15))
    w_berk      = float(sc_cfg.get('w_berkowitz', 0.20))
    w_parc      = float(sc_cfg.get('w_parcimonie',0.10))
    trace       = []

    # ── Etage 1 ──────────────────────────────────────────────────────────────
    df_val = filtrer_validite_statistique(df_garch, config)
    if len(df_val) == 0:
        _log.warning('  [WARN] Aucun modele valide apres filtrage — fallback IC global')
        col_ic_fb = critere_ic if critere_ic in df_garch.columns else 'BIC'
        return df_garch.sort_values(col_ic_fb).iloc[0], 'IC_seul_AUCUN_VALIDE', trace

    # ── Etage 2 ──────────────────────────────────────────────────────────────
    col_ic = critere_ic if critere_ic in df_val.columns else 'BIC'

    if rendements is not None:
        _log.info('  [SPEC] Tests de specification simultanees sur %d modeles...', len(df_val))
        lb_z_l, lb_z2_l, eng_l, pass_l, berk_l = [], [], [], [], []
        for _, row in df_val.iterrows():
            res = tester_specification(rendements, row, config, z_cache=z_cache)
            lb_z_l.append(res['lb_z_pval'])
            lb_z2_l.append(res['lb_z2_pval'])
            eng_l.append(res['engle_ng_pval'])
            pass_l.append(res['tous_passent'])
            berk_l.append(res.get('berkowitz_pval', float('nan')))
            nom = (f"{row['modele']}({int(row['p'])},{int(row['o'])},{int(row['q'])})"
                   f"[{row['dist']}]")
            v = 'PASS' if res['tous_passent'] else 'FAIL'
            _log.info('  [SPEC] %-45s LBz=%.3f LBz2=%.3f EngNg=%.3f -> %s',
                      nom, res['lb_z_pval'], res['lb_z2_pval'],
                      res['engle_ng_pval'], v)
        df_val = df_val.copy()
        df_val['lb_z_pval']     = lb_z_l
        df_val['lb_z2_pval']    = lb_z2_l
        df_val['engle_ng_pval'] = eng_l
        df_val['tous_passent']  = pass_l
        df_val['berkowitz_pval']= berk_l
        df_pass = df_val[df_val['tous_passent'] == True].copy()
        _log.info('  [SPEC] %d/%d modeles passent la specification', len(df_pass), len(df_val))
    else:
        df_val = df_val.copy()
        for col in ('lb_z_pval', 'lb_z2_pval', 'engle_ng_pval', 'berkowitz_pval'):
            df_val[col] = float('nan')
        df_val['tous_passent'] = True
        df_pass = df_val.copy()

    # ── Persistance des colonnes spec dans df_garch (in-place) ───────────────
    # Requis par verdict_specification() : le df_garch retourne par le pipeline
    # doit contenir les resultats des tests de specification pour que
    # all_candidates_failed_spec soit detecte correctement.
    # Modeles non passes par Etage 1 (filtres) → valeurs par defaut NaN/False.
    for _col in ('lb_z_pval', 'lb_z2_pval', 'engle_ng_pval'):
        df_garch[_col] = float('nan')
    df_garch['tous_passent'] = False
    df_garch.loc[df_val.index, 'lb_z_pval']     = df_val['lb_z_pval']
    df_garch.loc[df_val.index, 'lb_z2_pval']    = df_val['lb_z2_pval']
    df_garch.loc[df_val.index, 'engle_ng_pval'] = df_val['engle_ng_pval']
    df_garch.loc[df_val.index, 'tous_passent']  = df_val['tous_passent']

    if len(df_pass) == 0:
        _log.warning('  [WARN] Aucun modele ne passe la specification — fallback %s_min valide',
                     col_ic)
        df_fb = df_val.sort_values(col_ic).reset_index(drop=True)
        best  = df_fb.iloc[0]
        _construire_trace(df_val, trace, col_ic, ic_min=float(df_fb[col_ic].iloc[0]),
                          tol_delta=tol_delta)
        return best, f'{critere_ic}_seul_spec_echoue', trace

    # ── Etage 3 ──────────────────────────────────────────────────────────────
    df_pass = df_pass.sort_values(col_ic).reset_index(drop=True)
    ic_min  = float(df_pass[col_ic].iloc[0])

    fenetre = df_pass[df_pass[col_ic] <= ic_min + tol_delta].copy()
    fenetre['complexite'] = (fenetre['p'].astype(int)
                             + fenetre['o'].astype(int)
                             + fenetre['q'].astype(int))
    n_fen = len(fenetre)

    if sc_enabled and n_fen > 1:
        # Score composite (rang pondere) — Burnham & Anderson (2002), ch. 4
        # rang(x, asc): 1=meilleur (plus bas), rank/n ∈ (0,1]
        # rang(x, desc): 1=meilleur (plus haut p-value = moins de rejet)
        def _r_asc(s: pd.Series) -> np.ndarray:
            return _s_rankdata(s.values) / n_fen
        def _r_desc(s: pd.Series) -> np.ndarray:
            filled = s.fillna(s.min() if s.notna().any() else 0.0)
            return _s_rankdata(-filled.values) / n_fen

        score = (w_aic  * _r_asc( fenetre[col_ic])
               + w_lb_z2 * _r_desc(fenetre['lb_z2_pval'])
               + w_eng   * _r_desc(fenetre['engle_ng_pval'])
               + w_berk  * _r_desc(fenetre['berkowitz_pval'])
               + w_parc  * _r_asc( fenetre['complexite']))
        fenetre = fenetre.copy()
        fenetre['score_composite'] = score
        best_idx = int(fenetre['score_composite'].values.argmin())
        best = fenetre.iloc[best_idx]
        motif = f'spec_OK + {critere_ic} + score_composite'
        _log.info('[SELECT] %s_min=%.3f | fenetre B&A (delta<%.1f) : %d modeles | '
                  'score_composite : selectionne %s(%d,%d,%d)[%s] (score=%.4f)',
                  col_ic, ic_min, tol_delta, n_fen,
                  best['modele'], int(best['p']), int(best['o']), int(best['q']),
                  best['dist'], float(fenetre['score_composite'].iloc[best_idx]))
        _log.info('  [SELECT] %s_min=%.3f | fenetre B&A (delta<%.1f) : %d modeles | '
                  'selectionne %s(%d,%d,%d)[%s]',
                  col_ic, ic_min, tol_delta, n_fen,
                  best['modele'], int(best['p']), int(best['o']),
                  int(best['q']), best['dist'])
    else:
        fenetre = fenetre.sort_values(['complexite', col_ic],
                                      ascending=[True, True]).reset_index(drop=True)
        best  = fenetre.iloc[0]
        motif = f'spec_OK + {critere_ic} + parcimonie'
        nom_best = (f"{best['modele']}({int(best['p'])},{int(best['o'])},{int(best['q'])})"
                    f"[{best['dist']}]")
        _log.info('  [SELECT] %s_min=%.3f | fenetre B&A (delta<%.1f) : %d modeles | '
                  'selectionne %s',
                  col_ic, ic_min, tol_delta, n_fen, nom_best)

    _construire_trace(df_val, trace, col_ic, ic_min=ic_min, tol_delta=tol_delta)
    return best, motif, trace


def _construire_trace(df_val, trace, col_ic, ic_min, tol_delta):
    """Remplit la liste trace a partir de df_val trie par col_ic."""
    df_sorted = df_val.sort_values(col_ic).reset_index(drop=True)
    for rang, (_, row) in enumerate(df_sorted.iterrows()):
        ic_val = float(row.get(col_ic, float('nan')))
        trace.append({
            'rang':           rang + 1,
            'modele':         str(row['modele']),
            'p':              int(row['p']),
            'o':              int(row['o']),
            'q':              int(row['q']),
            'dist':           str(row['dist']),
            'AIC':            float(row.get('AIC',           float('nan'))),
            'BIC':            float(row.get('BIC',           float('nan'))),
            'persistance':    float(row.get('persistance',   float('nan'))),
            'lb_z_pval':      float(row.get('lb_z_pval',     float('nan'))),
            'lb_z2_pval':     float(row.get('lb_z2_pval',    float('nan'))),
            'engle_ng_pval':  float(row.get('engle_ng_pval', float('nan'))),
            'berkowitz_pval': float(row.get('berkowitz_pval',float('nan'))),
            'tous_passent':   bool(row.get('tous_passent', False)),
            'dans_fenetre':   (not math.isnan(ic_val)) and (ic_val <= ic_min + tol_delta),
            'complexite':     int(row['p']) + int(row['o']) + int(row['q']),
        })


# ── Estimation finale ────────────────────────────────────────────────────────

def estimer_final(rendements, vol, o, p, q, dist, power=None, **_):
    """
    Estime le modele GARCH final sur la serie complete.

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements en pourcentage.
    vol : str
        Type de modele ('Garch', 'EGARCH', 'APARCH').
    o : int
        Parametre d'asymetrie.
    p : int
        Ordre ARCH.
    q : int
        Ordre GARCH.
    dist : str
        Distribution des innovations ('normal', 't', 'skewt', 'ged').
    power : float or None
        Parametre delta pour APARCH/TGARCH. None ou NaN = ignore.
    **_ :
        Parametres supplementaires ignores (permet best.to_dict() direct).

    Returns
    -------
    arch.univariate.base.ARCHModelResult
    """
    model_kwargs = {'vol': vol, 'p': p, 'o': o, 'q': q, 'dist': dist}

    try:
        if power is not None and not math.isnan(float(power)):
            model_kwargs['power'] = float(power)
    except (TypeError, ValueError) as exc:
        _log.debug('power ignoré (non convertible en float) : %s', exc)

    est_aparch = vol == 'APARCH'
    fit_kwargs = {'disp': 'off'}
    if est_aparch:
        fit_kwargs['options'] = {'maxiter': 500}

    return arch_model(rendements, **model_kwargs).fit(**fit_kwargs)


# ── Verdict specification───────────────────────────────

def verdict_specification(best_dict: dict, df_garch: 'pd.DataFrame',
                          config: dict) -> dict:
    """
    Evalue la specification du modele retenu (blanchiment des residus).

    Returns un dict avec : spec_ok, lb_z_pval, lb_z2_pval, engle_ng_pval,
    all_candidates_failed_spec, gravite ('aucune'|'mineure'|'majeure'), message.
    """
    import numpy as _np
    seuil = float(config.get('garch', {}).get('seuil_significativite', 0.05))

    lb_z  = float(best_dict.get('lb_z_pval',     float('nan')))
    lb_z2 = float(best_dict.get('lb_z2_pval',    float('nan')))
    eng   = float(best_dict.get('engle_ng_pval',  float('nan')))

    pvals = [p for p in [lb_z, lb_z2, eng] if not _np.isnan(p)]
    spec_ok = all(p > seuil for p in pvals) if pvals else True

    # Detection defensive de la colonne spec dans df_garch
    tol_delta = float(config.get('garch', {}).get('tolerance_delta_critere_brut', 4.0))
    best_aic  = float(best_dict.get('AIC', 0))
    candidats = df_garch[df_garch['AIC'] - best_aic <= tol_delta] if 'AIC' in df_garch.columns else df_garch
    col_spec  = next((c for c in ['tous_passent', 'spec_OK', 'Sp. OK'] if c in candidats.columns), None)
    all_failed = not bool(candidats[col_spec].any()) if col_spec is not None else False

    p_min = min(pvals) if pvals else float('nan')

    if spec_ok:
        gravite = 'aucune'
        msg = 'Residus GARCH correctement blanchis.'
    elif not _np.isnan(p_min) and p_min > 0.01:
        gravite = 'mineure'
        msg = (f'Rejet faible du blanchiment (LB z_t p={lb_z:.3f}, marginal a 5%). '
               f'Surveillance recommandee.')
    else:
        gravite = 'majeure'
        ctx = (' Tous les candidats Burnham-Anderson echouent la specification.'
               if all_failed else '')
        msg = (f'Autocorrelation residuelle non capturee (LB z_t p={lb_z:.3f}).{ctx} '
               f'La VaR dynamique peut etre biaisee meme si Kupiec/Christoffersen passent.')

    # Regle de promotion : si tous les candidats Burnham-Anderson echouent,
    # l'alerte est au minimum 'majeure', meme si p_min > 0.01 (gravite 'mineure').
    # Signal d'asset class non modelisable par GARCH standard plutot que de
    # mauvaise specification d'un modele particulier (Haas-Mittnik-Paolella 2004).
    if all_failed and gravite == 'mineure':
        gravite = 'majeure'
        msg = (f'Tous les candidats Burnham-Anderson echouent la specification '
               f'(LB z_t p={lb_z:.3f}). Bien que le rejet du modele retenu soit marginal, '
               f'l\'echec systematique de la famille GARCH suggere une asset class mal '
               f'modelisable par GARCH standard — investiguer HAR-RV (Corsi 2009), MIDAS, '
               f'ou regime switching (Markov-Switching GARCH, Haas-Mittnik-Paolella 2004).')

    return {
        'spec_ok':                    spec_ok,
        'lb_z_pval':                  lb_z,
        'lb_z2_pval':                 lb_z2,
        'engle_ng_pval':              eng,
        'all_candidates_failed_spec': all_failed,
        'gravite':                    gravite,
        'message':                    msg,
    }
