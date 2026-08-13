"""Calcul VaR et TVaR — toutes méthodes × tous niveaux."""
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

_log = logging.getLogger(__name__)

__etape_version__ = '1'
_BTC001_FLOOR = -99.9  # plancher physique VaR GARCH (rendement simple borné à -100%)


def _apply_btc001_floor(r_g: float, ticker: str = '') -> tuple:
    """
    Floor la VaR GARCH a _BTC001_FLOOR (%) si inferieure. Retourne (r_g_floored, floored_bool).

    Motif : near-IGARCH + distribution t a queues epaisses (nu~3) peuvent produire
    des VaR < -100% physiquement impossibles (rendement simple borne a -100%).
    Reference : BTC-001 dans docs/DETTE_TECHNIQUE.md.
    """
    if r_g < _BTC001_FLOOR:
        warnings.warn(
            f'[BTC-001] VaR GARCH {r_g:.2f}% < {_BTC001_FLOOR}% — floore a {_BTC001_FLOOR}%.'
            f' Actif: {ticker or "?"}. Near-IGARCH + forte volat + queue epaisse.'
            ' Utiliser VaR FHS pour cet actif.',
            UserWarning,
            stacklevel=3,
        )
        return float(_BTC001_FLOOR), True
    return float(r_g), False


def tvar_historique(serie, alpha):
    """
    TVaR historique (moyenne de la queue gauche).

    Parameters
    ----------
    serie : array-like
        Série des rendements.
    alpha : float
        Niveau de confiance (ex: 0.95).

    Returns
    -------
    float
        TVaR au niveau alpha.
    """
    q = np.quantile(serie, 1 - alpha)
    tail = serie[serie <= q]
    return float(tail.mean()) if len(tail) > 0 else float(q)


def var_normale(mu, sigma, alpha):
    """
    VaR sous hypothèse de distribution normale.

    Parameters
    ----------
    mu : float
        Moyenne de la distribution.
    sigma : float
        Écart-type de la distribution.
    alpha : float
        Niveau de confiance.

    Returns
    -------
    float
        VaR au niveau alpha (négative — convention pertes).
    """
    z = norm.ppf(1 - alpha)
    return mu + z * sigma


def var_student(mu, sigma, nu, alpha):
    """
    VaR sous hypothèse de distribution Student STANDARDISÉE (variance unitaire).

    Parameters
    ----------
    mu : float
        Moyenne de la distribution.
    sigma : float
        Écart-type de la distribution. À ne pas confondre avec le paramètre
        d'échelle de la t brute (celui que renvoie scipy.stats.t.fit) :
        écart-type = échelle × sqrt(nu / (nu - 2)).
    nu : float
        Degrés de liberté.
    alpha : float
        Niveau de confiance.

    Returns
    -------
    float
        VaR au niveau alpha (négative — convention pertes).
    """
    z = t_dist.ppf(1 - alpha, df=nu) / np.sqrt(nu / (nu - 2.0))
    return mu + z * sigma


def var_historique(serie, alpha):
    """
    VaR historique (quantile empirique de la queue gauche).

    Parameters
    ----------
    serie : array-like
        Série des rendements.
    alpha : float
        Niveau de confiance (ex: 0.95).

    Returns
    -------
    float
        VaR au niveau alpha (négative — convention pertes).
    """
    return float(np.quantile(serie, 1 - alpha))


def tvar_normale(mu, sigma, alpha):
    """
    TVaR sous hypothèse de distribution normale.

    Parameters
    ----------
    mu : float
        Moyenne de la distribution.
    sigma : float
        Écart-type de la distribution.
    alpha : float
        Niveau de confiance.

    Returns
    -------
    float
        TVaR au niveau alpha.
    """
    z = norm.ppf(1 - alpha)
    return mu - sigma * norm.pdf(z) / (1 - alpha)


def tvar_student(mu, sigma, nu, alpha):
    """
    TVaR sous hypothèse de distribution Student STANDARDISÉE (variance unitaire).

    Parameters
    ----------
    mu : float
        Moyenne de la distribution.
    sigma : float
        Écart-type de la distribution. Même convention que var_student :
        écart-type = échelle × sqrt(nu / (nu - 2)).
    nu : float
        Degrés de liberté.
    alpha : float
        Niveau de confiance.

    Returns
    -------
    float
        TVaR au niveau alpha.
    """
    z = t_dist.ppf(1 - alpha, df=nu)
    s = np.sqrt(nu / (nu - 2.0))
    return mu - sigma * (t_dist.pdf(z, df=nu) / (1 - alpha)) \
                      * (nu + z**2) / (nu - 1) / s


def var_cornish_fisher(mu, sigma, S, K, alpha):
    """
    VaR de Cornish-Fisher (expansion d'Edgeworth ordre 2).

    Inclut un test de monotonie de la transformation z -> zc(z) selon
    Maillard (2018), "A User's Guide to the Cornish-Fisher Expansion".
    La derivee dzc/dz = 1 + z*S/3 + (3z^2-3)*K/24 - (6z^2-5)*S^2/36
    doit rester strictement positive au quantile considere.
    Si non-monotone, retourne NaN.

    Note : Cornish-Fisher diverge typiquement pour kurtosis excessif > 7.

    Reference : Maillard, D. (2018). A User's Guide to the Cornish-Fisher
    Expansion. SSRN 1997178.

    Parameters
    ----------
    mu : float
        Moyenne.
    sigma : float
        Ecart-type.
    S : float
        Skewness (asymetrie).
    K : float
        Kurtosis excessif.
    alpha : float
        Niveau de confiance.

    Returns
    -------
    float
        VaR Cornish-Fisher au niveau alpha, ou NaN si non-monotone.
    """
    z  = norm.ppf(1 - alpha)
    zc = z + (z**2 - 1)*S/6 + (z**3 - 3*z)*K/24 - (2*z**3 - 5*z)*S**2/36
    dzc_dz = 1.0 + z*S/3.0 + (3*z**2 - 3)*K/24.0 - (6*z**2 - 5)*S**2/36.0
    if dzc_dz <= 0:
        import warnings
        warnings.warn(
            f'Cornish-Fisher non-monotone au niveau {alpha:.0%} '
            f'(dzc/dz={dzc_dz:.4f}, K={K:.2f}, S={S:.2f}). '
            'Retour NaN. Kurtosis excessif > ~7 invalide l\'expansion.',
            stacklevel=2,
        )
        return float('nan')
    return mu + zc * sigma


def construire_df_vol(rendements, garch_final, alpha=0.99):
    """
    Construit le DataFrame rendements + volatilite conditionnelle + VaR/TVaR dynamiques.

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements en pourcentage.
    garch_final : arch.univariate.base.ARCHModelResult
        Résultat du modèle GARCH estimé sur la série complète.
    alpha : float, optional
        Niveau de confiance pour VaR/TVaR (défaut 0.99).

    Returns
    -------
    pd.DataFrame
        Index = dates. Colonnes : Rendement (%), Vol. conditionnelle (%),
        VaR {pct}% GARCH (%), TVaR {pct}% GARCH (%).
    """
    serie = rendements.dropna()
    mu = float(serie.mean())
    vol_cond = garch_final.conditional_volatility
    params = garch_final.params
    resid_std = (garch_final.resid / vol_cond).dropna()
    dist_name = garch_final.model.distribution.name.lower()
    nu_garch = float(params.get('nu', float('nan')))

    # q_z : quantile de la loi standardisee (mu=0, sigma=1), mis a l'echelle
    # par vol_cond plus bas.
    if dist_name == 'normal':
        q_z = var_normale(0.0, 1.0, alpha)
        tvar_z_sc = -norm.pdf(q_z) / (1 - alpha)
    elif dist_name == 't':
        # Les innovations GARCH sont standardisees (variance unitaire) : le
        # quantile doit l'etre aussi, d'ou l'appel a var_student/tvar_student
        # plutot qu'au quantile de la t brute.
        q_z = var_student(0.0, 1.0, nu_garch, alpha)
        tvar_z_sc = tvar_student(0.0, 1.0, nu_garch, alpha)
    else:
        z_innov = resid_std.values
        q_z = var_historique(z_innov, alpha)
        tail_z = z_innov[z_innov <= q_z]
        tvar_z_sc = float(tail_z.mean()) if len(tail_z) > 0 else q_z

    var_vec  = mu + q_z * vol_cond.values
    tvar_vec = mu + tvar_z_sc * vol_cond.values
    pct = int(alpha * 100)
    return pd.DataFrame({
        'Rendement (%)':           serie.reindex(vol_cond.index).values,
        'Vol. conditionnelle (%)': vol_cond.values,
        f'VaR {pct}% GARCH (%)':  var_vec.round(4),
        f'TVaR {pct}% GARCH (%)': tvar_vec.round(4),
    }, index=vol_cond.index).round(4)


def calculer_var_tvar(rendements, garch_final, niveaux=None, n_simulations_mc=50_000):
    """
    Calcule VaR et TVaR (toutes méthodes) pour chaque niveau de confiance.

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements en pourcentage.
    garch_final : arch.univariate.base.ARCHModelResult
        Résultat du modèle GARCH estimé.
    niveaux : list of float, optional
        Niveaux de confiance (défaut [0.90, 0.95, 0.99]).
    n_simulations_mc : int, optional
        Nombre de simulations Monte Carlo (défaut 50 000).

    Returns
    -------
    pd.DataFrame
        Index = niveaux formatés (ex: '99%'), colonnes = VaR/TVaR par méthode.
    """
    if niveaux is None:
        niveaux = [0.90, 0.95, 0.99]

    serie = rendements.dropna().values
    mu, sigma = serie.mean(), serie.std()
    S = float(pd.Series(serie).skew())
    K = float(pd.Series(serie).kurt())

    nu_fit, mu_fit_t, sigma_fit_t = t_dist.fit(serie, floc=mu)

    mc_sim     = garch_final.forecast(horizon=1, method='simulation',
                                      simulations=n_simulations_mc)
    mc_returns = mc_sim.simulations.values[-1, :, 0]

    vol_cond = garch_final.conditional_volatility
    last_vol = float(vol_cond.iloc[-1])
    params   = garch_final.params
    nu_garch = float(params.get('nu', nu_fit))

    dist_name = garch_final.model.distribution.name.lower()

    floored_99 = False
    resultats = []
    for alpha in niveaux:
        qh   = np.quantile(serie, 1 - alpha)
        rt_h = tvar_historique(serie, alpha)

        r_n  = norm.ppf(1 - alpha, loc=mu, scale=sigma)
        rt_n = tvar_normale(mu, sigma, alpha)

        # t ajustee par MLE : t_dist.fit renvoie une ECHELLE, tvar_student attend
        # un ECART-TYPE (= echelle * sqrt(nu/(nu-2))). Sans cette conversion la
        # TVaR Student ne correspondrait plus a la VaR Student de la ligne
        # au-dessus, qui est le quantile de la meme loi ajustee.
        r_t  = t_dist.ppf(1 - alpha, df=nu_fit, loc=mu_fit_t, scale=sigma_fit_t)
        rt_t = tvar_student(mu_fit_t, sigma_fit_t * np.sqrt(nu_fit / (nu_fit - 2.0)),
                            nu_fit, alpha)

        r_cf  = var_cornish_fisher(mu, sigma, S, K, alpha)
        if np.isnan(r_cf):
            rt_cf = float('nan')
        else:
            tail  = serie[serie <= r_cf]
            rt_cf = float(tail.mean()) if len(tail) > 0 else r_cf

        if dist_name == 'normal':
            q_g = norm.ppf(1 - alpha)
            r_g = mu + q_g * last_vol
            rt_g = mu - last_vol * norm.pdf(q_g) / (1 - alpha)
        elif dist_name == 't':
            # Innovations GARCH standardisees : quantile et multiplicateur ES
            # standardises eux aussi (cf. construire_df_vol).
            q_g  = var_student(0.0, 1.0, nu_garch, alpha)
            r_g  = mu + q_g * last_vol
            rt_g = mu + tvar_student(0.0, 1.0, nu_garch, alpha) * last_vol
        else:
            r_g    = float(np.quantile(mc_returns, 1 - alpha))
            tail_g = mc_returns[mc_returns <= r_g]
            rt_g   = float(tail_g.mean()) if len(tail_g) > 0 else r_g

        # BTC-001 : floor VaR GARCH au niveau 99% uniquement
        r_g_floored = r_g
        if alpha == 0.99:
            r_g_floored, _floored = _apply_btc001_floor(r_g)
            if _floored:
                floored_99 = True

        r_mc    = float(np.quantile(mc_returns, 1 - alpha))
        tail_mc = mc_returns[mc_returns <= r_mc]
        rt_mc   = float(tail_mc.mean()) if len(tail_mc) > 0 else r_mc

        resultats.append({
            'Niveau':                   f'{int(alpha*100)}%',
            'VaR Historique':           round(qh,   4),
            'TVaR Historique':          round(rt_h, 4),
            'VaR Normale':              round(r_n,  4),
            'TVaR Normale':             round(rt_n, 4),
            'VaR Student':              round(r_t,  4),
            'TVaR Student':             round(rt_t, 4),
            'VaR Cornish-Fisher':       round(r_cf, 4),
            'TVaR CF semi-empirique':   round(rt_cf,4),
            'VaR GARCH':                round(r_g_floored, 4),
            'TVaR GARCH':               round(rt_g, 4),
            'VaR Monte Carlo (1j)':     round(r_mc, 4),
            'TVaR Monte Carlo (1j)':    round(rt_mc,4),
        })

    df_result = pd.DataFrame(resultats).set_index('Niveau')
    df_result.attrs['var99_garch_floored'] = floored_99
    return df_result


def calculer_var_multi_horizon(garch_final, config=None):
    """
    VaR 99% sur horizons multiples : 1, 5, 10, 22 jours.

    Deux approches comparees par horizon H :
    - Regle racine carree de Bale : VaR_H = VaR_1 * sqrt(H)
    - Simulation directe GARCH h-step via arch forecast(horizon=H)

    Parameters
    ----------
    garch_final : arch.univariate.base.ARCHModelResult
    config : dict, optional
        Lit config['var']['horizons'] (defaut [1, 5, 10, 22])
        et config['var']['n_simulations_horizons'] (defaut 10000).

    Returns
    -------
    pd.DataFrame
        Index = horizons (jours), colonnes = VaR sqrt, VaR sim, ecart relatif.
    """
    horizons = [1, 5, 10, 22]
    n_sim = 10_000
    alpha = 0.99
    if config is not None:
        v = config.get('var', {})
        horizons = v.get('horizons', horizons)
        n_sim    = int(v.get('n_simulations_horizons', n_sim))

    dist_name = garch_final.model.distribution.name.lower()
    params    = garch_final.params
    nu_garch  = float(params.get('nu', 4.0))
    last_vol  = float(garch_final.conditional_volatility.iloc[-1])
    mu        = float(garch_final.params.get('mu', 0.0))

    # VaR_1 selon la distribution du modele
    if dist_name == 'normal':
        q1 = float(norm.ppf(1 - alpha))
    elif dist_name == 't':
        q1 = float(t_dist.ppf(1 - alpha, df=nu_garch))
    else:
        resid_std = (garch_final.resid / garch_final.conditional_volatility).dropna().values
        q1 = float(np.quantile(resid_std, 1 - alpha))
    var1_garch = mu + q1 * last_vol

    rows = []
    for H in horizons:
        var_sqrt = var1_garch * np.sqrt(H)
        try:
            fc = garch_final.forecast(horizon=H, method='simulation',
                                      simulations=n_sim, reindex=False)
            paths = fc.simulations.values[-1, :, :]  # (n_sim, H)
            cum_returns = paths.sum(axis=1)           # sum over H steps
            var_sim = float(np.quantile(cum_returns, 1 - alpha))
        except Exception:
            var_sim = float('nan')

        ecart = ((var_sim - var_sqrt) / abs(var_sqrt) * 100
                 if not np.isnan(var_sim) and abs(var_sqrt) > 1e-12
                 else float('nan'))
        rows.append({
            'Horizon (j)':      H,
            'VaR sqrt(H)':      round(var_sqrt, 4),
            'VaR simulation':   round(var_sim, 4) if not np.isnan(var_sim) else float('nan'),
            'Ecart rel. (%)':   round(ecart, 2) if not np.isnan(ecart) else float('nan'),
        })

    return pd.DataFrame(rows).set_index('Horizon (j)')


def calculer_bootstrap_ci_var(rendements, garch_final, config=None):
    """
    Intervalle de confiance bootstrap stationnaire conditionnel pour la VaR GARCH 1j.

    Deux modes :
    - express (config['bootstrap']['express'] = true) : 200 replications, block_length=10,
      niveaux_ic=[0.95], seed=42. Actif meme si enabled=false. Rapide (~2-4 s).
    - complet (config['bootstrap']['enabled'] = true) : n_boot replications, tous niveaux.
      Prend la precedence sur express.

    Bootstrap conditionnel (Pascual-Romo-Ruiz 2006) : parametres GARCH fixes a leur
    estimation ponctuelle. Quantifie l'incertitude de l'echantillonnage des residus,
    non l'incertitude parametrique.

    RNG local via numpy.random.default_rng + seed passe a StationaryBootstrap(seed=rng).
    Aucun effet de bord sur np.random global (isolation avec FHS seed=42).

    Parameters
    ----------
    rendements : pd.Series
    garch_final : arch result
    config : dict, optional

    Returns
    -------
    pd.DataFrame  Index = niveaux ('95%' etc.),
                  colonnes = ['VaR GARCH', 'CI lower', 'CI upper'].
    Retourne DataFrame vide si express=false et enabled=false.
    """
    from numpy.random import default_rng as _default_rng

    boot_cfg = {}
    if config is not None:
        boot_cfg = config.get('bootstrap', {})

    express = bool(boot_cfg.get('express', False))
    enabled = bool(boot_cfg.get('enabled', False))

    _empty = pd.DataFrame(
        columns=['VaR GARCH', 'CI lower', 'CI upper'],
        index=pd.Index([], name='Niveau'),
    )

    if not express and not enabled:
        return _empty

    if enabled:
        n_boot     = int(boot_cfg.get('n_boot', 500))
        block_size = int(boot_cfg.get('block_size', 10))
        niveaux    = [0.95, 0.99]
        ci_level   = float(boot_cfg.get('ci_level', 0.95))
    else:
        n_boot     = int(boot_cfg.get('n_replications', 200))
        block_size = int(boot_cfg.get('block_length', 10))
        niveaux    = [float(v) for v in boot_cfg.get('niveaux_ic', [0.95])]
        ci_level   = niveaux[0] if niveaux else 0.95

    seed_val = boot_cfg.get('seed', None)
    if seed_val is not None:
        rng = _default_rng(int(seed_val))
    else:
        _log.warning(
            'bootstrap.seed=null — mode recherche NON reproductible '
            '(variance MC etudiee).'
        )
        rng = _default_rng()

    try:
        from arch.bootstrap import StationaryBootstrap
    except ImportError:
        import warnings
        warnings.warn('arch.bootstrap non disponible — bootstrap CI ignore.')
        return _empty

    serie     = rendements.dropna().values
    last_vol  = float(garch_final.conditional_volatility.iloc[-1])
    mu        = float(garch_final.params.get('mu', float(serie.mean())))
    dist_name = garch_final.model.distribution.name.lower()
    nu        = float(garch_final.params.get('nu', 4.0))

    tail_alpha = (1 - ci_level) / 2

    rows = []
    for alpha in niveaux:
        niv = f'{int(alpha * 100)}%'

        if dist_name == 'normal':
            q_z = float(norm.ppf(1 - alpha))
        elif dist_name == 't':
            q_z = float(t_dist.ppf(1 - alpha, df=nu))
        else:
            resid_std = (garch_final.resid / garch_final.conditional_volatility).dropna().values
            q_z = float(np.quantile(resid_std, 1 - alpha))

        var_point = mu + q_z * last_vol

        try:
            vol        = garch_final.model.volatility
            params_arr = garch_final.params.values.astype(float)
            bs = StationaryBootstrap(block_size, serie, seed=rng)
            boot_vars = []
            for (x,), _ in bs.bootstrap(n_boot):
                x_arr  = x.reshape(-1).astype(float)
                sigma2 = np.zeros(len(x_arr))
                bc     = vol.backcast(x_arr)
                vb     = vol.variance_bounds(x_arr)
                var_t  = vol.compute_variance(params_arr, x_arr, sigma2, bc, vb)
                boot_vars.append(mu + q_z * float(var_t[-1] ** 0.5))
        except Exception:
            boot_vars = [
                float(np.quantile(rng.choice(serie, size=len(serie)), 1 - alpha))
                for _ in range(n_boot)
            ]

        boot_arr = np.array(boot_vars)
        ci_lo = float(np.quantile(boot_arr, tail_alpha))
        ci_hi = float(np.quantile(boot_arr, 1 - tail_alpha))

        rows.append({
            'Niveau':    niv,
            'VaR GARCH': round(var_point, 4),
            'CI lower':  round(ci_lo, 4),
            'CI upper':  round(ci_hi, 4),
        })

    return pd.DataFrame(rows).set_index('Niveau')
