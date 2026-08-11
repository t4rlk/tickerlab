# -*- coding: utf-8 -*-
"""
Backtesting rolling-window de la VaR GARCH.

Approche :
- Fenetre glissante de taille window_size (defaut 1000j ≈ 4 ans).
- Re-estimation du modele GARCH tous les refit_every jours (defaut 22j ≈ 1 mois).
- Entre deux ré-estimations : parametres fixes de la derniere estimation valide.
- Prevision one-step-ahead par la volatilite conditionnelle (pas de simulation).
- Gestion gracieuse des echecs de convergence : on garde les params precedents.

Sorties :
- df_violations  : pd.DataFrame (date, y_oos, VaR_rolling_95, VaR_rolling_99,
                   Hit_95, Hit_99) sur toute la periode hors fenetre initiale.
- df_params_drift : pd.DataFrame (date_estim × params GARCH estimes).
- stats_rolling   : dict {niveau: {LR_UC, p_UC, LR_IND, p_IND, LR_CC, p_CC,
                    DQ_stat, DQ_pval, N_viol, T_eff, verdict_UC, verdict_CC}}.
"""
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist
from arch import arch_model

from tickerlab.core.backtest import kupiec_test, christoffersen_test


# ── Quantile VaR selon la distribution ────────────────────────────────────────

def _q_z(dist_name: str, nu: float, alpha: float,
         resid_std: np.ndarray | None = None) -> float:
    if dist_name == 'normal':
        return float(norm.ppf(1 - alpha))
    elif dist_name == 't':
        nu_ = nu if nu > 2 else 4.0
        return float(t_dist.ppf(1 - alpha, df=nu_))
    else:
        if resid_std is not None and len(resid_std) > 0:
            return float(np.quantile(resid_std, 1 - alpha))
        return float(norm.ppf(1 - alpha))


# ── DQ test minimal (sans lags pour la serie rolling) ────────────────────────

def _dq_test(hit: np.ndarray, var_t: np.ndarray,
             alpha: float, n_lags: int = 4) -> dict:
    T = len(hit)
    s = n_lags
    if T <= s + 10:
        return {'DQ': float('nan'), 'p_value': float('nan')}
    H = hit[s:]
    X = np.column_stack([
        np.ones(T - s),
        *[hit[s - j - 1: T - j - 1] for j in range(n_lags)],
        var_t[s:],
    ])
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        dq_stat = float(max((H @ X @ XtX_inv @ X.T @ H) / ((1 - alpha) * alpha), 0))
        p_val   = float(1 - chi2.cdf(dq_stat, df=X.shape[1]))
    except Exception:
        dq_stat, p_val = float('nan'), float('nan')
    return {'DQ': dq_stat, 'p_value': p_val}


# ── Rolling backtest principal ─────────────────────────────────────────────────

def backtest_rolling_var(rendements: pd.Series, best: dict,
                         config: dict) -> tuple:
    """
    Rolling-window VaR backtest avec re-estimation periodique.

    Parameters
    ----------
    rendements : pd.Series
        Log-rendements complets (series avec dates en index).
    best : dict
        Parametres du modele GARCH retenu (vol, p, o, q, dist).
    config : dict
        Lit config['rolling_backtest'] pour les parametres.

    Returns
    -------
    tuple (df_violations, df_params_drift, stats_rolling)

    Notes
    -----
    Filtre de plausibilite : les estimations dont au moins un parametre
    depasse |param| > 10 sont considerees divergees (optimiseur mal
    conditionne) et ecartees — les parametres precedents sont conserves.
    """
    rb_cfg     = config.get('rolling_backtest', {})
    window     = int(rb_cfg.get('window_size', 1000))
    refit_ev   = int(rb_cfg.get('refit_every', 22))
    niveaux    = rb_cfg.get('niveaux_test', [0.95, 0.99])

    vol   = best['vol']
    p     = int(best['p'])
    o     = int(best['o'])
    q     = int(best['q'])
    dist  = best['dist']

    serie   = rendements.dropna()
    T       = len(serie)
    min_oos = 50  # predictions OOS minimales pour des statistiques interpretables

    # ── Validation window vs serie ────────────────────────────────────────────
    if window >= T:
        raise ValueError(
            f'rolling_backtest impossible : window_size={window} >= n_total={T}. '
            f'Serie trop courte. Minimum theorique ~500 obs. '
            f'Desactiver rolling_backtest.enabled ou utiliser une serie plus longue.'
        )
    if window >= T - min_oos:
        _new_window = max(int(0.5 * T), 250)
        if _new_window >= T:
            raise ValueError(
                f'rolling_backtest : auto-ajustement impossible '
                f'(n_total={T} trop court meme pour window={_new_window}). '
                f'Desactiver rolling_backtest ou utiliser une serie plus longue.'
            )
        warnings.warn(
            f'rolling_backtest : window_size={window} incompatible '
            f'(n_total={T}, min_oos={min_oos}). '
            f'Auto-ajustement a {_new_window}. '
            f'Robustesse statistique reduite : {_new_window} obs en fenetre rolling '
            f'sur {T - _new_window} predictions OOS. '
            f'Pour audit FRTB, viser >=1000 obs.',
            UserWarning, stacklevel=2
        )
        window = _new_window
    values = serie.values
    dates  = serie.index

    # Dates de re-estimation : premiere a window, puis toutes les refit_ev
    refit_dates_idx = list(range(window, T, refit_ev))
    n_refits        = len(refit_dates_idx)

    # ── Progression ─────────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        _iter = tqdm(refit_dates_idx, desc='Rolling GARCH', unit='estim',
                     ncols=80, leave=True)
    except ImportError:
        import logging as _logging
        _rolling_log = _logging.getLogger('tickerlab.backtest_rolling')
        _iter = refit_dates_idx
        _rolling_log.info('  [Rolling] %d re-estimations (tqdm non installe)...', n_refits)

    # ── Stockage ─────────────────────────────────────────────────────────────
    # violations[t] = {date, y, vol_t, hit_95, hit_99}
    viol_records  = []
    params_records = []

    # Etat courant
    current_params    = None   # None = pas encore estime
    current_params_arr = None  # numpy array pour arch_model.fix()
    current_mu        = 0.0
    current_nu        = 4.0
    current_resid_std = None
    current_q95       = float('nan')
    current_q99       = float('nan')
    last_vol          = float('nan')   # fallback : vol du dernier refit valide

    # Vol propagee entre refits (Bug 4)
    vol_buffer        = None   # array de vol pour [t_refit, t_refit+refit_ev]
    vol_buffer_start  = 0      # indice t du debut du buffer

    i_refit = 0
    n_divergent = 0
    t_fix_total = 0.0          # temps cumule des appels fix() (diagnostic)
    refit_set = set(refit_dates_idx)

    for t in range(window, T):
        # Re-estimation si t est une date de refit
        if t in refit_set:
            sub = serie.iloc[t - window: t]  # pandas Series avec dates
            try:
                fit = arch_model(
                    sub, vol=vol, p=p, o=o, q=q, dist=dist
                ).fit(disp='off', show_warning=False)
                params_series = fit.params  # pandas Series
                params_dict   = dict(params_series)

                # Filtre de plausibilite : params GARCH bornes entre -10 et 10.
                # Un |param| > 10 indique une convergence vers une zone aberrante
                # (optimiseur mal conditionne) — on conserve les params precedents.
                if any(abs(v) > 10
                       for v in params_dict.values()
                       if isinstance(v, (int, float)) and math.isfinite(v)):
                    n_divergent += 1
                    warnings.warn(
                        f'backtest_rolling: estimation a t={t} ({dates[t]}) '
                        f'a converge vers des params aberrants (|param| > 10) '
                        f'— params precedents conserves.'
                    )
                else:
                    current_params     = params_dict
                    current_params_arr = np.asarray(params_series)  # pour fix()
                    current_mu     = float(params_series.get('mu', float(np.mean(sub.values))))
                    current_nu     = float(params_series.get('nu', 4.0))
                    if math.isnan(current_nu) or current_nu <= 2:
                        current_nu = 4.0
                    cv_arr    = np.asarray(fit.conditional_volatility)
                    resid_arr = np.asarray(fit.resid)
                    mask      = (cv_arr > 1e-12) & np.isfinite(cv_arr) & np.isfinite(resid_arr)
                    current_resid_std = resid_arr[mask] / cv_arr[mask]
                    current_resid_std = current_resid_std[np.isfinite(current_resid_std)]
                    last_vol = float(cv_arr[-1])
                    current_q95 = _q_z(dist, current_nu, 0.95, current_resid_std)
                    current_q99 = _q_z(dist, current_nu, 0.99, current_resid_std)

                    # ── Pre-calcul du buffer de vol pour les refit_ev prochains pas ──
                    # Approche fix() : 1 appel par refit sur window+refit_ev obs.
                    # Timing mesure : ~4ms/appel, overhead total ~0.64s pour 163 refits.
                    # Plus robuste que la recursion manuelle pour EGARCH/APARCH/GJR.
                    import time as _time
                    _t0 = _time.perf_counter()
                    try:
                        end_prop    = min(t + refit_ev, T)
                        sub_prop    = serie.iloc[t - window: end_prop]
                        fit_prop    = arch_model(
                            sub_prop, vol=vol, p=p, o=o, q=q, dist=dist
                        ).fix(current_params_arr)
                        cv_prop = np.asarray(fit_prop.conditional_volatility)
                        # cv_prop[0:window] = in-sample ; cv_prop[window:] = OOS propagee
                        if (len(cv_prop) > window
                                and np.all(np.isfinite(cv_prop[window:]))
                                and np.all(cv_prop[window:] > 0)):
                            vol_buffer       = cv_prop[window:]
                            vol_buffer_start = t
                        else:
                            vol_buffer = None
                    except Exception as e_prop:
                        warnings.warn(
                            f'backtest_rolling: propagation vol a t={t} echouee '
                            f'— last_vol constant. ({e_prop})'
                        )
                        vol_buffer = None
                    t_fix_total += _time.perf_counter() - _t0

                    # Enregistrement derive des params
                    rec = {'date_estim': dates[t]}
                    rec.update(current_params)
                    params_records.append(rec)
                    i_refit += 1
            except Exception as e:
                warnings.warn(
                    f'backtest_rolling: estimation a t={t} ({dates[t]}) '
                    f'echouee — params precedents conserves. ({e})'
                )
                # Pas de mise a jour des params, pas de record

        # Prevision one-step-ahead
        if current_params is None:
            continue  # pas encore de premiere estimation

        # sigma_t : vol propagee via fix() si buffer disponible, sinon last_vol
        sigma_t = last_vol
        if vol_buffer is not None:
            buf_idx = t - vol_buffer_start
            if 0 <= buf_idx < len(vol_buffer):
                v = float(vol_buffer[buf_idx])
                if np.isfinite(v) and v > 0:
                    sigma_t = v

        if not np.isfinite(sigma_t) or sigma_t <= 0:
            continue  # pas encore de vol valide

        var95 = current_mu + current_q95 * sigma_t
        var99 = current_mu + current_q99 * sigma_t

        y_t   = values[t]
        hit95 = int(y_t < var95)
        hit99 = int(y_t < var99)

        viol_records.append({
            'date':        dates[t],
            'y':           y_t,
            'VaR_roll_95': var95,
            'VaR_roll_99': var99,
            'Hit_95':      hit95,
            'Hit_99':      hit99,
        })

        last_vol = sigma_t  # mise a jour du fallback pour la prochaine iteration

    import logging as _logging
    _logging.getLogger('tickerlab.backtest_rolling').info(
        '  [Rolling] Refits valides : %d / %d | Divergences filtrees : %d | fix() overhead : %.2fs',
        i_refit, n_refits, n_divergent, t_fix_total)

    df_violations = pd.DataFrame(viol_records)
    if not df_violations.empty:
        df_violations = df_violations.set_index('date')

    df_params_drift = pd.DataFrame(params_records)
    if not df_params_drift.empty:
        df_params_drift = df_params_drift.set_index('date_estim')

    # ── Statistiques rolling ─────────────────────────────────────────────────
    stats_rolling = {}
    if not df_violations.empty:
        for alpha, col_hit, col_var in [
            (0.95, 'Hit_95', 'VaR_roll_95'),
            (0.99, 'Hit_99', 'VaR_roll_99'),
        ]:
            niv     = f'{int(alpha*100)}%'
            hits    = df_violations[col_hit].values
            var_t   = df_violations[col_var].values
            T_eff   = len(hits)
            N_viol  = int(hits.sum())

            lr_uc,  p_uc  = kupiec_test(N_viol, T_eff, alpha)
            lr_ind, p_ind = christoffersen_test(hits)
            lr_cc   = (lr_uc + lr_ind
                       if not (math.isnan(lr_uc) or math.isnan(lr_ind))
                       else float('nan'))
            p_cc    = (float(1 - chi2.cdf(lr_cc, df=2))
                       if not math.isnan(lr_cc) else float('nan'))
            dq_res  = _dq_test(hits, var_t, alpha)

            stats_rolling[niv] = {
                'T_eff':     T_eff,
                'N_viol':    N_viol,
                'Taux_obs':  round(N_viol / max(T_eff, 1), 4),
                'Taux_theo': round(1 - alpha, 4),
                'LR_UC':     round(lr_uc,  4) if not math.isnan(lr_uc)  else float('nan'),
                'p_UC':      round(p_uc,   4) if not math.isnan(p_uc)   else float('nan'),
                'LR_IND':    round(lr_ind, 4) if not math.isnan(lr_ind) else float('nan'),
                'p_IND':     round(p_ind,  4) if not math.isnan(p_ind)  else float('nan'),
                'LR_CC':     round(lr_cc,  4) if not math.isnan(lr_cc)  else float('nan'),
                'p_CC':      round(p_cc,   4) if not math.isnan(p_cc)   else float('nan'),
                'DQ':        round(dq_res['DQ'], 4) if not math.isnan(dq_res['DQ']) else float('nan'),
                'DQ_pval':   round(dq_res['p_value'], 4) if not math.isnan(dq_res['p_value']) else float('nan'),
                'Verdict_UC': 'OK' if (not math.isnan(p_uc)  and p_uc  > 0.05) else 'NON',
                'Verdict_CC': 'OK' if (not math.isnan(p_cc)  and p_cc  > 0.05) else 'NON',
                'Verdict_DQ': 'OK' if (not math.isnan(dq_res['p_value']) and dq_res['p_value'] > 0.05) else 'NON',
            }

    return df_violations, df_params_drift, stats_rolling
