# -*- coding: utf-8 -*-
"""Mapping PipelineResult -> contrat `result` de l'API v1.

Lecture PURE des sorties existantes (aucune econometrie recalculee ici) :
  - modele_retenu / distribution   <- garch_best
  - persistance (somme alpha+beta)  <- core.rapport._stats.persistance_garch
  - taux_exception_var99            <- backtest_detail (n_exceptions_99 / T_eff_dyn)
  - ratio_tvar_var                  <- df_var (TVaR GARCH / VaR GARCH au 99%)
  - backtesting.detail (6 tests)    <- backtest_detail (FRTB + Berkowitz + FZ0/DM)

Conventions de verdict — EXPLICITES PAR TEST (jamais une regle globale) :
  kupiec, christoffersen  : H0 = modele correct ; accepte si p >= 0.05 (bilateral).
  dq (Engle-Manganelli)   : H0 = VaR correcte ; chi2 ; accepte si p >= 0.05.
  berkowitz_pit           : H0 = densite correcte ; LR chi2(3) ; accepte si p >= 0.05.
  acerbi_szekely (Z2)     : p-valeur SIMULEE UNILATERALE ; accepte si p_Z2 > 0.05.
  fissler_ziegel          : FZ0 est un SCORE, pas un test isole. Verdict = Diebold-
                            Mariano (HAC) du score FZ0 GARCH vs un benchmark NOMME
                            (Normale statique) ; accepte si GARCH non signif. pire.

Une p-valeur indisponible (NaN/None) => verdict 'n/a', NON compte dans `valides`.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

_log = logging.getLogger('tickerlab.web')

_DIST_LABEL = {
    'normal': 'Normale',
    't':      'Student-t',
    'skewt':  'Skew-Student',
    'ged':    'GED',
}

SEUIL_P = 0.05


# ── Helpers verdict ───────────────────────────────────────────────────────────

def _fini(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _verdict_p(pvalue) -> str:
    """H0 = modele correct : accepte si p >= seuil, rejete si p < seuil, n/a sinon."""
    if not _fini(pvalue):
        return 'n/a'
    return 'accepte' if float(pvalue) >= SEUIL_P else 'rejete'


def _verdict_fz(dm: dict) -> str:
    """Verdict Fissler-Ziegel via Diebold-Mariano vs benchmark.

    H0 : precision predictive egale. Si non significatif (p >= 0.05) -> accepte.
    Si significatif : accepte seulement si le GARCH a le meilleur score (plus
    faible = meilleur pour FZ0), sinon rejete. NaN -> n/a.
    """
    pv = dm.get('dm_pvalue')
    if not _fini(pv):
        return 'n/a'
    if float(pv) >= SEUIL_P:
        return 'accepte'
    sg, sb = dm.get('score_garch'), dm.get('score_benchmark')
    if _fini(sg) and _fini(sb):
        return 'accepte' if float(sg) <= float(sb) else 'rejete'
    return 'n/a'


def _rnd(x, n=4):
    return round(float(x), n) if _fini(x) else None


# ── Construction du detail 6 tests ────────────────────────────────────────────

def _detail_six_tests(bd: dict) -> dict:
    """Assemble les 6 tests (noms de cles EXACTS, non negociables) depuis
    backtest_detail. Tolerant : un bloc manquant => test 'n/a'."""
    frtb = bd.get('frtb') or {}
    dq   = frtb.get('dq') or {}
    as_  = frtb.get('acerbi_szekely') or {}
    berk = bd.get('berkowitz') or {}
    fzdm = bd.get('fissler_ziegel_dm') or {}
    kup  = bd.get('kupiec') or {}
    chr_ = bd.get('christoffersen') or {}

    detail = {
        'kupiec': {
            'stat': _rnd(kup.get('stat')), 'pvalue': _rnd(kup.get('pvalue')),
            'verdict': _verdict_p(kup.get('pvalue')),
            'convention': 'Kupiec (1995) LR_UC ; H0 couverture inconditionnelle correcte ; accepte si p>=0.05.',
        },
        'christoffersen': {
            'stat': _rnd(chr_.get('stat')), 'pvalue': _rnd(chr_.get('pvalue')),
            'verdict': _verdict_p(chr_.get('pvalue')),
            'convention': 'Christoffersen (1998) LR_IND ; H0 independance des violations ; accepte si p>=0.05.',
        },
        'dq': {
            'stat': _rnd(dq.get('DQ_stat')), 'pvalue': _rnd(dq.get('p_value')),
            'verdict': _verdict_p(dq.get('p_value')),
            'convention': 'Engle-Manganelli (2004) Dynamic Quantile ; chi2 ; H0 VaR correcte ; accepte si p>=0.05.',
        },
        'acerbi_szekely': {
            'stat': _rnd(as_.get('Z2')), 'pvalue': _rnd(as_.get('p_Z2')),
            'verdict': ('accepte' if (_fini(as_.get('p_Z2')) and float(as_.get('p_Z2')) > SEUIL_P)
                        else ('n/a' if not _fini(as_.get('p_Z2')) else 'rejete')),
            'convention': 'Acerbi-Szekely (2014) Z2 ; p-valeur simulee UNILATERALE ; accepte si p_Z2>0.05 (Z<0 = ES sous-estime).',
        },
        'fissler_ziegel': {
            'score': _rnd(fzdm.get('score_garch')),
            'benchmark': fzdm.get('benchmark'),
            'benchmark_score': _rnd(fzdm.get('score_benchmark')),
            'stat': _rnd(fzdm.get('dm_stat')), 'pvalue': _rnd(fzdm.get('dm_pvalue')),
            'verdict': _verdict_fz(fzdm),
            'convention': 'Fissler-Ziegel (2016) FZ0 : SCORE consistant. Verdict = Diebold-Mariano HAC vs benchmark nomme ; accepte si GARCH non signif. pire.',
        },
        'berkowitz_pit': {
            'stat': _rnd(berk.get('LR_stat')), 'pvalue': _rnd(berk.get('pvalue')),
            'verdict': _verdict_p(berk.get('pvalue')),
            'convention': 'Berkowitz (2001) PIT LR chi2(3) ; H0 densite correctement specifiee ; accepte si p>=0.05.',
        },
    }
    return detail


# ── Point d'entree ────────────────────────────────────────────────────────────

def construire_result(result: Any, config: dict,
                      rapport_pdf_url: Optional[str] = None) -> dict:
    """Construit le dict `result` de l'API v1 depuis un PipelineResult.

    Best-effort et defensif : chaque bloc indisponible est journalise (debug) et
    laisse a None / 'n/a' — jamais un crash, jamais un chiffre invente.
    """
    out: dict = {
        'modele_retenu': None, 'distribution': None, 'persistance': None,
        'taux_exception_var99': None, 'ratio_tvar_var': None,
        'backtesting': {'valides': 0, 'total': 6, 'detail': {}},
        'rapport_pdf_url': rapport_pdf_url, 'n_observations': None,
    }

    # ── Modele retenu + distribution ─────────────────────────────────────────
    b = None
    try:
        best = result.garch_best
        b = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
        modele = b.get('modele') or b.get('model') or 'GARCH'
        p, q = b.get('p'), b.get('q')
        out['modele_retenu'] = (f'{modele}({int(p)},{int(q)})'
                                if p is not None and q is not None else str(modele))
        dist = str(b.get('dist')) if b.get('dist') is not None else None
        out['distribution'] = _DIST_LABEL.get(dist, dist)
    except Exception as exc:
        _log.debug('construire_result: modele/distribution indisponible : %s', exc)

    # ── Persistance (alpha+beta selon famille) ───────────────────────────────
    try:
        from tickerlab.core.rapport._stats import persistance_garch
        modele_nom = (b or {}).get('modele') or (b or {}).get('model')
        params = getattr(result.garch_final, 'params', None)
        if modele_nom is not None and params is not None:
            val = persistance_garch(modele_nom, params)
            out['persistance'] = _rnd(val)
    except Exception as exc:
        _log.debug('construire_result: persistance indisponible : %s', exc)

    # ── Taux d'exception VaR 99 % (depuis le detail enrichi) ─────────────────
    bd = getattr(result, 'backtest_detail', None) or {}
    try:
        n_exc = bd.get('n_exceptions_99')
        t_eff = bd.get('T_eff_dyn')
        if n_exc is not None and t_eff:
            out['taux_exception_var99'] = _rnd(float(n_exc) / float(t_eff))
    except Exception as exc:
        _log.debug('construire_result: taux_exception indisponible : %s', exc)

    # ── Ratio TVaR/VaR au 99 % (depuis df_var) ───────────────────────────────
    try:
        dfv = result.var
        if dfv is not None:
            idx = '99%' if '99%' in dfv.index else (0.99 if 0.99 in dfv.index else None)
            if idx is not None and 'TVaR GARCH' in dfv.columns and 'VaR GARCH' in dfv.columns:
                var99 = float(dfv.loc[idx, 'VaR GARCH'])
                tvar99 = float(dfv.loc[idx, 'TVaR GARCH'])
                if var99 != 0:
                    out['ratio_tvar_var'] = _rnd(tvar99 / var99)
    except Exception as exc:
        _log.debug('construire_result: ratio_tvar_var indisponible : %s', exc)

    # ── n observations ───────────────────────────────────────────────────────
    try:
        if result.rendements is not None:
            out['n_observations'] = int(len(result.rendements.dropna()))
    except Exception as exc:
        _log.debug('construire_result: n_observations indisponible : %s', exc)

    # ── Backtesting : 6 tests + compte valides ───────────────────────────────
    if bd and 'erreur' not in bd:
        detail = _detail_six_tests(bd)
        out['backtesting'] = {
            'valides': sum(1 for t in detail.values() if t.get('verdict') == 'accepte'),
            'total': 6,
            'detail': detail,
        }
    elif bd.get('erreur'):
        out['backtesting']['erreur'] = bd['erreur']

    return out
