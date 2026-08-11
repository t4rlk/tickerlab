# -*- coding: utf-8 -*-
"""
14 tests — fidélité EViews Annexe C.6 :
  - libellés de loi (eviews_dist_label)
  - mapping par paramètre pour les 4 distributions (_param_section_label)
  - intégration sur fits arch réels (skew-t, GED)
"""
import warnings
import numpy as np
import pytest

from tickerlab.core.rapport._eviews import (
    eviews_dist_label,
    _param_section_label,
    bloc_eviews_estimation,
)


def _harvest_text(flowables):
    """Extrait le texte des Paragraph/Table d'une liste de flowables reportlab."""
    out = []
    def _para(x):
        try:
            return x.getPlainText()
        except Exception:
            return str(getattr(x, 'text', ''))
    for f in flowables:
        cells = getattr(f, '_cellvalues', None)
        if cells is not None:
            for row in cells:
                for c in row:
                    if isinstance(c, (list, tuple)):
                        out.extend(_para(z) for z in c)
                    elif isinstance(c, str):
                        out.append(c)
                    elif c is not None:
                        out.append(_para(c))
        else:
            out.append(_para(f))
    return '\n'.join(out)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def serie_200():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


def _fit_garch(serie, dist='normal'):
    from arch import arch_model
    m = arch_model(serie * 100, vol='Garch', p=1, q=1, dist=dist)
    return m.fit(disp='off', show_warning=False)


# =============================================================================
# Libellés de loi — eviews_dist_label (5 tests)
# =============================================================================

def test_label_normal():
    assert eviews_dist_label('Normal') == 'Normal'


def test_label_students_t():
    assert eviews_dist_label('StudentsT') == 'Student'


def test_label_ged():
    assert eviews_dist_label('GeneralizedError') == 'GED'


def test_label_skew_student():
    assert eviews_dist_label('SkewStudent') == 'Skew-Student'


def test_label_skew_before_student():
    """Régression bug : SkewStudent ne doit pas être classé 'Student'."""
    result = eviews_dist_label('SkewStudent')
    assert result == 'Skew-Student'
    assert result != 'Student'


# =============================================================================
# Mapping paramètres — _param_section_label (7 tests)
# =============================================================================

def test_param_mu_mean():
    sec, lbl = _param_section_label('mu', 'GARCH')
    assert sec == 'mean'
    assert lbl == 'C'


def test_param_omega_variance():
    sec, lbl = _param_section_label('omega', 'GARCH')
    assert sec == 'variance'
    assert lbl == 'C'


def test_param_beta1_variance():
    sec, lbl = _param_section_label('beta[1]', 'GARCH')
    assert sec == 'variance'
    assert lbl == 'GARCH(-1)'


def test_param_nu_student():
    sec, lbl = _param_section_label('nu', 'GARCH', nom_loi='Student')
    assert sec == 'dist'
    assert lbl == 'T-DIST. DOF'


def test_param_nu_ged():
    sec, lbl = _param_section_label('nu', 'GARCH', nom_loi='GED')
    assert sec == 'dist'
    assert lbl == 'GED PARAMETER'


def test_param_eta_dist_not_variance():
    """eta (dof skew-t) doit aller dans Distribution Parameters, pas Variance Equation."""
    sec, lbl = _param_section_label('eta', 'GARCH', nom_loi='Skew-Student')
    assert sec == 'dist'
    assert lbl == 'T-DIST. DOF'


def test_param_lambda_skewness():
    """lambda (asymétrie skew-t) doit être SKEWNESS PARAMETER, pas GED PARAMETER."""
    sec, lbl = _param_section_label('lambda', 'GARCH', nom_loi='Skew-Student')
    assert sec == 'dist'
    assert lbl == 'SKEWNESS PARAMETER'


# =============================================================================
# Intégration — fits arch réels (2 tests)
# =============================================================================

def _all_text(flowables) -> str:
    """Extrait tout le texte rendu des flowables (Paragraph + Table)."""
    parts = []
    for f in flowables:
        if hasattr(f, 'text'):
            parts.append(f.text)
        if hasattr(f, '_cellvalues'):
            for row in f._cellvalues:
                for cell in row:
                    if hasattr(cell, 'text'):
                        parts.append(cell.text)
    return ' '.join(parts)


def test_integration_skewt_labels(serie_200):
    fit = _fit_garch(serie_200, dist='skewt')
    flowables = bloc_eviews_estimation(
        fit, dep_var='DLTEST',
        method='ML ARCH -- Skew-Student distribution',
        nom_loi='Skew-Student',
    )
    assert flowables, 'bloc_eviews_estimation ne doit pas retourner []'
    txt = _all_text(flowables)
    assert 'T-DIST. DOF' in txt, f'T-DIST. DOF absent du bloc skewt : {txt[:300]}'
    assert 'SKEWNESS PARAMETER' in txt, f'SKEWNESS PARAMETER absent : {txt[:300]}'
    assert 'GED PARAMETER' not in txt, 'GED PARAMETER ne doit pas apparaître pour skewt'


def test_integration_ged_labels(serie_200):
    fit = _fit_garch(serie_200, dist='ged')
    flowables = bloc_eviews_estimation(
        fit, dep_var='DLTEST',
        method='ML ARCH -- GED distribution',
        nom_loi='GED',
    )
    assert flowables, 'bloc_eviews_estimation ne doit pas retourner []'
    txt = _all_text(flowables)
    assert 'GED PARAMETER' in txt, f'GED PARAMETER absent du bloc ged : {txt[:300]}'
    assert 'T-DIST. DOF' not in txt, 'T-DIST. DOF ne doit pas apparaître pour GED'
    assert 'SKEWNESS PARAMETER' not in txt, 'SKEWNESS PARAMETER ne doit pas apparaître pour GED'


# ── 4. Mapping ordres > 1 et puissance APARCH/TGARCH (audit 2026-06-30) ───────

def test_delta_dans_equation_variance():
    section, label = _param_section_label('delta', 'APARCH', 'Skew-Student')
    assert section == 'variance'        # surtout PAS 'mean'
    assert 'delta' in label.lower() or 'POWER' in label


@pytest.mark.parametrize('nom, attendu', [
    ('beta[2]',  'GARCH(-2)'),
    ('alpha[2]', 'RESID(-2)^2'),
    ('gamma[2]', 'RESID(-2)^2*(RESID(-2)<0)'),
    ('beta[3]',  'GARCH(-3)'),
])
def test_ordres_superieurs_mappes(nom, attendu):
    section, label = _param_section_label(nom, 'TGARCH', 'Skew-Student')
    assert section == 'variance'
    assert label == attendu
    assert label != nom.upper()          # plus de nom arch brut


def test_aparch_112_rendu_complet_et_footer():
    """Intégration : APARCH(1,1,2) skew-t — ordres>1 mappés, delta en variance,
    footer EViews enrichi présent. Sur fit arch réel."""
    warnings.filterwarnings('ignore')
    from arch import arch_model
    rng = np.random.default_rng(7)
    r = rng.standard_t(6, 1500) * 0.9
    res = arch_model(r, vol='APARCH', p=1, o=1, q=2, dist='skewt').fit(disp='off')
    lbl = eviews_dist_label(type(res.model.distribution).__name__)
    bloc = bloc_eviews_estimation(res, dep_var='DLTEST',
                                  method=f'ML ARCH -- {lbl} distribution', nom_loi=lbl)
    txt = _harvest_text(bloc)
    assert 'GARCH(-2)' in txt          # beta[2] mappé
    assert 'BETA[2]' not in txt        # plus de nom brut
    assert 'POWER' in txt or 'delta' in txt.lower()  # delta présent (variance)
    # footer EViews enrichi
    for champ in ('Mean dependent var', 'S.D. dependent var',
                  'Hannan-Quinn criter.', 'Durbin-Watson stat',
                  'S.E. of regression', 'Sum squared resid'):
        assert champ in txt, f"footer EViews incomplet : '{champ}' manquant"


# ── 5. Libellés équation de variance EGARCH vs GARCH (audit 2026-07-01) ───────

@pytest.mark.parametrize('nom, attendu', [
    ('alpha[1]', '|RESID(-1)/SQRT(GARCH(-1))|'),
    ('gamma[1]', 'RESID(-1)/SQRT(GARCH(-1))'),
    ('beta[1]',  'EGARCH(-1)'),
    ('beta[2]',  'EGARCH(-2)'),
])
def test_egarch_labels_variance(nom, attendu):
    section, label = _param_section_label(nom, 'ML ARCH -- SKEW-STUDENT DISTRIBUTION EGARCH',
                                          'Skew-Student')
    assert section == 'variance'
    assert label == attendu


@pytest.mark.parametrize('nom, attendu', [
    ('alpha[1]', 'RESID(-1)^2'),
    ('gamma[1]', 'RESID(-1)^2*(RESID(-1)<0)'),
    ('beta[1]',  'GARCH(-1)'),
])
def test_garch_labels_variance_sans_egarch(nom, attendu):
    """Sans 'EGARCH' dans model_upper, on garde la forme GARCH standard."""
    section, label = _param_section_label(nom, 'ML ARCH -- NORMAL DISTRIBUTION GARCH',
                                          'Normal')
    assert section == 'variance'
    assert label == attendu


def test_integration_egarch_rendu_reel():
    """Intégration : EGARCH(1,1,1) skew-t via bloc_eviews_estimation — la
    détection auto de la classe de volatilité doit produire les labels EGARCH."""
    warnings.filterwarnings('ignore')
    from arch import arch_model
    rng = np.random.default_rng(3)
    r = rng.standard_t(6, 1200) * 1.1
    res = arch_model(r, vol='EGARCH', p=1, o=1, q=1, dist='skewt').fit(disp='off')
    lbl = eviews_dist_label(type(res.model.distribution).__name__)
    bloc = bloc_eviews_estimation(res, dep_var='DLTEST',
                                  method=f'ML ARCH -- {lbl} distribution', nom_loi=lbl)
    txt = _harvest_text(bloc)
    assert '|RESID(-1)/SQRT(GARCH(-1))|' in txt   # alpha EGARCH
    assert 'EGARCH(-1)' in txt                     # beta EGARCH
    # aucune étiquette EGARCH ne contient RESID(-1)^2 → forme GARCH absente
    assert 'RESID(-1)^2' not in txt
