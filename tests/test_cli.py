# -*- coding: utf-8 -*-
"""Tests — CLI tickerlab + pyproject.toml.

Tests :
  1. tickerlab --version (appel direct main(), sans subprocess)
  2. tickerlab analyze --help (appel direct)
  3. tickerlab stress --help (appel direct)
  4. Import du package : __version__ defini, sous-packages importables
  5. Package ne contient pas dev/ ni scripts/ (find_spec retourne None)
  6. nu/eta skewt : extraction correcte depuis params GARCH
  7. Integration subprocess --version (skipif si tickerlab non installe)
"""
import importlib.util
import shutil
import sys

import pytest

import tickerlab
from tickerlab import __version__
from tickerlab.cli import main


# ── Test 1 : --version (appel direct) ────────────────────────────────────────

def test_cli_version(capsys, monkeypatch):
    monkeypatch.setattr('sys.argv', ['tickerlab', '--version'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in (captured.out + captured.err)


# ── Test 2 : analyze --help (appel direct) ───────────────────────────────────

def test_cli_analyze_help(capsys, monkeypatch):
    monkeypatch.setattr('sys.argv', ['tickerlab', 'analyze', '--help'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert '--ticker' in output
    assert '--frequency' in output
    assert '--no-ai' in output
    assert '--force-refresh' in output


# ── Test 3 : stress --help (appel direct) ────────────────────────────────────

def test_cli_stress_help(capsys, monkeypatch):
    monkeypatch.setattr('sys.argv', ['tickerlab', 'stress', '--help'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert '--scenario' in output
    assert '--ticker' in output


# ── Test 4 : package installe et importable ───────────────────────────────────

def test_package_installe_version_definie():
    assert hasattr(tickerlab, '__version__')
    assert isinstance(tickerlab.__version__, str)
    assert len(tickerlab.__version__.split('.')) >= 3

    from tickerlab.utils.llm_providers import fabriquer_provider  # noqa: F401
    from tickerlab.core.cache_v2 import CacheEtapes  # noqa: F401


# ── Test 5 : dev/ et scripts/ NON importables comme tickerlab.* ───────────

def test_package_ne_contient_pas_dev_scripts():
    """Verifie que les sous-repertoires non publics ne fuient pas dans le namespace."""
    for parasite in ['tickerlab.dev', 'tickerlab.scripts',
                     'tickerlab.resultats', 'tickerlab.templates']:
        spec = importlib.util.find_spec(parasite)
        assert spec is None, (
            f'{parasite} ne doit pas etre importable — verifier __init__.py blocking'
        )


# ── Test 6 : nu/eta extraction correcte pour skewt ───────────────────────────

def test_stress_nu_eta_skewt():
    """Fix: garch_final.params.get('nu') retourne None pour skewt (param = eta)."""
    # Simule les params d'un modele skewt (eta, lambda mais pas nu)
    params_skewt = {'omega': 1e-4, 'alpha[1]': 0.05, 'beta[1]': 0.90,
                    'eta': 8.5, 'lambda': -0.12}

    # Logique extraite de cmd_stress (cli.py) — test unitaire sans pipeline
    nu_raw = params_skewt.get('nu', params_skewt.get('eta'))
    assert nu_raw is not None, \
        "eta doit servir de fallback a nu pour les modeles skewt"
    assert float(nu_raw) == pytest.approx(8.5)

    # dist_stress : skewt avec nu disponible -> 't' pour appliquer_scenario
    dist = 'skewt'
    nu = float(nu_raw) if nu_raw is not None else None
    dist_stress = 't' if dist in ('t', 'skewt') and nu is not None else 'normal'
    assert dist_stress == 't', \
        "skewt avec eta disponible doit etre traite comme 't' dans appliquer_scenario"


# ── Test 7 : integration subprocess (skipif si entry point absent) ────────────

@pytest.mark.skipif(
    shutil.which('tickerlab') is None,
    reason='tickerlab entry point non installe (pip install -e . requis)',
)
def test_cli_version_subprocess():
    import subprocess
    res = subprocess.run(['tickerlab', '--version'], capture_output=True, text=True)
    assert res.returncode == 0
    assert __version__ in res.stdout
