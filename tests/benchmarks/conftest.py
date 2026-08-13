"""Infrastructure commune a la suite de benchmarks."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import zlib
import urllib.request
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"

# Serie DEM/GBP de Bollerslev & Ghysels (1996), 1974 rendements quotidiens en
# pourcentage (02/01/1984 - 31/12/1991). Miroir CRAN du package fGarch, ou le
# jeu est distribue sous le nom `dem2gbp`. Identique a `dmbp.dat` (colonne 1)
# de la page Econometric Benchmarks et a `dmbp` du package rugarch.
DEM2GBP_URL = (
    "https://raw.githubusercontent.com/cran/fGarch/master/data/dem2gbp.csv.gz"
)
DEM2GBP_SHA256 = "d9efaf5e4a57c72dead9ab59d637c5fe07c53cd061a17395c761a9e7af9efe5a"
DEM2GBP_N = 1974


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: benchmark long (Monte-Carlo lourd, plusieurs minutes)"
    )


def _mc_reps(default: int) -> int:
    """Nombre de replications Monte-Carlo, surchargeable par variable
    d'environnement (CI rapide : BENCH_REPS=500 ; publication : 20000)."""
    return int(os.environ.get("BENCH_REPS", default))


@pytest.fixture(scope="session")
def mc_reps():
    return _mc_reps


def require(value, name: str):
    """Renvoie `value`, ou SKIP le test si l'adaptateur n'est pas cable."""
    if value is None:
        pytest.skip(f"adapter.{name} non cable (voir tests/benchmarks/adapter.py)")
    return value


@pytest.fixture(scope="session")
def dem2gbp() -> np.ndarray:
    """Rendements DEM/GBP, telecharges une fois puis mis en cache sur disque.

    Le SHA-256 est verifie : un jeu de donnees silencieusement different
    invaliderait tout le benchmark FCP.
    """
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "dem2gbp.csv.gz"

    if not path.exists():
        if os.environ.get("BENCH_OFFLINE"):
            pytest.skip("BENCH_OFFLINE=1 et donnees absentes du cache")
        try:
            with urllib.request.urlopen(DEM2GBP_URL, timeout=30) as resp:
                blob = resp.read()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"telechargement du jeu FCP impossible : {exc}")
        path.write_bytes(blob)

    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == DEM2GBP_SHA256, (
        "empreinte SHA-256 du jeu DEM/GBP incorrecte : "
        f"{digest} != {DEM2GBP_SHA256}. Le fichier a ete altere ou la source a "
        "change ; ne pas ajuster l'empreinte sans verifier la serie."
    )

    with gzip.open(io.BytesIO(blob), "rt") as fh:
        values = [float(line) for line in fh.read().splitlines()[1:] if line.strip()]

    r = np.asarray(values, dtype=float)
    assert r.size == DEM2GBP_N, f"{r.size} observations au lieu de {DEM2GBP_N}"
    return r


@pytest.fixture
def rng(request):
    """Generateur seme a partir du NOM du test.

    Consequence : chaque test est reproductible independamment de l'ordre
    d'execution, des filtres -k et de la parallelisation (pytest-xdist). Une
    graine de session partagee rendrait les tests Monte-Carlo instables selon
    l'ordre -- une suite de benchmarks flaky ne prouve rien.
    """
    seed = zlib.crc32(request.node.nodeid.encode()) + 20260812
    return np.random.default_rng(seed)
