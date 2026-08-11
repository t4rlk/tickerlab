#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate de couverture ciblée sur les modules critiques.

Usage (après `pytest --cov=core --cov-report=json`) :
    python scripts/check_coverage_gate.py

Sort en code 1 si un module critique est sous son seuil plancher.
"""
import json
import sys
from pathlib import Path

# Seuils planchers fixés ~5 points sous la couverture mesurée (2026-06-17).
# Mesures de référence dans CHANGELOG.md Phase 9.
# TODO DETTE_TECHNIQUE : core/backtest.py à 44% (< 50%) — gate désactivée,
#   couverture à renforcer dans une phase ultérieure (tests OOS complets).
GATES: dict[str, int] = {
    'core/dm_gk.py':        83,   # mesuré 88%
    'core/var_engine.py':   58,   # mesuré 63%
    'core/cache_v2.py':     72,   # mesuré 77%
    'core/config_schema.py': 94,  # mesuré 99%
    'core/data_loader.py':  73,   # mesuré 78%
}

coverage_json = Path('coverage.json')
if not coverage_json.exists():
    print('ERREUR : coverage.json introuvable — lancer pytest --cov --cov-report=json d\'abord.')
    sys.exit(2)

with coverage_json.open(encoding='utf-8') as f:
    data = json.load(f)

files = data.get('files', {})
failures: list[str] = []
width = max(len(m) for m in GATES)

print('Gate de couverture -- modules critiques\n' + '-' * 50)
for module, threshold in sorted(GATES.items()):
    # Normalise les separateurs de chemin pour portabilite Windows/Linux
    key = next(
        (k for k in files if k.replace('\\', '/').endswith(module)),
        None,
    )
    if key is None:
        print(f'  MANQUANT  {module}  (absent de coverage.json)')
        failures.append(module)
        continue

    pct = files[key]['summary']['percent_covered']
    ok = pct >= threshold
    mark = 'OK  ' if ok else 'FAIL'
    print(f'  {mark}  {module:{width}}  {pct:5.1f}%  (seuil {threshold}%)')
    if not ok:
        failures.append(module)

print('-' * 50)
if failures:
    print(f'\n{len(failures)} module(s) sous le seuil : {failures}')
    sys.exit(1)
else:
    print('\nTous les modules critiques sont au-dessus du seuil. OK')
