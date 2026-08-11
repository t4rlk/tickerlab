# Release Notes — v6.0-phase2

**Date** : 11 juin 2026  
**Branche** : feat/v6-phase2-validation → master  
**Tag** : v6.0-phase2  
**Tests** : 75/75 pass (0 régression vs Phase 1 : 59/59)

---

## Résumé Phase 2 — Validation multi-actifs et stress testing

Phase 2 transforme le pipeline d'un outil de recherche académique (un seul actif, BZ=F) en un outil validé sur 6 actifs × 2 fréquences, avec catalogue des cas pathologiques et stress testing conforme FRTB/BCBS.

---

## Sous-tâches complétées

### 2.1 — Runner multi-tickers (commits dd07986, 08f773e, 2e53481)

- `scripts/run_multi_ticker.py` : exécution parallèle (ProcessPoolExecutor) de 12 runs (6 tickers × 2 fréquences) ; CSV incrémental 34 colonnes ; mode `--metrics-only` (défaut, ~25s/run) vs `--full-output` (~80s/run)
- Mode `_metrics_only` dans `main.py` : early return avant génération PDF/Excel/LaTeX
- `tests/test_phase21.py` : 4 tests smoke

**Résultats 12 runs (juin 2026)** :

| Ticker | Freq | Modèle | Persistance | spec_gravite | VaR99 GARCH |
|---|---|---|---|---|---|
| BZ=F | daily | EGARCH(1,1,1)[t] | 0.979 | ok | −5.00% |
| BZ=F | weekly | EGARCH(1,1,1)[t] | 0.974 | ok | −13.3% |
| ^GSPC | daily | GARCH(1,0,1)[t] | 0.995 | majeure | −1.73% |
| ^GSPC | weekly | GARCH(1,0,1)[t] | 0.995 | majeure | −5.57% |
| GC=F | weekly | — | — | Cat. 2 | — |
| EURUSD=X | weekly | — | — | Cat. 2 | — |
| BTC-USD | weekly | GARCH(1,0,1)[t] | 0.998 | majeure | **−99.9%** (floored BTC-001) |

**Bugs détectés et résolus** :
- BTC-001 (critique) : VaR GARCH = −357% pour BTC-USD weekly → floored à −99.9%
- DM-001 (cascade) : `math range error` résolu par BTC-001 floor

### 2.2 — Catalogue cas pathologiques (commit cde2bc0)

- `docs/CAS_PATHOLOGIQUES.md` : 4 catégories formalisées
  - Cat. 1 (modélisation impossible) : BTC-USD, ^GSPC daily
  - Cat. 2 (fallback IC_seul) : GC=F weekly, EURUSD=X weekly
  - Cat. 3 (near-IGARCH) : ^GSPC, EURUSD=X, BTC-USD
  - Cat. 4 (trop court) : actifs < 250 obs OOS
- `tests/test_phase22_catalogue.py` : 3 tests structurels

### 2.3 — Stress testing scénario-based (commits a018ad8, 5ff3cef)

- `core/stress_scenarios.py` : 4 scénarios FRTB (oil_shock_2022, covid_march_2020, fed_hike_2022, geopolitique_taiwan) ; distance Mahalanobis univariée (Studer 1997) ; t standardisée corrigée
- `core/reverse_stress.py` : résolution analytique univariée (BCBS d365 §44)
- `core/var_engine.py` : `_apply_btc001_floor()` — floor VaR GARCH < −99.9%
- `tests/test_phase23.py` : 9 tests (4 ajoutés en fix pour couvrir les 3 bugs scientifiques)
- `docs/PHASE2_3_STRESS_TESTING.md` : rapport complet avec probabilités corrigées

**Corrections scientifiques Phase 2.3-fix** :
1. t standardisée (Var=1) vs scipy.t direct — facteur ÷2 à 10× sur les probabilités
2. Probabilité directionnelle (queue supérieure pour gains)
3. σ_H via récurrence GARCH/IGARCH exacte (vs règle √H)

---

## Architecture après Phase 2

```
core/
  var_engine.py          +  _apply_btc001_floor(), BTC-001 floor
  stress_scenarios.py    +  NOUVEAU : stress testing univarié
  reverse_stress.py      +  NOUVEAU : reverse stress univarié
  exceptions.py              (Phase 3.1)
  config_validation.py       (Phase 3.1)
scripts/
  run_multi_ticker.py    +  mode metrics-only, CSV 34 colonnes
docs/
  CAS_PATHOLOGIQUES.md   +  NOUVEAU : catalogue actifs pathologiques
  PHASE2_1_ANALYSE_MULTI_TICKERS.md  +  NOUVEAU
  PHASE2_3_STRESS_TESTING.md         +  NOUVEAU
  DETTE_TECHNIQUE.md     +  BTC-001 RESOLU, 3 tickets restants
tests/
  test_phase21.py        +  4 tests runner multi-tickers
  test_phase22_catalogue.py  +  3 tests catalogue
  test_phase23.py        +  9 tests stress testing
```

---

## Métriques Phase 2

| Métrique | Valeur |
|---|---|
| Tests Phase 2 ajoutés | 16 (4 + 3 + 9) |
| Tests total | 75/75 pass |
| Runs multi-tickers validés | 12/12 |
| Bugs critiques résolus | 1 (BTC-001) |
| Bugs scientifiques corrigés | 3 (Phase 2.3-fix) |
| Dettes techniques ouvertes | 3 (PERF-001, LOG-001, DM-001 mineurs) |

---

## Prochaine étape — Phase 3 : Industrialisation

Phase 3.1 — API programmatique propre :
- Hiérarchie d'exceptions typées (`PipelineError`, `DataError`, `ModelError`, `ValidationError`, `ExportError`)
- Logging structuré `pea_brent.{module}` (remplacement des `print()`)
- `PipelineResult` dataclass rétrocompatible + bloc `meta` reproductibilité
- Validation de config centralisée
- Cible : 83/83 tests, 0 chiffre modifié vs Phase 2

*Voir `INSTRUCTION_CLAUDE_CODE.md` pour le plan complet Phase 3.*
