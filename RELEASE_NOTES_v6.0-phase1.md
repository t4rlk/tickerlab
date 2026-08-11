# Release Notes — v6.0-phase1

**Branche** : `feat/v6-phase1-methodologie` → `master`  
**Tag** : `v6.0-phase1`  
**Commits** : 24 (depuis `master` @ `c929cd5`)  
**Tests** : 59/59 pass — 0 régression  
**Date** : 2026-06-10

---

## Résumé

Phase 1 complète la méthodologie GARCH sur BZ=F (Brent futures, ~960 rendements
hebdomadaires 2006-2026). Le pipeline passe de la sélection ARIMA (Phase 1.1)
aux tests d'intégration cross-sous-tâches (Phase 1.7), en ajoutant 7 modules
de modélisation avancée et ~3 000 lignes de code de production.

---

## Nouvelles fonctionnalités (Phase 1.1 → 1.7)

### Phase 1.1 — Interprétation pédagogique ARIMA
- `core/arima_selector.py` : taxonomie 4 catégories (serie_predictible, bruit_faible,
  martingale_marche_efficient, incertain) selon Box-Jenkins strict + Burnham-Anderson 2002.
- Encadré pédagogique LaTeX dans section 1 du rapport PDF.

### Phase 1.2 — Diagnostic IGARCH post-hoc
- `core/igarch_diagnostic.py` : persistance GARCH, demi-vie (périodes + jours
  calendaires), test Wald H₀: pers=1, classification {igarch_strict, near_igarch,
  mean_reverting} selon seuil configurable (défaut 0.98).

### Phase 1.3 — Component GARCH Engle-Lee (1999)
- `core/component_garch.py` : décomposition permanente/transitoire de la volatilité.
  Estimation SLSQP multi-start (n_starts=3), contrainte C3 : α+β < ρ.
  Déclenchement conditionnel : `near_igarch=True` OU `force_estimation=True`.
  Projection C3 explicite avec warm-start ρ et FX-start.

### Phase 1.4 — Filtered Historical Simulation (FHS)
- `core/fhs.py` : VaR & ES multi-horizons via bootstrap résidus standardisés GARCH
  (Barone-Adesi, Giannopoulos & Vosper 1999). Mode simulation + mode one-step-ahead
  déterministe (OOS backtest). Pool résidus strictement TRAIN (anti look-ahead).

### Phase 1.5 — Diebold-Mariano + Giacomini-Komunjer
- `core/dm_gk.py` : test DM (1995) tick-loss + test GK instabilité temporelle (2005).
  `build_var_series()` : construit séries VaR OOS pour 6 méthodes (Historique, Normale,
  Student, Cornish-Fisher, GARCH dyn., FHS). `comparer_methodes_var()` : toutes paires.

### Phase 1.6 — Bootstrap stationnaire conditionnel
- `core/var_engine.py` : IC VaR bootstrap (Pascual-Romo-Ruiz 2006). Mode express
  (200 réplications, ~2-4s) et mode complet. Isolation RNG stricte (seed passé à
  StationaryBootstrap via `arch.bootstrap`).

### Phase 1.6bis — 3 correctifs audit PDF
- **Bug 1** : `core/data_loader.py` — `resampler_prix()` pour cohérence fréquence
  prix/rendements. Stats 1.2/corrélogramme 1.3 sur prix hebdomadaires, graphique 1.1
  garde la série journalière complète.
- **Bug 2** : `core/garch_selector.py` — `verdict_specification()` : double verdict
  Backtest + Specification dans le résumé exécutif. 5 cas de recommandation.
- **Bug 3** : `core/backtest_rolling.py` — ValueError si window≥n_total ;
  UserWarning + auto-ajustement si window≥n_total-50. `_valider_config_runtime()`
  dans `main.py`.

### Phase 1.6bis-fix — Persistance colonnes spec
- Correctif latent : `_selectionner_scientifique()` ne persistait pas les colonnes
  `tous_passent`, `lb_z_pval`, etc. dans le `df_garch` retourné. Résultat :
  `verdict_specification()` retournait toujours `all_candidates_failed_spec=False`
  sur données réelles.

### Phase 1.6bis-fix2 — Promotion gravité majeure
- Règle de promotion : `all_candidates_failed=True` AND `gravite='mineure'` →
  promu `'majeure'` avec message ALERTE asset-class. Validé sur BZ=F :
  0/5 candidats passent la spec → promotion déclenchée. Réf. : Haas-Mittnik-Paolella
  (2004), Corsi (2009).

### Phase 1.7 — Tests d'intégration cross-sous-tâches
- `tests/test_phase1.py` : 8 smoke tests pré-merge, < 90s.
  Transitions 1.2→1.3 (near_igarch → Component GARCH automatique) et
  1.4→1.5 (FHS → DM-GK matrice) explicitement testées.

---

## Infrastructure et qualité

- **Modules FRTB** (commits initiaux branche) : sVaR, ES 97.5%, capital IMA,
  backtests DQ/Acerbi-Szekely/Lopez/FZ0, traffic light, ICSS Inclan-Tiao,
  ruptures structurelles Zivot-Andrews.
- **Score composite GARCH** : Berkowitz PIT + rang pondéré (désactivé par défaut).
- **Pagination LaTeX** : PageBreak par sous-section, `_split_tableau()`.
- **Config v6** : `AIC/delta4`, `score_composite`, `tests_frtb`, pagination.

---

## Tests

| Fichier | Tests |
|---------|-------|
| `test_phase1.py` (intégrateur) | 8/8 |
| `test_phase11.py` (Phase 1.1) | 6/6 |
| `test_phase12.py` | 7/7 |
| `test_phase13.py` | 6/6 |
| `test_phase14.py` | 6/6 |
| `test_phase15.py` | 6/6 |
| `test_phase16.py` | 6/6 |
| `test_phase16bis.py` | 14/14 |
| **TOTAL** | **59/59** |

---

## Breaking changes

- `generer_pdf_unique()` : nouveaux paramètres `prix_stats=None`, `spec_verdict=None`
  (rétrocompatibles via valeurs par défaut).
- `section_1()` dans `_sections.py` : paramètre `prix_stats=None` ajouté.
- `_resume_executif()` : paramètre `spec_verdict=None` ajouté.
- `selectionner_meilleur()` : modifie `df_garch` in-place (ajout colonnes spec).
  Tout code qui sauvegarde `df_garch` avant cet appel doit recharger le DataFrame.

---

## Prochaine étape — Phase 2

Phase 2.1 : runner multi-tickers (validation sur actifs hors BZ=F).
Cette phase révélera les bugs latents sur asset classes non testées.
