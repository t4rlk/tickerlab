# CHANGELOG — PEA-Brent v5

## [Feat — Pass-through ordre des sorties (configurateur web)] — feat/eviews-annexe — 2026-07-01

Le configurateur web envoie `outputs` trié par ordre de clic utilisateur.
`construire_config` le préserve désormais dans `config['sorties_ordre']`
(dédupliqué, ordre conservé) pour un futur câblage de l'orchestrateur de rapport.

### Modif (`web/mapping.py`)
- Nouveau champ racine `sorties_ordre` : liste dédupliquée de `payload['outputs']`,
  ordre de clic préservé. **Pass-through inoffensif** — aucune section du rapport
  ne le lit encore (vérifié : 0 occurrence ailleurs dans le dépôt).
- Racine `TickerLabConfig` en `extra='allow'` → le nouveau champ passe
  `valider_config_structure` sans `ValidationError`.

### Tests (`tests/test_mapping.py`, nouveau)
- 10 tests couvrant le contrat de `construire_config` (aucune couverture
  auparavant) : dédup/ordre de `sorties_ordre`, opt-in `breaks`, `report` →
  IA + mode LaTeX, `price` → `auto_adjust`, désactivation rolling backtest,
  champs `data` pilotés, validité Pydantic, non-régression des blocs
  économétriques (`garch`/`arima`/`var`/`fhs` inchangés).

### Vérification
- Suite 212/212. `construire_config` → `valider_config_structure` OK ;
  `auto_adjust` bien consommé par `data_loader.telecharger_prix`.


## [Lisibilité — Kurtosis excès + brute dans le corps] — feat/eviews-annexe — 2026-07-01

Audit kurtosis (règle d'or, sur BZ=F régénéré) : aucune anomalie. La kurtosis
des rendements Brent est +5.91 en excès (8.91 brute) — leptokurtose normale.
La « valeur négative » suspectée précédemment était la série des PRIX (kurtosis
excès négative attendue pour une série tendancielle), pas les rendements.

Seul point réel : double convention (excès dans le corps §x.2 « Kurtosis (exc.) »
vs brute dans l'annexe EViews C.1 « Kurtosis »). Les deux sont justes et étiquetées
(raw = excès + 3), chacune native à son contexte. Pour lever toute ambiguïté de
lecture, le corps affiche désormais les deux.

### Modif (`sections/stationarite.py`)
- `_tab_stats_desc` : la ligne kurtosis affiche `excès [brute = excès+3]`,
  ex. `Kurtosis (exc.) 5.9069 [brute 8.9069]`. Fallback propre si valeur NaN.
- Convention conservée : excès dans le corps (standard académique, cohérent avec
  Jarque-Bera), brute dans l'annexe (fidélité EViews). Aucune valeur recalculée.

### Vérification
- Suite 202/202. PDF BZ=F régénéré (0 warnings) : brute affichée (8.9069)
  cohérente avec l'annexe C.1 (8.906852). Séries prix/log-prix confirmées à
  kurtosis excès négative (-0.7609 / -0.1613), sans anomalie.


## [Feat — Format EViews dans le corps (5.2 / 6.2)] — feat/eviews-annexe — 2026-07-01

Les tableaux d'estimation du corps du rapport passent du style maison
(Parametre | Estim. | Std. Err. | t-stat | Prob. | Sig.) au format EViews
authentique, identique a celui de l'annexe C.

### Modifs (`sections/arima_garch.py`)
- 5.2 (ARIMA) et 6.2 (GARCH) : les appels `_tab_arima_coefs` / `_tab_garch_coefs`
  sont remplaces par `bloc_eviews_estimation(...)`, avec la meme derivation que
  l'annexe (dep_var = 'DL'+ticker ; nom_loi via `eviews_dist_label`).
- Import ajoute : `bloc_eviews_estimation`, `eviews_dist_label`.
- `_tab_arima_coefs` / `_tab_garch_coefs` deviennent inutilises (non supprimes
  pour l'instant — conservables comme rendu alternatif / a nettoyer plus tard).

### Consequence visuelle
- 5.2 et 6.2 deviennent des blocs Courier N&B au milieu d'un rapport aux tables
  colorees : c'est voulu (fidelite EViews), mais cela cree deux "ilots"
  monochromes. Le bloc d'estimation apparait desormais deux fois (corps 5.2/6.2
  + annexe C.5/C.6). Un basculement par config (eviews vs style maison) est
  possible si besoin de conserver les deux rendus.

### Verification
- Suite 202/202, import sans cycle. PDF BZ=F regenere (0 warnings) : extraction
  confirme 5.2 « Least Squares -- ARIMA(2,0,2) » (C, AR(1), AR(2), MA(1), MA(2))
  et 6.2 « ML ARCH -- Skew-Student distribution » (EGARCH(1,1,1)) au format EViews.


## [Fix — Fidélité EViews annexe C.6 — équation de variance EGARCH] — feat/eviews-annexe — 2026-07-01

Audit visuel du PDF BZ=F (modèle retenu EGARCH(1,1,1)) : les termes de
l'équation de variance EGARCH étaient étiquetés en forme GARCH standard.
**Aucune logique économétrique modifiée — seuls les libellés d'affichage
changent.**

### Cause
- `_param_section_label()` route les labels EGARCH derrière `'EGARCH' in model_upper`,
  mais `model_upper = method.upper()` ne contient que « ML ARCH -- ... distribution » :
  le nom de famille du modèle n'y figure jamais → la branche EGARCH était du code mort.

### Fix (`_eviews.py`)
- `bloc_eviews_estimation()` détecte la classe de volatilité via
  `type(fit.model.volatility).__name__` (EGARCH / GARCH / APARCH) et l'ajoute à
  `model_up` pour le routage. La ligne « Method: » affichée reste inchangée.
- Ajout du libellé EViews `beta[k] → EGARCH(-k)` (log-variance retardée) pour EGARCH.
- Rendu EGARCH correct : `|RESID(-1)/SQRT(GARCH(-1))|`, `RESID(-1)/SQRT(GARCH(-1))`,
  `EGARCH(-1)` (au lieu de `RESID(-1)^2`, `RESID(-1)^2*(RESID(-1)<0)`, `GARCH(-1)`).

### Tests
- `tests/test_eviews_dist_labels.py` : +8 tests (mapping EGARCH vs GARCH par
  paramètre + intégration sur fit EGARCH réel). Total fichier 28/28 ; suite 202/202.
- Vérification règle d'or : BZ=F régénéré, extraction PDF confirme les 3 labels
  EGARCH, 0 warnings.


## [Fix — Fidélité EViews annexe C.6 (suite)] — feat/eviews-annexe — 2026-06-30

Audit visuel de la sortie réelle (PDF GSPC) : le bloc GARCH EViews comportait
encore des infidélités au-delà des libellés de distribution. Corrigées ici.

### Mapping des paramètres (`_eviews.py`)
- `delta` (puissance APARCH/TGARCH) était renvoyé par défaut en *Mean Equation*.
  → désormais *Variance Equation*, libellé « POWER (delta) ».
- Ordres > 1 affichés en noms arch bruts (`BETA[2]`, `ALPHA[2]`...).
  → généralisation regex : `beta[k]→GARCH(-k)`, `alpha[k]→RESID(-k)^2`,
  `gamma[k]→RESID(-k)^2*(RESID(-k)<0)` (idem variantes EGARCH).

### Footer enrichi façon EViews (`bloc_eviews_estimation`)
- Ajout de R-squared, Adjusted R-squared, S.E. of regression, Sum squared resid,
  Mean dependent var, S.D. dependent var, Hannan-Quinn criter., Durbin-Watson stat.
  (auparavant seulement Log L / AIC / Schwarz). Calcul défensif (NaN si indispo).

### Tests
- `tests/test_eviews_dist_labels.py` : +6 tests (delta→variance, ordres>1,
  intégration APARCH(1,1,2) skew-t vérifiant GARCH(-2)/POWER + footer complet).
  Total fichier 20/20.
- Vérification règle d'or : bloc C.6 régénéré sur rendements Brent réels.


## [Fix — Fidélité EViews annexe C.6] — feat/eviews-annexe — 2026-06-25

Audit de la sortie réelle (PDF BZ_F) : le bloc GARCH format EViews affichait des
libellés de distribution **faux** pour les lois GED et skew-Student. Deux bugs
coordonnés corrigés. **Aucune logique économétrique modifiée** — seuls les
libellés et le routage d'affichage des paramètres changent.

### Bug 1 — En-tête de distribution (`sections/annexes.py`)
- `'Student' in 'SkewStudent'` étant vrai, une skew-t était étiquetée
  « Method: ML ARCH -- Student distribution ». Test 'Skew' désormais effectué
  avant 'Student' via le nouveau helper `eviews_dist_label()` (testable).

### Bug 2 — Mapping des paramètres de forme (`_eviews.py`)
- `_param_section_label()` était aveugle à la loi. Rendu *distribution-aware* :
  - `nu` → « GED PARAMETER » si GED, sinon « T-DIST. DOF » (Student) ;
  - `eta` (ddl skew-t) → « T-DIST. DOF » dans *Distribution Parameters*
    (était avalé par le filtre des termes de variance) ;
  - `lambda` (asymétrie skew-t) → « SKEWNESS PARAMETER » (était « GED PARAMETER »).
- `eta` retiré du filtre des termes de variance ; `nom_loi` threadé dans les
  deux appels au classifieur.

### Tests
- `tests/test_eviews_dist_labels.py` — 14 tests : libellés de loi, mapping par
  paramètre pour les 4 distributions, et intégration sur fits `arch` réels
  (skew-t/GED) vérifiant le texte rendu du bloc.


## [Audit — Remédiation dette] — chore/audit-remediation — 2026-06-23

Suite à un audit externe : 5 chantiers identifiés, 4 traités ici. Le 5ᵉ
(révocation de la clé Groq exposée) est une action humaine hors périmètre code.
**Aucune logique économétrique modifiée — zéro changement de sortie numérique.**

### Chantier 1 — Réconciliation des dépendances (`pyproject.toml` / `requirements.txt`)

- `pyproject.toml` = source de vérité abstraite ; `requirements.txt` = lockfile épinglé.
- Ajout `reportlab>=4.0` aux dépendances de `pyproject.toml` : importé par 5 fichiers
  du cœur (`core/rapport/*`, `export_academique.py`). Sans lui, `pip install tickerlab`
  cassait à la génération du PDF (`ImportError`).
- Ajout `PyYAML==6.0.2` à `requirements.txt` : importé par `web/mapping.py` (`import yaml`).
  Sans lui, `pip install -r requirements.txt` cassait au lancement du configurateur.
- Ajout extra `[web]` (fastapi, uvicorn, python-dotenv), package `tickerlab.web`,
  package-data `static/*`, et script `tickerlab-web` (jusqu'ici référencé dans
  `web/app.py` mais non défini → entry-point fantôme corrigé).

### Chantier 2 — Découpe de `core/rapport/_sections.py` (3 829 → 33 lignes)

- Refactor pur (déplacement de code, aucune réécriture). Sous-package
  `core/rapport/sections/` créé :
  - `_common.py` — helpers transverses
  - `stationarite.py` — sections 1-3 (prix, log-prix, ADF/PP/KPSS)
  - `arima_garch.py` — sections 4-6 (rendements, ARIMA, GARCH + Component/IGARCH)
  - `var_backtest.py` — sections 7-9 (VaR/TVaR, backtest, violations)
  - `stress_synthese.py` — sections 10-11 (stress, synthèse)
  - `annexes.py` — annexes + résumé FRTB
- `_sections.py` devient une **façade de rétrocompatibilité** ré-exportant les 13
  sections (`section_1`…`section_11`, `section_annexes`, `section_frtb_resume`).
- `core/rapport/_orchestrateur.py` **inchangé** : import `_sections as SEC` préservé.
- `pyproject.toml` `[tool.coverage.run]` mis à jour (`*/rapport/_sections*.py`) pour
  couvrir aussi les nouveaux sous-modules.

### Chantier 3 — Instrumentation des `except` silencieux (observabilité)

- Les 18 blocs `except … : pass` muets reçoivent désormais un log nommant la cause
  attendue ; le type d'exception est restreint quand il est connu.
- **Priorité** `core/garch_selector.py` : un échec d'estimation GARCH émet maintenant
  un `_log.warning()` détaillant la spec (`modèle(p,q,o)`, distribution) et l'erreur,
  au lieu d'être avalé silencieusement.
- Aucun changement de flux de contrôle : amélioration d'observabilité pure, le
  fallback existant est conservé.

### Chantier 4 — Durcissement de la couche web

- **JobStore TTL** : les jobs `done`/`error` sont purgés après 1 h (`TTL_SECONDES = 3600`)
  à chaque appel à `creer()`. Les jobs actifs (`queued`/`running`) ne sont jamais purgés.
- **CORS configurable** : `allow_origins` lit `TICKERLAB_CORS_ORIGIN` (défaut `*` pour
  dev local). Pour durcir en déploiement : `TICKERLAB_CORS_ORIGIN=https://mon-domaine.com`.
- **Échappement `innerHTML`** : `state.ticker`, `state.name`, `state.from`, `state.to`
  passés via `_esc()` dans `render()` (index.html) — self-XSS local éliminé.
- **Data race documentée** : commentaire explicite dans `JobStore` — GIL CPython rend
  les affectations `progress`/`step` atomiques (tradeoff MVP acceptable).
- 8 tests unitaires ajoutés (`tests/test_web_jobs.py`) couvrant l'éviction TTL et la
  thread-safety des créations concurrentes.

### Chantier 5 — Hygiène dépôt

- Ajout d'un `README.md` à la racine (install, usage CLI `tickerlab`, usage web
  `tickerlab-web`, structure du dépôt, liens CHANGELOG/docs).
- `resultats/diff_v5_v6.csv` et `resultats/note_v6.md` déplacés dans `docs/`
  (référencés malgré `resultats/` dans `.gitignore`).

### Validation

- `pytest` complet vert (174/174 tests) ; compilation de l'ensemble du cœur sans erreur.
- Rétrocompat de la façade `_sections` vérifiée : les 13 sections consommées par
  `_orchestrateur.py` sont toutes accessibles, aucun import cassé.
- ⚠️ **Règle d'or — à confirmer dans l'environnement complet** : non-régression du
  PDF `BZ=F` généré avant/après la découpe (comparaison page à page). Tant que ce
  diff n'est pas vert, la découpe est *techniquement correcte mais pas prouvée*.

### Hors périmètre (action requise)

- 🔴 Révoquer la clé `GROQ_API_KEY` exposée (console.groq.com) et la régénérer.
  Le `.env` n'est pas tracké par git, mais la clé a transité dans des archives
  partagées — elle doit être considérée comme compromise.

---

## [Phase 9 — CI/CD] — feat/v6-phase9-ci — 2026-06-17

### Condition 1 — Config pytest centralisée (`pyproject.toml`)

- Ajout `[tool.pytest.ini_options]` : `testpaths = ["tests"]`, `addopts = "-q --strict-markers"`
- Markers déclarés pour usage futur : `network` (accès réseau) + `slow` (GARCH réel, >5 s)

### Condition 2 — Config coverage (`pyproject.toml`)

- `[tool.coverage.run]` : `source = ["core", "."]`, omit `tests/*`, `*/rapport/_sections.py`, `scripts/*`
  - `_sections.py` (1773 lignes, génération PDF) exclu : couverture unitaire nulle sans run réel
- `[tool.coverage.report]` : `precision = 1`, `show_missing = true`
- Pas de `fail_under` global : modules PDF/export à 0% tireraient la moyenne sous 40%

### Condition 3 — Workflow GitHub Actions (`.github/workflows/ci.yml`)

- Déclencheurs : `push` sur toute branche + `pull_request`
- Matrice Python : `3.11` + `3.12`
- Étapes : checkout → setup-python (cache pip) → `pip install -r requirements.txt` → `pip install pytest pytest-cov` → `pip install -e .` (entry point CLI) → `pytest --cov` → gate ciblée
- Upload artefact coverage XML (Python 3.12 seulement)
- Étape gate séparée : `python scripts/check_coverage_gate.py` — fail si régression

### Gate de couverture (`scripts/check_coverage_gate.py`)

Couvertures mesurées le 2026-06-17 (HEAD `094cea6`) et seuils planchers :

| Module | Couverture mesurée | Seuil gate |
|---|---|---|
| `core/dm_gk.py` | 88% | **83%** |
| `core/var_engine.py` | 63% | **58%** |
| `core/cache_v2.py` | 77% | **72%** |
| `core/config_schema.py` | 99% | **94%** |
| `core/data_loader.py` | 78% | **73%** |
| `core/backtest.py` | 44% | *(< 50% — gate désactivée, TODO DETTE_TECHNIQUE)* |

Gate locale validée : tous les 5 modules actifs au-dessus du seuil.

### Validation

- Suite complète : 165 tests, 0 régression (avec `--strict-markers` activé)
- Gate locale : `python scripts/check_coverage_gate.py` → exit 0 sur HEAD actuel
- YAML CI lint : fichier syntaxiquement valide

## [Phase 8 — Robustesse données] — feat/v6-phase8-robustesse-donnees — 2026-06-17

### Condition 1 — Garde-fous typés sur les données téléchargées (`core/data_loader.py`)

- Nouvelle fonction `valider_donnees(prix, rendements, config)` avec 4 contrôles par ordre de priorité :
  - **a) Longueur minimale** : `n_rendements >= data.min_observations` (défaut 250) — en dessous, GARCH non fiable
  - **b) Proportion de NaN** : `part_nan <= data.max_nan_ratio` (défaut 10%) — trous excessifs rejetés
  - **c) Prix strictement positifs** : `(prix > 0).all()` — log-rendements impossibles si prix ≤ 0
  - **d) Série non constante** : `std(rendements) >= 1e-8` — ticker suspendu/gelé détecté
- Chaque contrôle lève `DataError` (exceptions.py) avec `etape='data_validation'` et contexte documenté

### Condition 2 — Robustesse téléchargement (`telecharger_prix`)

- `yf.download` wrappé dans `try/except` → `DataError` typée si exception réseau/ticker invalide
- `raise ValueError` (ligne ~35) remplacé par `raise DataError(..., etape='download', ticker=ticker)`
- TODO DETTE_TECHNIQUE documenté : retry exponentiel — volontairement absent pour ne pas masquer un problème de ticker

### Condition 3 — Config + schéma Pydantic

- `config.yaml` : ajout `data.min_observations: 250` et `data.max_nan_ratio: 0.10`
- `core/config_schema.py` `DataConfig` : ajout `min_observations: Optional[int] = Field(ge=50)` et `max_nan_ratio: Optional[float] = Field(gt=0, lt=1)` — compatible extra='forbid'
- `main.py` : import `valider_donnees` + appel juste après `valider_config()`, avant l'étape ARIMA

### Tests ajoutés (`tests/test_phase8_robustesse.py`)

- R1 : 40 obs < 250 → `DataError` (+ variante seuil custom + borne incluse)
- R2 : 30% NaN > 10% → `DataError` (contexte `n_nan`, `part_nan` vérifiés)
- R3 : prix négatif et prix zéro → `DataError` (2 tests)
- R4 : série constante (std=0) → `DataError` (contexte `std_rendements`)
- R5 : série propre 800 obs → aucune exception levée
- R6 : `config.yaml` avec nouvelles clés passe `valider_config_structure` (Phase 7)
- R7 : `telecharger_prix` lève `DataError` (pas `ValueError`) sur DataFrame vide

Total : 154 → 165 tests passés, 0 régression.

## [Phase 7 — Validation Pydantic] — feat/v6-phase7-pydantic — 2026-06-17

### Condition 1 — Schéma Pydantic (`core/config_schema.py`)

- Nouveau module `core/config_schema.py` avec modèles Pydantic v2 pour tous les blocs scientifiques
- **Blocs scientifiques** (`extra='forbid'`, `strict=True`) : `data`, `arima`, `garch`, `fhs`, `dm_gk`, `component_garch`, `var`, `backtest`, `structural_breaks`, `bootstrap`, `rolling_backtest`
  - `extra='forbid'` : toute clé inconnue lève immédiatement une `ValidationError`
  - `strict=True` : rejette les coercions implicites (ex. `"0.05"` str → float échoue)
  - Contraintes numériques : `p_max` ∈ [1,10], `split_ratio` ∈ [0.5, 1.0), `seuil_*` ∈ (0,1), etc.
  - `frequency` : `Literal['daily', 'weekly', 'monthly']`
  - `critere_information` : `Literal['AIC', 'BIC', 'HQIC']`
  - `structural_breaks.mode` : `Literal['diagnostic', 'integrate', 'off']`
  - Clés LEGACY/DEPRECATED explicitement déclarées pour passer avec `extra='forbid'` : `tolerance_aic_parcimonie`, `seuil_engle_ng` (garch), `trim`, `icss`, `zivot_andrews`, `injecter_dans_garch` (structural_breaks)
  - Validation des dates : format `YYYY-MM-DD` + `start_date < end_date`
- **Blocs infra** (`extra='allow'` via racine) : `output`, `events`, `rapport`, `sorties_etendues`, `ai`, `ai_writer`, `monitoring`, `export_academique`, `frtb` — acceptés sans validation de structure
- **Racine** `TickerLabConfig` : `extra='allow'` pour `_reuse_cache`, `_metrics_only`, `cache_v2`, etc.
- `valider_config_structure(raw: dict) -> dict` : lève `ValidationError` (exceptions.py) avec message lisible, retourne le dict original inchangé si valide

### Condition 2 — Intégration `main.py`

- Import ajouté : `from tickerlab.core.config_schema import valider_config_structure`
- Appel inséré juste après `yaml.safe_load(f)`, avant toute autre étape

### Dépendances

- `requirements.txt` : ajout `pydantic>=2.0`
- `pyproject.toml` : ajout `pydantic>=2.0` dans `dependencies`

### Tests ajoutés (`tests/test_phase7_pydantic.py`)

- P0 : `config.yaml` de production passe la validation sans modification (test critique)
- P1 : `garch.seuil_significativite='0.05'` (string) → `ValidationError`
- P2 : `backtest.split_ratio=1.5` (hors borne) → `ValidationError`
- P3 : `data.frequency='hourly'` (hors Literal) → `ValidationError`
- P4 : clé inconnue `garch.parametre_inexistant_xyz` (extra='forbid') → `ValidationError`
- P5 : `start_date='2026-01-01'` > `end_date='2006-01-01'` → `ValidationError`
- P6 : non-régression — valeurs numériques et structurelles inchangées après validation

Total : 147 → 154 tests passés, 0 régression.

## [Phase 6 — Intégrité] — feat/v6-phase6-integrite — 2026-06-17

### Condition A (HIGH) — Écriture cache atomique

- `core/cache_v2.py` `set()` : écriture dans `{etape}_{cle12}.pkl.tmp` puis `os.replace()` vers le `.pkl` final. En cas d'interruption, le `.pkl` final reste intact ou absent — jamais tronqué
- `os.fsync()` avant le rename pour garantir l'écriture disque effective
- Le manifest (`manifest.json`) est mis à jour **seulement après** le `os.replace()` réussi
- `_sauver_manifest()` : même pattern atomique via `manifest.json.tmp`
- `_supprimer_anciens_pkls()` : nettoie aussi les `{etape}_*.pkl.tmp` résiduels d'un run interrompu
- `get()` : supprime silencieusement un `.pkl.tmp` résiduel au chargement (`logger.debug`)
- `_lire_manifest()` : supprime silencieusement un `manifest.json.tmp` résiduel au démarrage
- Utilise `os.replace()` (PAS `os.rename`) pour la compatibilité Windows

### Condition B (MEDIUM) — Déterminisme prouvé

- `core/var_engine.py` : ajout `import logging` + `_log = logging.getLogger(__name__)`
- Fallback `bootstrap.seed=null` : émet `_log.warning('... NON reproductible ...')` au lieu de passer silencieusement
- Comportement par défaut (`seed=42`) strictement inchangé numériquement
- Nouveau test de reproductibilité bit-exact (B1, B2, B3) : FHS run1 == run2, Bootstrap CI run1 == run2, warning émis si seed=null

### Tests ajoutés (`tests/test_phase6_determinisme.py`)

- B1 : FHS VaR et ES identiques entre deux runs avec seed=42 (`==`)
- B2 : Bootstrap CI DataFrame identique entre deux runs (`df.equals()`)
- B3 : Warning `NON reproductible` émis quand `bootstrap.seed=null`
- A1 : Interruption pickle.dump simulée → aucun `.pkl` tronqué, aucun `.pkl.tmp` résiduel
- A2 : `set()` réussi → aucun `.pkl.tmp` résiduel, valeur récupérable par `get()`
- A3 : `get()` supprime un `.pkl.tmp` résiduel laissé par un crash précédent

Total : 141 → 147 tests passés, 0 régression.

## [Phase 5 — Rigueur DM-GK] — feat/v6-phase5-dmgk-rigueur — 2026-06-17

### Phase 5 — Rigueur econometrique DM + GK

**Condition 1 (HIGH) — Degenerescence et verdict 'non_discriminable'**
- `core/dm_gk.py` : `dm_test()` retourne `'degenerate': True/False` dans tous les chemins de retour
- `gk_test()` : retourne `'degenerate': True/False` et `'reliable': True/False`
- `_determine_verdict()` : nouveau parametre `dm_degenerate`, `gk_degenerate` (defaut False) ; verdict `'non_discriminable'` pris en PREMIER avant les 4 verdicts existants
- `comparer_methodes_var()` : propage `dm.get('degenerate', False)` et `gk.get('degenerate', False)` a `_determine_verdict()`
- `VERDICTS_LABELS` : ajout de `'non_discriminable'`
- Docstring module : verdicts (4) → verdicts (5)
- Motivation : `se < 1e-10` (DM) ne prouve PAS l'equivalence des modeles — limite methodologique documentee

**Condition 2 (MEDIUM) — Omega singuliere : pval fiable vs nan**
- `gk_test()` branche pinv (cond > 1e10 ou LinAlgError) : `pval=nan`, `reliable=False`, `reject=False`
- `gk_test()` branche normale (inv()) : `reliable=True`
- Aucune cle existante supprimee (`stat`, `pval`, `df`, `reject` preservees)
- Motivation : chi2.cdf(stat, df=q) non fiable si Omega singuliere (pseudo-inverse biais la distribution)

**Condition 4 (LOW) — Commentaire H=1 dans build_var_series**
- `build_var_series()` : commentaire expliquant que GARCH dyn. et FHS partagent `vol_oos`, `mu_bt`, pool z a H=1 → differentiels `d_t ≈ 0` → DM potentiellement degenere → `'non_discriminable'`
- Zero changement de calcul

## [Phase 4.1] — feat/v6-phase4-performance — 2026-06-17

### Phase 4.1 — Performance Pipeline (~50% gain GARCH, profiling complet)

**Étape 0 — Profiling**
- `main.py` : timings ajoutés pour `igarch_diag`, `component_garch`, `fhs`, `dm_gk`
- Sortie `[PERF]` par étape en fin de run (logger.info)
- `result.meta['timings']` remplace `duree_par_etape_s` (9 étapes couvertes)

**Condition 1 (HIGH) — z_t cache GARCH**
- `core/garch_selector.py` : `grid_search_garch(build_z_cache=True)` collecte `(z, params)` lors de la grille
- `tester_specification(z_cache=)` : réutilise le z mis en cache, évite ~40-60 ré-estimations en Étage 2
- `selectionner_meilleur(z_cache=)` + `_selectionner_scientifique(z_cache=)` : propagation
- `main.py` : appel site mis à jour

**Condition 2 (MEDIUM) — Component GARCH timeout**
- `core/component_garch.py` : `maxiter` 500 → 200 sur les 2 appels `minimize()` (primary + extra starts)
- `scripts/run_multi_ticker.py` : `fut.result(timeout=240)` + catch `FutureTimeout` (mode `--workers > 1`)
- Ferme **PERF-001** dans `docs/DETTE_TECHNIQUE.md`

**Condition 3 (LOG) — LOG-001 déjà résolu**
- `core/dm_gk.py` : `_log.warning()` remplace `print()` (résolu Phase 3.1)
- Ferme **LOG-001** dans `docs/DETTE_TECHNIQUE.md`

**Condition 4 (LOW) — FHS version bump**
- `core/fhs.py` : `__etape_version__ = '2'` (pre-sampling `(n_boot, H_max)` déjà implémenté)
- Invalide l'ancien cache FHS pour forcer recalcul propre

## [Phase 3.3] — feat/v6-phase3-industrialisation — 2026-06-13

### Phase 3.3 — Multi-providers LLM + CLI pip-installable (4 commits, 103/103 tests)

**Added**

- **`utils/llm_providers.py`** : abstraction multi-providers LLM.
  - `LLMResponse` dataclass : `contenu`, `tokens_in`, `tokens_out`, `provider`, `modele`.
  - `LLMProvider` Protocol (runtime-checkable) : `generer(system, user, max_tokens)`.
  - `AnthropicProvider` : SDK officiel, comportement Phase 3.2 préservé, `temperature=0`.
  - `OpenAICompatibleProvider` : Groq, Gemini, Ollama, OpenAI via `openai` SDK.
    Quirks par provider : Groq → `Retry-After` header ; Gemini → `usage` fallback ;
    Ollama → dummy API key. Backoff exponentiel 3 tentatives.
  - `fabriquer_provider(config)` : lecture de `config['ai_writer']` — `fournisseur`,
    `modele`, `env_key` (nom de la variable d'env, **jamais la valeur**).
    Fallback rétrocompat : `config['ai']['model']` si `ai_writer` absent.
  - **Sécurité** : aucune clé API ne transite par config.yaml ni le code source.
- **`cli.py`** + **`pyproject.toml`** : CLI pip-installable.
  - `pea-brent analyze --ticker BZ=F --frequency weekly [--no-ai] [--force-refresh]`
  - `pea-brent batch --config ... [--output-root ...]` — délègue à `run_unique()` via
    `importlib.util` (scripts/ n'est pas un package Python).
  - `pea-brent stress --ticker BZ=F --scenario covid_crash [ukraine_war ...]` :
    appelle `run_pipeline()` en premier (HIT cache si disponible), extrait
    `sigma_t = garch_final.conditional_volatility.iloc[-1]`,
    `dist = best['dist']`, `nu = garch_final.params.get('nu')`,
    puis appelle `appliquer_scenario()`.
  - `pea-brent --version`
  - `pyproject.toml` : layout root-IS-package via `package-dir`, dépendances
    optionnelles `[ai] = [anthropic, openai]`, entry point `pea-brent = pea_brent_v5.cli:main`.

**Changed**

- **`utils/ai_writer.py`** :
  - `_substituer_variables()` : clé absente → `'[DONNÉE MANQUANTE]'`
    (défense structurelle anti-hallucination — ne repose pas sur les instructions LLM).
  - `SYSTEM_PROMPT` : clause explicite `[DONNÉE MANQUANTE]` ajoutée.
  - `rediger_section(section_id, contexte, provider, prompts_dir, max_tokens)` :
    paramètre `provider: LLMProvider` remplace `modele_anthropic: str`.
    Retour harmonisé en `tuple[str, int, int]` partout (fix ligne 331 : `str` → tuple).
  - `rediger_toutes_sections()` : idem, accepte `provider`.
- **`main.py`** `_etape_ia()` : instancie le provider via `fabriquer_provider(config)` ;
  supprime la vérification `ANTHROPIC_API_KEY` en dur (chaque provider gère sa validation).

**Tests**

- `tests/test_llm_providers.py` : 5 tests — provider par défaut Anthropic, provider Groq,
  fournisseur inconnu `ValueError`, clé env absente `RuntimeError` (sans appel réseau),
  clé contexte absente → `'[DONNÉE MANQUANTE]'` (défense anti-hallucination).
- `tests/test_cli.py` : 4 tests — `--version`, `analyze --help`, `stress --help`,
  package installé (`__version__` défini, sous-packages importables).
- **Total : 103/103 pass** (0 régression vs Phase 3.2 : 94/94).

---

## [Phase 3.2] — feat/v6-phase3-industrialisation — 2026-06-12

### Phase 3.2 — Cache versionné intelligent + correctif config_hash (4 commits, 92/92 tests)

**Added**

- **`core/cache_v2.py`** : `CacheEtapes` — cache par étape avec invalidation en cascade.
  - DAG acyclique : 9 étapes (download → arima → garch → igarch_diag → component_garch,
    fhs, dm_gk, var, backtest). Clé récursive pure SHA-256 12 hex = f(config_deps,
    upstream_keys, `__etape_version__`) — calcul déterministe, indépendant de l'état disque.
  - API : `get(etape)`, `set(etape, valeur)`, `cle(etape)`, `invalider(etape=None)`, `stats()`.
  - Storage : `{dossier}/.cache_v2/{etape}_{cle12}.pkl` + `manifest.json`.
  - TTL download : si `data.end_date` absent → re-téléchargement après `ttl_download_heures`
    (défaut 24h) ; si `end_date` explicite → cache illimité.
- **`__etape_version__ = '1'`** ajouté dans 9 modules :
  `data_loader`, `arima_selector`, `garch_selector`, `igarch_diagnostic`,
  `component_garch`, `var_engine`, `fhs`, `backtest`, `dm_gk`.
  Bumper ce numéro invalide automatiquement les descendants en cascade.

**Changed**

- **`main.py`** :
  - `run_pipeline(config, verbosity='normal', force_refresh=False) -> PipelineResult` :
    nouveau paramètre `force_refresh` (ignore les lectures, réécrit le cache).
  - Cache v2 activé par défaut (`config.cache_v2.enabled = true`).
  - Pattern single-get par étape : `_cached = cv2.get(e) if (cv2 and not force_refresh) else None`.
  - Étape 'garch' : tuple `(df_garch, best, motif, trace_garch, garch_final)` stocké **après**
    `selectionner_meilleur()` + `estimer_final()` — df_garch contient les colonnes spec enrichies
    in-place (`lb_z_pval`, `lb_z2_pval`, `engle_ng_pval`, `tous_passent`).
  - Ancien cache (`utils/cache.py`) uniquement si `cache_v2.enabled=False` (mode legacy explicite).
  - `meta['cache']` : dict étape → `'HIT'`/`'MISS'` dans `PipelineResult`.
- **`core/config_validation.py` — correctif 3.1** :
  `_CLES_SCIENTIFIQUES` + `'component_garch'` et `'stress_testing'` (manquaient, bug silencieux :
  modifier ces paramètres ne changeait pas le `config_hash`).

**Tests**

- `tests/test_phase32.py` : 9 nouveaux tests — round-trip pickle (df.attrs), invalidation
  aval (fhs change → fhs MISS, download/garch HIT), invalidation cascade amont (garch.p_max),
  invalidation version module (monkeypatch `__etape_version__`), TTL download (25h/end_date),
  df_garch enrichi dans le cache (colonnes spec présentes), force_refresh, disabled_strict,
  non-régression froid/chaud (var99/persistance/modele strictement identiques).
- `tests/test_phase31.py` : `test_config_hash_cles_scientifiques` enrichi de 2 assertions
  (`component_garch` et `stress_testing` → hash change).
- **Total : 92/92 pass** (0 régression vs Phase 3.1 : 83/83).

**Benchmark cache_v2 (BZ=F_synth weekly, n=1500, 3 modèles × 3 dist × p,q≤2)**

| Étape        | Cold     | Warm     | Speedup | HIT/MISS |
|---|---|---|---|---|
| download     | ~4 s *   | ~0.01 s  | ~400×   | HIT      |
| garch        | 8.59 s   | 0.16 s   | 52×     | HIT      |
| igarch_diag  | 0.02 s   | 0.32 s   | —       | HIT      |
| var          | 0.11 s   | 0.07 s   | 2×      | HIT      |
| backtest     | 0.21 s   | 0.05 s   | 4×      | HIT      |
| **TOTAL**    | **8.93 s** | **0.60 s** | **15×** | |

*download mesuré sur réseau yfinance réel (hors benchmark synthétique)  
Warm = **6.8% du cold** ≤ 20% ✓

**Non-régression (test 9)** : sur série synthétique seed=7, n=500, `var99` et `persistance`
identiques à 1e-6/1e-10 entre run froid et run chaud.

---

## [Phase 3.1] — feat/v6-phase3-industrialisation — 2026-06-12

### Phase 3.1 — API industrialisée (6 commits, 83/83 tests)

**Added**

- **`core/exceptions.py`** : hiérarchie `PipelineError` → `DataError`, `ModelError`,
  `ValidationError`, `ExportError`. Champs : `ticker`, `etape`, `contexte`.
- **`core/config_validation.py`** : `config_hash()` (SHA-256 12 hex sur clés scientifiques
  seulement — exclut `output`, `cache`, `ai`, `_metrics_only`) ; `valider_config()` migre
  `_valider_config_runtime()` de main.py avec `ValidationError` typée.
- **`core/pipeline_result.py`** : `PipelineResult` dataclass avec interface dict-like
  (`[]`, `get`, `keys`, `in`) pour rétrocompat Phase 1-2. Champ `meta` :
  `ticker`, `version`, `config_hash`, `duree_totale_s`, `duree_par_etape_s`, `timestamp`.
- **`core/_logging.py`** : `setup_pea_brent_logger(level)` idempotent (1 handler,
  `propagate=False`, niveau ajusté à chaque appel) ; `get_logger(name)`.

**Changed**

- **`main.py`** : `run_pipeline(config, verbosity='normal') -> PipelineResult` ;
  `verbosity='quiet'` → WARNING seulement (runner multi-tickers) ;
  `catch_warnings` autour de `calculer_var_tvar` re-émet `UserWarning` via logger
  (BTC-001 toujours visible) ; `_valider_config_runtime` supprimée → `valider_config()` ;
  `duree_par_etape_s` par étape (donnees, arima, garch, var, backtest).
- **122 `print()` → `logger.info/warning`** dans 17 fichiers :
  `main.py`, `core/garch_selector.py`, `core/reporter.py`, `core/sorties_complementaires.py`,
  `core/depassements_annuels.py`, `core/backtest_rolling.py`, `core/stats_descriptives.py`,
  `core/correlogrammes.py`, `core/export_academique.py`, `core/rapport/_helpers.py`,
  `core/rapport/_orchestrateur.py`, `core/rapport/_sections.py`, `core/rapport/_stats.py`,
  `core/rapport/_themes.py`, `utils/ai_writer.py`, `utils/cache.py`, `utils/latex_export.py`,
  `utils/plotter.py`.
- **`scripts/run_multi_ticker.py`** : `run_pipeline(cfg, verbosity='quiet')` ;
  `PipelineError` capturé séparément avec `etape` dans le champ erreur CSV ;
  `config_hash` ajouté (35e colonne) ; `_extraire_metriques` lit `meta['config_hash']`.
- **`__init__.py`** : `__version__ = "6.3.1"`.
- **`tests/test_phase16bis.py`** : `test_validation_config_runtime_split_trop_petit`
  migré de `_valider_config_runtime` (supprimée) vers `config_validation.valider_config`.

**Tests**

- `tests/test_phase31.py` : 8 nouveaux tests (exceptions, config_hash, valider_config,
  PipelineResult, logger idempotent, 35 colonnes runner).
- **Total : 83/83 pass** (0 régression vs Phase 2 : 75/75).

---

## [Phase 2.3-fix] — feat/v6-phase2-validation — commit 5ff3cef

### Phase 2.3-fix — Corrections scientifiques stress testing (3 bugs + BTC-001)

**Fixed**

- **`core/stress_scenarios.py`** — 3 corrections :
  - `_proba_queue_t_standardisee(z, nu)` : `z_scipy = z × √(ν/(ν−2))` — la t standardisée
    arch (Var=1) ≠ scipy.t (Var=ν/(ν−2)) ; correction divise les probabilités par 2–10×
  - Probabilité directionnelle : `proba_queue = CDF(z)` pour perte, `1−CDF(z)` pour gain.
    Renommé `proba_unilaterale` → `proba_queue`. Correction : oil_shock BZ=F affichait
    P=99.98% (queue inférieure d'un choc positif) → corrigé à P=0.0074% (queue supérieure)
  - `_sigma_horizon_exact(garch_final, sigma_t, H)` via récurrence GARCH/IGARCH analytique +
    `methode_sigma_H` dans le dict de retour. Fallback `sqrt_H_EGARCH` pour EGARCH.
    Paramètre `garch_final=None` ajouté à `appliquer_scenario()` et `appliquer_tous_scenarios()`

- **`core/reverse_stress.py`** : même correction t standardisée

- **`core/var_engine.py`** — BTC-001 floor :
  - `_apply_btc001_floor(r_g, ticker)` : floore VaR GARCH < −99.9% avec `UserWarning [BTC-001]`
  - Applied au niveau 99% dans `calculer_var_tvar()` ; flag `df.attrs['var99_garch_floored']`

- **`scripts/run_multi_ticker.py`** : colonne `var99_garch_floored` ajoutée (34 colonnes)

- **`tests/test_phase23.py`** : 4 nouveaux tests (test 6–9) + correction test 1

- **`docs/PHASE2_3_STRESS_TESTING.md`** : probabilités recalculées avec t standardisée corrigée

- **`docs/DETTE_TECHNIQUE.md`** : BTC-001 → statut RESOLU

**Résultats corrigés (BZ=F, ^GSPC, EURUSD=X daily, juin 2026)** :

| Scénario | Ticker | Choc | Distance | P(choc) corrigée | Multiple VaR99 |
|---|---|---|---|---|---|
| oil_shock_2022 | BZ=F | +30% / H=5 | 7.74σ | 0.0074% (gain) | 6.00× |
| covid_march_2020 | BZ=F | −65% / H=22 | 8.00σ | 0.0063% | 13.0× |
| covid_march_2020 | ^GSPC | −34% / H=22 | 11.99σ | 0.0006% | 19.7× |
| fed_hike_2022 | EURUSD=X | −3% / H=1 | 9.68σ | 0.00041% | 3.91× |

**Note :** suite complète 75/75 tests pass — 0 régression.

---

## [Phase 2.3] — feat/v6-phase2-validation — commit a018ad8

### Phase 2.3 — Stress testing scénario-based + reverse stress

**Added**

- **`core/stress_scenarios.py`** — module stress testing actif unique :
  - `SCENARIOS_BASE` : 4 scénarios (oil_shock_2022, covid_march_2020,
    fed_hike_2022, geopolitique_taiwan) avec shocks, horizon, référence,
    fenêtre historique optionnelle
  - `appliquer_scenario()`, `appliquer_tous_scenarios()`, `scenarios_pour_ticker()`

- **`core/reverse_stress.py`** — reverse stress univarié (BCBS d365 §44)

- **`tests/test_phase23.py`** — 5 tests (corrigés et étendus à 9 en Phase 2.3-fix)

- **`docs/PHASE2_3_STRESS_TESTING.md`** — rapport validation (recalculé en 2.3-fix)

---

## [Phase 2.2] — feat/v6-phase2-validation

### Phase 2.2 — Catalogue des cas pathologiques

**Added**

- **`docs/CAS_PATHOLOGIQUES.md`** — catalogue formel des actifs non modélisables
  ou à risque par GARCH standard. 4 catégories issues des 12 runs Phase 2.1 :
  - Cat. 1 — Non modélisables : BTC-USD (daily+weekly), ^GSPC daily
  - Cat. 2 — Hétéroscédasticité non détectable : GC=F weekly, EURUSD=X weekly
    (0/6 sig_vol=True — fallback IC_seul_AUCUN_VALIDE), CAS-01 EUR/USD Component GARCH
  - Cat. 3 — near-IGARCH : GC=F daily, ^GSPC daily, EURUSD=X daily, BTC-USD
    (pers ≥ 0.987, VaR H>1 déconseillée)
  - Cat. 4 — T_oos < 250 : toutes séries weekly (T_eff_dyn=173)
  - Checklist pré-analyse + tableau de décision rapide

- **`tests/test_phase22_catalogue.py`** — 3 tests structurels (< 20s) :
  - `test_catalogue_present_et_categories` — fichier présent + 4 catégories
  - `test_entrees_cat1_champs_obligatoires` — champs Symptômes/Diagnostic/
    Recommandation/Verdict présents, BTC-USD et ^GSPC daily documentés
  - `test_catalogue_markdown_parseable` — markdown parseable, ≥ 4 headers,
    ≥ 1 tableau rendu

**Note :** suite complète 66/66 tests pass — 0 régression.
  `pytest tests/test_phase22_catalogue.py -v` : 3/3, 18s.

---

## [Phase 2.1] — feat/v6-phase2-validation

### Phase 2.1 — Runner multi-tickers (validation croisée des actifs)

**Added**

- **`scripts/run_multi_ticker.py`** — runner CLI Phase 2.1 pour validation croisée
  sur 6 actifs × 2 fréquences = 12 runs.

  **CLI :**
  ```
  python scripts/run_multi_ticker.py                          # 12 runs complets
  python scripts/run_multi_ticker.py --quick                  # 3 runs (BZ=F, ^GSPC, EURUSD=X weekly)
  python scripts/run_multi_ticker.py --workers 2              # parallelisme Pool
  python scripts/run_multi_ticker.py --tickers BZ=F CL=F     # tickers custom
  ```

  **Tickers :** `BZ=F`, `CL=F`, `^GSPC`, `GC=F`, `EURUSD=X`, `BTC-USD`
  (start_date=2015-01-01 pour couvrir BTC-USD).

  **Grid GARCH réduit Phase 2.1 :** GARCH / GJR-GARCH / EGARCH × normal / t × p,q=1
  (9 fits/run). Choix délibéré pour itération rapide (45-60 min vs 6-8h grid complet).
  Les modèles retenus peuvent différer de Phase 1 (grid complet). À documenter dans
  l'analyse : trade-off itération vs comparabilité.

  **CSV incrémental 33 colonnes** (`dev/multi_ticker_summary.csv`) :
  `ticker`, `frequence`, `date_run`, `n_obs`, `duree_s`, `T_train`, `T_eff_dyn`,
  ARIMA (p/d/q/interpretation), `motif_selection`,
  GARCH (modele/aic/persistance/near_igarch/all_candidates_failed_spec/spec_gravite),
  VaR (95%/99% GARCH + TVaR 99% + FHS H=1),
  Backtest (Kupiec 95%/99% + Christoffersen 95%/99%),
  DM-GK (stat + pval GARCH vs FHS),
  Component GARCH (estime bool + rho),
  `bootstrap_ic_largeur_pct`, `erreur`.
  **Écriture après chaque run** — pas de perte si crash mid-run.

  **Isolation erreurs :** chaque run dans try/except ; `erreur.txt` par run_dir.
  `rolling_backtest.enabled=False` forcé (évite `_valider_config_runtime` ValueError
  sur séries courtes type BZ=F weekly).

  **Parallélisme :** `concurrent.futures.ProcessPoolExecutor` (picklable, Windows safe).

- **`tests/test_phase21.py`** — 4 tests < 30s (run_pipeline mocké) :
  - `test_run_unique_wrapper_ok` — pipeline mocké → erreur vide, métriques extraites
  - `test_run_unique_pipeline_exception` — RuntimeError → erreur capturée, erreur.txt créé
  - `test_csv_colonnes_obligatoires` — 33 colonnes attendues, CSV parseable
  - `test_csv_append_deux_runs` — 2 appels successifs, 2 lignes, sans collision

**Changed (refactor)**

- **`main.py` `run_pipeline()`** : return enrichi de 9 clés supplémentaires :
  `igarch_diag`, `component_garch`, `fhs`, `dm_gk`, `T_train`, `T_eff_dyn`,
  `df_garch`, `spec_verdict`, `motif_selection`.
  Rétrocompatible : les 8 clés originales sont inchangées.

- **`docs/PHASE2_1_ANALYSE_MULTI_TICKERS.md`** — rapport d'analyse complet post 12 runs.
  Contenu : tableau synthèse 12 tickers, 4 catégories de comportement, 4 bugs/anomalies
  identifiés (dont BTC-001 VaR critique), comparaison daily vs weekly, implications
  Phase 2.2 (catalogue) et Phase 2.3 (stress testing).

- **Mode `_metrics_only`** dans `run_pipeline()` et CLI `--full-output` :
  early return avant génération PDF/Excel/LaTeX quand `config['_metrics_only']=True`.
  Gain mesuré : BZ=F/^GSPC 80s → 25s. Outlier ^GSPC daily : 270s (Component GARCH
  near-IGARCH lent, voir PERF-001).

**Bugs découverts (enregistrés dans `docs/DETTE_TECHNIQUE.md`)** :

- **BTC-001 CRITIQUE** : VaR GARCH non plafonnée à -100%. Sur BTC-USD weekly avec
  pers=0.998 et σ_t élevée, VaR95=-200% et VaR99=-357%. Mathématiquement correct
  (σ_t × q_ν ≈ 360%) mais physiquement absurde. Correctif : floor à -99.9% ou
  substitution automatique FHS quand σ_t > 50% weekly. DM-GK retourne `math range error`
  en cascade.
- **DM-001 MINEUR** : DM-GK `math range error` sur BTC-USD weekly, causé par BTC-001.
- **PERF-001 MINEUR** : Component GARCH near-IGARCH (rho → 1) peut atteindre 270s sur
  dataset de 2765 observations. Aucun timeout par run dans le runner actuel.
- **LOG-001 COSMÉTIQUE** : `gk_test: Omega mal conditionnee` émis sur ~80% des runs.
  Comportement géré par `pinv`, mais verbeux. Réduire en niveau DEBUG en Phase 3.1.

**Résultats Phase 2.1 (12 runs, `--workers 2`, mode metriques-only)** :

| Catégorie | Tickers | Comportement |
|---|---|---|
| A — Sains | BZ=F weekly, CL=F weekly, ^GSPC weekly | Kupiec 95%+99% OK, spec_OK |
| B — VaR95 sous-couverte | BZ=F daily, CL=F daily, EURUSD=X daily | Kupiec 99% OK, daily plus difficile |
| C — near-IGARCH | GC=F daily, ^GSPC daily | VaR sous-couverte, persistance quasi-unitaire |
| D — Fallback sig_vol=0 | GC=F weekly, EURUSD=X weekly | GARCH détecte pas l'hétéroscédasticité weekly |
| E — Non modélisables | BTC-USD daily+weekly, ^GSPC daily | spec majeure ou VaR bug |

**Note :** suite complète 63/63 tests pass — 0 régression.
  `pytest tests/test_phase21.py -v` : 4/4, 26s.


## [v6.0-phase1] — feat/v6-phase1-methodologie

### Phase 1.7 — Tests d'intégration cross-sous-tâches (smoke test pré-merge)

**Added**

- **`tests/test_phase1.py`** (nouveau, ~450 lignes) — fichier intégrateur distinct des
  7 fichiers `test_phase1X.py` existants. Valide les ENCHAÎNEMENTS entre sous-tâches,
  pas chaque sous-tâche isolément. Critère CI : `pytest tests/test_phase1.py -v` → 8/8
  pass, < 90s.

  **8 tests d'intégration :**
  - `test_arima_interpretation_martingale` — série iid N(0,1) → interpretation ∈
    {`martingale_marche_efficient`, `bruit_faible`} (pas `serie_predictible` sur du bruit
    blanc). Pipeline 1.1 live.
  - `test_igarch_diagnostic_post_hoc` — série GARCH(1,1) pers≈0.99 → `near_igarch=True`,
    `half_life_periodes > 0`. Adapté : IGARCH non dans la grille (Phase 1.2 = post-hoc).
  - `test_component_garch_convergence` — simulation Component GARCH (ρ=0.97, 500 obs) →
    ρ estimé dans [0.97 ± 25%] ET contrainte Engle-Lee C3 : α+β < ρ. Tolérance 25% :
    multi-start SLSQP sur 500 obs.
  - `test_fhs_vs_garch_parametrique` — |VaR_FHS − VaR_GARCH| / |VaR_GARCH| < 30% aux
    niveaux 95% et 99%.
  - `test_dm_gk_symmetrie` — |DM(A,B) + DM(B,A)| < 0.01 (tolérance HAC Newey-West).
  - `test_bootstrap_express_taille_ic` — IC_inf ≤ VaR_GARCH ≤ IC_sup sur 3 séries
    (vol forte, vol modérée, queues épaisses).
  - `test_igarch_diagnostic_declenche_component_garch` — **intégration 1.2→1.3** :
    near_igarch=True → Component GARCH estimé avec `force_estimation=False`. Anti-régression
    contre découplage silencieux du pipeline. Contrainte C3 vérifiée.
  - `test_fhs_apparait_dans_dm_gk` — **intégration 1.4→1.5** : FHS dans `df_bt` via
    `backtest_oos()` automatique, `build_var_series` + `comparer_methodes_var` incluent FHS,
    au moins une paire DM impliquant FHS. Anti-régression contre "FHS sort du tableau
    silencieusement".

- **`tests/test_phase11.py`** — renommage de l'ancien `test_phase1.py` (Phase 1.1
  unitaire, 6 tests) pour libérer `test_phase1.py` pour le fichier intégrateur.

**Note** : suite complète 59/59 tests pass — 0 régression.
  `pytest tests/test_phase1.py -v` : 8/8, 24s. `pytest tests/` : 59/59, ~120s.

### Phase 1.6bis-fix2 — Promotion gravité si tous candidats B&A échouent

**Changed**

- **`core/garch_selector.py`** — `verdict_specification()` : ajout d'une règle de promotion
  lorsque `all_candidates_failed_spec=True` ET `gravite='mineure'` (p-value marginale > 0.01
  mais < 0.05). Dans ce cas la gravité est promue à `'majeure'` avec un message ALERTE
  référençant l'échec systématique de la famille GARCH (HAR-RV Corsi 2009, MIDAS,
  Markov-Switching GARCH Haas-Mittnik-Paolella 2004).

  **Motivation** : scénario reproduit sur BZ=F (960 rendements hebdomadaires) — EGARCH[skewt]
  retenu, `lb_z_pval=0.0103`, mais 5/5 candidats Burnham-Anderson échouent la spécification.
  Isolément, p=0.010 serait "mineure" ; mais l'échec systématique de toute la famille GARCH
  traduit un problème structurel de l'asset class, non un rejet marginal du modèle retenu.
  La promotion "majeure" force l'analyste à investiguer des modèles alternatifs.

- **`tests/test_phase16bis.py`** — 2 tests ajoutés :
  - `test_verdict_promotion_mineure_to_majeure_si_all_failed` : p_min=0.0103 (marginal) mais
    4/4 candidats `tous_passent=False` → promotion 'majeure' + message asset-class.
  - `test_verdict_pas_de_promotion_si_seul_modele_echoue` : p_min marginal mais 1 candidat
    passe → `all_candidates_failed_spec=False` → gravité reste 'mineure', pas de promotion abusive.
  - `test_verdict_specification_alerte_mineure` : df corrigé avec 1 candidat `tous_passent=True`
    pour éviter l'activation de la promotion (test unitaire ciblé sur la règle p-value seule).

  **Note** : la promotion est déclenchée uniquement si gravité est déjà 'mineure' (p_min marginal).
  Si gravité est déjà 'majeure' (p_min < 0.01 ou lb_z2/engle), aucun changement.

**Note** : suite tests 14/14 OK (Phase 1.6bis) + 6/6 OK (Phase 1.6) — pas de régression.
  Validation sur données réelles BZ=F : 0/5 candidats passent spec → promotion déclenchée →
  gravite='majeure', all_failed=True, message ALERTE généré correctement.

### Phase 1.6bis-fix — Correctif latent verdict_specification

**Fixed**

- **`core/garch_selector.py`** — `_selectionner_scientifique()` ne persistait pas les colonnes de
  specification (`lb_z_pval`, `lb_z2_pval`, `engle_ng_pval`, `tous_passent`) dans le `df_garch`
  retourne au pipeline. Ces colonnes etaient calculees uniquement dans `df_val` (variable locale).
  Consequence : `verdict_specification()` trouvait `col_spec=None` sur donnees reelles et retournait
  systematiquement `all_candidates_failed_spec=False`, masquant les alertes "Tous les candidats
  Burnham-Anderson echouent la specification".

  **Correctif** : persistance in-place a la fin d'Etage 2, avant les returns anticipatifs.
  Modeles non passes par Etage 1 (non-convergents, sig_vol=False, non-stationnaires) recoivent
  `tous_passent=False` et `lb_*_pval=NaN` comme valeurs par defaut. Ces modeles ont `AIC=NaN`
  (non-convergents) ou un AIC qui peut les exclure de la fenetre Burnham-Anderson selon le cas.

- **`tests/test_phase16bis.py`** — `test_verdict_specification_alerte_majeure` reecrit en test
  hybride : (1) integration sur serie GARCH(1,1) seed=7 — verifie que `df_garch` contient
  `tous_passent` et que `all_candidates_failed_spec` est coherent avec `df_garch` reel (pas
  d'injection artificielle) ; (2) classification directe sur p-values injectees — verifie la
  logique majeure/ALERTE de `verdict_specification`. Ancien test : faux positif car injectait
  `tous_passent=False` dans un df_garch simule, code jamais teste en conditions reelles.

  Nouveau test `test_df_garch_contient_colonnes_spec` : regression garantissant que `df_garch`
  contient les 4 colonnes spec apres tout appel `grid_search_garch` + `selectionner_meilleur`.

**Note** : suite tests 12/12 OK (Phase 1.6bis) + 6/6 OK (Phase 1.6) — pas de regression.

### Phase 1.6bis — Correctifs audit PDF (3 bugs)

- **Bug 1 — Coherence frequence prix/rendements** : `core/data_loader.py` ajoute
  `resampler_prix(prix, freq)` (resample W-FRI / ME / YE / no-op daily). `section_1` dans
  `_sections.py` accepte `prix_stats=None` : graphique 1.1 garde la serie journaliere complete,
  statistiques 1.2 et correlogramme 1.3 utilisent `prix_stats` (frequence d'analyse). Note
  discrete sous 1.1 si `freq != 'daily'` : N obs journalieres vs N obs hebdomadaires. Sections
  2 et 3 recoivent directement `prix_stats`. `generer_pdf_unique()` accepte `prix_stats=None`.
  `main.py` calcule `prix_stats = resampler_prix(prix, freq)` immediatement apres
  `calculer_rendements()` (branches cache et non-cache) et le passe au PDF.

- **Bug 2 — Double verdict Backtest + Specification** : `core/garch_selector.py` ajoute
  `verdict_specification(best_dict, df_garch, config)` : lit `lb_z_pval`, `lb_z2_pval`,
  `engle_ng_pval` depuis `best_dict` (deja calcules par `selectionner_meilleur`) ; detection
  defensive du nom de colonne spec (`tous_passent` / `spec_OK` / `Sp. OK`) ; calcule
  `all_candidates_failed_spec` sur les candidats Burnham-Anderson (fenetre `tolerance_delta_critere_brut`) ;
  retourne `gravite` ∈ {aucune, mineure, majeure} + `message` actionnable. `_resume_executif()`
  dans `_orchestrateur.py` accepte `spec_verdict=None` : nouvelle ligne "Specification" (OK /
  ATTENTION / ALERTE) et logique recommandation 5 cas (Backtest OK + Spec OK → Usage standard ;
  Backtest OK + Spec mineure → Surveillance ; Backtest OK + Spec majeure → Audit ; etc.).
  `generer_pdf_unique()` calcule le verdict et le passe au resume.

- **Bug 3 — Rolling backtest window > n_total** : `core/backtest_rolling.py` valide en tete de
  `backtest_rolling_var()` : `ValueError` si `window >= n_total` (serie trop courte, message
  actionnable avec recommandation FRTB ≥1000 obs) ; `UserWarning` + auto-ajustement a
  `max(0.5*n, 250)` si `window >= n_total - 50` (message enrichi : FRTB, nb predictions OOS
  apres ajustement). Deuxieme `ValueError` si l'ajustement lui-meme est impossible (n trop court).
  `main.py` ajoute `_valider_config_runtime(config, n_obs)` : validation precoce post-download
  (`rolling_backtest.window_size` vs n_obs, `backtest.split_ratio` insuffisant → Warning si
  T_oos < 50).

- **Tests** : `tests/test_phase16bis.py` — 11 tests (4 bug1 + 4 bug2 + 3 bug3) — 11/11 OK.
  Regression Phase 1.6 : 6/6 tests inchanges.

### Phase 1.6 — Bootstrap stationnaire express IC VaR conditionnel

- **`core/var_engine.py`** — refonte de `calculer_bootstrap_ci_var()` : ajout du mode
  express (config `bootstrap.express: true`) actif meme si `enabled: false` ; isolation
  RNG stricte via `numpy.random.default_rng(seed)` passe en `seed=rng` a
  `StationaryBootstrap` (arch signature `seed: int | Generator | RandomState | None`) —
  zero effet de bord sur `np.random` global (isolation avec FHS `seed=42`). Mode express :
  200 replications, `block_length=10`, `niveaux_ic=[0.95]`, `seed=42`. Mode complet
  (`enabled: true`) : 500 replications, tous niveaux, parametres separement lisibles.
  Bootstrap conditionnel (Pascual-Romo-Ruiz 2006) : parametres GARCH fixes a leur
  estimation ponctuelle — quantifie l'incertitude de l'echantillonnage des residus, non
  l'incertitude parametrique. Fallback RNG local si `StationaryBootstrap` leve une exception.
- **`config.yaml`** — section `bootstrap:` enrichie : `express: true`, `n_replications: 200`,
  `block_length: 10`, `niveaux_ic: [0.95]`, `inclure_tvar: false`, `seed: 42`. Commentaires
  sur la distinction mode express / complet et la nature conditionnelle du bootstrap.
- **`core/rapport/_sections.py`** — section 8.1 : suppression de la garde `enabled=true` ;
  `calculer_bootstrap_ci_var()` est toujours appele (retourne DataFrame vide si ni express ni
  enabled). Note methodologique explicite : "Bootstrap stationnaire conditionnel (Politis-Romano
  1994), B=N replications, bloc=b. Parametres GARCH fixes : incertitude liee a l'echantillonnage
  des residus, non parametrique (Pascual-Romo-Ruiz 2006)." Adaptatif selon mode express ou complet.
- **`main.py`** — etape 4b apres VaR & TVaR : log console IC bootstrap express par niveau
  (`VaR GARCH {niv} = X%  IC 95% [lo%, hi%]`). Appel conditionnel si `express: true` ou
  `enabled: true`.
- **`tests/test_phase16.py`** — 6 tests : reproductibilite (meme seed -> IC identiques),
  rapidite express (200 reps < 30 s), isolation RNG (np.random global non perturbe), IC grandit
  avec la taille du bloc sur AR(1) phi=0.5 (`largeur(b=20) > largeur(b=1) * 1.10`), format
  DataFrame correct (colonnes, index, coherence lower <= VaR <= upper), DataFrame vide si
  express=false et enabled=false.
- **`scripts/validate_bootstrap_tickers.py`** — validation sur 3 tickers (BZ=F, ^GSPC,
  EURUSD=X) : IC non degenere, VaR in IC, temps total < 60 s, largeur >= 0.01%.
- **`docs/references.bib`** — 2 entrees : `politis_romano_1994` (Stationary Bootstrap, JASA
  89:428), `pascual_romo_ruiz_2006` (Bootstrap Prediction GARCH, CSDA 50:9).

### Phase 1.5 — Comparaison statistique des methodes VaR (DM + GK)

- **`core/dm_gk.py`** — nouveau module : tests Diebold-Mariano (1995) et Giacomini-Komunjer
  (2005, JBES 23:4) sur la perte tick/quantile (Gonzalez-Rivera et al. 2004).
  Fonctions : `tick_loss()` (L_alpha(u_t) = u_t*(alpha - 1{u_t<0})), `dm_test()` (HAC
  Newey-West, stat N(0,1) bilateral, lags auto = int(T^(1/3))), `gk_test()` (instruments
  Z_t = [1, d_{t-1}], stat chi2(q) unilateral droit — direction lue dans DM),
  `_determine_verdict()` (4 verdicts : model1_wins, model2_wins, egalite_stable,
  egalite_avec_instabilite), `build_var_series()` (series VaR OOS pour toutes les methodes,
  vol OOS via recursion GARCH manuelle — zero dependance a arch.fix()),
  `comparer_methodes_var()` (toutes paires disponibles C(7,2)=21, dict exhaustif).
  Choix explicite GK 2005 vs GW 2006 (note : asymptotiquement equivalents sur ce cas
  d'usage avec instruments [1, d_{t-1}] — Patton & Timmermann 2007). Seuil alpha=0.05
  par defaut (expose en config), avertissement si T_oos < 250 (puissance insuffisante).
  Warning si HAC_lags > T_oos/4. Isolation stricte de backtest.py (zero modification).
- **`core/rapport/_sections.py`** — nouvelle sous-section 8.8 : tableau 3 paires structurantes
  x 2 niveaux (99%, 95%) x {DM stat, p DM, GK stat, p GK, verdict}. Le dict de retour
  complet (21 paires) est disponible pour un futur tableau annexe exhaustif. Verdict
  egalite_avec_instabilite mentionne explicitement dans la note : "pas de gagnant moyen,
  profil temporel instable — investiguer regime-dependance".
- **`core/rapport/_orchestrateur.py`** — `generer_pdf_unique()` accepte
  `dm_gk_result: dict = None` (kwarg optionnel retro-compatible) et le passe a `section_8()`.
- **`main.py`** — etape 3e apres FHS : appel `build_var_series()` + `comparer_methodes_var()`,
  log console paire principale GARCH dyn. vs FHS (DM stat, p, GK stat, p, verdict). Cache +
  PDF enrichis. `dm_gk_r` initialise a None pour compatibilite cache.
- **`config.yaml`** — section `dm_gk:` avec `enabled`, `test_type` ('gk' par defaut, avec
  note sur alternative 'gw' justifiee), `alpha_test` (0.05), `alpha_test_petit_echantillon`
  (0.10 si T_oos < 500), `hac_lags` ('auto'), `n_instruments_gk` (2), `paires_section_principale`.
- **`tests/test_phase15.py`** — 6 tests : perte tick nulle a la frontiere exacte, DM stat=0
  si VaR1=VaR2, DM detecte le meilleur modele sur serie simulee, GK non-rejet si d_t iid
  (instruments non-informatifs), structure dict complet `comparer_methodes_var`, test critique
  `test_gk_detecte_instabilite_temporelle` (regime changeant a mi-parcours : DM non rejete,
  GK rejete — valide la valeur ajoutee du test conditionnel).
- **`docs/references.bib`** — 4 entrees : diebold_mariano_1995, giacomini_komunjer_2005,
  gonzalez_rivera_2004, giacomini_white_2006 (reference alternative GW).

### Phase 1.4 — FHS (Filtered Historical Simulation, Barone-Adesi et al. 1999)

- **`core/fhs.py`** — nouveau module : `calculer_var_fhs()` implémente la FHS multi-horizons
  H ∈ {1, 5, 10, 22} jours. Algorithme : extraction des résidus standardisés z_t = ε_t/σ_t,
  tirage avec remise depuis le pool historique, propagation stochastique de la variance GARCH
  le long de chaque chemin simulé. Vectorisé sur l'axe des n_boot chemins simultanément (boucle
  séquentielle conservée sur H — récurrence GARCH intrinsèque) : 3–5 s pour n_boot=10 000, H=22
  vs 30–60 s en boucle Python pure. Supporte GARCH, GJR-GARCH (asymétrie via indicateur signe),
  et EGARCH (récurrence log-variance Nelson 1991) ; APARCH approximé en GARCH(1,1) avec warning.
  Drapeau `residus_iid_ok` (LB(z²) + Engle-Ng joint) : si False, avertissement PDF section 8.5bis
  indiquant que la VaR FHS aux horizons longs peut être sous-estimée. Seed fixé à 42 par défaut
  (reproductibilité scientifique / audit Bâle BCBS d457). `fhs_var_oos()` : variante backtest
  OOS one-step-ahead déterministe (équivalent n_boot → ∞), vectorisée sur T_eff_dyn dates.
  Dict de traçabilité `fenetre_residus` (date_debut, date_fin, n_residus, z_min, z_max).
- **`core/backtest.py`** — `backtest_oos()` : méthode FHS ajoutée aux lignes du `df_bt`.
  z_pool strictement train ({z_τ : τ ≤ T_train − 1}) — commentaire explicite anti look-ahead bias
  (erreur classique des implémentations naïves). VaR FHS one-step-ahead déterministe ; Kupiec et
  Christoffersen calculés comme pour les autres méthodes (cohérence pour la phase 1.5 DM-GK).
- **`core/rapport/_sections.py`** — nouvelle sous-section 8.5bis : tableau comparatif
  VaR sqrt(H) / VaR GARCH sim / VaR FHS par horizon ; avertissement automatique si
  `residus_iid_ok = False` (clustering résiduel non capté → sous-estimation H>1).
- **`core/rapport/_orchestrateur.py`** — `generer_pdf_unique()` accepte `fhs_result: dict = None`
  (kwarg optionnel rétrocompatible) et le passe à `section_8()`.
- **`main.py`** — étape 3d après Component GARCH : appel `calculer_var_fhs()`, log console
  H=1/H=22 VaR_99 + iid_ok. Cache + PDF enrichis.
- **`config.yaml`** — section `fhs:` avec `enabled`, `n_boot` (10 000), `n_boot_backtest` (2 000),
  `horizons` ([1, 5, 10, 22]), `seed` (42).
- **`tests/test_phase14.py`** — 6 tests : monotonicité VaR en α, propagation GARCH conforme à la
  formule (σ²_{T+1} déterministe), reproductibilité bit-à-bit, scaling sous-linéaire vs √H sur
  série leptokurtique, absence de look-ahead bias (`fhs_var_oos` train-only strict), drapeau
  `residus_iid_ok = False` sur série z² autocorrélée.
- **`docs/references.bib`** — entrée BibTeX Barone-Adesi, Giannopoulos & Vosper (1999).

### Fixed (commit 531250f — post Phase 1.3)

- **Component GARCH — warmstart violait C3 sur séries near-IGARCH** : pour des séries
  avec persistance GARCH ≈ 0.98+ (ex. BZ=F α+β ≈ 0.979), `rho0 = min(ab+0.04, 0.97)`
  donnait `rho0 = 0.97 < ab = 0.979` → point de départ infaisable → SLSQP divergeait
  vers un minimum dégénéré (φ ≈ 27, ν ≈ 2, ΔAIC ≈ +70000). Correction : projection
  explicite sur C3 via `rho0 = max(ab+0.01, 0.97)` — garantit `rho0 > ab` toujours ;
  multi-start déterministe inconditionnel ajouté (2 points fixes : ρ=0.99 et ρ=0.97,
  φ petit), sélection du global best par log-vraisemblance.
- **`_fig_component_decomposition()`** — ajout dans `core/rapport/_sections.py` : courbes
  σ_t (gris) et √q_t (rouge bordeaux épais) superposées, annotations crises depuis
  `config.events.crises`. Embarquée en Figure 6.0bis (section 6.2bis du PDF).

### Phase 1.3 — Component GARCH (Engle & Lee, 1999) : décomposition permanente/transitoire

- **`core/component_garch.py`** — nouveau module : `estimer_component_garch()` décompose σ²_t en composante
  permanente q_t (pilotée par ρ) et transitoire (σ²_t − q_t, pilotée par α, β). Contraintes non-négociables
  Engle-Lee 1999 eq.7 p.482 : (C1) α+β < 1, (C2) 0 < ρ < 1, **(C3) α+β < ρ** (séparation des
  composantes). Optimisation SLSQP (scipy) — seule méthode supportant les inégalités non-boîte. MLE
  Student-t avec ν estimé (warmstarté depuis le GARCH retenu, contrainte ν > 2). Warmstart : α₀/β₀/ν₀
  extraits du GARCH final, ρ₀ = max(α₀+β₀+0.01, 0.97) — projection explicite sur C3. Erreurs standard : Hessien
  numérique (différences finies centrées, h=1e-4). Sentinelles φ-warning (|φ|>0.5) et saturation-warning
  (|α+β−ρ| < 1e-3). Déclenchement conditionnel : `enabled=true` ET (`force_estimation=true` OU
  `near_igarch=true`).
- **`core/rapport/_sections.py`** — `_tab_component_garch()` : tableau 6 lignes (ω, ρ, φ, α, β, ν)
  colonnes (Estimé, Std. Err., p-value, Contrainte Engle-Lee ✓/✗), surlignage rouge si C1 ou C2 violée,
  note C3 en footer. `_encadre_component_garch()` : encadré avec titre 6.2bis inséré dans `section_6()`
  après l'encadré IGARCH (Phase 1.2), kwarg rétrocompatible `component_garch_result: dict = None`.
- **`core/rapport/_orchestrateur.py`** — `generer_pdf_unique()` accepte `component_garch_result: dict = None`
  (kwarg optionnel) et le passe à `section_6()`.
- **`main.py`** — étape 3c après le diagnostic IGARCH : appel `estimer_component_garch()` avec
  `near_igarch=igarch_diag.get('near_igarch')`. Stockage cache + log console ρ̂/α̂+β̂/ν̂/AIC.
- **`config.yaml`** — section `component_garch : {enabled, force_estimation, seuil_persistance}`.
- **`tests/test_phase13.py`** — 6 tests : contraintes C1/C2/C3 post-estimation, warmstart faisable,
  φ-warning bool, force_estimation override, enabled=False→None, ν>2 fini.

### Phase 1.2 — IGARCH : diagnostic post-hoc (évolution de spec)

> **Note de changement de cap** : la sous-tâche 1.2 initiale prévoyait d'ajouter IGARCH
> à la grille de sélection GARCH (`MODELES_DISPO`, flag `inclure_igarch`, colonne
> `reference_only=True`). Cette approche a été remplacée par un **diagnostic post-hoc**
> sur le modèle déjà retenu. Justification : Mikosch & Stărică (2004, *Review of
> Economics and Statistics*, 86(1), 378-390) montrent que l'apparence IGARCH est
> souvent un artefact de ruptures structurelles dans la variance (Lamoureux & Lastrapes
> 1990) — ajouter IGARCH à la grille biaiserait la sélection AIC/BIC vers un faux
> positif systématique sur des actifs à crises récurrentes comme le Brent.

- **`core/igarch_diagnostic.py`** — nouveau module : `diagnostiquer_igarch()` calcule
  persistance, demi-vie double unité (périodes natives + jours calendaires avec facteur
  (7/5) pour le daily), test de Wald δ-méthode H₀: persistance=1 (χ²(1)), classement
  `igarch_strict`|`near_igarch`|`mean_reverting`. Gradient analytique pour
  GARCH/GJR-GARCH/EGARCH, numérique (différences finies centrées) pour TGARCH/APARCH.
  Sentinelle `None` (pas `float('inf')`) pour les demi-vies non-stationnaires.
- **`main.py`** — hook après `estimer_final()` : calcul + stockage cache + passage à `generer_pdf_unique()`.
- **`core/rapport/_orchestrateur.py`** — `generer_pdf_unique()` accepte `igarch_diagnostic: dict = None`
  (kwarg optionnel rétrocompatible) et le passe à `section_6()`.
- **`core/rapport/_sections.py`** — `_encadre_igarch()` : box thème-adaptive (fond
  `entete_fond`, bordure `warn`) pour `near_igarch`/`igarch_strict` ; ligne simple
  persistance+demi-vie pour `mean_reverting` (pas de box). `section_6()` enrichi du
  kwarg optionnel `igarch_diagnostic`.
- **`config.yaml`** — `garch.seuil_igarch: 0.98` (0.98-0.99 recommandé, calibré multi-actifs).
- **`tests/test_phase12.py`** — 7 tests : 3 codes + Wald fallback param_cov=None + gradient EGARCH + persistance ≥ 1 + rétrocompatibilité.

### Phase 1.1 — Diagnostic explicite "ARIMA non significatif"

- **`core/arima_selector.py`** — nouvelle fonction `_determiner_interpretation()` : taxonomie à 4 catégories mutuellement exclusives (`serie_predictible`, `bruit_faible`, `martingale_marche_efficient`, `incertain`). Logique : motif B&J × magnitude max(|φ̂, θ̂|) × test Ljung-Box 10 lags sur rendements bruts. Garde n < 100 → force `incertain`. Références : Fama (1970), Lo & MacKinlay (1988), Burnham & Anderson (2002).
- **`core/arima_selector.py`** — `selectionner_arima()` enrichi de 4 nouvelles clés rétrocompatibles : `interpretation`, `message_pedagogique`, `lb_pval_rendements`, `max_magnitude_coef`. Correction bug latent : `grid_search_arima()` retourne un DataFrame vide typé (colonnes explicites) si toutes les estimations échouent ; `selectionner_arima()` gère ce cas avec un fallback `grille_vide`.
- **`core/rapport/_sections.py`** — nouvelle fonction `_encadre_pedagogique()` : encadré thème-adaptatif (fond = `entete_fond`, bordure = `accent`) inséré dans `section_5` entre l'introduction et la grille AIC.
- **`utils/ai_writer.py`** — `construire_contexte()` expose 4 nouvelles clés de substitution : `ARIMA_INTERPRETATION`, `ARIMA_MSG_PEDAGOGIQUE`, `ARIMA_LB_PVAL`, `ARIMA_MAX_MAG_COEF`.
- **`tests/test_phase1.py`** — créé : 6 tests unitaires (4 catégories d'interprétation + rétrocompatibilité + flowables ReportLab).
- **Nommage** : `lb_pval_brut` harmonisé en `lb_pval_rendements` dans toute la base de code (paramètre, variable locale, clé du dict, tests).

## [v5.1] — fix/audit-v5

### Corrections (4 bugs audit)

- **Bug 1 — Conflit YAML `var:`** (`config.yaml`) — deux blocs `var:` causaient l'ecrasement silencieux de `niveaux` et `n_simulations_mc` par `horizons` et `n_simulations_horizons`. Fusionnes en un seul bloc avec separateur de section.
- **Bug 2 — Filtre params aberrants rolling** (`core/backtest_rolling.py`) — les estimations dont |param| > 10 sont desormais ecartees (divergence optimiseur), params precedents conserves. Compteur `n_divergent` affiche dans les logs.
- **Bug 3 — CUSUM non-standardise** (`core/monitoring.py`) — increment standardise par sigma_hit = sqrt(p(1-p)), seuil de reference k_std = delta/2 (Page 1954), seuil critique h_std = 5.0 (Hawkins-Olwell 1998). Passe de ~4000 alertes fantomes a O(10-50) alertes sur les crises reelles.
- **Bug 4 — Vol constante entre refits** (`core/backtest_rolling.py`) — propagation GARCH one-step-ahead via `arch_model.fix()` pre-calculee par refit (163 appels, ~0.64s overhead). Amelioration attendue des verdicts Kupiec/Christoffersen en §8.7.

## [Unreleased] — feat/industrialisation

### Ajouts

- **Cornish-Fisher monotonicity guard** (`var_engine.py`) — test dzc/dz > 0 (Maillard 2018) ; retourne NaN si non monotone, affiche N/A dans le PDF.
- **Tests de robustesse avances** (`core/tests_robustesse.py`, §10.4 PDF) — Berkowitz (2001) PIT/LR, DQ test Engle-Manganelli (2004), Diebold-Mariano tick loss Giacomini-Komunjer (2005), Sign Bias.
- **Horizons multiples Bale** (`var_engine.calculer_var_multi_horizon`, §8.5 PDF) — VaR 99% 1/5/10/22j : regle sqrt(H) vs simulation GARCH directe.
- **Benchmark EVT-POT** (`core/benchmark_evt.py`, §9.5 PDF) — GPD MLE sur exceedances (seuil 95e pct), VaR/TVaR analytiques, test KS.
- **Ruptures structurelles** (`core/structural_breaks.py`, §3.5 PDF) — CUSUM-OLS (Ploberger-Kramer 1992) + Chow sequentiel sans le package ruptures.
- **Monitoring dynamique** (`core/monitoring.py`, §10.5 PDF) — CUSUM des violations, ratio glissant 250j.
- **Bootstrap IC VaR** (`var_engine.calculer_bootstrap_ci_var`) — arch.bootstrap.StationaryBootstrap, desactive par defaut (`bootstrap.enabled: false`).
- **Rolling backtest** (`core/backtest_rolling.py`, §8.6 + §8.7 PDF) — fenetre 1000j, re-estimation mensuelle (22j), gestion gracieuse des echecs de convergence, stockage df_params_drift (163 re-estimations), DQ test sur violations rolling. Cache bidirectionnel. Timing BZ=F : ~0.17s/estim → 28s total.
- **Resume executif** (`_orchestrateur.py`) — page 2 avec modele retenu, VaR/TVaR 99%, VaR 10j Bale, verdict backtest, recommandation.
- **config.yaml** — sections `var`, `bootstrap`, `monitoring`, `structural_breaks` avec paramètres commentes.

### Modifications

- **Depersonnalisation** (`_orchestrateur.py`, `utils/ai_writer.py`) — suppression de toute mention academique (Master, universite, encadrant). PDF : "Note technique automatisee — Outil PEA-Brent v5".
- **Persistance EGARCH multi-lag** (`core/rapport/_stats.py`, `core/reporter.py`) — persistance = somme de tous les bêta_j (Nelson 1991 eq.10).
- **Selection scientifique GARCH** (`core/garch_selector.py`) — BIC + tests spec simultanees (LB z_t, LB z_t², Engle-Ng) + fenetre Burnham-Anderson brute (delta_IC < 2).
- **A.3bis annexe** (`_sections.py`) — 12 colonnes incluant les p-values des 3 tests de spec et le flag Spec OK.
- **Section 8.5** — VaR multi-horizons Bale III (voir Ajouts).
- **Section 9.5** — EVT-POT (voir Ajouts).

### Infrastructure

- `.gitignore` — dossier `dev/` exclu du tracking.
- Nettoyage fichiers temporaires en racine de package.
- `config.yaml` — sections `rolling_backtest`, `var` (horizons), `bootstrap`, `monitoring`, `structural_breaks` entierement parametrees et commentees.
- `main.py` — fix extraction `var_cfg` pour eviter propagation des cles inconnues a `calculer_var_tvar`.
