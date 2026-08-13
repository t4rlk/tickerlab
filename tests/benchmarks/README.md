# Suite de benchmarks — validation externe de tickerlab

Cette suite ne teste pas que le code *tourne*. Elle teste qu'il est **juste**,
en le confrontant à trois sources d'autorité extérieures :

1. des **estimations publiées** dans la littérature économétrique ;
2. des **vérités analytiques** (formes fermées, propriétés d'élicitabilité) ;
3. le **comportement statistique attendu** des tests (taille et puissance).

Les trois sont nécessaires. Une suite qui ne vérifie que la taille des tests
reste verte avec `return 1.0` en dur ; une suite qui ne vérifie que des valeurs
ponctuelles ne dit rien sur la validité du score utilisé.

---

## Démarrage

```bash
# 1. câbler tickerlab (seul fichier à éditer)
$EDITOR tests/benchmarks/adapter.py

# 2. lancer
pytest tests/benchmarks -v

# CI rapide (~30 s)          publication / vérification approfondie
BENCH_REPS=400 pytest tests/benchmarks -m "not slow"
BENCH_REPS=20000 pytest tests/benchmarks
```

Toute entrée non câblée de `adapter.py` produit un **SKIP explicite**, jamais un
PASS silencieux.

---

## Niveau 1 — Benchmark FCP (`test_fcp_garch.py`)

GARCH(1,1) gaussien sur les rendements quotidiens DEM/GBP de Bollerslev &
Ghysels (1996), 1 974 observations (02/01/1984 – 31/12/1991).

| Paramètre | Référence FCP | Tolérance |
|---|---|---|
| μ | −0.00619041 | 1e−4 absolue |
| ω | 0.0107613 | 2e−3 relative |
| α | 0.153134 | 1e−3 relative |
| β | 0.805974 | 1e−3 relative |

*Sources.* Fiorentini, Calzolari & Panattoni (1996), *Journal of Applied
Econometrics* 11, 399–417 ; benchmark proposé par McCullough & Renfro (1999)
pour noter les logiciels économétriques, repris par Brooks, Burke & Persand
(2001) et par Hill & McCullough (2019) sur les packages R (`fGarch`, `rugarch`,
`tseries`), qui n'y sont **pas** également précis.

*Données.* Miroir CRAN du package `fGarch` (`dem2gbp.csv.gz`), téléchargé une
fois puis mis en cache dans `tests/benchmarks/data/`, avec vérification
SHA-256. Série identique à `dmbp.dat` (page Econometric Benchmarks) et à
`dmbp` du package `rugarch`.

### Initialisation — le point qui fait ou défait la reproduction

Le benchmark n'est reproductible qu'à **initialisation contrôlée** ; la page
Econometric Benchmarks publie d'ailleurs un jeu de valeurs par option
d'initialisation du presample. La suite impose

```
h_0 = e_0² = variance d'échantillon des rendements
```

Avec le backcast par défaut de `arch` (moyenne exponentiellement pondérée des
carrés), ω dévie d'environ 8 % et α de 5 %. **Ce n'est pas un bug, c'est une
convention différente** — mais elle doit être explicite et testée
(`test_backcast_sensitivity_is_documented` échoue si le paramètre est ignoré).

---

## Niveau 2 — Taille et puissance (`test_backtest_size_power.py`)

Pour Kupiec, Christoffersen, DQ, Berkowitz et Acerbi–Székely, il n'existe pas
de valeur publiée universelle à reproduire. La validation correcte est
statistique :

| Test | Sous H₀ | Alternative testée | Rejet attendu |
|---|---|---|---|
| Kupiec POF | 5.0 % | violations à 10 % | > 90 % |
| Christoffersen ind. | **8.0 %** (voir ci-dessous) | violations groupées à couverture correcte | > 90 % |
| Christoffersen cc | 5.5 % | — | — |
| Engle–Manganelli DQ | 5.0 % | VaR statique sur risque variable | > 75 % |
| Berkowitz | 4.8 % | volatilité sous-estimée de 25 % | > 90 % |
| Berkowitz | — | queues épaisses à variance correcte | 15–60 % |

Trois choix méthodologiques valent d'être signalés :

* **Distorsion de taille assumée.** À n = 1000 et α = 5 %, le LR d'indépendance
  de Christoffersen rejette ≈ 8 % du temps sous H₀, pas 5 % : la statistique est
  très discrète (peu de transitions 1→1 observées) et l'approximation χ²(1)
  sur-rejette. La suite teste cette valeur mesurée, pas le niveau nominal — un
  test qui « corrigerait » ce 8 % ne serait plus celui de Christoffersen.
* **Faiblesse encodée comme connaissance.** Berkowitz n'a qu'une puissance
  d'environ un tiers contre une mauvaise forme de queue à variance correcte : le
  test borne le taux de rejet **des deux côtés**. S'il devient très puissant,
  c'est que l'implémentation ne teste plus ce qu'elle annonce.
* **Degrés de liberté vérifiés en dur.** `test_degrees_of_freedom_are_correct`
  contrôle que chaque p-value est exactement la survie d'un χ² au bon ddl,
  évalué en la statistique renvoyée. Instantané et déterministe : une erreur de
  ddl ne fait « que » ramener le taux de rejet de 5 % à 1.4 %, ce que les tests
  Monte-Carlo ne détectent que marginalement.

### Acerbi & Székely (2014)

Invariants exacts contrôlés : Z₂ = 1.0 exactement en l'absence de violation
(et 1.0 est son maximum) ; Z₁ est indéfini (nan) et surtout pas nul dans ce cas ;
les deux statistiques sont négatives en cas de sous-estimation du risque.
`test_as_z2_threshold_matches_literature` reproduit par simulation le seuil
publié : sur 250 jours à 2.5 % sous hypothèse gaussienne, le quantile à 5 % de
Z₂ vaut ≈ −0.70, valeur citée par Acerbi & Székely et reprise dans la
littérature. Le seuil de Z₁ étant plus sensible au protocole de simulation, il
n'est encadré que largement.

---

## Niveau 4 — Invariants analytiques (`test_analytical.py`)

Aucune donnée externe, aucune autre implémentation : uniquement des vérités
mathématiques.

* **Formes fermées.** VaR/ES gaussiennes en dur ; VaR/ES Student comparées à
  l'intégration numérique de `E[X | X ≤ q_α]`. Le piège visé est l'oubli du
  facteur `sqrt(ν/(ν−2))` : sans lui, la VaR est surestimée d'environ 22 % à
  ν = 5.
* **Équivariance de localisation-échelle**, convergence Student → gaussienne,
  monotonie en α, ES ≥ VaR.
* **Queues épaisses.** À variance égale, la Student doit dominer en queue
  profonde *et* rester en dessous à 5 % — un test qui ne vérifierait que le
  premier sens laisse passer une erreur d'échelle.
* **Cohérence.** Contre-exemple explicite de non-sous-additivité de la VaR (deux
  obligations à défaut indépendant, proba 4 %, niveau 95 %) avec vérification
  simultanée que l'ES, elle, reste sous-additive.
* **Élicitabilité FZ0.** Le score de Fissler–Ziegel 0-homogène est balayé sur une
  grille : son espérance doit être minimale **au couple (VaR, ES) véritable et
  nulle part ailleurs**. C'est strictement plus fort qu'une comparaison de
  formules — cela prouve que la fonction implémentée est bien un score
  strictement consistant pour la paire.
* **Récupération des paramètres GARCH** sur 50 000 points simulés à partir de
  valeurs connues : teste simultanément la vraisemblance, la récursion de
  variance et l'optimiseur.
* **Feu tricolore bâlois** : vert ≤ 4, jaune 5–9, rouge ≥ 10 exceptions sur
  250 jours à 99 %.

---

## Reproductibilité

Chaque test tire sa graine du **nom du test** (`conftest.py`), donc reste
reproductible indépendamment de l'ordre d'exécution, des filtres `-k` et de
`pytest-xdist`. Une graine de session partagée rendrait les tests Monte-Carlo
instables selon l'ordre — et une suite de benchmarks flaky ne prouve rien.

Les intervalles d'acceptation des tests de taille sont des **intervalles
binomiaux exacts à 99.9 %**, élargis d'une marge de modèle, et non des bandes
choisies à la main.

---

## Tests de mutation

La suite a été validée par injection de bugs classiques dans une implémentation
de référence. Les sept mutations sont détectées :

| Bug injecté | Tests qui tombent |
|---|---|
| Student sans restandardisation | 13 |
| VaR prise dans la queue droite | 11 |
| Kupiec en χ²(2) au lieu de χ²(1) | 2 |
| Score FZ0 au signe inversé | 2 |
| Z₂ normalisé par le nombre d'exceptions | 1 |
| Paramètre `backcast` ignoré | 5 |
| Seuil bâlois vert/jaune décalé | 1 |

À relancer après toute modification substantielle de la suite : un test qui ne
tombe sous aucune mutation ne protège rien.

---

## Références

* Acerbi, C. & Székely, B. (2014), « Backtesting Expected Shortfall », *Risk* 27, 76–81.
* Berkowitz, J. (2001), « Testing Density Forecasts, with Applications to Risk Management », *JBES* 19, 465–474.
* Bollerslev, T. & Ghysels, E. (1996), « Periodic Autoregressive Conditional Heteroscedasticity », *JBES* 14, 139–151.
* Brooks, C., Burke, S. P. & Persand, G. (2001), « Benchmarks and the Accuracy of GARCH Model Estimation », *International Journal of Forecasting* 17, 45–56.
* Christoffersen, P. (1998), « Evaluating Interval Forecasts », *International Economic Review* 39, 841–862.
* Engle, R. F. & Manganelli, S. (2004), « CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles », *JBES* 22, 367–381.
* Fiorentini, G., Calzolari, G. & Panattoni, L. (1996), « Analytic Derivatives and the Computation of GARCH Estimates », *Journal of Applied Econometrics* 11, 399–417.
* Fissler, T. & Ziegel, J. F. (2016), « Higher Order Elicitability and Osband's Principle », *Annals of Statistics* 44, 1680–1707.
* Hill, C. & McCullough, B. D. (2019), « On the Accuracy of GARCH Estimation in R Packages », *Econometric Research in Finance* 4, 133–156.
* Kupiec, P. (1995), « Techniques for Verifying the Accuracy of Risk Management Models », *Journal of Derivatives* 3, 73–84.
* McCullough, B. D. & Renfro, C. G. (1999), « Benchmarks and Software Standards: A Case Study of GARCH Procedures », *Journal of Economic and Social Measurement* 25, 59–71.
* Nolde, N. & Ziegel, J. F. (2017), « Elicitability and Backtesting: Perspectives for Banking Regulation », *Annals of Applied Statistics* 11, 1833–1874.
