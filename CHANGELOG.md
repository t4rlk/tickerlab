# Journal des modifications

Toutes les modifications notables de ce projet sont consignées dans ce fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Corrigé

Cinq erreurs de calcul détectées par une nouvelle suite de validation externe
(`tests/benchmarks`). Les résultats produits par les versions antérieures sont
affectés sur les points suivants.

- **VaR et TVaR Student surestimées.** Le quantile utilisé était celui de la
  Student brute alors que les innovations GARCH sont standardisées à variance
  unitaire. Surestimation de 41 % à ν=4, 29 % à ν=5, 15 % à ν=8. Touche tout
  calcul avec `dist='t'`.

- **Test d'Acerbi-Székely inversé.** Les statistiques Z1 et Z2 valaient +2 sous
  H₀ au lieu de 0, et augmentaient lorsque le risque était sous-estimé — soit
  l'inverse de la lecture attendue. Un modèle sous-estimant le risque obtenait
  un meilleur score. Z1 est par ailleurs désormais conditionnel au nombre de
  violations observées, et indéfini en leur absence.

- **Score de Fissler-Ziegel non consistant.** La fonction n'était pas
  strictement consistante pour le couple (VaR, ES) : son minimum n'était pas
  atteint au vrai couple. Comme pour Acerbi-Székely, le classement récompensait
  la sous-estimation du risque. Tout comparatif de modèles fondé sur ce score
  est à refaire.

- **Niveaux mélangés dans le scoring FZ0.** Une VaR à 99 % était scorée contre
  un ES à 97.5 %, alors que la consistance conjointe exige un niveau unique.
  Le scoring s'effectue désormais entièrement à 97.5 %.

- **ES historique faussé en présence d'ex æquo.** Les observations situées
  exactement sur le quantile étaient comptées en entier. Sur des données
  comportant des points de masse, l'ES pouvait être très fortement sous-estimé.
  Sans effet sur des données continues.

- **Mode `integrate` sans effet sur la persistance.** Le mode annonçait
  corriger l'inflation de persistance induite par les ruptures de régime, sans
  la corriger : les dummies passaient par l'équation de MOYENNE — arch 8.0.0 ne
  supporte pas GARCHX — et captaient des sauts de niveau de rendement, non des
  changements de variance. Mesure : persistance avant = persistance après.
  Remplacé par une estimation à omega par régime, alpha et beta communs
  (Hillebrand 2005). Effet mesuré : persistance de 0.9809 à 0.9426 sur AAPL,
  de 0.9969 à 0.9611 sur ^GSPC.

- **Sélection des ruptures biaisée vers le début de série.** Le plafonnement
  retenait les ruptures les plus précoces, produisant des régimes de 8 à 36
  observations et une estimation dégénérée sur les trois séries testées.
  Remplacé par une sélection à espacement minimal.

- **Component GARCH : solution dégénérée présentée comme valide.** Une solution
  avec beta sur sa borne basse — composante transitoire absente, le modèle
  dégénérant en GARCH simple — était rapportée comme une décomposition
  permanente/transitoire valide, avec `constraints_ok['separation'] = True` :
  l'inégalité alpha+beta < rho était obtenue par effondrement, non par
  séparation. Constaté sur BTC-USD.

- **Alerte Lamoureux-Lastrapes trop restrictive.** Le critère exigeait au moins
  3 ruptures : une rupture de variance franche accompagnée d'une persistance
  supérieure au seuil ne la déclenchait pas. Critère rendu disjonctif — une
  rupture suffit, le seuil en nombre commandant désormais une alerte renforcée.

### Ajouté

- Suite de validation externe (66 tests) confrontant tickerlab à des
  estimations publiées, des vérités analytiques et le comportement statistique
  attendu des backtests. L'estimation GARCH reproduit le benchmark FCP de
  Fiorentini, Calzolari et Panattoni (1996) sur les rendements DEM/GBP de
  Bollerslev et Ghysels.
- Fonctions atomiques `var_normale`, `var_student` et `var_historique`.
- Paramètres `backcast` et `rescale` sur `estimer_final`.

### Modifié

- Les fonctions de calcul ne pré-arrondissent plus leurs valeurs de retour.
- `fissler_ziegel_loss` renvoie un `ndarray` plutôt qu'un `pd.Series`.
- Le niveau d'évaluation du score FZ est affiché dans l'interface.
- `config.yaml` : `max_dummies` devient `max_regimes`, avec changement de
  sémantique — la clé compte désormais des régimes et non des ruptures.
  L'ancien nom reste lu comme alias déprécié, converti en `max_dummies + 1`
  régimes avec un avertissement. Nouvelle clé `min_obs_regime`.
- Le cache `component_garch` est invalidé (version de schéma incrémentée) : les
  résultats antérieurs ne portent pas les indicateurs de dégénérescence et
  produiraient un rapport sans alerte.
