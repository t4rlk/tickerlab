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
