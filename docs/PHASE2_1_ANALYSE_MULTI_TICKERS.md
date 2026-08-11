# Phase 2.1 — Analyse multi-tickers : résultats et diagnostics

**Date d'exécution** : 11 juin 2026  
**Runner** : `scripts/run_multi_ticker.py --workers 2` (mode `metrics-only`)  
**Grid GARCH** : `['GARCH', 'GJR-GARCH', 'EGARCH'] × ['normal', 't']`, p,q ≤ 1 (réduit)  
**Données** : 2015-01-01 à aujourd'hui, via yfinance  
**Fichier CSV** : `dev/multi_ticker_phase21_full.csv`

---

## 1. Résumé exécutif

| Métrique | Valeur |
|---|---|
| Runs totaux | 12 (6 tickers × 2 fréquences) |
| Succès (aucun crash) | **12/12** |
| near_IGARCH (pers ≥ 0.98) | 5/12 |
| spec_gravite = majeure | 2/12 |
| all_candidates_failed_spec | 2/12 |
| fallback IC_seul_AUCUN_VALIDE | 2/12 |
| DM-GK ECHEC (math error) | 1/12 |
| **Bug VaR critique** | **1/12 (BTC-USD weekly)** |

**Critères de succès atteints** :
- ✅ 12/12 OK (aucun crash global, chaque ticker fail isolé)
- ✅ CSV exploitable pandas, 33 colonnes, types corrects
- ✅ Aucun crash inter-run (ProcessPoolExecutor stable)

**Critère non atteint** :
- ⚠ VaR GARCH numériquement aberrante sur BTC-USD weekly (voir section 4.1)

---

## 2. Tableau synthèse complet

| Ticker | Freq | n_obs | Modèle retenu | Motif | Pers. | near_IGARCH | Spec | Kupiec 95% | Kupiec 99% | Durée |
|---|---|---|---|---|---|---|---|---|---|---|
| BZ=F | weekly | 574 | EGARCH(1,1,1)[t] | spec_OK+AIC | 0.901 | ✗ | aucune | 0.646 ✓ | 0.380 ✓ | 29s |
| BZ=F | daily | 2766 | EGARCH(1,1,1)[t] | spec_OK+AIC | 0.979 | ✗ | aucune | 0.000 ✗ | 0.095 ✓ | 70s |
| CL=F | weekly | 574 | EGARCH(1,1,1)[t] | spec_OK+AIC | 0.920 | ✗ | aucune | 0.168 ✓ | 0.545 ✓ | 24s |
| CL=F | daily | 2763 | GJR-GARCH(1,1,1)[t] | spec_OK+AIC | 0.968 | ✗ | aucune | 0.001 ✗ | 0.034 ⚠ | 76s |
| ^GSPC | weekly | 574 | EGARCH(1,1,1)[t] | spec_OK+AIC | 0.886 | ✗ | aucune | 0.168 ✓ | 0.545 ✓ | 25s |
| ^GSPC | daily | 2765 | GARCH(1,0,1)[t] | AIC_spec_echoue | 0.995 | ✓ | **majeure** | 0.008 ✗ | 0.399 ✓ | **270s** |
| GC=F | weekly | 574 | EGARCH(1,1,1)[t] | IC_seul_AUCUN | 0.977 | ✗ | aucune | 0.023 ✗ | 1.000 ✓ | 25s |
| GC=F | daily | 2764 | GJR-GARCH(1,1,1)[t] | spec_OK+AIC | 0.987 | ✓ | aucune | 0.001 ✗ | 0.008 ✗ | 108s |
| EURUSD=X | weekly | 574 | GJR-GARCH(1,1,1)[t] | IC_seul_AUCUN | 0.976 | ✗ | aucune | 0.023 ✗ | 0.840 ✓ | 107s |
| EURUSD=X | daily | 2862 | GARCH(1,0,1)[t] | spec_OK+AIC | 0.995 | ✓ | aucune | 0.001 ✗ | 0.182 ✓ | 173s |
| BTC-USD | weekly | 574 | EGARCH(1,1,1)[t] | spec_OK+AIC | 0.998 | ✓ | aucune | 0.329 ✓ | 0.011 ⚠ | 48s |
| BTC-USD | daily | 4017 | EGARCH(1,1,1)[t] | AIC_spec_echoue | 0.988 | ✓ | **majeure** | 0.000 ✗ | 0.000 ✗ | 175s |

Légende : ✓ p > 0.05, ⚠ 0.01 < p ≤ 0.05, ✗ p ≤ 0.01

---

## 3. Analyse par catégorie de comportement

### 3.1 — Catégorie A : modélisables sans réserve

Tickers où le pipeline GARCH standard fonctionne correctement (spec_OK, VaR99 validée) :

**BZ=F weekly** et **CL=F weekly** et **^GSPC weekly** :
- EGARCH(1,1,1)[t] sélectionné systématiquement sur les 3 tickers weekly en série temporelle saine
- Persistance modérée (0.886–0.920) : volatilité mean-reverting
- Kupiec 95% et 99% validés
- FHS confirme la VaR GARCH (écarts < 20%)
- **Conclusion** : ces 3 tickers weekly sont les actifs de référence du pipeline. Résultats réutilisables pour Phase 2.3 (stress testing).

### 3.2 — Catégorie B : VaR 95% sous-couverte, VaR 99% valide

**BZ=F daily**, **CL=F daily**, **EURUSD=X daily** :
- Kupiec 95% échoue (p < 0.05) mais Kupiec 99% passe
- Lecture : le modèle correctement calibré en queue épaisse (1%) est trop lâche en queue "normale" (5%)
- Pattern typique pour la distribution t de Student sur données daily à forte persistance (Christoffersen & Pelletier 2004)
- **Conclusion pour usage opérationnel** : utiliser systématiquement VaR99 comme seuil de risque, pas VaR95, sur ces actifs.

### 3.3 — Catégorie C : near-IGARCH + VaR sous-couverte

**GC=F daily** (pers=0.987) et **^GSPC daily** (pers=0.995) :
- near_igarch=True : la persistance quasi-unitaire indique un choc permanent de volatilité — le modèle ne "revient" jamais à la moyenne
- GC=F daily : Kupiec 95% et 99% tous les deux FAIL — la VaR GARCH sous-estime systématiquement le risque
- ^GSPC daily : spec_gravite='majeure' (0/2 candidats passent les tests LB+Engle-Ng), forced fallback AIC_seul ; Kupiec 95% FAIL
- **Cause probable** : sur daily avec haute fréquence d'observations, l'Or (GC=F) et le S&P500 montrent des effets de longue mémoire que GARCH(1,1) ne capture pas intégralement (Ding-Granger-Engle 1993, FIGARCH)
- **Conclusion** : résultats VaR à interpréter avec précaution. Documenter dans Catalogue (Phase 2.2).

### 3.4 — Catégorie D : fallback IC_seul_AUCUN_VALIDE

**GC=F weekly** et **EURUSD=X weekly** :
- 0 modèle passe le filtre sig_vol=True ET stationnarité
- Pipeline déclenche fallback sur meilleur AIC global sans contrainte de spécification
- GC=F weekly : VaR99 Kupiec = 1.000 → 0 violations observées sur 173 OOS, modèle **surconservateur** (over-coverage)
- EURUSD=X weekly : même pathologie que Phase 1 (déjà documentée), VaR99 Kupiec OK (0.84) par chance
- **Cause probable pour GC=F weekly** : sur données weekly avec seulement ~574 obs, les rendements de l'Or ont une variance conditionnelle quasi-nulle (faible hétéroscédasticité sur l'horizon weekly), donc aucun modèle GARCH ne détecte d'effet ARCH significatif
- **Conclusion** : L'or hebdomadaire est intrinsèquement difficile à modéliser par GARCH. Recommandation : utiliser la VaR historique ou FHS comme seule référence sur GC=F weekly.

### 3.5 — Catégorie E : BTC-USD — asset classe hors normes

**BTC-USD weekly** :
- Spec OK, EGARCH(1,1,1)[t] sélectionné avec pers=0.998 (quasi near_IGARCH)
- **Bug VaR critique** : VaR95_GARCH = -200%, VaR99_GARCH = -356% (voir section 4.1)
- DM-GK : `math range error` (lié au VaR aberrant)
- Kupiec 99% borderline (p=0.011)

**BTC-USD daily** :
- spec_gravite='majeure', all_candidates_failed_spec=True : autocorrélation résiduelle persistante (LB p=0.002)
- VaR95 et VaR99 Kupiec toutes deux FAIL avec p=0.000 — sous-couverture massive
- near_igarch=True (pers=0.988)
- **Conclusion** : BTC-USD est non modélisable par GARCH standard sur les deux fréquences. À inscrire en Catégorie 1 du Catalogue (Phase 2.2). Modèles alternatifs : GARCH-Jump (Maheu-McCurdy 2004), GARCH avec distribution α-stable, ou pure approche FHS.

---

## 4. Bugs et anomalies techniques découverts

### 4.1 — BUG CRITIQUE : VaR GARCH absurde sur BTC-USD weekly

**Symptôme** :
```
var95_garch = -200.08%
var99_garch = -356.89%
var95_fhs   =   -5.54%   (valeur de référence réaliste)
var99_fhs   =   -9.13%
```

**Cause** : VaR GARCH = −(σ_t × q_{ν}(α)) où σ_t est la volatilité conditionnelle estimée. Avec pers=0.9979 et des chocs récents importants sur BTC weekly, σ_t atteint ~80-100% hebdomadaire. Le quantile de la Student(ν=3.35) à 1% est ≈ 3.6, ce qui produit VaR99 ≈ -360%. La formule est **mathématiquement correcte** mais physiquement absurde : un actif ne peut pas perdre plus de 100% de sa valeur.

**Conséquences en cascade** :
1. DM-GK : `math range error` lors du calcul de l'exponentielle de la statistique de test → NaN dans le CSV
2. Le signal near_IGARCH (pers=0.998) aurait dû alerter avant le calcul de VaR

**Correctif recommandé** (Phase 2.2 ou 3.1) :
```python
# Dans calculer_var_tvar() ou _extraire_metriques() :
VaR_garch = max(VaR_garch, -0.999)   # floor à -99.9% (rendement log-return)
```
Ou, plus proprement : détecter σ_t > 50% (hebdomadaire) comme signal d'anomalie et substituer automatiquement la VaR FHS.

**Ticket dette technique** : `DETTE_TECHNIQUE.md` — item BTC-001.

### 4.2 — Anomalie de performance : ^GSPC daily 270s

**Symptôme** : ^GSPC daily prend 270.4s vs ~70-175s pour les autres tickers daily de même taille.

**Cause probable** : Component GARCH avec rho=0.9993 (quasi IGARCH) sur 2765 observations daily. L'optimiseur SLSQP iterate davantage quand les paramètres sont proches des contraintes de bord (ρ → 1). Multi-start aggrave : chaque départ converge lentement.

**Impact** : avec `--workers 2`, ce run a bloqué un worker pendant 270s. En mode séquentiel, l'impact serait multiplicatif.

**Mitigation** : ajouter un timeout par run dans `run_unique()` (ex : `signal.alarm(300)` sur Unix ou `concurrent.futures.wait(timeout=300)` sur Windows). À évaluer en Phase 3.1.

### 4.3 — DM-GK Omega mal conditionnée (avertissements répétés)

**Symptôme** : `gk_test: Omega mal conditionnee (cond=2e+16) — passage a pinv` apparaît sur la majorité des runs.

**Évaluation** : comportement déjà connu (Phase 1.5), géré par `pinv`. Pas de crash, résultats cohérents. Avertissement informatif à supprimer ou réduire en niveau WARNING dans les logs finaux (Phase 3.1).

---

## 5. Comparaison fréquences daily vs weekly

| Ticker | Modèle weekly | Kupiec 99% weekly | Modèle daily | Kupiec 99% daily | Verdict |
|---|---|---|---|---|---|
| BZ=F | EGARCH[t] | 0.380 ✓ | EGARCH[t] | 0.095 ✓ | Cohérent |
| CL=F | EGARCH[t] | 0.545 ✓ | GJR-GARCH[t] | 0.034 ⚠ | Daily plus difficile |
| ^GSPC | EGARCH[t] | 0.545 ✓ | GARCH[t] (fallback) | 0.399 ✓ | Modèle différent, résultat OK |
| GC=F | EGARCH fallback | 1.000 (surcons.) | GJR-GARCH[t] | 0.008 ✗ | Pathologie des deux côtés |
| EURUSD=X | GJR-GARCH fallback | 0.840 ✓ | GARCH[t] | 0.182 ✓ | Daily modélisable, weekly non |
| BTC-USD | EGARCH[t] (VaR bug) | 0.011 ⚠ | EGARCH fallback | 0.000 ✗ | Non modélisable |

**Observation générale** : les données **weekly** donnent systématiquement de meilleurs résultats de backtest VaR que daily, sur 5/6 tickers. Explication : agrégation temporelle réduit les effets de microstructure et produit des rendements plus proches d'une distribution t-Student standard (Drost & Nijman 1993, aggregation theorem).

---

## 6. Tableau VaR comparé GARCH vs FHS

| Ticker | Freq | VaR99 GARCH | VaR99 FHS H=1 | Ratio FHS/GARCH | DM verdict |
|---|---|---|---|---|---|
| BZ=F | weekly | -9.10% | -10.73% | 1.18 | GARCH wins |
| BZ=F | daily | -4.57% | -4.93% | 1.08 | GARCH wins |
| CL=F | weekly | -10.10% | -11.89% | 1.18 | GARCH wins |
| CL=F | daily | -4.62% | -4.96% | 1.07 | GARCH wins |
| ^GSPC | weekly | -4.29% | -4.97% | 1.16 | GARCH wins |
| ^GSPC | daily | -1.67% | -1.84% | 1.10 | GARCH wins |
| GC=F | weekly | -6.81% | -6.03% | 0.89 | GARCH wins |
| GC=F | daily | -3.32% | -3.24% | 0.98 | GARCH wins |
| EURUSD=X | weekly | -2.48% | -2.23% | 0.90 | GARCH wins |
| EURUSD=X | daily | -0.78% | -0.80% | 1.03 | GARCH wins |
| BTC-USD | weekly | **-356.89%** | -9.13% | — | DM ECHEC |
| BTC-USD | daily | -4.77% | -4.37% | 0.92 | GARCH wins |

**Observations** :
- GARCH dyn. domine FHS sur 11/11 runs valides (DM p=0.000 systématiquement) — cohérent avec la littérature (Barone-Adesi et al. 1999)
- FHS est systématiquement plus conservatrice sur les énergies (BZ=F, CL=F) : ratio 1.07-1.18
- L'Or (GC=F) : FHS ≈ GARCH (ratio 0.89-0.98) — peu d'asymétrie temporelle
- EUR/USD daily : quasi identique FHS ≈ GARCH (ratio 1.03)

---

## 7. Implications pour Phase 2.2 — Catalogue cas pathologiques

Les runs Phase 2.1 alimentent directement le catalogue. Catégories identifiées :

| Catégorie | Actifs | Symptôme | Recommandation |
|---|---|---|---|
| Cat. 1 — Non modélisables | BTC-USD (daily+weekly), ^GSPC daily | spec majeure + VaR FAIL, VaR absurde | Exclure. Modèles alternatifs GARCH-Jump, α-stable |
| Cat. 2 — Fallback sig_vol=0 | GC=F weekly, EURUSD=X weekly | Hétéroscédasticité non détectable | Utiliser VaR historique / FHS seuls |
| Cat. 3 — near_IGARCH | GC=F daily, ^GSPC daily, EURUSD=X daily, BTC-USD weekly+daily | pers ≥ 0.987 | Interpréter VaR avec réserve, signaler à l'utilisateur |
| Cat. 4 — Bug VaR | BTC-USD weekly | VaR GARCH > 100% | Floor −99.9%, ou substitution automatique FHS |

---

## 8. Implications pour Phase 2.3 — Stress testing

**Tickers utilisables pour stress testing** (spec_OK ou backtest VaR99 validé) :
- BZ=F (weekly et daily) — scénarios oil shock applicable
- CL=F weekly — scénario oil shock
- ^GSPC weekly — scénarios equity applicable
- EURUSD=X daily — scénarios Fed/BCE (avec réserve near_IGARCH)

**Tickers à exclure des stress tests GARCH** :
- BTC-USD (les deux fréquences) : VaR absurde ou spec majeure
- GC=F weekly : surconservateur (0 violations), stress test non pertinent
- ^GSPC daily : spec majeure, résultats peu fiables

**Horizon recommandé pour stress** : H=1 (VaR FHS H=1 fiable sur tous les tickers sauf BTC). H=22 disponible via FHS mais uniquement sur tickers avec `residus_iid_ok=True`.

---

## 9. Dette technique identifiée

Voir `docs/DETTE_TECHNIQUE.md` pour les entrées formelles. Résumé :

| ID | Sévérité | Description |
|---|---|---|
| BTC-001 | Critique | VaR GARCH non plafonnée à -100% → valeurs absurdes sur actifs très volatils |
| DM-001 | Mineur | DM-GK math range error quand VaR GARCH > 100% |
| PERF-001 | Mineur | Component GARCH near-IGARCH peut prendre 270s+ (pas de timeout) |
| LOG-001 | Cosmétique | Omega mal conditionnée : avertissement verbeux, à réduire en niveau DEBUG |

---

## 10. Conclusion

**Phase 2.1 validée** avec 12/12 runs sans crash global. Le pipeline est robuste à la diversité des asset classes. Les cas de dégradation gracieuse (fallback, isolement des erreurs) fonctionnent correctement.

**Principal enseignement** : la frontière entre "modélisable" et "non modélisable" par GARCH standard n'est pas une frontière d'asset class simple (ex: "crypto = mauvais") — elle dépend de la combinaison fréquence × régime de volatilité × période. BTC daily est non modélisable ; ^GSPC daily a des problèmes de spécification que ^GSPC weekly n'a pas.

**Prochaine étape** : Phase 2.2 — Catalogue cas pathologiques formalisé, basé sur ce tableau. Plus correction du bug BTC-001 (VaR floor).

---

*Généré par `scripts/run_multi_ticker.py`, données yfinance 2015-2026, grid GARCH réduit (comparabilité Phase 1 non garantie).*
