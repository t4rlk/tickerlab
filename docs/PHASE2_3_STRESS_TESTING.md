# Phase 2.3 — Stress testing scénario-based : résultats

**Date** : 11 juin 2026 (corrigé Phase 2.3-fix)  
**Modules** : `core/stress_scenarios.py`, `core/reverse_stress.py`  
**Tickers validés** : BZ=F daily, ^GSPC daily, EURUSD=X daily  
**Modèles** : résultats Phase 2.1 (cache `dev/resultats/multi/`)

**Corrections Phase 2.3-fix (vs doc initial)** :
1. t standardisée (variance=1) au lieu de scipy.t direct — probabilités ÷ ~2 à 10×
2. Probabilité directionnelle : queue supérieure pour gains, inférieure pour pertes
3. σ_H via récurrence GARCH exacte (paramètre `garch_final` dans `appliquer_scenario`)
4. Floor BTC-001 : VaR GARCH < −99.9% → −99.9% avec `UserWarning`

---

## 1. Contexte méthodologique

### 1.1 Paramètres des modèles à la date de référence (juin 2026)

| Ticker | Modèle | σ_T (daily) | dist | ν | Persistance | VaR99 H=1 |
|---|---|---|---|---|---|---|
| BZ=F | EGARCH(1,1,1)[t] | 1.73% | t | 5.18 | 0.979 | −5.00% |
| ^GSPC | GARCH(1,0,1)[t] | 0.60% | t | 5.42 | 0.995 | −1.73% |
| EURUSD=X | GARCH(1,0,1)[t] | 0.31% | t | 7.08 | 0.995 | −0.77% |

**Note σ_H** : le tableau ci-dessous utilise `methode_sigma_H='sqrt_H'` (scaling par √H).
Quand `garch_final` est fourni : pour EGARCH (BZ=F) → fallback `sqrt_H_EGARCH` ;
pour GARCH/GJR-GARCH (^GSPC, EURUSD=X near-IGARCH) → `igarch_exact` diverge davantage.
L'impact sur les distances est < 5% à H=5 et jusqu'à 15% à H=22.

**Note sur ^GSPC daily** : ce ticker est en Catégorie 1 (spec_gravite='majeure') — voir `docs/CAS_PATHOLOGIQUES.md`. Les résultats de stress sont fournis à titre indicatif uniquement.

**Note sur la t standardisée** : le modèle ARCH utilise une t avec variance=1 (≠ scipy.t avec variance ν/(ν−2)). La conversion : P(Z_std ≤ z) = scipy.t.cdf(z × √(ν/(ν−2)), ν). Pour ν=5.18, √(5.18/3.18) = 1.276 — le z_scipy est 27.6% plus extrême que le z brut.

---

## 2. Application des 4 scénarios

### 2.1 Tableau de pertes attendues (position longue)

| Ticker | Scénario | Choc | H (j) | σ_H | Distance Maha. | P(choc) | × VaR99 | Sens |
|---|---|---|---|---|---|---|---|---|
| BZ=F | oil_shock_2022 | +30% | 5 | 3.87% | 7.74σ | **0.0074%** | 6.00× | **gain** |
| BZ=F | covid_march_2020 | −65% | 22 | 8.13% | 8.00σ | 0.0063% | 13.0× | perte |
| ^GSPC | oil_shock_2022 | −5% | 5 | 1.35% | 3.70σ | 0.226% | 2.89× | perte |
| ^GSPC | covid_march_2020 | −34% | 22 | 2.83% | 11.99σ | 0.0006% | 19.7× | perte |
| ^GSPC | fed_hike_2022 | −8% | 1 | 0.60% | 13.24σ | 0.00036% | 4.63× | perte |
| ^GSPC | geopolitique_taiwan | −15% | 5 | 1.35% | 11.10σ | 0.00093% | 8.68× | perte |
| EURUSD=X | fed_hike_2022 | −3% | 1 | 0.31% | 9.68σ | 0.00041% | 3.91× | perte |

*P(choc) = probabilité de queue directionnelle sous t standardisée(ν) :*
*`proba_queue = 1 − CDF_std(z)` pour un gain, `CDF_std(z)` pour une perte.*
*× VaR99 = |choc| / |VaR99 H=1 GARCH| (combien de fois pire que la VaR quotidienne).*

### 2.2 Interprétation par scénario

**oil_shock_2022 (Brent +30% en 1 semaine, Ukraine 2022)** :
- Pour une position **longue** BZ=F : ce scénario est un **gain** de 30%, soit 6× la VaR99 daily. Sous le modèle EGARCH[t_std] calibré juin 2026 (σ_T=1.73%), ce choc positif de +30% à l'horizon 5j est dans la queue supérieure extrême — P(r ≥ +30% sur 5j) = **0.0074%**.
- Correction vs doc initial : P était affichée à 99.98% (bug — queue inférieure d'un choc positif). Valeur correcte 0.0074% (queue supérieure, ~13 500× plus rare).
- Pour le S&P500 : la composante −5% est un événement de probabilité 0.23% (d=3.7σ) — rare mais non impossible à l'horizon 5 jours.
- **Conclusion** : scénario asymétrique — bénéfique pour long oil, pénalisant pour equity.

**covid_march_2020 (Krach COVID : −65% oil, −34% equity en 22 jours)** :
- BZ=F −65% en 22 jours : distance 8.0σ, P=0.0063%. Événement de probabilité infime sous le régime de volatilité juin 2026. **Multiple VaR99 = 13.0×** — ce scénario est 13 fois plus sévère que la VaR quotidienne.
- ^GSPC −34% en 22 jours : distance 12.0σ, P=0.0006%. Extrêmement rare sous le régime actuel calme (σ_T=0.60%). Multiple VaR99 = 19.7×.
- **Conclusion** : scénario le plus sévère du catalogue, notamment pour ^GSPC.

**fed_hike_2022 (Fed +75bp surprise : S&P −8%, EUR/USD −3% en 1 jour)** :
- ^GSPC −8% en H=1 : distance 13.24σ, P=0.00036%. Un recul de 8% en une séance est extrême (z_scipy = 13.24 × 1.258 = 16.65 sous scipy.t(5.42)).
- EURUSD=X −3% en H=1 : distance 9.68σ, P=0.00041%. Un recul EUR/USD de 3% en une journée est également extrême.
- **Note** : les distances élevées reflètent le régime de faible volatilité de juin 2026. Sous un régime de stress (σ_T ×2), les distances seraient divisées par 2 et les probabilités multipliées par ~10×.

**geopolitique_taiwan (Hypothétique : ^GSPC −15%, GC=F +12% en 5 jours)** :
- ^GSPC −15% en H=5 : distance 11.10σ, P=0.00093%. Calibré sur Crimée 2014 ×2.
- Note : ce scénario est hypothétique, sans fenêtre historique de référence. Distance et probabilité purement indicatives.

---

## 3. Reverse stress testing

Pour chaque ticker : quel choc (sur H=1 jour) atteint une perte cible donnée ?
Les probabilités utilisent la t standardisée corrigée.

| Ticker | Perte cible | Distance Maha. | P(choc) | Période de retour |
|---|---|---|---|---|
| BZ=F | −5% | 2.9σ | 0.668% | ~150 jours ouvrés |
| BZ=F | −10% | 5.8σ | 0.031% | ~3 222 jours (~13 ans) |
| BZ=F | −20% | 11.5σ | 0.0010% | ~100 440 jours (~400 ans) |
| BZ=F | −30% | 17.3σ | 0.00013% | ~797 000 jours (~3 170 ans) |
| ^GSPC | −5% | 8.3σ | 0.0043% | ~23 021 jours (~92 ans) |
| ^GSPC | −10% | 16.5σ | 0.00011% | ~905 000 jours (~3 600 ans) |
| ^GSPC | −20% | 33.1σ | ~3×10⁻⁶% | ~38 millions de jours |
| ^GSPC | −30% | 49.6σ | ~0 | ~340 millions de jours |
| EURUSD=X | −5% | 16.1σ | 1.2×10⁻⁷ | ~8.3 millions de jours |
| EURUSD=X | −10% | 32.3σ | ~9.3×10⁻¹⁰ | ~10⁹ jours |

**Interprétations** :

- **BZ=F** : une perte daily de −5% survient tous les ~150 jours ouvrés (~7 mois) sous le modèle t_std calibré. Cohérent avec la réalité historique (l'huile a des journées de −5% régulièrement). L'écart vs le doc initial (61j → 150j) reflète la correction t standardisée : les événements sont plus rares sous t_std que sous scipy.t direct.

- **^GSPC** : une perte daily de −5% est un événement 1-en-92-ans sous σ_T=0.60%. Cela traduit le régime de faible volatilité juin 2026. En période de crise (σ_T×3), la distance serait divisée par 3 et la période de retour passerait à ~12 ans.

- **EURUSD=X** : une perte daily de −5% est quasiment impossible (P≈10⁻⁷). Cohérent avec la réalité des FX majeurs (max historique daily ≈ 3−4%).

**Limite méthodologique** : les périodes de retour supposent une distribution stationnaire. Le modèle near-IGARCH (^GSPC, EURUSD=X) n'est **pas** stationnaire à long terme. Ces chiffres sont des indicateurs de sévérité relative, pas des prédictions de fréquence absolue.

---

## 4. Scénarios non couverts — actifs exclus

Conformément au catalogue Phase 2.2, les actifs suivants sont **exclus** du stress GARCH :

| Ticker | Raison | Alternative |
|---|---|---|
| BTC-USD | Cat. 1 — VaR GARCH absurde (floored BTC-001), spec majeure | FHS H=1 uniquement |
| GC=F weekly | Cat. 2 — sig_vol=0, fallback IC_seul | VaR historique |
| EURUSD=X weekly | Cat. 2 — sig_vol=0, fallback IC_seul | FHS H=1 |

---

## 5. Notes pour Phase 4 (extension multi-actifs)

Pour Phase 4 (DCC-GARCH), le reverse stress testing deviendra :
- Minimiser la distance de Mahalanobis multivariée `d = sqrt(s' Σ⁻¹ s)` (Studer 1997)
- Sous la contrainte `w' s = perte_portfolio` (perte cible sur le portefeuille)
- Solution : `s* = Σ w (w' Σ w)⁻¹ × perte_portfolio` (allocation proportionnelle à la covariance)

Ce cadre s'appliquera naturellement une fois le DCC-GARCH estimé sur les corrélations conditionnelles entre BZ=F, ^GSPC et GC=F.

---

*Modules : `core/stress_scenarios.py`, `core/reverse_stress.py`*  
*Tests : `tests/test_phase23.py` (9/9 pass)*  
*Données : Phase 2.1 — cache `dev/resultats/multi/`, juin 2026*  
*Correctifs : Phase 2.3-fix — t standardisée + proba queue directionnelle + σ_H exact + BTC-001*
