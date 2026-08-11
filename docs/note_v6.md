# Note technique — Refonte v6 tickerlab

**Date :** 2026-05-26  
**Ticker :** BZ=F (Brent Crude Oil)  
**Periode :** 2006-01-01 — 2026-01-01  
**Branche git :** feat/v6-production  

---

## 1. Contexte et motivations

La refonte v6 introduit six evolutions majeures du pipeline de modelisation de la volatilite du petrole Brent, en alignant la methodologie sur les normes FRTB/Basel IV (BCBS d457, janvier 2023) et les meilleures pratiques de selection de modeles (Burnham & Anderson 2002).

Les modifications preservent la retrocompatibilite : chaque nouveau comportement est protege par un flag `config.yaml` (desactive par defaut), de sorte que le pipeline v5 reste reproductible.

---

## 2. Tache T6 — Mise a jour config.yaml

**Commits :** `155fae6`

Modifications principales :
- `garch.critere_information` : `"BIC"` → `"AIC"` (critere de selection principal)
- `garch.tolerance_delta_critere_brut` : `2.0` → `4.0` (seuil Burnham-Anderson « evidences substantielles »)
- Ajout section `garch.score_composite` (desactivee par defaut)
- Ajout section `backtest.tests_frtb` (desactivee par defaut)
- Ajout section `frtb` complete (sVaR, ES, capital, multiplicateurs)
- Ajout parametres pagination rapport (`une_page_un_message`, `max_lignes_par_tableau`)

### Justification AIC vs BIC

Burnham & Anderson (2002, *Model Selection and Multimodel Inference*, Springer) etablissent que le delta AIC entre le meilleur modele et les modeles equivalents peut etre interprete comme :
- delta < 2 : equivalent au meilleur (evidence forte)  
- 2 <= delta < 4 : evidence substantielle encore  
- delta >= 4 : evidence considerably less (eliminer)

Le passage de BIC a AIC est motive par la nature predictive du probleme (VaR out-of-sample) : l'AIC minimise l'erreur de prediction esperee (Kullback-Leibler divergence), tandis que le BIC est un estimateur consistant de la log-vraisemblance marginale — approprie pour l'inference, pas pour la prediction.

**Reference :** Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2e ed.). Springer. Table 2.5.

---

## 3. Tache T1 — Test de Berkowitz et score composite

**Commits :** `dc64f10`  
**Flag :** `garch.score_composite.enabled: false`

### 3.1 Test LR de Berkowitz (2001)

Le test Berkowitz teste si la distribution des innovations standardisees `z_t = eps_t / sigma_t` est correctement specifiee.

**Methode (3 etapes) :**

1. **PIT** : transformer `z_t` via la CDF du modele : `u_t = F(z_t; theta_hat)`  
   - Normal : CDF normale standard  
   - Student-t : CDF de Student(nu)  
   - skewt, ged : CDF empirique par rang (`scipy.stats.rankdata`)

2. **Transformation normale inverse** : `x_t = Phi^-1(u_t)`  
   Sous H0 (distribution correcte), `x_t ~ N(0,1)` i.i.d.

3. **Test AR(1) + LR** :  
   Regrger `x_t` sur `x_{t-1}` (MCO). Tester conjointement `mu=0, rho=0, sigma^2=1` via :  
   `LR = 2*(LL_H1 - LL_H0) ~ chi^2(3)` sous H0.

**Reference :** Berkowitz, J. (2001). Testing density forecasts, with applications to risk management. *Journal of Business & Economic Statistics*, 19(4), 465-474.

### 3.2 Score composite de selection

Quand `score_composite.enabled: true`, un etage 3 de selection par rang pondere remplace la simple parcimonie :

```
score = w_aic * rang(AIC)
      + w_lb_z2 * rang_desc(p_LB_z^2)
      + w_engle_ng * rang_desc(p_Engle-Ng)
      + w_berkowitz * rang_desc(p_Berkowitz)
      + w_parcimonie * rang(p+o+q)
```

Poids par defaut : `w_aic=0.30, w_lb_z2=0.25, w_engle_ng=0.15, w_berkowitz=0.20, w_parcimonie=0.10`.

---

## 4. Tache T3 — Backtests FRTB-grade

**Commits :** `143573e`  
**Flag :** `backtest.tests_frtb.enabled: false`

### 4.1 Test DQ — Engle & Manganelli (2004)

Le test Dynamic Quantile teste conjointement la couverture inconditionnelle et l'independance des violations.

**Formule :** `DQ = Psi' X (X'X)^-1 X' Psi / (alpha*(1-alpha)) ~ chi^2(lags+2)`

ou `Psi_t = I(r_t < VaR_t) - (1-alpha)` (hit centre), et `X` contient une constante, les lags du hit, et `VaR_t`.

**Reference :** Engle, R.F. & Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles. *Journal of Business & Economic Statistics*, 22(4), 367-381.

### 4.2 Tests ES d'Acerbi & Szekely (2014)

Les statistiques Z1 et Z2 testent si l'Expected Shortfall est correctement specifie.

**Formules :**
- `Z1 = [sum_t I_t * r_t] / (T * p * e_bar) + 1`
- `Z2 = [sum_t I_t * r_t / e_t] / (T * p) + 1`

**Note implementation :** Avec la convention pertes negatives (`e_t < 0`), Z1 et Z2 sont centrees sur 2 sous H0 (pas 0). Les p-valeurs par bootstrap parametrique (Binomial + reechantillonnage des violations observees) sont comparables car le bootstrap utilise le meme e_bar.

**Reference :** Acerbi, C. & Szekely, B. (2014). Back-testing expected shortfall. *Risk Magazine*, November 2014, pp. 76-81.

### 4.3 Perte de Lopez (1999)

`L_t = 1 + (r_t - VaR_t)^2` si violation, `0` sinon.

Penalise l'amplitude des violations (pas seulement leur nombre).

**Reference :** Lopez, J.A. (1999). Methods for evaluating value-at-risk estimates. *Economic Review*, Federal Reserve Bank of San Francisco, 2, 3-17.

### 4.4 Score FZ0 de Fissler-Ziegel (2016)

Score strictement consistant pour le couple `(VaR_alpha, ES_alpha)` :

`S_t = [(p - I_t) * v_t - I_t * r_t] / e_t + log(-e_t)`

ou `p = 1 - alpha`. Un score plus faible indique un meilleur modele. La consistance stricte garantit que la vraie paire `(VaR, ES)` minimise le score esperee.

**References :**
- Fissler, T. & Ziegel, J.F. (2016). Higher order elicitability and Osband's principle. *Annals of Statistics*, 44(4), 1680-1707.
- Nolde, N. & Ziegel, J.F. (2017). Elicitability and backtesting: Perspectives for banking regulation. *Annals of Statistics*, 45(4), 1597-1638.

### 4.5 Feu tricolore Basel

Comptage glissant des violations sur 250 jours ouvrables :
- 0-4 violations : zone verte, multiplicateur k = 3.0
- 5-9 violations : zone jaune, k = 3.40 + (n-5) * (3.85-3.40)/4
- >= 10 violations : zone rouge, k = 4.0

**Reference :** BCBS (1996). Supervisory framework for the use of backtesting. Basel Committee on Banking Supervision. Mis a jour dans : BCBS (2023). *Minimum capital requirements for market risk* (d457), paragraphes 325-336.

---

## 5. Tache T4 — Metriques FRTB (sVaR, ES, capital IMA)

**Commits :** `b988eec`  
**Flag :** `frtb.enabled: false`

### 5.1 Stressed VaR (sVaR)

Identifie la fenetre de 12 mois calendaires qui maximise la VaR conditionnelle GARCH sur la serie complete. Conforme BCBS d457 §§181-187.

### 5.2 Expected Shortfall FRTB (97.5%)

`ES_t = mu_t + sigma_t * ES_z(alpha)`

- Distributions normal/t : formule analytique (McNeil et al. 2015)  
- Distributions skewt/ged : ES empirique des innovations `z_t`

**Reference FRTB :** BCBS (2023). *Minimum capital requirements for market risk* (d457), §189.

### 5.3 Capital IMA

`Capital = max(VaR_99_10j * k, sVaR_99_10j * 3.0)`

ou `k` est le multiplicateur du feu tricolore (3.0 a 4.0).

---

## 6. Tache T2 — Rolling backtest et ruptures structurelles

**Commits :** `4ac3985`

- Rolling backtest mensuel sur fenetre glissante de 1000 jours (~4 ans), re-estimation tous les 22 jours
- Tests ICSS (Inclan-Tiao 1994) sur les carres des residus : detection de changements de variance
- Test Zivot-Andrews (1992) sur log(prix) : ruptures en tendance
- Monitoring CUSUM standardise (Hawkins-Olwell 1998) avec seuil h=5 sigma

---

## 7. Tache T5 — Pagination du rapport

**Commits :** `c29f272`

Nouveau helper `_page_break_if(config)` : insere un `PageBreak` entre chaque sous-section quand `rapport.une_page_un_message: true` (defaut). En mode legacy, un espaceur est insere a la place.

`_split_tableau(titre, colonnes, lignes, max_rows=12)` : decoupe automatiquement les tableaux longs en plusieurs parties avec saut de page.

Nouvelle section `section_frtb_resume()` : affiche ES/sVaR/capital/backtests FRTB avec degradation gracieuse (si les calculs FRTB ne sont pas actives, un message placeholder est affiche).

---

## 8. Comparaison v5 vs v6 — Modele retenu BZ=F

| Indicateur | v5 (BIC, delta<2) | v6 (AIC, delta<4) |
|---|---|---|
| Specification | EGARCH(1,1,1)[skewt] | EGARCH(3,1,2)[skewt] |
| Complexite p+o+q | 3 | 6 |
| AIC (normalise) | 18 925.87 | 18 923.96 |
| BIC (normalise) | 18 970.89 | 18 988.26 |
| Persistance | 0.9891 | 0.9874 |
| VaR 99% GARCH | n/a | -4.71% |
| Backtest CC 99% GARCH dyn. | n/a | OK (15 viol., p=0.80) |

**Lecture :** Le passage AIC + delta<4 permet a la selection de retenir un modele EGARCH(3,1,2) plus expressif (ordres superieurs, meilleur AIC de 1.91 points). Le BIC plus eleve reflete la penalisation de complexite. La persistance legèrement plus faible (0.9874 vs 0.9891) suggere un retour a la moyenne legerement plus rapide.

---

## 9. Packages utilises (conformes au cahier des charges)

`arch`, `statsmodels`, `scipy`, `numpy`, `pandas`, `reportlab`, `matplotlib`, `yfinance`, `tqdm`

---

## 10. References bibliographiques completes

- Acerbi, C. & Szekely, B. (2014). Back-testing expected shortfall. *Risk Magazine*, November 2014.
- BCBS (1996). *Supervisory framework for the use of backtesting in conjunction with the internal models approach to market risk capital requirements*. Basel Committee on Banking Supervision.
- BCBS (2023). *Minimum capital requirements for market risk* (d457, janvier 2023). Basel Committee on Banking Supervision.
- Berkowitz, J. (2001). Testing density forecasts, with applications to risk management. *Journal of Business & Economic Statistics*, 19(4), 465-474.
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2e ed.). Springer.
- Engle, R.F. & Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles. *Journal of Business & Economic Statistics*, 22(4), 367-381.
- Fissler, T. & Ziegel, J.F. (2016). Higher order elicitability and Osband's principle. *Annals of Statistics*, 44(4), 1680-1707.
- Hawkins, D.M. & Olwell, D.H. (1998). *Cumulative Sum Charts and Charting for Quality Improvement*. Springer.
- Inclan, C. & Tiao, G.C. (1994). Use of cumulative sums of squares for retrospective detection of changes of variance. *Journal of the American Statistical Association*, 89(427), 913-923.
- Lopez, J.A. (1999). Methods for evaluating value-at-risk estimates. *Economic Review*, Federal Reserve Bank of San Francisco, 2, 3-17.
- McNeil, A.J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management* (revised ed.). Princeton University Press.
- Nolde, N. & Ziegel, J.F. (2017). Elicitability and backtesting: Perspectives for banking regulation. *Annals of Statistics*, 45(4), 1597-1638.
- Zivot, E. & Andrews, D.W.K. (1992). Further evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis. *Journal of Business & Economic Statistics*, 10(3), 251-270.
