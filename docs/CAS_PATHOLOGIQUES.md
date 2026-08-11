# Catalogue des cas pathologiques — Pipeline PEA-Brent v6

Asset classes où le pipeline GARCH standard échoue ou produit des résultats
non fiables. **Consulter ce document avant d'analyser un actif inconnu.**

Sources : Phase 1 (validation 3 tickers) + Phase 2.1 (12 runs, 6 tickers × 2 fréquences,
juin 2026). CSV de référence : `dev/multi_ticker_phase21_full.csv`.

---

## Catégorie 1 — Non modélisables par GARCH standard

Actifs sur lesquels le pipeline produit des résultats invalides ou trompeurs.
Ne pas utiliser les sorties VaR GARCH comme référence opérationnelle.

### BTC-USD (daily et weekly)

- **Symptômes** :
  - daily : `spec_gravite='majeure'`, `all_candidates_failed_spec=True`,
    LB(z_t) p=0.002 (autocorrélation résiduelle massive), Kupiec 95% p=0.000,
    Kupiec 99% p=0.000 — sous-couverture totale
  - weekly : VaR95_GARCH=-200%, VaR99_GARCH=-357% (physiquement impossible),
    DM-GK `math range error`, near_IGARCH (pers=0.998)
- **Diagnostic** : les queues de distribution de BTC sont trop épaisses pour une
  Student(ν) stationnaire. Combiné à near-IGARCH (σ_t n'est pas mean-reverting),
  le modèle extrapolé hors échantillon diverge numériquement.
- **Cause** : leptokurticité extrême (excès de kurtosis > 10 sur daily),
  chocs de marché discontinus (Maheu & McCurdy 2004), régimes multiples de volatilité.
  Bogue BTC-001 : VaR GARCH non bornée à −100% — voir `docs/DETTE_TECHNIQUE.md`.
- **Recommandation** : modèles GARCH-Jump (Maheu-McCurdy 2004), distribution α-stable
  (Mandelbrot 1963), ou approche non-paramétrique (FHS seul, EVT/GPD sur queues —
  McNeil & Frey 2000). La VaR FHS H=1 reste exploitable (weekly -5.5%, daily -4.4% à 99%).
- **Verdict pipeline** : ALERTE CRITIQUE. BUG BTC-001 déclenché (VaR non bornée).
  Ne jamais utiliser la VaR GARCH de BTC-USD en production.

### ^GSPC daily

- **Symptômes** : `spec_gravite='majeure'`, `all_candidates_failed_spec=True`,
  Engle-Ng p≈0.000 sur tous les candidats (asymétrie résiduelle non capturée),
  near_IGARCH (pers=0.995). Durée run anormale : 270s (Component GARCH instable).
- **Diagnostic** : le S&P500 daily présente un effet levier très prononcé (Black 1976,
  Nelson 1991) que ni GARCH symétrique ni GJR-GARCH(1,1) ne capturent sur grid p,q ≤ 1.
  EGARCH(1,1) non stationnaire (pers ≥ 1) exclu du filtre ; GJR et GARCH tous deux
  rejetés par Engle-Ng.
- **Cause** : effet levier asymétrique fort + longue mémoire (Ding-Granger-Engle 1993).
  Grid réduit Phase 2.1 trop contraint pour capturer la dynamique asymétrique complète.
- **Recommandation** : ré-estimer avec grid complet (p,q ≤ 2) et EGARCH(1,2) ou
  GJR-GARCH(2,1). Alternative : FIGARCH (Baillie-Bollerslev-Mikkelsen 1996) pour
  longue mémoire. ^GSPC **weekly** n'a pas ce problème (spec_OK, EGARCH(1,1)[t]).
- **Verdict pipeline** : résultat Kupiec 99% acceptable (p=0.399) malgré spec majeure.
  Pathologie spécifique au grid réduit Phase 2.1 — à réévaluer avec grid complet.

---

## Catégorie 2 — Hétéroscédasticité non détectable

Actifs pour lesquels aucun modèle GARCH ne passe le filtre `sig_vol=True`
(test ARCH-LM non significatif). Pipeline déclenche fallback `IC_seul_AUCUN_VALIDE` :
modèle sélectionné par AIC seul, sans garantie de validité statistique.

### GC=F weekly (Or — données hebdomadaires)

- **Symptômes** : 0/6 candidats `sig_vol=True`, `motif_selection='IC_seul_AUCUN_VALIDE'`,
  VaR99 Kupiec p=1.000 (0 violations sur 173 OOS — modèle **surconservateur**).
- **Diagnostic** : sur l'horizon hebdomadaire, les rendements de l'Or ont une variance
  conditionnelle quasi-constante. L'agrégation temporelle lisse les clusters de
  volatilité jusqu'à rendre l'effet ARCH indétectable (Drost & Nijman 1993).
- **Recommandation** : utiliser GC=F **daily** (GJR-GARCH(1,1)[t] valide, Kupiec99 p=0.008
  — borderline mais modèle statistiquement significatif). Si weekly imposé :
  VaR historique inconditionnelle ou VaR FHS H=1 (-6.03% à 99%).
- **Verdict pipeline** : signal `IC_seul_AUCUN_VALIDE` à traiter comme ALERTE MAJEURE.
  La VaR GARCH de -6.81% est surconservatrice (0 violations = sur-couverture).

### EURUSD=X weekly (EUR/USD — données hebdomadaires)

- **Symptômes** : 0/6 candidats `sig_vol=True`, `motif_selection='IC_seul_AUCUN_VALIDE'`,
  Kupiec 99% p=0.840 (acceptable par hasard, pas par construction du modèle).
- **Diagnostic** : même pathologie que GC=F weekly — volatilité EUR/USD hebdomadaire
  insuffisamment clustérisée pour un test ARCH significatif sur 574 obs (2015-2026).
  Sur daily (2862 obs), GARCH(1,0,1)[t] converge avec 2/6 sig_vol.
- **Recommandation** : utiliser EURUSD=X **daily**. Si weekly imposé : FHS H=1 seul.
  Identifié en Phase 1 (diagnostic ARIMA : série non autocorrélée sur weekly).
- **Verdict pipeline** : le résultat Kupiec acceptable est trompeur. Bonne couverture
  OOS ≠ modèle validé quand T_oos=173 (faible puissance des tests).

### Cas annexe — EUR/USD daily : Component GARCH α+β ≈ 0 (CAS-01)

- **Symptômes** : α̂+β̂ ≈ 0.005–0.008 pour EURUSD=X daily. La composante transitoire
  est quasi-nulle ; la composante permanente (ρ̂ ≈ 0.85–0.999) absorbe tout. ΔAIC = +199
  (Component GARCH inadapté).
- **Diagnostic** : la volatilité FX EUR/USD est quasi-entièrement portée par des facteurs
  structurels permanents (différentiel BCE/Fed, crises souveraines). Pas de chocs
  transitoires identifiables sur daily.
- **Implication** : Component GARCH rejeté automatiquement par le pipeline (ΔAIC > seuil).
  Mais si `force_estimation=True` : rho=0.9993 dans le CSV Phase 2.1 confirme le cas.
- **Référence** : Baillie & Bollerslev (1989) — volatilité FX dominée par longue mémoire,
  pas par chocs ARCH transitoires.

---

## Catégorie 3 — near-IGARCH : persistance quasi-unitaire

Actifs dont la persistance GARCH est ≥ 0.98. La volatilité ne revient pas à la
moyenne. Prévisions VaR à horizon > 1 peu fiables.

| Ticker | Freq | Persistance | half-life estimée | Kupiec 99% | Statut |
|---|---|---|---|---|---|
| GC=F | daily | 0.987 | ~52 jours | 0.008 ✗ | near-IGARCH + VaR FAIL |
| ^GSPC | daily | 0.995 | ~138 jours | 0.399 ✓ | near-IGARCH + spec majeure |
| EURUSD=X | daily | 0.995 | ~138 jours | 0.182 ✓ | near-IGARCH, VaR95 FAIL |
| BTC-USD | weekly | 0.998 | ~346 semaines | 0.011 ⚠ | near-IGARCH + bug BTC-001 |
| BTC-USD | daily | 0.988 | ~57 jours | 0.000 ✗ | near-IGARCH + spec majeure |

- **Diagnostic** : near-IGARCH (α+β ≈ 1) → chocs de volatilité "permanents". Sous ces
  paramètres, E[σ²_{t+H}] → ∞ quand H → ∞ (Nelson 1990). VaR H>1 via scaling √H
  ou FHS multi-horizon diverge.
- **Recommandation** : signaler `near_igarch=True` à l'utilisateur. VaR H=1 utilisable
  (dépend du Kupiec). VaR H>1 : utiliser FHS avec `residus_iid_ok=True` uniquement.
  Pour modélisation long terme : Component GARCH (Engle-Lee 1999) — estimé
  automatiquement et disponible dans le CSV (`component_garch_rho`).

---

## Catégorie 4 — Échantillons OOS trop courts

- **Seuil critique** : T_oos < 250 (Christoffersen 1998 pour puissance suffisante sur
  Kupiec et Christoffersen CC).
- **Symptôme** : warning `T_oos=173 < 250` sur toutes les séries **weekly** (split 70/30
  sur ~574 obs = T_oos=173). P-values Kupiec peu fiables en petits échantillons.
- **Impact mesuré** : DM-GK retourne systématiquement `model1_wins` sur tous les tickers
  weekly — la puissance du test est insuffisante à alpha=99% avec T_oos=173.
- **Recommandation** : split temporel fixe sur weekly (train jusqu'à 2020, test
  2020-2026 ≈ 312 obs). Sur daily (T_oos ≈ 830) : puissance satisfaisante.

---

## Recommandations pré-analyse

Checklist à exécuter avant d'analyser un actif inconnu :

1. **Catégorie 1 ?** — vérifier que l'actif n'est pas BTC-USD ou un actif avec
   excès de kurtosis > 10. Si oui : utiliser FHS seul, pas de VaR GARCH.

2. **Bonne fréquence ?** — préférer daily à weekly pour GC=F et EURUSD=X (Cat. 2).
   Sur weekly avec n < 600 obs, l'hétéroscédasticité peut être indétectable.

3. **near_igarch ?** — si pers ≥ 0.98 : interpréter VaR avec réserve, ne pas utiliser
   FHS H>1, signaler en sortie.

4. **spec_gravite ?** — si `majeure` : résidus non iid, VaR dynamique peu fiable sur
   les clusters de volatilité. Résultat acceptable uniquement si Kupiec 99% passe.

5. **motif_selection ?** — si `IC_seul_AUCUN_VALIDE` : aucun modèle n'a passé le
   filtre statistique. Utiliser FHS H=1 comme référence principale.

6. **T_eff_dyn < 250 ?** — si oui (toutes séries weekly) : p-values à interpréter
   avec marge ±0.05-0.10. Ne pas conclure sur un seul test.

---

## Tableau de décision rapide

| Condition observée dans le CSV | Action recommandée |
|---|---|
| `ticker` = BTC-USD | Exclure VaR GARCH. FHS H=1 uniquement. |
| `all_candidates_failed_spec=True` | ALERTE — ne pas utiliser VaR GARCH opérationnellement |
| `motif_selection` = `IC_seul_AUCUN_VALIDE` | Utiliser VaR historique / FHS comme seule référence |
| `near_igarch=True` | Signaler. VaR H=1 utilisable, VaR H>1 déconseillée |
| `T_eff_dyn < 250` | P-values peu fiables — augmenter T_oos ou bootstrap |
| `spec_gravite='mineure'` | Acceptable avec mention explicite |
| `spec_gravite='majeure'` | Valider manuellement avant usage |
| `kupiec_99_pval < 0.01` | Modèle rejeté pour usage risk management réglementaire |
| `var99_garch < -1.0` (< -100%) | BUG BTC-001 — voir `docs/DETTE_TECHNIQUE.md` |

---

*Ce catalogue est mis à jour à chaque phase de validation multi-actifs.*
*Dernière mise à jour : Phase 2.1, juin 2026.*
