# Dette technique — Pipeline PEA-Brent v6

Règle : tout `# TODO`, `# FIXME`, `# HACK` dans le code référence un ID ici.
Pas de "on verra plus tard" silencieux.

---

## BTC-001 — VaR GARCH non plafonnée à -100%
**Sévérité** : Critique  
**Découvert** : Phase 2.1, run BTC-USD weekly  
**Statut** : **RESOLU** — Phase 2.3-fix, commit `fix(phase2.3): t standardisee + proba queue + sigma_H exact + floor BTC-001`

**Fichier** : `core/var_engine.py` — fonctions `_apply_btc001_floor()` + `calculer_var_tvar()`

**Symptôme** : Sur BTC-USD weekly avec pers=0.998 et σ_t élevée en fin de série,
VaR95_GARCH = -200% et VaR99_GARCH = -357%. Valeur mathématiquement issue de
`-σ_t × q_ν(α)` avec σ_t ≈ 100% weekly et q_ν(1%) ≈ 3.6. Physiquement absurde :
un actif ne peut pas perdre plus de 100% de sa valeur en rendement simple.

**Correctif appliqué** :
```python
# core/var_engine.py — _apply_btc001_floor()
_BTC001_FLOOR = -99.9  # plancher physique (rendement simple borné à -100%)
if r_g < _BTC001_FLOOR:
    warnings.warn('[BTC-001] VaR GARCH floored...', UserWarning)
    r_g = -99.9
```
- Floor appliqué au niveau 99% uniquement dans `calculer_var_tvar()`
- Flag `df.attrs['var99_garch_floored']` (bool) dans le DataFrame retourné
- Colonne `var99_garch_floored` ajoutée au CSV multi-ticker (34 colonnes)
- Test `test_btc001_floor` dans `tests/test_phase23.py`

**Conséquences résolues** :
- DM-001 : `math range error` dans `gk_test()` résolu automatiquement (VaR floored)
- Le CSV signale désormais les runs BTC avec `var99_garch_floored=True`

---

## DM-001 — DM-GK math range error en cascade de BTC-001
**Sévérité** : Mineur (symptôme de BTC-001)  
**Découvert** : Phase 2.1, run BTC-USD weekly  
**Fichier** : `tickerlab/core/dm_gk.py`

**Symptôme** : `[DM-GK] ECHEC : math range error` quand VaR GARCH > 100%.
Le test GK calcule une exponentielle d'une valeur proportionnelle à la perte,
qui déborde `float` quand la VaR est de l'ordre de 200-360%.

**Correctif** : résolu automatiquement par BTC-001 (VaR floored). Alternativement,
ajouter un `try/except OverflowError` dans `dm_gk.py` avec retour de NaN explicite.

**Phase cible** : même que BTC-001

---

## PERF-001 — Component GARCH near-IGARCH sans timeout
**Sévérité** : Mineur  
**Découvert** : Phase 2.1, run ^GSPC daily (270.4s vs ~75s attendu)  
**Statut** : **RESOLU** — Phase 4.1  
**Fichier** : `scripts/run_multi_ticker.py` (`run_unique`), `tickerlab/core/component_garch.py`

**Symptôme** : quand rho → 1 (near-IGARCH), l'optimiseur SLSQP multiplie les itérations
car les contraintes de bord sont actives. Avec multi-start (3-5 points de départ) et
2765 observations, le total peut atteindre 270s par run.

**Correctifs appliqués** :
- `core/component_garch.py` : `maxiter` réduit de 500 → 200 sur les 2 appels `minimize()`
- `scripts/run_multi_ticker.py` : `fut.result(timeout=240)` + catch `FutureTimeout` en mode `--workers > 1`

**Phase cible** : Phase 3.1 (API propre + gestion erreurs typées)

---

## LOG-001 — Omega mal conditionnée : log verbeux
**Sévérité** : Cosmétique  
**Découvert** : Phase 2.1, ~80% des runs  
**Statut** : **RESOLU** — Phase 3.1 (logging structuré global)  
**Fichier** : `tickerlab/core/dm_gk.py` (`gk_test`)

**Symptôme** : `gk_test: Omega mal conditionnee (cond=2e+16) — passage a pinv`
imprimé sur stdout à chaque appel GK avec matrice mal conditionnée. Comportement
**correct** (fallback pinv fonctionne), mais verbeux et masque les vrais warnings.

**Correctif appliqué** : `_log.warning()` remplace `print()` dans `dm_gk.py gk_test()`.

**Phase cible** : Phase 3.1 (logging structuré global)
