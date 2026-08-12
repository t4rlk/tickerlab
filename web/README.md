# tickerlab — couche web (configurateur)

Fine façade FastAPI qui connecte le configurateur statique au pipeline
économétrique existant (`tickerlab.main.run_pipeline`). Aucune logique de
calcul n'est réimplémentée.

## Installation

```bash
pip install -e ".[web]"             # tickerlab + couche web (fastapi, uvicorn, python-dotenv)
```

## Secrets (optionnel — uniquement pour la sortie « Rapport rédigé »)

```bash
cp .env.example .env
# renseigner GROQ_API_KEY (ou le provider de config.yaml : ai_writer.env_key)
```

Les clés restent **100 % côté serveur** ; elles ne sont jamais transmises au
client. Sans clé, l'analyse aboutit quand même : le PDF ReportLab est produit,
seule la rédaction IA est ignorée (avertissement dans les logs).

## Lancement

```bash
python -m web.app
# → http://127.0.0.1:8742   (local uniquement, non exposé au réseau)
```

Le configurateur est servi sur `/`. Cliquer « Lancer l'analyse » envoie la
configuration au backend, affiche la progression réelle, puis un lien de
téléchargement du rapport PDF + un résumé des résultats.

## API (v1 — surface canonique)

Toutes les analyses passent par l'API versionnée `/api/v1`. Les jobs sont
sérialisés (un à la fois) ; les sorties vont dans `web_runs/{job_id}/` (ignoré
par git).

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/api/v1/analyses` | Crée une analyse. Corps : `{ symbol, module, date_from, date_to, freq, price, outputs[] }` → `202 { job_id, status, warning }` |
| `GET`  | `/api/v1/analyses/{id}` | Statut (polling) → `{ status, progress{etape,pct}, warning, result?, error? }` |
| `GET`  | `/api/v1/analyses/{id}/rapport` | PDF du rapport (`application/pdf`) |
| `GET`  | `/api/v1/signaux?symbol=…&days=…` | Signaux directionnels (démonstration) |

**Paramètres** — `module` ∈ `univarie` (défaut) · `ruptures` · `var` (indisponible
sur ce déploiement → `422`) ; `freq` ∈ `daily · weekly · monthly` ; `price` ∈
`adj_close · close` ; `outputs[]` : sorties additionnelles (ex. `breaks`, `report`).

**Statut** — `status` ∈ `queued · running · done · error`. `progress.pct` est
*honnête* : `null` tant qu'aucun jalon de log réel n'a été franchi, `100`
seulement à la fin.

**`result`** (quand `status = done`) :

```json
{
  "modele_retenu": "GJR-GARCH(1,1)",
  "distribution": "Student-t",
  "persistance": 0.98,
  "taux_exception_var99": 0.012,
  "ratio_tvar_var": 1.15,
  "backtesting": { "valides": 5, "total": 6,
                   "detail": { "kupiec": {…}, "christoffersen": {…}, "dq": {…},
                               "acerbi_szekely": {…}, "fissler_ziegel": {…},
                               "berkowitz_pit": {…} } },
  "n_observations": 4500,
  "rapport_pdf_url": "/api/v1/analyses/{id}/rapport"
}
```

**Erreurs** (`422`, champ `code`) : `MODULE_INDISPONIBLE`, `DATES_INCOHERENTES`,
`SERIE_TROP_COURTE`. Avertissement non bloquant : `SERIE_FRAGILE`.

> **Alias legacy** (rétrocompatibilité) : `POST /api/run`, `GET /api/run/{id}`,
> `GET /api/report/{id}`, `GET /api/figure/{id}/{name}` restent disponibles, mais
> l'API v1 ci-dessus est la surface canonique.

## Mapping front → config

| Front | Config pipeline |
|---|---|
| `ticker` | `data.ticker` |
| `from` / `to` | `data.start_date` / `data.end_date` |
| `freq` | `data.frequency` (`daily/weekly/monthly/annual`) |
| `price` | `data.auto_adjust` (`close`→False, `adjclose`→True) |
| `outputs: breaks` | `structural_breaks.enabled` |
| `outputs: report` | `ai.enabled` (via `activer_ia()`) |
| `outputs: garch/var/backtest/charts` | intrinsèques (toujours calculés) |

`monthly`/`annual` sur une période courte (< 250 obs) sont **rejetés
proprement** par `valider_donnees` avec un message affiché au front — pas de
triche, pas de blocage muet.
