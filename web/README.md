# tickerlab — couche web (configurateur)

Fine façade FastAPI qui connecte le configurateur statique au pipeline
économétrique existant (`tickerlab.main.run_pipeline`). Aucune logique de
calcul n'est réimplémentée.

## Installation

```bash
pip install -e .                    # rend le package tickerlab importable
pip install -r requirements-web.txt # fastapi, uvicorn, python-dotenv
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
# → http://127.0.0.1:8000   (local uniquement, non exposé au réseau)
```

Le configurateur est servi sur `/`. Cliquer « Lancer l'analyse » envoie la
configuration au backend, affiche la progression réelle, puis un lien de
téléchargement du rapport PDF + un résumé des résultats.

## API

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/api/run` | `{ ticker, from, to, freq, price, outputs[] }` → `{ job_id }` |
| `GET`  | `/api/run/{id}` | `{ status, step?, progress?, error?, result? }` |
| `GET`  | `/api/report/{id}` | PDF du rapport |
| `GET`  | `/api/figure/{id}/{name}` | figure PNG (si produite) |

`status` ∈ `queued | running | done | error`. Les jobs sont sérialisés
(un à la fois) ; les sorties vont dans `web_runs/{job_id}/` (ignoré par git).

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
