# tickerlab

Pipeline d'analyse de volatilité et de VaR pour actifs financiers (PEA, matières premières, indices). Modélisation ARIMA/GARCH, VaR multi-méthodes, backtests réglementaires, rapport PDF automatique.

## Installation

```bash
pip install -e .           # pipeline seul
pip install -e ".[web]"    # + configurateur web (FastAPI)
pip install -e ".[ai]"     # + rédaction IA du rapport (Groq/Anthropic)
pip install -e ".[dev]"    # + outils de test
```

## Usage CLI

```bash
tickerlab run BZ=F --from 2015-01-01 --to 2025-01-01 --freq daily --out report
tickerlab --help
```

## Usage web

```bash
tickerlab-web          # démarre sur http://localhost:8000
# ou
python -m web.app
```

Configurateur accessible sur `http://localhost:8000` — analyse en un clic, rapport PDF téléchargeable.

## Structure

```
core/           Logique économétrique (ARIMA, GARCH, VaR, rapport PDF)
  rapport/
    sections/   Sous-modules PDF : stationarite, arima_garch, var_backtest…
web/            Couche FastAPI (configurateur + API REST)
  static/       Pages HTML (landing, analyse, contact, méthodologie)
tests/          166+ tests unitaires et d'intégration
docs/           Notes techniques, RELEASE_NOTES, références
```

## Branches

- `main` / `master` — socle Phase 2 (stable)
- `feat/web-configurator` — configurateur web + pipeline complet (branche active)
- `chore/audit-remediation` — remédiation post-audit (en cours de merge)

> **Note** : la branche de référence à jour est `feat/web-configurator`. `master` est figé à la Phase 2.
> Décision de merge à valider avec Tarik (voir Chantier 5.2 de l'audit).

## Liens

- [CHANGELOG](CHANGELOG.md)
- [Méthodologie](web/README.md)
- [Documentation technique](docs/)
- [Audit & remédiation](INSTRUCTION_CLAUDE_CODE_AUDIT.md)
