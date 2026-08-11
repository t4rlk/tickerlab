# -*- coding: utf-8 -*-
"""API FastAPI du configurateur tickerlab.

Routes :
    GET  /                  → sert le configurateur (web/static/index.html)
    POST /api/run           → lance une analyse, renvoie { job_id }
    GET  /api/run/{id}       → statut { status, step?, progress?, error?, result? }
    GET  /api/report/{id}    → renvoie le PDF du rapport
    GET  /api/figure/{id}/{name} → renvoie une figure PNG (si produite)

L'exécution du pipeline est sérialisée (1 worker) : un job à la fois, les autres
restent 'queued'. La progression est dérivée des logs du pipeline, sans toucher
à la logique économétrique.
"""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .jobs import Job, JobStore
from .mapping import construire_config
from .schemas_v1 import AnalyseRequest
from .observations import evaluer_longueur, SEUIL_REFUS
from .signaux import generer_signaux

# ── Chemins & secrets ─────────────────────────────────────────────────────────
_RACINE     = Path(__file__).resolve().parent.parent
_STATIC_DIR = Path(__file__).resolve().parent / 'static'
_RUNS_DIR   = _RACINE / 'web_runs'
_RUNS_DIR.mkdir(exist_ok=True)

# Charge les variables d'environnement depuis .env à la racine (clés LLM, etc.).
# Les clés restent 100 % côté serveur — jamais transmises au client.
load_dotenv(_RACINE / '.env')

_log = logging.getLogger('tickerlab.web')

app = FastAPI(title='tickerlab — API configurateur', version='1.0')

# ── CORS : VOLONTAIREMENT AUCUN middleware ───────────────────────────────────
# Le configurateur (web/static) est servi MÊME ORIGINE par cette application
# (StaticFiles monté sur '/'). Les appels du front sont donc same-origin : aucun
# en-tête CORS n'est nécessaire. Ne PAS ajouter de CORSMiddleware permissif — ce
# serait élargir la surface d'attaque sans aucun besoin. (Décision explicite.)

_store = JobStore()
_executor = ThreadPoolExecutor(max_workers=1)  # pipeline sérialisé (MVP)


# ── Modèle de requête ─────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Payload du configurateur. 'from' est aliasé (mot réservé Python)."""
    model_config = ConfigDict(populate_by_name=True)

    ticker:  str
    from_:   str = Field(alias='from')
    to:      str
    freq:    Literal['daily', 'weekly', 'monthly', 'annual']
    price:   Literal['close', 'adjclose']
    outputs: list[str] = []


# ── Mapping logs → progression ────────────────────────────────────────────────
# (sous-chaîne du message, libellé FR, progression). Ordre = chronologie pipeline.
_ETAPES: list[tuple[str, str, int]] = [
    ('PIPELINE TICKERLAB', 'Initialisation',           3),
    (' rendements ',        'Données téléchargées',     12),
    ('>>> ARIMA(',          'Modèle ARIMA (moyenne)',   25),
    ('AIC/n=',              'Sélection GARCH',          45),
    ('[ICSS]',              'Ruptures structurelles',   52),
    ('[CompGARCH]',         'Component GARCH',          56),
    ('[FHS]',               'Simulation FHS',           62),
    ('[Bootstrap]',         'Bootstrap IC VaR',         72),
    ('[Rolling]',           'Backtest glissant',        78),
    ('ETAPE IA',            'Rédaction du rapport (IA)', 85),
    ('(TOTAL)',             'Génération du rapport',    95),
]


class _ProgressHandler(logging.Handler):
    """Handler de logs qui met à jour la progression d'un job (monotone)."""

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.INFO)
        self._job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        for fragment, label, pct in _ETAPES:
            if fragment in msg and pct > self._job.progress:
                self._job.progress = pct
                self._job.step = label
                # pct_reel = progression OBSERVEE (jalon de log reel) exposee par
                # l'API v1 ; distincte des valeurs coarse posees par l'executeur.
                self._job.pct_reel = pct
                break


# ── Exécution d'un job ────────────────────────────────────────────────────────

def _construire_resume(result, config: dict) -> dict:
    """Extrait un résumé lisible du PipelineResult (best-effort, défensif)."""
    meta = getattr(result, 'meta', {}) or {}
    resume: dict = {
        'ticker':  meta.get('ticker', config['data']['ticker']),
        'duree_s': meta.get('duree_totale_s'),
    }
    try:
        if result.rendements is not None:
            resume['n_obs'] = int(len(result.rendements.dropna()))
    except Exception as exc:
        _log.debug('_construire_resume : n_obs indisponible : %s', exc)

    # Modèle GARCH retenu
    try:
        best = result.garch_best
        b = best.to_dict() if hasattr(best, 'to_dict') else dict(best)
        modele = b.get('modele') or b.get('model') or 'GARCH'
        p, q = b.get('p'), b.get('q')
        dist = b.get('dist')
        label = f'{modele}({p},{q})' if p is not None and q is not None else str(modele)
        if dist:
            label += f'-{dist}'
        resume['modele_garch'] = label
    except Exception as exc:
        _log.debug('_construire_resume : modele_garch indisponible : %s', exc)

    # VaR 99 % GARCH (même accès défensif que la CLI stress)
    try:
        df_var = result.var
        if df_var is not None:
            idx = '99%' if '99%' in df_var.index else (0.99 if 0.99 in df_var.index else None)
            col = 'VaR GARCH' if 'VaR GARCH' in df_var.columns else df_var.columns[0]
            if idx is not None:
                resume['var_99'] = round(float(df_var.loc[idx, col]), 4)
    except Exception as exc:
        _log.debug('_construire_resume : var_99 indisponible : %s', exc)

    return resume


def _localiser_pdf(dossier: str, config: dict, report_demande: bool) -> Optional[Path]:
    """Localise le PDF à servir.

    Préférence : rapport IA compilé (latex_doc/main.pdf) si la rédaction a été
    demandée, sinon repli sur le PDF ReportLab consolidé (Rapport_*.pdf).
    Retourne None si aucun PDF n'est trouvé. Réutilisé par les deux surfaces
    (legacy /api/run et v1).
    """
    if report_demande:
        ia = sorted(Path(dossier).glob('**/latex_doc/main.pdf'))
        if ia:
            return ia[0]
    dossier_final = Path(config['output']['dossier_resultats'])
    pdfs = (sorted(dossier_final.glob('Rapport_*.pdf'))
            or sorted(Path(dossier).glob('**/Rapport_*.pdf')))
    return pdfs[0] if pdfs else None


def _pdf_response(job: Optional[Job]) -> FileResponse:
    """Réponse FileResponse du PDF d'un job terminé (404 sinon). Réutilisé par
    les deux surfaces de téléchargement du rapport."""
    if job is None or job.status != 'done' or not job.pdf_path:
        raise HTTPException(status_code=404, detail='Rapport indisponible.')
    pdf = Path(job.pdf_path)
    if not pdf.exists():
        raise HTTPException(status_code=404, detail='Fichier PDF introuvable.')
    nom = pdf.name
    if nom == 'main.pdf':  # rapport IA : nom parlant pour le téléchargement
        tk = (job.payload.get('ticker') or 'rapport').replace('=', '_').replace('^', '').replace('-', '_')
        nom = f'Rapport_IA_{tk}.pdf'
    return FileResponse(str(pdf), media_type='application/pdf', filename=nom)


def _executer_job(job: Job) -> None:
    """Corps exécuté dans le worker : pipeline + collecte des sorties."""
    from tickerlab.main import run_pipeline
    from tickerlab.core.config_schema import valider_config_structure

    job.status = 'running'
    job.step = 'Téléchargement des données'
    job.progress = 6

    dossier = str(_RUNS_DIR / job.id)
    job.dossier = dossier

    handler = _ProgressHandler(job)
    pipeline_logger = logging.getLogger('tickerlab')
    pipeline_logger.addHandler(handler)
    try:
        config = construire_config(job.payload, dossier)
        valider_config_structure(config)           # garde-fou structure (Phase 7)

        result = run_pipeline(config)              # pipeline réel, inchangé

        # Localisation du PDF à servir (helper partagé).
        report_demande = ('report' in (job.payload.get('outputs') or []))
        pdf_path = _localiser_pdf(dossier, config, report_demande)
        if pdf_path is None:
            if report_demande:
                raise RuntimeError(
                    'La rédaction IA n\'a pas abouti (quota ou clé API indisponible) '
                    'et aucun rapport PDF n\'a pu être produit. Réessayer plus tard '
                    'ou décocher « Rapport rédigé ».'
                )
            raise RuntimeError('Le rapport PDF n\'a pas été généré.')
        job.pdf_path = str(pdf_path)

        # Figures éventuelles (PNG) produites dans le dossier du job
        figures = []
        for png in sorted(Path(dossier).glob('**/*.png')):
            figures.append(f'/api/figure/{job.id}/{png.name}')

        job.result = {
            'report_url': f'/api/report/{job.id}',
            'figures':    figures,
            'summary':    _construire_resume(result, config),
        }
        job.progress = 100
        job.step = 'Terminé'
        job.status = 'done'

    except Exception as exc:                        # noqa: BLE001 (message remonté au front)
        job.status = 'error'
        job.error = str(exc)
        _log.exception('Job %s en échec : %s', job.id, exc)
    finally:
        pipeline_logger.removeHandler(handler)


# ── Routes API ────────────────────────────────────────────────────────────────

@app.post('/api/run')
def lancer_analyse(req: RunRequest) -> JSONResponse:
    """Valide légèrement, crée un job et le programme dans le worker."""
    payload = {
        'ticker':  req.ticker,
        'from':    req.from_,
        'to':      req.to,
        'freq':    req.freq,
        'price':   req.price,
        'outputs': req.outputs,
    }
    if not payload['ticker'].strip():
        raise HTTPException(status_code=422, detail='Ticker manquant.')
    if payload['from'] >= payload['to']:
        raise HTTPException(
            status_code=422,
            detail='La date de début doit être antérieure à la date de fin.',
        )
    job = _store.creer(payload)
    _executor.submit(_executer_job, job)
    return JSONResponse({'job_id': job.id})


@app.get('/api/run/{job_id}')
def statut_analyse(job_id: str) -> JSONResponse:
    """Statut courant du job (polling côté front)."""
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job inconnu.')
    return JSONResponse(job.to_status())


@app.get('/api/report/{job_id}')
def telecharger_rapport(job_id: str) -> FileResponse:
    """Renvoie le PDF du rapport (404 si le job n'est pas terminé)."""
    return _pdf_response(_store.get(job_id))


@app.get('/api/figure/{job_id}/{name}')
def telecharger_figure(job_id: str, name: str) -> FileResponse:
    """Renvoie une figure PNG du job (nom validé, pas de traversée de chemin)."""
    job = _store.get(job_id)
    if job is None or not job.dossier:
        raise HTTPException(status_code=404, detail='Job inconnu.')
    if '/' in name or '\\' in name or '..' in name or not name.endswith('.png'):
        raise HTTPException(status_code=400, detail='Nom de figure invalide.')
    matches = list(Path(job.dossier).glob(f'**/{name}'))
    if not matches:
        raise HTTPException(status_code=404, detail='Figure introuvable.')
    return FileResponse(str(matches[0]), media_type='image/png', filename=name)


# ── API v1 ────────────────────────────────────────────────────────────────────
# Surface canonique conforme aux contrats du cahier des charges. /api/run* reste
# disponible en alias legacy (front + tests existants) ; toute la logique lourde
# (pipeline, localisation PDF) est réutilisée via les helpers ci-dessus.

def _executer_job_v1(job: Job) -> None:
    """Corps worker pour l'API v1 : pipeline réel + contrat `result` v1."""
    from tickerlab.main import run_pipeline
    from tickerlab.core.config_schema import valider_config_structure
    from tickerlab.core.exceptions import ModuleUnavailableError, DataError
    from .resultat import construire_result

    job.status = 'running'
    job.step = 'Téléchargement des données'
    job.progress = 6

    dossier = str(_RUNS_DIR / job.id)
    job.dossier = dossier

    handler = _ProgressHandler(job)
    pipeline_logger = logging.getLogger('tickerlab')
    pipeline_logger.addHandler(handler)
    try:
        # detail_frtb=True : le pipeline calcule et expose les 6 tests de backtest.
        config = construire_config(job.payload, dossier, detail_frtb=True)
        valider_config_structure(config)

        result = run_pipeline(config)

        report_demande = ('report' in (job.payload.get('outputs') or []))
        pdf_path = _localiser_pdf(dossier, config, report_demande)
        job.pdf_path = str(pdf_path) if pdf_path else None
        rapport_url = f'/api/v1/analyses/{job.id}/rapport' if job.pdf_path else None

        job.result = construire_result(result, config, rapport_pdf_url=rapport_url)
        job.step = 'Terminé'
        job.progress = 100
        job.pct_reel = 100
        job.status = 'done'

    except ModuleUnavailableError as exc:
        job.status = 'error'
        job.error_code = 'MODULE_INDISPONIBLE'
        job.error = str(exc)
        _log.warning('Job v1 %s : module indisponible : %s', job.id, exc)
    except DataError as exc:
        job.status = 'error'
        job.error_code = 'DONNEES_INDISPONIBLES'
        job.error = str(exc)
        _log.warning('Job v1 %s : données indisponibles : %s', job.id, exc)
    except Exception as exc:                        # noqa: BLE001 — journalisé + remonté
        job.status = 'error'
        job.error_code = 'ERREUR_INTERNE'
        job.error = str(exc)
        _log.exception('Job v1 %s en échec : %s', job.id, exc)
    finally:
        pipeline_logger.removeHandler(handler)


@app.post('/api/v1/analyses', status_code=202)
def creer_analyse_v1(req: AnalyseRequest) -> JSONResponse:
    """Crée une analyse (job pattern). Validations serveur = source de vérité."""
    # Module multivarié indisponible sur cette branche.
    if req.module == 'var':
        return JSONResponse(status_code=422, content={
            'code': 'MODULE_INDISPONIBLE',
            'message': "Le module multivarié « var » (Granger, IRF, FEVD) n'est pas "
                       "disponible sur ce déploiement. Modules : univarie, ruptures.",
        })
    # Dates cohérentes (comparaison ISO YYYY-MM-DD).
    if str(req.date_from) >= str(req.date_to):
        return JSONResponse(status_code=422, content={
            'code': 'DATES_INCOHERENTES',
            'message': 'La date de début doit être antérieure à la date de fin.',
        })
    # Longueur de série estimée (garde 250 / 500).
    n_obs, statut = evaluer_longueur(req.date_from, req.date_to, req.freq)
    if statut == 'REFUS':
        return JSONResponse(status_code=422, content={
            'code': 'SERIE_TROP_COURTE',
            'message': f'Série trop courte : ~{n_obs} observations estimées '
                       f'(minimum requis {SEUIL_REFUS}).',
            'n_observations_estime': n_obs,
        })
    warning = 'SERIE_FRAGILE' if statut == 'FRAGILE' else None

    # Idempotence : hash scientifique => job_id stable. Même requête encore en
    # mémoire ⇒ on renvoie le job existant sans relancer le pipeline.
    job_id = req.hash_scientifique()
    existant = _store.get(job_id)
    if existant is not None:
        return JSONResponse(status_code=202, content={
            'job_id': existant.id, 'status': existant.status,
            'warning': existant.warning,
        })

    job = _store.creer(req.to_internal_payload(), job_id=job_id)
    job.warning = warning
    job.module = req.module
    _executor.submit(_executer_job_v1, job)
    return JSONResponse(status_code=202, content={
        'job_id': job.id, 'status': 'queued', 'warning': warning,
    })


@app.get('/api/v1/analyses/{job_id}')
def statut_analyse_v1(job_id: str) -> JSONResponse:
    """Statut v1 du job (polling front)."""
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job inconnu.')
    return JSONResponse(job.to_status_v1())


@app.get('/api/v1/analyses/{job_id}/rapport')
def rapport_analyse_v1(job_id: str) -> FileResponse:
    """PDF réel du rapport (Content-Type application/pdf)."""
    return _pdf_response(_store.get(job_id))


@app.get('/api/v1/signaux')
def signaux_v1(symbol: str, days: int = 30) -> JSONResponse:
    """Signaux directionnels (démonstration). Cf. web/signaux.py (SPEC PROVISOIRE)."""
    from tickerlab.core.exceptions import DataError
    if not symbol or not symbol.strip():
        return JSONResponse(status_code=422, content={
            'code': 'SYMBOLE_MANQUANT', 'message': 'Le paramètre symbol est requis.'})
    try:
        reponse = generer_signaux(symbol, days)
    except DataError as exc:
        return JSONResponse(status_code=422, content={
            'code': 'DONNEES_INDISPONIBLES', 'message': str(exc)})
    except Exception as exc:                        # noqa: BLE001 — journalisé + remonté
        _log.exception('signaux v1 : échec pour %s : %s', symbol, exc)
        return JSONResponse(status_code=500, content={
            'code': 'ERREUR_INTERNE', 'message': str(exc)})
    return JSONResponse(reponse)


# ── Accueil : page d'atterrissage du site multi-pages ─────────────────────────
# Le site (landing → analyse → methodologie → contact) a landing.html pour accueil.
# On sert donc landing.html sur '/'. Ces routes explicites sont enregistrees AVANT
# le mount statique ci-dessous : elles ont priorite sur le '/' du mount (qui, avec
# html=True, servirait sinon index.html). Le configurateur mono-page reste servi a
# /index.html (par le mount) et via l'alias parlant /configurateur.
@app.get('/', include_in_schema=False)
def accueil() -> FileResponse:
    """Page d'accueil du site : landing.html (même origine que l'API)."""
    return FileResponse(str(_STATIC_DIR / 'landing.html'), media_type='text/html')


@app.get('/configurateur', include_in_schema=False)
def configurateur() -> FileResponse:
    """Alias parlant vers le configurateur mono-page (index.html)."""
    return FileResponse(str(_STATIC_DIR / 'index.html'), media_type='text/html')


# ── Statique (site + configurateur) — monté en dernier pour ne pas masquer /api ─
app.mount('/', StaticFiles(directory=str(_STATIC_DIR), html=True), name='static')


def main(host: str = '127.0.0.1', port: int = 8742) -> None:
    """Point d'entrée : `tickerlab serve`, `tickerlab-web` ou `python -m web.app`.

    Défaut 127.0.0.1:8742 : local uniquement, non exposé au réseau. host/port sont
    paramétrables (cf. sous-commande CLI `tickerlab serve`). On passe l'objet `app`
    (et non une chaîne d'import) pour être robuste au contexte de lancement.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=int(port), reload=False)


if __name__ == '__main__':
    # S'assure que le package racine est importable si lancé hors install -e.
    sys.path.insert(0, str(_RACINE.parent))
    main()
