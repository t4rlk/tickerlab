# -*- coding: utf-8 -*-
"""Job store en mémoire pour les analyses asynchrones (MVP).

Un job = une exécution de pipeline. Le store est un simple dict protégé par un
verrou. Suffisant pour un MVP mono-process ; à remplacer par Redis/DB si besoin
de persistance ou de multi-process.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    """État d'une analyse. status ∈ {queued, running, done, error}."""
    id: str
    payload: dict
    status: str = 'queued'
    step: str = 'En file d\'attente'
    progress: int = 0
    error: str = ''
    result: Optional[dict] = None          # {report_url, figures[], summary{}} (v1: contrat result)
    pdf_path: Optional[str] = None         # chemin interne du PDF (non exposé)
    dossier: Optional[str] = None          # dossier résultats du job
    created: float = field(default_factory=time.time)
    # ── Champs v1 ────────────────────────────────────────────────────────────
    module: str = 'univarie'               # univarie | var | ruptures
    warning: Optional[str] = None          # ex. 'SERIE_FRAGILE'
    error_code: Optional[str] = None       # code machine de l'erreur (v1)
    pct_reel: Optional[int] = None         # progression OBSERVEE (jalons de log reels) ; None sinon

    def to_status(self) -> dict:
        """Représentation publique pour GET /api/run/{id} (surface legacy, inchangée)."""
        payload: dict = {'status': self.status}
        if self.status in ('running', 'queued'):
            payload['step'] = self.step
            payload['progress'] = self.progress
        if self.status == 'error':
            payload['error'] = self.error
        if self.status == 'done':
            payload['progress'] = 100
            payload['result'] = self.result
        return payload

    def to_status_v1(self) -> dict:
        """Représentation publique pour GET /api/v1/analyses/{id}.

        `progress.pct` est HONNÊTE : uniquement la progression OBSERVEE via les
        jalons de log réels (pct_reel). Tant qu'aucun jalon n'a été observé, pct
        vaut None et le front n'affiche que l'étape. `pct=100` seulement quand le
        job est réellement terminé.
        """
        if self.status == 'done':
            progress = {'etape': 'Terminé', 'pct': 100}
        else:
            progress = {'etape': self.step, 'pct': self.pct_reel}
        return {
            'job_id':   self.id,
            'status':   self.status,
            'progress': progress,
            'warning':  self.warning,
            'result':   self.result if self.status == 'done' else None,
            'error':    ({'code': self.error_code or 'ERREUR_INTERNE',
                          'message': self.error}
                         if self.status == 'error' else None),
        }


TTL_SECONDES: int = 3600  # jobs done/error conservés 1 h, puis évincés


class JobStore:
    """Conteneur thread-safe de jobs.

    Éviction : les jobs ``done`` ou ``error`` plus vieux que TTL_SECONDES
    sont supprimés à chaque appel à ``creer()``.  Les jobs ``queued`` et
    ``running`` ne sont jamais purgés automatiquement.

    Note de concurrence : ``job.progress`` et ``job.step`` sont mutés par le
    worker (thread pool) et lus par l'API (thread ASGI) sans verrou sur le Job
    lui-même.  Sous CPython le GIL rend ces affectations atomiques (tradeoff MVP
    documenté — acceptable pour un usage mono-utilisateur local).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def creer(self, payload: dict, job_id: Optional[str] = None) -> Job:
        """Crée (ou récupère) un job.

        Si `job_id` est fourni et déjà présent en mémoire, le job existant est
        retourné tel quel (IDEMPOTENCE : même requête normalisée => même job).
        Sinon un job est créé avec cet id (ou un id aléatoire si None).
        """
        with self._lock:
            self._purger_locked()
            if job_id is not None and job_id in self._jobs:
                return self._jobs[job_id]
            jid = job_id or uuid.uuid4().hex[:16]
            job = Job(id=jid, payload=payload,
                      module=str(payload.get('module', 'univarie')))
            self._jobs[jid] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _purger_locked(self) -> None:
        """Supprime les jobs terminés/en-erreur trop vieux. Appelé sous _lock."""
        limite = time.time() - TTL_SECONDES
        expirés = [
            jid for jid, j in self._jobs.items()
            if j.status in ('done', 'error') and j.created < limite
        ]
        for jid in expirés:
            del self._jobs[jid]
