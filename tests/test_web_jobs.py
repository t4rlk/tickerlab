# -*- coding: utf-8 -*-
"""Tests unitaires du JobStore (éviction TTL, thread-safety)."""
import time
import threading

import pytest

from web.jobs import Job, JobStore, TTL_SECONDES


def _job_vieux(store: JobStore, status: str) -> Job:
    """Crée un job et le vieillit artificiellement de TTL+1 secondes."""
    job = store.creer({'ticker': 'BZ=F'})
    job.status = status
    job.created = time.time() - TTL_SECONDES - 1
    return job


class TestEvictionTTL:
    def test_job_done_evince_apres_ttl(self):
        store = JobStore()
        vieux = _job_vieux(store, 'done')
        store.creer({'ticker': 'CL=F'})          # déclenche la purge
        assert store.get(vieux.id) is None

    def test_job_error_evince_apres_ttl(self):
        store = JobStore()
        vieux = _job_vieux(store, 'error')
        store.creer({'ticker': 'CL=F'})
        assert store.get(vieux.id) is None

    def test_job_running_non_evince(self):
        store = JobStore()
        vieux = _job_vieux(store, 'running')
        store.creer({'ticker': 'CL=F'})
        assert store.get(vieux.id) is not None

    def test_job_queued_non_evince(self):
        store = JobStore()
        vieux = _job_vieux(store, 'queued')
        store.creer({'ticker': 'CL=F'})
        assert store.get(vieux.id) is not None

    def test_job_done_recent_conserve(self):
        """Un job done encore dans le TTL ne doit pas être purgé."""
        store = JobStore()
        recent = store.creer({'ticker': 'BZ=F'})
        recent.status = 'done'
        # Ne pas vieillir — il est récent
        store.creer({'ticker': 'CL=F'})
        assert store.get(recent.id) is not None

    def test_plusieurs_vieux_purges(self):
        store = JobStore()
        vieux1 = _job_vieux(store, 'done')
        vieux2 = _job_vieux(store, 'error')
        store.creer({'ticker': 'CL=F'})
        assert store.get(vieux1.id) is None
        assert store.get(vieux2.id) is None

    def test_nouveau_job_accessible_apres_purge(self):
        store = JobStore()
        _job_vieux(store, 'done')
        nouveau = store.creer({'ticker': 'GC=F'})
        assert store.get(nouveau.id) is not None
        assert store.get(nouveau.id).payload['ticker'] == 'GC=F'


class TestJobStoreThreadSafety:
    def test_creations_concurrentes(self):
        """Plusieurs threads créent des jobs simultanément sans lever d'exception."""
        store = JobStore()
        erreurs = []

        def creer():
            try:
                store.creer({'ticker': 'BZ=F'})
            except Exception as e:
                erreurs.append(e)

        threads = [threading.Thread(target=creer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not erreurs, f'Erreurs concurrentes : {erreurs}'
