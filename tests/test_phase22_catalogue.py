# -*- coding: utf-8 -*-
"""
Tests Phase 2.2 — Catalogue des cas pathologiques.

3 tests structurels sur docs/CAS_PATHOLOGIQUES.md :
  Test 1 : presence du fichier + 4 categories requises
  Test 2 : chaque entree Cat. 1 contient les champs obligatoires
  Test 3 : markdown parseable sans erreur
"""
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_CATALOGUE = _ROOT / 'docs' / 'CAS_PATHOLOGIQUES.md'

_CATEGORIES_REQUISES = [
    r'##\s+Catégorie\s+1',
    r'##\s+Catégorie\s+2',
    r'##\s+Catégorie\s+3',
    r'##\s+Catégorie\s+4',
]

_CHAMPS_ENTREE_CAT1 = [
    r'\*\*Symptômes\*\*',
    r'\*\*Diagnostic\*\*',
    r'\*\*Recommandation\*\*',
    r'\*\*Verdict pipeline\*\*',
]

_ACTIFS_CAT1 = ['BTC-USD', '^GSPC daily']


# ── Test 1 : presence + categories ───────────────────────────────────────────

def test_catalogue_present_et_categories():
    """
    Le fichier docs/CAS_PATHOLOGIQUES.md existe et contient
    les 4 categories obligatoires (## Categorie 1 a 4).
    """
    assert _CATALOGUE.exists(), (
        f'docs/CAS_PATHOLOGIQUES.md absent : {_CATALOGUE}'
    )
    contenu = _CATALOGUE.read_text(encoding='utf-8')

    assert len(contenu) >= 1000, (
        f'Catalogue trop court ({len(contenu)} chars) — probablement vide ou tronque'
    )

    for pattern in _CATEGORIES_REQUISES:
        assert re.search(pattern, contenu, re.IGNORECASE), (
            f"Categorie manquante — pattern '{pattern}' absent du catalogue"
        )


# ── Test 2 : champs obligatoires dans Cat. 1 ─────────────────────────────────

def test_entrees_cat1_champs_obligatoires():
    """
    La section Categorie 1 contient les champs obligatoires :
    Symptomes, Diagnostic, Recommandation, Verdict pipeline.
    Les actifs BTC-USD et ^GSPC daily sont mentionnes.
    """
    contenu = _CATALOGUE.read_text(encoding='utf-8')

    # Extraire la section Cat. 1 (jusqu'a ## Categorie 2)
    match = re.search(
        r'##\s+Catégorie\s+1.*?(?=##\s+Catégorie\s+2)',
        contenu,
        re.DOTALL | re.IGNORECASE,
    )
    assert match, 'Section Categorie 1 non trouvee ou non terminee avant Categorie 2'

    section_cat1 = match.group(0)

    for champ in _CHAMPS_ENTREE_CAT1:
        assert re.search(champ, section_cat1, re.IGNORECASE), (
            f"Champ obligatoire manquant dans Cat. 1 : pattern '{champ}'"
        )

    for actif in _ACTIFS_CAT1:
        assert actif in section_cat1, (
            f"Actif '{actif}' non documente dans Categorie 1"
        )


# ── Test 3 : markdown parseable ───────────────────────────────────────────────

def test_catalogue_markdown_parseable():
    """
    Le fichier est parseable par la librairie markdown sans lever d'exception.
    Verifie aussi : au moins 1 tableau (|), au moins 5 headers (##).
    """
    import markdown  # pip install markdown

    contenu = _CATALOGUE.read_text(encoding='utf-8')

    try:
        html = markdown.markdown(contenu, extensions=['tables'])
    except Exception as exc:
        pytest.fail(f'markdown.markdown() a leve une exception : {exc}')

    assert len(html) > 500, 'HTML genere trop court — contenu suspect'

    # Au moins 4 headers de niveau 2 (## Categorie 1-4)
    h2_count = html.count('<h2>')
    assert h2_count >= 4, (
        f'Moins de 4 headers ## dans le HTML genere ({h2_count} trouves)'
    )

    # Au moins 1 tableau rendu (tag <table>)
    assert '<table>' in html, (
        'Aucun tableau rendu — verifier la syntaxe markdown des tableaux'
    )
