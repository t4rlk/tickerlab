"""
Rédaction automatique des sections LaTeX via l'API Anthropic (Claude).

Usage :
    from tickerlab.utils.ai_writer import construire_contexte, rediger_toutes_sections
    contexte = construire_contexte(...)
    rediger_toutes_sections(contexte, output_dir, model)
"""

import re
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ── Prompt système commun à toutes les sections ────────────────────────────
SYSTEM_PROMPT = """Tu es un analyste quantitatif rédigeant une note technique de risque \
de marché destinée à un comité des risques (niveau M2 finance / desk front office).

REGISTRE ET STYLE — PROSE ACADÉMIQUE FRANÇAISE :
- Français académique soigné, fluide et articulé. Non un style télégraphique, \
non une juxtaposition de phrases courtes. Les phrases peuvent être amples dès \
lors que la logique du raisonnement reste limpide.
- Articule le raisonnement avec les connecteurs logiques de l'argumentation \
française : « en effet », « dès lors », « il s'ensuit que », « or », « ainsi », \
« par conséquent », « néanmoins », « à cet égard », « en revanche ». Le lien \
constat → cause → implication doit transparaître dans l'enchaînement des \
propositions, à l'intérieur du paragraphe.
- Construis de véritables paragraphes : une idée directrice développée et \
nuancée, et non une liste de constats secs.
- Orthographe, grammaire et typographie françaises rigoureuses : accord des \
participes passés, concordance des temps, ponctuation correcte, espaces \
insécables avant « ; », « : », « ! », « ? » et dans les unités. Guillemets \
français «~…~» et non "...".
- Voix impersonnelle et objective : pas de « je », pas d'adresse au lecteur, \
pas de superlatifs ni de formulations commerciales.
- Vocabulaire technique exact : emploie les termes canoniques \
(hétéroscédasticité conditionnelle, persistance de la variance, queue de \
distribution, blanchiment des résidus, couverture inconditionnelle) à bon escient.

FORMAT LATEX :
- LaTeX uniquement, jamais de Markdown
- Pas de \\section ou \\subsection (déjà dans le fichier appelant)
- Échapper systématiquement \\%, \\_, \\&, \\$ dans le texte courant
- Utiliser \\textit, \\textbf avec parcimonie
- Si formule : environnement equation* (pas numéroté sauf demande explicite)
- Commencer directement dans le contenu, sans phrase d'introduction méta
- Décimales en notation française (virgule) dans le texte rédigé \
  (ex : 0,47 et non 0.47), sauf dans les environnements mathématiques LaTeX

INTÉGRITÉ SCIENTIFIQUE (impératif absolu) :
- Toute affirmation chiffrée renvoie à une valeur fournie dans le CONTEXTE.
- Tu n'inventes, ne recalcules ni n'extrapoles JAMAIS une valeur numérique.
- Si une donnée est absente du contexte, tu écris [DONNÉE MANQUANTE] et tu \
  ne combles pas le vide.
- Tu interprètes les résultats, tu ne les paraphrases pas : une p-value de \
  0,47 ne se commente pas par « égale 0,47 » mais par « ne permet pas de \
  rejeter l'hypothèse nulle au seuil de 5~\\% ».
- Tu cites les tests par leur nom et leur référence lorsque le contexte les \
  fournit (« test de Kupiec (1995) », « test d'Engle~\\&~Ng (1993) »).
"""

# Noms complets lisibles pour les tickers courants
_NOM_SOUS_JACENT = {
    'BZ=F': 'pétrole brut Brent',
    'CL=F': 'pétrole brut WTI (West Texas Intermediate)',
    'GC=F': 'or (Gold Futures)',
    'SI=F': 'argent (Silver Futures)',
    'NG=F': 'gaz naturel (Natural Gas Futures)',
    'HG=F': 'cuivre (Copper Futures)',
    'ZC=F': 'maïs (Corn Futures)',
    'ZS=F': 'soja (Soybean Futures)',
    'ZW=F': 'blé (Wheat Futures)',
}


def _nom_sous_jacent(ticker: str) -> str:
    return _NOM_SOUS_JACENT.get(ticker, ticker)


def _fmt(val, decimales: int = 4) -> str:
    """Formate un flottant ou retourne 'N/A' si NaN."""
    try:
        v = float(val)
        if np.isnan(v):
            return 'N/A'
        return f'{v:.{decimales}f}'
    except (TypeError, ValueError):
        return str(val)


def _fmt2(val) -> str:
    return _fmt(val, 2)


def construire_contexte(
    ticker, prix, rendements, arima_result, best, garch_final,
    df_var, df_bt, df_vol, config, T_train, T_eff_dyn,
    spec_verdict=None, igarch_diag=None,
) -> dict:
    """
    Construit un dictionnaire structuré avec toutes les métriques nécessaires
    à la rédaction des sections par l'IA.

    Returns
    -------
    dict
        Contexte complet prêt à être substitué dans les templates de prompts.
    """
    from tickerlab.core.data_loader import ensure_series
    rdt        = ensure_series(rendements).dropna()
    prix_clean = ensure_series(prix).dropna()

    # ── Statistiques descriptives des rendements ───────────────────────────
    jb_stat, jb_p = sp_stats.jarque_bera(rdt.values)
    skew_val  = float(sp_stats.skew(rdt.values))
    kurt_val  = float(sp_stats.kurtosis(rdt.values, fisher=False))  # kurtosis total

    # ── Paramètres GARCH ───────────────────────────────────────────────────
    params = garch_final.params
    vol_type = str(best.get('vol', best.get('modele', 'GARCH')))

    omega = float(params.get('omega', float('nan')))
    alpha = float(params.get('alpha[1]', float('nan')))
    beta  = float(params.get('beta[1]',  float('nan')))
    gamma = float(params.get('gamma[1]', float('nan')))
    nu    = float(params.get('nu', float('nan')))

    if vol_type == 'EGARCH':
        persistance = beta
    else:
        a = alpha if not np.isnan(alpha) else 0.0
        b = beta  if not np.isnan(beta)  else 0.0
        g = (gamma * 0.5) if not np.isnan(gamma) else 0.0
        persistance = a + b + g

    # ── Résultats VaR/TVaR ─────────────────────────────────────────────────
    def _var(niveau_str, methode):
        try:
            return float(df_var.loc[niveau_str, methode])
        except Exception:
            return float('nan')

    niv_90  = '90%'
    niv_95  = '95%'
    niv_99  = '99%'

    var_histo_95  = _var(niv_95, 'VaR Historique')
    var_norm_95   = _var(niv_95, 'VaR Normale')
    var_stud_95   = _var(niv_95, 'VaR Student')
    var_cf_95     = _var(niv_95, 'VaR Cornish-Fisher')
    var_garch_95  = _var(niv_95, 'VaR GARCH')
    var_mc_95     = _var(niv_95, 'VaR Monte Carlo (1j)')

    var_histo_99  = _var(niv_99, 'VaR Historique')
    var_norm_99   = _var(niv_99, 'VaR Normale')
    var_stud_99   = _var(niv_99, 'VaR Student')
    var_cf_99     = _var(niv_99, 'VaR Cornish-Fisher')
    var_garch_99  = _var(niv_99, 'VaR GARCH')
    var_mc_99     = _var(niv_99, 'VaR Monte Carlo (1j)')

    tvar_histo_95 = _var(niv_95, 'TVaR Historique')
    tvar_norm_95  = _var(niv_95, 'TVaR Normale')
    tvar_stud_95  = _var(niv_95, 'TVaR Student')
    tvar_cf_95    = _var(niv_95, 'TVaR Cornish-Fisher')
    tvar_garch_95 = _var(niv_95, 'TVaR GARCH')
    tvar_mc_95    = _var(niv_95, 'TVaR Monte Carlo (1j)')

    tvar_garch_99 = _var(niv_99, 'TVaR GARCH')
    tvar_stud_99  = _var(niv_99, 'TVaR Student')

    # Ratio TVaR/VaR Student 95%
    ratio_tvar_var = (tvar_stud_95 / var_stud_95) if (var_stud_95 and not np.isnan(var_stud_95)) else float('nan')

    # ── Backtesting : meilleure méthode (p_CC max) ─────────────────────────
    try:
        bt_95 = df_bt[df_bt['Niveau'].astype(str).str.contains('95')]
        bt_95_ok = bt_95[bt_95['Verdict CC'] == 'OK']
        if not bt_95_ok.empty:
            idx_best = bt_95_ok['p_CC'].astype(float).idxmax()
            meilleure_methode = bt_95_ok.loc[idx_best, 'Methode']
            meilleur_pcc      = float(bt_95_ok.loc[idx_best, 'p_CC'])
        else:
            meilleure_methode = 'N/A'
            meilleur_pcc      = float('nan')
    except Exception:
        meilleure_methode = 'N/A'
        meilleur_pcc      = float('nan')

    # ── Tableau backtest texte (pour le prompt 11) ─────────────────────────
    try:
        cols = ['Methode', 'Niveau', 'N viol.', 'Taux obs.', 'p_UC', 'p_CC', 'Verdict CC']
        cols_ok = [c for c in cols if c in df_bt.columns]
        tableau_bt_txt = df_bt[cols_ok].to_string(index=False)
    except Exception:
        tableau_bt_txt = 'Voir tableau \\ref{tab:backtest}'

    # ── Assemblage du contexte ─────────────────────────────────────────────
    ctx = {
        # Métadonnées
        'TICKER':           ticker,
        'NOM_SOUS_JACENT':  _nom_sous_jacent(ticker),
        'DATE_DEBUT':       str(config['data']['start_date']),
        'DATE_FIN':         str(config['data']['end_date']),
        'FREQUENCE':        str(config['data'].get('frequency', 'daily')),
        'N_OBS':            str(len(rdt)),
        'N_OBS_PRIX':       str(len(prix_clean)),
        'T_TRAIN':          str(T_train),
        'T_OOS':            str(T_eff_dyn),
        'SPLIT_RATIO_PCT':  str(int(config['backtest']['split_ratio'] * 100)),
        'TEST_RATIO_PCT':   str(100 - int(config['backtest']['split_ratio'] * 100)),

        # Stats descriptives prix
        'MOY_PRIX':  _fmt2(float(prix_clean.mean())),
        'MED_PRIX':  _fmt2(float(prix_clean.median())),
        'STD_PRIX':  _fmt2(float(prix_clean.std())),
        'MIN_PRIX':  _fmt2(float(prix_clean.min())),
        'MAX_PRIX':  _fmt2(float(prix_clean.max())),

        # Stats descriptives rendements
        'MOY_RDT':   _fmt(float(rdt.mean()), 6),
        'STD_RDT':   _fmt2(float(rdt.std())),
        'SKEW_RDT':  _fmt(skew_val, 4),
        'KURT_RDT':  _fmt(kurt_val, 4),
        'MIN_RDT':   _fmt2(float(rdt.min())),
        'MAX_RDT':   _fmt2(float(rdt.max())),
        'JB_STAT':   _fmt2(jb_stat),
        'JB_PVAL':   _fmt(jb_p, 6),

        # Tests de stationnarité (placeholders — calculés dans stationarity.py)
        'ADF_LOG_P':  'voir tableau',
        'PP_LOG_P':   'voir tableau',
        'KPSS_LOG_P': 'voir tableau',
        'ADF_DIF_P':  'voir tableau',
        'PP_DIF_P':   'voir tableau',
        'KPSS_DIF_P': 'voir tableau',

        # ARIMA
        'ARIMA_P':        str(arima_result.get('p_opt', '?')),
        'ARIMA_D':        str(arima_result.get('d_opt', '?')),
        'ARIMA_Q':        str(arima_result.get('q_opt', '?')),
        'ARIMA_AIC':      _fmt2(arima_result.get('aic', float('nan'))),
        'ARIMA_AIC_NORM': _fmt(arima_result.get('aic', float('nan')), 6),
        'ARIMA_PMAX':     str(config['arima'].get('p_max', 4)),
        'ARIMA_QMAX':     str(config['arima'].get('q_max', 4)),
        'ARIMA_SEUIL':    str(int(config['arima'].get('seuil_significativite', 0.10) * 100)),
        'ARIMA_COEFS_SIG': 'Oui',
        # ── Interprétation pédagogique ───────────────────────────
        'ARIMA_INTERPRETATION':  str(arima_result.get('interpretation', 'incertain')),
        'ARIMA_MSG_PEDAGOGIQUE': str(arima_result.get('message_pedagogique', '')),
        'ARIMA_LB_PVAL':         _fmt(arima_result.get('lb_pval_rendements', float('nan')), 4),
        'ARIMA_MAX_MAG_COEF':    _fmt(arima_result.get('max_magnitude_coef', float('nan')), 4),

        # GARCH
        'GARCH_MODELE':       str(best.get('modele', best.get('vol', '?'))),
        'GARCH_P':            str(int(best.get('p', 1))),
        'GARCH_O':            str(int(best.get('o', 1))),
        'GARCH_Q':            str(int(best.get('q', 1))),
        'GARCH_DIST':         str(best.get('dist', '?')),
        'GARCH_AIC':          _fmt2(best.get('AIC', float('nan'))),
        'GARCH_AIC_NORM':     _fmt(best.get('AIC_norm', float('nan')), 6),
        'GARCH_OMEGA':        _fmt(omega, 6),
        'GARCH_ALPHA':        _fmt(alpha, 4),
        'GARCH_BETA':         _fmt(beta,  4),
        'GARCH_GAMMA':        _fmt(gamma, 4),
        'GARCH_NU':           _fmt2(nu),
        'PERSISTANCE':        _fmt(persistance, 4),
        'GARCH_SEUIL':        str(int(config['garch'].get('seuil_significativite', 0.05) * 100)),
        'GARCH_PMAX':         str(config['garch'].get('p_max', 3)),
        'GARCH_MODELES_TESTES': ', '.join(config['garch'].get('modeles', [])),
        'GARCH_DISTS_TESTEES':  ', '.join(config['garch'].get('distributions', [])),

        # VaR 95%
        'VAR_HISTO_95':   _fmt2(var_histo_95),
        'VAR_NORM_95':    _fmt2(var_norm_95),
        'VAR_STUDENT_95': _fmt2(var_stud_95),
        'VAR_CF_95':      _fmt2(var_cf_95),
        'VAR_GARCH_95':   _fmt2(var_garch_95),
        'VAR_MC_95':      _fmt2(var_mc_95),

        # VaR 99%
        'VAR_HISTO_99':   _fmt2(var_histo_99),
        'VAR_NORM_99':    _fmt2(var_norm_99),
        'VAR_STUDENT_99': _fmt2(var_stud_99),
        'VAR_CF_99':      _fmt2(var_cf_99),
        'VAR_GARCH_99':   _fmt2(var_garch_99),
        'VAR_MC_99':      _fmt2(var_mc_99),

        # TVaR 95%
        'TVAR_HISTO_95':   _fmt2(tvar_histo_95),
        'TVAR_NORM_95':    _fmt2(tvar_norm_95),
        'TVAR_STUDENT_95': _fmt2(tvar_stud_95),
        'TVAR_CF_95':      _fmt2(tvar_cf_95),
        'TVAR_GARCH_95':   _fmt2(tvar_garch_95),
        'TVAR_MC_95':      _fmt2(tvar_mc_95),

        # TVaR 99%
        'TVAR_GARCH_99':   _fmt2(tvar_garch_99),
        'TVAR_STUDENT_99': _fmt2(tvar_stud_99),

        # Ratios & synthèse
        'RATIO_TVAR_VAR':      _fmt(ratio_tvar_var, 2),
        'MEILLEURE_METHODE':   str(meilleure_methode),
        'MEILLEUR_PCC':        _fmt(meilleur_pcc, 4),
        'N_MC':                str(config['var'].get('n_simulations_mc', 50000)),
        'TABLEAU_BACKTEST_TEXTE': tableau_bt_txt,
    }

    # ── Verdict de spécification (Bug 2 : anti-fausse-réassurance) ────────────
    sv = spec_verdict or {}
    ctx.update({
        'SPEC_GRAVITE':               str(sv.get('gravite', '[DONNÉE MANQUANTE]')),
        'SPEC_LB_Z_PVAL':             _fmt(sv.get('lb_z_pval',    float('nan')), 3),
        'SPEC_LB_Z2_PVAL':            _fmt(sv.get('lb_z2_pval',   float('nan')), 3),
        'SPEC_ENGLE_NG_PVAL':         _fmt(sv.get('engle_ng_pval', float('nan')), 3),
        'SPEC_MESSAGE':               str(sv.get('message', '[DONNÉE MANQUANTE]')),
        'SPEC_ALL_CANDIDATES_FAILED': 'oui' if sv.get('all_candidates_failed_spec') else 'non',
    })

    # ── Demi-vie GARCH (Bug 3) ─────────────────────────────────────────────────
    ig = igarch_diag or {}
    hl = ig.get('half_life_periodes')
    freq = config.get('data', {}).get('frequency', 'daily')
    unite_map = {'daily': 'jours ouvrables', 'weekly': 'semaines', 'monthly': 'mois'}
    ctx['DEMI_VIE']       = _fmt(hl, 1) if hl is not None else '[DONNÉE MANQUANTE]'
    ctx['DEMI_VIE_UNITE'] = unite_map.get(freq, 'périodes')

    return ctx


def _substituer_variables(texte: str, contexte: dict) -> str:
    """Remplace {{VARIABLE}} par la valeur du contexte.

    Cle absente → '[DONNÉE MANQUANTE]' (defense structurelle anti-hallucination).
    """
    def _remplacer(m):
        cle = m.group(1)
        return contexte.get(cle, '[DONNÉE MANQUANTE]')
    return re.sub(r'\{\{(\w+)\}\}', _remplacer, texte)


def rediger_section(section_id: str, contexte: dict, provider,
                    prompts_dir: Path, max_tokens: int = 4000) -> tuple:
    """Rédige une section du mémoire via un LLMProvider.

    Parameters
    ----------
    section_id : str
        Identifiant de la section (ex: '08_garch').
    contexte : dict
        Valeurs numériques à substituer dans le prompt.
    provider : LLMProvider
        Instance du provider LLM (AnthropicProvider, OpenAICompatibleProvider…).
    prompts_dir : Path
        Répertoire contenant les fichiers .txt de prompts.
    max_tokens : int
        Nombre maximum de tokens de la réponse.

    Returns
    -------
    tuple[str, int, int]
        (contenu_latex, tokens_in, tokens_out)
    """
    prompt_path = prompts_dir / f'{section_id}.txt'
    if not prompt_path.exists():
        logger.warning(f'Prompt manquant : {prompt_path}')
        return (f'% PROMPT MANQUANT : {section_id}.txt\n', 0, 0)

    prompt_brut = prompt_path.read_text(encoding='utf-8')
    prompt = _substituer_variables(prompt_brut, contexte)

    try:
        resp = provider.generer(SYSTEM_PROMPT, prompt, max_tokens)
        logger.info('  [%s] %d tokens entrée, %d tokens sortie',
                    section_id, resp.tokens_in, resp.tokens_out)
        return resp.contenu, resp.tokens_in, resp.tokens_out
    except Exception as exc:
        logger.error('  [%s] Échec définitif : %s', section_id, exc)
        return '% ECHEC IA - A REDIGER MANUELLEMENT\n', 0, 0


def rediger_toutes_sections(contexte: dict, output_dir: Path,
                             provider, sections: list,
                             prompts_dir: Path,
                             max_tokens: int = 4000) -> dict:
    """Itère sur toutes les sections, appelle l'IA et injecte le contenu
    entre les balises % CONTENU_IA_DEBUT et % CONTENU_IA_FIN.

    Parameters
    ----------
    contexte : dict
        Contexte numérique du pipeline.
    output_dir : Path
        Dossier latex_doc/ de destination.
    provider : LLMProvider
        Instance du provider LLM.
    sections : list[str]
        Liste des identifiants de sections à rédiger.
    prompts_dir : Path
        Dossier prompts/.
    max_tokens : int
        Limite de tokens par section.

    Returns
    -------
    dict
        Statistiques {'sections': int, 'tokens_total': int}.
    """
    tokens_total   = 0
    sections_ok    = 0
    sections_echec = 0

    for section_id in sections:
        tex_path = output_dir / f'{section_id}.tex'
        if not tex_path.exists():
            logger.warning(f'Squelette manquant : {tex_path}')
            sections_echec += 1
            continue

        t0 = time.time()
        logger.info('  Redaction [%s]...', section_id)

        try:
            contenu, tok_in, tok_out = rediger_section(
                section_id, contexte, provider, prompts_dir, max_tokens
            )
        except Exception as e:
            contenu  = f'% ECHEC IA - A REDIGER MANUELLEMENT\n% Erreur : {e}\n'
            tok_in   = 0
            tok_out  = 0

        echec = (tok_out == 0 or contenu.startswith('% ECHEC IA')
                 or contenu.startswith('% PROMPT MANQUANT'))
        tokens_total += tok_in + tok_out

        # Injection entre les balises dans le squelette (même en cas d'échec
        # pour laisser un placeholder visible dans le .tex)
        texte_squelette = tex_path.read_text(encoding='utf-8')
        pattern_debut = '% CONTENU_IA_DEBUT'
        pattern_fin   = '% CONTENU_IA_FIN'

        if pattern_debut in texte_squelette and pattern_fin in texte_squelette:
            avant = texte_squelette[:texte_squelette.index(pattern_debut)
                                    + len(pattern_debut)]
            apres = texte_squelette[texte_squelette.index(pattern_fin):]
            nouveau = f'{avant}\n{contenu}\n{apres}'
        else:
            nouveau = texte_squelette + '\n' + contenu

        tex_path.write_text(nouveau, encoding='utf-8')
        elapsed = time.time() - t0

        if echec:
            logger.warning('  ECHEC [%s] (%.1fs, 0 token) — placeholder injecte',
                           section_id, elapsed)
            sections_echec += 1
        else:
            logger.info('  OK [%s] (%.1fs, %d+%d tokens)', section_id, elapsed,
                        tok_in, tok_out)
            _scanner_hallucinations(section_id, contenu, contexte)
            sections_ok += 1

    return {
        'sections':       sections_ok,
        'sections_echec': sections_echec,
        'tokens_total':   tokens_total,
    }


def _scanner_hallucinations(section_id: str, texte: str, contexte: dict) -> None:
    """Détecte les nombres dans le texte généré absents du contexte — trace seulement.

    Faux positifs possibles (décimales reformatées, virgule vs point) :
    le scan est un filet de sécurité, pas un blocage.
    """
    # Valeurs numériques du contexte (normalisées : virgule → point)
    ctx_nums: set[str] = set()
    for v in contexte.values():
        s = str(v).replace(',', '.')
        ctx_nums.update(re.findall(r'\d+(?:\.\d+)?', s))

    # Nombres dans le texte généré (virgule → point pour comparaison)
    texte_norm = texte.replace(',', '.')
    text_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', texte_norm))

    # Exclure : années (1800-2199), entiers ≤ 100 sans décimale (niveaux de
    # confiance %, comptages, numéros de section — trop courants pour être utiles)
    def _est_annee_ou_num_section(n: str) -> bool:
        try:
            v = float(n)
            if 1800 <= v <= 2199:
                return True
            if v == int(v) and 0 < v <= 100:
                return True
            return False
        except ValueError:
            return False

    suspects = {n for n in text_nums
                if n not in ctx_nums and not _est_annee_ou_num_section(n)}
    for n in sorted(suspects):
        logger.warning('[IA-HALLUCINATION] section %s : nombre %r absent du contexte',
                       section_id, n)
