"""Chargement, préparation et validation des données financières."""
import yfinance as yf
import pandas as pd
import numpy as np

from .exceptions import DataError

__etape_version__ = '2'

# TODO DETTE_TECHNIQUE : ajouter un retry exponentiel (3 tentatives, back-off
# 1s/2s/4s) sur les erreurs réseau yfinance transitoires. Non implémenté
# volontairement pour ne pas masquer un problème de ticker derrière des
# tentatives silencieuses. Priorité P3 — phase 9+.


def telecharger_prix(ticker: str, start_date: str, end_date: str,
                     auto_adjust: bool = True) -> pd.DataFrame:
    """
    Télécharge les prix de clôture depuis Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Identifiant Yahoo Finance (ex: 'BZ=F', 'CL=F').
    start_date : str
        Date de début au format 'YYYY-MM-DD'.
    end_date : str
        Date de fin au format 'YYYY-MM-DD'.
    auto_adjust : bool, optional
        True (défaut) : clôture ajustée des dividendes et splits ('adjclose').
        False : clôture brute ('close'). L'ajustement est appliqué par yfinance
        sur la série journalière AVANT tout rééchantillonnage de fréquence.

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonne 'prix' et index DatetimeIndex.

    Raises
    ------
    DataError
        Si aucune donnée n'est disponible ou si le téléchargement échoue.
    """
    try:
        raw = yf.download(ticker, start=start_date, end=end_date,
                          progress=False, auto_adjust=auto_adjust)
    except Exception as e:
        raise DataError(
            f"Échec téléchargement {ticker} : {e}. "
            "Vérifier le ticker et la connexion réseau.",
            ticker=ticker,
            etape='download',
            contexte={'cause': str(e), 'start_date': start_date, 'end_date': end_date},
        ) from e

    if raw.empty:
        raise DataError(
            f"Aucune donnée pour {ticker} entre {start_date} et {end_date}. "
            "Vérifier le ticker et l'intervalle de dates.",
            ticker=ticker,
            etape='download',
            contexte={'start_date': start_date, 'end_date': end_date},
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    prix = raw[['Close']].copy()
    prix.columns = ['prix']
    prix.index = pd.to_datetime(prix.index)
    prix = prix.dropna(how='all')

    # Filtre les jours à prix nul ou négatif (ex : WTI le 20/04/2020 à -37 $/baril).
    # Ces observations sont réelles mais rendent les log-rendements non définis ;
    # on les élimine plutôt que de planter silencieusement plus loin.
    n_negatifs = int((prix['prix'] <= 0).sum())
    if n_negatifs > 0:
        import warnings
        warnings.warn(
            f"{ticker} : {n_negatifs} prix <= 0 supprimés (ex: prix négatif WTI 20/04/2020). "
            "Les rendements de ces journées sont retirés.",
            UserWarning, stacklevel=2,
        )
        prix = prix[prix['prix'] > 0]

    return prix


def calculer_rendements(prix: pd.DataFrame, freq: str = 'daily',
                        en_pourcentage: bool = True) -> pd.Series:
    """
    Calcule les log-rendements à la fréquence souhaitée.

    Parameters
    ----------
    prix : pd.DataFrame
        DataFrame avec colonne 'prix' (sortie de telecharger_prix).
    freq : str, optional
        Fréquence de resampling : 'daily', 'weekly', 'monthly', 'annual'.
        Défaut 'daily' (pas de resampling).
    en_pourcentage : bool, optional
        Multiplie les rendements par 100 si True (défaut True).

    Returns
    -------
    pd.Series
        Log-rendements avec index DatetimeIndex.
    """
    if freq == 'weekly':
        prix = prix.resample('W-FRI').last().dropna()
    elif freq == 'monthly':
        prix = prix.resample('ME').last().dropna()
    elif freq == 'annual':
        prix = prix.resample('YE').last().dropna()
    rdt = np.log(prix['prix'] / prix['prix'].shift(1)).dropna()
    return rdt * 100 if en_pourcentage else rdt


def valider_donnees(prix: pd.DataFrame, rendements: pd.Series,
                    config: dict) -> None:
    """Valide la qualité des données téléchargées avant tout calcul économétrique.

    Contrôles (par ordre de priorité) :
      a) Longueur minimale — n_rendements >= data.min_observations (défaut 250)
      b) Proportion de NaN — part_nan <= data.max_nan_ratio (défaut 0.10)
      c) Prix strictement positifs — (prix > 0).all() (log-rendements impossibles sinon)
      d) Série non constante — std(rendements) >= 1e-8 (GARCH dégénère sinon)

    Raises
    ------
    DataError
        Dès le premier contrôle qui échoue, avec etape='data_validation'
        et contexte documenté.
    """
    data_cfg = config.get('data', {})
    ticker   = data_cfg.get('ticker', '?')
    min_obs  = int(data_cfg.get('min_observations', 250))
    max_nan  = float(data_cfg.get('max_nan_ratio', 0.10))

    # a) Longueur minimale
    n = len(rendements.dropna())
    if n < min_obs:
        raise DataError(
            f"Série {ticker} : {n} rendements < minimum {min_obs}. "
            "Estimation GARCH non fiable. Élargir l'intervalle de dates.",
            ticker=ticker,
            etape='data_validation',
            contexte={'n_rendements': n, 'min_observations': min_obs},
        )

    # b) Proportion de NaN dans les prix
    n_total = len(prix)
    n_nan   = int(prix['prix'].isna().sum())
    if n_total > 0 and (n_nan / n_total) > max_nan:
        part_nan = n_nan / n_total
        raise DataError(
            f"Série {ticker} : {part_nan:.1%} de prix manquants "
            f"({n_nan}/{n_total}) > seuil {max_nan:.0%}. "
            "Vérifier la disponibilité des données sur la période.",
            ticker=ticker,
            etape='data_validation',
            contexte={'part_nan': round(part_nan, 4), 'max_nan_ratio': max_nan,
                      'n_nan': n_nan, 'n_total': n_total},
        )

    # c) Prix strictement positifs — déjà filtrés dans telecharger_prix,
    #    ce contrôle est gardé comme garde-fou pour les séries injectées directement.
    mauvais = prix['prix'][prix['prix'] <= 0].dropna()
    if not mauvais.empty:
        dates = mauvais.index.strftime('%Y-%m-%d').tolist()[:5]
        raise DataError(
            f"Série {ticker} : {len(mauvais)} prix <= 0 détectés "
            f"(log-rendements impossibles). Dates exemple : {dates}.",
            ticker=ticker,
            etape='data_validation',
            contexte={'n_prix_non_positifs': len(mauvais), 'dates_exemple': dates},
        )

    # d) Série non constante (std quasi nulle => ticker suspendu/gelé)
    std = float(rendements.dropna().std())
    if std < 1e-8:
        raise DataError(
            f"Série {ticker} : écart-type des rendements = {std:.2e} < 1e-8 "
            "(série quasi constante — ticker suspendu ou gelé ?). "
            "Estimation GARCH dégénérée.",
            ticker=ticker,
            etape='data_validation',
            contexte={'std_rendements': std},
        )


def ensure_series(x) -> pd.Series:
    """Garantit un pd.Series — compresse un DataFrame à une colonne si nécessaire."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def resampler_prix(prix: pd.DataFrame, freq: str = 'daily') -> pd.DataFrame:
    """Resample un DataFrame de prix à la fréquence cible (sans recalculer les rendements)."""
    if freq == 'weekly':
        return prix.resample('W-FRI').last().dropna()
    elif freq == 'monthly':
        return prix.resample('ME').last().dropna()
    elif freq == 'annual':
        return prix.resample('YE').last().dropna()
    return prix
