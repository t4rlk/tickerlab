# -*- coding: utf-8 -*-
"""Contrat de `web.mapping.construire_config` : traduction payload front → config.

Couvre le mapping des champs pilotés par le configurateur (aucune logique
économétrique n'est touchée) et la validité Pydantic du dict produit.
"""
import pytest

from tickerlab.web.mapping import construire_config
from tickerlab.core.config_schema import valider_config_structure


def _payload(**over):
    base = {'ticker': 'BZ=F', 'from': '2015-01-01', 'to': '2024-01-01',
            'freq': 'daily', 'price': 'adjclose', 'outputs': []}
    base.update(over)
    return base


def test_sorties_ordre_dedup_preserve_ordre_clic():
    cfg = construire_config(
        _payload(outputs=['charts', 'breaks', 'charts', 'report', 'var', 'breaks']),
        'web_runs/j')
    # dédupliqué, ordre de clic préservé
    assert cfg['sorties_ordre'] == ['charts', 'breaks', 'report', 'var']


def test_sorties_ordre_vide_si_aucune_sortie():
    cfg = construire_config(_payload(outputs=[]), 'web_runs/j')
    assert cfg['sorties_ordre'] == []


def test_breaks_opt_in():
    assert construire_config(_payload(outputs=['breaks']), 'web_runs/j')[
        'structural_breaks']['enabled'] is True
    assert construire_config(_payload(outputs=[]), 'web_runs/j')[
        'structural_breaks']['enabled'] is False


def test_report_active_ia_et_mode_classique():
    cfg = construire_config(_payload(outputs=['report']), 'web_runs/j')
    assert cfg['ai']['enabled'] is True
    assert cfg['output']['pdf_unique'] is False        # mode LaTeX classique
    cfg2 = construire_config(_payload(outputs=[]), 'web_runs/j')
    assert cfg2['ai']['enabled'] is False
    assert cfg2['output']['pdf_unique'] is True         # PDF ReportLab consolidé


@pytest.mark.parametrize('price, attendu', [
    ('adjclose', True),
    ('close',    False),
])
def test_price_pilote_auto_adjust(price, attendu):
    cfg = construire_config(_payload(price=price), 'web_runs/j')
    assert cfg['data']['price'] == price
    assert cfg['data']['auto_adjust'] is attendu


def test_rolling_backtest_desactive_pour_web():
    cfg = construire_config(_payload(outputs=['backtest']), 'web_runs/j')
    assert cfg['rolling_backtest']['enabled'] is False


def test_champs_data_pilotes_par_le_front():
    cfg = construire_config(
        _payload(ticker=' EURUSD=X ', **{'from': '2010-06-01'}, to='2020-06-01',
                 freq='weekly'), 'web_runs/j')
    d = cfg['data']
    assert d['ticker'] == 'EURUSD=X'          # strip appliqué
    assert d['start_date'] == '2010-06-01'
    assert d['end_date'] == '2020-06-01'
    assert d['frequency'] == 'weekly'


def test_config_produite_valide_pydantic():
    """Le dict produit doit passer valider_config_structure sans ValidationError."""
    cfg = construire_config(
        _payload(price='close', outputs=['charts', 'breaks', 'report']),
        'web_runs/j')
    # ne lève pas
    valider_config_structure(cfg)


def test_defauts_scientifiques_preserves():
    """Non-régression : le mapping ne touche pas aux blocs économétriques."""
    from tickerlab.web.mapping import charger_config_base
    base = charger_config_base()
    cfg = construire_config(_payload(outputs=['charts']), 'web_runs/j')
    for bloc in ('garch', 'arima', 'var', 'fhs'):
        assert cfg.get(bloc) == base.get(bloc), f'bloc {bloc} modifié par le mapping'


# ── Branchement du champ `module` (v1) ────────────────────────────────────────

def test_module_var_leve_module_unavailable():
    from tickerlab.core.exceptions import ModuleUnavailableError
    with pytest.raises(ModuleUnavailableError):
        construire_config(_payload(module='var'), 'web_runs/j')


def test_module_inconnu_leve_module_unavailable():
    from tickerlab.core.exceptions import ModuleUnavailableError
    with pytest.raises(ModuleUnavailableError):
        construire_config(_payload(module='n_importe_quoi'), 'web_runs/j')


def test_module_ruptures_force_breaks():
    # ruptures active la détection même si 'breaks' n'est pas dans outputs.
    cfg = construire_config(_payload(module='ruptures', outputs=[]), 'web_runs/j')
    assert cfg['structural_breaks']['enabled'] is True


def test_module_univarie_defaut_sans_breaks():
    cfg = construire_config(_payload(module='univarie', outputs=[]), 'web_runs/j')
    assert cfg['structural_breaks']['enabled'] is False


def test_detail_frtb_opt_in():
    # Legacy (defaut) : pas de detail ; v1 (detail_frtb=True) : active.
    cfg_legacy = construire_config(_payload(), 'web_runs/j')
    assert cfg_legacy['backtest']['detail_frtb'] is False
    cfg_v1 = construire_config(_payload(), 'web_runs/j', detail_frtb=True)
    assert cfg_v1['backtest']['detail_frtb'] is True


def test_adaptateur_v1_vers_payload_interne():
    """L'adaptateur AnalyseRequest -> payload interne mappe les noms de champs."""
    from tickerlab.web.schemas_v1 import AnalyseRequest
    req = AnalyseRequest(symbol=' BZ=F ', date_from='2006-01-02',
                         date_to='2024-12-31', price='adj_close',
                         outputs=['var', 'garch'])
    p = req.to_internal_payload()
    assert p['ticker'] == 'BZ=F'            # symbol -> ticker (+ strip)
    assert p['from'] == '2006-01-02' and p['to'] == '2024-12-31'
    assert p['price'] == 'adjclose'         # adj_close -> adjclose
    assert p['module'] == 'univarie'        # defaut
    assert p['outputs'] == ['var', 'garch']
