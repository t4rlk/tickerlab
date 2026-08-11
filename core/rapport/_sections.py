# -*- coding: utf-8 -*-
"""Façade de rétrocompatibilité — le code vit dans le sous-package sections/.

Ne pas modifier ce fichier : modifier le sous-module correspondant.
"""
from tickerlab.core.rapport.sections.stationarite import (
    section_1, section_2, section_3,
)
from tickerlab.core.rapport.sections._common import (
    _encadre_pedagogique,
)
from tickerlab.core.rapport.sections.arima_garch import (
    section_4, section_5, section_6,
    _encadre_igarch,
)
from tickerlab.core.rapport.sections.var_backtest import (
    section_7, section_8, section_9,
)
from tickerlab.core.rapport.sections.stress_synthese import (
    section_10, section_11,
)
from tickerlab.core.rapport.sections.annexes import (
    section_annexes, section_frtb_resume,
)

__all__ = [
    'section_1',  'section_2',  'section_3',
    'section_4',  'section_5',  'section_6',
    'section_7',  'section_8',  'section_9',
    'section_10', 'section_11',
    'section_annexes', 'section_frtb_resume',
    '_encadre_pedagogique', '_encadre_igarch',
]
