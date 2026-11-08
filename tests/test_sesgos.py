import numpy as np
import pandas as pd
import pytest
from src.sesgos import calcular_diagnostico, intervalo_wilson


def ejemplo():
    return pd.DataFrame({'comuna_destino':['A','B','C'], 'operaciones':[200,80,40],
      'retrasos':[40,16,8], 'alertas':[60,20,8], 'tp':[20,8,2], 'fp':[40,12,6]})


def test_intervalos_y_metricas():
    t,r=calcular_diagnostico(ejemplo())
    a=t.set_index('comuna_destino').loc['A']
    assert a['recall']==pytest.approx(.5)
    assert a['fpr']==pytest.approx(.25)
    assert a['recall_ic95_inf'] < .5 < a['recall_ic95_sup']
    assert r['total_operaciones']==320


def test_conteos_inconsistentes():
    t=ejemplo(); t.loc[0,'alertas']=500
    with pytest.raises(ValueError): calcular_diagnostico(t)


def test_sin_retrasos_no_fabrica_recall():
    t=ejemplo();t.loc[2,['retrasos','tp','alertas','fp']]=[0,0,0,0]
    t2,_=calcular_diagnostico(t)
    assert np.isnan(t2.set_index('comuna_destino').loc['C','recall'])
    assert np.isnan(intervalo_wilson(0,0)[0])
