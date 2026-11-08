"""H6-T2: pruebas de integridad y lectura de alertas; no entrena ni predice."""
import hashlib
import json

import pandas as pd
import pytest

from src.visualizacion import encontrar_ultima_corrida, generar_visualizacion, medir_predicciones


def crear_corrida(tmp_path, probabilidades=(.05, .12, .14, .18, .25)):
    carpeta = tmp_path / 'ejecuciones' / '20261104T170000050763Z_prueba'
    carpeta.mkdir(parents=True)
    ruta_csv = carpeta / 'predicciones.csv'
    alertas = [int(p >= .15) for p in probabilidades]
    pd.DataFrame({
        'fila_csv':range(1,len(probabilidades)+1),
        'probabilidad_retraso':probabilidades,
        'alerta_retraso':alertas,
    }).to_csv(ruta_csv,index=False,float_format='%.8f')
    bit = {
        'tarea':'H6-T1','estado':'completado',
        'identificador_ejecucion':carpeta.name,
        'resultados':{'filas_procesadas':len(probabilidades), 'alertas_emitidas':sum(alertas),
                      'umbral_oficial':.15,
                      'sha256_predicciones':hashlib.sha256(ruta_csv.read_bytes()).hexdigest()}
    }
    ruta=(carpeta/'registro.json')
    ruta.write_text(json.dumps(bit),encoding='utf-8')
    return ruta, ruta_csv


def test_lectura_inferencia_y_niveles_sin_etiquetas(tmp_path):
    _,csv=crear_corrida(tmp_path)
    ind, niveles,escenarios,hist=medir_predicciones(csv,.15,filas_esperadas=5,tam_lote=2)
    assert ind['operaciones']==5
    assert ind['alertas']==2
    assert ind['tasa_alertas']==pytest.approx(.4)
    assert niveles.set_index('nivel')['operaciones'].to_dict()=={'Bajo':1,'Observación':2,'Prioritario':2}
    assert escenarios['alertas'].to_list()==[4,2,1]
    assert hist['operaciones'].sum()==5


def test_generacion_panel_html_y_resultados(tmp_path):
    registro,_=crear_corrida(tmp_path)
    raiz=tmp_path/'salidas'
    r=generar_visualizacion(registro=registro,carpeta_resultados=raiz/'resultados',
                            carpeta_graficos=raiz/'graficos',carpeta_panel=raiz/'visualizacion',tam_lote=2)
    assert r['indicadores']['alertas']==2
    assert (raiz/'resultados'/'h6_t2_indicadores.json').exists()
    assert (raiz/'graficos'/'h6_t2_niveles_riesgo.png').exists()
    p=raiz/'visualizacion'/'panel_h6_t2.html'
    assert p.exists()
    html=p.read_text(encoding='utf-8')
    assert 'data:image/png;base64,' in html
    assert 'revisión humana' in html
    assert 'no una evaluación independiente' in html


def test_detectar_alteracion_y_filtrado_invalido(tmp_path):
    registro,csv=crear_corrida(tmp_path)
    csv.write_text(csv.read_text(encoding='utf-8').replace('0.18000000,1','0.18000000,0'),encoding='utf-8')
    with pytest.raises(ValueError,match='SHA-256'):
        generar_visualizacion(registro=registro,carpeta_resultados=tmp_path/'r',
                              carpeta_graficos=tmp_path/'g',carpeta_panel=tmp_path/'v')
    with pytest.raises(ValueError,match='no corresponden'):
        medir_predicciones(csv,.15,tam_lote=2)


def test_filas_y_registro_erroneos(tmp_path):
    registro,csv=crear_corrida(tmp_path)
    tabla=pd.read_csv(csv)
    tabla.loc[2,'fila_csv']=4
    tabla.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='Filas CSV'):
        medir_predicciones(csv,.15)
    registro.unlink()
    with pytest.raises(FileNotFoundError,match='No hay corridas'):
        encontrar_ultima_corrida(tmp_path/'ejecuciones')


def test_selecciona_ultima_corrida_completada(tmp_path):
    registro,_=crear_corrida(tmp_path)
    carpeta=tmp_path/'ejecuciones'/'20271104T170000050763Z_error'
    carpeta.mkdir()
    (carpeta/'registro.json').write_text('{"tarea":"H6-T1","estado":"error"}',encoding='utf-8')
    assert encontrar_ultima_corrida(tmp_path/'ejecuciones')==registro
