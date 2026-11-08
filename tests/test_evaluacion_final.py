from src.evaluacion import reporte_errores


def test_reporte_errores_cuadra():
    met = {"matriz_confusion": [[7, 2], [3, 5]]}
    e = reporte_errores(met)
    assert e["retrasos_reales"] == 8
    assert e["alertas_emitidas"] == 7
    assert e["retrasos_detectados"] == 5
    assert e["retrasos_no_detectados"] == 3
    assert e["falsos_positivos"] == 2


def test_reporte_errores_sin_falsos_positivos():
    e = reporte_errores({"matriz_confusion": [[10, 0], [2, 8]]})
    assert e["falsos_positivos"] == 0
    assert e["falsos_negativos"] == 2
