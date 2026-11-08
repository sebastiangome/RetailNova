"""Auditoría de equidad regional (RNF6): la diferencia de F1 entre cada región y el
F1 global no debe superar el 10 %. Una brecha pequeña no prueba ausencia de sesgo:
es la métrica que definió el plan, no la única posible."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from src import config as cfg


def auditar_regiones(y_true, pred, regiones, limite_pct: float = cfg.LIMITE_BRECHA_REGIONAL_PCT) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    regiones = np.asarray(regiones)
    if not (len(y_true) == len(pred) == len(regiones)):
        raise ValueError("y_true, pred y regiones deben tener el mismo largo")
    f1_global = f1_score(y_true, pred)
    if f1_global == 0:
        raise ValueError("F1 global igual a cero: no se puede calcular la brecha relativa")
    filas = []
    for region in sorted(set(regiones)):
        m = regiones == region
        f1_r = f1_score(y_true[m], pred[m], zero_division=0)
        brecha = abs(f1_r - f1_global) / f1_global * 100
        filas.append({
            "Región": region,
            "N clientes (test)": int(m.sum()),
            "Tasa abandono real": round(float(y_true[m].mean()), 3),
            "F1": round(float(f1_r), 3),
            "Recall": round(float(recall_score(y_true[m], pred[m], zero_division=0)), 3),
            "Diferencia vs. F1 general (%)": round(float(brecha), 1),
            "¿Cumple RNF6 (<10%)?": "Sí" if brecha < limite_pct else "No",
        })
    tabla = pd.DataFrame(filas)
    if tabla["N clientes (test)"].sum() != len(y_true):
        raise ValueError("La suma de clientes por región no coincide con el total")
    return tabla
