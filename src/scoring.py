"""Scoring de toda la base de 12 meses y categorización de riesgo (RF2: 3 niveles)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.modelado import cargar_modelo


def categorizar_riesgo(prob: pd.Series, percentiles=cfg.PERCENTILES_RIESGO) -> pd.Series:
    """Niveles por percentil de probabilidad: Alto = 10 % superior, Medio = siguiente 24 %,
    Bajo = resto. Es una herramienta de priorización, no una predicción de cuántos se irán."""
    p_medio, p_alto = prob.quantile(list(percentiles))
    return pd.Series(np.where(prob >= p_alto, "Alto", np.where(prob >= p_medio, "Medio", "Bajo")),
                     index=prob.index)


def puntuar(df: pd.DataFrame, pipeline=None, salida: Path = cfg.CSV_SCORED) -> pd.DataFrame:
    """Proceso `puntuar`: agrega probabilidad y riesgo y exporta el CSV que usa el dashboard."""
    pipeline = pipeline or cargar_modelo()
    X = df[cfg.FEATURES_NUMERICAS + cfg.FEATURES_CATEGORICAS]
    out = df.copy()
    out["prob_abandono"] = pipeline.predict_proba(X)[:, 1]
    out["riesgo_abandono"] = categorizar_riesgo(out["prob_abandono"])
    out = out[cfg.COLUMNAS_EXPORTACION]
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(salida, index=False)
    print(out["riesgo_abandono"].value_counts().reindex(cfg.NIVELES_RIESGO).to_string())
    print(f"Exportado {salida}")
    return out
