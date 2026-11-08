"""Preparación de datos: coherencia temporal, imputación, validación y variables RFM.

No modifica `data/raw/`; escribe en `data/processed/`. Continúa la secuencia del
generador aleatorio guardada por `generador_dataset.generar` para que el resultado
sea idéntico al del flujo completo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.generador_dataset import cargar_estado_rng


def diagnostico_nulos(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Nulos": df.isnull().sum(),
        "Porcentaje (%)": (df.isnull().mean() * 100).round(2),
    })


def corregir_coherencia_temporal(df: pd.DataFrame, rng: np.random.Generator) -> int:
    """Un cliente no puede comprar antes de registrarse: mueve el registro hacia atrás.

    Devuelve cuántos registros se corrigieron. No altera la definición de abandono.
    """
    df["fecha_registro"] = pd.to_datetime(df["fecha_registro"])
    df["fecha_ultima_compra"] = pd.to_datetime(df["fecha_ultima_compra"])
    idx = df[(df["fecha_ultima_compra"] - df["fecha_registro"]).dt.days < 0].index
    if len(idx) > 0:
        desfase = rng.integers(0, 60, size=len(idx))
        df.loc[idx, "fecha_registro"] = df.loc[idx, "fecha_ultima_compra"] - pd.to_timedelta(desfase, unit="D")
    return int(len(idx))


def imputar_nulos(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Imputación justificada: mediana (edad), moda (canal), mediana por canal (monto)
    y comuna aleatoria válida dentro de la misma región."""
    df["edad"] = df["edad"].fillna(df["edad"].median())
    df["canal_compra"] = df["canal_compra"].fillna(df["canal_compra"].mode()[0])
    mediana_canal = df.groupby("canal_compra")["monto_promedio_compra"].transform("median")
    df["monto_promedio_compra"] = df["monto_promedio_compra"].fillna(mediana_canal)
    for region in df["region"].unique():
        mascara = (df["region"] == region) & df["comuna"].isnull()
        if mascara.any():
            df.loc[mascara, "comuna"] = rng.choice(cfg.COMUNAS[region], size=int(mascara.sum()))


def validar_dataset(df: pd.DataFrame) -> None:
    """Validaciones de calidad; lanza ValueError con el motivo si alguna falla."""
    if df.isnull().sum().sum() != 0:
        raise ValueError("Aún existen valores nulos en el DataFrame")
    if not (pd.to_datetime(df["fecha_registro"]) <= pd.to_datetime(df["fecha_ultima_compra"])).all():
        raise ValueError("Hay fechas de registro posteriores a la última compra")
    for region, comunas in cfg.COMUNAS.items():
        invalidas = set(df.loc[df["region"] == region, "comuna"]) - set(comunas)
        if invalidas:
            raise ValueError(f"Comunas {sorted(invalidas)} no pertenecen a la región {region}")
    if not df["edad"].between(18, 80).all():
        raise ValueError("Edades fuera del rango permitido (18 a 80 años)")
    tasa = df["abandono"].mean()
    if not 0.20 <= tasa <= 0.28:
        raise ValueError(f"Tasa de abandono fuera del rango esperado: {tasa:.4f}")
    if not df["id_cliente"].is_unique:
        raise ValueError("Existen IDs de cliente duplicados")


def agregar_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Recencia, antigüedad y tasas por compra (RFM del plan, actividad 1.4.3)."""
    out = df.copy()
    ult = pd.to_datetime(out["fecha_ultima_compra"])
    reg = pd.to_datetime(out["fecha_registro"])
    compras = out["cantidad_compras"].clip(lower=1)
    out["recencia_dias"] = (cfg.FECHA_CORTE - ult).dt.days
    out["antiguedad_dias"] = (cfg.FECHA_CORTE - reg).dt.days
    out["tasa_reclamos"] = out["reclamos"] / compras
    out["tasa_devoluciones"] = out["devoluciones"] / compras
    return out


def preparar_datos(entrada: Path = cfg.CSV_12M, estado_rng: Path = cfg.ESTADO_RNG,
                   salida: Path = cfg.CSV_PREPARADO) -> pd.DataFrame:
    """Proceso `preparar`: limpieza + validación + RFM → data/processed/."""
    entrada, salida = Path(entrada), Path(salida)
    if not entrada.is_file():
        raise FileNotFoundError(f"No existe {entrada}; ejecuta primero --proceso generar")
    rng = cargar_estado_rng(estado_rng)
    df = pd.read_csv(entrada)

    print("Nulos iniciales:", diagnostico_nulos(df).query("Nulos > 0")["Nulos"].to_dict())
    corregidos = corregir_coherencia_temporal(df, rng)
    print(f"Inconsistencias temporales corregidas: {corregidos}")
    imputar_nulos(df, rng)
    validar_dataset(df)
    df = agregar_rfm(df)

    for col in ("fecha_registro", "fecha_ultima_compra"):
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"Dataset preparado: {len(df):,} filas · 0 nulos · {salida}")
    return df
