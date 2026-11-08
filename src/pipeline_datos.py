"""Pipeline simple de preparación de datos de LOGISMART (H2-T4 a H3-T4).

Uso desde la raíz del repositorio:
    python -m src.pipeline_datos --muestra 20000     # prueba rápida
    python -m src.pipeline_datos                     # dataset completo

Las transformaciones de este archivo son deterministas (no aprenden nada de los
datos). La imputación de nulos se aprende solo con el entrenamiento, usando
`construir_preprocesador()` en el notebook de modelado.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd

# --- Columnas ---------------------------------------------------------------
OBJETIVO = "retraso"
FECHA = "fecha"
IDENTIFICADORES = ["id_entrega", "id_conductor"]
POSTERIORES_AL_DESPACHO = ["retraso_salida_min", "duracion_real_min", "incidencias", "estado_final"]
PROHIBIDAS = IDENTIFICADORES + POSTERIORES_AL_DESPACHO  # nunca como predictoras

PREDICTORES = [
    "hora_programada", "comuna_origen", "comuna_destino", "distancia_km", "tipo_vehiculo",
    "experiencia_conductor_anios", "n_paquetes", "peso_kg", "volumen_m3",
    "duracion_estimada_min", "n_paradas", "dia_semana", "mes",
]
DERIVADAS = ["es_fin_de_semana", "es_hora_punta", "franja_horaria", "paquetes_por_parada"]
CATEGORICAS = ["comuna_origen", "comuna_destino", "tipo_vehiculo", "franja_horaria"]
NUMERICAS = [c for c in PREDICTORES + DERIVADAS if c not in CATEGORICAS]

HORAS_PUNTA = (8, 9, 17, 18, 19)
SAL = "logismart-seudonimo-v1"  # codificación simple; opcional: variable LOGISMART_CLAVE_SEUDONIMO


# --- H2-T4: seudonimización -------------------------------------------------
def seudonimizar_conductores(df: pd.DataFrame, clave: str | None = None) -> pd.DataFrame:
    """Reemplaza `id_conductor` por un código estable `COND-xxxx` (mismo conductor = mismo código).

    No guarda tabla de correspondencia. Es una codificación simple: no impide
    recalcular el código de un ID conocido.
    """
    clave_bytes = (clave or os.environ.get("LOGISMART_CLAVE_SEUDONIMO") or SAL).encode()
    codigos = {
        i: "COND-" + hmac.new(clave_bytes, str(i).encode(), hashlib.sha256).hexdigest()[:12]
        for i in df["id_conductor"].dropna().unique()
    }
    assert len(set(codigos.values())) == len(codigos), "Códigos repetidos entre conductores"
    salida = df.copy()
    salida["id_conductor"] = df["id_conductor"].map(codigos)
    return salida


# --- H3-T1: validación y limpieza -------------------------------------------
def validar_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta filas que incumplen cada regla básica. No modifica ni elimina filas."""
    reglas = {
        "id_entrega duplicado": df["id_entrega"].duplicated(),
        "fila completa duplicada": df.duplicated(),
        "hora_programada fuera de 0-23": ~df["hora_programada"].between(0, 23),
        "distancia_km <= 0": df["distancia_km"] <= 0,
        "n_paquetes < 1": df["n_paquetes"] < 1,
        "n_paradas < 1": df["n_paradas"] < 1,
        "peso_kg <= 0": df["peso_kg"] <= 0,  # los nulos no cuentan como inválidos
        "retraso no binario": ~df[OBJETIVO].isin([0, 1]),
    }
    return pd.DataFrame({"filas_afectadas": {k: int(v.sum()) for k, v in reglas.items()}})


def aplicar_limpieza(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza tipos. No elimina filas ni imputa (los nulos se conservan)."""
    resumen = validar_calidad(df)
    if resumen["filas_afectadas"].sum() > 0:
        raise ValueError(f"Datos con reglas incumplidas:\n{resumen}")
    salida = df.copy()
    salida[FECHA] = pd.to_datetime(salida[FECHA])
    return salida


# --- H3-T4: características -------------------------------------------------
def generar_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas calculables antes del despacho."""
    salida = df.copy()
    salida["es_fin_de_semana"] = (salida["dia_semana"] >= 5).astype(int)
    salida["es_hora_punta"] = salida["hora_programada"].isin(HORAS_PUNTA).astype(int)
    salida["franja_horaria"] = pd.cut(
        salida["hora_programada"], bins=[-1, 11, 16, 23], labels=["manana", "tarde", "cierre"]
    ).astype(str)
    salida["paquetes_por_parada"] = salida["n_paquetes"] / salida["n_paradas"]
    return salida


# --- H3-T2: minimización ----------------------------------------------------
def minimizar_para_modelado(df: pd.DataFrame) -> pd.DataFrame:
    """Deja solo predictores preventivos + fecha + retraso (sin identificadores)."""
    modelo = df[PREDICTORES + DERIVADAS + [FECHA, OBJETIVO]].copy()
    assert not set(PROHIBIDAS) & set(modelo.columns), "Hay columnas prohibidas en el modelado"
    return modelo


# --- Imputación (solo se ajusta con entrenamiento) ---------------------------
def construir_preprocesador():
    """Imputación por mediana + one-hot. Usar `fit` SOLO con train, luego `transform`."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])


# --- Orquestación -----------------------------------------------------------
def preparar_datos(
    entrada: str | Path = "data/raw/logismart_entregas_simuladas.csv",
    salida: str | Path = "data/processed",
    muestra: int | None = None,
):
    """Carga -> limpieza -> seudonimización -> características -> minimización -> guarda."""
    raw = pd.read_csv(entrada, parse_dates=[FECHA], nrows=muestra)
    preparado = generar_caracteristicas(seudonimizar_conductores(aplicar_limpieza(raw)))
    modelo = minimizar_para_modelado(preparado)
    assert len(raw) == len(preparado) == len(modelo), "Cambió la cantidad de registros"

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    preparado.to_csv(salida / "logismart_preparado.csv", index=False)
    modelo.to_csv(salida / "logismart_modelado.csv", index=False)

    print(f"Filas: {len(raw):,} | preparado: {preparado.shape[1]} cols | modelado: {modelo.shape[1]} cols")
    print("Nulos conservados (se imputan en el modelado):")
    print(modelo.isna().sum()[lambda s: s > 0].to_string())
    return preparado, modelo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preparación de datos LOGISMART")
    ap.add_argument("--entrada", default="data/raw/logismart_entregas_simuladas.csv")
    ap.add_argument("--salida", default="data/processed")
    ap.add_argument("--muestra", type=int, default=None, help="usar solo las primeras N filas")
    a = ap.parse_args()
    preparar_datos(a.entrada, a.salida, a.muestra)
