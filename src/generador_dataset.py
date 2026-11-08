"""Generación de la base sintética de RetailNova y recorte a los últimos 12 meses.

Los datos son 100 % sintéticos (RNF1/RNF5: sin información personal real). El
generador usa una única semilla y un único `Generator` de NumPy; el estado del
generador se guarda al terminar para que la etapa de limpieza continúe la misma
secuencia aleatoria y el resultado sea idéntico en cualquier máquina.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg


def probabilidad_abandono(frecuencia, reclamos, devoluciones, marketing, promociones,
                          tienda_fisica, ruido):
    """Modelo logístico que genera el comportamiento de abandono simulado."""
    z = (-1.3
         - 0.9 * (frecuencia - frecuencia.mean())
         + 0.6 * reclamos
         + 0.25 * devoluciones
         - 0.15 * marketing
         - 0.4 * promociones
         + 0.3 * tienda_fisica
         + ruido)
    return 1 / (1 + np.exp(-z))


def generar_base_historica(n: int = cfg.N_HISTORICO, semilla: int = cfg.SEMILLA):
    """Base histórica de `n` clientes (3 años). Devuelve (DataFrame, Generator)."""
    if n <= 0:
        raise ValueError("n debe ser mayor que cero")
    rng = np.random.default_rng(semilla)

    # Geografía, demografía y antigüedad
    region = rng.choice(list(cfg.COMUNAS), size=n, p=cfg.PESOS_REGION)
    comuna = np.array([rng.choice(cfg.COMUNAS[r]) for r in region])
    edad = rng.normal(40, 12, n).clip(18, 80).round().astype(int)
    dias_registro = rng.integers(30, 36 * 30, n)
    canal = rng.choice(cfg.CANALES, size=n, p=[0.40, 0.35, 0.25])

    # Hábitos de compra
    meses_activo = dias_registro / 30
    frecuencia = rng.gamma(2.0, 0.6, n).round(2)                    # compras por mes
    cantidad_compras = np.maximum(1, rng.poisson(frecuencia * meses_activo * 0.5))
    monto_promedio = rng.lognormal(10.8, 0.6, n).round(-2)          # CLP
    monto_total = (monto_promedio * cantidad_compras).round(-2)

    categorias_lista = ["Hogar", "Tecnología", "Electrodomésticos"]
    n_cat = rng.integers(1, 4, n)
    categorias = [", ".join(sorted(rng.choice(categorias_lista, k, replace=False))) for k in n_cat]

    promociones = rng.binomial(1, 0.45, n)
    devoluciones = rng.poisson(0.03 * cantidad_compras)
    reclamos = rng.poisson(0.4, n)
    marketing = rng.poisson(3, n)

    # Variable objetivo calibrada a la tasa del caso (~24 %)
    prob = probabilidad_abandono(frecuencia, reclamos, devoluciones, marketing, promociones,
                                 (canal == "Tienda física"), rng.normal(0, 0.8, n))
    abandono = (prob >= np.quantile(prob, 1 - cfg.TASA_ABANDONO)).astype(int)

    # Coherencia temporal: abandono = más de 180 días sin comprar
    dias_ult = np.where(abandono == 1, rng.integers(181, 540, n), rng.integers(0, 180, n))
    dias_registro = np.maximum(dias_registro, dias_ult + rng.integers(0, 60, n))
    fecha_registro = cfg.FECHA_CORTE - pd.to_timedelta(dias_registro, unit="D")
    fecha_ultima_compra = cfg.FECHA_CORTE - pd.to_timedelta(dias_ult, unit="D")

    df = pd.DataFrame({
        "id_cliente": [f"CL{i:06d}" for i in range(1, n + 1)],
        "edad": edad.astype("float"),
        "comuna": comuna,
        "region": region,
        "fecha_registro": fecha_registro.date,
        "canal_compra": canal,
        "cantidad_compras": cantidad_compras,
        "frecuencia_compra": frecuencia,
        "monto_promedio_compra": monto_promedio,
        "monto_total_acumulado": monto_total,
        "categorias_productos": categorias,
        "uso_promociones": promociones,
        "devoluciones": devoluciones,
        "reclamos": reclamos,
        "interacciones_marketing": marketing,
        "fecha_ultima_compra": fecha_ultima_compra.date,
        "abandono": abandono,
    })

    # Datos faltantes simulados (el caso describe problemas de calidad)
    idx_demo = rng.choice(n, int(n * 0.08), replace=False)
    mitad = len(idx_demo) // 2
    df.loc[idx_demo[:mitad], "edad"] = np.nan
    df.loc[idx_demo[mitad:], "comuna"] = np.nan
    df.loc[rng.choice(n, int(n * 0.02), replace=False), "monto_promedio_compra"] = np.nan
    df.loc[rng.choice(n, int(n * 0.01), replace=False), "canal_compra"] = np.nan
    return df, rng


def recortar_12_meses(df_historico: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Muestra de los últimos 12 meses (por fecha de registro) con el abandono recalculado."""
    inicio = cfg.FECHA_CORTE - pd.DateOffset(months=cfg.MESES_MUESTRA)
    fecha_reg = pd.to_datetime(df_historico["fecha_registro"])
    df = df_historico[fecha_reg >= inicio].copy()

    prob = probabilidad_abandono(
        df["frecuencia_compra"], df["reclamos"], df["devoluciones"],
        df["interacciones_marketing"], df["uso_promociones"],
        (df["canal_compra"] == "Tienda física"), rng.normal(0, 0.8, len(df)))
    df["abandono"] = (prob >= np.quantile(prob, 1 - cfg.TASA_ABANDONO)).astype(int)

    dias_ult = np.where(df["abandono"] == 1,
                        rng.integers(181, 350, len(df)), rng.integers(0, 180, len(df)))
    df["fecha_ultima_compra"] = pd.Series(
        cfg.FECHA_CORTE - pd.to_timedelta(dias_ult, unit="D"), index=df.index).dt.date
    return df


def validar_base(df: pd.DataFrame) -> None:
    """Validaciones básicas de la base cruda (no exige ausencia de nulos)."""
    if df["id_cliente"].isna().any() or not df["id_cliente"].is_unique:
        raise ValueError("Identificadores de cliente nulos o duplicados")
    if not df["abandono"].isin([0, 1]).all():
        raise ValueError("La variable abandono debe ser 0/1")
    tasa = df["abandono"].mean()
    if not 0.20 <= tasa <= 0.28:
        raise ValueError(f"Tasa de abandono fuera de rango esperado: {tasa:.3f}")


def guardar_estado_rng(rng: np.random.Generator, ruta: Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(rng.bit_generator.state), encoding="utf-8")


def cargar_estado_rng(ruta: Path) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return rng


def generar(registros: int = cfg.N_HISTORICO, semilla: int = cfg.SEMILLA,
            carpeta: Path = cfg.DATA_RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proceso `generar`: escribe la base de 3 años, la de 12 meses y el estado del RNG."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"Generando {registros:,} clientes sintéticos (semilla {semilla})...")
    historica, rng = generar_base_historica(registros, semilla)
    validar_base(historica)
    muestra = recortar_12_meses(historica, rng)
    validar_base(muestra)

    historica.to_csv(carpeta / cfg.CSV_3_ANIOS.name, index=False)
    muestra.to_csv(carpeta / cfg.CSV_12M.name, index=False)
    guardar_estado_rng(rng, carpeta / cfg.ESTADO_RNG.name)

    print(f"Base histórica (3 años): {len(historica):,} filas · abandono {historica['abandono'].mean():.1%}")
    print(f"Muestra 12 meses:        {len(muestra):,} filas · abandono {muestra['abandono'].mean():.1%}")
    nulos = muestra.isna().sum()
    print("Nulos en la muestra:", nulos[nulos > 0].to_dict())
    return historica, muestra
