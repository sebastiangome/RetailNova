"""Configuración central de RetailNova: constantes, rutas y definición de variables.

Todo parámetro que afecte resultados (semilla, tasa de abandono, umbrales del plan
EP3) vive aquí para que el pipeline sea reproducible y auditable.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# --- Reproducibilidad y población -------------------------------------------
SEMILLA = 42
N_HISTORICO = 185_000                       # clientes registrados según el caso (3 años)
FECHA_CORTE = pd.Timestamp("2026-09-30")
MESES_MUESTRA = 12                          # recorte de la base histórica
TASA_ABANDONO = 0.24                        # caso RetailNova: 24 % entre clientes frecuentes
DIAS_INACTIVIDAD_ABANDONO = 180             # abandono = sin compras en 6 meses

# --- Metas del plan de proyecto (EP3) ---------------------------------------
META_RECALL = 0.70                          # métrica de éxito del plan
RECALL_OOF_OBJETIVO = 0.75                  # 5 puntos de margen preventivo sobre la meta
LIMITE_BRECHA_REGIONAL_PCT = 10.0           # RNF6: diferencia de F1 por región < 10 %
DIFERENCIA_AUC_EMPATE = 0.01                # bajo este margen se prefiere el modelo simple
MODELO_SIMPLE = "Regresión Logística"

# --- Geografía simulada -----------------------------------------------------
COMUNAS = {
    "Metropolitana": ["Santiago", "Maipú", "Puente Alto", "Las Condes", "La Florida",
                      "Ñuñoa", "Providencia", "San Bernardo"],
    "Valparaíso": ["Valparaíso", "Viña del Mar", "Quilpué", "Villa Alemana", "San Antonio"],
    "Biobío": ["Concepción", "Talcahuano", "Los Ángeles", "Chiguayante", "San Pedro de la Paz"],
}
PESOS_REGION = [0.72, 0.16, 0.12]           # sobre-representa la Región Metropolitana
CANALES = ["Tienda física", "E-commerce", "Mixto"]

# --- Variables del modelo ---------------------------------------------------
# `recencia_dias` NO es predictora: define el abandono (regla de 180 días) y
# provocaría fuga de datos (AUC ~1,0). Se conserva solo como variable descriptiva.
FEATURES_NUMERICAS = [
    "edad", "cantidad_compras", "frecuencia_compra", "monto_promedio_compra",
    "monto_total_acumulado", "uso_promociones", "devoluciones", "reclamos",
    "interacciones_marketing", "antiguedad_dias", "tasa_reclamos", "tasa_devoluciones",
]
FEATURES_CATEGORICAS = ["canal_compra"]     # `region` queda fuera: solo se usa para auditar equidad
VARIABLE_FUGA = "recencia_dias"

# --- Riesgo (RF2: al menos 3 niveles) ---------------------------------------
NIVELES_RIESGO = ("Alto", "Medio", "Bajo")
PERCENTILES_RIESGO = (0.66, 0.90)           # Bajo < p66 <= Medio < p90 <= Alto

COLUMNAS_EXPORTACION = [
    "id_cliente", "region", "comuna", "canal_compra", "edad",
    "cantidad_compras", "frecuencia_compra", "monto_promedio_compra", "monto_total_acumulado",
    "recencia_dias", "antiguedad_dias", "reclamos", "devoluciones", "uso_promociones",
    "interacciones_marketing", "abandono", "prob_abandono", "riesgo_abandono",
]

# --- Rutas ------------------------------------------------------------------
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
SALIDA = RAIZ / "output"
RESULTADOS = SALIDA / "resultados"
GRAFICOS = SALIDA / "graficos"
DASHBOARD = RAIZ / "dashboard"
SEGUIMIENTO = RAIZ / "seguimiento"

CSV_3_ANIOS = DATA_RAW / "retailnova_clientes_3anios.csv"
CSV_12M = DATA_RAW / "retailnova_clientes_12m.csv"
ESTADO_RNG = DATA_RAW / "estado_rng.json"
CSV_PREPARADO = DATA_PROCESSED / "retailnova_clientes_12m_completo.csv"
CSV_SCORED = DATA_PROCESSED / "retailnova_clientes_12m_scored.csv"
MODELO_JOBLIB = MODELOS / "modelo_retailnova.joblib"
