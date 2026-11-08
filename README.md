# LOGISMART — Predicción de retrasos en entregas de última milla

Proyecto académico (Gestión de Proyectos de Datos, SCY1102 · EV3) organizado con CRISP-DM. Genera un dataset sintético de operaciones de reparto, lo prepara, entrena y evalúa modelos de predicción de retraso, audita la equidad territorial, ejecuta inferencia por lotes con respaldo local y entrega un **dashboard simple** que carga archivos JSON para mostrar las alertas y el seguimiento del proyecto.

Todo el trabajo usa **datos sintéticos**: no hay conexión a datos productivos, PostgreSQL ni AWS.

## Entregable funcional: dashboard

`dashboard/index.html` es una página autocontenida (sin servidor ni librerías externas). Se abre con doble clic o desde GitHub Pages.

- **Alertas:** botón «Cargar archivo JSON» con `output/resultados/visualizacion/h6_t2_indicadores.json` (lo genera `python main.py --proceso visualizar`). Actualiza indicadores, distribución por prioridad y escenarios de umbral. Un JSON inválido se rechaza con un mensaje y no cambia la pantalla.
- **Seguimiento del backlog:** el mismo botón acepta `seguimiento.json` (formato con lista `tareas`). Muestra tareas completadas / en proceso, avance por fase CRISP-DM y la tabla de las 24 tareas.
- Ejemplos para probar la carga en `dashboard/ejemplos/` (el escenario 02 es ficticio, solo demostración).
- Lectura local en el navegador: ningún archivo se envía a Internet.

Limitación: las predicciones exportadas no incluyen ID de entrega, comuna ni fecha, por lo que el dashboard muestra alertas agregadas, no una consola de despacho.

## Seguimiento de tareas y backlog

El backlog vive en el tablero de GitHub Projects (Planificación · Backlog LOGISMART). Para actualizar el dashboard:

```bash
gh project item-list 1 --owner carlos31255 --limit 100 --format json > seguimiento/backlog_github.json
python seguimiento/generar_seguimiento.py
```

El script genera `dashboard/ejemplos/03_seguimiento_backlog.json` y actualiza los datos incrustados en `dashboard/index.html`. Estado: «Completado» si el tablero lo marca así; cualquier otro estado se muestra como «En proceso». Las fechas son el cronograma planificado (simulado, 16 semanas).

## Ejecución

Requisitos: Python 3.10+ y `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q                # 42 pruebas
```

Procesos disponibles con `python main.py --proceso <nombre>`:

| Proceso | Qué hace |
|---|---|
| `generar` (por defecto) | Genera 1.050.000 operaciones sintéticas (semilla 42) en `data/raw/` |
| `preparar` | Validación, seudonimización y variables derivadas → `data/processed/` |
| `entrenar` | Entrena Random Forest y XGBoost con partición temporal |
| `optimizar` | Búsqueda de hiperparámetros (H4-T3) |
| `explicar` | Explicabilidad SHAP |
| `sesgos`, `equidad` | Diagnóstico de sesgos y auditoría de equidad territorial |
| `evaluacion-final`, `evaluar` | Evaluación en test y auditoría de umbrales |
| `validacion-negocio` | Escenarios de umbral de alerta (0,10 / 0,15 / 0,20) |
| `batch` | Pruebas de inferencia por lotes (tiempo y memoria) |
| `automatizar` | Preparación + inferencia por lotes + respaldo ZIP verificable |
| `visualizar` | Genera los indicadores JSON/CSV/PNG de la última corrida |
| `completo` | Generar, preparar, entrenar, evaluar y explicar |

Prueba rápida: `python main.py --registros 1000 --salida data/prueba.csv`. Los CSV grandes se generan localmente y no se versionan.

## Resultados principales (datos sintéticos)

División temporal: entrenamiento ago 2025–jun 2026 (825.299 filas), validación jul 2026 (74.766) y test ago–sep 2026 (149.935). Selección de modelo y umbral en validación; el test no se usó para elegir.

| Modelo (test) | Umbral | ROC-AUC | PR-AUC | Precisión | Recall |
|---|---|---|---|---|---|
| Random Forest | 0,10 | 0,7021 | 0,2980 | 25,91 % | 57,53 % |
| XGBoost (oficial) | 0,15 | 0,7074 | 0,3044 | 34,56 % | 30,95 % |

La discriminación es limitada. En la corrida H6-T1 se procesaron 1.050.000 operaciones y se emitieron 161.875 alertas (15,42 %) en 19,89 s. Las alertas son prioridades de revisión humana: no demuestran retrasos evitados ni la meta de reducir 20 % los retrasos.

## Organización

```text
main.py                  # Punto de entrada (--proceso)
src/                     # Generador, pipeline, modelado, explicabilidad, equidad, batch, automatización, visualización
tests/                   # Pruebas con pytest
notebooks/               # 01 EDA · 02 Modelado y evidencias H4–H6
dashboard/               # Dashboard HTML y ejemplos JSON
seguimiento/             # Export del backlog y generador de seguimiento.json
docs/                    # Documentación H6-T1 y H6-T2
output/                  # Resultados, gráficos, ejecuciones y respaldos
models/                  # Modelo oficial (.joblib)
```

CI en `.github/workflows/`: `tests.yml` (pytest) y `dashboard-pages.yml` (publica `dashboard/` en GitHub Pages).

## Procedencia y supuestos del dataset

Según el Caso 3, LogiSmart realiza ~75.000 entregas mensuales, tiene ~1.800.000 registros históricos y los retrasos subieron de ~8 % a ~15 % en seis meses. Se simulan **1.050.000 operaciones de reparto** entre agosto de 2025 y septiembre de 2026; la tendencia del caso se reproduce en abril–septiembre de 2026 y los meses previos son una línea base sintética (supuesto del generador). Una fila es una operación de reparto, que puede incluir varios paquetes y paradas. Comunas, conductores, distancias, faltantes y demás distribuciones son supuestos del equipo, no estadísticas medidas de LOGISMART.

## Preparación de datos

`src/pipeline_datos.py` no modifica `data/raw/`.

- **Seudonimización:** `id_conductor` → código `COND-xxxx` sin tabla de correspondencia. Es una medida básica de minimización, no garantiza anonimato irreversible.
- **Limpieza:** 8 reglas de validación (0 incumplimientos); no se eliminan filas. Los nulos de experiencia del conductor (1,51 %) y peso (1,01 %) se imputan por mediana ajustada solo con entrenamiento, para evitar fuga de información.
- **Minimización:** se excluyen identificadores y variables posteriores al despacho (`retraso_salida_min`, `duracion_real_min`, `incidencias`, `estado_final`).
- **Balanceo:** no se rebalanceó el modelo oficial; el rendimiento por comuna se audita en la equidad territorial.

## Limitaciones

- Datos sintéticos: los resultados no son evidencia de desempeño real.
- Sin integración con PostgreSQL ni S3; el respaldo es un ZIP local, no una copia externa.
- Las probabilidades no están calibradas como riesgo real y no hay dotación confirmada para atender las alertas.
- El modelo no debe usarse como sanción automática ni evaluación laboral de conductores.

Más detalle en `docs/H6_T1_AUTOMATIZACION_Y_RESPALDOS.md`, `docs/H6_T1_RESULTADOS_EJECUCION.md` y `docs/H6_T2_VISUALIZACION.md`.
