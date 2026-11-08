# LOGISMART — H6-T1: pipeline local y respaldos verificables

## Propósito y alcance

Implementar **una demostración local** que ejecute en una sola orden las etapas ya desarrolladas de preparación de datos (H3-T4), inferencia en bloques (H5-T2), persistencia de predicciones y respaldo verificable. Se **reutilizan** `src/pipeline_datos.py` y `src/batch.py`; no se duplican transformaciones ni se entrena un modelo nuevo.

La infraestructura **PostgreSQL / Amazon S3 no forma parte del alcance definido para la EV3**. La solución acordada utiliza preparación e inferencia locales, persistencia CSV/JSON y respaldo ZIP con verificación. No se ha configurado una ejecución nocturna automática ni un SLA productivo.

## Requisitos

- Ejecutar desde la raíz del repositorio con el ambiente de `requirements.txt` instalado.
- Existencia de `data/raw/logismart_entregas_simuladas.csv` (CSV sintético histórico, con la etiqueta `retraso`).
- Existencia del modelo oficial **propio y de confianza** en `models/modelo_logismart.joblib`.
- Existencia de `output/resultados/modelado/resultados_modelado.json` del entrenamiento completo, con modelo `XGBoost` y umbral oficial fijado en validación.
- Espacio en disco disponible para el CSV temporal, las predicciones y el respaldo ZIP; durante el ETL se carga el CSV en memoria.

## Ejecución

Primero ejecutar pruebas: `python -m pytest -q`.

**Prueba local con 20.000 registros** (no altera el modelo):

```powershell
python main.py --proceso automatizar --max-filas-automatizacion 20000
```

**Corrida local completa** sobre el CSV existente:

```powershell
python main.py --proceso automatizar
```

Opcionalmente: `--tam-lote-automatizacion 5000`, `--entrada-automatizacion ruta.csv`, `--carpeta-ejecuciones directorio` y `--carpeta-respaldos directorio`.

El programa retorna un error (exit code distinto de cero) si falta un archivo, la preparación falla, el artefacto no es el oficial o no se verifica el respaldo.

## Flujo y archivos

1. **Validación/carga:** verifica modelo oficial y lee el umbral de `resultados_modelado.json`. No selecciona hiperparámetros.
2. **Preparación:** invoca `preparar_datos()` sobre el CSV histórico y crea archivos intermedios dentro de un directorio temporal. Los datos intermedios se eliminan al terminar.
3. **Inferencia:** invoca `predecir_lotes()` con el pipeline oficial. Lee solo `FEATURES`, sin acceder a la etiqueta `retraso` durante la predicción.
4. **Registro:** persiste `registro.json`, con estado, fechas UTC, tiempos de carga/preparación/inferencia, número de filas, alertas, ruta y SHA-256 del CSV. Si hay error, registra tipo y mensaje.
5. **Respaldo:** genera un ZIP local con `predicciones.csv`, `registro.json` y `manifest.json` (SHA-256 + tamaño de ambos archivos). Comprueba CRC y SHA-256 después de crear el ZIP.

Cada ejecución genera un identificador único. Ejemplo de **estructura** (no son resultados ejecutados):

```text
output/
├── ejecuciones/
│   └── <identificador>/
│       ├── predicciones.csv
│       └── registro.json
└── respaldos/
    └── <identificador>.zip
```

El CSV de predicciones incluye `fila_csv` (posición original del CSV procesado, comenzando en 1), `probabilidad_retraso` y `alerta_retraso`. No contiene etiquetas reales ni identificadores de conductor. Si se cambia o reordena el CSV de entrada, la posición deja de corresponder a la corrida anterior: la trazabilidad depende de conservar la versión del origen.

### Recuperación y verificación

El ZIP no es una copia del dataset original ni del modelo; ambos deben conservarse por separado. Para verificar un ZIP desde Python en la raíz del repositorio:

```powershell
python -c "from src.automatizacion import verificar_respaldo; print(verificar_respaldo('output/respaldos/REEMPLAZA_IDENTIFICADOR.zip'))"
```

Un resultado `True` indica que el ZIP es legible y coincide con sus hashes internos; **no** prueba que existe una copia externa, que fue restaurado en otro servidor ni que puede recuperarse el modelo. Los archivos pueden recuperarse con la función *Extraer todo* del administrador ZIP del sistema o `Expand-Archive` en PowerShell. Deben almacenarse en una ubicación separada para protegerse ante fallas del equipo; eso no está automatizado aquí.

### Estrategia de respaldo propuesta (no implementada en producción)

- **Fuente para restauración:** conservar por separado el dataset original, la versión del modelo y `resultados_modelado.json`.
- **Resultados por corrida:** conservar `predicciones.csv`, `registro.json` y el ZIP con manifest para demostrar integridad.
- **Periodicidad:** una copia por ejecución; planificar frecuencia diaria/nocturna solo tras aprobación operacional.
- **Retención y fuera de sitio:** pendientes de decisión organizacional y autorización; este código guarda en disco local únicamente.
- **Verificación:** CRC + SHA-256 del ZIP al generarlo; se recomienda realizar pruebas de restauración periódicas en otro equipo antes de usarlo operativamente.

## Evidencias e interpretación

- Código en `src/automatizacion.py` y pruebas en `tests/test_automatizacion.py`.
- Registro generado en `output/ejecuciones/<identificador>/registro.json`.
- Predicciones generadas en `output/ejecuciones/<identificador>/predicciones.csv`.
- Respaldo local generado en `output/respaldos/<identificador>.zip`.
- Sección H6-T1 del notebook: **lee solamente** los archivos generados para mostrar la evidencia.

Los directorios de ejecución y respaldo incluyen `.gitignore` para evitar subir accidentalmente archivos grandes al repositorio. Para adjuntar evidencia académica puede elegirse una bitácora JSON pequeña, o registrar capturas de la ejecución, sin incorporar datasets ni ZIP masivos.

## Límites pendientes de resolver

1. El ETL existente exige la columna `retraso`: funciona como demostración histórica, pero **no procesa todavía entradas futuras sin etiqueta**. Adaptarlo para datos operacionales reales sería una tarea posterior.
2. Los ZIP son locales; no constituyen respaldo externo ni replicación fuera del equipo. Esto es una limitación operativa, aunque S3 y PostgreSQL no son requisitos de esta EV3.
3. No existe planificador nocturno ni registro de reintentos automáticos; la ejecución es manual por CLI.
4. No hay medición de tiempos/capacidad en infraestructura productiva ni SLA garantizado.
5. Las predicciones son apoyo para revisión humana, sin decisiones laborales automáticas.
6. La ausencia de errores durante la ejecución no certifica disponibilidad, continuidad ni trazabilidad de datos reales.

**Estado H6-T1:** flujo local de preparación, inferencia, persistencia y respaldo demostrado con la ejecución completa de 1.050.000 operaciones sintéticas. PostgreSQL y S3 están fuera del alcance acordado para la EV3; no se afirma implantación productiva.
