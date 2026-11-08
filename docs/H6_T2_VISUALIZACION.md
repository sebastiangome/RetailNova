# LOGISMART — H6-T2: visualización de alertas e indicadores

## Alcance del entregable

Se implementó un **prototipo local de visualización** mediante HTML autocontenido, generado desde Python. No es un dashboard conectado en tiempo real.

El entregable funcional principal es el **dashboard simple** en `dashboard/index.html`, que permite cargar un JSON para mostrar las alertas y el seguimiento del backlog (ver `dashboard/ejemplos/`). Este panel estático complementa al dashboard con los indicadores de la corrida H6-T1.

No reentrena modelos, no ajusta el umbral oficial ni reutiliza la etiqueta `retraso` en la visualización. Consume únicamente `registro.json` y `predicciones.csv` de una corrida exitosa de H6-T1.

## Cómo ejecutarlo

Desde la raíz del proyecto, con las dependencias de `requirements.txt` instaladas:

```powershell
python -m pytest tests/test_visualizacion.py -q
python main.py --proceso visualizar
```

Por defecto, se utiliza la corrida H6-T1 completada más reciente dentro de `output/ejecuciones/`. Para elegir una ejecución específica:

```powershell
python main.py --proceso visualizar --registro-visualizacion "output/ejecuciones/IDENTIFICADOR/registro.json"
```

Si no hay corridas completadas, ejecuta primero `python main.py --proceso automatizar` (utiliza datos sintéticos; no hace falta reentrenar).

Para abrir el panel local:

```powershell
Start-Process "output/visualizacion/panel_h6_t2.html"
```

## Archivos generados

```text
output/
├── resultados/visualizacion/
│   ├── h6_t2_indicadores.json
│   ├── h6_t2_niveles_riesgo.csv
│   ├── h6_t2_escenarios_umbrales.csv
│   └── h6_t2_histograma.csv
├── graficos/visualizacion/
│   ├── h6_t2_niveles_riesgo.png
│   ├── h6_t2_probabilidades.png
│   └── h6_t2_escenarios_umbrales.png
└── visualizacion/
    └── panel_h6_t2.html
```

El HTML contiene las imágenes incorporadas: se puede abrir sin servidor ni conexión de red. Las tablas CSV y JSON permiten reutilizar estos mismos cálculos posteriormente en otra herramienta.

## Indicadores y lectura de negocio

- **Operaciones:** filas inferidas desde la salida H6-T1, no necesariamente entregas reales.
- **Alertas:** casos con decisión de clasificación 1 al umbral oficial 0,15, no retrasos realmente observados.
- **Tasa de alertas:** proporción de operaciones marcadas para revisión, no error ni recall.
- **Niveles de riesgo:** bajo (p menor que 0,10); observación (desde 0,10 hasta menos de 0,15); prioritario (0,15 o más). Son *prioridades relativas*, no riesgos calibrados ni aprobaciones operacionales.
- **Escenarios 0,10 / 0,15 / 0,20:** volumen de casos hipotéticamente revisados sobre el MISMO CSV histórico. No se recalculan métricas de precisión/recall porque `predicciones.csv` no contiene etiquetas.

## Validación e integridad

El script comprueba estado completado en la bitácora, columnas requeridas, orden consecutivo, número de filas, probabilidades finitas entre 0 y 1, alertas binarias, concordancia con el umbral guardado, conteos totales y SHA-256 del archivo cuando está presente. Lee el CSV por bloques, sin cargar el millón de filas en una sola tabla. Se permite únicamente una posible ambigüedad de decisión de borde por redondeo a ocho decimales, que se cuenta explícitamente.

## Limitaciones

- Datos sintéticos históricos; el CSV incluye operaciones usadas en *train*, *validación* y *test*. No es evaluación independiente de precisión ni operación en vivo.
- El archivo de predicciones no incluye etiqueta real ni comuna/fecha; por ello no se muestran métricas por comuna, fechas, tasas reales de retraso ni falsos positivos por área. Esas métricas deben mantenerse en la auditoría H5-T3, **identificando que proceden de otro subconjunto**.
- No se acreditan un SLA, la capacidad diaria de atención, retrasos evitados ni impacto operacional del 20 %.
- El uso recomendado de alertas sigue siendo apoyo a decisiones humanas, sin sanciones automáticas a conductores.

## Resultado de la corrida de referencia (4 de noviembre de 2026)

- Corrida H6-T1: `20261104T170000050763Z_a51e87`.
- Operaciones analizadas: **1.050.000**.
- Alertas oficiales (0,15): **161.875** (15,42 %).
- Escenarios sobre el mismo histórico: umbral 0,10 → **329.021** alertas; umbral 0,20 → **78.166** alertas.

La diferencia en alertas representa una variación en la carga potencial de revisión, **no** una mejora demostrada del modelo ni una reducción de retrasos.
