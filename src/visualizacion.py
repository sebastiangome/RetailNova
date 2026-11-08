"""H6-T2 — Visualización reproducible de alertas LOGISMART (prototipo estático).

Consume las predicciones y la bitácora H6-T1 sin consultar etiquetas ni el modelo.
Produce un panel HTML autónomo, gráficos PNG y tablas/JSON para exportar.
No debe confundirse con un dashboard conectado a operaciones reales.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_archivo(archivo: Path) -> str:
    """Calcula SHA-256 sin cargar el archivo completo en memoria."""
    h = hashlib.sha256()
    with archivo.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def encontrar_ultima_corrida(carpeta: str | Path) -> Path:
    """Busca la última bitácora H6-T1 completada (orden por identificador UTC)."""
    rutas = sorted(Path(carpeta).glob('*/registro.json'), reverse=True)
    for ruta in rutas:
        try:
            registro = json.loads(ruta.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if registro.get('estado') == 'completado' and registro.get('tarea') == 'H6-T1':
            return ruta
    raise FileNotFoundError(f'No hay corridas H6-T1 completadas en {carpeta}; ejecuta --proceso automatizar')


def medir_predicciones(
    archivo: str | Path, umbral: float, filas_esperadas: int | None = None,
    tam_lote: int = 50_000,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audita CSV de predicciones por lotes y agrega métricas sin leer etiquetas.

    Compara alertas con las probabilidades *publicadas* (8 decimales); si el
    redondeo deja un valor exactamente sobre el umbral, permite esa única
    ambigüedad de borde (no modifica alertas ni decisiones del modelo).
    """
    archivo = Path(archivo)
    if not archivo.is_file():
        raise FileNotFoundError(f'No existe el CSV de predicciones: {archivo}')
    if not np.isfinite(umbral) or not (0.10 < umbral < 1):
        raise ValueError('Se requiere umbral oficial mayor que 0.10 y menor que 1')
    if tam_lote <= 0:
        raise ValueError('tam_lote debe ser positivo')
    columnas = pd.read_csv(archivo, nrows=0).columns.tolist()
    if columnas != ['fila_csv', 'probabilidad_retraso', 'alerta_retraso']:
        raise ValueError(f'Columnas de predicción inesperadas: {columnas}')

    limites = np.linspace(0, 1, 21)
    hist = np.zeros(20, dtype=np.int64)
    escenarios = [0.10, float(umbral), 0.20]
    escenarios = list(dict.fromkeys(escenarios))
    conteos_escenario = {x: 0 for x in escenarios}
    niveles = {'Bajo': 0, 'Observación': 0, 'Prioritario': 0}
    total = alertas = 0
    min_p = 1.0
    max_p = 0.0
    suma_p = 0.0
    frontera_ambigua = 0
    for bloque in pd.read_csv(archivo, chunksize=tam_lote):
        p = pd.to_numeric(bloque['probabilidad_retraso'], errors='raise').to_numpy(dtype=float)
        a = pd.to_numeric(bloque['alerta_retraso'], errors='raise').to_numpy(dtype=float)
        i = pd.to_numeric(bloque['fila_csv'], errors='raise').to_numpy(dtype=float)
        n = len(bloque)
        if n == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Probabilidades inválidas o bloque vacío')
        if not np.isfinite(i).all() or not np.array_equal(i, np.arange(total + 1, total + n + 1)):
            raise ValueError('Filas CSV faltantes, duplicadas o desordenadas')
        if not np.isin(a, [0, 1]).all():
            raise ValueError('Alerta debe ser binaria (0 o 1)')
        esperadas = p >= umbral
        discrepancias = (esperadas != a.astype(bool))
        # El CSV de batch redondea a ocho decimales: una discrepancia se tolera
        # SOLO si el valor publicado dista <= 0.000000005 del umbral.
        borde = discrepancias & (np.abs(p - umbral) <= 5e-9)
        if np.any(discrepancias & ~borde):
            raise ValueError('Las alertas no corresponden al umbral de la bitácora')
        frontera_ambigua += int(borde.sum())
        alertas += int(a.sum())
        total += n
        min_p = min(min_p, float(p.min()))
        max_p = max(max_p, float(p.max()))
        suma_p += float(p.sum())
        niveles['Bajo'] += int((p < 0.10).sum())
        niveles['Observación'] += int(((p >= 0.10) & (p < umbral)).sum())
        niveles['Prioritario'] += int(a.sum())
        hist += np.histogram(p, bins=limites)[0]
        for e in escenarios:
            if abs(e - umbral) < 1e-12:
                conteos_escenario[e] += int(a.sum())
            else:
                conteos_escenario[e] += int((p >= e).sum())
    if not total:
        raise ValueError('El CSV no contiene predicciones')
    if filas_esperadas is not None and total != filas_esperadas:
        raise ValueError(f'Filas distintas de bitácora: {total} != {filas_esperadas}')
    if sum(niveles.values()) != total:
        # Puede suceder con una discrepancia de borde redondeada; alertas
        # prevalecen y las otras categorías deben ajustarse explícitamente.
        if frontera_ambigua:
            niveles['Observación'] = total - niveles['Bajo'] - niveles['Prioritario']
        else:
            raise ValueError('Los niveles de riesgo no suman el total')
    if niveles['Observación'] < 0:
        raise ValueError('Niveles incompatibles con las alertas')
    niveles_df = pd.DataFrame([
        {'nivel': nivel, 'operaciones': cantidad, 'porcentaje': cantidad / total * 100}
        for nivel, cantidad in niveles.items()
    ])
    escenarios_df = pd.DataFrame([
        {'escenario': ('Mayor cobertura' if e == 0.10 else
                       'Umbral oficial' if abs(e - umbral) < 1e-12 else 'Menos alertas'),
         'umbral': e, 'alertas': cantidad, 'porcentaje_operaciones': cantidad / total * 100}
        for e, cantidad in conteos_escenario.items()
    ]).sort_values('umbral')
    hist_df = pd.DataFrame({
        'limite_inferior': limites[:-1], 'limite_superior': limites[1:], 'operaciones': hist
    })
    indicadores = {
        'operaciones': total,
        'alertas': alertas,
        'tasa_alertas': alertas / total,
        'umbral_oficial': umbral,
        'probabilidad_media': suma_p / total,
        'probabilidad_minima': min_p,
        'probabilidad_maxima': max_p,
        'casos_ambiguos_por_redondeo_en_umbral': frontera_ambigua,
    }
    return indicadores, niveles_df, escenarios_df, hist_df


def graficar(niveles: pd.DataFrame, escenarios: pd.DataFrame, hist: pd.DataFrame,
             destino: Path, umbral: float) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destino.mkdir(parents=True, exist_ok=True)
    archivos: list[Path] = []
    fig, ax = plt.subplots(figsize=(9, 4.7))
    colores = ['#64748b', '#eab308', '#e76f51']
    barras = ax.barh(niveles['nivel'], niveles['operaciones'], color=colores)
    ax.invert_yaxis()
    ax.bar_label(barras, labels=[f'{v:,.0f} ({p:.1f}%)' for v, p in zip(niveles['operaciones'], niveles['porcentaje'])], padding=4)
    ax.set(xlabel='Operaciones', title='LOGISMART — prioridades de revisión (histórico sintético)')
    ax.set_xlim(0, float(niveles['operaciones'].max()) * 1.32)
    fig.tight_layout()
    p = destino / 'h6_t2_niveles_riesgo.png'; fig.savefig(p, dpi=160); plt.close(fig); archivos.append(p)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.bar(hist['limite_inferior'], hist['operaciones'], width=.045, align='edge', color='#3691b2')
    ax.axvline(.10, ls='--', color='#b08a20', label='Observación 0,10')
    ax.axvline(umbral, ls='--', color='#be5044', label=f'Prioritario {umbral:.2f}')
    ax.legend()
    ax.set(xlabel='Probabilidad estimada de retraso', ylabel='Operaciones',
           title='Distribución de puntuaciones — datos históricos sintéticos')
    fig.tight_layout()
    p = destino / 'h6_t2_probabilidades.png'; fig.savefig(p, dpi=160); plt.close(fig); archivos.append(p)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    barras = ax.bar(escenarios['escenario'], escenarios['alertas'], color=['#76939d', '#df8f45', '#8c9db3'][:len(escenarios)])
    ax.bar_label(barras, labels=[f'{x:,.0f}\n({x2:.1f}%)' for x, x2 in zip(escenarios['alertas'], escenarios['porcentaje_operaciones'])], padding=3)
    ax.set(ylabel='Entregas con alerta', title='Carga hipotética a distintos umbrales — mismo CSV histórico')
    ax.set_ylim(0, float(escenarios['alertas'].max()) * 1.22)
    fig.tight_layout()
    p = destino / 'h6_t2_escenarios_umbrales.png'; fig.savefig(p, dpi=160); plt.close(fig); archivos.append(p)
    return archivos


def crear_html(ind: dict, niveles: pd.DataFrame, escenarios: pd.DataFrame,
               imagenes: list[Path], corrida: str, destino: Path) -> None:
    """Panel local HTML autónomo. Los gráficos se incrustan sin enlaces externos."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    def imagen(archivo: Path, subtitulo: str) -> str:
        b64 = base64.b64encode(archivo.read_bytes()).decode('ascii')
        return (f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(subtitulo)}">'
                f'<figcaption>{html.escape(subtitulo)}</figcaption></figure>')
    def tabla(df: pd.DataFrame, fmt: dict[str,str]) -> str:
        etiquetas = {'nivel':'Nivel', 'operaciones':'Operaciones', 'porcentaje':'Porcentaje',
                     'escenario':'Escenario', 'umbral':'Umbral', 'alertas':'Alertas',
                     'porcentaje_operaciones':'% de operaciones'}
        head = ''.join(f'<th>{html.escape(etiquetas.get(c,c))}</th>' for c in df.columns)
        rows = []
        for _,row in df.iterrows():
            cells=[]
            for c in df.columns:
                val=row[c]
                s=(fmt[c].format(val) if c in fmt else str(val))
                cells.append('<td>'+html.escape(s)+'</td>')
            rows.append('<tr>'+''.join(cells)+'</tr>')
        return '<table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    fnum=lambda v:f'{v:,.0f}'.replace(',', '.')
    cards=[('Operaciones analizadas', fnum(ind['operaciones'])),
           ('Alertas emitidas', fnum(ind['alertas'])),
           ('Tasa de alertas', f"{ind['tasa_alertas']*100:.2f}%".replace('.',',')),
           ('Umbral oficial', f"{ind['umbral_oficial']:.2f}".replace('.',','))]
    cajas=''.join('<section class="card"><small>'+html.escape(t)+'</small><strong>'+v+'</strong></section>' for t,v in cards)
    figuras='\n'.join(imagen(p, p.stem.replace('_', ' ')) for p in imagenes)
    doc = f'''<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LOGISMART — Panel de alertas H6-T2</title><style>
:root {{color-scheme:light;}} body {{margin:0;font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;background:#f2f5f9;color:#182536;}}
main {{max-width:1160px;margin:auto;padding:28px;}} h1 {{margin-bottom:4px;}} h2 {{margin-top:35px;font-size:1.25rem;}}
.sub {{color:#526378;margin-top:0;}} .kpis {{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;margin:25px 0;}}
.card {{padding:18px;background:white;border:1px solid #dce4ed;border-radius:13px;box-shadow:0 2px 10px #18324d0d;}}
.card small {{display:block;color:#64748b;}} .card strong {{display:block;font-size:1.7rem;letter-spacing:-.03em;}}
.gallery {{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px;}}
figure {{margin:0;padding:12px;background:white;border-radius:12px;border:1px solid #e0e7f0;}}
figure img {{display:block;width:100%;}} figcaption {{font-size:12px;color:#617185;text-align:center;}}
table {{width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;}}
th,td {{padding:9px 12px;border-bottom:1px solid #e8edf4;text-align:left;}} th {{background:#eaf0f7;}}
.notice {{padding:14px 18px;border-left:3px solid #7d96aa;background:#e8eef4;}}
footer {{margin:34px 0;color:#54677e;font-size:13px;}}</style></head><body><main>
<h1>LOGISMART · Alertas preventivas</h1><p class="sub">H6-T2 · Demostración local de resultados sintéticos · Corrida <code>{html.escape(corrida)}</code></p>
<div class="kpis">{cajas}</div>
<p class="notice">Las alertas son sugerencias para <strong>revisión humana</strong>. No son retrasos confirmados, sanciones ni reducción demostrada de retrasos.</p>
<h2>Prioridades y escenarios de atención</h2><div class="gallery">{figuras}</div>
<h2>Distribución por nivel</h2>{tabla(niveles, {'operaciones':'{:,.0f}', 'porcentaje':'{:.2f}%'})}
<h2>Escenarios de umbral (solo volumen de alertas)</h2>{tabla(escenarios, {'umbral':'{:.2f}', 'alertas':'{:,.0f}', 'porcentaje_operaciones':'{:.2f}%'})}
<footer>Base: predicciones batch históricas de H6-T1 (incluye períodos de entrenamiento, validación y test). El conteo de alertas es descriptivo, no una evaluación independiente del modelo. No existen etiquetas en este archivo para calcular precision, recall o retrasos evitados. El umbral oficial no fue modificado.</footer>
</main></body></html>'''
    destino.write_text(doc, encoding='utf-8')


def generar_visualizacion(
    registro: str | Path | None = None,
    carpeta_ejecuciones: str | Path = 'output/ejecuciones',
    carpeta_resultados: str | Path = 'output/resultados/visualizacion',
    carpeta_graficos: str | Path = 'output/graficos/visualizacion',
    carpeta_panel: str | Path = 'output/visualizacion',
    tam_lote: int = 50_000,
) -> dict:
    """Utiliza el CSV y la bitácora de una ejecución H6-T1 ya completada."""
    ruta = Path(registro) if registro is not None else encontrar_ultima_corrida(carpeta_ejecuciones)
    if not ruta.is_file():
        raise FileNotFoundError(f'Falta registro: {ruta}')
    bitacora = json.loads(ruta.read_text(encoding='utf-8'))
    if bitacora.get('tarea') != 'H6-T1' or bitacora.get('estado') != 'completado':
        raise ValueError('La visualización requiere bitácora H6-T1 completada')
    base = bitacora.get('resultados', {})
    predicciones = ruta.parent / 'predicciones.csv'
    if not predicciones.is_file():
        raise FileNotFoundError(f'Falta {predicciones}; restaura primero la corrida H6-T1')
    hash_guardado = base.get('sha256_predicciones')
    if hash_guardado and sha256_archivo(predicciones) != hash_guardado:
        raise ValueError('SHA-256 de predicciones no coincide con la bitácora')
    umbral = float(base['umbral_oficial'])
    ind, niveles, escenarios, hist = medir_predicciones(
        predicciones, umbral=umbral, filas_esperadas=int(base['filas_procesadas']),
        tam_lote=tam_lote,
    )
    if ind['alertas'] != int(base['alertas_emitidas']):
        raise ValueError('Conteo de alertas diferente al registro H6-T1')
    resultados = Path(carpeta_resultados); resultados.mkdir(parents=True, exist_ok=True)
    graficos = Path(carpeta_graficos)
    panel = Path(carpeta_panel); panel.mkdir(parents=True, exist_ok=True)
    archivos = graficar(niveles, escenarios, hist, graficos, umbral)
    niveles.to_csv(resultados/'h6_t2_niveles_riesgo.csv', index=False)
    escenarios.to_csv(resultados/'h6_t2_escenarios_umbrales.csv', index=False)
    hist.to_csv(resultados/'h6_t2_histograma.csv', index=False)
    indicadores = {
        'tarea':'H6-T2', 'tipo':'prototipo_visual_local',
        'formato_definitivo':'dashboard_html_autocontenido',
        'corrida_h6_t1':bitacora['identificador_ejecucion'],
        'origen':'predicciones históricas sintéticas, incluyen train/val/test; inferencia sin etiquetas',
        'indicadores':ind,
        'niveles_riesgo':niveles.to_dict(orient='records'),
        'escenarios':escenarios.to_dict(orient='records'),
        'alcance':'Cantidad de alertas y prioridades, NO métricas de exactitud ni impacto operativo',
        'limitaciones':[
            'Datos sintéticos, no representan operación real en vivo',
            'Las predicciones incluyen períodos de entrenamiento; no interpretar como precisión fuera de muestra',
            'Las probabilidades no están certificadas como calibradas para riesgo real',
            'Las alertas no demuestran que se hayan evitado retrasos',
            'No existe dotación confirmada para atender los casos prioritarios',
            'El entregable funcional es el dashboard HTML (dashboard/index.html)',
        ],
    }
    (resultados/'h6_t2_indicadores.json').write_text(json.dumps(indicadores,ensure_ascii=False,indent=2),encoding='utf-8')
    crear_html(ind, niveles, escenarios, archivos, bitacora['identificador_ejecucion'], panel/'panel_h6_t2.html')
    print(f"H6-T2: {ind['operaciones']:,} operaciones | {ind['alertas']:,} alertas | {ind['tasa_alertas']:.2%}")
    print(f"Panel: {panel / 'panel_h6_t2.html'}")
    print(f'Indicadores: {resultados} | Gráficos: {graficos}')
    print('Solo visualización: modelo, umbral y predicciones no modificados.')
    return indicadores


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='LOGISMART H6-T2: panel visual de alertas')
    p.add_argument('--registro', default=None)
    a = p.parse_args()
    generar_visualizacion(registro=a.registro)
