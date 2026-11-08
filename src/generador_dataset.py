"""Generación académica de operaciones de reparto sintéticas de LOGISMART.

Cada fila representa una operación de reparto que puede contener varios paquetes
y paradas; NO equivale necesariamente a un envío individual. La escala se
inspira en los volúmenes del caso, pero no replica directamente sus 1,8 millones
de envíos históricos. Distribuciones y relaciones son supuestos, no mediciones.
"""

import numpy as np
import pandas as pd

INTERCEPTO = -5.10


def generar_dataset(n: int = 1_050_000, semilla: int = 42) -> pd.DataFrame:
    """Devuelve operaciones simuladas sin escribir archivos ni alterar estado global.

    La semilla reproduce la simulación en un entorno de dependencias fijo.
    """
    if n <= 0:
        raise ValueError("La cantidad de registros debe ser positiva")
    rng = np.random.default_rng(semilla)
    N = n
    comunas = {  # comuna: (peso de volumen, zona, factor de congestión)
        'Santiago':(14,'RM',1.25),'Providencia':(8,'RM',1.15),'Las Condes':(10,'RM',1.20),'Ñuñoa':(7,'RM',1.05),
        'Maipú':(9,'RM',1.10),'La Florida':(8,'RM',1.10),'Puente Alto':(7,'RM',1.15),'San Bernardo':(5,'RM',1.00),
        'Quilicura':(5,'RM',1.00),'Pudahuel':(4,'RM',1.05),'Colina':(1.2,'RM',0.95),'Lampa':(0.8,'RM',0.95),
        'Valparaíso':(5,'V',1.15),'Viña del Mar':(5,'V',1.10),'Quilpué':(2,'V',1.00),'Villa Alemana':(1.5,'V',0.95),
        'Concón':(0.8,'V',0.95),'Casablanca':(0.4,'V',0.90),'Quintero':(0.3,'V',0.90),
    }
    nombres = list(comunas)
    pesos = np.array([comunas[c][0] for c in nombres]); pesos = pesos/pesos.sum()

    # Conductores: 600, cada uno con 2-3 comunas de preferencia
    n_cond = 600
    cond_exp = np.clip(rng.gamma(3.0, 1.1, n_cond), 0.2, 15).round(1)          # años de experiencia
    cond_comunas = [rng.choice(len(nombres), size=rng.integers(2,4), replace=False, p=pesos) for _ in range(n_cond)]

    destino_idx = rng.choice(len(nombres), size=N, p=pesos)
    cond_id = np.empty(N, dtype=int)
    por_comuna = {k: [j for j,l in enumerate(cond_comunas) if k in l] for k in range(len(nombres))}
    for k in range(len(nombres)):
        idx = np.where(destino_idx==k)[0]
        pool = por_comuna[k] if len(por_comuna[k])>0 else list(range(n_cond))
        cond_id[idx] = rng.choice(pool, size=len(idx))

    origen_idx = np.where(rng.random(N)<0.7,
                          np.where(rng.random(N)<0.5, nombres.index('Quilicura'), nombres.index('Pudahuel')),
                          rng.choice(len(nombres), size=N, p=pesos))

    # Histórico sintético de 14 meses: agosto de 2025 a septiembre de 2026.
    # Selección uniforme del mes para aproximar 75.000 entregas mensuales con N=1.050.000.
    inicios_mes = pd.date_range('2025-08-01', '2026-09-01', freq='MS')
    idx_mes = rng.integers(0, len(inicios_mes), size=N)
    dias_mes = inicios_mes.days_in_month.to_numpy()
    dia_aleatorio = rng.integers(0, dias_mes[idx_mes], size=N)
    fecha = pd.to_datetime(inicios_mes.to_numpy()[idx_mes] + dia_aleatorio.astype('timedelta64[D]'))
    hora_prog = rng.choice(np.arange(8,20), size=N, p=np.array([5,8,10,9,8,8,9,10,10,9,7,4])/97)
    vehiculo = rng.choice(['Moto','Furgón','Camioneta','Camión'], size=N, p=[0.22,0.38,0.30,0.10])
    distancia = np.clip(rng.gamma(2.4, 5.5, N), 1.5, 90).round(1)
    paquetes = np.clip(rng.poisson(14,N)+1, 1, 60)
    peso = np.clip(paquetes*rng.normal(2.1,0.6,N), 0.5, 400).round(1)
    volumen = np.clip(paquetes*rng.normal(0.018,0.006,N), 0.002, 4).round(3)
    paradas = np.clip((paquetes*rng.uniform(0.5,0.95,N)).round(), 1, 45).astype(int)
    dur_est = (distancia*np.where(vehiculo=='Moto',1.6,2.4) + paradas*np.where(vehiculo=='Moto',3.5,5.0)).round(0)

    df = pd.DataFrame({
        'id_entrega': np.arange(1, N+1),
        'fecha': fecha,
        'hora_programada': hora_prog,
        'comuna_origen': np.array(nombres)[origen_idx],
        'comuna_destino': np.array(nombres)[destino_idx],
        'distancia_km': distancia,
        'tipo_vehiculo': vehiculo,
        'id_conductor': cond_id,
        'experiencia_conductor_anios': cond_exp[cond_id],
        'n_paquetes': paquetes,
        'peso_kg': peso,
        'volumen_m3': volumen,
        'duracion_estimada_min': dur_est,
        'n_paradas': paradas,
    })
    df["dia_semana"] = df["fecha"].dt.day_of_week
    df['mes'] = df['fecha'].dt.month

    # --- Mecanismo generador del retraso (riesgo latente) ---
    hora_punta = ((df.hora_programada.between(8,9)) | (df.hora_programada.between(17,19))).astype(int)
    congestion = df['comuna_destino'].map({c: v[2] for c, v in comunas.items()}).to_numpy()
    mes_inv = df['mes'].isin([6,7]).astype(int)     # invierno: más lluvia (el clima NO está disponible como variable)
    hay_lluvia = (rng.random(N) < np.where(mes_inv==1, 0.18, 0.08)).astype(int)
    sal_retraso = np.clip(rng.exponential(4,N) + 8*hay_lluvia*rng.random(N) + 6*hora_punta*rng.random(N), 0, 90).round(0)

    # El incremento de retrasos se simula únicamente en abril-septiembre de 2026.
    # Los meses anteriores tienen una situación base equivalente a abril de 2026.
    meses_desde_abril = (df['fecha'].dt.year - 2026) * 12 + df['mes'] - 4
    tendencia = meses_desde_abril.clip(lower=0, upper=5) / 5 * 0.80
    z = (INTERCEPTO
         + 0.045*df.distancia_km
         + 0.055*df.n_paradas
         + 0.018*(df.n_paquetes-14)
         + 0.90*hora_punta
         + 2.6*(congestion-1.0)
         + 0.90*hay_lluvia
         + 0.095*sal_retraso
         - 0.11*df.experiencia_conductor_anios
         + df.tipo_vehiculo.map({'Moto':0.0,'Furgón':0.20,'Camioneta':0.15,'Camión':0.65}).values
         + 0.35*(df.dia_semana==4)
         + tendencia
         + rng.normal(0, 0.30, N))
    prob = 1/(1+np.exp(-z))
    retraso = (rng.random(N) < prob).astype(int)

    # VARIABLES NO DISPONIBLES ANTES DEL DESPACHO (fuga de información):
    # retraso_salida_min, duracion_real_min, incidencias y estado_final.
    # "retraso" es la variable objetivo. Excluir también id_entrega e
    # id_conductor del modelado inicial para evitar memorizar identificadores.
    # Mantener estas columnas en el CSV para análisis histórico, pero nunca
    # seleccionarlas automáticamente como predictores de riesgo preventivo.
    extra = np.where(retraso==1, rng.uniform(15,90,N), rng.normal(-3,6,N))
    df['retraso_salida_min'] = sal_retraso                         # minutos de atraso en la salida real (hora efectiva - programada)
    df['duracion_real_min'] = np.clip(df.duracion_estimada_min + extra, 10, None).round(0)
    df['incidencias'] = np.where(retraso==1, rng.choice([0,1,2,3],N,p=[.45,.30,.17,.08]), rng.choice([0,1,2],N,p=[.86,.11,.03]))
    df['retraso'] = retraso
    df['estado_final'] = np.where(retraso==1,'Entregada con retraso','Entregada a tiempo')

    # Faltantes artificiales (no derivados de registros reales)
    mask = rng.random(N) < 0.015
    df.loc[mask, 'experiencia_conductor_anios'] = np.nan
    mask = rng.random(N) < 0.01
    df.loc[mask, 'peso_kg'] = np.nan
    return df
