"""H4-T4: diagnóstico descriptivo de diferencias territoriales a partir de auditoría ya existente.
No reentrena, no consulta el modelo y no ajusta el umbral. No sustituye H5-T3.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def intervalo_wilson(exitos: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo descriptivo binomial (aprox. 95 %), no prueba de equidad."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = exitos / total
    d = 1 + z*z/total
    centro = (p + z*z/(2*total)) / d
    semiancho = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return float(max(0,centro-semiancho)),float(min(1,centro+semiancho))


def calcular_diagnostico(comunas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columnas = {'comuna_destino','operaciones','retrasos','alertas','tp','fp'}
    faltan=columnas-set(comunas.columns)
    if faltan: raise ValueError(f'Columnas faltantes: {sorted(faltan)}')
    d=comunas.copy()
    for col in ['operaciones','retrasos','alertas','tp','fp']:
        d[col]=pd.to_numeric(d[col],errors='raise')
        if d[col].isna().any() or (d[col]<0).any() or (d[col]%1!=0).any():
            raise ValueError(f'Conteos inválidos: {col}')
    if d['comuna_destino'].isna().any() or d['comuna_destino'].duplicated().any() or not len(d):
        raise ValueError('Comunas nulas, duplicadas o tabla vacía')
    if ((d['operaciones']<=0)|(d['retrasos']>d['operaciones'])|(d['alertas']>d['operaciones'])|
        (d['tp']>d[['retrasos','alertas']].min(axis=1))|(d['fp']>d['alertas'])|
        (d['tp']+d['fp']!=d['alertas'])).any():
        raise ValueError('Conteos territoriales inconsistentes')
    d['fn']=d['retrasos']-d['tp']
    d['tn']=d['operaciones']-d['retrasos']-d['fp']
    if (d['tn']<0).any(): raise ValueError('FP exceden operaciones sin retraso')
    d['precision']=d['tp']/d['alertas'].replace(0,np.nan)
    d['recall']=d['tp']/d['retrasos'].replace(0,np.nan)
    d['fpr']=d['fp']/(d['operaciones']-d['retrasos']).replace(0,np.nan)
    d['tasa_retrasos']=d['retrasos']/d['operaciones']
    d['tasa_alertas']=d['alertas']/d['operaciones']
    d['proporcion_operaciones']=d['operaciones']/d['operaciones'].sum()
    d['proporcion_retrasos']=d['retrasos']/d['retrasos'].sum() if d['retrasos'].sum()>0 else np.nan
    intervalos=[intervalo_wilson(int(tp),int(n)) for tp,n in zip(d['tp'],d['retrasos'])]
    d['recall_ic95_inf']=[x[0] for x in intervalos]
    d['recall_ic95_sup']=[x[1] for x in intervalos]
    d['recall_ic95_ancho']=d['recall_ic95_sup']-d['recall_ic95_inf']
    # Umbrales diagnósticos explícitos: NO ajustan las decisiones del modelo.
    d['muestra_baja_relativa']=d['operaciones']<d['operaciones'].median()
    d['recall_interpretable']=d['retrasos']>=30
    d['alerta_revision']=d['muestra_baja_relativa'] | (~d['recall_interpretable'])
    n=int(d['operaciones'].sum()); pos=int(d['retrasos'].sum()); alertas=int(d['alertas'].sum()); tp=int(d['tp'].sum()); fp=int(d['fp'].sum())
    recall_global=tp/pos if pos else None
    validas=d.loc[d['recall_interpretable'],'recall'].dropna()
    brecha=float(validas.max()-validas.min()) if len(validas)>=2 else None
    correlacion=None
    if len(d)>=3 and d['operaciones'].nunique()>1 and d['recall'].notna().sum()>=3:
        pares = d[["operaciones", "recall"]].dropna()
        corr: float = pares["operaciones"].corr(pares["recall"], method="spearman")
        correlacion = corr if np.isfinite(corr) else None
    resultado={
        'origen':'auditoria_comunas_test.json; datos sintéticos; test agosto-septiembre 2026',
        'alcance':'diagnóstico descriptivo H4-T4; NO optimiza umbral ni reentrena; H5-T3 realizará auditoría formal',
        'total_operaciones':n,'total_retrasos':pos,'total_alertas':alertas,
        'recall_global':recall_global,'precision_global':tp/alertas if alertas else None,
        'fpr_global':fp/(n-pos) if n>pos else None,
        'brecha_recall_comunas_con_al_menos_30_retrasos':brecha,
        'correlacion_spearman_operaciones_recall':correlacion,
        'nota_correlacion':'Asociación descriptiva entre 19 comunas; no demuestra causalidad, sesgo ni significancia.',
        'umbral_muestra_pequena':'Menos operaciones que mediana territorial o menos de 30 retrasos; solo etiqueta de revisión',
        'limitaciones':['Registros sintéticos; sin validación en operaciones reales',
          'Diferencias de desempeño no prueban discriminación ni causalidad',
          'Intervalos Wilson suponen observaciones independientes, posible agrupación por rutas/conductores',
          'Tasas base de retraso y composición territorial varían; comparaciones brutas pueden confundir',
          'No se deben ajustar umbrales usando el conjunto test'],
        'medidas_mitigacion_propuestas':[
          'Revisar cobertura y calidad de datos por comuna y por período; ampliar muestras donde sea necesario',
          'Examinar diferencias condicionando por distancia, franja horaria y tasa base de retraso',
          'Monitorear recall, precision y falsos positivos por comuna con incertidumbre',
          'Probar medidas de balanceo exclusivamente en train y comparar en validación temporal',
          'Aplicar alertas solo como apoyo operativo, sin sanciones automáticas a conductores']}
    return d.sort_values('operaciones',ascending=False),resultado


def analizar_sesgos(entrada='output/resultados/sesgos_equidad/auditoria_comunas_test.json',
                    salida='output/resultados/sesgos_equidad', graficos='output/graficos'):
    archivo=Path(entrada)
    if not archivo.exists():
        raise FileNotFoundError(f'Falta {archivo}. Ejecuta primero: python main.py --proceso evaluar')
    tabla, informe=calcular_diagnostico(pd.read_json(archivo))
    destino=Path(salida); destino.mkdir(parents=True,exist_ok=True)
    dg=Path(graficos); dg.mkdir(parents=True,exist_ok=True)
    tabla.to_csv(destino/'diagnostico_sesgos_comunas.csv',index=False)
    (destino/'diagnostico_sesgos.json').write_text(json.dumps(informe,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7))
    t=tabla.sort_values('recall')
    x=t['recall'].to_numpy(dtype=float)
    inf=t['recall_ic95_inf'].to_numpy(dtype=float)
    sup=t['recall_ic95_sup'].to_numpy(dtype=float)
    good=np.isfinite(x)&np.isfinite(inf)&np.isfinite(sup)
    ax.errorbar(x[good],np.arange(len(t))[good],xerr=[x[good]-inf[good],sup[good]-x[good]],fmt='o',capsize=2)
    ax.set_yticks(np.arange(len(t)),labels=t['comuna_destino'])
    ax.set_xlabel('Recall (IC Wilson aproximado 95 %)')
    ax.set_title('H4-T4: detección de retrasos por comuna — datos sintéticos')
    ax.set_xlim(0,min(1,max(.55,float(np.nanmax(sup))+.02)))
    fig.tight_layout();fig.savefig(dg/'sesgos_recall_comunas.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.scatter(tabla['operaciones'],tabla['recall'],s=45)
    for _,r in tabla.iterrows(): ax.annotate(r['comuna_destino'],(r['operaciones'],r['recall']),xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.set_xlabel('Operaciones de test por comuna');ax.set_ylabel('Recall')
    ax.set_title('H4-T4: representación territorial y recall (no causal)')
    fig.tight_layout();fig.savefig(dg/'sesgos_representacion_recall.png',dpi=160);plt.close(fig)
    print(f'Diagnóstico H4-T4: {len(tabla)} comunas; {informe["total_operaciones"]:,} operaciones')
    print(f'Recall global: {informe["recall_global"]:.3f} | brecha descriptiva: {informe["brecha_recall_comunas_con_al_menos_30_retrasos"] if informe["brecha_recall_comunas_con_al_menos_30_retrasos"] is not None else "no calculable"}')
    print(f'Evidencias: {destino} y {dg}. No se reentrenó ni cambió el umbral.')
    return tabla,informe

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--entrada',default='output/resultados/sesgos_equidad/auditoria_comunas_test.json')
    a=p.parse_args(); analizar_sesgos(entrada=a.entrada)
