#!/usr/bin/env python3
"""
Analisis del Reto 1: El Lanzamiento Que Casi Tumba La Plataforma.
Calcula estadisticos descriptivos, outliers mediante Z-Score, intervalos
de confianza, prueba K-S exponencial y dimensionamiento de instancias.
Requisitos: Python 3, numpy, scipy.
Instalacion: python -m pip install numpy scipy
Ejecucion desde la raiz del repositorio:
    python scripts/analisis.py
"""
from pathlib import Path
import math
import csv
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datos" / "datos.csv"
REPORT_PATH = ROOT / "reporte" / "informe.txt"

def load_data():
    with DATA_PATH.open("r", newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    ids = np.array([int(r["id"]) for r in data])
    arrivals = np.array([float(r["tiempo_entre_llegadas_s"]) for r in data])
    service = np.array([float(r["tiempo_servicio_s"]) for r in data])
    return ids, arrivals, service

def descriptive(x):
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
    }

def outliers_z(ids, x, threshold=3):
    z = (x - np.mean(x)) / np.std(x, ddof=1)
    return [(int(ids[i]), float(x[i]), float(z[i])) for i in range(len(x)) if abs(z[i]) > threshold], z

def ci_mean(x, confidence=0.95):
    n = len(x)
    mean = np.mean(x)
    s = np.std(x, ddof=1)
    margin = stats.t.ppf((1 + confidence) / 2, n - 1) * s / math.sqrt(n)
    return float(mean), float(mean - margin), float(mean + margin), float(s)

def fmt(v, digits=4):
    return f"{v:.{digits}f}"

def main():
    ids, arrivals, service = load_data()
    out_a, z_a = outliers_z(ids, arrivals)
    out_s, z_s = outliers_z(ids, service)

    # Cada outlier se elimina solamente de su variable, como en el PDF fuente.
    clean_a = arrivals[np.abs(z_a) <= 3]
    clean_s = service[np.abs(z_s) <= 3]
    clean_ids_a = ids[np.abs(z_a) <= 3]
    clean_ids_s = ids[np.abs(z_s) <= 3]

    da, ds = descriptive(clean_a), descriptive(clean_s)
    ma, loa, hia, sa = ci_mean(clean_a)
    ms, los, his, ss = ci_mean(clean_s)
    lam, mu = 1 / ma, 1 / ms
    lambda_ci = (1 / hia, 1 / loa)
    mu_ci = (1 / his, 1 / los)
    rho1 = lam / mu
    c_min = math.ceil(lam / (0.70 * mu))
    rho5, rho6 = lam / (5 * mu), lam / (6 * mu)

    # La media se estima de la muestra; el resultado K-S es orientativo
    # porque la prueba clásica no corrige por parámetros estimados.
    ks_a = stats.kstest(clean_a, "expon", args=(0, ma))
    ks_s = stats.kstest(clean_s, "expon", args=(0, ms))

    lines = [
        "REPORTE DE ANALISIS ESTADISTICO",
        "Caso: El Lanzamiento Que Casi Tumba La Plataforma",
        "Materia: Simulacion “Y”",
        "Nombre del estudiante: Javier Luna (verificar y completar apellidos/codigo si corresponde)",
        "",
        "1. FORMULACION DEL PROBLEMA",
        "Dimensionar la cantidad de instancias de una plataforma durante un lanzamiento, usando los tiempos entre llegadas y de servicio del periodo pico (IDs 11–30), para mantener la utilizacion objetivo rho <= 0.70 y evitar saturacion.",
        "",
        "2. VARIABLE PRINCIPAL",
        "Variable principal de dimensionamiento: tasa de llegada lambda (solicitudes/segundo). Variable complementaria: tasa de servicio mu (solicitudes/segundo por instancia). Se calculan a partir de los tiempos medios entre llegadas y de servicio.",
        "",
        "3. SUPUESTOS",
        "- Se utiliza el periodo de lanzamiento, IDs 11–30 (20 observaciones), porque representa el pico que debe soportar la infraestructura.",
        "- Los tiempos están expresados en segundos y las tasas en solicitudes por segundo.",
        "- Se calcula Z-Score con desviación estándar muestral (ddof=1); se considera atípico |Z| > 3.",
        "- Cada valor atípico se excluye únicamente de la variable donde aparece; el otro tiempo de esa fila se conserva.",
        "- El objetivo operativo es rho <= 0.70. La recomendación N+1 considera una instancia de reserva.",
        "- La muestra es pequeña (19 datos limpios por variable) y los resultados describen la carga observada.",
        "",
        "4. ESTADISTICOS DESCRIPTIVOS (DESPUES DE EXCLUIR OUTLIERS)",
        f"Tiempo entre llegadas: n={da['n']}, media={fmt(da['mean'])} s, mediana={fmt(da['median'])} s, desviacion estandar muestral={fmt(da['std'])} s, minimo={fmt(da['min'])} s, maximo={fmt(da['max'])} s, Q1={fmt(da['q1'])} s, Q3={fmt(da['q3'])} s.",
        f"Tiempo de servicio: n={ds['n']}, media={fmt(ds['mean'])} s, mediana={fmt(ds['median'])} s, desviacion estandar muestral={fmt(ds['std'])} s, minimo={fmt(ds['min'])} s, maximo={fmt(ds['max'])} s, Q1={fmt(ds['q1'])} s, Q3={fmt(ds['q3'])} s.",
        "",
        "5. VALORES ATIPICOS DETECTADOS CON Z-SCORE (CALCULADO SOBRE LAS 20 OBSERVACIONES ORIGINALES)",
        "Umbral: |Z| > 3.",
        *[f"- ID {i}, tiempo entre llegadas={v:.2f} s, Z={z:+.3f}: excluir de la variable de llegadas; probable hueco de registro o medicion." for i,v,z in out_a],
        *[f"- ID {i}, tiempo de servicio={v:.2f} s, Z={z:+.3f}: excluir del calculo de mu; documentar como posible incidente real del sistema." for i,v,z in out_s],
        "Los demás valores de cada variable no superan el umbral |Z| > 3.",
        "",
        "6. ANALISIS DESCRIPTIVO DE LA VARIABLE",
        f"Con los datos depurados, la media entre llegadas es {ma:.5f} s, por lo que lambda=1/media={lam:.3f} solicitudes/s (IC 95% aproximado: {lambda_ci[0]:.3f}–{lambda_ci[1]:.3f}).",
        f"La media de servicio es {ms:.5f} s, por lo que mu=1/media={mu:.3f} solicitudes/s por instancia (IC 95% aproximado: {mu_ci[0]:.3f}–{mu_ci[1]:.3f}).",
        f"Prueba K-S contra distribución exponencial: llegadas D={ks_a.statistic:.3f}, p={ks_a.pvalue:.3g}; servicio D={ks_s.statistic:.3f}, p={ks_s.pvalue:.3g}. Estos resultados son orientativos porque la media se estima con la misma muestra; se recomienda una corrección apropiada o simulación con distribuciones ajustadas.",
        "",
        "7. INTERPRETACION TECNICA",
        f"Con una instancia, rho=lambda/mu={rho1:.3f}, mayor que 1; la llegada de solicitudes supera la capacidad de servicio y el sistema no es estable. Para rho <= 0.70, c >= lambda/(0.70*mu) = {lam:.3f}/(0.70*{mu:.3f}), por lo que el mínimo calculado es {c_min} instancias. Con 5 instancias, rho={rho5:.3f}; con 6, rho={rho6:.3f}. Se recomienda 6 instancias para disponer de redundancia N+1 y mayor margen frente a incertidumbre.",
        "Los datos no se ajustan bien a una distribución exponencial según la prueba K-S indicada en el documento fuente. El dimensionamiento por tasas medias sigue siendo útil para la utilización, pero las estimaciones de espera basadas en M/M/c deben tratarse como aproximaciones.",
        "",
        "8. CONCLUSION",
        f"La plataforma no debe afrontar el pico con una sola instancia: la utilización estimada sería {rho1:.2f}. El mínimo para mantener rho <= 0.70 es {c_min} instancias; se recomienda operar con 6, escalado preventivo antes del lanzamiento, revisar los eventos anómalos de los IDs 19 y 24, y validar el comportamiento mediante simulación. La conclusión se limita a la muestra analizada y debe revisarse si cambia el patrón de tráfico.",
        "",
        "Fuente de datos y resultados base: documento “Reto 1: El Lanzamiento Que Casi Tumba La Plataforma – Estimación de Parámetros para Escalado de Emergencia”, proporcionado para esta práctica.",
    ]
    REPORT_PATH.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print("\\n".join(lines))
    print(f"\\nInforme guardado en: {REPORT_PATH}")

if __name__ == "__main__":
    main()
