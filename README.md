# Práctica 2 — Análisis de tiempos de llegada y servicio

## Caso
**El Lanzamiento Que Casi Tumba La Plataforma: Estimación de Parámetros para Escalado de Emergencia.**

## Estructura
```text
practica2_Luna_Javier/
├── datos/
│   └── datos.csv
├── scripts/
│   └── analisis.py
├── reporte/
│   └── informe.txt
└── README.md
```

## Objetivo
Analizar los tiempos entre llegadas y los tiempos de servicio del periodo de lanzamiento (IDs 11–30), detectar valores atípicos mediante Z-Score y estimar la cantidad de instancias necesarias para mantener la utilización objetivo `rho <= 0.70`.

## Requisitos
- Python 3.10 o posterior
- NumPy
- SciPy

Instala las dependencias con:

```bash
python -m pip install numpy scipy
```

## Ejecución
Desde la carpeta raíz del repositorio:

```bash
python scripts/analisis.py
```

El script lee `datos/datos.csv`, calcula estadísticos descriptivos, detecta outliers con `|Z| > 3`, calcula intervalos de confianza, realiza la prueba K-S contra la distribución exponencial y genera/actualiza `reporte/informe.txt`.

## Datos
Los 20 registros del periodo de lanzamiento se transcribieron del PDF proporcionado. El programa elimina cada outlier únicamente de la variable en la que fue detectado, manteniendo la otra medición de la misma fila.

## Nota
Verifica el nombre completo y el código de estudiante en `reporte/informe.txt` antes de entregar. El informe presenta la prueba K-S como orientativa, ya que los parámetros se estiman con la misma muestra.
