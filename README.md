# Laboratorio 5 - Minería de Textos y Análisis de Sentimiento

Este repositorio contiene la solución completa y ejecutable del **Laboratorio 5 de Data Science (CC3084)**. El proyecto clasifica si un tweet se refiere a un desastre natural o emergencia real (`target = 1`) o no (`target = 0`), utilizando minería de texto (TF-IDF de unigramas y bigramas), visualización con nubes de palabras, análisis de sentimiento con VADER, evaluación de la variable de negatividad y comparación de 4 clasificadores de Machine Learning.

El cuaderno principal de trabajo es:

```text
notebooks/laboratorio-5-mineria-de-texto-y-analisis-de-sentimiento.ipynb
```

El notebook está organizado de principio a fin con todas sus salidas (gráficas, tablas, métricas y explicaciones Markdown) completamente ejecutadas y reproducibles.

---

## Estructura del Repositorio

```text
.
├── data/
│   ├── raw/                  # train.csv original descargado y verificado (SHA-256)
│   └── processed/            # Tablas procesadas y scores de sentimiento
├── notebooks/                # Cuaderno Jupyter ejecutable principal
│   └── laboratorio-5-mineria-de-texto-y-analisis-de-sentimiento.ipynb
├── reports/                  # Informes finales y artefactos exportados
│   ├── figures/              # Nubes de palabras, histogramas, matrices y boxplots
│   ├── tables/               # Frecuencias de n-gramas, reportes y métricas en CSV
│   ├── avance.md             # Informe de entrega del avance
│   └── informe_final.md      # Informe académico final completo
├── src/
│   └── lab5/                 # Módulos Python reutilizables
│       ├── __init__.py
│       ├── data.py           # Carga, validación y preprocesamiento de texto
│       ├── eda.py            # Análisis exploratorio, nubes de palabras e histogramas
│       ├── ngrams.py         # Unigramas, bigramas y probabilidades empíricas
│       ├── sentiment.py      # VADER, negatividad/positividad y rankings de sentimiento
│       ├── modeling.py       # Regresión Logística, Naive Bayes, LinearSVC y Random Forest
│       ├── predictor.py      # Función predict_tweet(text) para inferencia de usuario
│       └── report.py         # Generación automatizada de reportes Markdown
├── scripts/                  # Scripts de descarga y ejecución del pipeline
│   ├── download_data.py      # Descarga atómica y validación por SHA-256
│   ├── run_advance.py        # Ejecución del avance
│   └── run_full_lab.py       # Ejecución del pipeline maestro
├── tests/                    # Pruebas unitarias completas
├── codebook.md               # Diccionario de datos, variables y criterios de preprocesamiento
├── requirements.txt          # Dependencias del proyecto Python
└── README.md
```

---

## Cómo Ejecutar el Laboratorio

### 1. Instalación de Dependencias
```bash
python3 -m pip install -r requirements.txt
```

### 2. Descarga del Dataset
```bash
python3 scripts/download_data.py
```

### 3. Abrir y Ejecutar el Jupyter Notebook
```bash
jupyter notebook notebooks/laboratorio-5-mineria-de-texto-y-analisis-de-sentimiento.ipynb
```
*El notebook se lee de arriba hacia abajo y contiene todas las explicaciones de los Ejercicios 1 al 11.*

### 4. Ejecución Alternativa por Consola / Pipeline Automatizado
```bash
python3 scripts/run_full_lab.py
```

### 5. Ejecutar Pruebas Unitarias
```bash
python3 -m unittest discover -s tests -v
```

---

## Resumen de Resultados

1. **Unigramas y Bigramas**: N-gramas como `forest fire` o `suicide bomber` aportan contexto local clave para la clasificación.
2. **Análisis de Sentimiento**: Los tweets de desastre real (`target=1`) presentan un nivel de negatividad significativamente mayor (54.2% clasificados como Negativo vs 28.4% en no desastre).
3. **Mejor Clasificador**: **Regresión Logística / LinearSVC con TF-IDF** obtuvo el mejor rendimiento global con un **F1-Score superior a 0.77** y **ROC-AUC superior a 0.85**.
4. **Impacto de la Variable Negatividad (Pregunta 10)**: La inclusión de la negatividad como característica adicional no alteró sustancialmente las métricas, confirmando que la representación vectorial TF-IDF absorbe adecuadamente la carga semántica de emergencia.
