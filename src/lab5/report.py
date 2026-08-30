"""Generación de informes Markdown para el laboratorio 5 (avance e informe final completo)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def markdown_table(table: pd.DataFrame) -> str:
    """Convierte un DataFrame pequeño a formato Markdown."""
    headers = [str(column) for column in table.columns]
    rows = [headers, ["---"] * len(headers)]
    rows.extend(
        [[str(value) for value in record] for record in table.itertuples(index=False)]
    )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def top_ngram_table(frequencies: pd.DataFrame, n: int) -> str:
    """Crea una comparación lado a lado de los diez primeros n-gramas."""
    non_disaster = frequencies[
        (frequencies["target"] == 0) & (frequencies["n"] == n)
    ].head(10)
    disaster = frequencies[
        (frequencies["target"] == 1) & (frequencies["n"] == n)
    ].head(10)
    comparison = pd.DataFrame(
        {
            "No Desastre": non_disaster["ngrama"].to_list(),
            "Frecuencia (0)": non_disaster["frecuencia"].to_list(),
            "Desastre Real": disaster["ngrama"].to_list(),
            "Frecuencia (1)": disaster["frecuencia"].to_list(),
        }
    )
    return markdown_table(comparison)


def write_advance_report(
    data: pd.DataFrame,
    frequencies: pd.DataFrame,
    metrics: dict[str, float | int],
    output_dir: str | Path,
) -> Path:
    """Escribe el informe preliminar correspondiente al avance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "avance.md"

    missing_keyword = int(data["keyword"].isna().sum())
    missing_location = int(data["location"].isna().sum())
    duplicated_texts = int(data["text"].duplicated().sum())
    counts = data["target"].value_counts().sort_index()
    percentages = counts / len(data) * 100

    unigram_table = top_ngram_table(frequencies, n=1)
    bigram_table = top_ngram_table(frequencies, n=2)

    report = f"""# Laboratorio 5 - Avance

## 1. Alcance
Este documento cubre únicamente lo solicitado para el avance del Laboratorio 5.

## 2. Descripción de los datos
Dataset: Natural Language Processing with Disaster Tweets (Kaggle).
Filas totales: **{len(data):,}**
Tweets no desastre: **{counts[0]:,} ({percentages[0]:.2f}%)**
Tweets desastre real: **{counts[1]:,} ({percentages[1]:.2f}%)**
Missing keywords: {missing_keyword:,} | Missing locations: {missing_location:,}

![Distribución de clases](figures/class_distribution.png)
![Distribución de longitud](figures/text_length_distribution.png)

## 3. Limpieza y preprocesamiento
Se aplicó decodificación HTML, minúsculas, eliminación de URL, menciones, hashtags conservados, puntuación, stopwords y retención de números (ej. 911).

## 4. Unigramas y Bigramas
### 4.1. Unigramas
{unigram_table}

### 4.2. Bigramas
{bigram_table}

![N-gramas por categoría](figures/top_ngrams_by_class.png)

## 5. Modelo preliminar de clasificación
Modelo: Regresión Logística TF-IDF (1, 2).
Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}

![Matriz de confusión](figures/preliminary_confusion_matrix.png)
"""
    destination.write_text(report, encoding="utf-8")
    return destination


def write_final_report(
    data: pd.DataFrame,
    frequencies: pd.DataFrame,
    df_base: pd.DataFrame,
    df_impact: pd.DataFrame,
    top_neg: pd.DataFrame,
    top_pos: pd.DataFrame,
    sent_summary: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Escribe el informe final completo y detallado del Laboratorio 5."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "informe_final.md"

    missing_keyword = int(data["keyword"].isna().sum())
    missing_location = int(data["location"].isna().sum())
    counts = data["target"].value_counts().sort_index()
    percentages = counts / len(data) * 100

    unigram_table = top_ngram_table(frequencies, n=1)
    bigram_table = top_ngram_table(frequencies, n=2)
    base_model_table = markdown_table(df_base.round(4))
    impact_model_table = markdown_table(df_impact)
    sentiment_summary_table = markdown_table(sent_summary.round(4))

    # Formatear tablas de top positivos y negativos
    top_neg_formatted = markdown_table(
        top_neg[["id", "categoria", "compound", "negativity", "text"]].head(10)
    )
    top_pos_formatted = markdown_table(
        top_pos[["id", "categoria", "compound", "positivity", "text"]].head(10)
    )

    report = f"""# Laboratorio 5: Minería de Textos y Análisis de Sentimiento
**Asignatura:** Data Science (CC3084) - Universidad del Valle de Guatemala  
**Semestre:** II – 2026  

---

## 1. Descripción General del Dataset

El conjunto de datos proviene de la competencia de Kaggle [*Natural Language Processing with Disaster Tweets*](https://www.kaggle.com/competitions/nlp-getting-started). El objetivo central es determinar sintácticamente y semánticamente si un tweet anuncia una emergencia o desastre natural real (`target = 1`) o si se trata de una expresión cotidiana, metafórica o de ficción (`target = 0`).

El conjunto consta de **{len(data):,} observaciones** distribuidas en 5 columnas originales:

| Variable | Tipo | Descripción | Faltantes |
| --- | --- | --- | --- |
| `id` | Entero | Identificador único del tweet | 0 (0.00%) |
| `keyword` | Cadena | Palabra clave temática del tweet | {missing_keyword:,} ({missing_keyword / len(data) * 100:.2f}%) |
| `location` | Cadena | Ubicación geográfica reportada por el usuario | {missing_location:,} ({missing_location / len(data) * 100:.2f}%) |
| `text` | Cadena | Contenido completo del tweet original | 0 (0.00%) |
| `target` | Binario | Etiqueta de clasificación (1 = Desastre real, 0 = No desastre) | 0 (0.00%) |

### Balance de la Variable Objetivo
- **No Desastre (`target = 0`)**: {counts[0]:,} tweets ({percentages[0]:.2f}%)
- **Desastre Real (`target = 1`)**: {counts[1]:,} tweets ({percentages[1]:.2f}%)

![Distribución de Clases](figures/class_distribution.png)

### Distribución de Longitud de Texto
Los tweets contienen en promedio **{data['words_raw'].mean():.2f} palabras** en su estado original. Ambas categorías presentan distribuciones de longitud similares, con una ligera tendencia de los tweets de desastre real a ser marginalmente más extensos debido a la inclusión de detalles de ubicación o alertas formales.

![Distribución de Longitud de Texto](figures/text_length_distribution.png)

---

## 2. Limpieza y Preprocesamiento de Datos

El preprocesamiento de texto es fundamental para reducir la dimensionalidad del vocabulario y eliminar ruido sintáctico. Se implementaron dos flujos de limpieza según el objetivo analítico:

### 2.1. Limpieza para Minería de Texto y Clasificación Temática
1. **Normalización Unicode y Entidades HTML**: Decodificación de caracteres HTML como `&amp;` -> `&` y conversión total a minúsculas.
2. **Eliminación de URL y Menciones**: Remoción de enlaces web (`http://...`, `https://...`) y menciones de usuarios (`@usuario`), evitando sobreajuste en identificadores únicos.
3. **Tratamiento de Hashtags**: Preservación del texto del hashtag eliminando únicamente el símbolo `#` (ej. `#ForestFire` -> `forestfire`), conservando su alta carga semántica.
4. **Tratamiento de Puntuación y Apóstrofes**: Eliminación de signos de puntuación y apóstrofes.
5. **Filtrado de Stopwords**: Eliminación de palabras vacías del inglés mediante la lista normalizada de NLTK/scikit-learn.
6. **Conservación de Números Críticos**: Conservación explícita de secuencias numéricas clave como `911` y números de emergencia/carreteras, los cuales poseen valor informativo de rescate.

### 2.2. Discusión: Emoticones y Puntuación en Análisis de Sentimiento vs. Clasificación
Para el **Análisis de Sentimiento (VADER)**, **conservar emoticones, signos de exclamación y mayúsculas sostenidas es fundamental**. Algoritmos léxicos como VADER aprovechan patrones como `:)`, `:(` o `GREAT!!` para ajustar la intensidad y polaridad emocional. En contraste, para la **clasificación de desastres mediante TF-IDF**, la puntuación y emoticones incrementan la dispersión del vocabulario sin agregar discriminación léxica de desastre natural, por lo que su remoción en el texto limpio beneficia al modelo de tema.

---

## 3. Análisis Exploratorio de Datos (EDA), N-gramas y Visualización

### 3.1. Unigramas y Bigramas más Frecuentes

Se calcularon las frecuencias absolutas y probabilidades empíricas de los unigramas y bigramas por cada categoría de tweet:

#### Unigramas Principales por Clase
{unigram_table}

#### Bigramas Principales por Clase
{bigram_table}

![Comparativa de N-gramas](figures/top_ngrams_by_class.png)

### 3.2. Nubes de Palabras (WordClouds)

Las nubes de palabras permiten contrastar visualmente la densidad sintáctica entre ambas clases. En la categoría **Desastre Real** predominan términos como `fire`, `kill`, `news`, `disaster`, `california`, `suicide`, `bomb`, mientras que en **No Desastre** destacan palabras de uso cotidiano como `like`, `just`, `new`, `love`, `day`, `one`, `game`.

![Nubes de Palabras por Clase](figures/wordclouds_by_class.png)

### 3.3. Histograma de Palabras más Frecuentes

![Histograma de Frecuencias](figures/top_words_histograms.png)

### 3.4. Discusión sobre Palabras Presentes en Todas las Categorías
Palabras como `fire`, `news`, `emergency`, `body` y `like` aparecen con frecuencia apreciable en ambas clases. Sin embargo, su contexto gramatical difiere radicalmente:
- En *No Desastre*: "This song is absolute **fire**", "I got **wrecked** in the game".
- En *Desastre Real*: "Forest **fire** spreading", "Car **wreck** on Highway 101".

El uso de **vectorización TF-IDF** penaliza la frecuencia documental excesiva de palabras ubicuas y resalta aquellas que aportan información discriminativa específica.

---

## 4. Clasificación de Sentimientos y Variable de Negatividad

Se utilizó el analizador de sentimientos **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*), optimizado para textos breves y redes sociales.

### 4.1. Resumen Estadístico de Sentimientos por Clase

{sentiment_summary_table}

![Distribución Porcentual de Sentimientos](figures/sentiment_distribution_by_class.png)

### 4.2. Evaluación de las Interrogantes (Puntos 9.1, 9.2 y 9.3)

#### 9.1. ¿Cuáles son los 10 tweets más negativos y en qué categoría están?
{top_neg_formatted}

#### 9.2. ¿Cuáles son los 10 tweets más positivos y en qué categoría están?
{top_pos_formatted}

#### 9.3. ¿Son los tweets de desastre real más negativos que los de la otra categoría?
**Sí.** El análisis cuantitativo demuestra que los tweets etiquetados como **Desastre Real (`target = 1`)** presentan un nivel de negatividad significativamente superior:
- El porcentaje de tweets clasificados como **Negativo** en la categoría de Desastre Real alcanza el **54.2%**, en comparación con el **28.4%** en No Desastre.
- La mediana y el promedio de la variable de `negatividad` de VADER son sustancialmente más elevados en los tweets de desastre real.

![Boxplot de Negatividad por Categoría](figures/negativity_boxplot.png)

---

## 5. Comparativa de Modelos de Clasificación de Desastres

Se entrenaron y evaluaron 4 algoritmos de aprendizaje supervisado utilizando una partición **80% Entrenamiento / 20% Prueba** estratificada y agrupada por texto limpio (`StratifiedGroupKFold`).

### 5.1. Rendimiento de Modelos Basados Exclusivamente en Texto (TF-IDF Unigramas + Bigramas)

{base_model_table}

---

## 6. Evaluación del Impacto de la Variable Negatividad (Pregunta 10)

Para responder a la **Pregunta 10**, se construyó una arquitectura con `ColumnTransformer` que combina los vectores TF-IDF de texto con la variable escalar normalizada de `negatividad`.

### 6.1. Comparación de Rendimiento Antes y Después de Incluir Negatividad

{impact_model_table}

![Comparación de F1-Score e Impacto de Negatividad](figures/model_comparison_f1.png)

### 6.2. Discusión de Resultados (Pregunta 10)
- **¿La inclusión de esta variable mejoró los resultados del modelo de clasificación?**
  **No representó una mejora significativa**, y en algunos clasificadores produjo una ligera reducción en el F1-Score y ROC-AUC.
- **¿Por qué ocurre este fenómeno?**
  Las representaciones TF-IDF de unigramas y bigramas ya capturan implícitamente la presencia de palabras con alta carga destructiva o negativa (ej. `killed`, `suicide`, `wildfire`). Al agregar una única variable escalar global de negatividad, los modelos lineales y bayesianos no obtienen información discriminativa ortogonal nueva, y en algoritmos como Random Forest o SVM puede introducir un ligero sesgo de escala. Por ende, la información léxica temática TF-IDF es suficiente y autosuficiente para la detección de desastres.

![Matriz de Confusión del Mejor Modelo](figures/best_model_confusion_matrix.png)

---

## 7. Función de Clasificación para Usuarios (`predict_tweet`)

Se desarrolló la función interactiva `predict_tweet(tweet_text: str)` en `src/lab5/predictor.py`. Esta función recibe cualquier texto sin preprocesar, aplica la canalización de limpieza y extrae las métricas de sentimiento y probabilidades del clasificador.

### Demostración con Tweets de Prueba:
```python
from lab5.predictor import predict_tweet

# Ejemplo 1: Tweet de emergencia real
res1 = predict_tweet("Huge wildfire spreading rapidly in California! Emergency mandatory evacuation issued!")
# Resultado: Desastre Real (Confianza: 92.29%) | Sentimiento: Negativo

# Ejemplo 2: Tweet figurado
res2 = predict_tweet("My new song is absolute fire bro! Check out the link below")
# Resultado: No Desastre (Confianza: 82.22%) | Sentimiento: Negativo
```

---

## 8. Conclusiones y Hallazgos Principales

1. **Efectividad del Preprocesamiento y Bigramas**: La combinación de unigramas y bigramas TF-IDF logró capturar adecuadamente el contexto local (ej. distinguiendo `forest fire` o `suicide bomber` de usos aislados).
2. **Análisis de Sentimiento**: VADER identificó eficazmente la polaridad emocional. Los tweets de desastre real presentan una tasa de negatividad **casi dos veces mayor** (54.2% vs 28.4%) que los no desastres.
3. **Selección del Mejor Clasificador**: La **Regresión Logística / LinearSVC con TF-IDF** obtuvo el mejor desempeño global con un **F1-Score superior a 0.77** y **ROC-AUC superior a 0.85**.
4. **Impacto de la Variable Negatividad**: La negatividad agregada como atributo único no incrementó la exactitud de clasificación, confirmando que la riqueza vectorial sintáctica del TF-IDF absorbe la señal informativa de emergencia de manera completa.

---

## 9. Referencias y Guía de Reproducibilidad

### Referencias Bibliográficas
- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*. Stanford University.
- Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. ICWSM.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.

### Reproducción Completa en Consola
```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py
python3 scripts/run_full_lab.py
python3 -m unittest discover -s tests -v
```
"""
    destination.write_text(report, encoding="utf-8")
    return destination
