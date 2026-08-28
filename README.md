# Laboratorio 5 - Mineria de textos

Repositorio del **avance** del Laboratorio 5 de CC3084 Data Science. El alcance
implementado es:

- descripcion y validacion del conjunto de datos;
- limpieza y preprocesamiento explicados;
- frecuencias y probabilidades de unigramas y bigramas por clase;
- analisis exploratorio con graficas;
- modelo preliminar TF-IDF + regresion logistica y evaluacion en holdout.

El informe reproducible con los hallazgos se encuentra en
[`outputs/avance.md`](outputs/avance.md).

## Ejecucion

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_advance.py
python -m unittest discover -s tests -v
```

El CSV se valida por esquema, dimensiones y hash antes de utilizarse. Todas las
tablas y figuras se regeneran dentro de `outputs/`.

## Estructura

```text
data/                 instrucciones para obtener train.csv
scripts/              descarga y ejecucion reproducible
src/lab5/             limpieza, EDA, n-gramas, modelo e informe
tests/                 pruebas de preprocesamiento y conteos
outputs/               tablas, figuras, metricas e informe del avance
```

No se incluyen todavia las actividades reservadas para la entrega final:
sentimiento, variable de negatividad, funcion de prediccion para usuario,
comparacion completa de modelos ni seleccion del mejor clasificador.

