# Datos

El laboratorio utiliza `train.csv` de la competencia de Kaggle
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data).

Para descargar una copia verificada del archivo de entrenamiento:

```powershell
python scripts/download_data.py
```

El script comprueba que el archivo tenga 7,613 filas, las columnas `id`,
`keyword`, `location`, `text` y `target`, y el hash SHA-256 esperado. El CSV no
se versiona para evitar duplicar los datos de la competencia en el repositorio.

Si se cuenta con credenciales y se aceptaron las reglas de Kaggle, también se
puede descargar el archivo desde la fuente oficial con la CLI:

```powershell
kaggle competitions download -c nlp-getting-started
```

