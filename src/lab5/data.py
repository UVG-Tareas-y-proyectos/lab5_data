"""Carga, validacion y preprocesamiento de tweets."""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


EXPECTED_COLUMNS = ["id", "keyword", "location", "text", "target"]
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga train.csv y valida el esquema minimo del laboratorio."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecute: python scripts/download_data.py"
        )

    data = pd.read_csv(path)
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Se esperaban las columnas {EXPECTED_COLUMNS}; "
            f"se recibieron {data.columns.tolist()}."
        )
    if data["id"].isna().any() or data["text"].isna().any():
        raise ValueError("Las columnas id y text no pueden tener valores faltantes.")
    if not set(data["target"].dropna().unique()).issubset({0, 1}):
        raise ValueError("target debe ser una etiqueta binaria (0 o 1).")
    return data


def clean_tweet(text: str, remove_stopwords: bool = True) -> str:
    """Normaliza un tweet conservando las palabras de hashtags y los numeros.

    Decisiones de limpieza:
    - convierte a minusculas y normaliza caracteres Unicode;
    - elimina URL y menciones completas;
    - quita el simbolo #, pero conserva la palabra del hashtag;
    - elimina apostrofes, puntuacion, simbolos y emoticones;
    - conserva numeros (incluido 911) por su posible valor semantico;
    - elimina stopwords inglesas y tokens de una letra.
    """

    normalized = html.unescape(str(text)).lower()
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = MENTION_PATTERN.sub(" ", normalized)
    normalized = normalized.replace("#", "")
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("'", "")

    tokens = TOKEN_PATTERN.findall(normalized)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    tokens = [token for token in tokens if len(token) > 1 or token.isdigit()]
    return " ".join(tokens)


def add_preprocessed_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Agrega texto limpio y medidas basicas sin modificar el DataFrame original."""

    result = data.copy()
    result["text_clean"] = result["text"].map(clean_tweet)
    result["characters"] = result["text"].str.len()
    result["words_raw"] = result["text"].str.split().str.len()
    result["words_clean"] = result["text_clean"].str.split().str.len()
    return result

