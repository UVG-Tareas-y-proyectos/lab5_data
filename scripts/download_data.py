"""Descarga y valida train.csv para el laboratorio 5."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw" / "train.csv"

# Copia publica del archivo original de la competencia. La fuente oficial de
# Kaggle requiere autenticacion y aceptacion previa de sus reglas.
MIRROR_URL = (
    "https://huggingface.co/datasets/VuduVations/disaster_tweets/resolve/"
    "main/data/train.csv"
)
EXPECTED_SHA256 = "61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df"
EXPECTED_COLUMNS = ["id", "keyword", "location", "text", "target"]
EXPECTED_ROWS = 7_613


def sha256(path: Path) -> str:
    """Calcula el SHA-256 de un archivo sin cargarlo completo en memoria."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(path: Path) -> None:
    """Verifica identidad, dimensiones, columnas y etiquetas del dataset."""

    file_hash = sha256(path)
    if file_hash != EXPECTED_SHA256:
        raise ValueError(
            "El archivo descargado no coincide con train.csv esperado. "
            f"SHA-256 recibido: {file_hash}"
        )

    data = pd.read_csv(path)
    if data.shape != (EXPECTED_ROWS, len(EXPECTED_COLUMNS)):
        raise ValueError(f"Dimensiones inesperadas: {data.shape}")
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"Columnas inesperadas: {data.columns.tolist()}")
    if set(data["target"].unique()) != {0, 1}:
        raise ValueError("La variable target debe contener unicamente 0 y 1.")


def download(destination: Path, force: bool = False) -> None:
    """Descarga el CSV de forma atomica y lo valida antes de conservarlo."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.part")

    if destination.exists() and not force:
        validate_dataset(destination)
        print(f"Dataset ya disponible y verificado: {destination}")
        return

    try:
        with urlopen(MIRROR_URL, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        validate_dataset(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Dataset descargado y verificado: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Ruta de salida (por defecto: data/raw/train.csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vuelve a descargar aunque el archivo ya exista.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download(arguments.destination.resolve(), arguments.force)
