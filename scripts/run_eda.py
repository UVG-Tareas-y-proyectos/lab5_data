"""Ejecuta la primera parte del avance: carga, limpieza y descripcion."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.data import add_preprocessed_columns, load_dataset  # noqa: E402
from lab5.eda import export_eda  # noqa: E402


def main() -> None:
    data = load_dataset(PROJECT_ROOT / "data" / "train.csv")
    processed = add_preprocessed_columns(data)
    outputs = export_eda(processed, PROJECT_ROOT / "outputs")

    print(f"Tweets procesados: {len(processed):,}")
    print(f"Desastres reales: {(processed['target'] == 1).sum():,}")
    print(f"No desastres: {(processed['target'] == 0).sum():,}")
    for name, path in outputs.items():
        print(f"{name}: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

