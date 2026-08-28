"""Genera todos los analisis, resultados y explicaciones del avance."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.data import add_preprocessed_columns, load_dataset  # noqa: E402
from lab5.eda import export_eda  # noqa: E402
from lab5.modeling import train_and_export  # noqa: E402
from lab5.ngrams import export_ngrams  # noqa: E402
from lab5.report import write_advance_report  # noqa: E402


def main() -> None:
    data = load_dataset(PROJECT_ROOT / "data" / "train.csv")
    processed = add_preprocessed_columns(data)
    output_dir = PROJECT_ROOT / "outputs"

    export_eda(processed, output_dir)
    frequencies, _ = export_ngrams(processed, output_dir)
    metrics, _ = train_and_export(processed, output_dir)
    report_path = write_advance_report(
        processed, frequencies, metrics, output_dir
    )

    print(f"Avance generado: {report_path.relative_to(PROJECT_ROOT)}")
    print(
        "Modelo preliminar: "
        f"F1={metrics['f1']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()

