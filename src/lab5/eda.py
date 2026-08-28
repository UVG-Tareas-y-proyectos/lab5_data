"""Descripcion y graficas exploratorias del conjunto de entrenamiento."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CLASS_NAMES = {0: "No desastre", 1: "Desastre real"}
COLORS = {0: "#4C78A8", 1: "#E45756"}


def dataset_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla compacta de descripcion del dataset."""

    values = {
        "filas": len(data),
        "columnas_originales": 5,
        "ids_unicos": data["id"].nunique(),
        "keywords_unicos": data["keyword"].nunique(dropna=True),
        "locations_unicas": data["location"].nunique(dropna=True),
        "duplicados_fila_completa": int(data.duplicated().sum()),
        "textos_duplicados": int(data["text"].duplicated().sum()),
        "caracteres_promedio": round(float(data["characters"].mean()), 2),
        "palabras_promedio_original": round(float(data["words_raw"].mean()), 2),
        "palabras_promedio_limpio": round(float(data["words_clean"].mean()), 2),
    }
    return pd.DataFrame(
        {"metrica": list(values.keys()), "valor": list(values.values())}
    )


def export_eda(data: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Exporta tablas y graficas descriptivas para el informe del avance."""

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = dataset_summary(data)
    missing = (
        data[["id", "keyword", "location", "text", "target"]]
        .isna()
        .sum()
        .rename("faltantes")
        .to_frame()
    )
    missing["porcentaje"] = (missing["faltantes"] / len(data) * 100).round(2)
    distribution = (
        data["target"]
        .value_counts()
        .sort_index()
        .rename_axis("target")
        .reset_index(name="cantidad")
    )
    distribution["categoria"] = distribution["target"].map(CLASS_NAMES)
    distribution["porcentaje"] = (
        distribution["cantidad"] / len(data) * 100
    ).round(2)

    summary_path = tables_dir / "dataset_summary.csv"
    missing_path = tables_dir / "missing_values.csv"
    distribution_path = tables_dir / "class_distribution.csv"
    summary.to_csv(summary_path, index=False)
    missing.to_csv(missing_path)
    distribution.to_csv(distribution_path, index=False)

    class_figure = figures_dir / "class_distribution.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        distribution["categoria"],
        distribution["cantidad"],
        color=[COLORS[target] for target in distribution["target"]],
    )
    ax.bar_label(
        bars,
        labels=[
            f"{count:,}\n({percentage:.1f}%)"
            for count, percentage in zip(
                distribution["cantidad"], distribution["porcentaje"]
            )
        ],
        padding=4,
    )
    ax.set_title("Distribución de la variable objetivo")
    ax.set_ylabel("Cantidad de tweets")
    ax.set_ylim(0, distribution["cantidad"].max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(class_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    length_figure = figures_dir / "text_length_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for target in (0, 1):
        ax.hist(
            data.loc[data["target"] == target, "words_raw"],
            bins=25,
            alpha=0.58,
            label=CLASS_NAMES[target],
            color=COLORS[target],
        )
    ax.set_title("Longitud de los tweets por categoría")
    ax.set_xlabel("Cantidad de palabras antes de la limpieza")
    ax.set_ylabel("Cantidad de tweets")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(length_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "summary": summary_path,
        "missing": missing_path,
        "distribution": distribution_path,
        "class_figure": class_figure,
        "length_figure": length_figure,
    }
