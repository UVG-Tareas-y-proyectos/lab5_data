"""Calculo y visualizacion de frecuencias de unigramas y bigramas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from lab5.eda import CLASS_NAMES, COLORS


def ngram_frequencies(
    texts: pd.Series, n: int, top_k: int = 20
) -> pd.DataFrame:
    """Devuelve frecuencia absoluta y probabilidad empirica de los n-gramas."""

    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        token_pattern=r"(?u)\b[a-z0-9]+\b",
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    total = int(counts.sum())

    frequencies = pd.DataFrame(
        {
            "ngrama": vectorizer.get_feature_names_out(),
            "frecuencia": counts.astype(int),
        }
    ).sort_values(["frecuencia", "ngrama"], ascending=[False, True])
    frequencies["probabilidad"] = frequencies["frecuencia"] / total
    return frequencies.head(top_k).reset_index(drop=True)


def frequencies_by_class(
    data: pd.DataFrame, top_k: int = 20
) -> pd.DataFrame:
    """Calcula unigramas y bigramas por valor de target."""

    tables: list[pd.DataFrame] = []
    for target in (0, 1):
        texts = data.loc[data["target"] == target, "text_clean"]
        for n in (1, 2):
            table = ngram_frequencies(texts, n=n, top_k=top_k)
            table.insert(0, "n", n)
            table.insert(0, "categoria", CLASS_NAMES[target])
            table.insert(0, "target", target)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def export_ngrams(
    data: pd.DataFrame, output_dir: str | Path, top_k: int = 20
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Exporta la tabla completa y una figura comparativa de n-gramas."""

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frequencies = frequencies_by_class(data, top_k=top_k)
    table_path = tables_dir / "ngram_frequencies.csv"
    frequencies.to_csv(table_path, index=False)

    figure_path = figures_dir / "top_ngrams_by_class.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for row, n in enumerate((1, 2)):
        for column, target in enumerate((0, 1)):
            ax = axes[row, column]
            subset = frequencies[
                (frequencies["n"] == n) & (frequencies["target"] == target)
            ].head(15)
            subset = subset.sort_values("frecuencia")
            ax.barh(
                subset["ngrama"],
                subset["frecuencia"],
                color=COLORS[target],
                alpha=0.88,
            )
            kind = "Unigramas" if n == 1 else "Bigramas"
            ax.set_title(f"{kind} - {CLASS_NAMES[target]}")
            ax.set_xlabel("Frecuencia")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("N-gramas más frecuentes por categoría", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return frequencies, {"table": table_path, "figure": figure_path}
