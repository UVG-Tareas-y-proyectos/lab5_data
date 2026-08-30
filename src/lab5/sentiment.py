"""Analisis de sentimiento y variable de negatividad para tweets."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import nltk

# Asegurar que lexicon de VADER este disponible
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from lab5.eda import CLASS_NAMES, COLORS


def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def compute_tweet_sentiment(text: str, sia: SentimentIntensityAnalyzer | None = None) -> dict[str, float | int | str]:
    """Calcula las metricas de sentimiento de un tweet individual.
    
    Se evalua sobre el texto original (o preservando puntuacion/emoticones) ya que
    VADER utiliza mayusculas, signos de exclamacion y emoticones para ajustar
    la intensidad del sentimiento.
    """
    if sia is None:
        sia = get_vader_analyzer()
    
    scores = sia.polarity_scores(str(text))
    compound = scores["compound"]
    
    if compound >= 0.05:
        label = "Positivo"
    elif compound <= -0.05:
        label = "Negativo"
    else:
        label = "Neutral"

    # Conteo explicito de palabras usando el lexico VADER
    tokens = str(text).lower().split()
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    for token in tokens:
        clean_tok = token.strip(".,!?\"'()[]{}")
        if clean_tok in sia.lexicon:
            val = sia.lexicon[clean_tok]
            if val > 0:
                pos_count += 1
            elif val < 0:
                neg_count += 1
            else:
                neu_count += 1
        else:
            neu_count += 1
            
    return {
        "negativity": float(scores["neg"]),
        "positivity": float(scores["pos"]),
        "neutrality": float(scores["neu"]),
        "compound": float(compound),
        "sentiment_label": label,
        "pos_word_count": pos_count,
        "neg_word_count": neg_count,
        "neu_word_count": neu_count,
    }


def add_sentiment_features(data: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de sentimiento al DataFrame sin modificar el original."""
    sia = get_vader_analyzer()
    sentiment_records = [compute_tweet_sentiment(txt, sia=sia) for txt in data["text"]]
    sentiment_df = pd.DataFrame(sentiment_records)
    
    result = data.copy()
    for col in sentiment_df.columns:
        result[col] = sentiment_df[col].values
        
    return result


def top_sentiment_tweets(data: pd.DataFrame, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Obtiene los top K tweets mas negativos y mas positivos con su categoria."""
    # Ordenar por compuesto de forma ascendente (mas negativos) y descendente (mas positivos)
    # Secundariamente por negatividad/positividad
    most_negative = (
        data.sort_values(by=["compound", "negativity"], ascending=[True, False])
        .head(top_k)[["id", "target", "text", "negativity", "compound", "sentiment_label"]]
        .reset_index(drop=True)
    )
    most_negative["categoria"] = most_negative["target"].map(CLASS_NAMES)

    most_positive = (
        data.sort_values(by=["compound", "positivity"], ascending=[False, False])
        .head(top_k)[["id", "target", "text", "positivity", "compound", "sentiment_label"]]
        .reset_index(drop=True)
    )
    most_positive["categoria"] = most_positive["target"].map(CLASS_NAMES)

    return most_negative, most_positive


def sentiment_by_class_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula resúmenes estadisticos de sentimiento por clase target."""
    summary = data.groupby("target").agg(
        total_tweets=("id", "count"),
        negatividad_promedio=("negativity", "mean"),
        negatividad_mediana=("negativity", "median"),
        negatividad_std=("negativity", "std"),
        positividad_promedio=("positivity", "mean"),
        compound_promedio=("compound", "mean"),
        tweets_negativos=("sentiment_label", lambda s: (s == "Negativo").sum()),
        tweets_positivos=("sentiment_label", lambda s: (s == "Positivo").sum()),
        tweets_neutrales=("sentiment_label", lambda s: (s == "Neutral").sum()),
    ).reset_index()

    summary["categoria"] = summary["target"].map(CLASS_NAMES)
    summary["pct_negativos"] = (summary["tweets_negativos"] / summary["total_tweets"] * 100).round(2)
    summary["pct_positivos"] = (summary["tweets_positivos"] / summary["total_tweets"] * 100).round(2)
    summary["pct_neutrales"] = (summary["tweets_neutrales"] / summary["total_tweets"] * 100).round(2)

    return summary


def export_sentiment_analysis(data: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Genera y guarda artefactos de analisis de sentimiento (tablas y graficas)."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    most_neg, most_pos = top_sentiment_tweets(data, top_k=10)
    summary = sentiment_by_class_summary(data)

    top_neg_path = tables_dir / "top_negative_tweets.csv"
    top_pos_path = tables_dir / "top_positive_tweets.csv"
    summary_path = tables_dir / "sentiment_summary_by_class.csv"

    most_neg.to_csv(top_neg_path, index=False)
    most_pos.to_csv(top_pos_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Grafica 1: Distribucion de Etiquetas de Sentimiento por Categoria
    dist_fig_path = figures_dir / "sentiment_distribution_by_class.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    sentiment_counts = (
        data.groupby(["target", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )
    sentiment_counts.index = [CLASS_NAMES[t] for t in sentiment_counts.index]
    # Normalizar a porcentaje
    sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

    sentiment_pct.plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=["#E45756", "#79706E", "#54A24B"],  # Negativo (rojo), Neutral (gris), Positivo (verde)
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Distribución porcentual de sentimientos por categoría")
    ax.set_xlabel("Categoría de Tweet")
    ax.set_ylabel("Porcentaje (%)")
    ax.legend(title="Sentimiento")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )
    fig.tight_layout()
    fig.savefig(dist_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Grafica 2: Boxplot de la Variable Negatividad por Categoria
    box_fig_path = figures_dir / "negativity_boxplot.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=data,
        x="target",
        y="negativity",
        hue="target",
        palette={0: COLORS[0], 1: COLORS[1]},
        ax=ax,
        width=0.4,
        legend=False,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([CLASS_NAMES[0], CLASS_NAMES[1]])
    ax.set_title("Comparación del score de Negatividad (VADER) por categoría")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Score de Negatividad (0 a 1)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(box_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "top_negative": top_neg_path,
        "top_positive": top_pos_path,
        "summary": summary_path,
        "dist_figure": dist_fig_path,
        "box_figure": box_fig_path,
    }
