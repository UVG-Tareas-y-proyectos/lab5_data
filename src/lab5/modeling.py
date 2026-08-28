"""Modelo preliminar TF-IDF con regresion logistica."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from lab5.eda import CLASS_NAMES


RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Crea el clasificador preliminar con unigramas y bigramas TF-IDF."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-z0-9]+\b",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def grouped_holdout_indices(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Genera una particion 80/20 estratificada y agrupada por texto limpio."""

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_indices, test_indices = next(
        splitter.split(
            data["text_clean"],
            data["target"],
            groups=data["text_clean"],
        )
    )
    return train_indices, test_indices


def most_informative_features(model: Pipeline, top_k: int = 20) -> pd.DataFrame:
    """Extrae los terminos con mayor asociacion lineal a cada clase."""

    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    terms = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    negative_indices = np.argsort(coefficients)[:top_k]
    positive_indices = np.argsort(coefficients)[-top_k:][::-1]
    rows = [
        {
            "termino": terms[index],
            "coeficiente": float(coefficients[index]),
            "asociacion": CLASS_NAMES[0],
        }
        for index in negative_indices
    ]
    rows.extend(
        {
            "termino": terms[index],
            "coeficiente": float(coefficients[index]),
            "asociacion": CLASS_NAMES[1],
        }
        for index in positive_indices
    )
    return pd.DataFrame(rows)


def train_and_export(
    data: pd.DataFrame, output_dir: str | Path
) -> tuple[dict[str, float | int], dict[str, Path]]:
    """Entrena, evalua y exporta resultados del modelo preliminar."""

    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_indices, test_indices = grouped_holdout_indices(data)
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]

    model = build_pipeline()
    model.fit(train["text_clean"], train["target"])
    predictions = model.predict(test["text_clean"])
    probabilities = model.predict_proba(test["text_clean"])[:, 1]

    metrics: dict[str, float | int] = {
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_fraction": float(len(test) / len(data)),
        "accuracy": float(accuracy_score(test["target"], predictions)),
        "precision": float(precision_score(test["target"], predictions)),
        "recall": float(recall_score(test["target"], predictions)),
        "f1": float(f1_score(test["target"], predictions)),
        "roc_auc": float(roc_auc_score(test["target"], probabilities)),
    }
    metrics_path = metrics_dir / "preliminary_model_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = pd.DataFrame(
        classification_report(
            test["target"],
            predictions,
            labels=[0, 1],
            target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_path = tables_dir / "classification_report.csv"
    report.to_csv(report_path)

    features = most_informative_features(model)
    features_path = tables_dir / "informative_features.csv"
    features.to_csv(features_path, index=False)

    matrix = confusion_matrix(test["target"], predictions, labels=[0, 1])
    figure_path = figures_dir / "preliminary_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
    ax.set_title("Matriz de confusión - modelo preliminar")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return metrics, {
        "metrics": metrics_path,
        "classification_report": report_path,
        "features": features_path,
        "confusion_matrix": figure_path,
    }
