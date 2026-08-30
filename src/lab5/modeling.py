"""Modelos de clasificacion, comparativa multi-algoritmo y evaluacion de la variable de negatividad."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

from lab5.eda import CLASS_NAMES, COLORS

RANDOM_STATE = 42


def get_classifiers() -> dict[str, object]:
    """Retorna un diccionario con las instancias de los clasificadores a comparar."""
    return {
        "Regresión Logística": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1_000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        "Naive Bayes (MultinomialNB)": MultinomialNB(alpha=0.5),
        "SVM Lineal (LinearSVC)": LinearSVC(
            C=0.5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=2_000,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_text_pipeline(classifier_name: str, classifier_obj: object) -> Pipeline:
    """Crea un pipeline clasificador solo con texto TF-IDF."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-z0-9]+\b",
                ),
            ),
            ("classifier", clone(classifier_obj)),
        ]
    )


def build_text_and_feature_pipeline(
    classifier_name: str, classifier_obj: object, feature_cols: list[str]
) -> Pipeline:
    """Crea un pipeline que combina texto TF-IDF con variables numericas (ej. negatividad)."""
    scaler = MinMaxScaler() if "Naive Bayes" in classifier_name else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-z0-9]+\b",
                ),
                "text_clean",
            ),
            (
                "num",
                scaler,
                feature_cols,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", clone(classifier_obj)),
        ]
    )


def grouped_holdout_indices(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Genera una partición 80/20 estratificada y agrupada por texto limpio."""
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


def evaluate_model_pipeline(
    pipeline: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    is_combined: bool = False,
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    """Entrena y evalua un pipeline devolviendo sus metricas principales."""
    if is_combined and feature_cols:
        X_train = train[["text_clean"] + feature_cols]
        X_test = test[["text_clean"] + feature_cols]
    else:
        X_train = train["text_clean"]
        X_test = test["text_clean"]

    y_train = train["target"]
    y_test = test["target"]

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        probabilities = pipeline.decision_function(X_test)
        probabilities = (probabilities - probabilities.min()) / (
            probabilities.max() - probabilities.min() + 1e-8
        )
    else:
        probabilities = predictions.astype(float)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def compare_all_models(
    data: pd.DataFrame, output_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Entrena y compara todos los modelos tanto solo con texto como incluyendo negatividad."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_indices, test_indices = grouped_holdout_indices(data)
    train = data.iloc[train_indices].copy()
    test = data.iloc[test_indices].copy()

    classifiers = get_classifiers()
    baseline_results = []
    with_negativity_results = []
    trained_pipelines = {}

    for name, clf_obj in classifiers.items():
        # Model 1: Solo texto
        pipe_base = build_text_pipeline(name, clf_obj)
        metrics_base = evaluate_model_pipeline(pipe_base, train, test, is_combined=False)
        metrics_base["modelo"] = name
        metrics_base["features"] = "Solo Texto (TF-IDF)"
        baseline_results.append(metrics_base)
        trained_pipelines[f"{name}_base"] = pipe_base

        # Model 2: Texto + Negatividad
        pipe_neg = build_text_and_feature_pipeline(name, clf_obj, ["negativity"])
        metrics_neg = evaluate_model_pipeline(
            pipe_neg, train, test, is_combined=True, feature_cols=["negativity"]
        )
        metrics_neg["modelo"] = name
        metrics_neg["features"] = "Texto TF-IDF + Negatividad"
        with_negativity_results.append(metrics_neg)
        trained_pipelines[f"{name}_neg"] = pipe_neg

    df_base = pd.DataFrame(baseline_results)
    df_neg = pd.DataFrame(with_negativity_results)

    # Comparativa de impacto de la negatividad
    impact_rows = []
    for name in classifiers.keys():
        row_base = df_base[df_base["modelo"] == name].iloc[0]
        row_neg = df_neg[df_neg["modelo"] == name].iloc[0]
        impact_rows.append(
            {
                "Modelo": name,
                "F1 (Solo Texto)": round(row_base["f1"], 4),
                "F1 (+ Negatividad)": round(row_neg["f1"], 4),
                "Diferencia F1": round(row_neg["f1"] - row_base["f1"], 4),
                "ROC-AUC (Solo Texto)": round(row_base["roc_auc"], 4),
                "ROC-AUC (+ Negatividad)": round(row_neg["roc_auc"], 4),
                "Diferencia ROC-AUC": round(row_neg["roc_auc"] - row_base["roc_auc"], 4),
            }
        )
    df_impact = pd.DataFrame(impact_rows)

    # Exportar Tablas
    df_base.to_csv(tables_dir / "model_comparison_baseline.csv", index=False)
    df_neg.to_csv(tables_dir / "model_comparison_with_negativity.csv", index=False)
    df_impact.to_csv(tables_dir / "negativity_impact_comparison.csv", index=False)

    # Generar Grafica Comparativa de F1-Score
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classifiers))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df_base["f1"],
        width,
        label="Solo Texto (TF-IDF)",
        color="#4C78A8",
    )
    bars2 = ax.bar(
        x + width / 2,
        df_neg["f1"],
        width,
        label="Texto + Negatividad",
        color="#E45756",
    )

    ax.set_ylabel("F1-Score")
    ax.set_title("Comparación del F1-Score entre modelos y efecto de la Negatividad")
    ax.set_xticks(x)
    ax.set_xticklabels(list(classifiers.keys()), rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)

    fig.tight_layout()
    comp_fig_path = figures_dir / "model_comparison_f1.png"
    fig.savefig(comp_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Seleccionar el MEJOR modelo general
    all_combined = pd.concat([df_base, df_neg], ignore_index=True)
    best_row = all_combined.sort_values(by=["f1", "roc_auc"], ascending=[False, False]).iloc[0]

    # Generar Matriz de confusion del MEJOR modelo
    best_name = best_row["modelo"]
    best_features = best_row["features"]
    is_neg = "Negatividad" in best_features
    best_pipe_key = f"{best_name}_neg" if is_neg else f"{best_name}_base"
    best_pipeline = trained_pipelines[best_pipe_key]

    if is_neg:
        X_test = test[["text_clean", "negativity"]]
    else:
        X_test = test["text_clean"]

    best_preds = best_pipeline.predict(X_test)
    matrix = confusion_matrix(test["target"], best_preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
    ax.set_title(f"Matriz de confusión - Mejor Modelo\n({best_name} | {best_features})")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    fig.tight_layout()
    best_cm_path = figures_dir / "best_model_confusion_matrix.png"
    fig.savefig(best_cm_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Exportar el resumen del mejor modelo a JSON
    best_metrics = {
        "modelo": best_name,
        "features": best_features,
        "f1": float(best_row["f1"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "accuracy": float(best_row["accuracy"]),
        "roc_auc": float(best_row["roc_auc"]),
    }
    (metrics_dir / "best_model_summary.json").write_text(
        json.dumps(best_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return df_base, df_impact, trained_pipelines
