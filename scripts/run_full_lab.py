"""Ejecucion completa, reproducible y automatizada de todo el Laboratorio 5."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.data import add_preprocessed_columns, load_dataset
from lab5.eda import export_eda
from lab5.modeling import compare_all_models
from lab5.ngrams import export_ngrams
from lab5.predictor import predict_tweet
from lab5.report import write_advance_report, write_final_report
from lab5.sentiment import add_sentiment_features, export_sentiment_analysis, top_sentiment_tweets, sentiment_by_class_summary


def main() -> None:
    print("=== INICIANDO PIPELINE COMPLETO - LABORATORIO 5 ===")
    
    # 1. Cargar y preprocesar datos
    csv_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    print(f"[1/6] Cargando y preprocesando dataset desde: {csv_path.relative_to(PROJECT_ROOT)}")
    raw_data = load_dataset(csv_path)
    data = add_preprocessed_columns(raw_data)
    
    # 2. Análisis de sentimiento y variable negatividad
    print("[2/6] Calculando sentimientos VADER y variable de negatividad...")
    data = add_sentiment_features(data)
    
    output_dir = PROJECT_ROOT / "reports"
    
    # 3. Exportar EDA (Estadísticas, distribuciones, WordClouds, Histogramas)
    print("[3/6] Generando análisis exploratorio de datos (EDA) y nubes de palabras...")
    eda_paths = export_eda(data, output_dir)
    
    # 4. Exportar N-gramas
    print("[4/6] Calculando frecuencias y probabilidades de unigramas y bigramas...")
    frequencies, _ = export_ngrams(data, output_dir)
    
    # 5. Exportar Análisis de Sentimiento
    print("[5/6] Exportando análisis de sentimiento, top tweets positivos y negativos...")
    sent_paths = export_sentiment_analysis(data, output_dir)
    top_neg, top_pos = top_sentiment_tweets(data, top_k=10)
    sent_summary = sentiment_by_class_summary(data)
    
    # 6. Comparación de Múltiples Clasificadores y Evaluación de Negatividad
    print("[6/6] Entrenando clasificadores (Regresión Logística, Naive Bayes, LinearSVC, Random Forest)...")
    df_base, df_impact, trained_pipelines = compare_all_models(data, output_dir)
    
    # Generar informes
    advance_path = write_advance_report(data, frequencies, {"accuracy": df_base.iloc[0]["accuracy"], "precision": df_base.iloc[0]["precision"], "recall": df_base.iloc[0]["recall"], "f1": df_base.iloc[0]["f1"], "roc_auc": df_base.iloc[0]["roc_auc"], "training_rows": int(len(data)*0.8), "test_rows": int(len(data)*0.2), "test_fraction": 0.2}, output_dir)
    final_report_path = write_final_report(
        data=data,
        frequencies=frequencies,
        df_base=df_base,
        df_impact=df_impact,
        top_neg=top_neg,
        top_pos=top_pos,
        sent_summary=sent_summary,
        output_dir=output_dir,
    )
    
    print("\n=== EJECUCIÓN COMPLETADA EXITOSAMENTE ===")
    print(f"Informe de avance: {advance_path.relative_to(PROJECT_ROOT)}")
    print(f"Informe final completo: {final_report_path.relative_to(PROJECT_ROOT)}")
    
    # Prueba rapida de la funcion predictor
    print("\n--- Demostración de Clasificación con predict_tweet() ---")
    sample_text = "Emergency team deployed after heavy earthquake damages buildings in California!"
    res = predict_tweet(sample_text, data_context=data)
    print(f"Tweet de entrada: \"{res['raw_text']}\"")
    print(f"Predicción: {res['prediction_label']} (Confianza: {res['confidence']:.2%})")
    print(f"Sentimiento: {res['sentiment_label']} | Negatividad: {res['negativity_score']}")


if __name__ == "__main__":
    main()
