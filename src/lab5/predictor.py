"""Modulo de prediccion de tweets individuales para el usuario final."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from lab5.data import clean_tweet
from lab5.eda import CLASS_NAMES
from lab5.sentiment import compute_tweet_sentiment


def train_production_model(data: pd.DataFrame) -> tuple[Pipeline, bool]:
    """Entrena el mejor clasificador sobre el dataset completo para inferencia en produccion."""
    from lab5.modeling import build_text_and_feature_pipeline, get_classifiers

    classifiers = get_classifiers()
    best_clf = classifiers["Regresión Logística"]

    pipeline = build_text_and_feature_pipeline("Regresión Logística", best_clf, ["negativity"])
    X = data[["text_clean", "negativity"]]
    pipeline.fit(X, data["target"])
    return pipeline, True


_GLOBAL_MODEL: Pipeline | None = None
_GLOBAL_USES_NEG: bool = True


def get_default_model(data: pd.DataFrame | None = None) -> tuple[Pipeline, bool]:
    """Obtiene o inicializa el modelo global de produccion."""
    global _GLOBAL_MODEL, _GLOBAL_USES_NEG
    if _GLOBAL_MODEL is None:
        if data is None:
            from lab5.data import add_preprocessed_columns, load_dataset
            from lab5.sentiment import add_sentiment_features
            raw_data = load_dataset("data/raw/train.csv")
            data = add_sentiment_features(add_preprocessed_columns(raw_data))
        _GLOBAL_MODEL, _GLOBAL_USES_NEG = train_production_model(data)
    return _GLOBAL_MODEL, _GLOBAL_USES_NEG


def predict_tweet(
    tweet_text: str,
    model: Pipeline | None = None,
    uses_negativity: bool = True,
    data_context: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Clasifica un tweet ingresado por el usuario sin preprocesar."""
    if model is None:
        model, uses_negativity = get_default_model(data_context)

    text_clean = clean_tweet(tweet_text)
    sentiment = compute_tweet_sentiment(tweet_text)

    if uses_negativity:
        X_input = pd.DataFrame(
            [{"text_clean": text_clean, "negativity": sentiment["negativity"]}]
        )
    else:
        X_input = pd.Series([text_clean])

    prediction_code = int(model.predict(X_input)[0])
    prediction_label = CLASS_NAMES[prediction_code]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        confidence = float(probs[prediction_code])
        prob_disaster = float(probs[1])
    elif hasattr(model, "decision_function"):
        dec = float(model.decision_function(X_input)[0])
        prob_disaster = 1.0 / (1.0 + float(np.exp(-dec)))
        confidence = prob_disaster if prediction_code == 1 else (1.0 - prob_disaster)
    else:
        confidence = 1.0
        prob_disaster = float(prediction_code)

    return {
        "raw_text": tweet_text,
        "clean_text": text_clean,
        "prediction_code": prediction_code,
        "prediction_label": prediction_label,
        "confidence": round(confidence, 4),
        "probability_disaster": round(prob_disaster, 4),
        "sentiment_label": sentiment["sentiment_label"],
        "negativity_score": round(sentiment["negativity"], 4),
        "compound_score": round(sentiment["compound"], 4),
    }


if __name__ == "__main__":
    test_tweets = [
        "Huge wildfire spreading rapidly in California! Emergency mandatory evacuation issued!",
        "Just having a coffee and watching the rain outside. Hope everyone has a great day!",
        "My new song is absolute fire bro! Check out the link below",
    ]
    print("--- Demostración de Clasificación de Tweets ---")
    for tw in test_tweets:
        res = predict_tweet(tw)
        print(f"\nTweet: \"{res['raw_text']}\"")
        print(f"Clasificación: {res['prediction_label']} (Confianza: {res['confidence']:.2%})")
        print(f"Sentimiento: {res['sentiment_label']} | Negatividad: {res['negativity_score']}")
