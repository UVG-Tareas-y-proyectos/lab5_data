from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.modeling import (
    build_text_and_feature_pipeline,
    build_text_pipeline,
    evaluate_model_pipeline,
    get_classifiers,
)


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "id": list(range(1, 11)),
                "text_clean": [
                    "fire disaster emergency",
                    "forest fire burning",
                    "wildfire evacuation highway",
                    "storm flood damage",
                    "bomb explosion hazard",
                    "happy sunny day",
                    "coffee with friends",
                    "listening to good music",
                    "great movie release",
                    "enjoying nice weekend",
                ],
                "negativity": [0.8, 0.7, 0.9, 0.6, 0.85, 0.0, 0.1, 0.0, 0.05, 0.0],
                "target": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            }
        )

    def test_get_classifiers_returns_expected_models(self) -> None:
        clfs = get_classifiers()
        self.assertIn("Regresión Logística", clfs)
        self.assertIn("Naive Bayes (MultinomialNB)", clfs)
        self.assertIn("SVM Lineal (LinearSVC)", clfs)
        self.assertIn("Random Forest", clfs)

    def test_evaluate_model_pipeline_baseline_and_combined(self) -> None:
        train = self.data.iloc[:8]
        test = self.data.iloc[8:]

        clfs = get_classifiers()
        pipe_base = build_text_pipeline("Regresión Logística", clfs["Regresión Logística"])
        metrics_base = evaluate_model_pipeline(pipe_base, train, test, is_combined=False)
        self.assertIn("f1", metrics_base)
        self.assertIn("roc_auc", metrics_base)

        pipe_neg = build_text_and_feature_pipeline(
            "Regresión Logística", clfs["Regresión Logística"], ["negativity"]
        )
        metrics_neg = evaluate_model_pipeline(
            pipe_neg, train, test, is_combined=True, feature_cols=["negativity"]
        )
        self.assertIn("f1", metrics_neg)


if __name__ == "__main__":
    unittest.main()
