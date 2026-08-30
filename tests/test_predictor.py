from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.predictor import predict_tweet, train_production_model


class PredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "id": list(range(1, 11)),
                "text": [
                    "Huge forest fire near town!",
                    "Wildfire emergency evacuation",
                    "Building explosion kills people",
                    "Massive flood destroying houses",
                    "Severe earthquake damages city",
                    "Love watching rain while drinking tea",
                    "Awesome party tonight with friends",
                    "Great song playing on radio",
                    "Good morning everyone have a nice day",
                    "Delicious lunch at restaurant",
                ],
                "text_clean": [
                    "huge forest fire near town",
                    "wildfire emergency evacuation",
                    "building explosion kills people",
                    "massive flood destroying houses",
                    "severe earthquake damages city",
                    "love watching rain drinking tea",
                    "awesome party tonight friends",
                    "great song playing radio",
                    "good morning everyone nice day",
                    "delicious lunch restaurant",
                ],
                "negativity": [0.6, 0.7, 0.8, 0.5, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0],
                "target": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            }
        )
        self.model, self.uses_neg = train_production_model(self.data)

    def test_predict_tweet_returns_valid_structure(self) -> None:
        res = predict_tweet(
            "Emergency wildfire evacuation!",
            model=self.model,
            uses_negativity=self.uses_neg,
        )
        self.assertIn("prediction_code", res)
        self.assertIn("prediction_label", res)
        self.assertIn("confidence", res)
        self.assertIn("sentiment_label", res)
        self.assertIn(res["prediction_code"], [0, 1])


if __name__ == "__main__":
    unittest.main()
