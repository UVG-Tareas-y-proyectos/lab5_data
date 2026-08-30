from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.sentiment import add_sentiment_features, compute_tweet_sentiment, top_sentiment_tweets


class SentimentTests(unittest.TestCase):
    def test_compute_tweet_sentiment_positive_and_negative(self) -> None:
        pos_res = compute_tweet_sentiment("This is wonderful and amazing news!")
        self.assertEqual(pos_res["sentiment_label"], "Positivo")
        self.assertGreater(pos_res["compound"], 0)

        neg_res = compute_tweet_sentiment("Terrible disaster and horrible death everywhere!")
        self.assertEqual(neg_res["sentiment_label"], "Negativo")
        self.assertGreater(neg_res["negativity"], 0)

    def test_add_sentiment_features_columns(self) -> None:
        df = pd.DataFrame({"id": [1], "text": ["Test tweet"], "target": [0]})
        df_sent = add_sentiment_features(df)
        self.assertIn("negativity", df_sent.columns)
        self.assertIn("compound", df_sent.columns)
        self.assertIn("sentiment_label", df_sent.columns)

    def test_top_sentiment_tweets(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "text": ["Horrible massacre disaster", "Lovely happy sunshine", "Just normal text"],
                "target": [1, 0, 0],
            }
        )
        df_sent = add_sentiment_features(df)
        neg, pos = top_sentiment_tweets(df_sent, top_k=2)
        self.assertEqual(len(neg), 2)
        self.assertEqual(len(pos), 2)
        self.assertEqual(neg.iloc[0]["id"], 1)
        self.assertEqual(pos.iloc[0]["id"], 2)


if __name__ == "__main__":
    unittest.main()
