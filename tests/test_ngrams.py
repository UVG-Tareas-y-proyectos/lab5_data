from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.ngrams import ngram_frequencies  # noqa: E402


class NgramTests(unittest.TestCase):
    def test_counts_bigrams_and_probabilities(self) -> None:
        texts = pd.Series(["forest fire warning", "forest fire"])
        result = ngram_frequencies(texts, n=2, top_k=10)

        forest_fire = result[result["ngrama"] == "forest fire"].iloc[0]
        self.assertEqual(forest_fire["frecuencia"], 2)
        self.assertAlmostEqual(result["probabilidad"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()

