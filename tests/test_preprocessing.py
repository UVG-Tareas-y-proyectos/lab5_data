from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab5.data import clean_tweet  # noqa: E402


class CleanTweetTests(unittest.TestCase):
    def test_removes_noise_and_keeps_useful_content(self) -> None:
        text = "HEY @rescue! #ForestFire near 911 https://t.co/example :("
        self.assertEqual(clean_tweet(text), "hey forestfire near 911")

    def test_decodes_html_and_normalizes_case(self) -> None:
        self.assertEqual(clean_tweet("FLOOD &amp; Storm"), "flood storm")


if __name__ == "__main__":
    unittest.main()
