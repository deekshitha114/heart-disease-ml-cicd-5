import unittest
import json
import os


class TestMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(os.path.exists("heart_disease_model.pkl"))

    def test_metrics_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy(self):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        self.assertGreaterEqual(metrics["accuracy"], 0.80)


if __name__ == "__main__":
    unittest.main()
