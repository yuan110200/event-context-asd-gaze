from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.classification.rf_model import (
    CLASSIFICATION_THRESHOLD,
    build_rf_pipeline,
    fit_final_train300_calibrated_rf,
    fit_oof_calibrated_rf,
    predict_labels_from_probability,
    predict_positive_probability,
)


class RandomForestModelTest(unittest.TestCase):
    def _training_frame(self):
        rows = []
        for idx in range(12):
            target = 0 if idx < 6 else 1
            base = float(idx if target == 0 else idx + 10)
            rows.append(
                {
                    "subject_id": f"S{idx:02d}",
                    "target": target,
                    "Feature.A": base,
                    "Feature.B": np.nan if idx in {1, 8} else base * 0.5,
                    "Leak.Feature": 9999.0 if target == 1 else -9999.0,
                }
            )
        return pd.DataFrame(rows)

    def test_pipeline_uses_configured_rf_and_imputer_parameters(self):
        pipeline = build_rf_pipeline()
        rf = pipeline.named_steps["model"]

        self.assertEqual(pipeline.named_steps["imputer"].strategy, "median")
        self.assertEqual(rf.n_estimators, 200)
        self.assertEqual(rf.max_depth, 8)
        self.assertEqual(rf.min_samples_leaf, 5)
        self.assertEqual(rf.class_weight, "balanced")
        self.assertEqual(rf.random_state, 42)

    def test_oof_model_fits_only_outer_train_and_outputs_probabilities(self):
        train = self._training_frame().iloc[:10].copy()
        val = self._training_frame().iloc[10:].copy()
        features = ["Feature.A", "Feature.B"]

        model = fit_oof_calibrated_rf(train, features)
        probabilities = predict_positive_probability(model, val, features)
        labels = predict_labels_from_probability(probabilities)

        self.assertEqual(model.cv.n_splits, 4)
        self.assertEqual(len(probabilities), len(val))
        self.assertTrue(np.all((0 <= probabilities) & (probabilities <= 1)))
        self.assertTrue(set(labels).issubset({0, 1}))
        self.assertEqual(CLASSIFICATION_THRESHOLD, 0.50)

    def test_final_model_uses_fixed_five_fold_sigmoid_calibration(self):
        train = self._training_frame()
        features = ["Feature.A", "Feature.B"]

        model = fit_final_train300_calibrated_rf(train, features)

        self.assertEqual(model.method, "sigmoid")
        self.assertEqual(model.cv.n_splits, 5)
        self.assertEqual(model.cv.random_state, 42)

    def test_feature_column_order_is_preserved_and_required(self):
        train = self._training_frame()
        features = ["Feature.B", "Feature.A"]
        model = fit_final_train300_calibrated_rf(train, features)
        shuffled = train[["Feature.A", "subject_id", "Feature.B", "target"]].copy()

        direct = predict_positive_probability(model, train, features)
        shuffled_pred = predict_positive_probability(model, shuffled, features)

        np.testing.assert_allclose(direct, shuffled_pred)
        with self.assertRaisesRegex(ValueError, "missing selected feature columns"):
            predict_positive_probability(model, train.drop(columns=["Feature.A"]), features)

    def test_missing_values_are_imputed_from_training_partition_only(self):
        train = self._training_frame().iloc[:10].copy()
        features = ["Feature.A", "Feature.B"]
        model = fit_oof_calibrated_rf(train, features)

        for calibrated in model.calibrated_classifiers_:
            imputer = calibrated.estimator.named_steps["imputer"]
            self.assertEqual(imputer.strategy, "median")
            self.assertFalse(np.isnan(imputer.statistics_).any())

    def test_fixed_seed_reproducibility(self):
        train = self._training_frame()
        features = ["Feature.A", "Feature.B"]

        model_a = fit_final_train300_calibrated_rf(train, features)
        model_b = fit_final_train300_calibrated_rf(train, features)
        prob_a = predict_positive_probability(model_a, train, features)
        prob_b = predict_positive_probability(model_b, train, features)

        np.testing.assert_allclose(prob_a, prob_b)

    def test_calibration_rejects_missing_class_and_small_oof_class(self):
        train = self._training_frame().iloc[:4].copy()
        with self.assertRaisesRegex(ValueError, "both target classes"):
            fit_oof_calibrated_rf(train.assign(target=0), ["Feature.A"])
        with self.assertRaisesRegex(ValueError, "at least 2 samples per class"):
            fit_oof_calibrated_rf(self._training_frame().iloc[[0, 6, 7]].copy(), ["Feature.A"])

    def test_final_calibration_requires_five_samples_per_class(self):
        train = self._training_frame().iloc[:8].copy()
        with self.assertRaisesRegex(ValueError, "at least 5 samples per class"):
            fit_final_train300_calibrated_rf(train, ["Feature.A"])


if __name__ == "__main__":
    unittest.main()
