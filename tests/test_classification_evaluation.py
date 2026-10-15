from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.classification.evaluation import (
    compute_binary_metrics,
    compute_confidence_intervals,
    evaluate_independent_test,
    load_model_bundle,
    load_selected_feature_columns,
    run_oof_protocol,
    save_model_bundle,
    save_selected_feature_columns,
    train_final_and_evaluate_test,
    validate_feature_order,
)
from asd_gaze.classification.feature_selection import SelectionConfig
from asd_gaze.classification.rf_model import RandomForestConfig, fit_final_train300_calibrated_rf, predict_positive_probability


class ClassificationEvaluationTest(unittest.TestCase):
    def _frame(self, prefix="S", n_per_class=6):
        rows = []
        for target in [0, 1]:
            for i in range(n_per_class):
                idx = target * n_per_class + i
                signal = float(i if target == 0 else i + 20)
                rows.append(
                    {
                        "subject_id": f"{prefix}{idx:02d}",
                        "target": target,
                        "Feature.Signal": signal,
                        "Feature.Noisy": signal + (i % 2),
                    }
                )
        return pd.DataFrame(rows)

    def test_metrics_match_configured_threshold_outputs(self):
        metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4], label="demo")

        self.assertEqual(metrics["model"], "demo")
        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["ACC"], 0.5)
        self.assertEqual(metrics["SEN"], 0.5)
        self.assertEqual(metrics["SPE"], 0.5)
        self.assertAlmostEqual(metrics["F1"], 0.5)
        self.assertEqual(metrics["AUC"], 0.75)

    def test_confidence_intervals_use_configured_methods(self):
        ci = compute_confidence_intervals([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4], replicates=100, seed=7)
        self.assertEqual(ci["bootstrap_replicates"], 100)
        self.assertEqual(ci["ACC_ci_method"], "exact Clopper-Pearson")
        self.assertEqual(ci["AUC_ci_method"], "subject bootstrap, 100 replicates")
        self.assertLessEqual(ci["AUC_lo"], ci["AUC_hi"])

    def test_model_bundle_roundtrip_preserves_predictions_and_features(self):
        train = self._frame()
        features = ["Feature.Signal", "Feature.Noisy"]
        config = RandomForestConfig(n_estimators=10, final_calibration_splits=2)
        model = fit_final_train300_calibrated_rf(train, features, config=config)
        before = predict_positive_probability(model, train, features)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            save_model_bundle(model, features, path, metadata={"kind": "unit-test"})
            loaded_model, loaded_features, metadata = load_model_bundle(path)

        after = predict_positive_probability(loaded_model, train, loaded_features)
        self.assertEqual(loaded_features, features)
        self.assertEqual(metadata["kind"], "unit-test")
        np.testing.assert_allclose(before, after)

    def test_selected_feature_columns_are_order_checked(self):
        features = ["B", "A"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected_features.csv"
            save_selected_feature_columns(features, path)
            loaded = load_selected_feature_columns(path)

        self.assertEqual(loaded, features)
        validate_feature_order(["B", "A"], features)
        with self.assertRaisesRegex(ValueError, "selected feature order"):
            validate_feature_order(["A", "B"], features)

    def test_independent_test_entry_fits_train_only_and_rejects_overlap(self):
        train = self._frame("TR")
        test = self._frame("TE")
        features = ["Feature.Signal", "Feature.Noisy"]
        config = RandomForestConfig(n_estimators=10, final_calibration_splits=2)

        model, predictions, metrics = train_final_and_evaluate_test(train, test, features, rf_config=config)

        self.assertEqual(len(predictions), len(test))
        self.assertEqual(metrics.iloc[0]["model"], "rf_top40")
        self.assertTrue(hasattr(model, "predict_proba"))
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            train_final_and_evaluate_test(train, train.iloc[:2].copy(), features, rf_config=config)

    def test_oof_protocol_selects_and_trains_with_fold_isolation(self):
        table = self._frame("OOF", n_per_class=6)
        fold_assignments = pd.DataFrame(
            {
                "subject_id": list(table["subject_id"].iloc[[0, 1, 2, 6, 7, 8]]) +
                               list(table["subject_id"].iloc[[3, 4, 5, 9, 10, 11]]) +
                               list(table["subject_id"].iloc[[0, 1, 2, 6, 7, 8]]) +
                               list(table["subject_id"].iloc[[3, 4, 5, 9, 10, 11]]),
                "fold": [0] * 12 + [1] * 12,
                "split": ["val"] * 6 + ["train"] * 6 + ["train"] * 6 + ["val"] * 6,
            }
        )
        selection_config = SelectionConfig(min_valid_n_per_group=2, fdr_alpha=1.0, top_n_features=1)
        rf_config = RandomForestConfig(n_estimators=10, final_calibration_splits=2)

        predictions, metrics, selected_by_fold = run_oof_protocol(
            table,
            fold_assignments,
            selection_config=selection_config,
            rf_config=rf_config,
        )

        self.assertEqual(set(predictions["fold"]), {0, 1})
        self.assertEqual(set(metrics["fold"]), {0, 1})
        self.assertEqual(len(selected_by_fold[0]), 1)
        self.assertEqual(len(selected_by_fold[1]), 1)
        self.assertTrue(set(selected_by_fold[0] + selected_by_fold[1]).issubset({"Feature.Signal", "Feature.Noisy"}))

        bad_folds = fold_assignments.copy()
        bad_folds = pd.concat([bad_folds, pd.DataFrame([{"subject_id": table.loc[0, "subject_id"], "fold": 1, "split": "val"}])])
        with self.assertRaisesRegex(ValueError, "exactly one OOF validation fold"):
            run_oof_protocol(table, bad_folds, selection_config=selection_config, rf_config=rf_config)

    def test_oof_protocol_rejects_missing_validation_assignment(self):
        table = self._frame("MISS", n_per_class=6)
        assignments = pd.DataFrame({
            "subject_id": table["subject_id"].iloc[:-1],
            "fold": [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
            "split": ["val", "train", "val", "train", "train", "train", "val", "train", "val", "train", "train"],
        })
        with self.assertRaisesRegex(ValueError, "cover every feature-table subject"):
            run_oof_protocol(table, assignments)


if __name__ == "__main__":
    unittest.main()
