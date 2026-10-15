from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.classification.feature_selection import (
    CORR_THRESHOLD,
    SelectionConfig,
    apply_bh_fdr_correction,
    compute_univariate_feature_stats,
    filter_fdr_effect_candidates,
    rank_and_score_features,
    select_oof_fold_features,
    select_features_from_training_groups,
    prune_correlated_features,
)


class FeatureSelectionTest(unittest.TestCase):
    def test_mann_whitney_fdr_effect_ranking_and_correlation_pruning(self):
        td = pd.DataFrame(
            {
                "id": [f"TD{i:02d}" for i in range(6)],
                "E01.Strong": [10, 11, 12, 13, 14, 15],
                "E02.Redundant": [20, 22, 24, 26, 28, 30],
                "E03.Weak": [1, 2, 1, 2, 1, 2],
            }
        )
        asd = pd.DataFrame(
            {
                "id": [f"ASD{i:02d}" for i in range(6)],
                "E01.Strong": [1, 2, 3, 4, 5, 6],
                "E02.Redundant": [2, 4, 6, 8, 10, 12],
                "E03.Weak": [1, 1, 2, 2, 1, 2],
            }
        )
        config = SelectionConfig(min_valid_n_per_group=6, min_abs_effect_size=0.35, top_n_features=2)

        selected, ranked, removed_pairs, raw_stats = select_features_from_training_groups(td, asd, config)

        self.assertIn("E01.Strong", raw_stats["Original_Feature_Key"].tolist())
        self.assertEqual(selected["Original_Feature_Key"].tolist(), ["E01.Strong"])
        self.assertEqual(len(removed_pairs), 1)
        self.assertEqual(removed_pairs[0][0], "E02.Redundant")
        self.assertEqual(removed_pairs[0][1], "E01.Strong")
        self.assertEqual(ranked.iloc[0]["Final_Selected_Reason"], "top_n_after_fdr_effect_corr")

    def test_individual_selection_steps_keep_original_rules(self):
        td = pd.DataFrame({"id": ["t1", "t2", "t3"], "Overall.A": [9, 10, 11], "E01.B": [1, 1, 1]})
        asd = pd.DataFrame({"id": ["a1", "a2", "a3"], "Overall.A": [1, 2, 3], "E01.B": [1, 1, 1]})

        stats = compute_univariate_feature_stats(td, asd, min_valid_n_per_group=3)
        corrected = apply_bh_fdr_correction(stats, alpha=1.0)
        candidates = filter_fdr_effect_candidates(corrected, min_abs_effect_size=0.35)
        ranked = rank_and_score_features(candidates, top_n_features=1)

        self.assertEqual(ranked.iloc[0]["Original_Feature_Key"], "Overall.A")
        self.assertTrue(bool(ranked.iloc[0]["Is_Macro_Metric"]))
        self.assertEqual(int(ranked.iloc[0]["Selected"]), 1)

    def test_oof_selection_uses_outer_training_subjects_only(self):
        rows = []
        for idx, target in enumerate([0, 0, 0, 1, 1, 1]):
            sid = f"TR{idx}"
            rows.append(
                {
                    "subject_id": sid,
                    "target": target,
                    "Train.Signal": 10 + idx if target == 0 else idx,
                    "Val.OnlySignal": None,
                }
            )
        rows.extend(
            [
                {"subject_id": "VA0", "target": 0, "Train.Signal": 3, "Val.OnlySignal": 100},
                {"subject_id": "VA1", "target": 1, "Train.Signal": 4, "Val.OnlySignal": -100},
            ]
        )
        feature_table = pd.DataFrame(rows)
        fold_assignments = pd.DataFrame(
            {
                "fold": [0] * 8,
                "split": ["train"] * 6 + ["val", "val"],
                "subject_id": feature_table["subject_id"].tolist(),
            }
        )
        config = SelectionConfig(min_valid_n_per_group=3, min_abs_effect_size=0.35, fdr_alpha=1.0, top_n_features=5)

        selected, _, _, raw_stats = select_oof_fold_features(feature_table, fold_assignments, fold=0, config=config)

        self.assertIn("Train.Signal", raw_stats["Original_Feature_Key"].tolist())
        self.assertNotIn("Val.OnlySignal", raw_stats["Original_Feature_Key"].tolist())
        self.assertEqual(selected["Original_Feature_Key"].tolist(), ["Train.Signal"])

    def test_default_spearman_threshold_is_point_nine_and_strictly_greater(self):
        ranked = pd.DataFrame(
            {
                "Original_Feature_Key": ["A", "B"],
                "Final_Score": [2.0, 1.0],
                "Abs_Cohen_D": [2.0, 1.0],
                "P_Value_Raw": [0.001, 0.002],
                "Is_Macro_Metric": [False, False],
            }
        )
        td_equal = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [1, 2, 3, 4, 0]})
        asd_equal = pd.DataFrame({"A": [6, 7, 8, 9, 10], "B": [6, 7, 8, 9, 5]})
        td_above = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [1, 2, 3, 4, 6]})
        asd_above = pd.DataFrame({"A": [6, 7, 8, 9, 10], "B": [7, 8, 9, 10, 11]})

        kept_equal, removed_equal = prune_correlated_features(ranked, td_equal, asd_equal)
        kept_above, removed_above = prune_correlated_features(ranked, td_above, asd_above)

        self.assertEqual(CORR_THRESHOLD, 0.90)
        self.assertEqual(kept_equal["Original_Feature_Key"].tolist(), ["A", "B"])
        self.assertEqual(removed_equal, [])
        self.assertEqual(kept_above["Original_Feature_Key"].tolist(), ["A"])
        self.assertEqual(removed_above[0][2], 1.0)


if __name__ == "__main__":
    unittest.main()
