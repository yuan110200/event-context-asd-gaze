from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.features.event_features import EventDefinition
from asd_gaze.features.feature_extraction import extract_subject_report
from asd_gaze.features.rqa_features import (
    calculate_rqa_metrics,
    collect_event_gaze_coords_for_rqa,
)
from asd_gaze.features.summary_features import calculate_overall_statistics


class RqaFeaturesTest(unittest.TestCase):
    def test_collect_event_gaze_coords_filters_and_downsamples(self):
        gaze = pd.DataFrame(
            {
                "Frame": [1, 2, 3, 4, 5, 6],
                "Gaze_AOI": ["Other"] * 6,
                "X": [10, 20, 0, None, 50, 60],
                "Y": [10, 20, 30, 40, -1, 60],
            }
        )
        event = EventDefinition("E01", 1, 6, ("Other",), ())
        self.assertEqual(collect_event_gaze_coords_for_rqa(gaze, event), [[10.0, 10.0], [60.0, 60.0]])

    def test_rqa_metrics_match_canonical_formula_for_full_recurrence(self):
        coords = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
        metrics = calculate_rqa_metrics(coords, epsilon=80)
        self.assertEqual(metrics["RQA_RR"], 1.0)
        self.assertEqual(metrics["RQA_CORM"], 50.0)

    def test_summary_exposes_final_rqa_fields(self):
        coords = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
        overall = calculate_overall_statistics([{"event_id": "E01"}], coords)
        functional = overall["Functional_Analysis"]
        self.assertEqual(functional["Avg_Gaze_Recurrence_RQA_RR"], 1.0)
        self.assertEqual(functional["Avg_Gaze_CORM_RQA_CORM"], 50.0)

    def test_subject_report_excludes_e12_from_rqa_collection(self):
        gaze = pd.DataFrame(
            {
                "Frame": list(range(1, 11)),
                "Gaze_AOI": ["Other"] * 10,
                "X": list(range(10)),
                "Y": [0] + [1] * 9,
            }
        )
        events = [
            EventDefinition("E01", 1, 10, ("Other",), ()),
            EventDefinition("E12", 1, 10, ("Other",), ()),
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            gaze.to_csv(handle.name, index=False)
            report = extract_subject_report(handle.name, events)

        functional = report["overall_statistics"]["Functional_Analysis"]
        self.assertEqual(functional["Avg_Gaze_Recurrence_RQA_RR"], 1.0)
        self.assertEqual(functional["Avg_Gaze_CORM_RQA_CORM"], 50.0)


if __name__ == "__main__":
    unittest.main()
