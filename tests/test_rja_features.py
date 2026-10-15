from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.features.event_features import (
    EventDefinition,
    analyze_event,
    calculate_perseverative_gaze_ratio,
    classify_rja_path_pattern,
)
from asd_gaze.features.summary_features import calculate_overall_statistics


class RjaFeaturesTest(unittest.TestCase):
    def test_classifies_checkback_after_target(self):
        path_facts = {
            "simplified_gaze_path": ["AOI_Stars", "AOI_Teacher_Face"],
            "targets_met_details": {
                "AOI_Stars": {"met": True, "duration_sec": 0.2},
                "AOI_Teacher_Face": {"met": True, "duration_sec": 0.6},
                "AOI_Teacher_Hands": {"met": False, "duration_sec": 0.0},
            },
            "post_target_face_visit_sec": 0.6,
        }
        self.assertEqual(
            classify_rja_path_pattern(path_facts, ["AOI_Stars"]),
            "SUCCESS_CHECKBACK_PATTERN",
        )

    def test_classifies_brief_check_after_target(self):
        path_facts = {
            "simplified_gaze_path": ["AOI_Stars", "AOI_Teacher_Face"],
            "targets_met_details": {
                "AOI_Stars": {"met": True, "duration_sec": 0.2},
                "AOI_Teacher_Face": {"met": True, "duration_sec": 0.2},
            },
            "post_target_face_visit_sec": 0.2,
        }
        self.assertEqual(
            classify_rja_path_pattern(path_facts, ["AOI_Stars"]),
            "SUCCESS_OTHER_BRIEF_CHECK",
        )

    def test_classifies_failed_significant_face_fixation(self):
        path_facts = {
            "simplified_gaze_path": ["AOI_Teacher_Face"],
            "targets_met_details": {
                "AOI_Stars": {"met": False, "duration_sec": 0.0},
                "AOI_Teacher_Face": {"met": True, "duration_sec": 1.2},
                "AOI_Teacher_Hands": {"met": False, "duration_sec": 0.0},
            },
        }
        self.assertEqual(
            classify_rja_path_pattern(path_facts, ["AOI_Stars"]),
            "FAILED_SIGNIFICANT_FIXATION_ON_FACE",
        )

    def test_classifies_missing_target_definition_as_unknown(self):
        self.assertEqual(
            classify_rja_path_pattern({"targets_met_details": {}, "simplified_gaze_path": []}, []),
            "UNKNOWN_ERROR",
        )

    def test_perseverative_ratio(self):
        self.assertAlmostEqual(
            calculate_perseverative_gaze_ratio({"AOI_Sun": 3}, "AOI_Sun", 10),
            0.3,
        )
        self.assertEqual(
            calculate_perseverative_gaze_ratio({"AOI_Sun": 3}, "AOI_Sun", 0),
            0,
        )

    def test_rja_event_and_overall_summary_fields(self):
        gaze = pd.DataFrame(
            {
                "Frame": list(range(1, 22)),
                "Gaze_AOI": (
                    ["AOI_Sun"] * 3
                    + ["AOI_Stars"] * 3
                    + ["AOI_Teacher_Face"] * 15
                ),
            }
        )
        event = EventDefinition(
            "E14",
            1,
            21,
            ("AOI_Stars",),
            ("AOI_Teacher_Face", "AOI_Teacher_Hands"),
        )

        report = analyze_event(gaze, event, fps=30, min_dwell_ms=100)
        rja = report["Event_Specific"]["RJA_Analysis"]

        self.assertEqual(rja["path_pattern"], "SUCCESS_CHECKBACK_PATTERN")
        self.assertEqual(rja["RJA_Is_CheckbackLike"], 1.0)
        self.assertAlmostEqual(
            report["Dimension_Indicators"]["Attention_Shifting"]["Perseverative_Gaze_Ratio"],
            3 / 21,
        )

        overall = calculate_overall_statistics([report])
        self.assertEqual(
            overall["Clinical_Groups"]["Joint_Attention"]["RJA_CheckbackLike_Rate"],
            1.0,
        )
        self.assertAlmostEqual(
            overall["Functional_Analysis"]["Global_Perseveration_Rate"],
            3 / 21,
        )


if __name__ == "__main__":
    unittest.main()
