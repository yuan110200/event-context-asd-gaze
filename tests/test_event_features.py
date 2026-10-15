from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.features.event_features import (
    EventDefinition,
    analyze_event,
    build_clean_aoi_sequence,
    build_fixation_blocks,
    load_event_definitions,
    select_event_window,
)
from asd_gaze.features.summary_features import calculate_overall_statistics


class EventFeaturesTest(unittest.TestCase):
    def test_event_window_is_inclusive(self):
        gaze = pd.DataFrame(
            {
                "Frame": [1, 2, 3, 4],
                "Gaze_AOI": ["Other", "AOI_Teacher_Face", "AOI_Sun", "Other"],
            }
        )
        event = EventDefinition("E13", 2, 3, ("AOI_Sun",), ("AOI_Teacher_Face",))
        window = select_event_window(gaze, event)
        self.assertEqual(window["Frame"].tolist(), [2, 3])

    def test_repository_event_config_loads_verified_boundaries(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "event_definitions.csv"
        events = {event.event_id: event for event in load_event_definitions(str(config))}

        self.assertEqual(len(events), 14)
        self.assertEqual((events["E11"].start_frame, events["E11"].end_frame), (1061, 1169))
        self.assertEqual((events["E12"].start_frame, events["E12"].end_frame), (1170, 1184))
        self.assertEqual((events["E13"].start_frame, events["E13"].end_frame), (1185, 1330))
        self.assertEqual(events["E13"].target_aois, ("AOI_Sun",))
        self.assertEqual(events["E13"].process_aois, ("AOI_Teacher_Face", "AOI_Teacher_Hands"))

    def test_missing_gaze_breaks_clean_sequence(self):
        blocks = build_fixation_blocks(
            ["AOI_Teacher_Face", "NoGazeData", "AOI_Teacher_Face", "AOI_Sun"],
            min_dwell_frames=1,
        )
        self.assertEqual(
            build_clean_aoi_sequence(blocks),
            ["AOI_Teacher_Face", "__MISSING_GAP__", "AOI_Teacher_Face", "AOI_Sun"],
        )

    def test_event_features_and_e12_summary_policy(self):
        gaze = pd.DataFrame(
            {
                "Frame": [1, 2, 3, 4, 5, 6, 7, 8],
                "Gaze_AOI": [
                    "AOI_Teacher_Face",
                    "AOI_Center_Cross",
                    "AOI_Center_Cross",
                    "AOI_Center_Cross",
                    "Other",
                    "AOI_Sun",
                    "AOI_Sun",
                    "AOI_Sun",
                ],
            }
        )
        e12 = EventDefinition("E12", 2, 4, ("AOI_Center_Cross",), ())
        e13 = EventDefinition("E13", 5, 8, ("AOI_Sun",), ("AOI_Teacher_Face",))

        e12_report = analyze_event(gaze, e12, fps=30)
        e13_report = analyze_event(gaze, e13, fps=30)
        overall = calculate_overall_statistics([e12_report, e13_report])

        self.assertEqual(e12_report["Event_Specific"]["Guidance_Success_Strict"], 1)
        self.assertIn("Data_Quality", overall["Clinical_Groups"])
        self.assertEqual(overall["Basic_Metrics"]["Overall_Task_Success_Rate"], 1.0)
        self.assertNotIn("Guidance_Latency_sec", overall["Basic_Metrics"])

    def test_e04_exploration_uses_full_event_window_denominator(self):
        gaze = pd.DataFrame(
            {
                "Frame": [1, 2, 3, 4, 5, 6],
                "Gaze_AOI": [
                    "AOI_Teacher_Face",
                    "AOI_Blue_Block",
                    "AOI_Blue_Block",
                    "NoGazeData",
                    "AOI_Teacher_Hands",
                    "Other",
                ],
            }
        )
        event = EventDefinition("E04", 1, 6, ("AOI_Blue_Block",), ("AOI_Teacher_Face", "AOI_Teacher_Hands"))
        report = analyze_event(gaze, event, fps=30, min_dwell_ms=1)

        self.assertAlmostEqual(report["Event_Specific"]["Exploration_Ratio"], 3 / 6)
        self.assertEqual(report["Event_Specific"]["Exploratory_Score"], 2 * (3 / 6))

    def test_e06_contributes_to_social_checkback_rate(self):
        gaze = pd.DataFrame(
            {
                "Frame": [1, 2, 3, 4, 5, 6],
                "Gaze_AOI": [
                    "AOI_Teacher_Hands",
                    "AOI_Teacher_Hands",
                    "AOI_Teacher_Face",
                    "AOI_Block_Tower",
                    "AOI_Block_Tower",
                    "Other",
                ],
            }
        )
        e06 = EventDefinition("E06", 1, 3, ("AOI_Teacher_Face",), ("AOI_Teacher_Hands", "AOI_Blue_Block", "AOI_Block_Tower"))
        e11 = EventDefinition("E11", 4, 6, ("AOI_Teacher_Face",), ("AOI_Block_Tower", "AOI_Teacher_Hands"))

        e06_report = analyze_event(gaze, e06, fps=30, min_dwell_ms=1)
        e11_report = analyze_event(gaze, e11, fps=30, min_dwell_ms=1)
        overall = calculate_overall_statistics([e06_report, e11_report])

        self.assertEqual(e06_report["Event_Specific"]["Social_Checkback_Occurred"], 1)
        self.assertEqual(e11_report["Event_Specific"]["Social_Checkback_Occurred"], 0)
        self.assertEqual(overall["Clinical_Groups"]["Social_Referencing"]["social_checkback_rate"], 0.5)

    def test_remaining_candidate_event_specific_features_are_emitted(self):
        gaze = pd.DataFrame(
            {
                "Frame": list(range(1, 31)),
                "Gaze_AOI": [
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Face",
                    "AOI_Blue_Block",
                    "AOI_Blue_Block",
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Hands",
                    "AOI_Teacher_Face",
                    "AOI_Red_Block",
                    "AOI_Red_Block",
                    "AOI_Red_Block",
                    "AOI_Block_Tower",
                    "AOI_Block_Tower",
                    "AOI_Teacher_Face",
                    "AOI_Block_Tower",
                    "AOI_Block_Tower",
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Hands",
                    "AOI_Left_Area",
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Hands",
                    "AOI_Green_Block",
                    "AOI_Block_Tower",
                    "AOI_Block_Tower",
                    "AOI_Block_Tower",
                    "Other",
                    "Other",
                    "Other",
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Face",
                ],
            }
        )

        e01 = analyze_event(gaze, EventDefinition("E01", 1, 6, ("AOI_Teacher_Face",), ()), fps=2, min_dwell_ms=1)
        e02 = analyze_event(
            gaze,
            EventDefinition("E02", 7, 16, ("AOI_Teacher_Hands", "AOI_Red_Block"), ("AOI_Teacher_Face",)),
            fps=2,
            min_dwell_ms=1,
        )
        e03 = analyze_event(gaze, EventDefinition("E03", 12, 17, ("AOI_Teacher_Face",), ()), fps=2, min_dwell_ms=1)
        e07 = analyze_event(
            gaze,
            EventDefinition("E07", 18, 20, ("AOI_Left_Area",), ("AOI_Teacher_Face", "AOI_Teacher_Hands")),
            fps=2,
            min_dwell_ms=1,
        )
        e10 = analyze_event(
            gaze,
            EventDefinition("E10", 21, 30, ("AOI_Teacher_Hands", "AOI_Green_Block", "AOI_Block_Tower"), ()),
            fps=2,
            min_dwell_ms=1,
        )
        overall = calculate_overall_statistics([e01, e02, e03, e07, e10])

        self.assertAlmostEqual(e01["Event_Specific"]["Face_Preference_Index"], 4 / 6)
        self.assertAlmostEqual(e01["Event_Specific"]["Eye_Contact_Maintenance_sec"], 1.0)
        self.assertEqual(e02["Event_Specific"]["Action_Completion_FinalPhase"], 0)
        self.assertEqual(e02["Event_Specific"]["Action_Completion_Anytime"], 1)
        self.assertEqual(e02["Event_Specific"]["Face_Object_Switch_Count"], 2)
        self.assertAlmostEqual(
            e03["Dimension_Indicators"]["Attention_Shifting"]["Disengagement_Latency_sec"],
            1.0,
        )
        self.assertEqual(e07["Event_Specific"]["Uncertainty_Checkback_Occurred"], 1)
        self.assertEqual(e10["Event_Specific"]["Action_Completion_FinalPhase"], 0)
        self.assertEqual(e10["Event_Specific"]["Action_Completion_Anytime"], 1)
        self.assertIn("Avg_Cue_to_Target_Latency", overall["Basic_Metrics"])


if __name__ == "__main__":
    unittest.main()
