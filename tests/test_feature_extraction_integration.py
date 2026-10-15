from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.features.event_features import EventDefinition
from asd_gaze.features.feature_extraction import (
    extract_numeric_feature_record,
    extract_subject_report,
)
from asd_gaze.reporting.structured_evidence import (
    build_claim_packet,
    build_structured_evidence,
    build_td_reference,
)


class FeatureExtractionIntegrationTest(unittest.TestCase):
    def test_subject_report_flattens_model_features(self):
        gaze = pd.DataFrame(
            {
                "Frame": list(range(1, 10)),
                "Gaze_AOI": [
                    "AOI_Blue_Block",
                    "AOI_Blue_Block",
                    "AOI_Blue_Block",
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Face",
                    "AOI_Teacher_Face",
                    "AOI_Center_Cross",
                    "AOI_Center_Cross",
                    "AOI_Center_Cross",
                ],
                "X": list(range(1, 10)),
                "Y": list(range(1, 10)),
            }
        )
        events = [
            EventDefinition("E06", 1, 6, ("AOI_Teacher_Face",), ("AOI_Teacher_Hands", "AOI_Blue_Block")),
            EventDefinition("E12", 7, 9, ("AOI_Center_Cross",), ()),
        ]
        path = Path(__file__).resolve().parent / "S00001_synthetic_labeled.csv"
        try:
            gaze.to_csv(path, index=False)
            report = extract_subject_report(path, events, fps=30)
        finally:
            path.unlink(missing_ok=True)

        record = extract_numeric_feature_record(report)
        self.assertIsNotNone(record)
        self.assertIn("Overall.Clinical_Groups.Social_Referencing.social_checkback_rate", record)
        self.assertIn("E06.Event_Specific.Social_Checkback_Occurred", record)
        self.assertNotIn("E12.Event_Specific.Guidance_Success_Strict", record)

    def test_real_flattened_keys_flow_into_evidence_and_claims(self):
        rows = []
        patterns = {
            "E01": (1, 9, ["AOI_Teacher_Face"] * 5 + ["AOI_Red_Block"] * 4),
            "E02": (10, 19, ["AOI_Red_Block"] * 10),
            "E03": (20, 29, ["AOI_Red_Block"] * 5 + ["AOI_Teacher_Face"] * 5),
            "E07": (30, 39, ["AOI_Teacher_Hands"] * 5 + ["AOI_Teacher_Face"] * 5),
            "E10": (40, 49, ["AOI_Green_Block"] * 10),
        }
        for start, _, labels in patterns.values():
            rows.extend(labels)
        gaze = pd.DataFrame({"Frame": range(1, len(rows) + 1), "Gaze_AOI": rows,
                             "X": range(1, len(rows) + 1), "Y": range(1, len(rows) + 1)})
        events = [
            EventDefinition("E01", 1, 9, ("AOI_Teacher_Face",), ()),
            EventDefinition("E02", 10, 19, ("AOI_Red_Block",), ("AOI_Teacher_Hands",)),
            EventDefinition("E03", 20, 29, ("AOI_Teacher_Face",), ("AOI_Red_Block",)),
            EventDefinition("E07", 30, 39, ("AOI_Left_Area",), ("AOI_Teacher_Hands",)),
            EventDefinition("E10", 40, 49, ("AOI_Green_Block",), ("AOI_Teacher_Hands",)),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SYN_EVID_labeled.csv"
            gaze.to_csv(path, index=False)
            report = extract_subject_report(path, events, fps=30)
        record = extract_numeric_feature_record({**report, "metadata": {"subject_id": "SYN_EVID"}})
        expected = {
            "E01.Event_Specific.Face_Preference_Index",
            "E02.Event_Specific.Action_Completion_Anytime",
            "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency",
        }
        self.assertTrue(expected.issubset(record))
        evidence_keys = {
            "E02.Event_Specific.Action_Completion_Anytime",
            "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency",
        }
        td = pd.DataFrame([{key: float(index + offset) for index, key in enumerate(evidence_keys)}
                           for offset in range(4)])
        reference = build_td_reference(td, sorted(evidence_keys))
        evidence = build_structured_evidence("SYN_EVID", record, reference)
        self.assertTrue(evidence_keys.issubset({item["feature_key"] for item in evidence}))
        packet = build_claim_packet("SYN_EVID", 0, 0.2, evidence, allow_unclassified=True)
        self.assertGreaterEqual(len(packet["claims"]), 1)


if __name__ == "__main__":
    unittest.main()
