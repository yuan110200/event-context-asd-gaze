"""Run the public pipeline on fictional gaze and AOI data only."""

from __future__ import annotations

import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.asd_gaze.aoi.gaze_mapping import build_aoi_map_from_detections, map_gaze_to_aoi
from src.asd_gaze.classification.rf_model import fit_final_train300_calibrated_rf, predict_positive_probability
from src.asd_gaze.features.feature_extraction import extract_numeric_feature_record, extract_subject_report_from_files
from src.asd_gaze.reporting import (build_claim_packet, build_structured_evidence,
                                    build_td_reference, generate_report)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        detections = pd.DataFrame([{
            "frame_number": frame, "aoi_label": "AOI_Teacher_Face",
            "x_min": 10, "y_min": 10, "x_max": 90, "y_max": 90,
        } for frame in range(120)])
        aoi_map = build_aoi_map_from_detections(detections, 120, 120, detections_are_expanded=True)
        gaze = pd.DataFrame({"Frame": range(120), "X": 50, "Y": 50})
        labeled = map_gaze_to_aoi(gaze, aoi_map)
        gaze_path = root / "synthetic_gaze.csv"
        labeled.to_csv(gaze_path, index=False)
        report = extract_subject_report_from_files(
            gaze_path, Path(__file__).parents[1] / "configs" / "event_definitions.csv", fps=30.0
        )
        print(f"AOI labels: {sorted(labeled['Gaze_AOI'].unique())}")
        print(f"Extracted events: {report['metadata']['event_count']}")

        record = extract_numeric_feature_record({**report, "metadata": {"subject_id": "SYNTHETIC"}})
        latency_key = "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency"
        td_reference = build_td_reference(pd.DataFrame({latency_key: [0.5, 0.6, 0.7, 0.8]}), [latency_key])
        evidence = build_structured_evidence("SYNTHETIC", record, td_reference)
        packet = build_claim_packet("SYNTHETIC", 0, 0.2, evidence)

        train = pd.DataFrame({"x": np.arange(10, dtype=float), "y": np.arange(10, dtype=float)[::-1],
                              "target": [0, 1] * 5})
        model = fit_final_train300_calibrated_rf(train, ["x", "y"])
        probability = float(predict_positive_probability(model, train.iloc[[0]], ["x", "y"])[0])
        packet["prediction"] = {"predicted_label": 0, "calibrated_probability": probability}
        def successful_mock(_system, prompt, _schema):
            claim = json.loads(prompt)["approved_claim"]
            return {"claim_id": claim["claim_id"], "event_id": claim["event_ids"][0] if claim["event_ids"] else None,
                    "text": "First qualifying gaze timing was shorter than the reference median."}
        rendered = generate_report(packet, successful_mock)
        failed = generate_report(packet, lambda *_: None)
        print(f"RF probability: {probability:.3f}")
        print(f"Report sources: {rendered['source']}, {failed['source']}")


if __name__ == "__main__":
    main()
