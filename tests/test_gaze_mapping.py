from pathlib import Path
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asd_gaze.aoi.gaze_mapping import (
    build_aoi_map_from_detections,
    expand_bbox,
    make_center_cross_bbox,
    map_gaze_to_aoi,
    prepare_aoi_map,
)


class GazeMappingTest(unittest.TestCase):
    def test_map_gaze_to_aoi_preserves_original_mapping_behavior(self):
        aoi_map = pd.DataFrame(
            [
                {
                    "frame_number": 1,
                    "aoi_label": "AOI_Teacher_Hands",
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 10,
                    "y_max": 10,
                },
                {
                    "frame_number": 1,
                    "aoi_label": "AOI_Block_Tower",
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 10,
                    "y_max": 10,
                },
                {
                    "frame_number": 2,
                    "aoi_label": "AOI_Red_Block",
                    "x_min": 5,
                    "y_min": 5,
                    "x_max": 15,
                    "y_max": 15,
                },
            ]
        )
        gaze = pd.DataFrame(
            [
                {"Frame": 1, "X": 5, "Y": 5},  # overlap: priority selects tower
                {"Frame": 2, "X": 5, "Y": 5},  # inclusive lower boundary
                {"Frame": 2, "X": 15, "Y": 15},  # inclusive upper boundary
                {"Frame": 2, "X": 16, "Y": 15},  # outside all AOIs
                {"Frame": 3, "X": 5, "Y": 5},  # no AOI for frame
                {"Frame": 1, "X": None, "Y": 5},  # invalid gaze
                {"Frame": 1, "X": 5, "Y": float("nan")},  # invalid gaze
            ]
        )

        mapped = map_gaze_to_aoi(gaze, prepare_aoi_map(aoi_map))

        self.assertEqual(
            mapped["Gaze_AOI"].tolist(),
            [
                "AOI_Block_Tower",
                "AOI_Red_Block",
                "AOI_Red_Block",
                "Other",
                "Other",
                "NoGazeData",
                "NoGazeData",
            ],
        )

    def test_missing_required_columns_raise_clear_error(self):
        aoi_map = prepare_aoi_map(
            pd.DataFrame(
                [
                    {
                        "frame_number": 1,
                        "aoi_label": "AOI_Red_Block",
                        "x_min": 0,
                        "y_min": 0,
                        "x_max": 10,
                        "y_max": 10,
                    }
                ]
            )
        )

        bad_gaze = pd.DataFrame([{"Frame": 1, "X": 1}])

        with self.assertRaisesRegex(ValueError, "Gaze data is missing required columns: Y"):
            map_gaze_to_aoi(bad_gaze, aoi_map)

    def test_non_finite_aoi_bounds_raise_clear_error(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            with self.subTest(invalid=invalid):
                bad_map = pd.DataFrame([{
                    "frame_number": 1, "aoi_label": "AOI_Red_Block",
                    "x_min": invalid, "y_min": 0, "x_max": 10, "y_max": 10,
                }])
                with self.assertRaisesRegex(ValueError, "non-finite AOI bounds"):
                    prepare_aoi_map(bad_map)

    def test_expand_bbox_clips_to_video_bounds(self):
        self.assertEqual(expand_bbox((10, 20, 40, 50), 100, 80), (0, 0, 65, 75))
        self.assertEqual(expand_bbox((90, 70, 120, 95), 100, 80), (65, 45, 100, 80))

    def test_center_cross_bbox_uses_frame_center_and_clips(self):
        self.assertEqual(make_center_cross_bbox(1920, 1080), (760, 340, 1160, 740))
        self.assertEqual(make_center_cross_bbox(300, 200), (0, 0, 300, 200))

    def test_build_aoi_map_expands_raw_detections_once(self):
        detections = pd.DataFrame(
            [
                {
                    "frame_number": 1,
                    "aoi_label": "AOI_Red_Block",
                    "x_min": 30,
                    "y_min": 30,
                    "x_max": 40,
                    "y_max": 40,
                }
            ]
        )

        expanded = build_aoi_map_from_detections(detections, frame_width=100, frame_height=100)
        already_expanded = build_aoi_map_from_detections(
            expanded.rename(columns={"frame_id": "frame_number"}),
            frame_width=100,
            frame_height=100,
            detections_are_expanded=True,
        )

        self.assertEqual(expanded[["x_min", "y_min", "x_max", "y_max"]].iloc[0].tolist(), [5, 5, 65, 65])
        self.assertEqual(already_expanded[["x_min", "y_min", "x_max", "y_max"]].iloc[0].tolist(), [5, 5, 65, 65])

    def test_overlap_priority_after_building_from_detections(self):
        detections = pd.DataFrame(
            [
                {
                    "frame_number": 1,
                    "aoi_label": "AOI_Teacher_Hands",
                    "x_min": 10,
                    "y_min": 10,
                    "x_max": 20,
                    "y_max": 20,
                },
                {
                    "frame_number": 1,
                    "aoi_label": "AOI_Block_Tower",
                    "x_min": 10,
                    "y_min": 10,
                    "x_max": 20,
                    "y_max": 20,
                },
            ]
        )
        prepared = build_aoi_map_from_detections(
            detections,
            frame_width=100,
            frame_height=100,
            expansion_pixels=0,
        )
        gaze = pd.DataFrame([{"Frame": 1, "X": 15, "Y": 15}])

        mapped = map_gaze_to_aoi(gaze, prepared)

        self.assertEqual(mapped["Gaze_AOI"].tolist(), ["AOI_Block_Tower"])


if __name__ == "__main__":
    unittest.main()
