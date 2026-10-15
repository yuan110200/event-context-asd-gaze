"""Map frame-aligned gaze samples to dynamic AOI labels.

The mapping preserves the original project behavior:

- gaze rows are matched to AOIs by the integer ``Frame`` value;
- AOI boxes use inclusive coordinate bounds;
- overlapping AOIs are resolved by a fixed priority order and first hit wins;
- missing gaze coordinates become ``NoGazeData``;
- valid gaze points outside every AOI become ``Other``.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


FRAME_COLUMN = "Frame"
GAZE_X_COLUMN = "X"
GAZE_Y_COLUMN = "Y"
AOI_OUTPUT_COLUMN = "Gaze_AOI"

AOI_FRAME_COLUMN = "frame_number"
AOI_LABEL_COLUMN = "aoi_label"
AOI_BOUNDS_COLUMNS = ("x_min", "y_min", "x_max", "y_max")
DEFAULT_AOI_EXPANSION_PIXELS = 25

DEFAULT_AOI_PRIORITY: tuple[str, ...] = (
    "AOI_Block_Tower",
    "AOI_Green_Block",
    "AOI_Blue_Block",
    "AOI_Red_Block",
    "AOI_Yellow_Block",
    "AOI_Teacher_Hands",
)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def clip_bbox_to_frame(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Clip an AOI box to video-frame bounds using the source coordinate convention."""

    if frame_width < 0 or frame_height < 0:
        raise ValueError("Frame width and height must be non-negative")
    x_min, y_min, x_max, y_max = bbox
    return (
        int(max(0, x_min)),
        int(max(0, y_min)),
        int(min(frame_width, x_max)),
        int(min(frame_height, y_max)),
    )


def expand_bbox(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    expansion_pixels: int = DEFAULT_AOI_EXPANSION_PIXELS,
) -> tuple[int, int, int, int]:
    """Expand a raw detection box and clip it to video-frame bounds.

    This mirrors the AOI-generation source: boxes are expanded by 25 pixels on
    each side before being saved to the AOI CSV. Use this only for raw
    detector/template boxes; public AOI CSVs are already expanded.
    """

    x_min, y_min, x_max, y_max = bbox
    expanded = (
        x_min - expansion_pixels,
        y_min - expansion_pixels,
        x_max + expansion_pixels,
        y_max + expansion_pixels,
    )
    return clip_bbox_to_frame(expanded, frame_width, frame_height)


def make_center_cross_bbox(
    frame_width: int,
    frame_height: int,
    half_size: int = 200,
) -> tuple[int, int, int, int]:
    """Return the center-cross AOI box used during the E12 transition window."""

    center_x = frame_width // 2
    center_y = frame_height // 2
    return clip_bbox_to_frame(
        (center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size),
        frame_width,
        frame_height,
    )


def _validate_aoi_bounds(df: pd.DataFrame, name: str) -> None:
    coordinates = df[list(AOI_BOUNDS_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} contains non-finite AOI bounds")
    invalid = df[
        (df["x_min"] > df["x_max"])
        | (df["y_min"] > df["y_max"])
        | (df["x_min"] < 0)
        | (df["y_min"] < 0)
    ]
    if not invalid.empty:
        raise ValueError(f"{name} contains invalid AOI bounds")


def prepare_aoi_map(
    aoi_map: pd.DataFrame,
    priority_order: Sequence[str] = DEFAULT_AOI_PRIORITY,
) -> pd.DataFrame:
    """Validate and sort a dynamic AOI map for deterministic hit testing."""

    required = (AOI_FRAME_COLUMN, AOI_LABEL_COLUMN, *AOI_BOUNDS_COLUMNS)
    _require_columns(aoi_map, required, "AOI map")

    prepared = aoi_map.copy()
    prepared = prepared.rename(columns={AOI_FRAME_COLUMN: "frame_id"})
    prepared["frame_id"] = prepared["frame_id"].astype(int)

    for column in AOI_BOUNDS_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    _validate_aoi_bounds(prepared, "AOI map")

    priority_lookup = {label: index for index, label in enumerate(priority_order)}
    prepared["_priority"] = prepared[AOI_LABEL_COLUMN].map(priority_lookup).fillna(99)
    prepared = prepared.sort_values(["frame_id", "_priority"], kind="mergesort")
    return prepared.drop(columns=["_priority"]).reset_index(drop=True)


def load_aoi_map(
    path: str | Path,
    priority_order: Sequence[str] = DEFAULT_AOI_PRIORITY,
) -> pd.DataFrame:
    """Read and prepare a dynamic AOI map CSV."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AOI map not found: {path}")
    return prepare_aoi_map(pd.read_csv(path), priority_order=priority_order)


def build_aoi_map_from_detections(
    detections: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    expansion_pixels: int = DEFAULT_AOI_EXPANSION_PIXELS,
    detections_are_expanded: bool = False,
    priority_order: Sequence[str] = DEFAULT_AOI_PRIORITY,
) -> pd.DataFrame:
    """Build a prepared AOI map from raw detector boxes.

    The original AOI-generation script expanded detector boxes once, clipped
    them to each video frame, then wrote ``frame_number,aoi_label,x_min,y_min,
    x_max,y_max``. Set ``detections_are_expanded=True`` only when the input
    table already contains those final expanded AOI boxes.
    """

    required = (AOI_FRAME_COLUMN, AOI_LABEL_COLUMN, *AOI_BOUNDS_COLUMNS)
    _require_columns(detections, required, "AOI detections")

    rows = []
    for row in detections.to_dict("records"):
        bbox = tuple(row[column] for column in AOI_BOUNDS_COLUMNS)
        final_bbox = (
            clip_bbox_to_frame(bbox, frame_width, frame_height)
            if detections_are_expanded
            else expand_bbox(bbox, frame_width, frame_height, expansion_pixels)
        )
        rows.append(
            {
                AOI_FRAME_COLUMN: int(row[AOI_FRAME_COLUMN]),
                AOI_LABEL_COLUMN: row[AOI_LABEL_COLUMN],
                "x_min": final_bbox[0],
                "y_min": final_bbox[1],
                "x_max": final_bbox[2],
                "y_max": final_bbox[3],
            }
        )

    return prepare_aoi_map(pd.DataFrame(rows).drop_duplicates(), priority_order=priority_order)


def map_gaze_to_aoi(
    gaze_data: pd.DataFrame,
    prepared_aoi_map: pd.DataFrame,
    output_column: str = AOI_OUTPUT_COLUMN,
) -> pd.DataFrame:
    """Return a copy of gaze data with one AOI label per row."""

    _require_columns(gaze_data, (FRAME_COLUMN, GAZE_X_COLUMN, GAZE_Y_COLUMN), "Gaze data")
    _require_columns(
        prepared_aoi_map,
        ("frame_id", AOI_LABEL_COLUMN, *AOI_BOUNDS_COLUMNS),
        "Prepared AOI map",
    )

    aoi_by_frame = {
        int(frame_id): frame_aois
        for frame_id, frame_aois in prepared_aoi_map.groupby("frame_id", sort=False)
    }

    labels: list[str] = []
    for _, row in gaze_data.iterrows():
        frame_id = int(row[FRAME_COLUMN])
        gaze_x = row[GAZE_X_COLUMN]
        gaze_y = row[GAZE_Y_COLUMN]

        if pd.isna(gaze_x) or pd.isna(gaze_y):
            labels.append("NoGazeData")
            continue

        label = "Other"
        frame_aois = aoi_by_frame.get(frame_id)
        if frame_aois is not None:
            for _, aoi in frame_aois.iterrows():
                if (
                    aoi["x_min"] <= gaze_x <= aoi["x_max"]
                    and aoi["y_min"] <= gaze_y <= aoi["y_max"]
                ):
                    label = aoi[AOI_LABEL_COLUMN]
                    break

        labels.append(label)

    mapped = gaze_data.copy()
    mapped[output_column] = labels
    return mapped


def map_gaze_file(
    gaze_csv: str | Path,
    aoi_map_csv: str | Path,
    output_csv: str | Path,
    priority_order: Sequence[str] = DEFAULT_AOI_PRIORITY,
) -> None:
    """Map one frame-aligned gaze CSV and write the labeled CSV."""

    prepared_map = load_aoi_map(aoi_map_csv, priority_order=priority_order)
    gaze_data = pd.read_csv(gaze_csv)
    mapped = map_gaze_to_aoi(gaze_data, prepared_map)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(output_csv, index=False)


def map_aligned_directory(
    input_dir: str | Path,
    aoi_map_csv: str | Path,
    pattern: str = "**/*_aligned.csv",
    output_suffix: str = "_labeled.csv",
) -> list[Path]:
    """Map all aligned gaze CSV files below a directory.

    Output files are written next to their inputs by replacing
    ``_aligned.csv`` with ``output_suffix``.
    """

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    prepared_map = load_aoi_map(aoi_map_csv)
    outputs: list[Path] = []
    for gaze_csv in sorted(input_dir.glob(pattern)):
        if not gaze_csv.name.endswith("_aligned.csv"):
            raise ValueError(f"Input file does not end with '_aligned.csv': {gaze_csv}")
        output_csv = gaze_csv.with_name(gaze_csv.name.replace("_aligned.csv", output_suffix))
        mapped = map_gaze_to_aoi(pd.read_csv(gaze_csv), prepared_map)
        mapped.to_csv(output_csv, index=False)
        outputs.append(output_csv)
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map frame-aligned gaze samples to AOIs.")
    parser.add_argument("--aoi-map", required=True, help="Dynamic AOI map CSV.")
    parser.add_argument("--gaze-csv", help="Single aligned gaze CSV to map.")
    parser.add_argument("--output-csv", help="Output CSV for --gaze-csv.")
    parser.add_argument("--input-dir", help="Directory for batch mapping.")
    parser.add_argument("--pattern", default="**/*_aligned.csv", help="Batch input glob.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.gaze_csv:
        if not args.output_csv:
            parser.error("--output-csv is required with --gaze-csv")
        map_gaze_file(args.gaze_csv, args.aoi_map, args.output_csv)
        return 0

    if args.input_dir:
        outputs = map_aligned_directory(args.input_dir, args.aoi_map, pattern=args.pattern)
        print(f"Mapped {len(outputs)} file(s).")
        return 0

    parser.error("Provide either --gaze-csv or --input-dir")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
