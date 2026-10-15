"""Feature extraction entry points for labeled gaze files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .event_features import EventDefinition, analyze_event, load_event_definitions, load_labeled_gaze
from .rqa_features import collect_event_gaze_coords_for_rqa
from .summary_features import calculate_overall_statistics

EXACT_EXCLUDE_KEYS = {
    "event_id",
    "subject_id",
    "start_frame",
    "end_frame",
    "timestamp",
    "Execution_Status",
    "Gaze_Trajectory_List",
}
EXCLUDE_SUBSTRINGS = ("Gaze_Trajectory_List", "details")
MODEL_EXCLUDE_SUBSTRINGS = ("Trajectory", "frame")
MODEL_EXCLUDE_EVENT_IDS = {"E12"}
REPORT_ONLY_SHORT_KEYS = {
    "path_pattern",
    "NoGazeData_Ratio",
    "valid",
    "RJA_Is_SuccessLike",
    "Attention_Shift_Freq",
}
MODEL_EXCLUDE_PATH_SUBSTRINGS = ("Clinical_Groups.Data_Quality.",)


def to_jsonable(value):
    """Convert numpy/pandas scalars to JSON-friendly Python values."""

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return round(float(value), 3)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def flatten_json(value: object) -> dict[str, object]:
    """Flatten nested dictionaries using dotted keys."""

    out: dict[str, object] = {}

    def flatten(item: object, prefix: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                flatten(child, f"{prefix}{key}.")
        elif isinstance(item, list):
            out[prefix[:-1]] = item
        else:
            out[prefix[:-1]] = item

    flatten(value)
    return out


def is_model_feature_key(key: str) -> bool:
    """Return whether a flattened report key is eligible as a numeric model feature."""

    key = str(key)
    short_key = key.split(".")[-1]
    if key in EXACT_EXCLUDE_KEYS or short_key in EXACT_EXCLUDE_KEYS:
        return False
    if any(item in key for item in EXCLUDE_SUBSTRINGS):
        return False
    if short_key in REPORT_ONLY_SHORT_KEYS:
        return False
    if any(item in key for item in MODEL_EXCLUDE_SUBSTRINGS):
        return False
    if any(item in key for item in MODEL_EXCLUDE_PATH_SUBSTRINGS):
        return False
    return True


def extract_numeric_feature_record(
    report: dict,
    subject_key: str = "subject_id",
    id_field_name: str = "id",
) -> dict | None:
    """Extract the numeric feature vector used by downstream selection/modeling code."""

    metadata = report.get("metadata", {})
    raw_subject_id = metadata.get(subject_key)
    subject_id = str(raw_subject_id).strip() if raw_subject_id is not None else ""
    if not subject_id:
        source_file = metadata.get("source_file")
        subject_id = Path(str(source_file)).name.split("_")[0] if source_file else ""
    if not subject_id:
        return None

    record: dict[str, object] = {id_field_name: subject_id, "subject_id": subject_id}
    overall = flatten_json(report.get("overall_statistics", {}))
    for key, value in overall.items():
        if value is None and is_model_feature_key(key):
            record[f"Overall.{key}"] = np.nan
        elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)) and is_model_feature_key(key):
            record[f"Overall.{key}"] = float(value)

    for event in report.get("events", []):
        event_id = event.get("event_id", "Unknown")
        if event_id in MODEL_EXCLUDE_EVENT_IDS:
            continue
        event_flat = flatten_json(event)
        for key, value in event_flat.items():
            if value is None and is_model_feature_key(key):
                record[f"{event_id}.{key}"] = np.nan
            elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)) and is_model_feature_key(key):
                record[f"{event_id}.{key}"] = float(value)

    return record


def extract_subject_report(
    labeled_gaze_csv: str | Path,
    events: list[EventDefinition],
    fps: float = 30.0,
) -> dict:
    """Extract a subject-level behavior report from one labeled gaze CSV."""

    gaze = load_labeled_gaze(str(labeled_gaze_csv))
    event_reports = []
    all_rqa_gaze_coords = []
    for event in events:
        if not event.target_aois:
            continue
        event_reports.append(analyze_event(gaze, event, fps=fps))
        if event.event_id != "E12":
            all_rqa_gaze_coords.extend(collect_event_gaze_coords_for_rqa(gaze, event))

    return {
        "metadata": {
            "source_file": str(labeled_gaze_csv),
            "total_frames": int(len(gaze)),
            "video_fps": float(fps),
            "event_count": int(len(event_reports)),
        },
        "overall_statistics": calculate_overall_statistics(event_reports, all_rqa_gaze_coords),
        "events": event_reports,
    }


def extract_subject_report_from_files(
    labeled_gaze_csv: str | Path,
    event_csv: str | Path,
    output_json: str | Path | None = None,
    fps: float = 30.0,
) -> dict:
    """Extract a report using file paths, optionally writing JSON output."""

    events = load_event_definitions(str(event_csv))
    report = extract_subject_report(labeled_gaze_csv, events, fps=fps)
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report
