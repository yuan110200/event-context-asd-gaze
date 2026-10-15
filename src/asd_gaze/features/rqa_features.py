"""Recurrence quantification analysis features for gaze coordinates."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .event_features import EventDefinition


def collect_event_gaze_coords_for_rqa(
    labeled_gaze: pd.DataFrame,
    event: EventDefinition,
) -> list[list[float]]:
    """Collect down-sampled valid gaze coordinates from one inclusive event window."""

    reaction_slice = labeled_gaze[
        (labeled_gaze["Frame"] >= int(event.start_frame))
        & (labeled_gaze["Frame"] <= int(event.end_frame))
    ].copy()

    if "X" not in reaction_slice.columns or "Y" not in reaction_slice.columns:
        return []

    valid_coords = reaction_slice[["X", "Y"]].dropna()
    valid_coords = valid_coords[(valid_coords["X"] > 0) & (valid_coords["Y"] > 0)]

    if len(valid_coords) <= 1:
        return []

    return valid_coords.iloc[::2].values.tolist()


def count_points_in_runs(binary_1d: Iterable[int], min_run_length: int = 2) -> int:
    """Count points that belong to qualifying consecutive one-runs."""

    total = 0
    run_len = 0
    for value in binary_1d:
        if int(value) == 1:
            run_len += 1
        else:
            if run_len >= min_run_length:
                total += run_len
            run_len = 0

    if run_len >= min_run_length:
        total += run_len
    return total


def calculate_rqa_metrics(fixation_coords: list[list[float]], epsilon: float = 80) -> dict[str, float]:
    """Calculate canonical RQA metrics from gaze coordinates."""

    if len(fixation_coords) < 5:
        return {"RQA_RR": 0, "RQA_DET": 0, "RQA_LAM": 0, "RQA_CORM": 0}

    coords = np.array(fixation_coords)
    max_rqa_frames = 2000
    if len(coords) > max_rqa_frames:
        coords = coords[:max_rqa_frames]

    deltas = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(deltas * deltas, axis=2))
    recurrence_matrix = (dist_matrix < epsilon).astype(int)
    np.fill_diagonal(recurrence_matrix, 0)

    n_points = len(coords)
    if n_points <= 1:
        return {"RQA_RR": 0, "RQA_DET": 0, "RQA_LAM": 0, "RQA_CORM": 0}

    total_recurrence = np.sum(recurrence_matrix)
    rr = total_recurrence / (n_points * (n_points - 1))

    rows, cols = np.indices((n_points, n_points))
    time_diff = rows - cols
    mask_lower = (time_diff > 0) & (recurrence_matrix == 1)
    recurrence_lower_count = np.sum(mask_lower)

    if recurrence_lower_count > 0 and n_points > 1:
        corm = np.sum(time_diff[mask_lower]) / (recurrence_lower_count * (n_points - 1)) * 100
    else:
        corm = 0

    det_points = 0
    for offset in range(-(n_points - 1), n_points):
        if offset == 0:
            continue
        diag = np.diagonal(recurrence_matrix, offset=offset)
        det_points += count_points_in_runs(diag, min_run_length=2)
    det = det_points / total_recurrence if total_recurrence > 0 else 0.0

    lam_points = 0
    for col in range(n_points):
        lam_points += count_points_in_runs(recurrence_matrix[:, col], min_run_length=2)
    lam = lam_points / total_recurrence if total_recurrence > 0 else 0.0

    return {
        "RQA_RR": round(rr, 3),
        "RQA_CORM": round(corm, 3),
        "RQA_DET": round(det, 3),
        "RQA_LAM": round(lam, 3),
    }
