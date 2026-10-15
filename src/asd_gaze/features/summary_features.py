"""Subject-level summaries from event reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rqa_features import calculate_rqa_metrics


def calculate_overall_statistics(
    events: list[dict],
    all_rqa_gaze_coords: list[list[float]] | None = None,
) -> dict:
    """Calculate the public subject-level summary skeleton.

    E12 is retained under Data_Quality but excluded from behavioral overall
    means, matching the canonical feature extraction policy.
    """

    if not events:
        return {"Basic_Metrics": {}, "Clinical_Groups": {}, "Functional_Analysis": {}}

    df = pd.DataFrame(events)
    behavior_df = df[df["event_id"] != "E12"].copy() if "event_id" in df.columns else df.copy()

    basic: dict[str, float | None] = {}
    if "Task_Success_Flag" in behavior_df:
        values = pd.to_numeric(behavior_df["Task_Success_Flag"], errors="coerce").dropna()
        basic["Overall_Task_Success_Rate"] = float(values.mean()) if not values.empty else np.nan

    for metric in [
        "Dwell_On_Face_Ratio",
        "Dwell_On_Hands_Ratio",
        "Dwell_On_Task_Object_Ratio",
        "Dwell_On_Other_Ratio",
        "Attention_Shift_Count",
    ]:
        if metric in behavior_df:
            values = pd.to_numeric(behavior_df[metric], errors="coerce").dropna()
            basic[metric] = float(values.mean()) if not values.empty else np.nan

    if "Cue_to_Target_Latency_sec" in behavior_df:
        target_orienting_latencies = pd.to_numeric(
            behavior_df["Cue_to_Target_Latency_sec"],
            errors="coerce",
        ).dropna()
        basic["Avg_Cue_to_Target_Latency"] = (
            float(target_orienting_latencies.mean()) if not target_orienting_latencies.empty else np.nan
        )

    clinical = {
        "Social_Motivation": _success_group(df, ["E01", "E03", "E04"]),
        "Joint_Attention": _joint_attention_group(df),
        "Social_Referencing": _social_referencing_group(df),
        "Action_Perception": _success_group(df, ["E02", "E10"]),
        "Data_Quality": _data_quality_group(df),
    }

    functional: dict[str, float | None] = {}
    if "Social_Switch_Rate" in behavior_df:
        values = pd.to_numeric(behavior_df["Social_Switch_Rate"], errors="coerce").dropna()
        functional["Avg_Social_Integration_Rate"] = float(values.mean()) if not values.empty else None
    if "Social_Preference_Bias" in behavior_df:
        values = pd.to_numeric(behavior_df["Social_Preference_Bias"], errors="coerce").dropna()
        functional["Mean_Social_Preference_Bias"] = float(values.mean()) if not values.empty else np.nan
    perseveration = behavior_df.apply(
        lambda row: row.get("Dimension_Indicators", {})
        .get("Attention_Shifting", {})
        .get("Perseverative_Gaze_Ratio", np.nan),
        axis=1,
    ).dropna()
    functional["Global_Perseveration_Rate"] = (
        float(perseveration.mean()) if not perseveration.empty else np.nan
    )
    rqa_result = calculate_rqa_metrics(all_rqa_gaze_coords or [], epsilon=80)
    functional["Avg_Gaze_Recurrence_RQA_RR"] = rqa_result["RQA_RR"]
    functional["Avg_Gaze_Determinism_RQA_DET"] = rqa_result["RQA_DET"]
    functional["Avg_Gaze_Laminarity_RQA_LAM"] = rqa_result["RQA_LAM"]
    functional["Avg_Gaze_CORM_RQA_CORM"] = rqa_result["RQA_CORM"]

    return {
        "Basic_Metrics": basic,
        "Clinical_Groups": clinical,
        "Functional_Analysis": functional,
    }


def _success_group(df: pd.DataFrame, event_ids: list[str]) -> dict:
    sub = df[df["event_id"].isin(event_ids)]
    if sub.empty:
        return {"valid": False}
    values = pd.to_numeric(sub.get("Task_Success_Flag", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "valid": not values.empty,
        "success_rate": float((values > 0).mean()) if not values.empty else np.nan,
    }


def _joint_attention_group(df: pd.DataFrame) -> dict:
    group = _success_group(df, ["E05", "E09", "E13", "E14"])
    sub = df[df["event_id"].isin(["E05", "E09", "E13", "E14"])]
    if sub.empty:
        return group

    rja_scores = sub.apply(
        lambda row: row.get("Event_Specific", {})
        .get("RJA_Analysis", {})
        .get("RJA_Path_Score", np.nan),
        axis=1,
    )
    hand_bias = sub.apply(
        lambda row: row.get("Event_Specific", {})
        .get("RJA_Analysis", {})
        .get("Hand_To_Target_Bias", np.nan),
        axis=1,
    )
    checkback_like = sub.apply(
        lambda row: row.get("Event_Specific", {})
        .get("RJA_Analysis", {})
        .get("RJA_Is_CheckbackLike", np.nan),
        axis=1,
    ).dropna()

    group["RJA_Path_Score_Mean"] = (
        float(rja_scores.dropna().mean()) if rja_scores.notna().any() else np.nan
    )
    group["Hand_To_Target_Bias_Mean"] = (
        float(hand_bias.dropna().mean()) if hand_bias.notna().any() else np.nan
    )
    group["RJA_CheckbackLike_Rate"] = (
        float(checkback_like.mean()) if not checkback_like.empty else np.nan
    )
    return group


def _social_referencing_group(df: pd.DataFrame) -> dict:
    group = _success_group(df, ["E06", "E07", "E08", "E11"])
    sub = df[df["event_id"].isin(["E06", "E07", "E08", "E11"])]
    if not sub.empty:
        checkbacks = sub.apply(
            lambda row: row.get("Event_Specific", {}).get("Social_Checkback_Occurred", np.nan),
            axis=1,
        ).dropna()
        group["social_checkback_rate"] = float(checkbacks.mean()) if not checkbacks.empty else np.nan
    return group


def _data_quality_group(df: pd.DataFrame) -> dict:
    sub = df[df["event_id"].isin(["E12"])]
    if sub.empty:
        return {"valid": False}
    success = sub.apply(
        lambda row: row.get("Event_Specific", {}).get("Guidance_Success_Strict", np.nan),
        axis=1,
    ).dropna()
    latency = sub.apply(
        lambda row: row.get("Event_Specific", {}).get("Guidance_Latency_sec", np.nan),
        axis=1,
    ).dropna()
    return {
        "valid": not success.empty,
        "success_rate": float((success > 0).mean()) if not success.empty else np.nan,
        "guidance_success_rate": float(success.mean()) if not success.empty else np.nan,
        "avg_guidance_latency_sec": float(latency.mean()) if not latency.empty else np.nan,
    }
