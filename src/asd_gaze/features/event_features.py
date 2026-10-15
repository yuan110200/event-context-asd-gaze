"""Event-level gaze behavior features.

This module contains the event-window and AOI-sequence logic used by the
public feature extraction code. Event windows are inclusive and use the
``Frame`` values already present in labeled gaze CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_GAP_TOKEN = "__MISSING_GAP__"

DEFAULT_MIN_DWELL_MS = 100
DEFAULT_VIDEO_FPS = 30.0
SIGNIFICANT_FIXATION_SEC = 1.5
RJA_SIGNIFICANT_FIXATION_SEC = 1.2
RJA_BRIEF_CHECK_SEC = 0.5

TASK_OBJECT_AOIS = {
    "AOI_Red_Block",
    "AOI_Blue_Block",
    "AOI_Green_Block",
    "AOI_Block_Tower",
    "AOI_Sun",
    "AOI_Stars",
}

RJA_PATTERN_SCORE_MAP = {
    "SUCCESS_CHECKBACK_PATTERN": 4,
    "SUCCESS_OTHER_BRIEF_CHECK": 3,
    "SUCCESS_FACE_TARGET": 3,
    "SUCCESS_HANDS_TARGET": 2,
    "SUCCESS_OTHER": 1,
    "FAILED_SOCIAL_LOOP_BRIEF": -1,
    "FAILED_BRIEF_LOOK_ON_FACE": -1,
    "FAILED_BRIEF_LOOK_ON_HANDS": -1,
    "FAILED_SIGNIFICANT_FIXATION_ON_FACE": -2,
    "FAILED_SIGNIFICANT_FIXATION_ON_HANDS": -2,
    "FAILED_DISENGAGED": -3,
    "UNKNOWN_ERROR": 0,
}

RJA_PREVIOUS_TASK_AOI = {
    "E05": "AOI_Teacher_Face",
    "E09": "AOI_Teacher_Face",
    "E13": "AOI_Center_Cross",
    "E14": "AOI_Sun",
}

EVENT_SPECIFIC_MIN_ACTIVE_SEC = 0.5
ACTION_LAST_SECONDS_CHECK = 1.0


@dataclass(frozen=True)
class EventDefinition:
    """One row of the event-definition table."""

    event_id: str
    start_frame: int
    end_frame: int
    target_aois: tuple[str, ...]
    process_aois: tuple[str, ...] = ()


def parse_aoi_list(value: object) -> tuple[str, ...]:
    """Parse comma-separated AOI fields from the event table."""

    if pd.isna(value) or str(value).strip() in {"", "None"}:
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def load_event_definitions(path: str) -> list[EventDefinition]:
    """Load event definitions from a CSV file."""

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Event_ID", "Start_Frame", "End_Frame", "Target_AOI", "Process_AOIs"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Event definition file is missing columns: {', '.join(missing)}")

    events: list[EventDefinition] = []
    for row in df.to_dict("records"):
        events.append(
            EventDefinition(
                event_id=str(row["Event_ID"]).strip(),
                start_frame=int(row["Start_Frame"]),
                end_frame=int(row["End_Frame"]),
                target_aois=parse_aoi_list(row["Target_AOI"]),
                process_aois=parse_aoi_list(row["Process_AOIs"]),
            )
        )
    return events


def load_labeled_gaze(path: str) -> pd.DataFrame:
    """Read a labeled gaze CSV and validate the columns used by extraction."""

    df = pd.read_csv(path)
    required = {"Frame", "Gaze_AOI"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Labeled gaze file is missing columns: {', '.join(missing)}")
    df = df.copy()
    df["Frame"] = df["Frame"].astype(int)
    df["Gaze_AOI"] = df["Gaze_AOI"].astype(str).str.strip()
    return df.sort_values("Frame").reset_index(drop=True)


def frames_from_seconds(seconds: float, fps: float) -> int:
    """Convert seconds to at least one frame."""

    return max(1, int(round(float(seconds) * float(fps))))


def select_event_window(labeled_gaze: pd.DataFrame, event: EventDefinition) -> pd.DataFrame:
    """Select an inclusive event window from labeled gaze data."""

    return labeled_gaze[
        (labeled_gaze["Frame"] >= event.start_frame)
        & (labeled_gaze["Frame"] <= event.end_frame)
    ].copy()


def build_fixation_blocks(gaze_aoi_series: Iterable[object], min_dwell_frames: int) -> pd.DataFrame:
    """Build duration-qualified consecutive AOI runs.

    ``is_valid`` only means the run reaches the minimum dwell duration. It is
    not a physiological fixation detector.
    """

    series = pd.Series(gaze_aoi_series)
    if series.empty:
        return pd.DataFrame(columns=["block_id", "aoi", "start_idx", "end_idx", "n_frames", "is_valid"])

    frame = series.reset_index(drop=False)
    frame.columns = ["orig_index", "aoi"]
    frame["block_id"] = (frame["aoi"] != frame["aoi"].shift()).cumsum()

    rows = []
    for block_id, group in frame.groupby("block_id", sort=True):
        n_frames = len(group)
        rows.append(
            {
                "block_id": int(block_id),
                "aoi": group["aoi"].iloc[0],
                "start_idx": int(group["orig_index"].iloc[0]),
                "end_idx": int(group["orig_index"].iloc[-1]),
                "n_frames": int(n_frames),
                "is_valid": bool(n_frames >= min_dwell_frames),
            }
        )
    return pd.DataFrame(rows)


def expand_valid_fixation_series(blocks: pd.DataFrame) -> pd.Series:
    """Expand valid AOI runs back to a sparse AOI series indexed by row offset."""

    if blocks is None or blocks.empty:
        return pd.Series(dtype=object)

    pieces = []
    for _, row in blocks[blocks["is_valid"]].iterrows():
        index = range(int(row["start_idx"]), int(row["end_idx"]) + 1)
        pieces.append(pd.Series([row["aoi"]] * len(list(index)), index=index))

    if not pieces:
        return pd.Series(dtype=object)
    out = pd.concat(pieces).sort_index()
    out.name = "Gaze_AOI"
    return out


def build_clean_aoi_sequence(blocks: pd.DataFrame) -> list[str]:
    """Build a sequence for transition/path metrics.

    ``NoGazeData`` becomes a gap token. Short non-missing runs are ignored but
    do not force a break.
    """

    if blocks is None or blocks.empty:
        return []

    sequence: list[str] = []
    for row in blocks.itertuples():
        if row.aoi == "NoGazeData":
            if sequence and sequence[-1] != MISSING_GAP_TOKEN:
                sequence.append(MISSING_GAP_TOKEN)
            continue
        if not bool(row.is_valid):
            continue
        if not sequence or sequence[-1] != row.aoi:
            sequence.append(row.aoi)
    return sequence


def count_aoi_transitions(clean_sequence: Iterable[str]) -> int:
    """Count AOI state changes without crossing missing-data gaps."""

    count = 0
    previous = None
    for aoi in clean_sequence:
        if aoi == MISSING_GAP_TOKEN:
            previous = None
            continue
        if previous is not None and aoi != previous:
            count += 1
        previous = aoi
    return int(count)


def count_aoi_visits(clean_sequence: Iterable[str], target_aoi: str) -> int:
    """Count separate visits to an AOI without crossing missing-data gaps."""

    count = 0
    previous = None
    for aoi in clean_sequence:
        if aoi == MISSING_GAP_TOKEN:
            previous = None
            continue
        if aoi == target_aoi and previous != target_aoi:
            count += 1
        previous = aoi
    return int(count)


def check_gaze_sequence(sequence: Iterable[str], trigger_aois: Iterable[str], target_aoi: str) -> int:
    """Return 1 when a trigger AOI is followed by a target AOI in the same observed segment."""

    triggers = set(trigger_aois)
    trigger_seen = False
    for aoi in sequence:
        if aoi == MISSING_GAP_TOKEN:
            trigger_seen = False
            continue
        if aoi in triggers:
            trigger_seen = True
            continue
        if trigger_seen and aoi == target_aoi:
            return 1
    return 0


def calculate_social_switch_metrics(
    blocks: pd.DataFrame,
    social_aoi: str,
    object_aois: Iterable[str],
    fps: float,
) -> tuple[int, float]:
    """Calculate direct social-object switch count and rate."""

    if blocks is None or blocks.empty or fps <= 0:
        return 0, np.nan

    object_set = set(object_aois)
    valid = blocks[(blocks["is_valid"]) & (blocks["aoi"] != "NoGazeData")]
    valid_duration = int(valid["n_frames"].sum()) / fps

    count = 0
    previous = None
    for row in blocks.itertuples():
        if row.aoi == "NoGazeData":
            previous = None
            continue
        if not bool(row.is_valid):
            continue
        if row.aoi == social_aoi:
            current = "social"
        elif row.aoi in object_set:
            current = "object"
        else:
            previous = None
            continue
        if previous is not None and current != previous:
            count += 1
        previous = current

    rate = count / valid_duration if valid_duration > 0 else np.nan
    return int(count), float(round(rate, 3)) if pd.notna(rate) else np.nan


def calculate_perseverative_gaze_ratio(
    aoi_counts: dict[str, int] | pd.Series,
    perseverative_aoi: str,
    window_total_frames: int,
) -> float:
    """Return the event proportion spent on the previous task AOI."""

    perseverative_frames = aoi_counts.get(perseverative_aoi, 0)
    return perseverative_frames / window_total_frames if window_total_frames > 0 else 0


def extract_path_facts(
    gaze_sequence: list[str],
    all_relevant_aois: Iterable[str],
    cleaned_counts: dict[str, int] | pd.Series,
    fps: float,
) -> dict:
    """Extract duration-qualified AOI path facts for RJA pattern scoring."""

    simplified_path: list[str] = []
    if gaze_sequence:
        simplified_path = [gaze_sequence[0]]
        for aoi in gaze_sequence[1:]:
            if aoi != simplified_path[-1]:
                simplified_path.append(aoi)

    targets_met_details = {}
    for target in set(all_relevant_aois):
        duration_frames = cleaned_counts.get(target, 0)
        duration_sec = round(duration_frames / fps, 3) if fps > 0 else 0
        targets_met_details[target] = {
            "met": target in simplified_path,
            "duration_sec": duration_sec,
        }

    return {
        "simplified_gaze_path": simplified_path,
        "targets_met_details": targets_met_details,
    }


def get_post_target_face_visit_duration_sec(
    block_df: pd.DataFrame,
    final_target: str | None,
    fps: float,
) -> float:
    """Return the first qualifying face-visit duration after the final target."""

    if block_df is None or block_df.empty or not final_target or fps is None or fps <= 0:
        return 0.0

    target_seen = False
    face_started = False
    face_frames = 0

    for row in block_df.itertuples():
        if row.aoi == "NoGazeData":
            if target_seen:
                break
            continue

        if not bool(row.is_valid):
            continue

        if not target_seen:
            if row.aoi == final_target:
                target_seen = True
            continue

        if row.aoi == "AOI_Teacher_Face":
            face_started = True
            face_frames += int(row.n_frames)
            continue

        if face_started:
            break

    return float(face_frames / fps)


def classify_rja_path_pattern(path_facts: dict, target_aois: list[str] | tuple[str, ...]) -> str:
    """Classify an RJA event path using the canonical rule order."""

    try:
        final_target = target_aois[-1]
        details = path_facts["targets_met_details"]

        saw_final_target = details.get(final_target, {}).get("met", False)
        saw_face = details.get("AOI_Teacher_Face", {}).get("met", False)
        saw_hands = details.get("AOI_Teacher_Hands", {}).get("met", False)
        path = path_facts["simplified_gaze_path"]

        def check_order(seq: list[str], first: str, second: str) -> bool:
            first_seen = False
            for aoi in seq:
                if aoi == MISSING_GAP_TOKEN:
                    first_seen = False
                    continue
                if aoi == first:
                    first_seen = True
                    continue
                if first_seen and aoi == second:
                    return True
            return False

        if saw_final_target:
            if saw_face and check_order(path, final_target, "AOI_Teacher_Face"):
                post_target_face_sec = float(path_facts.get("post_target_face_visit_sec", 0.0))
                if post_target_face_sec >= RJA_BRIEF_CHECK_SEC:
                    return "SUCCESS_CHECKBACK_PATTERN"
                return "SUCCESS_OTHER_BRIEF_CHECK"

            if check_order(path, "AOI_Teacher_Face", final_target):
                return "SUCCESS_FACE_TARGET"

            if check_order(path, "AOI_Teacher_Hands", final_target):
                return "SUCCESS_HANDS_TARGET"

            return "SUCCESS_OTHER"

        duration_face = details.get("AOI_Teacher_Face", {}).get("duration_sec", 0)
        duration_hands = details.get("AOI_Teacher_Hands", {}).get("duration_sec", 0)
        significant_sec = RJA_SIGNIFICANT_FIXATION_SEC or SIGNIFICANT_FIXATION_SEC

        is_long_hands_visit = saw_hands and duration_hands >= significant_sec
        is_long_face_visit = saw_face and duration_face >= significant_sec

        if is_long_hands_visit:
            return "FAILED_SIGNIFICANT_FIXATION_ON_HANDS"

        if is_long_face_visit:
            return "FAILED_SIGNIFICANT_FIXATION_ON_FACE"

        if saw_hands and saw_face:
            return "FAILED_SOCIAL_LOOP_BRIEF"

        if saw_hands:
            return "FAILED_BRIEF_LOOK_ON_HANDS"

        if saw_face:
            return "FAILED_BRIEF_LOOK_ON_FACE"

        return "FAILED_DISENGAGED"

    except (KeyError, IndexError, TypeError):
        return "UNKNOWN_ERROR"


def analyze_rja_event(context: dict, previous_task_aoi: str) -> dict:
    """Analyze an RJA event and return nested event-specific indicators."""

    aoi_counts = context["aoi_counts"]
    total_frames = context["window_total_frames"]
    target_aois = context["target_aois"]
    process_aois = context.get("process_aois", [])

    perseverative_ratio = calculate_perseverative_gaze_ratio(
        aoi_counts,
        previous_task_aoi,
        total_frames,
    )

    final_target = target_aois[-1] if target_aois else None
    hand_frames = aoi_counts.get("AOI_Teacher_Hands", 0)
    target_frames = aoi_counts.get(final_target, 0) if final_target else 0
    hand_to_target_dwell_ratio = hand_frames / target_frames if target_frames > 0 else np.nan

    path_facts = context.get("path_facts")
    if path_facts is None:
        all_relevant_aois = list(target_aois) + list(process_aois)
        path_facts = extract_path_facts(
            context.get("clean_aoi_sequence", []),
            all_relevant_aois,
            aoi_counts,
            context["fps"],
        )

    path_facts = dict(path_facts)
    path_facts["post_target_face_visit_sec"] = get_post_target_face_visit_duration_sec(
        context.get("fixation_blocks_df"),
        final_target,
        context["fps"],
    )

    path_pattern = str(classify_rja_path_pattern(path_facts, target_aois) or "UNKNOWN_ERROR")
    rja_path_score = float(RJA_PATTERN_SCORE_MAP.get(path_pattern, 0))

    return {
        "Dimension_Indicators": {
            "Attention_Shifting": {
                "Perseverative_Gaze_Ratio": perseverative_ratio,
            },
        },
        "Event_Specific": {
            "RJA_Analysis": {
                "Hand_To_Target_Bias": hand_to_target_dwell_ratio,
                "RJA_Path_Score": rja_path_score,
                "RJA_Is_SuccessLike": 1.0 if path_pattern.startswith("SUCCESS") else 0.0,
                "RJA_Is_CheckbackLike": 1.0 if "CHECKBACK" in path_pattern else 0.0,
                "RJA_Is_FixationFailure": 1.0 if "FIXATION" in path_pattern else 0.0,
                "path_pattern": path_pattern,
            }
        },
    }


def analyze_e12_transition(context: dict) -> dict:
    """Analyze the center-cross transition event."""

    cross_aoi = "AOI_Center_Cross"
    fixation_details = context["fixation_details"]
    success = 1 if cross_aoi in fixation_details else 0
    latency = (
        (fixation_details[cross_aoi]["first_frame"] - context["start_frame"]) / context["fps"]
        if success
        else np.nan
    )
    return {
        "Dimension_Indicators": {},
        "Event_Specific": {
            "Guidance_Success_Strict": success,
            "Guidance_Latency_sec": latency,
        },
    }


def analyze_e01_initial_face_preference(context: dict) -> dict:
    """Analyze Table S3 initial face preference and mean face-dwell episode duration."""

    counts = context["aoi_counts"]
    clean_sequence = context.get("clean_aoi_sequence", [])
    fps = context.get("fps", DEFAULT_VIDEO_FPS)
    min_active_frames = frames_from_seconds(EVENT_SPECIFIC_MIN_ACTIVE_SEC, fps)

    ignored_aois = {"AOI_Teacher_Face", "AOI_Teacher_Hands", "Other", "NoGazeData"}
    object_frames = sum(count for aoi, count in counts.items() if aoi not in ignored_aois)
    face_frames = counts.get("AOI_Teacher_Face", 0)
    active_attention_time = face_frames + object_frames
    initial_face_preference_index = (
        face_frames / active_attention_time if active_attention_time > min_active_frames else np.nan
    )

    face_visits = count_aoi_visits(clean_sequence, "AOI_Teacher_Face")
    mean_face_dwell_episode_duration_sec = (
        (face_frames / fps) / face_visits if face_visits > 0 and fps > 0 else np.nan
    )

    return {
        "Dimension_Indicators": {},
        "Event_Specific": {
            "Face_Preference_Index": initial_face_preference_index,
            "Eye_Contact_Maintenance_sec": mean_face_dwell_episode_duration_sec,
        },
    }


def analyze_action_event(context: dict, include_switch_metrics: bool = False) -> dict:
    """Analyze E02/E10 action-completion and end-window completion indicators."""

    reaction_slice = context["reaction_slice"]
    fps = context.get("fps", DEFAULT_VIDEO_FPS)
    last_frames_check = frames_from_seconds(ACTION_LAST_SECONDS_CHECK, fps)
    target_aois = context.get("target_aois", []) or []
    process_aois = context.get("process_aois", []) or []
    final_target = target_aois[-1] if target_aois else None
    completion_targets = [final_target] if final_target else []
    last_phase = reaction_slice.tail(last_frames_check) if len(reaction_slice) >= last_frames_check else reaction_slice

    if completion_targets:
        end_window_completion_indicator = last_phase["Gaze_AOI"].isin(completion_targets).any()
        action_completion_indicator = reaction_slice["Gaze_AOI"].isin(completion_targets).any()
    else:
        end_window_completion_indicator = False
        action_completion_indicator = False

    event_specific = {
        "Action_Completion_FinalPhase": int(end_window_completion_indicator),
        "Action_Completion_Anytime": int(action_completion_indicator),
    }

    if include_switch_metrics:
        object_aois = list(
            dict.fromkeys(
                ["AOI_Teacher_Hands"]
                + [aoi for aoi in process_aois if aoi not in {"AOI_Teacher_Face", "AOI_Teacher_Hands"}]
                + [aoi for aoi in target_aois if aoi not in {"AOI_Teacher_Face", "AOI_Teacher_Hands"}]
            )
        )
        switch_count, _ = calculate_social_switch_metrics(
            context.get("fixation_blocks_df"),
            "AOI_Teacher_Face",
            object_aois,
            fps,
        )
        event_specific["Face_Object_Switch_Count"] = int(switch_count)

    return {"Dimension_Indicators": {}, "Event_Specific": event_specific}


def analyze_e03_pre_face_task_object_dwell(context: dict) -> dict:
    """Measure Table S3 pre-face task-object dwell time."""

    fixation_details = context.get("fixation_details", {})
    reaction_slice = context.get("reaction_slice")
    fps = context.get("fps", DEFAULT_VIDEO_FPS)
    target_aois = context.get("target_aois", []) or []
    process_aois = context.get("process_aois", []) or []
    indicators = {}

    candidate_task_aois = [
        aoi
        for aoi in (process_aois + target_aois)
        if aoi not in {"AOI_Teacher_Face", "AOI_Teacher_Hands", "AOI_Center_Cross"}
    ]
    disengage_from_aoi = candidate_task_aois[-1] if candidate_task_aois else "AOI_Block_Tower"

    if "AOI_Teacher_Face" in fixation_details and reaction_slice is not None and not reaction_slice.empty:
        first_face_frame = fixation_details["AOI_Teacher_Face"]["first_frame"]
        pre_face_slice = reaction_slice[reaction_slice["Frame"] < first_face_frame]
        pre_face_task_object_frames = int((pre_face_slice["Gaze_AOI"] == disengage_from_aoi).sum())
        indicators["Disengagement_Latency_sec"] = pre_face_task_object_frames / fps if fps > 0 else np.nan
        indicators["Disengage_From_AOI"] = disengage_from_aoi

    return {"Dimension_Indicators": {"Attention_Shifting": indicators}, "Event_Specific": {}}


def analyze_social_checkback_event(context: dict, trigger_aois: Iterable[str], output_key: str) -> dict:
    """Analyze whether gaze checks back to the teacher face after trigger AOIs."""

    checkback_occurred = check_gaze_sequence(context["clean_aoi_sequence"], trigger_aois, "AOI_Teacher_Face")
    return {
        "Dimension_Indicators": {},
        "Event_Specific": {output_key: checkback_occurred},
    }


def analyze_exploration_event(context: dict) -> dict:
    """Analyze exploratory AOI behavior in the expectation-building event."""

    aoi_counts = context["aoi_counts"]
    total_frames = context["window_total_frames"]
    attention_shift_count = context["attention_shift_count"]

    if total_frames <= 0:
        return {
            "Dimension_Indicators": {},
            "Event_Specific": {
                "Exploratory_Score": np.nan,
                "Exploration_Ratio": np.nan,
            },
        }

    social_aois = {"AOI_Teacher_Face", "AOI_Teacher_Hands"}
    exploration_frames = sum(count for aoi, count in aoi_counts.items() if aoi not in social_aois)
    exploration_ratio = exploration_frames / total_frames
    exploratory_score = attention_shift_count * exploration_ratio

    return {
        "Dimension_Indicators": {},
        "Event_Specific": {
            "Exploratory_Score": exploratory_score,
            "Exploration_Ratio": exploration_ratio,
        },
    }


def analyze_event(
    labeled_gaze: pd.DataFrame,
    event: EventDefinition,
    fps: float = DEFAULT_VIDEO_FPS,
    min_dwell_ms: int = DEFAULT_MIN_DWELL_MS,
) -> dict:
    """Compute the core event-level behavior report for one event."""

    window = select_event_window(labeled_gaze, event)
    total_frames = len(window)
    event_info = {
        "event_id": event.event_id,
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "target_aois": list(event.target_aois),
        "process_aois": list(event.process_aois),
    }
    if total_frames == 0:
        return {
            **event_info,
            "Task_Success_Flag": np.nan,
            "Execution_Status": "NO_DATA",
            "Dimension_Indicators": {},
            "Event_Specific": {},
        }

    min_dwell_frames = frames_from_seconds(min_dwell_ms / 1000.0, fps)
    blocks = build_fixation_blocks(window["Gaze_AOI"], min_dwell_frames)
    valid_blocks = blocks[(blocks["is_valid"]) & (blocks["aoi"] != "NoGazeData")].copy()
    if valid_blocks.empty:
        return {
            **event_info,
            "Task_Success_Flag": np.nan,
            "Execution_Status": "NO_VALID_BEHAVIOR_AOI",
            "Dimension_Indicators": {},
            "Event_Specific": {},
        }

    clean_sequence = build_clean_aoi_sequence(blocks)
    valid_series = expand_valid_fixation_series(valid_blocks)
    counts = valid_blocks.groupby("aoi")["n_frames"].sum().sort_values(ascending=False)

    face_frames = counts.get("AOI_Teacher_Face", 0)
    hands_frames = counts.get("AOI_Teacher_Hands", 0)
    relevant_aois = list(dict.fromkeys((*event.target_aois, *event.process_aois)))
    task_objects = [aoi for aoi in relevant_aois if aoi in TASK_OBJECT_AOIS]
    task_object_frames = sum(counts.get(aoi, 0) for aoi in task_objects)
    other_frames = counts.get("Other", 0)

    duration_sec = total_frames / fps if fps > 0 else np.nan
    results: dict[str, object] = {
        "Dwell_On_Face_Ratio": face_frames / total_frames,
        "Dwell_On_Hands_Ratio": hands_frames / total_frames,
        "Dwell_On_Other_Ratio": other_frames / total_frames,
    }
    if task_objects:
        results["Dwell_On_Task_Object_Ratio"] = task_object_frames / total_frames

    results["Attention_Shift_Count"] = count_aoi_transitions(clean_sequence)
    results["Attention_Shift_Freq"] = (
        round(results["Attention_Shift_Count"] / duration_sec, 3)
        if pd.notna(duration_sec) and duration_sec > 0
        else np.nan
    )

    social_frames = face_frames + hands_frames
    physical_frames = sum(counts.get(aoi, 0) for aoi in counts.keys() if aoi not in {"AOI_Teacher_Face", "AOI_Teacher_Hands"})
    total_relevant_frames = social_frames + physical_frames
    results["Social_Preference_Bias"] = (
        round(social_frames / total_relevant_frames, 3)
        if total_relevant_frames > 0
        else np.nan
    )

    final_target = event.target_aois[-1] if event.target_aois else None
    latency = np.nan
    if final_target:
        hit_blocks = valid_blocks[valid_blocks["aoi"] == final_target]
        if not hit_blocks.empty:
            first_idx = int(hit_blocks.iloc[0]["start_idx"])
            first_frame = window.loc[first_idx, "Frame"]
            latency = (first_frame - event.start_frame) / fps if fps > 0 else np.nan
    results["Cue_to_Target_Latency_sec"] = latency
    results["Task_Success_Flag"] = 0.0 if np.isnan(latency) else 1.0

    if event.event_id in {"E02", "E05", "E06", "E07", "E09", "E10", "E11", "E13", "E14"}:
        object_aois = [aoi for aoi in set(relevant_aois) if "Face" not in aoi]
        _, switch_rate = calculate_social_switch_metrics(blocks, "AOI_Teacher_Face", object_aois, fps)
        results["Social_Switch_Rate"] = switch_rate

    path_aois = list(dict.fromkeys([*event.target_aois, *event.process_aois]))
    if event.event_id in RJA_PREVIOUS_TASK_AOI:
        path_aois = list(dict.fromkeys([*path_aois, "AOI_Teacher_Face", "AOI_Teacher_Hands"]))
    if not path_aois:
        path_aois = ["AOI_Teacher_Face"]
    path_facts = extract_path_facts(clean_sequence, path_aois, counts, fps)

    fixation_details = {}
    for aoi in dict.fromkeys([*path_aois, "AOI_Teacher_Face", "AOI_Teacher_Hands", "AOI_Center_Cross"]):
        matches = valid_series == aoi
        if matches.any():
            offset = valid_series[matches].index[0]
            fixation_details[aoi] = {"first_frame": int(window.loc[offset, "Frame"])}

    context = {
        "clean_aoi_sequence": clean_sequence,
        "fixation_details": fixation_details,
        "start_frame": event.start_frame,
        "fps": fps,
        "aoi_counts": counts.to_dict(),
        "fixation_blocks_df": blocks,
        "window_total_frames": total_frames,
        "attention_shift_count": results["Attention_Shift_Count"],
        "target_aois": list(event.target_aois),
        "process_aois": list(event.process_aois),
        "path_facts": path_facts,
        "reaction_slice": window,
    }
    if event.event_id in RJA_PREVIOUS_TASK_AOI:
        specific = analyze_rja_event(context, RJA_PREVIOUS_TASK_AOI[event.event_id])
    elif event.event_id == "E01":
        specific = analyze_e01_initial_face_preference(context)
    elif event.event_id == "E02":
        specific = analyze_action_event(context, include_switch_metrics=True)
    elif event.event_id == "E03":
        specific = analyze_e03_pre_face_task_object_dwell(context)
    elif event.event_id == "E04":
        specific = analyze_exploration_event(context)
    elif event.event_id == "E12":
        specific = analyze_e12_transition(context)
    elif event.event_id == "E06":
        specific = analyze_social_checkback_event(
            context,
            ["AOI_Teacher_Hands", "AOI_Blue_Block", "AOI_Block_Tower"],
            "Social_Checkback_Occurred",
        )
    elif event.event_id == "E11":
        specific = analyze_social_checkback_event(context, ["AOI_Block_Tower", "AOI_Teacher_Hands"], "Social_Checkback_Occurred")
    elif event.event_id == "E07":
        specific = analyze_social_checkback_event(
            context,
            ["AOI_Teacher_Hands", "AOI_Left_Area"],
            "Uncertainty_Checkback_Occurred",
        )
    elif event.event_id == "E10":
        specific = analyze_action_event(context, include_switch_metrics=False)
    else:
        specific = {"Dimension_Indicators": {}, "Event_Specific": {}}

    return {
        **event_info,
        "Execution_Status": "SUCCESS",
        **results,
        **specific,
    }
