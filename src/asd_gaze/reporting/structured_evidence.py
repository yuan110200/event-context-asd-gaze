"""Build structured behavioral evidence for rule-based report templates.

The functions in this module implement the deterministic evidence contract used
before LLM rendering. They do not call an LLM and do not modify classifier
labels or calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from pathlib import Path
import csv
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPORTING_DOMAINS: dict[str, str] = {
    "face_and_social_gaze": "Face and social gaze allocation",
    "cue_guided_target_response": "Cue-guided orienting and target response",
    "gaze_shifting_and_reallocation": "Gaze shifting and reallocation",
}

_FINAL40_METADATA_PATH = Path(__file__).resolve().parents[3] / "configs" / "final40_reporting_metadata.csv"


def load_final40_metadata(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load the checked-in final-Train300 reporting qualification table."""
    metadata_path = path or _FINAL40_METADATA_PATH
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata = {row["raw_feature_key"]: row for row in rows}
    if len(metadata) != 40 or len(rows) != 40:
        raise ValueError("final40 metadata must contain 40 unique feature keys")
    return metadata


FINAL40_METADATA = load_final40_metadata()


@dataclass(frozen=True)
class EvidenceSpec:
    """Mapping from a feature key to a human-facing evidence type."""

    feature_key: str
    domain_id: str
    cross_event_link_key: str
    plain_behavioral_meaning: str
    level: str = "cross_event"
    event_id: str = "NA"
    measurement: str | None = None
    lower_state: str = "lower_tail"
    upper_state: str = "upper_tail"
    central_state: str = "central_reference"


DEFAULT_EVIDENCE_SPECS: tuple[EvidenceSpec, ...] = (
    EvidenceSpec(
        "E01.Event_Specific.Face_Preference_Index",
        "face_and_social_gaze",
        "face_directed_gaze",
        "time looking at the instructor's face",
        level="event",
        event_id="E01",
        measurement="initial face preference index",
    ),
    EvidenceSpec(
        "E01.Event_Specific.Eye_Contact_Maintenance_sec",
        "face_and_social_gaze",
        "social_region_gaze_allocation",
        "mean face-dwell episode duration",
        level="event",
        event_id="E01",
        measurement="mean face-dwell episode duration",
    ),
    EvidenceSpec(
        "E02.Event_Specific.Action_Completion_Anytime",
        "cue_guided_target_response",
        "target_response",
        "event-defined target-region response",
        event_id="E02",
        measurement="target response indicator",
        lower_state="uncommon_relative_to_TD",
        upper_state="reference_consistent",
        central_state="reference_consistent",
    ),
    EvidenceSpec(
        "E02.Event_Specific.Action_Completion_FinalPhase",
        "cue_guided_target_response",
        "target_response",
        "end-window target-region response",
        level="event",
        event_id="E02",
        measurement="end-window completion indicator",
        lower_state="uncommon_relative_to_TD",
        upper_state="reference_consistent",
        central_state="reference_consistent",
    ),
    EvidenceSpec(
        "E10.Event_Specific.Action_Completion_Anytime",
        "cue_guided_target_response",
        "target_response",
        "event-defined target-region response",
        level="event",
        event_id="E10",
        measurement="target response indicator",
        lower_state="uncommon_relative_to_TD",
        upper_state="reference_consistent",
        central_state="reference_consistent",
    ),
    EvidenceSpec(
        "E10.Event_Specific.Action_Completion_FinalPhase",
        "cue_guided_target_response",
        "target_response",
        "end-window target-region response",
        event_id="E10",
        level="event",
        measurement="end-window completion indicator",
        lower_state="uncommon_relative_to_TD",
        upper_state="reference_consistent",
        central_state="reference_consistent",
    ),
    EvidenceSpec(
        "E03.Dimension_Indicators.Attention_Shifting.Disengagement_Latency_sec",
        "gaze_shifting_and_reallocation",
        "face_object_switching",
        "pre-face task-object dwell time",
        level="event",
        event_id="E03",
        measurement="pre-face task-object dwell time",
    ),
    EvidenceSpec(
        "E07.Event_Specific.Uncertainty_Checkback_Occurred",
        "face_and_social_gaze",
        "uncertainty_checkback",
        "uncertainty-event checkback to the instructor's face",
        level="event",
        event_id="E07",
        measurement="uncertainty checkback indicator",
        lower_state="uncommon_relative_to_TD",
        upper_state="reference_consistent",
        central_state="reference_consistent",
    ),
    EvidenceSpec(
        "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency",
        "cue_guided_target_response",
        "target_orienting_latency",
        "first qualifying gaze timing to the event-defined target region",
        level="cross_event",
        measurement="first qualifying gaze to the event-defined target region",
    ),
)


def build_td_reference(
    td_feature_rows: pd.DataFrame,
    feature_keys: Sequence[str] | None = None,
    *,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> dict[str, dict[str, float | int]]:
    """Build TD reference distributions from the provided training partition.

    Callers must pass only the eligible TD rows for the current fold or final
    Train300 fit. No validation or Test70 rows are read here.
    """

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("lower_quantile and upper_quantile must satisfy 0 <= lower < upper <= 1")
    keys = list(feature_keys) if feature_keys is not None else list(td_feature_rows.columns)
    reference: dict[str, dict[str, float | int]] = {}
    for key in keys:
        if key not in td_feature_rows.columns:
            continue
        values = pd.to_numeric(td_feature_rows[key], errors="coerce").dropna()
        if values.empty:
            continue
        reference[key] = {
            "n": int(values.shape[0]),
            "median": float(values.median()),
            "lower": float(values.quantile(lower_quantile)),
            "upper": float(values.quantile(upper_quantile)),
        }
    return reference


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _feature_state(value: float, ref: Mapping[str, float | int], spec: EvidenceSpec) -> str:
    lower = float(ref["lower"])
    upper = float(ref["upper"])
    if value < lower:
        return spec.lower_state
    if value > upper:
        return spec.upper_state
    return spec.central_state


def _as_feature_mapping(subject_features: Mapping[str, Any] | pd.Series) -> Mapping[str, Any]:
    if isinstance(subject_features, pd.Series):
        return subject_features.to_dict()
    return subject_features


def build_structured_evidence(
    subject_id: str,
    subject_features: Mapping[str, Any] | pd.Series,
    td_reference: Mapping[str, Mapping[str, float | int]],
    specs: Iterable[EvidenceSpec] = DEFAULT_EVIDENCE_SPECS,
) -> list[dict[str, Any]]:
    """Create evidence items with feature, event, and domain provenance."""

    features = _as_feature_mapping(subject_features)
    evidence: list[dict[str, Any]] = []
    for spec in specs:
        if spec.feature_key not in features or spec.feature_key not in td_reference:
            continue
        raw = features[spec.feature_key]
        if _is_missing(raw):
            continue
        value = float(raw)
        ref = td_reference[spec.feature_key]
        state = _feature_state(value, ref, spec)
        metadata = FINAL40_METADATA.get(spec.feature_key)
        evidence_id = f"{subject_id}_EVID_{spec.cross_event_link_key}_{len(evidence) + 1}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "subject_id": subject_id,
                "feature_key": spec.feature_key,
                "event_id": spec.event_id,
                "level": spec.level,
                "behavioral_domain": spec.domain_id,
                "domain_label": REPORTING_DOMAINS[spec.domain_id],
                "cross_event_link_key": spec.cross_event_link_key,
                "measurement": spec.measurement or spec.plain_behavioral_meaning,
                "plain_behavioral_meaning": spec.plain_behavioral_meaning,
                "feature_state": state,
                "raw_value": value,
                "reference_median": float(ref["median"]),
                "reference_lower": float(ref["lower"]),
                "reference_upper": float(ref["upper"]),
                "reference_n": int(ref["n"]),
                "comparison_object": "TD reference median"
                if spec.cross_event_link_key == "target_orienting_latency"
                else "TD reference",
                "final40_status": "final40" if metadata else "unclassified",
                "display_name": metadata["display_name"] if metadata else spec.measurement or spec.plain_behavioral_meaning,
                "reporting_linkage": metadata["reporting_linkage"] if metadata else "unconfirmed",
                "claim_eligible": bool(metadata and metadata["reporting_linkage"] in {"F", "C", "G"}),
            }
        )
    return evidence


def _approved_findings_for_domain(domain_id: str, evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Custom compatibility specs may still produce auditable evidence, but cannot
    # enter the default approved-claim path without final40 metadata.
    evidence = [item for item in evidence if item.get("claim_eligible") is True]
    states = {item.get("feature_state") for item in evidence}
    findings: list[dict[str, Any]] = []
    if domain_id == "face_and_social_gaze":
        if any(
            item.get("cross_event_link_key") == "face_directed_gaze"
            and item.get("feature_state") == "lower_tail"
            for item in evidence
        ):
            findings.append(
                {
                    "behavior": "time looking at the instructor's face",
                    "relation": "less",
                    "comparator": "TD reference",
                    "scope": "selected interaction contexts",
                }
            )
        social_states = {
            item.get("feature_state")
            for item in evidence
            if item.get("cross_event_link_key") == "social_region_gaze_allocation"
        }
        if "lower_tail" in social_states:
            relation = "less"
        elif "upper_tail" in social_states:
            relation = "more"
        elif social_states and social_states <= {"central_reference", "reference_consistent", "intermediate"}:
            relation = "consistent"
        else:
            relation = ""
        if relation:
            findings.append(
                {
                    "behavior": "gaze allocation to face and hand regions versus nonsocial task regions",
                    "relation": relation,
                    "comparator": "nonsocial task regions; TD reference"
                    if relation in {"less", "more"}
                    else "TD reference",
                    "scope": "selected interaction contexts",
                }
            )
    elif domain_id == "cue_guided_target_response":
        target = [item for item in evidence if item.get("cross_event_link_key") == "target_response"]
        latency = [item for item in evidence if item.get("cross_event_link_key") == "target_orienting_latency"]
        if target and all(item.get("feature_state") == "reference_consistent" for item in target):
            findings.append(
                {
                    "behavior": "event-defined target-region response",
                    "relation": "consistent",
                    "comparator": "TD reference",
                    "scope": "evaluable cue-guided events",
                }
            )
        if target and any(item.get("feature_state") == "uncommon_relative_to_TD" for item in target):
            findings.append(
                {
                    "behavior": "event-defined target-region response",
                    "relation": "fewer or not reached",
                    "comparator": "TD reference",
                    "scope": "evaluable cue-guided events",
                }
            )
        for item in latency:
            raw = item.get("raw_value")
            median = item.get("reference_median")
            if raw is None or median is None:
                continue
            raw_f = float(raw)
            med_f = float(median)
            relation = "longer than" if raw_f > med_f else "shorter than" if raw_f < med_f else "matching"
            findings.append(
                {
                    "behavior": "first qualifying gaze timing to the event-defined target region",
                    "relation": relation,
                    "comparator": "TD reference median",
                    "scope": item.get("event_id"),
                    "observed_seconds": raw_f,
                    "reference_median_seconds": med_f,
                }
            )
    elif domain_id == "gaze_shifting_and_reallocation" and any(
        item.get("cross_event_link_key") == "face_object_switching" for item in evidence
    ):
        if "lower_tail" in states:
            relation = "less often"
        elif "upper_tail" in states:
            relation = "more often"
        elif states and states <= {"central_reference", "reference_consistent"}:
            relation = "consistent"
        else:
            relation = ""
        if relation:
            findings.append(
                {
                    "behavior": "gaze shifts between the instructor and task objects",
                    "relation": relation,
                    "comparator": "TD reference",
                    "scope": "approved gaze-shifting evidence",
                }
            )
    return findings


def build_claim_packet(
    subject_id: str,
    predicted_label: int,
    calibrated_probability: float,
    evidence: Sequence[Mapping[str, Any]],
    *,
    allow_unclassified: bool = False,
) -> dict[str, Any]:
    """Build claims; unclassified specs require explicit compatibility opt-in."""

    claims: list[dict[str, Any]] = [
        {
            "claim_id": f"{subject_id}_CLAIM_CLASSIFICATION",
            "claim_type": "classification_result",
            "section": "Model result",
            "dimension_id": "NA",
            "event_ids": [],
            "evidence_ids": [],
            "fact_slots": {
                "predicted_label": int(predicted_label),
                "predicted_class": "ASD" if int(predicted_label) == 1 else "TD",
                "calibrated_asd_probability": float(calibrated_probability),
            },
            "display": True,
        }
    ]
    for domain_id in REPORTING_DOMAINS:
        domain_evidence = [item for item in evidence if item.get("behavioral_domain") == domain_id]
        if allow_unclassified:
            domain_evidence = [item for item in domain_evidence if item.get("final40_status") == "unclassified"] or domain_evidence
            domain_evidence = [{**item, "claim_eligible": True} for item in domain_evidence]
        findings = _approved_findings_for_domain(domain_id, domain_evidence)
        if not findings:
            continue
        claims.append(
            {
                "claim_id": f"{subject_id}_CLAIM_DIM_{domain_id}",
                "claim_type": "dimension_pattern",
                "section": "Behavioral patterns",
                "dimension_id": domain_id,
                "domain_label": REPORTING_DOMAINS[domain_id],
                "event_ids": sorted(
                    {str(item.get("event_id")) for item in domain_evidence if item.get("event_id") not in {None, "NA"}}
                ),
                "evidence_ids": [str(item["evidence_id"]) for item in domain_evidence],
                "approved_findings": findings,
                "display": True,
            }
        )
    return {
        "subject_id": subject_id,
        "prediction": {
            "predicted_label": int(predicted_label),
            "calibrated_probability": float(calibrated_probability),
        },
        "reporting_domains": REPORTING_DOMAINS.copy(),
        "evidence_items": [dict(item) for item in evidence],
        "claims": claims,
    }


def _fmt(value: Any) -> str:
    value_f = float(value)
    if np.isclose(value_f, 0.0):
        return "0.0"
    return f"{value_f:.3f}".rstrip("0").rstrip(".")


def _sentence_from_findings(domain_id: str, findings: Sequence[Mapping[str, Any]]) -> str:
    if domain_id == "face_and_social_gaze":
        parts = []
        for finding in findings:
            if finding["behavior"] == "time looking at the instructor's face":
                parts.append(f"the child spent {finding['relation']} time looking at the instructor's face")
            else:
                parts.append(
                    "gaze allocation to face and hand regions was "
                    f"{finding['relation']} relative to {finding['comparator']}"
                )
        return "Across selected interaction contexts, " + "; ".join(parts) + "."
    if domain_id == "cue_guided_target_response":
        parts = []
        for finding in findings:
            if finding["behavior"] == "event-defined target-region response":
                parts.append(f"event-defined target-region response was {finding['relation']}")
            else:
                observed = _fmt(finding["observed_seconds"])
                ref = _fmt(finding["reference_median_seconds"])
                scope = finding.get("scope") or "selected latency evidence"
                parts.append(
                    f"{scope} first qualifying gaze occurred after {observed} s, "
                    f"{finding['relation']} the TD reference median of {ref} s"
                )
        return "In cue-guided events, " + "; ".join(parts) + "."
    if domain_id == "gaze_shifting_and_reallocation":
        finding = findings[0]
        return (
            "For approved gaze-shifting evidence, gaze shifts between the instructor "
            f"and task objects occurred {finding['relation']} relative to the TD reference."
        )
    raise ValueError(f"Unknown reporting domain: {domain_id}")


def render_claim_fallback(claim: Mapping[str, Any]) -> str:
    """Render one approved dimension claim without involving the LLM."""
    return _sentence_from_findings(str(claim["dimension_id"]), claim.get("approved_findings", []))


def render_rule_based_report(packet: Mapping[str, Any]) -> str:
    """Render deterministic text from approved claims only."""

    prediction = packet["prediction"]
    pred_class = "ASD" if int(prediction["predicted_label"]) == 1 else "TD"
    lines = [
        f"Predicted class: {pred_class}",
        f"Calibrated ASD probability: {float(prediction['calibrated_probability']):.3f}",
    ]
    claims_by_domain = {
        claim["dimension_id"]: claim
        for claim in packet.get("claims", [])
        if claim.get("claim_type") == "dimension_pattern"
    }
    for domain_id, label in REPORTING_DOMAINS.items():
        lines.append("")
        lines.append(label)
        claim = claims_by_domain.get(domain_id)
        if not claim:
            lines.append("No predefined evidence-supported statement was generated for this domain.")
            continue
        lines.append(_sentence_from_findings(domain_id, claim.get("approved_findings", [])))
    return "\n".join(lines)
