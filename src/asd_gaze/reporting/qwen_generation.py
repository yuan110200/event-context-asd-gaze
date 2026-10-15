"""Per-claim Qwen realization and deterministic report assembly."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from .structured_evidence import REPORTING_DOMAINS, render_claim_fallback, render_rule_based_report

QWEN_CONFIG = {"model": "qwen2.5:32b", "quantization": "Q4_K_M",
               "temperature": 0.2, "top_p": 0.9, "context": 8192,
               "max_output_tokens": 220}

SYSTEM_PROMPT = """Role: You are a language editor for evidence-grounded behavioral reports. Your role is to express approved behavioral observations, not to provide clinical interpretations.

Task: Given one approved behavioral claim and its structured evidence, write one clear, natural English sentence. Preserve the supplied claim_id and event_id.

Evidence Constraints: Preserve all approved behavioral facts, including the event context, observed behavior, response status, timing information, and comparison direction, as applicable. Maintain an explicit TD-reference comparison when required by the claim. Do not introduce unsupported information or alter the meaning of the evidence. Avoid unsupported clinical interpretations, causal explanations, or judgments about cognitive abilities, impairment, diagnosis, or prognosis.

Writing Style: Use clear, natural, and concise English. Prefer direct behavioral descriptions over formulaic openings, repetitive phrasing, or unnecessary technical terminology.

Output: Return only a valid JSON object matching the provided schema, with exactly three fields: claim_id, event_id, and text. Use null for event_id when not applicable."""

CLAIM_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["claim_id", "event_id", "text"],
    "properties": {
        "claim_id": {"type": "string"},
        "event_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "text": {"type": "string"},
    },
}

_FORBIDDEN = re.compile(
    r"social cognition|social-processing difficulty|social information processing|social motivation|"
    r"executive dysfunction|impaired|deficit|disorder|rigidity|pathological|treatment|therapy|"
    r"intervention|diagnostic sign|ASD-like|TD-like|typical|atypical|normal|abnormal|healthy|"
    r"difficulty processing|suggesting|indicating|reflecting|because|therefore|diagnosis|prognosis|"
    r"z-score|percentile|Dwell_On|Social_Preference|Task_Success|Cue_to_Target|Social_Switch|"
    r"RJA|CORM|RQA|IQR|P10|P90|metric|feature", re.IGNORECASE)


def ollama_generate(system_prompt: str, user_prompt: str, schema: Mapping[str, Any], *,
                    base_url: str | None = None, timeout: float = 180.0) -> Any:
    """Call Ollama and return the decoded claim object, or ``None`` on failure."""
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": QWEN_CONFIG["model"],
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "stream": False, "format": schema,
        "options": {"temperature": QWEN_CONFIG["temperature"], "top_p": QWEN_CONFIG["top_p"],
                     "num_ctx": QWEN_CONFIG["context"], "num_predict": QWEN_CONFIG["max_output_tokens"]},
    }
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        content = envelope.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        return json.loads(content)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


def _claims(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(c["claim_id"]): c for c in packet.get("claims", [])
            if c.get("claim_type") != "classification_result" and c.get("display", True)}


def build_qwen_request(packet: Mapping[str, Any], claim_id: str | None = None) -> dict[str, Any]:
    """Build one S1 claim request; no classifier output is sent to Qwen."""
    claims = _claims(packet)
    if claim_id is None:
        if len(claims) != 1:
            raise ValueError("claim_id is required when the packet has multiple approved claims")
        claim_id = next(iter(claims))
    if claim_id not in claims:
        raise ValueError(f"Unknown approved claim: {claim_id}")
    claim = claims[claim_id]
    packet_evidence_ids = {str(item.get("evidence_id")) for item in packet.get("evidence_items", [])}
    if not set(claim.get("evidence_ids", [])).issubset(packet_evidence_ids):
        raise ValueError(f"Claim {claim_id} references evidence outside the packet")
    evidence = [item for item in packet.get("evidence_items", [])
                if str(item.get("evidence_id")) in set(claim.get("evidence_ids", []))]
    return {"subject_id": packet.get("subject_id"),
            "approved_claim": {"claim_id": claim_id,
                                "event_ids": list(claim.get("event_ids", [])),
                                "approved_findings": claim.get("approved_findings", []),
                                "evidence": evidence},
            "output_contract": {"claim_id": "string", "event_id": "string or null", "text": "string"}}


def validate_qwen_response(response: Any, claim: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one S1 response and its approved event/evidence provenance."""
    errors: list[str] = []
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = None
            errors.append("invalid_json")
    if not isinstance(response, dict):
        return {"valid": False, "errors": errors + ["response_not_object"], "response": None}
    if set(response) != {"claim_id", "event_id", "text"}:
        errors.append("schema_keys")
    if response.get("claim_id") != claim.get("claim_id"):
        errors.append("claim_id_mismatch")
    expected_events = list(claim.get("event_ids", []))
    event_id = response.get("event_id")
    if expected_events:
        if event_id not in expected_events:
            errors.append("event_id_mismatch")
    elif event_id is not None:
        errors.append("unexpected_event_id")
    text = response.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append("missing_text")
    else:
        if _FORBIDDEN.search(text):
            errors.append("forbidden_language")
        numbers_in_text = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text))
        approved_numbers = set()
        for item in packet.get("evidence_items", []):
            if str(item.get("evidence_id")) not in set(claim.get("evidence_ids", [])):
                continue
            for field in ("raw_value", "reference_median", "reference_lower", "reference_upper"):
                value = item.get(field)
                if value is not None:
                    approved_numbers.add(str(value))
                    approved_numbers.add(f"{float(value):.3f}".rstrip("0").rstrip("."))
        identifiers = re.findall(r"\b(?:E0[1-9]|E1[0-4])\b", text)
        if identifiers or not numbers_in_text.issubset(approved_numbers):
            errors.append("raw_identifier_or_number")
        relation = " ".join(str(item.get("relation", "")) for item in claim.get("approved_findings", [])).lower()
        low = text.lower()
        direction_terms = {
            "less": ("less", "lower", "fewer"),
            "fewer or not reached": ("fewer", "not reached", "less"),
            "less often": ("less", "fewer"),
            "more": ("more", "higher", "greater"),
            "more often": ("more", "higher", "greater"),
            "longer than": ("longer",),
            "shorter than": ("shorter",),
            "matching": ("matching", "consistent"),
            "consistent": ("consistent", "matching"),
        }
        expected_terms = direction_terms.get(relation)
        if expected_terms and not any(term in low for term in expected_terms):
            errors.append("direction_mismatch")
        opposite_terms = {
            "less": ("more", "higher", "greater"),
            "fewer or not reached": ("more", "higher", "reached the expected"),
            "less often": ("more", "higher"),
            "more": ("less", "lower", "fewer"),
            "more often": ("less", "lower", "fewer"),
            "longer than": ("shorter",),
            "shorter than": ("longer",),
        }
        if any(term in low for term in opposite_terms.get(relation, ())):
            errors.append("direction_mismatch")
    packet_evidence_ids = {str(item.get("evidence_id")) for item in packet.get("evidence_items", [])}
    if not set(claim.get("evidence_ids", [])).issubset(packet_evidence_ids):
        errors.append("evidence_id_mismatch")
    return {"valid": not errors, "errors": sorted(set(errors)), "response": response}


def _assemble_qwen_report(packet: Mapping[str, Any], responses: Mapping[str, Mapping[str, Any]]) -> str:
    prediction = packet["prediction"]
    lines = ["# Individual Behavioral Profile", "", "## Model result",
             f"Predicted label: {prediction['predicted_label']}",
             f"Calibrated probability: {float(prediction['calibrated_probability']):.3f}", ""]
    claims = _claims(packet)
    for domain_id, label in REPORTING_DOMAINS.items():
        lines += [f"## {label}"]
        domain_claims = [claim for claim in claims.values() if claim.get("dimension_id") == domain_id]
        texts = [responses[claim["claim_id"]]["text"] for claim in domain_claims if claim["claim_id"] in responses]
        lines += texts or ["No validated Qwen sentence was generated for this domain."]
        lines.append("")
    return "\n".join(lines)


def generate_report(packet: Mapping[str, Any], llm: Callable[[str, str, Mapping[str, Any]], Any]) -> dict[str, Any]:
    """Generate one sentence per approved claim, then assemble domains deterministically."""
    approved = _claims(packet)
    if not approved:
        return {"report": render_rule_based_report(dict(packet)), "source": "deterministic_fallback",
                "claim_sources": {}, "validation": {}}
    responses: dict[str, Mapping[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    claim_sources: dict[str, str] = {}
    try:
        for claim_id, claim in approved.items():
            request = build_qwen_request(packet, claim_id)
            raw = llm(SYSTEM_PROMPT, json.dumps(request), CLAIM_JSON_SCHEMA)
            validation = validate_qwen_response(raw, claim, packet)
            validations[claim_id] = validation
            if validation["valid"]:
                responses[claim_id] = validation["response"]
                claim_sources[claim_id] = "validated_qwen"
            else:
                responses[claim_id] = {"text": render_claim_fallback(claim)}
                claim_sources[claim_id] = "deterministic_fallback"
    except Exception as exc:
        validations["__error__"] = {"valid": False, "errors": [f"llm_error:{type(exc).__name__}"]}
        for claim_id, claim in approved.items():
            if claim_id not in responses:
                responses[claim_id] = {"text": render_claim_fallback(claim)}
                claim_sources[claim_id] = "deterministic_fallback"
    source_values = set(claim_sources.values())
    source = next(iter(source_values)) if len(source_values) == 1 else "mixed"
    return {"report": _assemble_qwen_report(packet, responses), "source": source,
            "claim_sources": claim_sources, "validation": validations}
