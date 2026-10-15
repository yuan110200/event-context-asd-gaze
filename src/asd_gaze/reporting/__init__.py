"""Structured evidence and deterministic reporting helpers."""

from .structured_evidence import (
    DEFAULT_EVIDENCE_SPECS,
    FINAL40_METADATA,
    load_final40_metadata,
    REPORTING_DOMAINS,
    build_claim_packet,
    build_structured_evidence,
    build_td_reference,
    render_rule_based_report,
    render_claim_fallback,
)
from .qwen_generation import (QWEN_CONFIG, SYSTEM_PROMPT, build_qwen_request,
                              generate_report, ollama_generate, validate_qwen_response)

__all__ = [
    "DEFAULT_EVIDENCE_SPECS",
    "FINAL40_METADATA",
    "load_final40_metadata",
    "REPORTING_DOMAINS",
    "build_claim_packet",
    "build_structured_evidence",
    "build_td_reference",
    "render_rule_based_report",
    "render_claim_fallback",
    "QWEN_CONFIG", "SYSTEM_PROMPT", "build_qwen_request", "generate_report", "validate_qwen_response",
    "ollama_generate",
]
