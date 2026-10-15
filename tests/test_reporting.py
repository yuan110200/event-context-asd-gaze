import unittest
import json
from unittest.mock import patch

import pandas as pd

from src.asd_gaze.reporting import (
    QWEN_CONFIG,
    build_qwen_request,
    REPORTING_DOMAINS,
    build_claim_packet,
    build_structured_evidence,
    build_td_reference,
    render_rule_based_report,
    SYSTEM_PROMPT,
    ollama_generate,
    generate_report,
    validate_qwen_response,
)


class ReportingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.feature_keys = [
            "E01.Event_Specific.Face_Preference_Index",
            "E01.Event_Specific.Eye_Contact_Maintenance_sec",
            "E02.Event_Specific.Action_Completion_Anytime",
            "E10.Event_Specific.Action_Completion_Anytime",
            "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency",
            "E03.Dimension_Indicators.Attention_Shifting.Disengagement_Latency_sec",
        ]
        self.td_train = pd.DataFrame(
            {
                "E01.Event_Specific.Face_Preference_Index": [0.55, 0.60, 0.65, 0.70],
                "E01.Event_Specific.Eye_Contact_Maintenance_sec": [1.0, 1.1, 1.2, 1.3],
                "E02.Event_Specific.Action_Completion_Anytime": [1, 1, 1, 1],
                "E10.Event_Specific.Action_Completion_Anytime": [1, 1, 1, 1],
                "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency": [0.5, 0.6, 0.7, 0.8],
                "E03.Dimension_Indicators.Attention_Shifting.Disengagement_Latency_sec": [0.2, 0.3, 0.4, 0.5],
            }
        )

    def test_td_reference_uses_only_supplied_partition(self):
        td_with_forbidden_validation = pd.concat(
            [
                self.td_train,
                pd.DataFrame({"Overall.Basic_Metrics.Avg_Cue_to_Target_Latency": [99.0]}),
            ],
            ignore_index=True,
        )
        train_ref = build_td_reference(self.td_train, self.feature_keys)
        leaked_ref = build_td_reference(td_with_forbidden_validation, self.feature_keys)

        self.assertAlmostEqual(train_ref["Overall.Basic_Metrics.Avg_Cue_to_Target_Latency"]["median"], 0.65)
        self.assertNotEqual(
            train_ref["Overall.Basic_Metrics.Avg_Cue_to_Target_Latency"]["median"],
            leaked_ref["Overall.Basic_Metrics.Avg_Cue_to_Target_Latency"]["median"],
        )

    def test_claims_reference_existing_evidence(self):
        ref = build_td_reference(self.td_train, self.feature_keys)
        subject = {
            "E01.Event_Specific.Face_Preference_Index": 0.2,
            "E01.Event_Specific.Eye_Contact_Maintenance_sec": 2.0,
            "E02.Event_Specific.Action_Completion_Anytime": 0,
            "E10.Event_Specific.Action_Completion_Anytime": 1,
            "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency": 1.4,
            "E03.Dimension_Indicators.Attention_Shifting.Disengagement_Latency_sec": 0.9,
        }
        evidence = build_structured_evidence("SYN001", subject, ref)
        packet = build_claim_packet("SYN001", predicted_label=1, calibrated_probability=0.73, evidence=evidence, allow_unclassified=True)
        evidence_ids = {item["evidence_id"] for item in evidence}

        for claim in packet["claims"]:
            self.assertTrue(set(claim["evidence_ids"]).issubset(evidence_ids))
        self.assertEqual(packet["prediction"], {"predicted_label": 1, "calibrated_probability": 0.73})

    def test_three_reporting_domains_are_present(self):
        self.assertEqual(
            set(REPORTING_DOMAINS.values()),
            {
                "Face and social gaze allocation",
                "Cue-guided orienting and target response",
                "Gaze shifting and reallocation",
            },
        )

    def test_missing_features_do_not_create_unsupported_claims(self):
        ref = build_td_reference(self.td_train, self.feature_keys)
        evidence = build_structured_evidence(
            "SYN002",
            {"E01.Event_Specific.Face_Preference_Index": None, "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency": 2.0},
            ref,
        )
        packet = build_claim_packet("SYN002", predicted_label=0, calibrated_probability=0.12, evidence=evidence, allow_unclassified=True)
        domain_ids = {claim["dimension_id"] for claim in packet["claims"]}

        self.assertNotIn("face_and_social_gaze", domain_ids)
        self.assertIn("cue_guided_target_response", domain_ids)

    def test_rule_report_keeps_prediction_output_fixed(self):
        ref = build_td_reference(self.td_train, self.feature_keys)
        evidence = build_structured_evidence(
            "SYN003",
            {
                "E01.Event_Specific.Face_Preference_Index": 0.62,
                "E01.Event_Specific.Eye_Contact_Maintenance_sec": 1.15,
                "E02.Event_Specific.Action_Completion_Anytime": 1,
                "E10.Event_Specific.Action_Completion_Anytime": 1,
                "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency": 0.65,
                "E03.Dimension_Indicators.Attention_Shifting.Disengagement_Latency_sec": 0.35,
            },
            ref,
        )
        packet = build_claim_packet("SYN003", predicted_label=0, calibrated_probability=0.23456, evidence=evidence, allow_unclassified=True)
        report = render_rule_based_report(packet)

        self.assertIn("Predicted class: TD", report)
        self.assertIn("Calibrated ASD probability: 0.235", report)
        self.assertEqual(packet["prediction"]["predicted_label"], 0)
        self.assertEqual(packet["prediction"]["calibrated_probability"], 0.23456)

    def _qwen_packet(self):
        ref = build_td_reference(self.td_train, self.feature_keys)
        evidence = build_structured_evidence(
            "SYN004", {"E01.Event_Specific.Face_Preference_Index": 0.2,
                        "Overall.Basic_Metrics.Avg_Cue_to_Target_Latency": 1.4}, ref)
        return build_claim_packet("SYN004", 1, 0.73, evidence, allow_unclassified=True)

    def test_qwen_config_and_valid_json(self):
        packet = self._qwen_packet()
        claim = next(c for c in packet["claims"] if c["claim_type"] != "classification_result")
        response = {"claim_id": claim["claim_id"], "event_id": claim["event_ids"][0],
                    "text": "Face-directed gaze was less frequent than the reference."}
        result = validate_qwen_response(response, claim, packet)
        self.assertTrue(result["valid"])
        self.assertEqual(QWEN_CONFIG["model"], "qwen2.5:32b")
        self.assertEqual(QWEN_CONFIG["max_output_tokens"], 220)
        self.assertEqual(build_qwen_request(packet, claim["claim_id"])["approved_claim"]["claim_id"], claim["claim_id"])

    def test_qwen_rejects_unknown_id_event_and_forbidden_sentence(self):
        packet = self._qwen_packet()
        claim = next(c for c in packet["claims"] if c["claim_type"] != "classification_result")
        response = {"claim_id": "UNKNOWN", "event_id": "E99", "text": "This indicates a deficit."}
        result = validate_qwen_response(response, claim, packet)
        self.assertFalse(result["valid"])
        self.assertIn("claim_id_mismatch", result["errors"])
        self.assertIn("event_id_mismatch", result["errors"])
        self.assertIn("forbidden_language", result["errors"])

    def test_qwen_fallback_and_prediction_assembly(self):
        packet = self._qwen_packet()
        result = generate_report(packet, lambda *_: "not json")
        self.assertEqual(result["source"], "deterministic_fallback")
        self.assertIn("Predicted label: 1", result["report"])
        self.assertTrue(all(source == "deterministic_fallback" for source in result["claim_sources"].values()))
        result = generate_report(packet, lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
        self.assertEqual(result["source"], "deterministic_fallback")

    def test_qwen_validated_report_keeps_classifier_output(self):
        packet = self._qwen_packet()
        def mock_llm(_system, prompt, _schema):
            request = json.loads(prompt)
            claim = request["approved_claim"]
            relation = claim["approved_findings"][0]["relation"]
            text = ("Face-directed gaze was less frequent than the reference."
                    if relation == "less" else
                    "First qualifying gaze timing was longer than the reference median."
                    if relation == "longer than" else
                    "Event-defined target-region response was consistent with the reference.")
            return {"claim_id": claim["claim_id"], "event_id": claim["event_ids"][0] if claim["event_ids"] else None,
                    "text": text}
        result = generate_report(packet, mock_llm)
        self.assertEqual(result["source"], "validated_qwen")
        self.assertIn("Predicted label: 1", result["report"])
        self.assertIn("Calibrated probability: 0.730", result["report"])

    def test_qwen_rejects_direction_mismatch(self):
        packet = self._qwen_packet()
        claim = next(c for c in packet["claims"] if c["claim_type"] != "classification_result")
        response = {"claim_id": claim["claim_id"], "event_id": claim["event_ids"][0],
                    "text": "Face-directed gaze was more frequent than the reference."}
        result = validate_qwen_response(response, claim, packet)
        self.assertFalse(result["valid"])
        self.assertIn("direction_mismatch", result["errors"])

    def test_qwen_partial_failure_uses_per_claim_fallback(self):
        packet = self._qwen_packet()
        calls = []
        def mock_llm(_system, prompt, _schema):
            request = json.loads(prompt)
            claim = request["approved_claim"]
            calls.append(claim["claim_id"])
            if len(calls) == 1:
                relation = claim["approved_findings"][0]["relation"]
                text = "Face-directed gaze was less frequent than the reference." if relation == "less" else "First qualifying gaze timing was longer than the reference median."
                return {"claim_id": claim["claim_id"], "event_id": claim["event_ids"][0] if claim["event_ids"] else None, "text": text}
            return None
        result = generate_report(packet, mock_llm)
        self.assertEqual(result["source"], "mixed")
        self.assertEqual(set(result["claim_sources"].values()), {"validated_qwen", "deterministic_fallback"})

    def test_empty_claims_are_deterministic(self):
        packet = {"subject_id": "EMPTY", "prediction": {"predicted_label": 0, "calibrated_probability": 0.1}, "claims": []}
        result = generate_report(packet, lambda *_: self.fail("LLM must not be called"))
        self.assertEqual(result["source"], "deterministic_fallback")
        self.assertEqual(result["claim_sources"], {})

    def test_qwen_rejects_unapproved_additional_field_contract(self):
        packet = self._qwen_packet()
        claim = next(c for c in packet["claims"] if c["claim_type"] != "classification_result")
        response = {"claim_id": claim["claim_id"], "event_id": None,
                    "text": "Face-directed gaze was less frequent than the reference.",
                    "additional_observation": "unsupported"}
        result = validate_qwen_response(response, claim, packet)
        self.assertFalse(result["valid"])
        self.assertIn("schema_keys", result["errors"])

    def test_ollama_http_payload_and_mock_response(self):
        body = {"message": {"content": '{"ok": true}'}}

        class MockResponse:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return json.dumps(body).encode("utf-8")

        with patch("src.asd_gaze.reporting.qwen_generation.urllib.request.urlopen", return_value=MockResponse()) as call:
            result = ollama_generate(SYSTEM_PROMPT, "{}", {}, base_url="http://mock", timeout=3)
        self.assertEqual(result, {"ok": True})
        request = call.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen2.5:32b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"], {"temperature": 0.2, "top_p": 0.9,
                                                "num_ctx": 8192, "num_predict": 220})
        self.assertEqual(call.call_args.kwargs["timeout"], 3)

    def test_ollama_failures_return_fallback_input(self):
        with patch("src.asd_gaze.reporting.qwen_generation.urllib.request.urlopen", side_effect=TimeoutError()):
            self.assertIsNone(ollama_generate("system", "user", {}, base_url="http://mock"))
        packet = self._qwen_packet()
        result = generate_report(packet, lambda *_: ollama_generate("system", "user", {}, base_url="http://mock"))
        self.assertEqual(result["source"], "deterministic_fallback")


if __name__ == "__main__":
    unittest.main()
