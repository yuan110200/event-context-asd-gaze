import unittest

from src.asd_gaze.reporting.structured_evidence import FINAL40_METADATA


class Final40ReportingMetadataTests(unittest.TestCase):
    def test_authoritative_count_and_linkage_totals(self):
        self.assertEqual(len(FINAL40_METADATA), 40)
        counts = {link: sum(row["reporting_linkage"] == link for row in FINAL40_METADATA.values()) for link in "FCGSM"}
        self.assertEqual(counts, {"F": 14, "C": 8, "G": 7, "S": 9, "M": 2})

    def test_event_and_cross_event_fields(self):
        for key, row in FINAL40_METADATA.items():
            if key.startswith("Overall."):
                self.assertEqual(row["cross_event"], "true")
                self.assertEqual(row["event_id"], "")
            else:
                self.assertEqual(row["cross_event"], "false")
                self.assertRegex(row["event_id"], r"^E\d{2}$")


if __name__ == "__main__":
    unittest.main()
