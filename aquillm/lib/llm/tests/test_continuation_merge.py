"""Unit tests for cutoff continuation prefix trimming and seam repair."""
from __future__ import annotations

import unittest

from lib.llm.providers.complete_turn import (
    _repair_continuation_seam,
    _trim_duplicate_continuation_prefix,
)


class ContinuationMergeTests(unittest.TestCase):
    def test_suffix_overlap_trims_short_token_duplicates(self):
        partial = "Area Under the Receiver Operating Characteristic Curve (AUROC"
        continuation = "(AUROC) scores typically range between 0.70 and 0.85."
        trimmed = _trim_duplicate_continuation_prefix(partial, continuation)
        self.assertEqual(trimmed, ") scores typically range between 0.70 and 0.85.")

    def test_percent_duplication_repaired_at_seam(self):
        partial = "As you lower the acceptable error rate (e.g., from 25%"
        continuation = "25% to 5%), utility drops."
        trimmed = _trim_duplicate_continuation_prefix(partial, continuation)
        merged = f"{partial.rstrip()}{trimmed}"
        repaired = _repair_continuation_seam(partial, merged)
        self.assertIn("from 25% to 5%", repaired)
        self.assertNotIn("25%25%", repaired)

    def test_restart_prefix_is_stripped_when_continuation_repeats_partial(self):
        partial = (
            "# The GalaxiesML Dataset: Comprehensive Technical Analysis\n\n"
            "Dataset Construction and Provenance\n"
            "The paper establishes **LS"
        )
        continuation = (
            "# The GalaxiesML Dataset: Comprehensive Technical Analysis\n\n"
            "Dataset Construction and Provenance\n"
            "The paper establishes **LSST deployment thresholds."
        )
        trimmed = _trim_duplicate_continuation_prefix(partial, continuation)
        self.assertEqual(trimmed, "ST deployment thresholds.")

    def test_duplicate_only_continuation_returns_empty(self):
        partial = (
            "Comprehensive rollout plan:\n"
            "- Validate ingestion pipeline behavior across three cohorts.\n"
            "Next, we should ev"
        )
        continuation = (
            "Comprehensive rollout plan:\n"
            "- Validate ingestion pipeline behavior across three cohorts.\n"
        )
        trimmed = _trim_duplicate_continuation_prefix(partial, continuation)
        self.assertEqual(trimmed, "")


if __name__ == "__main__":
    unittest.main()
