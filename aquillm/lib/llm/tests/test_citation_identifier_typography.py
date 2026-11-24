from lib.llm.providers.rag_citation_evidence import bind_citation_sources, misplaced_identifiers


def bound(*texts):
    refs = [f"[doc:d{index} chunk:{index}]" for index in range(1, len(texts) + 1)]
    rows = [dict(doc_id=f"d{i}", chunk_id=i, text=value, citation=refs[i-1])
            for i, value in enumerate(texts, 1)]
    return refs, bind_citation_sources(rows, set(refs))


def test_hyphenated_family_and_named_release_are_already_in_the_cited_source():
    (current, other), sources = bound("The ATLAS-PX3 calibration used a quartz lamp.",
                                     "ATLAS PX3 was also used for astrometry.")
    assert not misplaced_identifiers(f"ATLAS PX3 used a quartz lamp {current}.", sources)
    assert not misplaced_identifiers(f"ATLAS-PX3 was also used for astrometry {other}.", sources)


def test_typographic_variant_prevents_a_false_unique_cross_paper_redirect():
    (report, _, _), sources = bound("The observatory release used a quartz lamp.",
                                    "ATLAS-PX3 data also fed the new index.",
                                    "ATLAS PX3 appears in the appendix.")
    assert not misplaced_identifiers(f"ATLAS PX3 used a quartz lamp {report}.", sources)


def test_citation_check_still_distinguishes_actual_versions_and_families():
    (old, correct), sources = bound("ATLAS-PX2 and AURORA-PX3 were calibrated.",
                                    "ATLAS PX3 has a different solution.")
    assert misplaced_identifiers(f"ATLAS-PX3 has a different solution {old}.", sources) == {
        "ATLAS PX3": (correct,)
    }
