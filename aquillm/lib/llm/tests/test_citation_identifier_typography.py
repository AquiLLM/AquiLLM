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


def test_section_reference_must_be_in_a_cited_passage_of_the_same_paper():
    a, b, c = "[doc:report chunk:1]", "[doc:report chunk:2]", "[doc:other chunk:3]"
    rows = [
        dict(doc_id="report", chunk_id=1, text="The control values are shown in Section 2.11."),
        dict(doc_id="report", chunk_id=2, text="Calibration follows Section\n2.1 of the protocol."),
        dict(doc_id="other", chunk_id=3, text="Section 2.1 describes the sample."),
    ]
    sources = bind_citation_sources(rows, {a, b, c})
    assert misplaced_identifiers(f"Calibration follows §2.1 {a}.", sources) == {"section 2.1": (b,)}
    assert not misplaced_identifiers(f"Calibration follows Section 2.1 {b}.", sources)
    assert not misplaced_identifiers(f"The reading was 2.1 {a}.", sources)
    assert not misplaced_identifiers(f"Calibration follows Section 2.1 {a}.", (sources[0], sources[2]))


def test_ambiguous_or_already_present_section_locators_do_not_force_repair():
    refs = tuple(f"[doc:report chunk:{n}]" for n in range(1, 4))
    def check(draft, *texts):
        rows = [dict(doc_id="report", chunk_id=i, text=text) for i, text in enumerate(texts, 1)]
        return misplaced_identifiers(f"{draft} {refs[0]}.", bind_citation_sources(rows, set(refs)))
    assert not check("See Section 2.1", "2.1 Calibration\nMeasurements follow.", "See Section 2.1.")
    assert not check("See SECTION 2.1", "See Sec. 2.1.", "See SECTION 2.1.")
    assert not check("A cross section 2.1 times larger", "The effective area was larger.", "See Section 2.1.")
    assert not check("See §2.1(c)", "The details follow.", "See Section 2.1(b).")
    assert not check("See §2.1", "The details follow.", "See Section 2.1.", "We cite Section 2.1 too.")
