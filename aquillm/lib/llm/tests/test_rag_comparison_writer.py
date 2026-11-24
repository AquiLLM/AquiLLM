"""A model must select the premises for each comparison statement it writes."""
from dataclasses import replace
import json

import pytest

from lib.llm.answer_publication import FinalCitationPolicy, validate_final_answer
from lib.llm.providers.rag_citation_evidence import bind_citation_sources
from lib.llm.providers.rag_comparison_writer import ComparisonAnswerPlan, decode, response_format


A = "[doc:field-a chunk:10]"
A2 = "[doc:field-a chunk:12]"
B = "[doc:field-b chunk:19]"
C = "[doc:catalog chunk:27]"
PLAN = ComparisonAnswerPlan(
    "Compare the field inventories.",
    (("A", (A, A2)), ("B", (B,)), ("C", (C,)), ("S1", (A,)), ("S4", (C,))),
    ("BRIN coastal series", "BRIN K8"), ("West survey plates",),
)
ROWS = [
    ("field-a", 10, "The BRIN coastal series and West survey plates are listed."),
    ("field-a", 12, "Original guide: p < 0.05 for [O III]; see [12]."),
    ("field-b", 19, "The BRIN coastal series, West survey plates and upper bay maps are listed."),
    ("catalog", 27, "The register records the BRIN coastal series as BRIN K8."),
]
REFS = frozenset((A, A2, B, C))
POLICY = FinalCitationPolicy(REFS, True, "valid-evidence", "valid-scope", bind_citation_sources(
    [dict(doc_id=d, chunk_id=i, text=t) for d, i, t in ROWS], REFS))


def record(text="The later inventory adds upper bay maps.", keys=("A", "B"), kind="added"):
    return dict(relation=kind, statement=text, sources=list(keys))


def data(*claims):
    return json.dumps({"claims": list(claims or (record(),))})


def test_model_written_text_and_only_its_explicitly_selected_citations_are_published():
    answer = decode(data(record(), record("The register calls the BRIN coastal series BRIN K8.", ("A","B","C"), "designation")), PLAN, POLICY)
    assert answer == (
        f"The later inventory adds upper bay maps {A}{A2}{B}.\n\n"
        f"The register calls the BRIN coastal series BRIN K8 {A}{A2}{B}{C}."
    )
    assert validate_final_answer(answer, POLICY).valid
    single = decode(data(record("The register uses the name BRIN K8.", ("S4",), "other")), PLAN, POLICY)
    assert single == f"The register uses the name BRIN K8 {C}."


def test_schema_contains_only_reader_claims_and_real_local_pointers():
    schema = response_format(PLAN)["json_schema"]
    properties = schema["schema"]["properties"]
    assert schema["strict"] and set(schema["schema"]["required"]) == {"claims"}
    allowed = properties["claims"]["items"]["properties"]["sources"]["items"]["enum"]
    assert set(allowed) == {"A", "B", "C", "S1", "S4"}
    assert properties["claims"]["maxItems"] == 5


def test_equations_and_literal_brackets_survive_the_grounding_contract():
    assertion = "For [O III], the authors report p < 0.05 in [12]."
    assert decode(data(record(assertion, ("A",), "other")), PLAN, POLICY) == assertion[:-1]+f" {A}{A2}."


@pytest.mark.parametrize("raw", [
    "[]", '{"claims": []}', data()+" extra", data().replace('"claims":','"claims": [], "claims":',1),
    data(record(keys=("A","A"))), data(record(keys=("S99",))), data(record(keys=())),
    data(record("The new list cites [Doc:catalog chunk:27].")), data(record("The model said <img src=x>.")),
    data(record("The model said <script.")), data(record("![map](https://example.invalid/a)")),
    data(record("A claim about the register. A second, uncited claim.")), data(record("A source.\nAnother claim.")),
    data(record("The tool lists West survey plates as removed.",kind="removed")),
    data(*[record() for _ in range(6)]), data(record(kind="invented")),
    data(record("A separate register identifies the BRIN K8.", ("A", "C"), "designation")),
    json.dumps({"claims":[record()],"naming_decision":{}}),
])
def test_malformed_unsupported_or_unbound_records_do_not_acquire_citations(raw):
    assert decode(raw, PLAN, POLICY) is None


def test_stale_policy_unknown_pointer_or_different_source_owner_cannot_bind():
    missing = replace(POLICY, allowed_citations=REFS-{C})
    assert decode(data(), PLAN, missing) is None
    assert decode(data(), PLAN, replace(POLICY, sources=POLICY.sources[:2])) is None
    for groups in [
        (("A",(A,)), ("B",(A2,)), ("C",(C,))),
        (("A",(A,)), ("B",(B,)), ("C",(C,)), ("A",(A2,))),
        (("A",(A,)), ("B",(B,)), ("C",(C,)), ("S99",(B,C))),
    ]:
        with pytest.raises(ValueError):
            replace(PLAN,groups=groups)



def test_optional_scope_anchors_must_match_the_frozen_exact_source_offsets():
    texts = (ROWS[0][-1], ROWS[2][-1], ROWS[3][-1])
    anchors = tuple((ref, 0, len(t), t) for ref,t in zip((A,B,C),texts,strict=True))
    plan = replace(PLAN, references=anchors)
    assert decode(data(), plan, POLICY)
    changed = replace(plan, references=(*anchors[:2], (C, 0, len(texts[2]), "x"*len(texts[2]))))
    assert decode(data(), changed, POLICY) is None
    with pytest.raises(ValueError):
        replace(plan, references=(anchors[2], anchors[1], anchors[0]))


def test_preflight_rejects_a_stale_source_before_dispatch():
    from lib.llm.providers.rag_comparison_writer import plan_bound
    assert plan_bound(PLAN, POLICY)
    assert not plan_bound(PLAN, replace(POLICY, sources=POLICY.sources[1:]))
