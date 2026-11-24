"""Changes inferred by comparing inventories need model-selected evidence for both lists."""

import json
from dataclasses import replace

import pytest

from lib.llm.providers.rag_comparison_support import (
    apply_paired_choices, paired_schema, prepare_paired_claims,
)
from lib.llm.providers.rag_comparison_writer import decode, response_format
from lib.llm.tests.test_rag_comparison_writer import A, B, C, PLAN, POLICY


def claim(statement, sources, quote="", relation="added"):
    return dict(statement=statement, sources=sources, reporting_quote=quote, relation=relation)


def raw(*claims, naming=None):
    decision = {"decision":"insufficient_or_immaterial", "statement":"", "sources":[]}
    return json.dumps({"claims":list(claims), "naming_decision":decision if naming is None else naming})


def with_body(ref, body):
    return replace(POLICY, sources=tuple(replace(s,text=body) if s.citation==ref else s for s in POLICY.sources))


ADDED = "The inventory added oak strain gauges."
OTHER = "The register documents the coastal survey."


def test_schema_advertises_only_real_keys_and_the_optional_aggregate():
    form = response_format(PLAN)
    assert paired_schema(form) is form
    paired_schema(form)
    each = form["json_schema"]["schema"]["properties"]["claims"]["items"]
    assert each["required"].count("reporting_quote")==1
    assert each["properties"]["sources"]["items"]["enum"].count("AB")==1
    assert "S99" not in each["properties"]["sources"]["items"]["enum"]


def test_explicit_single_source_and_list_inference_are_distinct():
    quote="The inventory added oak strain gauges."
    obj,missing=prepare_paired_claims(raw(claim(ADDED,["B"],quote)),PLAN,with_body(B,quote))
    assert not missing and obj["claims"][0]["sources"]==["B"]
    obj,missing=prepare_paired_claims(raw(claim(ADDED,["B"],"The inventory lists oak strain gauges.")),PLAN,
                                    with_body(B,"The inventory lists oak strain gauges."))
    assert missing==(0,)
    for different in ("The inventory never added oak strain gauges.", "The inventory added strain gauges made of birch.",
                      "The inventory added strain gauges beside oak trees."):
        _,missing=prepare_paired_claims(raw(claim(ADDED,["B"],different)),PLAN,with_body(B,different))
        assert missing==(0,)
    _,missing=prepare_paired_claims(raw(claim(ADDED,["B"],quote)),PLAN,with_body(A,quote))
    assert missing==(0,)


def test_single_quote_may_not_drop_negation_from_the_same_clause():
    full="The report denies that the inventory added oak strain gauges."
    _,missing=prepare_paired_claims(raw(claim(ADDED,["B"],"the inventory added oak strain gauges")),PLAN,with_body(B,full))
    assert missing==(0,)


def test_multiple_real_pointers_and_explicit_alias_are_equivalent():
    designation={"decision":"explain","statement":"The two registers report the same probe.","sources":["AB","C"]}
    for selection in (["AB"], ["A","B"]):
        obj,needs=prepare_paired_claims(raw(claim(ADDED,selection),naming=designation),PLAN,POLICY)
        assert not needs and obj["claims"][0]["sources"]==["A","B"]
        assert obj["naming_decision"]["sources"]==["A","B","C"]
        assert 'reporting_quote' not in obj["claims"][0]


def test_only_the_model_can_select_additional_citations_for_frozen_claims():
    r=raw(claim(ADDED,["B"]),claim(OTHER,["C"],relation="qualification"))
    obj,needs=prepare_paired_claims(r,PLAN,POLICY)
    assert needs==(0,)
    result=apply_paired_choices(obj,needs,'{"choices":[{"i":0,"source":"AB"}]}')
    assert result["claims"][0]==dict(statement=ADDED,relation="added",sources=["A","B"])
    assert result["claims"][1]==obj["claims"][1] and obj["claims"][0]["sources"]==["B"]
    for reply in (None,'{"choices":[{"i":0,"source":"omit"}]}','{"choices":[{"i":1,"source":"AB"}]}',
                  '{"choices":[{"i":true,"source":"AB"}]}','{"choices":[{"i":0,"source":"AB"},{"i":0,"source":"AB"}]}',
                  '{"choices":[{"i":0,"source":"AB","statement":"different"}]}'):
        assert apply_paired_choices(obj,needs,reply)["claims"]==[obj["claims"][1]]


def test_scope_and_unknown_pointers_are_checked_before_rereading_source_claims():
    assert prepare_paired_claims(raw(claim(ADDED,["AB"])),PLAN,replace(POLICY,sources=POLICY.sources[1:])) is None
    for malformed in (raw(claim(ADDED,["S99"])), raw(claim(ADDED,["A","A"])),raw(claim(ADDED,["AB"],quote=17)),
                      raw(claim(ADDED,["AB"])).replace('"claims":','"claims":[],"claims":'),"[]"):
        assert prepare_paired_claims(malformed,PLAN,POLICY) is None


@pytest.mark.parametrize(("role", "statement", "expected"), [
    ("other", "The later inventory added upper bay maps.", "added"),
    ("qualification", "West survey plates were removed from the revised register.", "removed"),
    ("retained", "The new catalog replaced upper bay maps with field sketches.", "updated"),
    ("other", "The revised register retained West survey plates.", "retained"),
])
def test_explicit_change_prose_needs_evidence_even_if_the_model_miscategorizes_it(role, statement, expected):
    obj, needs = prepare_paired_claims(raw(claim(statement, ["B"], relation=role)), PLAN, POLICY)
    assert needs == (0,)
    assert obj["claims"][0]["relation"] == expected


@pytest.mark.parametrize("statement", [
    "The later inventory added West survey plates.",
    "West survey plates were removed from the revised register.",
])
def test_shared_names_cannot_be_reported_as_changes_by_using_another_relation(statement):
    obj, needs = prepare_paired_claims(raw(claim(statement, ["AB"], relation="other")), PLAN, POLICY)
    assert not needs
    assert decode(json.dumps({"claims": obj["claims"]}), PLAN, POLICY) is None


@pytest.mark.parametrize("statement", [
    "West survey plates were not removed from the later inventory.",
    "The later inventory never added West survey plates.",
    "The guide discusses why removed plates are harder to trace.",
    "The report found that added weight affected the gauges.",
    "The inventory added together the two measurements.",
    "It is unclear whether the later inventory added the plates.",
    "The later inventory added weight to the hypothesis.",
])
def test_qualifications_and_incidental_uses_keep_their_original_citations(statement):
    obj, needs = prepare_paired_claims(raw(claim(statement, ["B"], relation="qualification")), PLAN, POLICY)
    assert not needs and obj["claims"][0]["relation"] == "qualification"


def test_explicit_single_source_report_still_supports_a_miscategorized_change():
    statement = "The revised inventory added upper bay maps."
    obj, needs = prepare_paired_claims(raw(claim(statement, ["B"], statement, "other")),
                                       PLAN, with_body(B, statement))
    assert not needs and obj["claims"][0]["relation"] == "added"


def test_descriptive_passive_relation_and_original_schema_still_apply():
    statement = "The BRIN coastal series was updated during laboratory calibration."
    obj, needs = prepare_paired_claims(raw(claim(statement, ["B"], relation="qualification")), PLAN, POLICY)
    assert not needs and obj["claims"][0]["relation"] == "qualification"
    for role in ("invented", "designation"):
        assert prepare_paired_claims(raw(claim("The later inventory added West survey plates.", ["AB", "C"],
                                               relation=role)), PLAN, POLICY) is None
