"""Collection-backed Markdown prompt skill loading."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import connection
from django.test import override_settings
from django.test.utils import CaptureQueriesContext

from apps.chat.consumers.chat import CollectionsRef
from apps.chat.services.collection_prompt_skills import (
    load_collection_prompt_skills,
    validate_skill_overrides,
)
from apps.chat.services.skills_runtime import (
    effective_base_system_for_memory,
    effective_base_system_for_memory_async,
)
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TeXDocument
from aquillm.models import WSConversation

User = get_user_model()


def _raw_text_doc(
    collection: Collection, user, *, title: str, text: str
) -> RawTextDocument:
    doc = RawTextDocument(
        title=title,
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=user,
    )
    doc.save(dont_rechunk=True)
    return doc


def _consumer_for(user, db_convo: WSConversation, collection_ids: list[int]):
    return SimpleNamespace(
        user=user,
        db_convo=db_convo,
        col_ref=CollectionsRef(collection_ids),
    )


@pytest.mark.django_db
@override_settings(
    SKILLS_ENABLED=True,
    AQUILLM_SKILLS_EXTRA_MODULES=[],
    AQUILLM_SKILLS_MARKDOWN_DIR="",
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True,
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS=12000,
)
def test_effective_system_includes_marked_markdown_skill_from_selected_collection():
    user = User.objects.create_user(username="skill-user", password="pass")
    collection = Collection.objects.create(name="Astro Project")
    CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
    db_convo = WSConversation.objects.create(owner=user, system_prompt="Base system.")
    _raw_text_doc(
        collection,
        user,
        title="astro-python-scripts_skill.md",
        text=(
            "---\n"
            "name: astro-python-scripts\n"
            "description: >\n"
            "  Generate astrophysics Python scripts.\n"
            "  Prefer this skill for FITS and spectra work.\n"
            "---\n\n"
            "# Astro Python Scripts\n\nUse astropy units and never overwrite data."
        ),
    )

    system = effective_base_system_for_memory(
        _consumer_for(user, db_convo, [collection.id])
    )

    assert "## Collection Skill: astro-python-scripts" in system
    assert (
        "Description:\nGenerate astrophysics Python scripts. "
        "Prefer this skill for FITS and spectra work."
        in system
    )
    assert "Use astropy units and never overwrite data." in system


@pytest.mark.django_db
@override_settings(
    SKILLS_ENABLED=True,
    AQUILLM_SKILLS_EXTRA_MODULES=[],
    AQUILLM_SKILLS_MARKDOWN_DIR="",
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True,
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS=12000,
)
def test_effective_system_includes_markdown_docs_from_skill_pack_subcollection():
    user = User.objects.create_user(username="pack-user", password="pass")
    root = Collection.objects.create(name="Spectra Project")
    pack = Collection.objects.create(name="skill_pack", parent=root)
    CollectionPermission.objects.create(user=user, collection=root, permission="VIEW")
    db_convo = WSConversation.objects.create(owner=user, system_prompt="Base system.")
    _raw_text_doc(
        pack,
        user,
        title="spectra-style.md",
        text=(
            "---\n"
            "name: spectra-style\n"
            "---\n\n"
            "Always ask about wavelength units when writing spectra code."
        ),
    )

    system = effective_base_system_for_memory(_consumer_for(user, db_convo, [root.id]))

    assert "## Collection Skill: spectra-style" in system
    assert "Always ask about wavelength units" in system


@pytest.mark.django_db
@override_settings(
    SKILLS_ENABLED=True,
    AQUILLM_SKILLS_EXTRA_MODULES=[],
    AQUILLM_SKILLS_MARKDOWN_DIR="",
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True,
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS=12000,
)
def test_effective_system_ignores_unmarked_markdown_docs_in_regular_collection():
    user = User.objects.create_user(username="ignore-user", password="pass")
    collection = Collection.objects.create(name="Regular Notes")
    CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
    db_convo = WSConversation.objects.create(owner=user, system_prompt="Base system.")
    _raw_text_doc(
        collection,
        user,
        title="ordinary-notes.md",
        text="Do not treat this research note as system instructions.",
    )

    system = effective_base_system_for_memory(
        _consumer_for(user, db_convo, [collection.id])
    )

    assert "Do not treat this research note as system instructions." not in system


@pytest.mark.django_db
@override_settings(
    SKILLS_ENABLED=True,
    AQUILLM_SKILLS_EXTRA_MODULES=[],
    AQUILLM_SKILLS_MARKDOWN_DIR="",
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True,
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS=12000,
)
def test_collection_skill_loading_does_not_materialize_every_collection_document():
    user = User.objects.create_user(username="targeted-skill-user", password="pass")
    collection = Collection.objects.create(name="Large Research Collection")
    CollectionPermission.objects.create(
        user=user,
        collection=collection,
        permission="VIEW",
    )
    db_convo = WSConversation.objects.create(owner=user, system_prompt="Base system.")
    _raw_text_doc(
        collection,
        user,
        title="analysis_skill.md",
        text="Use the collection-specific analysis conventions.",
    )
    _raw_text_doc(
        collection,
        user,
        title="ordinary-research-note.md",
        text="This is evidence, not a prompt skill.",
    )

    with patch.object(
        Collection,
        "documents",
        new_callable=PropertyMock,
        side_effect=AssertionError("full collection materialization is too expensive"),
    ):
        system = effective_base_system_for_memory(
            _consumer_for(user, db_convo, [collection.id])
        )

    assert "collection-specific analysis conventions" in system
    assert "This is evidence" not in system


# Channels closes old connections; exercise that outside pytest's atomic wrapper.
@pytest.mark.django_db(transaction=True)
@override_settings(
    SKILLS_ENABLED=True,
    AQUILLM_SKILLS_EXTRA_MODULES=[],
    AQUILLM_SKILLS_MARKDOWN_DIR="",
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True,
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS=12000,
)
def test_effective_system_async_wrapper_loads_collection_skills():
    user = User.objects.create_user(username="async-skill-user", password="pass")
    collection = Collection.objects.create(name="Async Skills")
    CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
    db_convo = WSConversation.objects.create(owner=user, system_prompt="Base system.")
    _raw_text_doc(
        collection,
        user,
        title="skills.md",
        text=(
            "---\n"
            "name: async-safe-skill\n"
            "---\n\n"
            "This collection skill is safe to load from async consumers."
        ),
    )

    system = async_to_sync(effective_base_system_for_memory_async)(
        _consumer_for(user, db_convo, [collection.id])
    )

    assert "## Collection Skill: async-safe-skill" in system
    assert "safe to load from async consumers" in system


@pytest.mark.django_db
@override_settings(AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_parent_and_selected_pack_include_each_skill_once_with_overrides():
    user = User.objects.create_user(username="skill-overrides", password="pass")
    root = Collection.objects.create(name="Observatory")
    pack = Collection.objects.create(name="skill_pack", parent=root)
    CollectionPermission.objects.create(user=user, collection=root, permission="VIEW")
    inherited = _raw_text_doc(
        pack, user, title="inherited.md", text="Unique instruction body"
    )
    disabled = _raw_text_doc(
        pack, user, title="disabled.md", text="Disabled instruction body"
    )
    independent = _raw_text_doc(
        pack, user, title="independent.md", text="Explicit independent skill"
    )

    def skill_id(doc):
        return f"{doc._meta.label_lower}:{doc.pk}"

    prompt = load_collection_prompt_skills(
        user,
        [root.id, pack.id],
        {skill_id(disabled): False},
    )
    assert prompt.count("Unique instruction body") == 1
    assert "Disabled instruction body" not in prompt

    prompt = load_collection_prompt_skills(
        user,
        [],
        {skill_id(independent): True},
    )
    assert "Explicit independent skill" in prompt
    assert "Unique instruction body" not in prompt

    prompt = load_collection_prompt_skills(
        user, [root.id], {skill_id(inherited): False}
    )
    assert "Unique instruction body" not in prompt


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True, AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_chat_context_catalog_lists_readable_descendants_and_skill_metadata(client):
    user = User.objects.create_user(username="skill-catalog", password="pass")
    root = Collection.objects.create(name="Observatory")
    pack = Collection.objects.create(name="skill_pack", parent=root)
    hidden = Collection.objects.create(name="Private")
    CollectionPermission.objects.create(user=user, collection=root, permission="VIEW")
    doc = _raw_text_doc(
        pack,
        user,
        title="spectra.md",
        text=(
            "---\nname: Spectra Guide\ndescription: Handle spectra.\n"
            "---\n\nUse wavelength units."
        ),
    )
    _raw_text_doc(hidden, user, title="private_skill.md", text="Secret instruction")
    client.force_login(user)

    response = client.get("/api/collections/chat-context/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["skills_enabled"] is True
    assert payload["collections"] == [
        {
            "id": root.id,
            "name": "Observatory",
            "parent": None,
            "path": "Observatory",
            "is_skill_pack": False,
        },
        {
            "id": pack.id,
            "name": "skill_pack",
            "parent": root.id,
            "path": "Observatory/skill_pack",
            "is_skill_pack": True,
        },
    ]
    assert payload["skills"] == [
        {
            "id": f"{doc._meta.label_lower}:{doc.pk}",
            "name": "Spectra Guide",
            "description": "Handle spectra.",
            "instructions": "Use wavelength units.",
            "collection_id": str(pack.id),
            "collection_name": "skill_pack",
            "collection_path": "Observatory/skill_pack",
            "source_path": "Observatory/skill_pack/spectra.md",
            "pack_id": str(pack.id),
            "pack_name": "skill_pack",
            "default_collection_ids": [str(pack.id), str(root.id)],
        }
    ]


@pytest.mark.django_db
@override_settings(
    SKILLS_ENABLED=True, AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=False
)
def test_chat_context_catalog_hides_skills_when_feature_disabled(client):
    user = User.objects.create_user(username="skill-disabled", password="pass")
    root = Collection.objects.create(name="Research")
    CollectionPermission.objects.create(user=user, collection=root, permission="VIEW")
    _raw_text_doc(
        root, user, title="research_skill.md", text="Disabled feature instruction"
    )
    client.force_login(user)

    payload = client.get("/api/collections/chat-context/").json()

    assert payload["skills_enabled"] is False
    assert payload["skills"] == []
    assert payload["collections"][0]["id"] == root.id


@pytest.mark.django_db
@override_settings(AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_explicit_skill_rechecks_access_and_document_existence_at_runtime():
    user = User.objects.create_user(username="revoked-skill", password="pass")
    root = Collection.objects.create(name="Research")
    permission = CollectionPermission.objects.create(
        user=user, collection=root, permission="VIEW"
    )
    doc = _raw_text_doc(
        root, user, title="research_skill.md", text="Restricted instruction"
    )
    override = {f"{doc._meta.label_lower}:{doc.pk}": True}

    assert "Restricted instruction" in load_collection_prompt_skills(user, [], override)
    permission.delete()
    assert "Restricted instruction" not in load_collection_prompt_skills(
        user, [], override
    )
    CollectionPermission.objects.create(user=user, collection=root, permission="VIEW")
    doc.delete()
    assert "Restricted instruction" not in load_collection_prompt_skills(
        user, [], override
    )


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True, AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_catalog_excludes_unreadable_ancestor_from_paths_and_parent_ids(client):
    user = User.objects.create_user(username="child-only-reader", password="pass")
    hidden = Collection.objects.create(name="Private Division")
    visible = Collection.objects.create(name="Shared Project", parent=hidden)
    pack = Collection.objects.create(name="skill_pack", parent=visible)
    CollectionPermission.objects.create(
        user=user, collection=visible, permission="VIEW"
    )
    direct = _raw_text_doc(
        visible, user, title="direct_skill.md", text="Direct instructions"
    )
    packed = _raw_text_doc(pack, user, title="packed.md", text="Packed instructions")
    client.force_login(user)

    response = client.get("/api/collections/chat-context/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["collections"] == [
        {
            "id": visible.id,
            "name": "Shared Project",
            "parent": None,
            "path": "Shared Project",
            "is_skill_pack": False,
        },
        {
            "id": pack.id,
            "name": "skill_pack",
            "parent": visible.id,
            "path": "Shared Project/skill_pack",
            "is_skill_pack": True,
        },
    ]
    skills = {skill["id"]: skill for skill in payload["skills"]}
    assert (
        skills[f"{direct._meta.label_lower}:{direct.pk}"]["collection_path"]
        == "Shared Project"
    )
    assert (
        skills[f"{direct._meta.label_lower}:{direct.pk}"]["source_path"]
        == "Shared Project/direct_skill.md"
    )
    assert (
        skills[f"{packed._meta.label_lower}:{packed.pk}"]["collection_path"]
        == "Shared Project/skill_pack"
    )
    assert (
        skills[f"{packed._meta.label_lower}:{packed.pk}"]["source_path"]
        == "Shared Project/skill_pack/packed.md"
    )
    assert "Private Division" not in response.content.decode()


def _document_read_queries(captured):
    from apps.collections.models.collection import _get_document_types

    tables = tuple(f'FROM "{model._meta.db_table}"' for model in _get_document_types())
    return [
        query["sql"]
        for query in captured
        if query["sql"].lstrip().upper().startswith("SELECT")
        and any(table in query["sql"] for table in tables)
    ]


@pytest.mark.django_db
@override_settings(AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_prompt_candidate_reads_do_not_grow_with_unrelated_readable_collections():
    user = User.objects.create_user(username="bounded-runtime", password="pass")
    selected = Collection.objects.create(name="Selected")
    CollectionPermission.objects.create(
        user=user, collection=selected, permission="VIEW"
    )
    _raw_text_doc(
        selected, user, title="selected_skill.md", text="Selected instructions"
    )

    with CaptureQueriesContext(connection) as baseline_queries:
        baseline_prompt = load_collection_prompt_skills(user, [selected.pk])

    unrelated = Collection.objects.create(name="Unrelated")
    CollectionPermission.objects.create(
        user=user, collection=unrelated, permission="VIEW"
    )
    _raw_text_doc(
        unrelated, user, title="unrelated_skill.md", text="Unrelated instructions"
    )
    with CaptureQueriesContext(connection) as expanded_queries:
        expanded_prompt = load_collection_prompt_skills(user, [selected.pk])

    assert "Selected instructions" in baseline_prompt == expanded_prompt
    assert "Unrelated instructions" not in expanded_prompt
    baseline_reads = _document_read_queries(baseline_queries.captured_queries)
    expanded_reads = _document_read_queries(expanded_queries.captured_queries)
    assert baseline_reads
    assert len(expanded_reads) == len(baseline_reads)


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True, AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_override_validation_reads_only_referenced_document_models():
    user = User.objects.create_user(username="bounded-validation", password="pass")
    target = Collection.objects.create(name="Target")
    unrelated = Collection.objects.create(name="Unrelated")
    CollectionPermission.objects.create(user=user, collection=target, permission="VIEW")
    CollectionPermission.objects.create(
        user=user, collection=unrelated, permission="VIEW"
    )
    doc = _raw_text_doc(
        target, user, title="target_skill.md", text="Target instructions"
    )
    _raw_text_doc(unrelated, user, title="other_skill.md", text="Other instructions")
    overrides = {f"{doc._meta.label_lower}:{doc.pk}": True}

    with CaptureQueriesContext(connection) as queries:
        validated = validate_skill_overrides(user, overrides)

    assert validated == overrides
    assert len(_document_read_queries(queries.captured_queries)) == 1


@pytest.mark.django_db
@override_settings(AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_scoped_prompt_keeps_catalog_order_across_readable_ancestors():
    user = User.objects.create_user(username="scoped-order", password="pass")
    alpha = Collection.objects.create(name="Alpha Root")
    zulu = Collection.objects.create(name="Zulu Root")
    alpha_child = Collection.objects.create(name="Zulu Child", parent=alpha)
    zulu_child = Collection.objects.create(name="Alpha Child", parent=zulu)
    CollectionPermission.objects.create(user=user, collection=alpha, permission="VIEW")
    CollectionPermission.objects.create(user=user, collection=zulu, permission="VIEW")
    _raw_text_doc(alpha_child, user, title="first_skill.md", text="First ordered body")
    _raw_text_doc(zulu_child, user, title="second_skill.md", text="Second ordered body")

    prompt = load_collection_prompt_skills(user, [alpha_child.pk, zulu_child.pk])

    assert prompt.index("First ordered body") < prompt.index("Second ordered body")


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True, AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True)
def test_non_raw_pack_suffix_eligibility_matches_catalog_validation_and_runtime(client):
    user = User.objects.create_user(username="pack-suffix-reader", password="pass")
    pack = Collection.objects.create(name="skill_pack")
    CollectionPermission.objects.create(user=user, collection=pack, permission="VIEW")
    good = TeXDocument(
        title="good.md",
        full_text="Allowed TeX instruction",
        full_text_hash=TeXDocument.hash_fn("Allowed TeX instruction"),
        collection=pack,
        ingested_by=user,
    )
    good.save(dont_rechunk=True)
    trailing = TeXDocument(
        title="trailing.md ",
        full_text="Trailing TeX instruction",
        full_text_hash=TeXDocument.hash_fn("Trailing TeX instruction"),
        collection=pack,
        ingested_by=user,
    )
    trailing.save(dont_rechunk=True)
    good_id = f"{good._meta.label_lower}:{good.pk}"
    trailing_id = f"{trailing._meta.label_lower}:{trailing.pk}"
    client.force_login(user)

    catalog = client.get("/api/collections/chat-context/").json()
    assert {skill["id"] for skill in catalog["skills"]} == {good_id}
    assert validate_skill_overrides(user, {good_id: True}) == {good_id: True}
    with pytest.raises(ValidationError):
        validate_skill_overrides(user, {trailing_id: True})
    assert "Trailing TeX instruction" not in load_collection_prompt_skills(
        user, [], {trailing_id: True}
    )
    assert "Allowed TeX instruction" in load_collection_prompt_skills(
        user, [], {good_id: True}
    )
