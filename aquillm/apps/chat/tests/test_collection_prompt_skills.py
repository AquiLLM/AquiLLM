"""Collection-backed Markdown prompt skill loading."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.test import override_settings

from apps.chat.consumers.chat import CollectionsRef
from apps.chat.services.skills_runtime import (
    effective_base_system_for_memory,
    effective_base_system_for_memory_async,
)
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument
from aquillm.models import WSConversation


User = get_user_model()


def _raw_text_doc(collection: Collection, user, *, title: str, text: str) -> RawTextDocument:
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
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
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

    system = effective_base_system_for_memory(_consumer_for(user, db_convo, [collection.id]))

    assert "## Collection Skill: astro-python-scripts" in system
    assert "Description:\nGenerate astrophysics Python scripts. Prefer this skill for FITS and spectra work." in system
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
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    db_convo = WSConversation.objects.create(owner=user, system_prompt="Base system.")
    _raw_text_doc(
        collection,
        user,
        title="ordinary-notes.md",
        text="Do not treat this research note as system instructions.",
    )

    system = effective_base_system_for_memory(_consumer_for(user, db_convo, [collection.id]))

    assert "Do not treat this research note as system instructions." not in system


@pytest.mark.django_db
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
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
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
