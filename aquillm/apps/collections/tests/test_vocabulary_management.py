"""Import/authentication and history admin cannot bypass current CAS authority."""
from importlib import import_module, util
from io import StringIO
import json

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import RequestFactory

from apps.collections.models import CollectionPermission
from apps.collections.tests.test_vocabulary_scope import entry, publish, setup_scope


@pytest.mark.django_db(transaction=True)
def test_import_requires_authentication_current_manage_and_revision(tmp_path, monkeypatch):
    name = "apps.collections.management.commands.import_vocabulary"
    assert util.find_spec(name), "authenticated vocabulary import is missing"
    module = import_module(name)
    actor, collection, _ = setup_scope()
    vocabulary, _ = publish(actor, collection)
    actor.set_password("synthetic-test-password")
    actor.save()
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps([entry()]))
    args = (str(path), "--username", actor.username, "--vocabulary", str(vocabulary.pk),
            "--expected-revision", "2")
    monkeypatch.setattr(module, "getpass", lambda _: "wrong")
    with pytest.raises(CommandError):
        call_command("import_vocabulary", *args)
    monkeypatch.setattr(module, "getpass", lambda _: "synthetic-test-password")
    output = StringIO()
    call_command("import_vocabulary", *args, stdout=output)
    assert "revision 3" in output.getvalue()
    with pytest.raises(CommandError):
        call_command("import_vocabulary", *args)
    CollectionPermission.objects.filter(user=actor, collection=collection).delete()
    with pytest.raises(CommandError):
        call_command("import_vocabulary", *args[:-1], "3")


@pytest.mark.django_db(transaction=True)
def test_admin_history_is_read_only_and_filters_ungranted_owners():
    assert util.find_spec("apps.collections.admin"), "vocabulary history admin is missing"
    import_module("apps.collections.admin")
    from django.contrib import admin
    from apps.collections.models import VocabularyVersion

    actor, collection, _ = setup_scope()
    _, version = publish(actor, collection)
    other, foreign, _ = setup_scope("foreign")
    _, hidden = publish(other, foreign)
    request = RequestFactory().get("/")
    request.user = actor
    history = admin.site._registry[VocabularyVersion]
    assert list(history.get_queryset(request).values_list("pk", flat=True)) == [version.pk]
    assert not history.has_add_permission(request)
    assert not history.has_change_permission(request, version)
    assert not history.has_delete_permission(request, version)
    with pytest.raises(PermissionError):
        history.save_model(request, hidden, None, False)
