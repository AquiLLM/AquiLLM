"""Owner-scoped settings API, CSRF, and revision-controlled edits."""
import json
import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from apps.memory.models import UserMemoryFact

pytestmark = pytest.mark.django_db
URL = '/api/profile-memory/'


@pytest.fixture
def account():
    user = get_user_model().objects.create_user(username='profile-api')
    fact = UserMemoryFact.objects.create(user=user, fact='I prefer telescope A.')
    client = Client()
    client.force_login(user)
    return user, fact, client


def test_list_exposes_ordinary_origin_labels_without_internal_fields(account):
    user, fact, client = account
    other = get_user_model().objects.create_user(username='other-api')
    UserMemoryFact.objects.create(user=other, fact='Other private fact.')
    response = client.get(URL)
    assert response.status_code == 200
    row = response.json()['facts'][0]
    assert row['fact'] == 'I prefer telescope A.' and row['origin_label'] == 'Unknown'
    assert row['revision'] == 1 and row['status'] == 'active'
    assert set(row) == {'id', 'fact', 'revision', 'status', 'origin_label', 'source_label', 'reconciliation_status'}
    assert 'Other private fact.' not in response.content.decode()


def test_patch_then_delete_return_current_fact_and_no_discarded_text(account):
    user, fact, client = account
    response = client.patch(URL + str(fact.pk) + '/', data=json.dumps({'replacement': 'I prefer telescope B.', 'expected_revision': 1}), content_type='application/json')
    assert response.status_code == 200
    new = response.json()['fact']
    assert new['fact'] == 'I prefer telescope B.' and new['origin_label'] == 'Manual'
    assert 'telescope A' not in response.content.decode()
    response = client.delete(URL + str(new['id']) + '/', data=json.dumps({'expected_revision': new['revision']}), content_type='application/json')
    assert response.status_code == 200
    assert 'telescope' not in response.content.decode()
    assert client.get(URL).json()['facts'] == []
    assert not UserMemoryFact.objects.exclude(fact='').exists()


@pytest.mark.parametrize('method', ['patch', 'delete'])
def test_cross_user_and_stale_revision_are_rejected(account, method):
    user, fact, client = account
    other = get_user_model().objects.create_user(username='intruder')
    client.force_login(other)
    body = {'expected_revision': 1, 'replacement': 'I prefer telescope B.', 'user_id': user.pk}
    response = getattr(client, method)(URL + str(fact.pk) + '/', data=json.dumps(body), content_type='application/json')
    assert response.status_code == 404
    client.force_login(user)
    body['expected_revision'] = 2
    response = getattr(client, method)(URL + str(fact.pk) + '/', data=json.dumps(body), content_type='application/json')
    assert response.status_code == 409
    fact.refresh_from_db()
    assert fact.fact == 'I prefer telescope A.'


@pytest.mark.parametrize('revision', [True, None, '1', -1])
def test_revision_requires_a_positive_exact_integer(account, revision):
    response = account[2].delete(URL + str(account[1].pk) + '/', data=json.dumps({'expected_revision': revision}), content_type='application/json')
    assert response.status_code == 400


def test_api_requires_login_and_csrf(account):
    assert Client().get(URL).status_code == 401
    client = Client(enforce_csrf_checks=True)
    client.force_login(account[0])
    assert client.delete(URL + str(account[1].pk) + '/', data=json.dumps({'expected_revision': 1}), content_type='application/json').status_code == 403
