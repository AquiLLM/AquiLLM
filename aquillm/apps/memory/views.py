"""Authenticated owner-scoped profile memory controls in existing settings."""
import json
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import UserMemoryFact
from .services.profile_facts import (ProfileFactConflict, get_active_profile_facts, correct_profile_fact, revoke_profile_fact)


def _public_fact(fact):
    labels = {'explicit_request': 'Explicit request', 'user_statement': 'User statement', 'manual': 'Manual', 'legacy_unknown': 'Unknown'}
    return {'id': fact.pk, 'fact': fact.fact, 'revision': fact.revision, 'status': fact.status,
            'origin_label': labels.get(fact.origin, 'Unknown'),
            'source_label': 'Conversation' if fact.source_message_uuid else ('User correction' if fact.origin == 'manual' else 'Unknown'),
            'reconciliation_status': fact.reconciliation_status}


@require_http_methods(['GET'])
def profile_memory_list(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required.'}, status=401)
    facts = get_active_profile_facts(request.user)
    pending = list(UserMemoryFact.objects.filter(user=request.user, reconciliation_status__in=['pending', 'failed'])
                   .values('id', 'reconciliation_status'))
    return JsonResponse({'facts': [_public_fact(fact) for fact in facts], 'reconciliations': pending})


@require_http_methods(['PATCH', 'DELETE'])
def profile_memory_detail(request, fact_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required.'}, status=401)
    if len(request.body) > 16000:
        return JsonResponse({'error': 'Request too large.'}, status=400)
    try:
        data = json.loads(request.body)
        if not isinstance(data, dict):
            raise ValueError()
        if request.method == 'PATCH':
            fact = correct_profile_fact(user=request.user, fact_id=fact_id, replacement=data.get('replacement'), expected_revision=data.get('expected_revision'))
            return JsonResponse({'fact': _public_fact(fact)})
        revoke_profile_fact(user=request.user, fact_id=fact_id, expected_revision=data.get('expected_revision'))
        fact = UserMemoryFact.objects.get(pk=fact_id, user=request.user)
        return JsonResponse({'status': 'revoked', 'reconciliation_status': fact.reconciliation_status})
    except UserMemoryFact.DoesNotExist:
        return JsonResponse({'error': 'Memory not found.'}, status=404)
    except ProfileFactConflict:
        return JsonResponse({'error': 'Memory changed. Reload before editing.'}, status=409)
    except (ValueError, TypeError):
        return JsonResponse({'error': 'A valid revision and substantive replacement are required.'}, status=400)
