"""Bounded identity receipts; a usable subset does not prove full coverage."""


def capture_provider_receipt(result):
    events = result.get('results') if isinstance(result, dict) else None
    if not isinstance(events, list):
        return [], False
    identities, complete = [], True
    for event in events:
        kind = event.get('event') if isinstance(event, dict) else None
        if not isinstance(kind, str) or kind not in {'ADD', 'UPDATE', 'UPSERT', 'DELETE', 'NONE'}:
            complete = False
            continue
        if kind == 'NONE':
            continue
        identity = event.get('id')
        if not isinstance(identity, str) or not identity or len(identity) > 256:
            complete = False
        elif identity not in identities:
            if len(identities) < 50:
                identities.append(identity)
            else:
                complete = False
    return identities, complete
