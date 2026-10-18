"""Fault injection at real eager SQL fetch, preserving prior per-body assertions."""


def install_body_hook(monkeypatch, callback, *, before=False):
    from apps.documents.services import source_body_batch

    original = source_body_batch._fetch_body_rows

    def fetch(queryset):
        if before:
            callback(queryset, None)
        rows = original(queryset)
        if not before:
            # Existing controls capture ascending PK metadata. SQL batch order
            # is irrelevant; invoke their row assertions in that captured order.
            for row in sorted(rows, key=lambda item: item.pk):
                callback(queryset, row)
        return rows

    monkeypatch.setattr(source_body_batch, "_fetch_body_rows", fetch)
