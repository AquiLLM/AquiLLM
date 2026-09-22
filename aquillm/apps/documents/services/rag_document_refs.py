"""Serialize document references and hydrate them within the current access scope."""
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def document_refs_from_documents(documents: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"model": d.__class__.__name__, "pkid": int(d.pkid)} for d in documents]


def _validated_collection_allowlist(
    allowed_collection_ids: Sequence[int],
) -> tuple[int, ...]:
    try:
        collection_ids = tuple(allowed_collection_ids)
    except TypeError as exc:
        raise ValueError(
            "allowed_collection_ids must contain positive integers within the signed-bigint range"
        ) from exc

    if any(
        type(collection_id) is not int
        or collection_id <= 0
        or collection_id > 2**63 - 1
        for collection_id in collection_ids
    ):
        raise ValueError(
            "allowed_collection_ids must contain positive integers within the signed-bigint range"
        )
    return tuple(sorted(set(collection_ids)))


def rehydrate_documents_from_refs(
    refs: Sequence[Mapping[str, Any]],
    allowed_collection_ids: Sequence[int],
) -> list[Any]:
    from django.apps import apps

    collection_ids = _validated_collection_allowlist(allowed_collection_ids)
    if not refs or not collection_ids:
        return []

    by_model: dict[str, list[int]] = defaultdict(list)
    order: list[tuple[str, int]] = []
    for ref in refs:
        try:
            mname = str(ref["model"])
            pk = int(ref["pkid"])
        except Exception:
            continue
        by_model[mname].append(pk)
        order.append((mname, pk))

    fetched: dict[tuple[str, int], Any] = {}
    for mname, pks in by_model.items():
        try:
            model = apps.get_model("apps_documents", mname)
        except Exception:
            continue
        uniq_pks = list(dict.fromkeys(pks))
        try:
            qs = model.objects.filter(
                pkid__in=uniq_pks,
                collection_id__in=collection_ids,
            )
        except Exception:
            continue
        for doc in qs:
            try:
                fetched[(mname, int(doc.pkid))] = doc
            except Exception:
                continue

    out: list[Any] = []
    for mname, pk in order:
        doc = fetched.get((mname, pk))
        if doc is not None:
            out.append(doc)
    return out
