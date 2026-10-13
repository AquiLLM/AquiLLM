"""Private binding to the original captured selection; never reselects authority."""

from types import MappingProxyType
from uuid import UUID

from apps.knowledge_graph.retrieval.ready_scope import _encoded
from apps.knowledge_graph.retrieval.ready_scope_v2 import (
    GraphAuthorityGroupV2,
    SelectedGraphScopeV2,
    graph_authority_group_checksum_v2,
    graph_selection_digest_v2,
)


def _representation_values(selections):
    """Pure UUID/None plumbing; caller separately validates captured authority."""
    result = {}
    for row in selections:
        if row.document.id in result:
            raise ValueError("ambiguous captured representation")
        if row.generation_key is not None and type(row.generation_key) is not UUID:
            raise TypeError("representation generation must be exact UUID or None")
        result[row.document.id] = row.generation_key
    return result


class FrontierSelectionV2:
    def __init__(self, selection, codec):
        if type(selection) is not SelectedGraphScopeV2:
            raise TypeError("frontier requires the exact captured selection")
        selection.__post_init__()
        self.selection = selection
        self.digest = graph_selection_digest_v2(selection)
        representations = _representation_values(
            selection.representation_snapshot.selections
        )
        bindings = {}
        for group in selection.groups:
            encoded_generations, encoded_documents = [], []
            for authority in group.scope.projections:
                generation, documents = _encoded(authority, codec)
                encoded_generations.append(generation)
                encoded_documents.extend(documents)
                for (document_id, _), document in zip(
                    authority.documents, documents, strict=True
                ):
                    key = (generation.generation_key, document.document_key)
                    if key in bindings or document_id not in representations:
                        raise ValueError("missing or ambiguous representation binding")
                    bindings[key] = representations[document_id]
            if (
                tuple(sorted(encoded_generations, key=lambda g: g.collection_key))
                != group.scope.ready.selected_generations
                or tuple(
                    sorted(
                        encoded_documents,
                        key=lambda d: (d.collection_key, d.document_key),
                    )
                )
                != group.scope.ready.authorized_documents
            ):
                raise ValueError(
                    "captured opaque authority differs from identifier encoding"
                )
        self.representations = MappingProxyType(bindings)

    def require_group(self, group):
        if (
            type(group) is not GraphAuthorityGroupV2
            or group not in self.selection.groups
        ):
            raise ValueError("frontier group is outside captured selection")
        if graph_selection_digest_v2(self.selection) != self.digest:
            raise ValueError("captured selection changed")
        if group.group_checksum != graph_authority_group_checksum_v2(
            scope=group.scope,
            ontology_checksum=group.ontology_checksum,
            representation_snapshot=self.selection.representation_snapshot,
        ):
            raise ValueError("frontier group checksum differs")
        return group

    def representation(self, generation, document):
        try:
            return self.representations[(generation, document)]
        except KeyError as exc:
            raise ValueError(
                "evidence is outside captured document generation"
            ) from exc
