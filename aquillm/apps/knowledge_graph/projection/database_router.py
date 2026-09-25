from __future__ import annotations

from apps.knowledge_graph.models.projections import ProjectionAuthorityModel


class ProjectionDatabaseRouter:
    """Route only explicitly marked projection-worker operations."""

    def db_for_read(self, model, **hints):
        del model
        if hints.get("projection_source") is True:
            return "projection_source"
        return None

    def db_for_write(self, model, **hints):
        if hints.get("projection_worker_state") is not True:
            return None
        try:
            is_state_model = issubclass(model, ProjectionAuthorityModel)
        except TypeError:
            is_state_model = False
        return "projection_state" if is_state_model else None

    def allow_relation(self, obj1, obj2, **hints):
        del obj1, obj2, hints
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        del app_label, model_name, hints
        if db in {"projection_source", "projection_state"}:
            return False
        return None


__all__ = ["ProjectionDatabaseRouter"]
