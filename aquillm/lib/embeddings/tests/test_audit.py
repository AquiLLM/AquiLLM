import json
from types import SimpleNamespace


def test_audit_separates_declarations_from_unknown_history_and_redacts(monkeypatch):
    from lib.embeddings.audit import build_report

    monkeypatch.setenv(
        "APP_EMBED_BASE_URL", "https://user:secret@example.test/v1?key=secret"
    )
    monkeypatch.setenv("APP_EMBED_API_KEY", "never-print-this")
    monkeypatch.setenv("APP_EMBED_MODEL", "private-model-name")
    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "private-revision")
    report = build_report()
    assert report["historical_identity"] == "unknown"
    assert report["observed"] == {"status": "not_probed"}
    assert report["compatibility"] == "unproven"
    encoded = json.dumps(report)
    for private in (
        "secret",
        "example.test",
        "never-print-this",
        "private-model-name",
        "private-revision",
    ):
        assert private not in encoded


def test_probe_is_bounded_and_reports_raw_vector_behavior(monkeypatch):
    from lib.embeddings import audit

    monkeypatch.setenv("APP_EMBED_DIMS", "4")
    monkeypatch.setenv("APP_EMBED_MODEL", "test-model")
    requests = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 10.0
            assert kwargs["max_retries"] == 0
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            requests.append(kwargs)
            assert len(kwargs["input"]) == 2
            return SimpleNamespace(
                model="test-model",
                data=[
                    SimpleNamespace(index=0, embedding=[3.0, 4.0]),
                    SimpleNamespace(index=1, embedding=[3.0, 4.0]),
                ],
            )

        def close(self):
            pass

    monkeypatch.setattr(audit, "OpenAI", Client)
    report = audit.build_report(probe=True)
    assert len(requests) == 1
    assert report["observed"]["raw_dimensions"] == [2, 2]
    assert report["observed"]["raw_norms"] == [5.0, 5.0]
    assert report["observed"]["response_model_matches_declared"] is True
    assert report["observed"]["dimension_adaptation"] == ["pad", "pad"]
    assert report["compatibility"] == "unproven"


def test_probe_failure_does_not_echo_upstream_payload(monkeypatch):
    from lib.embeddings import audit

    def fail(**kwargs):
        raise RuntimeError("private endpoint secret body")

    monkeypatch.setattr(audit, "OpenAI", fail)
    report = audit.build_report(probe=True)
    assert report["observed"] == {"status": "upstream_failure"}
    assert "secret" not in json.dumps(report)
