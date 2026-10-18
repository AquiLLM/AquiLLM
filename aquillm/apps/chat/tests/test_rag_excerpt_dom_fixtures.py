"""Keep browser regressions bound to real fallback-renderer output."""

import json
from pathlib import Path
from types import SimpleNamespace

from apps.chat.services.rag_support_delivery import _render_unknown_excerpts

FIXTURES = (
    Path(__file__).resolve().parents[4]
    / "react/tests/fixtures/scientific-excerpts.json"
)


def rendered_fixture(source):
    citation = "[doc:11111111-1111-4111-8111-111111111111 chunk:1]"
    packet = SimpleNamespace(
        chunks=[{
            "doc_id": "11111111-1111-4111-8111-111111111111",
            "chunk_id": 1,
            "text": source,
        }],
        citation_tokens=[citation],
    )
    return _render_unknown_excerpts(packet, SimpleNamespace(state="unknown"))[0]


def test_browser_fixtures_are_actual_fallback_renderer_output():
    # These outputs are consumed by MessageBubble's DOM tests. If presentation
    # changes, regenerate them with this renderer and rerun the browser checks.
    for fixture in json.loads(FIXTURES.read_text(encoding="utf-8")):
        assert rendered_fixture(fixture["source"]) == fixture["published"]
