"""Markdown prompt skills (server-side)."""
from __future__ import annotations

from lib.skills.markdown import _parse_simple_front_matter_block, load_markdown_prompt_bodies


def test_parse_front_matter_basic() -> None:
    raw = "---\nname: My block\n---\n\nLine one."
    meta, body = _parse_simple_front_matter_block(raw)
    assert meta.get("name") == "My block"
    assert body.strip() == "Line one."


def test_parse_no_front_matter_returns_all_as_body() -> None:
    raw = "No delimiter here"
    meta, rest = _parse_simple_front_matter_block(raw)
    assert meta == {}
    assert rest == "No delimiter here"


def test_load_respects_underscore_readme_and_sorts(tmp_path) -> None:
    (tmp_path / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "README.md").write_text("skip", encoding="utf-8")
    (tmp_path / "_template.md").write_text("skip", encoding="utf-8")
    sub = tmp_path / "x"
    sub.mkdir()
    (sub / "z.md").write_text("# Z", encoding="utf-8")
    out = load_markdown_prompt_bodies(tmp_path)
    assert "A" in out and "B" in out and "Z" in out
    assert "skip" not in out
    a_pos = out.index("A")
    b_pos = out.index("B")
    z_pos = out.index("Z")
    assert a_pos < b_pos < z_pos
