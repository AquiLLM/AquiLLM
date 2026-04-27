# MCP + Skills + Agents Runtime Architecture

Status: Active (skills implemented; MCP and agents planned)  
Date: 2026-04-27

## Goals

- One registration path for tools that reach the LLM (`LLMTool` in `lib/llm`), whether they come from document RAG, astronomy helpers, **runtime skills**, or (later) MCP.
- Clear boundaries: orchestration and HTTP/Channels wiring stay in `apps/`; reusable contracts stay in `lib/`.
- Feature flags so each source can be disabled without behavior change for existing deployments.

## Skills (implemented)

There are two complementary kinds of “skill” in the **AquiLLM server** (neither is a Cursor-only `SKILL.md` for the IDE). The subsections below are the canonical split; for a **comparison table** aimed at authors, see `docs/skills/runtime/README.md`.

### Python skills (LLM tools + optional prompt text)

A **Python skill** is a module (under `lib/skills/` or any importable dotted path) that exposes:

| Entry point | Required | Purpose |
|-------------|----------|---------|
| `get_tools(ctx) -> list[LLMTool]` | Yes | Extra tools for the chat turn |
| `get_system_prompt_extra(ctx) -> str` | No | Text merged into the conversation base system prompt **before** profile/episodic memory injection |

`ctx` is a `SkillRuntimeContext` (`user_id`, `username`, optional `conversation_id`)—no ORM objects, so skills stay importable from `lib` without violating `apps` import boundaries (see `scripts/check_import_boundaries.py`).

### Markdown prompt skills (text only, `.md` files)

Drop plain Markdown (optional `---` / `---` front matter with `name:`) into a directory and point **`AQUILLM_SKILLS_MARKDOWN_DIR`** at it. Each loaded file becomes a `##` section in the system prompt. **No LLM tools** are registered this way; use Python skills for tool calls. Skips `README.md` and files starting with `_` (e.g. `_template.md`). Example tree: `docs/skills/runtime/`.

### Where it lives in code

| Area | Role |
|------|------|
| `lib/skills/types.py` | `SkillRuntimeContext`, `SkillModule` protocol |
| `lib/skills/loader.py` | Import skill modules by dotted path; log and skip failures |
| `lib/skills/registry.py` | Merge tools (first registration wins on name collision) and prompt extras |
| `lib/skills/markdown.py` | Load and merge `.md` prompt skills from a directory |
| `lib/skills/builtin/example_runtime_skill.py` | Default built-in skill when skills are on (quality-gate reminder tool + small prompt blurb) |
| `lib/skills/builtin/dummy_skill.py` | **Template only** — commented echo tool; not loaded by default; add path to `AQUILLM_SKILLS_EXTRA_MODULES` (or copy the file) when authoring new skills |
| `apps/chat/services/skills_runtime.py` | Reads Django settings, builds context from `ChatConsumer`, returns tools and effective base system string |

### Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `AQUILLM_SKILLS_ENABLED` | off (`0` / false) | When on, built-in + extra skill modules are loaded; markdown directory is also honored when set |
| `AQUILLM_SKILLS_EXTRA_MODULES` | empty | Comma-separated dotted paths **in addition** to the built-in example skill |
| `AQUILLM_SKILLS_MARKDOWN_DIR` | empty | Directory of `.md` files (path relative to **repository root** or absolute); empty = do not load markdown skills |

Built-in module list is `lib.skills.loader.DEFAULT_BUILTIN_MODULES` (includes `lib.skills.builtin.example_runtime_skill`).

### Runtime flow (chat)

1. On WebSocket connect, after a valid `WSConversation` is loaded, `ChatConsumer` appends `build_skill_tools(consumer)` when skills are enabled.
2. For memory augmentation (`augment_conversation_with_memory_async`), the base system string is `effective_base_system_for_memory(consumer)` = DB `system_prompt` + **Python** `get_system_prompt_extra` text (if any) + **markdown** file content (if any), then episodic/profile memory blocks are appended as today.

MCP and separate agent loops are not wired here yet; they will use the same `LLMTool` shape when implemented per `docs/roadmap/plans/pending/2026-03-20-mcp-skills-agents-structure.md`.

## MCP (planned)

- **Config:** env-driven server list, timeouts, allow/deny lists (see pending implementation plan).
- **Adapter:** map MCP tool definitions to `LLMTool` and register beside built-in and skill tools.
- **This document** will gain client lifecycle, failure containment, and troubleshooting when MCP lands.

## Agents (planned)

- **Policy:** caps on steps, tool calls, and model turns; opt-in feature flag.
- **Orchestration:** reuse `LLMInterface` and the unified tool list rather than a parallel tool system.

## Related references

- Quality gates and file ownership: `docs/documents/standards/code-style-guide.md`
- Roadmap: `docs/roadmap/plans/pending/2026-03-20-mcp-skills-agents-structure.md`
- Cursor / IDE “skills” can live under `.cursor/skills/<name>/SKILL.md` and are separate from the **AquiLLM** markdown directory (`AQUILLM_SKILLS_MARKDOWN_DIR`); the latter is read by the Django app for end-user chat.
