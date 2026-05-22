# AquiLLM markdown “skills” (system prompt add-ons)

**Documentation distinction:** AquiLLM has **two** server-side skill types. This page is only for **Markdown prompt skills** (text in `.md` files). **Python skills** are separate: importable modules with `get_tools` that register `LLMTool` callables. See the summary table in `docs/documents/architecture/mcp-skills-agents-runtime.md` (sections *Python skills* and *Markdown prompt skills*).

| | Markdown (this folder) | Python (`lib/skills/…`) |
|---|------------------------|-------------------------|
| **Delivers** | System prompt text | Optional prompt text **and/or** LLM tools |
| **Config** | `AQUILLM_SKILLS_MARKDOWN_DIR` | `AQUILLM_SKILLS_EXTRA_MODULES` + built-ins in `lib/skills/loader.py` |
| **Format** | `.md` files in a directory | Python modules |

---

These are **plain Markdown files** merged into the chat system prompt (before memory injection) when you enable AquiLLM skills and set `AQUILLM_SKILLS_MARKDOWN_DIR` to this folder (or another directory), for example:

`AQUILLM_SKILLS_MARKDOWN_DIR=docs/skills/runtime` (path is relative to the **repository root**, the parent of the `aquillm` Django project folder)

They are **not** the same as Claude/Cursor/IDE `SKILL.md` files: the **web app** reads them at runtime, not the editor.

## Collection Session Skills

AquiLLM can also load prompt-only Markdown skills from selected collections when:

```env
AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=1
AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS=12000
```

Collection skills are session-scoped; they are appended only for conversations with matching selected collections. They do not install global skills and they do not register Python tools.

Supported collection patterns:

- A Markdown document in a selected collection named `skills`, `skill`, or ending in `_skill`.
- A direct child subcollection named `skills` or `skill_pack`; Markdown/raw-text documents inside it are loaded as a skill pack.

The model-facing skill title comes from front matter such as `name: astro-python-scripts`, not from the filename. `description:` is preserved and shown to the model as activation guidance.

## What to add

- One concern per file, or one file with a `name` in front matter (see `_template.md`).
- Optional YAML-style front matter:

  ```text
  ---
  name: Short title shown to the model
  ---

  Your instructions in Markdown.
  ```

- Files named `README.md` or starting with `_` are **not** loaded (`_` is for templates you copy from).

## Tools (LLM function calls)

Markdown files here add **text only**. To expose tools to the model, add a **Python** skill under `lib/skills/` (see `lib/skills/builtin/dummy_skill.py`).
