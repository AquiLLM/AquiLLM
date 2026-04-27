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
