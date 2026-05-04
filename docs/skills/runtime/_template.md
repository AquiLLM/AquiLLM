---
name: Replace with a short section title
---

Copy this file to `my-skill.md` in the same directory (no leading `_`), or add a new file next to it.
Remove the `_` prefix on the copy so the runtime loads it.

Use normal Markdown: lists, tone, and constraints for the chat model. This block is inserted into
the system prompt; keep it focused so you do not bloat the context.

For tools the model can *call* (not just read), implement a Python skill in `lib/skills/` and
register the module in `AQUILLM_SKILLS_EXTRA_MODULES` or `DEFAULT_BUILTIN_MODULES` in
`lib/skills/loader.py`.
