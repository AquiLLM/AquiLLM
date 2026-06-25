---
name: aquillm-local-verification
description: >-
  Reminds the agent to run the repo’s structural and test checks before claiming
  backend or frontend work is done. Use when changing Python in `aquillm/`, React
  in `react/`, or repository hygiene, CI expectations, or “is this PR-ready?”
---

# AquiLLM local verification (Cursor skill)

## When this applies

Use when editing this repository and about to state that a change is complete, passing, or ready for review.

## Quality gates (must match the code style guide)

From the repository root, run the structural checks:

```bash
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
pwsh -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1
```

## Targeted test commands

- Backend: `cd aquillm` then `python -m pytest -q --tb=short` (narrow further when appropriate).
- Frontend: `cd react` then `npm run typecheck` and `npm run build` if TypeScript or UI changed.

## Consistency

Do not assert success without having run the relevant commands (or an equivalent the user has agreed to). Favor `rtk`-prefixed shell commands in agent transcripts when the environment provides RTK, per project rules.

For architecture details: `docs/documents/standards/code-style-guide.md` and `docs/documents/architecture/mcp-skills-agents-runtime.md`.
