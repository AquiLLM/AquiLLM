Build a Python skill
====================

This page is for people **building a Python skill from scratch** on an instance they
control. Writing tools requires editing files and environment variables on the server, so
it is only possible if you **run your own AquiLLM instance** (self-hosted or admin).

.. note::

   **On a shared / hosted instance you cannot add Python tools yourself** — they load from
   server configuration only. Ask whoever runs the instance. If you only need to change how
   the assistant *behaves* (not add tools), use :doc:`markdown` via collections instead.

What you are building
---------------------

A Python skill is a plain, importable module that exposes:

- **``get_tools(ctx) -> list[LLMTool]``** — required; the tools the model can call.
- **``get_system_prompt_extra(ctx) -> str``** — optional; extra system-prompt text.

For the exact signatures and the tool decorator, see the :doc:`python` reference
(:py:func:`~lib.skills.builtin.dummy_skill.get_tools`,
:py:func:`~lib.skills.builtin.dummy_skill.get_system_prompt_extra`).

Step 1 — Copy the template
--------------------------

Start from the template module shipped in the repo:

- ``aquillm/lib/skills/builtin/dummy_skill.py``

Copy it to a new module, either under ``aquillm/lib/skills/builtin/`` or in any package
that is importable by the server, for example ``my_package/my_skill.py``.

Step 2 — Write your tool(s)
---------------------------

In your new module:

1. Set a stable ``SKILL_ID`` string (used in logs and ops; need not match the file name).
2. Replace the example ``@llm_tool`` function with your own. Give every parameter a **type
   hint**, a ``param_descs`` entry, and list required names in ``required``.
3. Return the tools from :py:func:`~lib.skills.builtin.dummy_skill.get_tools`.
4. Optionally return static instructions from
   :py:func:`~lib.skills.builtin.dummy_skill.get_system_prompt_extra`.
5. Keep the module free of ``apps.*`` imports — use
   :py:class:`~lib.skills.types.SkillRuntimeContext` for user and conversation ids.

.. code-block:: python

   from lib.llm.decorators import llm_tool
   from lib.llm.types import LLMTool, ToolResultDict
   from ..types import SkillRuntimeContext

   SKILL_ID = "my_skill_id"

   @llm_tool(
       for_whom="assistant",
       description="What this tool does.",
       required=["query"],
       param_descs={"query": "What the model should pass in."},
   )
   def my_tool(query: str) -> ToolResultDict:
       return {"result": f"handled: {query}"}

   def get_tools(_ctx: SkillRuntimeContext) -> list[LLMTool]:
       return [my_tool]

Step 3 — Register the module
----------------------------

In the server's ``.env``::

   AQUILLM_SKILLS_ENABLED=1
   AQUILLM_SKILLS_EXTRA_MODULES=my_package.my_skill

Use a comma-separated list for several modules. To try the template itself, use
``AQUILLM_SKILLS_EXTRA_MODULES=lib.skills.builtin.dummy_skill``.

Step 4 — Restart and verify
---------------------------

Restart the web and worker processes, start a chat, and confirm the model can call your
tool (a tool-call block appears above the assistant message).

Author checklist
----------------

- **``SKILL_ID``** — stable identifier for logs and ops.
- **``get_tools(ctx)``** — required; returns ``LLMTool`` instances.
- **``get_system_prompt_extra(ctx)``** — optional; extra system-prompt text.
- **No ``apps.*`` imports**; use ``SkillRuntimeContext`` only.
- **Register** the dotted module path in ``AQUILLM_SKILLS_EXTRA_MODULES`` (with
  ``AQUILLM_SKILLS_ENABLED=1``).

Next: the :doc:`python` reference documents the exact types and the ``@llm_tool`` decorator,
pulled straight from the source.
