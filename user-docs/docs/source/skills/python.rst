Python skills — API reference
==============================

This is the **reference** for the exact types and the tool decorator. For a walkthrough of
building a skill, start with :doc:`python_setup` and come back here for the details.

.. note::

   Everything below is generated from the AquiLLM source when the docs are built, so the
   signatures always match the running code.

The three things you write
--------------------------

A Python skill is one module that provides:

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Function
     - Required?
     - What it does
   * - :py:func:`~lib.skills.builtin.dummy_skill.get_tools`
     - Yes
     - Returns the list of tools the model may call this chat.
   * - :py:func:`~lib.skills.builtin.dummy_skill.get_system_prompt_extra`
     - No
     - Returns extra system-prompt text (return ``""`` or omit if unused).
   * - Each tool, wrapped in :py:func:`~lib.llm.decorators.llm_tool`
     - —
     - An ordinary function the model can call. The decorator turns it into a tool.

Everything else on this page just describes the **inputs** (the context object) and the
**outputs** (tool objects) of those functions.

The context your skill receives
-------------------------------

Both entry points get a ``ctx`` argument. It is a plain dict (no database objects) with the
user and conversation ids — safe to read, JSON-friendly:

.. autoclass:: lib.skills.types.SkillRuntimeContext
   :members:

Entry points
------------

These are the two functions the loader looks for. The template's versions are shown here as
the canonical signatures:

.. autofunction:: lib.skills.builtin.dummy_skill.get_tools

.. autofunction:: lib.skills.builtin.dummy_skill.get_system_prompt_extra

Declaring a tool with ``@llm_tool``
-----------------------------------

You don't build tool objects by hand. Write a normal function, decorate it with
``@llm_tool``, and the decorator generates the JSON argument schema the model sees.

**The rules it enforces:**

- Every parameter needs a **type hint** *and* an entry in ``param_descs``.
- Only ``str``, ``int``, ``bool``, and lists of those are allowed as parameter types.
- ``required`` lists the parameters the model must always send.
- ``for_whom`` decides where the result goes: ``"assistant"`` (back to the model) or
  ``"user"`` (shown to the person).

.. autofunction:: lib.llm.decorators.llm_tool

:py:func:`~lib.skills.builtin.dummy_skill.get_tools` returns these as ``LLMTool`` objects —
you get one automatically from every :py:func:`~lib.llm.decorators.llm_tool` function, so you
rarely touch this type directly.

Full template
-------------

The template skill wires one echo tool through
:py:func:`~lib.skills.builtin.dummy_skill.get_tools`. Read it top to bottom to see the
context, the decorator, the entry points, and the optional prompt text together:

.. literalinclude:: ../../../../aquillm/lib/skills/builtin/dummy_skill.py
   :language: python
