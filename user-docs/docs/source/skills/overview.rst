======
Skills
======

**Skills** extend what AquiLLM can do inside a chat. There are two kinds:

- **Markdown skills** — extra written instructions added to the model's system prompt.
  They shape *how* the assistant behaves (tone, rules, workflow). They do **not** add
  new actions.
- **Python skills** — code that registers **tools** the model can call (search an API,
  run a calculation, hit an integration). They can also add prompt text.

.. tip::

   **On a shared or hosted instance, you can add Markdown skills yourself** — upload a
   ``.md`` file to a collection and attach that collection to your chat. You do **not**
   need access to the server's ``.env``. Python tools still require whoever runs the
   server to configure them.

How do I add a skill?
=====================

The right steps depend on **which kind** of skill you want and **whether you run the
server**. Find your row below and follow the linked guide — every case has a path.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Skill kind
     - Using a shared / hosted instance
     - Running your own instance
   * - **Markdown (prompt text)**
     - **You can do this yourself.** Upload a ``.md`` file to a collection and attach the
       collection to your chat — no server access needed. Follow
       :doc:`markdown` → *Adding a skill through collections*.
       (Your admin must have collection skills enabled.)
     - Do the same collection workflow, **or** load a server-wide folder for every chat
       with ``AQUILLM_SKILLS_MARKDOWN_DIR``. Follow :doc:`markdown` →
       *Server-wide Markdown*.
   * - **Python (tools)**
     - **You cannot add tools yourself** — they load from server config. Ask whoever runs
       the instance. (If you only need behavior changes, use a Markdown skill instead.)
     - **You build and register it.** Follow :doc:`python_setup` step by step, then use the
       :doc:`python` reference for exact types.

.. note::

   Skills are **not** turned on from the chat window. Markdown skills are added through
   **collections** (any user) or a **server folder** (operators). Python tools always
   require server-side setup by whoever runs the instance.

Markdown vs Python at a glance
==============================

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - 
     - Markdown
     - Python
   * - **Best for**
     - Tone, domain rules, workflow hints
     - APIs, computation, integrations
   * - **Adds tools the model can call?**
     - No
     - Yes (via :py:func:`~lib.skills.builtin.dummy_skill.get_tools`)
   * - **Typical way to add (chat user)**
     - ``.md`` in a collection, attach to chat
     - Not available without server access
   * - **Operator-only server setting**
     - ``AQUILLM_SKILLS_MARKDOWN_DIR``
     - ``AQUILLM_SKILLS_EXTRA_MODULES``

.. _operator-setup:

Setup for instance operators
============================

This section is for **people who run the AquiLLM server** (self-hosted or admin). Chat
users on a shared instance can skip this and use :doc:`markdown` for the collection
workflow instead.

All operator setup happens in the server's ``.env`` file. **Restart the web and worker
processes** after changing it.

Step 1 — Turn skills on
-----------------------

Skills are off by default::

   AQUILLM_SKILLS_ENABLED=1

Nothing loads until this is set.

Step 2 — Enable collection skills (recommended for shared instances)
--------------------------------------------------------------------

So chat users can add Markdown skills via collections without you editing files on disk::

   AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=1

Users then upload ``.md`` files following the naming rules in :doc:`markdown`.

Step 3 — Optional server-wide Markdown or Python
------------------------------------------------

Add whichever global skills you want (you can use one or both):

- **Server-wide Markdown** — every chat on this instance gets these prompt blocks::

     AQUILLM_SKILLS_MARKDOWN_DIR=docs/skills/runtime

  Path is relative to the **repository root**, or absolute. Loads every ``.md`` except
  ``README.md`` and files starting with ``_``. Details in :doc:`markdown`.

- **Python tools** — import modules that expose ``get_tools``::

     AQUILLM_SKILLS_EXTRA_MODULES=lib.skills.builtin.dummy_skill,my_package.my_skill

  To build one from scratch, follow :doc:`python_setup`.

Step 4 — Restart and verify
---------------------------

Restart the server, then confirm prompt text or tools appear in a test chat.

Next steps
==========

Pick the guide that matches what you're doing:

- :doc:`markdown` — add prompt instructions (via collections, or a server folder).
- :doc:`python_setup` — build a Python tool skill from scratch (own instance).
- :doc:`python` — the Python skill API reference (types and ``@llm_tool``).

.. toctree::
   :maxdepth: 2
   :caption: Skill guides
   :hidden:

   markdown
   python_setup
   python
