Markdown skills
===============

Markdown skills are **plain ``.md`` files** merged into the chat system prompt (before
memory injection). They change how the model behaves; they do **not** register tools.

Adding a skill through collections (most users)
===============================================

If you use a shared AquiLLM instance, this is usually how you add a Markdown skill — no
server access required.

**Prerequisite:** your administrator must enable collection skills
(``AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=1`` on the server). If skills never appear
when you attach collections, ask them to turn this on.

Steps
-----

1. Create or open a **collection** (see :doc:`../collections/creating`).
2. Upload a Markdown file using **one** of these patterns:

   - Name the file ``skill.md``, ``skills.md``, or ``something_skill.md`` (the ``_skill``
     suffix marks it as a prompt skill), **or**
   - Put multiple ``.md`` files inside a **subcollection** named ``skills`` or
     ``skill_pack`` under your project collection.
3. **Attach that collection to your chat** (see :doc:`../collections/using`).
4. Start chatting — the skill text is merged into the system prompt for that session.

Only Markdown/raw-text documents are used. Regular PDFs in the same collection are ignored
for skills unless they follow the naming rules above.

Optional front matter
---------------------

.. code-block:: text

   ---
   name: Short title shown to the model
   description: One-line summary of when to use this skill
   ---

   Your instructions in Markdown.

Server-wide Markdown (operators only)
=====================================

If you **run the server**, you can also load a folder of ``.md`` files for **every** chat
on the instance:

1. Set ``AQUILLM_SKILLS_ENABLED=1`` in ``.env``.
2. Set ``AQUILLM_SKILLS_MARKDOWN_DIR`` to a folder of ``.md`` files, for example::

      AQUILLM_SKILLS_MARKDOWN_DIR=docs/skills/runtime

   The path is relative to the **repository root** (the parent of the ``aquillm`` Django
   project folder), unless you use an absolute path.
3. Restart AquiLLM so the new settings are picked up.

Every ``**/*.md`` under that directory is loaded **except** ``README.md`` and files whose
names start with ``_``.

Example (shipped in the repo)
-----------------------------

.. literalinclude:: ../../../../docs/skills/runtime/example-behavior-hints.md
   :language: md

Authoring tips
--------------

- **One concern per file**, or use front matter with a ``name:`` title.
- For tools the model can **call**, build a :doc:`Python skill <python_setup>` instead
  (requires running your own instance).

Need tools?
-----------

Markdown can't add actions. To give the model callable tools, see :doc:`python_setup`
(build from scratch) and the :doc:`python` reference.
