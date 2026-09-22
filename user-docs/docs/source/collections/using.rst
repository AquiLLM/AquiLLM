=====================
Using a collection
=====================

Attach to chat
==============

Do this in the current conversation so the model is allowed to read documents from the collection.

1. Wait until uploads in your collection have **finished processing** (use the ingestion monitor if you are unsure).
2. Open a chat: use **New Conversation** in the left menu for a new thread, or stay in an existing one.
3. Click **Collections** on the bottom bar (right side, next to the message field).
4. In the dialog, check the collection(s) you want. You can select several, including **sub-collections** (for example a **Figures** folder from a PDF) when they appear.

.. figure:: /_static/images/usingCollections/home_page.png
   :alt: AquiLLM chat with New Conversation in the sidebar and Collections button on the bottom bar
   :width: 800px
   :align: center

   Sidebar: **New Conversation**. Bottom bar: **Collections**.

.. figure:: /_static/images/usingCollections/adding_collections_to_chat.png
   :alt: Select Collections dialog with checkboxes for a collection and a Figures sub-collection
   :width: 800px
   :align: center

   Pick libraries here; the bar updates (for example **Collections (2 selected)**).

Markdown skills in collections
==============================

If your administrator has enabled collection skills, you can add **prompt instructions**
(not new tools) by uploading Markdown to a collection you attach to chat:

- Name a file ``skill.md``, ``skills.md``, or ``my-topic_skill.md``, **or**
- Add a subcollection named ``skills`` or ``skill_pack`` and put ``.md`` files inside it.

When that collection is selected for the conversation, AquiLLM merges the skill text into
the system prompt for that session. See :doc:`../skills/markdown` for naming rules and
examples.

Ask about documents
===================

Ask aquillm your question. If you see **vector_search** (or similar) above the assistant message, the model ran a search over the collections you attached.

.. figure:: /_static/images/usingCollections/ask_about_collection_document.png
   :alt: Chat showing a user question about a paper, vector_search tool calls, and a detailed model answer citing the document
   :width: 800px
   :align: center

   Example: a paper-specific question with retrieval-backed answer.

Open **vector_search** to read the query and the retrieved **chunks** when you want to check what text the model actually pulled in.

.. figure:: /_static/images/usingCollections/tool_call_output.png
   :alt: Expanded vector_search tool call showing search arguments and numbered chunk excerpts from the collection
   :width: 800px
   :align: center

   Expanded retrieval: query parameters and snippet-level matches from your documents.

.. important::

   If the model does not use your documents, the most common issue is that the **collection was not attached to the chat**.
