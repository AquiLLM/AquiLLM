=====================
Using a collection
=====================

Attach to chat
==============

Do this in the current conversation so the model is allowed to read documents from the collection.

1. Wait until uploads in your collection have **finished processing** (use the ingestion monitor if you are unsure).
2. Open a chat: use **New Conversation** in the left menu for a new thread, or stay in an existing one.
3. Click the **Collections and skills** control beside the message field. It shows separate collection and skill counts.
4. In **Collections**, check the collections you want. Selecting or removing a parent also selects or removes its accessible descendants. Expand or collapse folders with the arrow, or search by name or parent path.
5. Optionally open **Skills** to inspect or change the response instructions for this chat. A collection's skills badge opens its associated skills.
6. Review the separate **Collections** and **Skills** groups under **In this chat**, then choose **Apply to chat**. The dialog closes after the server confirms the save.

Changes stay in the dialog until you apply them. **Cancel**, Escape, or clicking outside
the dialog discards your draft. If saving fails or times out, the draft stays open with
a message so you can retry. Wait until the current reply finishes before applying
changes. Your applied choices are restored when you reopen the conversation.

.. figure:: /_static/images/usingCollections/home_page.png
   :alt: AquiLLM chat with New Conversation in the sidebar and Collections button on the bottom bar
   :width: 800px
   :align: center

   Sidebar: **New Conversation**. Bottom bar: **Collections**.

.. figure:: /_static/images/usingCollections/adding_collections_to_chat.png
   :alt: Select Collections dialog with checkboxes for a collection and a Figures sub-collection
   :width: 800px
   :align: center

   The earlier collection-only dialog is shown here. The current picker has Collections and Skills tabs, an In this chat summary, and an Apply to chat button.

Markdown skills in collections
==============================

If your administrator has enabled collection skills, you can add **prompt instructions**
(not new tools) by uploading Markdown to a collection you attach to chat:

- Name a file ``skill.md``, ``skills.md``, or ``my-topic_skill.md``, **or**
- Add a subcollection named ``skills`` or ``skill_pack`` and put ``.md`` files inside it.

Collection selection enables associated skills by default. In **Skills**, you can enable
an individual skill without selecting its collection for document retrieval, or turn off
an inherited skill. **Follow collection selection** resets an individual override.
Removing a collection removes inherited skills; explicitly enabled skills stay enabled.
**Clear all** clears retrieval collections and disables all currently available collection
skills. Server-wide operator skills are managed separately and are unaffected.

Skill-pack collections remain selectable for retrieval and are labeled **Skill pack**.
Selecting a parent and its pack does not duplicate the pack's skills. See
:doc:`../skills/markdown` for naming rules and examples.

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
