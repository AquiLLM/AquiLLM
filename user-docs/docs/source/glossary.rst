====================
Glossary
====================

.. glossary::

   Collection
      A group of documents that AquiLLM can use when answering questions. Collections are the primary organizational unit.

   Chunk
      A smaller piece that documents are split into during processing. Chunks enable efficient information retrieval.

   Ingestion
      The process of uploading, parsing, chunking, and indexing documents so they can be used by AquiLLM.

   Sub-Collection
      A nested collection within a document, typically containing extracted content like figures, tables, or images.

   Skill
      An extension that adds instructions (Markdown) or callable tools (Python) to AquiLLM.
      **Markdown skills** on a shared instance: upload a ``*_skill.md`` file (or a ``skills``
      / ``skill_pack`` subcollection) to a collection and attach it to chat — no server
      configuration by the end user. **Python skills** require the instance operator to
      register modules on the server.

   Vector Search
      The retrieval mechanism AquiLLM uses to find relevant document chunks based on a user's question.

   Zotero Integration
      A feature that allows you to connect your Zotero account and import libraries directly into AquiLLM collections.
