# Manual chat search

Use these commands at the beginning of a chat message to run retrieval directly:

- `/collection calibration methods` searches the collections selected in the chat picker.
- `/search [Paper title] calibration methods` searches one document. A full UUID,
  a unique ID prefix, or a unique title fragment can be used inside brackets.
  A quoted title or a bare full UUID also works.
- `/search calibration methods` reuses the most recently searched/opened single
  document if it is still in the selected collections, or the only accessible
  document in those collections. If there is no unique target, chat asks for one.
- `/collection` and `/search [Paper title]` reuse the previous question. Without
  a previous question, chat asks what to search for.

For example, `/search "Calibration Guide" dark frames` searches that document;
`/collection dark frames` searches all selected collections.

Commands bypass automatic intent detection and run even when automatic direct
RAG is disabled. They execute one search with the supplied query, then produce
the usual cited answer. A failed or ambiguous `/search` never expands into a
collection search. Document permissions are checked by the existing search
tools; a full document UUID can be used without a selected collection when the
user has access. Commands are case-insensitive and must begin the message.
