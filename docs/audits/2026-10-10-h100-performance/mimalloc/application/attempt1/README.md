# Aborted application replay and scoped recovery

These captures belong only to `allocator20261010`, the first application attempt.
Its 12 system-allocator requests passed exact answer and persistence checks, but
the run stopped before a mimalloc block. None are used in paired latency results.

After replacement of the web container, `docker cp` restored the state journal
with root ownership. Conversation deletion succeeded; writing the updated
journal then raised `PermissionError`. Final cleanup refused the resulting
database/journal mismatch. The runner restored both original service images and
verified their health, as recorded in `runner-output.txt`.

The recovery checked the exact 12 recorded conversation IDs against the retained
proof, confirmed all were already absent, and repaired only that private journal.
It then cleared the test principal's live Mem0 namespace and deleted its own
fixture and principal. The recovery cleanup capture confirms that deletion and
empty live vector/graph namespace. Mem0 deletion history remains deliberately
retained; complete background-task quiescence was not proved.

Fixture deletion triggered the existing application's synchronous post-commit
canonical reconciliation. That rebuild can lock all active collection scopes and
update derived canonical entities, links, and membership projections. It does not
delete unrelated source collections, documents, chunks, or graph inputs. This
maintenance was outside request timing. It completed naturally; the guarded
attempt to stop the owned recovery found its process already absent and sent no
signal.

The retry uses a fresh fixture and repairs ownership of the exact state/fixture
files after each copy. It also checks journal writability before deletion. Exact
executed retry and recovery scripts are preserved in
[reproduction](../../reproduction/README.md). `manifest.json` binds the 14 captured
evidence files in this directory; this explanatory README is not runtime evidence.
