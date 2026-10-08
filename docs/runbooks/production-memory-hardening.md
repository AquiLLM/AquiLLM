# Production memory isolation and retention

Production Compose publishes Qdrant only on `127.0.0.1:6333:6333` and requires
`MEM0_QDRANT_API_KEY` to be nonempty when Compose interpolates its configuration.
The same protected value must reach Qdrant and every process that uses Mem0,
including web, workers, and any separately managed helper clients. Do not put
the value in source control, logs, command arguments, or this runbook.

## Deploy and rotate authentication

1. Record the current deployment revision and take the normal Qdrant backup.
   Preserve the existing Qdrant storage mount, collection name, embedding width,
   and model settings. This change requires no collection migration.
2. Generate and store a strong key in the deployment's protected environment
   file as `MEM0_QDRANT_API_KEY`. Ensure the environment used for Compose
   interpolation and the environment delivered to each client use the same value.
   Service `env_file` alone does not supply Compose interpolation variables.
3. Set `MEM0_QDRANT_URL=http://qdrant:6333` for Docker clients if an explicit URL
   is desired. Leaving it blank constructs that HTTP URL from host/port when
   a key is present. Explicit URLs take precedence; use an HTTPS URL for a
   separately managed TLS endpoint. Anonymous development clients retain the
   host/port configuration when both optional variables are blank.
4. During a coordinated maintenance window, stop requests and memory writers,
   update the server and all clients together, and recreate Qdrant and the client
   containers. A key rotation must repeat this coordination; restarting only
   the server or only the clients causes authentication failures.
5. Verify loopback-only publication, unauthenticated request rejection, and
   authenticated read/write access from the application network. Check the
   normal memory retrieval flow without printing request headers or the key.
   Restore traffic only after these checks pass.

If rollback is needed, restore the previous protected key on both server and
clients together, plus the prior application revision if necessary. Preserve
loopback publication and authentication. Never reopen the wildcard port as a
rollback shortcut. Container network HTTP is deliberate; do not publish that
endpoint publicly.

## Graph artifact retention

The dedicated `scheduler_knowledge_graph_maintenance` service requires the
`knowledge-graph` profile and `KG_MAINTENANCE_SCHEDULER_ENABLED=1`. Artifact
pruning additionally requires `KG_ARTIFACT_PRUNING_ENABLED=1`; its default is
`0`. `KG_ARTIFACT_PRUNING_INTERVAL_SECONDS` defaults to `86400`. Keep pruning
disabled until the production retention policy and backup/recovery checks are
approved. The scheduler uses its allowlisted environment and Redis dependency,
without graph database credentials. Application beat always sets both graph
maintenance and artifact pruning to `0`.

Before opting in, inspect artifact usage, establish which artifacts are eligible
under the current retention policy, validate recovery, and confirm that only the
dedicated graph beat schedules pruning. These settings do not change model
configuration, persistent memory storage paths, or Qdrant collection names.
