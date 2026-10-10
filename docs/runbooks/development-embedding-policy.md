# Development embedding fallback policy

The ordinary embedding facade reads `APP_EMBED_FALLBACK_POLICY` before any
single, batch, or multimodal provider call. Accepted values are exact:

| Value | Local provider failure |
| --- | --- |
| `local-only` | Raises redacted `EmbeddingUpstreamUnavailableError`; never calls Cohere |
| `legacy-cohere` | Retains existing Cohere text fallback |
| Absent | Retains `legacy-cohere` for compatibility outside development |
| Empty, whitespace, or any other value | Permanent `EmbeddingContractError`, before provider calls |

Malformed vectors remain permanent errors under both policies. Multimodal pooling
may still fall back to the caption's local text embedding. Transient local outages
remain retryable by existing callers; provider retry counts and deadlines are
unchanged. In particular, chunk ingestion's pre-existing unbounded retry policy
for transient errors is unchanged; operators should monitor backlog during outages.

Development Compose fixes `local-only` in each Django web, worker, and scheduler
environment using its existing environment overrides. These override the shared
`.env`, including a legacy, empty, or misspelled value there. This is an explicit
development policy, not a parser default. A process launched outside that Compose
configuration must set its own policy. Never use `${VAR:-legacy-cohere}` to consume
this setting: it would hide an explicitly empty value. Base, production, no-GPU,
and model service definitions are unchanged.

The audit's `declared.transport_failure_policy` reports the selected exact value.
History remains unknown and compatibility remains unproven. This prevents new
silent provider substitution; it neither identifies nor repairs historical vectors.

## Bounded coordinator rollout and smoke

Only the coordinator operates development. Do not print resolved environments,
provider errors, source text, credentials, or private identifiers into tracked logs.

1. Review the commit and run the focused embedding, Compose policy, chunk diagnostics,
   and chunk candidate tests in the isolated Linux test environment. Render only
   the selected policy key for each affected app process using the reviewed Compose
   input; require `local-only`. Do not run a generic full config dump.
2. Recreate the affected development application containers using the normal
   deployment procedure; do not recreate or change model services. Verify the
   policy key inside every running web/worker process. Pending optional workers
   must receive the same configuration when started.
3. In one web process and one ingestion worker environment, run
   `python -m lib.embeddings.audit` from the application Python path. Require
   `transport_failure_policy=local-only`, `historical_identity=unknown`, and
   `compatibility=unproven`. An optional single `--probe` per process sends two
   synthetic strings with a 10-second timeout and zero client retries; no DB writes.
4. Run the synthetic policy regressions, including transport failures, within an
   isolated test process. Do not stop the live embedding service to induce failure.
   Require no Cohere calls, a typed redacted transient error, and the existing
   chunk search `embedding_unavailable` diagnostic with scoped lexical results
   (or an explicit empty result when no lexical candidates exist).
5. Run the coordinator's established authorized cold combined/parent/graph-off
   replay once, with existing caps and deadlines. Compare source support and
   availability honestly; do not infer retrieval quality from successful calls.

If application availability regresses, pause new development ingestion and inspect
the local provider/backlog. Restore the prior reviewed application revision/config
only by a deliberate coordinator decision; doing so can restore cross-provider
fallback risk. Changing shared `.env` alone cannot undo the development override.
Do not re-embed, switch models, alter indexes, or enable gated retrieval modes as
part of this rollout.
