# Development embedding fallback policy

Task 5A, 2026-10-09. Local development implementation only; coordinator owns rollout.

## Evidence and scope

The existing facade catches transport errors then requests Cohere vectors. The
existing sentinel regression `test_transport_failure_retains_legacy_cohere_fallback_and_role`
passed before changes (1 passed), proving this route returns the sentinel vector.
No evidence establishes compatibility between local, Cohere, or historical vectors.

## Exact subplan

1. Add failing single/batch/multimodal policy tests at the real local transport
   boundary. Cover absent and explicit legacy policy, strict invalid configuration,
   valid local wire inputs, redacted transient errors, and audit declarations.
2. Add `get_embed_fallback_policy()` in embedding config: only exact `local-only`
   and `legacy-cohere` strings; absent means legacy. Invalid present values raise
   the existing permanent `EmbeddingContractError` without reflecting their value.
   Use a function-local error import to avoid the current config/utils cycle.
3. Validate policy before provider calls in the three ordinary facade entry points.
   Preserve contract failures. For local-only transport failure, raise a redacted
   `EmbeddingUpstreamUnavailableError(RuntimeError)` with suppressed chaining;
   no Cohere client lookup or call. Preserve successful local inputs/dimension
   fitting and all existing local provider retry/deadline behavior.
4. Report the selected policy in the existing audit transport-policy field; retain
   unknown history and unproven compatibility. Document the exact environment values.
5. Apply local-only to development application process environments, reusing shared
   YAML configuration where feasible. Do not edit production/base/no-GPU Compose,
   model services, models, schemas, requests, or stored vectors.
6. Test adjacent chunk search's existing lexical fallback through the real facade
   and prove permanent malformed policy does not retry chunk ingestion. No change
   to query authorization or candidate acquisition is planned.
7. Run focused embedding/adjacent suites and lint. Record unavailable Linux/Django
   integration separately. Write bounded coordinator smoke/rollback instructions,
   commit changes, and produce private reports with exact commands and results.

## Limits

No quality claim, historical compatibility claim, model switch, re-embedding,
activation-gate bypass, provider retry increase, server contact, or deployment.
