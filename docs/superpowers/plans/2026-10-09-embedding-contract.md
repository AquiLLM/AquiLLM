# Embedding contract bounded fix plan

Reproduction on base15a8abaf: fit_embedding_dims accepts empty vectors padded to 1024, all-zero vectors, and NaN values; short nonzero vectors are padded as legacy behavior. The facade drops the validated input_type for local requests and catches every local error to cross provider. Chunk embedding retries every exception without a stop condition. Runtime identity and stored-vector provenance remain unverified.

1. Add failing behavioral tests through the real local provider/facade boundaries using a fake external embedding transport: invalid vectors, malformed batch counts/indices, role propagation with unchanged wire payload, fallback on transport failure versus no fallback on contract failure, and chunk permanent-error retry exclusion.
2. Add EmbeddingContractError and validate numeric, finite, nonempty, nonzero vectors before and after existing dimension fitting. Preserve padding/truncation and current raw request formatting.
3. Validate provider response count/index binding, restore input order, and propagate input_type internally while declaring local role handling unsupported. No request-format experiment protocol is introduced without endpoint evidence. Existing strict KG request formatting remains unchanged.
4. Update the facade to propagate contract errors rather than crossing providers; preserve current transport-failure fallback explicitly as a compatibility risk. Prevent chunk retry/backoff on permanent contract errors. No settings/model/migration changes.
5. Add a redacted, bounded read-only audit CLI separating declared configuration, observed synthetic vector behavior, and unknown historical provenance. Default is configuration only; an explicit probe performs at most two synthetic local calls with a fixed timeout and no transport retries.
6. Run pure embedding regression tests and lint. Ask coordinator to run adjacent chunk and strict KG facade tests in isolated Linux. Record exact red/green results and limitations. Commit owned files only.

Production APIs: EmbeddingContractError(ValueError); fit_embedding_dims retains signature; get_embedding_via_local_openai(query, input_type=search_query), get_embeddings_via_local_openai(queries, input_type=search_query); public facade signatures unchanged. Audit is additive and never proves historical model, precision, template, or compatibility from vector shape.
