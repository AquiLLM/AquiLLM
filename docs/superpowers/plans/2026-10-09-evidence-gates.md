# Evidence gates: bounded integration fixes

Date: 2026-10-09. Base: 15a8abaf7fbb8fdb1b850b094ebedde6c714b861.

## Observed path and hypothesis

The quality and operational CLIs both call `evidence_quality_live.run_live`,
which explicitly owns a `PairCapabilityController`. Ordinary management commands
do not initialize serving-worker capability. The controller verifies locally
attested identity, `/tokenize` output and three bounded `/score` requests.

The normal warm `/score` response is currently accepted on HTTP status and token
usage alone. It never checks a score. The subsequent window HTTP scorer requires
a finite parsed score. Thus token-count success can incorrectly establish
capability even when scoring is unusable. Reproduce this discrepancy before
editing behavior; no runtime protocol assumption will be relaxed without evidence.

## Exact fix/test sequence

1. Add regression inputs for missing, nonnumeric and nonfinite scores to the
   existing warm protocol test. Drive the actual verifier with bounded HTTP
   boundary doubles and verify the registry remains unknown on rejected replies.
2. Run those tests red. Require the existing production score parser to accept a
   finite result for both normal probes before capability can be published.
   Preserve all token-identity, overflow-rejection, expiry and pair-budget checks.
3. Audit and reproduce malformed window response behavior using the real scorer;
   if it escapes instead of returning unavailable, add a fail-closed regression
   and the minimum shape guard. Do not change transport retries or allowances.
4. Run capability/window/lifecycle and pure quality/operational/review/replay
   suites. Keep the frozen quality and operational files byte-identical.
5. Execute fixture CLIs and immutable rescoring; verify exit 2 for blocked
   activation. Prepare an isolated development live recipe and evidence index,
   with verified runtime identities and independent human labels explicitly
   pending. Ask the coordinator for bounded runtime diagnostics and Linux
   integration tests; do not contact shared services from this checkout.

## Constraints and acceptance

No embedding, graph, schema, frozen corpus or human label changes. No new enabled
modes. Preserve extractor/direct/extended/overall graph budgets 3000/4500/4500/5000
ms and all existing evidence limits. Pure tests use an ignored pytest config
derived from the repository config without `DJANGO_SETTINGS_MODULE`; Django and
service integration remain coordinator-owned. A passing protocol regression is
mechanical evidence only and never constitutes live quality or human approval.
