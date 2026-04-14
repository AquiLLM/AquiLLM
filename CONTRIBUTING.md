# Contributing to AquiLLM

## Versioning

AquiLLM follows [Semantic Versioning 2.0.0](https://semver.org/). The full policy is documented in [`docs/specs/2026-03-29-semantic-versioning-design.md`](docs/specs/2026-03-29-semantic-versioning-design.md).

### Version sources

| Artifact | Location |
|----------|----------|
| Canonical version | `aquillm/aquillm/version.py` (`VERSION`) |
| Python import | `aquillm.__version__` |
| Frontend | `react/package.json` `version` field |
| Git tag | `vMAJOR.MINOR.PATCH` |

All locations must agree at release boundaries.

### When to bump

| Bump | When |
|------|------|
| **MAJOR** | Incompatible changes to the public surface: HTTP API, env vars, database migrations requiring manual intervention, Celery task contracts, container contracts. |
| **MINOR** | New functionality added in a backward-compatible manner (additive endpoints, new optional env vars, new JSON response fields). |
| **PATCH** | Backward-compatible bug fixes, security fixes that don't change the public contract. |

When in doubt, prefer MAJOR or document a deprecation window.

### Release checklist

1. Update `VERSION` in `aquillm/aquillm/version.py`.
2. Update `version` in `react/package.json` to match.
3. Add a changelog entry under `docs/changelog/` (see [`docs/changelog/README.md`](docs/changelog/README.md)).
4. Commit: `release: vX.Y.Z`.
5. Tag: `git tag vX.Y.Z`.
6. Push: `git push origin main --tags`.

### Changelog

Each release should have a changelog entry under `docs/changelog/`. Follow the existing branch-scoped or release-scoped conventions documented in [`docs/changelog/README.md`](docs/changelog/README.md).
