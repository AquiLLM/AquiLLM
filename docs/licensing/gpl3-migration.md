# Proposed GPLv3 migration — approvals pending

This is a proposal, not a license grant. Keep the PR in draft until the necessary
copyright-holder permissions are recorded and the remaining release checks pass.
The maintainer confirmed on 2026-11-12 that some contributors still need to approve.
The root LICENSE must not be replaced on the assumption that repository access,
commit authorship, silence, or employment automatically confers relicensing rights.

## Permission evidence

Review all code and assets that remain in the intended release, including
imported history, copied source, coauthors, and employer/university ownership.
The following is only an initial author inventory from
`git shortlog -sn 2439ca866fa353fc51e9b43b8e8ecf9bc78e1702`.
Names can be aliases; authors are not necessarily the copyright holders.

| Recorded author | Ownership/permission evidence |
| --- | --- |
| PhantomG27249 | Pending maintainer verification |
| Chandler Campbell | Pending maintainer verification |
| Bernie Boscoe | Pending maintainer verification |
| Jacob489 | Pending maintainer verification |
| jacksgodsey | Pending maintainer verification |
| Regonir | Pending maintainer verification |
| Srinath Saikrishnan | Pending maintainer verification |
| Tuan Do | Pending maintainer verification |
| bboscoe | Pending maintainer verification |
| github-actions[bot] | Identify the origin/owner of generated or imported changes |

Record the consenting rights holder, covered contributions, date, and a durable
reference to their written permission. Keep personal/legal records in an
appropriate location; a public PR can link to an approved summary. Do not mark
a row complete merely because its author approves the technical code changes.

Suggested permission request for a rights holder to review (not sent automatically):

> I confirm that I hold, or am authorized to license, the copyright in the
> AquiLLM contributions identified here: [specific contributions]. I permit
> those contributions to be distributed as part of AquiLLM under the GNU
> General Public License version 3 only (GPL-3.0-only). This permission does
> not change third-party licenses or revoke previously granted licenses.

## Changes to make once permission is established

- [ ] Resolve the owner/permission inventory and separately licensed source.
- [ ] Replace root LICENSE with the unmodified official GPLv3 text.
- [ ] Add an explicit project GPL-3.0-only grant with accurate copyright credits.
- [ ] Update README, Python metadata, root/frontend package metadata and locks;
      update the frontend license copy and any separately distributed plugins.
- [ ] Keep AGPL, Apache, OFL, dataset and other third-party terms and notices.
- [ ] Rebuild browser assets and inspect their generated notices.
- [ ] Publish and verify corresponding source for the exact deployed build,
      including required dependency source, modifications and build materials.
- [ ] Set and verify AQUILLM_SOURCE_URL as an anonymous end user.
- [ ] Review historical releases and all redistributed container components.
- [ ] Run the licensing tests and deployment check, then remove draft status
      only after permission and source-distribution evidence are reviewed.

GPLv2 is not the recommended destination for the current dependency set:
Apache-2.0 and AGPLv3 integrations cannot simply be declared GPLv2. A GPLv2-only
route would require dependency replacements or adequate separate permissions.
