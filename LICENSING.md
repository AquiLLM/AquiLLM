# Licensing status and release requirements

## Current status

The root [LICENSE](LICENSE) contains GNU GPL version 2. It is unchanged by this
cleanup. Previous MIT and ISC package labels were inconsistent with that file;
metadata now points to the existing text instead of asserting another grant.
This does not revoke any license validly granted by a copyright holder in an
earlier release. Confirm intended component-specific grants with their owners.

**GPLv3 is proposed, not effective. Contributor permissions are still pending.**
This branch is not a declaration that the application is fully license-compliant.
The GPLv2/AGPLv3 and GPLv2/Apache-2.0 combination issues remain release blockers.
See [the permission and migration checklist](docs/licensing/gpl3-migration.md).

Third-party code, fonts, model weights, datasets, logos, and services retain
their own licenses. The root license never replaces their terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Recommended destination

After obtaining and recording the necessary permissions, license AquiLLM's own
code under **GPL-3.0-only**. GPLv3 permits combination with Apache-2.0 and, under
section 13, AGPLv3 components. PyMuPDF and EbookLib remain under their AGPL terms;
the network-source requirements apply to the covered combined application.
Avoiding those obligations would require replacing or separately licensing the
relevant components. A change of project label does not relicense dependencies.

## Source access for each hosted deployment

Set `AQUILLM_SOURCE_URL` to a publicly accessible, no-charge source download or
source landing page for the **actual deployed version**, including operator
modifications. `/legal/` presents this link without requiring an account. A
public GitHub commit or release can provide the application-source portion.
Do not substitute a moving development branch, private repository, or upstream
commit that omits local modifications.

The linked materials must include the complete corresponding source required
by the applicable licenses: preferred editable source, relevant dependency
source (including modifications), interface definitions, and build/install/run
scripts. Lockfiles identify versions but are not a substitute for dependency
source. Provide or arrange enduring access to those sources; a list of homepage
links alone is not a complete source offer. Exclude credentials, user uploads,
database contents and other private operational data from published artifacts.

Run `python manage.py check --deploy --tag licensing` inside the web container.
This verifies that a usable HTTP(S) source URL is configured; it cannot verify
the URL's contents, authorization, completeness, or matching deployed revision.
Verify those manually before exposing a release. An unset/invalid value is
reported rather than silently advertising upstream as the deployed source.

For forks and local patches, the operator must update the source URL. The
Compose web services already read `.env`. Add this setting to that file and
recreate the web container. Do not include access tokens in the URL.

## Redistribution and notices

- Keep the project license and third-party notices with source distributions.
- `npm run build` emits `THIRD_PARTY_LICENSES.txt` alongside browser assets,
  including the separately copied PDF.js worker and KaTeX fonts. Keep the
  whole output directory together. The production image already copies it.
- Keep the font OFL files with the TTFs. Comic Sans is selected only from the
  user's installed fonts, with a generic fallback; no Microsoft font is shipped.
- Preserve licenses and NOTICE files in Python packages, container layers,
  patched upstream code, and separately distributed plugins.
- Binary/container releases require their own complete corresponding-source
  arrangements for covered components, including bundled system packages.
  This application-source cleanup does not certify all container licenses.
- Removing files from a new revision does not erase earlier distributions.
  Review historical release assets and any required remediation separately;
  do not rewrite shared history as part of this cleanup.

## Primary references

- [GPLv3 section 13](https://www.gnu.org/licenses/gpl-3.0.html#section13)
- [AGPLv3 section 13](https://www.gnu.org/licenses/agpl-3.0.html#section13)
- [Apache/GPL compatibility](https://www.apache.org/licenses/GPL-compatibility)
- [GNU licensing FAQ](https://www.gnu.org/licenses/gpl-faq.html)
- [Microsoft font redistribution](https://learn.microsoft.com/en-us/typography/fonts/font-faq)
