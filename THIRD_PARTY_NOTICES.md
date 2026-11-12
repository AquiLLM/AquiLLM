# Third-party notices

These components retain their own terms; inclusion does not relicense them under
AquiLLM's project license. This is a maintained inventory of identified components,
not an exhaustive certification of every transitive dependency or container image.
See LICENSING.md for unresolved project compatibility and release requirements.

## Application dependencies

- **PyMuPDF / MuPDF**: Artifex and upstream contributors; open-source AGPLv3
  distribution used here. Source: https://github.com/pymupdf/PyMuPDF . Exact
  package source URL and hash are in uv.lock under pymupdf. Licensing:
  https://pypi.org/project/pymupdf/1.27.2.2/ . Retain the package COPYING files
  and supply the required corresponding source, including the linked MuPDF code.
- **EbookLib**: Aleksandar Erkalovic and contributors; AGPLv3-or-later.
  Source and LICENSE.txt: https://github.com/aerkalov/ebooklib . Exact package
  source URL and hash are in uv.lock under ebooklib (0.20 in the reviewed lock).
- **Neo4j Python driver**, **Google Gen AI SDK**, and other Apache components:
  retain their Apache-2.0 licenses and applicable notices. Sources:
  https://github.com/neo4j/neo4j-python-driver and https://github.com/googleapis/python-genai .
- Remaining Python packages: versions/source archives are identified in uv.lock
  and deploy/docker/web/runtime-constraints.txt; preserve each distribution's
  license files. Lockfiles and source links alone do not fulfill all source duties.

## Browser distribution

React, React DOM, Cytoscape, Lucide, KaTeX, PDF.js and other bundled packages keep
their upstream terms. The build collects full license/NOTICE files from the
packages actually included in the bundle, plus the separately copied PDF.js
worker and KaTeX assets. The result is served at
`/static/js/dist/THIRD_PARTY_LICENSES.txt`, linked from `/legal/`, and must travel
with the browser build. A missing package license text fails the build.
Additional CDN-loaded libraries retain their notices in the upstream resources.

## Fonts

- **Lexend**: Copyright 2019 The Lexend Project Authors (embedded notice).
  Full upstream OFL and additional attribution are in
  aquillm/aquillm/static/fonts/OFL-Lexend.txt. Source: https://github.com/googlefonts/lexend .
- **OpenDyslexic3**: 2012 Abelardo Gonzalez, SIL Open Font License 1.1.
  Full terms and the embedded attribution are in
  aquillm/aquillm/static/fonts/OFL-OpenDyslexic3.txt. Author's release:
  https://antijingoist.itch.io/opendyslexic .
- **Comic Sans**: a local system-font preference only; no font file is included.
  Users without it receive a generic cursive fallback.
- The externally hosted Latin Modern stylesheet/fonts retain their upstream
  notices: https://github.com/lalten/lmweb .

## Vendored inference code and templates

- The Qwen fixed chat template is attributed in
  deploy/docker/vllm/chat_templates/QWEN_FIXED_CHAT_TEMPLATE.md. The accompanying
  Apache-2.0 license is in that directory's LICENSE-Apache-2.0.txt.
- Genesis-derived H100 kernels retain their Apache SPDX headers, full license
  and Genesis NOTICE in deploy/vllm_plugins/h100_kernels/src/aquillm_vllm_h100/kernels/.
  The package includes these documents. Source provenance is recorded beside
  the implementations. Other downloaded inference packages keep their own terms.
- Model weights are separate from AquiLLM source. Nemotron's OpenMDW terms are
  documented in deploy/NEMOTRON_ASR.md; retain each deployed model's license,
  model card, required notices and any redistribution conditions.

## Data and branding

- LibriSpeech audio fixtures: attribution, exact source hashes and CC-BY-4.0
  license are retained in tests/fixtures/audio/README.md and LICENSE-CC-BY-4.0.txt.
- The PDF upload fixture is synthetic project test content; its generator and
  provenance are in react/test-data/README.md. It replaces a textbook chapter
  whose redistribution permissions were not documented in this repository.
- Thank you to arXiv for use of its open access interoperability. This service
  was not reviewed or approved by, nor does it necessarily express or reflect
  the policies or opinions of, arXiv. The arXiv name and logo are registered
  trademarks; AquiLLM's license grants no rights to them. Follow
  https://info.arxiv.org/brand/brand-guidelines.html when reusing it.

Separately running services (including MinIO, Redis, Memgraph and observability
images) have their own licenses. Their network/source, modification, trademark
and redistribution conditions need review for the exact deployed image versions;
this notice neither relicenses them nor certifies their compliance.
