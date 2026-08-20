# AquiLLM User Docs

This is the **AquiLLM documentation site**, built with [Sphinx](https://www.sphinx-doc.org/) and the [sphinx-rtd-theme](https://sphinx-rtd-theme.readthedocs.io/).

## Features

- **Multi-page navigation** with collapsible sidebar
- **Full-text search** across all documentation
- **Cross-references** between pages (`:doc:`, `:ref:`, `:term:`)
- **Glossary** of key terms
- **Responsive design** (desktop, tablet, mobile)
- **No ads, no tracking** — fully self-hosted static HTML
- **Multiple output formats**: HTML, ePub, single-page HTML

## Project Structure

```
docs/
├── Makefile                  # Build commands
├── requirements.txt          # Python dependencies
└── source/
    ├── conf.py             # Sphinx configuration
    ├── index.rst           # Documentation homepage
    ├── glossary.rst        # Terminology glossary
    ├── _static/            # Logos, images, custom CSS
    │   ├── css/
    │   │   └── aquillm_custom.css
    │   ├── images/
    │   └── AquiLLMLogo.png
    ├── getting-started/
    │   └── account.rst
    ├── collections/
    │   ├── overview.rst
    │   ├── creating.rst
    │   ├── tracking.rst
    │   ├── using.rst
    │   └── sharing.rst
    ├── integrations/
    │   └── zotero.rst
    ├── skills/
    │   ├── overview.rst
    │   ├── markdown.rst
    │   └── python.rst
    └── more/
        └── feedback.rst
```

## Building

From the **AquiLLM repo root**, work inside `user-docs/`:

### Prerequisites

- Python 3.12+
- Install dependencies:

```bash
cd user-docs
python -m pip install -r docs/requirements.txt
```

### Build Commands

```bash
cd user-docs

# Build HTML (autodoc reads skills modules from ../aquillm/)
python -m sphinx -b html docs/source docs/build/html
```

### Viewing Locally

```bash
python -m http.server 8765 --directory docs/build/html
```

Open http://localhost:8765/

## Deploying with GitHub Actions

Workflow: [`.github/workflows/user-docs.yml`](../.github/workflows/user-docs.yml)

Rebuilds when `user-docs/`, `aquillm/lib/skills/`, or `docs/skills/runtime/` change.

### One-time setup on GitHub

1. Open **Settings → Pages**.
2. Set **Source** to **GitHub Actions**.
3. Push to `main`/`master` (or run the workflow manually).

Site URL: `https://<owner>.github.io/<repo>/` (e.g. `https://aquillm.github.io/AquiLLM/`)

## Engineering docs

Internal roadmap/specs live in [`docs/`](../docs/) at the repo root — separate from this user-facing Sphinx site.
