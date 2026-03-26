# AquiLLM


![AquiLLM Logo](aquillm/aquillm/static/images/aquila.svg)

[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.1-green.svg)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**AquiLLM is an open-source RAG (Retrieval-Augmented Generation) application designed specifically for researchers.** It helps you manage, search, and interact with your research documents using AI, streamlining your literature review and knowledge discovery process. Upload various document formats, organize them into collections, and chat with an AI that understands your library's content.

More info can be found at [https://aquillm.org](https://aquillm.org). See also our paper ["AquiLLM: a RAG Tool for Capturing Tacit Knowledge in Research Groups"](https://arxiv.org/abs/2508.05648) by Chandler Campbell, Bernie Boscoe, and Tuan Do

<!-- ![AquiLLM Screenshot](path/to/screenshot.gif) -->

## Key Features

*   **Versatile Document Ingestion**: Upload PDFs, fetch arXiv papers by ID, import VTT transcripts, scrape webpages, process handwritten notes (with OCR), and ingest mixed-format file batches through one unified upload flow.
*   **Intelligent Organization**: Group documents into logical `Collections` for focused research projects.
*   **AI-Powered Chat**: Engage in context-aware conversations with your documents, ask follow-up questions, and get answers with source references.

### Unified Upload Format Support

The unified upload endpoint supports:

* Documents: `pdf`, `doc`, `docx`, `odt`, `rtf`, `txt`, `md`, `html`, `htm`, `epub`
* Spreadsheets/tabular: `csv`, `tsv`, `xls`, `xlsx`, `ods`
* Presentations: `ppt`, `pptx`, `odp`
* Structured: `json`, `jsonl`, `xml`, `yaml`, `yml`
* Captions/transcripts: `vtt`, `srt`
* Images (OCR): `png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`, `webp`, `heic`, `heif`
* Audio/video transcription: `mp3`, `wav`, `m4a`, `aac`, `flac`, `ogg`, `opus`, `mp4`, `mov`, `m4v`, `webm`, `mkv`, `avi`, `mpeg`, `mpg`
* Archives: `zip` (supported files inside are expanded and ingested)

## Tech Stack

*   **Backend**: Python, Django
*   **Frontend**: React
*   **Database**: PostgreSQL
*   **Vector Store**: pgvector (PostgreSQL extension)
*   **LLM Integration**: Local LLMs, Claude, OpenAI, Gemini as desired
*   **Asynchronous Tasks**: Celery, Redis, Django Channels

*   **Optional RAG / cost controls**: Django cache–backed retrieval TTL caches (`RAG_CACHE_*`), cross-provider prompt preflight trimming (`TOKEN_EFFICIENCY_*`), optional LM-Lingua2 compression (`LM_LINGUA2_*`), and optional vLLM LMCache wiring (`LMCACHE_*`). See `.env.example` and `docs/superpowers/plans/2026-03-23-caching-rag-token-efficiency-rollout-notes.md` for rollout and rollback.

### RAG retrieval defaults and benchmarking

Default `VECTOR_TOP_K`, `TRIGRAM_TOP_K`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` target a balance of latency and recall for typical research corpora. Tune `RAG_CANDIDATE_MULTIPLIER`, `RAG_*_MIN_LIMIT`, and `RAG_TRIGRAM_SIMILARITY_MIN` when you need more aggressive candidate fan-out or stricter trigram filtering. To compare old versus new defaults without code changes, snapshot your current `.env`, restore prior values (for example higher `VECTOR_TOP_K` / `TRIGRAM_TOP_K`), run the same fixed set of chat queries, and compare p95 end-to-end chat latency plus qualitative answer quality. Enable `RAG_CACHE_ENABLED=1` only after you have a shared cache backend so measurements are not dominated by cold embed calls.

*   **Authentication**: django-allauth
*   **Containerization**: Docker, Docker Compose

## Module layout and boundaries

* **`aquillm/apps/*`**: Domain Django apps (models, views, consumers, and Celery tasks owned per app). Prefer importing concrete models and services from `apps.<domain>` rather than the `aquillm.models` compatibility module in new application code.
* **`aquillm/lib/*`**: Shared, provider-style helpers (for example LLM adapters and tool types). Keep this tree free of direct `apps.*` imports; pass Django or ORM behavior in from `apps` callers.
* **`aquillm/lib/tools/*`**: Reusable tool logic without Django (`search` chunk formatting, `documents` ID parsing and payloads, `astronomy` FITS/array operations, `debug` test tools). Chat-specific binding (collections, `TextChunk`, `ConversationFile`, user permissions) lives in **`aquillm/apps/chat/services/tool_wiring/`** (package: `documents.py`, `astronomy.py`, `__init__.py`). New tool code should stay import-clean under `lib/tools/` and wire through that package.
* **React `src/features/*`**: Domain UI lives under `react/src/features/<area>/` (for example `features/chat` for the WebSocket chat shell and composer, `features/collections` for collection view, `features/documents` for the filesystem table, `features/platform_admin` for user management, `features/ingestion` for ingest rows). `react/src/components/*.tsx` may re-export shims for Django template mount points; prefer importing from `features/` in new code.
* **`aquillm/aquillm/models.py`**: Legacy barrel that re-exports models and a few helpers for older call sites. Integration tests under `aquillm/tests/integration/test_architecture_import_boundaries.py` and `scripts/check_import_boundaries.py` discourage new `from aquillm.models import` usage under `apps/` and `lib/`.
* **WebSockets**: `aquillm/asgi.py` wires `apps.chat.routing` and `apps.ingestion.routing` into the Channels URL router (legacy `chat.routing` / `ingest.routing` remain thin re-exports).
* **Structure checks** (also run in CI): `python scripts/check_file_lengths.py` and `python scripts/check_import_boundaries.py`.

## Quick start (development and local use):

This assumes you have Docker and Docker Compose installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AquiLLM/AquiLLM.git 
    cd AquiLLM
    ```
2.  **Copy the environment template:**
    ```bash
    cp .env.example .env
    ```
3.  **Edit the .env file with your specific configuration:**
    - Database settings: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_NAME, POSTGRES_HOST
    - At least one LLM API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)
    - Set LLM_CHOICE to your preferred provider (`CLAUDE`, `OPENAI`, `GEMINI`, `GEMMA3`, `LLAMA3.2`, `GPT-OSS`, or `QWEN3_30B`). To switch models after initial setup, update LLM_CHOICE in `.env` and do a full restart: `docker compose down && docker compose up` — a simple restart may not pick up the change.
    - If using local vLLM-backed choices (`GEMMA3`, `LLAMA3.2`, `GPT-OSS`, `QWEN3_30B`), use `--profile vllm` when starting compose. This profile launches `vllm` (chat), `vllm_ocr` (OCR), `vllm_transcribe` (audio/video transcription), `vllm_embed` (embeddings), and `vllm_rerank` (reranker).
    - For image OCR through local vLLM, set `APP_OCR_PROVIDER=qwen` and point `APP_OCR_QWEN_BASE_URL` to `http://vllm_ocr:8000/v1`.
    - For audio/video transcription through local vLLM, set `INGEST_TRANSCRIBE_PROVIDER=openai` and `INGEST_TRANSCRIBE_OPENAI_BASE_URL=http://vllm_transcribe:8000/v1`.
    - GGUF note: set model as `repo:filename.gguf` or `repo:selector` (for example `repo:i1-Q4_K_M`). Startup resolves the best matching GGUF file in the repo, downloads it, and launches vLLM with the local file path.
    - For embedding/reranker models like `Qwen/Qwen3-Embedding-4B` and `Qwen/Qwen3-Reranker-4B`, set `MEM0_EMBED_VLLM_TRUST_REMOTE_CODE=1` and `APP_RERANK_VLLM_TRUST_REMOTE_CODE=1`.
    - Optional memory backend:
      - `MEMORY_BACKEND=local` (default): AquiLLM pgvector memory tables
      - `MEMORY_BACKEND=mem0`: Mem0 episodic memory retrieval/write with local fallback
      - OSS setup:
        - `MEM0_QDRANT_HOST=qdrant`
        - `MEM0_LLM_BASE_URL=http://host.docker.internal:8000/v1`
        - `MEM0_EMBED_BASE_URL=http://host.docker.internal:8002/v1`
        - Leave `MEM0_EMBED_DIMS` blank unless you need to force a known dimension

4.  **Build and run using Docker Compose (development):**
    ```bash
    # Default: use hosted LLMs configured in .env (e.g., OpenAI, Claude, Gemini)
    docker compose up -d

    # Local vLLM-backed startup (serial health-gated launch:
    # vllm -> vllm_ocr -> vllm_transcribe -> vllm_embed -> vllm_rerank -> web/worker)
    bash deploy/scripts/start_dev.sh

    # Optional edge/TLS dev startup with nginx:
    # - first cert issue/renewal run: USE_EDGE=1 RUN_CERTBOT=1 bash deploy/scripts/start_dev.sh
    # - normal restarts after cert exists: USE_EDGE=1 bash deploy/scripts/start_dev.sh
    ```

5. **Add a superuser:**
   ```bash
   docker compose exec web ./manage.py addsuperuser
   ```

6.  **Access the application (development):**

    Open your browser to `http://localhost:8080`, then sign in with the superuser account you just created.

7.  **Common dev commands:**

    ```bash
    # View logs for all services
    docker compose logs -f

    # View status of services
    docker compose ps
    ```

8.  **Stop the application (development):**
    ```bash
    docker compose down
    ```

## Updating (development)

Pull the latest changes and rebuild. Migrations run automatically on startup.

```bash
git pull origin main
docker compose down
docker compose up --build -d

# Or, if you are using local vLLM-backed models:
bash deploy/scripts/start_dev.sh
```

## Updating (production)

```bash
git pull origin main
docker compose -f deploy/compose/production.yml down
docker compose -f deploy/compose/production.yml up --build -d

# Or, if you are using local vLLM-backed models in production:
docker compose -f deploy/compose/production.yml --profile vllm up --build -d
```

If vLLM is running but individual services need to be force-recreated (for example, after changing model configuration):

```bash
docker compose -f deploy/compose/production.yml --profile vllm up -d --force-recreate vllm vllm_ocr vllm_transcribe vllm_embed vllm_rerank

# Recreate only OCR + transcription model services:
docker compose -f deploy/compose/production.yml --profile vllm up -d --force-recreate vllm_ocr vllm_transcribe
```

## Small-scale deployment:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AquiLLM/AquiLLM.git 
    cd AquiLLM
    ```
2.  **Copy the environment template:**
    ```bash
    cp .env.example .env
    ```
3.  **Edit the .env file with your specific configuration:**
    - Database settings: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_NAME, POSTGRES_HOST
    - At least one LLM API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)
    - Set LLM_CHOICE to your preferred provider (`CLAUDE`, `OPENAI`, `GEMINI`, `GEMMA3`, `LLAMA3.2`, `GPT-OSS`, or `QWEN3_30B`). To switch models after initial setup, update LLM_CHOICE in `.env` and do a full restart: `docker compose down && docker compose -f deploy/compose/production.yml up` — a simple restart may not pick up the change.
    - If using local vLLM-backed choices (`GEMMA3`, `LLAMA3.2`, `GPT-OSS`, `QWEN3_30B`), use `--profile vllm` when starting compose. This profile launches `vllm` (chat), `vllm_ocr` (OCR), `vllm_transcribe` (audio/video transcription), `vllm_embed` (embeddings), and `vllm_rerank` (reranker).
    - GGUF note: set model as `repo:filename.gguf` or `repo:selector` (for example `repo:i1-Q4_K_M`). Startup resolves the best matching GGUF file in the repo, downloads it, and launches vLLM with the local file path.
    - For embedding/reranker models like `Qwen/Qwen3-Embedding-4B` and `Qwen/Qwen3-Reranker-4B`, set `MEM0_EMBED_VLLM_TRUST_REMOTE_CODE=1` and `APP_RERANK_VLLM_TRUST_REMOTE_CODE=1`.
    - Optional memory backend:
      - `MEMORY_BACKEND=local` (default): AquiLLM pgvector memory tables
      - `MEMORY_BACKEND=mem0`: Mem0 episodic memory retrieval/write with local fallback
      - OSS setup:
        - `MEM0_QDRANT_HOST=qdrant`
        - `MEM0_LLM_BASE_URL=http://host.docker.internal:8000/v1`
        - `MEM0_EMBED_BASE_URL=http://host.docker.internal:8002/v1`
        - Leave `MEM0_EMBED_DIMS` blank unless you need to force a known dimension
    - Optional: Google OAuth credentials (GOOGLE_OAUTH2_CLIENT_ID, GOOGLE_OAUTH2_CLIENT_SECRET)
    - Optional: Email access permissions (ALLOWED_EMAIL_DOMAINS, ALLOWED_EMAIL_ADDRESSES). Required if OAuth is to be used.
    - Set HOST_NAME for your domain or use 'localhost' for development

4.  **Build and run using Docker Compose (HTTPS deployment):**
    ```bash
    # First-time cert issue/renewal (must have port 80 free):
    docker compose -f deploy/compose/production.yml stop nginx
    docker compose -f deploy/compose/production.yml --profile certbot up --build get_certs

    # Default: hosted LLMs (OpenAI, Claude, Gemini, etc.)
    docker compose -f deploy/compose/production.yml up -d 

    # Optional: with local vLLM-backed models
    docker compose -f deploy/compose/production.yml --profile vllm up -d
    ```
    `get_certs` now runs only on the explicit `certbot` profile so regular stack restarts do not collide on port 80.

4. **Add a superuser for administration:**
   ```bash
   docker compose -f deploy/compose/production.yml exec web ./manage.py addsuperuser
   ```

## Configuring and deploying Mem0 with vLLM

AquiLLM can use [Mem0](https://github.com/mem0ai/mem0) for **episodic memory**: storing and retrieving past conversation turns so the assistant can refer to them in new chats. You can run Mem0 entirely locally by backing it with vLLM for the LLM and embedding models, and Qdrant (included in the stack) for the vector store.

AquiLLM integrates Mem0 in OSS SDK mode.

This uses AquiLLM's built-in Mem0 SDK integration: no extra Mem0 server, and no `MEM0_API_KEY` required.

**1. Enable Mem0 and use the SDK**

In `.env`:

```bash
MEMORY_BACKEND=mem0
MEM0_QDRANT_HOST=qdrant
MEM0_QDRANT_PORT=6333
MEM0_COLLECTION_NAME=mem0_1024_v1
```

Leave `MEM0_EMBED_DIMS` blank unless you need to force a specific embedding dimension.

**2. Point Mem0 at vLLM for LLM and embeddings**

Use OpenAI-compatible endpoints (vLLM exposes these):

```bash
MEM0_LLM_PROVIDER=openai
MEM0_EMBED_PROVIDER=openai
MEM0_LLM_API_KEY=EMPTY
MEM0_EMBED_API_KEY=EMPTY
```

- **When vLLM runs in the same Docker Compose** (e.g. `docker compose --profile vllm up`):
  - `MEM0_LLM_BASE_URL=http://vllm:8000/v1`
  - `MEM0_EMBED_BASE_URL=http://vllm_embed:8000/v1`
- **When vLLM runs on the host** (e.g. bare metal or another compose):
  - `MEM0_LLM_BASE_URL=http://host.docker.internal:8000/v1`
  - `MEM0_EMBED_BASE_URL=http://host.docker.internal:8002/v1`

Set the model names to match what your vLLM instances serve:

```bash
MEM0_LLM_MODEL=your-chat-model-name
MEM0_EMBED_MODEL=Qwen/Qwen3-Embedding-4B
```

For embedding models that need it (e.g. Qwen3-Embedding-4B), set:

```bash
MEM0_EMBED_VLLM_TRUST_REMOTE_CODE=1
```

**3. Start the stack with vLLM**

Development:

```bash
docker compose --profile vllm up -d
```

Production:

```bash
docker compose -f deploy/compose/production.yml --profile vllm up -d
```

The `vllm` profile starts dedicated model services: chat (`vllm` on 8000), OCR (`vllm_ocr` on 8004), transcription (`vllm_transcribe` on 8005), embeddings (`vllm_embed` on 8002), and reranker (`vllm_rerank` on 8003). Mem0 uses chat + embed services; Qdrant is already part of the stack.

**4. Optional: dual-write to local DB**

To keep a copy of episodic memories in AquiLLM's local pgvector tables as well:

```bash
MEM0_DUAL_WRITE_LOCAL=1
```

**5. Relaunch Mem0 (OSS mode)**

In OSS mode, relaunching Mem0 means recreating AquiLLM services that host/use it (`qdrant`, `web`, `worker`).

```bash
./deploy/scripts/relaunch_mem0_oss.sh
```

Optional env vars:

- `AQUILLM_COMPOSE_FILE` - e.g. `deploy/compose/development.yml` or `deploy/compose/production.yml`
- `RELAUNCH_MEM0_MODELS=1` - also recreate `vllm`, `vllm_ocr`, `vllm_transcribe`, `vllm_embed`, and `vllm_rerank`

`./deploy/scripts/start_mem0_local.sh` now forwards to this OSS relaunch flow for backward compatibility.

For standard development launches with local vLLM services in serial order, use:

```bash
bash deploy/scripts/start_dev.sh
```

### Summary: key env vars for Mem0 + vLLM

| Variable | Purpose |
|----------|--------|
| `MEMORY_BACKEND=mem0` | Use Mem0 for episodic memory. |
| `MEM0_LLM_BASE_URL` | OpenAI-compatible URL for chat (e.g. `http://vllm:8000/v1` or `http://host.docker.internal:8000/v1`). |
| `MEM0_EMBED_BASE_URL` | OpenAI-compatible URL for embeddings (e.g. `http://vllm_embed:8000/v1` or `http://host.docker.internal:8002/v1`). |
| `MEM0_LLM_MODEL` / `MEM0_EMBED_MODEL` | Model names as served by vLLM. |
| `MEM0_QDRANT_HOST=qdrant` | Qdrant service name (same stack). |
| `MEM0_EMBED_VLLM_TRUST_REMOTE_CODE=1` | Required for some embedding models (e.g. Qwen3-Embedding-4B). |

## Using AquiLLM

### Uploading Documents

1. **Add a Collection First**
   - Click on "Collections" in the navigation menu
   - Click the "New Collection" button
   - Enter a name for your collection

2. **Upload Documents**
   - Go to the collection you created
   - Choose the document type you want to upload using the buttons, the buttons are as follows, in order from left to right:
     - PDF: Upload PDF files
     - ArXiv Paper: Enter an arXiv ID to import 
     - VTT File: Upload VTT transcript files
     - Webpage: Enter the URL of a site 
     - Handwritten Notes: Upload images of handwritten notes, select the Convert to LaTeX box if they contain formulas
     - All documents will appear in your collection, if they don't show up automatically, refresh the page
     
3. **View Your Documents**
   - If you leave your collection page, do the following
   - Select Collections button from the sidebar and choose which collection you want to view
   - Click on any document to view its contents

### Chatting With Your Documents

1. **Start a New Conversation**
   - From the sidebar, click "New Conversation"
   - Select which collections to include in your search context

2. **Using the Chat**
   - Type your questions about the documents in natural language
   - The AI will search your documents and provide answers with references
   - You can follow up with additional questions
   - The AI may quote specific parts of your documents as references

3. **Managing Conversations**
   - All conversations are saved automatically
   - Access past conversations from the "Your Conversations" menu in the sidebar
   - Each conversation maintains its collection context

### Exporting chat feedback (superusers)

Ratings and free-text feedback on assistant messages are stored on the chat `Message` rows. Superusers can download them as CSV for analysis.

- **UI:** While viewing the Email Whitelist page (`/aquillm/email_whitelist/`), superusers see **Download Feedback CSV** in the **top navigation bar** (next to the account control), aligned with the rest of the header.
- **API:** `GET /api/feedback/ratings.csv` (same permission: Django superuser only; otherwise HTTP 403). If the request sends **`Accept-Encoding: gzip`** (browsers and `curl --compressed` do), the body is **gzip-compressed** with `Content-Encoding: gzip` to keep large exports light on the wire; the payload is still UTF-8 CSV after decompression.
- **Columns (in order):** `date` (ISO 8601 UTC), `user_number` (conversation owner user id), `rating` (1–5, or empty if only comments were submitted), `question_number` (1-based count of user prompts in that conversation up to and including the assistant turn), `comments`.
- **Optional query parameters:** `start_date`, `end_date` (inclusive; `YYYY-MM-DD` or parseable datetime), `min_rating` (integer; rows without a numeric rating are excluded when set), `user_number` (filter by conversation owner id).

Example (after saving session cookies to `cookies.txt`):

```bash
curl --compressed -L -b cookies.txt "http://localhost:8000/api/feedback/ratings.csv?start_date=2026-03-01&end_date=2026-03-31" -o feedback_ratings.csv
```

## Tests and hygiene

Backend tests use pytest from the `aquillm/` directory (where `manage.py` lives), with `DJANGO_SETTINGS_MODULE=aquillm.settings`. Set the same environment variables as runtime (at minimum `SECRET_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and Google OAuth variables if `DJANGO_DEBUG` is off). PostgreSQL must be reachable for tests that use `@pytest.mark.django_db`.

```bash
cd aquillm
python -m pytest aquillm/tests aquillm/apps/chat/tests aquillm/apps/ingestion/tests -q
```

To ensure generated paths such as `node_modules/` are not committed, run `pwsh -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1` from the repository root.

## Contributors

### Project Leads

* Bernie Boscoe (Southern Oregon University)
* Tuan Do (UCLA)

### Other Contributors

* Chandler Campbell (Lead Developer, Southern Oregon University)
* Jack Stark
* Jacob Nowack (Southern Oregon University)
* Skyler Acosta (Southern Oregon University)
* Zhuo Chen (University of Washington)
* Kevin Donlon (Southern Oregon University)
* Jackson Godsey (Southern Oregon University)
* Tee Grant (Southern Oregon University)
* Elyjah Kiehne (Southern Oregon University)
* Jonathan Soriano (UCLA)

## Contributing

We welcome contributions! AquiLLM is an open-source project, and we appreciate help from the community.
*   **Using AquiLLM**: Please use AquiLLM to help you with your research. This will help us identify bugs and areas for improvement.
*   **Reporting Bugs**: Please open an issue on GitHub detailing the problem, expected behavior, and steps to reproduce.
*   **Feature Requests**: Open an issue describing the feature and its potential benefits.
*   **Pull Requests**: Send a pull request!
*   **Code style and structure**: Follow [docs/code-style-guide.md](docs/code-style-guide.md) for repository standards and quality gates.
