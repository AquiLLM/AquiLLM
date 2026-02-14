# AquiLLM


![AquiLLM Logo](aquillm/aquillm/static/images/aquila.svg)

[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.1-green.svg)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**AquiLLM is an open-source RAG (Retrieval-Augmented Generation) application designed specifically for researchers.** It helps you manage, search, and interact with your research documents using AI, streamlining your literature review and knowledge discovery process. Upload various document formats, organize them into collections, and chat with an AI that understands your library's content.

More info can be found at [[https://aquillm.org]]

<!-- ![AquiLLM Screenshot](path/to/screenshot.gif) -->

## Key Features

*   **Versatile Document Ingestion**: Upload PDFs, fetch arXiv papers by ID, import VTT transcripts, scrape webpages, and process handwritten notes (with OCR).
*   **Intelligent Organization**: Group documents into logical `Collections` for focused research projects.
*   **AI-Powered Chat**: Engage in context-aware conversations with your documents, ask follow-up questions, and get answers with source references.

## Tech Stack

*   **Backend**: Python, Django
*   **Frontend**: React
*   **Database**: PostgreSQL
*   **Vector Store**: pgvector (PostgreSQL extension)
*   **LLM Integration**: Claude, OpenAI, Gemini, or local models via Ollama
*   **Asynchronous Tasks**: Celery, Redis, Django Channels

*   **Authentication**: django-allauth
*   **Containerization**: Docker, Docker Compose

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
    - **LLM Provider**: Choose between cloud or local options:
      - **Cloud providers** (require API keys):
        - Set LLM_CHOICE to CLAUDE, OPENAI, or GEMINI
        - Add the corresponding API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)
      - **Local models via Ollama** (no API keys required):
        - Set LLM_CHOICE to GEMMA3, LLAMA3.2, or GPT-OSS
        - No API keys needed for local models

4.  **Build and run using Docker Compose:**
    ```bash
    # For cloud LLM providers (Claude, OpenAI, Gemini)
    docker compose up -d 
    
    # For local Ollama models (Gemma3, Llama3.2, GPT-OSS)
    docker compose --profile ollama up -d
    ```

5. **Add a superuser:**
   ```bash
   docker compose exec web ./manage.py addsuperuser
   ```

6.  **Access the application:**

    Open your browser to `http://localhost:8080`, sign in with superuser account.

7.  **Stop the application:**
    ```bash
    docker compose down
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
    - **LLM Provider**: Choose between cloud or local options:
      - **Cloud providers** (require API keys): CLAUDE, OPENAI, or GEMINI
      - **Local models via Ollama** (no API keys required): GEMMA3, LLAMA3.2, or GPT-OSS
    - Optional: Google OAuth credentials (GOOGLE_OAUTH2_CLIENT_ID, GOOGLE_OAUTH2_CLIENT_SECRET)
    - Optional: Email access permissions (ALLOWED_EMAIL_DOMAINS, ALLOWED_EMAIL_ADDRESSES). Required if OAuth is to be used.
    - Set HOST_NAME for your domain or use 'localhost' for development

4.  **Build and run using Docker Compose:**
    ```bash
    docker compose -f docker-compose-prod.yml up  -d 
    ```
    This will automatically use letsencrypt to get TLS certificates.

4. **Add a superuser for administration:**
   ```bash
   docker compose -f docker-compose-prod.yml exec web ./manage.py addsuperuser
   ```


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

## Using Ollama for Local LLM Models

AquiLLM supports running completely offline using local LLM models via Ollama. This eliminates the need for API keys and allows you to run models on your own hardware.

### Setup

1. **Enable Ollama in your environment:**
   ```bash
   # Edit your .env file
   LLM_CHOICE=GEMMA3  # or LLAMA3.2 or GPT-OSS
   ```

2. **Start services with Ollama enabled:**
   ```bash
   docker compose --profile ollama up -d
   ```

3. **Available models:**
   - `GEMMA3`: Google's Gemma 3 enhanced model (12B parameters)
   - `LLAMA3.2`: Meta's Llama 3.2 model
   - `GPT-OSS`: Open-source GPT model (120B parameters variant)

### Requirements

- **GPU recommended**: Ollama models run best with NVIDIA GPU support (required for GPT-OSS)
- **RAM**: At least 16GB recommended for smaller models (GEMMA3, LLAMA3.2), 64GB+ for larger models (GPT-OSS)
- **Disk Space**: Models are downloaded on first use and can be several GB each
- The Ollama service will automatically download models on first use

### Using Custom Ollama Models

To use a different Ollama model not listed above, you can modify `aquillm/aquillm/apps.py` to add your preferred model. The Ollama service connects via OpenAI-compatible API at `http://ollama:11434/v1/`.

## Contributing

We welcome contributions! AquiLLM is an open-source project, and we appreciate help from the community.

*   **Reporting Bugs**: Please open an issue on GitHub detailing the problem, expected behavior, and steps to reproduce.
*   **Feature Requests**: Open an issue describing the feature and its potential benefits.
*   **Pull Requests**: Send a pull request!
