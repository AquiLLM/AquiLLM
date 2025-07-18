# AquiLLM


![AquiLLM Logo](aquillm/aquillm/static/images/aquila.svg)

[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.1-green.svg)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**AquiLLM is an open-source RAG (Retrieval-Augmented Generation) application designed specifically for researchers.** It helps you manage, search, and interact with your research documents using AI, streamlining your literature review and knowledge discovery process. Upload various document formats, organize them into collections, and chat with an AI that understands your library's content.

<!-- ![AquiLLM Screenshot](path/to/screenshot.gif) -->

## Key Features

*   **Versatile Document Ingestion**: Upload PDFs, fetch arXiv papers by ID, import VTT transcripts, scrape webpages, and process handwritten notes (with OCR).
*   **Intelligent Organization**: Group documents into logical `Collections` for focused research projects.
*   **AI-Powered Chat**: Engage in context-aware conversations with your documents, ask follow-up questions, and get answers with source references.
*   **Customization**: Personalize the UI with themes (including accessible options) and various fonts.
*   **Admin Utilities**: Monitor Gemini API costs, track document ingestion status, and manage user access via an email whitelist.

## Tech Stack

*   **Backend**: Python, Django
*   **Frontend**: React, TypeScript
*   **Database**: PostgreSQL on AWS (using `psycopg2-binary`)
*   **Vector Store**: `pgvector` (PostgreSQL extension)
*   **AI/LLM Integration**: Anthropic Claude (Primary RAG LLM, `anthropic`), Google Gemini (`google-generativeai`), potentially others (`openai`, `cohere`)
*   **Asynchronous Tasks**: Celery, Redis, Django Channels
*   **PDF Handling**: `pypdf`, `pdf2image`
*   **Web Scraping**: `trafilatura`, `BeautifulSoup4`, `Selenium`
*   **OCR**: `pytesseract` (requires Tesseract installation)
*   **Cloud Storage**: AWS S3 (`django-storages[s3]`, `boto3`)
*   **Authentication**: `django-allauth`
*   **Containerization**: Docker, Docker Compose

## Quick Start (Developers)

This assumes you have Docker and Docker Compose installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AquiLLM.git # Replace with your repo URL
    cd AquiLLM
    ```
2.  **Copy the environment template:**
    ```bash
    cp .env.example .env
    ```
3.  **Edit the .env file with your specific configuration:**
    - Database settings: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_NAME, POSTGRES_HOST
    - At least one LLM API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)
    - Set LLM_CHOICE to your preferred provider (CLAUDE, OPENAI, GEMINI)
    - Google OAuth credentials (GOOGLE_OAUTH2_CLIENT_ID, GOOGLE_OAUTH2_CLIENT_SECRET)
    - Email access permissions (ALLOWED_EMAIL_DOMAINS, ALLOWED_EMAIL_ADDRESSES)
    - Set HOST_NAME for your domain or use 'localhost' for development

4.  **Build and run using Docker Compose:**
    ```bash
    docker compose up --build -d # -d runs in detached mode
    ```
5.  **Access the application:**
    Open your browser to `http://localhost:8080`

6.  **Log in:** Use a whitelisted SOU or UCLA email address.

7.  **Stop the application:**
    ```bash
    docker compose down
    ```

## Getting Started (End Users)

This guide provides detailed steps for users less familiar with development tools.

### Step 1: Install Docker Desktop

Docker is required to run AquiLLM. Follow these instructions for your operating system:

#### Windows
1. Download Docker Desktop from [Docker's official website](https://www.docker.com/products/docker-desktop/)
2. Double-click the installer (.exe file) and follow the installation wizard
3. When prompted, ensure the "Use WSL 2 instead of Hyper-V" option is selected (if available)
4. After installation, restart your computer
5. Docker Desktop should start automatically. Look for the Docker icon in your system tray

#### Mac
1. Download Docker Desktop for Mac from [Docker's official website](https://www.docker.com/products/docker-desktop/)
   - For Intel Macs: Choose the Intel chip option
   - For newer Macs with M1/M2 chips: Choose the Apple chip option
2. Open the downloaded .dmg file
3. Drag the Docker icon to the Applications folder
4. Open Docker from your Applications folder
5. You may need to provide administrator permissions during first launch

#### Ubuntu Linux
1. Open Terminal (press Ctrl+Alt+T)
2. Update your package index:
   ```
   sudo apt-get update
   ```
3. Install necessary packages:
   ```
   sudo apt-get install ca-certificates curl gnupg
   ```
4. Add Docker's official GPG key:
   ```
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   ```
5. Set up the repository:
   ```
   echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```
6. Update package index again:
   ```
   sudo apt-get update
   ```
7. Install Docker packages:
   ```
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```
8. Verify Docker is installed correctly:
   ```
   sudo docker run hello-world
   ```

### Step 2: Download AquiLLM

1. Download this repository by clicking the green "Code" button on the GitHub page and selecting "Download ZIP"
2. Extract the ZIP file to a location on your computer where you can easily find it (like your Desktop)

### Step 3: Configure Environment Variables

AquiLLM requires several environment variables to be set up for proper operation. The application uses these variables to connect to various services like the database, storage, and AI providers.

1. Copy the example environment file:
   ```
   cp .env.example .env
   ```

2. Open the `.env` file in a text editor and configure the following critical settings:

   - **Authentication**: The application uses Google OAuth for login
     - `GOOGLE_OAUTH2_CLIENT_ID` and `GOOGLE_OAUTH2_CLIENT_SECRET`: Create these at [Google Cloud Console](https://console.cloud.google.com/)
     - `ALLOWED_EMAIL_DOMAINS`: Comma-separated list of email domains that can register (e.g., `example.edu,example.org`)
     - `ALLOWED_EMAIL_ADDRESSES`: Specific email addresses allowed to access the application

   - **LLM API Keys**: At least one of these is required:
     - `ANTHROPIC_API_KEY`: For Claude models (recommended)
     - `OPENAI_API_KEY`: For GPT models
     - `GEMINI_API_KEY`: For Google Gemini models
     - Set `LLM_CHOICE` to your preferred provider (`CLAUDE`, `OPENAI`, or `GEMINI`)

   - **Other Settings**:
     - `HOST_NAME`: Your domain name (for production) or `localhost` (for development)
     - Database and Storage settings (default values work with Docker Compose)

   For development purposes, you may also want to set:
   ```
   DJANGO_DEBUG=1
   ```

### Step 4: Run AquiLLM (First Time)

1. Open a command prompt/terminal:
   - Windows: Press Win+R, type "cmd" and press Enter
   - Mac: Open Terminal from Applications > Utilities
   - Ubuntu: Press Ctrl+Alt+T

2. Navigate to the AquiLLM folder:
   ```
   cd path/to/AquiLLM
   ```
   (Replace "path/to/AquiLLM" with the actual path where you extracted the files)

3. Build and start the application for the first time:
   ```
   docker compose up --build
   ```

   This will take several minutes to complete. You'll see lots of text output as the application builds and starts.

4. When you see messages that look like the application is ready (such as "Starting development server"), open your web browser and go to:
   ```
   http://localhost:8080
   ```

5. Log in with your Google account

### Step 5: Using AquiLLM After Installation

After you've built AquiLLM once, you can start it faster next time:

1. Open a command prompt/terminal
2. Navigate to the AquiLLM folder:
   ```
   cd path/to/AquiLLM
   ```
3. Start the application:
   ```
   docker compose up web
   ```
4. Open your web browser and go to:
   ```
   http://localhost:8080
   ```
5. Log in with a SOU or UCLA email

### Step 6: Stopping AquiLLM

To stop AquiLLM:
1. Go to the command prompt/terminal where it's running
2. Press Ctrl+C
3. Wait for the application to shut down gracefully

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

### Customizing Your Experience

1. **Change Theme and Font**:
   - Click your username in the upper right corner
   - Select "Manage Account"
   - Choose your preferred theme and font, them descriptions are shown on the right
   - Click "Save Settings"

2. **Available Themes**:
   - Aquillm Default Light
   - Aquillm Default Dark
   - Aquillm Light Accessible Chat
   - Aquillm Dark Accessible Chat
   - High Contrast

3. **Font Options**:
   - Sans-serif
   - Verdana
   - Times New Roman
   - OpenDyslexic (for improved readability)
   - Lexend
   - Comic Sans

### Admin Utilities

AquiLLM includes several utilities:

1. **Gemini Costs Monitor**
   - Tracks and displays API usage costs for Gemini AI services
   - Monitors OCR processing costs
   - Provides detailed breakdown of API usage, shows total input tokens, total output tokens, and total tokens used
   - Helps manage budget allocation for AI services

2. **Ingestion Monitor**
   - Real-time status tracking for document uploads and processing
   - Displays which documents are currently being processed
   - Identifies failed ingestions with error messages
   - Allows administrators to monitor system performance

3. **Email Whitelist**

## Troubleshooting

### If Docker Won't Start
- Windows/Mac: Make sure Docker Desktop is running (look for the icon in your system tray/menu bar)
- Ubuntu: Try running `sudo systemctl start docker`

### If You Can't Access the Web Interface
- Make sure no other applications are using port 8080
- Check that Docker containers are running with: `docker ps`
- Try restarting the application: stop it with Ctrl+C, then run `docker compose up web` again

### If You're Getting Database Errors
- Try rebuilding the application with: `docker compose down` followed by `docker compose up --build`

## Contributing

We welcome contributions! AquiLLM is an open-source project, and we appreciate help from the community.

*   **Reporting Bugs**: Please open an issue on GitHub detailing the problem, expected behavior, and steps to reproduce.
*   **Feature Requests**: Open an issue describing the feature and its potential benefits.
*   **Pull Requests**: If you'd like to contribute code:
    1.  Fork the repository.
    2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
    3.  Make your changes.
    4.  Ensure your code follows the project's style guidelines.
    5.  Write tests for new functionality.
    6.  Commit your changes (`git commit -am 'Add some feature'`).
    7.  Push to the branch (`git push origin feature/your-feature-name`).
    8.  Open a Pull Request against the `main` branch.

Please review the `CONTRIBUTING.md` file (if it exists) for more detailed guidelines.

## License

This project is licensed under MIT

## Contact & Support

*   For issues, bugs, or feature requests, please use the [GitHub Issues](https://github.com/your-username/AquiLLM/issues) page. <!-- Replace with your repo URL -->

## Additional Information

*   The application runs on port `8080` by default.
*   Documents are processed asynchronously and chunked for efficient vector search.
*   System administrators have access to additional monitoring and management tools via the `/admin` page (requires admin privileges).

## Environment Variables

AquiLLM uses environment variables for configuration. Below is a comprehensive explanation of all available variables.

### Core Settings

| Variable | Description | Required | Default |
|----------|-------------|:--------:|---------|
| `SECRET_KEY` | Django secret key for security | Yes* | Auto-generated in debug mode |
| `DJANGO_DEBUG` | Enable debug mode | No | 0 (disabled) |
| `PORT` | Application port | No | 8080 |
| `HOST_NAME` | Application hostname | Yes | N/A |

### Database Configuration

| Variable | Description | Required | Default |
|----------|-------------|:--------:|---------|
| `POSTGRES_USER` | PostgreSQL username | Yes | postgres |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes | postgres |
| `POSTGRES_NAME` | PostgreSQL database name | Yes | postgres |
| `POSTGRES_HOST` | PostgreSQL host address | Yes | postgres |

### Storage Configuration (S3-compatible)

| Variable | Description | Required | Default |
|----------|-------------|:--------:|---------|
| `STORAGE_HOST` | S3-compatible storage host address (MinIO in Docker setup) | Yes | storage |
| `STORAGE_ACCESS_KEY` | S3 access key | Yes | test |
| `STORAGE_SECRET_KEY` | S3 secret key | Yes | test |

*Note: AquiLLM uses MinIO as an S3-compatible object storage service in the Docker development environment. The application connects to the storage using Django's S3 storage backend from the django-storages package.*

### Authentication

| Variable | Description | Required | Default |
|----------|-------------|:--------:|---------|
| `GOOGLE_OAUTH2_CLIENT_ID` | Google OAuth client ID | Yes | N/A |
| `GOOGLE_OAUTH2_CLIENT_SECRET` | Google OAuth client secret | Yes | N/A |
| `ALLOWED_EMAIL_DOMAINS` | Comma-separated list of allowed email domains | No | N/A |
| `ALLOWED_EMAIL_ADDRESSES` | Comma-separated list of allowed email addresses | No | N/A |
| `WM_EMAIL` | Webmaster email address | Yes | N/A |

### LLM Configuration

| Variable | Description | Required | Default |
|----------|-------------|:--------:|---------|
| `LLM_CHOICE` | Default LLM provider (CLAUDE, OPENAI, GEMINI) | Yes | N/A |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | No | N/A |
| `OPENAI_API_KEY` | OpenAI API key | No | N/A |
| `GEMINI_API_KEY` | Google Gemini API key | Yes** | N/A |
| `COHERE_KEY` | Cohere API key | No | N/A |

\* Required in production, auto-generated in development mode  
\*\* Required for Handwritten Notes Ingestion  
| At least one LLM API key is required, corresponding to your selected LLM_CHOICE


