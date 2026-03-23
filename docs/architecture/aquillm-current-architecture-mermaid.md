# AquiLLM Architecture (Mermaid)

This document captures the current architecture as implemented in the repository and compose configuration.

## 1) System Container View

```mermaid
flowchart LR
  U[Browser user]
  FE[React UI built assets]
  ZOT[Zotero API OAuth]

  %% Edge / ingress
  NGINX[nginx and certbot production]
  DEV[Direct web access port 8080 development]

  %% App runtime
  WEB[web container Django ASGI Channels]
  WORKER[worker container Celery async tasks]
  MODS[Domain modules chat collections documents ingestion memory core admin zotero]
  ORCH[Runtime orchestration websocket ingestion llm routing memory injection]

  %% Data / infra
  PG[PostgreSQL pgvector]
  REDIS[Redis channels celery broker result]
  MINIO[MinIO object storage]
  QDRANT[Qdrant optional for Mem0]

  %% AI integrations
  HOSTED[Hosted model APIs OpenAI Claude Gemini]
  VLLM[Local vLLM profile optional chat ocr transcribe embed rerank]
  OCR[OCR transcription providers Tesseract Gemini Qwen Whisper]
  MEM0[Mem0 SDK backend optional]

  %% Core request paths
  FE --> U
  U --> NGINX
  U -.-> DEV
  NGINX --> WEB
  DEV -.-> WEB

  WEB --> MODS
  WEB --> ORCH
  WEB --> WORKER
  WORKER --> ORCH

  %% App to data
  WEB --> PG
  WEB --> REDIS
  WEB --> MINIO
  WORKER --> PG
  WORKER --> REDIS
  WORKER --> MINIO
  ORCH -.-> QDRANT

  %% App to AI
  ORCH --> HOSTED
  ORCH -.-> VLLM
  ORCH --> OCR
  ORCH -.-> MEM0
  MEM0 -.-> QDRANT
  MEM0 -.-> VLLM

  %% External integration
  MODS --> ZOT
```

## 2) Chat Request Runtime Flow

```mermaid
flowchart TD
  USER[User message in UI] --> WSHTTP[HTTP and WebSocket to Django web]
  WSHTTP --> CHATAPI[Chat views and consumers]
  CHATAPI --> BUILDCTX[Build conversation context]
  BUILDCTX --> MEMSEL{Memory backend}
  MEMSEL --> LOCALMEM[Local memory retrieval pgvector backed]
  MEMSEL --> MEM0R[Mem0 retrieval path]
  LOCALMEM --> PROMPT[Assemble system and retrieved memory]
  MEM0R --> PROMPT

  PROMPT --> ROUTE{LLM provider route}
  ROUTE --> HOSTED[OpenAI Claude Gemini]
  ROUTE --> VLLM[Local vLLM endpoints]

  HOSTED --> RESP[LLM response]
  VLLM --> RESP
  RESP --> TOOL{Tool call}
  TOOL --> TOOLS[Execute tools and append tool results]
  TOOLS --> ROUTE
  TOOL --> SAVE[Persist messages]

  SAVE --> ASYNC[Enqueue memory task]
  ASYNC --> CELERY[Celery worker]
  CELERY --> MEMWRITE{Memory write mode}
  MEMWRITE --> LWRITE[Write local episodic memory]
  MEMWRITE --> MWRITE[Write Mem0 optional dual write local]
  LWRITE --> RET[Send final answer to client]
  MWRITE --> RET
```

## 3) Unified Ingestion Runtime Flow

```mermaid
flowchart TD
  UPLOAD[User upload or ingest request] --> API[Ingestion API endpoint]
  API --> BATCH[Create ingestion batch and batch item]
  BATCH --> QUEUE[Queue ingest uploaded file task]
  QUEUE --> WORKER[Celery worker]

  WORKER --> READ[Read source file from storage]
  READ --> PARSE[Extract text payloads type and modality detection]

  PARSE --> DECIDE{Payload modality type}
  DECIDE --> DOCS[Create PDFDocument or RawTextDocument]
  DECIDE --> IMG[Create ImageUploadDocument or DocumentFigure]
  DECIDE --> MEDIA[Create MediaUploadDocument and transcript]

  IMG --> OCRPATH[OCR provider path Tesseract Gemini Qwen]
  MEDIA --> TRANSPATH[Transcription provider path Whisper or local endpoint]
  OCRPATH --> SAVE
  TRANSPATH --> SAVE
  DOCS --> SAVE

  SAVE[Persist documents chunks metadata] --> STORE[PostgreSQL and MinIO]
  STORE --> STATUS[Update batch item status and parser metadata]
  STATUS --> MONITOR[Ingestion monitor API and WebSocket]
  MONITOR --> UI[UI refresh and progress visibility]
```

## 4) Deployment Profiles (compose-level)

```mermaid
flowchart LR
  BASE[Base stack web worker db redis minio createbuckets]
  PROD[Production extras nginx certbot get certs]
  VLLM[Optional vllm profile chat ocr transcribe embed rerank]
  MEM0[Optional Mem0 backend mode with qdrant and mem0 sdk]

  BASE --> PROD
  BASE --> VLLM
  BASE --> MEM0
  VLLM --> MEM0
```

## Notes

- Dashed nodes/edges represent optional or mode-dependent paths.
- React is built during web container startup and served as static assets through Django.
- WebSockets are handled by Channels (Redis channel layer). The ASGI stack registers `apps.chat.routing` and `apps.ingestion.routing` websocket patterns (alongside project-level crawl status routes) in `aquillm/asgi.py`.
- Celery worker handles asynchronous ingestion and memory-writing tasks.
