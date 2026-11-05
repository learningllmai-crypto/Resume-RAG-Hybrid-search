# Resume Desk

A local browser chatbot using your existing LangChain/LangGraph engine, PostgreSQL hybrid retrieval, and Ollama models. No paid API or frontend build tools are required.

## Launch on this Mac

Double-click **Launch Resume Desk.command** in Finder, or run:

```sh
./"Launch Resume Desk.command"
```

The launcher starts PostgreSQL, installs the Python dependencies, starts Ollama and downloads missing models. On first launch it creates local credentials and indexes any bundled PDFs before opening **http://127.0.0.1:8000**. Keep the Terminal window open. Press Control-C to stop the web app.

Install Docker Desktop, Ollama and Python 3.10 or later on the Mac. First launch needs internet access for missing downloads. See [the handoff guide](HANDOFF.md) for what to share.

For VS Code:

```sh
docker compose up -d --wait
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:8000**. Optional port: `app.py --port 8001`. OpenAPI schema: **http://127.0.0.1:8000/openapi.json**.

## Use the app

- Ask questions in the composer. Follow-ups retain the current conversation's candidate context.
- **New conversation** starts a separate session. Previous chats remain in the sidebar and survive application restarts. The menu next to a conversation lets you rename or delete it.
- **Resume library** shows searchable, paginated resumes and their full indexed text. Click source references in an answer to open the corresponding resume.
- **Add resumes** accepts multiple PDFs, up to 15 MB and 100 pages each. Progress shows extraction/chunking, embedding and saving. A “saved and ready to chat” notification means the resume is searchable. Identical PDFs are deduplicated. Scanned PDFs without extractable text need OCR before upload.
- **Replace PDF** keeps the candidate identity and updates its text, vectors, lexical index and saved PDF. Use it to preserve custom formatting.
- **Edit text** creates a new PDF with a simple text layout and re-indexes it. Keep the name first and headings on separate lines. This changes the PDF's layout.
- **Delete** removes the source PDF, full text and all chunks/vectors. Old conversation answers remain historical snapshots. New answers use the current library. Deleting the last resume is supported; upload another before asking a question.

## How the pieces work

`app.py` serves the FastAPI routes, the frontend and a durable PostgreSQL job queue. `web/index.html`, `web/styles.css` and `web/app.js` provide the browser interface without a build step. `ingest_resumes.py` supplies the shared PDF preparation, dense embeddings and transactional index writes. `resume_engine.py` remains the query engine, with LangGraph planning/routing and LangChain Ollama model/embedding calls.

Each chat has a unique engine session. Conversation titles, all messages, answers and jobs are stored in PostgreSQL (`rag_web_*` tables). Engine context is persisted under `.cache/chatbot/engine_memory.sqlite3`. A signed browser cookie separates chat histories between browser profiles. Keep `.cache/chatbot/browser.secret` when moving the workspace if you want existing cookies to remain valid. Clearing cookies loses that browser's access to its old chats.

Resume jobs and questions are serialized through one worker to avoid overloading the local models and to make index refresh deterministic. Every query queued after a completed resume change uses refreshed data. API endpoints remain responsive while model work runs. Duplicate message submissions are idempotent; concurrent edits use file hashes to reject stale changes. File backups and persisted job states allow unfinished operations to resume on restart.

Uploaded source PDFs go to `Resume_RAW/WEB_<id>.pdf`, matching the CLI's stable candidate identity convention. Keep this folder and the PostgreSQL Docker volume backed up together. The launcher uses a ready dataset directly. A later CLI ingestion should use the same resume folder, collection and embedding model.

Run **one app process / one Uvicorn worker** per dataset. The launcher binds only to this computer. Resume libraries are shared among browsers on this local workspace; chat histories are browser-specific. This is ready for local use. Public or multi-tenant hosting would require account authentication, per-user resume permissions, HTTPS and a separate deployment configuration.

## Verification

```sh
.venv/bin/python resume_engine.py --self-test
.venv/bin/python tests/test_chatbot.py
```

The second command uses the real local PostgreSQL and Ollama services in a disposable dataset and temporary resume folder. It covers upload, deduplication, malformed/blank PDFs, automatic filters, follow-up context, full resumes, replacement, text editing, immediate search refresh, deletion, chat isolation, history persistence, API idempotency, rollback after a simulated file-write failure, retries and interrupted-job recovery. It deletes its own test records afterward; it does not modify the main library.

If the service indicator is offline, start Docker/Ollama and check the models. Failed resume operations show their reason and a Retry button for recoverable service errors. Resume changes are only exposed to search after the database transaction commits. Questions submitted while a resume job runs wait in the queue. If another bulk ingestion is running, finish it before updating resumes in the app.
