# Resume Desk handoff

Resume Desk searches local PDFs, answers questions with resume citations, and lets users manage resumes and saved chats in the browser.

| What it uses | Why it’s there |
|---|---|
| **Ollama + Llama 3.2 3B** (`llama3.2:3b`) | Runs locally to interpret questions, resolve follow-ups, and select evidence for cited answers. |
| **Nomic Embed Text** (`nomic-embed-text`) | Creates 768-dimensional dense vectors for resume passages and questions so search can match meaning. |
| **PyMuPDF + LangChain + tiktoken** | Extracts PDF text and keeps recognized projects and jobs in separate chunks. Long entries use a 320-token splitter with 48-token overlap and carry their heading into continuation chunks. Full extracted text supports complete-resume requests. |
| **BM25 sparse retrieval** | Matches technical terms using term frequency, chunk length, and how rare a term is across the library. The default settings are `k1=1.2` and `b=0.75`. |
| **Hybrid search + RRF** | Combines pgvector cosine ranking and BM25 sparse inner-product ranking with `1 / (60 + rank)` per branch. Explicit skill, location and experience filters determine eligibility before search. Counts use candidate records, and full-resume requests use stored text. |
| **LangChain + LangGraph** | LangChain provides document objects and Ollama calls. LangGraph routes questions through planning, filtering, retrieval or a direct answer path. Python verifies quoted evidence against source text. |
| **PostgreSQL 17 + pgvector + psycopg** | Stores resumes, both vector types, chat transcripts and the job queue. Docker Compose starts the database and persists its data in a volume. SQLite stores the engine’s follow-up context. |
| **FastAPI + Uvicorn + HTML/CSS/JavaScript** | Serves the local chat and resume library, accepts PDF uploads, and lets the browser poll for answers and indexing progress. |

## Launch on another Mac

Share the **whole project folder**, including `Resume_RAW/` if the recipient should get the same resume library. They need **Docker Desktop, Ollama, and Python 3.10+** installed. First launch needs internet access for missing downloads.

**From the project folder, run one command** (or double-click the launcher in Finder):

```bash
zsh 'Launch Resume Desk.command'
```

The launcher creates local database credentials, starts PostgreSQL, installs Python dependencies, downloads missing models, indexes bundled PDFs into a fresh library, and opens the [chatbot](http://127.0.0.1:8000). First indexing may take a few minutes. The recipient can start with an empty `Resume_RAW/` and upload PDFs in the browser.

Leave your `.env`, `.venv/`, and `.cache/` out of the handoff. The recipient gets a fresh local database and conversation history.
