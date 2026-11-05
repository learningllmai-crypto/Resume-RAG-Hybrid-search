# Resume Desk

A local resume chatbot built with LangChain, LangGraph, FastAPI, PostgreSQL/pgvector and BM25 hybrid search. Includes 1,500 synthetic resumes.

**Requirements:** macOS, Python 3.10+, Docker Desktop with Docker Compose, and Ollama installed. First launch needs internet access for downloads.

From the downloaded or cloned project folder, run:

```bash
zsh 'Launch Resume Desk.command'
```

The launcher starts Docker and Ollama, creates local database credentials, installs Python packages from `requirements.txt`, downloads `llama3.2:3b` and `nomic-embed-text` if missing, indexes the bundled PDFs and launches the [web chatbot](http://127.0.0.1:8000). Keep the Terminal window open. First indexing can take a few minutes.

**Try your own resumes:** use **Resume library → Upload PDFs** in the chatbot. For an initial library containing only your files, replace the PDFs in `Resume_RAW/` before the first launch.

**Uploading to GitHub:** include `Resume_RAW/` and `.env.example`. The existing `.gitignore` excludes local credentials, virtual environments, caches and chat-memory files.
