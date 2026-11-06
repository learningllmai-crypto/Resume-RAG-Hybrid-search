# Resume Desk

A resume chatbot with local search and a choice of Ollama or API chat models. Built with LangChain, LangGraph, FastAPI, PostgreSQL/pgvector and BM25 hybrid search. Includes 1,500 synthetic resumes.

**Requirements:** macOS, Python 3.10+, Docker Desktop with Docker Compose, and Ollama installed. First launch needs internet access for downloads.

From the downloaded or cloned project folder, run:

```bash
zsh 'Launch Resume Desk.command'
```

The launcher starts Docker and Ollama, creates `.env` with local database credentials, installs Python packages, downloads the configured Ollama models if missing and indexes the bundled PDFs. It opens the [web chatbot](http://127.0.0.1:8000). Keep Terminal open. First indexing can take a few minutes.

### Choose a chat model

Edit or add the settings below in the local `.env` file, then restart the launcher. The defaults are also listed in `.env.example`.

For another Ollama model, set `CHAT_PROVIDER=ollama` and change `CHAT_MODEL` to its Ollama tag. The launcher downloads it if needed.

For OpenAI, add these settings and replace the key placeholder with your API key.

```dotenv
CHAT_PROVIDER=openai
CHAT_MODEL=gpt-4.1-mini
CHAT_API_KEY=your-provider-key
CHAT_API_BASE_URL=https://api.openai.com/v1
```

An OpenAI-compatible provider can use its own base URL, key and model ID. The chat endpoint must support JSON Schema structured outputs, which the engine uses to plan searches and select citations. The status indicator also checks the provider's `/models` endpoint. Hosted chat sends questions and selected resume excerpts to the provider and uses your API credits. Keep the key in the Git-ignored `.env` file.

Ollama still runs the local `nomic-embed-text` search model in API chat mode. Switching chat models reuses the existing index. To change the embedding model, set `EMBED_MODEL` to the new Ollama tag and `RESUME_COLLECTION` to an unused name, then relaunch to build a separate index from the PDFs on disk.

**Try your own resumes:** use **Resume library → Upload PDFs** in the chatbot. For an initial library containing only your files, replace the PDFs in `Resume_RAW/` before the first launch.

**Uploading to GitHub:** include `Resume_RAW/` and `.env.example`. The existing `.gitignore` excludes local credentials, virtual environments, caches and chat-memory files.
