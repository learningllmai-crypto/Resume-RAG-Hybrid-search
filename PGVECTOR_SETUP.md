# PostgreSQL / pgvector resume RAG

The active entry points are `ingest_resumes.py` and `resume_engine.py`. They now
use PostgreSQL; the engine has no Qdrant or FastEmbed dependency. The separate
`Resume_rag_engine.py` file is an existing legacy Qdrant copy, not the migrated
entry point.

## Run this project

Docker Desktop and Ollama must be running. The migration has already created
the local `.env`, built the database image, and ingested the PDF folder.

```bash
docker compose up -d --build --wait
.venv/bin/python resume_engine.py
```

Example conversation:

```text
Find Python and Docker developers in Mumbai with at least five years of experience
What projects have they made?
Show their resumes
```

For subsequent PDF additions or edits, finish ingestion before restarting or
refreshing query workers:

```bash
.venv/bin/python ingest_resumes.py
.venv/bin/python resume_engine.py
```

Unchanged files skip embedding and database replacement. `--prune` explicitly
removes indexed resumes whose source PDFs were deleted. `--check` checks PDF
extraction, Nomic connectivity, and the database connection without index writes.

## Fresh machine setup

1. Create a Python 3.10+ virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and set a password before the first database start.
3. Start Docker Desktop and Ollama. Install `llama3.2:3b` and `nomic-embed-text` in Ollama.
4. Run `docker compose up -d --build --wait`.
5. Run `.venv/bin/python ingest_resumes.py`, then `.venv/bin/python resume_engine.py`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The database is bound to **127.0.0.1:5437**. Docker's persistent named volume
stores PostgreSQL data. `docker compose stop` stops the service while retaining
that data. The image is PostgreSQL 17 with pgvector 0.8.7. The Dockerfile builds
the database service; the Python engine continues running from VS Code on the host.

## Connection settings

Both scripts read literal `KEY=value` settings from the project `.env` without
executing shell expressions. Existing process environment variables take priority.

```text
POSTGRES_USER=resume_app
POSTGRES_DB=resume_rag
POSTGRES_PASSWORD=<your local password>
PGHOST=127.0.0.1
PGPORT=5437
```

`DATABASE_URL` can override these settings for another PostgreSQL instance.
Use a normal `postgresql://...` URI, not a SQLAlchemy driver URI. The database
owner must be able to enable `vector`, or an administrator must enable it first.
Changing `.env` credentials after the volume is initialized does not change the
existing PostgreSQL role's password; update the role and connection settings together.

The local `.env` contains credentials and is excluded from Git. No database
password is embedded in either Python file or in the Dockerfile.

## Retrieval and data layout

- **Dense:** the same Nomic model, prefixes, and cosine similarity. The current
  dataset has 768-dimensional vectors. Dimension consistency is enforced per dataset.
- **Keyword:** BM25 on English lexemes from stored chunk text. Document sparse vectors
  hold saturated term frequencies and length normalization. The query sparse vector
  holds corpus IDF weights. PostgreSQL ranks their pgvector inner products.
- **Technical terms:** C++, C#, .NET, Node.js, REST API/APIs, PostgreSQL/Postgres,
  and scikit-learn/sklearn receive matching normalization in indexed text and queries.
- **Fusion:** RRF adds `1 / (RRF_K + rank)` for each branch before candidate-level ranking.
- **Coverage:** the dense query materializes every eligible row before sorting.
  There is no ANN index or retrieval `LIMIT`; a named server cursor streams batches
  of 256. The engine verifies every expected dense chunk was returned.
- **Eligibility:** strict skill/role/location/experience rules still run before
  ranking. The sparse branch ranks any overlap within the confirmed candidate pool.
- **Memory and answers:** the LangGraph intent flow, SQLite sessions, source checks,
  complete project sections, and full-resume fetching retain their existing behavior.

Tables are namespaced by `RESUME_COLLECTION` (default `resume_rag_v1`):

| Table | Purpose |
|---|---|
| `rag_datasets` | Recipe, embedding dimension, and ingestion readiness |
| `rag_resumes` | Complete source text, candidate ID, file hash, and metadata |
| `rag_chunks` | Chunk text, metadata, dense vectors and BM25 sparse vectors |
| `rag_bm25_models` | Per-dataset vocabulary, IDF, scoring settings and indexed revision |

Each resume replacement is a database transaction. An embedding or insert failure
preserves the previous committed resume. Failed ingestion leaves the dataset marked
incomplete; rerun ingestion to resume safely. A dataset advisory lock prevents
concurrent ingestions of the same dataset. Query branches share a repeatable-read
snapshot. After ingestion, restart the engine or call `engine.refresh()`.

## Upgrade an existing dataset to BM25

The engine backfills sparse vectors from existing indexed chunk text the first time
it opens an older dataset. To run the same rebuild explicitly:

```bash
.venv/bin/python ingest_resumes.py --rebuild-bm25
```

The rebuild uses the current database image and preserves the existing dense embeddings.
Uploads, replacements and deletions rebuild the sparse model with the corpus revision
in the same transaction. Bulk ingestion builds it once before the dataset becomes ready.

Optional `.env` settings are `BM25_K1=1.2` (term-frequency saturation), `BM25_B=0.75`
(length normalization) and `RRF_K=60` (rank smoothing). Restart the engine after
changing settings. Startup rebuilds a stale BM25 model automatically. Use
`resume_engine.py --mode sparse` for BM25-only CLI retrieval or `--mode dense`
for dense-only retrieval. The CLI and chatbot default to hybrid.

## Checks

```bash
.venv/bin/python resume_engine.py --self-test
.venv/bin/python resume_engine.py --evaluate --samples 12
.venv/bin/python ingest_resumes.py --check
```

The offline tests include exhaustive batching and a candidate beyond the first
256 results ranking first. Retrieval benchmarks use generated corpus probes;
they are smoke tests, not human-judged answer accuracy.

The BM25 benchmark and a real query trace are available in `.cache/bm25-benchmark.json` and `.cache/bm25-example.json`.

References: [pgvector](https://github.com/pgvector/pgvector),
[PostgreSQL full-text ranking](https://www.postgresql.org/docs/17/textsearch-controls.html),
[GIN text indexes](https://www.postgresql.org/docs/17/textsearch-indexes.html).
