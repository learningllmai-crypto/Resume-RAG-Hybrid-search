from __future__ import annotations

import argparse
from collections import Counter
import math
from enum import Enum
import hashlib
import json
import os
import re
from pathlib import Path

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pgvector import SparseVector, Vector
from pgvector.psycopg import register_vector
import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).resolve().parent
# Read local KEY=value settings and keep values already supplied by the shell.
def load_local_env():
    """Read literal KEY=value settings; never execute or expand shell expressions."""
    path = BASE / '.env'
    if path.exists():
        for raw in path.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            if re.fullmatch(r'[A-Z][A-Z0-9_]*', key.strip()):
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)

load_local_env()
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(BASE / '.cache' / 'tiktoken'))
COLLECTION = os.getenv('RESUME_COLLECTION', 'resume_rag_v1')
OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
CHAT_MODEL = os.getenv('CHAT_MODEL', 'llama3.2:3b')
# Retain the historical recipe tag so the existing dense index passes compatibility checks.
RECIPE = {'schema': 2, 'backend': 'postgres-pgvector', 'embedding': EMBED_MODEL,
          'lexical': 'postgres-english-ts_rank_cd-or-v1', 'technical_normalization': 1,
          'chunk_tokens': 320, 'overlap': 48, 'prefix': 'nomic-search-v1'}
# New indexes use entry boundaries. The sections mode can still open the rollback index.
CHUNKING = os.getenv('RESUME_CHUNKING', 'entries')
if CHUNKING not in ('sections', 'entries'):
    raise ValueError('RESUME_CHUNKING must be sections or entries.')
if CHUNKING == 'entries':
    RECIPE['entry_chunking'] = 1
# Version sparse scoring separately so it can be rebuilt from text without new dense embeddings.
BM25_SETTINGS = {
    'version': 1,
    'k1': float(os.getenv('BM25_K1', '1.2')),
    'b': float(os.getenv('BM25_B', '0.75')),
}
if not math.isfinite(BM25_SETTINGS['k1']) or BM25_SETTINGS['k1'] <= 0 or not 0 <= BM25_SETTINGS['b'] <= 1:
    raise ValueError('BM25_K1 must be finite and positive, and BM25_B must be between 0 and 1.')

HEADINGS = re.compile(r'^(professional summary|summary|profile|technical skills|skills|'
                      r'professional experience|work experience|experience|projects|education|'
                      r'certifications|achievements|keywords)\s*:?$', re.I)
# Pick meaning search, word search, or both. “Sparse” is another name for BM25 word search here.
class RetrievalMode(str, Enum):
    DENSE = 'dense'
    LEXICAL = 'lexical'
    HYBRID = 'hybrid'
    SPARSE = 'lexical'

    # Translate the CLI spelling sparse to the internal lexical mode.
    @classmethod
    def _missing_(cls, value):
        if value == 'sparse':
            return cls.LEXICAL
# Adapt Ollama's embedding API to LangChain's Embeddings interface.
class SearchEmbeddings(Embeddings):
    # Configure LangChain's OllamaEmbeddings client with the local model and timeout.
    def __init__(self):
        self.model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL,
                                      client_kwargs={'timeout': 180})
    # Prefix each Nomic document passage, then ask LangChain for a batch of dense vectors.
    def embed_documents(self, texts):
        prefix = 'search_document: ' if 'nomic' in EMBED_MODEL else ''
        return self.model.embed_documents([prefix + t for t in texts])
    # Prefix a Nomic search query so it can be compared with the stored document vectors.
    def embed_query(self, text):
        prefix = 'search_query: ' if 'nomic' in EMBED_MODEL else ''
        return self.model.embed_query(prefix + text)

    # Release the HTTP connections held by the embedding client.
    def close(self):
        self.model._client.close()


# Prefer DATABASE_URL and otherwise build a connection from individual PostgreSQL settings.
def connection_info():
    if os.getenv('DATABASE_URL'):
        return os.environ['DATABASE_URL']
    password = os.getenv('PGPASSWORD') or os.getenv('POSTGRES_PASSWORD')
    if not password:
        raise ValueError('Set DATABASE_URL or POSTGRES_PASSWORD in the project .env file.')
    return make_conninfo(host=os.getenv('PGHOST', '127.0.0.1'), port=os.getenv('PGPORT', '5437'),
        dbname=os.getenv('PGDATABASE') or os.getenv('POSTGRES_DB', 'resume_rag'),
        user=os.getenv('PGUSER') or os.getenv('POSTGRES_USER', 'resume_app'), password=password,
        connect_timeout=10, application_name='resume-rag')
# Open a psycopg connection and register pgvector adapters when the extension is available.
def client():
    db = psycopg.connect(connection_info(), autocommit=True, row_factory=dict_row)
    if db.execute("SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone():
        register_vector(db)
    return db
# Give punctuation-heavy skills stable words before PostgreSQL tokenizes the text.
def lexical_query_text(text):
    """Preserve punctuation-sensitive technologies before English tokenization."""
    aliases = [(r'(?<!\w)c\+\+(?!\w)', ' skillcpp '),
               (r'(?<!\w)c#(?!\w)', ' skillcsharp '),
               (r'(?<!\w)\.net(?!\w)', ' skilldotnet '),
               (r'\bnode[.\s_-]*js\b', ' skillnodejs '),
               (r'\brest[\s_-]+apis?\b', ' skillrestapi '),
               (r'\b(?:postgresql|postgres)\b', ' skillpostgres '),
               (r'\b(?:scikit[\s_-]*learn|sklearn)\b', ' skillsklearn ')]
    for pattern, replacement in aliases:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text
# For each allowed candidate, compare the question’s meaning numbers with the current chunks.
# Only score chunks from the requested sections and current PDF versions.
DENSE_SQL = """
WITH eligible AS MATERIALIZED (
    SELECT chunk_id, embedding FROM rag_chunks
    WHERE dataset = %(dataset)s AND candidate_id = ANY(%(candidate_ids)s)
      AND file_hash = ANY(%(file_hashes)s) AND section = ANY(%(sections)s)
)
SELECT chunk_id, 1.0 - (embedding <=> %(dense)s) AS score
FROM eligible ORDER BY score DESC, chunk_id
"""
# Compare the question’s word weights with each chunk’s word weights.
# pgvector returns the negative score, so flip its sign and keep positive matches.
LEXICAL_SQL = """
WITH eligible AS MATERIALIZED (
    SELECT chunk_id, bm25_embedding FROM rag_chunks
    WHERE dataset = %(dataset)s AND candidate_id = ANY(%(candidate_ids)s)
      AND file_hash = ANY(%(file_hashes)s) AND section = ANY(%(sections)s)
), scored AS MATERIALIZED (
    SELECT chunk_id, -(bm25_embedding <#> %(lexical)s) AS score FROM eligible
)
SELECT chunk_id, score FROM scored WHERE score > 0 ORDER BY score DESC, chunk_id
"""
SEARCH_SQL = {'dense': DENSE_SQL, 'lexical': LEXICAL_SQL}
# Create the resume tables, pgvector columns, corpus metadata and supporting indexes.
def ensure_schema(db):
    db.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(db)
    db.execute('''CREATE TABLE IF NOT EXISTS rag_datasets (
        name text PRIMARY KEY, recipe jsonb NOT NULL, embedding_dim integer NOT NULL CHECK (embedding_dim > 0),
        ready boolean NOT NULL DEFAULT false, updated_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (name, embedding_dim))''')
    db.execute('''CREATE TABLE IF NOT EXISTS rag_resumes (
        dataset text NOT NULL REFERENCES rag_datasets(name) ON DELETE CASCADE,
        candidate_id text NOT NULL, file_hash text NOT NULL, page_content text NOT NULL,
        metadata jsonb NOT NULL, PRIMARY KEY (dataset, candidate_id))''')
    db.execute('''CREATE TABLE IF NOT EXISTS rag_chunks (
        dataset text NOT NULL, candidate_id text NOT NULL, chunk_id text NOT NULL,
        file_hash text NOT NULL, section text NOT NULL, page integer NOT NULL, chunk_index integer NOT NULL,
        page_content text NOT NULL, metadata jsonb NOT NULL, embedding vector NOT NULL,
        embedding_dim integer NOT NULL, search_text text NOT NULL,
        PRIMARY KEY (dataset, chunk_id),
        FOREIGN KEY (dataset, candidate_id) REFERENCES rag_resumes(dataset, candidate_id) ON DELETE CASCADE,
        FOREIGN KEY (dataset, embedding_dim) REFERENCES rag_datasets(name, embedding_dim) ON DELETE CASCADE,
        CHECK (vector_dims(embedding) = embedding_dim))''')
    db.execute('ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS bm25_embedding sparsevec')
    db.execute('''CREATE TABLE IF NOT EXISTS rag_bm25_models (
        dataset text PRIMARY KEY REFERENCES rag_datasets(name) ON DELETE CASCADE,
        indexed_revision timestamptz NOT NULL,
        settings jsonb NOT NULL,
        vocabulary jsonb NOT NULL
    )''')
    db.execute('CREATE INDEX IF NOT EXISTS rag_chunks_candidate_section ON rag_chunks(dataset, candidate_id, section)')
    constraint = db.execute("""SELECT confdeltype FROM pg_constraint
        WHERE conrelid='rag_chunks'::regclass AND conname='rag_chunks_dataset_embedding_dim_fkey'""").fetchone()
    if constraint and constraint['confdeltype'] != 'c':
        # Group related database writes so an error rolls back the current unit of work.
        with db.transaction():
            db.execute('ALTER TABLE rag_chunks DROP CONSTRAINT rag_chunks_dataset_embedding_dim_fkey')
            db.execute('''ALTER TABLE rag_chunks ADD CONSTRAINT rag_chunks_dataset_embedding_dim_fkey
                FOREIGN KEY (dataset, embedding_dim) REFERENCES rag_datasets(name, embedding_dim) ON DELETE CASCADE''')


# Build the word-search numbers for the whole library.
# Save the chunks’ numbers and the shared word list together so their numbered slots agree.
def rebuild_bm25(db):
    """Rebuild within the caller's dataset lock and transaction, after updating its revision."""
    rows = db.execute('''SELECT chunk_id, ARRAY(
        SELECT unnest(lexemes) FROM ts_debug('english', search_text)
    ) AS tokens FROM rag_chunks WHERE dataset=%s ORDER BY chunk_id''', (COLLECTION,)).fetchall()
    frequencies = [Counter(row['tokens']) for row in rows]
    lengths = [sum(counts.values()) for counts in frequencies]
    # Count how many chunks contain each word. A word repeated ten times in one chunk still counts as one chunk.
    document_frequency = Counter()
    for counts in frequencies:
        document_frequency.update(counts.keys())

    size = len(rows)
    average_length = sum(lengths) / max(1, size) or 1.0
    # Give each word a numbered slot and a rarity weight. Words in fewer chunks get more weight.
    vocabulary = {
        term: (index, math.log1p((size - frequency + 0.5) / (frequency + 0.5)))
        for index, (term, frequency) in enumerate(sorted(document_frequency.items()))
    }
    dimension = max(1, len(vocabulary))
    k1, b = BM25_SETTINGS['k1'], BM25_SETTINGS['b']
    vectors = []
    for row, counts, length in zip(rows, frequencies, lengths):
        # Adjust for chunk length and give repeated words smaller extra boosts each time they repeat.
        norm = k1 * (1 - b + b * length / average_length)
        weights = {vocabulary[term][0]: tf * (k1 + 1) / (tf + norm) for term, tf in counts.items()}
        vectors.append((SparseVector(weights, dimension), COLLECTION, row['chunk_id']))
    # Send each batch of writes through a psycopg cursor.
    with db.cursor() as cursor:
        cursor.executemany('UPDATE rag_chunks SET bm25_embedding=%s WHERE dataset=%s AND chunk_id=%s', vectors)
    db.execute('''INSERT INTO rag_bm25_models(dataset, indexed_revision, settings, vocabulary)
        SELECT name, updated_at, %s, %s FROM rag_datasets WHERE name=%s
        ON CONFLICT(dataset) DO UPDATE SET indexed_revision=excluded.indexed_revision,
        settings=excluded.settings, vocabulary=excluded.vocabulary''',
        (Jsonb(BM25_SETTINGS), Jsonb(vocabulary), COLLECTION))
    return size


# Read the term coordinates and IDF only when their settings and corpus revision match.
def load_bm25_vocabulary(db):
    row = db.execute('''SELECT m.vocabulary FROM rag_bm25_models m JOIN rag_datasets d
        ON d.name=m.dataset AND d.updated_at=m.indexed_revision
        WHERE m.dataset=%s AND m.settings=%s''', (COLLECTION, Jsonb(BM25_SETTINGS))).fetchone()
    return row['vocabulary'] if row else None


# Put the question’s known words into the same numbered slots as the chunks, using their rarity weights.
def bm25_query_vector(db, query):
    vocabulary = load_bm25_vocabulary(db)
    if vocabulary is None:
        raise ValueError('BM25 index is stale. Restart the engine or run ingest_resumes.py --rebuild-bm25.')
    # Use PostgreSQL's English dictionary so query terms match the indexed document terms.
    terms = db.execute("SELECT tsvector_to_array(to_tsvector('english', %s)) AS terms", (query,)).fetchone()['terms']
    weights = {vocabulary[term][0]: vocabulary[term][1] for term in terms if term in vocabulary}
    return SparseVector(weights, len(vocabulary)) if weights else None


# Build or refresh word-search numbers while holding the same lock as PDF indexing. Only one writer can change the library at a time.
def ensure_bm25(db, *, force=False):
    if not force and db.execute("SELECT to_regclass('rag_bm25_models') AS name").fetchone()['name']:
        if load_bm25_vocabulary(db) is not None:
            return None
    # Group related database writes so an error rolls back the current unit of work.
    with db.transaction():
        # Share an advisory lock with other index writers while building a consistent corpus revision.
        locked = db.execute("SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS locked",
                            ('resume-rag:' + COLLECTION,)).fetchone()['locked']
        if not locked:
            raise ValueError('Another ingestion owns this dataset. Wait for it to finish.')
        manifest = load_manifest(db)
        if not manifest or not manifest['ready'] or manifest['recipe'] != RECIPE:
            raise ValueError('A ready, compatible resume index is required to build BM25.')
        ensure_schema(db)
        if force or load_bm25_vocabulary(db) is None:
            return rebuild_bm25(db)
    return None


# Read the dataset's recipe, dense dimension and readiness before opening its index.
def load_manifest(db):
    if not db.execute("SELECT to_regclass('rag_datasets') AS name").fetchone()['name']:
        return None
    return db.execute('SELECT recipe, embedding_dim, ready FROM rag_datasets WHERE name=%s', (COLLECTION,)).fetchone()
# Read the complete resumes and their chunk text together from one database snapshot.
def load_records(db):
    """One consistent snapshot of current parent and child payloads."""
    # Read the whole resumes and their chunks from the same database version, even if another process edits the library.
    with db.transaction():
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        manifest = load_manifest(db)
        if not manifest or manifest['recipe'] != RECIPE or not manifest['ready']:
            raise ValueError('PostgreSQL index is missing, incompatible, or incomplete. Run ingest_resumes.py.')
        parents = db.execute('SELECT page_content, metadata FROM rag_resumes WHERE dataset=%s ORDER BY candidate_id',
                             (COLLECTION,)).fetchall()
        chunks = db.execute('SELECT page_content, metadata FROM rag_chunks WHERE dataset=%s ORDER BY candidate_id, chunk_index',
                            (COLLECTION,)).fetchall()
    return parents, chunks
# Look for the start of each project or job. If the layout is unfamiliar, keep the section together for the normal splitter.
def section_entries(label, body):
    if CHUNKING != 'entries' or label not in (
        'projects', 'experience', 'professional experience', 'work experience',
    ):
        return [(None, body)]
    lines = body.splitlines()
    date_range = re.compile(r'\b(?:19|20)\d{2}\s*[-–—]\s*(?:[A-Za-z]{3,9}\s+)?(?:\d{4}|present|current)\b', re.I)
    boundaries = []
    for i, line in enumerate(lines):
        title = line.strip()
        if not title or title.startswith(('•', '-', '*')) or HEADINGS.fullmatch(title):
            continue
        following = lines[i + 1].strip() if i + 1 < len(lines) else ''
        if label == 'projects':
            # A project title with a tool list, or a title after a break, starts an entry when bullets follow it.
            section_break = i == 0 or not lines[i - 1].strip() or HEADINGS.fullmatch(lines[i - 1].strip())
            if following.startswith(('•', '- ', '* ')) and (' | ' in title or section_break):
                boundaries.append(i)
        elif (' | ' in title and date_range.search(title)) or date_range.match(following):
            # A job heading can carry its dates on the same line or just below it. Recognize both layouts.
            boundaries.append(i)
    if not boundaries:
        return [(None, body)]
    entries = []
    for start, end in zip(boundaries, boundaries[1:] + [len(lines)]):
        title = lines[start].strip()
        # Keep any leading section heading or unclassified text with the first entry.
        first_line = 0 if start == boundaries[0] else start
        entries.append((title, '\n'.join(lines[first_line:end])))
    return entries


# Extract PDF page text and keep project/job entries together before the token fallback.
def extract(path, root):
    """Preserve PDF page and section boundaries, with token bounded overlapping chunks."""
    relative = path.relative_to(root).as_posix()
    cid = path.stem if path.parent == root else relative[:-4]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pymupdf.open(path) as pdf:
        pages = [p.get_text('text', sort=True).strip() for p in pdf]
    if not any(pages):
        raise ValueError(f'{relative}: no extractable text; OCR is required')
    name = next(line.strip() for page in pages for line in page.splitlines() if line.strip())
    # LangChain counts small pieces of words called tokens. It splits long text and repeats a bit at each cut for context.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base', chunk_size=RECIPE['chunk_tokens'],
        chunk_overlap=RECIPE['overlap'], separators=['\n\n', '\n', '. ', ' ', ''])
    metadata = dict(kind='chunk', candidate_id=cid, candidate_name=name,
                    source=relative, file_hash=digest)
    chunks = []
    section = 'identity'
    # Add a version to the chunk ID. An old citation should never accidentally name a different piece of text.
    version = ':e1' if CHUNKING == 'entries' else ''
    for page_no, page in enumerate(pages, 1):
        groups, lines = [], []
        for line in page.splitlines():
            if HEADINGS.fullmatch(line.strip()):
                if lines:
                    groups.append((section, '\n'.join(lines)))
                section, lines = line.strip().rstrip(':').lower(), []
            lines.append(line)
        if lines:
            groups.append((section, '\n'.join(lines)))
        for label, body in groups:
            for title, entry in section_entries(label, body):
                for part_no, part in enumerate(splitter.split_text(entry)):
                    if title and part_no:
                        # If a project or job is too long for one chunk, repeat its title so later pieces still say what they belong to.
                        part = f'{title}\n{part}'
                    index = len(chunks)
                    chunk_id = f'{cid}:{digest[:12]}{version}:{index:04d}' # creates unique chunk id.
                    # Store the entry title alongside the page and section so answers can name the source.
                    chunks.append(Document(page_content=f'Candidate: {name}\n{part}', metadata={
                        **metadata, 'section': label, 'page': page_no, 'chunk_index': index,
                        'chunk_id': chunk_id, **({'entry_title': title} if title else {})}))
    # Keep the complete extracted text separately so full-resume requests can bypass retrieval.
    full_text = '\n\n'.join(f'--- Page {i} ---\n{t}' for i, t in enumerate(pages, 1))
    parent = {'page_content': full_text, 'metadata': {**metadata, 'kind': 'parent',
              'pages': len(pages), 'chunks': len(chunks)}}
    return cid, digest, parent, chunks
# Assign the intended candidate ID and source path after extracting a staged PDF.
def prepare_resume(path, *, candidate_id=None, source=None, original_filename=None):
    """Prepare a staged PDF for single-resume indexing without reprocessing the corpus."""
    cid, digest, parent, chunks = extract(Path(path), Path(path).parent)
    cid = candidate_id or cid
    source = source or Path(path).name
    for metadata in [parent['metadata'], *[d.metadata for d in chunks]]:
        metadata.update(candidate_id=cid, source=source)
        if original_filename:
            metadata['original_filename'] = original_filename
    for document in chunks:
        _, _, suffix = document.metadata['chunk_id'].partition(':')
        document.metadata['chunk_id'] = f'{cid}:{suffix}'
    return cid, digest, parent, chunks
# Send chunk text to Ollama in batches and verify that all vector dimensions match.
def embed_chunks(chunks, *, embeddings=None, progress=None):
    owned = embeddings is None
    embeddings = embeddings or SearchEmbeddings()
    vectors = []
    try:
        for start in range(0, len(chunks), 32):
            # Send up to 32 chunk texts to local Ollama through LangChain. The model turns each text into meaning numbers.
            vectors.extend(embeddings.embed_documents([d.page_content for d in chunks[start:start+32]]))
            if progress:
                progress(min(start+32, len(chunks)), len(chunks))
    finally:
        if owned:
            embeddings.close()
    if len(vectors) != len(chunks) or not vectors or not len(vectors[0]):
        raise ValueError('Unexpected number of embeddings.')
    dimension = len(vectors[0])
    if any(len(v) != dimension for v in vectors):
        raise ValueError('Inconsistent embedding dimensions.')
    return vectors, dimension
# Replace one candidate's rows and write the new chunk text, metadata and dense vectors.
def save_resume_records(db, cid, digest, parent, chunks, vectors, dimension):
    """Write one complete version. The caller owns the encompassing transaction."""
    if len(vectors) != len(chunks) or any(len(v) != dimension for v in vectors):
        raise ValueError(f'Unexpected embedding count/dimension for {cid}')
    db.execute('DELETE FROM rag_chunks WHERE dataset=%s AND candidate_id=%s', (COLLECTION, cid))
    db.execute('''INSERT INTO rag_resumes(dataset,candidate_id,file_hash,page_content,metadata)
        VALUES(%s,%s,%s,%s,%s) ON CONFLICT(dataset,candidate_id) DO UPDATE SET
        file_hash=excluded.file_hash,page_content=excluded.page_content,metadata=excluded.metadata''',
        (COLLECTION,cid,digest,parent['page_content'],Jsonb(parent['metadata'])))
    rows = [(COLLECTION,cid,d.metadata['chunk_id'],digest,d.metadata['section'],d.metadata['page'],
             d.metadata['chunk_index'],d.page_content,Jsonb(d.metadata),Vector(v),dimension,
             lexical_query_text(d.page_content)) for d,v in zip(chunks,vectors)]
    # Send each batch of writes through a psycopg cursor.
    with db.cursor() as cursor:
        cursor.executemany('''INSERT INTO rag_chunks(dataset,candidate_id,chunk_id,file_hash,section,page,
            chunk_index,page_content,metadata,embedding,embedding_dim,search_text)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''', rows)
# Walk the input folder, embed changed PDFs and publish a ready dataset after BM25 is fitted.
def ingest(args):
    root = args.folder.resolve()
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() == '.pdf')
    if not paths:
        raise ValueError(f'No PDFs in {root}')
    records = [extract(p, root) for p in paths]
    if len({r[0] for r in records}) != len(records):
        raise ValueError('Source paths produce duplicate candidate IDs.')
    embeddings = SearchEmbeddings()
    dimension = len(embeddings.embed_query('connection check'))
    with client() as db:
        print(json.dumps({'pdfs': len(records), 'chunks': sum(len(r[3]) for r in records),
              'embedding_dimensions': dimension, 'backend': 'postgres-pgvector', 'dataset': COLLECTION}, indent=2), flush=True)
        if args.check:
            status = db.execute("SELECT current_database() AS database, version() AS postgres, (SELECT extversion FROM pg_extension WHERE extname='vector') AS pgvector").fetchone()
            print(json.dumps(status, indent=2))
            return
        ensure_schema(db)
        # Share an advisory lock with other index writers while building a consistent corpus revision.
        locked = db.execute("SELECT pg_try_advisory_lock(hashtext(%s)) AS locked", ('resume-rag:' + COLLECTION,)).fetchone()['locked']
        if not locked:
            raise ValueError('Another ingestion owns this dataset. Wait for it to finish.')
        try:
            manifest = load_manifest(db)
            if manifest and (manifest['recipe'] != RECIPE or manifest['embedding_dim'] != dimension):
                raise ValueError('Index recipe/dimension differs. Set RESUME_COLLECTION to a new dataset name.')
            db.execute('''INSERT INTO rag_datasets(name, recipe, embedding_dim, ready) VALUES(%s,%s,%s,false)
                ON CONFLICT(name) DO UPDATE SET ready=false, updated_at=now()''',
                (COLLECTION, Jsonb(RECIPE), dimension))
            existing = {r['candidate_id']: r for r in db.execute('''
                SELECT r.candidate_id, r.file_hash, r.metadata,
                    (SELECT count(*) FROM rag_chunks c WHERE c.dataset=r.dataset AND c.candidate_id=r.candidate_id
                       AND c.file_hash=r.file_hash) AS actual_chunks
                FROM rag_resumes r WHERE r.dataset=%s''', (COLLECTION,))}
            updated = skipped = 0
            for cid, digest, parent, chunks in records:
                previous = existing.get(cid)
                if previous and previous['file_hash'] == digest and previous['actual_chunks'] == len(chunks):
                    skipped += 1
                    continue
                vectors, actual_dimension = embed_chunks(chunks, embeddings=embeddings)
                if actual_dimension != dimension:
                    raise ValueError(f'Unexpected embedding count/dimension for {cid}')
                # Group related database writes so an error rolls back the current unit of work.
                with db.transaction():
                    save_resume_records(db,cid,digest,parent,chunks,vectors,dimension)
                updated += 1
                if updated % 25 == 0:
                    print(f'Indexed {updated}; unchanged {skipped}; last {cid}', flush=True)
            removed = set(existing) - {r[0] for r in records}
            if args.prune and removed:
                db.execute('DELETE FROM rag_resumes WHERE dataset=%s AND candidate_id=ANY(%s)', (COLLECTION,sorted(removed)))
            check = db.execute('''SELECT
                (SELECT coalesce(sum((metadata->>'chunks')::int),0) FROM rag_resumes WHERE dataset=%s) AS expected,
                (SELECT count(*) FROM rag_chunks WHERE dataset=%s) AS actual''', (COLLECTION,COLLECTION)).fetchone()
            if check['expected'] != check['actual']:
                raise ValueError(f'Index consistency failed: {check}')
            db.execute('ANALYZE rag_chunks')
            # Group related database writes so an error rolls back the current unit of work.
            with db.transaction():
                db.execute('UPDATE rag_datasets SET ready=true,updated_at=now() WHERE name=%s', (COLLECTION,))
                # Recount the library’s words after saving all the changed PDFs so BM25 uses the new totals.
                rebuild_bm25(db)
            print(json.dumps({'dataset': COLLECTION,'backend':'postgres-pgvector','updated':updated,'unchanged':skipped,
                'chunks':check['actual'],'deleted':len(removed) if args.prune else 0,
                'missing_files_retained':len(removed) if not args.prune else 0}, indent=2))
        finally:
            db.execute('SELECT pg_advisory_unlock(hashtext(%s))', ('resume-rag:' + COLLECTION,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=Path, default=Path(os.getenv('RESUME_DIR', str(BASE / 'Resume_RAW'))))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--rebuild-bm25', action='store_true', help='Rebuild sparse vectors from stored chunk text only.')
    args = parser.parse_args()
    if args.rebuild_bm25:
        with client() as db:
            chunks = ensure_bm25(db, force=True)
        print(f'BM25 rebuilt for {chunks} chunks in {COLLECTION}.')
    else:
        ingest(args)
