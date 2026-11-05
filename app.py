"""Resume Desk: local FastAPI chatbot. Run `.venv/bin/python app.py`.

A durable PostgreSQL job queue serializes local-model work. Chat histories are
browser-isolated; the resume library is shared within this local workspace.
Use one Uvicorn worker. The default bind is loopback; this is not public-login hosting.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
from contextlib import asynccontextmanager
import hashlib
import hmac
import html
import logging
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import time
import uuid

import httpx
import psycopg
from psycopg.types.json import Jsonb
import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ingest_resumes import (BASE, COLLECTION, CHAT_MODEL, EMBED_MODEL, OLLAMA_URL,
    RECIPE, client, embed_chunks, ensure_schema, load_manifest, prepare_resume, rebuild_bm25, save_resume_records)
from resume_engine import ResumeEngine, profile_fields

log = logging.getLogger('resume-desk')
MAX_BYTES = int(os.getenv('RESUME_UPLOAD_MAX_MB', '15')) * 1024 * 1024
PAGE_MARKER = re.compile(r'^--- Page \d+ ---\s*$', re.M)


# Validate submitted question text and its deduplication key.
class MessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    client_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=80)


# Check the shape of a conversation title sent by the browser.
class ChatInput(BaseModel):
    title: str = Field(default='New conversation', min_length=1, max_length=100)


# Require edited text and the file hash used to reject stale replacements.
class EditInput(BaseModel):
    text: str = Field(min_length=1, max_length=150000)
    expected_hash: str = Field(min_length=64, max_length=64)
# Flush bytes into a temporary file and replace the target path in one filesystem operation.
def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.resume-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
# Escape edited text and render it as a paginated PDF with PyMuPDF.
def text_to_pdf(text: str, destination: Path):
    text = PAGE_MARKER.sub('', text).strip()
    if not text:
        raise ValueError('The resume text cannot be empty.')
    paragraphs = ''.join('<p>' + html.escape(line or ' ') + '</p>' for line in text.splitlines())
    story = pymupdf.Story(html='<body>' + paragraphs + '</body>',
        user_css='body { font-family: sans-serif; font-size: 10pt; } p { margin: 0 0 3pt; white-space: pre-wrap; }')
    writer = pymupdf.DocumentWriter(str(destination))
    try:
        story.write(writer, lambda number, filled: (pymupdf.Rect(0,0,595,842), pymupdf.Rect(44,44,551,798), None))
    finally:
        writer.close()


# Hold the storage paths, browser signing secret, engine and serial background worker.
class Workspace:
    # Prepare workspace directories and load the key used to sign browser cookies.
    def __init__(self, data_dir=None, resume_dir=None):
        self.data_dir = Path(data_dir or os.getenv('CHAT_DATA_DIR', str(BASE / '.cache' / 'chatbot'))).resolve()
        self.resume_dir = Path(resume_dir or os.getenv('RESUME_DIR', str(BASE / 'Resume_RAW'))).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'jobs').mkdir(exist_ok=True)
        secret_path = self.data_dir / 'browser.secret'
        if not secret_path.exists():
            try:
                with secret_path.open('xb') as handle:
                    handle.write(secrets.token_bytes(32))
                secret_path.chmod(0o600)
            except FileExistsError:
                pass
        self.secret = secret_path.read_bytes()
        self.engine = None
        self.revision = None
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.initialized = False
        self.health_cache = (0, None)

    # Attach an HMAC signature to the browser owner ID.
    def cookie(self, owner):
        return owner + '.' + hmac.new(self.secret, owner.encode(), hashlib.sha256).hexdigest()

    # Check the cookie's format and signature before accepting its owner ID.
    def owner_from_cookie(self, cookie):
        try:
            owner, signature = cookie.split('.', 1)
            uuid.UUID(owner)
            if hmac.compare_digest(self.cookie(owner).split('.',1)[1], signature):
                return owner
        except (ValueError, AttributeError):
            pass
        return None
    # Create database tables and recover unfinished jobs and their staged file backups.
    def initialize(self):
        with client() as db:
            ensure_schema(db)
            db.execute('''CREATE TABLE IF NOT EXISTS rag_web_chats (
                id text PRIMARY KEY, dataset text NOT NULL, owner text NOT NULL, title text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now())''')
            db.execute('CREATE INDEX IF NOT EXISTS rag_web_chats_owner ON rag_web_chats(dataset, owner, updated_at DESC)')
            db.execute('''CREATE TABLE IF NOT EXISTS rag_web_messages (
                id text PRIMARY KEY, chat_id text NOT NULL REFERENCES rag_web_chats(id) ON DELETE CASCADE,
                role text NOT NULL CHECK(role IN ('user','assistant')), content text NOT NULL DEFAULT '',
                status text NOT NULL DEFAULT 'done', result jsonb, client_id text,
                created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(chat_id, client_id))''')
            db.execute('CREATE INDEX IF NOT EXISTS rag_web_messages_chat ON rag_web_messages(chat_id, created_at, id)')
            db.execute('''CREATE TABLE IF NOT EXISTS rag_web_jobs (
                id text PRIMARY KEY, dataset text NOT NULL, owner text NOT NULL, kind text NOT NULL,
                status text NOT NULL DEFAULT 'queued', stage text NOT NULL DEFAULT 'queued',
                message text NOT NULL DEFAULT 'Waiting to start', spec jsonb NOT NULL,
                result jsonb, error text, retryable boolean NOT NULL DEFAULT false,
                created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now())''')
            db.execute('CREATE INDEX IF NOT EXISTS rag_web_jobs_queue ON rag_web_jobs(dataset,status,created_at)')
            # Restore PDFs from interrupted mutations before making their jobs runnable again.
            pending = db.execute("SELECT * FROM rag_web_jobs WHERE dataset=%s AND status IN ('queued','running')", (COLLECTION,)).fetchall()
            for job in pending:
                backup = self.backup_path(job['id'])
                if backup.exists() and job['spec'].get('source'):
                    atomic_write(self.source_path(job['spec']['source']), backup.read_bytes())
                    backup.unlink()
                db.execute("UPDATE rag_web_jobs SET status='queued',stage='queued',message='Resuming after restart',updated_at=now() WHERE id=%s", (job['id'],))
        self.initialized = True

    # Start the daemon thread that handles chat and resume jobs.
    def start(self):
        self.thread = threading.Thread(target=self.worker, name='resume-desk-worker', daemon=True)
        self.thread.start()

    # Ask the worker to exit and close the engine once it has stopped.
    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=3)
        if not self.thread or not self.thread.is_alive():
            if self.engine:
                self.engine.close()
                self.engine = None

    # Locate the recovery copy associated with a resume mutation job.
    def backup_path(self, job_id):
        return self.data_dir / 'jobs' / (job_id + '.backup')

    # Resolve a stored file path and keep it inside the configured resume directory.
    def source_path(self, source):
        path = (self.resume_dir / source).resolve()
        if not path.is_relative_to(self.resume_dir):
            raise ValueError('The source path is outside the resume folder.')
        return path
    # Namespace the engine's chat memory by browser owner and conversation.
    def session_key(self, owner, chat_id):
        return 'web:' + owner + ':' + chat_id
    # Open the search engine when a chat needs it. Reload its resume list when the database reports a change.
    def get_engine(self):
        with client() as db:
            row = db.execute('SELECT updated_at,ready FROM rag_datasets WHERE name=%s', (COLLECTION,)).fetchone()
            count = db.execute('SELECT count(*) AS n FROM rag_resumes WHERE dataset=%s', (COLLECTION,)).fetchone()['n']
        if not row or not count:
            raise ValueError('Your resume library is empty. Upload a resume first.')
        if not row['ready']:
            raise ValueError('Bulk ingestion is still running. Wait for it to finish.')
        if self.engine:
            try:
                self.engine.db.execute('SELECT 1')
            except psycopg.Error:
                self.engine.close()
                self.engine = None
        if self.engine is None:
            self.engine = ResumeEngine(memory_path=self.data_dir / 'engine_memory.sqlite3')
            self.revision = row['updated_at']
        elif self.revision != row['updated_at']:
            self.engine.refresh()
            self.revision = row['updated_at']
        return self.engine
    # Dispose of the cached engine after a resume mutation commits.
    def invalidate_engine(self):
        if self.engine:
            self.engine.close()
            self.engine = None
        self.revision = None
        self.health_cache = (0, None)

    # Look up a conversation by browser owner and optionally lock it for an update.
    def check_chat(self, db, owner, chat_id, lock=False):
        sql = 'SELECT * FROM rag_web_chats WHERE id=%s AND owner=%s AND dataset=%s'
        row = db.execute(sql + (' FOR UPDATE' if lock else ''), (chat_id,owner,COLLECTION)).fetchone()
        if not row:
            raise HTTPException(404, 'Conversation not found.')
        return row

    # Read a candidate's current database record or return a not-found error.
    def resume(self, db, candidate_id):
        row = db.execute('SELECT * FROM rag_resumes WHERE dataset=%s AND candidate_id=%s', (COLLECTION,candidate_id)).fetchone()
        if not row:
            raise HTTPException(404, 'This resume is no longer in the library.')
        return row

    # Write a job stage and message for the browser's next polling request.
    def progress(self, job_id, stage, message):
        with client() as db:
            db.execute('UPDATE rag_web_jobs SET stage=%s,message=%s,updated_at=now() WHERE id=%s', (stage,message,job_id))
    # Atomically claim the oldest queued job using a PostgreSQL row lock.
    def claim(self):
        with client() as db:
            # Pick and mark a waiting job in one database step so two workers cannot grab it together.
            return db.execute('''UPDATE rag_web_jobs SET status='running',stage='starting',updated_at=now()
                WHERE id=(SELECT id FROM rag_web_jobs WHERE dataset=%s AND status='queued'
                ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *''', (COLLECTION,)).fetchone()
    # Process claimed jobs serially and persist their failures for the browser.
    def worker(self):
        while not self.stop_event.is_set():
            try:
                if not self.initialized:
                    self.initialize()
                job = self.claim()
                if job is None:
                    self.wake.wait(timeout=1)
                    self.wake.clear()
                    continue
                try:
                    # Use the saved job kind to dispatch to the chat or resume handler.
                    if job['kind'] == 'chat':
                        self.run_chat(job)
                    else:
                        self.run_resume(job)
                except Exception as exc:
                    log.exception('Job %s failed', job['id'])
                    # Hold this job until its failure can be recorded after a database outage.
                    while not self.stop_event.is_set():
                        try:
                            self.fail(job, exc)
                            break
                        except psycopg.Error:
                            self.stop_event.wait(2)
            except Exception:
                log.exception('Background worker will retry database connectivity')
                self.wake.wait(timeout=2)
                self.wake.clear()
    # Record the job error and mark its pending assistant message failed when applicable.
    def fail(self, job, exc):
        error = exc.detail if isinstance(exc, HTTPException) else str(exc)
        if isinstance(exc, (psycopg.Error, ConnectionError, httpx.HTTPError)):
            error = 'A local service is unavailable. Check Docker and Ollama, then retry.'
        error = str(error)[:1000]
        with client() as db, db.transaction():
            db.execute("UPDATE rag_web_jobs SET status='failed',stage='failed',message='Could not complete this request',error=%s,retryable=%s,updated_at=now() WHERE id=%s",
                       (error,not isinstance(exc,(ValueError,HTTPException)),job['id']))
            # Replace the pending assistant placeholder with the same error shown on the job.
            if job['kind'] == 'chat':
                db.execute("UPDATE rag_web_messages SET status='failed',content=%s WHERE id=%s", (error,job['spec']['assistant_id']))
    # Call ResumeEngine with saved session context and commit the completed assistant message.
    def run_chat(self, job):
        spec = job['spec']
        with client() as db:
            self.check_chat(db,job['owner'],spec['chat_id'])
        engine = self.get_engine()
        key = self.session_key(job['owner'],spec['chat_id'])
        # Put the chat’s memory back where it was before retrying the same question.
        if 'memory_before' in spec:
            engine._save_memory(key,spec['memory_before'])
        else:
            spec['memory_before'] = engine._load_memory(key)
            with client() as db:
                db.execute('UPDATE rag_web_jobs SET spec=%s WHERE id=%s', (Jsonb(spec),job['id']))
        self.progress(job['id'],'answering','Reviewing resume evidence')
        try:
            # Hand the question to ResumeEngine, which runs its LangGraph workflow.
            result = engine.ask(spec['question'], session_id=key)
            with client() as db, db.transaction():
                db.execute("UPDATE rag_web_messages SET content=%s,status='done',result=%s WHERE id=%s",
                           (result['answer'],Jsonb(result),spec['assistant_id']))
                db.execute('UPDATE rag_web_chats SET updated_at=now() WHERE id=%s', (spec['chat_id'],))
                db.execute("UPDATE rag_web_jobs SET status='ready',stage='ready',message='Answer ready',result=%s,error=NULL,updated_at=now() WHERE id=%s",
                           (Jsonb({'chat_id':spec['chat_id'],'message_id':spec['assistant_id']}),job['id']))
        except Exception:
            engine._save_memory(key,spec['memory_before'])
            raise
    # Process an upload, replacement, text edit or deletion with recoverable file and index writes.
    def run_resume(self, job):
        spec = job['spec']
        kind = job['kind']
        cid = spec['candidate_id']
        stage = Path(spec['staged']) if spec.get('staged') else None
        target = self.source_path(spec['source'])
        backup = self.backup_path(job['id'])
        parent = chunks = vectors = None
        dimension = digest = None
        if kind != 'delete':
            # Turn edited text into a PDF before calling the shared extraction pipeline.
            if kind == 'edit':
                self.progress(job['id'],'validating','Creating the updated PDF')
                text_to_pdf(spec['text'],stage)
            self.progress(job['id'],'chunking','Extracting and chunking the resume')
            with pymupdf.open(stage) as pdf:
                if pdf.needs_pass:
                    raise ValueError('Password-protected PDFs must be unlocked before uploading.')
                if len(pdf)>100:
                    raise ValueError('Upload a resume with at most 100 pages.')
            cid,digest,parent,chunks = prepare_resume(stage,candidate_id=cid,source=spec['source'],
                original_filename=spec['original_filename'])
            with client() as db:
                # Check the file hash before embedding so identical uploads reuse the existing candidate.
                if kind=='upload':
                    duplicate=db.execute('SELECT candidate_id,metadata FROM rag_resumes WHERE dataset=%s AND file_hash=%s LIMIT 1', (COLLECTION,digest)).fetchone()
                    if duplicate:
                        db.execute("UPDATE rag_web_jobs SET status='ready',stage='ready',message='This PDF is already indexed',result=%s,updated_at=now() WHERE id=%s",
                            (Jsonb({'candidate_id':duplicate['candidate_id'],'candidate_name':duplicate['metadata']['candidate_name'],'duplicate':True}),job['id']))
                        stage.unlink(missing_ok=True)
                        return
            # Make the embeddings first. Take the library’s write lock only when the update is ready to save.
            self.progress(job['id'],'embedding',f'Embedding {len(chunks)} chunks')
            vectors,dimension = embed_chunks(chunks, progress=lambda n,total:self.progress(
                job['id'],'embedding',f'Embedded {n} of {total} chunks'))
        with client() as db:
            lock = db.execute('SELECT pg_try_advisory_lock(hashtext(%s)) AS locked', ('resume-rag:' + COLLECTION,)).fetchone()['locked']
            if not lock:
                raise ValueError('Another ingestion is running. Wait for it to finish and try again.')
            changed_file = False
            try:
                manifest = load_manifest(db)
                if manifest and (manifest['recipe']!=RECIPE or not manifest['ready']):
                    raise ValueError('The resume index is incompatible or bulk ingestion is running.')
                if kind != 'delete' and manifest and manifest['embedding_dim'] != dimension:
                    raise ValueError('The embedding dimension differs from the existing index.')
                # Compare the opened hash with the current version before changing an existing resume.
                if kind != 'upload':
                    current = self.resume(db,cid)
                    if current['file_hash'] != spec['expected_hash']:
                        raise HTTPException(409,'This resume changed since you opened it. Reload it and try again.')
                self.progress(job['id'],'saving','Saving the resume and updating search')
                # Save a spare copy of the PDF. The database can undo its rows, but the app has to put the file back.
                if target.exists():
                    atomic_write(backup,target.read_bytes())
                try:
                    with db.transaction():
                        # Delete the parent record and its PDF. The foreign key removes its searchable chunks.
                        if kind=='delete':
                            db.execute('DELETE FROM rag_resumes WHERE dataset=%s AND candidate_id=%s',(COLLECTION,cid))
                            target.unlink(missing_ok=True)
                            changed_file=True
                            result={'candidate_id':cid,'deleted':True}
                        else:
                            if manifest is None:
                                db.execute('INSERT INTO rag_datasets(name,recipe,embedding_dim,ready) VALUES(%s,%s,%s,true)',
                                           (COLLECTION,Jsonb(RECIPE),dimension))
                            save_resume_records(db,cid,digest,parent,chunks,vectors,dimension)
                            atomic_write(target,stage.read_bytes())
                            changed_file=True
                            result={'candidate_id':cid,'candidate_name':parent['metadata']['candidate_name'],
                                    'chunks':len(chunks),'pages':parent['metadata']['pages'],'file_hash':digest}
                        db.execute('UPDATE rag_datasets SET updated_at=now() WHERE name=%s',(COLLECTION,))
                        # Rebuild the word-search numbers before marking the upload, edit or deletion complete.
                        rebuild_bm25(db)
                        db.execute("UPDATE rag_web_jobs SET status='ready',stage='ready',message=%s,result=%s,error=NULL,updated_at=now() WHERE id=%s",
                                   ('Resume removed' if kind=='delete' else 'Saved and ready to chat',Jsonb(result),job['id']))
                except Exception:
                    # Restore the previous file if the database transaction fails after a file change.
                    if changed_file:
                        if backup.exists():atomic_write(target,backup.read_bytes())
                        else:target.unlink(missing_ok=True)
                    backup.unlink(missing_ok=True)
                    raise
            finally:
                db.execute('SELECT pg_advisory_unlock(hashtext(%s))',('resume-rag:' + COLLECTION,))
        backup.unlink(missing_ok=True)
        if stage:stage.unlink(missing_ok=True)
        self.invalidate_engine()

    # Return only the job fields that the browser needs to display status and results.
    def public_job(self,row):
        return {k:row[k] for k in ('id','kind','status','stage','message','result','error','retryable','created_at','updated_at')}
    # Cache brief PostgreSQL and Ollama checks for the status indicator.
    def health(self):
        cached_at,cached=self.health_cache
        if cached and time.monotonic()-cached_at<5:
            return cached
        database=False;count=0;model=False;embedding=False;ready=False
        try:
            with client() as db:
                count=db.execute('SELECT count(*) AS n FROM rag_resumes WHERE dataset=%s',(COLLECTION,)).fetchone()['n']
                manifest=load_manifest(db);ready=bool(manifest and manifest['ready']);database=True
        except Exception:pass
        try:
            response=httpx.get(OLLAMA_URL+'/api/tags',timeout=2)
            response.raise_for_status()
            names=[x['name'] for x in response.json().get('models',[])]
            model=CHAT_MODEL in names or CHAT_MODEL+':latest' in names
            embedding=EMBED_MODEL in names or EMBED_MODEL+':latest' in names
        except Exception:pass
        result={'database':database,'model':model,'embedding':embedding,'ready':ready,
                'resume_count':count,'model_name':CHAT_MODEL,'max_upload_mb':MAX_BYTES//1024//1024}
        self.health_cache=(time.monotonic(),result)
        return result


# Register FastAPI routes and middleware around one local workspace.
def create_app(*,data_dir=None,resume_dir=None):
    workspace=Workspace(data_dir,resume_dir)

    # Take the workspace process lock, start its worker and release resources on shutdown.
    @asynccontextmanager
    async def lifespan(app):
        process_lock=(workspace.data_dir/'server.lock').open('a')
        try:
            fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            process_lock.close()
            raise RuntimeError('Resume Desk is already running for this workspace. Use its existing browser tab.')
        try:
            try:
                await asyncio.to_thread(workspace.initialize)
            except Exception:
                log.exception('Database setup unavailable; the UI will show service status and retry')
            workspace.start()
            yield
        finally:
            workspace.close()
            process_lock.close()

    app=FastAPI(title='Resume Desk',version='1.0.0',lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.workspace=workspace
    allowed=os.getenv('CHAT_ALLOWED_HOSTS','localhost,127.0.0.1,::1,testserver').split(',')
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=[x.strip() for x in allowed])
    # Validate browser writes, establish an owner cookie and attach response security headers.
    @app.middleware('http')
    async def browser_session(request,call_next):
        if request.headers.get('content-length','').isdigit() and int(request.headers['content-length'])>MAX_BYTES+1024*1024:
            return JSONResponse({'detail':'The upload is too large.'},status_code=413)
        if request.method in ('POST','PUT','PATCH','DELETE'):
            origin=request.headers.get('origin')
            if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'This request must come from the local chat page.'},status_code=403)
        owner=workspace.owner_from_cookie(request.cookies.get('resume_desk',''))
        fresh=owner is None
        request.state.owner=owner or str(uuid.uuid4())
        response=await call_next(request)
        if fresh:
            response.set_cookie('resume_desk',workspace.cookie(request.state.owner),httponly=True,
                                samesite='strict',secure=request.url.scheme=='https',max_age=60*60*24*365)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        return response

    # Return a concise service-unavailable response for PostgreSQL errors.
    @app.exception_handler(psycopg.Error)
    async def database_error(request,exc):
        log.error('Database request failed: %s',type(exc).__name__)
        return JSONResponse({'detail':'PostgreSQL is unavailable. Start Docker and run docker compose up -d.'},status_code=503)
    # Serve the HTML that initializes the browser chatbot.
    @app.get('/',include_in_schema=False)
    def index():return FileResponse(BASE/'web'/'index.html')

    # Expose model, database and resume-library status to the frontend.
    @app.get('/api/status')
    def status():return workspace.health()

    # List the current browser owner's saved conversation titles.
    @app.get('/api/chats')
    def chats(request:Request):
        with client() as db:
            return db.execute('SELECT id,title,created_at,updated_at FROM rag_web_chats WHERE dataset=%s AND owner=%s ORDER BY updated_at DESC,id',
                              (COLLECTION,request.state.owner)).fetchall()
    # Create a conversation ID and persist its title and owner.
    @app.post('/api/chats',status_code=201)
    def new_chat(request:Request,payload:ChatInput):
        with client() as db:
            return db.execute('INSERT INTO rag_web_chats(id,dataset,owner,title) VALUES(%s,%s,%s,%s) RETURNING id,title,created_at,updated_at',
                (str(uuid.uuid4()),COLLECTION,request.state.owner,payload.title.strip() or 'New conversation')).fetchone()

    # Return the selected conversation's messages and pending answer jobs.
    @app.get('/api/chats/{chat_id}')
    def chat(request:Request,chat_id:str):
        with client() as db:
            row=workspace.check_chat(db,request.state.owner,chat_id)
            messages=db.execute('SELECT id,role,content,status,result,created_at FROM rag_web_messages WHERE chat_id=%s ORDER BY created_at,id',(chat_id,)).fetchall()
            pending=db.execute("SELECT id,stage,message,status FROM rag_web_jobs WHERE dataset=%s AND owner=%s AND spec->>'chat_id'=%s AND status IN ('queued','running') ORDER BY created_at",
                               (COLLECTION,request.state.owner,chat_id)).fetchall()
            return {'id':row['id'],'title':row['title'],'messages':messages,'pending':pending}

    # Update the title after checking who owns the conversation.
    @app.patch('/api/chats/{chat_id}')
    def rename_chat(request:Request,chat_id:str,payload:ChatInput):
        with client() as db:
            workspace.check_chat(db,request.state.owner,chat_id)
            return db.execute('UPDATE rag_web_chats SET title=%s,updated_at=now() WHERE id=%s RETURNING id,title',
                              (payload.title.strip() or 'New conversation',chat_id)).fetchone()

    # Remove saved conversation data and engine memory once pending work is finished.
    @app.delete('/api/chats/{chat_id}')
    def delete_chat(request:Request,chat_id:str):
        with client() as db,db.transaction():
            workspace.check_chat(db,request.state.owner,chat_id,lock=True)
            if db.execute("SELECT 1 FROM rag_web_jobs WHERE dataset=%s AND spec->>'chat_id'=%s AND status IN ('queued','running')",(COLLECTION,chat_id)).fetchone():
                raise HTTPException(409,'Wait for the current answer before deleting this conversation.')
            db.execute("DELETE FROM rag_web_jobs WHERE dataset=%s AND owner=%s AND spec->>'chat_id'=%s",(COLLECTION,request.state.owner,chat_id))
            db.execute('DELETE FROM rag_web_chats WHERE id=%s',(chat_id,))
        path=workspace.data_dir/'engine_memory.sqlite3'
        if path.exists():
            import sqlite3
            with sqlite3.connect(path) as memory:
                memory.execute('DELETE FROM sessions WHERE id=?',(workspace.session_key(request.state.owner,chat_id),))
        return {'deleted':True}
    # Save an accepted question and create one durable answer job per client request ID.
    @app.post('/api/chats/{chat_id}/messages',status_code=202)
    def send_message(request:Request,chat_id:str,payload:MessageInput):
        question=payload.content.strip()
        if not question:raise HTTPException(422,'Write a question first.')
        owner=request.state.owner
        with client() as db,db.transaction():
            row=workspace.check_chat(db,owner,chat_id,lock=True)
            duplicate=db.execute('SELECT id FROM rag_web_messages WHERE chat_id=%s AND client_id=%s',(chat_id,payload.client_id)).fetchone()
            # Reuse the first job for this client request ID and avoid saving a second question.
            if duplicate:
                job=db.execute("SELECT id FROM rag_web_jobs WHERE dataset=%s AND owner=%s AND spec->>'user_message_id'=%s",(COLLECTION,owner,duplicate['id'])).fetchone()
                return {'job_id':job['id'] if job else None,'duplicate':True}
            if db.execute("SELECT 1 FROM rag_web_messages WHERE chat_id=%s AND status='pending'",(chat_id,)).fetchone():
                raise HTTPException(409,'An answer is already in progress for this conversation.')
            user_id,assistant_id,job_id=(str(uuid.uuid4()) for _ in range(3))
            db.execute("INSERT INTO rag_web_messages(id,chat_id,role,content,client_id) VALUES(%s,%s,'user',%s,%s)",
                       (user_id,chat_id,question,payload.client_id))
            db.execute("INSERT INTO rag_web_messages(id,chat_id,role,status,created_at) VALUES(%s,%s,'assistant','pending',clock_timestamp())",(assistant_id,chat_id))
            spec={'chat_id':chat_id,'question':question,'assistant_id':assistant_id,'user_message_id':user_id}
            db.execute("INSERT INTO rag_web_jobs(id,dataset,owner,kind,spec,message) VALUES(%s,%s,%s,'chat',%s,'Waiting for the local model')",
                       (job_id,COLLECTION,owner,Jsonb(spec)))
            db.execute('UPDATE rag_web_chats SET title=%s,updated_at=now() WHERE id=%s',
                       (question[:72] if row['title']=='New conversation' else row['title'],chat_id))
        workspace.wake.set()
        return {'job_id':job_id,'message_id':assistant_id}

    # Return the latest job statuses visible to this browser.
    @app.get('/api/jobs')
    def jobs(request:Request):
        with client() as db:
            rows=db.execute('SELECT * FROM rag_web_jobs WHERE dataset=%s AND owner=%s ORDER BY created_at DESC LIMIT 30',
                            (COLLECTION,request.state.owner)).fetchall()
            return [workspace.public_job(r) for r in rows]

    # Look up a single owned job for status polling.
    @app.get('/api/jobs/{job_id}')
    def job(request:Request,job_id:str):
        with client() as db:
            row=db.execute('SELECT * FROM rag_web_jobs WHERE id=%s AND owner=%s AND dataset=%s',(job_id,request.state.owner,COLLECTION)).fetchone()
            if not row:raise HTTPException(404,'Request not found.')
            return workspace.public_job(row)

    # Requeue a retryable failure only while its original chat turn can be safely retried.
    @app.post('/api/jobs/{job_id}/retry',status_code=202)
    def retry_job(request:Request,job_id:str):
        with client() as db,db.transaction():
            row=db.execute('SELECT * FROM rag_web_jobs WHERE id=%s AND owner=%s AND dataset=%s FOR UPDATE',
                           (job_id,request.state.owner,COLLECTION)).fetchone()
            if not row:raise HTTPException(404,'Request not found.')
            if row['status']!='failed' or not row['retryable']:
                raise HTTPException(409,'This request cannot be retried. Submit a new request instead.')
            if row['kind']=='chat':
                spec=row['spec']
                workspace.check_chat(db,request.state.owner,spec['chat_id'],lock=True)
                latest=db.execute("SELECT id FROM rag_web_messages WHERE chat_id=%s AND role='assistant' ORDER BY created_at DESC,id DESC LIMIT 1",(spec['chat_id'],)).fetchone()
                if not latest or latest['id']!=spec['assistant_id']:
                    raise HTTPException(409,'This conversation has moved on. Ask your question again.')
                db.execute("UPDATE rag_web_messages SET status='pending',content='' WHERE id=%s",(spec['assistant_id'],))
            db.execute("UPDATE rag_web_jobs SET status='queued',stage='queued',message='Waiting to retry',error=NULL,retryable=false,updated_at=now() WHERE id=%s",(job_id,))
        workspace.wake.set()
        return {'job_id':job_id}

    # Match the library search against resume text and metadata, then return one page.
    @app.get('/api/resumes')
    def resumes(q:str='',offset:int=0,limit:int=40):
        if offset<0 or not 1<=limit<=100 or len(q)>200:raise HTTPException(422,'Invalid search or page size.')
        params=(COLLECTION,'%'+q+'%','%'+q+'%','%'+q+'%')
        where="dataset=%s AND (metadata->>'candidate_name' ILIKE %s OR metadata->>'source' ILIKE %s OR page_content ILIKE %s)"
        with client() as db:
            total=db.execute('SELECT count(*) AS n FROM rag_resumes WHERE '+where,params).fetchone()['n']
            rows=db.execute('SELECT candidate_id,file_hash,page_content,metadata FROM rag_resumes WHERE '+where+
                            " ORDER BY lower(metadata->>'candidate_name'),candidate_id LIMIT %s OFFSET %s",params+(limit,offset)).fetchall()
        return {'total':total,'items':[{'candidate_id':r['candidate_id'],'file_hash':r['file_hash'],
            'name':r['metadata']['candidate_name'],'filename':r['metadata'].get('original_filename',Path(r['metadata']['source']).name),
            'pages':r['metadata']['pages'],'chunks':r['metadata']['chunks'],**profile_fields(r)} for r in rows]}
    # Stream an uploaded file to temporary storage and check the size and PDF signature.
    async def stage_upload(file:UploadFile):
        filename=Path((file.filename or '').replace('\\','/')).name
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(422,'Please select a PDF resume.')
        staged=workspace.data_dir/'jobs'/(str(uuid.uuid4())+'.pdf')
        size=0
        try:
            with staged.open('wb') as handle:
                while piece:=await file.read(1024*1024):
                    size+=len(piece)
                    if size>MAX_BYTES:raise HTTPException(413,f'The PDF must be smaller than {MAX_BYTES//1024//1024} MB.')
                    handle.write(piece)
            if size==0 or b'%PDF-' not in staged.read_bytes()[:1024]:
                raise HTTPException(422,'This file is not a valid PDF.')
            return staged,filename
        except Exception:
            staged.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
    # Persist the requested resume mutation and wake the background worker.
    def queue_resume(owner,kind,spec):
        job_id=str(uuid.uuid4())
        with client() as db:
            db.execute('INSERT INTO rag_web_jobs(id,dataset,owner,kind,spec) VALUES(%s,%s,%s,%s,%s)',
                       (job_id,COLLECTION,owner,kind,Jsonb(spec)))
        workspace.wake.set()
        return {'job_id':job_id,'candidate_id':spec['candidate_id']}
    # Stage the new PDF and queue indexing under a fresh candidate ID.
    @app.post('/api/resumes',status_code=202)
    async def upload(request:Request,file:UploadFile=File(...)):
        staged,filename=await stage_upload(file)
        cid='WEB_'+uuid.uuid4().hex[:16]
        try:
            return queue_resume(request.state.owner,'upload',{'candidate_id':cid,'source':cid+'.pdf',
                'staged':str(staged),'original_filename':filename})
        except Exception:
            staged.unlink(missing_ok=True)
            raise
    # Stage a replacement PDF under the candidate's existing identity and version check.
    @app.put('/api/resumes/{candidate_id:path}/file',status_code=202)
    async def replace(request:Request,candidate_id:str,file:UploadFile=File(...),expected_hash:str=Form(...)):
        with client() as db:row=workspace.resume(db,candidate_id)
        staged,filename=await stage_upload(file)
        try:
            return queue_resume(request.state.owner,'replace',{'candidate_id':candidate_id,'source':row['metadata']['source'],
                'expected_hash':expected_hash,'staged':str(staged),'original_filename':filename})
        except Exception:
            staged.unlink(missing_ok=True)
            raise
    # Serve the saved original PDF as an attachment.
    @app.get('/api/resumes/{candidate_id:path}/file')
    def download(candidate_id:str):
        with client() as db:row=workspace.resume(db,candidate_id)
        path=workspace.source_path(row['metadata']['source'])
        if not path.is_file():raise HTTPException(404,'The source PDF is missing. You can still view its indexed text.')
        return FileResponse(path,media_type='application/pdf',filename=row['metadata'].get('original_filename',path.name))

    # Validate edited text and queue generation of a replacement PDF and index.
    @app.patch('/api/resumes/{candidate_id:path}',status_code=202)
    def edit(request:Request,candidate_id:str,payload:EditInput):
        text=payload.text.strip()
        if not text:raise HTTPException(422,'The resume text cannot be empty.')
        with client() as db:row=workspace.resume(db,candidate_id)
        staged=workspace.data_dir/'jobs'/(str(uuid.uuid4())+'.pdf')
        return queue_resume(request.state.owner,'edit',{'candidate_id':candidate_id,'source':row['metadata']['source'],
            'expected_hash':payload.expected_hash,'staged':str(staged),'text':text,
            'original_filename':row['metadata'].get('original_filename',Path(row['metadata']['source']).name)})

    # Queue deletion with the current version hash to detect intervening changes.
    @app.delete('/api/resumes/{candidate_id:path}',status_code=202)
    def remove(request:Request,candidate_id:str,expected_hash:str):
        with client() as db:row=workspace.resume(db,candidate_id)
        return queue_resume(request.state.owner,'delete',{'candidate_id':candidate_id,'source':row['metadata']['source'],
                                                         'expected_hash':expected_hash})

    # Return the full indexed text, editable text and parsed profile for one candidate.
    @app.get('/api/resumes/{candidate_id:path}')
    def resume_detail(candidate_id:str):
        with client() as db:row=workspace.resume(db,candidate_id)
        return {'candidate_id':candidate_id,'file_hash':row['file_hash'],'metadata':row['metadata'],
                'text':row['page_content'],'edit_text':PAGE_MARKER.sub('',row['page_content']).strip(),
                **profile_fields(row)}

    app.mount('/static',StaticFiles(directory=BASE/'web'),name='static')
    return app


app=create_app()

if __name__=='__main__':
    import uvicorn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=int(os.getenv('CHAT_PORT','8000')))
    args=parser.parse_args()
    uvicorn.run(app,host=args.host,port=args.port)
