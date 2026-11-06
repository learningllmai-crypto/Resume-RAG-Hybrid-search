#!/bin/zsh
# Run from the project folder and stop if a required setup command fails.
set -e
cd -- "${0:A:h}"
# Require Docker, then wait for its daemon before starting PostgreSQL.
if ! command -v docker >/dev/null 2>&1; then
  echo 'Install Docker Desktop, then open this launcher again.'
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  open -a Docker
  echo 'Waiting for Docker Desktop…'
  for attempt in {1..60}; do
    if docker info >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
# Give a fresh local database its own password on the first launch.
if [[ ! -f .env ]]; then
  python3 - <<'PY'
import os
import secrets
from pathlib import Path

config = Path('.env.example').read_text()
config = config.replace('replace-with-a-local-password', secrets.token_hex(24))
os.umask(0o077)
with Path('.env').open('x') as handle:
    handle.write(config)
PY
fi

# Start the database and wait for its health check.
docker compose up -d --wait --wait-timeout 90
# Create the local Python environment when it is missing.
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
# Reinstall dependencies only when the requirements file changes.
requirements_digest=$(shasum -a 256 requirements.txt | cut -d ' ' -f 1)
if [[ ! -f .venv/.chatbot-requirements ]] || [[ "$(cat .venv/.chatbot-requirements)" != "$requirements_digest" ]]; then
  .venv/bin/python -m pip install -r requirements.txt
  print -r -- "$requirements_digest" > .venv/.chatbot-requirements
fi
# Wait for Ollama, download missing models, then index any bundled PDFs on first run.
.venv/bin/python -u - <<'PY'
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from ingest_resumes import BASE, CHAT_PROVIDER, CHAT_MODEL, EMBED_MODEL, OLLAMA_URL, client, load_manifest
from resume_engine import chat_api_key

if CHAT_PROVIDER == 'openai':
    try:
        chat_api_key()
    except ValueError as error:
        raise SystemExit(str(error)) from None

with httpx.Client(base_url=OLLAMA_URL, timeout=2) as http:
    for attempt in range(31):
        try:
            response = http.get('/api/tags')
            response.raise_for_status()
            installed = {item['name'] for item in response.json()['models']}
            break
        except httpx.HTTPError:
            if attempt == 30:
                raise SystemExit('Ollama is unavailable. Start the Ollama app, then reopen this launcher.')
            if attempt == 0:
                subprocess.run(['open', '-a', 'Ollama'], check=False)
                print('Waiting for Ollama…')
            time.sleep(1)

ollama_models = (CHAT_MODEL, EMBED_MODEL) if CHAT_PROVIDER == 'ollama' else (EMBED_MODEL,)
for model in dict.fromkeys(ollama_models):
    if model not in installed and model + ':latest' not in installed:
        ollama = shutil.which('ollama') or '/Applications/Ollama.app/Contents/Resources/ollama'
        if not Path(ollama).is_file():
            raise SystemExit('Install Ollama on this Mac to download the required models.')
        print(f'Downloading {model}…')
        subprocess.run([ollama, 'pull', model], env={**os.environ, 'OLLAMA_HOST': OLLAMA_URL}, check=True)

with client() as db:
    manifest = load_manifest(db)
resume_dir = Path(os.getenv('RESUME_DIR', str(BASE / 'Resume_RAW')))
if (not manifest or not manifest['ready']) and any(
    path.is_file() and path.suffix.lower() == '.pdf' for path in resume_dir.rglob('*')
):
    print('Indexing the bundled resumes…')
    subprocess.run([sys.executable, 'ingest_resumes.py', '--folder', str(resume_dir)], check=True)

# Use the same port for the browser and the server, including settings read from .env.
port = int(os.getenv('CHAT_PORT', '8000'))
url = f'http://127.0.0.1:{port}'
print(f'Resume Desk: {url} — press Control-C to stop.')
subprocess.Popen(['sh', '-c', 'sleep 3; open "$1"', 'resume-desk', url])
os.execv(sys.executable, [sys.executable, 'app.py'])
PY
