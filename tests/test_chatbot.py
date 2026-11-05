"""Real PostgreSQL + Ollama integration test; uses a disposable dataset, never the corpus.
Run: .venv/bin/python tests/test_chatbot.py
"""
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['RESUME_COLLECTION']='web_test_'+uuid.uuid4().hex
temporary=tempfile.TemporaryDirectory(prefix='resume-desk-test-')
os.environ['CHAT_DATA_DIR']=str(Path(temporary.name)/'data')
os.environ['RESUME_DIR']=str(Path(temporary.name)/'resumes')

from fastapi.testclient import TestClient
from app import create_app, text_to_pdf, atomic_write
from ingest_resumes import BM25_SETTINGS, COLLECTION, bm25_query_vector, client, load_bm25_vocabulary
from psycopg.types.json import Jsonb

ALICE='''Alice Testwood
Python Developer | Mumbai, Maharashtra
Email: alice@example.com
Professional Summary
Python developer with 7 years of professional experience.
Skills
Python, Docker, PostgreSQL, FastAPI
Projects
Atlas Resume Search
Built an AI resume search engine using Python, Docker and PostgreSQL.
Experience
Senior Developer at Example Systems | 2019 - Present
Education
B.Tech Computer Science
'''
BOB='''Bob Testfield
Java Developer | Pune, Maharashtra
Email: bob@example.com
Professional Summary
Java developer with 3 years of professional experience.
Skills
Java, Spring Boot, REST APIs
Projects
Beacon Payment API
Built a Java payment API using Spring Boot.
Education
B.Tech Computer Science
'''
# Exercise the web API and retrieval against temporary PostgreSQL and Ollama test data.
class ChatbotIntegration(unittest.TestCase):
    # Poll a submitted job until it reaches the expected terminal state or times out.
    def wait(self,browser,job_id,expected='ready'):
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            response=browser.get('/api/jobs/'+job_id)
            self.assertEqual(response.status_code,200,response.text)
            job=response.json()
            if job['status'] in ('ready','failed'):
                self.assertEqual(job['status'],expected,job)
                return job
            time.sleep(.2)
        self.fail('Timed out waiting for '+job_id)

    # Turn fixture text into PDF bytes using the same renderer as the application.
    def pdf(self,name,text):
        path=Path(temporary.name)/name
        text_to_pdf(text,path)
        return path.read_bytes()

    # Post a fixture PDF and wait for the worker to make it searchable.
    def upload(self,browser,name,text):
        pdf=self.pdf(name,text)
        result=browser.post('/api/resumes',files={'file':(name,pdf,'application/pdf')})
        self.assertEqual(result.status_code,202,result.text)
        job=self.wait(browser,result.json()['job_id'])
        return job['result']['candidate_id'],pdf

    # Post the same request twice to check deduplication before reading the finished answer.
    def ask(self,browser,chat,question):
        token=str(uuid.uuid4())
        result=browser.post(f'/api/chats/{chat}/messages',json={'content':question,'client_id':token})
        self.assertEqual(result.status_code,202,result.text)
        duplicate=browser.post(f'/api/chats/{chat}/messages',json={'content':question,'client_id':token})
        self.assertEqual(duplicate.json()['job_id'],result.json()['job_id'])
        self.wait(browser,result.json()['job_id'])
        messages=browser.get('/api/chats/'+chat).json()['messages']
        self.assertEqual(messages[-1]['status'],'done')
        print('\nQUESTION:',question,'\nANSWER:',messages[-1]['content'],flush=True)
        return messages[-1]['result']

    # Verify indexing, BM25 scores, chat isolation, file updates and worker recovery.
    def test_complete_workflow(self):
        app=create_app()
        with TestClient(app) as browser:
            self.assertEqual(browser.get('/').status_code,200)
            self.assertEqual(browser.get('/api/status').json()['resume_count'],0)
            alice,original=self.upload(browser,'alice.pdf',ALICE)
            bob,_=self.upload(browser,'bob.pdf',BOB)
            with client() as db:
                # Score a repeated query term and compare the sparse inner product with scalar BM25.
                query = bm25_query_vector(db, 'Atlas Atlas')
                rows = db.execute("""SELECT -(bm25_embedding <#> %s) AS score,
                    ARRAY(SELECT unnest(lexemes) FROM ts_debug('english', search_text)) AS tokens
                    FROM rag_chunks WHERE dataset=%s""", (query, COLLECTION)).fetchall()
                term = db.execute("SELECT (tsvector_to_array(to_tsvector('english', 'Atlas')))[1] AS term").fetchone()['term']
                df = sum(term in row['tokens'] for row in rows)
                self.assertGreater(df, 0)
                self.assertLess(df, len(rows))
                idf = math.log(1 + (len(rows) - df + .5) / (df + .5))
                avgdl = sum(len(row['tokens']) for row in rows) / len(rows)
                k1, b = BM25_SETTINGS['k1'], BM25_SETTINGS['b']
                for row in rows:
                    tf = row['tokens'].count(term)
                    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(row['tokens']) / avgdl))
                    self.assertAlmostEqual(row['score'], expected, places=6)
                self.assertIsNone(bm25_query_vector(db, 'zzznosuchlexemezzz'))
            self.assertEqual(browser.get('/api/resumes').json()['total'],2)
            self.assertEqual(browser.get('/api/resumes/'+alice+'/file').content,original)
            duplicate=browser.post('/api/resumes',files={'file':('same.pdf',original,'application/pdf')})
            self.assertTrue(self.wait(browser,duplicate.json()['job_id'])['result']['duplicate'])
            self.assertEqual(browser.get('/api/resumes').json()['total'],2)
            self.assertEqual(browser.post('/api/resumes',files={'file':('fake.pdf',b'bad','application/pdf')}).status_code,422)
            blank=Path(temporary.name)/'blank.pdf'
            import pymupdf
            pdf=pymupdf.open();pdf.new_page();pdf.save(blank);pdf.close()
            bad=browser.post('/api/resumes',files={'file':('blank.pdf',blank.read_bytes(),'application/pdf')})
            self.wait(browser,bad.json()['job_id'],'failed')
            chat=browser.post('/api/chats',json={}).json()['id']
            result=self.ask(browser,chat,'Find Python and Docker developers in Mumbai with at least five years of experience.')
            self.assertIn(alice,result['candidate_ids'])
            self.assertNotIn(bob,result['candidate_ids'])
            self.assertEqual(result['audit']['lexical_algorithm'], 'BM25 sparsevec inner product')
            self.assertGreater(result['audit']['lexical_chunks_scored'], 0)
            result=self.ask(browser,chat,'What projects have they built?')
            self.assertIn('Atlas',result['answer'])
            self.assertNotIn('Beacon',result['answer'])
            result=self.ask(browser,chat,'Show the full resume for Alice Testwood.')
            self.assertIn('Education',result['answer'])
            self.assertIn('alice@example.com',result['answer'])
            old=browser.get('/api/chats/'+chat).json()
            self.assertEqual(len(old['messages']),6)
            new=browser.post('/api/chats',json={}).json()['id']
            self.assertEqual(browser.get('/api/chats/'+new).json()['messages'],[])
            self.assertEqual(len(browser.get('/api/chats').json()),2)
            outsider=TestClient(app)
            self.assertEqual(outsider.get('/api/chats/'+chat).status_code,404)
            self.assertEqual(outsider.get('/api/chats').json(),[])
            self.assertEqual(browser.post('/api/chats',json={},headers={'origin':'https://wrong.example'}).status_code,403)
            detail=browser.get('/api/resumes/'+alice).json()
            replacement=self.pdf('updated.pdf',ALICE.replace('Python, Docker, PostgreSQL, FastAPI','Python, Docker, PostgreSQL, FastAPI, Rust'))
            task=browser.put('/api/resumes/'+alice+'/file',data={'expected_hash':detail['file_hash']},files={'file':('updated.pdf',replacement,'application/pdf')})
            self.wait(browser,task.json()['job_id'])
            current=browser.get('/api/resumes/'+alice).json()
            self.assertNotEqual(current['file_hash'],detail['file_hash'])
            self.assertIn('Rust',current['text'])
            stale=browser.patch('/api/resumes/'+alice,json={'text':ALICE,'expected_hash':detail['file_hash']})
            self.assertIn('changed',self.wait(browser,stale.json()['job_id'],'failed')['error'])
            edited=ALICE.replace('Python, Docker, PostgreSQL, FastAPI','Python, Docker, PostgreSQL, FastAPI, Kubernetes')
            task=browser.patch('/api/resumes/'+alice,json={'text':edited,'expected_hash':current['file_hash']})
            self.wait(browser,task.json()['job_id'])
            # Capture the current version before injecting a file-write failure.
            before=browser.get('/api/resumes/'+alice).json()
            target=app.state.workspace.resume_dir/(alice+'.pdf')
            fired=[]
            # Interrupt the first target-file write so the test can verify rollback and retry.
            def disk_failure(path,data):
                if path==target and not fired:
                    fired.append(True)
                    raise OSError('Simulated temporary disk failure')
                return atomic_write(path,data)
            with patch('app.atomic_write',side_effect=disk_failure):
                task=browser.patch('/api/resumes/'+alice,json={'text':edited+'\nAdditional delivery notes.','expected_hash':before['file_hash']})
                failed=self.wait(browser,task.json()['job_id'],'failed')
            self.assertTrue(failed['retryable'])
            self.assertEqual(browser.get('/api/resumes/'+alice).json()['file_hash'],before['file_hash'])
            retry=browser.post('/api/jobs/'+task.json()['job_id']+'/retry')
            self.assertEqual(retry.status_code,202,retry.text)
            self.wait(browser,task.json()['job_id'])
            result=self.ask(browser,new,'How many candidates know Kubernetes?')
            self.assertIn(alice,result['eligible_candidate_ids'])
            self.assertNotIn(bob,result['eligible_candidate_ids'])
            current=browser.get('/api/resumes/'+alice).json()
            saved=browser.get('/api/resumes/'+alice+'/file').content
            with pymupdf.open(stream=saved,filetype='pdf') as pdf:
                self.assertIn('Kubernetes',''.join(p.get_text() for p in pdf))
            task=browser.delete('/api/resumes/'+alice,params={'expected_hash':current['file_hash']})
            self.wait(browser,task.json()['job_id'])
            self.assertEqual(browser.get('/api/resumes/'+alice).status_code,404)
            self.assertFalse((app.state.workspace.resume_dir/(alice+'.pdf')).exists())
            result=self.ask(browser,new,'How many candidates know Kubernetes?')
            self.assertEqual(result['eligible_candidate_ids'],[])
            self.assertEqual(len(browser.get('/api/chats/'+chat).json()['messages']),6)
            detail=browser.get('/api/resumes/'+bob).json()
            task=browser.delete('/api/resumes/'+bob,params={'expected_hash':detail['file_hash']})
            self.wait(browser,task.json()['job_id'])
            self.assertEqual(browser.get('/api/resumes').json()['total'],0)
            with client() as db:
                self.assertEqual(db.execute('SELECT count(*) AS n FROM rag_chunks WHERE dataset=%s',(COLLECTION,)).fetchone()['n'],0)
                self.assertEqual(load_bm25_vocabulary(db), {})
            browser.patch('/api/chats/'+chat,json={'title':'Saved recruitment shortlist'})
            cookie=browser.cookies.get('resume_desk')
            owner=app.state.workspace.owner_from_cookie(cookie)
        staged=Path(temporary.name)/'recover.pdf'
        text_to_pdf(BOB,staged)
        recovery_id=str(uuid.uuid4())
        with client() as db:
            db.execute("INSERT INTO rag_web_jobs(id,dataset,owner,kind,status,spec) VALUES(%s,%s,%s,'upload','running',%s)",
                (recovery_id,COLLECTION,owner,Jsonb({'candidate_id':'WEB_recovery','source':'WEB_recovery.pdf',
                    'staged':str(staged),'original_filename':'recover.pdf'})))
        with TestClient(create_app()) as reopened:
            reopened.cookies.set('resume_desk',cookie)
            self.wait(reopened,recovery_id)
            self.assertEqual(reopened.get('/api/resumes').json()['total'],1)
            self.assertEqual(reopened.get('/api/chats/'+chat).json()['title'],'Saved recruitment shortlist')
            self.assertEqual(len(reopened.get('/api/chats/'+chat).json()['messages']),6)
            self.assertEqual(reopened.delete('/api/chats/'+chat).status_code,200)
            self.assertEqual(reopened.get('/api/chats/'+chat).status_code,404)


if __name__=='__main__':
    try:
        result=unittest.main(verbosity=2,exit=False).result
    finally:
        with client() as db:
            db.execute('DELETE FROM rag_web_jobs WHERE dataset=%s',(COLLECTION,))
            db.execute('DELETE FROM rag_web_chats WHERE dataset=%s',(COLLECTION,))
            db.execute('DELETE FROM rag_datasets WHERE name=%s',(COLLECTION,))
        temporary.cleanup()
    sys.exit(0 if result.wasSuccessful() else 1)
