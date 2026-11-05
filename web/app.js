'use strict';
// Look up page controls by ID so event handlers can reuse the same elements.
const $ = id => document.getElementById(id);
// Keep the selected chat, library page and running jobs in browser memory.
const state = {chats: [], chatId: null, chat: null, jobs: [], resumes: [], selected: null, editing: false, sending: false, seenJobs: new Set(), searchTimer: null};
// Use a symbol from the HTML icon sprite to build an inline SVG.
const icon = name => {const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');const use=document.createElementNS('http://www.w3.org/2000/svg','use');use.setAttribute('href','#i-'+name);svg.append(use);return svg;};
// Create DOM nodes with textContent so resume and answer text renders literally.
const el = (tag, cls, text) => {const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
// Show a short-lived notice after an action or an error.
function toast(text, error=false){const node=el('div','toast'+(error?' error':''),text);$('toasts').append(node);setTimeout(()=>node.remove(),error?8500:5000);}
// Send same-origin HTTP requests and decode the server's response or error.
async function api(path, options={}){
  const init={credentials:'same-origin',...options};
  // Leave multipart encoding to the browser so it can set the upload boundary.
  if(init.body && !(init.body instanceof FormData)){init.headers={'Content-Type':'application/json',...init.headers};init.body=JSON.stringify(init.body);}
  const response=await fetch(path,init);
  let data;try{data=await response.json();}catch{data={detail:'The server returned an unexpected response.'};}
  if(!response.ok){const error=new Error(typeof data.detail==='string'?data.detail:Array.isArray(data.detail)?data.detail.map(e=>e.msg).join('; '):'Could not complete this request.');error.status=response.status;throw error;}
  return data;
}
// Escape search text and record IDs before putting them in a URL.
const encoded = value => encodeURIComponent(value);
// Close the conversation drawer on narrow screens.
function closeSidebar(){$('sidebar').classList.remove('open');}
// Gate the send button on both the composer contents and the pending reply state.
function updateSend(){const pending=Boolean(state.chat?.pending?.length);$('sendButton').disabled=!$('question').value.trim()||state.sending||pending;$('composerHint').textContent=pending?'Answer in progress':'Enter to send';}
// Reset the message panel for a workspace with no conversation selected.
function showWelcome(){state.chat=null;$('messages').replaceChildren();$('welcome').hidden=false;$('chatTitle').textContent='Workspace chat';updateSend();}
// Reload saved chats from FastAPI before rebuilding the sidebar.
async function refreshChats(){state.chats=await api('/api/chats');renderChats();}
// Render each chat with controls that open or manage its saved conversation.
function renderChats(){
  const nav=$('chatList');nav.replaceChildren();
  if(!state.chats.length){nav.append(el('p','empty-chats','Your conversations will appear here. Start a new one whenever you like.'));return;}
  for(const chat of state.chats){
    const row=el('div','chat-item'+(chat.id===state.chatId?' active':''));
    const button=el('button','chat-open');button.append(icon('chat'),el('span','',chat.title));button.title=chat.title;button.addEventListener('click',()=>selectChat(chat.id));
    const menu=el('button','chat-menu');menu.setAttribute('aria-label','Manage '+chat.title);menu.title='Rename or delete conversation';menu.append(icon('more'));
    menu.addEventListener('click',async()=>{
      const choice=window.prompt('Rename this conversation, or type DELETE to remove it:',chat.title);
      if(choice===null)return;
      try{
        if(choice==='DELETE'){
          if(!window.confirm('Delete this conversation and its saved messages?'))return;
          await api('/api/chats/'+encoded(chat.id),{method:'DELETE'});
          if(state.chatId===chat.id){state.chatId=null;localStorage.removeItem('resumeDesk.chat');showWelcome();}
        }else if(choice.trim()){await api('/api/chats/'+encoded(chat.id),{method:'PATCH',body:{title:choice.trim()}});if(state.chatId===chat.id)$('chatTitle').textContent=choice.trim();}
        await refreshChats();
      }catch(error){toast(error.message,true);}
    });
    row.append(button,menu);nav.append(row);
  }
}
// Ask the backend for a new conversation and make it the current one.
async function newChat(){
  try{const chat=await api('/api/chats',{method:'POST',body:{title:'New conversation'}});await refreshChats();await selectChat(chat.id);$('question').focus();}
  catch(error){toast(error.message,true);}
}
// Store the selected chat ID and fetch its messages for display.
async function selectChat(id){
  if(state.chatId!==id){state.chatId=id;localStorage.setItem('resumeDesk.chat',id);state.chat=null;$('messages').replaceChildren(el('p','loading-text','Loading conversation…'));$('welcome').hidden=true;}
  closeSidebar();renderChats();
  try{const chat=await api('/api/chats/'+encoded(id));if(state.chatId!==id)return;state.chat=chat;renderMessages();}
  catch(error){toast(error.message,true);if(error.status===404){state.chatId=null;localStorage.removeItem('resumeDesk.chat');showWelcome();}}
}
// Replace bracketed citations with buttons when they match a returned source ID.
function appendCitedText(container,text,sources){
  const map=new Map((sources||[]).map(s=>[s.source_id,s]));
  const regex=/\[([^\]\n]{1,180})\]/g;let last=0;let match;
  while((match=regex.exec(text))!==null){
    container.append(document.createTextNode(text.slice(last,match.index)));
    const source=map.get(match[1]);
    if(source){const button=el('button','source-link','['+match[1]+']');button.title='View '+source.candidate_name+' · '+(source.section||'full resume');button.addEventListener('click',()=>openLibrary(source.candidate_id));container.append(button);}
    else container.append(document.createTextNode(match[0]));
    last=regex.lastIndex;
  }
  container.append(document.createTextNode(text.slice(last)));
}
// Draw the conversation, including waiting states, cited sources and evaluation counts.
function renderMessages(){
  const chat=state.chat;if(!chat)return;
  const area=$('conversation');const wasNearBottom=area.scrollHeight-area.scrollTop-area.clientHeight<140;
  $('chatTitle').textContent=chat.title==='New conversation'?'Workspace chat':chat.title;
  $('welcome').hidden=chat.messages.length>0;
  const list=$('messages');list.replaceChildren();
  for(const message of chat.messages){
    const row=el('article','message '+message.role+(message.status==='failed'?' failed':''));
    if(message.role==='assistant'){
      const head=el('div','message-head');const mark=el('span','assistant-mark');mark.append(icon('spark'));head.append(mark,el('span','','Resume Desk'));row.append(head);
    }
    if(message.status==='pending'){
      const line=el('div','pending-line');const dots=el('span','dots');for(let i=0;i<3;i++)dots.append(el('i'));
      line.append(dots,el('span','',chat.pending?.[0]?.message||'Reviewing resume evidence…'));row.append(line);
    }else{
      const body=el('div','message-text');appendCitedText(body,message.content,message.result?.sources);row.append(body);
      if(message.role==='assistant'&&message.status==='done'){
        const result=message.result||{};const meta=el('div','message-meta');
        const reviewed=result.audit?.candidates_scored??result.audit?.candidates_checked;
        if(reviewed!==undefined)meta.append(el('span','reviewed-tag',reviewed.toLocaleString()+' candidates reviewed'));
        if(result.audit?.complete_resumes_returned)meta.append(el('span','reviewed-tag',result.audit.complete_resumes_returned+' complete resume'+(result.audit.complete_resumes_returned===1?'':'s')));
        if(result.timings?.total_seconds)meta.append(el('span','',Math.round(result.timings.total_seconds)+'s'));
        row.append(meta);
        if(result.sources?.length){
          const details=el('details','source-details');details.append(el('summary','',result.sources.length+' source passage'+(result.sources.length===1?'':'s')));
          const cards=el('div','source-cards');const seen=new Set();
          for(const source of result.sources){const key=source.candidate_id+':'+(source.section||'full');if(seen.has(key))continue;seen.add(key);
            const card=el('button','source-card',source.candidate_name||source.candidate_id);card.append(el('small','',(source.section||'Complete resume')+(source.page?' · page '+source.page:'')));card.addEventListener('click',()=>openLibrary(source.candidate_id));cards.append(card);}
          details.append(cards);row.append(details);
        }
      }
    }
    list.append(row);
  }
  updateSend();
  if(wasNearBottom||state.sending)requestAnimationFrame(()=>{area.scrollTop=area.scrollHeight;});
}
// Create a chat if necessary and submit the question with a fresh deduplication ID.
async function send(event){
  event?.preventDefault();const question=$('question').value.trim();if(!question||state.sending||state.chat?.pending?.length)return;
  state.sending=true;updateSend();
  try{
    if(!state.chatId){const chat=await api('/api/chats',{method:'POST',body:{title:'New conversation'}});state.chatId=chat.id;localStorage.setItem('resumeDesk.chat',chat.id);}
    const id=state.chatId;
    await api('/api/chats/'+encoded(id)+'/messages',{method:'POST',body:{content:question,client_id:crypto.randomUUID()}});
    $('question').value='';$('question').rows=1;
    await Promise.all([refreshChats(),selectChat(id)]);
    requestAnimationFrame(()=>{$('conversation').scrollTop=$('conversation').scrollHeight;});
  }catch(error){toast(error.message,true);}
  finally{state.sending=false;updateSend();}
}
// Reflect database and model availability in the header and service notices.
async function refreshStatus(){
  try{
    const status=await api('/api/status');const ready=status.database&&status.model&&status.embedding;
    $('serviceStatus').classList.toggle('offline',!ready);$('serviceStatus').replaceChildren(el('i'),document.createTextNode(ready?'Local model online':'Service needs attention'));
    $('sidebarCount').textContent=status.resume_count.toLocaleString()+' resumes indexed';$('headerCount').textContent=status.resume_count.toLocaleString();
    $('welcomeCount').textContent=status.resume_count?status.resume_count.toLocaleString()+' resumes, ready to explore':'Add your first resume to get started';
    $('uploadLimit').textContent='PDF files · up to '+status.max_upload_mb+' MB each';
    const missing=[];if(!status.database)missing.push('Start Docker and run docker compose up -d.');if(!status.model||!status.embedding)missing.push('Start Ollama with llama3.2:3b and nomic-embed-text installed.');
    $('serviceNotice').hidden=missing.length===0;$('serviceNotice').textContent=missing.join(' ');
  }catch(error){$('serviceStatus').classList.add('offline');$('serviceStatus').textContent='Server unavailable';$('serviceNotice').hidden=false;$('serviceNotice').textContent='The local server is unavailable. Start the app and refresh this page.';}
}
// Show the library and open a specific candidate when following a source link.
async function openLibrary(candidateId=null){
  if(!$('libraryDialog').open)$('libraryDialog').showModal();
  try{await refreshResumes();if(candidateId)await selectResume(candidateId);await refreshJobs();}
  catch(error){toast(error.message,true);}
}
// Ask before closing the library if the text editor contains unsaved changes.
function closeLibrary(){if(state.editing&&!window.confirm('Discard your unsaved resume edits?'))return;state.editing=false;$('libraryDialog').close();}
// Request the current search and page offset from the resume-list endpoint.
async function refreshResumes(append=false){
  const offset=append?state.resumes.length:0;const q=$('resumeSearch').value.trim();
  const result=await api('/api/resumes?q='+encoded(q)+'&offset='+offset+'&limit=40');
  state.resumes=append?[...state.resumes,...result.items]:result.items;
  $('libraryTotal').textContent=result.total.toLocaleString();$('loadMore').hidden=state.resumes.length>=result.total;
  renderResumes();
}
// Turn the current result page into clickable candidate rows.
function renderResumes(){
  const list=$('resumeList');list.replaceChildren();
  if(!state.resumes.length){list.append(el('p','loading-text',$('resumeSearch').value?'No resumes match this search.':'Your library is empty. Upload a PDF to start.'));return;}
  for(const resume of state.resumes){
    const button=el('button','resume-row'+(state.selected?.candidate_id===resume.candidate_id?' selected':''));button.setAttribute('aria-label','View resume for '+resume.name);
    const initial=el('span','resume-initial',resume.name.split(/\s+/).map(x=>x[0]).slice(0,2).join(''));
    const text=el('span','resume-row-text');text.append(el('strong','',resume.name),el('small','',resume.role||resume.filename),el('small','row-location',resume.location||resume.candidate_id));
    button.append(initial,text);button.addEventListener('click',()=>selectResume(resume.candidate_id));list.append(button);
  }
}
// Fetch a candidate's full details after confirming any unsaved edit can be discarded.
async function selectResume(id){
  if(state.editing&&!window.confirm('Discard your unsaved resume edits?'))return;
  state.editing=false;
  try{const resume=await api('/api/resumes/'+encoded(id));state.selected=resume;renderResumes();renderDetail();}
  catch(error){toast(error.message,true);if(error.status===404){state.selected=null;renderDetail();}}
}
// Build a reusable action button with an icon, label and click callback.
function action(label,name,handler,cls='secondary'){const button=el('button','button '+cls);button.append(icon(name),document.createTextNode(label));button.addEventListener('click',handler);return button;}
// Switch between resume details and the text editor while exposing the available file actions.
function renderDetail(){
  const panel=$('resumeDetail');panel.replaceChildren();const resume=state.selected;
  if(!resume){const empty=el('div','detail-empty');empty.append(el('h3','','A closer look'),el('p','','Choose a resume to view, edit, or replace.'));panel.append(empty);return;}
  const meta=resume.metadata;const top=el('div','detail-top');top.append(el('h3','',meta.candidate_name),el('p','',[resume.role,resume.location].filter(Boolean).join(' · ')));
  const tags=el('div','detail-tags');tags.append(el('span','detail-tag',meta.pages+' page'+(meta.pages===1?'':'s')),el('span','detail-tag',meta.chunks+' searchable chunks'),el('span','detail-tag',meta.original_filename||meta.source));top.append(tags);panel.append(top);
  const actions=el('div','detail-actions');
  const download=el('a','button secondary');download.href='/api/resumes/'+encoded(resume.candidate_id)+'/file';download.append(icon('down'),document.createTextNode('Download PDF'));
  actions.append(download,action('Edit text','edit',()=>{state.editing=true;renderDetail();}),action('Replace PDF','upload',()=>$('replaceInput').click()),action('Delete','trash',deleteResume,'danger'));panel.append(actions);
  if(state.editing){
    const notice=el('p','edit-notice','Edits create a new PDF with a simple text layout and re-index the resume. The original formatting may change. Keep the candidate name on the first line and section headings on their own lines.');
    const editor=el('textarea','resume-editor');editor.id='resumeEditor';editor.value=resume.edit_text;editor.setAttribute('aria-label','Edit resume text');
    const buttons=el('div','edit-buttons');buttons.append(action('Save and re-index','check',saveEdit,'primary'),action('Cancel','close',()=>{state.editing=false;renderDetail();}));panel.append(notice,editor,buttons);
  }else{panel.append(el('pre','resume-text',resume.text));}
}
// Send edited text with the opened file hash so the server can detect a stale update.
async function saveEdit(){
  const text=$('resumeEditor').value.trim();if(!text){toast('The resume text cannot be empty.',true);return;}
  const resume=state.selected;
  try{const result=await api('/api/resumes/'+encoded(resume.candidate_id),{method:'PATCH',body:{text,expected_hash:resume.file_hash}});state.editing=false;renderDetail();toast('Updating the PDF and search index…');await refreshJobs();}
  catch(error){toast(error.message,true);}
}
// Confirm removal and send the version the user actually selected.
async function deleteResume(){
  const resume=state.selected;
  if(!window.confirm('Delete '+resume.metadata.candidate_name+' from the resume library and remove the saved PDF? Existing conversation messages will be kept.'))return;
  try{await api('/api/resumes/'+encoded(resume.candidate_id)+'?expected_hash='+encoded(resume.file_hash),{method:'DELETE'});toast('Removing the resume…');await refreshJobs();}
  catch(error){toast(error.message,true);}
}
// Upload PDFs as multipart form data and include the expected hash for replacements.
async function uploadFiles(files,replace=false){
  if(!files.length)return;
  if(replace&&(!state.selected||!window.confirm('Replace '+state.selected.metadata.candidate_name+'’s PDF and searchable resume?')))return;
  if(!$('libraryDialog').open)$('libraryDialog').showModal();
  for(const file of files){
    const form=new FormData();form.append('file',file);
    let path='/api/resumes';if(replace){path+='/'+encoded(state.selected.candidate_id)+'/file';form.append('expected_hash',state.selected.file_hash);}
    try{await api(path,{method:replace?'PUT':'POST',body:form});toast(file.name+' added to the processing queue.');}
    catch(error){toast(file.name+': '+error.message,true);}
  }
  await refreshJobs();
}
// Read queued work and refresh the library when an indexing job finishes.
async function refreshJobs(){
  const jobs=await api('/api/jobs');state.jobs=jobs;
  const panel=$('uploadJobs');panel.replaceChildren();
  for(const job of jobs){
    if(job.kind==='chat')continue;
    const live=job.status==='queued'||job.status==='running';
    if(live||job.status==='failed'){
      const row=el('div','upload-job'+(job.status==='failed'?' failed':''));row.append(el('span','job-dot'));
      const text=el('span','job-text',job.status==='failed'?'Resume update failed':job.message);if(job.error)text.append(el('small','',job.error));row.append(text);
      if(job.status==='failed'&&job.retryable){const retry=el('button','retry-job','Retry');retry.addEventListener('click',async()=>{try{await api('/api/jobs/'+encoded(job.id)+'/retry',{method:'POST'});await refreshJobs();}catch(error){toast(error.message,true);}});row.append(retry);}
      panel.append(row);
    }
    if(job.status==='ready'&&!state.seenJobs.has(job.id)){
      state.seenJobs.add(job.id);
      if(Date.now()-new Date(job.updated_at).getTime()<120000){
        const result=job.result||{};
        toast(result.deleted?'Resume removed. Search is up to date.':result.duplicate?'This PDF is already in your library.':(result.candidate_name||'Resume')+' is saved and ready to chat.');
        await refreshStatus();if($('libraryDialog').open)await refreshResumes();
        if(result.deleted&&state.selected?.candidate_id===result.candidate_id){state.selected=null;state.editing=false;renderDetail();}
        else if(!state.editing&&result.candidate_id&&$('libraryDialog').open)await selectResume(result.candidate_id);
      }
    }
  }
}
// Poll chat and job endpoints more often while an operation is active.
async function poll(){
  try{
    if(state.chatId&&state.chat?.pending?.length)await selectChat(state.chatId);
    await refreshJobs();
    if(!state.chat?.pending?.length&&state.chats.length)await refreshChats();
  }catch{
    // The health check displays service failures. Keep polling so the page can recover.
  }
  setTimeout(poll,state.chat?.pending?.length||state.jobs.some(j=>j.status==='queued'||j.status==='running')?1200:5000);
}
// Wire up the page controls, restore the last chat and begin polling.
async function init(){
  // Wire chat input first, then connect the library and file controls.
  $('newChat').addEventListener('click',newChat);$('composer').addEventListener('submit',send);
  $('question').addEventListener('input',()=>{updateSend();$('question').rows=Math.min(6,$('question').value.split('\n').length);});
  $('question').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();send();}});
  document.querySelectorAll('.suggestion').forEach(button=>button.addEventListener('click',()=>{$('question').value=button.dataset.prompt;updateSend();$('question').focus();}));
  for(const id of ['openLibrary','sidebarLibrary'])$(id).addEventListener('click',()=>{closeSidebar();openLibrary();});
  for(const id of ['attachButton','welcomeUpload','libraryUpload'])$(id).addEventListener('click',()=>$('uploadInput').click());
  $('uploadInput').addEventListener('change',async()=>{await uploadFiles([...$('uploadInput').files]);$('uploadInput').value='';});
  $('replaceInput').addEventListener('change',async()=>{await uploadFiles([...$('replaceInput').files],true);$('replaceInput').value='';});
  $('closeLibrary').addEventListener('click',closeLibrary);$('libraryDialog').addEventListener('cancel',event=>{if(state.editing){event.preventDefault();closeLibrary();}});
  $('resumeSearch').addEventListener('input',()=>{clearTimeout(state.searchTimer);state.searchTimer=setTimeout(()=>refreshResumes().catch(e=>toast(e.message,true)),250);});
  $('loadMore').addEventListener('click',()=>refreshResumes(true).catch(e=>toast(e.message,true)));
  $('menuButton').addEventListener('click',()=>$('sidebar').classList.toggle('open'));$('sidebarScrim').addEventListener('click',closeSidebar);
  // Restore the previous selection only after the backend lists the owner's chats.
  await refreshStatus();
  try{await refreshChats();const saved=localStorage.getItem('resumeDesk.chat');if(state.chats.some(c=>c.id===saved))await selectChat(saved);else if(state.chats.length)await selectChat(state.chats[0].id);await refreshJobs();}
  catch(error){toast(error.message,true);}
  poll();setInterval(refreshStatus,15000);
}
init();
