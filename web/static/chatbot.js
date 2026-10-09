/* <AIChatbot> — floating Q&A assistant. Replies stream over a POST response
   (server-sent-event frames read with fetch), render as plain text while
   streaming, then swap to the server's sanitized Markdown HTML when done. */
(() => {
'use strict';
const T = window.TA;
const { $, esc } = T;

const STORE = 'ta.chat.v1';
const MAX_KEEP = 40;
let history = [];      // [{role, content, html?, tickers?}]
let controller = null; // AbortController of the reply in flight

/* -------------------------------- storage -------------------------------- */

function restore() {
  try { history = JSON.parse(localStorage.getItem(STORE) || '[]'); } catch (_) { history = []; }
  if (!Array.isArray(history)) history = [];
}
function persist() {
  try { localStorage.setItem(STORE, JSON.stringify(history.slice(-MAX_KEEP))); } catch (_) { /* private mode */ }
}

/* -------------------------------- rendering ------------------------------ */

function avatar(role) {
  return `<span class="avatar">${T.icon(role === 'user' ? 'user' : 'sparkles')}</span>`;
}

function addMessage(role, { text = '', html = null, error = false } = {}) {
  const body = $('#chat-body');
  const empty = $('.chat-empty', body);
  if (empty) empty.remove();
  const msg = document.createElement('div');
  msg.className = `msg ${role === 'user' ? 'user' : 'bot'}`;
  msg.innerHTML = `${avatar(role)}<div class="bubble"></div>`;
  const bubble = $('.bubble', msg);
  if (error) bubble.classList.add('error');
  if (html) { bubble.classList.add('prose'); bubble.innerHTML = html; } else bubble.textContent = text;
  body.appendChild(msg);
  scrollDown();
  return msg;
}

function addActions(msg, tickers) {
  if (!tickers || !tickers.length) return;
  const row = document.createElement('div');
  row.className = 'msg-actions';
  row.innerHTML = tickers.map(t =>
    `<button class="suggestion" data-analyze="${esc(t)}">${T.icon('search')} Analyze ${esc(t)}</button>` +
    `<button class="suggestion" data-chart="${esc(t)}">${T.icon('chart')} Chart ${esc(t)}</button>`).join('');
  msg.after(row);
}

function scrollDown() {
  const body = $('#chat-body');
  body.scrollTop = body.scrollHeight;
}

function renderAll() {
  const body = $('#chat-body');
  body.innerHTML = '';
  if (!history.length) {
    body.innerHTML = `<div class="chat-empty"><span class="brand-mark">${T.icon('sparkles')}</span>` +
      `<h3>Ask about your book or any ticker</h3>` +
      `<p class="small">Answers use your saved portfolio and the latest cached prices.</p></div>`;
  }
  history.forEach(m => {
    const el = addMessage(m.role, m.role === 'user' ? { text: m.content } : { html: m.html, text: m.content });
    if (m.role === 'assistant') addActions(el, m.tickers);
  });
  renderSuggestions();
}

function renderSuggestions() {
  const box = $('#chat-suggestions');
  const meta = T.meta() || {};
  const list = [...(meta.chat_suggestions || [])];
  if (T.currentTicker && !list.some(s => s.includes(T.currentTicker))) {
    list.unshift(`Explain ${T.currentTicker}'s recent move`);
  }
  box.innerHTML = history.length > 1 ? '' : list.slice(0, 4)
    .map(s => `<button type="button" class="suggestion" data-prompt="${esc(s)}">${esc(s)}</button>`).join('');
}

function setBusy(busy) {
  const btn = $('#chat-send');
  btn.innerHTML = T.icon(busy ? 'stop' : 'send');
  btn.setAttribute('aria-label', busy ? 'Stop' : 'Send');
  btn.classList.toggle('danger', busy);
  btn.classList.toggle('primary', !busy);
}

/* -------------------------------- streaming ------------------------------ */

async function send(text) {
  text = text.trim();
  if (!text || controller) return;
  history.push({ role: 'user', content: text });
  persist();
  addMessage('user', { text });
  $('#chat-suggestions').innerHTML = '';

  const msg = addMessage('assistant');
  const bubble = $('.bubble', msg);
  bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  controller = new AbortController();
  setBusy(true);

  let reply = ''; let tickers = []; let html = null; let failed = null;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        messages: history.slice(-12).map(({ role, content }) => ({ role, content })),
        settings: T.collectModelSettings(),
      }),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const line = frame.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line.slice(6)); } catch (_) { continue; }
        if (ev.type === 'delta') {
          if (!reply) { bubble.textContent = ''; bubble.classList.add('streaming'); }
          reply += ev.text;
          bubble.textContent = reply;
          scrollDown();
        } else if (ev.type === 'actions') {
          tickers = ev.tickers || [];
        } else if (ev.type === 'done') {
          html = ev.html;
        } else if (ev.type === 'error') {
          failed = ev.message;
        }
      }
    }
  } catch (err) {
    failed = err.name === 'AbortError' ? (reply ? null : 'Stopped.') : err.message;
  } finally {
    controller = null;
    setBusy(false);
  }

  bubble.classList.remove('streaming');
  if (failed && !reply) {
    bubble.classList.add('error');
    bubble.textContent = failed;
    history.pop();             // drop the unanswered question so a retry is clean
    persist();
    return;
  }
  if (html) { bubble.classList.add('prose'); bubble.innerHTML = html; }
  history.push({ role: 'assistant', content: reply, html, tickers });
  persist();
  addActions(msg, tickers);
  scrollDown();
}

/* ---------------------------------- wire --------------------------------- */

function open() {
  $('#chat-drawer').classList.remove('hidden');
  $('#chat-fab').classList.add('open');
  renderSuggestions();
  const model = T.collectModelSettings();
  if (model.llm_provider) $('#chat-model').textContent = `${model.llm_provider} · ${model.quick_think_llm || 'default model'}`;
  setTimeout(() => $('#chat-input').focus(), 50);
}
function close() {
  $('#chat-drawer').classList.add('hidden');
  $('#chat-fab').classList.remove('open');
  $('#chat-fab').focus();
}

function init() {
  restore();
  renderAll();
  $('#chat-fab').addEventListener('click', open);
  $('#chat-close').addEventListener('click', close);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('#chat-drawer').classList.contains('hidden')) close();
  });
  $('#chat-clear').addEventListener('click', () => {
    if (controller) controller.abort();
    history = [];
    persist();
    renderAll();
  });
  const input = $('#chat-input');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(140, input.scrollHeight) + 'px';
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#chat-form').requestSubmit(); }
  });
  $('#chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    if (controller) { controller.abort(); return; }
    const text = input.value;
    input.value = '';
    input.style.height = 'auto';
    send(text);
  });
  $('#chat-drawer').addEventListener('click', (e) => {
    const s = e.target.closest('[data-prompt]');
    if (s) { send(s.dataset.prompt); return; }
    const a = e.target.closest('[data-analyze]');
    if (a) { close(); T.openAnalysisFor(a.dataset.analyze); return; }
    const c = e.target.closest('[data-chart]');
    if (c) { close(); T.showChart(c.dataset.chart); }
  });
  document.addEventListener('ta:ready', renderSuggestions);
}

document.addEventListener('DOMContentLoaded', init);
})();
