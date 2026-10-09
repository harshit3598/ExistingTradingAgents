/* TradingAgents web UI — vanilla JS, no build step.
   Streams runs over SSE; runs are replayable, so a page reload picks the
   stream back up from the first event it has not seen yet. */
(() => {
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const TERMINAL = ['done', 'error'];

const ANALYST_NAMES = {
  market: 'Market Analyst',
  social: 'Sentiment Analyst',
  news: 'News Analyst',
  fundamentals: 'Fundamentals Analyst',
};
const FIXED_GROUPS = [
  ['Research Team', ['Bull Researcher', 'Bear Researcher', 'Research Manager']],
  ['Trading Team', ['Trader']],
  ['Risk Management', ['Aggressive Analyst', 'Neutral Analyst', 'Conservative Analyst']],
  ['Portfolio Management', ['Portfolio Manager']],
];
const SECTION_ORDER = [
  'market_report', 'sentiment_report', 'news_report', 'fundamentals_report',
  'investment_plan', 'trader_investment_plan', 'final_trade_decision',
];
const CRYPTO_SUFFIXES = ['-USD', '-USDT', '-USDC', '-BTC', '-ETH'];

let META = null;
const state = {
  runId: null,
  runConn: null,
  lastSeq: 0,
  agents: {},
  sections: {},          // section -> {title, html}
  sectionOrder: [],
  startedAt: 0,
  done: null,            // done event payload
  runPayload: null,
  runKind: null,          // 'analysis' | 'backtest' | 'suggest'
  btRunId: null,
  btConn: null,
  sgRunId: null,
  sgConn: null,
  reports: [],
  activeReport: null,
  reportFiles: [],
  activeFile: null,
};

/* ------------------------------- utilities ------------------------------- */

function toast(msg, isError) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.toggle('error', !!isError);
  el.classList.add('hidden');
  void el.offsetWidth; // restart the entry animation for back-to-back toasts
  el.classList.remove('hidden');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add('hidden'), 5200);
}

async function api(path, opts = {}) {
  const init = { ...opts, headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) } };
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* body was not JSON */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function icon(name) {
  return `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

function setBadge(badge, status) {
  badge.classList.remove('hidden');
  badge.textContent = status;
  badge.className = 'status-badge ' + status;
  if (status === 'running' || status === 'queued') badge.classList.add('running');
}

function setProgress(wrap, done, total, describe) {
  wrap.classList.remove('hidden');
  const pct = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
  $('.bar', wrap).style.width = pct + '%';
  $('.label', wrap).textContent = `${describe(done, total)} (${pct}%)`;
}

// One log line in any of the three logs; old lines are trimmed to ``max``.
function appendLog(log, kind, content, { clock = '', cls = '', max = 2500, follow = true } = {}) {
  const placeholder = log.querySelector('.empty');
  if (placeholder) placeholder.remove();
  const line = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '') + ' k-' + (kind || 'System');
  line.innerHTML = `<span class="t">${esc(clock)}</span>` +
    `<span class="k">${esc(kind || '')}</span>` +
    `<span class="c">${esc(content || '')}</span>`;
  log.appendChild(line);
  while (log.children.length > max) log.removeChild(log.firstChild);
  if (follow) log.scrollTop = log.scrollHeight;
}

function fmtTokens(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

function showView(name) {
  $$('.view').forEach(v => { v.hidden = v.id !== 'view-' + name; });
  $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === name));
  if (name === 'reports') loadReports();
  window.scrollTo({ top: 0 });
  document.dispatchEvent(new CustomEvent('ta:view', { detail: name }));
}

$$('.tab').forEach(tab => tab.addEventListener('click', () => showView(tab.dataset.view)));

/* --------------------------------- SSE ---------------------------------- */

function connectEvents(runId, onEvent) {
  let seq = 0;
  let es = null;
  let closed = false;

  function open() {
    if (closed) return;
    es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events?after=${seq}`);
    es.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch (_) { return; }
      if (ev.seq) seq = ev.seq;
      try { onEvent(ev); } catch (err) { console.error(err); }
      if ((ev.type === 'status' && TERMINAL.includes(ev.status)) ||
          ev.type === 'error' || ev.type === 'end') {
        closed = true;
        es.close();
      }
    };
    es.onerror = () => {
      es.close();
      if (!closed) setTimeout(open, 2000);
    };
  }
  open();
  return { close: () => { closed = true; if (es) es.close(); }, seq: () => seq };
}

/* ============================== ANALYZE FORM ============================= */

function isCryptoTicker(value) {
  const v = String(value || '').trim().toUpperCase();
  return CRYPTO_SUFFIXES.some(s => v.endsWith(s));
}

function buildAnalystChips(container) {
  container.innerHTML = '';
  META.analysts.forEach(a => {
    const label = document.createElement('label');
    label.className = 'chip';
    label.innerHTML = `<input type="checkbox" value="${a.key}" checked><span>${esc(a.label)}</span>`;
    container.appendChild(label);
  });
}

function checkedAnalysts(container) {
  return $$('input:checked', container).map(i => i.value);
}

function applyAssetType() {
  const crypto = isCryptoTicker($('#ticker').value);
  $('#asset-hint').textContent = crypto ? 'Detected asset type: crypto' : '';
  $$('#analyst-chips .chip').forEach(chip => {
    const input = $('input', chip);
    if (input.value === 'fundamentals') {
      chip.classList.toggle('disabled', crypto);
      input.disabled = crypto;
      if (crypto) input.checked = false;
    }
  });
}

function baseProvider(key) {
  return key.endsWith('-cn') ? key.slice(0, -3) : key;
}

function fillModelSelect(select, customInput, options, remembered) {
  select.innerHTML = '';
  options.forEach(([label, value]) => {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    select.appendChild(opt);
  });
  const customExists = options.some(([, v]) => v === 'custom');
  if (remembered && options.some(([, v]) => v === remembered)) select.value = remembered;
  else if (!customExists) select.value = 'custom';
  customInput.classList.toggle('hidden', select.value !== 'custom');
}

function refreshProviderUI() {
  const provider = $('#llm_provider').value;
  const base = baseProvider(provider);

  // models
  const remembered = (META.prefs.llm_provider === provider) ? META.prefs : {};
  if (base === 'azure') {
    ['#quick_think_llm', '#deep_think_llm'].forEach(sel => { $(sel).innerHTML = ''; });
    $('#quick_custom').classList.remove('hidden');
    $('#deep_custom').classList.remove('hidden');
    $('#quick_custom').placeholder = 'Azure deployment name (quick-thinking)';
    $('#deep_custom').placeholder = 'Azure deployment name (deep-thinking)';
  } else if (base === 'openrouter') {
    const opts = [['Custom model ID', 'custom']];
    fillModelSelect($('#quick_think_llm'), $('#quick_custom'), opts, remembered.quick_think_llm);
    fillModelSelect($('#deep_think_llm'), $('#deep_custom'), opts, remembered.deep_think_llm);
    api('/api/openrouter-models').then(data => {
      if (provider !== $('#llm_provider').value || !data.models.length) return;
      const list = [...data.models, ['Custom model ID', 'custom']];
      fillModelSelect($('#quick_think_llm'), $('#quick_custom'), list, remembered.quick_think_llm);
      fillModelSelect($('#deep_think_llm'), $('#deep_custom'), list, remembered.deep_think_llm);
    }).catch(() => { /* offline: custom IDs still work */ });
  } else {
    const models = META.models[base] || { quick: [], deep: [] };
    fillModelSelect($('#quick_think_llm'), $('#quick_custom'), models.quick, remembered.quick_think_llm);
    fillModelSelect($('#deep_think_llm'), $('#deep_custom'), models.deep, remembered.deep_think_llm);
  }

  // provider-specific reasoning knob
  const knob = META.knobs[base];
  const row = $('#knob-row');
  if (knob) {
    row.classList.remove('hidden');
    $('#knob-label').textContent = knob.label;
    const sel = $('#knob-select');
    sel.innerHTML = '<option value="">Provider default</option>' +
      knob.options.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
    const locked = META.env_locked[knob.field];
    if (locked) sel.value = String(locked);
    row.dataset.field = knob.field;
  } else {
    row.classList.add('hidden');
    row.dataset.field = '';
  }

  // API key requirement
  const keyInfo = META.api_keys[provider] || {};
  const missing = keyInfo.env && !keyInfo.set && !keyInfo.optional;
  $('#api-key-block').classList.toggle('hidden', !missing);
  $('#api-key-env').textContent = keyInfo.env || '';

  // backend URL placeholder
  const providerObj = META.providers.find(p => p.key === provider);
  $('#backend_url').placeholder = (providerObj && providerObj.url) || 'provider default';
}

function fillPrefilledForm() {
  const prefs = META.prefs || {};
  const locked = META.env_locked || {};

  $('#analysis_date').value = META.defaults.date;
  $('#analysis_date').max = META.defaults.date;
  $('#ticker').value = META.defaults.ticker;
  $('#output_language').value = locked.output_language || prefs.output_language || 'English';

  const dl = $('#language-list');
  dl.innerHTML = META.languages.map(l => `<option value="${esc(l)}">`).join('');

  buildAnalystChips($('#analyst-chips'));
  buildAnalystChips($('#bt-analyst-chips'));
  const wanted = (prefs.analysts && prefs.analysts.length)
    ? prefs.analysts : META.analysts.map(a => a.key);
  $$('#analyst-chips input').forEach(i => { i.checked = wanted.includes(i.value); });
  $$('#bt-analyst-chips input').forEach(i => { i.checked = wanted.includes(i.value); });

  $('#depth-cards').innerHTML = META.depths.map(d => `
    <label class="choice">
      <input type="radio" name="research_depth" value="${d.value}">
      <b>${d.value} · ${esc(d.label)}</b><span>${esc(d.hint)}</span>
    </label>`).join('');
  const depth = (prefs.research_depth === 1 || prefs.research_depth === 3 || prefs.research_depth === 5)
    ? prefs.research_depth : 1;
  const radio = $(`#depth-cards input[value="${depth}"]`);
  if (radio) radio.checked = true;

  $('#llm_provider').innerHTML = META.providers
    .map(p => `<option value="${p.key}">${esc(p.label)}</option>`).join('');
  $('#llm_provider').value = (prefs.llm_provider &&
    META.providers.some(p => p.key === prefs.llm_provider)) ? prefs.llm_provider : 'openai';
  $('#backend_url').value = prefs.backend_url || '';
  refreshProviderUI();

  if (Object.keys(locked).length) {
    const banner = $('#env-lock-banner');
    banner.classList.remove('hidden');
    banner.innerHTML = '<strong>Environment overrides active:</strong> ' +
      Object.entries(locked).map(([k, v]) => `<code>${esc(k)} = ${esc(v)}</code>`).join(' · ') +
      ' — these win over the form.';
  }
}

function collectModelSettings() {
  // The provider/model choices shared by a run and a backtest sweep.
  const knobField = $('#knob-row').dataset.field;
  const settings = {
    llm_provider: $('#llm_provider').value,
    backend_url: $('#backend_url').value.trim() || null,
    quick_think_llm: ($('#quick_think_llm').value === 'custom')
      ? $('#quick_custom').value.trim() : $('#quick_think_llm').value,
    deep_think_llm: ($('#deep_think_llm').value === 'custom')
      ? $('#deep_custom').value.trim() : $('#deep_think_llm').value,
    output_language: $('#output_language').value.trim() || 'English',
  };
  if (knobField) settings[knobField] = $('#knob-select').value || null;
  return settings;
}

function updateBacktestHint() {
  const s = collectModelSettings();
  const el = $('#bt-model-hint');
  if (el) {
    el.textContent = `Sweeps run on ${s.llm_provider} · quick=${s.quick_think_llm} · ` +
      `deep=${s.deep_think_llm} — change these on New Analysis.`;
  }
}

function collectAnalysisPayload() {
  const depth = $('#depth-cards input:checked');
  const payload = {
    ticker: $('#ticker').value.trim() || 'SPY',
    analysis_date: $('#analysis_date').value,
    analysts: checkedAnalysts($('#analyst-chips')),
    research_depth: depth ? parseInt(depth.value, 10) : 1,
    checkpoint: $('#checkpoint').value === '' ? null : $('#checkpoint').value === 'true',
    portfolio: $('#portfolio').value.trim() || null,
    ...collectModelSettings(),
  };
  const keyBlock = $('#api-key-block');
  if (!keyBlock.classList.contains('hidden') && $('#api-key-input').value.trim()) {
    payload.api_key = $('#api-key-input').value.trim();
  }
  return payload;
}

async function saveKeyFromForm() {
  const provider = $('#llm_provider').value;
  const key = $('#api-key-input').value.trim();
  if (!key) { toast('Paste an API key first.', true); return; }
  try {
    const res = await api('/api/apikey', { method: 'POST', body: JSON.stringify({ provider, key }) });
    META.api_keys[provider] = { env: res.env, set: true, optional: false };
    $('#api-key-input').value = '';
    toast(`Saved ${res.env} to .env`);
    refreshProviderUI();
  } catch (err) {
    toast(err.message, true);
  }
}

/* =============================== RUN VIEW =============================== */

function resetRunView() {
  state.agents = {};
  state.sections = {};
  state.sectionOrder = [];
  state.done = null;
  state.startedAt = Date.now();
  state.lastSeq = 0;
  $('#run-log').innerHTML = '';
  $('#run-sections').innerHTML = '<p class="empty">Sections appear here as each agent finishes.</p>';
  $('#final-card').classList.add('hidden');
  $('#exec-card').classList.add('hidden');
  $('#exec-result').classList.add('hidden');
  state.runKind = null;
  $('#run-error').classList.add('hidden');
  $('#btn-save-report').classList.add('hidden');
  $('#run-elapsed').textContent = '';
  ['st-llm', 'st-tools', 'st-tokens', 'st-tokens-in'].forEach(id => { $('#' + id).textContent = '0'; });
  $$('#pipeline li').forEach(li => li.classList.remove('active', 'done'));
}

function setStatusBadge(status) {
  setBadge($('#run-status'), status);
}

function selectedAnalystNames() {
  const payload = state.runPayload || {};
  return (payload.analysts || []).map(a => ANALYST_NAMES[a]).filter(Boolean);
}

function renderAgents() {
  const wrap = $('#agent-chips');
  const groups = [];
  const analystNames = selectedAnalystNames();
  if (analystNames.length) groups.push(['Analysts', analystNames]);
  FIXED_GROUPS.forEach(([name, members]) => groups.push([name, members]));

  const parts = [];
  groups.forEach(([gname, members]) => {
    members.forEach((member, i) => {
      const st = state.agents[member] || 'pending';
      parts.push(`<span class="chip s-${esc(st)}">${i === 0 ? `<em>${esc(gname)}</em>` : ''}${esc(member)}</span>`);
    });
  });
  wrap.innerHTML = parts.join('');
}

function updatePipeline() {
  const groups = [
    { id: 'analysts', names: selectedAnalystNames() },
    { id: 'research', names: FIXED_GROUPS[0][1] },
    { id: 'trader', names: FIXED_GROUPS[1][1] },
    { id: 'risk', names: FIXED_GROUPS[2][1] },
    { id: 'portfolio', names: FIXED_GROUPS[3][1] },
  ];
  groups.forEach(g => {
    const li = $(`#pipeline li[data-stage="${g.id}"]`);
    if (!li || !g.names.length) return;
    const statuses = g.names.map(n => state.agents[n] || 'pending');
    li.classList.remove('active', 'done');
    if (statuses.every(s => s === 'completed')) li.classList.add('done');
    else if (statuses.some(s => s === 'in_progress')) li.classList.add('active');
  });
}

function addLogLine(clock, kind, content, cls) {
  appendLog($('#run-log'), kind, content,
    { clock: clock || '', cls, follow: $('#log-autoscroll').checked });
}

function renderSection(section, title, html) {
  const wrap = $('#run-sections');
  if (!state.sections[section]) {
    state.sectionOrder.push(section);
    const first = state.sectionOrder.length === 1;
    if (first) wrap.innerHTML = '';
    const el = document.createElement('div');
    el.className = 'section';
    el.dataset.section = section;
    el.innerHTML = `<h3>${esc(title)}</h3><div class="prose"></div>`;
    // keep the canonical section order, not arrival order
    wrap.appendChild(el);
    sortSections();
  }
  state.sections[section] = { title, html };
  const el = $(`#run-sections .section[data-section="${section}"] .prose`);
  if (el) el.innerHTML = html;
}

function sortSections() {
  const wrap = $('#run-sections');
  state.sectionOrder
    .slice()
    .sort((a, b) => SECTION_ORDER.indexOf(a) - SECTION_ORDER.indexOf(b))
    .forEach(section => {
      const el = $(`#run-sections .section[data-section="${section}"]`);
      if (el) wrap.appendChild(el);
    });
}

function updateStats(ev) {
  if (ev.llm_calls != null) $('#st-llm').textContent = ev.llm_calls;
  if (ev.tool_calls != null) $('#st-tools').textContent = ev.tool_calls;
  if (ev.tokens_out != null) $('#st-tokens').textContent = fmtTokens(ev.tokens_out);
  if (ev.tokens_in != null) $('#st-tokens-in').textContent = fmtTokens(ev.tokens_in);
  if (ev.elapsed != null) $('#st-elapsed').textContent = Math.round(ev.elapsed) + 's';
}

function onRunDone(ev) {
  state.done = ev;
  const card = $('#final-card');
  const rating = $('#final-rating');
  const decision = state.sections.final_trade_decision;
  $('#final-body').innerHTML = decision
    ? decision.html
    : '<p class="muted">The run finished without a final decision text.</p>';
  rating.textContent = ev.rating || '—';
  rating.className = 'rating' +
    (ev.review ? ' review' : (['Sell', 'Underweight'].includes(ev.rating) ? ' sell' : ''));
  card.classList.remove('hidden');
  $('#btn-save-report').classList.remove('hidden');
  const note = $('#final-note');
  if (ev.review) {
    note.classList.remove('hidden');
    note.innerHTML = '<strong>No rating could be read from the final decision,</strong> so this run is ' +
      'recorded for review rather than as a position. Re-run, or read the decision above and judge it yourself.';
  } else {
    note.classList.add('hidden');
  }
  if (ev.wall_time) addLogLine('', 'System', ev.wall_time);
  $('#st-elapsed').textContent = Math.round(ev.elapsed || 0) + 's';
  updateExecCard(ev);
}

const TRADABLE_RATINGS = ['Buy', 'Overweight', 'Sell', 'Underweight'];

function updateExecCard(ev) {
  const show = state.runKind === 'analysis' && TRADABLE_RATINGS.includes(ev.rating);
  $('#exec-card').classList.toggle('hidden', !show);
  if (!show) return;
  $('#exec-badge').textContent = ev.rating;
  $('#exec-badge').className = 'rating' +
    (['Sell', 'Underweight'].includes(ev.rating) ? ' sell' : '');
  $('#exec-result').classList.add('hidden');
}

async function executeTrade(mode) {
  if (!state.runId) return;
  if (mode === 'live' && !confirm(
    'Send a REAL limit order to Robinhood for this decision?\n\n' +
    'This trades your live account. Continue?')) return;
  const btn = mode === 'live' ? $('#btn-exec-live') : $('#btn-exec-preview');
  const out = $('#exec-result');
  const priceRaw = $('#exec-price').value.trim();
  btn.disabled = true;
  out.classList.remove('hidden');
  out.innerHTML = '<p class="muted small">Running risk check → sizing → broker…</p>';
  try {
    const res = await api(`/api/runs/${encodeURIComponent(state.runId)}/execute`, {
      method: 'POST',
      body: JSON.stringify({
        mode,
        price: priceRaw === '' ? null : Number(priceRaw),
        mfa_code: $('#exec-mfa').value.trim() || null,
        confirm: mode === 'live',
      }),
    });
    renderExec(res);
  } catch (err) {
    out.innerHTML = `<div class="notice warn"><strong>Could not execute:</strong> ${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function renderExec(res) {
  const r = res.result || {};
  const order = r.order || {};
  const sizing = r.sizing || {};
  const risk = r.risk || null;
  const cells = [];
  const add = (label, value) => {
    if (value != null && value !== '') cells.push(
      `<div><span class="muted small">${esc(label)}</span><strong>${esc(String(value))}</strong></div>`);
  };
  add('Mode', res.mode === 'live' ? 'LIVE order' : 'dry run');
  add('Signal', r.signal);
  add('Action', (order.action || r.action || '').toUpperCase());
  add('Quantity', order.quantity);
  add('Limit price', order.limit_price != null ? '$' + Number(order.limit_price).toFixed(2) : null);
  if (order.quantity && order.limit_price) {
    add('Order value', '$' + (order.quantity * order.limit_price)
      .toLocaleString(undefined, { maximumFractionDigits: 2 }));
  }
  const riskLine = risk
    ? `<p class="exec-line"><b>Risk</b><span class="${risk.passed ? 'ok-text' : 'bad-text'}">` +
      `${risk.passed ? 'Passed' : 'Blocked'}</span> ${esc(risk.reason || '')}</p>`
    : '';
  const sizingLine = sizing.reasoning
    ? `<p class="exec-line"><b>Sizing</b>${esc(sizing.reasoning)}</p>` : '';
  const status = r.executed
    ? (r.dry_run ? 'Order simulated' : 'Order submitted')
    : 'Not executed';
  $('#exec-result').innerHTML =
    `<div class="exec-grid">${cells.join('')}</div>${riskLine}${sizingLine}` +
    `<div class="notice ${r.executed ? 'ok' : 'warn'}">` +
    `<strong>${status}:</strong> ${esc(r.message || '')}</div>`;
}

function onRunError(ev) {
  $('#run-error').classList.remove('hidden');
  $('#run-error-msg').textContent = ev.message || 'The run failed.';
  $('#run-error-tb').textContent = ev.traceback || '';
}

function applyRunEvent(ev) {
  if (ev.seq) state.lastSeq = ev.seq;
  switch (ev.type) {
    case 'status': setStatusBadge(ev.status); break;
    case 'message': addLogLine(ev.clock, ev.kind, ev.content); break;
    case 'tool_call': {
      const args = Object.entries(ev.args || {}).map(([k, v]) => `${k}=${v}`).join(', ');
      addLogLine(ev.clock, 'Tool', `${ev.name}(${args})`, 'tool');
      break;
    }
    case 'agents': state.agents = ev.status || {}; renderAgents(); updatePipeline(); break;
    case 'section': renderSection(ev.section, ev.title, ev.html); break;
    case 'stats': updateStats(ev); break;
    case 'done': onRunDone(ev); break;
    case 'error': onRunError(ev); break;
    default: break;
  }
}

function openRun(runId) {
  if (state.runConn) state.runConn.close();
  state.runId = runId;
  resetRunView();
  $('#tab-run').hidden = false;
  showView('run');
  setStatusBadge('queued');

  api(`/api/runs/${runId}`).then(run => {
    state.runKind = run.kind;
    state.runPayload = run.payload || {};
    const p = state.runPayload;
    $('#run-title').textContent = `${p.ticker || ''}${p.analysis_date ? ' · ' + p.analysis_date : ''}`;
    $('#run-sub').textContent =
      `${(p.analysts || []).join(', ')} · depth ${p.research_depth} · ${p.llm_provider} · ` +
      `quick=${p.quick_think_llm} · deep=${p.deep_think_llm}`;
    renderAgents();
  }).catch(err => toast(err.message, true));

  state.runConn = connectEvents(runId, applyRunEvent);
}

function startElapsedTicker() {
  setInterval(() => {
    if (!state.runId || state.done) return;
    if ($('#view-run').hidden) return;
    const secs = Math.floor((Date.now() - state.startedAt) / 1000);
    $('#run-elapsed').textContent = secs + 's';
  }, 1000);
}

async function saveReport() {
  if (!state.runId) return;
  try {
    const res = await api(`/api/runs/${state.runId}/save-report`, {
      method: 'POST', body: JSON.stringify({}),
    });
    toast(`Report saved to ${res.path}`);
  } catch (err) {
    toast(err.message, true);
  }
}

/* =============================== REPORTS ================================ */

async function loadReports() {
  try {
    const data = await api('/api/reports');
    state.reports = data.reports;
    $('#reports-dir').textContent = `Saved under ${data.results_dir}`;
    const list = $('#report-list');
    if (!data.reports.length) {
      list.innerHTML = '<div class="empty">No saved reports yet — finish a run and press “Save report”.</div>';
      return;
    }
    list.innerHTML = data.reports.map((r, i) => `
      <div class="list-item${i === 0 ? ' active' : ''}" data-path="${esc(r.path)}">
        ${esc(r.name)}
        <span class="sub">${new Date(r.mtime * 1000).toLocaleString()}</span>
      </div>`).join('');
    $$('#report-list .list-item').forEach(item => item.addEventListener('click', () => {
      $$('#report-list .list-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      openReport(item.dataset.path);
    }));
    openReport(data.reports[0].path);
  } catch (err) {
    toast(err.message, true);
  }
}

async function openReport(path) {
  try {
    const data = await api(`/api/reports/files?path=${encodeURIComponent(path)}`);
    state.reportFiles = data.files;
    const tabs = $('#report-tabs');
    tabs.innerHTML = data.files.map((f, i) => `
      <button class="tab${i === 0 ? ' active' : ''}" data-file="${esc(f.name)}">${esc(f.name)}</button>`).join('');
    $$('#report-tabs .tab').forEach(tab => tab.addEventListener('click', () => {
      $$('#report-tabs .tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      showReportFile(tab.dataset.file);
    }));
    showReportFile(data.files[0].name);
  } catch (err) {
    $('#report-body').innerHTML = `<p class="empty">${esc(err.message)}</p>`;
  }
}

function showReportFile(name) {
  const file = state.reportFiles.find(f => f.name === name);
  if (file) $('#report-body').innerHTML = file.html;
}

/* =============================== BACKTEST =============================== */

function resetBacktestView() {
  $('#bt-log').innerHTML = '';
  $('#bt-summary').classList.add('hidden');
  $('#bt-progress').classList.add('hidden');
  const badge = $('#bt-status');
  badge.classList.add('hidden');
  badge.textContent = '';
}

function addBtLine(kind, content) {
  appendLog($('#bt-log'), kind || 'System', content, { max: 1500 });
}

function applyBacktestEvent(ev) {
  switch (ev.type) {
    case 'status': setBadge($('#bt-status'), ev.status); break;
    case 'message': addBtLine(ev.kind, ev.content); break;
    case 'progress':
      setProgress($('#bt-progress'), ev.done, ev.total, (d, t) => `${d} / ${t} cells settled`);
      break;
    case 'done': {
      addBtLine('System', `Finished: ${ev.cells_run} cell(s) run, ${ev.skipped} skipped. Log: ${ev.log_path}`);
      (ev.failures || []).forEach(f => addBtLine('System', `failed: ${f[0]} ${f[1]} — ${f[2]}`));
      (ev.settlement_failures || []).forEach(f => addBtLine('System', `unsettled: ${f[0]} — ${f[1]}`));
      const summary = $('#bt-summary');
      summary.classList.remove('hidden');
      if (ev.summary_html) summary.innerHTML = `<h2>Summary</h2>${ev.summary_html}`;
      else summary.textContent = ev.summary || '';
      break;
    }
    case 'error': addBtLine('System', 'ERROR: ' + (ev.message || 'failed')); break;
    default: break;
  }
}

async function startBacktest(ev) {
  ev.preventDefault();
  const payload = {
    tickers: $('#bt-tickers').value,
    asset_type: $('#bt-asset').value,
    start: $('#bt-start').value,
    end: $('#bt-end').value,
    every: parseInt($('#bt-every').value, 10) || 7,
    run_id: $('#bt-run-id').value.trim() || null,
    portfolio: $('#bt-portfolio').value.trim() || null,
    analysts: checkedAnalysts($('#bt-analyst-chips')),
    ...collectModelSettings(),
  };
  $('#bt-error').textContent = '';
  $('#bt-submit').disabled = true;
  try {
    const run = await api('/api/backtest', { method: 'POST', body: JSON.stringify(payload) });
    resumeBacktest(run.run_id, true);
  } catch (err) {
    $('#bt-error').textContent = err.message;
  } finally {
    $('#bt-submit').disabled = false;
  }
}

function resumeBacktest(runId, reset) {
  if (state.btConn) state.btConn.close();
  state.btRunId = runId;
  if (reset) resetBacktestView();
  $('#bt-progress').classList.remove('hidden');
  state.btConn = connectEvents(runId, applyBacktestEvent);
}

/* ============================= STOCK IDEAS ============================ */

function updateSuggestHint() {
  const s = collectModelSettings();
  const el = $('#sg-model-hint');
  if (el) {
    el.textContent = `Suggestions are curated with ${s.llm_provider} · ${s.quick_think_llm} ` +
      '(changed on New Analysis); without an API key the quantitative ranking is used as-is.';
  }
}

let sgParseTimer = null;
function scheduleGoalParse() {
  clearTimeout(sgParseTimer);
  sgParseTimer = setTimeout(parseGoalNow, 350);
}

async function parseGoalNow() {
  const goal = $('#sg-goal').value.trim();
  const out = $('#sg-detected');
  if (!goal) { out.textContent = ''; return; }
  try {
    const found = await api('/api/suggest/parse', { method: 'POST', body: JSON.stringify({ goal }) });
    if (found.target_pct != null) $('#sg-target').value = found.target_pct;
    if (found.horizon_days != null) $('#sg-days').value = found.horizon_days;
    const bits = [];
    if (found.target_pct != null) bits.push(`+${found.target_pct}%`);
    if (found.horizon_days != null) {
      bits.push(`within ${found.horizon_days} day${found.horizon_days === 1 ? '' : 's'}`);
    }
    out.textContent = bits.length
      ? `Detected: ${bits.join(' ')} — editable below.`
      : 'No numbers detected — set the target and horizon below.';
  } catch (_) { /* parse is best-effort feedback */ }
}

function resetSuggestView() {
  $('#sg-log').innerHTML = '';
  $('#sg-results').classList.add('hidden');
  $('#sg-table').innerHTML = '';
  $('#sg-progress').classList.add('hidden');
  $('#sg-status').classList.add('hidden');
  $('#sg-error').textContent = '';
}

function addSuggestLine(text) {
  appendLog($('#sg-log'), 'System', text, { max: 500 });
}

function fmtPct(value) {
  return value == null ? '—' : `${value > 0 ? '+' : ''}${Number(value).toFixed(1)}%`;
}

function applySuggestEvent(ev) {
  switch (ev.type) {
    case 'status': setBadge($('#sg-status'), ev.status); break;
    case 'message': addSuggestLine(ev.content || ''); break;
    case 'progress':
      setProgress($('#sg-progress'), ev.done, ev.total,
        (d, t) => `${ev.stage === 'score' ? 'Scoring' : 'Pricing'} ${d} / ${t}`);
      break;
    case 'done': renderSuggestions(ev); break;
    case 'error':
      addSuggestLine('ERROR: ' + (ev.message || 'failed'));
      $('#sg-error').textContent = ev.message || 'The screen failed.';
      break;
    default: break;
  }
}

function renderSuggestions(ev) {
  $('#sg-progress').classList.add('hidden');
  $('#sg-results').classList.remove('hidden');
  $('#sg-disclaimer').textContent = ev.disclaimer || '';

  const r = ev.request || {};
  const meta = `Screened ${ev.universe_size} tickers · ${ev.scored} priced · ` +
    `${ev.failed} unavailable · ${ev.llm_used ? 'LLM-curated' : 'quantitative ranking only'} · ` +
    `target +${r.target_pct}% within ${r.horizon_days} day(s)`;

  let html = `<p class="muted small">${esc(meta)}</p>`;
  if (!ev.suggestions || !ev.suggestions.length) {
    html += '<div class="empty">Nothing cleared the screen — try a lower target, a longer horizon, or a different universe.</div>';
  }
  (ev.suggestions || []).forEach((s, i) => {
    const chips = [
      `price ${s.price}`,
      `5d ${fmtPct(s.ret_5d_pct)}`,
      `${r.horizon_days}d ±${s.horizon_vol_pct == null ? '—' : s.horizon_vol_pct}%`,
      `vol ${s.vol_ann_pct == null ? '—' : s.vol_ann_pct + '%'}`,
      `from high ${fmtPct(s.dist_high_pct)}`,
    ];
    if (s.dollar_vol_m) chips.push(`$${s.dollar_vol_m}M/day`);
    html += `
      <div class="sg-item">
        <div class="sg-head">
          <span class="sg-rank">${i + 1}</span>
          <strong class="sg-ticker">${esc(s.ticker)}</strong>
          <span class="muted">${esc(s.name || '')}</span>
          <span class="sg-score mono" title="screen score out of 100">${Math.round(s.score || 0)}<small>/100</small></span>
        </div>
        <div class="sg-metrics">${chips.map(c => `<span class="chip">${esc(c)}</span>`).join('')}</div>
        <p class="sg-why">${esc(s.rationale || '')}</p>
        <p class="sg-risk">${icon('alert')}<span>${esc(s.risks || '')}</span></p>
        <button class="btn sg-analyze" data-ticker="${esc(s.ticker)}">${icon('search')}Deep-analyze ${esc(s.ticker)}</button>
      </div>`;
  });
  $('#sg-table').innerHTML = html;
}

async function startSuggest() {
  const payload = {
    goal: $('#sg-goal').value.trim(),
    target_pct: parseFloat($('#sg-target').value),
    horizon_days: parseInt($('#sg-days').value, 10),
    count: parseInt($('#sg-count').value, 10),
    risk: $('#sg-risk').value,
    universe: $('#sg-universe').value.trim() || null,
    ...collectModelSettings(),
  };
  $('#sg-error').textContent = '';
  $('#sg-run').disabled = true;
  try {
    const run = await api('/api/suggest', { method: 'POST', body: JSON.stringify(payload) });
    resumeSuggest(run.run_id, true);
  } catch (err) {
    $('#sg-error').textContent = err.message;
  } finally {
    $('#sg-run').disabled = false;
  }
}

function resumeSuggest(runId, reset) {
  if (state.sgConn) state.sgConn.close();
  state.sgRunId = runId;
  if (reset) resetSuggestView();
  $('#sg-status').classList.remove('hidden');
  state.sgConn = connectEvents(runId, applySuggestEvent);
}

function openSuggestionInAnalysis(ticker) {
  $('#ticker').value = ticker;
  applyAssetType();
  showView('analyze');
  window.scrollTo({ top: 0 });
}

/* =============================== SETTINGS =============================== */

function renderKeyStatus() {
  $('#set-key-status').innerHTML = META.providers.map(p => {
    const info = META.api_keys[p.key] || {};
    const ok = info.set || info.optional;
    const text = info.env ? `${info.env} ${info.set ? '✓' : '— not set'}` : 'no key needed';
    return `<span class="keytag ${ok ? 'ok' : 'missing'}">${esc(p.key)}: ${esc(text)}</span>`;
  }).join('');
}

async function loadSettings() {
  $('#set-results').textContent = META.defaults.results_dir;
  renderKeyStatus();

  const locks = Object.entries(META.env_locked || {});
  $('#set-env-locks').innerHTML = locks.length
    ? locks.map(([k, v]) => `${esc(k)}=${esc(v)}`).join('<br>')
    : 'no TRADINGAGENTS_* overrides detected';

  try {
    const runs = await api('/api/runs');
    const active = runs.runs.find(r => !TERMINAL.includes(r.status));
    $('#set-active').textContent = active
      ? `${active.kind} ${active.ticker || ''} (${active.status})`
      : 'none';
  } catch (_) { /* informational only */ }

  try {
    const data = await api('/api/announcements');
    const box = $('#announcements');
    if (data.announcements && data.announcements.length) {
      box.classList.remove('hidden');
      box.innerHTML = '<strong>Announcements</strong><ul>' +
        data.announcements.map(a => `<li>${esc(a)}</li>`).join('') + '</ul>';
    }
  } catch (_) { /* announcements are optional */ }
}

async function saveSettingsKey() {
  const provider = $('#set-provider').value;
  const key = $('#set-key').value.trim();
  const msg = $('#set-msg');
  if (!key) { msg.textContent = 'Paste a key first.'; return; }
  try {
    const res = await api('/api/apikey', { method: 'POST', body: JSON.stringify({ provider, key }) });
    META.api_keys[provider] = { env: res.env, set: true, optional: false };
    $('#set-key').value = '';
    msg.textContent = `Saved ${res.env} to ${res.saved_to}`;
    renderKeyStatus();
    refreshProviderUI();
  } catch (err) {
    msg.textContent = err.message;
  }
}

/* ============================== ACTIVE RUN ============================== */

async function checkActiveRun() {
  try {
    const data = await api('/api/runs');
    const unfinished = data.runs.filter(r => !TERMINAL.includes(r.status));
    const pill = $('#active-run-pill');
    if (!unfinished.length) { pill.classList.add('hidden'); return; }
    const run = unfinished[0];
    pill.classList.remove('hidden');
    pill.classList.add('busy');
    const label = run.kind === 'analysis'
      ? `Live run · ${run.ticker || ''} ${run.analysis_date || ''}`
      : run.kind === 'backtest' ? 'Backtest running' : 'Screening stock ideas';
    pill.textContent = label;
    pill.onclick = () => {
      if (run.kind === 'analysis') openRun(run.run_id);
      else if (run.kind === 'backtest') { showView('backtest'); resumeBacktest(run.run_id, true); }
      else { showView('suggest'); resumeSuggest(run.run_id, true); }
    };
    pill.onkeydown = (e) => { if (e.key === 'Enter') pill.click(); };

    // Re-open the run so a page reload never loses a run in flight.
    if (run.kind === 'analysis' && !state.runId) openRun(run.run_id);
    if (run.kind === 'analysis') {
      const badge = $('#run-status');
      if (badge.classList.contains('queued')) badge.classList.add('running');
    }
  } catch (_) { /* the API may not be up yet */ }
}

/* ================================= WIRE ================================== */

function wire() {
  $('#analyze-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const btn = $('#start-run');
    const errBox = $('#form-error');
    errBox.textContent = '';
    btn.disabled = true;
    try {
      const run = await api('/api/analyze', {
        method: 'POST', body: JSON.stringify(collectAnalysisPayload()),
      });
      openRun(run.run_id);
    } catch (err) {
      errBox.textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  });

  $('#api-key-save').addEventListener('click', saveKeyFromForm);
  $('#llm_provider').addEventListener('change', () => {
    refreshProviderUI(); updateBacktestHint(); updateSuggestHint();
  });
  ['#quick_think_llm', '#deep_think_llm', '#backend_url'].forEach(sel =>
    $(sel).addEventListener('change', () => { updateBacktestHint(); updateSuggestHint(); }));
  $('#quick_think_llm').addEventListener('change',
    () => $('#quick_custom').classList.toggle('hidden', $('#quick_think_llm').value !== 'custom'));
  $('#deep_think_llm').addEventListener('change',
    () => $('#deep_custom').classList.toggle('hidden', $('#deep_think_llm').value !== 'custom'));
  $('#ticker').addEventListener('input', applyAssetType);
  $('#btn-save-report').addEventListener('click', saveReport);
  $('#btn-new-analysis').addEventListener('click', () => showView('analyze'));
  $('#btn-exec-preview').addEventListener('click', () => executeTrade('preview'));
  $('#btn-exec-live').addEventListener('click', () => executeTrade('live'));

  $('#backtest-form').addEventListener('submit', startBacktest);

  $('#sg-goal').addEventListener('input', scheduleGoalParse);
  $('#sg-goal').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); startSuggest(); }
  });
  $('#sg-run').addEventListener('click', startSuggest);
  $('#sg-table').addEventListener('click', (e) => {
    const btn = e.target.closest('.sg-analyze');
    if (btn) openSuggestionInAnalysis(btn.dataset.ticker);
  });

  $('#set-save').addEventListener('click', saveSettingsKey);
  $('#btn-clear-checkpoints').addEventListener('click', async () => {
    try {
      const res = await api('/api/clear-checkpoints', { method: 'POST', body: '{}' });
      $('#clear-msg').textContent = `Cleared ${res.cleared} checkpoint(s).`;
    } catch (err) {
      $('#clear-msg').textContent = err.message;
    }
  });

  $$('.tab[data-view="settings"]').forEach(t => t.addEventListener('click', loadSettings));
}

async function boot() {
  wire();
  try {
    META = await api('/api/meta');
  } catch (err) {
    toast('Could not load settings: ' + err.message, true);
    return;
  }
  $('#set-provider').innerHTML = META.providers
    .map(p => `<option value="${p.key}">${esc(p.label)}</option>`).join('');
  fillPrefilledForm();
  applyAssetType();
  updateBacktestHint();
  updateSuggestHint();
  startElapsedTicker();
  checkActiveRun();
  setInterval(checkActiveRun, 15000);
  document.dispatchEvent(new CustomEvent('ta:ready'));
}

// Shrink the sticky header once the page scrolls.
window.addEventListener('scroll', () => {
  $('.topbar').classList.toggle('scrolled', window.scrollY > 12);
}, { passive: true });

// Shared with chart.js, portfolio.js and chatbot.js (loaded after this file).
window.TA = {
  $, $$, api, esc, toast, icon, showView, setBadge,
  meta: () => META,
  collectModelSettings: () => (META ? collectModelSettings() : {}),
  openAnalysisFor: openSuggestionInAnalysis,
};

document.addEventListener('DOMContentLoaded', boot);
})();
