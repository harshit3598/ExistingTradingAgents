/* <PortfolioTab> — summary cards, allocation donut, holdings table, editor,
   Robinhood sync, the price chart, and dry-run Buy/Sell previews. */
(() => {
'use strict';
const T = window.TA;
const { $, $$, esc } = T;

const state = { data: null, allocMode: 'category', loaded: false, chart: null, range: '1M', selected: null };
const REFRESH_MS = 60000;   // matches the server's one-minute intraday cache

/* ------------------------------- formatting ------------------------------ */

function money(v, { signed = false, compact = false } = {}) {
  if (v == null) return '—';
  const abs = Math.abs(v);
  const body = compact && abs >= 1e6
    ? (abs / 1e6).toFixed(2) + 'M'
    : abs.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const sign = v < 0 ? '−' : (signed ? '+' : '');
  return `${sign}$${body}`;
}
function pct(v) {
  if (v == null) return '—';
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}%`;
}
function tone(v) { return v == null ? '' : v >= 0 ? 'up' : 'down'; }
function qty(v) { return Number(v).toLocaleString(undefined, { maximumFractionDigits: 6 }); }

/* --------------------------------- loading ------------------------------- */

// ``silent`` is the background refresh: no error banner, and a failed poll
// keeps what is on screen.
async function load({ silent = false } = {}) {
  if (!silent) $('#pf-error').classList.add('hidden');
  try {
    render(await T.api('/api/portfolio'));
    state.updatedAt = new Date();
  } catch (err) {
    if (!silent) showError(err.message);
  }
}

function showError(msg) {
  const box = $('#pf-error');
  box.textContent = msg;
  box.classList.remove('hidden');
}

function render(data) {
  state.data = data;
  state.loaded = true;
  const src = { robinhood: 'Synced from Robinhood', manual: 'Saved holdings', none: 'No holdings yet' }[data.source] || data.source;
  $('#pf-source').textContent = src + (data.updated_at ? ` · ${new Date(data.updated_at).toLocaleString()}` : '');
  const open = data.market_status === 'open';
  const market = $('#pf-market');
  market.textContent = open ? 'Market open' : 'Market closed';
  market.classList.toggle('on', open);
  if (data.error) showError(data.error);
  market.title = state.updatedAt ? `Updated ${state.updatedAt.toLocaleTimeString()} · refreshes every minute` : '';

  renderMetrics(data);
  renderAllocation();
  renderHoldings(data);
  if (!state.selected) {
    const first = data.holdings && data.holdings[0];
    selectTicker(first ? first.ticker : 'SPY');
  }
}

/* --------------------------------- metrics ------------------------------- */

function metric(label, value, sub, cls) {
  return `<div class="metric ${cls}"><span class="metric-k">${esc(label)}</span>` +
    `<span class="metric-v">${value}</span><span class="metric-s">${sub}</span></div>`;
}

function renderMetrics(data) {
  const box = $('#pf-metrics');
  if (data.empty) {
    box.innerHTML = `<div class="metric neutral"><span class="metric-k">Get started</span>` +
      `<span class="metric-v">Add your holdings</span><span class="metric-s muted">` +
      `Use “Edit holdings”${T.meta()?.robinhood_configured ? ' or sync from Robinhood' : ''} to value your book here.</span></div>`;
    return;
  }
  const s = data.summary;
  const dayCls = s.day_change == null ? 'neutral' : s.day_change >= 0 ? 'pos' : 'neg';
  const allCls = s.all_time_return == null ? 'neutral' : s.all_time_return >= 0 ? 'pos' : 'neg';
  box.innerHTML = [
    metric('Total portfolio value', money(s.total_value),
      `<span class="muted">Invested ${money(s.positions_value)}</span>`, 'neutral'),
    metric("Day's gain / loss", `<span class="${tone(s.day_change)}">${money(s.day_change, { signed: true })}</span>`,
      `<span class="${tone(s.day_change_pct)}">${pct(s.day_change_pct)} today</span>`, dayCls),
    metric('All-time return', `<span class="${tone(s.all_time_return)}">${money(s.all_time_return, { signed: true })}</span>`,
      s.all_time_return == null
        ? '<span class="muted">Add average costs to track it</span>'
        : `<span class="${tone(s.all_time_return_pct)}">${pct(s.all_time_return_pct)} on cost</span>`, allCls),
    metric('Buying power', money(s.buying_power),
      s.unpriced && s.unpriced.length
        ? `<span class="down">Unpriced: ${esc(s.unpriced.join(', '))}</span>`
        : '<span class="muted">Free cash in the book</span>', 'neutral'),
  ].join('');
}

/* -------------------------------- allocation ----------------------------- */

function renderAllocation() {
  const data = state.data;
  const box = $('#pf-alloc');
  const list = data && !data.empty
    ? (state.allocMode === 'holding' ? data.allocation_by_holding : data.allocation) : [];
  if (!list || !list.length) {
    box.innerHTML = '<p class="empty">Nothing to allocate yet.</p>';
    return;
  }
  const shown = list.slice(0, 7);
  const rest = list.slice(7);
  if (rest.length) {
    shown.push({ label: `Other (${rest.length})`,
      value: rest.reduce((a, r) => a + r.value, 0), pct: rest.reduce((a, r) => a + r.pct, 0) });
  }
  const R = 80; const C = 2 * Math.PI * R;
  let offset = 0;
  const arcs = shown.map((a, i) => {
    const len = Math.max(0, (a.pct / 100) * C - 2);   // 2px gap between segments
    const arc = `<circle class="c${i % 8}" data-i="${i}" r="${R}" cx="110" cy="110" ` +
      `stroke-dasharray="${len.toFixed(2)} ${(C - len).toFixed(2)}" stroke-dashoffset="${(-offset).toFixed(2)}"/>`;
    offset += (a.pct / 100) * C;
    return arc;
  }).join('');
  const total = data.summary.total_value;
  box.innerHTML =
    `<div class="donut-wrap"><svg viewBox="0 0 220 220" aria-hidden="true">` +
    `<circle class="track" r="${R}" cx="110" cy="110"/>${arcs}</svg>` +
    `<div class="donut-center"><b id="pf-donut-v">${money(total, { compact: true })}</b>` +
    `<span id="pf-donut-k">Total value</span></div></div>` +
    `<div class="legend">${shown.map((a, i) => `<div class="legend-row" data-i="${i}"><i class="c${i % 8}"></i>` +
      `<span>${esc(a.label)}</span><b>${money(a.value, { compact: true })}</b><em>${a.pct.toFixed(1)}%</em></div>`).join('')}</div>`;

  const hot = (i) => {
    $$('.donut-wrap circle[data-i]', box).forEach(c => c.classList.toggle('seg-hot', c.dataset.i === String(i)));
    const a = shown[i];
    $('#pf-donut-v').textContent = a ? `${a.pct.toFixed(1)}%` : money(total, { compact: true });
    $('#pf-donut-k').textContent = a ? a.label : 'Total value';
  };
  $$('[data-i]', box).forEach(el => {
    el.addEventListener('pointerenter', () => hot(el.dataset.i));
    el.addEventListener('pointerleave', () => hot(null));
  });
}

/* --------------------------------- holdings ------------------------------ */

function renderHoldings(data) {
  const box = $('#pf-table');
  const when = data.as_of && data.as_of.includes('T') ? new Date(data.as_of) : null;
  $('#pf-asof').textContent = !data.as_of ? ''
    : when ? `${data.live ? 'Live · ' : ''}prices as of ${when.toLocaleTimeString()} (delayed up to ~15 min)`
      : `Prices as of ${data.as_of} close`;
  if (data.empty || !data.holdings.length) {
    box.innerHTML = '<p class="empty">No positions yet — press “Edit holdings” to add some.</p>';
    return;
  }
  const rows = data.holdings.map(h => `
    <tr data-ticker="${esc(h.ticker)}"${h.ticker === state.selected ? ' class="selected"' : ''}>
      <td><div class="tk"><span class="tk-badge">${esc(h.ticker.slice(0, 4))}</span>
        <div><b>${esc(h.ticker)}</b><small>${esc(h.name || '')} · ${esc(h.category)}</small></div></div></td>
      <td>${qty(h.quantity)}</td>
      <td>${money(h.average_price)}</td>
      <td>${money(h.price)}</td>
      <td class="${tone(h.day_change)}">${money(h.day_change, { signed: true })}<br><small>${pct(h.day_change_pct)}</small></td>
      <td><b>${money(h.value)}</b></td>
      <td class="${tone(h.gain)}">${money(h.gain, { signed: true })}<br><small>${pct(h.gain_pct)}</small></td>
      <td><span class="actions">
        <button class="btn xs buy" data-act="buy" title="Dry-run buy preview">Buy</button>
        <button class="btn xs sell" data-act="sell" title="Dry-run sell preview">Sell</button>
        <button class="btn xs" data-act="analyze" title="Run a full analysis">${T.icon('search')}Analyze</button>
      </span></td>
    </tr>`).join('');
  box.innerHTML = `<table class="holdings"><thead><tr>
      <th>Asset</th><th>Shares</th><th>Avg cost</th><th>Price</th><th>Day change</th>
      <th>Total value</th><th>Return</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function onTableClick(e) {
  const row = e.target.closest('tr[data-ticker]');
  if (!row) return;
  const ticker = row.dataset.ticker;
  const act = e.target.closest('[data-act]')?.dataset.act;
  if (act === 'analyze') { T.openAnalysisFor(ticker); return; }
  if (act === 'buy' || act === 'sell') { previewOrder(ticker, act); return; }
  selectTicker(ticker);
}

async function previewOrder(ticker, side) {
  const out = $('#pf-preview');
  out.classList.remove('hidden');
  out.innerHTML = `<div class="notice">Running risk check → sizing → dry-run broker for ${esc(side.toUpperCase())} ${esc(ticker)}…</div>`;
  try {
    const res = await T.api('/api/portfolio/preview-order', {
      method: 'POST', body: JSON.stringify({ ticker, side }),
    });
    const r = res.result; const o = r.order || {};
    const cells = [
      ['Mode', 'Dry run — no order sent'], ['Action', (o.action || side).toUpperCase()],
      ['Quantity', o.quantity != null ? qty(o.quantity) : '—'], ['Limit price', money(o.limit_price)],
      ['Order value', o.quantity && o.limit_price ? money(o.quantity * o.limit_price) : '—'],
    ].map(([k, v]) => `<div><span class="muted small">${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');
    const risk = r.risk ? `<p class="exec-line"><b>Risk</b><span class="${r.risk.passed ? 'ok-text' : 'bad-text'}">` +
      `${r.risk.passed ? 'Passed' : 'Blocked'}</span> ${esc(r.risk.reason)}</p>` : '';
    const sizing = r.sizing ? `<p class="exec-line"><b>Sizing</b>${esc(r.sizing.reasoning)}</p>` : '';
    out.innerHTML = `<div class="notice ${r.executed ? 'ok' : 'warn'}"><div class="card-head">` +
      `<strong>${r.executed ? 'Order simulated' : 'Not executable'}: ${esc(r.message || '')}</strong>` +
      `<button class="icon-btn" data-close aria-label="Dismiss">${T.icon('x')}</button></div>` +
      `<div class="exec-grid">${cells}</div>${risk}${sizing}</div>`;
    $('[data-close]', out).addEventListener('click', () => out.classList.add('hidden'));
  } catch (err) {
    out.innerHTML = `<div class="notice danger">${esc(err.message)}</div>`;
  }
}

/* ---------------------------------- chart -------------------------------- */

function selectTicker(ticker) {
  state.selected = ticker;
  T.currentTicker = ticker;
  $$('#pf-table tr[data-ticker]').forEach(r => r.classList.toggle('selected', r.dataset.ticker === ticker));
  $('#ch-input').value = '';
  state.chart.load(ticker, state.range);
}

function onChartLoaded(d) {
  if (!d) return;
  $('#ch-ticker').textContent = d.ticker;
  $('#ch-name').textContent = d.name || '';
  $('#ch-price').textContent = T.fmtPrice(d.last);
  const ch = $('#ch-change');
  ch.textContent = `${d.change >= 0 ? '+' : '−'}${T.fmtPrice(Math.abs(d.change))} (${pct(d.change_pct)}) · ${d.range}`;
  ch.className = 'metric-s ' + tone(d.change);
  const live = $('#ch-live');
  const isLive = d.market_status === 'open' && d.range === '1D';
  live.textContent = isLive ? 'Live · 5m' : d.market_status === 'open' ? 'Market open' : 'Market closed';
  live.classList.toggle('on', isLive);
  const when = new Date(d.as_of);
  $('#ch-asof').textContent = isNaN(when) ? '' : `as of ${when.toLocaleString([], d.interval === '1d'
    ? { year: 'numeric', month: 'short', day: 'numeric' } : { hour: '2-digit', minute: '2-digit' })}`;
}

/* --------------------------------- editor -------------------------------- */

function editorRow(p = {}) {
  const row = document.createElement('div');
  row.className = 'editor-row';
  row.innerHTML =
    `<input class="ed-t" type="text" placeholder="AAPL" spellcheck="false" value="${esc(p.ticker || '')}">` +
    `<input class="ed-q" type="number" step="any" placeholder="10" value="${p.quantity ?? ''}">` +
    `<input class="ed-a" type="number" step="any" min="0" placeholder="optional" value="${p.average_price ?? ''}">` +
    `<button type="button" class="icon-btn" aria-label="Remove">${T.icon('trash')}</button>`;
  $('button', row).addEventListener('click', () => row.remove());
  return row;
}

function openEditor() {
  const book = state.data?.book || { positions: [] };
  const rows = $('#pf-rows');
  rows.innerHTML = '';
  (book.positions.length ? book.positions : [{}]).forEach(p => rows.appendChild(editorRow(p)));
  $('#pf-cash').value = book.cash ?? '';
  $('#pf-save-msg').textContent = '';
  $('#pf-rh').classList.toggle('hidden', !T.meta()?.robinhood_configured);
  $('#pf-editor').classList.remove('hidden');
  $('#pf-editor').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function saveEditor() {
  const positions = $$('#pf-rows .editor-row').map(r => ({
    ticker: $('.ed-t', r).value.trim().toUpperCase(),
    quantity: parseFloat($('.ed-q', r).value),
    average_price: $('.ed-a', r).value === '' ? null : parseFloat($('.ed-a', r).value),
  })).filter(p => p.ticker || !Number.isNaN(p.quantity));
  const cash = $('#pf-cash').value === '' ? null : parseFloat($('#pf-cash').value);
  const btn = $('#pf-save');
  btn.disabled = true;
  try {
    render(await T.api('/api/portfolio', {
      method: 'PUT', body: JSON.stringify({ cash, currency: 'USD', positions }),
    }));
    $('#pf-editor').classList.add('hidden');
    T.toast('Holdings saved');
  } catch (err) {
    $('#pf-save-msg').textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

async function syncRobinhood() {
  const btn = $('#pf-rh-sync');
  btn.disabled = true;
  btn.classList.add('loading');
  try {
    const res = await T.api('/api/portfolio/sync-robinhood', {
      method: 'POST', body: JSON.stringify({ mfa_code: $('#pf-rh-mfa').value.trim() || null }),
    });
    $('#pf-rh-mfa').value = '';
    $('#pf-editor').classList.add('hidden');
    render(res);
    T.toast(`Synced ${res.synced.positions} position(s) from Robinhood`);
  } catch (err) {
    $('#pf-save-msg').textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
  }
}

/* ---------------------------------- wire --------------------------------- */

// Once a minute, while the tab is showing: re-value the book, and redraw the
// chart when it is on the 1D view (the only one with intraday bars).
function refreshLive() {
  if (document.hidden || $('#view-portfolio').hidden || !state.loaded) return;
  load({ silent: true });
  if (state.range === '1D' && state.selected) state.chart.load(state.selected, '1D', { silent: true });
}

function segment(root, attr, onPick) {
  root.addEventListener('click', (e) => {
    const b = e.target.closest(`button[data-${attr}]`);
    if (!b) return;
    $$('button', root).forEach(x => x.classList.toggle('active', x === b));
    onPick(b.dataset[attr]);
  });
}

function init() {
  state.chart = new T.StockChart({ canvas: $('#ch-canvas'), tip: $('#ch-tip'), onLoad: onChartLoaded });
  $('#pf-table').addEventListener('click', onTableClick);
  $('#pf-refresh').addEventListener('click', load);
  $('#pf-edit').addEventListener('click', openEditor);
  $('#pf-cancel').addEventListener('click', () => $('#pf-editor').classList.add('hidden'));
  $('#pf-add-row').addEventListener('click', () => $('#pf-rows').appendChild(editorRow()));
  $('#pf-save').addEventListener('click', saveEditor);
  $('#pf-rh-sync').addEventListener('click', syncRobinhood);
  segment($('#pf-alloc-mode'), 'mode', (m) => { state.allocMode = m; renderAllocation(); });
  segment($('#ch-range'), 'range', (r) => { state.range = r; state.chart.load(state.selected || 'SPY', r); });
  segment($('#ch-mode'), 'mode', (m) => state.chart.setMode(m));
  $('#ch-search').addEventListener('submit', (e) => {
    e.preventDefault();
    const t = $('#ch-input').value.trim().toUpperCase();
    if (t) selectTicker(t);
  });
  document.addEventListener('ta:view', (e) => { if (e.detail === 'portfolio') load(); });
  setInterval(refreshLive, REFRESH_MS);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshLive(); });
}

// Lets the assistant's "Chart" action open a ticker here.
T.showChart = (ticker) => {
  T.showView('portfolio');
  selectTicker(ticker);
};

document.addEventListener('DOMContentLoaded', init);
})();
