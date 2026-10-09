/* <StockChart> — candlestick / area chart in plain SVG: no library, no CSP
   exception. Volume bars sit under the price pane; SMA 20/50 come from the
   server (computed on the full history, so short ranges still have them). */
(() => {
'use strict';
const T = window.TA;

const PAD = { top: 12, right: 58, bottom: 24, left: 6 };
const VOL_H = 56;          // volume pane height
const MAX_CANDLES = 180;   // beyond this, candles are too thin — draw an area

function fmtPrice(v) {
  if (v == null) return '—';
  const digits = Math.abs(v) >= 1000 ? 0 : Math.abs(v) >= 1 ? 2 : 4;
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function fmtVol(v) {
  if (!v) return '—';
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  return String(Math.round(v));
}
function fmtTime(iso, intraday, long) {
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  if (intraday) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return d.toLocaleDateString([], long ? { month: 'short', year: '2-digit' } : { month: 'short', day: 'numeric' });
}

class StockChart {
  constructor({ canvas, tip, onLoad }) {
    this.canvas = canvas;
    this.tip = tip;
    this.onLoad = onLoad;
    this.mode = 'candle';
    this.data = null;
    this.layout = null;
    this.reqId = 0;
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener('pointermove', (e) => this.hover(e));
    canvas.addEventListener('pointerleave', () => this.clearHover());
  }

  setMode(mode) { this.mode = mode; this.draw(); }

  // ``silent`` is a background refresh: keep the chart on screen while the
  // request is in flight, and keep it if the request fails.
  async load(ticker, range, { silent = false } = {}) {
    const id = ++this.reqId;            // ignore answers to superseded requests
    this.ticker = ticker;
    this.range = range;
    if (!silent || !this.data) this.state('Loading ' + ticker + '…');
    try {
      const data = await T.api(`/api/chart/${encodeURIComponent(ticker)}?range=${encodeURIComponent(range)}`);
      if (id !== this.reqId) return;
      this.data = data;
      this.draw();
      if (this.onLoad) this.onLoad(data);
    } catch (err) {
      if (id !== this.reqId || silent) return;
      this.data = null;
      this.state(err.message);
      if (this.onLoad) this.onLoad(null, err);
    }
  }

  state(text) {
    this.canvas.innerHTML = `<div class="chart-state">${T.esc(text)}</div>`;
    this.clearHover();
  }

  draw() {
    const d = this.data;
    if (!d || !d.candles.length) return;
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    if (W < 50) return;
    const bars = d.candles;
    const n = bars.length;
    const intraday = d.interval !== '1d';
    const candles = this.mode === 'candle' && n <= MAX_CANDLES;

    const priceBottom = H - PAD.bottom - VOL_H - 8;
    const plotW = W - PAD.left - PAD.right;
    const step = plotW / n;
    const x = (i) => PAD.left + step * (i + 0.5);

    let lo = Infinity; let hi = -Infinity;
    for (const b of bars) {
      lo = Math.min(lo, candles ? b.l : b.c, b.sma20 ?? Infinity, b.sma50 ?? Infinity);
      hi = Math.max(hi, candles ? b.h : b.c, b.sma20 ?? -Infinity, b.sma50 ?? -Infinity);
    }
    const span = (hi - lo) || hi || 1;
    lo -= span * 0.06; hi += span * 0.06;
    const y = (v) => PAD.top + (hi - v) / (hi - lo) * (priceBottom - PAD.top);
    const vMax = Math.max(...bars.map(b => b.v || 0)) || 1;
    const volTop = H - PAD.bottom - VOL_H;
    const vy = (v) => volTop + VOL_H - (v / vMax) * VOL_H;

    const up = (d.change ?? 0) >= 0;
    const parts = [];

    // horizontal grid + price axis
    for (let k = 0; k <= 4; k++) {
      const v = lo + (hi - lo) * (k / 4);
      const yy = y(v).toFixed(1);
      parts.push(`<line class="grid" x1="${PAD.left}" x2="${W - PAD.right}" y1="${yy}" y2="${yy}"/>`);
      parts.push(`<text class="axis" x="${W - PAD.right + 8}" y="${(+yy + 3.5).toFixed(1)}">${fmtPrice(v)}</text>`);
    }
    // time axis
    const ticks = Math.min(6, n);
    for (let k = 0; k < ticks; k++) {
      const i = Math.round((n - 1) * (k / Math.max(1, ticks - 1)));
      const anchor = k === 0 ? 'start' : k === ticks - 1 ? 'end' : 'middle';
      parts.push(`<text class="axis" x="${x(i).toFixed(1)}" y="${H - 6}" text-anchor="${anchor}">` +
        `${T.esc(fmtTime(bars[i].t, intraday, d.range === 'ALL' || d.range === '1Y'))}</text>`);
    }

    // volume
    const bw = Math.max(1, step * 0.66);
    bars.forEach((b, i) => {
      const cls = b.c >= b.o ? 'vol-up' : 'vol-down';
      const top = vy(b.v || 0);
      parts.push(`<rect class="${cls}" x="${(x(i) - bw / 2).toFixed(1)}" y="${top.toFixed(1)}" ` +
        `width="${bw.toFixed(1)}" height="${(volTop + VOL_H - top).toFixed(1)}" rx="1"/>`);
    });

    if (candles) {
      bars.forEach((b, i) => {
        const dir = b.c >= b.o ? 'up' : 'down';
        const xx = x(i).toFixed(1);
        const top = y(Math.max(b.o, b.c));
        const bodyH = Math.max(1, Math.abs(y(b.o) - y(b.c)));
        parts.push(`<line class="wick-${dir}" x1="${xx}" x2="${xx}" y1="${y(b.h).toFixed(1)}" y2="${y(b.l).toFixed(1)}"/>`);
        parts.push(`<rect class="body-${dir}" x="${(x(i) - bw / 2).toFixed(1)}" y="${top.toFixed(1)}" ` +
          `width="${bw.toFixed(1)}" height="${bodyH.toFixed(1)}" rx="1"/>`);
      });
    } else {
      const dir = up ? 'up' : 'down';
      const line = bars.map((b, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(b.c).toFixed(1)}`).join('');
      const area = `${line}L${x(n - 1).toFixed(1)},${priceBottom}L${x(0).toFixed(1)},${priceBottom}Z`;
      parts.unshift(`<defs><linearGradient id="ch-fill" x1="0" y1="0" x2="0" y2="1">` +
        `<stop offset="0" class="stop-${dir}" stop-opacity=".32"/>` +
        `<stop offset="1" class="stop-${dir}" stop-opacity="0"/></linearGradient></defs>`);
      parts.push(`<path d="${area}" fill="url(#ch-fill)"/>`);
      parts.push(`<path class="line-${dir}" d="${line}"/>`);
    }

    // moving averages
    for (const key of ['sma20', 'sma50']) {
      let path = ''; let pen = false;
      bars.forEach((b, i) => {
        if (b[key] == null) { pen = false; return; }
        path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(b[key]).toFixed(1)}`;
        pen = true;
      });
      if (path) parts.push(`<path class="${key}" d="${path}"/>`);
    }

    parts.push('<g id="ch-cross"></g>');
    this.canvas.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" ` +
      `aria-label="${T.esc(d.ticker)} price chart">${parts.join('')}</svg>`;
    this.layout = { x, y, step, W, H, n, intraday, priceBottom };
  }

  hover(e) {
    const d = this.data; const L = this.layout;
    if (!d || !L) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const i = Math.max(0, Math.min(L.n - 1, Math.floor((px - PAD.left) / L.step)));
    const b = d.candles[i];
    const cx = L.x(i); const cy = L.y(b.c);
    const g = this.canvas.querySelector('#ch-cross');
    if (g) {
      g.innerHTML = `<line class="cross" x1="${cx}" x2="${cx}" y1="${PAD.top}" y2="${L.H - PAD.bottom}"/>` +
        `<line class="cross" x1="${PAD.left}" x2="${L.W - PAD.right}" y1="${cy}" y2="${cy}"/>` +
        `<circle class="cross-dot" cx="${cx}" cy="${cy}" r="4"/>`;
    }
    const when = new Date(b.t);
    const label = isNaN(when) ? b.t : (L.intraday
      ? when.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
      : when.toLocaleDateString([], { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' }));
    const chg = b.c - b.o;
    this.tip.innerHTML = `<div class="tt">${T.esc(label)}</div><dl>` +
      `<dt>Open</dt><dd>${fmtPrice(b.o)}</dd><dt>High</dt><dd>${fmtPrice(b.h)}</dd>` +
      `<dt>Low</dt><dd>${fmtPrice(b.l)}</dd>` +
      `<dt>Close</dt><dd class="${chg >= 0 ? 'up' : 'down'}">${fmtPrice(b.c)}</dd>` +
      `<dt>Volume</dt><dd>${fmtVol(b.v)}</dd>` +
      `<dt>SMA 20</dt><dd>${fmtPrice(b.sma20)}</dd><dt>SMA 50</dt><dd>${fmtPrice(b.sma50)}</dd></dl>`;
    this.tip.classList.remove('hidden');
    const tw = this.tip.offsetWidth; const th = this.tip.offsetHeight;
    let left = cx + 16; if (left + tw > L.W) left = cx - tw - 16;
    this.tip.style.left = Math.max(0, left) + 'px';
    this.tip.style.top = Math.max(0, Math.min(L.H - th, cy - th / 2)) + 'px';
  }

  clearHover() {
    if (this.tip) this.tip.classList.add('hidden');
    const g = this.canvas.querySelector('#ch-cross');
    if (g) g.innerHTML = '';
  }
}

T.StockChart = StockChart;
T.fmtPrice = fmtPrice;
})();
