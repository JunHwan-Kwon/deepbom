const DAY = 86400000;
const LABELS = { chatgpt: 'ChatGPT', web: 'Web', claude: 'Claude', cli: 'CLI', local_mcp: 'Local MCP' };
const COLORS = { chatgpt: '#158074', web: '#5975cc' };
const count = (rows, event, detail) => rows.filter((r) => r.event === event && (detail === undefined || r.detail === detail)).reduce((n, r) => n + r.count, 0);
const rate = (value) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const node = (tag, text = '', className = '') => { const n = document.createElement(tag); n.textContent = text; n.className = className; return n; };
const button = (text) => { const b = node('button', text); b.type = 'button'; return b; };
function select(label, entries, value) {
  const n = document.createElement('select'); n.setAttribute('aria-label', label);
  for (const [id, text] of entries) { const o = node('option', text); o.value = id; o.selected = id === value; n.append(o); }
  return n;
}
function table(headers, rows, caption) {
  const wrap = node('div', '', 'table-wrap'), t = document.createElement('table'); t.append(node('caption', caption));
  const head = document.createElement('thead'), tr = document.createElement('tr'); tr.append(...headers.map((h) => node('th', h))); head.append(tr);
  const body = document.createElement('tbody');
  for (const row of rows) { const tr = document.createElement('tr'); tr.append(...row.map((v) => node('td', v == null ? 'Not measured' : String(v)))); body.append(tr); }
  t.append(head, body); wrap.append(t); return wrap;
}
function chart(title, dates, series) {
  const wrap = node('div', '', 'usage-chart'); wrap.append(node('h3', title));
  const svgNode = (tag, attrs, text) => { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (text !== undefined) n.textContent = text; return n; };
  const svg = svgNode('svg', { viewBox: '0 0 900 220', role: 'img', 'aria-label': title });
  const max = Math.max(1, ...series.flatMap((s) => s.values.filter((v) => v != null)));
  for (const fraction of [0, 0.5, 1]) {
    const y = 180 - fraction * 150;
    svg.append(svgNode('line', { x1: 45, x2: 885, y1: y, y2: y, stroke: 'currentColor', opacity: 0.15 }));
    svg.append(svgNode('text', { x: 38, y: y + 4, 'text-anchor': 'end', fill: 'currentColor', 'font-size': 12 }, String(Math.round(max * fraction))));
  }
  const slot = 840 / Math.max(1, dates.length), width = Math.max(1, slot / Math.max(1, series.length) - 2);
  dates.forEach((date, i) => {
    series.forEach((s, j) => {
      const value = s.values[i]; if (value == null) return;
      const height = value / max * 150;
      const rect = svgNode('rect', { x: 45 + i * slot + j * slot / series.length, y: 180 - height, width, height });
      rect.style.fill = s.color; rect.append(svgNode('title', {}, `${date} UTC · ${s.label}: ${value}`)); svg.append(rect);
    });
    if (i === 0 || i === dates.length - 1 || i % Math.max(1, Math.ceil(dates.length / 6)) === 0) svg.append(svgNode('text', { x: 45 + i * slot, y: 204, fill: 'currentColor', 'font-size': 12, 'text-anchor': i === dates.length - 1 ? 'end' : 'start' }, date.slice(5)));
  });
  wrap.append(svg);
  const legend = node('p');
  for (const s of series) { const item = node('span', `■ ${s.label}  `); item.style.color = s.color; legend.append(item); }
  wrap.append(legend, node('p', 'Hover a bar for its UTC date and value. Missing dates are not measured.'));
  return wrap;
}
function dailyRows(data) {
  const channels = data.channels.filter((c) => data.channel === 'all' || c.id === data.channel), rows = [];
  for (let i = 0; i < data.period.days; i++) {
    const day = new Date(Date.parse(data.period.since) + i * DAY).toISOString().slice(0, 10);
    for (const c of channels) {
      const measured = c.status === 'measured_opt_in' && day >= c.measured_from.slice(0, 10);
      const events = data.events.filter((r) => r.channel === c.id && r.day === day);
      rows.push({ day, channel: c.id, measurement_status: measured ? (day === c.measured_from.slice(0, 10) || day === data.generated_at.slice(0, 10) ? 'partial_day' : 'observed') : 'not_measured',
        started: measured ? count(events, 'analysis_started') : null, completed: measured ? count(events, 'analysis_completed') : null, failed: measured ? count(events, 'analysis_failed') : null,
        browsers: measured ? data.daily_browsers.find((r) => r.channel === c.id && r.day === day)?.browsers || 0 : null,
        exports: measured ? count(events, 'export_prepared') : null, handoffs: measured ? count(events, 'file_shared') : null });
    }
  }
  return rows;
}
const download = (name, text, type) => {
  const url = URL.createObjectURL(new Blob([text], { type })), a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
export function createAdminUsage(container, apiFetch) {
  const head = node('div', '', 'admin-section-head'), title = node('div');
  title.append(node('h2', 'Usage by channel'), node('p', 'Opt-in analysis reports and browser estimates. UTC dates; today and edge weeks may be partial.'));
  const controls = node('div', '', 'admin-bench-controls');
  const period = select('Usage period', [7, 30, 90].map((n) => [String(n), `${n} days`]), '30');
  const cohort = select('Usage population', [['public', 'Public use'], ['test', 'Declared tests']], 'public');
  const channel = select('Usage channel', [['all', 'All measured channels'], ...Object.entries(LABELS)], 'all');
  const refresh = button('Refresh usage'), csv = button('Download daily CSV'), json = button('Download evidence JSON'); csv.disabled = json.disabled = true;
  controls.append(period, channel, cohort, refresh, csv, json); head.append(title, controls);
  const status = node('p', 'Sign in as an administrator to view usage.'); status.setAttribute('role', 'status');
  const overview = node('div'), cards = node('div', '', 'admin-metrics'), charts = node('div'), tables = node('div', '', 'usage-tables');
  const service = node('div'), packages = node('div'), limits = node('p', '', 'usage-note');
  container.append(head, status, overview, cards, charts, tables, service, packages, limits);
  let latest = null, requestId = 0;
  const card = (label, value, detail) => { const n = node('div', '', 'admin-metric-card'); n.append(node('span', label), node('strong', value == null ? 'Not measured' : String(value)), node('p', detail)); return n; };
  function render(data) {
    status.textContent = `${data.enabled ? 'Collection enabled' : 'Collection disabled'} · ${data.cohort === 'test' ? 'Declared tests only' : 'Declared tests excluded'} · Snapshot ${data.generated_at}`;
    overview.replaceChildren(table(['Channel', 'Coverage begins (UTC)', 'Completed reports', 'Active browsers', 'Returning browsers'], data.channels.map((c) => [c.label, c.measured_from || 'Not measured', c.totals?.completed, c.totals?.unique_browsers, c.totals?.returning_browsers]), 'Channel coverage — browser counts are not additive across channels'));
    cards.replaceChildren(); charts.replaceChildren(); tables.replaceChildren();
    const t = data.totals, rows = dailyRows(data);
    if (!t) cards.append(node('p', 'Execution is not measured for this channel. No zero usage has been inferred.'));
    else {
      cards.append(card('Analysis reports', t.completed, `${t.started} started / ${t.failed} failed`), card('Without a terminal report', t.unfinished, 'Running, interrupted or abandoned; not automatically failures'), card('Terminal success rate', rate(t.terminal_success_rate), 'Completed ÷ (completed + failed); excludes unfinished runs'), card('Prepared exports', t.export_preparations, 'Once per type per run; not confirmed downloads'));
      if (data.channel !== 'all') cards.append(card('Estimated active browsers', t.unique_browsers, 'Consenting browser instances; not people'), card('Returning browsers', t.returning_browsers, `${rate(t.returning_browser_rate)} observed on multiple UTC dates`));
      const selected = data.channels.filter((c) => c.status === 'measured_opt_in' && (data.channel === 'all' || c.id === data.channel));
      const dates = [...new Set(rows.map((r) => r.day))];
      charts.append(chart('Daily completed analysis reports', dates, selected.map((c) => ({ label: c.label, color: COLORS[c.id], values: dates.map((day) => rows.find((r) => r.channel === c.id && r.day === day)?.completed ?? null) }))));
      const weeks = [...new Set(dates.map((day) => { const d = new Date(day); d.setUTCDate(d.getUTCDate() - (d.getUTCDay() + 6) % 7); return d.toISOString().slice(0, 10); }))];
      const weeklyValue = (c, week, metric) => Date.parse(week) + 7 * DAY <= Date.parse(c.measured_from) ? null : data.weekly_browsers.find((r) => r.channel === c.id && r.week === week)?.[metric] || 0;
      charts.append(chart('Weekly active and returning browsers — week beginning Monday', weeks, selected.flatMap((c) => [
        { label: `${c.label} active`, color: COLORS[c.id], values: weeks.map((w) => weeklyValue(c, w, 'browsers')) },
        { label: `${c.label} returning`, color: c.id === 'web' ? '#a8b7e1' : '#8abfb8', values: weeks.map((w) => weeklyValue(c, w, 'returning_browsers')) },
      ])));
      const formats = [...new Set(data.events.filter((r) => r.event === 'analysis_completed').map((r) => r.format))].sort();
      const exports = [...new Set(selected.flatMap((c) => c.export_types))];
      tables.append(
        table(['Format', 'Completed reports', 'Share'], formats.map((f) => { const n = count(data.events.filter((r) => r.format === f), 'analysis_completed'); return [f, n, rate(t.completed ? n / t.completed : null)]; }), 'Artifact format distribution'),
        table(['Export', 'Prepared', 'ChatGPT file handoffs'], exports.map((type) => [type, count(data.events, 'export_prepared', type), data.channel === 'web' ? 'Not offered' : count(data.events, 'file_shared', type)]), 'Export use — only instrumented export actions'),
        table(['View', 'Runs rendered'], [...new Set(data.events.filter((r) => r.event === 'visualization_rendered').map((r) => r.detail))].map((v) => [v, count(data.events, 'visualization_rendered', v)]), 'Visualizations rendered; not proof of human viewing'),
        table(['Week (UTC)', 'Channel', 'Active browsers', 'Returning browsers'], selected.flatMap((c) => weeks.map((w) => [w, c.label, weeklyValue(c, w, 'browsers'), weeklyValue(c, w, 'returning_browsers')])), 'Weekly browser estimates; partial edge weeks'),
      );
    }
    tables.append(table(['UTC date', 'Channel', 'Coverage', 'Started', 'Completed', 'Failed', 'Browsers', 'Prepared exports', 'File handoffs'], rows.map((r) => [r.day, LABELS[r.channel], r.measurement_status, r.started, r.completed, r.failed, r.browsers, r.exports, r.handoffs]), 'Daily usage — missing coverage is not zero activity'));
    const metrics = ['http_request', 'rpc_error', 'analyze_requested', 'analysis_report_received', 'analysis_error_received'];
    service.replaceChildren(node('h3', 'MCP service traffic — /mcp endpoint, all callers'), node('p', 'Independent of the channel and test filters above. Includes tests, bots and repeated reports; not unique users.'), table(['UTC date', 'HTTP requests', 'Protocol errors', 'Analysis requests', 'Result reports', 'Failure reports'], [...new Set(data.service.map((r) => r.day))].sort().map((day) => [day, ...metrics.map((metric) => data.service.filter((r) => r.day === day && r.metric === metric).reduce((n, r) => n + r.count, 0))]), 'Operational counts; not ChatGPT-only usage'));
    limits.textContent = Object.values(data.measurement).join(' ') + ' Records expire after 90 days. Keep these definitions with published evidence snapshots.';
  }
  function renderDownloads(data) {
    packages.replaceChildren(node('h3', 'CLI / local MCP distribution — npm downloads'), node('p', data.measurement), node('p', `${data.reported_total} reported downloads · ${data.missing_days} missing day(s) · Fetched ${data.fetched_at}`));
    const link = node('a', 'Source: npm download-count API'); link.href = data.source; link.target = '_blank'; link.rel = 'noopener'; packages.append(link);
    packages.append(chart('Daily npm package downloads — excludes today', data.daily.map((r) => r.day), [{ label: 'npm downloads', color: '#8a6caa', values: data.daily.map((r) => r.downloads) }]), table(['UTC date', 'Reported downloads'], data.daily.map((r) => [r.day, r.downloads]), 'Distribution evidence; never added to analysis or browser totals'));
  }
  function clear(message) {
    requestId++; latest = null; csv.disabled = json.disabled = true; status.textContent = message;
    for (const section of [overview, cards, charts, tables, service, packages]) section.replaceChildren(); limits.textContent = '';
  }
  async function load() {
    const id = ++requestId; refresh.disabled = true; csv.disabled = json.disabled = true; status.textContent = 'Loading usage evidence…';
    try {
      const data = await apiFetch(`/api/admin/usage?days=${period.value}&cohort=${cohort.value}&channel=${channel.value}`);
      if (id !== requestId) return;
      latest = data; render(data); packages.replaceChildren(node('p', 'Loading public npm download counts…'));
      try {
        const downloads = await apiFetch(`/api/admin/usage/downloads?days=${period.value}`);
        if (id !== requestId) return;
        latest = { ...data, package_downloads: downloads }; renderDownloads(downloads);
      } catch {
        if (id !== requestId) return;
        latest = { ...data, package_downloads: { status: 'unavailable' } };
        packages.replaceChildren(node('h3', 'CLI / local MCP distribution — npm downloads'), node('p', 'Download statistics are unavailable. No zero counts have been substituted.'));
      }
      csv.disabled = json.disabled = false;
    } catch (error) { if (id === requestId) { clear(error.message || 'Usage unavailable.'); refresh.disabled = false; } }
    finally { if (id === requestId) refresh.disabled = false; }
  }
  for (const control of [period, cohort, channel]) control.addEventListener('change', load);
  refresh.addEventListener('click', load);
  json.addEventListener('click', () => { if (latest) download(`deepbom-usage-${latest.channel}-${latest.cohort}-${latest.generated_at.slice(0, 10)}.json`, JSON.stringify(latest, null, 2) + '\n', 'application/json'); });
  csv.addEventListener('click', () => {
    if (!latest) return;
    const header = 'utc_date,channel,cohort,measurement_status,started,completed,failed,estimated_active_browsers,export_preparations,file_handoffs';
    const rows = dailyRows(latest).map((r) => [r.day, r.channel, latest.cohort, r.measurement_status, r.started, r.completed, r.failed, r.browsers, r.exports, r.handoffs].map((v) => v ?? '').join(','));
    download(`deepbom-usage-${latest.channel}-${latest.cohort}-${latest.generated_at.slice(0, 10)}.csv`, '\uFEFF' + [header, ...rows].join('\r\n'), 'text/csv;charset=utf-8');
  });
  return { load, clear };
}
