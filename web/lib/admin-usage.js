export function createAdminUsage(container, apiFetch) {
  const node = (tag, text, className = '') => {
    const item = document.createElement(tag); item.textContent = text; item.className = className; return item;
  };
  const head = node('div', '', 'admin-section-head');
  const title = node('div', ''); title.append(node('h2', 'ChatGPT usage evidence'), node('p', 'Consenting browser estimates and operational request counts. All dates are UTC.'));
  const controls = node('div', '', 'admin-bench-controls');
  const period = document.createElement('select'); period.setAttribute('aria-label', 'Usage period');
  for (const days of [7, 30, 90]) { const option = node('option', `${days} days`); option.value = days; option.selected = days === 30; period.append(option); }
  const cohort = document.createElement('select'); cohort.setAttribute('aria-label', 'Usage population');
  for (const [value, label] of [['public', 'Public use'], ['test', 'Declared tests']]) { const option = node('option', label); option.value = value; cohort.append(option); }
  const refresh = node('button', 'Refresh usage'); refresh.type = 'button';
  const csv = node('button', 'Download daily CSV'); csv.type = 'button'; csv.disabled = true;
  const json = node('button', 'Download evidence JSON'); json.type = 'button'; json.disabled = true;
  controls.append(period, cohort, refresh, csv, json); head.append(title, controls);
  const status = node('p', 'Sign in as an administrator to view usage.'); status.setAttribute('role', 'status');
  const cards = node('div', '', 'admin-metrics');
  const serviceCards = node('div', '', 'admin-metrics');
  const chart = node('div', '', 'usage-chart');
  const tables = node('div', '', 'usage-tables');
  const limits = node('p', '', 'usage-note');
  container.append(head, status, cards, chart, tables, node('h3', 'MCP service traffic — all callers'), serviceCards, limits);
  let latest = null;
  let requestId = 0;
  const card = (label, value, detail) => {
    const item = node('div', '', 'admin-metric-card'); item.append(node('span', label), node('strong', value), node('p', detail)); return item;
  };
  const rate = (value) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
  const count = (events, type, detail) => events.filter((event) => event.event === type && (detail === undefined || event.detail === detail)).reduce((total, event) => total + event.count, 0);
  const download = (filename, text, type) => {
    const href = URL.createObjectURL(new Blob([text], { type }));
    const link = document.createElement('a'); link.href = href; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  };
  function dailyRows(data) {
    const rows = [];
    for (let i = 0; i < data.period.days; i += 1) {
      const day = new Date(Date.parse(data.period.since) + i * 86400000).toISOString().slice(0, 10);
      const events = data.events.filter((event) => event.day === day);
      rows.push({ day, started: count(events, 'analysis_started'), completed: count(events, 'analysis_completed'), failed: count(events, 'analysis_failed'),
        browsers: data.daily_browsers.find((row) => row.day === day)?.browsers || 0,
        exports: count(events, 'export_prepared'), handoffs: count(events, 'file_shared') });
    }
    return rows;
  }
  function table(headers, rows, caption) {
    const wrap = node('div', '', 'table-wrap'); const table = document.createElement('table');
    table.append(node('caption', caption));
    const thead = document.createElement('thead'); const header = document.createElement('tr');
    header.append(...headers.map((value) => node('th', value))); thead.append(header);
    const body = document.createElement('tbody');
    for (const row of rows) { const tr = document.createElement('tr'); tr.append(...row.map((value) => node('td', String(value)))); body.append(tr); }
    table.append(thead, body); wrap.append(table); return wrap;
  }
  function render(data) {
    const t = data.totals;
    status.textContent = `${data.enabled ? 'Collection enabled' : 'Collection disabled'} · ${data.cohort === 'test' ? 'Declared tests only' : 'Declared tests excluded'} · Snapshot ${data.generated_at}`;
    cards.replaceChildren(
      card('Estimated active browsers', t.unique_browsers, 'Consenting persistent browser IDs; not people'),
      card('Returning browsers', t.returning_browsers, `${rate(t.returning_browser_rate)} observed on multiple UTC dates`),
      card('Analysis reports', t.completed, `${t.started} started / ${t.failed} failed`),
      card('Terminal success rate', rate(t.terminal_success_rate), 'Completed ÷ (completed + failed); excludes unfinished runs'),
      card('Prepared exports', t.export_preparations, 'Once per format per run; not confirmed downloads'),
      card('File handoffs', t.file_handoffs, 'Uploads accepted by ChatGPT; not proof of a rendered reply'),
    );
    const serviceCount = (metric) => data.service.filter((row) => row.metric === metric).reduce((sum, row) => sum + row.count, 0);
    serviceCards.replaceChildren(
      card('HTTP requests', serviceCount('http_request'), 'Includes connection checks, tests, and retries'),
      card('Analysis requests', serviceCount('analyze_requested'), 'Validated tool calls; no user identifier'),
      card('Completed result reports', serviceCount('analysis_report_received'), 'Validated reports, including duplicates and tests'),
      card('Reported failures', serviceCount('analysis_error_received'), `${serviceCount('rpc_error')} separate protocol errors`),
    );
    const rows = dailyRows(data);
    const max = Math.max(1, ...rows.map((row) => row.completed));
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${rows.length * 18} 120`); svg.setAttribute('role', 'img'); svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('aria-label', 'Daily completed analysis reports for the selected population');
    rows.forEach((row, i) => {
      const rect = document.createElementNS(svg.namespaceURI, 'rect');
      const height = row.completed / max * 105;
      for (const [key, value] of Object.entries({ x: i * 18 + 2, y: 110 - height, width: 14, height, fill: 'currentColor' })) rect.setAttribute(key, value);
      const title = document.createElementNS(svg.namespaceURI, 'title'); title.textContent = `${row.day}: ${row.completed} completed reports`; rect.append(title); svg.append(rect);
    });
    chart.replaceChildren(node('h3', 'Completed reports by day'), svg, node('p', `${rows[0].day} — ${rows.at(-1).day} · chart maximum ${max}`));
    const formats = ['onnx', 'tflite', 'gguf', 'safetensors', 'coreml', 'executorch'];
    const exports = ['svg', 'png', 'word', 'cyclonedx', 'spdx'];
    tables.replaceChildren(
      table(['Format', 'Completed reports', 'Share'], formats.map((format) => { const n = data.events.filter((row) => row.event === 'analysis_completed' && row.format === format).reduce((sum, row) => sum + row.count, 0); return [format, n, rate(t.completed ? n / t.completed : null)]; }), 'Artifact format distribution'),
      table(['Export', 'Prepared', 'ChatGPT handoffs'], exports.map((type) => [type, count(data.events, 'export_prepared', type), count(data.events, 'file_shared', type)]), 'Export use'),
      table(['View', 'Runs rendered'], ['identity-boundary', 'architecture-overview', 'block-detail', 'exhaustive', 'static-runtime', 'observed-runtime'].map((view) => [view, count(data.events, 'visualization_rendered', view)]), 'Model IR views rendered'),
      table(['UTC date', 'Started', 'Completed', 'Failed', 'Browsers', 'Prepared exports', 'File handoffs'], rows.map((row) => Object.values(row)), 'Daily usage; browser counts must not be summed to infer unique people'),
      table(['UTC date', 'HTTP requests', 'Protocol errors', 'Analysis requests', 'Result reports', 'Failure reports'], rows.map(({ day }) => [day, ...['http_request', 'rpc_error', 'analyze_requested', 'analysis_report_received', 'analysis_error_received'].map((metric) => data.service.filter((row) => row.day === day && row.metric === metric).reduce((sum, row) => sum + row.count, 0))]), 'Daily MCP traffic; all callers, including tests and retries'),
    );
    limits.textContent = Object.values(data.measurement).filter((value) => typeof value === 'string').join(' ') + ' Records expire after 90 days. Export snapshots with the measurement notes when presenting adoption evidence.';
  }
  async function load() {
    const id = ++requestId; refresh.disabled = true; csv.disabled = json.disabled = true;
    status.textContent = 'Loading usage evidence…';
    try {
      const data = await apiFetch(`/api/admin/usage?days=${period.value}&cohort=${cohort.value}`);
      if (id !== requestId) return;
      latest = data; render(data); csv.disabled = json.disabled = false;
    } catch (error) {
      if (id !== requestId) return;
      latest = null; cards.replaceChildren(); serviceCards.replaceChildren(); chart.replaceChildren(); tables.replaceChildren(); limits.textContent = '';
      status.textContent = error.message || 'Usage unavailable. Check migration 0012 and the collection setting.';
    } finally { if (id === requestId) refresh.disabled = false; }
  }
  refresh.addEventListener('click', load); period.addEventListener('change', load); cohort.addEventListener('change', load);
  json.addEventListener('click', () => { if (latest) download(`deepbom-usage-${latest.cohort}-${latest.generated_at.slice(0, 10)}.json`, JSON.stringify(latest, null, 2) + '\n', 'application/json'); });
  csv.addEventListener('click', () => {
    if (!latest) return;
    const header = 'utc_date,cohort,started,completed,failed,estimated_active_browsers,export_preparations,file_handoffs';
    const rows = dailyRows(latest).map(({ day, ...values }) => [day, latest.cohort, ...Object.values(values)].join(','));
    download(`deepbom-usage-${latest.cohort}-${latest.generated_at.slice(0, 10)}.csv`, '\uFEFF' + [header, ...rows].join('\r\n'), 'text/csv;charset=utf-8');
  });
  return { load, clear(message) { requestId += 1; latest = null; csv.disabled = json.disabled = true; status.textContent = message; cards.replaceChildren(); serviceCards.replaceChildren(); chart.replaceChildren(); tables.replaceChildren(); limits.textContent = ''; } };
}
