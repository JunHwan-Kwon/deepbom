import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { launchChromium } from './browser-launch.mjs';

const sample = {
  schema: 'deepbom.usage_summary.v2', channel: 'all', enabled: true, generated_at: '2026-09-17T12:00:00Z', cohort: 'public',
  period: { days: 7, since: '2026-09-11T00:00:00Z', through: '2026-09-17T12:00:00Z', timezone: 'UTC' },
  totals: { unique_browsers: 3, returning_browsers: 1, returning_browser_rate: 1 / 3, started: 5, completed: 4, failed: 1, unfinished: 0, terminal_success_rate: 0.8, export_preparations: 2, file_handoffs: 1 },
  events: [
    { day: '2026-09-17', event: 'analysis_started', detail: '', format: 'unknown', count: 5 },
    { day: '2026-09-17', event: 'analysis_completed', detail: '', format: 'onnx', count: 3 },
    { day: '2026-09-17', event: 'analysis_completed', detail: '', format: 'tflite', count: 1 },
    { day: '2026-09-17', event: 'analysis_failed', detail: '', format: 'unknown', count: 1 },
    { day: '2026-09-17', event: 'export_prepared', detail: 'cyclonedx', format: 'onnx', count: 2 },
    { day: '2026-09-17', event: 'file_shared', detail: 'cyclonedx', format: 'onnx', count: 1 },
  ],
  daily_browsers: [{ day: '2026-09-17', browsers: 3 }],
  service: [{ day: '2026-09-17', metric: 'http_request', format: 'unknown', count: 20 }],
  measurement: { population: 'Consenting browser estimates, not verified people.', limits: 'Synthetic test data only. Production has no backfilled usage.', retention_days: 90 },
};
sample.events = sample.events.map((r) => ({ ...r, channel: 'chatgpt' }));
sample.daily_browsers = sample.daily_browsers.map((r) => ({ ...r, channel: 'chatgpt' }));
sample.weekly_browsers = [{ channel: 'chatgpt', week: '2026-09-14', browsers: 3, returning_browsers: 1 }];
sample.channels = [
  { id: 'chatgpt', label: 'ChatGPT', status: 'measured_opt_in', measured_from: '2026-09-11T00:00:00Z', totals: { ...sample.totals }, export_types: ['svg','png','word','cyclonedx','spdx'] },
  { id: 'web', label: 'Web', status: 'measured_opt_in', measured_from: '2026-09-17T09:00:00Z', totals: Object.fromEntries(Object.keys(sample.totals).map((key) => [key, key.includes('rate') ? null : 0])), export_types: ['svg','png','cyclonedx','model_views','evidence_package'] },
  ...[['cli','CLI'],['claude','Claude'],['local_mcp','Local MCP']].map(([id,label]) => ({ id,label,status: 'not_measured',measured_from:null,totals:null,export_types:[] })),
];
sample.totals.unique_browsers = sample.totals.returning_browsers = sample.totals.returning_browser_rate = null;
const npm = { schema: 'deepbom.package_downloads.v1', provider: 'npm', package: 'deepbom', source:'https://api.npmjs.org/downloads/range/2026-09-10:2026-09-16/deepbom', fetched_at:sample.generated_at, reported_total:42, missing_days:1, measurement:'Registry downloads, not executions or people; synthetic fixture.', daily:[{day:'2026-09-15',downloads:null},{day:'2026-09-16',downloads:42}] };
const browser = await launchChromium(chromium);
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = []; page.on('pageerror', (e) => errors.push(e.message));
  await page.route('https://deepbom.test/**', async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/') {
      await route.fulfill({ contentType: 'text/html; charset=utf-8', body: `<meta charset="utf-8"><link rel="stylesheet" href="/web/styles.css"><body class="admin-page"><main class="admin-shell"><p>LOCAL TEST FIXTURE — synthetic counts</p><section class="admin-bench-section" id="adminUsage"></section></main><script type="module">import {createAdminUsage} from '/web/lib/admin-usage.js'; window.calls=[]; window.fixture=${JSON.stringify(sample)}; window.npm=${JSON.stringify(npm)}; window.view=createAdminUsage(document.getElementById('adminUsage'), async (url)=>{window.calls.push(url); if(window.fail)throw new Error('Access revoked');if(url.includes('/downloads?')){if(window.failDownloads)throw new Error('Provider offline');return structuredClone(window.npm)}const data=structuredClone(window.fixture);data.channel=new URL(url,location.origin).searchParams.get('channel');if(data.channel!=='all'){data.totals=data.channels.find(c=>c.id===data.channel).totals;for(const key of ['events','daily_browsers','weekly_browsers'])data[key]=data[key].filter(r=>r.channel===data.channel)}return data}); window.view.load();</script>` });
    } else if (/^\/web\/[a-z-]+\.css$/.test(pathname) || pathname === '/web/lib/admin-usage.js') {
      await route.fulfill({ contentType: pathname.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(`.${pathname}`) });
    } else await route.fulfill({ status: 404, body: '' });
  });
  await page.goto('https://deepbom.test/');
  await page.getByRole('button', { name: 'Download evidence JSON' }).waitFor();
  await page.waitForFunction(() => !document.querySelector('button:last-child')?.disabled && document.querySelectorAll('.admin-metric-card').length === 4);
  assert.equal(await page.getByText('Estimated active browsers', {exact:true}).count(),0,'No combined unique browser estimate');
  assert.match(await page.getByRole('table', {name:'Artifact format distribution'}).innerText(), /75.0%/);
  assert.equal(await page.getByRole('img',{name:'Daily completed analysis reports',exact:true}).locator('rect').count(),8,'Only covered dates have bars; pre-coverage Web dates are absent');
  assert.match(await page.getByRole('table',{name:'Operational counts; not ChatGPT-only usage'}).innerText(),/20/);
  await page.waitForFunction(()=>document.body.textContent.includes('42 reported downloads'));
  const csvPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download daily CSV' }).click();
  const csv = await readFile(await (await csvPromise).path(), 'utf8');
  assert.match(csv, /estimated_active_browsers/);
  assert.match(csv, /2026-09-17,chatgpt,public,partial_day,5,4,1,3,2,1/);
  const jsonPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download evidence JSON' }).click();
  const document = JSON.parse(await readFile(await (await jsonPromise).path(), 'utf8'));
  assert.deepEqual(document, {...sample, package_downloads:npm}, 'Evidence export retains definitions, channels and separate registry source');
  assert.match(csv,/2026-09-11,web,public,not_measured,,,,,,/);
  await page.getByLabel('Usage channel').selectOption('cli');
  await page.getByText('Execution is not measured for this channel.',{exact:false}).waitFor();
  assert.equal(await page.locator('.admin-metric-card').count(),0);
  await page.getByLabel('Usage channel').selectOption('chatgpt');
  await page.getByText('Estimated active browsers',{exact:true}).waitFor();
  await page.getByLabel('Usage channel').selectOption('all');
  await page.getByLabel('Usage population').selectOption('test');
  await page.waitForFunction(() => window.calls.some(url=>url.includes('cohort=test&channel=all')));
  await page.getByLabel('Usage population').selectOption('public');
  await page.getByLabel('Usage period').selectOption('7');
  await mkdir('.local-validation/usage', { recursive: true });
  await page.screenshot({ path: '.local-validation/usage/dashboard-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Dashboard must fit a narrow viewport');
  await page.screenshot({ path: '.local-validation/usage/dashboard-mobile.png', fullPage: true });
  await page.evaluate(async()=>{window.failDownloads=true;await window.view.load()});
  assert.match(await page.locator('#adminUsage').innerText(),/Download statistics are unavailable/);
  assert.equal(await page.locator('.admin-metric-card').count(),4,'Provider outage preserves first-party metrics');
  await page.evaluate(async () => { window.fail = true; await window.view.load(); });
  assert.equal(await page.locator('.admin-metric-card').count(), 0, 'Failed authorization must clear previous data');
  assert(await page.getByRole('button', { name: 'Download evidence JSON' }).isDisabled());
  assert.deepEqual(errors, []);
  console.log('Usage dashboard passed: counts, share/rate labels, zero bars, cohort filter, CSV/JSON exports, mobile layout, and stale-data clearing. Screenshots contain synthetic fixture data only.');
} finally { await browser.close(); }
