import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { launchChromium } from './browser-launch.mjs';

const sample = {
  schema: 'deepbom.usage_summary.v1', enabled: true, generated_at: '2026-09-17T12:00:00Z', cohort: 'public',
  period: { days: 7, since: '2026-09-11T00:00:00Z', through: '2026-09-17T12:00:00Z', timezone: 'UTC' },
  totals: { unique_browsers: 3, returning_browsers: 1, returning_browser_rate: 1 / 3, started: 5, completed: 4, failed: 1, terminal_success_rate: 0.8, export_preparations: 2, file_handoffs: 1 },
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
const browser = await launchChromium(chromium);
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = []; page.on('pageerror', (e) => errors.push(e.message));
  await page.route('https://deepbom.test/**', async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/') {
      await route.fulfill({ contentType: 'text/html; charset=utf-8', body: `<meta charset="utf-8"><link rel="stylesheet" href="/web/styles.css"><body class="admin-page"><main class="admin-shell"><p>LOCAL TEST FIXTURE — synthetic counts</p><section class="admin-bench-section" id="adminUsage"></section></main><script type="module">import {createAdminUsage} from '/web/lib/admin-usage.js'; window.calls=[]; window.fixture=${JSON.stringify(sample)}; window.view=createAdminUsage(document.getElementById('adminUsage'), async (url)=>{window.calls.push(url); if(window.fail)throw new Error('Access revoked');return structuredClone(window.fixture)}); window.view.load();</script>` });
    } else if (/^\/web\/[a-z-]+\.css$/.test(pathname) || pathname === '/web/lib/admin-usage.js') {
      await route.fulfill({ contentType: pathname.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(`.${pathname}`) });
    } else await route.fulfill({ status: 404, body: '' });
  });
  await page.goto('https://deepbom.test/');
  await page.getByRole('button', { name: 'Download evidence JSON' }).waitFor();
  await page.waitForFunction(() => document.querySelectorAll('.admin-metric-card').length === 10);
  assert.match(await page.locator('#adminUsage').innerText(), /Estimated active browsers[\s\S]*3/i);
  assert.match(await page.locator('table').first().innerText(), /75.0%/);
  assert.equal(await page.locator('svg rect[height="0"]').count(), 6, 'Zero days must not render a nonzero bar');
  assert.match(await page.locator('table').last().innerText(), /20/);
  const csvPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download daily CSV' }).click();
  const csv = await readFile(await (await csvPromise).path(), 'utf8');
  assert.match(csv, /estimated_active_browsers/);
  assert.match(csv, /2026-09-17,public,5,4,1,3,2,1/);
  const jsonPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download evidence JSON' }).click();
  const document = JSON.parse(await readFile(await (await jsonPromise).path(), 'utf8'));
  assert.deepEqual(document, sample, 'Evidence export must retain definitions and limitations');
  await page.getByLabel('Usage population').selectOption('test');
  await page.waitForFunction(() => window.calls.at(-1).endsWith('cohort=test'));
  await page.getByLabel('Usage population').selectOption('public');
  await page.getByLabel('Usage period').selectOption('7');
  await mkdir('.local-validation/usage', { recursive: true });
  await page.screenshot({ path: '.local-validation/usage/dashboard-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Dashboard must fit a narrow viewport');
  await page.screenshot({ path: '.local-validation/usage/dashboard-mobile.png', fullPage: true });
  await page.evaluate(async () => { window.fail = true; await window.view.load(); });
  assert.equal(await page.locator('.admin-metric-card').count(), 0, 'Failed authorization must clear previous data');
  assert(await page.getByRole('button', { name: 'Download evidence JSON' }).isDisabled());
  assert.deepEqual(errors, []);
  console.log('Usage dashboard passed: counts, share/rate labels, zero bars, cohort filter, CSV/JSON exports, mobile layout, and stale-data clearing. Screenshots contain synthetic fixture data only.');
} finally { await browser.close(); }
