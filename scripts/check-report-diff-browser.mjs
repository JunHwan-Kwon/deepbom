import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {chromium} from 'playwright';
import {launchChromium} from './browser-launch.mjs';

const root=path.resolve(process.argv[2]||'.local-validation/report-diff-tests');
const browser=await launchChromium(chromium),results=[];
try {
  for(const name of ['selected','keras']) {
    const dir=path.join(root,name),data=JSON.parse(await readFile(path.join(dir,'report.json'),'utf8'));
    const page=await browser.newPage({viewport:{width:1400,height:1000},acceptDownloads:true}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(pathToFileURL(path.join(dir,'report.html')).href);
    assert.equal(await page.locator('#diff-filter').inputValue(),'changes');
    assert.equal(await page.locator('#diff-mode').inputValue(),'unified');
    assert((await page.locator('#diff-detail h3').textContent()).startsWith('~'));
    assert(await page.locator('#diff-detail .add').count()>0);
    assert(await page.locator('#diff-detail .remove').count()>0);
    assert.equal(await page.locator('.complete-evidence').evaluate(e=>e.open),false);
    await page.locator('#diff-detail .diff-reason summary').first().click();
    assert((await page.locator('#diff-detail').textContent()).includes('training'));
    await page.locator('#diff-mode').selectOption('split');
    assert.equal(await page.locator('.diff-split section').count(),2);
    const before=JSON.parse(await page.locator('.diff-split pre').first().textContent());
    const after=JSON.parse(await page.locator('.diff-split pre').last().textContent());
    assert.notDeepEqual(before.module,after.module);
    await page.locator('#diff-mode').selectOption('unified');
    await page.locator('#diff-filter').selectOption('added');
    assert.equal(await page.locator('#diff-list button').count(),data.comparison.nodes.filter(n=>n.status==='added').length);
    await page.locator('#diff-search').fill('absent-query');
    assert.equal(await page.locator('#diff-list button').count(),0);
    assert(await page.locator('#diff-next').isDisabled());
    // Export scope remains all changed evidence even with no matching UI rows.
    const pending=page.waitForEvent('download');await page.locator('#diff-download').click();
    const download=await pending,patch=await readFile(await download.path(),'utf8');
    assert.equal(download.suggestedFilename(),'model-evidence.diff');
    assert(patch.includes(data.report_sha256));
    for(const n of data.comparison.nodes.filter(n=>n.status!=='unchanged'))assert(patch.includes(`After/${JSON.stringify(n.id)}`));
    await page.locator('#diff-search').fill('');await page.locator('#diff-filter').selectOption('changes');
    await page.locator('#model-diff').screenshot({path:path.join(root,`${name}-model-diff.png`)});
    await page.locator('.complete-evidence>summary').click();
    for(const side of ['baseline','candidate'])assert.deepEqual(await page.locator(`rect[data-model-side="${side}"]`).evaluateAll(ns=>ns.map(n=>n.dataset.subjectRef)),data[side].nodes.map(n=>n.id));
    await page.locator('.complete-evidence>summary').click();
    await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
    assert.equal(await page.locator('.complete-evidence').evaluate(e=>e.open),true);
    await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
    assert.equal(await page.locator('.complete-evidence').evaluate(e=>e.open),false);
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);results.push({name,status:'passed'});await page.close();
  }
  const page=await browser.newPage({viewport:{width:1400,height:1000},acceptDownloads:true}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(root,'large-diff.html')).href);
  assert.equal(await page.locator('#diff-list button').count(),3);
  assert((await page.locator('#diff-detail h3').textContent()).includes('block_0005'));
  await page.locator('#diff-next').click();assert((await page.locator('#diff-detail h3').textContent()).includes('block_0700'));
  await page.locator('#diff-next').click();assert((await page.locator('#diff-detail h3').textContent()).includes('block_1499'));
  await page.locator('#diff-next').click();assert((await page.locator('#diff-detail h3').textContent()).includes('block_0005'));
  await page.locator('#diff-prev').click();assert((await page.locator('#diff-detail h3').textContent()).includes('block_1499'));
  await page.locator('#diff-filter').selectOption('all');
  assert.equal(await page.locator('#diff-list button').count(),50);
  assert.equal(await page.locator('#diff-page-label').textContent(),'30 / 30');
  await page.locator('#diff-page-prev').click();assert.equal(await page.locator('#diff-page-label').textContent(),'29 / 30');
  await page.locator('#diff-page-next').click();assert.equal(await page.locator('#diff-list button').count(),50);
  await page.locator('#diff-search').fill('block_0700');assert.equal(await page.locator('#diff-list button').count(),1);
  const pending=page.waitForEvent('download');await page.locator('#diff-download').click();
  const patch=await readFile(await (await pending).path(),'utf8');
  for(const id of ['0005','0700','1499'])assert(patch.includes(`block_${id}`));
  for(const [file,filter] of [['state-only-diff.html','state'],['edge-only-diff.html','connections']]){
    await page.goto(pathToFileURL(path.join(root,file)).href);await page.locator('#diff-filter').selectOption(filter);
    assert(await page.locator('#diff-list button').count()>0);assert((await page.locator('#diff-list button').first().textContent()).includes('unchanged'));
  }
  await page.goto(pathToFileURL(path.join(root,'rename-diff.html')).href);
  await page.locator('#diff-filter').selectOption('removed');assert.equal(await page.locator('#diff-list button').count(),1);
  assert(await page.locator('#diff-detail .remove').count()>0);assert.equal(await page.locator('#diff-detail .add').count(),0);
  await page.locator('#diff-mode').selectOption('split');assert.equal(await page.locator('.diff-split pre').last().textContent(),'Absent');
  await page.locator('#diff-filter').selectOption('added');assert.equal(await page.locator('.diff-split pre').first().textContent(),'Absent');
  await page.goto(pathToFileURL(path.join(root,'long/report.html')).href);
  await page.locator('#diff-filter').selectOption('all');await page.locator('#diff-search').fill('<script>');
  assert.equal(await page.locator('#diff-list button').count(),1);assert((await page.locator('#diff-detail h3').textContent()).includes('한글<script>'));
  assert.deepEqual(errors,[]);results.push({name:'1500-module-navigation-and-evidence-boundaries',status:'passed'});
  await writeFile(path.join(root,'diff-browser-validation.json'),JSON.stringify({status:'passed',results},null,2)+'\n');
  console.log('Model diff browser: filters, unified/split records, 1,500-module pagination, change navigation, complete exports, state/edge-only changes, escaping, printing and mobile layout passed.');
} finally {await browser.close();}
