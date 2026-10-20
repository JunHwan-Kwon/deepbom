"""Produce a real 410-tensor capture and inspect bounded DeepBoard navigation."""
import subprocess,tempfile
from pathlib import Path
import torch
import deepbom
torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='deepbom-board-scale-') as tmp:
    model=torch.nn.Sequential(*[torch.nn.Linear(2,2) for _ in range(205)])
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    with deepbom.monitor(model,optimizer=optimizer,logdir=Path(tmp)/'run',preset='weights',every=1,max_values=2,report=True) as session:
        optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step()
        for phase in ("train","validation"):
            for step in range(2):session.record_metrics({"loss":float(step)},at={"step":step,"phase":phase})
        for loader in (0,1):
            for step in range(2):session.record_metrics({"loss":float(step)},at={"step":step,"phase":"validation","dataloader_idx":loader})
    script='''
    import fs from 'node:fs/promises';import assert from 'node:assert/strict';
    import {chromium} from 'playwright';import {launchChromium} from './scripts/browser-launch.mjs';
    const browser=await launchChromium(chromium);try{const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.setContent(await fs.readFile(process.argv[1],'utf8'));
    assert((await page.locator('#weights').textContent()).includes('of 410 matched / 410 recorded'));
    assert.equal(await page.locator('#weights button').count(),52);assert.equal(await page.locator('#curve polyline').count(),4);
    await page.locator('#phase').selectOption('train');assert.equal(await page.locator('#curve polyline').count(),1);await page.locator('#phase').selectOption('');
    await page.getByRole('button',{name:'Next weights page',exact:true}).click();
    assert((await page.locator('#weights').textContent()).includes('51–100'));
    await page.locator('#module').selectOption('204');assert.equal(await page.locator('#weights button').count(),2);await page.locator('#module').selectOption('');
    await page.locator('#tensor-search').fill('204.');assert.equal(await page.locator('#weights button').count(),2);
    await page.locator('#tensor-search').fill('no-such-tensor');assert.equal(await page.locator('#weights button').count(),0);
    await page.setViewportSize({width:320,height:700});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);console.log('DeepBoard 410 real tensor capture: bounded page DOM, search, empty match and 320px reflow passed');
    }finally{await browser.close();}
    '''
    subprocess.run(['node','--input-type=module','-e',script,session.receipt['report']],check=True)
