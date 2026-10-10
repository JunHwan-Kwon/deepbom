# A reproducible model-change evidence review

This example compares **synthetic reported measurements**, not clinical results.
The adaptation population improves from 2 to 1; the original population worsens
from 2 to 3. With a maximum regression of 0.25, one check meets the criterion and
one does not. Omitting the original files makes binding incomplete, rather than
silently approving a change.

From this source checkout, using Node 20 or later:

```sh
node examples/evidence-workflow/write-synthetic.mjs /tmp/deepbom-example
node --input-type=module <<'JS'
import {readFile,writeFile} from 'node:fs/promises';
import {evidenceWorkflow} from './sdk/index.mjs';
const root='/tmp/deepbom-example';
const request=JSON.parse(await readFile(root+'/request.json'));
const files=JSON.parse(await readFile(root+'/files.json'));
const result=await evidenceWorkflow(request,{
  files:files.map(f=>({...f,path:root+'/'+f.path}))
});
await writeFile(root+'/result.json',JSON.stringify(result,null,2)+'\n');
console.log(result.document.status,result.document.counts);
JS
node bin/deepbom.mjs evidence-workflow /tmp/deepbom-example/result.json --format html --output /tmp/deepbom-example/report.html
```

For an installed package, use `import {evidenceWorkflow} from 'deepbom'` and the
`deepbom` command. PDF requires the matching Python `deepbom[report]` package.
Load `request.json` and all ten `source-*.bin` files in
[the browser review](https://deepbom.org/reports/evidence/), or open `result.json`
to inspect previously recorded evidence. The latter does not verify files anew.
Local MCP offers `deepbom_evidence_workflow` with an allowed request path and the
same `{sha256,path}` file selections. Run `scripts/check-evidence-workflow-channels.mjs`
to reproduce the shared-core/CLI/SDK/MCP/browser comparison.

Replace the example records with measurements produced by your evaluator. Bind
them to exact before/after model identities, dataset and cohort identities, code,
environment, configuration and run evidence. Preserve the original JSON beside
HTML/PDF or logger projections. See [the workflow contract](../../docs/evidence-ir/SNAPSHOT_WORKFLOW.md)
for supported operations, identity scopes, limits and migration rules.
