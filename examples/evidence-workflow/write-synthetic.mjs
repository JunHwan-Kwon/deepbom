// Run from the source checkout. The destination must not exist.
import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {syntheticScenario} from './synthetic.mjs';
const directory=path.resolve(process.argv[2]||'synthetic-evidence');
await mkdir(directory);
const {request,files}=syntheticScenario();
await writeFile(path.join(directory,'request.json'),JSON.stringify(request,null,2)+'\n');
const selections=[];
for(const [i,f] of files.entries()){
  const filename=`source-${i}.bin`;
  await writeFile(path.join(directory,filename),f.bytes);
  selections.push({sha256:f.ref.sha256,path:filename});
}
await writeFile(path.join(directory,'files.json'),JSON.stringify(selections,null,2)+'\n');
console.log(`Synthetic inputs created at ${directory}. These are not trained models or patient measurements.`);
