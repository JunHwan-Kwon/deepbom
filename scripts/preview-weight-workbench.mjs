// Local, read-only preview of the production component using the bundled ONNX
// sample. Bind only to loopback; never expose the repository or credentials.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyzeOnnxModel } from '../web/onnx.js';
import { getArtifactIrContext } from '../web/lib/artifact-ir-context.js';
import { sha256BytesHex } from '../web/lib/sha256-sync.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.DEEPBOM_PREVIEW_PORT || 4179);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('DEEPBOM_PREVIEW_PORT must be between 1024 and 65535');
const bytes = new Uint8Array(await readFile(path.join(root, 'web/samples/sample_cnn_float.onnx')));
const filename = 'sample_cnn_float.onnx', analysis = analyzeOnnxModel(bytes, filename), sha256 = sha256BytesHex(bytes);
analysis.model_sha256 = sha256;
const model = getArtifactIrContext(analysis, { filename, format: 'onnx', sha256, size: bytes.length }).model_ir;
const page = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DEEPBOM · Interactive weight workbench</title><link rel="stylesheet" href="/web/research-theme.css"><link rel="stylesheet" href="/web/weight-workspace.css"><style>body{max-width:1440px;margin:24px auto;padding:0 16px}body>header{display:flex;gap:16px;align-items:center;justify-content:space-between;font:13px system-ui;color:var(--muted)}button{cursor:pointer}</style></head><body><header><span>DEEPBOM · Local interaction preview · bundled ONNX sample</span><button id="theme">Change theme</button></header><main></main><script type="module">
import {installNumericalPanel} from '/web/lib/numerical-ir-panel.js';
const context=await(await fetch('/context.json')).json();context.source=await(await fetch('/sample.onnx')).blob();
installNumericalPanel(document.querySelector('main'),()=>context);
document.querySelector('#theme').onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';};
</script></body></html>`;
const server = createServer(async (req, res) => {
  try {
    if (req.method !== 'GET') { res.writeHead(405).end(); return; }
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/') { res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }).end(page); return; }
    if (url.pathname === '/context.json') { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ model, analysis })); return; }
    if (url.pathname === '/sample.onnx') { res.writeHead(200, { 'content-type': 'application/octet-stream' }).end(bytes); return; }
    const pathname = decodeURIComponent(url.pathname);
    if (!['/web/', '/pkg/'].some(prefix => pathname.startsWith(prefix))) throw new Error('not a preview asset');
    const file = path.resolve(root, '.' + pathname);
    if (![path.join(root, 'web') + path.sep, path.join(root, 'pkg') + path.sep].some(prefix => file.startsWith(prefix))) throw new Error('outside preview assets');
    const content = await readFile(file), type = { '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm' }[path.extname(file)] || 'application/octet-stream';
    res.writeHead(200, { 'content-type': type, 'cache-control': 'no-store' }).end(content);
  } catch { res.writeHead(404).end('Not found'); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
console.log(`DEEPBOM weight preview: http://127.0.0.1:${port}\nChoose Analyze weights → Interactive pruning. Ctrl+C stops the preview.`);
process.on('SIGINT', () => server.close());
process.on('SIGTERM', () => server.close());
