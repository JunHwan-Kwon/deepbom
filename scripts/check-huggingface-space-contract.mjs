import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const read = (relative) => readFile(path.join(root, relative), "utf8");
const dockerfile = await read("channels/huggingface/Dockerfile");
const readme = await read("channels/huggingface/README.md");
const nginx = await read("channels/huggingface/nginx.conf");
const pagesBuilder = await read("scripts/build-pages.mjs");

assert.match(readme, /^---\r?\n[\s\S]*?sdk:\s*docker\r?\napp_port:\s*7860\r?\n[\s\S]*?---/m);
assert.match(readme, /analyzed in the visitor's browser/i);
assert.match(readme, /are not posted to the Space\s+container/i);
assert.match(readme, /hash-verified `dist\/`/i);
assert.match(dockerfile, /^FROM nginx:1\.29\.5-alpine@sha256:[a-f0-9]{64}$/m);
assert.match(dockerfile, /^COPY dist\/ \/usr\/share\/nginx\/html\/$/m);
assert.doesNotMatch(dockerfile, /curl|wget|pip install|npm install|ENTRYPOINT|CMD/i);
assert.match(nginx, /listen\s+7860/);
assert.match(nginx, /Cross-Origin-Opener-Policy\s+same-origin/);
assert.match(nginx, /Cross-Origin-Embedder-Policy\s+require-corp/);
assert.match(pagesBuilder, /skills["'],\s*["']deepbom["']/);
assert.match(pagesBuilder, /llms\.txt/);

try {
  const distSkill = path.join(root, "dist", "skills", "deepbom", "SKILL.md");
  if ((await stat(distSkill)).isFile()) {
    const [sourceSkill, builtSkill] = await Promise.all([read("skills/deepbom/SKILL.md"), readFile(distSkill, "utf8")]);
    assert.equal(builtSkill, sourceSkill, "Built Space/site skill must equal the reviewed source skill.");
  }
} catch (error) {
  if (error?.code !== "ENOENT") throw error;
}

console.log("Hugging Face Space contract passed: digest-pinned static image, browser-only artifact boundary, security headers, and agent discovery assets.");
