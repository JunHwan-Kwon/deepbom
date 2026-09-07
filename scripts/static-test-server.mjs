import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";

// A read that fails for any reason other than a missing path must not be
// reported as 404. Under load these servers hit transient descriptor and
// sharing errors, and answering "not found" turns them into module-fetch
// failures that look like product defects. Retry those once, then fail loudly.
const TRANSIENT_READ_CODES = new Set(["EMFILE", "ENFILE", "EBUSY", "EAGAIN", "EPERM", "EACCES"]);

export function createStaticTestServer(root) {
  return createServer(async (request, response) => {
    let file;
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
      file = path.resolve(root, relative);
      if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
    } catch {
      return send(response, 400, "text/plain", "bad request");
    }
    try {
      send(response, 200, mimeType(file), await readContents(file));
    } catch (error) {
      if (error?.code === "ENOENT" || error?.code === "EISDIR") {
        return send(response, 404, "text/plain", "not found");
      }
      console.error(`static-test-server: ${error?.code || "read failure"} for ${file}`);
      send(response, 500, "text/plain", "read failure");
    }
  });
}

async function readContents(file) {
  try {
    return await readFile(file);
  } catch (error) {
    if (!TRANSIENT_READ_CODES.has(error?.code)) throw error;
    await new Promise((resolve) => setTimeout(resolve, 25));
    return readFile(file);
  }
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css",
    ".json": "application/json",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
