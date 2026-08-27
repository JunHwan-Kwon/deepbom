import { spawn } from "node:child_process";
import process from "node:process";

export async function runCommand(command, args, { timeoutMs = 0 } = {}) {
  console.log(`\n> ${[command, ...args].join(" ")}`);
  let timedOut = false;
  const code = await new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: "inherit", shell: false });
    const timer = timeoutMs > 0 ? setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs) : null;
    child.on("error", reject);
    child.on("exit", resolve);
    child.on("close", () => {
      if (timer) clearTimeout(timer);
    });
  });
  if (timedOut) {
    throw new Error(`${command} ${args.join(" ")} exceeded ${timeoutMs} ms`);
  }
  if (code !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with exit code ${code}`);
  }
}

export async function runNode(script, args = [], options = {}) {
  await runCommand(process.execPath, [script, ...args], options);
}

export async function runNpm(args) {
  const invocation = resolveNpmCommand(args);
  await runCommand(invocation.command, invocation.args);
}

export function resolveNpmCommand(args) {
  const npmExecPath = process.env.npm_execpath?.trim();
  if (npmExecPath) return { command: process.execPath, args: [npmExecPath, ...args] };
  if (process.platform === "win32") {
    return { command: process.env.ComSpec || "cmd.exe", args: ["/d", "/s", "/c", "npm.cmd", ...args] };
  }
  return { command: "npm", args };
}
