import { existsSync } from "node:fs";

const DEFAULT_WINDOWS_CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";

export function chromiumLaunchOptions(options = {}, playwrightExecutablePath = "") {
  const configuredPath = String(
    process.env.DEEPBOM_CHROME_EXECUTABLE
      || process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      || "",
  ).trim();
  if (configuredPath && !existsSync(configuredPath)) {
    throw new Error(`Configured Chromium executable does not exist: ${configuredPath}`);
  }
  const pinnedPath = String(playwrightExecutablePath || "").trim();
  const executablePath = configuredPath
    || (pinnedPath && existsSync(pinnedPath) ? pinnedPath : "")
    || (process.platform === "win32" && existsSync(DEFAULT_WINDOWS_CHROME) ? DEFAULT_WINDOWS_CHROME : "");
  return executablePath ? { ...options, executablePath } : { ...options };
}

export function launchChromium(chromium, options = {}) {
  return chromium.launch(chromiumLaunchOptions(
    { headless: true, ...options },
    chromium.executablePath(),
  ));
}

export function waitForAnimationFrames(page, frameCount = 2) {
  return page.evaluate((requestedFrames) => new Promise((resolve) => {
    let remaining = Math.max(1, Number(requestedFrames) || 1);
    const advance = () => {
      remaining -= 1;
      if (remaining === 0) resolve();
      else requestAnimationFrame(advance);
    };
    requestAnimationFrame(advance);
  }), frameCount);
}
