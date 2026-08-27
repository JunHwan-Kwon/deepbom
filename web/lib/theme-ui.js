(function initializeTheme() {
  if (typeof window === "undefined" || typeof document === "undefined") return;

  const STORAGE_KEY = "deepbom-color-theme-v1";
  const root = document.documentElement;
  const systemPreference = window.matchMedia("(prefers-color-scheme: dark)");

  function savedTheme() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch {
      return null;
    }
  }

  function systemTheme() {
    return systemPreference.matches ? "dark" : "light";
  }

  function updateControl(theme) {
    const button = document.getElementById("themeToggle");
    const label = document.getElementById("themeToggleLabel");
    if (!button || !label) return;
    const next = theme === "dark" ? "light" : "dark";
    label.textContent = next === "dark" ? "Dark" : "Light";
    button.dataset.nextTheme = next;
    button.setAttribute("aria-label", `Use ${next} theme`);
    button.title = `Use ${next} theme`;
    button.setAttribute("aria-pressed", String(theme === "dark"));
  }

  function updateBrowserChrome(theme) {
    const themeColor = document.getElementById("themeColor");
    if (themeColor) themeColor.content = theme === "dark" ? "#111614" : "#fffefa";
  }

  function applyTheme(theme, source) {
    root.dataset.theme = theme;
    root.dataset.themeSource = source;
    root.style.colorScheme = theme;
    updateBrowserChrome(theme);
    updateControl(theme);
    window.dispatchEvent(new CustomEvent("deepbom:themechange", { detail: { theme, source } }));
  }

  const initialSavedTheme = savedTheme();
  applyTheme(initialSavedTheme || systemTheme(), initialSavedTheme ? "user" : "system");

  document.addEventListener("DOMContentLoaded", () => {
    updateControl(root.dataset.theme || systemTheme());
    document.getElementById("themeToggle")?.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch {
        // The selected theme still applies for this document when storage is unavailable.
      }
      applyTheme(next, "user");
    });
  }, { once: true });

  const followSystemTheme = () => {
    if (!savedTheme()) applyTheme(systemTheme(), "system");
  };
  if (typeof systemPreference.addEventListener === "function") {
    systemPreference.addEventListener("change", followSystemTheme);
  } else if (typeof systemPreference.addListener === "function") {
    systemPreference.addListener(followSystemTheme);
  }
}());
