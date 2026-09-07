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
    // Elements that transition a colour resolved from a custom property keep
    // the previous theme's value when only the token changes: the transition
    // never starts, so nothing repaints them until an unrelated reflow. That
    // left the evidence-state badges in light colours on the dark ground.
    // Suppressing transitions across the swap forces the new value to settle.
    root.dataset.themeSwapping = "true";
    root.dataset.theme = theme;
    root.dataset.themeSource = source;
    root.style.colorScheme = theme;
    // Reading a computed style on the root is not enough: descendants keep
    // their cached colour. Detaching the root from layout for one tick forces
    // the whole tree to restyle while transitions are suppressed, so the new
    // token values settle instead of waiting for an unrelated invalidation.
    const previousDisplay = root.style.display;
    root.style.display = "none";
    void root.offsetHeight;
    root.style.display = previousDisplay;
    const release = () => delete root.dataset.themeSwapping;
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => requestAnimationFrame(release));
    else release();
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
