export function createAuditProgressController({ root, bar, label }) {
  let percent = 0;
  let phase = "Not started";
  let phaseStartedAt = 0;
  let auditStartedAt = 0;
  let phaseStartPercent = 0;
  let ceiling = 0;
  let step = 0;
  let totalSteps = 12;
  let timer = null;

  function stopTimer() {
    if (timer != null) clearInterval(timer);
    timer = null;
  }

  function elapsedSeconds(startedAt) {
    return Math.max(0, (performance.now() - startedAt) / 1000);
  }

  function render(state = "running") {
    if (!root || !bar || !label) return;
    const phaseElapsed = phaseStartedAt ? elapsedSeconds(phaseStartedAt) : 0;
    const prefix = step ? `Step ${String(step).padStart(2, "0")}/${totalSteps} · ` : "";
    const suffix = state === "complete" && auditStartedAt
      ? ` · ${elapsedSeconds(auditStartedAt).toFixed(1)}s total`
      : state === "running" ? ` · ${phaseElapsed.toFixed(1)}s` : "";
    const text = `${prefix}${percent}% · ${phase}${suffix}`;
    root.hidden = false;
    root.dataset.state = state;
    root.setAttribute("aria-valuenow", String(percent));
    root.setAttribute("aria-valuetext", text);
    bar.style.width = `${percent}%`;
    label.textContent = text;
  }

  function set(value, nextPhase, state = "running", options = {}) {
    if (!root || !bar || !label) return;
    stopTimer();
    if (value != null) percent = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
    phase = nextPhase || phase;
    step = Number(options.step || step || 0);
    totalSteps = Number(options.totalSteps || totalSteps || 12);
    phaseStartedAt = performance.now();
    phaseStartPercent = percent;
    if (!auditStartedAt || percent <= 2 || root.dataset.state === "complete") auditStartedAt = phaseStartedAt;
    render(state);
  }

  function begin(value, nextPhase, options = {}) {
    set(value, nextPhase, "running", options);
    ceiling = Math.max(percent, Math.min(99, Number(options.ceiling ?? percent)));
    timer = setInterval(() => {
      const elapsed = elapsedSeconds(phaseStartedAt);
      const projected = phaseStartPercent
        + Math.floor((ceiling - phaseStartPercent) * (1 - Math.exp(-elapsed / 4)));
      if (projected > percent) percent = Math.min(ceiling - 1, projected);
      render("running");
    }, 250);
  }

  function describe(nextPhase) {
    if (nextPhase && nextPhase !== phase) {
      phase = nextPhase;
      phaseStartedAt = performance.now();
      phaseStartPercent = percent;
    }
    render(root?.dataset.state || "running");
  }

  function reset() {
    stopTimer();
    percent = 0;
    phase = "Not started";
    phaseStartedAt = 0;
    auditStartedAt = 0;
    phaseStartPercent = 0;
    ceiling = 0;
    step = 0;
    if (!root || !bar || !label) return;
    root.hidden = true;
    root.dataset.state = "idle";
    root.setAttribute("aria-valuenow", "0");
    root.setAttribute("aria-valuetext", "Not started");
    bar.style.width = "0%";
    label.textContent = "0%";
  }

  return { begin, describe, reset, set };
}
