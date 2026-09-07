import { readFileSync } from "node:fs";

/*
 * The application module is interaction-loaded. Until a pointer, focus, change,
 * click, or drop event imports web/app.js, nothing has run that can set the
 * workflow state, so every rule keyed on that state is inert during the first
 * paint. The shell therefore has to declare its own starting state in markup.
 *
 * Without that declaration the pre-audit panels render on arrival and then
 * disappear at the first interaction, and collapsing whatever the visitor
 * opened does not bring them back, because idle is their correct steady state.
 *
 * This check fails when any leg of that coupling is removed.
 */

const failures = [];
const html = readFileSync("web/index.html", "utf8");
const bootstrap = readFileSync("web/bootstrap.js", "utf8");
const shellCss = readFileSync("web/app-shell.css", "utf8");

// 1. The module really is interaction-loaded. If this stops being true the
//    declaration below is harmless, but the reason for it is no longer honest.
const INTERACTION_EVENTS = ["pointerdown", "focusin", "change", "click", "drop"];
const missingEvents = INTERACTION_EVENTS.filter(
  (event) => !new RegExp(`addEventListener\\(\\s*["']${event}["']`).test(bootstrap),
);
if (missingEvents.length) {
  failures.push(`web/bootstrap.js no longer defers the application on: ${missingEvents.join(", ")}`);
}
if (!/import\(\s*["']\.\/app\.js["']\s*\)/.test(bootstrap)) {
  failures.push("web/bootstrap.js no longer imports ./app.js dynamically; the first-paint coupling changed.");
}

// 2. The first paint declares the state the module will re-affirm.
const bodyTag = html.match(/<body\b[^>]*>/);
if (!bodyTag) {
  failures.push("web/index.html has no body tag.");
} else if (!/data-workflow-state\s*=\s*["']idle["']/.test(bodyTag[0])) {
  failures.push(
    `web/index.html body must declare data-workflow-state="idle" so the idle rules apply before app.js loads. Found: ${bodyTag[0]}`,
  );
}

// 3. The rules that depend on it still exist and still cover every panel that
//    holds a placeholder until an artifact is audited.
const IDLE_HIDDEN_PANELS = ["#preAuditReference", "#formatCapabilityPanel", "#modelPlan"];
const idleRule = shellCss.match(/(?:body\[data-workflow-state="idle"\][^{,]*,?\s*)+\{[^}]*\}/);
if (!idleRule) {
  failures.push('web/app-shell.css no longer hides any panel under body[data-workflow-state="idle"].');
} else {
  for (const panel of IDLE_HIDDEN_PANELS) {
    if (!idleRule[0].includes(panel)) {
      failures.push(`web/app-shell.css idle rule no longer covers ${panel}.`);
    }
  }
  if (!/display:\s*none/.test(idleRule[0])) {
    failures.push("web/app-shell.css idle rule no longer hides its selectors.");
  }
}

// 4. Those panels must not carry a conflicting inline state in markup, or the
//    declared idle state would be overridden before the module runs.
for (const panel of IDLE_HIDDEN_PANELS) {
  const id = panel.slice(1);
  const tag = html.match(new RegExp(`<[a-z]+\\b[^>]*id="${id}"[^>]*>`, "i"));
  if (tag && /\sstyle="[^"]*display\s*:\s*(?!none)/i.test(tag[0])) {
    failures.push(`web/index.html #${id} sets an inline display that overrides the idle rule.`);
  }
}

if (failures.length) {
  for (const failure of failures) console.error(`FAIL ${failure}`);
  process.exit(1);
}

console.log(
  `Pre-interaction shell check passed (interaction-loaded module, declared idle first paint, ${IDLE_HIDDEN_PANELS.length} idle-hidden panels).`,
);
