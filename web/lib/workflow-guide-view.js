import { guideWorkflows } from "./workflow-catalog.js";

export function mountWorkflowGuide(container, { format, openai, inspect }) {
  const details = document.createElement("details"); details.dataset.testid = "workflow-guide";
  const heading = document.createElement("summary"); heading.textContent = "Explore next · Chat, web and CLI";
  const select = document.createElement("select"); select.setAttribute("aria-label", "DEEPBOM workflow"); select.style.cssText = "max-width:100%;font:inherit;padding:8px;margin:12px 0";
  const guide = guideWorkflows({ format });
  for (const row of guide.workflows) { const option = document.createElement("option"); option.value = row.id; option.textContent = row.title; select.append(option); }
  const description = document.createElement("p"), boundary = document.createElement("p"), commands = document.createElement("pre"), status = document.createElement("p");
  commands.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;max-height:180px;overflow:auto";
  const run = document.createElement("button"); run.type = "button"; run.textContent = "Inspect here and reply in chat"; run.dataset.action = "workflow-inspect";
  const link = document.createElement("a"); link.target = "_blank"; link.rel = "noopener noreferrer"; link.style.marginLeft = "12px"; link.dataset.action = "workflow-web";
  const privacy = document.createElement("p"); privacy.className = "detail"; privacy.textContent = "Web continuation opens the selected workspace after you select your model and run its audit. No attachment or result is transferred; compare SHA-256 with your conversation. CLI commands are templates for your local terminal.";
  let current;
  const update = () => {
    current = guide.workflows.find(row => row.id === select.value);
    description.textContent = current.description; boundary.textContent = current.limits;
    run.hidden = !current.chat.section;
    link.href = current.url; link.textContent = current.web ? "Continue on deepbom.org ↗" : "Open CLI Handbook ↗";
    commands.textContent = current.cli.map(command => [command.executable, ...command.args.map(arg => arg.includes("<") ? `"${arg}"` : arg)].join(" ")).join("\n");
    status.textContent = current.chat.widget_action || "";
  };
  run.addEventListener("click", async () => {
    run.disabled = true;
    try { await inspect({ section: current.chat.section, ...(current.chat.weight_view ? { weight_view: current.chat.weight_view } : {}) }); status.textContent = "See the requested evidence above. Use the web workspace for additional interaction."; }
    catch (error) { status.textContent = `Could not complete this request: ${error.message}`; }
    finally { run.disabled = false; }
  });
  link.addEventListener("click", event => {
    if (typeof openai?.openExternal !== "function") return;
    event.preventDefault();
    try { Promise.resolve(openai.openExternal({ href: link.href })).catch(() => { status.textContent = `Open ${link.href} in your browser to continue.`; }); }
    catch { status.textContent = `Open ${link.href} in your browser to continue.`; }
  });
  select.addEventListener("change", update);
  details.append(heading, select, description, boundary, run, link, status, commands, privacy); container.append(details); update();
}
