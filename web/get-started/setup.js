const tabs = [...document.querySelectorAll("[data-os]")];
const tabList = document.querySelector(".os-tabs");
const status = document.querySelector("#copy-status");
let noticeTimer;

function selectOS(tab, focus = false) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
  }
  if (focus) tab.focus();
}

tabList.setAttribute("role", "tablist");
for (const [index, tab] of tabs.entries()) {
  tab.setAttribute("role", "tab");
  const panel = document.getElementById(tab.getAttribute("aria-controls"));
  panel.setAttribute("role", "tabpanel");
  panel.tabIndex = 0;
  tab.addEventListener("click", () => selectOS(tab));
  tab.addEventListener("keydown", event => {
    const next = event.key === "ArrowRight" ? (index + 1) % tabs.length
      : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length
      : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
    if (next !== null) { event.preventDefault(); selectOS(tabs[next], true); }
  });
}
const platform = navigator.userAgentData?.platform || navigator.platform || "";
const preferred = /win/i.test(platform) ? "windows" : /mac/i.test(platform) ? "macos" : /linux/i.test(platform) ? "linux" : "windows";
selectOS(tabs.find(tab => tab.dataset.os === preferred));
document.documentElement.classList.add("enhanced");

for (const button of document.querySelectorAll("[data-copy]")) {
  button.hidden = false;
  button.addEventListener("click", async () => {
    const code = document.getElementById(button.dataset.copy);
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code.textContent);
      status.textContent = "Copied to clipboard.";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Text selected. Use your device’s copy command.";
    }
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { status.textContent = ""; }, 4500);
  });
}
