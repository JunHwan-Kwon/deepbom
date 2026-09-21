const status = document.querySelector("#book-status");
let noticeTimer;
for (const [index, block] of [...document.querySelectorAll(".book-code")].entries()) {
  const code = block.querySelector("code");
  if (!code) continue;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "book-copy";
  button.textContent = "Copy";
  button.setAttribute("aria-label", `Copy code example ${index + 1}`);
  button.addEventListener("click", async () => {
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
  block.prepend(button);
}
for (const button of document.querySelectorAll("[data-print]")) {
  button.hidden = false;
  button.addEventListener("click", () => window.print());
}
const search = document.querySelector("#command-search");
if (search) {
  search.closest(".command-search").hidden = false;
  const rows = [...document.querySelectorAll("[data-command-row]")];
  const count = document.querySelector("#command-count");
  const empty = document.querySelector("#command-empty");
  search.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();
    for (const row of rows) row.hidden = !row.textContent.toLowerCase().includes(query);
    const visible = rows.filter(row => !row.hidden).length;
    count.textContent = `${visible} of ${rows.length} commands`;
    empty.hidden = visible !== 0;
  });
}
