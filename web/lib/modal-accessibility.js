const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';
const returnFocusByModal = new WeakMap();

export function openModal(backdrop, { focus = null } = {}) {
  if (!backdrop) return;
  const active = document.activeElement;
  if (active instanceof HTMLElement && active !== document.body && active !== document.documentElement) {
    returnFocusByModal.set(backdrop, active);
  }
  backdrop.hidden = false;
  backdrop.inert = false;
  backdrop.removeAttribute("aria-hidden");
  document.body.classList.add("modal-open");
  requestAnimationFrame(() => (focus || visibleFocusable(backdrop)[0])?.focus());
}

export function closeModal(backdrop, { fallbackFocus = null } = {}) {
  if (!backdrop) return;
  const wasOpen = !backdrop.hidden;
  backdrop.hidden = true;
  backdrop.inert = true;
  backdrop.setAttribute("aria-hidden", "true");
  if (!document.querySelector('[role="dialog"]:not([hidden])')) document.body.classList.remove("modal-open");
  const returnFocus = returnFocusByModal.get(backdrop);
  returnFocusByModal.delete(backdrop);
  if (!wasOpen) return;
  requestAnimationFrame(() => {
    const target = validFocusTarget(returnFocus) ? returnFocus : fallbackFocus;
    target?.focus();
  });
}

export function installModalKeyboard(backdrop, onEscape = null) {
  const handler = (event) => {
    if (!backdrop || backdrop.hidden) return;
    if (event.key === "Escape" && onEscape) {
      event.preventDefault();
      onEscape();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = visibleFocusable(backdrop);
    if (!focusable.length) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };
  document.addEventListener("keydown", handler);
  return () => document.removeEventListener("keydown", handler);
}

function visibleFocusable(backdrop) {
  return [...backdrop.querySelectorAll(FOCUSABLE)].filter((element) => element.getClientRects().length > 0);
}

function validFocusTarget(target) {
  return target instanceof HTMLElement && target !== document.body && target !== document.documentElement
    && target.isConnected && !target.hidden && !target.closest("[inert]");
}
