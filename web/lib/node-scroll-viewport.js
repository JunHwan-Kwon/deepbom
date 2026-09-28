const boundScale = (value) => Math.max(0.4, Math.min(3, value));

// The displayed percentage is CSS pixels per SVG unit, independent of graph length.
export function zoomScrollableGraph(state, factor, anchorX, anchorY) {
  const [x, y, width, height] = state.viewBox;
  const previous = state.graphScale;
  state.graphScale = boundScale(previous / factor);
  const ratio = previous / state.graphScale;
  state.viewBox = [anchorX - (anchorX - x) * ratio, anchorY - (anchorY - y) * ratio, width * ratio, height * ratio];
}

export function mountScrollableGraph({ root, svg, layout, state, onChange }) {
  root.classList.add("nv-scrollable");
  const port = document.createElement("div");
  port.className = "nv-scrollport";
  port.tabIndex = 0;
  port.setAttribute("role", "region");
  port.setAttribute("aria-label", "Scrollable model graph");
  const space = document.createElement("div");
  space.className = "nv-scrollspace";
  space.append(svg);
  port.append(space);
  root.prepend(port);
  let originX = 0;
  let originY = 0;
  let ready = false;
  let disposed = false;

  function sync() {
    if (!ready || disposed || !port.clientWidth || !port.clientHeight) return;
    const scale = state.graphScale;
    state.viewBox = [originX + port.scrollLeft / scale, originY + port.scrollTop / scale, port.clientWidth / scale, port.clientHeight / scale];
    svg.setAttribute("viewBox", state.viewBox.join(" "));
    svg.classList.toggle("overview-scale", scale < 0.6);
    onChange(Math.round(scale * 100));
  }

  function place() {
    if (disposed || !root.isConnected || !port.clientWidth || !port.clientHeight) return;
    const [x, y] = state.viewBox;
    const scale = state.graphScale;
    const bounds = layout.bounds;
    space.style.width = `${Math.max(port.clientWidth, bounds.width * scale)}px`;
    space.style.height = `${Math.max(port.clientHeight, bounds.height * scale)}px`;
    // Re-read after the native scrollbars have taken their share of the viewport.
    const width = port.clientWidth;
    const height = port.clientHeight;
    originX = bounds.x - Math.max(0, width / scale - bounds.width) / 2;
    originY = bounds.y;
    svg.style.width = `${width}px`;
    svg.style.height = `${height}px`;
    const target = state.scrollTarget;
    const left = target ? target.x - width / scale / 2 : x;
    const top = target ? target.y - (target.top ? 0 : height / scale / 2) : y;
    port.scrollLeft = (left - originX) * scale;
    port.scrollTop = (top - originY) * scale;
    state.scrollTarget = null;
    ready = true;
    sync();
  }

  function navigate(next) {
    state.viewBox = next;
    place();
  }

  function zoom(factor, px = 0.5, py = 0.5) {
    const [x, y, width, height] = state.viewBox;
    zoomScrollableGraph(state, factor, x + width * px, y + height * py);
    place();
  }

  port.addEventListener("scroll", sync, { passive: true });
  port.addEventListener("wheel", (event) => {
    // Ordinary wheel/trackpad gestures use native scrolling, including Shift+wheel.
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const rect = port.getBoundingClientRect();
    const px = Math.max(0, Math.min(1, (event.clientX - rect.left) / port.clientWidth));
    const py = Math.max(0, Math.min(1, (event.clientY - rect.top) / port.clientHeight));
    zoom(Math.exp(Math.max(-360, Math.min(360, event.deltaY)) * 0.00135), px, py);
  }, { passive: false });
  port.addEventListener("keydown", (event) => {
    if (event.key === "+" || event.key === "=") zoom(0.8);
    else if (event.key === "-") zoom(1.2);
    else if (event.key === "0") zoom(state.graphScale);
    else if (event.key === "ArrowDown") port.scrollTop += 60;
    else if (event.key === "ArrowUp") port.scrollTop -= 60;
    else if (event.key === "ArrowLeft") port.scrollLeft -= 60;
    else if (event.key === "ArrowRight") port.scrollLeft += 60;
    else if (event.key === "PageDown") port.scrollTop += port.clientHeight * 0.8;
    else if (event.key === "PageUp") port.scrollTop -= port.clientHeight * 0.8;
    else if (event.key === "Home") port.scrollTop = 0;
    else if (event.key === "End") port.scrollTop = port.scrollHeight;
    else return;
    event.preventDefault();
  });
  let drag = null;
  svg.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "mouse" || event.button !== 0 || event.target.closest?.(".nv-node")) return;
    event.preventDefault();
    svg.setPointerCapture(event.pointerId);
    drag = { x: event.clientX, y: event.clientY, left: port.scrollLeft, top: port.scrollTop };
    svg.classList.add("dragging");
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    port.scrollLeft = drag.left - (event.clientX - drag.x);
    port.scrollTop = drag.top - (event.clientY - drag.y);
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) svg.addEventListener(name, () => {
    drag = null;
    svg.classList.remove("dragging");
  });
  const observer = new ResizeObserver(place);
  observer.observe(port);
  return { navigate, dispose: () => { disposed = true; observer.disconnect(); } };
}
