// Small DOM helpers shared by the views.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// Stroke icons (24px grid, drawn at 16px), currentColor.
const PATHS = {
  undo: "M9 14 4 9l5-5M4 9h10.5a5.5 5.5 0 0 1 0 11H11",
  redo: "m15 14 5-5-5-5M20 9H9.5a5.5 5.5 0 0 0 0 11H13",
  up: "M12 19V5M5 12l7-7 7 7",
  down: "M12 5v14M19 12l-7 7-7-7",
  trash: "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6",
  merge: "M8 6h8M8 12h8M8 18h8M4 6v12",
  split: "M4 8h16M4 16h16M9 12h6",
  alignLeft: "M3 6h18M3 12h12M3 18h16",
  alignCenter: "M3 6h18M6 12h12M4 18h16",
  alignRight: "M3 6h18M9 12h12M5 18h16",
  alignJustify: "M3 6h18M3 12h18M3 18h18",
  rowAdd: "M3 5h18v6H3zM12 15v6M9 18h6",
  colAdd: "M5 3h6v18H5zM15 12h6M18 9v6",
  rowDel: "M3 5h18v6H3zM9 18h6",
  colDel: "M5 3h6v18H5zM15 12h6",
  chevLeft: "m15 18-6-6 6-6",
  chevRight: "m9 18 6-6-6-6",
  chevDown: "m6 9 6 6 6-6",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  upload: "M12 16V4M7 9l5-5 5 5M4 16v4h16v-4",
  reset: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5",
  plus: "M12 5v14M5 12h14",
};

export function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", PATHS[name]);
  svg.append(p);
  return svg;
}

export function iconButton(name, label, onclick, extra = {}) {
  return h("button", { class: "icon-btn", type: "button", title: label, "aria-label": label, onclick, ...extra }, icon(name));
}

let toastTimer;
export function toast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

export function formatDate(ts) {
  try {
    return new Intl.DateTimeFormat("en-GB",
      { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(ts * 1000));
  } catch {
    return new Date(ts * 1000).toLocaleString();
  }
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Clipboard API needs a secure context; plain-HTTP LAN deployments fall back.
    const ta = h("textarea", { style: "position:fixed;opacity:0" });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

export function isKhmer(ch) {
  const c = ch.codePointAt(0);
  return (c >= 0x1780 && c <= 0x17ff) || (c >= 0x19e0 && c <= 0x19ff);
}
