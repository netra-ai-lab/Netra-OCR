import { api, setKey } from "./api.js";
import { renderApi } from "./apipage.js";
import { renderHome } from "./home.js";
import { applyStatic, t } from "./i18n.js";
import { renderJob } from "./job.js";
import { h } from "./util.js";

const view = document.getElementById("view");
let cleanup = null;
let info = null;

const engine = document.getElementById("engine-status");
let engineTimer = null;

// Header indicator: the models load in the background after start-up, so
// the first upload may wait; say so instead of looking stuck.
function showEngine() {
  clearTimeout(engineTimer);
  if (!info) {
    engine.className = "status status-error engine-status";
    engine.textContent = t("engine.offline");
    return;
  }
  const device = info.device_name || (info.device.startsWith("cuda") ? "GPU" : "CPU");
  if (info.models_ready) {
    engine.className = "status status-done engine-status";
    engine.textContent = t("engine.ready", { device });
  } else {
    engine.className = "status status-running engine-status";
    engine.textContent = t("engine.loading");
    engineTimer = setTimeout(async () => { await loadInfo(); showEngine(); }, 2000);
  }
}

async function loadInfo() {
  try {
    info = await api.info();
    document.getElementById("footer-version").textContent = `Netra OCR ${info.version}`;
  } catch {
    info = null;
  }
}

function route() {
  if (cleanup) { cleanup(); cleanup = null; }
  const hash = location.hash || "#/";
  const job = hash.match(/^#\/job\/([0-9a-f]{32})$/);
  document.querySelectorAll("[data-nav]").forEach((a) => a.removeAttribute("aria-current"));
  if (job) {
    document.querySelector('[data-nav="home"]').setAttribute("aria-current", "page");
    cleanup = renderJob(view, job[1]);
  } else if (hash === "#/api") {
    document.querySelector('[data-nav="api"]').setAttribute("aria-current", "page");
    cleanup = renderApi(view, info);
  } else {
    document.querySelector('[data-nav="home"]').setAttribute("aria-current", "page");
    cleanup = renderHome(view, info);
  }
  window.scrollTo(0, 0);
}


window.addEventListener("netra:auth", () => {
  if (document.getElementById("auth-panel")) return;
  const input = h("input", { type: "password", autocomplete: "off", "aria-label": "API key" });
  const panel = h("section", { class: "wrap", id: "auth-panel" }, h("div", { class: "processing" },
    h("p", { class: "eyebrow muted2" }, "401"),
    h("h1", { class: "h2" }, t("auth.title")),
    h("p", { class: "note" }, t("auth.text")),
    h("div", { class: "field", style: "max-width:24rem" }, input),
    h("p", { style: "margin-top:16px" }, h("button", { class: "pill pill-primary", type: "button",
      onclick: async () => { setKey(input.value.trim()); await loadInfo(); route(); } }, t("auth.save")))));
  if (cleanup) { cleanup(); cleanup = null; }
  view.replaceChildren(panel);
  input.focus();
});

window.addEventListener("hashchange", route);
applyStatic();
loadInfo().then(() => { showEngine(); route(); });
