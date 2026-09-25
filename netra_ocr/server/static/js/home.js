import { api } from "./api.js";
import { num, t } from "./i18n.js";
import { formatDate, h, icon, toast } from "./util.js";

function statusLabel(job) {
  const key = job.status === "done" && job.edited ? "edited" : job.status;
  return h("span", { class: `status status-${job.status}` }, t(`status.${key}`));
}

export function renderHome(root, info) {
  let file = null;
  let pollTimer = null;
  const limits = info?.limits || { max_upload_mb: 50, max_pdf_pages: 100, job_ttl_hours: 24 };

  // ── upload ─────────────────────────────────────────────────────────
  const fileInput = h("input", { type: "file", accept: ".pdf,.jpg,.jpeg,.png,.tif,.tiff,.bmp,.webp,application/pdf,image/*",
    "aria-label": t("drop.label") });
  const dropBody = h("div", { class: "drop-body" });
  const drop = h("label", { class: "drop" }, fileInput, dropBody);

  function showDropState() {
    dropBody.replaceChildren(
      h("span", { class: "drop-icon" }, icon("upload")),
      file ? h("p", { class: "drop-file" }, file.name) : h("p", { class: "drop-title", html: t("drop.title") }),
      h("p", { class: "meta" }, file
        ? `${(file.size / 1024 / 1024).toFixed(1)} MB · ${t("drop.change")}`
        : t("drop.meta", { mb: num(limits.max_upload_mb), pages: num(limits.max_pdf_pages) })));
  }
  showDropState();

  function pick(f) {
    if (!f) return;
    file = f;
    error.hidden = true;
    showDropState();
    submit.focus();
  }
  fileInput.addEventListener("change", () => pick(fileInput.files[0]));

  // A file dropped anywhere on the page counts, not only on the drop zone.
  const onDragOver = (e) => {
    if (!e.dataTransfer?.types.includes("Files")) return;
    e.preventDefault();
    drop.classList.add("dragging");
  };
  const onDragLeave = (e) => { if (!e.relatedTarget) drop.classList.remove("dragging"); };
  const onDrop = (e) => {
    if (!e.dataTransfer?.files.length) return;
    e.preventDefault();
    drop.classList.remove("dragging");
    pick(e.dataTransfer.files[0]);
  };
  window.addEventListener("dragover", onDragOver);
  window.addEventListener("dragleave", onDragLeave);
  window.addEventListener("drop", onDrop);

  function check(input, label, hint) {
    return h("label", { class: "check" }, input, h("span", {}, label),
      h("span", { class: "hint", title: hint, "aria-label": hint }, "?"));
  }
  const layout = h("input", { type: "checkbox", checked: true });
  const headers = h("input", { type: "checkbox", checked: true });
  const detector = h("select", { id: "opt-detector" },
    ["yolo", "legacy", "tesseract"].map((d) => h("option", { value: d }, t(`det.${d}`))));
  const decoder = h("select", { id: "opt-decoder" },
    ["ar", "blockwise"].map((d) => h("option", { value: d }, t(`dec.${d}`))));
  if (info?.defaults) {
    detector.value = info.defaults.detector;
    decoder.value = info.defaults.decoder;
  }
  const advanced = h("div", { class: "advanced", id: "opt-advanced", hidden: true },
    h("div", { class: "field" }, h("label", { for: "opt-detector" }, t("opt.detector")), detector),
    h("div", { class: "field" }, h("label", { for: "opt-decoder" }, t("opt.decoder")), decoder));
  const advToggle = h("button", { class: "adv-toggle", type: "button", "aria-expanded": "false", "aria-controls": "opt-advanced",
    onclick: () => {
      advanced.hidden = !advanced.hidden;
      advToggle.setAttribute("aria-expanded", String(!advanced.hidden));
    } }, t("opt.advanced"));
  const error = h("p", { class: "error-text upload-error", role: "alert", hidden: true });
  const submit = h("button", { class: "pill pill-primary", type: "button" }, t("opt.submit"));

  submit.addEventListener("click", async () => {
    if (!file) {
      error.textContent = t("opt.pick");
      error.hidden = false;
      return;
    }
    const form = new FormData();
    form.append("file", file);
    form.append("layout", layout.checked);
    form.append("include_headers_footers", headers.checked);
    form.append("detector", detector.value);
    form.append("decoder", decoder.value);
    form.append("wait", "false");
    submit.disabled = true;
    error.hidden = true;
    try {
      const res = await api.upload(form);
      location.hash = `#/job/${res.job.id}`;
    } catch (e) {
      error.textContent = e.status === 0 ? t("err.network") : (e.message || t("err.generic"));
      error.hidden = false;
      submit.disabled = false;
    }
  });

  const upload = h("div", { class: "upload reveal-2" },
    drop,
    h("div", { class: "upload-bar" },
      check(layout, t("opt.layout"), t("opt.layoutDesc")),
      check(headers, t("opt.headers"), t("opt.headersDesc")),
      advToggle,
      h("span", { class: "spacer" }),
      submit),
    advanced,
    error);

  // ── recent documents ───────────────────────────────────────────────
  const tableHost = h("div");

  function deleteButton(job) {
    // Two clicks, no browser dialog: the first arms it, the second deletes.
    const btn = h("button", { class: "icon-btn", type: "button", title: t("recent.delete"), "aria-label": t("recent.delete") },
      icon("trash"));
    let armed = false;
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!armed) {
        armed = true;
        btn.classList.add("pill", "pill-sm", "armed");
        btn.replaceChildren(t("recent.confirm"));
        return;
      }
      btn.disabled = true;
      try {
        await api.deleteJob(job.id);
        loadJobs();
      } catch {
        toast(t("err.generic"));
        btn.disabled = false;
      }
    });
    btn.addEventListener("blur", () => {
      if (!armed || btn.disabled) return;
      armed = false;
      btn.classList.remove("pill", "pill-sm", "armed");
      btn.replaceChildren(icon("trash"));
    });
    btn.addEventListener("keydown", (e) => e.stopPropagation());
    return btn;
  }

  async function loadJobs() {
    let jobs = [];
    try {
      jobs = (await api.jobs()).jobs;
    } catch (e) {
      if (e.status !== 401) toast(t("err.network"));
      return;
    }
    if (!jobs.length) {
      tableHost.replaceChildren(h("p", { class: "empty" }, t("recent.empty")));
    } else {
      tableHost.replaceChildren(h("table", { class: "index" },
        h("thead", {}, h("tr", {},
          h("th", {}, t("col.name")), h("th", {}, t("col.pages")),
          h("th", {}, t("col.status")), h("th", { class: "hide-sm" }, t("col.created")),
          h("th", { class: "act" }, h("span", { class: "sr-only" }, t("col.actions"))))),
        h("tbody", {}, jobs.map((j) => h("tr", { tabindex: "0", onclick: () => { location.hash = `#/job/${j.id}`; },
          onkeydown: (e) => { if (e.key === "Enter") location.hash = `#/job/${j.id}`; } },
        h("td", { class: "name" }, j.filename),
        h("td", { class: "num-cell" }, num(j.progress.total)),
        h("td", {}, statusLabel(j)),
        h("td", { class: "num-cell hide-sm" }, formatDate(j.created)),
        h("td", { class: "act" },
          j.status === "queued" || j.status === "running" ? null : deleteButton(j),
          h("span", { class: "arrow", "aria-hidden": "true" }, " →")))))));
    }
    clearTimeout(pollTimer);
    if (jobs.some((j) => j.status === "queued" || j.status === "running")) {
      pollTimer = setTimeout(loadJobs, 2000);
    }
  }
  loadJobs();

  root.replaceChildren(h("section", { class: "wrap page" },
    h("header", { class: "page-head reveal" },
      h("h1", { class: "h1" }, t("home.title")),
      h("p", { class: "lede muted" }, t("home.lede"))),
    upload,
    h("div", { class: "list-head" },
      h("h2", { class: "h3" }, t("recent.title")),
      h("span", { class: "meta muted2" }, t("recent.note", { h: num(limits.job_ttl_hours) }))),
    tableHost));

  return () => {
    clearTimeout(pollTimer);
    window.removeEventListener("dragover", onDragOver);
    window.removeEventListener("dragleave", onDragLeave);
    window.removeEventListener("drop", onDrop);
  };
}
