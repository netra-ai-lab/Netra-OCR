import { api } from "./api.js";
import { num, t } from "./i18n.js";
import { copyText, h, icon, iconButton, isKhmer, toast } from "./util.js";

const TEXT_TYPES = new Set(["heading", "paragraph", "caption", "page_header", "page_footer"]);
const TYPE_CHOICES = ["heading-1", "heading-2", "heading-3", "paragraph", "caption", "page_header", "page_footer"];
const FORMATS = ["docx", "html", "md", "json", "txt"];
const HISTORY_LIMIT = 200;
const SAVE_DELAY = 700;

// contenteditable="plaintext-only" keeps pasted/dropped content as plain
// text natively; older engines fall back to "true" + our own paste handler.
const PLAINTEXT_ONLY = (() => {
  const d = document.createElement("div");
  try { d.contentEditable = "plaintext-only"; } catch { return false; }
  return d.contentEditable === "plaintext-only";
})();

function typeKey(b) {
  return b.type === "heading" ? `heading-${b.level || 1}` : b.type;
}

export function renderJob(root, jobId) {
  let disposed = false;
  let pollTimer = null;
  let cleanupEditor = null;

  async function poll() {
    if (disposed) return;
    let job;
    try {
      job = (await api.job(jobId)).job;
    } catch (e) {
      if (e.status === 404) return renderMissing();
      pollTimer = setTimeout(poll, 3000);
      return;
    }
    if (job.status === "done") {
      try {
        const doc = await api.document(jobId);
        if (!disposed) cleanupEditor = renderEditor(root, job, doc);
      } catch {
        toast(t("err.generic"));
      }
      return;
    }
    renderProgress(job);
    if (job.status !== "error") pollTimer = setTimeout(poll, 1000);
  }

  function renderMissing() {
    root.replaceChildren(h("section", { class: "wrap" }, h("div", { class: "processing" },
      h("p", { class: "eyebrow muted2" }, t("status.error")),
      h("h1", { class: "h2" }, t("proc.missing")),
      h("p", { class: "note" }, h("a", { href: "#/" }, `← ${t("proc.back")}`)))));
  }

  let progressEls = null;
  function renderProgress(job) {
    if (!progressEls) {
      progressEls = {
        eyebrow: h("p", { class: "eyebrow muted2" }),
        bar: h("div", { class: "bar" }, h("span")),
        meta: h("p", { class: "meta" }),
        note: h("p", { class: "note" }),
      };
      root.replaceChildren(h("section", { class: "wrap" }, h("div", { class: "processing reveal" },
        progressEls.eyebrow, h("h1", { class: "h2" }, job.filename),
        progressEls.bar, progressEls.meta, progressEls.note)));
    }
    const { eyebrow, bar, meta, note } = progressEls;
    if (job.status === "error") {
      eyebrow.textContent = t("status.error");
      bar.hidden = true;
      meta.textContent = t("proc.failed");
      note.replaceChildren(h("span", { class: "error-text" }, job.error || ""), h("br"), h("br"),
        h("a", { href: "#/" }, `← ${t("proc.back")}`));
      return;
    }
    eyebrow.textContent = t("proc.eyebrow");
    const { done, total } = job.progress;
    bar.classList.toggle("indeterminate", job.status === "queued" || done === 0);
    bar.firstChild.style.width = total ? `${Math.max(4, (done / total) * 100)}%` : "0";
    meta.textContent = job.status === "queued" ? t("proc.queued")
      : done === 0 ? t("proc.starting") : t("proc.page", { done: num(done), total: num(total) });
    note.textContent = t("proc.note");
  }

  poll();
  return () => {
    disposed = true;
    clearTimeout(pollTimer);
    if (cleanupEditor) cleanupEditor();
  };
}

// ════════════════════════════════════════════════════════════════════════
// Editor
// ════════════════════════════════════════════════════════════════════════

function renderEditor(root, job, initialDoc) {
  const state = {
    doc: initialDoc,
    revision: initialDoc.revision,
    page: 0,
    activeId: null,
    cell: null,               // {r, c} of the focused table cell
    undo: [],
    redo: [],
    lastTypingSnapshot: 0,
    dirty: false,
    saving: false,
    saveTimer: null,
    conflict: false,
  };
  const pages = state.doc.pages;
  const firstBlock = state.doc.blocks[0];
  if (firstBlock) state.page = firstBlock.page;

  // ── toolbar ─────────────────────────────────────────────────────────
  const contextRow = h("div", { class: "context-row" });
  const saveState = h("span", { class: "save-state", role: "status" }, t("ed.saved"));
  const undoBtn = iconButton("undo", t("ed.undo"), () => undo());
  const redoBtn = iconButton("redo", t("ed.redo"), () => redo());
  const menuList = h("div", { class: "menu-list", role: "menu", hidden: true },
    FORMATS.map((f) => h("button", { type: "button", role: "menuitem", onclick: () => { closeMenu(); download(f); } },
      h("span", {}, t(`fmt.${f}`)), h("span", { class: "meta" }, `.${f === "md" ? "md" : f}`))));
  const moreBtn = h("button", { class: "pill pill-sm", type: "button", "aria-haspopup": "menu", "aria-expanded": "false",
    onclick: (e) => { e.stopPropagation(); menuList.hidden ? openMenu() : closeMenu(); } }, t("ed.more"), icon("chevDown"));
  moreBtn.querySelector("svg").style.width = "14px";

  function openMenu() { menuList.hidden = false; moreBtn.setAttribute("aria-expanded", "true"); }
  function closeMenu() { menuList.hidden = true; moreBtn.setAttribute("aria-expanded", "false"); }
  const onDocClick = () => closeMenu();
  document.addEventListener("click", onDocClick);

  const toolbar = h("div", { class: "toolbar" }, h("div", { class: "toolbar-inner" },
    h("a", { class: "icon-btn", href: "#/", title: t("proc.back"), "aria-label": t("proc.back") }, icon("chevLeft")),
    h("div", { class: "doc-title" },
      h("strong", {}, job.filename),
      h("span", { class: "meta" }, `${t("col.pages")}: ${num(pages.length)}`)),
    saveState,
    h("div", { class: "tool-group" }, undoBtn, redoBtn,
      iconButton("reset", t("ed.revert"), () => revert())),
    h("span", { class: "divider", "aria-hidden": "true" }),
    h("button", { class: "pill pill-sm", type: "button", onclick: copyAll }, icon("copy"), t("ed.copy")),
    h("div", { class: "menu" }, moreBtn, menuList),
    h("button", { class: "pill pill-primary pill-sm", type: "button", onclick: () => download("docx") }, t("ed.download"))),
    // Tools for the selected block live here rather than floating over the
    // text, so they never cover the block above.
    contextRow);
  toolbar.querySelectorAll(".pill svg").forEach((s) => { s.style.width = "14px"; s.style.height = "14px"; });

  const banner = h("div", { class: "banner", hidden: true }, h("div", { class: "banner-inner" },
    h("span", {}, t("ed.conflict")),
    h("button", { class: "pill pill-sm", type: "button", onclick: reloadLatest }, t("ed.reload"))));

  // ── page viewer ─────────────────────────────────────────────────────
  const pageImg = h("img", { alt: t("ed.original"), draggable: "false" });
  const overlay = h("div", { class: "overlay" });
  const stage = h("div", { class: "page-stage" }, pageImg, overlay);
  const frame = h("div", { class: "page-frame" }, stage);
  const pageLabel = h("span", { class: "meta" });
  const prevBtn = iconButton("chevLeft", t("ed.prev"), () => showPage(state.page - 1));
  const nextBtn = iconButton("chevRight", t("ed.next"), () => showPage(state.page + 1));
  const thumbs = h("div", { class: "thumbs", hidden: pages.length < 2 },
    pages.map((p) => h("button", { class: "thumb", type: "button", "aria-label": t("ed.pageLabel", { n: num(p.index + 1) }),
      onclick: () => showPage(p.index) },
    h("img", { src: p.image ? api.assetUrl(job.id, p.image) : "", alt: "", loading: "lazy" }))));
  const viewer = h("aside", { class: "viewer" },
    h("div", { class: "viewer-head" },
      h("span", { class: "eyebrow muted2" }, t("ed.original")),
      h("div", { class: "tool-group" }, prevBtn, pageLabel, nextBtn)),
    frame, thumbs);

  function showPage(index) {
    if (index < 0 || index >= pages.length) return;
    const changed = index !== state.page || !pageImg.src;
    state.page = index;
    const p = pages[index];
    if (changed) pageImg.src = p.image ? api.assetUrl(job.id, p.image) : "";
    pageLabel.textContent = t("ed.page", { n: num(index + 1), total: num(pages.length) });
    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === pages.length - 1;
    thumbs.querySelectorAll(".thumb").forEach((el, i) => el.setAttribute("aria-current", String(i === index)));
    renderOverlay();
  }

  function renderOverlay() {
    const p = pages[state.page];
    if (!p || !p.width) return overlay.replaceChildren();
    overlay.replaceChildren(...state.doc.blocks
      .filter((b) => b.page === state.page && b.bbox)
      .map((b) => {
        const [x1, y1, x2, y2] = b.bbox;
        const el = h("button", {
          class: `ov${b.id === state.activeId ? " active" : ""}`, type: "button", dataset: { id: b.id, type: b.type },
          style: `left:${(x1 / p.width) * 100}%;top:${(y1 / p.height) * 100}%;`
            + `width:${((x2 - x1) / p.width) * 100}%;height:${((y2 - y1) / p.height) * 100}%`,
          "aria-label": t(`type.${typeKey(b)}`),
          onclick: () => focusBlock(b.id, true),
        }, h("span", { class: "ov-tag" }, t(`type.${typeKey(b)}`)));
        return el;
      }));
  }

  // ── document ────────────────────────────────────────────────────────
  const docEl = h("article", { class: "doc" });
  const hint = h("p", { class: "doc-hint meta" }, t("ed.hint"));
  const addBtn = h("button", { class: "pill pill-sm add-block", type: "button", onclick: () => addParagraph() },
    icon("plus"), t("ed.addParagraph"));
  addBtn.querySelector("svg").style.width = "14px";
  const docCol = h("div", {}, docEl);

  function blockIndex(id) { return state.doc.blocks.findIndex((b) => b.id === id); }
  function blockById(id) { return state.doc.blocks.find((b) => b.id === id); }

  function renderDoc() {
    const children = [];
    let prevPage = null;
    for (const b of state.doc.blocks) {
      if (pages.length > 1 && b.page !== prevPage) {
        children.push(h("div", { class: "page-break eyebrow" }, t("ed.pageLabel", { n: num(b.page + 1) })));
      }
      prevPage = b.page;
      children.push(renderBlock(b));
    }
    if (!state.doc.blocks.length) children.push(h("p", { class: "empty" }, t("ed.empty")));
    docEl.replaceChildren(...children, addBtn, hint);
    renderOverlay();
    renderContext();
  }

  function renderBlock(b) {
    const key = typeKey(b);
    const el = h("div", {
      class: `block b-${b.type === "heading" ? `heading-${b.level || 1}` : b.type} align-${b.align || "left"}`
        + `${b.id === state.activeId ? " active" : ""}`,
      dataset: { id: b.id },
    });
    const label = h("div", { class: "block-label", onclick: () => focusBlock(b.id) }, t(`type.${key}`));
    const body = h("div", { class: "block-body" });

    if (TEXT_TYPES.has(b.type)) {
      const ed = editable(b.text, (text) => { b.text = text; });
      ed.dataset.placeholder = t("ed.placeholder");
      body.append(ed);
    } else if (b.type === "table") {
      if (b.rows.length === 1 && b.rows[0].length === 1 && b.image) {
        body.append(h("figure", { class: "tbl-image" }, h("img", { src: api.assetUrl(job.id, b.image), alt: "", loading: "lazy" })));
      }
      body.append(h("div", { class: "tbl-wrap" }, h("table", {}, h("tbody", {},
        b.rows.map((row, r) => h("tr", {}, row.map((cell, c) => {
          const td = h("td");
          const ed = editable(cell, (text) => { b.rows[r][c] = text; });
          ed.addEventListener("focus", () => { state.cell = { r, c }; });
          td.append(ed);
          return td;
        })))))));
    } else {
      body.append(h("figure", {}, h("img", { src: api.assetUrl(job.id, b.image), alt: t(`type.${key}`), loading: "lazy" })));
      body.tabIndex = 0;
    }
    el.append(label, body);
    el.addEventListener("focusin", () => setActive(b.id));
    el.addEventListener("click", () => setActive(b.id));
    return el;
  }

  function editable(text, onChange) {
    const ed = h("div", { class: "editable", spellcheck: "false" });
    ed.contentEditable = PLAINTEXT_ONLY ? "plaintext-only" : "true";
    ed.textContent = text;
    let composing = false;
    ed.addEventListener("compositionstart", () => { composing = true; });
    ed.addEventListener("compositionend", () => { composing = false; commitText(); });
    ed.addEventListener("input", () => { if (!composing) commitText(); });
    function commitText() {
      snapshotForTyping();
      onChange(ed.textContent);
      scheduleSave();
    }
    ed.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        splitActive(ed);
      } else if (e.key === "Enter" && !e.isComposing) {
        // A literal newline, never a <div>/<br> -- the model is plain text.
        e.preventDefault();
        document.execCommand("insertText", false, "\n");
      }
    });
    ed.addEventListener("paste", (e) => {
      e.preventDefault();
      const text = (e.clipboardData || window.clipboardData).getData("text/plain");
      document.execCommand("insertText", false, text.replace(/\r\n?/g, "\n"));
    });
    ed.addEventListener("drop", (e) => { if (!PLAINTEXT_ONLY) e.preventDefault(); });
    return ed;
  }

  function renderTools(b) {
    const tools = h("div", { class: "block-tools", role: "toolbar", "aria-label": t(`type.${typeKey(b)}`) });
    const i = blockIndex(b.id);
    if (TEXT_TYPES.has(b.type)) {
      const select = h("select", { "aria-label": "Block type",
        onchange: () => mutate(() => {
          const v = select.value;
          if (v.startsWith("heading-")) { b.type = "heading"; b.level = Number(v.slice(8)); }
          else { b.type = v; delete b.level; }
        }) }, TYPE_CHOICES.map((c) => h("option", { value: c, selected: c === typeKey(b) }, t(`type.${c}`))));
      tools.append(select, h("span", { class: "sep" }));
      for (const [a, name] of [["left", "alignLeft"], ["center", "alignCenter"], ["right", "alignRight"], ["justify", "alignJustify"]]) {
        tools.append(iconButton(name, t(`tool.${name}`), () => mutate(() => { b.align = a; }),
          { "aria-pressed": String((b.align || "left") === a) }));
      }
      tools.append(h("span", { class: "sep" }),
        iconButton("split", t("tool.split"), () => splitActive(document.querySelector(`.block[data-id="${b.id}"] .editable`))));
      const next = state.doc.blocks[i + 1];
      tools.append(iconButton("merge", t("tool.merge"), () => mergeWithNext(b.id), { disabled: !(next && TEXT_TYPES.has(next.type)) }));
    } else if (b.type === "table") {
      tools.append(
        iconButton("rowAdd", t("tool.addRow"), () => tableOp(b, "addRow")),
        iconButton("colAdd", t("tool.addCol"), () => tableOp(b, "addCol")),
        iconButton("rowDel", t("tool.delRow"), () => tableOp(b, "delRow"), { disabled: b.rows.length < 2 }),
        iconButton("colDel", t("tool.delCol"), () => tableOp(b, "delCol"), { disabled: b.rows[0].length < 2 }));
    }
    tools.append(h("span", { class: "sep" }),
      iconButton("up", t("tool.moveUp"), () => move(b.id, -1), { disabled: i === 0 }),
      iconButton("down", t("tool.moveDown"), () => move(b.id, 1), { disabled: i === state.doc.blocks.length - 1 }),
      iconButton("trash", t("tool.delete"), () => remove(b.id)));
    // Keep focus in the text while clicking tools (so split knows the caret).
    tools.addEventListener("mousedown", (e) => { if (e.target.closest("button")) e.preventDefault(); });
    return tools;
  }

  function renderContext() {
    const b = state.activeId && blockById(state.activeId);
    contextRow.replaceChildren(h("div", { class: "context-inner" }, ...(b
      ? [h("span", { class: "eyebrow muted2" }, t(`type.${typeKey(b)}`)), renderTools(b)]
      : [h("span", { class: "meta muted2" }, t("ed.hint"))])));
  }

  function setActive(id) {
    if (state.activeId === id) return;
    docEl.querySelector(".block.active")?.classList.remove("active");
    state.activeId = id;
    const el = docEl.querySelector(`.block[data-id="${CSS.escape(id)}"]`);
    const b = blockById(id);
    if (el && b) el.classList.add("active");
    renderContext();
    if (b && b.page !== state.page) showPage(b.page);
    else renderOverlay();
    const ov = overlay.querySelector(`.ov[data-id="${CSS.escape(id)}"]`);
    if (ov) scrollIntoFrame(ov);
  }

  function scrollIntoFrame(ov) {
    const f = frame.getBoundingClientRect();
    const r = ov.getBoundingClientRect();
    if (r.top < f.top || r.bottom > f.bottom) frame.scrollTop += r.top - f.top - f.height / 3;
  }

  function focusBlock(id, fromOverlay = false) {
    setActive(id);
    const el = docEl.querySelector(`.block[data-id="${CSS.escape(id)}"]`);
    if (!el) return;
    const target = el.querySelector(".editable") || el.querySelector(".block-body");
    target?.focus({ preventScroll: fromOverlay });
    if (fromOverlay) el.scrollIntoView({ block: "center" });
  }

  // ── edits ───────────────────────────────────────────────────────────
  function snapshot() {
    state.undo.push(JSON.stringify(state.doc.blocks));
    if (state.undo.length > HISTORY_LIMIT) state.undo.shift();
    state.redo = [];
    updateHistoryButtons();
  }

  // Typing is recorded as one undo step per burst (a pause > 1.2 s starts a new one).
  function snapshotForTyping() {
    const now = Date.now();
    if (now - state.lastTypingSnapshot > 1200) snapshot();
    state.lastTypingSnapshot = now;
  }

  function mutate(fn, { focusId } = {}) {
    snapshot();
    state.lastTypingSnapshot = 0;
    fn();
    renderDoc();
    scheduleSave();
    const id = focusId || state.activeId;
    if (id && blockById(id)) {
      state.activeId = null;
      focusBlock(id);
    }
  }

  function newId() { return `u-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`; }

  function joinText(a, b) {
    if (!a) return b;
    if (!b) return a;
    return isKhmer(a.at(-1)) && isKhmer(b[0]) ? a + b : `${a} ${b}`;
  }

  function caretOffset(ed) {
    const sel = window.getSelection();
    if (!sel.rangeCount || !ed.contains(sel.anchorNode)) return ed.textContent.length;
    const range = sel.getRangeAt(0).cloneRange();
    const pre = document.createRange();
    pre.selectNodeContents(ed);
    pre.setEnd(range.endContainer, range.endOffset);
    return pre.toString().length;
  }

  function splitActive(ed) {
    const b = blockById(state.activeId);
    if (!b || !ed || !TEXT_TYPES.has(b.type)) return;
    const at = caretOffset(ed);
    const text = ed.textContent;
    const first = text.slice(0, at).replace(/\s+$/, "");
    const rest = text.slice(at).replace(/^\s+/, "");
    const nb = { id: newId(), type: b.type === "heading" ? "paragraph" : b.type, text: rest, page: b.page,
      align: b.align || "left", bbox: null };
    mutate(() => {
      b.text = first;
      state.doc.blocks.splice(blockIndex(b.id) + 1, 0, nb);
    }, { focusId: nb.id });
  }

  function mergeWithNext(id) {
    const i = blockIndex(id);
    const a = state.doc.blocks[i];
    const b = state.doc.blocks[i + 1];
    if (!a || !b || !TEXT_TYPES.has(b.type)) return;
    mutate(() => {
      a.text = joinText(a.text, b.text);
      if (a.bbox && b.bbox && a.page === b.page) {
        a.bbox = [Math.min(a.bbox[0], b.bbox[0]), Math.min(a.bbox[1], b.bbox[1]),
          Math.max(a.bbox[2], b.bbox[2]), Math.max(a.bbox[3], b.bbox[3])];
      }
      if (b.lines) a.lines = [...(a.lines || []), ...b.lines];
      state.doc.blocks.splice(i + 1, 1);
    }, { focusId: a.id });
  }

  function move(id, dir) {
    const i = blockIndex(id);
    const j = i + dir;
    if (j < 0 || j >= state.doc.blocks.length) return;
    mutate(() => {
      const blocks = state.doc.blocks;
      [blocks[i], blocks[j]] = [blocks[j], blocks[i]];
      // Crossing a page boundary moves the block onto that page, so page
      // numbers stay in reading order (a page change = a page break).
      if (blocks[j].page !== blocks[i].page) blocks[j].page = blocks[i].page;
    }, { focusId: id });
  }

  function remove(id) {
    const i = blockIndex(id);
    const neighbour = state.doc.blocks[i + 1] || state.doc.blocks[i - 1];
    mutate(() => {
      state.doc.blocks.splice(i, 1);
      state.activeId = null;
    }, { focusId: neighbour?.id });
  }

  function addParagraph() {
    const i = state.activeId ? blockIndex(state.activeId) : state.doc.blocks.length - 1;
    const ref = state.doc.blocks[i];
    const nb = { id: newId(), type: "paragraph", text: "", page: ref ? ref.page : state.page, align: "left", bbox: null };
    mutate(() => { state.doc.blocks.splice(i + 1, 0, nb); }, { focusId: nb.id });
  }

  function tableOp(b, op) {
    const { r = b.rows.length - 1, c = b.rows[0].length - 1 } = state.cell || {};
    mutate(() => {
      const width = b.rows[0].length;
      if (op === "addRow") b.rows.splice(r + 1, 0, Array(width).fill(""));
      if (op === "addCol") b.rows.forEach((row) => row.splice(c + 1, 0, ""));
      if (op === "delRow" && b.rows.length > 1) b.rows.splice(Math.min(r, b.rows.length - 1), 1);
      if (op === "delCol" && width > 1) b.rows.forEach((row) => row.splice(Math.min(c, width - 1), 1));
    }, { focusId: b.id });
    state.cell = null;
  }

  function undo() {
    if (!state.undo.length) return;
    state.redo.push(JSON.stringify(state.doc.blocks));
    state.doc.blocks = JSON.parse(state.undo.pop());
    afterHistoryJump();
  }

  function redo() {
    if (!state.redo.length) return;
    state.undo.push(JSON.stringify(state.doc.blocks));
    state.doc.blocks = JSON.parse(state.redo.pop());
    afterHistoryJump();
  }

  function afterHistoryJump() {
    state.lastTypingSnapshot = 0;
    if (state.activeId && !blockById(state.activeId)) state.activeId = null;
    renderDoc();
    updateHistoryButtons();
    scheduleSave();
  }

  function updateHistoryButtons() {
    undoBtn.disabled = !state.undo.length;
    redoBtn.disabled = !state.redo.length;
  }

  async function revert() {
    await flushSave();
    try {
      const doc = await api.resetDocument(job.id);
      state.undo.push(JSON.stringify(state.doc.blocks));
      state.redo = [];
      state.doc = doc;
      state.revision = doc.revision;
      state.activeId = null;
      renderDoc();
      updateHistoryButtons();
      toast(t("ed.reverted"));
    } catch {
      toast(t("err.generic"));
    }
  }

  // ── saving ──────────────────────────────────────────────────────────
  function setSaveState(kind) {
    saveState.className = `save-state${kind === "unsaved" ? " dirty" : kind === "error" ? " error" : ""}`;
    saveState.textContent = t({ saved: "ed.saved", saving: "ed.saving", unsaved: "ed.unsaved", error: "ed.saveError" }[kind]);
  }

  function scheduleSave() {
    if (state.conflict) return;
    state.dirty = true;
    setSaveState("unsaved");
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(save, SAVE_DELAY);
  }

  async function save(keepalive = false) {
    clearTimeout(state.saveTimer);
    if (!state.dirty || state.conflict) return;
    if (state.saving) { state.saveTimer = setTimeout(save, 300); return; }
    state.saving = true;
    state.dirty = false;
    setSaveState("saving");
    const payload = { pages: state.doc.pages, blocks: state.doc.blocks };
    try {
      const res = await api.saveDocument(job.id, payload, state.revision, keepalive);
      state.revision = res.revision;
      setSaveState(state.dirty ? "unsaved" : "saved");
    } catch (e) {
      if (e.status === 409) {
        state.conflict = true;
        banner.hidden = false;
        setSaveState("error");
      } else if (e.status === 422) {
        // A structural problem the server rejected; surface it, don't loop.
        setSaveState("error");
        toast(e.message);
      } else {
        state.dirty = true;
        setSaveState("error");
        state.saveTimer = setTimeout(save, 3000);
      }
    } finally {
      state.saving = false;
    }
  }

  async function flushSave() {
    clearTimeout(state.saveTimer);
    while (state.saving) await new Promise((r) => setTimeout(r, 50));
    if (state.dirty) await save();
  }

  async function reloadLatest() {
    try {
      const doc = await api.document(job.id);
      state.doc = doc;
      state.revision = doc.revision;
      state.conflict = false;
      state.dirty = false;
      state.undo = [];
      state.redo = [];
      banner.hidden = true;
      setSaveState("saved");
      updateHistoryButtons();
      renderDoc();
    } catch {
      toast(t("err.generic"));
    }
  }

  // ── output ──────────────────────────────────────────────────────────
  function plainText() {
    return state.doc.blocks.map((b) => {
      if (TEXT_TYPES.has(b.type)) return b.text;
      if (b.type === "table") return b.rows.map((r) => r.join("\t")).join("\n");
      return "";
    }).filter((s) => s.trim()).join("\n\n");
  }

  async function copyAll() {
    if (await copyText(plainText())) toast(t("ed.copied"));
  }

  async function download(format) {
    await flushSave();
    const a = h("a", { href: api.exportUrl(job.id, format), download: "" });
    document.body.append(a);
    a.click();
    a.remove();
  }

  // ── keyboard ────────────────────────────────────────────────────────
  function onKey(e) {
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) {
      if (e.key === "Escape") closeMenu();
      return;
    }
    const k = e.key.toLowerCase();
    if (k === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
    else if ((k === "z" && e.shiftKey) || k === "y") { e.preventDefault(); redo(); }
    else if (k === "s") { e.preventDefault(); flushSave(); }
  }
  document.addEventListener("keydown", onKey);

  function onHide() { if (document.visibilityState === "hidden" && state.dirty) save(true); }
  document.addEventListener("visibilitychange", onHide);

  // ── mount ───────────────────────────────────────────────────────────
  root.replaceChildren(toolbar, banner,
    h("div", { class: "wrap" }, h("div", { class: "split reveal" }, viewer, docCol)));
  renderDoc();
  showPage(state.page);
  updateHistoryButtons();

  return () => {
    document.removeEventListener("click", onDocClick);
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("visibilitychange", onHide);
    if (state.dirty) save(true);
  };
}
