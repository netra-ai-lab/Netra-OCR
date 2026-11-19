import { api } from "./api.js";
import { num, t } from "./i18n.js";
import { copyText, h, icon, iconButton, isKhmer, toast } from "./util.js";

const TEXT_TYPES = new Set(["heading", "paragraph", "caption", "page_header", "page_footer"]);
const TYPE_CHOICES = ["heading-1", "heading-2", "heading-3", "paragraph", "caption", "page_header", "page_footer"];
// "docx" rebuilds the original page; "docx-reflow" is the clean, re-flowed Word document.
const FORMATS = ["docx", "docx-reflow", "html", "md", "json", "txt"];
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

// ── the Word layout ─────────────────────────────────────────────────────
// Sheets are drawn from the same page plan the original-layout Word export
// follows (GET /v1/jobs/{id}/layout, in points): exact line pitch and space
// before, indents, the scan's line breaks with each line's character scale,
// tables with the scan's column widths and row heights, side-by-side
// columns, and logos and seals floating behind the text. Text is set in
// the faces the .docx embeds (Moul, Noto Sans/Serif Khmer) with Arial for
// Latin, as in the .docx. A block you edit is reflowed, as Word does.

const DOC_FONTS = { muol: '"Doc Moul"', sans: '"Doc Noto Sans Khmer"', serif: '"Doc Noto Serif Khmer"' };
const LATIN_FONT = 'Arial, "Liberation Sans", Helvetica, sans-serif';
const DEFAULT_STYLE = { font: "sans", size: 11, bold: false, color: "000000" };
const TABLE_RULE = "rgb(128, 128, 128)";      // the .docx grid colour

const squash = (s) => (s || "").replace(/\s+/g, "");
const sum = (xs) => xs.reduce((a, b) => a + b, 0);

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Text style of a block or cell, as CSS (sizes through ``u``, pt -> cqw). */
function fontCss(st, u) {
  const colour = /^[0-9a-fA-F]{6}$/.test(st.color || "") ? st.color : "000000";
  return `font-family:${DOC_FONTS[st.font] || DOC_FONTS.sans}, ${LATIN_FONT};font-size:${u(st.size)};`
    + `font-weight:${st.bold ? 700 : 400};color:#${colour}`;
}

/** A plain layout for a page the server couldn't plan: blocks in order. */
function fallbackPage(page, blocks) {
  const W = 595.28;
  const H = page.width && page.height ? (W * page.height) / page.width : 841.89;
  const m = 56;
  const bands = [];
  for (const b of blocks) {
    if (TEXT_TYPES.has(b.type)) {
      bands.push({ space_before: 6, columns: [], item: { kind: "para", id: b.id, style: DEFAULT_STYLE,
        align: b.align || "left", pitch: 17, keep_lines: false, text: b.text, rows: [],
        left: 0, right: 0, first: 0, space_before: 0 } });
    } else if (b.type === "table") {
      const n = Math.max(1, ...b.rows.map((r) => r.length));
      bands.push({ space_before: 6, columns: [], item: { kind: "table", id: b.id, style: DEFAULT_STYLE,
        header_style: null, col_widths: Array(n).fill((W - 2 * m) / n), row_heights: b.rows.map(() => 18),
        left: m, borders: true, padding: 2, pitch: 15, cells: [] } });
    }
  }
  return { index: page.index, width: W, height: H, margins: [m, m, m, m], bands, floats: [] };
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
      // The layout first: it measures pages that haven't been, and the
      // document fetched after it carries those measurements.
      let layout = null;
      renderPreparing(job);
      try {
        layout = await api.layout(jobId);
      } catch { /* drawn without it: blocks in order */ }
      if (disposed) return;
      try {
        const doc = await api.document(jobId);
        if (!disposed) cleanupEditor = renderEditor(root, job, doc, layout);
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
      h("p", { class: "label" }, t("status.error")),
      h("h1", { class: "h2" }, t("proc.missing")),
      h("p", { class: "note" }, h("a", { href: "#/" }, `← ${t("proc.back")}`)))));
  }

  // Documents read before pages were measured at OCR time are measured on
  // first open: a few seconds a page.
  function renderPreparing(job) {
    progressEls = null;
    root.replaceChildren(h("section", { class: "wrap" }, h("div", { class: "processing reveal" },
      h("p", { class: "label" }, t("proc.layoutLabel")), h("h1", { class: "h2" }, job.filename),
      h("div", { class: "bar indeterminate" }, h("span")),
      h("p", { class: "meta" }, t("proc.layout")))));
  }

  let progressEls = null;
  function renderProgress(job) {
    if (!progressEls) {
      progressEls = {
        eyebrow: h("p", { class: "label" }),
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

function renderEditor(root, job, initialDoc, initialLayout) {
  const state = {
    doc: initialDoc,
    layout: initialLayout?.pages || null,
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
      h("span", {}, t(`fmt.${f}`)), h("span", { class: "meta" }, `.${f.split("-")[0]}`))));
  const moreBtn = h("button", { class: "btn btn-sm", type: "button", "aria-haspopup": "menu", "aria-expanded": "false",
    onclick: (e) => { e.stopPropagation(); menuList.hidden ? openMenu() : closeMenu(); } }, t("ed.more"), icon("chevDown"));

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
    h("button", { class: "btn btn-sm", type: "button", onclick: copyAll }, icon("copy"), t("ed.copy")),
    h("div", { class: "menu" }, moreBtn, menuList),
    h("button", { class: "btn btn-primary btn-sm", type: "button", onclick: () => download("docx") }, t("ed.download"))),
    // Tools for the selected block live here rather than floating over the
    // text, so they never cover the block above.
    contextRow);

  const banner = h("div", { class: "banner", hidden: true }, h("div", { class: "banner-inner" },
    h("span", {}, t("ed.conflict")),
    h("button", { class: "btn btn-sm", type: "button", onclick: reloadLatest }, t("ed.reload"))));

  // ── pages ───────────────────────────────────────────────────────────
  // One row per page: the scan on the left and its rebuilt sheet on the
  // right, at the same scale, so a single scroll moves both and every line
  // sits level with the line it came from.
  const docEl = h("div", { class: "pages" });
  const hint = h("p", { class: "doc-hint meta" }, t("ed.hint"));
  const addBtn = h("button", { class: "btn btn-sm add-block", type: "button", onclick: () => addParagraph() },
    icon("plus"), t("ed.addParagraph"));
  const scans = new Map();          // page index -> {el, inner, overlay}

  function blockIndex(id) { return state.doc.blocks.findIndex((b) => b.id === id); }
  function blockById(id) { return state.doc.blocks.find((b) => b.id === id); }

  function pageLayout(index) {
    return state.layout?.[index] || fallbackPage(pages[index] || {}, state.doc.blocks.filter((b) => b.page === index));
  }

  /** pt -> cqw for a sheet ``W`` points wide. */
  function unit(W) {
    return (v) => `${((v / W) * 100).toFixed(4)}cqw`;
  }

  /** Where the scan sits on the sheet: centred when it was fitted by height. */
  function scanPlacement(page, lp) {
    const dpi = page.dpi || (page.width ? (page.width * 72) / lp.width : 72);
    const w = ((page.width || 0) * 72) / dpi;
    return { k: 72 / dpi, ox: Math.max(0, (lp.width - w) / 2), w, h: ((page.height || 0) * 72) / dpi };
  }

  function scanPanel(index, lp) {
    let s = scans.get(index);
    const page = pages[index] || {};
    const u = unit(lp.width);
    const at = scanPlacement(page, lp);
    if (!s) {
      const img = h("img", { src: page.image ? api.assetUrl(job.id, page.image) : "", alt: t("ed.original"),
        draggable: "false", loading: index > 1 ? "lazy" : "eager" });
      const overlay = h("div", { class: "overlay" });
      const inner = h("div", { class: "scan-inner" }, img, overlay);
      s = { el: h("div", { class: "scan" }, inner), inner, img, overlay };
      scans.set(index, s);
    }
    s.inner.style.height = u(lp.height);
    s.img.style.cssText = `left:${u(at.ox)};width:${u(at.w)};height:${u(at.h)}`;
    s.overlay.replaceChildren(...state.doc.blocks
      .filter((b) => b.page === index && b.bbox)
      .map((b) => {
        const [x1, y1, x2, y2] = b.bbox;
        return h("button", {
          class: `ov${b.id === state.activeId ? " active" : ""}`, type: "button", dataset: { id: b.id, type: b.type },
          style: `left:${u(at.ox + x1 * at.k)};top:${u(y1 * at.k)};width:${u((x2 - x1) * at.k)};height:${u((y2 - y1) * at.k)}`,
          "aria-label": t(`type.${typeKey(b)}`),
          onclick: () => focusBlock(b.id, true),
        }, h("span", { class: "ov-tag" }, t(`type.${typeKey(b)}`)));
      }));
    return s.el;
  }

  // Rendered pages; mutate() and saves redraw them.
  function renderDoc() {
    const rows = pages.map((page, i) => {
      const lp = pageLayout(i);
      const label = pages.length > 1 ? t("ed.page", { n: num(i + 1), total: num(pages.length) }) : null;
      return h("section", { class: "page-row", dataset: { page: i },
        "aria-label": t("ed.pageLabel", { n: num(i + 1) }) },
      h("div", { class: "page-col" }, label ? h("p", { class: "label page-label" }, label) : null, scanPanel(i, lp)),
      h("div", { class: "page-col" }, label ? h("p", { class: "label page-label" }, " ") : null, renderSheet(i, lp)));
    });
    if (!state.doc.blocks.length) rows.push(h("p", { class: "empty" }, t("ed.empty")));
    docEl.replaceChildren(...rows, h("div", { class: "doc-foot" }, addBtn, hint));
    renderContext();
  }

  function renderSheet(index, lp) {
    const u = unit(lp.width);
    const [ml, mr, mt] = lp.margins;
    const contentW = lp.width - ml - mr;
    const placed = new Map();       // block id -> its top-level element in the flow
    const flow = h("div", { class: "flow", style: `padding:${u(mt)} ${u(mr)} 0 ${u(ml)}` });

    // A tall paragraph can be planned as parts (its top lines beside a
    // letterhead, the rest below): each part draws its own lines, and the
    // block is unedited while all its parts' lines still make its text.
    const scanned = new Map();      // block id -> its parts' lines, joined
    for (const band of lp.bands) {
      for (const it of band.item ? [band.item] : band.columns.flatMap((c) => c.items)) {
        if (it.kind === "para" && it.keep_lines) {
          scanned.set(it.id, (scanned.get(it.id) || "") + squash(it.rows.map((r) => r.text).join("")));
        }
      }
    }
    const drawn = new Set();

    function item(it, width, offset) {
      const b = blockById(it.id);
      if (!b || b.page !== index) return null;      // deleted, merged or moved since the plan
      if (it.kind === "table") return b.type === "table" ? renderTable(b, it, u, offset) : null;
      if (!TEXT_TYPES.has(b.type)) return null;
      const unedited = it.keep_lines && it.rows.length && scanned.get(b.id) === squash(b.text);
      if (!unedited && drawn.has(b.id)) return null;   // an edited block reflows once, in its first part
      drawn.add(b.id);
      return renderPara(b, it, u, width, unedited);
    }

    for (const band of lp.bands) {
      let el = null;
      const ids = [];
      if (band.item) {
        el = item(band.item, contentW, ml);
        ids.push(band.item.id);
      } else {
        let x = ml;
        const cols = band.columns.map((c) => {
          const col = h("div", { class: "col", style: `margin-left:${u(c.x1 - x)};width:${u(c.x2 - c.x1)}` });
          x = c.x2;
          for (const it of c.items) {
            const child = item(it, c.x2 - c.x1, c.x1);
            if (child) {
              child.style.marginTop = u(it.space_before);
              col.append(child);
            }
            ids.push(it.id);
          }
          return col;
        });
        el = h("div", { class: "cols" }, cols);
      }
      if (!el) continue;
      el.style.marginTop = u(band.space_before);
      flow.append(el);
      ids.forEach((id) => placed.set(id, el));
    }

    // Blocks the plan doesn't have yet (just added, split or moved here):
    // in the flow right after the block before them, until the next plan.
    const body = { style: { ...DEFAULT_STYLE, font: pages[index]?.body_font || "sans",
      size: pages[index]?.body_size || DEFAULT_STYLE.size } };
    let prev = null;
    for (const b of state.doc.blocks.filter((x) => x.page === index)) {
      if (!placed.has(b.id) && (TEXT_TYPES.has(b.type) || b.type === "table")) {
        const it = b.type === "table" ? fallbackPage(pages[index] || {}, [b]).bands[0].item
          : { kind: "para", id: b.id, style: body.style, align: b.align || "left",
            pitch: body.style.size * 1.75, keep_lines: false, text: b.text, rows: [], left: 0, right: 0, first: 0 };
        const el = item(it, contentW, ml);
        if (el) {
          const after = prev && placed.get(prev.id);
          if (after) after.after(el); else flow.prepend(el);
          placed.set(b.id, el);
        }
      }
      if (placed.has(b.id)) prev = b;
    }

    const floats = h("div", { class: "floats", "aria-hidden": "true" }, lp.floats.map((f) => {
      const fig = f.block_id && blockById(f.block_id);
      if (f.block_id && !fig) return null;          // a deleted logo
      return h("img", {
        class: `float${fig ? " float-fig" : ""}${fig && fig.id === state.activeId ? " active" : ""}`,
        src: api.assetUrl(job.id, f.image), alt: "", loading: "lazy", draggable: "false",
        dataset: fig ? { id: fig.id } : {},
        style: `left:${u(f.x)};top:${u(f.y)};width:${u(f.width)};height:${u(f.height)}`,
        onclick: fig ? () => focusBlock(fig.id) : null,
      });
    }).filter(Boolean));

    return h("div", { class: "sheet" },
      h("div", { class: "sheet-inner", style: `min-height:${u(lp.height)}` }, floats, flow));
  }

  // A paragraph keeps the scan's lines (each line's character scale too)
  // until it is edited; while you type, and after, it reflows like Word.
  function renderPara(b, it, u, width, keep) {
    const st = it.style || DEFAULT_STYLE;
    const key = typeKey(b);
    const align = b.align && b.align !== "left" && it.align !== b.align ? b.align : it.align || b.align || "left";
    const el = h("div", {
      class: `block para b-${key}${b.id === state.activeId ? " active" : ""}`,
      dataset: { id: b.id },
      style: `${fontCss(st, u)};line-height:${u(it.pitch)}`,
    });
    const tag = h("span", { class: "block-tag", "aria-hidden": "true" }, t(`type.${key}`));
    const ed = editable(b.text, (text) => { b.text = text; });
    ed.dataset.placeholder = t("ed.placeholder");
    ed.style.cssText = `padding-left:${u(it.left)};padding-right:${u(it.right)};text-indent:${u(it.first)};`
      + `text-align:${align}`;
    el.append(tag, ed);

    if (keep) {
      const view = h("div", { class: "para-view", tabindex: "0" }, it.rows.map((r, i) => {
        const s = Math.max(0.3, (r.scale || 100) / 100);
        const indent = it.left + (i === 0 ? it.first : 0);
        const room = Math.max(1, width - indent - it.right);
        const last = i === it.rows.length - 1;
        const rowAlign = align === "justify" ? (last ? "left" : "justify") : align;
        return h("div", { class: "line", style: `height:${u(it.pitch)};margin-left:${u(indent)};`
          + `width:${u(room / s)};transform:scaleX(${s});text-align:${rowAlign};`
          + `text-align-last:${rowAlign}` }, r.text);
      }));
      ed.hidden = true;
      // Editing starts where you click: the caret goes to the same letter
      // in the block's text (whitespace aside: lines join without breaks).
      view.addEventListener("mousedown", (e) => {
        e.preventDefault();
        startEditing(view, ed, b, it, caretInView(view, e.clientX, e.clientY));
      });
      view.addEventListener("focus", () => setActive(b.id));
      view.addEventListener("keydown", (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey || e.key === "Tab" || e.key === "Shift") return;
        if (e.key === "Enter") e.preventDefault();
        startEditing(view, ed, b, it, null);
      });
      const original = b.text;
      ed.addEventListener("focusout", () => {
        // Nothing changed: back to the scan's lines.
        if (squash(b.text) === squash(original)) {
          ed.hidden = true;
          view.hidden = false;
        }
      });
      el.append(view);
    }
    el.addEventListener("focusin", () => setActive(b.id));
    el.addEventListener("click", () => setActive(b.id));
    return el;
  }

  /** Letters (not counting whitespace) before the point clicked in a view. */
  function caretInView(view, x, y) {
    const pos = document.caretPositionFromPoint?.(x, y);
    const range = pos ? null : document.caretRangeFromPoint?.(x, y);
    const node = pos ? pos.offsetNode : range?.startContainer;
    const offset = pos ? pos.offset : range?.startOffset;
    const line = node && (node.nodeType === 3 ? node.parentElement : node)?.closest(".line");
    if (!line || !view.contains(line)) return null;
    let count = 0;
    for (const l of view.querySelectorAll(".line")) {
      if (l === line) return count + squash(l.textContent.slice(0, offset)).length;
      count += squash(l.textContent).length;
    }
    return count;
  }

  function startEditing(view, ed, b, it, letters) {
    view.hidden = true;
    ed.hidden = false;
    ed.focus();
    const text = ed.firstChild;
    let at = text ? text.length : 0;
    if (letters != null && text) {
      let seen = 0;
      at = 0;
      while (at < text.length && seen < letters) {
        if (!/\s/.test(text.data[at])) seen++;
        at++;
      }
    }
    const sel = window.getSelection();
    const r = document.createRange();
    if (text) r.setStart(text, at); else r.setStart(ed, 0);
    r.collapse(true);
    sel.removeAllRanges();
    sel.addRange(r);
  }

  // A table with the scan's column widths and exact row heights; each
  // cell's text keeps its alignment and is narrowed to fit, as in Word.
  function renderTable(b, it, u, offset) {
    const nc = Math.max(1, ...b.rows.map((r) => r.length));
    let widths = it.col_widths || [];
    if (widths.length !== nc) widths = Array(nc).fill(sum(widths.length ? widths : [400]) / nc);
    const med = median(it.row_heights || []) || 18;
    const heights = b.rows.map((_, i) => (it.row_heights || [])[i] ?? med);
    const pad = it.padding ?? 2;
    const rule = it.borders ? `1px solid ${TABLE_RULE}` : "0";
    const el = h("div", {
      class: `block tbl b-table${b.id === state.activeId ? " active" : ""}`,
      dataset: { id: b.id },
      style: `margin-left:${u(Math.max(0, it.left - offset))};width:${u(sum(widths))}`,
    });
    const table = h("table", { style: `width:${u(sum(widths))}` },
      h("colgroup", {}, widths.map((w) => h("col", { style: `width:${u(w)}` }))),
      h("tbody", {}, b.rows.map((row, r) => {
        const header = r === 0 && it.header_style;
        const st = header ? it.header_style : it.style || DEFAULT_STYLE;
        const natural = (header ? it.header_pitch : it.pitch) || st.size * 1.4;
        return h("tr", { style: `height:${u(heights[r])}` }, Array.from({ length: nc }, (_, c) => {
          const meta = it.cells?.[r]?.[c] || { align: "left", scale: 100, valign: "center" };
          const text = row[c] ?? "";
          const n = Math.max(1, text.split("\n").length);
          const pitch = Math.max(st.size * 1.05, Math.min(natural, heights[r] / n));
          const s = Math.max(0.3, (meta.scale || 100) / 100);
          const room = Math.max(1, widths[c] - 2 * pad);
          const ed = editable(text, (v) => { b.rows[r][c] = v; });
          ed.style.cssText = `width:${u(room / s)};transform:scaleX(${s});text-align:${meta.align};`
            + `line-height:${u(pitch)}`;
          ed.addEventListener("focus", () => { state.cell = { r, c }; });
          return h("td", { style: `${fontCss(st, u)};padding:0 ${u(pad)};border:${rule};`
            + `vertical-align:${meta.valign === "center" ? "middle" : meta.valign}` }, ed);
        }));
      })));
    el.append(h("span", { class: "block-tag", "aria-hidden": "true" }, t("type.table")), table);
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
      ? [h("span", { class: "label" }, t(`type.${typeKey(b)}`)), renderTools(b)]
      : [h("span", { class: "meta muted2" }, t("ed.hint"))])));
  }

  // The selected block is marked on the sheet and on the scan beside it.
  function setActive(id) {
    if (state.activeId === id) return;
    docEl.querySelectorAll(".active").forEach((el) => el.classList.remove("active"));
    state.activeId = id;
    const b = blockById(id);
    if (b) {
      state.page = b.page;
      docEl.querySelectorAll(`[data-id="${CSS.escape(id)}"]`).forEach((el) => el.classList.add("active"));
    }
    renderContext();
  }

  function focusBlock(id, fromOverlay = false) {
    setActive(id);
    const el = docEl.querySelector(`.block[data-id="${CSS.escape(id)}"], .float-fig[data-id="${CSS.escape(id)}"]`);
    if (!el) return;
    if (fromOverlay) el.scrollIntoView({ block: "nearest" });
    const view = el.querySelector(".para-view:not([hidden])");
    const target = view || el.querySelector(".editable:not([hidden])");
    target?.focus({ preventScroll: true });
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
      scheduleLayout();
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
      if (!keepalive) scheduleLayout();
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

  // After a save, ask for the plan again (what the .docx now looks like)
  // and redraw -- but never under the caret: a block being typed in is
  // redrawn once it loses focus.
  let layoutTimer = null;
  let layoutPending = false;
  function scheduleLayout() {
    clearTimeout(layoutTimer);
    layoutTimer = setTimeout(async () => {
      try {
        const res = await api.layout(job.id);
        if (res.revision !== state.revision || state.dirty) return;   // a newer save will follow
        state.layout = res.pages;
        if (typing()) layoutPending = true;
        else renderDoc();
      } catch { /* keep the last plan */ }
    }, 250);
  }
  function typing() {
    const a = document.activeElement;
    return !!(a && a.classList?.contains("editable") && docEl.contains(a));
  }
  docEl.addEventListener("focusout", () => {
    setTimeout(() => {
      if (layoutPending && !typing()) {
        layoutPending = false;
        const active = state.activeId;
        renderDoc();
        if (active) docEl.querySelectorAll(`[data-id="${CSS.escape(active)}"]`).forEach((el) => el.classList.add("active"));
      }
    }, 0);
  });

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
      scheduleLayout();
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
    const [fmt, layout] = format.split("-");
    const a = h("a", { href: api.exportUrl(job.id, fmt, layout), download: "" });
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
    h("div", { class: "wrap editor reveal" },
      h("div", { class: "page-row page-heads" },
        h("p", { class: "label" }, t("ed.original")), h("p", { class: "label" }, t("ed.result"))),
      docEl));
  renderDoc();
  updateHistoryButtons();

  return () => {
    document.removeEventListener("click", onDocClick);
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("visibilitychange", onHide);
    if (state.dirty) save(true);
  };
}
