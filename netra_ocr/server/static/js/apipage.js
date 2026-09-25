import { t } from "./i18n.js";
import { h } from "./util.js";

function section(n, title, text, ...content) {
  return h("section", { class: "rail" },
    h("div", {}, h("div", { class: "rail-label eyebrow" },
      h("span", { class: "num" }, String(n).padStart(2, "0")), h("span", { class: "slash" }, "/"),
      h("span", { class: "muted2" }, title))),
    h("div", { class: "prose" }, h("p", { class: "body-text muted" }, text), ...content));
}

function code(text) {
  return h("pre", { class: "code" }, h("code", {}, text));
}

export function renderApi(root, info) {
  const origin = location.origin;
  const auth = info?.auth ? ' \\\n  -H "X-API-Key: $NETRA_API_KEY"' : "";
  const pyAuth = info?.auth ? ', headers={"X-API-Key": KEY}' : "";

  const endpoints = [
    ["POST", "/v1/ocr", "Upload a PDF or image (multipart `file`)."],
    ["GET", "/v1/jobs/{id}", "Status and page progress."],
    ["GET", "/v1/jobs/{id}/document", "The structured document: pages + blocks."],
    ["PUT", "/v1/jobs/{id}/document", "Save edits (send If-Match: revision)."],
    ["GET", "/v1/jobs/{id}/export?format=docx", "docx · html · md · json · txt"],
    ["GET", "/v1/jobs/{id}/assets/{asset}", "Page renders and figure crops."],
    ["DELETE", "/v1/jobs/{id}", "Delete the job and its files."],
    ["GET", "/v1/info", "Device, limits, available detectors/decoders."],
  ];

  root.replaceChildren(h("div", { class: "wrap" },
    h("header", { class: "page page-head api-head reveal" },
      h("h1", { class: "h1" }, t("api.title")),
      h("p", { class: "lede muted" }, t("api.lede"))),
    section(1, t("api.s1"), t("api.s1.text"),
      code(`curl -F file=@scan.jpg${auth} \\\n  ${origin}/v1/ocr\n\n# PDF: returns 202 + job id; poll until "status": "done"\ncurl -F file=@letter.pdf${auth} ${origin}/v1/ocr\ncurl${auth.replace(" \\\n ", "")} ${origin}/v1/jobs/<id>`)),
    section(2, t("api.s2"), t("api.s2.text"),
      code(`curl -OJ${auth.replace(" \\\n ", "")} "${origin}/v1/jobs/<id>/export?format=docx"`)),
    section(3, t("api.s3"), t("api.s3.text"),
      code(`import time, requests

URL = "${origin}"${info?.auth ? '\nKEY = "..."' : ""}

with open("letter.pdf", "rb") as f:
    job = requests.post(f"{URL}/v1/ocr", files={"file": f}${pyAuth}).json()["job"]

while job["status"] not in ("done", "error"):
    time.sleep(1)
    job = requests.get(f"{URL}/v1/jobs/{job['id']}"${pyAuth}).json()["job"]

doc = requests.get(f"{URL}/v1/jobs/{job['id']}/document"${pyAuth}).json()
for block in doc["blocks"]:
    print(block["type"], block.get("text", ""))

docx = requests.get(f"{URL}/v1/jobs/{job['id']}/export", params={"format": "docx"}${pyAuth})
open("letter.docx", "wb").write(docx.content)`)),
    section(4, t("api.s4"), t("api.s4.text"),
      h("p", {}, h("a", { href: "/docs", target: "_blank", rel: "noopener" }, `${origin}/docs`)),
      h("div", {}, endpoints.map(([verb, path, desc]) => h("div", { class: "endpoint" },
        h("span", { class: "verb" }, verb),
        h("div", {}, h("code", {}, path), h("p", { class: "meta muted", style: "margin-top:6px" }, desc))))))));
  return null;
}
