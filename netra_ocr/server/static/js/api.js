// REST client. Adds the API key when the server requires one.

function getKey() {
  try { return localStorage.getItem("netra.apiKey") || ""; } catch { return ""; }
}

export function setKey(key) {
  try { localStorage.setItem("netra.apiKey", key); } catch { /* storage unavailable */ }
}

export class ApiError extends Error {
  constructor(status, detail, body) {
    super(detail || `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

async function request(method, path, { body, headers = {}, keepalive = false } = {}) {
  const key = getKey();
  if (key) headers["X-API-Key"] = key;
  let res;
  try {
    res = await fetch(path, { method, body, headers, keepalive });
  } catch (e) {
    throw new ApiError(0, "network");
  }
  if (res.status === 401) window.dispatchEvent(new CustomEvent("netra:auth"));
  const type = res.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await res.json().catch(() => null) : null;
  if (!res.ok) {
    const detail = data && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    throw new ApiError(res.status, detail, data);
  }
  return data;
}

// For <img src> / downloads, which can't send headers.
export function withKey(url) {
  const key = getKey();
  return key ? `${url}${url.includes("?") ? "&" : "?"}key=${encodeURIComponent(key)}` : url;
}

export const api = {
  info: () => request("GET", "/v1/info"),
  jobs: () => request("GET", "/v1/jobs?limit=50"),
  job: (id) => request("GET", `/v1/jobs/${id}`),
  deleteJob: (id) => request("DELETE", `/v1/jobs/${id}`),
  upload: (form) => request("POST", "/v1/ocr", { body: form }),
  document: (id) => request("GET", `/v1/jobs/${id}/document`),
  saveDocument: (id, doc, revision, keepalive = false) => request("PUT", `/v1/jobs/${id}/document`, {
    body: JSON.stringify(doc), keepalive,
    headers: { "Content-Type": "application/json", "If-Match": `"${revision}"` },
  }),
  resetDocument: (id) => request("POST", `/v1/jobs/${id}/document/reset`),
  assetUrl: (id, asset) => withKey(`/v1/jobs/${id}/assets/${asset}`),
  exportUrl: (id, format) => withKey(`/v1/jobs/${id}/export?format=${format}`),
};
