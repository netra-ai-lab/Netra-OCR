// Model folders (see netra_ocr/inference/hub.py): `netra-model.json` plus the
// ONNX graphs and side files it names, read from a URL, an
// "hf://<org>/<repo>/<folder>[@revision]" path, or any `readFile(name)` function
// (node:fs in tests, files the user dropped into the page).

import { Session } from "./session.js";

export const CONFIG_NAME = "netra-model.json";

/** "hf://<org>/<repo>[/<folder>][@revision]" -> the folder's file URL prefix on huggingface.co. */
export function resolveSource(source) {
  if (!source.startsWith("hf://")) return source;
  const [path, revision = "main"] = source.slice("hf://".length).split("@");
  const [org, repo, ...folder] = path.split("/").filter(Boolean);
  return `https://huggingface.co/${org}/${repo}/resolve/${revision}/${folder.length ? folder.join("/") + "/" : ""}`;
}

function urlReader(base) {
  const root = base.endsWith("/") ? base : `${base}/`;
  return async (name) => {
    const response = await fetch(new URL(name, new URL(root, globalThis.location?.href)));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return response.arrayBuffer();
  };
}

export class ModelFolder {
  constructor(config, readFile) {
    this.config = config;
    this.readFile = readFile;
    this.name = `${config.name} ${config.version}`;
    this.settings = config.components;
  }

  async bytes(name) {
    if (name.includes("..") || name.startsWith("/")) throw new Error(`'${name}' points outside the model folder`);
    return new Uint8Array(await this.readFile(name));
  }

  async json(name) {
    return JSON.parse(new TextDecoder().decode(await this.bytes(name)));
  }

  /** The first variant the config prefers on `device`, else null (the fp32 graphs). */
  pickVariant(device) {
    for (const [name, variant] of Object.entries(this.config.variants ?? {})) {
      if ((variant.devices ?? []).includes(device)) return name;
    }
    return null;
  }

  graphFiles(variant = null) {
    const graphs = { ...this.config.graphs };
    if (variant !== null) {
      if (!(variant in (this.config.variants ?? {}))) throw new Error(`No variant '${variant}' in ${this.name}`);
      Object.assign(graphs, this.config.variants[variant].graphs ?? {});
    }
    return graphs;
  }

  /** One onnxruntime-web session per graph; `keep[role]` lists outputs to leave on the GPU. */
  async sessions(ort, device, variant, keep = {}) {
    const out = {};
    for (const [role, file] of Object.entries(this.graphFiles(variant))) {
      out[role] = await Session.create(ort, await this.bytes(file), device, keep[role] ?? []);
    }
    return out;
  }
}

export async function openModel(source, { readFile } = {}) {
  const reader = readFile ?? urlReader(resolveSource(source));
  const config = JSON.parse(new TextDecoder().decode(new Uint8Array(await reader(CONFIG_NAME))));
  for (const key of ["name", "version", "graphs", "components"]) {
    if (!(key in config)) throw new Error(`${CONFIG_NAME} is missing '${key}'`);
  }
  return new ModelFolder(config, reader);
}
