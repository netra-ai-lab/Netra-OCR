// onnxruntime-web sessions. Devices: "webgpu", "wasm", or "auto" (WebGPU when
// the browser exposes it, else WebAssembly).

export function resolveDevice(device = "auto") {
  if (device === "auto") return globalThis.navigator?.gpu ? "webgpu" : "wasm";
  if (device !== "webgpu" && device !== "wasm") throw new Error(`Unknown device '${device}' (webgpu, wasm, auto)`);
  return device;
}

export class Session {
  /**
   * @param ort       the onnxruntime-web module
   * @param keep      output names to leave in GPU memory on WebGPU (e.g. a KV
   *                  cache fed straight back into the next run)
   */
  static async create(ort, bytes, device, keep = []) {
    const options = { executionProviders: [device], graphOptimizationLevel: "all" };
    if (device === "webgpu" && keep.length) {
      options.preferredOutputLocation = Object.fromEntries(keep.map((name) => [name, "gpu-buffer"]));
    }
    const session = await ort.InferenceSession.create(bytes, options);
    return new Session(ort, session, device);
  }

  constructor(ort, session, device) {
    this.ort = ort;
    this.session = session;
    this.device = device;
    this.inputNames = session.inputNames;
    this.outputNames = session.outputNames;
  }

  tensor(type, data, dims) {
    return new this.ort.Tensor(type, data, dims);
  }

  run(feeds) {
    return this.session.run(feeds);
  }
}

/** Free a GPU-resident tensor (no-op for CPU tensors). */
export function release(tensor) {
  if (tensor?.location === "gpu-buffer") tensor.dispose();
}
