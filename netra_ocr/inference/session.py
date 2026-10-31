"""ONNX Runtime sessions: device -> execution providers, and a session wrapper
that can keep tensors on the GPU between calls.

Devices:
    "cpu"                 CPU
    "cuda" / "cuda:N"     NVIDIA GPU (onnxruntime-gpu)
    "tensorrt[:N]"        TensorRT, falling back to CUDA then CPU per op
    "openvino"            Intel CPU/iGPU (onnxruntime-openvino)
    "auto"                cuda if usable, else cpu (tensorrt is never picked automatically)
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GPU_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
_dlls_preloaded = False


def _ort():
    try:
        import onnxruntime
    except ImportError as e:
        raise ImportError(
            "Netra OCR needs ONNX Runtime. Install exactly one of:\n"
            "  pip install 'netra-ocr[cpu]'   # CPU\n"
            "  pip install 'netra-ocr[gpu]'   # NVIDIA GPU (+ CPU)"
        ) from e
    # Environment-level messages (e.g. provider discovery notes) are noise for
    # users; real provider failures are detected and reported by Session.
    onnxruntime.set_default_logger_severity(3)
    return onnxruntime


def available_devices() -> list[str]:
    providers = set(_ort().get_available_providers())
    devices = ["cpu"]
    if "CUDAExecutionProvider" in providers:
        devices.append("cuda")
    if "TensorrtExecutionProvider" in providers:
        devices.append("tensorrt")
    if "OpenVINOExecutionProvider" in providers:
        devices.append("openvino")
    return devices


def _split_device(device: str) -> tuple[str, int]:
    kind, _, index = device.partition(":")
    return kind, int(index) if index else 0


def resolve_device(device: str | None) -> str:
    """"auto" / None ($NETRA_DEVICE, else auto) -> a concrete, available device."""
    device = device or os.environ.get("NETRA_DEVICE") or "auto"
    if device == "auto":
        return "cuda" if "cuda" in available_devices() else "cpu"
    kind, _ = _split_device(device)
    if kind not in ("cpu", "cuda", "tensorrt", "openvino"):
        raise ValueError(f"Unknown device '{device}'")
    if kind not in available_devices():
        raise RuntimeError(f"Device '{device}' is not available with the installed ONNX Runtime "
                           f"(available: {available_devices()})")
    return device


def device_kind(device: str) -> str:
    """"cuda:1" -> "cuda"."""
    return _split_device(device)[0]


def providers_for(device: str, cache_dir: Path | None = None) -> list:
    kind, index = _split_device(device)
    # Input shapes vary per call (image sizes, sequence lengths); the default
    # exhaustive cuDNN search re-benchmarks every new shape, which costs more
    # than it saves.
    cuda = ("CUDAExecutionProvider", {"device_id": index, "cudnn_conv_algo_search": "HEURISTIC"})
    if kind == "cpu":
        return ["CPUExecutionProvider"]
    if kind == "cuda":
        return [cuda, "CPUExecutionProvider"]
    if kind == "tensorrt":
        trt = {"device_id": index, "trt_fp16_enable": True}
        if cache_dir is not None:
            # Building a TensorRT engine takes minutes; cache it next to the model.
            trt.update(trt_engine_cache_enable=True, trt_engine_cache_path=str(cache_dir),
                       trt_timing_cache_enable=True, trt_timing_cache_path=str(cache_dir))
        return [("TensorrtExecutionProvider", trt), cuda, "CPUExecutionProvider"]
    if kind == "openvino":
        return [("OpenVINOExecutionProvider", {"device_type": "CPU"}), "CPUExecutionProvider"]
    raise ValueError(f"Unknown device '{device}'")


def _preload_gpu_libraries():
    # onnxruntime-gpu can load CUDA/cuDNN from the nvidia-* pip wheels instead
    # of needing a system CUDA toolkit.
    global _dlls_preloaded
    if _dlls_preloaded:
        return
    _dlls_preloaded = True
    ort = _ort()
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # missing optional libs are reported, not fatal
            logger.debug("preload_dlls: %s", e)


class DeviceTensor:
    """An ONNX Runtime value left where the graph produced it (GPU memory on CUDA)."""

    def __init__(self, value):
        self.value = value

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.value.shape())

    def numpy(self) -> np.ndarray:
        return self.value.numpy()


class Session:
    """One ONNX graph on one device.

    ``run()`` takes numpy arrays or ``DeviceTensor``s and returns numpy arrays,
    except for outputs named in ``keep``, which come back as ``DeviceTensor``s
    that stay on the device. Feeding those back into the next ``run()`` avoids
    a host round-trip -- that's what keeps the decoder's KV cache on the GPU
    between steps.
    """

    def __init__(self, path: str | Path, device: str, threads: int | None = None):
        ort = _ort()
        kind, index = _split_device(device)
        if kind in ("cuda", "tensorrt"):
            _preload_gpu_libraries()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        # Idle worker threads busy-wait by default; with several sessions in
        # one process (detector + layout + recognizer) they steal CPU from
        # whichever session is actually running.
        if os.environ.get("NETRA_ORT_SPIN", "0") != "1":
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.log_severity_level = 3

        path = Path(path)
        self.session = ort.InferenceSession(
            str(path), sess_options=options,
            providers=providers_for(device, cache_dir=path.parent / ".trt_cache"),
        )
        active = self.session.get_providers()
        if kind in ("cuda", "tensorrt") and not _GPU_PROVIDERS & set(active):
            # ORT silently falls back to CPU when the CUDA libraries can't be
            # loaded -- say so instead of pretending to run on the GPU.
            logger.warning("%s: GPU provider failed to load, running on CPU", path.name)
            kind = "cpu"
        self.device = device if kind != "cpu" else "cpu"
        self._output_device = ("cuda", index) if kind in ("cuda", "tensorrt") else ("cpu", 0)
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, inputs: dict, keep: tuple[str, ...] = ()) -> dict:
        binding = self.session.io_binding()
        for name, value in inputs.items():
            if isinstance(value, DeviceTensor):
                binding.bind_ortvalue_input(name, value.value)
            else:
                binding.bind_cpu_input(name, np.ascontiguousarray(value))
        for name in self.output_names:
            if name in keep:
                binding.bind_output(name, self._output_device[0], self._output_device[1])
            else:
                binding.bind_output(name, "cpu")
        self.session.run_with_iobinding(binding)
        values = binding.get_outputs()
        return {
            name: DeviceTensor(value) if name in keep else value.numpy()
            for name, value in zip(self.output_names, values)
        }
