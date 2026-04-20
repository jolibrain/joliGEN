import argparse
import ctypes
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

JG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(JG_DIR)

from data.online_creation import crop_image, fill_mask_with_color, fill_mask_with_random


def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def prepare_tensors(tensor_list):
    if all(tensor is not None for tensor in tensor_list):
        return torch.stack(tensor_list, dim=0).permute(1, 0, 2, 3, 4)
    return None


def separate_tensors(tensor):
    if tensor is not None and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
        return [img.unsqueeze(0) for img in tensor.unbind(0)]
    return None


def import_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "tensorrt is required for TensorRT inference. "
            "Install the TensorRT Python bindings first."
        ) from exc
    return trt


class CudaRuntimeError(RuntimeError):
    pass


class CudaRuntime:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self) -> None:
        self.lib = self._load_libcudart()
        self._bind_symbols()

    @staticmethod
    def _load_libcudart() -> ctypes.CDLL:
        tried: List[str] = []
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                return ctypes.CDLL(name)
            except OSError as exc:
                tried.append(f"{name}: {exc}")
        details = "\n".join(f"  - {row}" for row in tried)
        raise RuntimeError(
            "Failed to load CUDA runtime library (libcudart).\n"
            "Tried:\n"
            f"{details}"
        )

    def _bind_symbols(self) -> None:
        self._cudaGetDeviceCount = self.lib.cudaGetDeviceCount
        self._cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._cudaGetDeviceCount.restype = ctypes.c_int

        self._cudaStreamCreate = self.lib.cudaStreamCreate
        self._cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._cudaStreamCreate.restype = ctypes.c_int

        self._cudaStreamDestroy = self.lib.cudaStreamDestroy
        self._cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._cudaStreamDestroy.restype = ctypes.c_int

        self._cudaMalloc = self.lib.cudaMalloc
        self._cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._cudaMalloc.restype = ctypes.c_int

        self._cudaFree = self.lib.cudaFree
        self._cudaFree.argtypes = [ctypes.c_void_p]
        self._cudaFree.restype = ctypes.c_int

        self._cudaMemcpyAsync = self.lib.cudaMemcpyAsync
        self._cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._cudaMemcpyAsync.restype = ctypes.c_int

        self._cudaStreamSynchronize = self.lib.cudaStreamSynchronize
        self._cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._cudaStreamSynchronize.restype = ctypes.c_int

        self._cudaGetErrorName = getattr(self.lib, "cudaGetErrorName", None)
        if self._cudaGetErrorName is not None:
            self._cudaGetErrorName.argtypes = [ctypes.c_int]
            self._cudaGetErrorName.restype = ctypes.c_char_p

        self._cudaGetErrorString = getattr(self.lib, "cudaGetErrorString", None)
        if self._cudaGetErrorString is not None:
            self._cudaGetErrorString.argtypes = [ctypes.c_int]
            self._cudaGetErrorString.restype = ctypes.c_char_p

    def _error_text(self, code: int) -> str:
        name = b""
        text = b""
        if self._cudaGetErrorName is not None:
            name = self._cudaGetErrorName(code) or b""
        if self._cudaGetErrorString is not None:
            text = self._cudaGetErrorString(code) or b""
        if name and text:
            return (
                f"{name.decode('utf-8', errors='replace')}: "
                f"{text.decode('utf-8', errors='replace')}"
            )
        if name:
            return name.decode("utf-8", errors="replace")
        if text:
            return text.decode("utf-8", errors="replace")
        return "unknown CUDA error"

    def _check(self, code: int, fn_name: str) -> None:
        if code != 0:
            raise CudaRuntimeError(
                f"{fn_name} failed with code {code} ({self._error_text(code)})"
            )

    def device_count(self) -> int:
        out = ctypes.c_int(0)
        self._check(self._cudaGetDeviceCount(ctypes.byref(out)), "cudaGetDeviceCount")
        return int(out.value)

    def stream_create(self) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        self._check(self._cudaStreamCreate(ctypes.byref(out)), "cudaStreamCreate")
        return out

    def stream_destroy(self, stream: ctypes.c_void_p) -> None:
        if stream and stream.value:
            self._check(self._cudaStreamDestroy(stream), "cudaStreamDestroy")

    def stream_synchronize(self, stream: ctypes.c_void_p) -> None:
        self._check(self._cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        size = max(int(nbytes), 1)
        out = ctypes.c_void_p()
        self._check(
            self._cudaMalloc(ctypes.byref(out), ctypes.c_size_t(size)), "cudaMalloc"
        )
        return out

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr and ptr.value:
            self._check(self._cudaFree(ptr), "cudaFree")

    def memcpy_h2d_async(
        self, dst_device: ctypes.c_void_p, src_host: np.ndarray, stream: ctypes.c_void_p
    ) -> None:
        if not src_host.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "Host source array must be C contiguous for cudaMemcpyAsync H2D"
            )
        self._check(
            self._cudaMemcpyAsync(
                dst_device,
                ctypes.c_void_p(src_host.ctypes.data),
                ctypes.c_size_t(src_host.nbytes),
                self.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def memcpy_d2h_async(
        self, dst_host: np.ndarray, src_device: ctypes.c_void_p, stream: ctypes.c_void_p
    ) -> None:
        if not dst_host.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "Host destination array must be C contiguous for cudaMemcpyAsync D2H"
            )
        self._check(
            self._cudaMemcpyAsync(
                ctypes.c_void_p(dst_host.ctypes.data),
                src_device,
                ctypes.c_size_t(dst_host.nbytes),
                self.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )


def _num_bytes(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def load_engine(engine_path: Path):
    trt = import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    return logger, runtime, engine


def is_input(engine, name: str) -> bool:
    trt = import_tensorrt()
    return engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT


def describe_engine_io(engine) -> List[Tuple[str, str, Tuple[int, ...], np.dtype]]:
    trt = import_tensorrt()
    rows: List[Tuple[str, str, Tuple[int, ...], np.dtype]] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = "INPUT" if is_input(engine, name) else "OUTPUT"
        shape = tuple(int(d) for d in engine.get_tensor_shape(name))
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        rows.append((name, mode, shape, dtype))
    return rows


def read_paths_file(paths_in_file):
    pairs = []
    with open(paths_in_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line in paths file: {line!r}")
            pairs.append((parts[0], parts[1]))
    pairs.sort(key=lambda x: natural_key(x[0]))
    return pairs


def resolve_dataset_root(paths_in_file, dataset_root):
    if dataset_root:
        return os.path.abspath(dataset_root)
    return os.path.dirname(os.path.dirname(os.path.abspath(paths_in_file)))


def load_bbox_txt(path):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid bbox line in {path}: {line!r}")
            cls, x1, y1, x2, y2 = map(int, parts[:5])
            boxes.append((cls, x1, y1, x2, y2))
    if not boxes:
        raise ValueError(f"No bbox entries found in {path}")
    return boxes


def load_train_config(model_in_file, train_config):
    if train_config is None:
        train_config = os.path.join(
            os.path.dirname(os.path.abspath(model_in_file)), "train_config.json"
        )
    with open(train_config, "r") as f:
        return json.load(f), train_config


def get_train_shape(train_json):
    data = train_json.get("data", {})
    online = data.get("online_creation", {})
    crop_size = online.get("crop_size_A", data.get("crop_size", 256))
    load_size = online.get("load_size_A", data.get("load_size", crop_size))
    temporal = data.get("temporal_number_frames", 2)

    if isinstance(crop_size, (list, tuple)):
        if len(crop_size) >= 2:
            height, width = int(crop_size[0]), int(crop_size[1])
        else:
            height = width = int(crop_size[0])
    else:
        height = width = int(crop_size)

    if isinstance(load_size, (list, tuple)):
        if len(load_size) == 0:
            load_h = load_w = int(height)
        elif len(load_size) >= 2:
            load_h, load_w = int(load_size[0]), int(load_size[1])
        else:
            load_h = load_w = int(load_size[0])
    else:
        load_h = load_w = int(load_size)

    return temporal, height, width, load_h, load_w


def resolve_denoise_steps(train_json, denoise_steps):
    if denoise_steps is not None:
        return int(denoise_steps)
    steps = train_json.get("alg", {}).get("b2b_denoise_timesteps", 2)
    if isinstance(steps, int):
        return int(steps)
    if not steps:
        raise ValueError("alg.b2b_denoise_timesteps is empty in train_config.json")
    return int(steps[0])


def get_b2b_params(train_json, image_size):
    alg = train_json.get("alg", {})
    params = {
        "P_mean": float(alg.get("b2b_P_mean", -0.8)),
        "P_std": float(alg.get("b2b_P_std", 0.8)),
        "cfg_scale": float(alg.get("b2b_cfg_scale", 1.0)),
        "clip_denoised": bool(alg.get("b2b_clip_denoised", False)),
        "disable_inference_clipping": bool(
            alg.get("b2b_disable_inference_clipping", False)
        ),
        "t_eps": float(alg.get("b2b_t_eps", 5e-2)),
        "cond_mode": alg.get("diffusion_cond_image_creation", "y_t"),
        "mask_as_channel": bool(alg.get("b2b_mask_as_channel", False)),
    }
    requested_noise_scale = float(alg.get("b2b_noise_scale", -1.0))
    if requested_noise_scale > 0:
        params["noise_scale"] = requested_noise_scale
    else:
        params["noise_scale"] = 1.0 if int(image_size) <= 256 else 2.0
    params["num_classes"] = int(
        train_json.get("G", {}).get("vit_num_classes", 1)
        or train_json.get("G", {}).get("num_classes", 1)
        or 1
    )
    params["cfg_interval"] = (0.1, 1.0)
    return params


def select_bbox(boxes, bbox_index):
    if bbox_index < 0 or bbox_index >= len(boxes):
        raise ValueError(f"bbox_index {bbox_index} out of range for {len(boxes)} boxes")
    return boxes[bbox_index]


def to_tensor(img):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )(img)


def normalize_mask_delta(mask_delta):
    if isinstance(mask_delta, (int, float)):
        return [[mask_delta]]
    if not mask_delta:
        return [[]]
    if isinstance(mask_delta[0], (int, float)):
        return [list(mask_delta)]
    return mask_delta


def compute_bbox_select(
    bbox, cls, mask_delta, mask_square, crop_coordinates, context_pixels
):
    x_crop, y_crop, crop_size = crop_coordinates
    bbox_select = list(bbox)
    mask_delta = normalize_mask_delta(mask_delta)

    if mask_delta != [[]]:
        if len(mask_delta) == 1:
            index_cls = 0
        else:
            index_cls = max(0, int(cls) - 1)
        delta_values = mask_delta[index_cls]
        if delta_values:
            if isinstance(delta_values[0], float):
                delta_x = int(delta_values[0] * max(1, bbox_select[2] - bbox_select[0]))
                if len(delta_values) == 1:
                    delta_y = delta_x
                else:
                    delta_y = int(
                        delta_values[1] * max(1, bbox_select[3] - bbox_select[1])
                    )
            else:
                delta_x = int(delta_values[0])
                delta_y = int(
                    delta_values[0] if len(delta_values) == 1 else delta_values[1]
                )
            bbox_select[0] -= delta_x
            bbox_select[1] -= delta_y
            bbox_select[2] += delta_x
            bbox_select[3] += delta_y

    if mask_square:
        sdiff = (bbox_select[2] - bbox_select[0]) - (bbox_select[3] - bbox_select[1])
        if sdiff > 0:
            bbox_select[3] += int(sdiff / 2)
            bbox_select[1] -= int(sdiff / 2)
        else:
            bbox_select[2] += -int(sdiff / 2)
            bbox_select[0] -= -int(sdiff / 2)

    bbox_select[1] += y_crop
    bbox_select[0] += x_crop
    bbox_select[3] = bbox_select[1] + crop_size
    bbox_select[2] = bbox_select[0] + crop_size
    bbox_select[1] -= context_pixels
    bbox_select[0] -= context_pixels
    bbox_select[3] += context_pixels
    bbox_select[2] += context_pixels
    return bbox_select


def compute_paste_bbox(crop_meta):
    full_x0 = crop_meta["x_crop"] - crop_meta["context_pixels"]
    full_y0 = crop_meta["y_crop"] - crop_meta["context_pixels"]
    full_x1 = crop_meta["x_crop"] + crop_meta["crop_size"] + crop_meta["context_pixels"]
    full_y1 = crop_meta["y_crop"] + crop_meta["crop_size"] + crop_meta["context_pixels"]

    orig_loaded_x0 = crop_meta["x_padding"]
    orig_loaded_y0 = crop_meta["y_padding"]
    orig_loaded_x1 = orig_loaded_x0 + crop_meta["loaded_width"]
    orig_loaded_y1 = orig_loaded_y0 + crop_meta["loaded_height"]

    inter_x0 = max(full_x0, orig_loaded_x0)
    inter_y0 = max(full_y0, orig_loaded_y0)
    inter_x1 = min(full_x1, orig_loaded_x1)
    inter_y1 = min(full_y1, orig_loaded_y1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return None

    bbox_x0 = int(
        np.floor(
            (inter_x0 - orig_loaded_x0)
            * crop_meta["orig_width"]
            / crop_meta["loaded_width"]
        )
    )
    bbox_y0 = int(
        np.floor(
            (inter_y0 - orig_loaded_y0)
            * crop_meta["orig_height"]
            / crop_meta["loaded_height"]
        )
    )
    bbox_x1 = int(
        np.ceil(
            (inter_x1 - orig_loaded_x0)
            * crop_meta["orig_width"]
            / crop_meta["loaded_width"]
        )
    )
    bbox_y1 = int(
        np.ceil(
            (inter_y1 - orig_loaded_y0)
            * crop_meta["orig_height"]
            / crop_meta["loaded_height"]
        )
    )
    return [bbox_x0, bbox_y0, bbox_x1, bbox_y1]


def compute_paste_mapping(crop_meta, src_width, src_height):
    full_x0 = crop_meta["x_crop"] - crop_meta["context_pixels"]
    full_y0 = crop_meta["y_crop"] - crop_meta["context_pixels"]
    full_x1 = crop_meta["x_crop"] + crop_meta["crop_size"] + crop_meta["context_pixels"]
    full_y1 = crop_meta["y_crop"] + crop_meta["crop_size"] + crop_meta["context_pixels"]
    full_width = full_x1 - full_x0
    full_height = full_y1 - full_y0

    orig_loaded_x0 = crop_meta["x_padding"]
    orig_loaded_y0 = crop_meta["y_padding"]
    orig_loaded_x1 = orig_loaded_x0 + crop_meta["loaded_width"]
    orig_loaded_y1 = orig_loaded_y0 + crop_meta["loaded_height"]

    inter_x0 = max(full_x0, orig_loaded_x0)
    inter_y0 = max(full_y0, orig_loaded_y0)
    inter_x1 = min(full_x1, orig_loaded_x1)
    inter_y1 = min(full_y1, orig_loaded_y1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return None

    src_x0 = int(np.floor((inter_x0 - full_x0) * src_width / full_width))
    src_y0 = int(np.floor((inter_y0 - full_y0) * src_height / full_height))
    src_x1 = int(np.ceil((inter_x1 - full_x0) * src_width / full_width))
    src_y1 = int(np.ceil((inter_y1 - full_y0) * src_height / full_height))

    dst_x0 = int(
        np.floor(
            (inter_x0 - orig_loaded_x0)
            * crop_meta["orig_width"]
            / crop_meta["loaded_width"]
        )
    )
    dst_y0 = int(
        np.floor(
            (inter_y0 - orig_loaded_y0)
            * crop_meta["orig_height"]
            / crop_meta["loaded_height"]
        )
    )
    dst_x1 = int(
        np.ceil(
            (inter_x1 - orig_loaded_x0)
            * crop_meta["orig_width"]
            / crop_meta["loaded_width"]
        )
    )
    dst_y1 = int(
        np.ceil(
            (inter_y1 - orig_loaded_y0)
            * crop_meta["orig_height"]
            / crop_meta["loaded_height"]
        )
    )

    src_x0 = max(0, min(src_x0, src_width))
    src_x1 = max(src_x0 + 1, min(src_x1, src_width))
    src_y0 = max(0, min(src_y0, src_height))
    src_y1 = max(src_y0 + 1, min(src_y1, src_height))
    dst_x0 = max(0, min(dst_x0, crop_meta["orig_width"]))
    dst_x1 = max(dst_x0 + 1, min(dst_x1, crop_meta["orig_width"]))
    dst_y0 = max(0, min(dst_y0, crop_meta["orig_height"]))
    dst_y1 = max(dst_y0 + 1, min(dst_y1, crop_meta["orig_height"]))

    return {
        "src": (src_x0, src_y0, src_x1, src_y1),
        "dst": (dst_x0, dst_y0, dst_x1, dst_y1),
    }


def preprocess_with_repo_crop(img_path, bbox_path, bbox_index, train_json, device):
    img_path = os.path.realpath(img_path)
    bbox_path = os.path.realpath(bbox_path)
    data = train_json.get("data", {})
    online = data.get("online_creation", {})

    img_orig = cv2.imread(img_path)
    if img_orig is None:
        raise ValueError(f"Failed to read image: {img_path}")

    boxes = load_bbox_txt(bbox_path)
    bbox_entry = select_bbox(boxes, bbox_index)
    cls = int(bbox_entry[0])
    bbox = [int(v) for v in bbox_entry[1:]]

    mask_delta = normalize_mask_delta(online.get("mask_delta_A", [[]]))
    mask_random_offset = online.get("mask_random_offset_A", [0.0])
    mask_square = online.get("mask_square_A", False)
    _, crop_h, crop_w, output_h, output_w = get_train_shape(train_json)
    crop_dim = crop_h if crop_h == crop_w else [crop_h, crop_w]
    output_dim = output_h if output_h == output_w else [output_h, output_w]
    load_size = online.get("load_size_A", output_dim)
    if isinstance(load_size, int):
        load_size = [load_size]
    elif isinstance(load_size, tuple):
        load_size = list(load_size)
    context_pixels = data.get("online_context_pixels", 0)
    min_crop_bbox_ratio = online.get("min_crop_bbox_ratio_A", 0)

    crop_coordinates = crop_image(
        img_path=img_path,
        bbox_path=bbox_path,
        mask_delta=mask_delta,
        mask_random_offset=mask_random_offset,
        crop_delta=0,
        mask_square=mask_square,
        crop_dim=crop_dim,
        output_dim=output_dim,
        context_pixels=context_pixels,
        load_size=load_size,
        get_crop_coordinates=True,
        crop_center=True,
        bbox_ref_id=bbox_index,
        min_crop_bbox_ratio=min_crop_bbox_ratio,
    )

    img, mask, _, _, crop_meta = crop_image(
        img_path=img_path,
        bbox_path=bbox_path,
        mask_delta=mask_delta,
        mask_random_offset=mask_random_offset,
        crop_delta=0,
        mask_square=mask_square,
        crop_dim=crop_dim,
        output_dim=output_dim,
        context_pixels=context_pixels,
        load_size=load_size,
        crop_coordinates=crop_coordinates,
        crop_center=True,
        bbox_ref_id=bbox_index,
        override_class=cls,
        return_meta=True,
    )

    bbox_select = compute_paste_bbox(crop_meta)
    if bbox_select is None:
        bbox_select = compute_bbox_select(
            bbox=bbox,
            cls=cls,
            mask_delta=mask_delta,
            mask_square=mask_square,
            crop_coordinates=crop_coordinates,
            context_pixels=context_pixels,
        )

    img = np.array(img)
    mask = np.array(mask) if mask is not None else None

    if mask is not None and data.get("inverted_mask", False):
        mask = mask.copy()
        mask[mask > 0] = 2
        mask[mask == 0] = 1
        mask[mask == 2] = 0

    img_tensor = to_tensor(img).to(device)
    mask_tensor = None
    if mask is not None:
        mask_tensor = (
            torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0).to(device)
        )

    if mask_tensor is not None:
        if online.get("rand_mask_A", False):
            y_t = fill_mask_with_random(
                img_tensor.clone().detach(), mask_tensor.clone().detach(), -1
            )
        elif online.get("color_mask_A", False):
            y_t = fill_mask_with_color(
                img_tensor.clone().detach(), mask_tensor.clone().detach(), {}
            )
        else:
            y_t = img_tensor.clone().detach()
    else:
        y_t = torch.randn_like(img_tensor)
        mask_tensor = torch.zeros(
            (1, img_tensor.shape[-2], img_tensor.shape[-1]), device=device
        )

    cond_mode = train_json.get("alg", {}).get("diffusion_cond_image_creation", "y_t")
    cond_image = None
    if cond_mode != "y_t":
        raise NotImplementedError(
            "This runner currently supports only alg.diffusion_cond_image_creation = 'y_t'."
        )

    return {
        "img_path": img_path,
        "index": None,
        "bbox": bbox,
        "bbox_select": bbox_select,
        "img_orig": img_orig.copy(),
        "y_t": y_t.unsqueeze(0).detach().cpu(),
        "cond_image": cond_image,
        "y0_tensor": img_tensor.unsqueeze(0).detach().cpu(),
        "mask": mask_tensor.unsqueeze(0).float().detach().cpu(),
        "img_tensor": img_tensor.unsqueeze(0).detach().cpu(),
        "crop_meta": crop_meta,
        "has_bbox": True,
        "generated_bbox": None,
    }


def chw_to_bgr_uint8(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, -1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def mask_to_uint8(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    return (np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def project_known_pixels(x, y_known, mask):
    if mask is None or y_known is None:
        return x
    return x * mask + y_known * (1.0 - mask)


def match_prediction_channels(pred, reference):
    if pred.ndim != reference.ndim:
        raise RuntimeError(
            f"Prediction dims {pred.ndim} do not match reference dims {reference.ndim}"
        )
    channel_dim = 2 if pred.ndim == 5 else 1
    pred_channels = pred.shape[channel_dim]
    ref_channels = reference.shape[channel_dim]
    if pred_channels == ref_channels:
        return pred
    if pred_channels > ref_channels:
        if channel_dim == 1:
            return pred[:, -ref_channels:, :, :]
        return pred[:, :, -ref_channels:, :, :]
    raise RuntimeError(
        f"Prediction channels ({pred_channels}) are fewer than reference channels ({ref_channels})"
    )


def _shape_has_dynamic_dims(shape: Tuple[int, ...]) -> bool:
    return any(int(d) < 0 for d in shape)


def _is_small_condition_tensor(shape: Tuple[int, ...]) -> bool:
    return len(shape) <= 2


def _ensure_static_dims_match(
    name: str, raw_shape: Tuple[int, ...], host_shape: Tuple[int, ...]
) -> None:
    if len(raw_shape) != len(host_shape):
        raise RuntimeError(
            f"Tensor {name!r} rank mismatch: engine expects {raw_shape}, host provided {host_shape}"
        )
    for index, (raw_dim, host_dim) in enumerate(zip(raw_shape, host_shape)):
        if int(raw_dim) >= 0 and int(raw_dim) != int(host_dim):
            raise RuntimeError(
                f"Tensor {name!r} dim {index} mismatch: engine expects {raw_shape}, host provided {host_shape}"
            )


def _make_scalar_input(value, raw_shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if len(raw_shape) == 0:
        return np.asarray(value, dtype=dtype)
    shape = tuple(1 if int(dim) < 0 else int(dim) for dim in raw_shape)
    return np.full(shape, value, dtype=dtype)


class TensorRTDenoiserSession:
    def __init__(
        self,
        engine_path: str,
        model_input_name: Optional[str] = None,
        timesteps_input_name: Optional[str] = None,
        labels_input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> None:
        self.trt = import_tensorrt()
        self.cuda = CudaRuntime()
        if self.cuda.device_count() < 1:
            raise RuntimeError(
                "CUDA is required for TensorRT inference, but no CUDA-capable device was found."
            )

        self.engine_path = Path(engine_path)
        _, self.runtime, self.engine = load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.io_rows = describe_engine_io(self.engine)
        self.input_names = [name for name, mode, _, _ in self.io_rows if mode == "INPUT"]
        self.output_names = [
            name for name, mode, _, _ in self.io_rows if mode == "OUTPUT"
        ]
        if not self.input_names:
            raise RuntimeError("TensorRT engine has no input tensors")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors")

        self.model_input_name = self._resolve_model_input_name(model_input_name)
        self.timesteps_input_name = self._resolve_timesteps_input_name(
            timesteps_input_name
        )
        self.labels_input_name = self._resolve_labels_input_name(labels_input_name)
        self.output_name = self._resolve_output_name(output_name)

        self.stream = self.cuda.stream_create()
        self.device_buffers: Dict[str, ctypes.c_void_p] = {}
        self.host_outputs: Dict[str, np.ndarray] = {}
        self.tensor_shapes: Dict[str, Tuple[int, ...]] = {}
        self.tensor_dtypes: Dict[str, np.dtype] = {}
        self.bound_input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.reset_timing_stats()

    def _resolve_tensor_name(
        self,
        override_name: Optional[str],
        expected_name: str,
        candidates: List[str],
        kind: str,
    ) -> str:
        if override_name is not None:
            if override_name not in candidates:
                raise RuntimeError(
                    f"Configured {kind} tensor {override_name!r} not found in engine candidates: {candidates}"
                )
            return override_name
        if expected_name in candidates:
            return expected_name
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            f"Unable to infer {kind} tensor name. Expected {expected_name!r}; candidates={candidates}"
        )

    def _resolve_model_input_name(self, override_name: Optional[str]) -> str:
        candidates = []
        for name in self.input_names:
            raw_shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            if len(raw_shape) == 5:
                candidates.append(name)
        return self._resolve_tensor_name(
            override_name, "model_input", candidates, "model input"
        )

    def _resolve_timesteps_input_name(self, override_name: Optional[str]) -> str:
        candidates = []
        for name in self.input_names:
            if name == self.model_input_name:
                continue
            raw_shape = self._raw_shape_for(name)
            if not _is_small_condition_tensor(raw_shape):
                continue
            dtype = self._dtype_for(name)
            if np.issubdtype(dtype, np.floating):
                candidates.append(name)
        return self._resolve_tensor_name(
            override_name, "timesteps", candidates, "timesteps input"
        )

    def _resolve_labels_input_name(self, override_name: Optional[str]) -> str:
        candidates = []
        for name in self.input_names:
            if name in (self.model_input_name, self.timesteps_input_name):
                continue
            raw_shape = self._raw_shape_for(name)
            if not _is_small_condition_tensor(raw_shape):
                continue
            dtype = self._dtype_for(name)
            if np.issubdtype(dtype, np.integer):
                candidates.append(name)
        return self._resolve_tensor_name(
            override_name, "labels", candidates, "labels input"
        )

    def _resolve_output_name(self, override_name: Optional[str]) -> str:
        return self._resolve_tensor_name(
            override_name, "output", self.output_names, "output"
        )

    def _dtype_for(self, name: str) -> np.dtype:
        return np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))

    def _raw_shape_for(self, name: str) -> Tuple[int, ...]:
        return tuple(int(d) for d in self.engine.get_tensor_shape(name))

    def _release_buffers(self) -> None:
        for name, ptr in list(self.device_buffers.items()):
            try:
                self.cuda.free(ptr)
            except Exception as exc:
                print(
                    f"Warning: failed to free device buffer for {name}: {exc}",
                    file=sys.stderr,
                )
        self.device_buffers.clear()
        self.host_outputs.clear()
        self.tensor_shapes.clear()
        self.tensor_dtypes.clear()
        self.bound_input_shapes.clear()

    def close(self) -> None:
        self._release_buffers()
        if self.stream is not None:
            try:
                self.cuda.stream_destroy(self.stream)
            except Exception as exc:
                print(
                    f"Warning: failed to destroy CUDA stream: {exc}",
                    file=sys.stderr,
                )
            self.stream = None
        self.context = None
        self.engine = None
        self.runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def get_model_input_shape(self) -> Tuple[int, ...]:
        return self._raw_shape_for(self.model_input_name)

    def format_io_summary(self) -> List[str]:
        rows = []
        for name, mode, shape, dtype in self.io_rows:
            rows.append(f"  - {mode:<6} {name}: shape={shape}, dtype={dtype}")
        return rows

    def reset_timing_stats(self) -> None:
        self.timing_stats = {
            "calls": 0,
            "h2d_ms": 0.0,
            "execute_ms": 0.0,
            "d2h_ms": 0.0,
            "total_ms": 0.0,
        }

    def get_timing_stats(self) -> Dict[str, float]:
        return dict(self.timing_stats)

    def _prepare_model_input(self, model_input: np.ndarray) -> np.ndarray:
        raw_shape = self._raw_shape_for(self.model_input_name)
        host = np.ascontiguousarray(
            model_input.astype(self._dtype_for(self.model_input_name), copy=False)
        )
        if host.ndim != 5:
            raise RuntimeError(
                f"Expected model_input rank 5, got shape {host.shape} for TensorRT inference"
            )
        if host.shape[0] != 1:
            raise RuntimeError(
                f"Expected model_input batch size 1, got shape {host.shape}"
            )
        _ensure_static_dims_match(self.model_input_name, raw_shape, host.shape)
        return host

    def _prepare_timestep_input(self, t_scalar: float) -> np.ndarray:
        raw_shape = self._raw_shape_for(self.timesteps_input_name)
        return np.ascontiguousarray(
            _make_scalar_input(
                float(t_scalar), raw_shape, self._dtype_for(self.timesteps_input_name)
            )
        )

    def _prepare_labels_input(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels).reshape(-1)
        if labels.size != 1:
            raise RuntimeError(
                f"Expected one label for batch=1 TensorRT inference, got shape {labels.shape}"
            )
        raw_shape = self._raw_shape_for(self.labels_input_name)
        return np.ascontiguousarray(
            _make_scalar_input(
                int(labels[0]), raw_shape, self._dtype_for(self.labels_input_name)
            )
        )

    def _ensure_bindings(self, host_inputs: Dict[str, np.ndarray]) -> None:
        desired_shapes = {
            name: tuple(int(dim) for dim in array.shape)
            for name, array in host_inputs.items()
        }
        if self.bound_input_shapes == desired_shapes and self.device_buffers:
            return

        self._release_buffers()

        for name, array in host_inputs.items():
            raw_shape = self._raw_shape_for(name)
            _ensure_static_dims_match(name, raw_shape, tuple(array.shape))
            if _shape_has_dynamic_dims(raw_shape):
                self.context.set_input_shape(name, tuple(int(dim) for dim in array.shape))

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Tensor shape unresolved for {name}: {shape}")

            dtype = self._dtype_for(name)
            ptr = self.cuda.malloc(_num_bytes(shape, dtype))

            self.device_buffers[name] = ptr
            self.tensor_shapes[name] = shape
            self.tensor_dtypes[name] = dtype
            self.context.set_tensor_address(name, int(ptr.value))

            if not is_input(self.engine, name):
                self.host_outputs[name] = np.empty(shape, dtype=dtype)

        self.bound_input_shapes = desired_shapes

    def infer(
        self,
        model_input: np.ndarray,
        t_scalar: float,
        labels: np.ndarray,
        dump_dir: Optional[str] = None,
        dump_tag: Optional[str] = None,
    ) -> np.ndarray:
        model_input_host = self._prepare_model_input(model_input)
        timestep_host = self._prepare_timestep_input(t_scalar)
        labels_host = self._prepare_labels_input(labels)

        if dump_dir and dump_tag:
            os.makedirs(dump_dir, exist_ok=True)
            np.save(os.path.join(dump_dir, f"{dump_tag}_model_input.npy"), model_input_host)
            np.save(os.path.join(dump_dir, f"{dump_tag}_timesteps.npy"), timestep_host)
            np.save(os.path.join(dump_dir, f"{dump_tag}_labels.npy"), labels_host)

        host_inputs = {
            self.model_input_name: model_input_host,
            self.timesteps_input_name: timestep_host,
            self.labels_input_name: labels_host,
        }
        self._ensure_bindings(host_inputs)

        t_total_start = time.perf_counter()
        t_h2d_start = time.perf_counter()
        for name, array in host_inputs.items():
            self.cuda.memcpy_h2d_async(self.device_buffers[name], array, self.stream)
        self.cuda.stream_synchronize(self.stream)
        t_h2d_end = time.perf_counter()

        t_exec_start = time.perf_counter()
        ok = self.context.execute_async_v3(int(self.stream.value))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        self.cuda.stream_synchronize(self.stream)
        t_exec_end = time.perf_counter()

        t_d2h_start = time.perf_counter()
        output_host = self.host_outputs[self.output_name]
        self.cuda.memcpy_d2h_async(
            output_host, self.device_buffers[self.output_name], self.stream
        )
        self.cuda.stream_synchronize(self.stream)
        t_d2h_end = time.perf_counter()

        self.timing_stats["calls"] += 1
        self.timing_stats["h2d_ms"] += (t_h2d_end - t_h2d_start) * 1000.0
        self.timing_stats["execute_ms"] += (t_exec_end - t_exec_start) * 1000.0
        self.timing_stats["d2h_ms"] += (t_d2h_end - t_d2h_start) * 1000.0
        self.timing_stats["total_ms"] += (t_d2h_end - t_total_start) * 1000.0
        return output_host.copy()


def get_engine_shape(session: TensorRTDenoiserSession):
    shape = session.get_model_input_shape()
    if len(shape) != 5:
        raise ValueError(f"Expected model_input rank 5, got {shape}")
    batch, frames, channels, height, width = shape
    if batch not in (-1, 1):
        raise ValueError(
            f"Expected batch size 1 or dynamic batch for denoiser TensorRT, got {batch}"
        )
    if channels not in (-1, 3):
        raise NotImplementedError(
            f"Expected 3 denoiser input channels for y_t mode, got {channels}"
        )
    return batch, frames, channels, height, width


def denoiser_forward(
    session: TensorRTDenoiserSession,
    model_input,
    t_scalar,
    labels,
    dump_dir=None,
    dump_tag=None,
):
    return session.infer(
        model_input=model_input,
        t_scalar=t_scalar,
        labels=labels,
        dump_dir=dump_dir,
        dump_tag=dump_tag,
    )


def forward_sample_restoration(
    session,
    x,
    t_scalar,
    y_cond,
    mask,
    labels,
    y_known,
    params,
    dump_dir=None,
    dump_prefix=None,
):
    x_in = project_known_pixels(x, y_known, mask)
    if y_cond is None:
        model_input = x_in
    else:
        model_input = np.concatenate([y_cond, x_in], axis=2)

    x_cond = denoiser_forward(
        session,
        model_input,
        t_scalar,
        labels,
        dump_dir=dump_dir,
        dump_tag=f"{dump_prefix}_cond" if dump_prefix else None,
    )
    x_cond = match_prediction_channels(x_cond, x_in)
    x_cond = project_known_pixels(x_cond, y_known, mask)

    den = 1.0 - float(t_scalar)
    if not params["disable_inference_clipping"]:
        den = max(den, params["t_eps"])
    v_cond = (x_cond - x_in) / den

    if math.isclose(params["cfg_scale"], 1.0, rel_tol=0.0, abs_tol=1e-12):
        return v_cond

    low, high = params["cfg_interval"]
    interval_active = (t_scalar < high) and (low == 0 or t_scalar > low)
    if not interval_active:
        return v_cond

    x_uncond = denoiser_forward(
        session,
        model_input,
        t_scalar,
        np.full_like(labels, params["num_classes"]),
        dump_dir=dump_dir,
        dump_tag=f"{dump_prefix}_uncond" if dump_prefix else None,
    )
    x_uncond = match_prediction_channels(x_uncond, x_in)
    x_uncond = project_known_pixels(x_uncond, y_known, mask)
    v_uncond = (x_uncond - x_in) / den
    return v_uncond + params["cfg_scale"] * (v_cond - v_uncond)


def restoration_with_denoiser(
    session,
    y,
    y_cond,
    denoise_steps,
    mask,
    labels,
    params,
    init_noise,
    dump_dir=None,
    dump_prefix=None,
):
    y_known = y if mask is not None else None
    if mask is not None:
        mask = np.clip(mask, 0.0, 1.0)
        y_background = y * (1.0 - mask)
    else:
        y_background = y

    x = y_background + init_noise * params["noise_scale"]
    if mask is not None:
        x = x * mask + y * (1.0 - mask)

    timesteps = np.linspace(0.0, 1.0, int(denoise_steps) + 1, dtype=np.float32)
    for i in range(int(denoise_steps) - 1):
        t = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        v_t = forward_sample_restoration(
            session,
            x,
            t,
            y_cond,
            mask,
            labels,
            y_known,
            params,
            dump_dir=dump_dir,
            dump_prefix=f"{dump_prefix}_step{i}_a" if dump_prefix else None,
        )
        x_euler = x + (t_next - t) * v_t
        v_t_next = forward_sample_restoration(
            session,
            x_euler,
            t_next,
            y_cond,
            mask,
            labels,
            y_known,
            params,
            dump_dir=dump_dir,
            dump_prefix=f"{dump_prefix}_step{i}_b" if dump_prefix else None,
        )
        x = x + (t_next - t) * 0.5 * (v_t + v_t_next)
        if params["clip_denoised"]:
            x = np.clip(x, -1.0, 1.0)
        if mask is not None:
            x = x * mask + y * (1.0 - mask)

    x = x + (float(timesteps[-1]) - float(timesteps[-2])) * forward_sample_restoration(
        session,
        x,
        float(timesteps[-2]),
        y_cond,
        mask,
        labels,
        y_known,
        params,
        dump_dir=dump_dir,
        dump_prefix=f"{dump_prefix}_final" if dump_prefix else None,
    )
    if params["clip_denoised"]:
        x = np.clip(x, -1.0, 1.0)
    if mask is not None:
        x = x * mask + y * (1.0 - mask)
    return np.clip(x, -1.0, 1.0)


def write_frame(frame_index, out_tensor, frame_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_img_for_paste = chw_to_bgr_uint8(out_tensor)
    img_orig = frame_data["img_orig"].copy()
    bbox_select = frame_data["bbox_select"]
    has_bbox = frame_data["has_bbox"]
    mask_np_for_paste = None
    if frame_data["mask"] is not None:
        mask_np_for_paste = (mask_to_uint8(frame_data["mask"]) > 0).astype(np.uint8)

    paste_bbox = bbox_select
    if has_bbox:
        mapping = None
        if frame_data.get("crop_meta") is not None:
            mapping = compute_paste_mapping(
                frame_data["crop_meta"],
                src_width=out_img_for_paste.shape[1],
                src_height=out_img_for_paste.shape[0],
            )
        if mapping is not None:
            sx0, sy0, sx1, sy1 = mapping["src"]
            x0, y0, x1, y1 = mapping["dst"]
            out_img_crop = out_img_for_paste[sy0:sy1, sx0:sx1]
            out_img_resized = cv2.resize(out_img_crop, (x1 - x0, y1 - y0))
            paste_bbox = [x0, y0, x1, y1]
        else:
            y0 = max(0, min(int(bbox_select[1]), img_orig.shape[0] - 1))
            y1 = max(y0 + 1, min(int(bbox_select[3]), img_orig.shape[0]))
            x0 = max(0, min(int(bbox_select[0]), img_orig.shape[1] - 1))
            x1 = max(x0 + 1, min(int(bbox_select[2]), img_orig.shape[1]))
            out_img_resized = cv2.resize(out_img_for_paste, (x1 - x0, y1 - y0))
            paste_bbox = [x0, y0, x1, y1]
        out_img_real_size = img_orig.copy()
        if mask_np_for_paste is not None:
            if mapping is not None:
                mask_np_for_paste = mask_np_for_paste[sy0:sy1, sx0:sx1]
            mask_resized = cv2.resize(
                mask_np_for_paste, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
            )
            mask_3 = mask_resized.astype(bool)[:, :, None]
            orig_crop = out_img_real_size[y0:y1, x0:x1].copy()
            out_img_real_size[y0:y1, x0:x1] = np.where(
                mask_3, out_img_resized, orig_crop
            )
        else:
            out_img_real_size[y0:y1, x0:x1] = out_img_resized
        orig_crop = img_orig[y0:y1, x0:x1].copy()
    else:
        out_img_resized = out_img_for_paste
        out_img_real_size = img_orig.copy()
        orig_crop = img_orig.copy()

    name_out = f"{frame_index:06d}"
    cv2.imwrite(os.path.join(output_dir, name_out + "_orig.png"), img_orig)
    cv2.imwrite(
        os.path.join(output_dir, name_out + "_generated.png"), out_img_real_size
    )
    cv2.imwrite(
        os.path.join(output_dir, name_out + "_generated_crop.png"), out_img_for_paste
    )
    cv2.imwrite(os.path.join(output_dir, name_out + "_orig_crop.png"), orig_crop)
    cv2.imwrite(
        os.path.join(output_dir, name_out + "_y_t.png"),
        chw_to_bgr_uint8(frame_data["y_t"]),
    )
    cv2.imwrite(
        os.path.join(output_dir, name_out + "_y_0.png"),
        chw_to_bgr_uint8(frame_data["y0_tensor"]),
    )
    cv2.imwrite(
        os.path.join(output_dir, name_out + "_mask.png"),
        mask_to_uint8(frame_data["mask"]),
    )
    with open(os.path.join(output_dir, name_out + "_bbox_select.json"), "w") as f:
        json.dump([int(v) for v in paste_bbox], f)
    with open(os.path.join(output_dir, name_out + "_orig_bbox.json"), "w") as f:
        json.dump([int(v) for v in frame_data["bbox"]], f)


def run_sequence(
    session,
    pairs,
    dataset_root,
    output_dir,
    bbox_index,
    label,
    seed,
    train_json,
    denoise_steps,
    debug_dump_dir,
    warmup,
    repeat,
):
    rng = np.random.default_rng(seed)
    _, crop_h, _, _, _ = get_train_shape(train_json)
    params = get_b2b_params(train_json, crop_h)
    inputmix = True
    prev_frame = None
    last_seq_half_y_t = None
    last_seq_half_cond_image = None
    last_seq_half_y0_tensor = None
    last_seq_half_mask = None
    frames_written = []
    seq_half = 1
    num_buckets = 2
    max_sigma = 0.71
    timing_summary = {
        "runs": 0,
        "frames": 0,
        "restoration_ms": 0.0,
        "trt_calls": 0,
        "trt_h2d_ms": 0.0,
        "trt_execute_ms": 0.0,
        "trt_d2h_ms": 0.0,
        "trt_total_ms": 0.0,
    }

    for sequence_count, (img_rel, bbox_rel) in enumerate(pairs):
        img_path = os.path.join(dataset_root, img_rel)
        bbox_path = os.path.join(dataset_root, bbox_rel)
        frame_data = preprocess_with_repo_crop(
            img_path=img_path,
            bbox_path=bbox_path,
            bbox_index=bbox_index,
            train_json=train_json,
            device=torch.device("cpu"),
        )
        frame_data["index"] = sequence_count
        frame_data["img_rel"] = img_rel

        if prev_frame is None:
            prev_frame = frame_data
            if frame_data["y0_tensor"] is not None:
                y0_np = frame_data["y0_tensor"].numpy()
                sigma = rng.integers(0, num_buckets, size=(y0_np.shape[0],)).astype(
                    np.float32
                )
                sigma = sigma / float(num_buckets - 1) * max_sigma
                sigma = sigma.reshape(y0_np.shape[0], 1, 1, 1)
                eps_ctx = rng.standard_normal(size=y0_np.shape, dtype=np.float32)
                y0_noisy_first = torch.from_numpy(y0_np + sigma * eps_ctx)
            else:
                y0_noisy_first = None
            continue

        if last_seq_half_y_t is None:
            y_t_batch = prepare_tensors([prev_frame["y_t"], frame_data["y_t"]])
            y0_tensor_batch = prepare_tensors(
                [prev_frame["y0_tensor"], frame_data["y0_tensor"]]
            )
            mask_batch = prepare_tensors([prev_frame["mask"], frame_data["mask"]])
            if inputmix:
                cond_image_batch = prepare_tensors(
                    [y0_noisy_first, frame_data["cond_image"]]
                )
            else:
                cond_image_batch = prepare_tensors(
                    [prev_frame["cond_image"], frame_data["cond_image"]]
                )
        else:
            y_t_batch = prepare_tensors(last_seq_half_y_t + [frame_data["y_t"]])
            cond_image_batch = prepare_tensors(
                last_seq_half_cond_image + [frame_data["cond_image"]]
            )
            y0_tensor_batch = prepare_tensors(
                last_seq_half_y0_tensor + [frame_data["y0_tensor"]]
            )
            mask_batch = prepare_tensors(last_seq_half_mask + [frame_data["mask"]])

        labels = np.asarray([0 if label is None else int(label)], dtype=np.int64)
        init_noise = rng.standard_normal(size=y_t_batch.shape, dtype=np.float32)
        y_np = y_t_batch.numpy().astype(np.float32)
        y_cond_np = (
            None
            if cond_image_batch is None
            else cond_image_batch.numpy().astype(np.float32)
        )
        mask_np = mask_batch.numpy().astype(np.float32)

        for _ in range(warmup):
            session.reset_timing_stats()
            restoration_with_denoiser(
                session=session,
                y=y_np,
                y_cond=y_cond_np,
                denoise_steps=denoise_steps,
                mask=mask_np,
                labels=labels,
                params=params,
                init_noise=init_noise,
                dump_dir=None,
                dump_prefix=None,
            )

        out_tensor = None
        for repeat_index in range(repeat):
            session.reset_timing_stats()
            restoration_start = time.perf_counter()
            out_tensor = restoration_with_denoiser(
                session=session,
                y=y_np,
                y_cond=y_cond_np,
                denoise_steps=denoise_steps,
                mask=mask_np,
                labels=labels,
                params=params,
                init_noise=init_noise,
                dump_dir=debug_dump_dir if repeat_index == repeat - 1 else None,
                dump_prefix=(
                    f"frame_{sequence_count:06d}" if repeat_index == repeat - 1 else None
                ),
            )
            restoration_ms = (time.perf_counter() - restoration_start) * 1000.0
            trt_stats = session.get_timing_stats()
            timing_summary["runs"] += 1
            timing_summary["frames"] += int(y_t_batch.shape[1])
            timing_summary["restoration_ms"] += restoration_ms
            timing_summary["trt_calls"] += int(trt_stats["calls"])
            timing_summary["trt_h2d_ms"] += trt_stats["h2d_ms"]
            timing_summary["trt_execute_ms"] += trt_stats["execute_ms"]
            timing_summary["trt_d2h_ms"] += trt_stats["d2h_ms"]
            timing_summary["trt_total_ms"] += trt_stats["total_ms"]

        out_tensor_torch = torch.from_numpy(out_tensor).squeeze(0)
        out_img_tensor_list_batch = [
            out_tensor_torch[i : i + 1] for i in range(out_tensor_torch.shape[0])
        ]

        y_t_temp_list = separate_tensors(y_t_batch)
        y0_tensor_temp_list = separate_tensors(y0_tensor_batch)
        mask_temp_list = separate_tensors(mask_batch)
        last_seq_half_y_t = y_t_temp_list[-seq_half:]
        last_seq_half_cond_image = out_img_tensor_list_batch[-seq_half:]
        last_seq_half_y0_tensor = y0_tensor_temp_list[-seq_half:]
        last_seq_half_mask = mask_temp_list[-seq_half:]

        write_frame(
            prev_frame["index"], out_img_tensor_list_batch[0], prev_frame, output_dir
        )
        write_frame(
            frame_data["index"], out_img_tensor_list_batch[-1], frame_data, output_dir
        )
        frames_written.extend([prev_frame["index"], frame_data["index"]])
        prev_frame = frame_data

    return sorted(set(frames_written)), timing_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_in_file",
        "--engine",
        dest="engine",
        required=True,
        help="Path to denoiser TensorRT engine",
    )
    parser.add_argument(
        "--paths_in_file",
        required=True,
        help="paths.txt containing 'image_rel_path bbox_rel_path' pairs",
    )
    parser.add_argument(
        "--dataset_root",
        help="Dataset root used to resolve relative paths. Defaults to dirname(dirname(paths_in_file))",
    )
    parser.add_argument(
        "--train_config",
        help="Optional train_config.json for crop, mask, and denoising settings",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start line index in paths.txt"
    )
    parser.add_argument(
        "--bbox_index", type=int, default=0, help="Which bbox line to use"
    )
    parser.add_argument("--label", type=int, default=None, help="Override class label")
    parser.add_argument("--seed", type=int, default=0, help="Seed for init_noise")
    parser.add_argument(
        "--denoise_steps",
        type=int,
        help="Override denoise step count. Defaults to first entry from alg.b2b_denoise_timesteps",
    )
    parser.add_argument(
        "--single_step",
        action="store_true",
        help="Only process one 2-frame window",
    )
    parser.add_argument(
        "--debug_dump_dir",
        help="Optional directory to dump per-step denoiser TensorRT inputs",
    )
    parser.add_argument(
        "--model_input_name",
        "--model-input-name",
        help="Optional override for TensorRT model_input tensor name",
    )
    parser.add_argument(
        "--timesteps_input_name",
        "--timesteps-input-name",
        help="Optional override for TensorRT timesteps tensor name",
    )
    parser.add_argument(
        "--labels_input_name",
        "--labels-input-name",
        help="Optional override for TensorRT labels tensor name",
    )
    parser.add_argument(
        "--output_name",
        "--output-name",
        help="Optional override for TensorRT output tensor name",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup restoration runs per 2-frame window before timing",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed restoration runs per 2-frame window",
    )
    return parser.parse_args()


args = None


def main():
    global args
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeat <= 0:
        raise ValueError("--repeat must be >= 1")

    session = TensorRTDenoiserSession(
        engine_path=args.engine,
        model_input_name=args.model_input_name,
        timesteps_input_name=args.timesteps_input_name,
        labels_input_name=args.labels_input_name,
        output_name=args.output_name,
    )

    trt_batch, trt_frames, trt_channels, trt_height, trt_width = get_engine_shape(
        session
    )

    train_json, train_config_path = load_train_config(args.engine, args.train_config)
    train_frames, train_height, train_width, _, _ = get_train_shape(train_json)
    if train_json.get("alg", {}).get("diffusion_cond_image_creation", "y_t") != "y_t":
        raise NotImplementedError(
            "This runner currently supports only alg.diffusion_cond_image_creation = 'y_t'."
        )
    if (
        trt_frames > 0
        and trt_height > 0
        and trt_width > 0
        and (trt_frames, trt_height, trt_width)
        != (train_frames, train_height, train_width)
    ):
        print(
            "warning: TensorRT shape does not match train_config.json: "
            f"trt={(trt_frames, trt_height, trt_width)} "
            f"train_config={(train_frames, train_height, train_width)}"
        )
    if trt_channels not in (-1, 3):
        raise NotImplementedError(
            f"Expected 3 denoiser input channels for y_t mode, got {trt_channels}"
        )
    if trt_batch not in (-1, 1):
        raise ValueError(f"Expected batch size 1 denoiser TensorRT, got {trt_batch}")

    dataset_root = resolve_dataset_root(args.paths_in_file, args.dataset_root)
    pairs = read_paths_file(args.paths_in_file)
    if args.start_index + 2 > len(pairs):
        raise ValueError(
            f"paths.txt has {len(pairs)} entries, need at least 2 from start_index={args.start_index}"
        )
    if args.single_step:
        pairs = pairs[args.start_index : args.start_index + 2]
    else:
        pairs = pairs[args.start_index :]

    denoise_steps = resolve_denoise_steps(train_json, args.denoise_steps)

    try:
        frames_written, timing_summary = run_sequence(
            session=session,
            pairs=pairs,
            dataset_root=dataset_root,
            output_dir=args.output_dir,
            bbox_index=args.bbox_index,
            label=args.label,
            seed=args.seed,
            train_json=train_json,
            denoise_steps=denoise_steps,
            debug_dump_dir=args.debug_dump_dir,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    finally:
        session.close()

    print("engine_io:")
    for row in session.format_io_summary():
        print(row)
    print(f"dataset_root : {dataset_root}")
    print(f"train_config : {train_config_path}")
    print(f"engine       : {args.engine}")
    print(f"model_input  : {session.model_input_name}")
    print(f"timesteps    : {session.timesteps_input_name}")
    print(f"labels       : {session.labels_input_name}")
    print(f"output       : {session.output_name}")
    print(f"denoise_steps: {denoise_steps}")
    print(f"warmup       : {args.warmup}")
    print(f"repeat       : {args.repeat}")
    if timing_summary["runs"] > 0:
        avg_restoration_ms = timing_summary["restoration_ms"] / timing_summary["runs"]
        avg_frames_per_run = timing_summary["frames"] / timing_summary["runs"]
        fps = (
            timing_summary["frames"] * 1000.0 / timing_summary["restoration_ms"]
            if timing_summary["restoration_ms"] > 0
            else float("inf")
        )
        restorations_per_sec = (
            timing_summary["runs"] * 1000.0 / timing_summary["restoration_ms"]
            if timing_summary["restoration_ms"] > 0
            else float("inf")
        )
        print(
            "restoration_timing: "
            f"avg={avg_restoration_ms:.3f} ms/restoration, "
            f"frames_per_restoration={avg_frames_per_run:.3f}, "
            f"throughput={fps:.3f} fps, "
            f"restoration_rate={restorations_per_sec:.3f} restorations/s"
        )
        if timing_summary["trt_calls"] > 0:
            avg_h2d_ms = timing_summary["trt_h2d_ms"] / timing_summary["trt_calls"]
            avg_execute_ms = (
                timing_summary["trt_execute_ms"] / timing_summary["trt_calls"]
            )
            avg_d2h_ms = timing_summary["trt_d2h_ms"] / timing_summary["trt_calls"]
            avg_total_ms = timing_summary["trt_total_ms"] / timing_summary["trt_calls"]
            avg_calls_per_run = timing_summary["trt_calls"] / timing_summary["runs"]
            denoiser_calls_per_sec = (
                timing_summary["trt_calls"] * 1000.0 / timing_summary["trt_total_ms"]
                if timing_summary["trt_total_ms"] > 0
                else float("inf")
            )
            print(
                "trt_denoiser_timing: "
                f"avg_total={avg_total_ms:.3f} ms/model_call, "
                f"h2d={avg_h2d_ms:.3f} ms/call, "
                f"execute={avg_execute_ms:.3f} ms/call, "
                f"d2h={avg_d2h_ms:.3f} ms/call, "
                f"calls_per_restoration={avg_calls_per_run:.3f}, "
                f"call_rate={denoiser_calls_per_sec:.3f} calls/s"
            )
    print(f"written      : {len(frames_written)} frames")
    print(f"saved        : {args.output_dir}")


if __name__ == "__main__":
    main()
