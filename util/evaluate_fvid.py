r"""Offline FVD on full-frame MP4 collections, independent of training/inference.

Example (missing weights are downloaded automatically on first use):
    python util/evaluate_fvid.py --real-dir /videos/original \
        --generated-dir /videos/generated

Default results: ./tmp/fvd_results.json, relative to the current working directory.
Each input accepts a single MP4 file or a folder of MP4 files (including subfolders).

Backbones and numerical helpers: ZhengJun-AI/vfid-metrics
(see fvd_backbones/LICENSE).
Weights: https://github.com/ZhengJun-AI/vfid-metrics#usage
FVD compares collections, not paired videos. Original-input references measure
source-distribution distance, including intentional changes to the ring.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from torchvision.transforms.functional import resize, to_tensor
from tqdm import tqdm

try:  # Support both direct execution and import from tests.
    from .fvd_backbones.frechet import calculate_fid
    from .fvd_backbones.inception3d import InceptionI3d
    from .fvd_backbones.resnext3d import resnet101
except ImportError:
    from fvd_backbones.frechet import calculate_fid
    from fvd_backbones.inception3d import InceptionI3d
    from fvd_backbones.resnext3d import resnet101


WEIGHTS = {"i3d": "i3d.pt", "resnext": "resnext-101.pth"}
# Matching checkpoints from the weights folder linked in the upstream README.
WEIGHT_URLS = {
    "i3d": "https://drive.usercontent.google.com/download?id=1AUTu5cbou7JWxeBP_BL9V12M1jhNWQDd&export=download&confirm=t",
    "resnext": "https://drive.usercontent.google.com/download?id=1I5OJ3WU4UfI78UZCpC5hK5HsYpHohQvx&export=download&confirm=t",
}


def ensure_weights(name, path):
    """Reuse local weights, or stream a download and publish only on success."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        print(f"Downloading {name} weights to {path}", flush=True)
        with requests.get(
            WEIGHT_URLS[name], stream=True, timeout=(30, 120)
        ) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError(
                    "Weight server returned an HTML page instead of a checkpoint"
                )
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".part", delete=False
            ) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
            size = temporary.stat().st_size
            expected = response.headers.get("Content-Length")
            if size == 0 or (expected is not None and size != int(expected)):
                raise ValueError("Empty or incomplete weight download")
        temporary.replace(path)
    except (requests.RequestException, OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot download {name} weights from {WEIGHT_URLS[name]}: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_model(name, weights, clip_length, device):
    # Same architectures and checkpoint formats as upstream fid.py.
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if name == "i3d":
        model = InceptionI3d(400, in_channels=3)
    else:
        model = resnet101(
            num_classes=400,
            shortcut_type="B",
            sample_size=224,
            sample_duration=clip_length,
            last_fc=False,
        )
        state = {k.removeprefix("module."): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def video_batches(path, clip_length, batch_size, info):
    """Decode once; never carry an incomplete clip into the next video."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    info.update(path=str(path), fps=capture.get(cv2.CAP_PROP_FPS), frames=0, clips=0)
    frames, clips = [], []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            info["frames"] += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Upstream image preprocessing: ToTensor -> Resize -> [-1, 1].
            frames.append(resize(to_tensor(rgb), [224, 224], antialias=True) * 2 - 1)
            if len(frames) == clip_length:
                clips.append(torch.stack(frames, dim=1))  # C,T,H,W
                frames = []
                info["clips"] += 1
            if len(clips) == batch_size:
                yield torch.stack(clips)
                clips = []
        if info["frames"] == 0 or (
            declared_frames > 0 and info["frames"] != declared_frames
        ):
            raise ValueError(
                f"Incomplete decode of {path}: {info['frames']} frames, "
                f"container reports {declared_frames}"
            )
        info["discarded_frames"] = len(frames)
        if clips:
            yield torch.stack(clips)
    finally:
        capture.release()


@torch.inference_mode()
def extract_features(model, name, batch, device):
    pred = model.extract_features(batch.to(device))
    if name == "i3d":
        pred = pred.squeeze(3).squeeze(3).mean(2)
    if pred.ndim != 2 or pred.shape[0] != batch.shape[0]:
        raise ValueError(f"Invalid {name} features: {tuple(pred.shape)}")
    features = pred.cpu().numpy()
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite {name} features")
    return features


def collect_features(
    directory,
    names,
    weights,
    clip_length,
    batch_size,
    device,
    models,
):
    paths = sorted(
        p.resolve()
        for p in ([directory] if directory.is_file() else directory.rglob("*"))
        if p.is_file() and p.suffix.lower() == ".mp4"
    )
    if not paths:
        raise ValueError(f"No MP4 videos in {directory}")
    features = {name: [] for name in names}
    videos = []
    for path in tqdm(paths, desc=directory.name, unit="video"):
        info = {}
        parts = {name: [] for name in names}
        for batch in video_batches(path, clip_length, batch_size, info):
            for name in names:
                if name not in models:
                    models[name] = build_model(name, weights[name], clip_length, device)
                parts[name].append(extract_features(models[name], name, batch, device))
        for name in names:
            dims = 1024 if name == "i3d" else 2048
            features[name].append(
                np.concatenate(parts[name]) if parts[name] else np.empty((0, dims))
            )
        videos.append(info)
    features = {name: np.concatenate(parts) for name, parts in features.items()}
    if any(len(value) < 2 for value in features.values()):
        raise ValueError(f"Need at least two complete clips in {directory}")
    return features, {
        "directory": str(directory.resolve()),
        "video_count": len(videos),
        "clip_count": sum(v["clips"] for v in videos),
        "discarded_frames": sum(v["discarded_frames"] for v in videos),
        "videos": videos,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--real-dir", type=Path, required=True, help="Real MP4 file or folder"
    )
    parser.add_argument(
        "--generated-dir", type=Path, required=True, help="Generated MP4 file or folder"
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Weight storage directory (default: <torch hub cache>/checkpoints/fvd); missing weights are downloaded",
    )
    parser.add_argument(
        "--backbones", nargs="+", choices=list(WEIGHTS), default=list(WEIGHTS)
    )
    parser.add_argument("--clip-length", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/fvd_results.json"),
        help="Results JSON (default: ./tmp/fvd_results.json under the working directory)",
    )
    args = parser.parse_args()
    # Small per-frame resizes suffer from excessive CPU thread overhead.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.clip_length < 2:
        parser.error("--clip-length must be at least 2")
    names = list(dict.fromkeys(args.backbones))
    weights_dir = args.weights_dir or Path(torch.hub.get_dir()) / "checkpoints" / "fvd"
    weights = {name: weights_dir / WEIGHTS[name] for name in names}
    for directory in (args.real_dir, args.generated_dir):
        if not (
            directory.is_dir()
            or (directory.is_file() and directory.suffix.lower() == ".mp4")
        ):
            parser.error(f"Expected an MP4 file or video folder: {directory}")
    try:
        for name, path in weights.items():
            ensure_weights(name, path)
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"FVD evaluation failed: {error}\n")
    weight_hashes = {}
    for name, path in weights.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        weight_hashes[name] = digest.hexdigest()
    models, collections = {}, {}
    try:
        for label, directory in (
            ("real", args.real_dir),
            ("generated", args.generated_dir),
        ):
            collections[label] = collect_features(
                directory,
                names,
                weights,
                args.clip_length,
                args.batch_size,
                args.device,
                models,
            )
        scores = {
            name: float(
                calculate_fid(
                    collections["real"][0][name].astype(np.float64),
                    collections["generated"][0][name].astype(np.float64),
                )
            )
            for name in names
        }
        if not all(np.isfinite(score) for score in scores.values()):
            raise ValueError("FVD computation returned a non-finite score")
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"FVD evaluation failed: {error}\n")
    result = {
        "scores": {f"fvd_{name}": score for name, score in scores.items()},
        "settings": {
            "clip_length": args.clip_length,
            "batch_size": args.batch_size,
            "device": args.device,
            "resolution": [224, 224],
            "frame_sampling": "consecutive, non-overlapping, native FPS",
            "weight_sha256": weight_hashes,
        },
        "real": collections["real"][1],
        "generated": collections["generated"][1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["scores"], indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
