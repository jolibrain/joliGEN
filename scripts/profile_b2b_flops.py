"""THOP estimates for representative B2B forwards, not sampled generation costs."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.overrides import TorchFunctionMode

VERSION = "2.2.0"


class Products(TorchFunctionMode):
    """Supplement THOP 2.2.0's missing baddbmm product only."""

    def __init__(self):
        self.ops = 0
        self.unsupported = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if func in (torch.baddbmm, torch.Tensor.baddbmm):
            first = args[1] if len(args) > 1 else kwargs["batch1"]
            self.ops += result.numel() * first.shape[-1]
        if func in (torch.einsum, torch.addmm, torch.addbmm, torch.mm, torch.mv):
            # The known unpatchify einsum is a pure permutation of dimensions.
            if func is not torch.einsum or args[0].replace(" ", "") != "nhwpqc->nchpwq":
                self.unsupported.add(str(func))
        return result


class Predict(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, times, labels, conditions):
        call = getattr(self.model, "predict_clean", self.model)
        return call(image, times, labels, **conditions)


def load_thop(directory):
    directory = Path(directory).resolve()
    if not (directory / "thop").is_dir():
        raise ValueError(
            f"Install ultralytics-thop=={VERSION} --no-deps --target {directory}"
        )
    distributions = list(importlib.metadata.distributions(path=[str(directory)]))
    if not any(
        d.metadata["Name"] == "ultralytics-thop" and d.version == VERSION
        for d in distributions
    ):
        raise ValueError(f"Expected isolated ultralytics-thop=={VERSION}")
    sys.path.insert(0, str(directory))
    import thop

    if not Path(thop.__file__).resolve().is_relative_to(directory):
        raise ValueError(
            "Another thop package was imported before the isolated profiler"
        )
    return thop


def inputs(config, label, stage, device):
    from scripts.b2b_onnx_denoiser_infer_autoregressive_progress_bbox import (
        get_b2b_params,
        resolve_temporal_frame_step,
    )

    # define_G and the denoiser export construct the network from data_crop_size.
    # online_creation.crop_size_A is source context, not network resolution.
    data = config.get("data", {})
    frames = int(data.get("temporal_number_frames", 2))
    height = width = int(data.get("crop_size", 256))
    if frames < 1 or height < 1:
        raise ValueError("Network resolution and frame count must be positive")
    params = get_b2b_params(config, height)
    if params["object_ref_conditioning"]:
        raise ValueError(
            "Object-reference conditioning is not supported by this profiler yet"
        )
    channels = int(config.get("model", {}).get("input_nc", 3)) + int(
        params["mask_as_channel"]
    )
    if not 0 <= label < params["num_classes"]:
        raise ValueError("Label outside configured class range")
    image = torch.randn(1, frames, channels, height, width, device=device)
    conditions = {}
    if params["mask_size_conditioning"]:
        conditions["mask_size_cond"] = torch.full((1, frames, 6), 0.5, device=device)
    if params["mask_prediction"]:
        mode = torch.full(
            (1, frames), 2 if stage == "coarse" else 0, dtype=torch.long, device=device
        )
        mode[:, 0] = 0
        conditions["mask_precision_mode"] = mode
        conditions["mask_precision_severity"] = torch.zeros(1, frames, device=device)
    if params["temporal_frame_step_conditioning"]:
        conditions["temporal_frame_step"] = torch.full(
            (1, frames), resolve_temporal_frame_step(config), device=device
        )
    if params["global_context_conditioning"]:
        size = params["global_context_size"]
        conditions["global_context"] = torch.randn(
            1, frames, 3, size, size, device=device
        )
    return (
        image,
        torch.full((frames,), 0.5, device=device),
        torch.tensor([label], device=device),
        conditions,
    )


def measure(model, args, thop):
    # Known scalar/data-only leaves excluded explicitly; no parent attention hooks.
    from thop.profile import register_hooks

    exclusions = {
        "RMSNorm",
        "VisionRotaryEmbeddingFast",
        "SiLU",
        "GELU",
        "Dropout",
        "Dropout2d",
    }
    missing = sorted(
        {
            type(m).__name__
            for m in model.modules()
            if not list(m.children())
            and type(m).__name__ not in exclusions
            and not any(t in register_hooks for t in type(m).__mro__)
        }
    )
    counter = Products()
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad(), counter:
        native, _ = thop.profile(
            model, inputs=args, verbose=False, report_missing=False
        )
    total = float(native) + counter.ops
    complete = not missing and not counter.unsupported
    return dict(
        thop_ops=float(native),
        supplemental_baddbmm_macs=counter.ops,
        unsupported_functional_products=sorted(counter.unsupported),
        estimated_gflops=2 * total / 1e9 if complete else None,
        params=params,
        complete=complete,
        unsupported_leaf_modules=missing,
    )


def summary(name, details):
    base, coarse, refine = (details[k] for k in ("baseline", "coarse", "refinement"))

    def value(d):
        return d["estimated_gflops"]

    return [
        dict(
            model_set=name,
            mode="baseline",
            params=base["params"],
            coarse_gflops=value(base),
            refinement_gflops=None,
            representative_gflops=value(base),
        ),
        dict(
            model_set=name,
            mode="smartbrush_bbox",
            params=coarse["params"],
            coarse_gflops=value(coarse),
            refinement_gflops=None,
            representative_gflops=value(coarse),
        ),
        dict(
            model_set=name,
            mode="smartbrush_hybrid",
            params=coarse["params"],
            coarse_gflops=value(coarse),
            refinement_gflops=value(refine),
            representative_gflops=(
                (value(coarse) + value(refine))
                if coarse["complete"] and refine["complete"]
                else None
            ),
        ),
    ]


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="JSON manifest for multiple model sets"
    )
    parser.add_argument("--baseline", type=Path, help="Baseline .pth checkpoint")
    parser.add_argument("--smartbrush", type=Path, help="SmartBrush .pth checkpoint")
    parser.add_argument(
        "--label", type=int, help="Class label for direct checkpoint inputs (default 0)"
    )
    parser.add_argument("--model-set")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--thop-dir", type=Path, default=ROOT / "results/thop_deps")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        if args.baseline or args.smartbrush or args.label is not None:
            parser.error(
                "--config cannot be combined with --baseline, --smartbrush or --label"
            )
        sets = json.loads(args.config.read_text())["model_sets"]
        if args.model_set:
            sets = {args.model_set: sets[args.model_set]}
    else:
        if not (args.baseline and args.smartbrush):
            parser.error("provide both --baseline and --smartbrush, or use --config")
        item = {"label": args.label if args.label is not None else 0}
        for role, weights in (
            ("baseline", args.baseline),
            ("smartbrush", args.smartbrush),
        ):
            item[role] = {
                "weights": str(weights),
                "config": str(weights.with_name("train_config.json")),
            }
        sets = {args.model_set or "default": item}
    if not sets:
        raise ValueError("No model sets configured")
    for item in sets.values():
        for role in ("baseline", "smartbrush"):
            for key in ("weights", "config"):
                if not Path(item[role][key]).is_file():
                    raise FileNotFoundError(item[role][key])
            train = json.loads(Path(item[role]["config"]).read_text())
            if train.get("model_type") != "b2b":
                raise ValueError("Only B2B checkpoints are supported")
    thop = load_thop(args.thop_dir)
    if args.check_only:
        print(
            f"CHECK OK: {len(sets)} model sets; ultralytics-thop {VERSION}; no forwards run"
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir required for profiling")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from scripts.b2b_pth_denoiser_infer_autoregressive_progress_bbox import (
        load_pth_session,
    )

    device = torch.device(args.device)
    report = dict(
        policy="thop_2.2.0_plus_baddbmm_v1",
        torch_version=torch.__version__,
        thop_version=VERSION,
        convention="estimated FLOPs = 2 * (native THOP ops + supplemental baddbmm MACs)",
        scope="representative denoiser forwards; Hybrid is one coarse plus one refinement forward, not generation cost",
        exclusions="functional scalar arithmetic, functional normalization, activations and data movement; native THOP module rules retained",
        seed=42,
        model_sets={},
    )
    rows = []
    for name, item in sets.items():
        details = {}
        for role, stages in [
            ("baseline", ["baseline"]),
            ("smartbrush", ["coarse", "refinement"]),
        ]:
            spec = item[role]
            train = json.loads(Path(spec["config"]).read_text())
            session, _, _ = load_pth_session(
                spec["weights"], spec["config"], device, False
            )
            if role == "smartbrush" and not session.mask_prediction:
                raise ValueError("SmartBrush requires a mask-predicting checkpoint")
            adapter = Predict(session.model).eval()
            for stage in stages:
                torch.manual_seed(42)
                tensors = inputs(train, item["label"], stage, device)
                record = measure(adapter, tensors, thop)
                record.update(
                    weights_sha256=fingerprint(spec["weights"]),
                    config_sha256=fingerprint(spec["config"]),
                    weights=spec["weights"],
                    config=spec["config"],
                    label=item["label"],
                    input_shape=list(tensors[0].shape),
                    timestep=0.5,
                    conditioning={
                        k: dict(
                            shape=list(v.shape),
                            values=v.tolist() if v.numel() < 20 else None,
                        )
                        for k, v in tensors[3].items()
                    },
                )
                if not record["complete"]:
                    print(
                        "Incomplete profile:",
                        name,
                        stage,
                        "unsupported leaves",
                        record["unsupported_leaf_modules"],
                        "unsupported products",
                        record["unsupported_functional_products"],
                    )
                details[stage] = record
                print(
                    name,
                    stage,
                    record["estimated_gflops"],
                    "estimated GFLOPs",
                    flush=True,
                )
            del adapter, session
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.extend(summary(name, details))
        report["model_sets"][name] = details
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "details.json").write_text(json.dumps(report, indent=2) + "\n")
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Profiling complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
