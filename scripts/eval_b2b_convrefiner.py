"""Evaluate a trained refiner against its own frozen backbone on held-out windows."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import create_dataloader, create_dataset, list_test_sets
from models.b2b_model import B2BModel
from options.train_options import TrainOptions
from util.convrefiner_metrics import evaluate_refiner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--raw", action="store_true", help="Evaluate non-EMA weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.train_config).read_text())
    device = torch.device(args.device)
    config["gpu_ids"] = str(device.index or 0) if device.type == "cuda" else "-1"
    opt = TrainOptions().parse_json(config, save_config=False, set_device=False)
    opt.isTrain = False
    opt.jg_dir = str(Path(__file__).resolve().parents[1])
    opt.total_iters = 0
    opt.use_cuda = device.type == "cuda"
    opt.train_G_ema = not args.raw
    opt.train_epoch = args.epoch
    opt.train_continue_from = ""
    opt.data_num_threads = 0
    opt.test_batch_size = 1
    opt.alg_b2b_perceptual_loss = ["LPIPS", "DISTS"]
    if opt.alg_b2b_convrefiner == "none":
        raise ValueError(
            "Evaluation requires a refiner-enabled checkpoint/configuration"
        )
    model = B2BModel(opt, 0)
    model.load_networks(args.epoch)
    reports = {}
    for name in list_test_sets(opt):
        dataset = create_dataset(opt, "test", name)
        loader = create_dataloader(opt, 0, dataset, 1)
        reports[name] = evaluate_refiner(model, loader, args.max_batches, args.seed)
    if not reports:
        raise ValueError("No held-out test manifests found in the configuration")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
