"""Prepare an isolated full-state resume branch with sampler-aligned GAN training."""

import argparse
import json
from pathlib import Path


def prepare_branch(source_config, output_config, epoch="5", name=None):
    source_config = Path(source_config).resolve()
    config = json.loads(source_config.read_text())
    if config.get("model_type") != "b2b" or not config.get("alg", {}).get(
        "b2b_convrefiner_finetune"
    ):
        raise ValueError("Expected a B2B ConvRefiner finetuning configuration")
    source = (Path(config["checkpoints_dir"]).expanduser() / config["name"]).resolve()
    required = ["net_G_A", "net_D_refiner", "refiner_training"]
    if config["train"].get("G_ema", False):
        required.append("net_G_A_ema")
    for suffix in required:
        path = source / f"{epoch}_{suffix}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Full-state resume requires {path}")
    destination = Path(output_config).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    branch_name = name or config["name"] + f"_sampler_aligned_from_{epoch}"
    if Path(branch_name).name != branch_name or branch_name in (".", ".."):
        raise ValueError("Branch name must be a single directory name")
    run_dir = (Path(config["checkpoints_dir"]).expanduser() / branch_name).resolve()
    if run_dir == source or run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run {run_dir}")
    config["name"] = branch_name
    config["alg"].update(
        b2b_convrefiner_gan_schedule="sampler_aligned",
        b2b_convrefiner_gan_near_clean_prob=0.2,
        b2b_convrefiner_gan_timestep_radius=0.05,
        b2b_convrefiner_resume_from=str(source),
    )
    config["train"].update({"continue": True, "continue_from": "", "epoch": str(epoch)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_config", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--epoch", default="5")
    parser.add_argument("--name")
    args = parser.parse_args()
    print(prepare_branch(args.source_config, args.output_config, args.epoch, args.name))
