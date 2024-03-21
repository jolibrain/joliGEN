import argparse
import os
import torch
import random
import numpy as np
import time
import json

from data import (
    create_dataloader,
    create_dataset,
    create_dataset_temporal,
    create_iterable_dataloader,
)
from models import create_model
from util.parser import get_opt
from util.util import MAX_INT
from models.modules.diffusion_utils import set_new_noise_schedule


def launch_testing(opt, main_opt):
    rank = 0

    opt.jg_dir = os.path.join("/".join(__file__.split("/")[:-1]))
    opt.use_cuda = torch.cuda.is_available() and opt.gpu_ids and opt.gpu_ids[0] >= 0
    if opt.use_cuda:
        torch.cuda.set_device(opt.gpu_ids[rank])
    opt.isTrain = False

    testset = create_dataset(opt, phase="test")
    print("The number of testing images = %d" % len(testset))
    opt.num_test_images = len(testset)
    opt.train_nb_img_max_fid = min(opt.train_nb_img_max_fid, len(testset))

    dataloader_test = create_dataloader(
        opt, rank, testset, batch_size=opt.test_batch_size
    )  # create a dataset given opt.dataset_mode and other options

    use_temporal = ("temporal" in opt.D_netDs) or opt.train_temporal_criterion

    if use_temporal:
        testset_temporal = create_dataset_temporal(opt, phase="test")

        dataloader_test_temporal = create_iterable_dataloader(
            opt, rank, testset_temporal, batch_size=opt.test_batch_size
        )
    else:
        dataloader_test_temporal = None

    model = create_model(opt, rank)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # sampling options
    if main_opt.sampling_steps is not None:
        model.netG_A.denoise_fn.model.beta_schedule["test"][
            "n_timestep"
        ] = main_opt.sampling_steps
        if main.opt.model_type == "palette":
            set_new_noise_schedule(model.netG_A.denoise_fn.model, "test")
    if main_opt.sampling_method is not None:
        model.netG_A.set_new_sampling_method(main_opt.sampling_method)
    if main_opt.ddim_num_steps is not None:
        model.ddim_num_steps = main_opt.ddim_num_steps
    if main_opt.ddim_eta is not None:
        model.ddim_eta = main_opt.ddim_eta

    model.use_temporal = use_temporal
    model.eval()
    if opt.use_cuda:
        model.single_gpu()
    model.init_metrics(dataloader_test)

    if use_temporal:
        dataloaders_test = zip(dataloader_test, dataloader_test_temporal)
    else:
        dataloaders_test = zip(dataloader_test)

    epoch = "test"
    total_iters = "test"
    with torch.no_grad():
        model.compute_metrics_test(dataloaders_test, epoch, total_iters)

    metrics = model.get_current_metrics([""])
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    metrics_dir = os.path.join(opt.test_model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, time.strftime("%Y%m%d-%H%M%S") + ".json")
    with open(metrics_file, "w") as f:
        f.write(json.dumps(metrics, indent=4))
    print("metrics written to:", metrics_file)


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument(
        "--test_model_dir", type=str, required=True, help="path to model directory"
    )
    main_parser.add_argument(
        "--test_epoch",
        type=str,
        default="latest",
        help="which epoch to load? set to latest to use latest cached model",
    )
    main_parser.add_argument(
        "--test_metrics_list",
        type=str,
        nargs="*",
        choices=["FID", "KID", "MSID", "PSNR", "LPIPS"],
        default=["FID", "KID", "MSID", "PSNR", "LPIPS"],
    )
    main_parser.add_argument(
        "--test_nb_img",
        type=int,
        default=MAX_INT,
        help="Number of samples to compute metrics. If the dataset directory contains more, only a subset is used.",
    )
    main_parser.add_argument(
        "--test_batch_size", type=int, default=1, help="input batch size"
    )
    main_parser.add_argument(
        "--test_seed", type=int, default=42, help="seed to use for tests"
    )
    main_parser.add_argument(
        "--sampling_steps", type=int, help="number of sampling steps"
    )
    main_parser.add_argument(
        "--sampling_method",
        type=str,
        choices=["ddpm", "ddim"],
        help="choose the sampling method between ddpm and ddim",
    )
    main_parser.add_argument(
        "--ddim_num_steps",
        type=int,
        help="number of steps for ddim sampling method",
    )
    main_parser.add_argument(
        "--ddim_eta",
        type=float,
        help="eta parameter for ddim variance",
    )

    main_opt, remaining_args = main_parser.parse_known_args()
    main_opt.config_json = os.path.join(main_opt.test_model_dir, "train_config.json")

    opt = get_opt(main_opt, remaining_args)

    # override global options with local test options
    opt.train_compute_metrics_test = True
    opt.test_model_dir = main_opt.test_model_dir
    opt.train_epoch = main_opt.test_epoch
    opt.train_metrics_list = main_opt.test_metrics_list
    opt.train_nb_img_max_fid = main_opt.test_nb_img
    opt.test_batch_size = main_opt.test_batch_size
    opt.alg_diffusion_generate_per_class = False

    random.seed(main_opt.test_seed)
    torch.manual_seed(main_opt.test_seed)
    np.random.seed(main_opt.test_seed)

    launch_testing(opt, main_opt)
