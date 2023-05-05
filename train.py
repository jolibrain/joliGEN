"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import argparse
import json
import os
import signal
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data import (
    create_dataloader,
    create_dataset,
    create_dataset_temporal,
    create_iterable_dataloader,
)
from models import create_model
from options.train_options import TrainOptions
from util.util import flatten_json
from util.visualizer import Visualizer


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def optim(opt, params, lr, betas):
    print("Using ", opt.train_optim, " as optimizer")
    if opt.train_optim == "adam":
        return torch.optim.Adam(params, lr, betas)
    elif opt.train_optim == "radam":
        return torch.optim.RAdam(params, lr, betas)
    elif opt.train_optim == "adamw":
        return torch.optim.AdamW(params, lr, betas)


def signal_handler(sig, frame):
    dist.destroy_process_group()


def train_gpu(rank, world_size, opt, trainset, trainset_temporal):
    if not opt.warning_mode:
        warnings.simplefilter("ignore")

    if opt.use_cuda:
        torch.cuda.set_device(opt.gpu_ids[rank])

    signal.signal(signal.SIGINT, signal_handler)  # to really kill the process
    signal.signal(signal.SIGTERM, signal_handler)
    if len(opt.gpu_ids) > 1:
        setup(rank, world_size, opt.ddp_port)

    dataloader = create_dataloader(
        opt, rank, trainset, batch_size=opt.train_batch_size
    )  # create a dataset given opt.dataset_mode and other options

    use_temporal = ("temporal" in opt.D_netDs) or opt.train_temporal_criterion

    if use_temporal:
        dataloader_temporal = create_iterable_dataloader(
            opt, rank, trainset_temporal, batch_size=opt.train_batch_size
        )

    trainset_size = len(trainset)  # get the number of images in the trainset.

    if rank == 0:
        if opt.train_compute_metrics_test:
            testset = create_dataset(opt, phase="test")
            print("The number of testing images = %d" % len(testset))

            dataloader_test = create_dataloader(
                opt, rank, testset, batch_size=opt.test_batch_size
            )  # create a dataset given opt.dataset_mode and other options

            if use_temporal:
                testset_temporal = create_dataset_temporal(opt, phase="test")

                dataloader_test_temporal = create_iterable_dataloader(
                    opt, rank, testset_temporal, batch_size=opt.test_batch_size
                )
            else:
                dataloader_test_temporal = None
        else:
            dataloader_test = None
            dataloader_test_temporal = None

    opt.optim = optim  # set optimizer
    model = create_model(opt, rank)  # create a model given opt.model and other options

    if hasattr(model, "data_dependent_initialize"):
        data = next(iter(dataloader))
        model.data_dependent_initialize(data)

    rank_0 = rank == 0

    if rank_0:
        model.init_metrics(dataloader, dataloader_test)

    model.setup(opt)  # regular setup: load and print networks; create schedulers

    model.use_temporal = use_temporal

    if opt.use_cuda:
        if len(opt.gpu_ids) > 1:
            model.parallelize(rank)
        else:
            model.single_gpu()

    if rank_0:
        visualizer = Visualizer(
            opt
        )  # create a visualizer that display/save images and plots

        visualizer.print_networks(nets=model.get_nets(), verbose=opt.output_verbose)

        model.print_flop()

        if opt.train_continue:
            opt.train_epoch_count = visualizer.load_data()

        model.print_flop()

    total_iters = 0  # the total number of training iterations

    if rank_0 and opt.output_display_networks:
        data = next(iter(dataloader))
        for path in model.save_networks_img(data):
            visualizer.display_img(path + ".png")

    for epoch in range(
        opt.train_epoch_count, opt.train_n_epochs + opt.train_n_epochs_decay + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        if rank_0:
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        if use_temporal:
            dataloaders = zip(dataloader, dataloader_temporal)
        else:
            dataloaders = zip(dataloader)

        for i, data_list in enumerate(
            dataloaders
        ):  # inner loop (minibatch) within one epoch
            data = data_list[0]
            if use_temporal:
                temporal_data = data_list[1]

            iter_start_time = time.time()  # timer for computation per iteration
            t_data_mini_batch = iter_start_time - iter_data_time

            model.set_input(data)  # unpack data from dataloader and apply preprocessing
            if use_temporal:
                model.set_input_temporal(temporal_data)

            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            t_comp = (time.time() - iter_start_time) / opt.train_batch_size

            batch_size = model.get_current_batch_size() * len(opt.gpu_ids)
            total_iters += batch_size
            epoch_iter += batch_size

            if (
                total_iters % opt.output_print_freq < batch_size
            ):  # print training losses and save logging information to the disk
                losses = model.get_current_losses()

                float_losses = {}

                for name, value in losses.items():
                    if len(opt.gpu_ids) > 1:
                        torch.distributed.all_reduce(
                            value, op=torch.distributed.ReduceOp.SUM
                        )  # loss value is summed accross gpus

                    float_losses[name] = float(value / len(opt.gpu_ids))

                losses = float_losses

                if rank_0:
                    visualizer.print_current_losses(
                        epoch, epoch_iter, losses, t_comp, t_data_mini_batch
                    )
                    if opt.output_display_id > 0:
                        visualizer.plot_current_losses(
                            epoch, float(epoch_iter) / trainset_size, losses
                        )

            if rank_0:
                if (
                    total_iters % opt.output_display_freq < batch_size
                ):  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.output_update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(
                        model.get_current_visuals(),
                        epoch,
                        save_result,
                        params=model.get_display_param(),
                        first=(total_iters == batch_size),
                    )

                if (
                    total_iters % opt.train_save_latest_freq < batch_size
                ):  # cache our latest model every <save_latest_freq> iterations
                    print(
                        "saving the latest model (epoch %d, total_iters %d)"
                        % (epoch, total_iters)
                    )

                    model.save_networks("latest")
                    model.export_networks("latest")

                    if opt.train_save_by_iter:
                        save_suffix = "iter_%d" % total_iters
                        model.save_networks(save_suffix)
                        model.export_networks(save_suffix)

                if total_iters % opt.train_metrics_every < batch_size and (
                    opt.train_compute_metrics or opt.train_compute_metrics_test
                ):
                    with torch.no_grad():
                        if opt.train_compute_metrics:
                            model.compute_metrics()

                        if opt.train_compute_metrics_test:
                            if use_temporal:
                                dataloaders_test = zip(
                                    dataloader_test, dataloader_test_temporal
                                )
                            else:
                                dataloaders_test = zip(dataloader_test)

                            model.compute_metrics_test(
                                dataloaders_test, epoch, total_iters
                            )

                    if opt.output_display_id > 0:
                        metrics = model.get_current_metrics()
                        visualizer.plot_current_metrics(
                            epoch, float(epoch_iter) / trainset_size, metrics
                        )

                if (
                    total_iters % opt.train_D_accuracy_every < batch_size
                    and opt.train_compute_D_accuracy
                ):
                    model.compute_D_accuracy()
                    if opt.output_display_id > 0:
                        accuracies = model.get_current_D_accuracies()
                        visualizer.plot_current_D_accuracies(
                            epoch, float(epoch_iter) / trainset_size, accuracies
                        )

                if (
                    total_iters % opt.output_display_freq < batch_size
                    and opt.dataaug_APA
                ):
                    if opt.output_display_id > 0:
                        p = model.get_current_APA_prob()
                        visualizer.plot_current_APA_prob(
                            epoch, float(epoch_iter) / trainset_size, p
                        )

                if (
                    total_iters % opt.train_mask_miou_every < batch_size
                    and opt.train_mask_compute_miou
                ):
                    model.compute_miou()
                    if opt.output_display_id > 0:
                        miou = model.get_current_miou()
                        visualizer.plot_current_miou(
                            epoch, float(epoch_iter) / trainset_size, miou
                        )

                iter_data_time = time.time()

        if (
            epoch % opt.train_save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            if rank_0:
                print(
                    "saving the model at the end of epoch %d, iters %d"
                    % (epoch, total_iters)
                )
                model.save_networks("latest")
                model.save_networks(epoch)

                model.export_networks("latest")
                model.export_networks(epoch)

        if rank_0:
            print(
                "End of epoch %d / %d \t Time Taken: %d sec"
                % (
                    epoch,
                    opt.train_n_epochs + opt.train_n_epochs_decay,
                    time.time() - epoch_start_time,
                )
            )
        model.update_learning_rate()  # update learning rates at the end of every epoch.

    ###Let's compute final FID
    if rank_0 and opt.train_compute_metrics_test:
        cur_fid = model.compute_metrics_test()
        path_json = os.path.join(opt.checkpoints_dir, opt.name, "eval_results.json")

        if os.path.exists(path_json):
            with open(path_json, "r") as loadfile:
                data = json.load(loadfile)

        with open(path_json, "w+") as outfile:
            data = {}
            data["fid_%s_img_%s_epochs" % (opt.data_max_dataset_size, epoch)] = float(
                cur_fid.item()
            )
            json.dump(data, outfile)

    if rank_0:
        print("End of training")


def launch_training(opt=None):
    signal.signal(signal.SIGINT, signal_handler)  # to really kill the process
    if opt is None:
        opt = TrainOptions().parse()  # get training options
    opt.jg_dir = os.path.join("/".join(__file__.split("/")[:-1]))
    world_size = len(opt.gpu_ids)

    if not opt.warning_mode:
        warnings.simplefilter("ignore")

    trainset = create_dataset(opt, phase="train")
    print("The number of training images = %d" % len(trainset))

    use_temporal = ("temporal" in opt.D_netDs) or opt.train_temporal_criterion

    if use_temporal:
        trainset_temporal = create_dataset_temporal(opt, phase="train")
    else:
        trainset_temporal = None

    if opt.with_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    opt.use_cuda = torch.cuda.is_available() and opt.gpu_ids and opt.gpu_ids[0] >= 0
    if opt.use_cuda:
        mp.spawn(
            train_gpu,
            args=(world_size, opt, trainset, trainset_temporal),
            nprocs=world_size,
            join=True,
        )
    else:
        train_gpu(0, world_size, opt, trainset, trainset_temporal)


def compute_test_metrics(model, dataloader):

    return metrics


def get_override_options_names(remaining_args):
    return_options_names = []

    for arg in remaining_args:
        if arg.startswith("--"):
            return_options_names.append(arg[2:])

    return return_options_names


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(add_help=False)

    main_parser.add_argument(
        "--config_json", type=str, default="", help="path to json config"
    )

    main_opt, remaining_args = main_parser.parse_known_args()

    if main_opt.config_json != "":
        override_options_names = get_override_options_names(remaining_args)

        if not "--dataroot" in remaining_args:
            remaining_args += ["--dataroot", "unused"]
        override_options_json = flatten_json(
            TrainOptions().parse_to_json(remaining_args)
        )

        with open(main_opt.config_json, "r") as jsonf:
            train_json = flatten_json(json.load(jsonf))

        for name in override_options_names:
            train_json[name] = override_options_json[name]

        opt = TrainOptions().parse_json(train_json)

        print("%s config file loaded" % main_opt.config_json)
    else:
        opt = None

    launch_training(opt)
