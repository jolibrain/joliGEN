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
import time
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
from util.visualizer import Visualizer
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import signal
import torch
import json


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


def train_gpu(rank, world_size, opt, dataset):
    torch.cuda.set_device(opt.gpu_ids[rank])
    signal.signal(signal.SIGINT, signal_handler)  # to really kill the process
    signal.signal(signal.SIGTERM, signal_handler)
    if len(opt.gpu_ids) > 1:
        setup(rank, world_size, opt.ddp_port)
    dataloader = create_dataloader(
        opt, rank, dataset
    )  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    opt.optim = optim  # set optimizer
    model = create_model(opt, rank)  # create a model given opt.model and other options

    if hasattr(model, "data_dependent_initialize"):
        data = next(iter(dataloader))
        model.data_dependent_initialize(data)

    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if len(opt.gpu_ids) > 1:
        model.parallelize(rank)
    else:
        model.single_gpu()

    if rank == 0:
        visualizer = Visualizer(
            opt
        )  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    if rank == 0 and opt.train_compute_fid_val:
        model.real_A_val, model.real_B_val = dataset.get_validation_set(
            opt.train_pool_size
        )
        model.real_A_val, model.real_B_val = (
            model.real_A_val.to(model.device),
            model.real_B_val.to(model.device),
        )

    if rank == 0 and opt.output_display_networks:
        data = next(iter(dataloader))
        for path in model.save_networks_img(data):
            visualizer.display_img(path + ".png")

    for epoch in range(
        opt.train_epoch_count, opt.train_n_epochs + opt.train_n_epochs_decay + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        if rank == 0:
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataloader):  # inner loop (minibatch) within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            t_data_mini_batch = iter_start_time - iter_data_time

            model.set_input(data)  # unpack data from dataloader and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            t_comp = (time.time() - iter_start_time) / opt.train_batch_size

            batch_size = model.get_current_batch_size() * len(opt.gpu_ids)
            total_iters += batch_size
            epoch_iter += batch_size

            if rank == 0:
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
                    )

                if (
                    total_iters % opt.output_print_freq < batch_size
                ):  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    visualizer.print_current_losses(
                        epoch, epoch_iter, losses, t_comp, t_data_mini_batch
                    )
                    if opt.output_display_id > 0:
                        visualizer.plot_current_losses(
                            epoch, float(epoch_iter) / dataset_size, losses
                        )

                if (
                    total_iters % opt.train_save_latest_freq < batch_size
                ):  # cache our latest model every <save_latest_freq> iterations
                    print(
                        "saving the latest model (epoch %d, total_iters %d)"
                        % (epoch, total_iters)
                    )
                    save_suffix = (
                        "iter_%d" % total_iters if opt.train_save_by_iter else "latest"
                    )
                    model.save_networks(save_suffix)
                    model.export_networks(save_suffix)

                if (
                    total_iters % opt.train_fid_every < batch_size
                    and opt.train_compute_fid
                ):
                    model.compute_fid(epoch, total_iters)
                    if opt.output_display_id > 0:
                        fids = model.get_current_fids()
                        visualizer.plot_current_fid(
                            epoch, float(epoch_iter) / dataset_size, fids
                        )

                if (
                    total_iters % opt.train_D_accuracy_every < batch_size
                    and opt.train_compute_D_accuracy
                ):
                    model.compute_D_accuracy()
                    if opt.output_display_id > 0:
                        accuracies = model.get_current_D_accuracies()
                        visualizer.plot_current_D_accuracies(
                            epoch, float(epoch_iter) / dataset_size, accuracies
                        )

                if (
                    total_iters % opt.output_display_freq < batch_size
                    and opt.dataaug_APA
                ):
                    if opt.output_display_id > 0:
                        p = model.get_current_APA_prob()
                        visualizer.plot_current_APA_prob(
                            epoch, float(epoch_iter) / dataset_size, p
                        )

                iter_data_time = time.time()

        if (
            epoch % opt.train_save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            if rank == 0:
                print(
                    "saving the model at the end of epoch %d, iters %d"
                    % (epoch, total_iters)
                )
                model.save_networks("latest")
                model.save_networks(epoch)

                model.export_networks("latest")
                model.export_networks(epoch)

        if rank == 0:
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
    if rank == 0 and opt.train_compute_fid_val:
        cur_fid = model.compute_fid_val()
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

    if rank == 0:
        print("End of training")


def launch_training(opt=None):
    signal.signal(signal.SIGINT, signal_handler)  # to really kill the process
    if opt is None:
        opt = TrainOptions().parse()  # get training options
    opt.jg_dir = os.path.join("/".join(__file__.split("/")[:-1]))
    world_size = len(opt.gpu_ids)

    dataset = create_dataset(opt)
    print("The number of training images = %d" % len(dataset))

    mp.spawn(train_gpu, args=(world_size, opt, dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    launch_training()
