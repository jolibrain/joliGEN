"""Opt-in ConvRefiner training lifecycle for B2BModel."""

import copy
import math
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .modules.convrefiner import (
    RefinerDiscriminator,
    load_refiner_generator_state,
    weighted_patch_mean,
)


def add_refiner_options(parser):
    parser.add_argument(
        "--alg_b2b_convrefiner", choices=["none", "spatial", "temporal"], default="none"
    )
    parser.add_argument("--alg_b2b_convrefiner_width", type=int, default=64)
    parser.add_argument("--alg_b2b_convrefiner_finetune", action="store_true")
    parser.add_argument("--alg_b2b_convrefiner_source_ema", action="store_true")
    parser.add_argument("--alg_b2b_convrefiner_gan_weight", type=float, default=0.02)
    parser.add_argument(
        "--alg_b2b_convrefiner_gan_schedule",
        choices=["near_clean", "sampler_aligned"],
        default="near_clean",
        help="Adversarial-batch timestep distribution; inference is unchanged",
    )
    parser.add_argument(
        "--alg_b2b_convrefiner_gan_near_clean_prob",
        type=float,
        default=0.2,
        help="Near-clean batch probability in sampler_aligned mode",
    )
    parser.add_argument(
        "--alg_b2b_convrefiner_gan_timestep_radius",
        type=float,
        default=0.05,
        help="Uniform jitter around each sampler's last evaluation time",
    )
    parser.add_argument(
        "--alg_b2b_convrefiner_resume_from",
        default="",
        help="Resume full refiner training state from another run without changing save directory",
    )
    parser.add_argument(
        "--alg_b2b_convrefiner_eval_batches",
        type=int,
        default=0,
        help="Fixed-seed baseline/refiner evaluation batches per test set each epoch; 0 disables",
    )
    return parser


def validate_refiner_options(opt):
    schedule = getattr(opt, "alg_b2b_convrefiner_gan_schedule", "near_clean")
    if schedule not in ("near_clean", "sampler_aligned"):
        raise ValueError("Unknown ConvRefiner GAN schedule")
    probability = getattr(opt, "alg_b2b_convrefiner_gan_near_clean_prob", 0.2)
    radius = getattr(opt, "alg_b2b_convrefiner_gan_timestep_radius", 0.05)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("ConvRefiner near-clean probability must be in [0, 1]")
    if not math.isfinite(radius) or not 0 <= radius < 0.5:
        raise ValueError("ConvRefiner timestep radius must be in [0, 0.5)")
    if schedule == "sampler_aligned":
        steps = getattr(opt, "alg_b2b_denoise_timesteps", [])
        steps = [steps] if isinstance(steps, int) else steps
        if not steps or any(not isinstance(s, int) or s < 2 for s in steps):
            raise ValueError(
                "Sampler-aligned ConvRefiner requires sampling step counts >= 2"
            )
    if getattr(opt, "alg_b2b_convrefiner_resume_from", "") and getattr(
        opt, "isTrain", False
    ):
        if not getattr(opt, "alg_b2b_convrefiner_finetune", False) or not getattr(
            opt, "train_continue", False
        ):
            raise ValueError(
                "ConvRefiner resume_from requires finetuning and train_continue"
            )
        if getattr(opt, "train_continue_from", ""):
            raise ValueError(
                "ConvRefiner resume_from cannot be combined with fresh train_continue_from"
            )
    mode = getattr(opt, "alg_b2b_convrefiner", "none")
    if mode != "none":
        if opt.G_netG not in ("vit", "vit_vid"):
            raise ValueError("ConvRefiner supports vit/vit_vid B2B backbones")
        width = getattr(opt, "alg_b2b_convrefiner_width", 64)
        if width < 32 or width % 32:
            raise ValueError("ConvRefiner width must be a positive multiple of 32")
        if getattr(opt, "model_output_nc", 3) != 3:
            raise ValueError("ConvRefiner currently supports RGB output only")
    if getattr(opt, "alg_b2b_convrefiner_finetune", False):
        if mode == "none":
            raise ValueError(
                "ConvRefiner finetuning requires a spatial or temporal refiner"
            )
        if getattr(opt, "alg_b2b_lora", False):
            raise ValueError(
                "ConvRefiner finetuning cannot be combined with LoRA training"
            )
        if getattr(opt, "isTrain", False):
            if not (
                getattr(opt, "train_continue_from", "")
                or getattr(opt, "train_continue", False)
            ):
                raise ValueError(
                    "ConvRefiner finetuning requires train_continue_from or train_continue"
                )
            if getattr(opt, "train_finetune", False):
                raise ValueError(
                    "Use train_continue_from, not train_finetune (which suppresses load errors)"
                )
            if getattr(opt, "with_torch_compile", False):
                raise ValueError(
                    "ConvRefiner finetuning currently requires with_torch_compile=false"
                )
            if getattr(opt, "alg_b2b_minsnr", False):
                raise ValueError(
                    "ConvRefiner finetuning does not support the incomplete B2B min-SNR path"
                )
            if getattr(opt, "data_crop_size", 256) < 32:
                raise ValueError(
                    "ConvRefiner PatchGAN training requires crops of at least 32 pixels"
                )
    if getattr(opt, "alg_b2b_convrefiner_gan_weight", 0.02) < 0:
        raise ValueError("ConvRefiner GAN weight must be nonnegative")


class B2BRefinerTrainingMixin:
    def _refiner_training(self):
        return (
            bool(getattr(self.opt, "alg_b2b_convrefiner_finetune", False))
            and self.isTrain
        )

    def _init_refiner_training(self):
        if not self._refiner_training():
            return
        self.netD_refiner = RefinerDiscriminator(self.opt.model_output_nc).to(
            self.device
        )
        self.model_names.append("D_refiner")
        self.optimizer_D_refiner = self.opt.optim(
            self.opt,
            self.netD_refiner.parameters(),
            lr=self.opt.train_D_lr,
            betas=(self.opt.train_beta1, self.opt.train_beta2),
            weight_decay=self.opt.train_optim_weight_decay,
            eps=self.opt.train_optim_eps,
        )
        self.optimizers.append(self.optimizer_D_refiner)
        self._refiner_adversarial_loss_name = (
            "G_refiner_adv"
            if getattr(self.opt, "alg_b2b_convrefiner_gan_schedule", "near_clean")
            == "sampler_aligned"
            else "G_refiner_near"
        )
        self.loss_names = [
            "G_tot",
            "G_refiner_all",
            self._refiner_adversarial_loss_name,
            "G_refiner_gan",
            "D_refiner",
        ]
        self.loss_names_G = self.loss_names[:-1].copy()
        self._refiner_accumulated = 0
        self._refiner_log_sums = {}
        self.refiner_optimizer_steps = 0
        self.refiner_resume_batch = 0
        self._refiner_resume_rng = None
        for name in ("criterionLPIPS", "criterionDISTS"):
            if hasattr(self, name):
                getattr(self, name).requires_grad_(False).eval()

    def set_requires_grad(self, nets, requires_grad=False):
        super().set_requires_grad(nets, requires_grad)
        if self._refiner_training():
            for net in nets if isinstance(nets, list) else [nets]:
                net = self._unwrap_net(net)
                if hasattr(net, "convrefiner"):
                    net.b2b_model.requires_grad_(False)
                    net.b2b_model.eval()

    def parallelize(self, rank):
        if not self._refiner_training():
            return super().parallelize(rank)
        for name in self.model_names:
            net = getattr(self, "net" + name).to(self.gpu_ids[rank])
            self.set_requires_grad(net, True)
            setattr(
                self,
                "net" + name,
                nn.parallel.DistributedDataParallel(
                    net,
                    device_ids=[self.gpu_ids[rank]],
                    broadcast_buffers=False,
                ),
            )

    def _refiner_adversarial_timestep_range(self):
        """Pick a band per microbatch; samples draw independently within that band.

        Python RNG is already checkpointed per rank. No additional forward/backward
        passes are introduced, and all-t reconstruction keeps its existing sampler.
        """
        if (
            getattr(self.opt, "alg_b2b_convrefiner_gan_schedule", "near_clean")
            == "near_clean"
        ):
            return (0.9, 0.98)
        if random.random() < getattr(
            self.opt, "alg_b2b_convrefiner_gan_near_clean_prob", 0.2
        ):
            return (0.9, 0.98)
        steps = self.opt.alg_b2b_denoise_timesteps
        steps = [steps] if isinstance(steps, int) else steps
        center = 1.0 - 1.0 / random.choice(sorted(set(steps)))
        radius = getattr(self.opt, "alg_b2b_convrefiner_gan_timestep_radius", 0.05)
        return (max(0.0, center - radius), min(1.0, center + radius))

    def _refiner_schedule_metadata(self):
        return {
            "schedule": getattr(
                self.opt, "alg_b2b_convrefiner_gan_schedule", "near_clean"
            ),
            "near_clean_probability": getattr(
                self.opt, "alg_b2b_convrefiner_gan_near_clean_prob", 0.2
            ),
            "timestep_radius": getattr(
                self.opt, "alg_b2b_convrefiner_gan_timestep_radius", 0.05
            ),
            "sampling_steps": getattr(self.opt, "alg_b2b_denoise_timesteps", None),
        }

    def _refiner_forward(self, near=False):
        kwargs = {}
        for argument, attribute in (
            ("temporal_frame_step", "temporal_frame_step"),
            ("global_context", "global_context"),
            ("object_refs", "object_refs"),
            ("mask_precision_mode", "mask_precision_mode"),
            ("mask_precision_severity", "mask_precision_severity"),
        ):
            if getattr(self, attribute, None) is not None:
                kwargs[argument] = getattr(self, attribute)
        return self.netG_A(
            self.gt_image,
            self.mask,
            self.cond_image,
            label=self.label_cls if self.num_classes > 0 else None,
            use_gt=getattr(self, "use_gt", None),
            ref_idx=getattr(self, "ref_idx", None),
            timestep_range=self._refiner_adversarial_timestep_range() if near else None,
            return_training_details=True,
            **kwargs,
        )

    def _refiner_reconstruction(self, output):
        pred, target = output["v_pred"], output["v"]
        mask = (
            self.mask.clamp(0, 1)
            if self.mask is not None
            else torch.ones_like(pred[..., :1, :, :])
        )
        if self.opt.alg_b2b_loss_masked_region_only:
            if "multiscale" in self.opt.alg_b2b_loss:
                loss = self.loss_fn(pred, target, mask=mask, masked_region_only=True)
            else:
                loss = self._masked_region_loss(pred, target, mask)
        else:
            loss = self.loss_fn(pred * mask, target * mask)
        if isinstance(loss, dict):
            loss = sum(loss.values())
        loss = loss * self.opt.alg_diffusion_lambda_G
        if self.opt.alg_b2b_perceptual_loss != [""]:
            # Flatten video into frames and exclude references/empty masks.
            image = output["x_pred"].reshape(-1, *pred.shape[-3:])
            truth = self.gt_image.reshape_as(image)
            frame_mask = mask.reshape(-1, 1, *pred.shape[-2:])
            valid = (output["t"].flatten() >= 0.3) & (frame_mask.flatten(1).sum(1) > 0)
            if valid.any():
                lpips, dists = self._compute_perceptual_losses(
                    truth[valid] * frame_mask[valid],
                    image[valid] * frame_mask[valid],
                )
                loss = loss + self.opt.alg_b2b_lambda_perceptual * (lpips + dists)
        return loss

    def _refiner_disc_batch(self, output):
        fake = output["x_pred"]
        real = self.gt_image
        mask = (
            self.mask.clamp(0, 1)
            if self.mask is not None
            else torch.ones_like(real[..., :1, :, :])
        )
        # forward() already projects known pixels; repeat explicitly for clarity.
        fake = fake * mask + real * (1 - mask)
        return (
            fake.reshape(-1, *fake.shape[-3:]),
            real.reshape(-1, *real.shape[-3:]),
            mask.reshape(-1, 1, *mask.shape[-2:]),
            output["t"].flatten(),
        )

    def optimize_parameters(self):
        if not self._refiner_training():
            return super().optimize_parameters()
        self.niter += 1
        self._refiner_accumulated += 1
        sync = self._refiner_accumulated == self.opt.train_iter_size
        self.set_requires_grad(self.netG_A, True)
        self.set_requires_grad(self.netD_refiner, False)
        # Bypass DDP for G's frozen discriminator so its reducer is not armed.
        discriminator = self._unwrap_net(self.netD_refiner)

        def no_sync(net, synchronize):
            return (
                net.no_sync()
                if hasattr(net, "no_sync") and not synchronize
                else nullcontext()
            )

        with no_sync(self.netG_A, False):
            with torch.autocast(self.device.type, enabled=self.with_amp):
                all_output = self._refiner_forward()
                all_loss = self._refiner_reconstruction(all_output)
            self.scaler.scale(all_loss / self.opt.train_iter_size).backward()
        del all_output
        with no_sync(self.netG_A, sync):
            with torch.autocast(self.device.type, enabled=self.with_amp):
                near_output = self._refiner_forward(near=True)
                near_loss = self._refiner_reconstruction(near_output)
                fake, real, mask, timesteps = self._refiner_disc_batch(near_output)
                weights = discriminator.patch_weights(mask)
                gan_loss = -weighted_patch_mean(discriminator(fake, timesteps), weights)
                weighted_gan = self.opt.alg_b2b_convrefiner_gan_weight * gan_loss
                near_total = 0.25 * near_loss + weighted_gan
            self.scaler.scale(near_total / self.opt.train_iter_size).backward()
        fake = fake.detach()
        del near_output
        self.set_requires_grad(self.netD_refiner, True)
        with no_sync(self.netD_refiner, sync):
            with torch.autocast(self.device.type, enabled=self.with_amp):
                scores = self.netD_refiner(
                    torch.cat([real, fake]), torch.cat([timesteps, timesteps])
                )
                real_scores, fake_scores = scores.chunk(2)
                d_loss = weighted_patch_mean(
                    F.relu(1 - real_scores) + F.relu(1 + fake_scores), weights
                )
            self.scaler.scale(d_loss / self.opt.train_iter_size).backward()
        values = {
            "G_tot": all_loss + near_total,
            "G_refiner_all": all_loss,
            self._refiner_adversarial_loss_name: near_loss,
            "G_refiner_gan": weighted_gan,
            "D_refiner": d_loss,
        }
        for name, value in values.items():
            value = value.detach()
            setattr(self, "loss_" + name, value)
            self._refiner_log_sums[name] = self._refiner_log_sums.get(name, 0) + value
        if sync:
            self._step_refiner_optimizers()

    def _step_refiner_optimizers(self):
        count = self._refiner_accumulated
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
        previous_scale = self.scaler.get_scale()
        self.scaler.update()
        if self.scaler.get_scale() >= previous_scale:
            self.refiner_optimizer_steps += 1
            if self.opt.train_G_ema:
                self.ema_step("G_A")
        for name, total in self._refiner_log_sums.items():
            setattr(self, "loss_" + name + "_avg", total / count)
        self._refiner_accumulated = 0
        self._refiner_log_sums = {}

    def finish_refiner_epoch(self):
        """Flush a short final accumulation group on all ranks before saving."""
        if not self._refiner_training() or not self._refiner_accumulated:
            return
        factor = self.opt.train_iter_size / self._refiner_accumulated
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(parameter.grad)
                            parameter.grad.div_(torch.distributed.get_world_size())
                        parameter.grad.mul_(factor)
        self._step_refiner_optimizers()

    @staticmethod
    def _refiner_rng_state():
        return {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }

    def prepare_refiner_checkpoint(self, epoch, next_batch, end_epoch=False):
        if not self._refiner_training() or self._refiner_accumulated:
            return
        rng = self._refiner_rng_state()
        rngs = [rng]
        if torch.distributed.is_initialized():
            rngs = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rngs, rng)
        self._refiner_checkpoint_cursor = (
            epoch + int(end_epoch),
            0 if end_epoch else next_batch,
        )
        self._refiner_checkpoint_rngs = rngs

    def restore_refiner_rng(self):
        rng = self._refiner_resume_rng
        if rng is None:
            return
        torch.set_rng_state(rng["torch"].cpu())
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"].cpu())
        self._refiner_resume_rng = None

    def load_networks(self, epoch, load_dir=None):
        enabled = getattr(self.opt, "alg_b2b_convrefiner", "none") != "none"
        if not enabled:
            return super().load_networks(epoch, load_dir=load_dir)
        branch_source = getattr(self.opt, "alg_b2b_convrefiner_resume_from", "")
        if self._refiner_training() and branch_source and os.path.isdir(self.save_dir):
            if any(
                name.endswith("_refiner_training.pth")
                for name in os.listdir(self.save_dir)
            ):
                raise ValueError(
                    "Branch already contains refiner checkpoints; clear "
                    "alg_b2b_convrefiner_resume_from and select the branch's train_epoch to resume"
                )
        load_dir = (
            os.path.abspath(os.path.expanduser(branch_source))
            if self._refiner_training() and branch_source
            else load_dir or self.save_dir
        )
        fresh = self._refiner_training() and bool(
            getattr(self.opt, "train_continue_from", "")
        )
        use_ema = (fresh and self.opt.alg_b2b_convrefiner_source_ema) or (
            not self.isTrain and getattr(self.opt, "train_G_ema", False)
        )
        suffix = "_ema" if use_ema else ""
        path = os.path.join(load_dir, f"{epoch}_net_G_A{suffix}.pth")
        state = torch.load(path, map_location=self.device, weights_only=True)
        load_refiner_generator_state(self._unwrap_net(self.netG_A), state, fresh=fresh)
        self.refiner_source_checkpoint = path
        if self._refiner_training() and not fresh:
            self.netD_refiner.load_state_dict(
                torch.load(
                    os.path.join(load_dir, f"{epoch}_net_D_refiner.pth"),
                    map_location=self.device,
                    weights_only=True,
                )
            )
            # This is our trusted training-state file, which includes Python/NumPy RNG states.
            state = torch.load(
                os.path.join(load_dir, f"{epoch}_refiner_training.pth"),
                map_location="cpu",
                weights_only=False,
            )
            saved_schedule = state.get(
                "adversarial_schedule", {"schedule": "near_clean"}
            )
            if saved_schedule != self._refiner_schedule_metadata():
                print(
                    "ConvRefiner training schedule:",
                    saved_schedule,
                    "->",
                    self._refiner_schedule_metadata(),
                )
            for optimizer, saved in zip(self.optimizers, state["optimizers"]):
                optimizer.load_state_dict(saved)
            for scheduler, saved in zip(self.schedulers, state["schedulers"]):
                scheduler.load_state_dict(saved)
            self.scaler.load_state_dict(state["scaler"])
            self.niter = state["niter"]
            self.refiner_optimizer_steps = state["optimizer_steps"]
            self.opt.total_iters = state["total_iters"]
            self.opt.train_epoch_count, self.refiner_resume_batch = state["cursor"]
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            world = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            if len(state["rngs"]) != world:
                raise ValueError(
                    "ConvRefiner resume requires the saved distributed world size"
                )
            self._refiner_resume_rng = state["rngs"][rank]
            if self.opt.train_G_ema:
                self.netG_A_ema = copy.deepcopy(self._unwrap_net(self.netG_A)).eval()
                load_refiner_generator_state(
                    self.netG_A_ema,
                    torch.load(
                        os.path.join(load_dir, f"{epoch}_net_G_A_ema.pth"),
                        map_location=self.device,
                        weights_only=True,
                    ),
                )

    def save_networks(self, epoch):
        if not self._refiner_training():
            return super().save_networks(epoch)
        if self._refiner_accumulated:
            print(
                "ConvRefiner: skipping mid-accumulation checkpoint; epoch checkpoints flush gradients."
            )
            return
        super().save_networks(epoch)
        state = {
            "adversarial_schedule": self._refiner_schedule_metadata(),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "scaler": self.scaler.state_dict(),
            "niter": self.niter,
            "optimizer_steps": self.refiner_optimizer_steps,
            "total_iters": self.opt.total_iters,
            "cursor": getattr(
                self, "_refiner_checkpoint_cursor", (self.opt.train_epoch_count, 0)
            ),
            "rngs": getattr(
                self, "_refiner_checkpoint_rngs", [self._refiner_rng_state()]
            ),
        }
        torch.save(state, os.path.join(self.save_dir, f"{epoch}_refiner_training.pth"))
